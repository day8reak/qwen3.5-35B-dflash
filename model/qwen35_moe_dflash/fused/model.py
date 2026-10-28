"""Independent stateful target fork with explicitly supplied fused operations.

This is a logical PyTorch integration path, not a claim that a yellow-zone GE
schema has been installed. Existing ge_target/export/runtime paths are untouched.
Only batch-one, unpadded, contiguous token streams are accepted in this version.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..artifacts import checked_file
from ..ge_export import weight_identity
from ..target_weight_shapes import required_weight_shapes
from .ops import ModuleFusedOps, SelectiveFusedOps
from .session import LayerState, TargetState


@dataclass(frozen=True)
class FusedTargetOutput:
    top1: Tensor
    features: Tensor
    state: TargetState
    logits: Tensor | None = None


def _configuration(config: Mapping, feature_layers: tuple[int, ...]) -> dict:
    c = deepcopy(dict(config.get("text_config", config)))
    positive = (
        "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "linear_num_key_heads",
        "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
        "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok",
        "moe_intermediate_size", "shared_expert_intermediate_size",
        "max_position_embeddings",
    )
    for name in positive:
        v = c.get(name)
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive integer")
    capacity = c.get("kv_cache_max_len", c["max_position_embeddings"])
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("kv_cache_max_len must be a positive integer")
    kinds = c.get("layer_types", [])
    if len(kinds) != c["num_hidden_layers"] or set(kinds) - {"linear_attention", "full_attention"}:
        raise ValueError("layer_types must declare every full/linear attention layer")
    if c["num_attention_heads"] % c["num_key_value_heads"] or c["linear_num_value_heads"] % c["linear_num_key_heads"]:
        raise ValueError("query/value head counts must be multiples of KV/key heads")
    if c["num_experts_per_tok"] > c["num_experts"]:
        raise ValueError("top-k exceeds expert count")
    if c.get("attention_bias", False) or c.get("attention_dropout", 0) != 0 or c.get("hidden_act", "silu") != "silu":
        raise ValueError("only bias-free zero-dropout SiLU inference is supported")
    if c.get("norm_topk_prob", True) is not True:
        raise ValueError("this fork requires normalized selected-expert probabilities")
    eps = float(c.get("rms_norm_eps", 1e-6))
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("rms_norm_eps must be finite and positive")
    c["rms_norm_eps"] = eps
    rope = c.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default":
        raise ValueError("only default text RoPE is supported")
    theta = float(rope.get("rope_theta", c.get("rope_theta", 10000.0)))
    fraction = float(rope.get("partial_rotary_factor", c.get("partial_rotary_factor", .25)))
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("invalid partial_rotary_factor")
    width = int(c["head_dim"] * fraction)
    if width < 2 or width % 2 or not math.isfinite(theta) or theta <= 0:
        raise ValueError("RoPE requires positive even rotary width and positive finite theta")
    c["rotary_dim"], c["rope_theta"] = width, theta
    if (not feature_layers or any(isinstance(i, bool) or not isinstance(i, int) for i in feature_layers)
            or tuple(sorted(set(feature_layers))) != feature_layers
            or feature_layers[0] < 0 or feature_layers[-1] >= c["num_hidden_layers"]):
        raise ValueError("feature_layers must be sorted unique valid decoder layer IDs")
    return c


class FusedTargetModel(nn.Module):
    """Stateful target with independent weights and mandatory explicit op binding.

    The logical adapter uses ND caches. Paged/NZ conversion belongs to the real
    vendor adapter. FP32 recurrent state requires an explicit constructor choice;
    no dtype mismatch is silently converted. Original W4 payloads require an
    explicitly bound OriginalQuantizedWeights adapter.
    """

    def __init__(self, config: Mapping, weights: Mapping[str, Tensor], *,
                 feature_layers: tuple[int, ...], ops: ModuleFusedOps,
                 recurrent_dtype: torch.dtype = torch.float16, quantization=None):
        super().__init__()
        if not isinstance(ops, ModuleFusedOps):
            raise TypeError("ops must be an explicitly validated ModuleFusedOps binding")
        if recurrent_dtype not in (torch.float16, torch.float32):
            raise ValueError("recurrent_dtype must explicitly select float16 or float32")
        self.config = _configuration(config, feature_layers)
        if isinstance(ops, SelectiveFusedOps):
            ops.validate_model_config(self.config, recurrent_dtype)
        self.feature_layers = tuple(feature_layers)
        self.ops, self.recurrent_dtype = ops, recurrent_dtype
        self.quantization = quantization
        shapes = required_weight_shapes(self.config)
        if quantization is not None:
            from .quantization import OriginalQuantizedWeights
            if not isinstance(quantization, OriginalQuantizedWeights):
                raise TypeError("quantization must be an OriginalQuantizedWeights adapter")
            quantization.validate_shapes(shapes)
        self._weight_names = {}
        device = quantization.device if quantization is not None else None
        for index, (name, shape) in enumerate(shapes.items()):
            if quantization is not None and quantization.has_weight(name):
                continue
            if name not in weights:
                raise ValueError(f"missing weight: {name}")
            value = weights[name]
            if not isinstance(value, Tensor) or value.dtype != torch.float16 or tuple(value.shape) != shape:
                raise ValueError(f"{name}: expected float16 {shape}")
            if device is not None and value.device != device:
                raise ValueError("all weights must be on the same device")
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite weight: {name}")
            device = value.device
            buffer_name = f"weight_{index:05d}"
            self._weight_names[name] = buffer_name
            self.register_buffer(buffer_name, value.detach().clone().contiguous())
        c = self.config
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, c["rotary_dim"], 2, dtype=torch.float32) / c["rotary_dim"]))
        self.register_buffer("inv_freq", inv.to(device=device, dtype=torch.float16))
        self.eval()

    @property
    def device(self):
        return self.inv_freq.device

    @property
    def cache_capacity(self):
        return min(self.config["max_position_embeddings"],
                   self.config.get("kv_cache_max_len", self.config["max_position_embeddings"]))

    def weight(self, name: str) -> Tensor:
        return getattr(self, self._weight_names[name])

    def initial_state(self) -> TargetState:
        c, layers = self.config, []
        for kind in c["layer_types"]:
            if kind == "linear_attention":
                kw = c["linear_num_key_heads"] * c["linear_key_head_dim"]
                vw = c["linear_num_value_heads"] * c["linear_value_head_dim"]
                layers.append(LayerState(
                    conv=torch.zeros(1, 2 * kw + vw, c["linear_conv_kernel_dim"] - 1,
                                     device=self.device, dtype=torch.float16),
                    recurrent=torch.zeros(1, c["linear_num_value_heads"], c["linear_key_head_dim"],
                                          c["linear_value_head_dim"], device=self.device, dtype=self.recurrent_dtype)))
            else:
                shape = (1, c["num_key_value_heads"], 0, c["head_dim"])
                layers.append(LayerState(key=torch.zeros(shape, device=self.device, dtype=torch.float16),
                                         value=torch.zeros(shape, device=self.device, dtype=torch.float16)))
        return TargetState(position=0, layers=tuple(layers))

    def _tensor(self, value, shape, name, dtype=torch.float16):
        if (not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != dtype or value.device != self.device):
            raise ValueError(f"{name}: expected {dtype} {tuple(shape)} on {self.device}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name}: nonfinite tensor")
        return value

    def _validate_state(self, state):
        if not isinstance(state, TargetState) or len(state.layers) != len(self.config["layer_types"]):
            raise ValueError("state does not match target layer count")
        if state.position > self.cache_capacity:
            raise ValueError("state exceeds target context capacity")
        c = self.config
        for i, (layer, kind) in enumerate(zip(state.layers, c["layer_types"])):
            if kind == "linear_attention":
                if layer.key is not None or layer.value is not None:
                    raise ValueError("linear attention state cannot contain KV")
                width = 2 * c["linear_num_key_heads"] * c["linear_key_head_dim"] + c["linear_num_value_heads"] * c["linear_value_head_dim"]
                self._tensor(layer.conv, (1, width, c["linear_conv_kernel_dim"] - 1), f"layer {i} conv state")
                self._tensor(layer.recurrent, (1, c["linear_num_value_heads"], c["linear_key_head_dim"],
                                              c["linear_value_head_dim"]), f"layer {i} recurrent state", self.recurrent_dtype)
            else:
                if layer.conv is not None or layer.recurrent is not None:
                    raise ValueError("full attention state cannot contain GDN state")
                shape = (1, self.config["num_key_value_heads"], state.position, self.config["head_dim"])
                self._tensor(layer.key, shape, f"layer {i} key state")
                self._tensor(layer.value, shape, f"layer {i} value state")

    def _linear(self, x, prefix):
        if self.quantization is not None and self.quantization.has_weight(prefix + ".weight"):
            return self.quantization.linear(x, prefix + ".weight")
        return F.linear(x, self.weight(prefix + ".weight"))

    def _norm(self, x, prefix, *, zero_centered=True):
        gamma = self.weight(prefix + ".weight")
        if zero_centered:
            gamma = gamma + 1.0
        return self._tensor(self.ops.rms_norm(x, gamma, self.config["rms_norm_eps"]), x.shape, prefix)

    def _gdn(self, x, prefix, cache):
        c, s = self.config, x.shape[1]
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        kd, vd, ksize = c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
        kw, vw = kh * kd, vh * vd
        projected = self._linear(x, prefix + ".in_proj_qkv")
        conv, padded = self.ops.conv_state_update(projected, cache.conv)
        self._tensor(conv, (1, kw * 2 + vw, ksize - 1), "conv_state_update history")
        self._tensor(padded, (1, kw * 2 + vw, s + ksize - 1), "conv_state_update padded")
        # Freeze state before another layer/operator can reuse a backend buffer.
        conv = conv.detach().clone()
        mixed = self.ops.causal_conv1d(padded, self.weight(prefix + ".conv1d.weight"))
        mixed = F.silu(self._tensor(mixed, projected.shape, "causal_conv1d"))
        q, k, v = mixed.split((kw, kw, vw), dim=-1)
        q, k = [t.reshape(1, s, kh, kd).repeat_interleave(vh // kh, dim=2).contiguous() for t in (q, k)]
        v = v.reshape(1, s, vh, vd).contiguous()
        beta = torch.sigmoid(self._linear(x, prefix + ".in_proj_b"))
        a = self._linear(x, prefix + ".in_proj_a") + self.weight(prefix + ".dt_bias")
        g = -torch.exp(self.weight(prefix + ".A_log")) * F.softplus(a)
        valid = torch.tensor([s], dtype=torch.int16, device=self.device)
        core, recurrent = self.ops.chunk_gated_delta_rule(
            q, k, v, g, beta, cache.recurrent, valid,
            chunk_size=1 if s == 1 else 64, use_qk_l2norm_in_kernel=True)
        self._tensor(core, (1, s, vh, vd), "ChunkGatedDeltaRule core")
        self._tensor(recurrent, (1, vh, kd, vd), "ChunkGatedDeltaRule state", self.recurrent_dtype)
        recurrent = recurrent.detach().clone()
        z = self._linear(x, prefix + ".in_proj_z").reshape(1, s, vh, vd)
        core = self._norm(core, prefix + ".norm", zero_centered=False) * F.silu(z)
        return self._linear(core.reshape(1, s, vw), prefix + ".out_proj"), LayerState(conv=conv, recurrent=recurrent)

    def _rotary(self, x, start):
        width = self.config["rotary_dim"]
        positions = torch.arange(start, start + x.shape[-2], device=self.device).to(torch.float16)
        frequency = positions[:, None] * self.inv_freq[None, :]
        angle = torch.cat((frequency, frequency), dim=-1)[None, None]
        a, tail = x[..., :width], x[..., width:]
        first, second = a.chunk(2, dim=-1)
        a = a * angle.cos() + torch.cat((-second, first), dim=-1) * angle.sin()
        return torch.cat((a, tail), dim=-1).contiguous()

    def _attention(self, x, prefix, cache, start):
        c, s = self.config, x.shape[1]
        heads, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q, gate = self._linear(x, prefix + ".q_proj").reshape(1, s, heads, d * 2).chunk(2, dim=-1)
        q = self._norm(q, prefix + ".q_norm").transpose(1, 2)
        k = self._norm(self._linear(x, prefix + ".k_proj").reshape(1, s, kv, d), prefix + ".k_norm").transpose(1, 2)
        v = self._linear(x, prefix + ".v_proj").reshape(1, s, kv, d).transpose(1, 2).contiguous()
        q, k = self._rotary(q, start), self._rotary(k, start)
        key, value = self.ops.cache_update(cache.key, cache.value, k, v)
        shape = (1, kv, start + s, d)
        self._tensor(key, shape, "CacheUpdate key")
        self._tensor(value, shape, "CacheUpdate value")
        key, value = key.detach().clone(), value.detach().clone()
        mixed = self.ops.attention(q, key, value, query_start=start, scale=d ** -.5)
        self._tensor(mixed, (1, heads, s, d), "fused attention")
        mixed = mixed.transpose(1, 2).reshape(1, s, heads * d) * torch.sigmoid(gate.reshape(1, s, heads * d))
        return self._linear(mixed, prefix + ".o_proj"), LayerState(key=key, value=value)

    def _moe(self, x, prefix):
        if self.quantization is not None:
            result = self.quantization.moe(x, prefix, top_k=self.config["num_experts_per_tok"])
        else:
            result = self.ops.moe(x, self.weight(prefix + ".gate.weight"),
                                  self.weight(prefix + ".experts.gate_up_proj"),
                                  self.weight(prefix + ".experts.down_proj"),
                                  top_k=self.config["num_experts_per_tok"])
        self._tensor(result, x.shape, "fused MoE")
        shared = F.silu(self._linear(x, prefix + ".shared_expert.gate_proj")) * self._linear(x, prefix + ".shared_expert.up_proj")
        shared = self._linear(shared, prefix + ".shared_expert.down_proj")
        return result + shared * torch.sigmoid(self._linear(x, prefix + ".shared_expert_gate"))

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, state: TargetState | None = None, *,
                return_logits: bool = False) -> FusedTargetOutput:
        if (not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1
                or input_ids.dtype not in (torch.int32, torch.int64) or input_ids.device != self.device):
            raise ValueError("input_ids must be int32/int64 [1,S] on the weight device")
        s = input_ids.shape[1]
        if not 1 <= s <= 32767:
            raise ValueError("unpadded sequence length must fit a positive INT16 effective_length")
        if (input_ids < 0).any() or (input_ids >= self.config["vocab_size"]).any():
            raise ValueError("input token is outside the vocabulary")
        state = self.initial_state() if state is None else state
        self._validate_state(state)
        if state.position + s > self.cache_capacity:
            raise ValueError("input would exceed target context capacity")
        # Vendor operations may mutate their input caches. They only see this
        # private branch, never the caller's committed state.
        working = state.clone()
        hidden = (self.quantization.embedding(input_ids) if self.quantization is not None else
                  F.embedding(input_ids.long(), self.weight("target.embed_tokens.weight")))
        states, features = [], {}
        for index, kind in enumerate(self.config["layer_types"]):
            p = f"target.layers.{index}"
            normed = self._norm(hidden, p + ".input_layernorm")
            mixed, new_state = (self._gdn(normed, p + ".linear_attn", working.layers[index])
                                if kind == "linear_attention" else
                                self._attention(normed, p + ".self_attn", working.layers[index], working.position))
            hidden = hidden + mixed
            hidden = hidden + self._moe(self._norm(hidden, p + ".post_attention_layernorm"), p + ".mlp")
            states.append(new_state)
            if index in self.feature_layers:
                features[index] = hidden
        hidden = self._norm(hidden, "target.norm")
        if self.config["num_hidden_layers"] - 1 in features:
            features[self.config["num_hidden_layers"] - 1] = hidden
        # Per-row head bounds temporary vocabulary logits without changing Top1.
        top1, full_logits = [], []
        for i in range(s):
            logits = self._linear(hidden[:, i:i+1], "target.lm_head")
            self._tensor(logits, (1, 1, self.config["vocab_size"]), "vocabulary logits")
            top1.append(logits.argmax(-1).to(torch.int32))
            if return_logits:
                full_logits.append(logits.float())
        captured = torch.cat([features[i] for i in self.feature_layers], dim=-1)
        self._tensor(captured, (1, s, len(self.feature_layers) * self.config["hidden_size"]), "target features")
        return FusedTargetOutput(torch.cat(top1, dim=1), captured,
                                 TargetState(position=working.position + s, layers=tuple(states)),
                                 torch.cat(full_logits, dim=1) if return_logits else None)


def load_prepared_target(prepared, *, ops, device="cpu", recurrent_dtype=torch.float16):
    root = Path(prepared).expanduser().resolve()
    plan = json.loads((root / "prepared.json").read_text())
    if (plan.get("artifact_kind") != "qwen35-ge-prepared" or plan.get("status") != "PREPARED"
            or plan.get("precision") != "float16" or plan.get("quantization") is not None):
        raise ValueError("fused target currently requires unquantized FP16 prepared weights")
    records = plan.get("weights", {})
    if not isinstance(records, dict) or not records:
        raise ValueError("prepared weights are missing")
    if plan.get("source", {}).get("fp16_payload_sha256") != weight_identity(records):
        raise ValueError("prepared weight identity mismatch")
    features = tuple(plan["draft_config"]["dflash_config"]["target_layer_ids"])
    config = _configuration(plan["target_config"], features)
    shapes = required_weight_shapes(config)
    if {key for key in records if key.startswith("target.")} != set(shapes):
        raise ValueError("prepared target weights do not cover the fused model exactly")
    weights = {}
    for key, shape in shapes.items():
        record = records[key]
        if (record.get("dtype") != "float16" or record.get("shape") != list(shape)
                or record.get("bytes") != math.prod(shape) * 2):
            raise ValueError(f"invalid prepared target descriptor: {key}")
        payload = checked_file(root, record)
        weights[key] = torch.from_numpy(np.fromfile(payload, dtype="<f2").reshape(shape)).to(device)
    model = FusedTargetModel(config, weights, feature_layers=features,
                             ops=ops, recurrent_dtype=recurrent_dtype)
    # Metadata describes the input artifact, not device validation of this fork.
    model.prepared_source = plan["source"].copy()
    return model

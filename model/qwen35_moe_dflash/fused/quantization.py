"""Explicit original-checkpoint W4 adapter for the stateful PyTorch target.

Weights are repacked, never requantized. Expert banks remain packed INT4;
only activation tensors are dynamically quantized during inference. This is
an operator invocation contract, not a declaration of target-device support.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from qwen35_moe_dflash.graph.quant_plan import target_tensor_quantization
from qwen35_moe_dflash.graph.weights import SafeTensorFiles, checkpoint_key


class OriginalQuantOps:
    """Require every vendor operation before allocating model weights.

    CPU/mock implementations must explicitly opt into reference mode. There
    is no automatic replacement of a missing vendor operation with float ops.
    """

    _OPERATIONS = (
        "npu_dynamic_quant", "npu_quant_matmul", "npu_trans_quant_param",
        "npu_moe_gating_top_k_softmax", "npu_moe_init_routing_v2",
        "npu_grouped_matmul", "npu_moe_finalize_routing",
    )

    def __init__(self, module, *, backend_id: str, is_reference: bool,
                 allow_reference: bool = False, device_type: str | None = None):
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("quant backend_id is required")
        if type(is_reference) is not bool or (is_reference and not allow_reference):
            raise ValueError("reference quant backend requires explicit allow_reference=True")
        missing = [name for name in self._OPERATIONS if not callable(getattr(module, name, None))]
        if missing:
            raise RuntimeError("original quant backend lacks required operators: " + ", ".join(missing))
        self.backend_id, self.is_reference = backend_id, is_reference
        self.device_type = device_type or ("cpu" if is_reference else "npu")
        if self.device_type not in {"cpu", "npu"}:
            raise ValueError("quant backend must declare cpu or npu execution device type")
        for name in self._OPERATIONS:
            setattr(self, name, getattr(module, name))

    @classmethod
    def from_torch_npu(cls):
        try:
            module = importlib.import_module("torch_npu")
        except ImportError as exc:
            raise RuntimeError("original quant inference requires an explicitly installed torch_npu backend") from exc
        return cls(module, backend_id="torch_npu-original-w4", is_reference=False)

    def validate_device(self, device):
        # Check before tensor allocation: torch_npu's QPARAM converter requires
        # NPU input just like its quantized matmuls. CPU is reference-only by
        # default, and must never silently dispatch a real NPU binding.
        if str(device).split(":", 1)[0] != self.device_type:
            raise ValueError(f"quant backend {self.backend_id} requires device='{self.device_type}:0'; "
                             f"received {device!r}")


class OriginalQuantizedWeights(nn.Module):
    """Registered I4ZN INT32 carriers and original scales for one target.

    Ordinary matrix scales stay FP32. Expert scales must be converted by the
    supplied backend to its INT64 QPARAM representation; a cast is not valid.
    Main/gated norm weights remain outside this adapter, as raw HF tensors.
    """

    def __init__(self, shapes: Mapping, reader, embedding, *, ops: OriginalQuantOps,
                 device="cpu"):
        super().__init__()
        if not isinstance(ops, OriginalQuantOps):
            raise TypeError("ops must be an explicit OriginalQuantOps binding")
        ops.validate_device(device)
        self.ops = ops
        self.shapes = {key: tuple(shape) for key, shape in shapes.items()}
        self.kinds = {key: target_tensor_quantization(key, shape)["kind"]
                      for key, shape in self.shapes.items()}
        self._names = {}
        weight, scale = embedding
        expected = self.shapes["target.embed_tokens.weight"]
        if (np.asarray(weight).dtype != np.int8 or tuple(weight.shape) != expected
                or np.asarray(scale).dtype != np.float32 or tuple(scale.shape) != (expected[0], 1)):
            raise ValueError("embedding requires original INT8 [vocab,hidden] and FP32 [vocab,1]")
        if not np.isfinite(scale).all():
            raise ValueError("embedding scale is nonfinite")
        # Copy the INT8 embedding only; never materialize a dequantized table.
        self.register_buffer("embedding_weight", torch.from_numpy(np.array(weight, copy=True)).to(device))
        self.register_buffer("embedding_scale", torch.from_numpy(np.array(scale, copy=True)).to(device))

        def convert_qparam(values):
            source = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(device)
            converted = ops.npu_trans_quant_param(source.reshape(-1), None, 1)
            if not isinstance(converted, torch.Tensor) or converted.dtype != torch.int64 or converted.numel() != source.numel():
                raise ValueError("npu_trans_quant_param must return INT64 QPARAM with unchanged element count")
            return converted.reshape(source.shape).detach().cpu().numpy()

        for index, (key, shape) in enumerate(self.shapes.items()):
            kind = self.kinds[key]
            if kind not in {"linear_int4", "expert_int4"}:
                continue
            item = reader.packed(key, shape, qparam_converter=convert_qparam if kind == "expert_int4" else None)
            packed, scales = np.asarray(item["weight"]), np.asarray(item["scale"])
            e = shape[:-2]
            n, k = shape[-2:]
            if (packed.dtype != np.int32 or packed.shape != (*e, k, n // 8)
                    or scales.shape != (*e, k // 256, n)
                    or scales.dtype != (np.int64 if e else np.float32)):
                raise ValueError(f"invalid original I4ZN/QPARAM descriptor: {key}")
            if scales.dtype == np.float32 and not np.isfinite(scales).all():
                raise ValueError(f"nonfinite quant scale: {key}")
            wname, sname = f"packed_{index}", f"scale_{index}"
            self.register_buffer(wname, torch.from_numpy(np.ascontiguousarray(packed)).to(device))
            self.register_buffer(sname, torch.from_numpy(np.ascontiguousarray(scales)).to(device))
            self._names[key] = wname, sname
        self.eval()

    @property
    def device(self):
        return self.embedding_weight.device

    def has_weight(self, key):
        return key == "target.embed_tokens.weight" or key in self._names

    def validate_shapes(self, shapes):
        if {key: tuple(shape) for key, shape in shapes.items()} != self.shapes:
            raise ValueError("original quant adapter geometry does not match target model")

    def _pair(self, key):
        wname, sname = self._names[key]
        return getattr(self, wname), getattr(self, sname)

    def embedding(self, token_ids):
        # Cpro applies the original row scale once and keeps the residual
        # stream in FP32. Individual quantized projections still return FP16.
        return (self.embedding_weight[token_ids.long()].float()
                * self.embedding_scale[token_ids.long()])

    def _dynamic_quant(self, x):
        quant, scale = self.ops.npu_dynamic_quant(x)
        if (not isinstance(quant, torch.Tensor) or quant.dtype != torch.int8
                or quant.shape != x.shape or quant.device != x.device
                or not isinstance(scale, torch.Tensor) or scale.numel() != x.shape[0]
                or scale.device != x.device or scale.dtype not in (torch.float16, torch.float32)):
            raise ValueError("DynamicQuant must return INT8 activations and one floating scale per row")
        return quant, scale.reshape(-1)

    def linear(self, x, key):
        if self.kinds[key] != "linear_int4" or x.shape[-1] != self.shapes[key][-1]:
            raise ValueError(f"invalid original quant linear invocation: {key}")
        flat = x.reshape(-1, x.shape[-1])
        quant, per_token = self._dynamic_quant(flat)
        weight, scale = self._pair(key)
        out = self.ops.npu_quant_matmul(quant, weight, scale,
            pertoken_scale=per_token.reshape(-1, 1), output_dtype=torch.float16)
        expected = (flat.shape[0], self.shapes[key][-2])
        if not isinstance(out, torch.Tensor) or tuple(out.shape) != expected or out.dtype != torch.float16 or out.device != x.device:
            raise ValueError(f"QuantMatmul returned wrong tensor for {key}; expected FP16 {expected}")
        return out.reshape(*x.shape[:-1], expected[-1])

    def _grouped(self, x, key, expert_tokens):
        quant, per_token = self._dynamic_quant(x)
        weight, scale = self._pair(key)
        result = self.ops.npu_grouped_matmul([quant], [weight], scale=[scale],
            per_token_scale=[per_token], group_list=expert_tokens, split_item=3,
            output_dtype=torch.float16, group_type=0, group_list_type=1, act_type=0)
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise ValueError("GroupedMatmul must return one output tensor")
        output = result[0]
        if (not isinstance(output, torch.Tensor) or output.dtype != torch.float16
                or output.device != x.device or tuple(output.shape) != (x.shape[0], self.shapes[key][-2])):
            raise ValueError("GroupedMatmul returned an invalid expert output")
        return output

    def moe(self, x, prefix, *, top_k):
        flat = x.reshape(-1, x.shape[-1])
        router = self.linear(flat, prefix + ".gate.weight")
        probabilities, indices, _ = self.ops.npu_moe_gating_top_k_softmax(router, None, k=top_k)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        probabilities = probabilities.to(x.dtype)
        experts = self.shapes[prefix + ".experts.gate_up_proj"][0]
        routed, row_map, counts, _ = self.ops.npu_moe_init_routing_v2(flat,
            expert_idx=indices.to(torch.int32), active_num=flat.shape[0] * top_k,
            expert_num=experts, expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, experts], quant_mode=-1)
        counts = counts.to(torch.int64)
        hidden = self._grouped(routed, prefix + ".experts.gate_up_proj", counts)
        gate, up = hidden.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        hidden = self._grouped(hidden, prefix + ".experts.down_proj", counts)
        output = self.ops.npu_moe_finalize_routing(hidden, skip1=None, skip2=None, bias=None,
            scales=probabilities.to(hidden.dtype), expanded_src_to_dst_row=row_map,
            export_for_source_row=None, drop_pad_mode=2)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != flat.shape or output.dtype != torch.float16 or output.device != x.device:
            raise ValueError("MoeFinalizeRouting returned an invalid output")
        return output.reshape(x.shape)


def _raw_base_weights(model_file, shapes, kinds, device):
    """Read only norms/conv/gates, never HF embedding/linear/expert banks."""
    source = SafeTensorFiles(model_file)
    result = {}
    for key, shape in shapes.items():
        if kinds[key] != "float16":
            continue
        _, source_key = checkpoint_key(key)
        path, offset, count, dtype = source.tensor(source_key, shape)
        numpy_dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[dtype]
        values = np.fromfile(path, dtype=numpy_dtype, count=count, offset=offset)
        if dtype == "BF16":
            values = (values.astype("<u4") << 16).view("<f4")
        # Do not fold +1 into any norm: the model applies +1 only to main/q/k.
        result[key] = torch.from_numpy(values.astype("<f2").reshape(shape)).to(device)
    return result


def load_original_quantized_target(config, *, model_file, quanted_pth,
        embedding_weight_path, embedding_scale_path, feature_layers, ops,
        quant_ops: OriginalQuantOps, device="cpu", recurrent_dtype=torch.float16):
    """Executable loader; paths come from caller/YAML, with no machine defaults."""
    from qwen35_moe_dflash.target.original_quant import OriginalQuantWeights, load_embedding
    from qwen35_moe_dflash.fused.model import FusedTargetModel, _configuration, required_weight_shapes
    normalized = _configuration(config, tuple(feature_layers))
    if not isinstance(quant_ops, OriginalQuantOps):
        raise TypeError("quant_ops must be an explicit OriginalQuantOps binding")
    quant_ops.validate_device(device)
    shapes = required_weight_shapes(normalized)
    quantization = OriginalQuantizedWeights(shapes, OriginalQuantWeights(quanted_pth),
        load_embedding(embedding_weight_path, embedding_scale_path,
                       shapes["target.embed_tokens.weight"]), ops=quant_ops, device=device)
    weights = _raw_base_weights(model_file, shapes, quantization.kinds, device)
    model = FusedTargetModel(normalized, weights, feature_layers=tuple(feature_layers),
                            ops=ops, recurrent_dtype=recurrent_dtype, quantization=quantization)
    model.original_quant_contract = {
        "weight_source": "ORIGINAL_W4_NO_REQUANTIZATION", "layout": "I4ZN_INT32",
        "ordinary_scale": "FP32", "expert_scale": "VENDOR_INT64_QPARAM",
        "embedding": "ORIGINAL_INT8_TIMES_FP32_ROW_SCALE",
        "residual_dtype": "float32", "quantized_projection_output_dtype": "float16",
        "base_weight_storage": "float16",
        "precision_limits": "Base weights remain FP16 snapshots; full bitwise original-AIR parity is not established.",
        "gdn_gate_math": "FP32_THEN_EXPLICIT_FP16_PORT_CAST",
        "gated_norm_math": "FP32_RMS_AND_SILU_PRODUCT_THEN_FP16",
        "rotary_math": "FP32_POSITIONS_FREQUENCIES_TRIG_THEN_ACTIVATION_CAST",
        "final_norm_and_head_input": "float32", "dflash_features": "float16",
        "norm_main_addend": 1, "norm_gated_addend": 0,
        "backend": quant_ops.backend_id, "reference_backend": quant_ops.is_reference,
        "execution_device_type": quant_ops.device_type,
        "accuracy": "UNVERIFIED", "placement": "UNVERIFIED",
    }
    return model

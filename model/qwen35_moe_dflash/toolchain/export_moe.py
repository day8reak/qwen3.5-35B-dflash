"""Exact sparse expert dispatch with a static expert enumeration for export.

The upstream eager path enumerates only the experts hit by the example input.
Tracing that Python loop would freeze routing coverage. Enumerate every expert,
keeping token selection, weights, and accumulation inside the tensor graph.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ExportExperts(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.num_experts = experts.num_experts
        self.gate_up_proj = experts.gate_up_proj
        self.down_proj = experts.down_proj
        self.act_fn = experts.act_fn

    def forward(self, hidden, selected, weights):
        # Append a zero-weight sentinel so even unselected experts have a
        # nonempty MatMul. Discard its output after accumulation. This keeps
        # zero-length Gather/MatMul out of the converter's runtime shape path.
        count, width = hidden.shape
        padded = torch.cat((hidden, hidden.new_zeros((1, width))), dim=0)
        result = torch.zeros_like(padded)
        for expert in range(self.num_experts):
            slot, token = torch.where(selected.transpose(0, 1) == expert)
            route_weights = weights[token, slot]
            token = torch.cat((token, torch.full(
                (1,), count, dtype=torch.long, device=hidden.device)))
            route_weights = torch.cat((route_weights, route_weights.new_zeros((1,))))
            gate, up = F.linear(padded[token], self.gate_up_proj[expert]).chunk(2, dim=-1)
            values = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            values = (values * route_weights[:, None]).to(result.dtype)
            result = result.index_add(0, token, values)
        return result[:count]


class StaticExportExperts(ExportExperts):
    """Fixed shapes for AIR export; expensive all-expert correctness baseline.

    No example-dependent loop or NonZero. The selected top-k weights remain
    dynamic. Sum contributions in ascending expert order like upstream eager.
    """
    def forward(self, hidden, selected, weights):
        result = torch.zeros_like(hidden)
        for expert in range(self.num_experts):
            route = selected == expert
            weight = torch.where(route, weights, torch.zeros_like(weights)).sum(-1)
            gate, up = F.linear(hidden, self.gate_up_proj[expert]).chunk(2, dim=-1)
            values = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            contribution = (values * weight[:, None]).to(result.dtype)
            result = result + torch.where(route.any(-1, keepdim=True), contribution,
                                          torch.zeros_like(contribution))
        return result


def enable_export_dispatch(language_model, mode="routing"):
    if mode not in ("routing", "static"):
        raise ValueError("unknown export expert dispatch")
    cls = StaticExportExperts if mode == "static" else ExportExperts
    replaced = 0
    for layer in language_model.layers:
        moe = layer.mlp
        if hasattr(moe, "experts") and getattr(moe.experts, "export_dispatch", None) == mode:
            replaced += 1
            continue
        if hasattr(moe, "experts") and type(moe.experts) is not cls:
            moe.experts = cls(moe.experts)
            replaced += 1
    if not replaced:
        raise ValueError("no supported Qwen3.5 MoE expert modules found")
    # TargetGraph has no multimodal inputs: all three position axes are the
    # same. Selecting T/H/W frequencies therefore returns identical elements.
    # Avoid in-place strided MRoPE writes, including empty tiny-fixture slices.
    rope_type = language_model.config.rope_parameters.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError("export supports the locked default text RoPE only")
    language_model.rotary_emb = TextOnlyRotary(language_model.rotary_emb)
    return replaced


class TextOnlyRotary(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.register_buffer("inv_freq", original.inv_freq, persistent=False)
        self.attention_scaling = original.attention_scaling

    def forward(self, hidden, positions):
        text_positions = positions[0] if positions.ndim == 3 else positions
        inv = self.inv_freq[None, :, None].float().expand(text_positions.shape[0], -1, 1)
        frequency = (inv @ text_positions[:, None, :].float()).transpose(1, 2)
        embedding = torch.cat((frequency, frequency), dim=-1)
        return ((embedding.cos() * self.attention_scaling).to(hidden.dtype),
                (embedding.sin() * self.attention_scaling).to(hidden.dtype))


class ExportTargetGraph(nn.Module):
    """Explicit upstream decoder loop without Python hidden-state capture hooks.

    The reference TargetGraph still calls the unwrapped upstream text model.
    This loop has the same text positions, masks, layer order and final norm.
    """
    def __init__(self, original):
        super().__init__()
        self.language_model = original.language_model
        self.lm_head = original.lm_head
        self.config = original.config
        self.head_chunk_rows = original.head_chunk_rows

    def forward(self, input_ids, attention_mask):
        model = self.language_model
        hidden = model.embed_tokens(input_ids.long())
        rows = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = rows.view(1, -1)
        rope_positions = positions.unsqueeze(0).expand(3, -1, -1)
        rotary = model.rotary_emb(hidden, rope_positions)
        visible = (rows[:, None] >= rows[None, :]).view(1, 1, rows.shape[0], rows.shape[0])
        visible = visible & attention_mask.bool()[:, None, None, :]
        full_mask = torch.where(visible, torch.zeros((), dtype=hidden.dtype, device=hidden.device),
                               torch.full((), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device))
        captures = {}
        for index, layer in enumerate(model.layers):
            # Calling forward also avoids reference-only capture hooks if the
            # original module was previously executed with output_hidden_states.
            hidden = layer.forward(hidden, position_embeddings=rotary,
                                   attention_mask=(attention_mask if layer.block_type == "linear_attention" else full_mask),
                                   position_ids=positions, past_key_values=None, use_cache=False)
            if index in self.config.target_layer_ids:
                captures[index] = hidden
        hidden = model.norm(hidden)
        last = len(model.layers) - 1
        if last in captures:
            captures[last] = hidden  # Transformers replaces the last recorded output with the final norm.
        features = torch.cat([captures[i] for i in self.config.target_layer_ids], dim=-1)
        top1 = [self.lm_head(hidden[:, start:start+self.head_chunk_rows]).argmax(-1).to(torch.int32)
                for start in range(0, input_ids.shape[1], self.head_chunk_rows)]
        return torch.cat(top1, dim=1), features.float()


def prepare_export_target(target, mode="static"):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule, Qwen3_5MoeRMSNormGated)
    enable_export_dispatch(target.language_model, mode=mode)
    for layer in target.language_model.layers:
        if hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn
            gdn.causal_conv1d_fn = None
            gdn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            gdn.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
            if not isinstance(gdn.norm, Qwen3_5MoeRMSNormGated):
                norm = Qwen3_5MoeRMSNormGated(gdn.head_v_dim, eps=gdn.layer_norm_epsilon)
                norm.weight = gdn.norm.weight
                gdn.norm = norm
    return ExportTargetGraph(target).eval()

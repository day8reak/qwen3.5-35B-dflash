"""Canonical target tensor inventory, independent of Torch and GE runtimes."""
from __future__ import annotations

from typing import Mapping


def required_weight_shapes(c: Mapping) -> dict[str, tuple[int, ...]]:
    """Canonical target.* names, compatible with existing prepared FP16 payloads."""
    h, d = c["hidden_size"], c["head_dim"]
    shapes = {"target.embed_tokens.weight": (c["vocab_size"], h),
              "target.norm.weight": (h,), "target.lm_head.weight": (c["vocab_size"], h)}
    for i, kind in enumerate(c["layer_types"]):
        p = f"target.layers.{i}"
        shapes[p + ".input_layernorm.weight"] = (h,)
        shapes[p + ".post_attention_layernorm.weight"] = (h,)
        if kind == "linear_attention":
            q = p + ".linear_attn"
            kw = c["linear_num_key_heads"] * c["linear_key_head_dim"]
            vw = c["linear_num_value_heads"] * c["linear_value_head_dim"]
            vh = c["linear_num_value_heads"]
            shapes.update({q + ".in_proj_qkv.weight": (kw * 2 + vw, h),
                           q + ".in_proj_z.weight": (vw, h),
                           q + ".in_proj_b.weight": (vh, h), q + ".in_proj_a.weight": (vh, h),
                           q + ".A_log": (vh,), q + ".dt_bias": (vh,),
                           q + ".conv1d.weight": (kw * 2 + vw, 1, c["linear_conv_kernel_dim"]),
                           q + ".norm.weight": (c["linear_value_head_dim"],),
                           q + ".out_proj.weight": (h, vw)})
        else:
            q = p + ".self_attn"
            shapes.update({q + ".q_proj.weight": (c["num_attention_heads"] * d * 2, h),
                           q + ".k_proj.weight": (c["num_key_value_heads"] * d, h),
                           q + ".v_proj.weight": (c["num_key_value_heads"] * d, h),
                           q + ".o_proj.weight": (h, c["num_attention_heads"] * d),
                           q + ".q_norm.weight": (d,), q + ".k_norm.weight": (d,)})
        e, m, shared = c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
        q = p + ".mlp"
        shapes.update({q + ".gate.weight": (e, h),
                       q + ".experts.gate_up_proj": (e, 2 * m, h),
                       q + ".experts.down_proj": (e, h, m),
                       q + ".shared_expert.gate_proj.weight": (shared, h),
                       q + ".shared_expert.up_proj.weight": (shared, h),
                       q + ".shared_expert.down_proj.weight": (h, shared),
                       q + ".shared_expert_gate.weight": (1, h)})
    return shapes

"""Explicit CPU oracle for the new fused-model adapter contract.

This module implements mathematical reference operations, not vendor kernels.
It must be selected with ``allow_reference=True``.  The FP32 norm, softmax and
recurrent calculations here do not change deployment precision, emulate an
OMC, or establish board accuracy.  State returns use the caller's declared
state dtype; FP16 state therefore rounds at each invocation boundary.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

FUSED_OPS_ABI = 1
BACKEND_ID = "host-reference-only"
IS_REFERENCE_BACKEND = True


def _cpu(*values: Tensor | None) -> None:
    for value in values:
        if value is not None and (not isinstance(value, Tensor) or value.device.type != "cpu"):
            raise ValueError("host-reference-only requires CPU tensors")


def rms_norm(x: Tensor, gamma: Tensor, eps: float) -> Tensor:
    """The supplied gamma is effective: this function never adds one."""
    _cpu(x, gamma)
    if not math.isfinite(eps) or eps <= 0 or gamma.shape != x.shape[-1:]:
        raise ValueError("RMSNorm needs positive epsilon and one gamma per feature")
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * gamma


def conv_state_update(projected: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
    """Canonical ND history has exactly K-1 elements, independently of vendor ABI."""
    _cpu(projected, history)
    if (projected.ndim != 3 or history.ndim != 3 or projected.shape[1] < 1
            or projected.shape[0] != history.shape[0] or projected.shape[2] != history.shape[1]
            or projected.dtype != history.dtype):
        raise ValueError("conv state requires projected [B,S,C] and matching history [B,C,K-1]")
    padded = torch.cat((history, projected.transpose(1, 2)), dim=-1)
    count = history.shape[-1]
    updated = padded[..., -count:] if count else padded[..., :0]
    return updated.clone(), padded


def causal_conv1d(padded: Tensor, weight: Tensor) -> Tensor:
    _cpu(padded, weight)
    if (padded.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1
            or padded.shape[1] != weight.shape[0] or weight.shape[-1] < 1
            or padded.shape[-1] < weight.shape[-1] or padded.dtype != weight.dtype):
        raise ValueError("depthwise conv requires [B,C,S+K-1] and [C,1,K]")
    return F.conv1d(padded, weight, groups=weight.shape[0]).transpose(1, 2).contiguous()


def chunk_gated_delta_rule(
    query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor,
    initial_state: Tensor, effective_length: Tensor, *, chunk_size: int,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[Tensor, Tensor]:
    """Sequential gated delta-rule oracle; chunk_size is an adapter attribute.

    A correct chunk implementation must implement the same recurrence.  This
    oracle deliberately uses the sequential form and does not claim matching
    chunk-kernel rounding.  Only valid prefix tokens update recurrent state.
    """
    _cpu(query, key, value, g, beta, initial_state, effective_length)
    if isinstance(chunk_size, bool) or chunk_size not in (1, 64):
        raise ValueError("chunk_size must be 1 or 64")
    if not isinstance(use_qk_l2norm_in_kernel, bool):
        raise ValueError("use_qk_l2norm_in_kernel must be bool")
    if (query.ndim != 4 or key.shape != query.shape or value.ndim != 4
            or value.shape[:3] != query.shape[:3] or g.shape != query.shape[:3]
            or beta.shape != query.shape[:3] or any(d < 1 for d in query.shape)
            or value.shape[-1] < 1):
        raise ValueError("GDN requires compatible [B,S,H,D] tensors and [B,S,H] gates")
    batch, sequence, heads, key_width = query.shape
    if (initial_state.shape != (batch, heads, key_width, value.shape[-1])
            or initial_state.dtype not in (torch.float16, torch.float32)):
        raise ValueError("initial_state must be FP16/FP32 [B,H,Dk,Dv]")
    if (effective_length.dtype != torch.int16 or effective_length.shape != (batch,)
            or bool(((effective_length < 0) | (effective_length > sequence)).any())):
        raise ValueError("effective_length must be INT16 [B] within [0,S]")
    if query.dtype not in (torch.float16, torch.float32) or any(
        item.dtype != query.dtype for item in (key, value, g, beta)
    ):
        raise ValueError("GDN floating inputs must share FP16 or FP32 dtype")
    q, k, v, decay, gate = [item.float() for item in (query, key, value, g, beta)]
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * key_width ** -0.5
    state = initial_state.float().clone()
    outputs = []
    for position in range(sequence):
        valid = position < effective_length
        decayed = state * decay[:, position].exp()[..., None, None]
        prediction = (decayed * k[:, position, :, :, None]).sum(-2)
        delta = (v[:, position] - prediction) * gate[:, position, :, None]
        candidate = decayed + k[:, position, :, :, None] * delta[..., None, :]
        state = torch.where(valid[:, None, None, None], candidate, state)
        output = (state * q[:, position, :, :, None]).sum(-2)
        outputs.append(torch.where(valid[:, None, None], output, torch.zeros_like(output)))
    return torch.stack(outputs, dim=1).to(query.dtype), state.to(initial_state.dtype)


def cache_update(key: Tensor | None, value: Tensor | None,
                 new_key: Tensor, new_value: Tensor) -> tuple[Tensor, Tensor]:
    """Return privately owned logical ND caches; no in-place source writes."""
    _cpu(key, value, new_key, new_value)
    if ((key is None) != (value is None) or new_key.ndim != 4
            or new_value.ndim != 4 or new_key.shape[:3] != new_value.shape[:3]):
        raise ValueError("KV caches must appear together and use [B,H,T,D]")
    if key is None:
        return new_key.clone(), new_value.clone()
    for old, new in ((key, new_key), (value, new_value)):
        if old.ndim != 4 or old.shape[:2] != new.shape[:2] or old.shape[-1] != new.shape[-1] or old.dtype != new.dtype:
            raise ValueError("old and new KV cache dimensions/dtypes do not match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("old key and value cache lengths differ")
    return torch.cat((key, new_key), -2), torch.cat((value, new_value), -2)


def attention(query: Tensor, key: Tensor, value: Tensor, *,
              query_start: int, scale: float) -> Tensor:
    _cpu(query, key, value)
    if (query.ndim != 4 or key.ndim != 4 or value.ndim != 4
            or query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]
            or query.shape[-1] != key.shape[-1] or key.shape[1] < 1
            or query.shape[1] % key.shape[1] or isinstance(query_start, bool)
            or not isinstance(query_start, int) or query_start < 0
            or query_start + query.shape[-2] != key.shape[-2]
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("attention requires causal append-only GQA [B,H,S,D] inputs")
    groups = query.shape[1] // key.shape[1]
    repeated_key = key.repeat_interleave(groups, 1).float()
    repeated_value = value.repeat_interleave(groups, 1).float()
    scores = query.float() @ repeated_key.transpose(-1, -2) * scale
    positions = torch.arange(query.shape[-2]) + query_start
    visible = torch.arange(key.shape[-2])[None, :] <= positions[:, None]
    probabilities = torch.softmax(scores.masked_fill(~visible[None, None], -torch.inf), -1)
    return (probabilities @ repeated_value).to(query.dtype)


def moe(hidden: Tensor, router_weight: Tensor, gate_up: Tensor, down: Tensor, *,
        top_k: int) -> Tensor:
    """Explicit sparse experts, normalized selected probabilities, ascending sum."""
    _cpu(hidden, router_weight, gate_up, down)
    if (hidden.ndim != 3 or router_weight.ndim != 2 or gate_up.ndim != 3
            or down.ndim != 3 or gate_up.shape[1] % 2
            or router_weight.shape[0] != gate_up.shape[0] or down.shape[0] != gate_up.shape[0]
            or router_weight.shape[1] != hidden.shape[-1] or gate_up.shape[2] != hidden.shape[-1]
            or down.shape[1:] != (hidden.shape[-1], gate_up.shape[1] // 2)
            or isinstance(top_k, bool) or not isinstance(top_k, int)
            or not 0 < top_k <= router_weight.shape[0]):
        raise ValueError("MoE needs compatible expert weights and a valid top_k")
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(flat, router_weight).float()
    probabilities = torch.softmax(logits, dim=-1)
    selected_weights, selected = torch.topk(probabilities, top_k, dim=-1)
    selected_weights = (selected_weights / selected_weights.sum(-1, keepdim=True)).to(hidden.dtype)
    result = torch.zeros_like(flat)
    for expert in range(router_weight.shape[0]):
        tokens, slots = torch.where(selected == expert)
        if not tokens.numel():
            continue
        gate, up = F.linear(flat[tokens], gate_up[expert]).chunk(2, dim=-1)
        values = F.linear(F.silu(gate) * up, down[expert])
        result[tokens] += values * selected_weights[tokens, slots, None]
    return result.reshape_as(hidden)

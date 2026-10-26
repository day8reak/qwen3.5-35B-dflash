"""Static selected-expert Gather/BatchMatMul construction for direct GE.

The caller supplies the existing router's unique TopK IDs and normalized FP16
weights. This module changes neither routing, normalization, nor checkpoint
layout. It builds one token at a time so each gathered bank contains K experts,
not S*K experts; the compiler still determines the actual live memory peak.
"""
from __future__ import annotations

import numpy as np


def selected_experts(b, flat, indices, normalized_weights, gate_up, down, middle):
    """Return FP16 ``[tokens, hidden]`` selected-expert contributions.

    ``indices`` must contain unique in-range expert IDs in each row, as produced
    by router TopK. Rank and dtype constraints are checked here; runtime values
    remain the caller's contract. Expert IDs are sorted exactly through FP16
    only for at most 2048 experts. Contributions are then added sequentially in
    ascending expert-ID order, retaining the all-expert baseline's FP16 addition
    boundaries. BatchMatMul kernels may have different internal accumulation;
    this construction alone is not device numerical or placement evidence.

    The banks keep their checkpoint shapes ``[E, 2*I, H]`` and ``[E, H, I]``.
    All generated tensors have rank at most three. No integer-data Gather,
    GatherElements, NonZero or custom operator is required.
    """
    if type(middle) is not int or middle <= 0:
        raise ValueError("selected experts require a positive integer intermediate width")
    if len(flat.shape) != 2 or flat.dtype != "float16" or min(flat.shape) <= 0:
        raise ValueError("selected experts require nonempty FP16 [tokens, hidden] input")
    tokens, hidden = flat.shape
    if len(indices.shape) != 2 or indices.dtype != "int32" or indices.shape[0] != tokens:
        raise ValueError("selected experts require INT32 [tokens, top_k] indices")
    top_k = indices.shape[1]
    if normalized_weights.shape != indices.shape or normalized_weights.dtype != "float16":
        raise ValueError("selected experts require FP16 route weights matching indices")
    if len(gate_up.shape) != 3 or len(down.shape) != 3:
        raise ValueError("selected experts require rank-three checkpoint banks")
    experts = gate_up.shape[0]
    if not 1 <= top_k <= experts <= 2048:
        raise ValueError("selected experts require 1 <= top_k <= experts <= 2048 for exact FP16 IDs")
    if (gate_up.dtype != "float16" or down.dtype != "float16"
            or gate_up.shape != (experts, 2 * middle, hidden)
            or down.shape != (experts, hidden, middle)):
        raise ValueError("selected experts require FP16 banks [E, 2*I, H] and [E, H, I]")

    rows = []
    zero = b.constant(np.zeros((1, hidden), dtype=np.float16))
    for token in range(tokens):
        ids = b.view(b.slice(indices, 0, token, 1), (top_k,))
        negative_ids = b.op("neg", b.op("cast", ids, dtype="float16"))
        sorted_negative_ids = b.op("topk_values", negative_ids, shape=(top_k,), k=top_k, axis=-1)
        order = b.op("topk_indices", negative_ids, shape=(top_k,), dtype="int32", k=top_k, axis=-1)
        sorted_ids = b.op("cast", b.op("neg", sorted_negative_ids), dtype="int32")
        weights = b.view(b.slice(normalized_weights, 0, token, 1), (top_k,))
        weights = b.op("gather", weights, order, shape=(top_k,), axis=0)

        selected_gate_up = b.op("gather", gate_up, sorted_ids,
                                shape=(top_k, 2 * middle, hidden), axis=0)
        selected_down = b.op("gather", down, sorted_ids,
                             shape=(top_k, hidden, middle), axis=0)
        row = b.view(b.slice(flat, 0, token, 1), (1, 1, hidden))
        # Equal explicit batch prefixes avoid implicit BatchMatMul broadcasting.
        repeated = b.cat([row] * top_k, 0)
        projected = b.op("matmul", repeated, selected_gate_up,
                         shape=(top_k, 1, 2 * middle), transpose_a=False, transpose_b=True)
        gate = b.slice(projected, -1, 0, middle)
        up = b.slice(projected, -1, middle, middle)
        activated = b.op("mul", b.op("silu", gate), up)
        projected = b.op("matmul", activated, selected_down,
                         shape=(top_k, 1, hidden), transpose_a=False, transpose_b=True)
        weighted = b.op("mul", projected, b.view(weights, (top_k, 1, 1)))

        result = zero
        for slot in range(top_k):
            contribution = b.view(b.slice(weighted, 0, slot, 1), (1, hidden))
            result = b.op("add", result, contribution)
        rows.append(result)
    return b.cat(rows, 0)

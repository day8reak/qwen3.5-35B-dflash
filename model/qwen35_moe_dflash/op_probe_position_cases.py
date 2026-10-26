"""Bounded position/RoPE/mask probes; no production graph policy changes.

All external tensors are FP16 or INT32. Mask lookup candidates accept only
binary INT32 masks; they do not implement arbitrary nonzero-to-true casting.
References retain the original operation boundaries and finite mask fill.
"""
from __future__ import annotations

import numpy as np

from .ge_ir import Graph


_MASK_SHAPE = (1, 1, 8, 8)
_BLOCKED = np.finfo(np.float16).min


def _view(graph, value, shape, name):
    return graph.op("reshape", value, shape=shape, dtype=value.dtype,
                    attrs={"shape": list(shape)}, name=name)


def _rope_graph(name, mode):
    graph = Graph(name)
    inverse = graph.input("inverse", (1, 2, 1))
    positions = graph.input("positions", (1, 1, 8))
    if mode == "bmm":
        result = graph.op("matmul", inverse, positions, shape=(1, 2, 8), name="result")
    elif mode == "matmul":
        left = _view(graph, inverse, (2, 1), "inverse_matrix")
        right = _view(graph, positions, (1, 8), "position_matrix")
        result = graph.op("matmul", left, right, shape=(2, 8), name="frequency_matrix")
        result = _view(graph, result, (1, 2, 8), "result")
    elif mode == "outer_mul":
        result = graph.op("mul", inverse, positions, shape=(1, 2, 8), name="result")
    else:
        raise ValueError("unknown RoPE probe mode")
    graph.outputs = [result]
    return graph


def _position_graph(name, compact):
    graph = Graph(name)
    positions = graph.input("positions", (1, 8), "int32")
    if compact:
        positions = _view(graph, positions, (8,), "position_vector")
    result = graph.op("cast", positions, shape=positions.shape, dtype="float16",
                      name="float_positions" if compact else "result")
    if compact:
        result = _view(graph, result, (1, 8), "result")
    graph.outputs = [result]
    return graph


def _mask_graph(name, mode):
    graph = Graph(name)
    mask = graph.input("attention_mask", (1, 8), "int32")
    causal = graph.const(np.tril(np.ones(_MASK_SHAPE, dtype=bool)), "bool", name="causal")
    blocked = graph.const(_BLOCKED, name="blocked")
    if mode in {"lut_int32", "lut_fp16"}:
        # Under the declared {0,1} domain, Gather exactly selects the original
        # fill value. INT32 -65504 is exactly representable in FP16.
        dtype = "int32" if mode == "lut_int32" else "float16"
        table = graph.const([int(_BLOCKED), 0], dtype, name="binary_mask_lut")
        key_mask = graph.op("gather", table, mask, shape=(1, 8), dtype=dtype,
                            attrs={"axis": 0}, name="mask_values")
        if dtype == "int32":
            key_mask = graph.op("cast", key_mask, shape=key_mask.shape,
                                dtype="float16", name="float_mask_values")
        key_mask = _view(graph, key_mask, (1, 1, 1, 8), "key_mask")
        result = graph.op("where", causal, key_mask, blocked, shape=_MASK_SHAPE, name="result")
    else:
        if mode not in {"and_select", "reshape_before_cast", "sequential_select"}:
            raise ValueError("unknown mask probe mode")
        if mode == "reshape_before_cast":
            mask = _view(graph, mask, (1, 1, 1, 8), "integer_mask_view")
        valid = graph.op("cast", mask, shape=mask.shape, dtype="bool", name="valid")
        if mode != "reshape_before_cast":
            valid = _view(graph, valid, (1, 1, 1, 8), "valid_view")
        zero = graph.const(0, name="allowed")
        if mode == "sequential_select":
            key_mask = graph.op("where", valid, zero, blocked, shape=(1, 1, 1, 8), name="key_mask")
            result = graph.op("where", causal, key_mask, blocked, shape=_MASK_SHAPE, name="result")
        else:
            visible = graph.op("logical_and", causal, valid, shape=_MASK_SHAPE,
                               dtype="bool", name="visible")
            result = graph.op("where", visible, zero, blocked, shape=_MASK_SHAPE, name="result")
    graph.outputs = [result]
    return graph


def position_cases():
    """Yield 12 dynamic-input controls/candidates with unchanged external IO."""
    # Lazy import permits the shared catalogue to compose this generator.
    from .op_probe_cases import _finish

    rope_inputs = {"inverse": np.array([1, 0.01], np.float16).reshape(1, 2, 1),
                   "positions": np.arange(8, dtype=np.float16).reshape(1, 1, 8)}
    rope_reference = _rope_graph("rope_reference", "bmm")
    for suffix, mode in (("bmm", "bmm"), ("matmul2d", "matmul"), ("outer_mul", "outer_mul")):
        graph = _rope_graph(f"rope_frequency_{suffix}", mode)
        yield _finish(graph, {key: value.copy() for key, value in rope_inputs.items()},
                      "RoPE FP16 [1,2,1] x [1,1,8] -> [1,2,8], K=1; both operands dynamic. "
                      f"{suffix} preserves one FP16 product; positions 0..7, tiny inverse frequencies. "
                      "No claim about longer positions, trigonometric kernels or NPU execution.",
                      pair="rope_frequency_k1", reference_graph=rope_reference)

    for fixture, values in (
        ("tiny", np.arange(8, dtype=np.int32)),
        ("rounding", np.array([2047, 2048, 2049, 2051, 4095, 4096, 4097, 65504], np.int32)),
    ):
        reference = _position_graph("position_reference", False)
        for compact in (False, True):
            suffix = "compact" if compact else "native"
            graph = _position_graph(f"position_cast_{fixture}_{suffix}", compact)
            yield _finish(graph, {"positions": values.reshape(1, 8).copy()},
                          "INT32 [1,8] to FP16 [1,8]; "
                          + ("reshape to rank1 before Cast, then restore. " if compact else "native rank2 Cast. ")
                          + ("Actual tiny positions 0..7." if fixture == "tiny" else
                             "Rounding-boundary probe: 2049/2051/4095/4097 lose integer precision as in the original Cast; no truncation or large-ID exactness claim."),
                          exact=True, pair=f"position_cast_{fixture}", reference_graph=reference)

    reference = _mask_graph("mask_reference", "and_select")
    mask_input = np.array([[1, 0, 1, 1, 0, 1, 0, 1]], np.int32)
    for mode in ("and_select", "reshape_before_cast", "sequential_select", "lut_int32", "lut_fp16"):
        graph = _mask_graph(f"attention_mask_{mode}", mode)
        case = _finish(graph, {"attention_mask": mask_input.copy()},
                       "Dynamic binary INT32 attention_mask [1,8], static lower-triangular causal "
                       "[1,1,8,8], FP16 output Select(valid, +0, -65504). "
                       f"Variant {mode}; "
                       + ("two-entry Gather lookup is valid ONLY for input values {0,1}; "
                          "it does not replace arbitrary nonzero-to-true semantics. " if mode.startswith("lut_") else "")
                       + "Retain output selection semantics and original finite fill; no attention score or softmax rewrite.",
                       exact=True, pair="attention_mask_binary", reference_graph=reference)
        case["input_domain"] = {"attention_mask": {"allowed_values": [0, 1]}}
        yield case

"""Small deterministic operator contracts for AIR/OMC/device probing.

Each case establishes only its own shape, dtype, attributes and placement. A
passing case is not a claim of universal support for that operator, the full
model, or every device. External tensors use FP16 or INT32; booleans stay inside
constructed graphs. References use the existing host semantic oracle, including
its FP16 rounding boundaries, and are not device execution results.
"""
from __future__ import annotations

import math

import numpy as np

from .ge_eval import evaluate
from .ge_ir import Graph


# Exact operations copy/reorder values or use integer arithmetic. Other cases
# allow ordinary FP16 implementation differences, without changing graph dtype.
FP16_TOLERANCE = {"rtol": 0.01, "atol": 0.002}
EXACT_TOLERANCE = {"rtol": 0.0, "atol": 0.0}


def _fp(shape, offset=0):
    i = np.arange(math.prod(shape), dtype=np.int32)
    # Dyadic, mixed-sign, nonconstant values, independent of NumPy RNG versions.
    return (((i * 7 + offset * 3) % 37 - 18) / 16).astype(np.float16).reshape(shape)


def _finish(graph, inputs, description, *, exact=False, pair=None):
    graph.validate()
    expected = [np.ascontiguousarray(a) for a in evaluate(graph, inputs)]
    if any(not np.all(np.isfinite(a)) for a in expected):
        raise ValueError(f"{graph.name}: probe reference must be finite")
    return {"name": graph.name, "graph": graph, "inputs": inputs,
            "expected": expected, "description": description,
            "tolerances": [dict(EXACT_TOLERANCE if exact or a.dtype.kind in "biu"
                                else FP16_TOLERANCE) for a in expected],
            "equivalence_group": pair}


def _dense_cases():
    for m, n in [(m, 16) for m in (1, 2, 8, 16)] + [(8, n) for n in (1, 2, 8)]:
        for constant in (True, False):
            dims = f"m{m}" if n == 16 else f"m{m}_n{n}"
            name = f"matmul_{dims}_b_{'const' if constant else 'dynamic'}"
            g = Graph(name)
            x = g.input("x", (m, 16))
            rhs = _fp((n, 16), 5)
            inputs = {"x": _fp(x.shape, 2)}
            if constant:
                b = g.const(rhs, name="b")
            else:
                b = g.input("b", rhs.shape)
                inputs["b"] = rhs
            g.outputs = [g.op("matmul", x, b, shape=(m, n),
                              attrs={"transpose_b": True}, name="result")]
            yield _finish(g, inputs, f"FP16 dense projection [{m},16] × [{n},16]^T; "
                          f"right operand {'constant' if constant else 'dynamic'}.")


def _bmm_cases():
    for name, batch, transpose_a, transpose_b in (
        ("bmm_rank3", (2,), False, False),
        ("bmm_rank3_left_transpose", (2,), True, False),
        ("bmm_rank5_normalized", (1, 2, 1), False, True),
    ):
        g = Graph(name)
        x = g.input("x", batch + ((16, 8) if transpose_a else (8, 16)))
        b = g.input("b", batch + ((8, 16) if transpose_b else (16, 8)))
        g.outputs = [g.op("matmul", x, b, shape=batch + (8, 8),
                          attrs={"transpose_a": transpose_a, "transpose_b": transpose_b},
                          name="result")]
        yield _finish(g, {"x": _fp(x.shape, 1), "b": _fp(b.shape, 4)},
                      "FP16 equal-batch BMM; production serializer normalizes high ranks "
                      "to rank3 and makes a left transpose explicit when requested.")
    g = Graph("bmm_conv_batch48_m8_k4_n1")
    x, b = g.input("x", (48, 8, 4)), g.input("b", (48, 4, 1))
    g.outputs = [g.op("matmul", x, b, shape=(48, 8, 1), name="result")]
    yield _finish(g, {"x": _fp(x.shape, 1), "b": _fp(b.shape, 4)},
                  "FP16 convolution-like BMM with batch48, M8, K4, N1; both operands dynamic.")


def _rank_cases():
    # Same flattened values across every rank5/rank4 pair; no arithmetic rewrite.
    for kind in ("neg", "add", "mul", "where", "slice"):
        label = "select" if kind == "where" else kind
        for rank, shape in ((5, (1, 2, 1, 8, 16)), (4, (1, 2, 8, 16))):
            g = Graph(f"{label}_rank{rank}")
            x = g.input("x", shape)
            inputs = {"x": _fp(shape, 1)}
            if kind in {"add", "mul", "where"}:
                y = g.input("y", shape)
                inputs["y"] = _fp(shape, 5)
            if kind == "where":
                mask = g.input("mask", shape, "int32")
                inputs["mask"] = (np.arange(math.prod(shape)) % 3 != 1).astype(np.int32).reshape(shape)
                cond = g.op("cast", mask, shape=shape, dtype="bool", name="condition")
                result = g.op("where", cond, x, y, shape=shape, name="result")
            elif kind == "slice":
                starts = [0] * rank
                starts[-1] = 3
                sizes = list(shape)
                sizes[-1] = 9
                result = g.op("slice", x, shape=tuple(sizes),
                              attrs={"starts": starts, "sizes": sizes}, name="result")
            elif kind in {"add", "mul"}:
                result = g.op(kind, x, y, shape=shape, name="result")
            else:
                result = g.op(kind, x, shape=shape, name="result")
            g.outputs = [result]
            yield _finish(g, inputs, f"FP16 {label} at rank{rank}; compare with its "
                          "paired rank after flattening only the singleton dimension.",
                          exact=kind in {"neg", "where", "slice"}, pair=f"{label}_rank_pair")


def _other_cases():
    for kind in ("silu", "softplus"):
        g = Graph(kind)
        x = g.input("x", (2, 16))
        g.outputs = [g.op(kind, x, shape=x.shape, name="result")]
        data = np.linspace(-8, 8, math.prod(x.shape), dtype=np.float16).reshape(x.shape)
        yield _finish(g, {"x": data}, f"Native FP16 {kind}; mixed positive and negative input.")

    for source, dest in (("float16", "int32"), ("int32", "float16")):
        g = Graph(f"cast_{source}_to_{dest}")
        x = g.input("x", (2, 8), source)
        g.outputs = [g.op("cast", x, shape=x.shape, dtype=dest, name="result")]
        data = np.arange(-8, 8, dtype=np.int32).reshape(x.shape)
        data = (data + 0.375).astype(np.float16) if source == "float16" else data
        yield _finish(g, {"x": data}, f"Cast {source} to {dest}, including negative values.", exact=True)

    for dtype in ("int32", "float16"):
        g = Graph(f"gather_{dtype}")
        x = g.input("x", (6, 4), dtype)
        ids = g.input("indices", (3,), "int32")
        g.outputs = [g.op("gather", x, ids, shape=(3, 4), dtype=dtype,
                          attrs={"axis": 0}, name="result")]
        data = ((np.arange(24, dtype=np.int32) * 41 + 16777217).reshape(x.shape)
                if dtype == "int32" else _fp(x.shape, 2))
        yield _finish(g, {"x": data, "indices": np.array([4, 0, 3], dtype=np.int32)},
                      f"GatherV2 with {dtype} data and nonmonotonic INT32 indices. "
                      + ("Integer payload exceeds exact FP32 integer range."
                         if dtype == "int32" else "Preserves selected FP16 payloads."), exact=True)

    g = Graph("gather_elements_float16")
    x = g.input("x", (2, 8))
    ids = g.input("indices", (2, 3), "int32")
    g.outputs = [g.op("gather_elements", x, ids, shape=ids.shape,
                      attrs={"axis": 1}, name="result")]
    yield _finish(g, {"x": _fp(x.shape, 2), "indices": np.array([[7, 1, 4], [0, 6, 2]], np.int32)},
                  "FP16 GatherElements; row-specific INT32 indices.", exact=True)

    for kind in ("reduce_mean", "reduce_sum"):
        g = Graph(kind)
        x = g.input("x", (1, 2, 8, 16))
        g.outputs = [g.op(kind, x, shape=(1, 2, 8, 1),
                          attrs={"axes": [-1], "keepdims": True}, name="result")]
        yield _finish(g, {"x": _fp(x.shape, 3)}, f"FP16 {kind} over final axis with keepdims.")

    g = Graph("reshape")
    x = g.input("x", (2, 3, 8))
    g.outputs = [g.op("reshape", x, shape=(6, 8), attrs={"shape": [6, 8]}, name="result")]
    yield _finish(g, {"x": _fp(x.shape, 3)}, "Static FP16 reshape with INT32 serialized shape.", exact=True)

    g = Graph("transpose")
    x = g.input("x", (1, 2, 4, 8))
    g.outputs = [g.op("transpose", x, shape=(1, 4, 2, 8),
                      attrs={"perm": [0, 2, 1, 3]}, name="result")]
    yield _finish(g, {"x": _fp(x.shape, 2)}, "FP16 rank4 head/sequence transpose.", exact=True)

    g = Graph("concat")
    x, y = g.input("x", (1, 2, 3, 8)), g.input("y", (1, 2, 2, 8))
    g.outputs = [g.op("concat", x, y, shape=(1, 2, 5, 8), attrs={"axis": 2}, name="result")]
    yield _finish(g, {"x": _fp(x.shape, 1), "y": _fp(y.shape, 6)},
                  "FP16 concat along a nonfinal axis with unequal input extents.", exact=True)

    g = Graph("softmax")
    x = g.input("x", (1, 2, 4, 8))
    g.outputs = [g.op("softmax", x, shape=x.shape, attrs={"axis": -1}, name="result")]
    yield _finish(g, {"x": _fp(x.shape, 1)}, "FP16 rank4 last-axis SoftmaxV2.")

    # Unique scores avoid unspecified TopK tie-breaking obscuring numerical checks.
    scores = np.array([[1, -2, 7, 0, 4, 3, -1, 5], [6, 0, -3, 4, 2, 9, 1, -2]], np.float16)
    g = Graph("argmax")
    x = g.input("x", scores.shape)
    g.outputs = [g.op("argmax", x, shape=(2, 1), dtype="int32",
                      attrs={"axis": -1, "keepdims": True}, name="result")]
    yield _finish(g, {"x": scores.copy()}, "FP16 ArgMaxV2 with INT32 result and keepdims.", exact=True)

    g = Graph("topk")
    x = g.input("x", scores.shape)
    values = g.op("topk_values", x, shape=(2, 3), attrs={"k": 3}, name="values")
    indices = g.op("topk_indices", x, shape=(2, 3), dtype="int32", attrs={"k": 3}, name="indices")
    g.outputs = [values, indices]
    yield _finish(g, {"x": scores.copy()}, "Shared TopKV2; validates ordered FP16 values and INT32 indices.", exact=True)

    for kind in ("add", "mul"):
        g = Graph(f"{kind}_broadcast")
        x, y = g.input("x", (1, 2, 8, 16)), g.input("y", (16,))
        g.outputs = [g.op(kind, x, y, shape=x.shape, name="result")]
        yield _finish(g, {"x": _fp(x.shape, 2), "y": _fp(y.shape, 6)},
                      f"FP16 {kind} with vector-to-rank4 broadcasting.")

    g = Graph("cumsum_int32")
    x = g.input("x", (1, 8), "int32")
    g.outputs = [g.op("cumsum", x, shape=x.shape, dtype="int32", attrs={"axis": 1}, name="result")]
    yield _finish(g, {"x": np.array([[1, 0, 1, 1, 0, 1, 0, 1]], dtype=np.int32)},
                  "INT32 prefix sum through the production static scan lowering.", exact=True)

    g = Graph("power_scale_shift")
    x = g.input("x", (1, 2, 8, 16))
    scaled = g.op("scalar_mul", x, shape=x.shape, attrs={"value": 0.3535533905932738}, name="scaled")
    g.outputs = [g.op("scalar_add", scaled, shape=x.shape, attrs={"value": 0.125}, name="result")]
    yield _finish(g, {"x": _fp(x.shape, 3)}, "Two FP16 Power boundaries for scale then shift.")


def _reduction_layout_cases():
    # Bounded controls for the current tiny model's CPU-only reductions. Each
    # pair keeps FP16 payloads and reduction order, varying only singleton
    # layout or keepdims followed by an exact reshape.
    for kind, label, shapes, group in (
        ("reduce_mean", "rms", ((1, 8, 16), (8, 16)), "reduce_rms_layout_pair"),
        ("reduce_sum", "scan", ((2, 1, 64), (2, 64)), "reduce_scan_layout_pair"),
    ):
        for shape in shapes:
            g = Graph(f"{kind}_{label}_rank{len(shape)}")
            x = g.input("x", shape)
            g.outputs = [g.op(kind, x, shape=shape[:-1] + (1,),
                              attrs={"axes": [len(shape) - 1], "keepdims": True}, name="result")]
            data = _fp(shape, 3)
            if label == "scan":
                data = -np.abs(data)  # Match the GDN nonpositive decay domain.
            yield _finish(g, {"x": data},
                          f"FP16 {label} {kind} over the final axis of {list(shape)}; "
                          "keepdims, INT32 constant axes; paired singleton-layout control.", pair=group)

    for keepdims, label in ((False, "dropdims"), (True, "keepdims")):
        g = Graph(f"reduce_sum_gdn_{label}")
        x = g.input("x", (2, 1, 64, 64))
        result = g.op("reduce_sum", x, shape=(2, 1, 1, 64) if keepdims else (2, 1, 64),
                      attrs={"axes": [2], "keepdims": keepdims},
                      name="reduced" if keepdims else "result")
        if keepdims:
            result = g.op("reshape", result, shape=(2, 1, 64),
                          attrs={"shape": [2, 1, 64]}, name="result")
        g.outputs = [result]
        yield _finish(g, {"x": _fp(x.shape, 3)},
                      "FP16 GDN correction ReduceSum [2,1,64,64] over axis 2; "
                      + ("keepdims then exact reshape to [2,1,64]." if keepdims
                         else "drop reduced dimension to [2,1,64]."), pair="reduce_gdn_keepdims_pair")


def cases():
    """Yield fresh case dictionaries; callers may filter by stable case name."""
    yield from _dense_cases()
    yield from _bmm_cases()
    yield from _rank_cases()
    yield from _other_cases()
    yield from _reduction_layout_cases()

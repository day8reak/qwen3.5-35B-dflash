"""FP16 contraction plan shared by probes and optional GE emission lowering.

The real-number reduction is unchanged and there is one FP16 output boundary.
MatMul accumulation order/internal precision can differ from ReduceSum/Mean;
host agreement does not establish device numerical agreement or NPU placement.
"""
from __future__ import annotations

import math

import numpy as np

from qwen35_moe_dflash.graph.ir import Graph, Value


def reduction_matmul_plan(source: Value, *, kind: str, axis: int, keepdims: bool) -> dict:
    """Validate and describe the same single-axis contraction for both callers."""
    if source.dtype != "float16":
        raise ValueError("MatMul reduction candidates require FP16 input; integer reductions stay integer")
    if kind not in {"reduce_sum", "reduce_mean"}:
        raise ValueError("MatMul reduction kind must be reduce_sum or reduce_mean")
    rank = len(source.shape)
    if not rank or any(d <= 0 for d in source.shape):
        raise ValueError("MatMul reduction requires positive static dimensions")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or not -rank <= axis < rank:
        raise ValueError("MatMul reduction requires one in-range integer axis")
    if not isinstance(keepdims, bool):
        raise ValueError("MatMul reduction keepdims must be boolean")
    axis = int(axis) % rank
    length = source.shape[axis]
    coefficient = 1.0
    if kind == "reduce_mean":
        coefficient = 1.0 / length
        if length & (length - 1) or float(np.float16(coefficient)) != coefficient:
            raise ValueError("MatMul mean requires a power-of-two axis with an exact nonzero FP16 reciprocal")

    remaining = [i for i in range(rank) if i != axis]
    shape = list(source.shape)
    if keepdims:
        shape[axis] = 1
    else:
        shape.pop(axis)
    return {"axis": axis, "K": length, "M": math.prod(source.shape[i] for i in remaining),
            "coefficient": coefficient, "permutation": remaining + [axis], "output_shape": tuple(shape)}


def reduce_via_matmul(graph: Graph, source: Value, *, kind: str, axis: int,
                      keepdims: bool, name: str = "result") -> Value:
    """Contract one axis, preserving the existing FP16 producer boundary.

    Move only the reduced axis, flatten independent rows, and restore their
    order and requested shape. Even K=1 emits a real MatMul.
    """
    if not isinstance(graph, Graph) or not isinstance(source, Value) or graph._values.get(source.name) is not source:
        raise ValueError("reduction source must belong to the supplied graph")
    plan = reduction_matmul_plan(source, kind=kind, axis=axis, keepdims=keepdims)
    perm, rows, length = plan["permutation"], plan["M"], plan["K"]
    value = source
    if perm != list(range(len(source.shape))):
        shape = tuple(source.shape[i] for i in perm)
        value = graph.op("transpose", value, shape=shape, attrs={"perm": perm})
    value = graph.op("reshape", value, shape=(rows, length), attrs={"shape": [rows, length]})
    # Match the tested contraction probes' right-transpose convention. Their
    # device evidence applies only to the tested shapes and runtime.
    weight = graph.const(np.full((1, length), plan["coefficient"], dtype=np.float16))
    value = graph.op("matmul", value, weight, shape=(rows, 1),
                     attrs={"transpose_a": False, "transpose_b": True})
    shape = list(plan["output_shape"])
    return graph.op("reshape", value, shape=tuple(shape), attrs={"shape": shape}, name=name)

"""Isolated FP16 contraction candidates for operator probes, not production.

The real-number reduction is unchanged and there is one FP16 output boundary.
MatMul accumulation order/internal precision can differ from ReduceSum/Mean;
host agreement does not establish device numerical agreement or NPU placement.
"""
from __future__ import annotations

import math

import numpy as np

from .ge_ir import Graph, Value


def reduce_via_matmul(graph: Graph, source: Value, *, kind: str, axis: int,
                      keepdims: bool, name: str = "result") -> Value:
    """Contract one static axis with FP16 ones (or exact mean coefficients).

    Consume the existing FP16 source, including any preceding rounded product;
    do not fuse that producer. Move only the reduced axis to the end, flatten
    independent rows, and restore their original order and requested shape.
    Even K=1 emits MatMul so a probe actually exercises that kernel contract.
    """
    if not isinstance(graph, Graph) or not isinstance(source, Value) or graph._values.get(source.name) is not source:
        raise ValueError("reduction source must belong to the supplied graph")
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
    perm = remaining + [axis]
    value = source
    if perm != list(range(rank)):
        shape = tuple(source.shape[i] for i in perm)
        value = graph.op("transpose", value, shape=shape, attrs={"perm": perm})
    rows = math.prod(source.shape[i] for i in remaining)
    value = graph.op("reshape", value, shape=(rows, length), attrs={"shape": [rows, length]})
    # Match the existing dense projection probe's right-transpose convention;
    # N=1 is still its own unverified device contract.
    weight = graph.const(np.full((1, length), coefficient, dtype=np.float16))
    value = graph.op("matmul", value, weight, shape=(rows, 1),
                     attrs={"transpose_a": False, "transpose_b": True})
    shape = list(source.shape)
    if keepdims:
        shape[axis] = 1
    else:
        shape.pop(axis)
    return graph.op("reshape", value, shape=tuple(shape), attrs={"shape": shape}, name=name)

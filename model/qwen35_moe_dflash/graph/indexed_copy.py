"""Bounded Gather copies shared by probes and the opt-in target GDN policy.

Masking selects existing FP16 elements or one positive-zero sentinel. Outer
subtraction copies each operand to the broadcast result shape before one FP16
Sub. Neither construction introduces arithmetic into its copy/mask steps.
The target policy separately restricts integration to previously tested shapes;
compiler support and device execution remain specific to the compiled graph.
"""
from __future__ import annotations

import math
from numbers import Integral
import re

import numpy as np

from qwen35_moe_dflash.graph.ir import Graph, Value


MAX_OUTPUT_ELEMENTS = 1 << 20
_INT32_MAX = (1 << 31) - 1


def _inputs(graph, *values):
    if not isinstance(graph, Graph) or any(
            not isinstance(value, Value) or graph._values.get(value.name) is not value
            for value in values):
        raise ValueError("indexed-copy inputs must belong to the supplied graph")
    for value in values:
        if value.dtype != "float16":
            raise ValueError("indexed-copy probes require FP16 inputs without casts")
        if len(value.shape) != 4:
            raise ValueError("indexed-copy probes require rank-four inputs")
        if any(isinstance(dim, bool) or not isinstance(dim, Integral)
               or not 0 < dim <= _INT32_MAX for dim in value.shape):
            raise ValueError("indexed-copy dimensions must be positive static INT32 values")


def _size(shape):
    size = math.prod(shape)
    if not 0 < size <= MAX_OUTPUT_ELEMENTS:
        raise ValueError(f"indexed-copy output is limited to {MAX_OUTPUT_ELEMENTS} elements")
    return size


def _names(graph, name, suffixes):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", name):
        raise ValueError("invalid indexed-copy output name")
    result = {suffix: f"__indexed_copy/{name}/{suffix}" for suffix in suffixes}
    names = [*result.values(), name]
    if len(set(names)) != len(names) or any(item in graph._values for item in names):
        raise ValueError("indexed-copy output or intermediate name already exists")
    return result


def _view(graph, value, shape, name):
    return graph.op("reshape", value, shape=shape, dtype="float16",
                    attrs={"shape": list(shape)}, name=name)


def triangular_mask_gather(graph: Graph, x: Value, *, strict: bool = False,
                           name: str = "result") -> Value:
    """Copy a lower triangle of [B,C,N,N], replacing other cells by +0.

    All batches/channels retain their own flattened offsets. Strict mode also
    replaces the diagonal. Exceptional values in retained cells are never
    multiplied by a mask or passed through floating-point arithmetic.
    """
    _inputs(graph, x)
    if not isinstance(strict, bool):
        raise ValueError("triangular mask strict must be boolean")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("triangular mask requires [B,C,N,N] square matrices")
    size = _size(x.shape)
    if size + 1 > _INT32_MAX:
        raise ValueError("triangular sentinel index/shape must fit INT32")
    names = _names(graph, name, ("flat", "zero", "with_zero", "indices", "gather"))
    order = np.arange(size, dtype=np.int32)
    width = x.shape[-1]
    local = order % (width * width)
    rows, columns = local // width, local % width
    keep = rows > columns if strict else rows >= columns
    indices = np.where(keep, order, size).astype(np.int32, copy=False)

    flat = _view(graph, x, (size,), names["flat"])
    zero = graph.const(np.zeros(1, dtype=np.float16), name=names["zero"])
    padded = graph.op("concat", flat, zero, shape=(size + 1,), dtype="float16",
                      attrs={"axis": 0}, name=names["with_zero"])
    index = graph.const(indices, dtype="int32", name=names["indices"])
    gathered = graph.op("gather", padded, index, shape=(size,), dtype="float16",
                        attrs={"axis": 0}, name=names["gather"])
    return _view(graph, gathered, x.shape, name)


def outer_sub_gather(graph: Graph, left: Value, right: Value, name: str = "result") -> Value:
    """Copy [B,1,M,1] and [B,1,1,N] to [B,1,M,N], then subtract once.

    Each Gather index refers only to the matching input batch. The original
    FP16 arithmetic boundary and operand order are retained.
    """
    _inputs(graph, left, right)
    batch, singleton, rows, tail = left.shape
    if singleton != 1 or tail != 1 or right.shape[:3] != (batch, 1, 1):
        raise ValueError("outer subtraction requires [B,1,M,1] minus [B,1,1,N]")
    width = right.shape[-1]
    shape = (batch, 1, rows, width)
    size = _size(shape)
    left_size, right_size = math.prod(left.shape), math.prod(right.shape)
    if max(left_size, right_size, size) > _INT32_MAX:
        raise ValueError("outer subtraction flattened dimensions/indices must fit INT32")
    names = _names(graph, name, ("left_flat", "right_flat", "left_indices", "right_indices",
                                "left_gather", "right_gather", "left_full", "right_full"))
    order = np.arange(size, dtype=np.int32)
    left_indices = (order // width).astype(np.int32, copy=False)
    right_indices = ((order // (rows * width)) * width + order % width).astype(np.int32, copy=False)

    expanded = []
    for side, source, count, indices in (("left", left, left_size, left_indices),
                                        ("right", right, right_size, right_indices)):
        flat = _view(graph, source, (count,), names[side + "_flat"])
        index = graph.const(indices, dtype="int32", name=names[side + "_indices"])
        gathered = graph.op("gather", flat, index, shape=(size,), dtype="float16",
                            attrs={"axis": 0}, name=names[side + "_gather"])
        expanded.append(_view(graph, gathered, shape, names[side + "_full"]))
    return graph.op("sub", *expanded, shape=shape, dtype="float16", name=name)

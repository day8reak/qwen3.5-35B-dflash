"""Isolated FP16 binary-layout candidates for probes, not model policy.

Only carrier shapes change. Each candidate preserves operand order and the
single FP16 arithmetic boundary; host equivalence does not prove NPU placement
or device handling of subnormals, signed zeros, or nonfinite values.
"""
from __future__ import annotations

import math
from numbers import Integral
import re

from .ge_ir import Graph, Value


def _inputs(graph, left, right):
    if not isinstance(graph, Graph) or any(
            not isinstance(value, Value) or graph._values.get(value.name) is not value
            for value in (left, right)):
        raise ValueError("binary layout inputs must belong to the supplied graph")
    if left.dtype != "float16" or right.dtype != "float16":
        raise ValueError("binary layout probes require FP16 inputs without casts")
    _shapes(left.shape, right.shape)


def _shapes(*shapes):
    for shape in shapes:
        if any(isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0 for dim in shape):
            raise ValueError("binary layout probes require positive static dimensions")
        if any(dim > (1 << 31) - 1 for dim in shape):
            raise ValueError("binary layout Reshape dimensions must fit INT32")


def _names(graph, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", name):
        raise ValueError("invalid binary layout result name")
    names = (f"__binary_layout/{name}/left", f"__binary_layout/{name}/right",
             f"__binary_layout/{name}/arithmetic", name)
    if any(candidate in graph._values for candidate in names):
        raise ValueError("binary layout result or intermediate name already exists")
    return names


def _emit(graph, left, right, *, left_shape, right_shape, arithmetic_shape, output_shape, kind, name):
    _shapes(left_shape, right_shape, arithmetic_shape, output_shape)
    left_name, right_name, arithmetic_name, output_name = _names(graph, name)

    def view(value, shape, output):
        return graph.op("reshape", value, shape=shape, dtype="float16",
                        attrs={"shape": list(shape)}, name=output)

    lhs = view(left, left_shape, left_name)
    rhs = view(right, right_shape, right_name)
    value = graph.op(kind, lhs, rhs, shape=arithmetic_shape, dtype="float16", name=arithmetic_name)
    return view(value, output_shape, output_name)


def row_scale_mul(graph: Graph, left: Value, right: Value, name: str = "result") -> Value:
    """Flatten equal prefixes: (..., N) * (..., 1) -> (M, N) * (M, 1).

    This intentionally excludes other broadcast patterns. Reshape preserves
    the original prefix order; no token/head permutation or expanded weights
    are introduced. Restore the original left/output shape after one Mul.
    """
    _inputs(graph, left, right)
    if len(left.shape) < 2 or len(left.shape) != len(right.shape):
        raise ValueError("row scale requires equal input ranks of at least two")
    if left.shape[:-1] != right.shape[:-1] or right.shape[-1] != 1:
        raise ValueError("row scale requires identical prefixes and right last dimension one")
    rows, width = math.prod(left.shape[:-1]), left.shape[-1]
    return _emit(graph, left, right, left_shape=(rows, width), right_shape=(rows, 1),
                 arithmetic_shape=(rows, width), output_shape=left.shape, kind="mul", name=name)


def outer_sub_compact(graph: Graph, left: Value, right: Value, name: str = "result") -> Value:
    """Remove only the shared singleton in [B,1,M,1] - [B,1,1,N].

    The arithmetic remains left minus right, with one FP16 Sub on
    [B,M,1] - [B,1,N], then restoration to [B,1,M,N].
    """
    _inputs(graph, left, right)
    if len(left.shape) != 4 or len(right.shape) != 4:
        raise ValueError("outer subtraction requires rank-four inputs")
    batch, singleton, rows, tail = left.shape
    if singleton != 1 or tail != 1 or right.shape[:3] != (batch, 1, 1):
        raise ValueError("outer subtraction requires [B,1,M,1] minus [B,1,1,N]")
    width = right.shape[-1]
    return _emit(graph, left, right, left_shape=(batch, rows, 1), right_shape=(batch, 1, width),
                 arithmetic_shape=(batch, rows, width), output_shape=(batch, 1, rows, width),
                 kind="sub", name=name)

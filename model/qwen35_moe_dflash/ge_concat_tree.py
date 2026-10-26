"""Ordered, bounded-fan-in concatenation candidate for host graphs and probes.

Only copy operations are introduced; input order, dtype, and element bits are
unchanged. This construction does not establish a device fan-in limit or NPU
placement, and is not automatically applied to production graphs.
"""
from __future__ import annotations

from numbers import Integral
import re

from .ge_ir import Graph, Value


def concat_tree(graph: Graph, values, axis: int, max_fan_in: int = 8,
                name: str = "result") -> Value:
    """Concatenate adjacent groups at each level, passing singleton groups on.

    Concatenation dimensions may differ; every other dimension must match.
    A single input is returned unchanged without adding a naming/copy node.
    """
    if not isinstance(graph, Graph):
        raise ValueError("concat inputs must belong to the supplied graph")
    values = list(values)
    if not values:
        raise ValueError("concat tree requires at least one input")
    if any(not isinstance(value, Value) or graph._values.get(value.name) is not value
           for value in values):
        raise ValueError("concat inputs must belong to the supplied graph")
    if isinstance(max_fan_in, bool) or not isinstance(max_fan_in, Integral) or max_fan_in < 2:
        raise ValueError("concat max_fan_in must be an integer of at least two")
    max_fan_in = int(max_fan_in)
    rank, dtype = len(values[0].shape), values[0].dtype
    if not rank or any(any(dim <= 0 for dim in value.shape) for value in values):
        raise ValueError("concat tree requires positive static dimensions and nonzero rank")
    if isinstance(axis, bool) or not isinstance(axis, Integral) or not -rank <= axis < rank:
        raise ValueError("concat axis must be an in-range integer")
    axis = int(axis) % rank
    for value in values:
        if value.dtype != dtype:
            raise ValueError("concat input dtypes must match without casts")
        if len(value.shape) != rank or any(value.shape[i] != values[0].shape[i]
                                          for i in range(rank) if i != axis):
            raise ValueError("concat input ranks and non-concatenated dimensions must match")
    if len(values) == 1:
        return values[0]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", name):
        raise ValueError("invalid concat result name")

    # Plan names before adding nodes, so even a later-level collision leaves
    # the caller's graph intact. Generated names do not consume v0/v1 names.
    names = []
    count, level = len(values), 0
    while count > 1:
        final_level = count <= max_fan_in
        for start in range(0, count, max_fan_in):
            if min(max_fan_in, count - start) > 1:
                names.append(name if final_level else
                             f"__concat/{name}/l{level}_g{start // max_fan_in}")
        count = (count + max_fan_in - 1) // max_fan_in
        level += 1
    if len(set(names)) != len(names) or any(candidate in graph._values for candidate in names):
        raise ValueError("concat result or intermediate name already exists")

    names = iter(names)
    while len(values) > 1:
        following = []
        for start in range(0, len(values), max_fan_in):
            group = values[start:start + max_fan_in]
            if len(group) == 1:
                following.append(group[0])
                continue
            shape = list(group[0].shape)
            shape[axis] = sum(value.shape[axis] for value in group)
            following.append(graph.op("concat", *group, shape=shape, dtype=dtype,
                                      attrs={"axis": axis}, name=next(names)))
        values = following
    return values[0]

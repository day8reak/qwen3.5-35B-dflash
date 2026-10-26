"""Bounded tiny-MoE routing probes; these do not replace production routing.

The contracts use eight tokens, two INT32 expert IDs and four experts. Lookup
tables are exact only on the declared integer domains. Finite board fixtures
include selected zero weights; they do not establish nonfinite device behavior,
TopK tie handling, or NPU execution. Expert arithmetic is supplied as an input so
these cases isolate routing from expert projections.
"""
from __future__ import annotations

import numpy as np

from .ge_ir import Graph


ROUTING_CASE_NAMES = (
    "moe_equal_cast_int32", "moe_cast_bool_roundtrip", "moe_count_reduce_int32",
    "moe_active_greater_int32", "moe_membership_native", "moe_membership_gather",
    "moe_count_native", "moe_count_gather", "moe_active_native", "moe_active_gather",
    "moe_weights_native", "moe_weights_gather", "moe_gate_select", "moe_gate_gather",
    "moe_route_native", "moe_route_gather",
)


def _ids():
    return np.array([[0, 1], [1, 3], [2, 0], [3, 2],
                     [0, 2], [1, 0], [2, 3], [3, 1]], dtype=np.int32)


def _weights():
    # Every expert occurs with positive and with selected zero weights. Each
    # row sums to one; no positivity shortcut may infer the membership mask.
    return np.array([[1, 0], [0, 1], [1, 0], [0, 1],
                     [1, 0], [.75, .25], [.5, .5], [.25, .75]], dtype=np.float16)


def _reshape(g, x, shape):
    return g.op("reshape", x, shape=shape, dtype=x.dtype, attrs={"shape": list(shape)})


def _gather(g, table, indices):
    return g.op("gather", table, indices, shape=indices.shape + table.shape[1:],
                dtype=table.dtype, attrs={"axis": 0})


def _cast(g, x, dtype):
    return g.op("cast", x, shape=x.shape, dtype=dtype)


def _sum(g, x):
    return g.op("reduce_sum", x, shape=(8, 1), dtype=x.dtype,
                attrs={"axes": [-1], "keepdims": True})


def _member(g, ids, expert, lookup=False):
    if lookup:
        table = g.const(np.arange(4) == expert, "int32")
        return _gather(g, table, ids)
    equal = g.op("eq", ids, g.const(expert, "int32"), shape=(8, 2), dtype="bool")
    return _cast(g, equal, "int32")


def _pair_code(g, ids):
    columns = [g.op("slice", ids, shape=(8, 1), dtype="int32",
                    attrs={"starts": [0, j], "sizes": [8, 1]}) for j in (0, 1)]
    high = g.op("mul", columns[0], g.const(4, "int32"), shape=(8, 1), dtype="int32")
    return g.op("add", high, columns[1], shape=(8, 1), dtype="int32")


def _count_or_active(g, ids, expert, *, active=False, lookup=False, bool_output=False):
    if lookup:
        counts = np.array([int(a == expert) + int(b == expert)
                           for a in range(4) for b in range(4)], dtype=np.int32)
        table = g.const(counts > 0 if active else counts, "int32")
        return _gather(g, table, _pair_code(g, ids))
    count = _sum(g, _member(g, ids, expert))
    if active:
        nonzero = g.op("gt", count, g.const(0, "int32"), shape=(8, 1), dtype="bool")
        return nonzero if bool_output else _cast(g, nonzero, "int32")
    return count


def _with_zero(g, x):
    flat = _reshape(g, x, (int(np.prod(x.shape)),))
    return g.op("concat", flat, g.const([0], "float16"), shape=(flat.shape[0] + 1,),
                attrs={"axis": 0})


def _routed_weights(g, ids, weights, expert, lookup=False):
    if not lookup:
        selected = g.op("eq", ids, g.const(expert, "int32"), shape=(8, 2), dtype="bool")
        return g.op("where", selected, weights, g.const(0), shape=(8, 2))
    # For slot p, encode p * 4 + id, then gather either p or the +0 sentinel.
    # No floating-point mask multiplication: unselected NaN/Inf is not touched.
    position = np.arange(16, dtype=np.int32)
    code = g.op("add", ids, g.const((position * 4).reshape(8, 2), "int32"),
                shape=(8, 2), dtype="int32")
    table = g.const(np.where(np.arange(4)[None, :] == expert,
                            position[:, None], 16).reshape(-1), "int32")
    return _gather(g, _with_zero(g, weights), _gather(g, table, code))


def _gate(g, active, values, lookup=False):
    if not lookup:
        condition = active if active.dtype == "bool" else _cast(g, active, "bool")
        return g.op("where", condition, values, g.const(0), shape=(8, 16))
    # All index arithmetic is INT32. Repeat each token's 0/1 flag by copying,
    # then select source p or sentinel 128. This also retains FP16 NaN payloads
    # and signed zero in the host copy oracle, unlike multiplying by a mask.
    position = np.arange(128, dtype=np.int32)
    flags = _gather(g, _reshape(g, active, (8,)),
                    g.const(position // 16, "int32"))
    code = g.op("add", flags, g.const(position * 2, "int32"), shape=(128,), dtype="int32")
    table = g.const(np.stack([np.full(128, 128, dtype=np.int32), position], axis=1).ravel(), "int32")
    selected = _gather(g, _with_zero(g, values), _gather(g, table, code))
    return _reshape(g, selected, (8, 16))


def _routing_graph(name, stage, lookup=False, *, matmul_scale=False):
    g = Graph(name)
    ids = g.input("ids", (8, 2), "int32")
    if stage in {"weights", "route"}:
        weights = g.input("weights", (8, 2))
    outputs = []
    result = g.const(np.zeros((8, 16), np.float16)) if stage == "route" else None
    for expert in range(4):
        if stage == "membership":
            value = _member(g, ids, expert, lookup)
        elif stage in {"count", "active"}:
            value = _count_or_active(g, ids, expert, active=stage == "active", lookup=lookup)
        else:
            value = _routed_weights(g, ids, weights, expert, lookup)
        if stage == "route":
            if matmul_scale:
                from .ge_reduction_matmul import reduce_via_matmul
                scale = reduce_via_matmul(g, value, kind="reduce_sum", axis=1,
                                          keepdims=True, name=f"scale_{expert}")
            else:
                scale = _sum(g, value)
            expert_values = g.input(f"expert_{expert}", (8, 16))
            contribution = g.op("mul", expert_values, scale, shape=(8, 16))
            active = _count_or_active(g, ids, expert, active=True, lookup=lookup,
                                      bool_output=not lookup)
            contribution = _gate(g, active, contribution, lookup)
            result = g.op("add", result, contribution, shape=(8, 16))
        else:
            outputs.append(value)
    g.outputs = [result] if stage == "route" else outputs
    return g


def routing_cases():
    # Lazy imports permit registration in op_probe_cases without an import cycle.
    from .op_probe_cases import _finish, _fp

    def finish(g, inputs, description, *, exact=True, pair=None, reference_graph=None):
        case = _finish(g, inputs, description, exact=exact, pair=pair,
                       reference_graph=reference_graph)
        case["input_domain"] = {
            name: {"allowed_values": ([0, 1, 2, 3] if name == "ids" else
                                      [0, 1, 2] if name == "counts" else [0, 1])}
            for name in inputs if name in {"ids", "membership", "active", "counts"}
        }
        return case

    g = Graph(ROUTING_CASE_NAMES[0])
    ids = g.input("ids", (8, 2), "int32")
    g.outputs = [_member(g, ids, 2)]
    yield finish(g, {"ids": _ids()}, "Equal against expert 2, followed by BOOL-to-INT32 Cast.")

    g = Graph(ROUTING_CASE_NAMES[1])
    flags = g.input("membership", (8, 2), "int32")
    g.outputs = [_cast(g, _cast(g, flags, "bool"), "int32")]
    yield finish(g, {"membership": (_ids() == 2).astype(np.int32)},
                 "Isolated 0/1 INT32 to BOOL to INT32 Cast contract; BOOL remains internal.")

    g = Graph(ROUTING_CASE_NAMES[2])
    flags = g.input("membership", (8, 2), "int32")
    g.outputs = [_sum(g, flags)]
    yield finish(g, {"membership": np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 2, np.int32)},
                 "INT32 membership ReduceSum over two slots, retaining counts 0, 1 and 2.")

    g = Graph(ROUTING_CASE_NAMES[3])
    counts = g.input("counts", (8, 1), "int32")
    greater = g.op("gt", counts, g.const(0, "int32"), shape=(8, 1), dtype="bool")
    g.outputs = [_cast(g, greater, "int32")]
    yield finish(g, {"counts": np.array([0, 1, 2, 0, 2, 1, 0, 1], np.int32).reshape(8, 1)},
                 "INT32 membership count Greater-than-zero with BOOL-to-INT32 output.")

    for stage in ("membership", "count", "active", "weights"):
        for lookup in (False, True):
            name = f"moe_{stage}_{'gather' if lookup else 'native'}"
            g = _routing_graph(name, stage, lookup)
            inputs = {"ids": _ids()}
            if stage == "weights":
                inputs["weights"] = _weights()
            reference = _routing_graph(name + "_reference", stage)
            yield finish(g, inputs, f"All four experts' {stage} for INT32 IDs [8,2]; "
                         + ("exact bounded INT32 lookup/Gather candidate." if lookup else
                            "original Equal/Cast/reduction/selection control."),
                         pair=f"moe_{stage}", reference_graph=reference)

    for lookup in (False, True):
        name = f"moe_gate_{'gather' if lookup else 'select'}"
        g, reference = Graph(name), Graph(name + "_reference")
        for graph, use_lookup in ((g, lookup), (reference, False)):
            active = graph.input("active", (8, 1), "int32")
            values = graph.input("values", (8, 16))
            graph.outputs = [_gate(graph, active, values, use_lookup)]
        yield finish(g, {"active": np.any(_ids() == 2, axis=1, keepdims=True).astype(np.int32),
                         "values": _fp((8, 16), 4)},
                     "Membership-based contribution gate [8,1] to [8,16]; "
                     + ("INT32 indexed-copy candidate with +0 sentinel." if lookup else
                        "BOOL BroadcastTo and Select control."),
                     pair="moe_gate", reference_graph=reference)

    for lookup in (False, True):
        name = f"moe_route_{'gather' if lookup else 'native'}"
        g = _routing_graph(name, "route", lookup, matmul_scale=True)
        inputs = {"ids": _ids(), "weights": _weights(),
                  **{f"expert_{e}": _fp((8, 16), e + 1) for e in range(4)}}
        yield finish(g, inputs, "Four-expert routing with precomputed FP16 expert values; "
                     "Both arms use a FP16 MatMul for the two-slot weight sum and retain "
                     "the product and ordered expert additions; reference uses native ReduceSum. "
                     "Membership, including selected zero weights, controls the final gate. "
                     + ("Lookup/indexed-copy candidate." if lookup else "Native routing control."),
                     exact=False, pair="moe_route",
                     reference_graph=_routing_graph(name + "_reference", "route"))

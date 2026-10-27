"""Source-bound native reduction candidates and exact FP16 boundary controls.

Only diagnostic models are created. Native ReduceSum preserves the source
primitive and FP16 boundaries; it is not a declaration of device support or a
fix for a separately observed BatchMatMul output loss.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from .ge_ir import Graph
from .ge_reduction_matmul import reduce_via_matmul
from .ge_target import _Builder
from .op_probe_gdn_comprehensive import _input, _make, _normal_rows


_BOUNDARY_BITS = (0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x8002,
                  0x0200, 0x8200, 0x03FF, 0x83FF, 0x0400, 0x8400,
                  0x0401, 0x8401, 0x0800, 0x8800)


def _record(array):
    raw = np.ascontiguousarray(array, dtype="<f2").tobytes()
    return {"shape": list(array.shape), "dtype": "float16", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _source_inputs(originals, provenance):
    if ([item.get("name") for item in originals] != ["gdn_layer0_core_stages", "gdn_core_replay"]
            or provenance.get("policies") != {"fp16_reductions": "matmul", "fp16_row_mul": "compact",
                                              "gdn_indexed_copy": "gather"}):
        raise ValueError("precision controls require the validated original GDN core pair and policies")
    full, replay = originals
    stages = dict(zip(full["output_labels"], full["expected"]))
    actual = {}
    for name in ("query", "key"):
        array = replay["inputs"].get(name)
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype("float16")
                or array.shape != (1, 8, 2, 8) or not np.isfinite(array).all()):
            raise ValueError(f"invalid frozen precision input: {name}")
        bound = provenance.get("cut_inputs", {}).get(name, {})
        if any(bound.get(key) != value for key, value in _record(array).items()):
            raise ValueError(f"frozen precision input hash or descriptor changed: {name}")
        source = stages.get(name + "_input")
        if (not isinstance(source, np.ndarray) or source.dtype != array.dtype or source.shape != array.shape
                or source.tobytes() != np.ascontiguousarray(array).tobytes()):
            raise ValueError(f"frozen precision input differs from the weighted first layer: {name}")
        actual[name] = np.ascontiguousarray(array).copy()
    return actual


def _l2(builder, x):
    square = builder.op("mul", x, x)
    total = builder.reduce("reduce_sum", square, -1)
    epsilon = builder.op("scalar_add", total, value=1e-6)
    gain = builder.op("rsqrt", epsilon)
    result = builder.op("mul", x, gain)
    return {"square": square, "sum": total, "epsilon": epsilon, "gain": gain, "result": result}


def _sum_input(builder, x):
    # Observing the input through Neg avoids a direct Data graph output.
    negative = builder.op("neg", x)
    epsilon = builder.op("scalar_add", x, value=1e-6)
    return {"sum_neg": negative, "epsilon": epsilon, "gain": builder.op("rsqrt", epsilon)}


def _boundaries():
    return np.array(_BOUNDARY_BITS, dtype="<u2").view("<f2")


def _cancellation_rows(values):
    """Normal operands whose exact real sum is each requested FP16 value."""
    tiny = float(np.finfo(np.float16).tiny)
    rows = np.zeros((16, 8), np.float16)
    for index, value in enumerate(values):
        magnitude = abs(float(value))
        if 0 < magnitude < tiny:
            sign = -1 if value < 0 else 1
            rows[index, :2] = [sign * (tiny + magnitude), -sign * tiny]
        else:
            # Three terms cover nextafter(tiny,+inf), whose difference from
            # two larger normal values is awkward at their coarser spacing.
            rows[index, :3] = [value, tiny, -tiny]
    return rows.reshape(1, 8, 2, 8)


def _bmm_boundaries(length):
    tiny, unit = float(np.finfo(np.float16).tiny), float(np.nextafter(np.float16(0), np.float16(1)))
    if length == 1:
        left = np.array([1, -1, .5, -.5, 2, -2, 2**-10, -(2**-10)], np.float16)[:, None]
        right = np.array([tiny, tiny + unit, 2*tiny, 4*tiny, 8*tiny, 16*tiny, 1, .5], np.float16)[None, :]
    else:
        scale = np.array([1, -1, .5, -.5, 2, -2, .25, -.25], np.float16)
        left = np.stack([scale, -scale], axis=-1)
        pairs = [(tiny + unit, tiny), (tiny + 2*unit, tiny), (1.5*tiny, tiny),
                 (2*tiny - unit, tiny), (2*tiny, tiny),
                 (2*tiny + 2*unit, tiny + unit), (tiny, tiny), (3*tiny, tiny)]
        right = np.asarray(pairs, np.float16).T
    # Both heads differ; every operand is normal and every product is dyadic.
    return (np.stack([left, -left[::-1]]).reshape(2, 1, 8, length),
            np.stack([right, np.roll(right, 3, axis=-1)]).reshape(2, 1, length, 8))


def build_cases(original_core_cases, provenance):
    """Return eleven bounded probes using an already validated source pair.

    No prepared payload is regenerated or reread. Q/K bytes must still match
    the cut-input hashes and original weighted layer outputs. The caller owns
    the complete prepared/reference/OMC provenance validation.
    """
    actual = _source_inputs(original_core_cases, provenance)
    policies = provenance["policies"]
    cases, stage_maps, native_names, source_sums = [], {}, {}, {}

    def add(graph, feeds, outputs, description, *, native=(), exact=False, fixture="actual"):
        item, stages = _make(graph, feeds, outputs, description, policies,
                             exact=exact, fixture=fixture)
        item["probe_suite"] = "gdn-repair"
        if native:
            item["emit_options"]["fp16_native_reductions"] = list(native)
        cases.append(item)
        stage_maps[item["name"]] = stages
        native_names[item["name"]] = list(native)
        return item

    for kind in ("query", "key"):
        for fixture in ("actual", "normal"):
            graph, feeds = Graph(f"gp_l2_{kind}_{fixture}_native"), {}
            builder = _Builder(graph)
            array = actual[kind] if fixture == "actual" else _normal_rows(kind == "key")
            outputs = _l2(builder, _input(graph, feeds, "x", array))
            dense = _l2(builder, _input(graph, feeds, "dense_x", array.reshape(16, 8)))
            outputs.update(("dense_" + label, value) for label, value in dense.items())
            item = add(graph, feeds, outputs,
                       "Original FP16 square/sum/epsilon/Rsqrt/product boundaries, with only the two "
                       "rank4/rank2 sums retained as native ReduceSum. Paired with gc_l2 MatMul baseline; "
                       "normal control uses exactly the same frozen control values.",
                       native=(outputs["sum"].name, dense["sum"].name), fixture=fixture)
            if fixture == "actual":
                source_sums[kind] = item["expected"][item["output_labels"].index("sum")].copy()

    for kind in ("query", "key"):
        graph, feeds = Graph("gp_sum_input_" + kind), {}
        builder = _Builder(graph)
        array = source_sums[kind]
        outputs = _sum_input(builder, _input(graph, feeds, "source_sum", array))
        outputs.update(("dense_" + label, value) for label, value in
                       _sum_input(builder, _input(graph, feeds, "dense_source_sum", array.reshape(16, 1))).items())
        add(graph, feeds, outputs,
            "Correct same-weight host FP16 sum supplied independently to Neg, +1e-6 and Rsqrt; "
            "distinguishes reduction output loss from downstream scalar/Rsqrt behavior. No sum is recomputed on device.",
            exact={"sum_neg", "dense_sum_neg"})

    boundary = _boundaries()
    for length in (1, 8):
        graph, feeds = Graph(f"gp_reduce_boundary_k{length}"), {}
        builder = _Builder(graph)
        array = boundary.reshape(1, 8, 2, 1) if length == 1 else _cancellation_rows(boundary)
        x = _input(graph, feeds, "x", array)
        native = builder.reduce("reduce_sum", x, -1)
        matmul = reduce_via_matmul(graph, x, kind="reduce_sum", axis=-1, keepdims=True, name="matmul_sum")
        add(graph, feeds, {"input_neg": builder.op("neg", x),
                           "native_sum_neg": builder.op("neg", native),
                           "matmul_sum_neg": builder.op("neg", matmul)},
            "Exact signed FP16 boundary sums from identical runtime operands: native ReduceSum versus an explicit "
            "MatMul contraction. K1 includes subnormal inputs; K8 uses only normal or zero operands and exact "
            "cancellation to produce the boundary outputs. Sums are observed through Neg so K1 reduction "
            "elimination does not create a direct Data output; K1 cannot prove native reduction kernel execution. "
            "This is a new zero-tolerance control contract.",
            native=(native.name,), exact=True, fixture="signed_boundary")

    graph, feeds = Graph("gp_scalar_boundary"), {}
    builder = _Builder(graph)
    x = _input(graph, feeds, "signed_boundary", boundary.reshape(1, 8, 2, 1))
    unit = float(np.nextafter(np.float16(0), np.float16(1)))
    positive = np.maximum(np.abs(boundary), np.float16(unit)).reshape(1, 8, 2, 1)
    positive_x = _input(graph, feeds, "positive_boundary", positive)
    epsilon = builder.op("scalar_add", positive_x, value=1e-6)
    add(graph, feeds, {"neg": builder.op("neg", x),
                      "plus_min_subnormal": builder.op("scalar_add", x, value=unit),
                      "minus_min_subnormal": builder.op("scalar_add", x, value=-unit),
                      "positive_epsilon": epsilon, "positive_gain": builder.op("rsqrt", epsilon)},
        "Signed zero/subnormal/minimum-normal boundary Neg and exact dyadic scalar additions; positive boundary "
        "+1e-6/Rsqrt keeps the original arithmetic tolerance and has no negative/zero Rsqrt input. "
        "These inputs are explicit controls, not captured original model tensors.",
        exact={"neg", "plus_min_subnormal", "minus_min_subnormal"}, fixture="signed_boundary")

    for length in (1, 2):
        graph, feeds = Graph(f"gp_bmm_boundary_k{length}"), {}
        builder = _Builder(graph)
        left, right = _bmm_boundaries(length)
        lhs, rhs = _input(graph, feeds, "left", left), _input(graph, feeds, "right", right)
        result = builder.op("matmul", lhs, rhs, shape=(2, 1, 8, 8), transpose_a=False, transpose_b=False)
        add(graph, feeds, {"left_neg": builder.op("neg", lhs), "right_neg": builder.op("neg", rhs), "product": result},
            "Exact dyadic BMM boundary control with all-normal runtime operands, both signs and distinct heads. "
            "K1 produces subnormals by multiplication; K2 cancels dyadic products, including both normal "
            "and subnormal product magnitudes across its rows. "
            "This does not duplicate the separately retained same-weight final BMM and is a new zero-tolerance contract.",
            exact=True, fixture="normal_operand_boundary")

    metadata = {"schema_version": 1, "case_names": [item["name"] for item in cases],
                "source_bindings": copy.deepcopy(provenance["bindings"]),
                "source_cut_inputs": {kind: copy.deepcopy(provenance["cut_inputs"][kind]) for kind in actual},
                "source_sum_inputs": {kind: {**_record(array), "derivation":
                    "same-weight FP16 square -> ReduceSum last axis, keepdims -> FP16; no epsilon included"}
                    for kind, array in source_sums.items()},
                "stage_maps": stage_maps, "native_reduction_names": native_names,
                "signed_boundary_fp16_bits": list(_BOUNDARY_BITS),
                "fixture_inputs": {item["name"]: {name: _record(array) for name, array in item["inputs"].items()}
                                   for item in cases},
                "limitations": ["Native ReduceSum candidates have no changed dtype, epsilon, output rounding or tolerance.",
                    "Boundary controls use a new exact contract; actual L2/scalar/Rsqrt retain their original tolerances.",
                    "The host oracle accumulates reductions/MatMul in FP32 and rounds at each original FP16 output boundary; "
                    "this does not add FP32 graph operations or establish device accumulation behavior.",
                    "No FP32 carrier, Add-unrolled reduction, production graph rewrite or accuracy-gate relaxation is included.",
                    "Native norm support and partitioning require device validation; this does not repair a final BMM output loss."]}
    return cases, metadata

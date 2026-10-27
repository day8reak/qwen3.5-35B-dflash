"""Independent FP16 contracts for diagnostic native reduction candidates."""
import copy
from fractions import Fraction
import hashlib
import json
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.op_probe_cases import EXACT_TOLERANCE, FP16_TOLERANCE
from qwen35_moe_dflash.op_probe_gdn_accuracy import cases_from_prepared
from qwen35_moe_dflash.op_probe_gdn_precision import build_cases
from test_ge_select_lowering import RecipeReader
from test_op_probe_gdn_accuracy import case, prepared_source, _same_half


def _source(case):
    return cases_from_prepared(case["prepared"], case["reference_path"], case["omc"], core_stages=True)


def _suite(case):
    original, provenance = _source(case)
    items, metadata = build_cases(original, provenance)
    return {item["name"]: item for item in items}, metadata, original, provenance


def _outputs(item):
    return dict(zip(item["output_labels"], item["expected"]))


def _l2(x):
    square = (x.astype(np.float32) ** 2).astype(np.float16)
    total = square.astype(np.float32).sum(axis=-1, keepdims=True).astype(np.float16)
    epsilon = (total.astype(np.float32) + np.float32(1e-6)).astype(np.float16)
    gain = (1 / np.sqrt(epsilon.astype(np.float32))).astype(np.float16)
    result = (x.astype(np.float32) * gain.astype(np.float32)).astype(np.float16)
    return dict(square=square, sum=total, epsilon=epsilon, gain=gain, result=result)


def test_precision_case_metadata_and_inputs_bind_existing_source_without_mutation(case, monkeypatch):
    original, provenance = _source(case)
    before = pickle.dumps((original, provenance))
    def forbidden(*args, **kwargs):
        raise AssertionError("precision factory must consume supplied arrays, not regenerate weights")
    monkeypatch.setattr("qwen35_moe_dflash.ge_weights.tiny_weights", forbidden)
    items, metadata = build_cases(original, provenance)
    assert len(items) == 11
    assert len(set(metadata["case_names"])) == 11
    assert metadata["case_names"] == [item["name"] for item in items]
    assert metadata["source_bindings"] == provenance["bindings"]
    assert pickle.dumps((original, provenance)) == before
    json.dumps(metadata, allow_nan=False)
    for item in items:
        assert item["probe_suite"] == "gdn-repair" and item["probe_family"] == "gdn"
        assert item["weights"] == {}
        assert item["emit_options"]["fp16_reductions"] == "matmul"
        assert item["emit_options"]["fp16_row_mul"] == "compact"
        assert len(item["graph"].inputs) + len(item["graph"].outputs) <= 32
        assert all(v.dtype == "float16" and len(v.shape) <= 4 and np.prod(v.shape)*2 <= 32768
                   for v in item["graph"].inputs + item["graph"].outputs)
        runtime = {v.name for v in item["graph"].inputs}
        for node in item["graph"].nodes:
            if any(v.name in runtime for v in node.inputs):
                runtime.add(node.output.name)
        direct = {v.name for v in item["graph"].inputs}
        assert all(v.name in runtime and v.name not in direct for v in item["graph"].outputs)
        for name, array in item["inputs"].items():
            assert metadata["fixture_inputs"][item["name"]][name] == {
                "shape": list(array.shape), "dtype": "float16", "bytes": array.nbytes,
                "sha256": hashlib.sha256(array.astype("<f2").tobytes()).hexdigest()}


@pytest.mark.parametrize("kind", ["query", "key"])
@pytest.mark.parametrize("fixture", ["actual", "normal"])
def test_native_l2_retains_all_original_rounding_boundaries_and_dense_independent_input(case, kind, fixture):
    batch, metadata, original, _ = _suite(case)
    item = batch[f"gp_l2_{kind}_{fixture}_native"]
    x = item["inputs"]["x"]
    _same_half(item["inputs"]["dense_x"], x.reshape(16, 8))
    if fixture == "actual":
        _same_half(x, original[1]["inputs"][kind])
    else:
        assert np.all(np.abs(x).astype(np.float32)**2 >= np.finfo(np.float16).tiny)
        assert np.unique(np.sum(x.astype(np.float32)**2, axis=-1)).size >= 4
    values, expected = _outputs(item), _l2(x)
    for label, array in expected.items():
        _same_half(values[label], array)
        _same_half(values["dense_" + label], array.reshape(16, -1))
    if fixture == "actual":
        total = expected["sum"]
        assert np.count_nonzero((total > 0) & (total < np.finfo(np.float16).tiny)) == (2 if kind == "query" else 1)
    assert all(t == FP16_TOLERANCE for t in item["tolerances"])
    overrides = item["emit_options"]["fp16_native_reductions"]
    assert len(overrides) == 2
    assert overrides == metadata["native_reduction_names"][item["name"]]
    assert {n.output.name for n in item["graph"].nodes if n.kind == "reduce_sum"} == set(overrides)
    # The rank2 control really consumes its separate runtime input.
    nodes = {n.output.name: n for n in item["graph"].nodes}
    def leaves(v):
        return {v.name} if v.name not in nodes else set().union(*(leaves(x) for x in nodes[v.name].inputs))
    assert leaves(item["graph"].outputs[-1]) == {"dense_x"}


@pytest.mark.parametrize("kind", ["query", "key"])
def test_frozen_sum_inputs_bypass_device_reduction_without_epsilon_or_extra_rounding(case, kind):
    batch, metadata, original, _ = _suite(case)
    item = batch["gp_sum_input_" + kind]
    expected = _l2(original[1]["inputs"][kind])
    _same_half(item["inputs"]["source_sum"], expected["sum"])
    _same_half(item["inputs"]["dense_source_sum"], expected["sum"].reshape(16, 1))
    values = _outputs(item)
    for prefix in ("", "dense_"):
        shape = (16, 1) if prefix else (1, 8, 2, 1)
        _same_half(values[prefix + "sum_neg"], -expected["sum"].reshape(shape))
        _same_half(values[prefix + "epsilon"], expected["epsilon"].reshape(shape))
        _same_half(values[prefix + "gain"], expected["gain"].reshape(shape))
    assert {n.kind for n in item["graph"].nodes} == {"neg", "scalar_add", "rsqrt"}
    assert "fp16_native_reductions" not in item["emit_options"]
    for label, tolerance in zip(item["output_labels"], item["tolerances"]):
        assert tolerance == (EXACT_TOLERANCE if label.endswith("sum_neg") else FP16_TOLERANCE)
    assert metadata["source_sum_inputs"][kind]["sha256"] == hashlib.sha256(expected["sum"].astype("<f2").tobytes()).hexdigest()


@pytest.mark.parametrize("length", [1, 8])
def test_signed_reduction_boundaries_separate_input_ftz_from_normal_operand_cancellation(case, length):
    batch, metadata, _, _ = _suite(case)
    item = batch[f"gp_reduce_boundary_k{length}"]
    x = item["inputs"]["x"]
    boundary = np.asarray(metadata["signed_boundary_fp16_bits"], dtype="<u2").view("<f2")
    assert set(boundary.view(np.uint16)) >= {0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x0401, 0x8401}
    # FP16 values are exact integers in units of 2**-24 over this domain.
    ticks = (x.astype(np.float64) * 2**24).astype(np.int64)
    summed = (ticks.sum(axis=-1, keepdims=True).astype(np.float64) / 2**24).astype(np.float16)
    np.testing.assert_array_equal(summed.reshape(-1), boundary)
    outputs = _outputs(item)
    np.testing.assert_array_equal(outputs["native_sum_neg"], -summed)
    np.testing.assert_array_equal(outputs["matmul_sum_neg"], -summed)
    _same_half(outputs["input_neg"], -x)
    if length == 8:
        nonzero = np.abs(x[x != 0])
        assert np.all(nonzero >= np.finfo(np.float16).tiny)
        assert np.all(np.count_nonzero(x, axis=-1) >= 2)
    else:
        assert np.any((np.abs(x) > 0) & (np.abs(x) < np.finfo(np.float16).tiny))
    assert all(t == EXACT_TOLERANCE for t in item["tolerances"])
    assert sum(n.kind == "reduce_sum" for n in item["graph"].nodes) == 1
    assert sum(n.kind == "matmul" for n in item["graph"].nodes) == 1
    producers = {n.output.name: n for n in item["graph"].nodes}
    assert all(producers[v.name].kind == "neg" for v in item["graph"].outputs)


def test_boundary_scalar_control_preserves_sign_bits_and_dyadic_additions(case):
    batch, _, _, _ = _suite(case)
    item = batch["gp_scalar_boundary"]
    x = item["inputs"]["signed_boundary"]
    values = _outputs(item)
    np.testing.assert_array_equal(values["neg"].view(np.uint16), x.view(np.uint16) ^ np.uint16(0x8000))
    ticks = (x.astype(np.float64) * 2**24).astype(np.int64)
    for label, delta in (("plus_min_subnormal", 1), ("minus_min_subnormal", -1)):
        expected = ((ticks + delta).astype(np.float64) / 2**24).astype(np.float16)
        np.testing.assert_array_equal(values[label], expected)
    assert np.all(values["positive_epsilon"] > 0)
    assert np.isfinite(values["positive_gain"]).all()
    assert item["tolerances"][:3] == [EXACT_TOLERANCE] * 3
    assert item["tolerances"][3:] == [FP16_TOLERANCE] * 2


@pytest.mark.parametrize("length", [1, 2])
def test_bmm_boundary_controls_use_normal_operands_and_exact_rational_results(case, length):
    batch, _, _, _ = _suite(case)
    item = batch[f"gp_bmm_boundary_k{length}"]
    left, right = item["inputs"]["left"], item["inputs"]["right"]
    assert all(np.all(np.abs(x) >= np.finfo(np.float16).tiny) for x in (left, right))
    assert not np.array_equal(left[0], left[1])
    expected = np.empty((2, 1, 8, 8), np.float16)
    for head in range(2):
        for row in range(8):
            for column in range(8):
                exact = sum(Fraction(float(left[head, 0, row, k])) * Fraction(float(right[head, 0, k, column]))
                            for k in range(length))
                expected[head, 0, row, column] = float(exact)
    result = _outputs(item)["product"]
    _same_half(result, expected)
    assert np.any((np.abs(result) > 0) & (np.abs(result) < np.finfo(np.float16).tiny))
    assert np.any(result >= np.finfo(np.float16).tiny) and np.any(result <= -np.finfo(np.float16).tiny)
    if length == 2:
        assert np.count_nonzero(result == 0) > 0
    assert all(t == EXACT_TOLERANCE for t in item["tolerances"])


def test_real_recipes_keep_selected_native_sums_and_explicit_matmul_boundaries(case, tmp_path):
    batch, metadata, _, _ = _suite(case)
    for name, item in batch.items():
        report, output = {}, tmp_path / name
        emit_cpp(item["graph"], {}, recipe_dir=output, reduction_report=report, **item["emit_options"])
        ops, _, native_outputs = RecipeReader(output / "graph.recipe").read()
        assert len(native_outputs) == len(item["expected"])
        overrides = set(metadata["native_reduction_names"][name])
        retained = {v["name"] for v in report["retained"] if v["reason"] == "explicit_native_override"}
        assert retained == overrides
        for selected in overrides:
            assert next(op for op in ops if op["name"] == selected)["type"] == "ReduceSum"
        if name.startswith("gp_reduce_boundary_"):
            assert any(op["type"] in {"MatMul", "BatchMatMul"} for op in ops)
        assert all(port["dtype"] != "float32" for op in ops for port in op["inputs"] + op["outputs"])


@pytest.mark.parametrize("corruption", ["hash", "input", "weighted_stage", "policy"])
def test_precision_rejects_mismatched_frozen_source_before_constructing_cases(case, corruption):
    original, provenance = _source(case)
    original, provenance = copy.deepcopy(original), copy.deepcopy(provenance)
    if corruption == "hash":
        provenance["cut_inputs"]["query"]["sha256"] = "0" * 64
    elif corruption == "input":
        original[1]["inputs"]["query"][0, 0, 0, 0] *= np.float16(2)
    elif corruption == "weighted_stage":
        index = original[0]["output_labels"].index("query_input")
        original[0]["expected"][index][0, 0, 0, 0] *= np.float16(2)
    else:
        provenance["policies"]["fp16_reductions"] = "native"
    with pytest.raises(ValueError):
        build_cases(original, provenance)

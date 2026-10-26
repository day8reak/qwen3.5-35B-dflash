"""Host contracts for device probes, without pretending to execute a device."""
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.op_probe_cases import cases
from test_ge_select_lowering import RecipeReader


REDUCTION_CASES = (
    "reduce_mean_rms_rank3", "reduce_mean_rms_rank2",
    "reduce_sum_scan_rank3", "reduce_sum_scan_rank2",
    "reduce_sum_gdn_dropdims", "reduce_sum_gdn_keepdims",
)
MATMUL_REDUCTION_CASES = (
    "reduce_mean_rms_matmul", "reduce_mean_head_matmul", "reduce_sum_scan_matmul",
    "reduce_sum_gdn_matmul_k64", "reduce_sum_gdn_matmul_k7", "reduce_sum_gdn_matmul_k1",
)


def test_case_inputs_references_and_serialization_are_complete():
    suite = list(cases())
    assert len(suite) == 59
    assert tuple(c["name"] for c in suite[47:53]) == REDUCTION_CASES
    assert tuple(c["name"] for c in suite[53:]) == MATMUL_REDUCTION_CASES
    assert len({c["name"] for c in suite}) == len(suite)
    for case in suite:
        graph = case["graph"]
        assert not graph.weights
        assert set(case["inputs"]) == {v.name for v in graph.inputs}
        assert len(case["expected"]) == len(graph.outputs) == len(case["tolerances"])
        assert all(v.dtype in {"float16", "int32"} for v in graph.inputs + graph.outputs)
        assert all(np.isfinite(a).all() for a in case["expected"])
        assert all(np.any(a != 0) for a in case["inputs"].values())
        assert "SaveToFile" in emit_cpp(graph)
        for actual, expected in zip(evaluate(graph, case["inputs"]), case["expected"]):
            np.testing.assert_array_equal(actual, expected)


def test_rank_pairs_have_identical_flattened_inputs_and_results():
    groups = {}
    for case in cases():
        if (case["equivalence_group"] or "").endswith("_rank_pair"):
            groups.setdefault(case["equivalence_group"], []).append(case)
    assert len(groups) == 5
    for pair in groups.values():
        assert len(pair) == 2
        high, flat = pair
        assert len(high["graph"].inputs[0].shape) == 5
        assert len(flat["graph"].inputs[0].shape) == 4
        for name in high["inputs"]:
            np.testing.assert_array_equal(high["inputs"][name].ravel(), flat["inputs"][name].ravel())
        np.testing.assert_array_equal(high["expected"][0].ravel(), flat["expected"][0].ravel())
        assert high["expected"][0].size > 1
        assert np.unique(high["expected"][0]).size > 2


def test_dense_const_and_dynamic_preserve_values_and_topk_port_order():
    suite = {case["name"]: case for case in cases()}
    for dims in [f"m{m}" for m in (1, 2, 8, 16)] + [f"m8_n{n}" for n in (1, 2, 8)]:
        const = suite[f"matmul_{dims}_b_const"]
        dynamic = suite[f"matmul_{dims}_b_dynamic"]
        np.testing.assert_array_equal(const["expected"][0], dynamic["expected"][0])
        assert '"MatMul"' in emit_cpp(const["graph"])
    topk = suite["topk"]
    assert topk["expected"][0].dtype == np.float16
    assert topk["expected"][1].dtype == np.int32
    np.testing.assert_array_equal(topk["expected"][1], [[2, 7, 4], [5, 0, 3]])
    np.testing.assert_array_equal(topk["expected"][0], [[7, 5, 4], [9, 6, 4]])
    gather = suite["gather_int32"]
    assert np.all(gather["expected"][0] > 2 ** 24)
    assert all(t == {"rtol": 0.0, "atol": 0.0} for t in gather["tolerances"])


def test_probe_values_are_deterministic_and_bmm_uses_normalized_native_kind():
    left, right = list(cases()), list(cases())
    for a, b in zip(left, right):
        assert a["name"] == b["name"]
        for name in a["inputs"]:
            np.testing.assert_array_equal(a["inputs"][name], b["inputs"][name])
    bmm = next(case for case in left if case["name"] == "bmm_rank5_normalized")
    code = emit_cpp(bmm["graph"])
    assert '"BatchMatMulV2"' in code
    assert '"__ge/result/batch_left"' in code
    assert '"__ge/result/batch_right"' in code
    conv = next(case for case in left if case["name"] == "bmm_conv_batch48_m8_k4_n1")
    assert conv["inputs"]["x"].shape == (48, 8, 4)
    assert conv["inputs"]["b"].shape == (48, 4, 1)
    assert conv["expected"][0].shape == (48, 8, 1)
    np.testing.assert_array_equal(conv["expected"][0], np.matmul(conv["inputs"]["x"], conv["inputs"]["b"]))


@pytest.mark.parametrize("names,shapes,axis", [
    (REDUCTION_CASES[:2], ((1, 8, 16), (8, 16)), -1),
    (REDUCTION_CASES[2:4], ((2, 1, 64), (2, 64)), -1),
    (REDUCTION_CASES[4:], ((2, 1, 64, 64), (2, 1, 64, 64)), 2),
])
def test_reduction_layout_controls_keep_identical_fp16_payloads_and_results(names, shapes, axis):
    suite = {case["name"]: case for case in cases()}
    left, right = (suite[name] for name in names)
    assert left["equivalence_group"] == right["equivalence_group"]
    for case, shape in zip((left, right), shapes):
        x = case["inputs"]["x"]
        assert x.shape == shape and x.dtype == np.float16
        assert case["tolerances"] == [{"rtol": 0.01, "atol": 0.002}]
        expected_shape = list(shape)
        expected_shape[axis] = 1
        expected_shape = (2, 1, 64) if "gdn" in case["name"] else tuple(expected_shape)
        assert case["expected"][0].shape == expected_shape
        operation = np.mean if case["name"].startswith("reduce_mean") else np.sum
        expected = operation(x.astype(np.float32), axis=axis, keepdims=True).astype(np.float16)
        np.testing.assert_array_equal(case["expected"][0].ravel().view(np.uint16),
                                      expected.ravel().view(np.uint16))
    np.testing.assert_array_equal(left["inputs"]["x"].ravel().view(np.uint16),
                                  right["inputs"]["x"].ravel().view(np.uint16))
    np.testing.assert_array_equal(left["expected"][0].ravel().view(np.uint16),
                                  right["expected"][0].ravel().view(np.uint16))
    if "scan" in left["name"]:
        assert np.all(left["inputs"]["x"] <= 0)


@pytest.mark.parametrize("name", REDUCTION_CASES)
def test_reduction_layout_controls_serialize_fp16_data_and_int32_axes(name, tmp_path):
    case = next(case for case in cases() if case["name"] == name)
    emit_cpp(case["graph"], recipe_dir=tmp_path / "recipe")
    records, _, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    reduction, = [r for r in records if r["type"] in {"ReduceSum", "ReduceMean"}]
    x, axes = reduction["inputs"]
    assert x["dtype"] == reduction["outputs"][0]["dtype"] == "float16"
    assert x["shape"] == case["inputs"]["x"].shape
    assert axes["port"] == "axes" and axes["dtype"] == "int32" and axes["shape"] == (1,)
    constant = records[axes["src"]]
    assert constant["type"] == "Const" and constant["outputs"][0]["dtype"] == "int32"
    path, offset, size = constant["constant"]
    assert size == 4
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(size)
    expected_axis = 2 if "gdn" in name else len(x["shape"]) - 1
    assert raw == np.asarray([expected_axis], dtype="<i4").tobytes()
    assert bool(reduction["attrs"]["keep_dims"]) == (name != "reduce_sum_gdn_dropdims")
    assert not any(r["type"] in {"Cast", "CastT"} for r in records)
    final = records[outputs[0][0]]
    assert final["outputs"][0]["shape"] == case["expected"][0].shape
    if name == "reduce_sum_gdn_keepdims":
        assert final["type"] == "Reshape"
        assert final["inputs"][0]["shape"] == (2, 1, 1, 64)
        assert final["outputs"][0]["shape"] == (2, 1, 64)
    else:
        assert final is reduction


@pytest.mark.parametrize("name", MATMUL_REDUCTION_CASES)
def test_matmul_reduction_candidates_keep_original_reference_and_fp16_thresholds(name, tmp_path):
    case = next(case for case in cases() if case["name"] == name)
    data = case["inputs"]["x"]
    is_mean, is_gdn = name.startswith("reduce_mean"), "gdn" in name
    axis, keepdims = (2, False) if is_gdn else (-1, True)
    reference = (np.mean if is_mean else np.sum)(data.astype(np.float64), axis=axis, keepdims=keepdims).astype(np.float16)
    np.testing.assert_array_equal(case["expected"][0], reference)
    assert case["tolerances"] == [{"rtol": 0.01, "atol": 0.002}]
    if "scan" in name:
        assert np.all(data <= 0)
    emit_cpp(case["graph"], recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert not any(r["type"] in {"ReduceSum", "ReduceMean", "Cast", "CastT", "Mul", "Power"} for r in records)
    matmul, = [r for r in records if r["type"] == "MatMul"]
    assert all(port["dtype"] == "float16" for port in matmul["inputs"] + matmul["outputs"])
    length = data.shape[axis]
    assert matmul["inputs"][0]["shape"] == (data.size // length, length)
    assert matmul["inputs"][1]["shape"] == (1, length)
    assert matmul["attrs"]["transpose_x1"] == 0 and matmul["attrs"]["transpose_x2"] == 1
    constant = records[matmul["inputs"][1]["src"]]
    path, offset, size = constant["constant"]
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        weights = np.frombuffer(stream.read(size), dtype="<f2")
    np.testing.assert_array_equal(weights, np.full(length, 1 / length if is_mean else 1, np.float16))

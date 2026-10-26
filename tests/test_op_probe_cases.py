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
FP16_LAYOUT_CASES = (
    "mul_rows16_rank4", "mul_rows16_compact", "mul_rows128_rank4", "mul_rows128_compact",
    "sub_outer_rank4", "sub_outer_compact", "concat64_rank4", "concat64_tree8", "concat64_tree2",
)

GDN_INDEXED_COPY_CASES = (
    "gdn_mask_lower_slice_concat", "gdn_mask_lower_gather",
    "gdn_mask_strict_slice_concat", "gdn_mask_strict_gather",
    "gdn_mask_exp_chain_slice_concat", "gdn_mask_exp_chain_gather", "sub_outer_gather",
)


def test_case_inputs_references_and_serialization_are_complete():
    suite = list(cases())
    assert len(suite) == 75
    assert tuple(c["name"] for c in suite[47:53]) == REDUCTION_CASES
    assert tuple(c["name"] for c in suite[53:59]) == MATMUL_REDUCTION_CASES
    assert tuple(c["name"] for c in suite[59:68]) == FP16_LAYOUT_CASES
    assert tuple(c["name"] for c in suite[68:]) == GDN_INDEXED_COPY_CASES
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


@pytest.mark.parametrize("prefix,compact_shape,kind", [
    ("mul_rows16", (16, 8), "Mul"),
    ("mul_rows128", (128, 8), "Mul"),
    ("sub_outer", (2, 64, 64), "Sub"),
])
def test_binary_layout_probe_pairs_keep_io_and_original_pointwise_reference(prefix, compact_shape, kind, tmp_path):
    suite = {case["name"]: case for case in cases()}
    control, candidate = (suite[prefix + "_" + mode] for mode in ("rank4", "compact"))
    assert control["equivalence_group"] == candidate["equivalence_group"]
    for name, payload in control["inputs"].items():
        assert payload.shape == candidate["inputs"][name].shape
        assert payload.tobytes() == candidate["inputs"][name].tobytes()
    inputs = control["inputs"]
    expected = ((inputs["x"].astype(np.float64) * inputs["y"].astype(np.float64)) if kind == "Mul"
                else (inputs["x"].astype(np.float64) - inputs["y"].astype(np.float64))).astype(np.float16)
    for case in (control, candidate):
        assert case["expected"][0].tobytes() == expected.tobytes()
        assert case["tolerances"] == [{"rtol": 0.01, "atol": 0.002}]
        emit_cpp(case["graph"], recipe_dir=tmp_path / case["name"])
        records, _, outputs = RecipeReader(tmp_path / case["name"] / "graph.recipe").read()
        arithmetic, = [row for row in records if row["type"] == kind]
        assert all(row["type"] in {"Data", "Const", "Reshape", kind} for row in records)
        assert all(port["dtype"] == "float16" for port in arithmetic["inputs"] + arithmetic["outputs"])
        assert arithmetic["outputs"][0]["shape"] == (compact_shape if case is candidate else expected.shape)
        assert records[outputs[0][0]]["outputs"][0]["shape"] == expected.shape


@pytest.mark.parametrize("mode,fan_in,concat_count", [("rank4", 64, 1), ("tree8", 8, 9), ("tree2", 2, 63)])
def test_concat64_probe_preserves_distinct_dynamic_port_order_and_original_reference(mode, fan_in, concat_count, tmp_path):
    case = next(c for c in cases() if c["name"] == "concat64_" + mode)
    assert len(case["graph"].inputs) == 64
    assert not case["graph"].constants
    ordered = [case["inputs"][f"x{i}"] for i in range(64)]
    assert np.unique(np.concatenate([a.ravel().view(np.uint16) for a in ordered])).size == 8192
    expected = np.concatenate(ordered, axis=2)
    assert case["expected"][0].tobytes() == expected.tobytes()
    assert case["tolerances"] == [{"rtol": 0.0, "atol": 0.0}]
    emit_cpp(case["graph"], recipe_dir=tmp_path / "recipe")
    records, inputs, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert [records[i]["name"] for i in inputs] == [f"x{i}" for i in range(64)]
    assert all(row["type"] in {"Data", "ConcatD"} for row in records)
    concats = [row for row in records if row["type"] == "ConcatD"]
    assert len(concats) == concat_count
    assert all(2 <= len(row["inputs"]) <= fan_in for row in concats)
    assert all(row["attrs"]["concat_dim"] == 2 for row in concats)
    assert records[outputs[0][0]]["outputs"][0]["shape"] == (2, 1, 64, 64)


@pytest.mark.parametrize("label,diagonal", [("lower", 0), ("strict", -1)])
def test_gdn_mask_probes_use_shared_matrix_and_original_zero_fill_reference(label, diagonal, tmp_path):
    suite = {case["name"]: case for case in cases()}
    control = suite[f"gdn_mask_{label}_slice_concat"]
    candidate = suite[f"gdn_mask_{label}_gather"]
    mask = np.tril(np.ones((64, 64), dtype=bool), k=diagonal)
    data = control["inputs"]["x"]
    assert np.unique(data.view(np.uint16)).size == 8192 and np.all(data != 0)
    expected = np.where(mask, data, np.float16(0))
    assert candidate["inputs"]["x"].tobytes() == data.tobytes()
    assert control["equivalence_group"] == candidate["equivalence_group"]
    assert control["expected"][0].tobytes() == candidate["expected"][0].tobytes() == expected.tobytes()
    for case in (control, candidate):
        assert len(case["graph"].inputs) == 1
        assert case["tolerances"] == [{"rtol": 0.0, "atol": 0.0}]
        emit_cpp(case["graph"], recipe_dir=tmp_path / case["name"])
        records, inputs, outputs = RecipeReader(tmp_path / case["name"] / "graph.recipe").read()
        assert [records[i]["name"] for i in inputs] == ["x"]
        assert records[outputs[0][0]]["outputs"][0]["shape"] == (2, 1, 64, 64)
        kinds = [row["type"] for row in records]
        if case is control:
            assert "GatherV2" not in kinds and kinds.count("ConcatD") == 64
            outer, = [r for r in records if r["name"] == "result"]
            parents = [records[p["src"]] for p in outer["inputs"]]
            assert [p["type"] for p in parents].count("ConcatD") == 63
            assert parents[0 if diagonal == -1 else -1]["type"] == ("Const" if diagonal == -1 else "Slice")
        else:
            assert kinds.count("GatherV2") == 1 and kinds.count("ConcatD") == 1
            assert set(kinds) <= {"Data", "Const", "Reshape", "ConcatD", "GatherV2"}


def test_gdn_exp_context_keeps_both_masks_and_original_fp16_exp_boundary(tmp_path):
    suite = {case["name"]: case for case in cases()}
    control, candidate = (suite[f"gdn_mask_exp_chain_{mode}"] for mode in ("slice_concat", "gather"))
    mask = np.tril(np.ones((64, 64), dtype=bool))
    data = control["inputs"]["x"]
    expected = np.where(mask, np.exp(np.where(mask, data, np.float16(0)).astype(np.float32)).astype(np.float16), np.float16(0))
    assert candidate["inputs"]["x"].tobytes() == data.tobytes()
    assert control["expected"][0].tobytes() == candidate["expected"][0].tobytes() == expected.tobytes()
    for case in (control, candidate):
        assert case["tolerances"] == [{"rtol": 0.01, "atol": 0.002}]
        kinds = [node.kind for node in case["graph"].nodes]
        assert kinds.count("exp") == 1
        emit_cpp(case["graph"], recipe_dir=tmp_path / case["name"])
        records, _, outputs = RecipeReader(tmp_path / case["name"] / "graph.recipe").read()
        exponential, = [r for r in records if r["type"] == "Exp"]
        assert records[exponential["inputs"][0]["src"]]["name"] == "before_exp"
        assert records[outputs[0][0]]["name"] == "result"
        assert sum(r["type"] == "GatherV2" for r in records) == (2 if case is candidate else 0)


def test_outer_gather_probe_retains_existing_control_input_reference_and_tolerance(tmp_path):
    suite = {case["name"]: case for case in cases()}
    control, candidate = suite["sub_outer_rank4"], suite["sub_outer_gather"]
    assert control["equivalence_group"] == candidate["equivalence_group"]
    for name, data in control["inputs"].items():
        assert data.tobytes() == candidate["inputs"][name].tobytes()
    assert control["expected"][0].tobytes() == candidate["expected"][0].tobytes()
    assert control["tolerances"] == candidate["tolerances"]
    emit_cpp(candidate["graph"], recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert sum(r["type"] == "GatherV2" for r in records) == 2
    sub, = [r for r in records if r["type"] == "Sub"]
    assert [p["shape"] for p in sub["inputs"]] == [(2, 1, 64, 64)] * 2
    assert not any(r["type"] in {"Mul", "Cast", "BroadcastTo", "SelectV2"} for r in records)

"""Explicit residual CPU candidate keeps the source FP16 sum and whole draft."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from qwen35_moe_dflash.draft_compile_precision import collect_precision_plan
from qwen35_moe_dflash.draft_standalone import _graph
from qwen35_moe_dflash.ge_cpp import _Emitter, emit_cpp
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from qwen35_moe_dflash.ge_residual import EXPORT_SUFFIX, emit_residual_add, residual_selection


def _lowered_add(left, right):
    graph = Graph("draft")
    x = graph.input("left", left.shape)
    y = graph.input("right", right.shape)
    graph.outputs = [graph.op("add", x, y, shape=x.shape, name="residual")]
    e = _Emitter(graph)
    for value in graph.inputs:
        e.refs[value.name] = (e.op(value.name, "Data", [], [("y", value)]), "y")
    emit_residual_add(e, graph.nodes[-1], {"layer": 0, "role": "output"})
    assert [r["type"] for r in e.records[2:]] == ["Cast", "Cast", "Add", "Cast"]
    assert e.records[4]["outputs"][0]["dtype"] == "float32"
    assert e.records[5]["outputs"][0]["dtype"] == "float16"
    assert e.records[5]["name"] == "residual"
    arrays = [left, right]
    with np.errstate(all="ignore"):
        for record in e.records[2:]:
            args = [arrays[port["src"]] for port in record["inputs"]]
            dtype = record["outputs"][0]["dtype"]
            result = np.asarray(args[0], dtype=dtype) if record["type"] == "Cast" else np.add(*args)
            arrays.append(result.astype(dtype))
    return arrays[-1]


@pytest.mark.parametrize("anchor", [0.0, -0.0, 2**-24, -(2**-24), 2**-14, -(2**-14),
                                    1.0, -1.0, 32752.0, -32752.0, 65504.0, -65504.0])
def test_all_finite_fp16_values_against_rounding_anchors_match_torch_and_exact_double(anchor):
    bits = np.arange(65536, dtype=np.uint16)
    finite = bits[(bits & 0x7c00) != 0x7c00].view(np.float16)
    other = np.full(finite.shape, anchor, dtype=np.float16)
    actual = _lowered_add(finite, other)
    with np.errstate(all="ignore"):
        # Every sum of two finite binary16 values is exact in binary64, so this
        # tests an independent precision route rather than FP32 against itself.
        exact_double = (finite.astype(np.float64) + other.astype(np.float64)).astype(np.float16)
    torch_half = (torch.from_numpy(finite) + torch.from_numpy(other)).numpy()
    np.testing.assert_array_equal(actual.view(np.uint16), exact_double.view(np.uint16))
    np.testing.assert_array_equal(actual.view(np.uint16), torch_half.view(np.uint16))


def test_random_finite_pairs_cover_mixed_exponents_cancellation_and_midpoints():
    rng = np.random.default_rng(23903)
    bits = rng.integers(0, 65536, size=(2, 400000), dtype=np.uint16)
    bits[(bits & 0x7c00) == 0x7c00] &= np.uint16(0xfbff)
    left, right = bits.view(np.float16)
    right[:1000] = -left[:1000]
    actual = _lowered_add(left, right)
    with np.errstate(all="ignore"):
        expected = (left.astype(np.float64) + right.astype(np.float64)).astype(np.float16)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_signed_zeros_and_overflow_keep_the_original_fp16_bits():
    left = np.array([-0., -0., 0., 65504., -65504., 1., -1.], dtype=np.float16)
    right = np.array([-0., 0., -0., 65504., -65504., -1., 1.], dtype=np.float16)
    actual = _lowered_add(left, right)
    expected = (torch.from_numpy(left) + torch.from_numpy(right)).numpy()
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert np.isposinf(actual[3]) and np.isneginf(actual[4])
    assert actual.view(np.uint16)[0] == 0x8000


def test_full_recipe_keeps_source_graph_values_and_adds_twelve_cpu_requirements(tmp_path):
    graph = _graph(64)[2]
    before = pickle.dumps(graph)
    weights = {key: tmp_path / "unopened" / key for key in graph.weights}
    reports, recipes = [], []
    for policy in ("native", "cpu-fp32-add"):
        report, folder = {}, tmp_path / policy
        emit_cpp(graph, weights, external_weights=True, fold_constant_arithmetic=False, recipe_dir=folder,
                 draft_residual_policy=policy, draft_residual_report=report)
        reports.append(report)
        recipes.append(read_recipe(folder / "graph.recipe"))
    native, candidate = recipes
    assert pickle.dumps(graph) == before
    assert len(reports[1]["replaced"]) == 12 and reports[0]["replaced"] == []
    assert len(candidate["records"]) == len(native["records"]) + 36
    assert [candidate["records"][index]["name"] for index, _ in candidate["outputs"]] == [v.name for v in graph.outputs]
    plan = collect_precision_plan(tmp_path / "cpu-fp32-add/graph.recipe")
    assert len(plan["required_nodes"]) == 165
    for row in reports[1]["replaced"]:
        final = next(r for r in candidate["records"] if r["name"] == row["name"])
        addition = next(r for r in plan["required_nodes"] if r["name"] == row["sum_node"])
        assert final["type"] == "Cast" and final["outputs"][0]["dtype"] == "float16"
        assert addition["type"] == "Add"
        assert [p["dtype"] for p in addition["inputs"] + addition["outputs"]] == ["float32"] * 3
        for side, original in zip(addition["inputs"], row["inputs"]):
            promoted = next(r for r in candidate["records"] if r["name"] == side["producer"])
            assert promoted["type"] == "Cast" and promoted["inputs"][0]["dtype"] == "float16"
            assert candidate["records"][promoted["inputs"][0]["src"]]["name"] == original
    assert not (tmp_path / "unopened").exists()


def test_explicit_native_policy_keeps_emitted_cpp_identical():
    graph = _graph(8, tiny=True)[2]
    weights = {key: Path("/not-opened") / key for key in graph.weights}
    args = dict(external_weights=True, fold_constant_arithmetic=False)
    assert emit_cpp(graph, weights, **args) == emit_cpp(graph, weights, draft_residual_policy="native", **args)


@pytest.mark.parametrize("policy", ["unknown", None, True])
def test_invalid_policy_is_rejected(policy):
    graph = _graph(8, tiny=True)[2]
    with pytest.raises(ValueError, match="residual policy"):
        residual_selection(graph, policy)


def test_export_records_distinct_candidate_contract_and_binds_original_fp16_outputs(tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_standalone
    from test_draft_standalone import fake_ge
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared, output = tmp_path / "prepared", tmp_path / "candidate-air"
    original = draft_standalone.prepare_draft(prepared, 8, tiny=True)
    before = (prepared / "prepared.json").read_bytes()

    def tool(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock exporter")
        else:
            Path(command[1]).write_bytes(b"mock AIR; no device evidence")

    monkeypatch.setattr(draft_standalone, "_logged", tool)
    manifest = draft_standalone.export_draft(prepared, output, fake_ge(tmp_path), residual_policy="cpu-fp32-add")
    assert manifest["export_contract"] == draft_standalone.CURRENT_DRAFT_EXPORT_CONTRACT + EXPORT_SUFFIX
    assert manifest["residual_policy"] == "cpu-fp32-add"
    report = json.loads((output / "draft/residual-lowering.json").read_text())
    assert len(report["replaced"]) == original["draft_layers"] * 2
    assert manifest["graphs"][0]["residual_lowering"] == report
    assert report["compiler_validation"] == report["device_validation"] == "NOT_RUN"
    assert (prepared / "prepared.json").read_bytes() == before

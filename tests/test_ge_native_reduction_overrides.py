"""Named diagnostic reductions retain source FP16 boundaries and scan policy."""
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph
from test_ge_reduction_lowering import _execute
from test_ge_select_lowering import RecipeReader


def _graph():
    graph = Graph("native_overrides")
    source = graph.input("x", (2, 1, 8))
    integer = graph.input("integer", (2, 1, 8), "int32")
    for name, kind, value in (("selected_sum", "reduce_sum", source),
                              ("selected_mean", "reduce_mean", source),
                              ("other_sum", "reduce_sum", source),
                              ("integer_sum", "reduce_sum", integer)):
        graph.outputs.append(graph.op(kind, value, shape=(2, 1, 1), dtype=value.dtype,
                                       name=name, attrs={"axes": [-1], "keepdims": True}))
    graph.outputs.append(graph.op("cumsum", source, shape=source.shape, name="scan",
                                   attrs={"axis": -1, "scope": "gdn_nonpositive"}))
    return graph


@pytest.mark.parametrize("selection", [("selected_sum", "selected_mean"), ["selected_sum", "selected_mean"]])
def test_only_selected_graph_reductions_stay_native_with_fp16_boundaries(tmp_path, selection):
    graph = _graph()
    before = pickle.dumps(graph)
    report = {}
    emit_cpp(graph, recipe_dir=tmp_path / "recipe", fp16_reductions="matmul",
             fp16_native_reductions=selection, reduction_report=report)
    records, _, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    retained = {item["name"]: item for item in report["retained"]}
    assert report["native_overrides"] == list(selection)
    assert retained["integer_sum"]["reason"] == "non_fp16_source_or_output"
    for name in selection:
        assert retained[name]["reason"] == "explicit_native_override"
        assert retained[name]["origin"] == "graph"
        assert retained[name]["source"] == {"name": "x", "shape": [2, 1, 8], "dtype": "float16"}
        assert retained[name]["output"] == {"name": name, "shape": [2, 1, 1], "dtype": "float16"}
        node, = [record for record in records if record["name"] == name]
        assert node["type"] == ("ReduceMean" if name.endswith("mean") else "ReduceSum")
        assert node["inputs"][0]["dtype"] == node["outputs"][0]["dtype"] == "float16"
        assert node["inputs"][1]["dtype"] == "int32"
    assert report["counts"] == {"total": 12, "replaced": 9, "retained": 3,
                                 "fp16_replaced": 9, "integer_retained": 1,
                                 "scan_prefix_replaced": 8}
    assert {item["name"] for item in report["replaced"]} == {
        "other_sum", *(f"__ge/scan/sum_{i}" for i in range(1, 9))}
    feeds = {"x": -np.arange(1, 17, dtype=np.float16).reshape(2, 1, 8) / 16,
             "integer": np.arange(16, dtype=np.int32).reshape(2, 1, 8)}
    actual, expected = _execute(records, outputs, feeds), evaluate(graph, feeds)
    for got, wanted in zip(actual, expected):
        np.testing.assert_array_equal(got.view(np.uint16) if got.dtype == np.float16 else got,
                                      wanted.view(np.uint16) if wanted.dtype == np.float16 else wanted)
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("mode", ["native", "matmul"])
def test_empty_option_preserves_default_serialization_and_report(mode):
    graph = _graph()
    reports = [{}, {}, {}]
    default = emit_cpp(graph, fp16_reductions=mode, reduction_report=reports[0])
    for selection, report in zip(((), []), reports[1:]):
        assert emit_cpp(graph, fp16_reductions=mode, reduction_report=report,
                        fp16_native_reductions=selection) == default
    assert reports[0] == reports[1] == reports[2]
    assert reports[0]["native_overrides"] == []


@pytest.mark.parametrize("selection", ["selected_sum", {"selected_sum"}, None, True, 1,
                                         [""], [1], [True], [["selected_sum"]],
                                         ["selected_sum", "selected_sum"],
                                         ["missing"], ["x"], ["integer_sum"], ["scan"],
                                         ["__ge/scan/sum_1"]])
def test_invalid_or_non_graph_fp16_reduction_names_fail_before_writing(tmp_path, selection):
    folder = tmp_path / "recipe"
    with pytest.raises(ValueError, match="fp16_native_reductions"):
        emit_cpp(_graph(), recipe_dir=folder, fp16_reductions="matmul",
                 fp16_native_reductions=selection)
    assert not folder.exists()


@pytest.mark.parametrize("source_dtype,output_dtype", [("int32", "float16"), ("float16", "int32")])
def test_mixed_dtype_reduction_cannot_be_selected(tmp_path, source_dtype, output_dtype):
    graph = Graph("invalid_reduction")
    source = graph.input("x", (2, 8), source_dtype)
    graph.outputs = [graph.op("reduce_sum", source, shape=(2, 1), dtype=output_dtype,
                              name="bad", attrs={"axes": [-1], "keepdims": True})]
    with pytest.raises(ValueError, match="existing FP16 graph ReduceSum/Mean"):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe", fp16_native_reductions=["bad"])
    assert not (tmp_path / "recipe").exists()


def test_explicit_choice_is_reported_even_when_global_policy_already_native(tmp_path):
    report = {}
    emit_cpp(_graph(), recipe_dir=tmp_path / "recipe", reduction_report=report,
             fp16_native_reductions=["selected_sum"])
    assert report["counts"]["replaced"] == 0
    assert next(item for item in report["retained"] if item["name"] == "selected_sum")["reason"] == "explicit_native_override"
    assert all(item["reason"] == "mode_native" for item in report["retained"] if item["name"] != "selected_sum")

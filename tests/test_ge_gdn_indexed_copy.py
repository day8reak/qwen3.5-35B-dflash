"""Host gates for the bounded GDN integration; no device placement claim."""
from collections import Counter
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations, prepare_pair
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_target import _Builder, _chunk_gdn, build_target
from qwen35_moe_dflash.hashing import sha256_file
from test_ge_select_lowering import RecipeReader


def _target(mode="native", report=None):
    target, draft = configurations(tiny=True)
    return build_target(target, Qwen35DFlashConfig.from_dict(draft), 8,
                        gdn_indexed_copy=mode, gdn_copy_report=report)


def _chunk(sequence=8, heads=2, mode="native", report=None, observe=False):
    graph = Graph("chunk")
    inputs = [graph.input(name, (1, sequence, heads, dim) if dim else (1, sequence, heads))
              for name, dim in (("q", 3), ("k", 3), ("v", 4), ("g", 0), ("beta", 0))]
    states = [] if observe else None
    output = _chunk_gdn(_Builder(graph, gdn_indexed_copy=mode, gdn_copy_report=report),
                        *inputs, state_outputs=states)
    graph.outputs = [output, *(states or [])]
    return graph


def test_default_graph_and_cpp_bytes_are_unchanged_by_explicit_native_policy(tmp_path):
    target, draft = configurations(tiny=True)
    default = build_target(target, Qwen35DFlashConfig.from_dict(draft), 8)
    report = {"stale": True}
    native = _target(report=report)
    assert pickle.dumps(default) == pickle.dumps(native)
    paths = {key: tmp_path / "weights" / key for key in default.weights}
    # Descriptor-only emission is enough for the default-policy byte check;
    # the integration test below emits with actual prepared weight payloads.
    assert emit_cpp(default, paths, fold_constant_arithmetic=False) == emit_cpp(
        native, paths, fold_constant_arithmetic=False)
    assert report["schema_version"] == 1 and report["graph"] == "target"
    assert report["counts"] == {"total": 12, "replaced": 0, "retained": 12,
                                "mask_replaced": 0, "sub_replaced": 0}
    assert all(entry["strategy"] == "native" and entry["reason"] == "native_policy"
               for entry in report["entries"])
    assert "stale" not in report


def test_integrated_recipe_changes_only_nine_masks_and_three_outer_subtractions(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, tiny=True, sequence_length=8)
    report, reductions, rows = {}, {}, {}
    graph = _target("gather", report)
    assert report["counts"] == {"total": 12, "replaced": 12, "retained": 0,
                                "mask_replaced": 9, "sub_replaced": 3}
    assert Counter(entry["kind"] for entry in report["entries"]) == {
        "triangular_mask": 9, "outer_sub": 3}
    assert Counter(entry.get("site") for entry in report["entries"]) == {
        None: 3, "before_exp": 3, "after_exp": 3, "before_neg": 3}
    assert all(entry["dtype"] == "float16" and entry["shape"] == [2, 1, 64, 64]
               and entry["strategy"] == "gather" and entry["reason"] == "tested_shape"
               for entry in report["entries"])
    outputs = {node.output.name: node for node in graph.nodes}
    for entry in report["entries"]:
        node = outputs[entry["node"]]
        if entry["kind"] == "outer_sub":
            assert node.kind == "sub" and node.inputs[0].shape == node.inputs[1].shape == (2, 1, 64, 64)
        elif entry["site"] == "after_exp":
            exp = outputs[entry["inputs"][0]]
            assert exp.kind == "exp"
            assert exp.inputs[0].name in {item["node"] for item in report["entries"]
                                          if item.get("site") == "before_exp"}
        elif entry["site"] == "before_neg":
            assert entry["strict"] is True
            assert any(node.kind == "neg" and node.inputs[0].name == entry["node"]
                       for node in graph.nodes)
    before = pickle.dumps(graph)
    paths = {key: prepared / item["path"] for key, item in plan["weights"].items()}
    emit_cpp(graph, paths, recipe_dir=tmp_path / "recipe", fp16_reductions="matmul",
             reduction_report=reductions, fp16_row_mul="compact", row_mul_report=rows)
    assert pickle.dumps(graph) == before
    assert reductions["counts"]["replaced"] == 421
    assert rows["counts"]["replaced"] == 19
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    gathers = [record for record in records if record["type"] == "GatherV2"
               and record["name"].startswith("__indexed_copy/__gdn_copy/")]
    assert len(gathers) == 15
    assert all(record["inputs"][0]["dtype"] == "float16"
               and record["inputs"][1]["dtype"] == "int32" for record in gathers)
    # Every large triangular Concat is removed, while the recurrence's own
    # Slice/Concat updates and original arithmetic are retained.
    assert not any(record["type"] == "ConcatD" and record["attrs"]["N"] == 64
                   and record["outputs"][0]["shape"] == (2, 1, 64, 64)
                   for record in records)


@pytest.mark.parametrize("sequence,heads", [(65, 2), (8, 1), (8, 3)])
def test_untested_gdn_shapes_retain_identical_native_graph(sequence, heads):
    report = {}
    native = _chunk(sequence, heads)
    requested = _chunk(sequence, heads, "gather", report)
    assert pickle.dumps(native) == pickle.dumps(requested)
    assert report["counts"] == {"total": 4, "replaced": 0, "retained": 4,
                                "mask_replaced": 0, "sub_replaced": 0}
    assert all(entry["reason"] == "outside_tested_shape" for entry in report["entries"])


@pytest.mark.parametrize("kind,shapes", [
    ("triangular_mask", [(2, 1, 64, 64)]),
    ("outer_sub", [(2, 1, 64, 1), (2, 1, 1, 64)]),
])
@pytest.mark.parametrize("dtype", ["int32", "bool"])
def test_exact_shape_never_bypasses_fp16_guard(kind, shapes, dtype):
    graph = Graph("dtype_guard")
    values = [graph.input(f"x{index}", shape, dtype) for index, shape in enumerate(shapes)]
    builder = _Builder(graph, gdn_indexed_copy="gather")
    assert builder.gdn_copy_strategy(kind, *values) == ("native", "outside_tested_shape")
    assert not graph.nodes and not graph.constants


@pytest.mark.parametrize("sequence", [8, 64])
def test_gdn_copies_preserve_output_and_observed_final_state_bits(sequence):
    rng = np.random.default_rng(843)
    feeds = {"q": rng.normal(0, .25, (1, sequence, 2, 3)).astype(np.float16),
             "k": rng.normal(0, .25, (1, sequence, 2, 3)).astype(np.float16),
             "v": rng.normal(0, .25, (1, sequence, 2, 4)).astype(np.float16),
             "g": -rng.uniform(.01, .5, (1, sequence, 2)).astype(np.float16),
             "beta": rng.uniform(.05, .95, (1, sequence, 2)).astype(np.float16)}
    expected = evaluate(_chunk(sequence, observe=True), feeds)
    actual = evaluate(_chunk(sequence, mode="gather", observe=True), feeds)
    assert len(expected) == len(actual) == 2
    for old, new in zip(expected, actual):
        assert np.isfinite(old).all() and np.isfinite(new).all()
        np.testing.assert_array_equal(new.view(np.uint16), old.view(np.uint16))


def test_complete_tiny_pair_preserves_weight_payloads_abi_and_output_bits(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, tiny=True, sequence_length=8)
    draft_config = Qwen35DFlashConfig.from_dict(plan["draft_config"])
    native = build_target(plan["target_config"], draft_config, 8)
    candidate = build_target(plan["target_config"], draft_config, 8, gdn_indexed_copy="gather")
    draft = build_draft(draft_config, 8)
    assert native.weights == candidate.weights
    abi = lambda graph: [(value.shape, value.dtype) for value in graph.inputs + graph.outputs]
    assert abi(native) == abi(candidate)
    weights = {}
    for key, item in plan["weights"].items():
        path = prepared / item["path"]
        assert sha256_file(path) == item["sha256"]
        weights[key] = np.fromfile(path, dtype="<f2").reshape(item["shape"])
        weights[key].setflags(write=False)
    for length in (1, 3, 8):
        ids = np.zeros((1, 8), dtype=np.int32)
        mask = np.zeros_like(ids)
        ids[0, :length] = [7, 3, 26, 8, 16, 42, 29, 31][:length]
        mask[0, :length] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        top1, features = evaluate(native, feeds, weights)
        actual_top1, actual_features = evaluate(candidate, feeds, weights)
        np.testing.assert_array_equal(actual_top1, top1)
        assert np.isfinite(features).all() and np.isfinite(actual_features).all()
        np.testing.assert_array_equal(actual_features.view(np.uint16), features.view(np.uint16))
        expected_draft, = evaluate(draft, {**feeds, "target_features": features}, weights)
        actual_draft, = evaluate(draft, {**feeds, "target_features": actual_features}, weights)
        np.testing.assert_array_equal(actual_draft, expected_draft)
    for item in plan["weights"].values():
        assert sha256_file(prepared / item["path"]) == item["sha256"]


@pytest.mark.parametrize("mode", ["other", None, []])
def test_bad_policy_fails_before_graph_or_report_changes(mode):
    report = {"sentinel": True}
    with pytest.raises(ValueError, match="gdn_indexed_copy"):
        build_target({}, None, 8, gdn_indexed_copy=mode, gdn_copy_report=report)
    assert report == {"sentinel": True}

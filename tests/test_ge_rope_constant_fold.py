"""Whole-pair host regressions for the constant target RoPE K=1 MatMul.

These checks cover graph construction and the existing semantic oracle. They
do not establish compiler partitioning, target Load success or NPU placement.
"""
from copy import deepcopy
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash import ge_constants, ge_recipe
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import build_pair, configurations, weight_shapes
from qwen35_moe_dflash.ge_weights import tiny_weights
from qwen35_moe_dflash.hashing import sha256_file


_POLICIES = [
    pytest.param("native", {}, id="default-native"),
    pytest.param("gather", {"fp16_reductions": "matmul", "fp16_row_mul": "compact"},
                 id="matmul-compact-gather"),
]


def _abi(graph):
    return [[(value.name, value.shape, value.dtype) for value in values]
            for values in (graph.inputs, graph.outputs)]


def _rope_matmul(graph):
    nodes = [node for node in graph.nodes if node.kind == "matmul"
             and all(value.name in graph.constants for value in node.inputs)]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.output.name == "v10"
    assert [value.shape for value in node.inputs] == [(1, 2, 1), (1, 1, 8)]
    assert node.output.shape == (1, 2, 8)
    return node


@pytest.fixture(scope="module")
def tiny_payloads(tmp_path_factory):
    target_config, draft_config = configurations(tiny=True)
    graphs = build_pair(target_config, draft_config, 8)
    weights = tiny_weights(weight_shapes(graphs))
    folder = tmp_path_factory.mktemp("rope-fold-weights")
    paths = {}
    for index, (key, array) in enumerate(weights.items()):
        paths[key] = folder / f"{index}.bin"
        paths[key].write_bytes(array.tobytes())
        array.setflags(write=False)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    yield target_config, draft_config, weights, paths
    assert {key: sha256_file(path) for key, path in paths.items()} == hashes


@pytest.mark.parametrize("gdn_policy,emission_policy", _POLICIES)
def test_actual_pair_emission_folds_target_rope_and_preserves_abi_and_external_weights(
        tmp_path, monkeypatch, tiny_payloads, gdn_policy, emission_policy):
    target_config, draft_config, _, paths = tiny_payloads
    graphs = build_pair(target_config, draft_config, 8, gdn_indexed_copy=gdn_policy)
    originals = [pickle.dumps(graph) for graph in graphs]
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    target_rope = _rope_matmul(graphs[0])
    original_fold = ge_constants.fold_constants
    original_emit = ge_recipe.emit_recipe_cpp
    captured = []

    def record_emit(records, name, inputs, outputs, recipe_dir):
        captured.append((deepcopy(records), list(inputs), list(outputs)))
        return original_emit(records, name, inputs, outputs, recipe_dir)

    def previous_folding(graph, *args, **kwargs):
        # Retain the pre-existing unary/view folds to isolate the new K=1
        # change from their already established external-weight pruning.
        folded = original_fold(graph, *args, **kwargs)
        for node in graph.nodes:
            if node.kind == "matmul":
                folded.pop(node.output.name, None)
        return folded

    monkeypatch.setattr(ge_recipe, "emit_recipe_cpp", record_emit)
    for graph in graphs:
        monkeypatch.setattr(ge_constants, "fold_constants", previous_folding)
        emit_cpp(graph, paths, external_weights=True,
                 recipe_dir=tmp_path / f"{graph.name}-before", **emission_policy)
        before, before_inputs, before_outputs = captured[-1]
        monkeypatch.setattr(ge_constants, "fold_constants", original_fold)
        emit_cpp(graph, paths, external_weights=True,
                 recipe_dir=tmp_path / f"{graph.name}-after", **emission_policy)
        after, after_inputs, after_outputs = captured[-1]

        def emitted_abi(records, inputs, outputs):
            def item(index, port):
                record = records[index]
                descriptor = record["outputs"][port]
                return record["name"], tuple(descriptor["shape"]), descriptor["dtype"]
            return [[item(index, 0) for index in inputs],
                    [item(index, port) for index, port in outputs]]

        assert emitted_abi(before, before_inputs, before_outputs) == _abi(graph)
        assert emitted_abi(after, after_inputs, after_outputs) == _abi(graph)
        before_files = [row for row in before if row["type"] == "FileConstant"]
        after_files = [row for row in after if row["type"] == "FileConstant"]
        assert before_files and after_files == before_files
        if graph.name == "target":
            old = next(row for row in before if row["name"] == target_rope.output.name)
            new = next(row for row in after if row["name"] == target_rope.output.name)
            assert old["type"] == "BatchMatMulV2"
            assert new["type"] == "Const"
            assert new["outputs"] == old["outputs"]
            left, right = [graph.constants[value.name].astype(np.float32)
                           for value in target_rope.inputs]
            expected = np.matmul(left, right).astype("<f2")
            assert new["constant"]["data"] == expected.tobytes()
            assert not new["inputs"]
            assert not {value.name for value in target_rope.inputs} & {row["name"] for row in after}
        else:
            assert original_fold(graph, paths, set(graph._values)) == {}
            assert after == before
            frequency, = [node for node in graph.nodes if node.kind == "mul"
                          and node.output.shape == (1, 12, 4)
                          and any(value.name in graph.constants for value in node.inputs)]
            assert any(value.name not in graph.constants for value in frequency.inputs)
            assert next(row for row in after if row["name"] == frequency.output.name)["type"] == "Mul"
    assert [pickle.dumps(graph) for graph in graphs] == originals
    assert {key: sha256_file(path) for key, path in paths.items()} == hashes


@pytest.mark.parametrize("gdn_policy", ["native", "gather"])
@pytest.mark.parametrize("length", [1, 3, 8])
def test_complete_target_and_draft_output_bits_match_with_folded_rope(
        tiny_payloads, gdn_policy, length):
    target_config, draft_config, weights, paths = tiny_payloads
    target, draft = build_pair(target_config, draft_config, 8, gdn_indexed_copy=gdn_policy)
    rope = _rope_matmul(target)
    folded = ge_constants.fold_constants(target, paths, set(target._values))
    assert rope.output.name in folded
    lowered = deepcopy(target)
    lowered.constants.update(folded)
    lowered.nodes = [node for node in lowered.nodes if node.output.name not in folded]
    assert lowered.weights == target.weights
    assert _abi(lowered) == _abi(target)
    assert ge_constants.fold_constants(draft, paths, set(draft._values)) == {}
    ids = np.zeros((1, 8), dtype=np.int32)
    mask = np.zeros_like(ids)
    ids[0, :length] = [7, 3, 26, 8, 16, 42, 29, 31][:length]
    mask[0, :length] = 1
    feeds = {"input_ids": ids, "attention_mask": mask}
    expected = evaluate(target, feeds, weights)
    actual = evaluate(lowered, feeds, weights)
    for old, new in zip(expected, actual):
        assert new.dtype == old.dtype and new.shape == old.shape
        assert np.isfinite(old).all() and np.isfinite(new).all()
        assert new.tobytes() == old.tobytes()
    expected_draft, = evaluate(draft, {**feeds, "target_features": expected[1]}, weights)
    actual_draft, = evaluate(draft, {**feeds, "target_features": actual[1]}, weights)
    assert actual_draft.dtype == expected_draft.dtype
    assert actual_draft.shape == expected_draft.shape
    assert actual_draft.tobytes() == expected_draft.tobytes()

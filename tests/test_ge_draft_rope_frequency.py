"""Bounded draft RoPE BMM preserves runtime positions and FP16 boundaries."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash import ge_recipe
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_constants import fold_constants
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_ir import Graph


def _config(tiny=True):
    return Qwen35DFlashConfig.from_dict(configurations(tiny=tiny)[1])


def _frequency(graph):
    node, = [node for node in graph.nodes if node.output.name == "v29"]
    return node


def _old_mul(graph):
    old = deepcopy(graph)
    node = _frequency(old)
    node.kind = "mul"
    node.attrs = {}
    return old


def _abi(graph):
    return [[(value.name, value.shape, value.dtype) for value in values]
            for values in (graph.inputs, graph.outputs)]


def _position_slice(graph):
    """Keep the actual source position computation, with no model weights."""
    graph = deepcopy(graph)
    named = {node.output.name: node for node in graph.nodes}
    graph.outputs = [named[name].output for name in ("positions", "v27", "v29")]
    live = {value.name for value in graph.outputs}
    for node in reversed(graph.nodes):
        if node.output.name in live:
            live.update(value.name for value in node.inputs)
    graph.nodes = [node for node in graph.nodes if node.output.name in live]
    graph.inputs = [value for value in graph.inputs if value.name in live]
    graph.constants = {name: array for name, array in graph.constants.items() if name in live}
    graph.weights = {name: value for name, value in graph.weights.items() if name in live}
    assert not graph.weights
    assert [value.name for value in graph.inputs] == ["attention_mask"]
    return graph


@pytest.mark.parametrize("tiny", [True, False], ids=["tiny", "full-config"])
def test_source_changes_only_the_runtime_frequency_operation_and_preserves_contracts(tiny):
    config = _config(tiny)
    graph = build_draft(config, 8)
    old = _old_mul(graph)
    assert _abi(graph) == _abi(old)
    assert graph.weights == old.weights
    assert list(graph.constants) == list(old.constants)
    for name, data in graph.constants.items():
        assert data.tobytes() == old.constants[name].tobytes()
    frequency = _frequency(graph)
    assert frequency.kind == "matmul"
    assert [value.shape for value in frequency.inputs] == [(1, 8 + config.block_size, 1),
                                                         (1, 1, config.head_dim // 2)]
    assert frequency.output.shape == (1, 8 + config.block_size, config.head_dim // 2)
    assert frequency.output.dtype == "float16"
    assert all(value.dtype == "float16" for value in frequency.inputs)
    assert frequency.inputs[0].name not in graph.constants
    assert frequency.inputs[1].name in graph.constants
    nodes = {node.output.name: node for node in graph.nodes}
    reshaped = nodes[frequency.inputs[0].name]
    cast = nodes[reshaped.inputs[0].name]
    assert reshaped.kind == "reshape" and reshaped.output.name == "v28"
    assert cast.kind == "cast" and cast.output.name == "v27"
    assert cast.inputs[0].name == "positions" and cast.inputs[0].dtype == "int32"
    assert cast.output.dtype == "float16"
    assert fold_constants(graph, {}, set(graph._values)) == {}
    changed = deepcopy(graph)
    _frequency(changed).kind = "mul"
    _frequency(changed).attrs = {}
    assert changed.nodes == old.nodes


@pytest.mark.parametrize("tiny", [True, False], ids=["tiny", "full-config"])
@pytest.mark.parametrize("active", [False, True], ids=["native", "matmul-compact"])
def test_actual_recipe_uses_dynamic_left_bmm_and_original_constant_rhs(
        tmp_path, monkeypatch, tiny, active):
    graph = build_draft(_config(tiny), 8)
    old = _old_mul(graph)
    before = pickle.dumps(graph)
    captured = []
    original_emit = ge_recipe.emit_recipe_cpp

    def capture(records, name, inputs, outputs, recipe_dir):
        captured.append((deepcopy(records), list(inputs), list(outputs)))
        return original_emit(records, name, inputs, outputs, recipe_dir)

    monkeypatch.setattr(ge_recipe, "emit_recipe_cpp", capture)
    # These are descriptor-only external weights. A hidden payload read would
    # fail; neither the new BMM nor the draft's constant folder needs one.
    paths = {name: tmp_path / "unread-external-weights" / name for name in graph.weights}
    policy = {"fp16_reductions": "matmul", "fp16_row_mul": "compact"} if active else {}
    for label, source in (("old", old), ("candidate", graph)):
        emit_cpp(source, paths, external_weights=True, recipe_dir=tmp_path / label, **policy)
    old_records, old_inputs, old_outputs = captured[0]
    records, inputs, outputs = captured[1]
    assert [(row["name"], row["outputs"]) for row in records if row["type"] == "Data"] == [
        (row["name"], row["outputs"]) for row in old_records if row["type"] == "Data"]
    assert [row for row in records if row["type"] == "FileConstant"] == [
        row for row in old_records if row["type"] == "FileConstant"]
    assert inputs == old_inputs and outputs == old_outputs
    emitted = next(row for row in records if row["name"] == "v29")
    assert emitted["type"] == "BatchMatMulV2"
    assert next(row for row in old_records if row["name"] == "v29")["type"] == "Mul"
    left, right = [records[value["src"]] for value in emitted["inputs"]]
    assert left["name"] == "v28" and left["type"] == "Reshape"
    assert right["type"] == "Const"
    inverse = graph.constants[_frequency(graph).inputs[1].name]
    assert right["constant"]["data"] == inverse.tobytes()
    assert right["outputs"][0]["shape"] == list(inverse.shape)
    assert all(value["dtype"] == "float16" for value in emitted["inputs"] + emitted["outputs"])
    cast = next(row for row in records if row["name"] == "v27")
    assert cast["type"] == "Cast"
    assert cast["inputs"][0]["dtype"] == "int32" and cast["outputs"][0]["dtype"] == "float16"
    assert pickle.dumps(graph) == before


@pytest.mark.parametrize("tiny", [True, False], ids=["tiny", "full-config"])
def test_all_current_prefix_lengths_match_old_mul_source_slice_bytes(tiny):
    graph = _position_slice(build_draft(_config(tiny), 8))
    old = _old_mul(graph)
    for length in range(1, 9):
        mask = np.zeros((1, 8), dtype=np.int32)
        mask[0, :length] = 1
        expected = evaluate(old, {"attention_mask": mask})
        actual = evaluate(graph, {"attention_mask": mask})
        for left, right in zip(expected, actual):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert np.isfinite(left).all() and np.isfinite(right).all()
            assert left.tobytes() == right.tobytes()
        positions = actual[0]
        assert positions.min() >= 0 and positions.max() <= 8 + _config(tiny).block_size - 2
        assert not np.signbit(actual[1]).any()


def test_position_cast_rounding_precedes_product_at_2048_boundary():
    config = replace(_config(), max_position_embeddings=4096)
    graph = _position_slice(build_draft(config, 2052))
    mask = np.ones((1, 2052), dtype=np.int32)
    positions, converted, actual = evaluate(graph, {"attention_mask": mask})
    expected = evaluate(_old_mul(graph), {"attention_mask": mask})[-1]
    assert actual.tobytes() == expected.tobytes()
    assert 2049 in positions and 2051 in positions
    assert np.all(converted[positions == 2049] == np.float16(2048))
    assert np.all(converted[positions == 2051] == np.float16(2052))
    inverse = graph.constants[_frequency(graph).inputs[1].name].astype(np.float32)
    late_cast = (positions.astype(np.float32)[..., None] * inverse).astype("<f2")
    assert late_cast.tobytes() != actual.tobytes()


@pytest.mark.parametrize("offset,expected", [(0, "matmul"), (1, "mul")])
def test_sequence_domain_gate_uses_the_largest_block_position(offset, expected):
    config = replace(_config(), max_position_embeddings=70000)
    sequence = 65504 - config.block_size + 2 + offset
    graph = build_draft(config, sequence)
    assert _frequency(graph).kind == expected
    assert _frequency(graph).output.shape == (1, sequence + config.block_size, config.head_dim // 2)


def test_positive_theta_with_overflowing_fp16_inverse_retains_mul():
    config = replace(_config(), rope_theta=1e-20)
    with np.errstate(over="ignore", invalid="ignore"):
        graph = build_draft(config, 8)
    frequency = _frequency(graph)
    assert frequency.kind == "mul"
    assert not np.isfinite(graph.constants[frequency.inputs[1].name]).all()


def test_finite_nonnegative_operands_preserve_positive_overflow_at_output_boundary():
    graph = _position_slice(build_draft(replace(_config(), rope_theta=1e-4), 80))
    assert _frequency(graph).kind == "matmul"
    mask = np.ones((1, 80), dtype=np.int32)
    actual = evaluate(graph, {"attention_mask": mask})[-1]
    expected = evaluate(_old_mul(graph), {"attention_mask": mask})[-1]
    assert np.isposinf(actual).any()
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("replacement,expected", [
    (-0.0, "mul"), (-1.0, "mul"), (np.nan, "mul"), (np.inf, "mul"),
    (-np.inf, "mul"), (0.0, "matmul"),
])
def test_inverse_domain_gate_distinguishes_positive_and_negative_zero(monkeypatch, replacement, expected):
    config = _config()
    original_const = Graph.const

    def replace_inverse(self, data, dtype="float16", name=None):
        value = original_const(self, data, dtype, name)
        if self.name == "draft" and dtype == "float16" and value.shape == (1, 1, config.head_dim // 2):
            self.constants[value.name].flat[0] = replacement
        return value

    monkeypatch.setattr(Graph, "const", replace_inverse)
    graph = build_draft(config, 8)
    assert _frequency(graph).kind == expected
    if expected == "matmul":
        source_slice = _position_slice(graph)
        mask = np.ones((1, 8), dtype=np.int32)
        actual = evaluate(source_slice, {"attention_mask": mask})[-1]
        previous = evaluate(_old_mul(source_slice), {"attention_mask": mask})[-1]
        assert actual.tobytes() == previous.tobytes()
        assert not np.signbit(actual[..., 0]).any()

"""Checked INT32 BroadcastTo controls at real draft serializer boundaries.

Recipe/unrolled host checks preserve payloads and semantics; they do not prove
the selected OMG or board accepts/executes the generated graph.
"""
import copy
from pathlib import Path
import re

import numpy as np
import pytest

from qwen35_moe_dflash.draft_standalone import _graph
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_ir import Graph
from test_ge_select_lowering import _assert_select_contract, _evaluate_recipe
from test_ge_static_broadcast import _payload, _serialize_both


def _check_shape(records, record):
    assert record["type"] == "BroadcastTo"
    data, control = record["inputs"]
    output, = record["outputs"]
    assert data["port"] == "x" and control["port"] == "shape"
    assert data["dtype"] == output["dtype"]
    assert control["dtype"] == "int32" and control["shape"] == (len(output["shape"]),)
    literal = records[control["src"]]
    assert literal["type"] == "Const"
    descriptor, = literal["outputs"]
    assert descriptor["port"] == control["src_port"]
    assert descriptor["shape"] == control["shape"] and descriptor["dtype"] == control["dtype"]
    data = _payload(literal)
    assert len(data) == 4 * len(output["shape"])
    assert np.frombuffer(data, dtype="<i4").tolist() == list(output["shape"])
    return literal


def test_full_six_layer_s64_block16_v501_and_all_runtime_broadcast_controls(tmp_path):
    _, config, graph = _graph(64)
    assert config.num_hidden_layers == 6 and config.block_size == 16
    original = copy.deepcopy(graph.manifest())
    observed = next(node for node in graph.nodes if node.output.name == "v501")
    assert observed.kind == "broadcast_to" and observed.output.dtype == "bool"
    assert observed.inputs[0].shape == (1, 1, 1, 80) and observed.output.shape == (1, 1, 16, 80)
    # Full checkpoint weights remain unopened references: this test concerns
    # the actual six-layer graph topology and serialized shape controls only.
    paths = {key: tmp_path / f"unopened-{index}.bin" for index, key in enumerate(graph.weights)}
    records, _, outputs = _serialize_both(graph, tmp_path, weight_paths=paths,
        external_weights=True, fold_constant_arithmetic=False, draft_fc_policy="conv2d")
    assert (tmp_path / "recipe/graph.recipe").read_bytes()[:8] == b"Q35GE002"
    broadcasts = [record for record in records if record["type"] == "BroadcastTo"]
    assert len(broadcasts) == 12
    for record in broadcasts:
        _check_shape(records, record)
    explicit = next(record for record in broadcasts if record["name"] == "v501")
    assert explicit["inputs"][0]["shape"] == observed.inputs[0].shape
    assert explicit["outputs"][0]["shape"] == observed.output.shape
    assert explicit["inputs"][0]["dtype"] == explicit["outputs"][0]["dtype"] == "bool"
    selected = _assert_select_contract(records)
    masks = [record for record in selected if record["outputs"][0]["dtype"] == "float16"]
    assert len(masks) == 6
    for record in masks:
        condition = records[record["inputs"][0]["src"]]
        assert condition["type"] == "BroadcastTo" and condition["outputs"][0]["dtype"] == "bool"
        assert record["outputs"][0]["shape"] == (1, 32, 16, 80)
    assert [records[index]["outputs"][port]["shape"] for index, port in outputs] == [(1, 15), (1, 15, 2048)]
    assert graph.manifest() == original
    assert not any(path.exists() for path in paths.values())


def test_dynamic_v501_mask_to_select_preserves_selected_nan_and_signed_zero_bits(tmp_path):
    graph = Graph("draft_dynamic_mask")
    condition = graph.input("condition", (1, 1, 1, 80), "bool")
    expanded = graph.op("broadcast_to", condition, shape=(1, 1, 16, 80), dtype="bool",
                        attrs={"shape": [1, 1, 16, 80]}, name="v501")
    scores = graph.input("scores", (1, 32, 16, 80))
    other = graph.const(np.float16(-np.inf))
    graph.outputs = [graph.op("where", expanded, scores, other, shape=scores.shape, name="v505")]
    records, _, outputs = _serialize_both(graph, tmp_path)
    broadcasts = [record for record in records if record["type"] == "BroadcastTo"]
    assert len(broadcasts) == 2 and {record["name"] for record in broadcasts} == {"v501", "__ge/v505/broadcast_0"}
    for record in broadcasts:
        _check_shape(records, record)
    _assert_select_contract(records)
    mask = ((np.arange(80) // 8) % 2 == 0).reshape(condition.shape)
    # Every value appears on both selected and excluded positions; where must
    # preserve selected signalling/quiet NaN payloads without arithmetic masks.
    bits = np.resize(np.array([0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01, 1, 0x8001], dtype="<u2"), scores.shape)
    feed = bits.view("<f2")
    expected = np.where(mask, feed, np.float16(-np.inf))
    actual, = _evaluate_recipe(records, outputs, {"condition": mask, "scores": feed})
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"))
    assert not any(record["type"] in {"Mul", "Add", "Cast"} for record in records)


@pytest.mark.parametrize("dtype", ["float16", "int32", "int64", "bool"])
def test_explicit_runtime_broadcast_only_changes_shape_control_dtype(tmp_path, dtype):
    graph = Graph("runtime_broadcast")
    source = graph.input("source", (1, 3), dtype)
    graph.outputs = [graph.op("broadcast_to", source, shape=(2, 3), dtype=dtype,
                              attrs={"shape": [2, 3]}, name="expanded")]
    records, _, outputs = _serialize_both(graph, tmp_path)
    record = records[outputs[0][0]]
    _check_shape(records, record)
    assert record["inputs"][0]["shape"] == (1, 3) and record["outputs"][0]["shape"] == (2, 3)
    assert record["outputs"][0]["dtype"] == dtype
    assert {record["type"] for record in records} == {"Data", "Const", "BroadcastTo"}


@pytest.mark.parametrize("kind", ["explicit", "select"])
@pytest.mark.parametrize("dimension", [1 << 31, 1 << 40])
def test_runtime_broadcast_rejects_overflow_without_narrowing_or_expansion(tmp_path, monkeypatch, kind, dimension):
    graph = Graph("overflow_broadcast")
    source = graph.input("source", (1,), "bool")
    if kind == "explicit":
        graph.outputs = [graph.op("broadcast_to", source, shape=(dimension,), dtype="bool",
                                  attrs={"shape": [dimension]})]
    else:
        then = graph.input("then", (dimension,))
        other = graph.input("other", (dimension,))
        graph.outputs = [graph.op("where", source, then, other, shape=(dimension,))]
    def no_expand(*args, **kwargs):
        pytest.fail("out-of-range shape must not expand tensor data")
    monkeypatch.setattr(np, "broadcast_to", no_expand)
    with pytest.raises(ValueError, match="BroadcastTo dimensions must fit nonnegative INT32"):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("bad_shape", [[3, 2], [-1, 3], [True, 3], [2.0, 3]])
def test_explicit_shape_controls_reject_mismatch_or_noninteger_values(tmp_path, bad_shape):
    graph = Graph("invalid_broadcast_control")
    source = graph.input("source", (1, 3))
    graph.outputs = [graph.op("broadcast_to", source, shape=(2, 3), attrs={"shape": bad_shape})]
    with pytest.raises(ValueError, match="BroadcastTo.*(?:descriptor|INT32)"):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("backend", ["legacy", "stateful"])
def test_both_exporters_emit_int32_for_explicit_and_select_generated_bool_broadcast(tmp_path, backend):
    if backend == "legacy":
        from qwen35_moe_dflash.ge_ir import Graph as Builder
        from qwen35_moe_dflash.ge_cpp import emit_cpp as emit
        from qwen35_moe_dflash.ge_recipe_read import read_recipe
    else:
        from qwen35_moe_dflash.ge_fused.ir import Graph as Builder
        from qwen35_moe_dflash.ge_fused.emitter import emit_cpp as emit
        from qwen35_moe_dflash.ge_fused.recipe import read_recipe
    graph = Builder("dynamic_mask_controls")
    condition = graph.input("condition", (1, 1, 3), "bool")
    expanded = graph.op("broadcast_to", condition, shape=(1, 2, 3), dtype="bool",
                        attrs={"shape": [1, 2, 3]}, name="explicit_mask")
    scores = graph.input("scores", (4, 2, 3))
    graph.outputs = [graph.op("where", expanded, scores, graph.const(np.float16(-np.inf)),
                              shape=scores.shape, name="selected")]
    emit(graph, recipe_dir=tmp_path / "recipe")
    records = read_recipe(tmp_path / "recipe/graph.recipe")["records"]
    cpp = emit(graph)
    literals = {int(index): bytes(int(byte) for byte in data.split(",") if byte)
                for index, data in re.findall(
                    r'SetConst\(ops\[(\d+)\], [^\n]*std::vector<uint8_t>\{([\d,]*)\}\);', cpp)}
    broadcasts = [record for record in records if record["type"] == "BroadcastTo"]
    assert {record["name"] for record in broadcasts} == {"explicit_mask", "__ge/selected/broadcast_0"}
    assert cpp.count(', "BroadcastTo",') == 2
    for record in broadcasts:
        data, shape = record["inputs"]
        output, = record["outputs"]
        assert data["dtype"] == output["dtype"] == "bool" and shape["dtype"] == "int32"
        constant = records[shape["src"]]
        descriptor, = constant["outputs"]
        assert descriptor["dtype"] == "int32" and descriptor["shape"] == shape["shape"]
        storage = constant["constant"]
        with Path(storage["path"]).open("rb") as stream:
            stream.seek(storage["offset"])
            payload = stream.read(storage["size"])
        assert len(payload) == 4 * len(output["shape"])
        assert np.frombuffer(payload, dtype="<i4").tolist() == output["shape"]
        assert literals[shape["src"]] == payload

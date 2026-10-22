"""Byte-level tests for bounded static broadcasts in both GE serializer modes.

These inspect host construction artifacts; they do not assert OMG/device support.
"""
import json
from pathlib import Path
import re

import numpy as np
import pytest

import qwen35_moe_dflash.ge_cpp as ge_cpp
from qwen35_moe_dflash.ge_ir import Graph
from test_ge_select_lowering import RecipeReader, _evaluate_recipe, _assert_select_contract


def _payload(record):
    path, offset, size = record["constant"]
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        data = stream.read(size)
    assert len(data) == size
    return data


def _serialize_both(graph, tmp_path):
    ge_cpp.emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records, inputs, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    cpp = ge_cpp.emit_cpp(graph)
    # Verify the unrolled source emits the same operation order and constant
    # bytes as the independently decoded recipe, including NaN payload bits.
    declared = [(json.loads(name), json.loads(kind)) for name, kind in re.findall(
        r'ops\.push_back\(Op\(("[^"\n]+"), ("[^"\n]+"),', cpp)]
    assert declared == [(record["name"], record["type"]) for record in records]
    literal_data = {int(index): bytes(int(byte) for byte in data.split(",") if byte)
                    for index, data in re.findall(
                        r'SetConst\(ops\[(\d+)\], [^\n]*std::vector<uint8_t>\{([\d,]*)\}\);', cpp)}
    assert literal_data == {index: _payload(record) for index, record in enumerate(records)
                            if record["type"] == "Const"}
    # Lazy constants must not leave disconnected nodes in GE's expected list.
    live = {index for index, _ in outputs}
    for index in reversed(range(len(records))):
        if index in live:
            live.update(port["src"] for port in records[index]["inputs"])
    assert all(index in live for index, record in enumerate(records) if record["type"] == "Const")
    return records, inputs, outputs


def _where_graph(scalars, *, shape=(2, 3)):
    graph = Graph("static_broadcast")
    condition = graph.input("condition", shape, "bool")
    then = graph.input("then", shape)
    graph.outputs = [graph.op("where", condition, then,
                             graph.const(scalar, name=f"literal_{index}"), shape=shape)
                     for index, scalar in enumerate(scalars)]
    return graph


@pytest.mark.parametrize("bits", [0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01],
                         ids=["positive-zero", "negative-zero", "positive-inf", "negative-inf",
                              "quiet-nan-payload", "signalling-nan-payload"])
def test_static_scalar_broadcast_preserves_raw_fp16_bits(tmp_path, bits):
    scalar = np.array(bits, dtype="<u2").view("<f2")
    graph = _where_graph([scalar])
    records, _, outputs = _serialize_both(graph, tmp_path)
    selected, = _assert_select_contract(records)
    constant = records[selected["inputs"][2]["src"]]
    assert constant["type"] == "Const"
    assert _payload(constant) == scalar.tobytes() * 6
    assert not any(record["type"] == "BroadcastTo" for record in records)
    assert not any(record["name"] == "literal_0" for record in records)
    actual, = _evaluate_recipe(records, outputs, {
        "condition": np.zeros((2, 3), dtype=bool), "then": np.ones((2, 3), dtype=np.float16)})
    np.testing.assert_array_equal(actual.view("<u2"), np.full((2, 3), bits, dtype="<u2"))


def test_static_boolean_condition_and_carrier_keep_boolean_semantics(tmp_path):
    graph = Graph("bool_literals")
    then = graph.input("then", (2, 3), "bool")
    condition = graph.const([[True], [False]], "bool", name="condition_literal")
    other = graph.const(False, "bool", name="false_literal")
    graph.outputs = [graph.op("where", condition, then, other, shape=(2, 3), dtype="bool")]
    records, _, outputs = _serialize_both(graph, tmp_path)
    selected, = _assert_select_contract(records)
    condition_record = records[selected["inputs"][0]["src"]]
    branch_record = records[selected["inputs"][2]["src"]]
    assert condition_record["outputs"][0]["dtype"] == "bool"
    assert branch_record["outputs"][0]["dtype"] == "int32"
    assert condition_record["type"] == branch_record["type"] == "Const"
    assert _payload(branch_record) == bytes(2 * 3 * 4)
    assert not any(record["type"] == "BroadcastTo" for record in records)
    feeds = {"then": np.array([[True, False, True], [True, True, True]])}
    actual, = _evaluate_recipe(records, outputs, feeds)
    np.testing.assert_array_equal(actual, np.array([[True, False, True], [False, False, False]]))


def test_equal_constants_with_distinct_names_share_one_expansion(tmp_path):
    graph = _where_graph([np.float16(0), np.float16(0), np.float16(-0.0)])
    records, _, _ = _serialize_both(graph, tmp_path)
    selected = _assert_select_contract(records)
    source_indices = [record["inputs"][2]["src"] for record in selected]
    assert source_indices[0] == source_indices[1]
    assert source_indices[0] != source_indices[2]
    assert sum(record["type"] == "Const" for record in records) == 2


def _set_budgets(monkeypatch, *, tensor_bytes, total_bytes):
    original = ge_cpp._Emitter.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.max_broadcast_tensor_bytes = tensor_bytes
        self.max_broadcast_total_bytes = total_bytes

    monkeypatch.setattr(ge_cpp._Emitter, "__init__", initialize)


def test_per_tensor_budget_keeps_runtime_broadcast_without_expansion(tmp_path, monkeypatch):
    _set_budgets(monkeypatch, tensor_bytes=10, total_bytes=100)
    graph = _where_graph([np.float16(0)])  # Expanded FP16 tensor needs 12 bytes.
    records, _, _ = _serialize_both(graph, tmp_path)
    selected, = _assert_select_contract(records)
    expanded = records[selected["inputs"][2]["src"]]
    assert expanded["type"] == "BroadcastTo"
    assert records[expanded["inputs"][0]["src"]]["name"] == "literal_0"


def test_total_budget_counts_unique_expansions_and_reuses_after_exhaustion(tmp_path, monkeypatch):
    _set_budgets(monkeypatch, tensor_bytes=12, total_bytes=12)
    graph = _where_graph([np.float16(0), np.float16(1), np.float16(0)])
    records, _, _ = _serialize_both(graph, tmp_path)
    selected = _assert_select_contract(records)
    source_indices = [record["inputs"][2]["src"] for record in selected]
    assert source_indices[0] == source_indices[2]
    assert records[source_indices[0]]["type"] == "Const"
    assert records[source_indices[1]]["type"] == "BroadcastTo"
    assert not any(record["name"] in {"literal_0", "literal_2"} for record in records)


def test_budget_guard_precedes_attempt_to_expand_large_tensor(tmp_path, monkeypatch):
    graph = Graph("large_broadcast")
    graph.input("unused", (1,))
    scalar = graph.const(np.float16(0))
    shape = (1 << 40,)
    graph.outputs = [graph.op("broadcast_to", scalar, shape=shape, attrs={"shape": shape})]

    def forbidden(*args, **kwargs):
        raise AssertionError("over-budget broadcast must not allocate or expand")

    monkeypatch.setattr(ge_cpp.np, "broadcast_to", forbidden)
    records, _, outputs = _serialize_both(graph, tmp_path)
    assert records[outputs[0][0]]["type"] == "BroadcastTo"


def test_explicit_broadcast_alias_outputs_and_original_constant_output(tmp_path):
    graph = Graph("output_alias")
    graph.input("unused", (1,))
    source = graph.const(np.float16(-0.0), name="original")
    copy = graph.const(np.float16(-0.0), name="same_value")
    a = graph.op("broadcast_to", source, shape=(2, 3), attrs={"shape": (2, 3)})
    b = graph.op("broadcast_to", copy, shape=(2, 3), attrs={"shape": (2, 3)})
    graph.outputs = [a, b, source]
    records, _, outputs = _serialize_both(graph, tmp_path)
    assert outputs[0] == outputs[1]
    assert outputs[0] != outputs[2]
    assert not any(record["name"] == "same_value" for record in records)
    actual = _evaluate_recipe(records, outputs, {"unused": np.zeros((1,), np.float16)})
    assert actual[0].tobytes() == actual[1].tobytes() == b"\x00\x80" * 6
    assert actual[2].shape == () and actual[2].tobytes() == b"\x00\x80"


def test_explicit_alias_is_available_to_downstream_select(tmp_path):
    graph = Graph("alias_consumer")
    condition = graph.input("condition", (2, 3), "bool")
    then = graph.input("then", (2, 3))
    first = graph.const(np.float16(0))
    second = graph.const(np.float16(0))
    a = graph.op("broadcast_to", first, shape=(2, 3), attrs={"shape": (2, 3)})
    b = graph.op("broadcast_to", second, shape=(2, 3), attrs={"shape": (2, 3)})
    graph.outputs = [a, graph.op("where", condition, then, b, shape=(2, 3))]
    records, _, outputs = _serialize_both(graph, tmp_path)
    selected, = _assert_select_contract(records)
    assert selected["inputs"][2]["src"] == outputs[0][0]
    assert not any(record["type"] == "BroadcastTo" for record in records)

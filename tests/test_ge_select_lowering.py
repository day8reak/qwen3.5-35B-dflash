"""Inspect serialized GE Select inputs and execute their exact mask semantics.

This is host-only validation of the generated recipe, not an OMG/device test.
"""
from pathlib import Path
import struct

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_export import build_pair, configurations
from qwen35_moe_dflash.ge_ir import DTYPES, Graph


class RecipeReader:
    """Independent reader for the v1/v2 files consumed by the C++ exporter."""

    def __init__(self, path):
        self.data = path.read_bytes()
        self.offset = 0

    def take(self, size):
        value = self.data[self.offset:self.offset + size]
        assert len(value) == size
        self.offset += size
        return value

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.take(self.number("I")).decode()

    def descriptor(self):
        dtype = {1: "float16", 2: "int32", 3: "int64", 4: "bool"}[self.number("B")]
        layout = {0: "ND", 1: "NCHW"}[self.number("B")] if self.version == 2 else None
        shape = tuple(self.number("q") for _ in range(self.number("I")))
        return {"shape": shape, "dtype": dtype, **({"format": layout} if layout else {})}

    def attribute(self):
        kind = self.number("B")
        if kind in (1, 2, 3, 5):
            return self.number({1: "q", 2: "f", 3: "B", 5: "B"}[kind])
        if kind == 4:
            return [self.number("q") for _ in range(self.number("I"))]
        assert kind == 6
        return self.string()

    def read(self):
        self.version = {b"Q35GE001": 1, b"Q35GE002": 2}[self.take(8)]
        self.string()  # graph name
        records = []
        for _ in range(self.number("I")):
            record = {"name": self.string(), "type": self.string(), "inputs": [], "outputs": []}
            for _ in range(self.number("I")):
                port = {"port": self.string(), "src": self.number("I"), "src_port": self.string()}
                port.update(self.descriptor())
                record["inputs"].append(port)
            for _ in range(self.number("I")):
                port = {"port": self.string()}
                port.update(self.descriptor())
                record["outputs"].append(port)
            attrs = {}
            for _ in range(self.number("I")):
                key = self.string()
                attrs[key] = self.attribute()
            record["attrs"] = attrs
            constant = self.number("B")
            assert constant in (0, 1, 2)
            if constant:
                record["constant"] = (self.string(), self.number("Q"), self.number("Q"))
            records.append(record)
        inputs = [self.number("I") for _ in range(self.number("I"))]
        outputs = [(self.number("I"), self.number("I")) for _ in range(self.number("I"))]
        assert self.offset == len(self.data)
        return records, inputs, outputs


def _assert_select_contract(records):
    selected = [record for record in records if record["type"] == "SelectV2"]
    assert selected
    for record in selected:
        condition, then, other = record["inputs"]
        result, = record["outputs"]
        assert [port["port"] for port in record["inputs"]] == ["condition", "then", "else"]
        assert result["port"] == "result"
        assert condition["dtype"] == "bool"
        assert then["dtype"] == other["dtype"] == result["dtype"]
        assert condition["shape"] == then["shape"] == other["shape"] == result["shape"]
        for port in record["inputs"]:
            source = next(item for item in records[port["src"]]["outputs"]
                          if item["port"] == port["src_port"])
            assert (source["shape"], source["dtype"]) == (port["shape"], port["dtype"])
    return selected


def _evaluate_recipe(records, outputs, feeds):
    values = {}
    for index, record in enumerate(records):
        args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
        output, = record["outputs"]
        kind = record["type"]
        if kind == "Data":
            value = feeds[record["name"]]
        elif kind == "Const":
            path, offset, size = record["constant"]
            with Path(path).open("rb") as stream:
                stream.seek(offset)
                payload = stream.read(size)
            assert len(payload) == size
            value = np.frombuffer(payload, dtype=DTYPES[output["dtype"]]).reshape(output["shape"])
        elif kind == "Cast":
            cast_dtype = {1: "float16", 3: "int32", 9: "int64", 12: "bool"}[record["attrs"]["dst_type"]]
            assert cast_dtype == output["dtype"]
            value = args[0].astype(DTYPES[cast_dtype])
        elif kind == "BroadcastTo":
            value = np.broadcast_to(args[0], tuple(int(d) for d in args[1]))
        elif kind == "GatherV2":
            value = np.take(args[0], args[1], axis=int(args[2]))
        elif kind == "SelectV2":
            assert args[0].shape == args[1].shape == args[2].shape
            value = np.where(*args)
        else:
            raise AssertionError(f"unexpected operation in mask fixture: {kind}")
        assert value.shape == output["shape"]
        assert value.dtype == DTYPES[output["dtype"]]
        values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


def _mask_cases():
    yield "scalar-branches", np.array([[[[True, False], [False, True]]]]), np.array(-0.0, np.float16), np.array(-65504, np.float16)
    visible = np.array([[[[True, False, True], [False, True, False]]]])
    scores = np.arange(12, dtype=np.float16).reshape(1, 2, 2, 3)
    scores[~np.broadcast_to(visible, scores.shape)] = np.nan
    yield "draft-heads", visible, scores, np.array(-np.inf, np.float16)
    yield "gdn-triangle", np.tril(np.ones((3, 3), dtype=bool)), np.arange(36, dtype=np.float16).reshape(1, 2, 2, 3, 3), np.array(0, np.float16)
    yield "moe-rows", np.array([[True], [False], [True]]), np.array([[0, -0.0, 1], [np.nan, np.inf, -np.inf], [2, 3, 4]], np.float16), np.array(0, np.float16)
    yield "boolean-output", np.array([[True], [False]]), np.array([[True, False, True], [True, True, True]]), np.array(False)


MASK_CASES = list(_mask_cases())


@pytest.mark.parametrize("name,condition,then,other", MASK_CASES, ids=[case[0] for case in MASK_CASES])
def test_serialized_select_preserves_exact_mask_values(tmp_path, name, condition, then, other):
    graph = Graph(name)
    inputs = [graph.input("condition", condition.shape, "bool"),
              graph.input("then", then.shape, "bool" if then.dtype == np.bool_ else "float16"),
              graph.const(other, "bool" if other.dtype == np.bool_ else "float16")]
    expected = np.where(condition, then, other)
    dtype = "bool" if expected.dtype == np.bool_ else "float16"
    graph.outputs = [graph.op("where", *inputs, shape=expected.shape, dtype=dtype)]
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records, _, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    _assert_select_contract(records)
    actual, = _evaluate_recipe(records, outputs, {"condition": condition, "then": then})
    # Bit comparison covers signed zero, -inf, and NaNs excluded by the mask.
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
    broadcast_count = sum(record["type"] == "BroadcastTo" for record in records)
    # Only dynamic inputs require runtime expansion; literal branches are
    # serialized already expanded to avoid CPUCL FP16 constant execution.
    assert broadcast_count == sum(value.shape != expected.shape for value in (condition, then)) - (name == "draft-heads")
    unrolled = emit_cpp(graph)
    assert unrolled.count(', "BroadcastTo",') == broadcast_count
    assert unrolled.count(', "SelectV2",') == 1
    if dtype == "bool":
        select, = [record for record in records if record["type"] == "SelectV2"]
        assert select["outputs"][0]["dtype"] == "int32"
        assert records[outputs[0][0]]["type"] == "Cast"


@pytest.mark.parametrize("graph_index", [0, 1], ids=["target", "draft"])
def test_entire_tiny_pair_serializes_equal_shaped_select_inputs(tmp_path, graph_index):
    graph = build_pair(*configurations(tiny=True), 8)[graph_index]
    paths = {key: tmp_path / f"weight-{index}.bin" for index, key in enumerate(graph.weights)}
    # External references keep this descriptor audit independent of payload generation.
    emit_cpp(graph, paths, external_weights=True, recipe_dir=tmp_path / "recipe",
             fold_constant_arithmetic=False)
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    selected = _assert_select_contract(records)
    assert len(selected) == sum(node.kind == "where" for node in graph.nodes)


@pytest.mark.parametrize("shapes", [((2, 3), (2, 3), (3,), (2, 2)),
                                    ((1, 2, 3), (2, 3), (), (2, 3))])
def test_select_rejects_inputs_that_cannot_broadcast_to_declared_output(tmp_path, shapes):
    graph = Graph("invalid_shape")
    condition = graph.input("condition", shapes[0], "bool")
    then = graph.input("then", shapes[1])
    other = graph.input("other", shapes[2])
    graph.outputs = [graph.op("where", condition, then, other, shape=shapes[3])]
    with pytest.raises(ValueError):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("condition_dtype,then_dtype,other_dtype,output_dtype", [
    ("int32", "float16", "float16", "float16"),
    ("bool", "float16", "int32", "float16"),
    ("bool", "int32", "int32", "float16"),
    ("bool", "bool", "int32", "bool"),
])
def test_select_rejects_invalid_condition_or_branch_dtypes(
        tmp_path, condition_dtype, then_dtype, other_dtype, output_dtype):
    graph = Graph("invalid_dtype")
    condition = graph.input("condition", (2, 2), condition_dtype)
    then = graph.input("then", (2, 2), then_dtype)
    other = graph.input("other", (2, 2), other_dtype)
    graph.outputs = [graph.op("where", condition, then, other, shape=(2, 2), dtype=output_dtype)]
    with pytest.raises(ValueError, match="BOOL condition|matching branch dtypes"):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()

"""Exact row-Mul reshape-chain rewrites, with host-only recipe/CPP checks."""
import pickle
import re

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_ir import DTYPES, Graph
from qwen35_moe_dflash.ge_recipe import _validate
from qwen35_moe_dflash.ge_reshape_chains import compact_row_mul_reshape_chains
from test_ge_select_lowering import RecipeReader


def _attr(value):
    return {"type": "bool" if isinstance(value, bool) else "int", "value": value}


class Records:
    def __init__(self):
        self.records, self.inputs = [], []

    def add(self, name, kind, shape, sources=(), ports=(), dtype="float16", attrs=None, constant=None):
        inputs = []
        for source, port in zip(sources, ports):
            output, = self.records[source]["outputs"]
            inputs.append({**output, "port": port, "src": source, "src_port": output["port"]})
        self.records.append({"name": name, "type": kind, "inputs": inputs,
                             "outputs": [{"port": "y", "shape": list(shape), "dtype": dtype}],
                             "attrs": attrs or {}, "constant": constant})
        return len(self.records) - 1

    def data(self, name, shape, dtype="float16"):
        index = self.add(name, "Data", shape, dtype=dtype, attrs={"index": _attr(len(self.inputs))})
        self.inputs.append(index)
        return index

    def const(self, name, array, dtype):
        array = np.asarray(array, dtype=DTYPES[dtype])
        return self.add(name, "Const", array.shape, dtype=dtype, constant={"data": array.tobytes()})

    def reshape(self, name, source, shape, dtype=None):
        shape_index = self.const(name + "/shape", shape, "int32")
        dtype = dtype or self.records[source]["outputs"][0]["dtype"]
        return self.add(name, "Reshape", shape, (source, shape_index), ("x", "shape"), dtype,
                        {"axis": _attr(0), "num_axes": _attr(-1)})


def _run(builder, outputs, seeds=("m",), validate=True):
    before = pickle.dumps(builder.records)
    input_before, output_before = list(builder.inputs), list(outputs)
    result = compact_row_mul_reshape_chains(builder.records, builder.inputs, outputs, seeds)
    assert len(result) == 4
    records, inputs, pairs, report = result
    assert pickle.dumps(builder.records) == before
    assert builder.inputs == input_before and outputs == output_before
    assert [records[i]["name"] for i in inputs] == [builder.records[i]["name"] for i in builder.inputs]
    assert [(records[i]["name"], p) for i, p in pairs] == [(builder.records[i]["name"], p) for i, p in outputs]
    if validate:
        _validate(records, "rewritten", inputs, pairs)
    return records, inputs, pairs, report


def _named(records, name):
    return next(record for record in records if record["name"] == name)


def _source(records, name, port="x"):
    edge = next(edge for edge in _named(records, name)["inputs"] if edge["port"] == port)
    return records[edge["src"]]["name"], edge


def _copy_recipe(records, outputs, feeds):
    values = {}
    for index, record in enumerate(records):
        output, = record["outputs"]
        args = [values[port["src"]] for port in record["inputs"]]
        if record["type"] == "Data":
            value = feeds[record["name"]]
        elif record["type"] == "Const":
            value = np.frombuffer(record["constant"]["data"], DTYPES[output["dtype"]]).reshape(output["shape"])
        else:
            assert record["type"] == "Reshape"
            assert args[1].dtype == np.int32
            value = args[0].reshape(tuple(args[1]))
        assert value.shape == tuple(output["shape"]) and value.dtype == DTYPES[output["dtype"]]
        values[index] = value
    return [values[index] for index, port in outputs]


@pytest.mark.parametrize("side,source_shape,view_shape,final_shape", [
    ("left", (1, 8, 16), (1, 8, 2, 8), (16, 8)),
    ("right", (1, 2, 64), (1, 2, 64, 1), (128, 1)),
])
def test_input_view_chain_keeps_outer_name_int32_shape_and_every_element_bit(side, source_shape, view_shape, final_shape):
    builder = Records()
    data = builder.data("input", source_shape)
    inner = builder.reshape("input_view", data, view_shape)
    name = f"__ge/m/row_mul/{side}"
    outer = builder.reshape(name, inner, final_shape)
    records, _, outputs, report = _run(builder, [(outer, 0)])
    assert _source(records, name)[0] == "input"
    assert _source(records, name)[1]["shape"] == list(source_shape)
    assert "input_view" not in {record["name"] for record in records}
    assert report["counts"]["rewrites"] >= 1
    shape_source = _named(records, name)["inputs"][1]["src"]
    assert records[shape_source]["outputs"][0]["dtype"] == "int32"
    assert records[shape_source]["constant"]["data"] == np.array(final_shape, "<i4").tobytes()
    bits = np.resize(np.array([0, 0x8000, 0x7e31, 0x7d01, 0xfe07, 0x7c00,
                               0xfc00, 1, 0x8001, 0x3c00], np.uint16), source_shape)
    actual, = _copy_recipe(records, outputs, {"input": bits.view(np.float16)})
    np.testing.assert_array_equal(actual.view(np.uint16), bits.reshape(final_shape))


def test_output_restore_chain_into_bmm_helper_stops_at_mul():
    builder = Records()
    left, right = builder.data("left", (128, 8)), builder.data("right", (128, 1))
    product = builder.add("__ge/m/row_mul/product", "Mul", (128, 8), (left, right), ("x1", "x2"))
    restored = builder.reshape("m", product, (1, 2, 64, 8))
    view = builder.reshape("v90", restored, (2, 1, 64, 8))
    helper = builder.reshape("__ge/bmm/batch_left", view, (2, 64, 8))
    rhs = builder.data("bmm_right", (2, 8, 4))
    output = builder.add("bmm", "BatchMatMulV2", (2, 64, 4), (helper, rhs), ("x1", "x2"))
    records, _, _, _ = _run(builder, [(output, 0)])
    assert _source(records, "__ge/bmm/batch_left")[0] == "__ge/m/row_mul/product"
    assert _source(records, "bmm", "x1")[0] == "__ge/bmm/batch_left"
    assert not {"m", "v90"} & {record["name"] for record in records}
    product_record = _named(records, "__ge/m/row_mul/product")
    assert product_record["type"] == "Mul"
    assert [records[edge["src"]]["name"] for edge in product_record["inputs"]] == ["left", "right"]


@pytest.mark.parametrize("keep", ["consumer", "graph_output", "both"])
def test_shared_parent_is_not_globally_aliased_or_deleted(keep):
    builder = Records()
    data = builder.data("beta", (1, 2, 64))
    inner = builder.reshape("beta_view", data, (1, 2, 64, 1))
    name = "__ge/m/row_mul/right"
    child = builder.reshape(name, inner, (128, 1))
    outputs = [(child, 0)]
    if keep in {"consumer", "both"}:
        other = builder.data("other", (1, 2, 64, 1))
        added = builder.add("sum", "Add", (1, 2, 64, 1), (inner, other), ("x1", "x2"))
        outputs.append((added, 0))
    if keep in {"graph_output", "both"}:
        # Repeated ordered boundary entries must not be collapsed into an alias.
        outputs.extend([(inner, 0), (inner, 0)])
    records, _, _, _ = _run(builder, outputs)
    assert _source(records, name)[0] == "beta"
    assert _source(records, "beta_view")[0] == "beta"
    if keep in {"consumer", "both"}:
        assert _source(records, "sum", "x1")[0] == "beta_view"
        assert _source(records, "sum", "x1")[1]["shape"] == [1, 2, 64, 1]


def test_output_view_branches_keep_each_output_name_and_shape():
    builder = Records()
    data = builder.data("product", (128, 8))
    restored = builder.reshape("m", data, (1, 2, 64, 8))
    a = builder.reshape("output_a", restored, (2, 1, 64, 8))
    b = builder.reshape("output_b", restored, (128, 8))
    records, _, pairs, _ = _run(builder, [(a, 0), (b, 0)])
    assert [records[i]["name"] for i, _ in pairs] == ["output_a", "output_b"]
    assert _source(records, "output_a")[0] == _source(records, "output_b")[0] == "product"
    # Even the identity-shaped outer view remains the named output operation.
    assert _named(records, "output_b")["type"] == "Reshape"


@pytest.mark.parametrize("barrier", ["Transpose", "Mul"])
def test_chain_does_not_cross_a_transpose_or_arithmetic_boundary(barrier):
    builder = Records()
    data = builder.data("input", (2, 8, 8))
    before = builder.reshape("native_view", data, (1, 2, 64))
    if barrier == "Transpose":
        parameter = builder.const("perm", [1, 0, 2], "int32")
        source = builder.add("barrier", barrier, (2, 1, 64), (before, parameter), ("x", "perm"))
    else:
        rhs = builder.data("factor", (1, 2, 64))
        source = builder.add("barrier", barrier, (1, 2, 64), (before, rhs), ("x1", "x2"))
    inner = builder.reshape("__ge/m/row_mul/right", source, (128, 1))
    output = builder.reshape("output", inner, (2, 64))
    records, _, _, _ = _run(builder, [(output, 0)])
    assert _source(records, "output")[0] == "barrier"
    assert _source(records, "barrier", "x" if barrier == "Transpose" else "x1")[0] == "native_view"
    assert _source(records, "native_view")[0] == "input"


@pytest.mark.parametrize("kind", ["Const", "FileConstant"])
def test_constant_data_root_does_not_require_opening_external_weights(monkeypatch, kind):
    builder = Records()
    builder.data("unused_input_boundary", (1,))
    if kind == "Const":
        source = builder.const("constant_data", np.arange(128).reshape(1, 2, 64), "float16")
    else:
        source = builder.add("constant_data", "FileConstant", (1, 2, 64), attrs={
            "file_path": {"type": "string", "value": "/declared/not-read.bin"},
            "shape": {"type": "ints", "value": [1, 2, 64]},
            "dtype": {"type": "dtype", "value": "float16"}})
    inner = builder.reshape("view", source, (1, 2, 64, 1))
    outer = builder.reshape("__ge/m/row_mul/right", inner, (128, 1))
    monkeypatch.setattr("builtins.open", lambda *a, **kw: pytest.fail("canonicalizer must not read weight files"))
    records, inputs, _, _ = _run(builder, [(outer, 0)])
    assert _source(records, "__ge/m/row_mul/right")[0] == "constant_data"
    assert [records[i]["name"] for i in inputs] == ["unused_input_boundary"]
    assert _named(records, "constant_data")["type"] == kind


def test_unrelated_native_reshape_chain_is_preserved():
    builder = Records()
    data = builder.data("data", (1, 2, 64))
    inner = builder.reshape("native_inner", data, (1, 2, 64, 1))
    native_output = builder.reshape("native_output", inner, (128, 1))
    row_data = builder.data("row_data", (1, 2, 64))
    row_inner = builder.reshape("row_inner", row_data, (1, 2, 64, 1))
    row_output = builder.reshape("__ge/m/row_mul/right", row_inner, (128, 1))
    records, _, _, _ = _run(builder, [(native_output, 0), (row_output, 0)])
    assert _source(records, "native_output")[0] == "native_inner"
    assert _source(records, "native_inner")[0] == "data"
    assert _source(records, "__ge/m/row_mul/right")[0] == "row_data"


@pytest.mark.parametrize("fault", ["int64_shape", "shape_words", "zero", "numel", "dtype",
                                  "partial_axis", "partial_num_axes", "unknown_attribute",
                                  "dynamic_shape", "file_shape"])
def test_unsupported_or_invalid_parent_is_not_bypassed(fault):
    builder = Records()
    data = builder.data("data", (1, 2, 64))
    replacement = None
    if fault == "dynamic_shape":
        replacement = builder.data("dynamic_shape", (4,), "int32")
    elif fault == "file_shape":
        replacement = builder.add("file_shape", "FileConstant", (4,), dtype="int32", attrs={
            "file_path": {"type": "string", "value": "/declared/not-read-shape.bin"},
            "shape": {"type": "ints", "value": [4]}, "dtype": {"type": "dtype", "value": "int32"}})
    shape = (1, 2, 0, 1) if fault == "zero" else (1, 2, 32, 1) if fault == "numel" else (1, 2, 64, 1)
    dtype = "int32" if fault == "dtype" else "float16"
    inner = builder.reshape("parent", data, shape, dtype)
    child = builder.reshape("__ge/m/row_mul/right", inner, (int(np.prod(shape)), 1), dtype)
    parent = builder.records[inner]
    shape_port = parent["inputs"][1]
    parameter = builder.records[shape_port["src"]]
    if fault == "int64_shape":
        parameter["outputs"][0]["dtype"] = shape_port["dtype"] = "int64"
        parameter["constant"]["data"] = np.array(shape, "<i8").tobytes()
    elif fault == "shape_words":
        parameter["constant"]["data"] = np.array([1, 1, 128, 1], "<i4").tobytes()
    elif fault in {"partial_axis", "partial_num_axes"}:
        parent["attrs"]["axis" if fault == "partial_axis" else "num_axes"] = _attr(1)
    elif fault == "unknown_attribute":
        parent["attrs"]["unknown"] = _attr(True)
    elif replacement is not None:
        desc = builder.records[replacement]["outputs"][0]
        parent["inputs"][1] = {**desc, "port": "shape", "src": replacement, "src_port": "y"}
    records, _, _, report = _run(builder, [(child, 0)], validate=False)
    assert _source(records, "__ge/m/row_mul/right")[0] == "parent"
    assert report["counts"]["rewrites"] == 0


@pytest.mark.parametrize("fault", ["source_index", "output_index", "output_port"])
def test_unremappable_indices_are_rejected_without_mutation(fault):
    builder = Records()
    data = builder.data("data", (1, 2, 64))
    inner = builder.reshape("inner", data, (1, 2, 64, 1))
    child = builder.reshape("__ge/m/row_mul/right", inner, (128, 1))
    outputs = [(child, 0)]
    if fault == "source_index":
        builder.records[child]["inputs"][0]["src"] = len(builder.records)
    elif fault == "output_index":
        outputs = [(len(builder.records), 0)]
    else:
        outputs = [(child, 2)]
    original = pickle.dumps(builder.records)
    with pytest.raises(ValueError):
        compact_row_mul_reshape_chains(builder.records, builder.inputs, outputs, ["m"])
    assert pickle.dumps(builder.records) == original


def _integration_graph():
    graph = Graph("row_chain_integration")
    source = graph.input("source", (2, 1, 64, 8))
    beta = graph.input("beta", (1, 2, 64))
    right = graph.input("bmm_right", (2, 1, 8, 4))

    def view(value, name, shape):
        return graph.op("reshape", value, shape=shape, attrs={"shape": list(shape)}, name=name)

    left_view = view(source, "left_view", (1, 2, 64, 8))
    beta_view = view(beta, "beta_view", (1, 2, 64, 1))
    scaled = graph.op("mul", left_view, beta_view, shape=left_view.shape, name="scaled")
    output_view = view(scaled, "output_view", (2, 1, 64, 8))
    result = graph.op("matmul", output_view, right, shape=(2, 1, 64, 4), name="result")
    graph.outputs = [result, output_view, beta_view]
    return graph


def _inline_signature(code):
    nodes = re.findall(r'ops\.push_back\(Op\("([^"]+)", "([^"]+)"', code)
    assert nodes
    edges = re.findall(r'ops\[(\d+)\]\.SetInput\("([^"]+)", ops\[(\d+)\], "([^"]+)"\)', code)
    return nodes, [(nodes[int(target)][0], port, nodes[int(source)][0], source_port)
                   for target, port, source, source_port in edges]


def test_compact_emitter_uses_canonical_edges_in_recipe_and_inline_cpp(tmp_path):
    graph = _integration_graph()
    before, report = pickle.dumps(graph), {}
    emit_cpp(graph, recipe_dir=tmp_path / "recipe", fp16_row_mul="compact", row_mul_report=report)
    records, _, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert report["counts"]["replaced"] == 1
    assert report["reshape_chains"]["counts"]["rewrites"] > 0
    assert _source(records, "__ge/scaled/row_mul/left")[0] == "source"
    assert _source(records, "__ge/scaled/row_mul/right")[0] == "beta"
    assert _source(records, "__ge/result/batch_left")[0] == "__ge/scaled/row_mul/product"
    assert _source(records, "output_view")[0] == "__ge/scaled/row_mul/product"
    assert [records[i]["name"] for i, _ in outputs] == ["result", "output_view", "beta_view"]
    inline_report = {}
    code = emit_cpp(graph, fp16_row_mul="compact", row_mul_report=inline_report)
    assert inline_report == report and pickle.dumps(graph) == before
    nodes, edges = _inline_signature(code)
    assert nodes == [(record["name"], record["type"]) for record in records]
    assert edges == [(record["name"], port["port"], records[port["src"]]["name"], port["src_port"])
                     for record in records for port in record["inputs"]]
    assert emit_cpp(graph) == emit_cpp(graph, fp16_row_mul="native")


def test_native_emitter_preserves_unrelated_original_reshape_chains(tmp_path):
    graph = _integration_graph()
    report = {}
    emit_cpp(graph, recipe_dir=tmp_path / "native", row_mul_report=report)
    records, _, _ = RecipeReader(tmp_path / "native/graph.recipe").read()
    assert _source(records, "__ge/result/batch_left")[0] == "output_view"
    assert report["counts"]["replaced"] == 0


@pytest.mark.parametrize("recipe", [False, True], ids=["inline", "recipe"])
def test_both_emitters_reject_bad_input_descriptor_after_a_valid_rewrite(tmp_path, monkeypatch, recipe):
    from qwen35_moe_dflash import ge_reshape_chains

    original = ge_reshape_chains.compact_row_mul_reshape_chains

    def corrupt_descriptor(*args, **kwargs):
        records, inputs, outputs, report = original(*args, **kwargs)
        assert report["counts"]["rewrites"] > 0
        product = _named(records, "__ge/scaled/row_mul/product")
        # Keep numel and the source index valid, but contradict its actual
        # output descriptor. Rendering must not silently repair this edge.
        product["inputs"][0]["shape"] = [16, 64]
        return records, inputs, outputs, report

    monkeypatch.setattr(ge_reshape_chains, "compact_row_mul_reshape_chains", corrupt_descriptor)
    folder = tmp_path / "invalid-recipe"
    with pytest.raises(ValueError, match="source port or connected descriptor mismatch"):
        emit_cpp(_integration_graph(), fp16_row_mul="compact", recipe_dir=folder if recipe else None)
    assert not folder.exists()

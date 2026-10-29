"""Strict encoded AIR/irpb shape evidence; no native library or model payload."""
import hashlib
import json
from pathlib import Path
import struct

from google.protobuf import descriptor_pb2
import pytest

from qwen35_moe_dflash.target_reshape_metadata import inspect_target_reshape_metadata
from test_air_schema import ge_descriptor, pb2
from test_draft_ir_inspect import _attr, _blob, _desc, _int
from test_original_air_inspect import _node, _scalar_node, selected
import test_draft_ir_inspect as fixture


def op(name, kind, inputs=(), inputs_desc=(), outputs_desc=()):
    return (_blob(1, name) + _blob(2, kind)
            + b"".join(_blob(5, edge) for edge in inputs)
            + b"".join(_blob(33, desc) for desc in inputs_desc)
            + b"".join(_blob(34, desc) for desc in outputs_desc))


@pytest.fixture
def schemas(tmp_path, selected, monkeypatch):
    ddk, library = selected
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(fixture._schema())
    enum = next(item for item in descriptor.enum_type if item.name == "DataType")
    for name, number in [("DT_INT32", 42), ("DT_INT64", 91), ("DT_INT8", 3)]:
        item = enum.value.add()
        item.name, item.number = name, number
    monkeypatch.setattr(fixture, "_schema", lambda: descriptor.SerializeToString())
    fixture._library(library)
    ge = pb2(tmp_path / "ge_ir_pb2.py", ge_descriptor())
    return ddk, ge


def graph(*, dtype=91, data=None, shape=(2,), port_shape="same", values=(64, 2048), external=False,
          src=False, disagree=False, actual_type="Reshape", fanout=0):
    hidden = _desc(dtype=1, shape=(1, 64, 2048))
    flat = _desc(dtype=1, shape=(64, 2048))
    integer = _desc(dtype=dtype, shape=shape)
    out_shape = shape if port_shape == "same" else port_shape
    if data is None:
        data = struct.pack("<" + ("i" if dtype == 42 else "q") * len(values), *values)
    nodes = [_node("embedding", "Data", [1, 64, 2048], dtype=1),
             _node("gamma", "FileConstant", [2048], dtype=1) + _attr("file_path", _blob(2, "./weights/input_norm")),
             op("norm", "RmsNorm", ["embedding:0", "gamma:0"], [hidden, _desc(dtype=1, shape=(2048,))], [hidden])]
    if external:
        nodes.append(_node("shape", "FileConstant", out_shape, dtype=dtype)
                     + _attr("file_path", _blob(2, "./private-shape-payload")))
    else:
        nodes.append(_node("shape", "Const", out_shape, dtype=dtype)
                     + _attr("value", _blob(12, _blob(1, integer) + _blob(2, data))))
    reshape = op("Reshape", actual_type, [] if src else ["norm:0", "shape:0"], [hidden, integer], [flat])
    if src or disagree:
        reshape += _blob(24, "norm") + _blob(24, "wrong" if disagree else "shape") + _int(25, 0) + _int(25, 0)
    nodes.extend([reshape, op("quant", "DynamicQuant", ["Reshape:0"], [flat], [flat]),
                  _node("projection_weight", "FileConstant", [2048, 8192], dtype=3)
                  + _attr("file_path", _blob(2, "./weights/layer0/in_proj_qkv")),
                  op("qmm", "QuantBatchMatMulV3", ["quant:0", "projection_weight:0"], [flat, _desc(dtype=3, shape=(2048, 8192))], [_desc(dtype=1, shape=(64, 8192))]),
                  op("Reshape_1", "Reshape", ["qmm:0", "shape:0"], [flat, integer], [flat])])
    for index in range(fanout):
        nodes.append(op(f"fan_{index}", "Consumer", ["Reshape:0"], [flat], [flat]))
    return _blob(1, "exact-source-graph") + b"".join(_blob(6, row) for row in nodes)


def inspect(tmp_path, schemas, raw, kind="hiai", **options):
    path = tmp_path / "input.irpb"
    path.write_bytes(raw)
    ddk, ge = schemas
    kwargs = {"ddk_root": ddk} if kind == "hiai" else {"torchair_pb2": ge}
    return inspect_target_reshape_metadata(path, **kwargs, **options)


@pytest.mark.parametrize("root", ["ModelDef", "GraphDef"])
@pytest.mark.parametrize("kind", ["hiai", "torchair"])
@pytest.mark.parametrize("dtype", [42, 91])
def test_selected_schema_controls_integer_type_and_exact_small_shape_bytes(tmp_path, schemas, root, kind, dtype):
    body = graph(dtype=dtype)
    raw = _blob(7, body) if root == "ModelDef" else body
    result = inspect(tmp_path, schemas, raw, kind)
    assert result["status"] == "INSPECTED", result
    assert result["protobuf_decode"] == "PASSED" and result["observed_format"] == root
    assert result["matching_node_count"] == 1
    match = result["matches"][0]
    assert match["node_name"] == "Reshape" and match["node_type"] == "Reshape"
    assert match["data_edge"]["reference"] == "norm:0" and match["shape_edge"]["reference"] == "shape:0"
    small = match["shape_constant"]
    assert small["status"] == "DECODED_TINY_INTEGER_TENSOR" and small["decoded_values"] == [64, 2048]
    assert small["descriptor"]["raw_dtype"] == dtype
    assert small["descriptor"]["schema_dtype_names"] == ["DT_INT32" if dtype == 42 else "DT_INT64"]
    assert small["data"]["bytes"] == (8 if dtype == 42 else 16)
    assert small["data"]["sha256"] == hashlib.sha256(struct.pack("<ii" if dtype == 42 else "<qq", 64, 2048)).hexdigest()
    assert any(node["name"] == "qmm" for node in match["neighborhood"]["nodes"])
    assert any(node["file_path"] == "./weights/layer0/in_proj_qkv" for node in match["file_constant_operands"])
    assert result["dynamic_gear_semantics"] == "UNVERIFIED" and result["graph_changes"] == "NONE"
    assert len(json.dumps(result)) < 50000


@pytest.mark.parametrize("options,reason", [
    ({"shape": ()}, "explicit nonempty dimensions"),
    ({"shape": (-1,)}, "explicit nonempty dimensions"),
    ({"shape": (9,), "values": tuple(range(9))}, "1..8 elements"),
    ({"dtype": 1}, "INT32 or INT64"),
    ({"data": b"short"}, "byte count"),
    ({"port_shape": (3,)}, "dimensions disagree"),
    ({"external": True}, "embedded Const"),
    ({"disagree": True}, "disagree"),
    ({"actual_type": "Cast"}, "not Reshape"),
])
def test_no_shape_values_are_invented_for_unknown_or_inconsistent_const_metadata(tmp_path, schemas, options, reason):
    result = inspect(tmp_path, schemas, _blob(7, graph(**options)))
    constant = result["matches"][0]["shape_constant"]
    assert constant["status"] == "UNRESOLVED" and "decoded_values" not in constant
    assert reason in constant["reason"], constant


def test_sparse_output_dimensions_remain_unresolved_but_explicit_tensor_data_can_be_decoded(tmp_path, schemas):
    result = inspect(tmp_path, schemas, _blob(7, graph(port_shape=())))
    match = result["matches"][0]
    assert match["shape_constant"]["decoded_values"] == [64, 2048]
    assert match["shape_constant"]["output_dimensions_status"].startswith("UNRESOLVED")
    assert match["shape_edge"]["producer_descriptor"]["shape"] == []
    assert "NOT_ASSUMED_SCALAR" in match["shape_edge"]["producer_descriptor"]["shape_semantics"]


def test_hiai_src_name_port_representation_is_explicitly_resolved(tmp_path, schemas):
    result = inspect(tmp_path, schemas, _blob(7, graph(src=True)))
    assert result["matches"][0]["shape_edge"]["representation"] == "src_name/src_index"
    assert result["matches"][0]["shape_constant"]["decoded_values"] == [64, 2048]


def test_all_nested_graphs_are_separate_domains_and_more_than_eight_matches_are_bounded(tmp_path, schemas):
    holder = _node("holder", "GraphOp", [1], dtype=1) + _attr("subgraph", _blob(13, graph()))
    nested = _blob(1, "parent") + _blob(6, holder)
    result = inspect(tmp_path, schemas, _blob(7, nested) + b"".join(_blob(7, graph()) for _ in range(9)))
    assert result["status"] == "INSPECTED" and result["matching_node_count"] == 10
    assert result["graph_count"] == 11 and result["matches_truncated"] is True and len(result["matches"]) == 8
    assert any("/attr/" in row["graph_path"] for row in result["matches"])


def test_neighborhood_and_exact_name_matching_are_bounded(tmp_path, schemas):
    result = inspect(tmp_path, schemas, _blob(7, graph(fanout=100)))
    assert len(result["matches"][0]["neighborhood"]["nodes"]) <= 32
    assert result["matches"][0]["neighborhood"]["truncated"] is True
    missing = inspect(tmp_path, schemas, _blob(7, graph()), node_name="Reshape_missing")
    assert missing["status"] == "NODE_NOT_FOUND" and missing["matches"] == []


def test_no_weights_or_vendor_code_is_opened_and_source_bytes_are_unchanged(tmp_path, schemas, monkeypatch):
    payload = tmp_path / "private-shape-payload"
    payload.write_bytes(b"PRIVATE_BYTES_MUST_NOT_BE_OPENED")
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if path == payload:
            pytest.fail("FileConstant must never be opened")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    raw = _blob(7, graph(external=True))
    result = inspect(tmp_path, schemas, raw)
    assert result["status"] == "INSPECTED"
    assert (tmp_path / "input.irpb").read_bytes() == raw
    assert "PRIVATE_BYTES" not in json.dumps(result)
    assert result["source_preservation"] == "UNCHANGED_BY_STAT"


def test_unknown_wire_or_duplicate_operation_names_do_not_produce_shape_evidence(tmp_path, schemas):
    unknown = inspect(tmp_path, schemas, _blob(7, graph()) + _blob(199, b""))
    assert unknown["status"] == "OPAQUE_SCHEMA_MISMATCH" and unknown["matches"] == []
    duplicate = inspect(tmp_path, schemas, _blob(7, graph() + _blob(6, _node("Reshape", "Reshape", [1]))))
    assert duplicate["status"] == "DECODED_METADATA_UNSUPPORTED" and duplicate["matches"] == []
    assert duplicate["protobuf_decode"] == "PASSED"


def test_oversize_file_is_rejected_before_opening_contents(tmp_path, schemas, monkeypatch):
    path = tmp_path / "large.irpb"
    with path.open("wb") as stream:
        stream.truncate((64 << 20) + 1)
    original_open = Path.open
    def guarded(selected, *args, **kwargs):
        if selected == path:
            pytest.fail("oversize contents must not be opened")
        return original_open(selected, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    result = inspect_target_reshape_metadata(path, ddk_root=schemas[0])
    assert result["status"] == "UNKNOWN" and "64 MiB" in result["error"]

"""Schema-bound protobuf decoding without executing a DDK library or compiler."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from qwen35_moe_dflash import omc_wire


SYMBOL = "_ZL49descriptor_table_protodef_proto_2fge_5fir_2eproto"


def varint(value):
    value &= (1 << 64) - 1
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def integer(number, value):
    return varint(number << 3) + varint(value)


def blob(number, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((number << 3) | 2) + varint(len(value)) + value


def fixed(number, value, fmt):
    wire = 1 if struct.calcsize(fmt) == 8 else 5
    return varint((number << 3) | wire) + struct.pack(fmt, value)


def field(name, number, kind, repeated=False, reference="", oneof=None):
    result = (blob(1, name) + integer(3, number) + integer(4, 3 if repeated else 1)
              + integer(5, kind))
    if reference:
        result += blob(6, ".hiai.proto." + reference)
    if oneof is not None:
        result += integer(9, oneof)
    return blob(2, result)


def message(name, fields, nested=(), oneofs=()):
    return (blob(1, name) + b"".join(fields)
            + b"".join(blob(3, child) for child in nested)
            + b"".join(blob(8, blob(1, item)) for item in oneofs))


def descriptor():
    entry = message("AttrEntry", [field("key", 1, 9),
        field("value", 2, 11, reference="AttrDef")])
    list_value = message("ListValue", [field("s", 2, 12, True),
        field("i", 3, 3, True), field("f", 4, 2, True)])
    messages = [
        message("ModelDef", [field("graph", 7, 11, True, "GraphDef")]),
        message("GraphDef", [field("name", 1, 9),
            field("op", 6, 11, True, "OpDef")]),
        message("OpDef", [field("name", 1, 9), field("type", 2, 9),
            field("attr", 10, 11, True, "OpDef.AttrEntry"),
            field("src_name", 24, 9, True), field("src_index", 25, 3, True),
            field("input_desc", 33, 11, True, "TensorDescriptor")], [entry]),
        message("AttrDef", [field("s", 2, 12), field("i", 3, 3),
            field("f", 4, 2), field("b", 5, 8),
            field("g", 13, 11, reference="GraphDef"),
            field("list", 1, 11, reference="AttrDef.ListValue")], [list_value]),
        message("TensorDescriptor", [field("dtype", 2, 14, reference="DataType"),
            field("shape", 3, 11, reference="ShapeDef")]),
        message("ShapeDef", [field("dim", 1, 3, True)]),
        message("TensorDef", [field("data", 2, 12)]),
        message("Recursive", [field("child", 1, 11, reference="Recursive")]),
        message("Numeric", [field("double", 1, 1), field("float", 2, 2),
            field("int64", 3, 3), field("uint64", 4, 4), field("int32", 5, 5),
            field("fixed64", 6, 6), field("fixed32", 7, 7), field("bool", 8, 8),
            field("uint32", 13, 13), field("sfixed32", 15, 15),
            field("sfixed64", 16, 16), field("sint32", 17, 17),
            field("sint64", 18, 18), field("packed_float", 19, 2, True)]),
        message("Choice", [field("one", 1, 3, oneof=0),
            field("two", 2, 9, oneof=0)], oneofs=["value"]),
    ]
    enum = blob(1, "DataType") + b"".join(blob(2, blob(1, key) + integer(2, value))
        for key, value in [("UNDEFINED", 0), ("FLOAT", 1), ("HALF", 2), ("NEGATIVE", -1)])
    return (blob(1, "proto/ge_ir.proto") + blob(2, "hiai.proto")
            + b"".join(blob(4, value) for value in messages)
            + blob(5, enum) + blob(12, "proto3"))


def elf(descriptor_data=None, *, duplicate=False, symbol_name=SYMBOL):
    """Place descriptor away from its section start and use a nonzero address."""
    payload = descriptor() if descriptor_data is None else descriptor_data
    rodata = b"prefix!" + payload + b"\0suffix"
    names = b"\0" + symbol_name.encode() + b"\0"
    section_offset, section_address = 0x120, 0x8000
    symbol_value = section_address + 7
    symbol_size = len(payload) + 1
    strings_offset = section_offset + len(rodata)
    symbols_offset = strings_offset + len(names)
    symbols = bytes(24) + struct.pack("<IBBHQQ", 1, 1, 0, 1, symbol_value, symbol_size)
    if duplicate:
        symbols += struct.pack("<IBBHQQ", 1, 1, 0, 1, symbol_value + 1, symbol_size)
    table_offset = symbols_offset + len(symbols)
    sections = [bytes(64),
        struct.pack("<IIQQQQIIQQ", 0, 1, 0, section_address, section_offset,
                    len(rodata), 0, 0, 1, 0),
        struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, strings_offset, len(names), 0, 0, 1, 0),
        struct.pack("<IIQQQQIIQQ", 0, 2, 0, 0, symbols_offset, len(symbols), 2, 0, 8, 24)]
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 3, 62, 1, 0, 0,
                         table_offset, 0, 64, 0, 0, 64, 4, 0)
    data = header + bytes(section_offset - len(header)) + rodata + names + symbols
    return bytearray(data + b"".join(sections)), {
        "descriptor_offset": section_offset + 7, "table_offset": table_offset,
        "symbols_offset": symbols_offset, "symbol_value": symbol_value,
        "symbol_size": symbol_size}


@pytest.fixture
def schema():
    return omc_wire.Schema(descriptor())


def test_elf_symbol_extraction_preserves_exact_schema_evidence(tmp_path):
    data, info = elf()
    path = tmp_path / "declared-library.so"
    path.write_bytes(data)
    schema = omc_wire.load_schema(path)
    evidence = schema.evidence
    assert evidence["library"] == str(path.resolve())
    assert evidence["library_sha256"] == hashlib.sha256(data).hexdigest()
    assert evidence["descriptor_sha256"] == hashlib.sha256(descriptor()).hexdigest()
    assert evidence["symbol_file_offset"] == info["descriptor_offset"]
    assert evidence["symbol_value"] == info["symbol_value"]
    assert evidence["symbol_size"] == info["symbol_size"]
    assert evidence["descriptor_bytes"] == len(descriptor())
    assert schema.enums == {"hiai.proto.DataType": {
        "UNDEFINED": 0, "FLOAT": 1, "HALF": 2, "NEGATIVE": -1}}


def test_cast_attributes_maps_and_nested_graph_keep_raw_type_domains(schema):
    inner = blob(1, "SubGraph_0") + blob(6, blob(1, "nested") + blob(2, "Data"))
    attributes = [("src_dtype", integer(3, 1)), ("dst_dtype", integer(3, 0)),
                  ("subgraph", blob(13, inner))]
    cast = blob(1, "insert_cast_4") + blob(2, "CastT")
    cast += b"".join(blob(10, blob(1, key) + blob(2, value)) for key, value in attributes)
    cast += blob(24, "producer") + blob(25, varint(-1) + varint(0))
    cast += integer(25, 2)
    cast += blob(33, integer(2, 2) + blob(3, blob(1, varint(1) + varint(-1))))
    wire = blob(1, "target") + blob(6, cast)
    result = schema.decode(wire, "GraphDef")
    assert result == schema.decode(wire, ".hiai.proto.GraphDef")
    op = result["op"][0]
    assert op["attr"][:2] == [{"key": "src_dtype", "value": {"i": 1}},
                              {"key": "dst_dtype", "value": {"i": 0}}]
    assert op["input_desc"] == [{"dtype": 2, "shape": {"dim": [1, -1]}}]
    assert op["src_name"] == ["producer"]
    assert op["src_index"] == [-1, 0, 2]
    assert op["attr"][2]["value"]["g"]["op"] == [{"name": "nested", "type": "Data"}]
    assert schema.decode(b"", "GraphDef") == {}  # No fabricated defaults.


def test_numeric_wire_types_and_signed_varints(schema):
    wire = fixed(1, 1.25, "<d") + fixed(2, -0.5, "<f")
    wire += integer(3, -(1 << 63)) + integer(4, (1 << 64) - 1) + integer(5, -7)
    wire += fixed(6, 1 << 63, "<Q") + fixed(7, (1 << 32) - 1, "<I")
    wire += integer(8, 1) + integer(13, (1 << 32) - 1)
    wire += fixed(15, -(1 << 31), "<i") + fixed(16, -(1 << 63), "<q")
    wire += integer(17, 13) + integer(18, (1 << 64) - 1)
    wire += blob(19, struct.pack("<ff", 2.0, -3.5)) + fixed(19, 4.5, "<f")
    assert schema.decode(wire, "Numeric") == {
        "double": 1.25, "float": -0.5, "int64": -(1 << 63),
        "uint64": (1 << 64) - 1, "int32": -7, "fixed64": 1 << 63,
        "fixed32": (1 << 32) - 1, "bool": True, "uint32": (1 << 32) - 1,
        "sfixed32": -(1 << 31), "sfixed64": -(1 << 63), "sint32": -7,
        "sint64": -(1 << 63), "packed_float": [2.0, -3.5, 4.5]}
    assert schema.decode(integer(2, -1), "TensorDescriptor") == {"dtype": -1}
    assert schema.decode(integer(2, 123), "TensorDescriptor") == {"dtype": 123}


@pytest.mark.parametrize("payload", [b"\xff", b"hello\n", b"a" * 257])
def test_attribute_binary_or_large_bytes_are_hashed(schema, payload):
    assert schema.decode(blob(2, payload), "AttrDef") == {
        "s": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}


def test_small_attribute_text_and_tensor_bytes_have_distinct_contracts(schema):
    assert schema.decode(blob(2, "CPUTask"), "AttrDef") == {"s": "CPUTask"}
    assert schema.decode(blob(2, "图"), "AttrDef") == {"s": "图"}
    assert schema.decode(blob(2, b""), "AttrDef") == {"s": ""}
    assert schema.decode(blob(1, blob(2, "one") + blob(2, "two")), "AttrDef") == {
        "list": {"s": ["one", "two"]}}
    payload = b"weights"
    assert schema.decode(blob(2, payload), "TensorDef") == {
        "data": {"bytes": 7, "sha256": hashlib.sha256(payload).hexdigest()}}


def test_nonfinite_float_can_be_serialized_without_json_nan(schema):
    value = schema.decode(fixed(4, float("inf"), "<f"), "AttrDef")
    assert value == {"f": {"nonfinite_float": "inf", "wire_hex": "0000807f"}}
    json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("type_name,wire,match", [
    ("GraphDef", integer(500, 1), "unknown protobuf field"),
    ("GraphDef", integer(1, 1), "wire type mismatch"),
    ("GraphDef", b"\x80", "truncated protobuf varint"),
    ("GraphDef", b"\x80" * 10 + b"\x00", "exceeds uint64"),
    ("GraphDef", b"\x0a\x05ab", "exceeds buffer bounds"),
    ("GraphDef", b"\x0b", "unsupported protobuf wire type"),
    ("GraphDef", b"\x00\x00", "invalid protobuf field number"),
    ("GraphDef", blob(1, b"\xff"), "utf-8"),
    ("GraphDef", blob(1, "one") + blob(1, "two"), "duplicate singular"),
    ("Numeric", blob(19, b"\x00"), "packed protobuf field"),
    ("Numeric", integer(13, 1 << 32), "uint32 field overflow"),
    ("Numeric", integer(17, 1 << 32), "sint32 field overflow"),
    ("Numeric", integer(8, 2), "invalid boolean"),
    ("Choice", integer(1, 0) + blob(2, "two"), "multiple values for oneof"),
])
def test_malformed_or_schema_mismatched_wire_is_rejected(schema, type_name, wire, match):
    with pytest.raises(ValueError, match=match):
        schema.decode(wire, type_name)


def test_depth_and_field_budgets_apply_across_nested_and_packed_fields(schema, monkeypatch):
    wire = b""
    for _ in range(64):
        wire = blob(1, wire)
    schema.decode(wire, "Recursive")
    with pytest.raises(ValueError, match="nesting depth"):
        schema.decode(blob(1, wire), "Recursive")
    monkeypatch.setattr(omc_wire, "_MAX_FIELDS", 4)
    assert schema.decode(blob(1, b"\x00\x01\x02"), "ShapeDef") == {"dim": [0, 1, 2]}
    with pytest.raises(ValueError, match="field budget"):
        schema.decode(blob(1, b"\x00\x01\x02\x03"), "ShapeDef")
    with pytest.raises(ValueError, match="field budget"):
        schema.decode(blob(7, blob(6, blob(1, "a") + blob(2, "Data")))
                      + blob(7, b""), "ModelDef")


@pytest.mark.parametrize("damage,match", [
    ("class", "ELF64 little-endian"), ("endian", "ELF64 little-endian"),
    ("truncated", "ELF section table"), ("section", "ELF section"),
    ("symbol", "owning section"), ("name", "symbol name bounds"),
    ("symtab", "invalid ELF symbol table"), ("duplicate", "exactly one"),
    ("absent", "exactly one"),
])
def test_invalid_elf_metadata_is_rejected_before_schema_decode(tmp_path, damage, match):
    data, info = elf(duplicate=damage == "duplicate", symbol_name="wrong" if damage == "absent" else SYMBOL)
    if damage == "class":
        data[4] = 1
    elif damage == "endian":
        data[5] = 2
    elif damage == "truncated":
        del data[-1:]
    elif damage == "section":
        struct.pack_into("<Q", data, info["table_offset"] + 64 + 32, len(data) + 1)
    elif damage == "symbol":
        struct.pack_into("<Q", data, info["symbols_offset"] + 24 + 16, len(data))
    elif damage == "name":
        struct.pack_into("<I", data, info["symbols_offset"] + 24, len(data))
    elif damage == "symtab":
        struct.pack_into("<Q", data, info["table_offset"] + 3 * 64 + 56, 16)
    path = tmp_path / "bad.so"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=match):
        omc_wire.load_schema(path)


@pytest.mark.parametrize("data,match", [
    (blob(1, "proto/ge_ir.proto") + blob(2, "ge.proto"), "package hiai.proto"),
    (descriptor() + blob(3, "other.proto"), "dependencies"),
    (descriptor() + blob(4, message("GraphDef", [])), "duplicate schema type"),
    (descriptor() + blob(4, message("Bad", [field("x", 1, 11, reference="Absent")])), "unknown message"),
    (descriptor() + blob(4, message("Bad", [field("x", 1, 14, reference="Absent")])), "unknown enum"),
    (descriptor() + blob(4, message("Bad", [field("x", 1, 3), field("y", 1, 3)])), "duplicate field"),
])
def test_schema_identity_and_references_must_be_unambiguous(data, match):
    with pytest.raises(ValueError, match=match):
        omc_wire.Schema(data)


def test_library_and_descriptor_byte_budgets(tmp_path, monkeypatch):
    path = tmp_path / "too-large.so"
    path.write_bytes(b"a" * 33)
    monkeypatch.setattr(omc_wire, "_MAX_LIBRARY", 32)
    with pytest.raises(ValueError, match="library exceeds byte budget"):
        omc_wire.load_schema(path)
    monkeypatch.setattr(omc_wire, "_MAX_DESCRIPTOR", 32)
    with pytest.raises(ValueError, match="descriptor exceeds byte budget"):
        omc_wire.Schema(descriptor())


def test_import_and_real_extraction_path_need_only_python_standard_library(tmp_path):
    path = tmp_path / "schema.so"
    path.write_bytes(elf()[0])
    script = ("import json, sys; from qwen35_moe_dflash.omc_wire import load_schema; "
              "s=load_schema(sys.argv[1]); print(json.dumps(s.decode(bytes.fromhex('0a0178'), 'GraphDef')))")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "model"))
    completed = subprocess.run([sys.executable, "-S", "-c", script, str(path)],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"name": "x"}

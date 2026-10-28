"""Static IR diagnosis uses its selected synthetic schema, never GE enum guesses."""
import hashlib
import json
import struct

import pytest

from qwen35_moe_dflash import draft_ir_inspect as di


def _varint(value):
    value &= (1 << 64) - 1
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result) + bytes([value])


def _int(number, value):
    return _varint(number << 3) + _varint(value)


def _blob(number, value):
    value = value.encode() if isinstance(value, str) else value
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _field(name, number, kind, repeated=False, reference=""):
    value = _blob(1, name) + _int(3, number) + _int(4, 3 if repeated else 1) + _int(5, kind)
    if reference:
        value += _blob(6, ".hiai.proto." + reference)
    return _blob(2, value)


def _message(name, *fields):
    return _blob(4, _blob(1, name) + b"".join(fields))


def _schema():
    messages = [
        _message("ModelDef", _field("graph", 7, 11, True, "GraphDef")),
        _message("GraphDef", _field("name", 1, 9), _field("op", 6, 11, True, "OpDef")),
        _message("OpDef", _field("name", 1, 9), _field("type", 2, 9), _field("input", 5, 9, True),
                 _field("attr", 10, 11, True, "AttrEntry"), _field("src_name", 24, 9, True),
                 _field("src_index", 25, 3, True), _field("input_desc", 33, 11, True, "TensorDescriptor"),
                 _field("output_desc", 34, 11, True, "TensorDescriptor")),
        _message("AttrEntry", _field("key", 1, 9), _field("value", 2, 11, False, "AttrDef")),
        _message("AttrDef", _field("i", 3, 3), _field("s", 2, 12),
                 _field("t", 12, 11, False, "TensorDef"), _field("g", 13, 11, False, "GraphDef")),
        _message("TensorDescriptor", _field("dtype", 2, 14, False, "DataType"),
                 _field("shape", 3, 11, False, "ShapeDef"), _field("layout", 4, 9)),
        _message("ShapeDef", _field("dim", 1, 3, True)),
        _message("TensorDef", _field("data", 2, 12)),
    ]
    # Deliberately differ from public GE IDs. These are test fixture values,
    # not claims about any deployed DDK.
    enum = _blob(1, "DataType") + b"".join(_blob(2, _blob(1, name) + _int(2, number))
        for name, number in [("UNDEFINED", 0), ("FLOAT", 1), ("HALF", 7)])
    return _blob(1, "proto/ge_ir.proto") + _blob(2, "hiai.proto") + b"".join(messages) + _blob(5, enum)


def _library(path):
    descriptor = _schema()
    payload = descriptor + b"\0"
    names = b"\0_ZL49descriptor_table_protodef_proto_2fge_5fir_2eproto\0"
    symbols = bytes(24) + struct.pack("<IBBHQQ", 1, 1, 0, 1, 0x1000, len(payload))
    start = 64
    names_start, symbols_start = start + len(payload), start + len(payload) + len(names)
    table = symbols_start + len(symbols)
    header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01" + bytes(9),
                         3, 62, 1, 0, 0, table, 0, 64, 0, 0, 64, 4, 0)
    sections = [bytes(64), struct.pack("<IIQQQQIIQQ", 0, 1, 0, 0x1000, start, len(payload), 0, 0, 1, 0),
                struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, names_start, len(names), 0, 0, 1, 0),
                struct.pack("<IIQQQQIIQQ", 0, 2, 0, 0, symbols_start, len(symbols), 2, 0, 8, 24)]
    data = header + payload + names + symbols + b"".join(sections)
    path.write_bytes(data)
    return data


def _desc(dtype=7, shape=(64, 16384)):
    return _int(2, dtype) + _blob(3, b"".join(_int(1, x) for x in shape)) + _blob(4, "ND")


def _attr(name, value):
    return _blob(10, _blob(1, name) + _blob(2, value))


def _graph(*, duplicate=False, target=True):
    inputs = _blob(1, "__ge/v20/flat_input") + _blob(2, "Reshape") + _blob(34, _desc())
    weight = (_blob(1, "draft.fc.weight") + _blob(2, "FileConstant") +
              _blob(34, _desc(shape=(2048, 16384))) + _attr("dtype", _int(3, 1)) +
              _attr("value", _blob(12, _blob(2, b"PRIVATE_WEIGHT_BYTES"))))
    matmul = (_blob(1, "__ge/v20/flat_output") + _blob(2, "MatMul") +
              _blob(5, "__ge/v20/flat_input:0") + _blob(5, "draft.fc.weight:0") +
              _blob(33, _desc()) + _blob(33, _desc(shape=(2048, 16384))) +
              _blob(34, _desc(shape=(64, 2048))) + _attr("transpose_x2", _int(3, 1)))
    return (_blob(1, "draft") + _blob(6, inputs) + _blob(6, weight) +
            (_blob(6, weight) if duplicate else b"") + (_blob(6, matmul) if target else b""))


@pytest.fixture
def case(tmp_path):
    lib = tmp_path / "selected.so"
    lib_bytes = _library(lib)
    root = tmp_path / "compilation"
    folder = root / di._DUMP_DIRECTORY
    folder.mkdir(parents=True)
    path = folder / "00_before_air_parser.irpb"
    path.write_bytes(_blob(7, _graph()))
    return root, lib, lib_bytes, path


def test_selected_schema_descriptors_and_attribute_enum_domains_remain_separate(case):
    root, lib, lib_bytes, path = case
    before = {p: p.read_bytes() for p in (lib, path)}
    result = di.inspect_draft_ir(root, lib)
    assert result["status"] == "INSPECTED"
    assert result["schema"]["library_sha256"] == hashlib.sha256(lib_bytes).hexdigest()
    assert result["schema"]["descriptor_sha256"] == hashlib.sha256(_schema()).hexdigest()
    row = result["files"][0]
    assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    match = row["matches"][0]
    assert match["node"]["input_desc"][0]["schema_enum_names"]["dtype"]["names"] == ["HALF"]
    assert match["node"]["input_desc"][0]["raw"]["shape"]["dim"] == [64, 16384]
    assert [x["status"] for x in match["input_producers"]] == ["RESOLVED", "RESOLVED"]
    weight = match["input_producers"][1]["producer"]
    assert weight["attributes"] == [{"key": "dtype", "value": {"i": 1}}]
    assert weight["output_desc"][0]["schema_enum_names"]["dtype"]["names"] == ["HALF"]
    assert weight["other_attribute_count"] == 1
    encoded = json.dumps(result)
    assert "PRIVATE_WEIGHT_BYTES" not in encoded and '"data"' not in encoded
    assert before == {p: p.read_bytes() for p in before}


def test_duplicate_same_graph_producers_are_not_associated(case):
    root, lib, _, path = case
    path.write_bytes(_blob(7, _graph(duplicate=True)))
    match = di.inspect_draft_ir(root, lib)["files"][0]["matches"][0]
    assert match["input_producers"][1]["status"] == "AMBIGUOUS"
    assert "producer" not in match["input_producers"][1]
    assert match["named_fc_weight_count"] == 2


@pytest.mark.parametrize("raw", [_graph(), b"not a protobuf", _blob(88, b"unknown field")])
def test_schema_mismatch_is_recorded_without_graphdef_or_text_fallback(case, raw):
    root, lib, _, path = case
    path.write_bytes(raw)
    result = di.inspect_draft_ir(root, lib)
    assert result["status"] == "PARTIAL"
    assert result["files"][0]["status"] == "UNSUPPORTED_DUMP"
    assert "error" in result["files"][0] and "matches" not in result["files"][0]


def test_recorded_dump_hash_mismatch_never_decodes(case):
    root, lib, _, path = case
    record = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": "0" * 64}
    (root / "report.json").write_text(json.dumps({"generated_artifacts": [record]}))
    result = di.inspect_draft_ir(root, lib)
    assert result["status"] == "INTEGRITY_MISMATCH"
    assert "matches" not in result["files"][0]
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "report.json").write_text(json.dumps({"generated_artifacts": [record]}))
    assert di.inspect_draft_ir(root, lib)["files"][0]["inventory_binding"] == "MATCHED"


def test_bounds_and_nonrecursive_fixed_directory(case, monkeypatch):
    root, lib, _, path = case
    other = path.parent / "nested"
    other.mkdir()
    (other / "ignored.irpb").write_bytes(b"unknown")
    (root / "other.irpb").write_bytes(b"unknown")
    monkeypatch.setattr(di, "_MAX_FILE_BYTES", 4)
    result = di.inspect_draft_ir(root, lib)
    assert result["candidate_count"] == 1 and result["read_bytes"] == 0
    assert "byte budget" in result["files"][0]["error"]


def test_missing_and_invalid_schema_are_explicit(case):
    root, lib, _, _ = case
    assert di.inspect_draft_ir(root, lib.with_name("missing.so"))["status"] == "SCHEMA_UNAVAILABLE"
    lib.write_bytes(b"not ELF")
    result = di.inspect_draft_ir(root, lib)
    assert result["status"] == "SCHEMA_UNAVAILABLE" and "ELF64" in result["schema_error"]


def test_symlink_is_not_followed(case):
    root, lib, _, path = case
    target = root / "outside.irpb"
    path.rename(target)
    path.symlink_to(target)
    result = di.inspect_draft_ir(root, lib)
    assert result["read_bytes"] == 0
    assert "non-symlink" in result["files"][0]["error"]


def test_target_absence_is_not_support_evidence(case):
    root, lib, _, path = case
    path.write_bytes(_blob(7, _graph(target=False)))
    result = di.inspect_draft_ir(root, lib)
    assert result["status"] == "INSPECTED"
    assert result["files"][0]["status"] == "TARGET_NOT_FOUND"


def test_output_and_file_count_budgets_are_explicit(case, monkeypatch):
    root, lib, _, path = case
    (path.parent / "01.irpb").write_bytes(path.read_bytes())
    monkeypatch.setattr(di, "_MAX_FILES", 1)
    result = di.inspect_draft_ir(root, lib)
    assert result["status"] == "PARTIAL" and result["files_omitted"] == 1
    monkeypatch.setattr(di, "_MAX_OUTPUT_BYTES", 256)
    result = di.inspect_draft_ir(root, lib)
    assert result["output_limited"] and result["files"][0]["status"] == "OUTPUT_LIMIT"
    assert "matches" not in result["files"][0]

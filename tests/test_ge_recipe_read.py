"""Structural recipe diagnostics must not read or interpret tensor payloads."""
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from qwen35_moe_dflash.ge_recipe import emit_recipe_cpp
from qwen35_moe_dflash.ge_recipe_read import read_recipe


def _records():
    def port(name, **extra):
        return {"port": name, "shape": [1, 4], "dtype": "float16", **extra}
    return [
        {"name": "tokens", "type": "Data", "inputs": [], "outputs": [port("y")],
         "attrs": {"index": {"type": "int", "value": 0}}},
        {"name": "weight", "type": "Const", "inputs": [], "outputs": [port("y")],
         "constant": {"data": b"\x00\x3c" * 4}},
        {"name": "__ge/v42/result", "type": "Add", "inputs": [port("x1", src=0, src_port="y"), port("x2", src=1, src_port="y")],
         "outputs": [port("y")], "attrs": {
             "alpha": {"type": "float", "value": 0.5},
             "axes": {"type": "ints", "value": [-1, 0]},
             "flag": {"type": "bool", "value": False},
             "kind": {"type": "dtype", "value": "int32"},
             "note": {"type": "string", "value": "不推测重命名"},
         }},
    ]


def _write_valid(tmp_path, records=None):
    records = records or _records()
    folder = tmp_path / "recipe"
    emit_recipe_cpp(records, "target", [0], [(2, 0)], folder)
    return folder / "graph.recipe"


def test_real_writer_round_trip_preserves_names_ports_shapes_and_typed_attributes(tmp_path):
    original = _records()
    path = _write_valid(tmp_path, original)
    result = read_recipe(path)
    assert result["graph_name"] == "target"
    assert result["inputs"] == [0] and result["outputs"] == [[2, 0]]
    for got, want in zip(result["records"], original):
        assert {key: got[key] for key in ("name", "type", "inputs", "outputs")} == {
            key: want[key] for key in ("name", "type", "inputs", "outputs")}
        assert got["attrs"] == want.get("attrs", {})
    assert result["records"][1]["constant"] == {
        "kind": 1, "path": str(path.with_name("constants.bin")), "offset": 0, "size": 8}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("dtype,width", [("float16", 2), ("int32", 4), ("int64", 8), ("bool", 1)])
@pytest.mark.parametrize("shape,items", [([], 1), ([0, 4], 0)])
def test_scalar_and_zero_element_descriptors_preserve_all_format_dtypes(tmp_path, dtype, width, shape, items):
    records = _records()
    for record in records:
        for port in record["inputs"] + record["outputs"]:
            port.update(shape=shape, dtype=dtype)
    records[1]["constant"]["data"] = b"\0" * items * width
    result = read_recipe(_write_valid(tmp_path, records))
    assert result["records"][0]["outputs"] == [{"port": "y", "shape": shape, "dtype": dtype}]
    assert result["records"][1]["constant"]["size"] == items * width


@pytest.mark.parametrize("storage", ["inline", "raw", "fileconstant"])
def test_metadata_reader_never_opens_constant_or_fileconstant_paths(tmp_path, monkeypatch, storage):
    records = _records()
    missing = tmp_path / "nonexistent-weight.bin"
    if storage == "raw":
        records[1]["constant"] = {"path": str(missing), "size": 8}
    elif storage == "fileconstant":
        records[1].pop("constant")
        records[1].update(type="FileConstant", attrs={
            "file_path": {"type": "string", "value": str(missing)},
            "shape": {"type": "ints", "value": [1, 4]},
            "dtype": {"type": "dtype", "value": "float16"}})
    path = _write_valid(tmp_path, records)
    path.with_name("constants.bin").unlink()
    original_open = Path.open
    opened = []

    def only_recipe(self, *args, **kwargs):
        opened.append(self)
        if self != path:
            raise AssertionError(f"unexpected tensor file read: {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", only_recipe)
    result = read_recipe(path)
    assert len(result["records"]) == 3 and opened == [path]
    if storage == "raw":
        assert result["records"][1]["constant"] == {"kind": 2, "path": str(missing), "offset": 0, "size": 8}
    if storage == "fileconstant":
        assert result["records"][1]["attrs"]["file_path"]["value"] == str(missing)


def _text(value):
    payload = value.encode()
    return struct.pack("<I", len(payload)) + payload


def _desc(shape=(1, 4), dtype=1):
    return struct.pack("<BI", dtype, len(shape)) + b"".join(struct.pack("<q", item) for item in shape)


def _corrupt(payload, fault):
    payload = bytearray(payload)
    def replace(old, new):
        start = payload.index(old)
        payload[start:start + len(old)] = new
    if fault == "magic":
        payload[0] ^= 1
    elif fault == "truncated":
        return payload[:-2]
    elif fault == "trailing":
        payload.extend(b"extra")
    elif fault == "huge_string":
        payload[8:12] = struct.pack("<I", 2**32 - 1)
    elif fault == "empty_graph":
        replace(_text("target"), _text(""))
    elif fault == "nul_string":
        replace(_text("target"), _text("tar\0et"))
    elif fault == "invalid_utf8":
        replace(_text("target"), struct.pack("<I", 1) + b"\xff")
    elif fault in ("too_many_ops", "zero_ops"):
        start = 8 + len(_text("target"))
        payload[start:start + 4] = struct.pack("<I", 2_000_001 if fault == "too_many_ops" else 0)
    elif fault == "duplicate_name":
        replace(_text("weight"), _text("tokens"))
    elif fault == "empty_name":
        replace(_text("weight"), _text(""))
    elif fault == "empty_type":
        replace(_text("Add"), _text(""))
    elif fault in ("forward_source", "missing_source_port"):
        replace(_text("x1") + struct.pack("<I", 0) + _text("y"),
                _text("x1") + struct.pack("<I", 2 if fault == "forward_source" else 0)
                + _text("missing" if fault == "missing_source_port" else "y"))
    elif fault == "duplicate_input":
        replace(_text("x2"), _text("x1"))
    elif fault == "duplicate_output":
        replace(_text("Data") + struct.pack("<II", 0, 1) + _text("y") + _desc(),
                _text("Data") + struct.pack("<II", 0, 2) + (_text("y") + _desc()) * 2)
    elif fault == "port_count_limit":
        replace(_text("Data") + struct.pack("<I", 0), _text("Data") + struct.pack("<I", 65537))
    elif fault == "descriptor_mismatch":
        replace(_text("x1") + struct.pack("<I", 0) + _text("y") + _desc(),
                _text("x1") + struct.pack("<I", 0) + _text("y") + _desc((1, 5)))
    elif fault in ("unknown_dtype", "rank_limit", "negative_dimension", "shape_overflow"):
        new = {"unknown_dtype": _desc(dtype=9), "rank_limit": _desc((1,) * 17),
               "negative_dimension": _desc((1, -4)), "shape_overflow": _desc((2**62, 4))}[fault]
        replace(_desc(), new)
    elif fault == "invalid_bool":
        replace(_text("flag") + b"\x03\x00", _text("flag") + b"\x03\x02")
    elif fault == "unknown_attribute":
        replace(_text("flag") + b"\x03\x00", _text("flag") + b"\x09\x00")
    elif fault == "unknown_dtype_attribute":
        replace(_text("kind") + b"\x05\x02", _text("kind") + b"\x05\x09")
    elif fault == "nonfinite_attribute":
        replace(_text("alpha") + b"\x02" + struct.pack("<f", 0.5),
                _text("alpha") + b"\x02" + struct.pack("<f", float("nan")))
    elif fault == "duplicate_attribute":
        replace(_text("kind"), _text("flag"))
    elif fault == "huge_attribute_list":
        replace(_text("axes") + b"\x04" + struct.pack("<I", 2),
                _text("axes") + b"\x04" + struct.pack("<I", 65537))
    elif fault == "bad_data_index":
        replace(_text("index") + b"\x01" + struct.pack("<q", 0),
                _text("index") + b"\x01" + struct.pack("<q", 1))
    elif fault == "bad_input_boundary":
        payload[-16:-12] = struct.pack("<I", 2)
    elif fault == "bad_output_boundary":
        payload[-4:] = struct.pack("<I", 1)
    elif fault == "bad_concat_ports":
        replace(_text("Add"), _text("ConcatD"))
    else:
        raise AssertionError(fault)
    return payload


@pytest.mark.parametrize("fault", [
    "magic", "truncated", "trailing", "huge_string", "empty_graph", "nul_string", "invalid_utf8",
    "too_many_ops", "zero_ops", "duplicate_name", "empty_name", "empty_type", "forward_source",
    "missing_source_port", "duplicate_input", "duplicate_output", "port_count_limit", "descriptor_mismatch", "unknown_dtype", "rank_limit",
    "negative_dimension", "shape_overflow", "invalid_bool", "unknown_attribute", "unknown_dtype_attribute",
    "nonfinite_attribute", "duplicate_attribute", "huge_attribute_list", "bad_data_index",
    "bad_input_boundary", "bad_output_boundary", "bad_concat_ports",
])
def test_rejects_malformed_metadata_without_partial_mapping(tmp_path, fault):
    path = _write_valid(tmp_path)
    path.write_bytes(_corrupt(path.read_bytes(), fault))
    with pytest.raises(ValueError):
        read_recipe(path)


@pytest.mark.parametrize("fault", ["unknown_storage", "missing_storage", "wrong_size", "raw_nonzero_offset"])
def test_rejects_constant_metadata_mismatch_without_loading_payload(tmp_path, fault):
    path = _write_valid(tmp_path)
    payload = path.read_bytes()
    old = b"\x01" + _text(str(path.with_name("constants.bin"))) + struct.pack("<QQ", 0, 8)
    new = {"unknown_storage": b"\x03" + old[1:], "missing_storage": b"\x00",
           "wrong_size": old[:-8] + struct.pack("<Q", 7),
           "raw_nonzero_offset": b"\x02" + _text("/never/open") + struct.pack("<QQ", 1, 8)}[fault]
    assert old in payload
    path.write_bytes(payload.replace(old, new, 1))
    with pytest.raises(ValueError):
        read_recipe(path)


def test_reader_enforces_byte_limit_before_opening(tmp_path, monkeypatch):
    import qwen35_moe_dflash.ge_recipe_read as module
    path = _write_valid(tmp_path)
    monkeypatch.setattr(module, "_MAX_BYTES", path.stat().st_size - 1)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("oversized recipe was opened"))
    with pytest.raises(ValueError, match="size limit"):
        read_recipe(path)


def test_directory_is_rejected_without_opening(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        read_recipe(tmp_path)


def test_reader_imports_and_runs_without_site_packages(tmp_path):
    path = _write_valid(tmp_path)
    model = Path(__file__).resolve().parents[1] / "model"
    program = "import sys,json;sys.path.insert(0,sys.argv[1]);from qwen35_moe_dflash.ge_recipe_read import read_recipe;print(json.dumps(read_recipe(sys.argv[2])))"
    completed = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(model), str(path)],
                               check=True, capture_output=True, text=True, timeout=10)
    assert json.loads(completed.stdout)["records"][2]["name"] == "__ge/v42/result"

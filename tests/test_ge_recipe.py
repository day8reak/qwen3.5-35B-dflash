"""Execute the real C++ recipe parser, plus fail-closed writer validation.

The parser-only build contains no GE implementation or fake AIR exporter.
Real GE header compilation is recorded separately in run evidence.
"""
import copy
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from qwen35_moe_dflash.ge_recipe import emit_recipe_cpp


def records():
    def desc(port, **extra):
        return {"port": port, "shape": [1, 4], "dtype": "float16", **extra}
    return [
        {"name": "x", "type": "Data", "inputs": [], "outputs": [desc("y")],
         "attrs": {"index": {"type": "int", "value": 0}}},
        {"name": "bias", "type": "Const", "inputs": [], "outputs": [desc("y")],
         "constant": {"data": b"\x00\x3c" * 4}},
        {"name": "sum", "type": "Add", "inputs": [desc("x1", src=0, src_port="y"), desc("x2", src=1, src_port="y")],
         "outputs": [desc("y")]},
        {"name": "power", "type": "Power", "inputs": [desc("x", src=2, src_port="y")], "outputs": [desc("y")],
         "attrs": {name: {"type": "float", "value": value} for name, value in (("power", 1.0), ("scale", 0.3535533905932738), ("shift", 0.0))}},
        {"name": "softmax", "type": "SoftmaxV2", "inputs": [desc("x", src=3, src_port="y")], "outputs": [desc("y")],
         "attrs": {"axes": {"type": "ints", "value": [-1]}, "half_to_float": {"type": "bool", "value": False}}},
    ]


@pytest.fixture(scope="module")
def parser(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable for recipe parser execution")
    folder = tmp_path_factory.mktemp("real-recipe-parser")
    source = folder / "reader.cpp"
    source.write_text(emit_recipe_cpp(records(), "fixture", [0], [(4, 0)], folder / "data"))
    binary = folder / "reader"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-DQ35_RECIPE_PARSE_ONLY", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary, folder / "data/graph.recipe"


def run(parser, path):
    return subprocess.run([str(parser[0]), str(path)], capture_output=True, text=True, timeout=15)


def test_real_cpp_parser_accepts_typed_attributes_and_shared_constant_blob(parser):
    result = run(parser, parser[1])
    assert result.returncode == 0, result.stderr
    assert result.stdout == "RECIPE_VALID 5\n"
    assert "AIR_EXPORTED" not in result.stdout
    assert parser[1].with_name("constants.bin").read_bytes() == b"\x00\x3c" * 4


@pytest.mark.parametrize("fault", ["magic", "truncated", "trailing", "oversized_string", "forward_reference", "invalid_bool", "unknown_dtype"])
def test_actual_cpp_reader_rejects_corrupt_recipe(parser, tmp_path, fault):
    payload = bytearray(parser[1].read_bytes())
    def text(value):
        value = value.encode()
        return struct.pack("<I", len(value)) + value
    if fault == "magic":
        payload[0] ^= 1
    elif fault == "truncated":
        payload = payload[:-3]
    elif fault == "trailing":
        payload.extend(b"extra")
    elif fault == "oversized_string":
        payload[8:12] = struct.pack("<I", 0xffffffff)
    elif fault == "forward_reference":
        where = payload.index(text("x1") + struct.pack("<I", 0) + text("y")) + len(text("x1"))
        payload[where:where+4] = struct.pack("<I", 99999)
    elif fault == "invalid_bool":
        where = payload.index(text("half_to_float") + b"\3\0") + len(text("half_to_float")) + 1
        payload[where] = 2
    else:
        where = payload.index(text("y") + b"\1" + struct.pack("<I", 2) + struct.pack("<qq", 1, 4)) + len(text("y"))
        payload[where] = 9
    path = tmp_path / "bad.recipe"
    path.write_bytes(payload)
    result = run(parser, path)
    assert result.returncode != 0
    assert "GE recipe failed:" in result.stderr
    assert "RECIPE_VALID" not in result.stdout


@pytest.mark.parametrize("fault", ["missing_payload", "wrong_payload_size", "bad_source", "descriptor_mismatch", "bad_data_index", "nonfinite", "overflowing_shape"])
def test_writer_rejects_invalid_graph_before_creating_outputs(tmp_path, fault):
    graph = records()
    if fault == "missing_payload":
        del graph[1]["constant"]
    elif fault == "wrong_payload_size":
        graph[1]["constant"]["data"] = b"x"
    elif fault == "bad_source":
        graph[2]["inputs"][0]["src"] = 2
    elif fault == "descriptor_mismatch":
        graph[2]["inputs"][0]["shape"] = [1, 5]
    elif fault == "bad_data_index":
        graph[0]["attrs"]["index"]["value"] = 1
    elif fault == "nonfinite":
        graph[3]["attrs"]["scale"]["value"] = float("inf")
    else:
        graph[0]["outputs"][0]["shape"] = [2**62, 4]
    folder = tmp_path / "new-output"
    with pytest.raises(ValueError):
        emit_recipe_cpp(graph, "bad", [0], [(4, 0)], folder)
    assert not folder.exists()


def test_writer_preserves_existing_output_directory(tmp_path):
    folder = tmp_path / "existing"
    folder.mkdir()
    marker = folder / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        emit_recipe_cpp(records(), "new", [0], [(4, 0)], folder)
    assert marker.read_text() == "preserve"
    assert not (folder / "graph.recipe").exists()


def test_fileconstant_attrs_and_external_size_validation(parser, tmp_path):
    graph = records()
    weight = tmp_path / "weights.bin"
    weight.write_bytes(b"\x00\x3c" * 4)
    graph[1] = {"name": "bias", "type": "FileConstant", "inputs": [], "outputs": graph[1]["outputs"],
                "attrs": {"file_path": {"type": "string", "value": str(weight)},
                          "shape": {"type": "ints", "value": [1, 4]},
                          "dtype": {"type": "dtype", "value": "float16"},
                          "file_id": {"type": "string", "value": ""}}}
    folder = tmp_path / "external"
    emit_recipe_cpp(graph, "external", [0], [(4, 0)], folder)
    assert run(parser, folder / "graph.recipe").returncode == 0
    weight.write_bytes(b"short")
    result = run(parser, folder / "graph.recipe")
    assert result.returncode != 0
    assert "external weight file size mismatch" in result.stderr


def test_raw_const_file_requires_exact_size(parser, tmp_path):
    graph = records()
    weight = tmp_path / "raw.bin"
    weight.write_bytes(b"\x00\x3c" * 4)
    graph[1]["constant"] = {"path": str(weight), "offset": 0, "size": 8}
    folder = tmp_path / "raw"
    emit_recipe_cpp(graph, "raw", [0], [(4, 0)], folder)
    assert run(parser, folder / "graph.recipe").returncode == 0
    weight.write_bytes(weight.read_bytes() + b"padding")
    result = run(parser, folder / "graph.recipe")
    assert result.returncode != 0
    assert "weight file size mismatch" in result.stderr


def test_cpp_size_is_independent_of_graph_operator_count(tmp_path):
    small = records()
    large = copy.deepcopy(small)
    for i in range(10000):
        previous = len(large) - 1
        large.append({"name": f"identity_{i}", "type": "Power",
                      "inputs": [{"port": "x", "src": previous, "src_port": "y", "shape": [1, 4], "dtype": "float16"}],
                      "outputs": [{"port": "y", "shape": [1, 4], "dtype": "float16"}],
                      "attrs": {"power": {"type": "float", "value": 1}, "scale": {"type": "float", "value": 1}, "shift": {"type": "float", "value": 0}}})
    small_code = emit_recipe_cpp(small, "small", [0], [(4, 0)], tmp_path / "small")
    large_code = emit_recipe_cpp(large, "large", [0], [(len(large)-1, 0)], tmp_path / "large")
    assert abs(len(small_code) - len(large_code)) < 10
    assert len(large_code) < 25000
    assert (tmp_path / "large/graph.recipe").stat().st_size > 1_000_000

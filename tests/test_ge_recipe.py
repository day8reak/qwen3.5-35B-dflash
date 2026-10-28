"""Execute the real C++ recipe parser, plus fail-closed writer validation.

The parser-only build contains no GE implementation or fake AIR exporter.
Real GE header compilation is recorded separately in run evidence.
"""
import copy
import os
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


def test_explicit_nd_keeps_version_one_bytes_unchanged(parser, tmp_path):
    graph = [records()[0]]
    emit_recipe_cpp(graph, "nd", [0], [(0, 0)], tmp_path / "implicit")
    graph[0]["outputs"][0]["format"] = "ND"
    emit_recipe_cpp(graph, "nd", [0], [(0, 0)], tmp_path / "explicit")
    first = tmp_path / "implicit/graph.recipe"
    second = tmp_path / "explicit/graph.recipe"
    assert first.read_bytes().startswith(b"Q35GE001")
    assert first.read_bytes() == second.read_bytes()
    assert run(parser, first).returncode == run(parser, second).returncode == 0


def _layout_graph():
    graph = records()
    for record in graph:
        for port in record["inputs"] + record["outputs"]:
            port.update(shape=[1, 1, 1, 4], format="NCHW")
    graph.append({"name": "nd_input", "type": "Data", "inputs": [],
                  "outputs": [{"port": "y", "shape": [1, 4], "dtype": "int32"}],
                  "attrs": {"index": {"type": "int", "value": 1}}})
    return graph


def test_real_cpp_parser_accepts_mixed_nd_nchw_descriptors(parser, tmp_path):
    folder = tmp_path / "mixed"
    code = emit_recipe_cpp(_layout_graph(), "mixed", [0, 5], [(4, 0), (5, 0)], folder)
    path = folder / "graph.recipe"
    assert path.read_bytes().startswith(b"Q35GE002")
    result = run(parser, path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "RECIPE_VALID 6\n"
    # This checks generated construction/verification, not actual GE execution.
    assert "case 1: return ge::FORMAT_NCHW" in code
    assert "ge::Shape(d.shape), Format(d.layout), DType(d.dtype)" in code
    assert "got.GetFormat() == Format(want.layout)" in code


@pytest.mark.parametrize("fault", ["unknown_layout", "connected_layout_mismatch", "nchw_rank"])
def test_cpp_reader_rejects_corrupt_version_two_layout(parser, tmp_path, fault):
    folder = tmp_path / "mixed"
    emit_recipe_cpp(_layout_graph(), "mixed", [0, 5], [(4, 0)], folder)
    path = folder / "graph.recipe"
    payload = bytearray(path.read_bytes())
    def text(value):
        return struct.pack("<I", len(value)) + value.encode()
    desc = struct.pack("<BBIqqqq", 1, 1, 4, 1, 1, 1, 4)
    prefix = text("x1") + struct.pack("<I", 0) + text("y") if fault == "connected_layout_mismatch" else text("y")
    offset = payload.index(prefix + desc) + len(prefix) + 1
    if fault == "nchw_rank":
        payload[offset+1:offset+5] = struct.pack("<I", 3)
    else:
        payload[offset] = 2 if fault == "unknown_layout" else 0
    path.write_bytes(payload)
    result = run(parser, path)
    assert result.returncode != 0
    assert {"unknown_layout": "unsupported recipe descriptor format",
            "connected_layout_mismatch": "connected descriptor mismatch",
            "nchw_rank": "NCHW recipe descriptors require rank 4"}[fault] in result.stderr


@pytest.mark.parametrize("shape", [[], [1, 4], [1, 1, 1, 1, 4]])
def test_writer_requires_rank_four_for_nchw(tmp_path, shape):
    graph = [records()[0]]
    graph[0]["outputs"][0].update(format="NCHW", shape=shape)
    with pytest.raises(ValueError, match="NCHW recipe descriptors require rank 4"):
        emit_recipe_cpp(graph, "bad", [0], [(0, 0)], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("layout", ["NHWC", "nchw", "", None, 1, ["NCHW"]])
def test_writer_rejects_invalid_layout_before_creating_outputs(tmp_path, layout):
    graph = [records()[0]]
    graph[0]["outputs"][0]["format"] = layout
    with pytest.raises(ValueError, match="descriptor format"):
        emit_recipe_cpp(graph, "bad", [0], [(0, 0)], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("explicit_nd", [False, True])
def test_writer_rejects_connected_layout_mismatch_before_outputs(tmp_path, explicit_nd):
    graph = _layout_graph()
    port = graph[2]["inputs"][0]
    port.pop("format")
    if explicit_nd:
        port["format"] = "ND"
    with pytest.raises(ValueError, match="connected descriptor mismatch"):
        emit_recipe_cpp(graph, "bad", [0, 5], [(4, 0)], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


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


@pytest.fixture(params=["recipe", "unrolled"])
def factory_regression_cpp(request, tmp_path):
    """Exercise all boundary dtypes, input fan-out and repeated source ports."""
    from qwen35_moe_dflash.ge_cpp import emit_cpp
    from qwen35_moe_dflash.ge_ir import Graph

    graph = Graph("data_factory_regression")
    ids = graph.input("input_ids", (1, 4), "int32")
    mask = graph.input("attention_mask", (1, 4), "int32")
    features = graph.input("target_features", (1, 4), "float16")
    positions = graph.input("positions", (1, 4), "int64")
    visible = graph.input("visible", (1, 4), "bool")
    positions32 = graph.op("cast", positions, shape=(1, 4), dtype="int32")
    visible32 = graph.op("cast", visible, shape=(1, 4), dtype="int32")
    tokens = graph.op("add", ids, mask, shape=(1, 4), dtype="int32")
    tokens = graph.op("add", tokens, positions32, shape=(1, 4), dtype="int32")
    tokens = graph.op("add", tokens, visible32, shape=(1, 4), dtype="int32")
    doubled = graph.op("add", features, features, shape=(1, 4))
    graph.outputs = [tokens, doubled]
    options = {"recipe_dir": tmp_path / "recipe"} if request.param == "recipe" else {}
    source = tmp_path / (request.param + ".cpp")
    source.write_text(emit_cpp(graph, **options))
    return source


def _real_ge_compiler():
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    abi = os.environ.get("Q35_GE_TEST_CXX11_ABI", "0")
    assert abi in ("0", "1"), "Q35_GE_TEST_CXX11_ABI must be 0 or 1"
    return [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-D_GLIBCXX_USE_CXX11_ABI=" + abi]


def test_data_factory_real_ge_header_compilation(factory_regression_cpp, tmp_path):
    """Compile actual registered Data schemas in both emitters; no GE run claim.

    Opt in with Q35_GE_TEST_INCLUDE pointing at the matching public GE headers.
    This catches errors in registration macro expansion and public API usage.
    """
    declared = os.environ.get("Q35_GE_TEST_INCLUDE")
    if not declared:
        pytest.skip("Q35_GE_TEST_INCLUDE not configured; real GE header test requires explicit headers")
    include = Path(declared)
    assert (include / "graph/operator_reg.h").is_file()
    command = _real_ge_compiler() + ["-I", str(include), "-c", str(factory_regression_cpp),
                                    "-o", str(tmp_path / "factory.o")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "factory.o").stat().st_size > 0


def test_data_factory_real_ge_save_load(factory_regression_cpp, tmp_path):
    """Actual GE SetInputs/SaveToFile/LoadFromFile regression, without an NPU.

    Opt in with Q35_GE_TEST_ROOT (include/ and lib64/ from the same GE toolkit).
    Without real libraries this is skipped, never replaced with a fake GE API.
    """
    declared = os.environ.get("Q35_GE_TEST_ROOT")
    if not declared:
        pytest.skip("Q35_GE_TEST_ROOT not configured; actual GE serialization remains unverified")
    root = Path(declared).resolve()
    include, lib = root / "include", root / "lib64"
    assert (include / "graph/operator_reg.h").is_file()
    assert (lib / "libgraph.so").is_file()
    binary, output = tmp_path / "factory-exporter", tmp_path / "fixture.air"
    command = _real_ge_compiler() + ["-I", str(include), str(factory_regression_cpp),
                                    "-L", str(lib), "-Wl,-rpath," + str(lib), "-lgraph"]
    if (lib / "libgraph_base.so").is_file():
        command.append("-lgraph_base")
    command.extend(["-o", str(binary)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(lib), env.get("LD_LIBRARY_PATH", ""))))
    result = subprocess.run([str(binary), str(output)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AIR_EXPORTED" in result.stdout
    assert output.is_file() and output.stat().st_size > 0
    assert not output.with_suffix(".air.partial").exists()

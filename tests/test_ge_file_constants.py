"""FileConstant compatibility policy, binary metadata and real host parser.

The C++ executable below is the production recipe reader compiled without GE.
These tests do not serialize AIR, invoke OMG, or establish device support for
the explicit CPro integer-attribute compatibility candidate.
"""
import copy
import shutil
import struct
import subprocess

import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_ir import DTYPES, Graph
from qwen35_moe_dflash.ge_recipe import emit_recipe_cpp
from qwen35_moe_dflash.ge_recipe_read import read_recipe


# Independently state the external GE IDs and this repository's recipe tags:
# they share float16=1, but must never be used interchangeably for other types.
TYPES = (("float16", 1, 1), ("int32", 3, 2), ("int64", 9, 3), ("bool", 12, 4))


def _text(value):
    value = value.encode()
    return struct.pack("<I", len(value)) + value


def _fixture(root):
    (root / "weights").mkdir(parents=True)
    graph = Graph("file_constants")
    source = graph.input("features", (1, 4))
    paths = {}
    for dtype, _, _ in TYPES:
        name = "weight_" + dtype
        graph.outputs.append(graph.weight(name, (1, 4), dtype))
        path = root / "weights" / (name + ".bin")
        path.write_bytes(bytes(4 * DTYPES[dtype].itemsize))
        paths[name] = path
    graph.outputs.append(graph.op("cast", source, shape=(1, 4), dtype="int32"))
    return graph, paths


def _export_recipe(root):
    graph, paths = _fixture(root)
    code = emit_cpp(graph, paths, external_weights=True, recipe_dir=root / "recipe",
                    file_constant_policy="cpro-relative-int", file_constant_root=root)
    return graph, paths, code, root / "recipe/graph.recipe"


@pytest.fixture(scope="module")
def parser(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable for actual host recipe reader")
    root = tmp_path_factory.mktemp("fileconstant-parser")
    _, _, code, _ = _export_recipe(root)
    source, binary = root / "reader.cpp", root / "reader"
    source.write_text(code)
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-DQ35_RECIPE_PARSE_ONLY", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


def _run(parser, recipe, cwd):
    return subprocess.run([str(parser), str(recipe)], cwd=cwd, capture_output=True,
                          text=True, timeout=15)


def test_integer_policy_binary_tags_relative_paths_and_inline_setters(tmp_path):
    graph, paths, _, recipe = _export_recipe(tmp_path)
    metadata = read_recipe(recipe)
    payload = recipe.read_bytes()
    records = {record["name"]: record for record in metadata["records"]}
    inline = emit_cpp(graph, paths, external_weights=True,
                      file_constant_policy="cpro-relative-int", file_constant_root=tmp_path)
    assert str(tmp_path) not in inline
    for dtype, ge_id, descriptor_tag in TYPES:
        record = records["weight_" + dtype]
        relative = "weights/weight_" + dtype + ".bin"
        assert record["attrs"]["dtype"] == {"type": "int", "value": ge_id}
        assert record["outputs"][0]["dtype"] == dtype
        assert record["attrs"]["file_path"] == {"type": "string", "value": relative}
        assert _text("dtype") + b"\x01" + struct.pack("<q", ge_id) in payload
        assert (_text("y") + bytes([descriptor_tag]) + struct.pack("<Iqq", 2, 1, 4)) in payload
        assert '.SetAttr("dtype", int64_t{' + str(ge_id) + '})' in inline
        assert '.SetAttr("file_path", ge::AscendString("' + relative + '"))' in inline
        # Inline reconstruction must preserve the path for its size check too.
        assert 'fs::is_regular_file("' + relative + '")' in inline


def test_default_keeps_ge_type_attributes_and_absolute_paths(tmp_path):
    graph, paths = _fixture(tmp_path)
    inline = emit_cpp(graph, paths, external_weights=True)
    emit_cpp(graph, paths, external_weights=True, recipe_dir=tmp_path / "default")
    metadata = read_recipe(tmp_path / "default/graph.recipe")
    constants = [record for record in metadata["records"] if record["type"] == "FileConstant"]
    for record in constants:
        dtype = record["outputs"][0]["dtype"]
        assert record["attrs"]["dtype"] == {"type": "dtype", "value": dtype}
        assert record["attrs"]["file_path"]["value"] == str(paths[record["name"]].resolve())
    assert '.SetAttr("dtype", ge::DT_FLOAT16)' in inline
    assert '.SetAttr("dtype", ge::DT_INT32)' in inline
    assert '.SetAttr("dtype", ge::DT_INT64)' in inline
    assert '.SetAttr("dtype", ge::DT_BOOL)' in inline


def test_policy_does_not_change_argmax_type_or_cast_integer_attributes(tmp_path):
    graph, paths = _fixture(tmp_path)
    graph.outputs.append(graph.op("argmax", graph.inputs[0], shape=(1,), dtype="int32",
                                   attrs={"axis": -1}))
    for name, options in (("default", {}), ("cpro", {
        "file_constant_policy": "cpro-relative-int", "file_constant_root": tmp_path,
    })):
        emit_cpp(graph, paths, external_weights=True, recipe_dir=tmp_path / name, **options)
    default = read_recipe(tmp_path / "default/graph.recipe")["records"]
    cpro = read_recipe(tmp_path / "cpro/graph.recipe")["records"]
    for before, after in zip(default, cpro, strict=True):
        if before["type"] != "FileConstant":
            assert before["attrs"] == after["attrs"]
    argmax = next(record for record in cpro if record["type"] == "ArgMaxV2")
    cast = next(record for record in cpro if record["type"] == "Cast")
    assert argmax["attrs"]["dtype"] == {"type": "dtype", "value": "int32"}
    assert cast["attrs"]["dst_type"] == {"type": "int", "value": 3}


def test_actual_reader_accepts_air_parent_and_relocated_tree(parser, tmp_path):
    original, moved = tmp_path / "original", tmp_path / "moved"
    _, _, _, recipe = _export_recipe(original)
    result = _run(parser, recipe, original)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("RECIPE_VALID ")
    assert "AIR_EXPORTED" not in result.stdout
    original.rename(moved)
    relocated = moved / "recipe/graph.recipe"
    result = _run(parser, relocated, moved)
    assert result.returncode == 0, result.stderr
    assert not original.exists()
    assert read_recipe(relocated)["graph_name"] == "file_constants"


@pytest.mark.parametrize("fault", ["wrong_cwd", "missing_file", "wrong_size"])
def test_actual_reader_rejects_inaccessible_or_wrong_size_sidecars(parser, tmp_path, fault):
    root = tmp_path / "air_parent"
    _, paths, _, recipe = _export_recipe(root)
    cwd = root
    if fault == "wrong_cwd":
        cwd = tmp_path
    elif fault == "missing_file":
        paths["weight_float16"].unlink()
    else:
        paths["weight_float16"].write_bytes(b"short")
    result = _run(parser, recipe, cwd)
    assert result.returncode != 0
    assert "external weight file size mismatch" in result.stderr
    assert "RECIPE_VALID" not in result.stdout


@pytest.mark.parametrize("dtype,wrong_id", [("int32", 2), ("int64", 3), ("bool", 4)])
def test_actual_reader_and_metadata_reader_reject_descriptor_tag_as_ge_id(parser, tmp_path, dtype, wrong_id):
    _, _, _, recipe = _export_recipe(tmp_path)
    ge_id = next(ge_id for name, ge_id, _ in TYPES if name == dtype)
    payload = recipe.read_bytes()
    encoded = _text("dtype") + b"\x01" + struct.pack("<q", ge_id)
    assert payload.count(encoded) == 1
    recipe.write_bytes(payload.replace(encoded, _text("dtype") + b"\x01" + struct.pack("<q", wrong_id)))
    with pytest.raises(ValueError, match="FileConstant descriptor and metadata disagree"):
        read_recipe(recipe)
    result = _run(parser, recipe, tmp_path)
    assert result.returncode != 0
    assert "FileConstant metadata mismatch" in result.stderr


def test_recipe_writer_rejects_mismatched_fileconstant_integer_dtype(tmp_path):
    _, _, _, recipe = _export_recipe(tmp_path)
    metadata = read_recipe(recipe)
    records = copy.deepcopy(metadata["records"])
    record = next(record for record in records if record["name"] == "weight_int32")
    record["attrs"]["dtype"]["value"] = 2
    with pytest.raises(ValueError, match="FileConstant descriptor and metadata disagree"):
        emit_recipe_cpp(records, metadata["graph_name"], metadata["inputs"], metadata["outputs"], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_relative_policy_rejects_weight_outside_air_tree(tmp_path, symlink):
    root = tmp_path / "air_parent"
    graph, paths = _fixture(root)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(paths["weight_float16"].read_bytes())
    if symlink:
        link = root / "weights/outside_link.bin"
        link.symlink_to(outside)
        paths["weight_float16"] = link
    else:
        paths["weight_float16"] = outside
    with pytest.raises(ValueError, match="must be inside file_constant_root"):
        emit_cpp(graph, paths, external_weights=True,
                 file_constant_policy="cpro-relative-int", file_constant_root=root)


@pytest.mark.parametrize("options,match", [
    ({"file_constant_policy": "cpro-relative-int", "external_weights": True}, "requires external_weights and file_constant_root"),
    ({"file_constant_policy": "cpro-relative-int"}, "requires external_weights and file_constant_root"),
    ({"file_constant_policy": "typo"}, "unsupported FileConstant policy"),
])
def test_invalid_policy_settings_rejected_before_export(tmp_path, options, match):
    graph, paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match=match):
        emit_cpp(graph, paths, **options)

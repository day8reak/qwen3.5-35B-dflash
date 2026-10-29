"""Standalone host GE branch construction; no original target or weights."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash.original_ge_append_cpp import (
    CAST_NAMES, CONCAT_NAME, INPUT_NAMES, emit_feature_branch_cpp,
)


@pytest.mark.parametrize("length", [1, 55, 64, -1])
def test_generator_has_only_the_declared_branch_and_separate_dtype_domains(length):
    source = emit_feature_branch_cpp(length)
    assert f"constexpr int64_t kSequence = {length};" in source
    assert 'ge::Graph graph("__dflash_features_template")' in source
    assert 'concat.SetAttr("concat_dim", int64_t{2})' in source
    assert 'concat.SetAttr("N", int64_t{8})' in source
    assert 'cast.SetAttr("dst_type", static_cast<int64_t>(ge::DT_FLOAT16))' in source
    assert "O_EXCL | O_NOFOLLOW" in source
    assert "LoadFromFile(argv[1])" in source  # Reload only the newly generated output.
    assert "if (argc != 3)" in source
    assert "FileConstant" not in source and "Read(" not in source
    assert "graph.Compile(" not in source and "graph.Run(" not in source
    assert "FindOpByName" not in source
    assert "@SEQUENCE_LENGTH@" not in source
    assert source.count("REG_OP(Data)") == 1


@pytest.mark.parametrize("length", [True, False, 0, 65, -2, 55.0, "55", None])
def test_invalid_or_implicit_sequence_lengths_are_not_silently_changed(length):
    with pytest.raises(ValueError):
        emit_feature_branch_cpp(length)


def test_template_names_define_the_exact_nine_nodes_for_transplant():
    assert INPUT_NAMES == tuple(f"__dflash_features/input_{index}" for index in range(8))
    assert CAST_NAMES == tuple(f"__dflash_features/cast_{index}" for index in range(8))
    assert CONCAT_NAME == "__dflash_features/concat"
    assert len(set((*INPUT_NAMES, *CAST_NAMES, CONCAT_NAME))) == 17


@pytest.mark.parametrize("length", [55, -1])
def test_declared_real_ge_template_saves_reloads_and_strictly_decodes_without_weights(tmp_path, length):
    ge_root = os.environ.get("CANN_ROOT")
    pb2 = os.environ.get("TORCHAIR_SCHEMA_PB2")
    if not ge_root or not pb2:
        pytest.skip("requires explicitly declared CANN_ROOT and TORCHAIR_SCHEMA_PB2")
    from qwen35_moe_dflash.ge_fused.export import _toolchain
    from qwen35_moe_dflash.ge_export import _logged
    from qwen35_moe_dflash.air_schema import load_torchair_schema
    from qwen35_moe_dflash.hashing import sha256_file

    root = tmp_path / "branch"
    root.mkdir()
    cpp = root / "branch.cpp"
    cpp.write_text(emit_feature_branch_cpp(length))
    _, include, lib, compiler = _toolchain(ge_root, "c++", 0, 120)
    binary, air, receipt = root / "branch", root / "branch.air", root / "native.json"
    command = [compiler, "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-I", str(include),
               str(cpp), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
    if (lib / "libgraph_base.so").is_file():
        command.append("-lgraph_base")
    command.extend(["-o", str(binary)])
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(lib)
    _logged(command, folder=root, env=env, log=root / "build.log", timeout=120)
    _logged([str(binary), str(air), str(receipt)], folder=root, env=env, log=root / "export.log", timeout=120)
    native = json.loads(receipt.read_text())
    assert native["status"] == "BRANCH_TEMPLATE_EXPORTED"
    assert native["save_status"] == native["reload_status"] == 0
    assert native["model_execution"] == "NOT_RUN"
    schema = load_torchair_schema(pb2)
    graph = schema.decode(air.read_bytes(), "ModelDef")["graph"][0]
    nodes = {node["name"]: node for node in graph["op"]}
    assert len(nodes) == 18
    assert set(INPUT_NAMES + CAST_NAMES + (CONCAT_NAME,)) <= set(nodes)
    proto_types = schema.enums["ge.proto.DataType"]
    for index, name in enumerate(CAST_NAMES):
        cast = nodes[name]
        assert cast["input"] == [INPUT_NAMES[index] + ":0"]
        assert cast["input_desc"][0]["dtype"] == proto_types["DT_FLOAT"]
        assert cast["output_desc"][0]["dtype"] == proto_types["DT_FLOAT16"]
        assert cast["input_desc"][0]["shape"]["dim"] == [1, length, 2048]
        assert cast["output_desc"][0]["shape"]["dim"] == [1, length, 2048]
        attrs = {row["key"]: row["value"] for row in cast["attr"]}
        assert attrs["dst_type"] == {"i": 1}  # GE DataType domain, not protobuf FLOAT=1.
    concat = nodes[CONCAT_NAME]
    assert concat["input"] == [name + ":0" for name in CAST_NAMES]
    attrs = {row["key"]: row["value"] for row in concat["attr"]}
    assert attrs["concat_dim"] == {"i": 2} and attrs["N"] == {"i": 8}
    assert concat["output_desc"][0]["shape"]["dim"] == [1, length, 16384]
    net = next(node for node in graph["op"] if node["type"] == "NetOutput")
    assert net["input"] == [CONCAT_NAME + ":0"]
    assert len(net["input_desc"]) == 1
    assert not any(node["type"] == "FileConstant" for node in graph["op"])
    before = sha256_file(air), receipt.read_bytes()
    repeated = subprocess.run([str(binary), str(air), str(root / "retry-native.json")],
                              cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert repeated.returncode != 0
    assert sha256_file(air) == before[0] and receipt.read_bytes() == before[1]
    assert not (root / "retry-native.json").exists()


"""Host preparation/packing contracts; compiler and OMG boundaries are mocked."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from qwen35_moe_dflash.ge_weights import SafeTensorFiles, checkpoint_key
from qwen35_moe_dflash import ge_export


def write_safetensor(root, values, *, dtype="BF16", offsets=None):
    root.mkdir()
    words = values.view("<u4") >> 16
    data = words.astype("<u2").tobytes() if dtype == "BF16" else values.astype("<f2").tobytes()
    header = json.dumps({"weight": {"dtype": dtype, "shape": list(values.shape),
                                   "data_offsets": offsets or [0, len(data)]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)


def test_stream_bf16_in_small_chunks_and_verify_identity(tmp_path):
    values = np.array([[1.0, -2.0, .5], [0, 4, 8]], dtype="<f4")
    root = tmp_path / "checkpoint"
    write_safetensor(root, values)
    result = SafeTensorFiles(root).write_fp16("weight", values.shape, tmp_path / "weights.bin", chunk_elements=2)
    np.testing.assert_array_equal(np.fromfile(tmp_path / "weights.bin", dtype="<f2").reshape(values.shape), values)
    assert result["bytes"] == 12 and result["source_dtype"] == "BF16"
    assert len(result["sha256"]) == 64
    with pytest.raises(FileExistsError):
        SafeTensorFiles(root).write_fp16("weight", values.shape, tmp_path / "weights.bin")


@pytest.mark.parametrize("problem", ["shape", "offset", "overflow", "dtype"])
def test_stream_rejects_invalid_or_non_original_payload(tmp_path, problem):
    values = np.array([1e10 if problem == "overflow" else 1], dtype="<f4")
    root = tmp_path / "checkpoint"
    write_safetensor(root, values, dtype="F16" if problem == "dtype" else "BF16",
                     offsets=[0, 20] if problem == "offset" else None)
    with pytest.raises(ValueError):
        SafeTensorFiles(root).write_fp16("weight", (2,) if problem == "shape" else (1,), tmp_path / "out.bin")


def test_canonical_names_and_full_fp16_opt_in(tmp_path, monkeypatch):
    assert checkpoint_key("target.embed_tokens.weight") == ("target", "model.language_model.embed_tokens.weight")
    assert checkpoint_key("target.lm_head.weight") == ("target", "lm_head.weight")
    assert checkpoint_key("draft.fc.weight") == ("draft", "fc.weight")
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="float-baseline"):
        ge_export.prepare_pair(tmp_path / "full", tiny=False)
    assert not (tmp_path / "full").exists()


def test_preparation_never_imports_torch_or_torchair(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    blocked = tmp_path / "blocked"; blocked.mkdir()
    for module in ("torch", "torch_npu", "torchair", "transformers"):
        (blocked / f"{module}.py").write_text("raise AssertionError('framework imported by direct GE')\n")
    env = dict(os.environ, AI_RUN_DIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join((str(blocked), str(repo / "model"))))
    result = subprocess.run([sys.executable, str(repo / "targets/cpro/scripts/direct_model_air.py"),
                             "prepare", "--tiny", "--sequence-length", "8", "--output", str(tmp_path / "prepared")],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    plan = json.loads((tmp_path / "prepared/prepared.json").read_text())
    assert plan["status"] == "PREPARED" and plan["quantization"] is None
    assert [g["name"] for g in plan["graphs"]] == ["target", "draft"]
    assert not list((tmp_path / "prepared").rglob("*.air"))


def fake_ge(tmp_path):
    root = tmp_path / "cann"
    (root / "include/graph").mkdir(parents=True)
    (root / "lib64").mkdir()
    for name in ("graph.h", "operator.h"):
        (root / "include/graph" / name).write_text("mock header")
    (root / "lib64/libgraph.so").write_bytes(b"mock GE library")
    return root


@pytest.mark.parametrize("external", [False, True])
def test_prepared_pair_export_and_existing_omg_abi(tmp_path, monkeypatch, external):
    from qwen35_moe_dflash.compile import compile_air
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = ge_export.prepare_pair(prepared, tiny=True, external_weights=external)
    # Existing prepared payloads remain usable after a generator-only fix.
    # Export must regenerate construction files, never compile this stale copy.
    (prepared / "target/target.cpp").write_text("STALE_PREPARED_GENERATOR\n")
    ge = fake_ge(tmp_path)
    def tool(command, **kwargs):
        if "-o" in command:
            source = next(Path(arg) for arg in command if arg.endswith(".cpp"))
            assert "STALE_PREPARED_GENERATOR" not in source.read_text()
            Path(command[command.index("-o") + 1]).write_bytes(b"mock exporter")
        elif "--framework=1" in command:
            name = next(s.split("=", 1)[1] for s in command if s.startswith("--output="))
            (Path(kwargs["cwd"]) / f"{name}.omc").write_bytes(b"mock omc")
        else:
            Path(command[1]).write_bytes(b"mock serialized air")
        kwargs["stdout"].write("mock tool log\n")
    monkeypatch.setattr(subprocess, "run", tool)
    manifest = ge_export.export_pair(prepared, tmp_path / "air", ge)
    assert manifest["status"] == "EXPORTED" and manifest["abi_version"] == 2
    assert manifest["pipeline"][0] == "ge-cpp" and manifest["quantization"] is None
    assert manifest["source"]["fp16_payload_sha256"] == ge_export.weight_identity(plan["weights"])
    for graph in manifest["graphs"]:
        assert bool(len(graph["files"]) > 1) == external
        construction = graph["construction_files"]
        assert {Path(item["path"]).name for item in construction} == {
            f"{graph['name']}.cpp", "graph.recipe", "constants.bin"}
    ddk = tmp_path / "ddk"; master = ddk / "tools/tools_omg/master"; master.mkdir(parents=True)
    (master / "omg").write_bytes(b"mock omg"); (master / "omg").chmod(0o700)
    result = compile_air(tmp_path / "air", tmp_path / "omc", ddk, ddk_id="mock-ddk")
    assert result["status"] == "COMPILED" and result["device_validation"] == "PENDING"
    assert result["graphs"][0]["output_shapes"] == [[1, 8], [1, 8, 32]]
    assert result["graphs"][1]["output_shapes"] == [[1, 3]]
    assert plan["unique_weight_bytes"] > 0


def test_export_detects_weight_corruption_before_compile(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    plan = ge_export.prepare_pair(tmp_path / "prepared", tiny=True)
    item = next(iter(plan["weights"].values()))
    (tmp_path / "prepared" / item["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256/size"):
        ge_export.export_pair(tmp_path / "prepared", tmp_path / "air", fake_ge(tmp_path))
    assert not (tmp_path / "air").exists()


def test_export_failure_does_not_promote_partial_air(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ge_export.prepare_pair(tmp_path / "prepared", tiny=True)
    def tool(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock exporter")
        else:
            Path(command[1]).write_bytes(b"partial air")
            kwargs["stdout"].write("[ERROR] BuildGraph: tensor_desc is null.\n")
            kwargs["stdout"].write("[DEBUG] ParseJsonFormatString:add error_code E68059 success\n" * 300)
            kwargs["stdout"].write("GE recipe failed: invalid GE graph\n")
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", tool)
    with pytest.raises(RuntimeError, match="tensor_desc is null") as caught:
        ge_export.export_pair(tmp_path / "prepared", tmp_path / "air", fake_ge(tmp_path))
    assert not (tmp_path / "air/manifest.json").exists()
    failure = json.loads((tmp_path / "air/failure.json").read_text())
    assert failure["status"] == "FAILED" and failure["stage"] == "target-serialize"
    assert "First error messages:\nL1: [ERROR] BuildGraph: tensor_desc is null." in str(caught.value)
    assert "GE recipe failed: invalid GE graph" in str(caught.value)
    assert "tensor_desc is null" in failure["error"]
    assert len(str(caught.value)) < 14000


def test_log_excerpt_bounds_oversized_lines_and_ignores_error_catalog(tmp_path):
    log = tmp_path / "tool.log"
    log.write_bytes(b"[DEBUG] add error_code E68059 success\n"
                    + b"[ERROR] root cause " + b"x" * 30000 + b"\n"
                    + b"[FATAL] second cause\n" + b"tail padding\n" * 1000)
    result = ge_export._log_excerpt(log)
    first = result.split("Log tail:", 1)[0]
    assert "L2: [ERROR] root cause" in first and "[truncated]" in first
    assert "L3: [FATAL] second cause" in first
    assert "add error_code" not in first
    assert len(result) < 13000
    assert ge_export._log_excerpt(tmp_path / "missing") == ""


def test_export_detects_recipe_mutation_during_tool_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ge_export.prepare_pair(tmp_path / "prepared", tiny=True)
    def tool(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock exporter")
        else:
            Path(command[1]).write_bytes(b"mock air")
            (Path(kwargs["cwd"]) / "recipe/graph.recipe").write_bytes(b"changed graph")
    monkeypatch.setattr(subprocess, "run", tool)
    with pytest.raises(ValueError, match="SHA256/size"):
        ge_export.export_pair(tmp_path / "prepared", tmp_path / "air", fake_ge(tmp_path))
    assert not (tmp_path / "air/manifest.json").exists()

"""Mocked tool boundaries; these tests do not establish GE/OMG compatibility."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_air_smoke.py"


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("direct_air_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "source/targets/cpro/air/ge_add_air.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("// boundary-test source; not a real GE exporter\n")
    monkeypatch.setattr(module, "REPO", tmp_path / "source")
    ge = tmp_path / "ge"
    (ge / "include/graph").mkdir(parents=True)
    for name in ("graph.h", "operator_reg.h", "tensor.h"):
        (ge / "include/graph" / name).write_text("// mock GE header\n")
    (ge / "lib64").mkdir()
    (ge / "lib64/libgraph.so").write_bytes(b"mock graph library")
    (ge / "lib64/libgraph_base.so").write_bytes(b"mock graph_base library")
    compiler = tmp_path / "fake-cxx"
    compiler.write_text("mock compiler")
    monkeypatch.setattr(module.shutil, "which", lambda value: str(compiler))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == str(compiler):
            Path(command[command.index("-o") + 1]).write_bytes(b"mock exporter")
        else:
            Path(command[1]).write_bytes(b"mock AIR protobuf")

    monkeypatch.setattr(module.subprocess, "run", run)
    return module, ge, calls


def test_export_records_fixed_fp16_case_and_ge_toolchain(smoke, tmp_path):
    module, ge, calls = smoke
    result = module.export_smoke(ge, tmp_path / "air", cxx11_abi=1)
    assert result["artifact_kind"] == "ge-air-smoke"
    assert result["status"] == "AIR_EXPORTED"
    assert result["device_validation"] == "PENDING"
    assert result["graph"]["inputs"] == [module.tensor("x")]
    assert result["graph"]["outputs"] == [module.tensor("y")]
    assert (tmp_path / "air/payload/input.bin").read_bytes() == struct.pack("<4e", 0, 1, 2, 3)
    assert (tmp_path / "air/payload/expected.bin").read_bytes() == struct.pack("<4e", 1, 3, 5, 7)
    assert "-D_GLIBCXX_USE_CXX11_ABI=1" in calls[0][0]
    assert "-lgraph" in calls[0][0] and "-lgraph_base" in calls[0][0]
    assert calls[0][1]["env"]["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(ge / "lib64")
    assert calls[1][0] == [str(tmp_path / "air/build/ge_add_air"), str(tmp_path / "air/payload/add.air")]
    assert calls[1][1]["stderr"] == subprocess.STDOUT
    assert len(result["files"]) == 3
    assert result["source"]["sha256"] and result["additional_libraries"][0]["sha256"]


def test_export_rejects_success_without_air(smoke, tmp_path, monkeypatch):
    module, ge, _ = smoke
    def run(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock exporter")
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="without a nonempty add.air"):
        module.export_smoke(ge, tmp_path / "air")
    assert not (tmp_path / "air/manifest.json").exists()
    assert json.loads((tmp_path / "air/failure.json").read_text())["stage"] == "ge-serialize"


def test_export_cli_exposes_real_graph_failure_and_preserves_complete_log(smoke, tmp_path, monkeypatch, capsys):
    module, ge, _ = smoke
    root_cause = "AIR smoke failed: x failed, graph status=4294967295"
    full_log = "first-log-line\n" + "diagnostic padding\n" * 1200 + root_cause + "\n"
    def run(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock exporter")
        else:
            kwargs["stdout"].write(full_log)
            Path(command[1]).write_bytes(b"incomplete AIR must not become success")
            raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "export", "--ge-root", str(ge),
                                     "--output", str(tmp_path / "air")])
    assert module.main() == 1
    terminal = capsys.readouterr()
    assert not terminal.out
    assert root_cause in terminal.err
    assert "ge-serialize" in terminal.err and "CalledProcessError" in terminal.err
    assert str(tmp_path / "air/export.log") in terminal.err
    assert "first-log-line" not in terminal.err
    tail = terminal.err.split("Log tail (at most 8192 bytes):\n", 1)[1].rstrip()
    assert len(tail.encode("utf-8")) <= 8192
    assert (tmp_path / "air/export.log").read_text() == full_log
    failure = json.loads((tmp_path / "air/failure.json").read_text())
    assert failure["status"] == "FAILED" and failure["stage"] == "ge-serialize"
    assert failure["log_path"] == str(tmp_path / "air/export.log")
    assert failure["returncode"] == 2
    assert root_cause in failure["error"]
    assert not (tmp_path / "air/manifest.json").exists()


def test_logged_timeout_keeps_original_exception_and_flushed_tail(smoke, tmp_path, monkeypatch):
    module, _, _ = smoke
    timeout = subprocess.TimeoutExpired(["mock-exporter"], 1)
    def run(*args, **kwargs):
        kwargs["stdout"].write("last exporter progress before timeout\n")
        raise timeout
    monkeypatch.setattr(module.subprocess, "run", run)
    log = tmp_path / "timeout.log"
    with pytest.raises(module.LoggedCommandError) as caught:
        module._logged(["mock-exporter"], stage="ge-serialize", cwd=tmp_path,
                       env={}, log=log, timeout=1)
    assert caught.value.__cause__ is timeout
    assert caught.value.returncode is None
    assert caught.value.log_path == str(log)
    assert "last exporter progress before timeout" in str(caught.value)
    assert log.read_text() == "last exporter progress before timeout\n"


def test_explicit_ge_resources_take_priority_over_inherited_cann_root(smoke, tmp_path, monkeypatch):
    module, ge, _ = smoke
    resources = tmp_path / "selected-resources.json"
    resources.write_text(json.dumps({"schema_version": 1, "ge_root": str(ge)}))
    monkeypatch.setenv("CANN_ROOT", str(tmp_path / "wrong-ge"))
    monkeypatch.setattr(module, "load_resources", lambda path:
                        (json.loads(Path(path).read_text()), Path(path)))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "export", "--resources", str(resources),
                                     "--output", str(tmp_path / "air")])
    assert module.main() == 0
    manifest = json.loads((tmp_path / "air/manifest.json").read_text())
    assert manifest["ge_root"] == str(ge)


def _ddk(tmp_path):
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"mock OMG")
    omg.chmod(0o700)
    return ddk


def test_compile_uses_short_names_and_preserves_sidecars(smoke, tmp_path, monkeypatch):
    module, ge, _ = smoke
    module.export_smoke(ge, tmp_path / "air")
    ddk = _ddk(tmp_path)
    def run(command, **kwargs):
        assert command[1:] == ["--framework=1", "--model=input.air", "--output=add",
            "--target=omc", "--platform=kirinx90", "--use_origin_format=true", "--weight_data_type=FP16"]
        scratch = Path(kwargs["cwd"])
        assert tmp_path / "omc" in scratch.parents
        assert kwargs["env"]["TMPDIR"] == str(scratch)
        assert (scratch / "input.air").read_bytes() == b"mock AIR protobuf"
        (scratch / "add.omc").write_bytes(b"mock OMC")
        (scratch / "add.weights").mkdir()
        (scratch / "add.weights/data.bin").write_bytes(b"mock sidecar")
    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.compile_smoke(tmp_path / "air", tmp_path / "omc", ddk, ddk_id="mock-ddk")
    assert result["artifact_kind"] == "ge-omc-smoke"
    assert result["status"] == "COMPILED" and result["device_validation"] == "PENDING"
    assert result["graph"]["omc"] == "payload/add.omc"
    assert {record["path"] for record in result["files"]} == {
        "payload/add.omc", "payload/add.weights/data.bin", "payload/input.bin", "payload/expected.bin"}
    assert result["air_manifest_sha256"]
    assert not list((tmp_path / "omc").glob("omg-add-*"))


@pytest.mark.parametrize("mutation", ["hash", "pair-kind", "shape"])
def test_compile_rejects_untrusted_or_incompatible_air(smoke, tmp_path, mutation):
    module, ge, calls = smoke
    module.export_smoke(ge, tmp_path / "air")
    manifest_path = tmp_path / "air/manifest.json"
    if mutation == "hash":
        (tmp_path / "air/payload/add.air").write_bytes(b"changed")
    else:
        manifest = json.loads(manifest_path.read_text())
        if mutation == "pair-kind":
            manifest["artifact_kind"] = "qwen35-cpro-air-pair"
        else:
            manifest["graph"]["inputs"][0]["shape"] = [1, 8]
        manifest_path.write_text(json.dumps(manifest))
    calls.clear()
    with pytest.raises(ValueError):
        module.compile_smoke(tmp_path / "air", tmp_path / "omc", _ddk(tmp_path), ddk_id="mock")
    assert not calls
    assert not (tmp_path / "omc/manifest.json").exists()


@pytest.mark.parametrize("failure", ["nonzero", "empty", "missing"])
def test_compile_rejects_tool_failure_or_invalid_omc(smoke, tmp_path, monkeypatch, failure):
    module, ge, _ = smoke
    module.export_smoke(ge, tmp_path / "air")
    def run(command, **kwargs):
        kwargs["stdout"].write("mock OMG diagnostic\n")
        if failure == "nonzero":
            raise subprocess.CalledProcessError(1, command)
        if failure == "empty":
            (Path(kwargs["cwd"]) / "add.omc").write_bytes(b"")
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        module.compile_smoke(tmp_path / "air", tmp_path / "omc", _ddk(tmp_path), ddk_id="mock")
    assert not (tmp_path / "omc/manifest.json").exists()
    report = json.loads((tmp_path / "omc/failure.json").read_text())
    assert report["status"] == "FAILED"
    assert report["log_path"] == str(tmp_path / "omc/omg.log")
    assert report["returncode"] == (1 if failure == "nonzero" else None)
    if failure == "nonzero":
        assert "mock OMG diagnostic" in report["error"]
    assert "mock OMG diagnostic" in (tmp_path / "omc/omg.log").read_text()
    assert not list((tmp_path / "omc").glob("omg-add-*"))


@pytest.mark.parametrize("missing", ["include/graph/graph.h", "lib64/libgraph.so"])
def test_export_requires_owned_new_output_and_real_ge_paths(smoke, tmp_path, missing):
    module, ge, _ = smoke
    with pytest.raises(ValueError, match="below AI_RUN_DIR"):
        module.export_smoke(ge, tmp_path)
    output = tmp_path / "existing"; output.mkdir()
    with pytest.raises(FileExistsError):
        module.export_smoke(ge, output)
    (ge / missing).unlink()
    with pytest.raises(FileNotFoundError, match=Path(missing).name):
        module.export_smoke(ge, tmp_path / "missing-ge")
    assert not (tmp_path / "missing-ge/manifest.json").exists()


@pytest.mark.parametrize("command", ["export", "compile"])
def test_existing_output_requests_new_path_and_keeps_old_evidence(smoke, tmp_path, command):
    module, ge, _ = smoke
    output = tmp_path / "previous-failure"; output.mkdir()
    old_log = output / "export.log"
    old_log.write_text("original failure evidence\n")
    with pytest.raises(FileExistsError, match="choose a new --output path"):
        if command == "export":
            module.export_smoke(ge, output)
        else:
            module.compile_smoke(tmp_path / "unused-air", output, tmp_path / "unused-ddk", ddk_id="mock")
    assert old_log.read_text() == "original failure evidence\n"
    assert not (output / "failure.json").exists()


def test_smoke_cli_does_not_import_python_model_frameworks(tmp_path):
    for name in ("torch", "torch_npu", "torchair", "transformers"):
        (tmp_path / f"{name}.py").write_text(f'raise AssertionError("must not import {name}")\n')
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "export" in result.stdout and "compile" in result.stdout

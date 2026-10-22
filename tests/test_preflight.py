"""A broken accelerator plugin must not prevent environment diagnostics."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from qwen35_moe_dflash import cli, diagnostics


def test_preflight_reports_broken_torch_in_both_policies(tmp_path):
    shadow = tmp_path / "broken_backend"
    shadow.mkdir()
    (shadow / "torch.py").write_text(
        'raise RuntimeError("Failed to load backend: torch_npu") from '
        'ImportError("libhccl.so: cannot open shared object file")\n')
    output = tmp_path / "preflight.json"
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"schema_version": 1}))
    model_root = Path(cli.__file__).resolve().parents[1]
    env = dict(os.environ, AI_RUN_DIR=str(tmp_path),
               PYTHONPATH=os.pathsep.join([str(shadow), str(model_root)]),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "qwen35_moe_dflash.cli",
                             "--resources", str(resources), "preflight",
                             "--output", str(output)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "INVENTORY"
    assert report["torch_import_ready"] is False
    assert report["cuda_available"] is None
    assert report["reference_import_ready"] is False
    assert "libhccl.so" in report["reference_import_error"]
    assert "libhccl.so" in report["torch_error"]
    assert "Failed to load backend" in report["torch_error"]
    assert report["air_export_ready"] is False
    assert report["dopt_calibration_ready"] is False
    assert report["dopt_air_ready"] is False
    assert report["air_export_validation"] == "PENDING"
    assert report["device_validation"] == "PENDING"
    assert "missing" in report["target_weights"]


def test_air_import_loads_npu_before_bundled_torchair(monkeypatch):
    from qwen35_moe_dflash import export
    names = []
    backend = SimpleNamespace(dynamo_export=lambda: None)
    def imported(name):
        names.append(name)
        return backend if name == "torchair" else SimpleNamespace()
    monkeypatch.setattr(export.importlib, "import_module", imported)
    assert export.require_torchair() is backend
    assert names == ["torch_npu", "torchair"]


def test_air_import_retains_missing_shared_library(monkeypatch):
    from qwen35_moe_dflash import export
    def imported(name):
        raise ImportError("libhccl.so: cannot open shared object file")
    monkeypatch.setattr(export.importlib, "import_module", imported)
    with pytest.raises(RuntimeError, match="libhccl.so"):
        export.require_torchair()


def _real_preflight_env(tmp_path, extra_pythonpath=()):
    model_root = Path(cli.__file__).resolve().parents[1]
    return dict(os.environ, AI_RUN_DIR=str(tmp_path),
                PYTHONPATH=os.pathsep.join([*map(str, extra_pythonpath), str(model_root)]),
                PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")


def _run_real_preflight(tmp_path, extra_pythonpath=(), *, enable_autoload=False):
    output = tmp_path / "preflight.json"
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"schema_version": 1}))
    env = _real_preflight_env(tmp_path, extra_pythonpath)
    if enable_autoload:
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    result = subprocess.run([sys.executable, "-m", "qwen35_moe_dflash.cli",
                             "--resources", str(resources), "preflight",
                             "--output", str(output)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text())


def _broken_npu_package(tmp_path, *, backend_entrypoint=False):
    shadow = tmp_path / "broken_npu"
    package = shadow / "torch_npu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise ImportError("libhccl.so: cannot open shared object file")\n')
    metadata = shadow / "torch_npu-2.11.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch-npu\nVersion: 2.11.0\n")
    (metadata / "top_level.txt").write_text("torch_npu\n")
    if backend_entrypoint:
        (metadata / "entry_points.txt").write_text(
            "[torch.backends]\ntorch_npu = torch_npu:_autoload\n")
    return shadow


def test_real_reference_import_isolates_broken_npu_but_air_still_reports_it(tmp_path):
    # Reproduce Transformers' explicit backend import even when Torch autoload
    # is disabled. The real Torch and Qwen model imports are used in the child.
    shadow = _broken_npu_package(tmp_path)
    natural = subprocess.run(
        [sys.executable, "-c", "from transformers import Qwen3_5MoeForCausalLM"],
        cwd=tmp_path, env=_real_preflight_env(tmp_path, [shadow]),
        capture_output=True, text=True, timeout=60)
    assert natural.returncode != 0
    assert "libhccl.so" in natural.stderr
    assert "transformers" in natural.stderr
    report = _run_real_preflight(tmp_path, [shadow])
    assert report["torch_import_ready"] is True
    assert report["reference_backend"] == "cpu"
    assert report["reference_import_ready"] is True
    assert report["reference_import_error"] is None
    assert report["air_export_ready"] is False
    assert "libhccl.so" in report["air_dependency_error"]
    assert report["device_validation"] == "PENDING"


@pytest.mark.skipif(importlib.util.find_spec("torch_npu") is not None,
                    reason="this case validates the reference-only environment")
def test_real_reference_import_succeeds_without_npu_backend(tmp_path):
    report = _run_real_preflight(tmp_path)
    assert report["torch_import_ready"] is True
    assert report["reference_backend"] == "cpu"
    assert report["reference_import_ready"] is True
    assert report["reference_import_error"] is None
    assert report["air_export_ready"] is False
    assert "torch_npu" in report["air_dependency_error"]
    assert report["air_export_validation"] == "PENDING"


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired(["python", "-c", "import models"], 60,
                              stderr=b"ImportError: libhccl.so dependency stalled"),
    OSError("cannot start interpreter"),
])
def test_reference_import_probe_handles_process_failures(monkeypatch, failure):
    def failed(*args, **kwargs):
        assert args[0][0] == sys.executable
        assert kwargs["timeout"] <= 60
        assert "env" not in kwargs  # inherit the real selected environment
        raise failure
    monkeypatch.setattr(diagnostics.subprocess, "run", failed)
    report = diagnostics.probe_reference_import()
    assert report["reference_import_ready"] is False
    if isinstance(failure, subprocess.TimeoutExpired):
        assert "timed out" in report["reference_import_error"]
        assert "libhccl.so" in report["reference_import_error"]
    else:
        assert "cannot start interpreter" in report["reference_import_error"]


def test_reference_import_failure_output_is_bounded(monkeypatch):
    output = "ImportError: libhccl.so\n" + "x" * 20000 + "\nRuntimeError: model import failed"
    monkeypatch.setattr(diagnostics.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stderr=output, stdout=""))
    report = diagnostics.probe_reference_import()
    error = report["reference_import_error"]
    assert len(error) <= 8192
    assert "libhccl.so" in error
    assert "model import failed" in error
    assert "truncated" in error


def test_preflight_reports_broken_draft_dependency_with_torch_ready(tmp_path):
    shadow = tmp_path / "broken_model_dependency"
    package = shadow / "qwen35_moe_dflash"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n")
    (package / "model.py").write_text(
        'raise ImportError("draft model dependency is unavailable")\n')
    report = _run_real_preflight(tmp_path, [shadow])
    assert report["torch_import_ready"] is True
    assert report["reference_backend"] == "cpu"
    assert report["reference_import_ready"] is False
    assert "draft model dependency is unavailable" in report["reference_import_error"]
    assert report["device_validation"] == "PENDING"


def test_cpu_probe_succeeds_even_when_parent_torch_backend_autoload_fails(tmp_path):
    shadow = _broken_npu_package(tmp_path, backend_entrypoint=True)
    report = _run_real_preflight(tmp_path, [shadow], enable_autoload=True)
    assert report["torch_import_ready"] is False
    assert "libhccl.so" in report["torch_error"]
    assert report["reference_backend"] == "cpu"
    assert report["reference_import_ready"] is True
    assert report["reference_import_error"] is None
    assert report["air_export_ready"] is False
    assert report["air_dependency_error"].startswith("Skipped:")

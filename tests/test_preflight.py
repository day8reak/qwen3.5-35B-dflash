"""A broken accelerator plugin must not prevent environment diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from qwen35_moe_dflash import cli


def test_preflight_reports_broken_backend_without_importing_models(tmp_path):
    shadow = tmp_path / "broken_backend"
    shadow.mkdir()
    (shadow / "torch.py").write_text(
        'raise RuntimeError("Failed to load backend: torch_npu") from '
        'ImportError("libhccl.so: cannot open shared object file")\n')
    (shadow / "transformers.py").write_text('raise AssertionError("model import attempted")\n')
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
    assert "libhccl.so" in report["torch_error"]
    assert "Failed to load backend" in report["torch_error"]
    assert "model import attempted" not in output.read_text()
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

"""Real CPU reference must survive an installed but broken NPU plugin."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from qwen35_moe_dflash import cli


@pytest.fixture
def broken_npu(tmp_path):
    shadow = tmp_path / "broken_npu"
    package = shadow / "torch_npu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise ImportError("libhccl.so: cannot open shared object file")\n')
    metadata = shadow / "torch_npu-2.11.0.post2.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch-npu\nVersion: 2.11.0.post2\n")
    (metadata / "entry_points.txt").write_text(
        "[torch.backends]\ntorch_npu = torch_npu:_autoload\n")
    (metadata / "top_level.txt").write_text("torch_npu\n")
    return shadow


def run_env(tmp_path, shadow=None):
    model_root = Path(cli.__file__).resolve().parents[1]
    paths = ([str(shadow)] if shadow is not None else []) + [str(model_root), str(model_root.parent)]
    return dict(os.environ, AI_RUN_DIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=os.pathsep.join(paths), TORCH_DEVICE_BACKEND_AUTOLOAD="1")


def child(tmp_path, args, env):
    return subprocess.run([sys.executable, *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=90)


def test_cpu_demo_with_installed_broken_npu_matches_clean_reference(tmp_path, broken_npu):
    env = run_env(tmp_path, broken_npu)
    # Confirm this fixture represents the observed Torch automatic-load error.
    natural = child(tmp_path, ["-c", "import torch"], env)
    assert natural.returncode != 0 and "libhccl.so" in natural.stderr
    reports = []
    for name, selected_env in [("clean", run_env(tmp_path)), ("installed-npu", env)]:
        output = tmp_path / (name + ".json")
        result = child(tmp_path, ["-m", "qwen35_moe_dflash.cli", "demo", "--device", "cpu",
                       "--precision", "fp16", "--sequence-length", "8", "--prompt-token-ids", "1,2,3",
                       "--max-new-tokens", "4", "--output", str(output)], selected_env)
        assert result.returncode == 0, result.stderr
        reports.append(json.loads(output.read_text()))
    assert all(r["status"] == "PASS" for r in reports)
    for mode in ("ordinary", "dflash"):
        assert reports[0][mode]["token_ids"] == reports[1][mode]["token_ids"]
        assert reports[0][mode]["stop_reason"] == reports[1][mode]["stop_reason"] == "length"
    assert (broken_npu / "torch_npu/__init__.py").read_text().startswith("raise ImportError")


def test_air_cli_does_not_hide_broken_npu(tmp_path, broken_npu):
    env = run_env(tmp_path, broken_npu)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    output = tmp_path / "air"
    result = child(tmp_path, ["-m", "qwen35_moe_dflash.cli", "export-tiny-air",
                            "--sequence-length", "8", "--output", str(output)], env)
    assert result.returncode != 0
    assert "libhccl.so" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_process_policy_changes_both_detectors_and_rejects_mixed_air(tmp_path, broken_npu, device):
    script = """
import os, sys
from qwen35_moe_dflash.backends import configure_reference_backend
assert configure_reference_backend(sys.argv[1])
import transformers.utils as utils
import transformers.utils.import_utils as imports
assert not utils.is_torch_npu_available()
assert not imports.is_torch_npu_available(check_device=True)
assert 'torch_npu' not in sys.modules
assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] == '0'
from qwen35_moe_dflash.export import require_torchair
try:
    require_torchair()
except RuntimeError as error:
    assert 'fresh' in str(error) and 'same process' in str(error)
else:
    raise AssertionError('AIR must not inherit disabled NPU detection')
"""
    result = child(tmp_path, ["-c", script, device], run_env(tmp_path, broken_npu))
    assert result.returncode == 0, result.stderr


def test_npu_selection_is_not_disabled(tmp_path):
    script = """
import os, sys
from qwen35_moe_dflash.backends import configure_reference_backend
before = os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')
assert not configure_reference_backend('npu:0')
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == before
assert 'transformers' not in sys.modules
"""
    result = child(tmp_path, ["-c", script], run_env(tmp_path))
    assert result.returncode == 0, result.stderr

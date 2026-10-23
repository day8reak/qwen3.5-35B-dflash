"""Diagnostics and conversion help must survive a broken Torch backend import."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import qwen35_moe_dflash as package


@pytest.fixture
def broken_model_environment(tmp_path):
    shadow = tmp_path / "broken-packages"
    shadow.mkdir()
    (shadow / "torch.py").write_text(
        "raise ImportError('libhccl.so: cannot open shared object file')\n")
    (shadow / "transformers.py").write_text(
        "raise ImportError('transformers is unavailable for this diagnostic')\n")
    model_root = Path(package.__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(shadow), str(model_root), str(model_root.parent)]))
    result = subprocess.run([sys.executable, "-c", "import torch"], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and "libhccl.so" in result.stderr
    return env


def test_package_and_artifact_tools_without_model_imports(broken_model_environment, tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    script = """
import sys
from pathlib import Path
import qwen35_moe_dflash as package
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.artifacts import checked_file
from qwen35_moe_dflash.compile import omg_environment
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
assert set(package.__all__) <= set(dir(package))
assert package.DFLASH_TARGET_FEATURE_SIZE == 16384
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
path = Path(sys.argv[1])
digest = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
assert sha256_file(path, chunk_size=1) == digest
assert checked_file(path.parent, {'path': path.name, 'bytes': 3, 'sha256': digest}) == path
try:
    package.no_such_public_name
except AttributeError:
    pass
else:
    raise AssertionError('unknown names must raise AttributeError')
try:
    package.DFlashDraftModel
except ImportError as error:
    assert 'libhccl.so' in str(error)
else:
    raise AssertionError('model access must still surface its actual dependency failure')
"""
    result = subprocess.run([sys.executable, "-c", script, str(payload)], cwd=tmp_path,
                            env=broken_model_environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("args", [
    ["--help"], ["preflight", "--help"], ["compile-omc", "--help"],
])
def test_cli_help_with_broken_torch(args, broken_model_environment, tmp_path):
    result = subprocess.run([sys.executable, "-m", "qwen35_moe_dflash.cli", *args],
                            cwd=tmp_path, env=broken_model_environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("script, option", [
    ("build_hiai.py", "--headers-only"), ("run_hdc.py", "--reference"),
])
def test_hiai_tools_help_without_torch(script, option, broken_model_environment, tmp_path):
    repo = Path(package.__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(repo / "targets/cpro/scripts" / script), "--help"],
                            cwd=tmp_path, env=broken_model_environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert option in result.stdout


def test_lazy_exports_preserve_public_objects_and_hash_alias():
    exports = {
        "config": ["DFLASH_CONFIG", "DFLASH_TARGET_FEATURE_SIZE", "DFLASH_TARGET_HIDDEN_SIZE",
                   "DFLASH_TARGET_LAYER_IDS", "DFLASH_TARGET_NUM_HIDDEN_LAYERS", "Qwen35DFlashConfig",
                   "audit_official_35b_dflash_config"],
        "model": ["DFlashDraftModel", "extract_context_feature"],
        "ops": ["DFlashOps", "ModuleDFlashOps", "TorchDFlashOps"],
        "target_features": ["DFlashBaseModelOutputWithPast", "DFlashCausalLMOutputWithPast",
                            "DFlashFeatureCollector", "DFlashTargetFeatureSpec",
                            "QWEN35_35B_DFLASH_TARGET_FEATURES"],
        "weights": ["audit_dflash_checkpoint"],
    }
    expected_names = {name for names in exports.values() for name in names}
    assert set(package.__all__) == expected_names
    assert expected_names <= set(dir(package))
    namespace = {}
    exec("from qwen35_moe_dflash import *", namespace)
    for module_name, names in exports.items():
        module = importlib.import_module(f"qwen35_moe_dflash.{module_name}")
        for name in names:
            assert namespace[name] is getattr(package, name) is getattr(module, name)
    from qwen35_moe_dflash.hashing import sha256_file
    from qwen35_moe_dflash.weights import sha256_file as legacy_hash
    assert legacy_hash is sha256_file

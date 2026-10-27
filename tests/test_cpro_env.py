"""Bash environment restoration only; these fixtures do not execute vendor tools."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "targets/cpro/scripts/env.sh"


@pytest.fixture
def config(tmp_path):
    values = {
        "REPO_DIR": str(REPO), "AI_RUN_DIR": str(tmp_path / "runs"),
        "MODEL_PYTHON": str(tmp_path / "model env/bin/python3"),
        "CANN_ROOT": str(tmp_path / "cann"), "CPRO_DDK_ROOT": str(tmp_path / "ddk"),
        "CPRO_DDK_ID": "TEST-DDK", "OHOS_NATIVE_SDK": str(tmp_path / "native"),
        "HIAI_INCLUDE_DIR": str(tmp_path / "headers"),
        "HIAI_LIBRARIES": str(tmp_path / "libs/libhiai $(touch SHOULD_NOT_EXIST).so"),
        "HDC": str(tmp_path / "hdc"), "BIP": "127.0.0.1:8710", "SN": "TEST-SERIAL",
    }
    files = [values["MODEL_PYTHON"], values["HDC"], values["HIAI_LIBRARIES"],
             values["HIAI_INCLUDE_DIR"] + "/HiAiModelManagerService.h",
             values["CANN_ROOT"] + "/include/graph/graph.h", values["CANN_ROOT"] + "/lib64/libgraph.so",
             values["CPRO_DDK_ROOT"] + "/tools/tools_omg/master/omg",
             values["OHOS_NATIVE_SDK"] + "/build/cmake/ohos.toolchain.cmake",
             values["OHOS_NATIVE_SDK"] + "/llvm/bin/clang++"]
    for name in files:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
        path.chmod(0o700)
    (Path(values["OHOS_NATIVE_SDK"]) / "sysroot").mkdir()
    return tmp_path / "private.env", values


def write_config(path, values):
    path.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n")


def bash(path, script, *, env=None):
    return subprocess.run(["bash", "-c", script, "env-test", str(SCRIPT), str(path), sys.executable],
                          env={**os.environ, **(env or {})}, cwd=path.parent,
                          capture_output=True, text=True)


def test_restores_once_and_twice_without_path_growth_or_stale_artifacts(config):
    path, values = config
    values.update(FUSED_PROBE_RUN=str(path.parent / "deleted"), TBE_DIAG_JOB=str(path.parent / "deleted"),
                  GE_CUSTOM_BINDINGS=str(path.parent / "deleted.json"))
    write_config(path, values)
    result = bash(path, '''
set -eu
source "$1" "$2"
first_path=$PATH
source "$1" "$2"
test "$PATH" = "$first_path"
"$3" -c 'import json,os; print(json.dumps(dict(os.environ)))'
''', env={"FIRMWARE_ID": "STALE-FIRMWARE", "QWEN35_RESOURCES": "/deleted/resources.json"})
    assert result.returncode == 0, result.stderr
    restored = json.loads(result.stdout)
    for key, value in values.items():
        if key not in {"FUSED_PROBE_RUN", "TBE_DIAG_JOB", "GE_CUSTOM_BINDINGS"}:
            assert restored[key] == value
    for key in ("FUSED_PROBE_RUN", "TBE_DIAG_JOB", "GE_CUSTOM_BINDINGS", "FIRMWARE_ID"):
        assert key not in restored
    assert restored["PYTHONPATH"] == f"{REPO}/model:{REPO}"
    assert restored["PATH"].split(":")[0] == str(Path(values["MODEL_PYTHON"]).parent)
    assert restored["QWEN35_RESOURCES"] == str(REPO / "specs/resources.json")
    assert Path(values["AI_RUN_DIR"]).is_dir()
    assert not (path.parent / "SHOULD_NOT_EXIST").exists()


@pytest.mark.parametrize("field,value", [("BIP", "127.0.0.1"), ("BIP", "127.0.0.1:99999"),
                                         ("SN", ""), ("MODEL_PYTHON", "/missing/python"),
                                         ("AI_RUN_DIR", str(REPO / "forbidden-run"))])
def test_invalid_config_is_atomic_and_reports_field(config, field, value):
    path, values = config
    values[field] = value
    write_config(path, values)
    result = bash(path, '''
set -eu
export REPO_DIR=ORIGINAL
before_path=$PATH
if source "$1" "$2"; then exit 80; fi
test "$REPO_DIR" = ORIGINAL
test "$PATH" = "$before_path"
''')
    assert result.returncode == 0, result.stderr
    assert "cpro-env:" in result.stderr
    assert not (path.parent / "runs").exists()


def test_first_source_creates_private_config_without_overwriting(config):
    path, values = config
    result = bash(path, 'source "$1" "$2"', env=values)
    assert result.returncode == 1
    assert "created" in result.stderr
    assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()
    assert "TEST-SERIAL" in original.decode()
    again = bash(path, 'source "$1" "$2"')
    assert again.returncode == 0, again.stderr
    assert path.read_bytes() == original

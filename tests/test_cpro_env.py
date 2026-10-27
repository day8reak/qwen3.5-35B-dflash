"""Bash environment restoration only; these fixtures do not execute vendor tools."""
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "targets/cpro/scripts/env.sh"


@pytest.fixture
def config(tmp_path):
    values = {
        "REPO_DIR": str(REPO), "AI_RUN_DIR": str(tmp_path / "runs"),
        "CANN_ROOT": str(tmp_path / "cann"), "CPRO_DDK_ROOT": str(tmp_path / "ddk"),
        "CPRO_DDK_ID": "TEST-DDK", "OHOS_NATIVE_SDK": str(tmp_path / "native"),
        "HIAI_INCLUDE_DIR": str(tmp_path / "headers"),
        "HIAI_LIBRARIES": str(tmp_path / "libs/libhiai $(touch SHOULD_NOT_EXIST).so"),
        "HDC": str(tmp_path / "hdc"), "BIP": "127.0.0.1:8710", "SN": "TEST-SERIAL",
    }
    files = [values["HDC"], values["HIAI_LIBRARIES"],
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


def test_restores_once_and_twice_without_path_growth_or_changing_run_variables(config):
    path, values = config
    runtime = {"FUSED_PROBE_RUN": "/invocation/probes", "TBE_DIAG_JOB": "/invocation/diagnosis",
               "GE_CUSTOM_BINDINGS": "/invocation/bindings.json", "FIRMWARE_ID": "INVOCATION-FIRMWARE",
               "HIAI_RUNTIME_VERSION": "INVOCATION-RUNTIME"}
    write_config(path, values)
    # Legacy config entries must not overwrite command-specific values either.
    with path.open("a") as stream:
        stream.write("FUSED_PROBE_RUN=/old/config/probes\n")
    result = bash(path, '''
set -eu
unset MODEL_PYTHON CONDA_PREFIX
original_python=$(type -P python3)
source "$1" "$2"
first_path=$PATH
source "$1" "$2"
test "$PATH" = "$first_path"
test "$(type -P python3)" = "$original_python"
test -z "${MODEL_PYTHON+x}"
"$3" -c 'import json,os; print(json.dumps(dict(os.environ)))'
''', env={**runtime, "QWEN35_RESOURCES": "/deleted/resources.json", "PYTHONPATH": "/vendor/tbe:/vendor/python"})
    assert result.returncode == 0, result.stderr
    restored = json.loads(result.stdout)
    for key, value in values.items():
        assert restored[key] == value
    for key, value in runtime.items():
        assert restored[key] == value
    assert restored["PYTHONPATH"] == f"{REPO}/model:{REPO}:/vendor/tbe:/vendor/python"
    assert restored["PATH"] == os.environ["PATH"]
    assert restored["QWEN35_RESOURCES"] == str(REPO / "specs/resources.json")
    assert Path(values["AI_RUN_DIR"]).is_dir()
    assert not (path.parent / "SHOULD_NOT_EXIST").exists()


@pytest.mark.parametrize("field,value", [("BIP", "127.0.0.1"), ("BIP", "127.0.0.1:99999"),
                                         ("SN", ""), ("OHOS_NATIVE_SDK", "/missing/native"),
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
    activated = {"DDK_PATH": values["CPRO_DDK_ROOT"],
                 "CONDA_PREFIX": "", "MODEL_PYTHON": "/obsolete/python",
                 "PATH": str(path.parent) + os.pathsep + os.environ["PATH"],
                 "HIAI_INCLUDE_DIR": values["HIAI_INCLUDE_DIR"],
                 "HIAI_LIBRARIES": values["HIAI_LIBRARIES"]}
    result = bash(path, 'source "$1" "$2"', env={**values, "FUSED_PROBE_RUN": "/temporary/probes",
                                                "FIRMWARE_ID": "TEMPORARY", **activated})
    assert result.returncode == 1
    assert "created" in result.stderr
    assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()
    assert "TEST-SERIAL" in original.decode()
    assert "FUSED_PROBE_RUN" not in original.decode() and "FIRMWARE_ID" not in original.decode()
    assignments = {line.split("=", 1)[0] for line in original.decode().splitlines()
                   if line and not line.startswith("#")}
    assert assignments == {"REPO_DIR", "AI_RUN_DIR", "CANN_ROOT", "CPRO_DDK_ID", "OHOS_NATIVE_SDK", "BIP", "SN"}
    assert "HDC服务端的IP:端口" in original.decode()
    again = bash(path, 'source "$1" "$2"', env=activated)
    assert again.returncode == 0, again.stderr
    assert path.read_bytes() == original


def test_reuses_selected_vendor_environment_without_redeclaring_its_variables(config):
    path, values = config
    configured = {key: value for key, value in values.items()
                  if key not in {"CPRO_DDK_ROOT", "HDC"}}
    write_config(path, configured)
    with path.open("a") as stream:
        stream.write("MODEL_PYTHON=/old/config/python\n")
    inherited = {"DDK_PATH": values["CPRO_DDK_ROOT"], "CPRO_DDK_ROOT": "/previous/ddk",
                 "CONDA_PREFIX": "", "MODEL_PYTHON": "/obsolete/python", "HDC": "",
                 "TOOLCHAIN_HOME": values["CPRO_DDK_ROOT"], "HIAI_VERSION": "/vendor/omg",
                 "LD_LIBRARY_PATH": "/vendor/omg/lib64:/vendor/tbe/lib64",
                 "PYTHONPATH": "/vendor/tbe/ops:/vendor/package/python",
                 "PATH": (str(path.parent) + os.pathsep) * 2 + os.environ["PATH"]}
    result = bash(path, '''
set -eu
source "$1" "$2"
"$3" -c 'import json,os; print(json.dumps(dict(os.environ)))'
''', env=inherited)
    assert result.returncode == 0, result.stderr
    restored = json.loads(result.stdout)
    for key in ("CPRO_DDK_ROOT", "HDC"):
        assert restored[key] == values[key]
    for key in ("DDK_PATH", "TOOLCHAIN_HOME", "HIAI_VERSION", "LD_LIBRARY_PATH"):
        assert restored[key] == inherited[key]
    assert restored["PYTHONPATH"] == f"{REPO}/model:{REPO}:" + inherited["PYTHONPATH"]
    assert restored["PATH"] == inherited["PATH"]
    assert restored["MODEL_PYTHON"] == inherited["MODEL_PYTHON"]


@pytest.mark.parametrize("mode", ["unique", "ambiguous", "host-only"])
def test_auto_hiai_is_confined_to_selected_ddk_and_checks_architecture(config, mode):
    path, values = config
    ddk = Path(values["CPRO_DDK_ROOT"])
    header = ddk / "tools/tools_ascendc/include/ddk/HiAiModelManagerService.h"
    header.parent.mkdir(parents=True)
    header.write_text("// selected DDK header fixture\n")

    def elf(name, machine):
        library = ddk / name
        library.parent.mkdir(parents=True, exist_ok=True)
        ident = b"\x7fELF\x02\x01\x01" + bytes(9)
        library.write_bytes(struct.pack("<16sHHIQQQIHHHHHH", ident, 3, machine, 1,
                                        0, 0, 0, 0, 64, 56, 0, 64, 0, 0))
        return library

    elf("tools/tools_omg/master/lib64/libhiai.so", 62)
    elf("libs/linux/arm64/libhiai.so", 183)  # Architecture alone cannot select OHOS.
    selected = elf("libs/ohos/arm64/libhiai.so", 62 if mode == "host-only" else 183)
    if mode == "ambiguous":
        elf("another/ohos/arm64/libhiai.so", 183)
    values.update(HIAI_INCLUDE_DIR="auto", HIAI_LIBRARIES="auto")
    write_config(path, values)
    result = bash(path, '''
set -eu
source "$1" "$2"
"$3" -c 'import json,os; print(json.dumps([os.environ["HIAI_INCLUDE_DIR"], os.environ["HIAI_LIBRARIES"]]))'
''')
    if mode == "unique":
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == [str(header.parent), str(selected)]
    else:
        assert result.returncode != 0
        assert "set HIAI_LIBRARIES explicitly" in result.stderr
        assert not (path.parent / "runs").exists()

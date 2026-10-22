"""HIAI build CLI configuration with fake SDK inputs; no native compilation."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def build_case(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/build_hiai.py"
    spec = importlib.util.spec_from_file_location("hiai_build_config_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sdk = tmp_path / "sdk/native"
    toolchain = sdk / "build/cmake/ohos.toolchain.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text("# fake OHOS toolchain; never executed\n")
    compiler = sdk / "llvm/bin/clang++"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\nexit 99\n")
    compiler.chmod(0o700)
    (sdk / "sysroot").mkdir()
    include = tmp_path / "hiai-include"
    include.mkdir()
    (include / "HiAiModelManagerService.h").write_text("// fake header; no native compilation\n")
    resources = tmp_path / "resources.json"
    config = {"schema_version": 1, "ohos_native_sdk": str(sdk),
              "hiai_include_dir": str(include), "hiai_libraries": []}
    resources.write_text(json.dumps(config))
    output = tmp_path / "output"
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("HIAI_LIBRARIES", raising=False)
    calls = []

    def no_compile(argv, **kwargs):
        assert kwargs["check"] is True
        assert Path(kwargs["cwd"]) == output
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(module.subprocess, "run", no_compile)

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", [str(script), "--resources", str(resources),
                                          "--output", str(output), "--headers-only", *arguments])
        return module.main()

    return {"run": run, "sdk": sdk, "toolchain": toolchain, "compiler": compiler,
            "resources": resources, "config": config, "output": output, "calls": calls}


@pytest.mark.parametrize("api", [None, "21", "24"])
def test_sdk_default_is_preserved_and_explicit_api_is_forwarded(build_case, api):
    case = build_case
    arguments = () if api is None else ("--compatible-sdk-version", api)
    case["run"](*arguments)
    assert len(case["calls"]) == 2
    configure, build = case["calls"]
    prefix = "-DOHOS_COMPATIBLE_SDK_VERSION="
    assert [argument for argument in configure if argument.startswith(prefix)] == (
        [] if api is None else [prefix + api])
    assert "-DCMAKE_TOOLCHAIN_FILE=" + str(case["toolchain"]) in configure
    assert "-DBUILD_HIAI_RUNNER=OFF" in configure
    assert build[:2] == ["cmake", "--build"]
    command = json.loads((case["output"] / "command.json").read_text())
    assert command["argv"] == configure and command["headers_only"] is True
    assert command["compatible_sdk_version_requested"] == (None if api is None else int(api))
    assert command["sdk_api_policy"] == ("sdk_default" if api is None else "explicit")
    assert command["sdk_toolchain_sha256"] == hashlib.sha256(case["toolchain"].read_bytes()).hexdigest()
    result = json.loads((case["output"] / "result.json").read_text())
    assert result["status"] == "HEADERS_COMPILED" and result["device_validation"] == "PENDING"


@pytest.mark.parametrize("api", ["0", "-1", "2.5", "21suffix", ""])
def test_invalid_api_is_an_argparse_error_before_output_or_build(build_case, capsys, api):
    with pytest.raises(SystemExit) as caught:
        build_case["run"]("--compatible-sdk-version", api)
    assert caught.value.code == 2
    assert "--compatible-sdk-version" in capsys.readouterr().err
    assert not build_case["output"].exists()
    assert not build_case["calls"]


@pytest.mark.parametrize("problem,resource_name", [
    ("sdk_parent", "ohos.toolchain.cmake"),
    ("toolchain_missing", "ohos.toolchain.cmake"),
    ("toolchain_directory", "ohos.toolchain.cmake"),
    ("compiler_missing", "clang++"),
    ("compiler_nonexecutable", "clang++"),
    ("sysroot_missing", "sysroot"),
])
def test_incomplete_sdk_fails_clearly_before_creating_output(build_case, problem, resource_name):
    case = build_case
    if problem == "sdk_parent":
        config = dict(case["config"], ohos_native_sdk=str(case["sdk"].parent))
        case["resources"].write_text(json.dumps(config))
    elif problem == "toolchain_missing":
        case["toolchain"].unlink()
    elif problem == "toolchain_directory":
        case["toolchain"].unlink()
        case["toolchain"].mkdir()
    elif problem == "compiler_missing":
        case["compiler"].unlink()
    elif problem == "compiler_nonexecutable":
        case["compiler"].chmod(0o600)
    elif problem == "sysroot_missing":
        (case["sdk"] / "sysroot").rmdir()
    with pytest.raises((ValueError, FileNotFoundError, PermissionError)) as caught:
        case["run"]()
    assert resource_name in str(caught.value)
    assert not case["output"].exists()
    assert not case["calls"]

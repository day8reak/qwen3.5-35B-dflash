"""Offline tests for the explicit native-only refresh; no compiler/device claims."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from qwen35_moe_dflash import hiai_probe_build as build
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.hiai_runtime_probe import Snapshot


@pytest.fixture
def refresh(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source_run = tmp_path / "historical-probe"
    old_runner = source_run / "native/build/qwen35_hiai_op_probe"
    old_runner.parent.mkdir(parents=True)
    old_runner.write_bytes(b"historical diagnostic binary")
    repository = tmp_path / "current-checkout"
    source = repository / "targets/cpro/runtime"
    (source / "src").mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("current probe CMake source")
    current_cpp = source / "src/hiai_op_probe.cpp"
    current_cpp.write_text("current diagnostic source")
    toolchain = tmp_path / "recorded-sdk/native/build/cmake/ohos.toolchain.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text("recorded SDK toolchain")
    header = tmp_path / "recorded-headers/HiAiModelManagerService.h"
    header.parent.mkdir()
    header.write_text("recorded HIAI header")
    originals = []
    for name in ("libhiai.so", "libhiai_ir_build.so"):
        library = tmp_path / "recorded-ddk" / name
        library.parent.mkdir(exist_ok=True)
        library.write_bytes(b"recorded library " + name.encode())
        originals.append(library)
    original = {
        "argv": ["do-not-execute-historical-program", "-S", "/old/source", "-B", "/old/build",
                 f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", f"-DHIAI_INCLUDE_DIR={header.parent}",
                 "-DOHOS_ARCH=arm64-v8a", "-DOHOS_PLATFORM=OHOS", "-DOHOS_STL=c++_shared",
                 "-DOHOS_COMPATIBLE_SDK_VERSION=21", "-DCMAKE_CXX_FLAGS=-DUNTRUSTED",
                 "-DHIAI_LIBRARIES=/untrusted/lib.so", "-DBUILD_HOST_TESTS=ON"],
        "libraries": [{"path": str(path), "sha256": sha256_file(path)} for path in originals],
        "sdk_toolchain_sha256": sha256_file(toolchain), "header_sha256": sha256_file(header),
        "sources": [{"path": "/removed/historical/source.cpp", "sha256": "a" * 64}],
    }
    command_path = source_run / "native/command.json"
    command_path.write_text(json.dumps(original))
    snapshot, frozen = Snapshot(), Snapshot()
    snapshot.add(command_path, role="source-metadata")
    snapshot.add(old_runner, role="native-runner")
    root = tmp_path / "new-comparison"
    (root / "reused/lib").mkdir(parents=True)
    libraries = []
    for path in originals:
        snapshot.add(path, role="runtime-library")
        target = root / "reused/lib" / path.name
        shutil.copyfile(path, target)
        frozen.add(target, role="frozen-payload")
        libraries.append(target)
    state = {"root": root, "source_run": source_run, "old_runner": old_runner,
             "source": source, "current_cpp": current_cpp, "toolchain": toolchain, "header": header,
             "originals": originals, "libraries": libraries, "frozen": frozen, "calls": [], "hook": None,
             "validated": {"source": source_run, "runner": old_runner, "command": original, "snapshot": snapshot},
             "dependencies": {"needed": ["libhiai.so", "libhiai_ir_build.so", "libc.so"],
                              "soname": None, "rpath": None, "runpath": None}}
    monkeypatch.setattr(build, "repository_root", lambda: repository)

    def logged(argv, **kwargs):
        state["calls"].append({"argv": list(argv), **kwargs})
        if argv[:2] == ["cmake", "--build"]:
            runner = Path(argv[2]) / "qwen35_hiai_op_probe"
            runner.parent.mkdir(parents=True)
            runner.write_bytes(b"new diagnostic binary")
        if state["hook"]:
            state["hook"](argv)

    monkeypatch.setattr(build, "_logged", logged)
    monkeypatch.setattr(build, "dynamic_info", lambda path: copy.deepcopy(state["dependencies"]))
    return state


def _run(state):
    return build.build_probe(state["root"], state["validated"], state["libraries"], state["frozen"], 43)


def test_refresh_builds_current_probe_once_against_frozen_recorded_libraries(refresh):
    original_bytes = {path: path.read_bytes() for path in [refresh["old_runner"], *refresh["originals"]]}
    runner, record = _run(refresh)
    configure, compile_call = refresh["calls"]
    configured = configure["argv"]
    assert configured[:5] == ["cmake", "-S", str(refresh["source"]), "-B", str(runner.parent)]
    expected_settings = {
        "CMAKE_TOOLCHAIN_FILE": str(refresh["toolchain"]), "HIAI_INCLUDE_DIR": str(refresh["header"].parent),
        "OHOS_ARCH": "arm64-v8a", "OHOS_PLATFORM": "OHOS", "OHOS_STL": "c++_shared",
        "OHOS_COMPATIBLE_SDK_VERSION": "21", "BUILD_HOST_TESTS": "OFF", "BUILD_HIAI_RUNNER": "ON",
        "BUILD_HIAI_OBJECTS": "OFF", "HIAI_LIBRARIES": ";".join(map(str, refresh["libraries"])),
    }
    assert dict(arg[2:].split("=", 1) for arg in configured[5:]) == expected_settings
    assert compile_call["argv"] == ["cmake", "--build", str(runner.parent), "--target", "qwen35_hiai_op_probe", "-j", "4"]
    assert all(call["timeout"] == 43 and call["folder"] == runner.parent.parent for call in refresh["calls"])
    assert all(path.read_bytes() == before for path, before in original_bytes.items())
    assert record["mode"] == "REFRESHED_NATIVE" and record["sha256"] == sha256_file(runner)
    assert record["historical_binary"] == {"path": str(refresh["old_runner"]), "sha256": sha256_file(refresh["old_runner"])}
    assert record["dependencies"] == refresh["dependencies"]
    provenance = record["build_command"]
    assert provenance["libraries"] == [
        {"path": str(new), "original_path": str(old), "sha256": sha256_file(old)}
        for new, old in zip(refresh["libraries"], refresh["originals"])
    ]
    assert {Path(item["path"]) for item in provenance["sources"]} == {
        refresh["source"] / "CMakeLists.txt", refresh["current_cpp"]}
    assert all(item["sha256"] == sha256_file(Path(item["path"])) for item in provenance["sources"])
    assert provenance["source_command_sha256"] == sha256_file(refresh["source_run"] / "native/command.json")
    assert provenance["header_sha256"] == sha256_file(refresh["header"])
    assert provenance["sdk_toolchain_sha256"] == sha256_file(refresh["toolchain"])
    folder = refresh["root"] / "refreshed-native"
    result = json.loads((folder / "result.json").read_text())
    assert result == record["build_result"]
    assert result["device_validation"] == "NOT_RUN"
    assert result["command_sha256"] == sha256_file(folder / "command.json")
    assert result["build_command_sha256"] == sha256_file(folder / "build-command.json")
    assert str(runner) in refresh["frozen"].files
    refresh["frozen"].verify()


def test_unspecified_sdk_api_is_not_invented(refresh):
    original = refresh["validated"]["command"]
    original["argv"] = [arg for arg in original["argv"] if not arg.startswith("-DOHOS_COMPATIBLE_SDK_VERSION=")]
    _run(refresh)
    assert not any("OHOS_COMPATIBLE_SDK_VERSION" in arg for arg in refresh["calls"][0]["argv"])


@pytest.mark.parametrize("changed", ["toolchain", "header", "original-library", "frozen-library"])
def test_changed_recorded_or_frozen_inputs_fail_before_any_build(refresh, changed):
    path = {"toolchain": refresh["toolchain"], "header": refresh["header"],
            "original-library": refresh["originals"][0], "frozen-library": refresh["libraries"][0]}[changed]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256|source changed"):
        _run(refresh)
    assert refresh["calls"] == []
    assert not (refresh["root"] / "refreshed-native").exists()


@pytest.mark.parametrize("key", ["header_sha256", "sdk_toolchain_sha256"])
@pytest.mark.parametrize("value", [None, "not-a-sha256"])
def test_missing_or_invalid_sdk_header_binding_cannot_silently_refresh(refresh, key, value):
    refresh["validated"]["command"][key] = value
    with pytest.raises(ValueError, match="recorded .*SHA256"):
        _run(refresh)
    assert refresh["calls"] == []


def test_frozen_inventory_must_include_each_recorded_link_input(refresh):
    refresh["libraries"].pop()
    with pytest.raises(ValueError, match="frozen runtime set lacks recorded link input"):
        _run(refresh)
    assert refresh["calls"] == []


@pytest.mark.parametrize("setting", [
    "-DOHOS_ARCH=x86_64", "-DOHOS_PLATFORM=ANDROID", "-DOHOS_STL=c++_static",
    "-DOHOS_COMPATIBLE_SDK_VERSION=0", "-DOHOS_COMPATIBLE_SDK_VERSION=21;execute",
    "-DCMAKE_TOOLCHAIN_FILE=relative/toolchain.cmake", "-DHIAI_INCLUDE_DIR=relative/headers",
    "-DHIAI_INCLUDE_DIR=/absolute/path;inject", "-DCMAKE_TOOLCHAIN_FILE=/path\nwithcontrol",
])
def test_invalid_recorded_settings_are_rejected_before_build(refresh, setting):
    key = setting.split("=", 1)[0] + "="
    original = refresh["validated"]["command"]
    original["argv"] = [arg for arg in original["argv"] if not arg.startswith(key)] + [setting]
    with pytest.raises(ValueError, match="recorded|positive integer"):
        _run(refresh)
    assert refresh["calls"] == []


def test_conflicting_allowlisted_settings_are_not_resolved_by_order(refresh):
    refresh["validated"]["command"]["argv"].append("-DOHOS_ARCH=x86_64")
    with pytest.raises(ValueError, match="conflicting recorded CMake setting"):
        _run(refresh)
    assert refresh["calls"] == []


@pytest.mark.parametrize("stage", ["configure", "build"])
def test_build_failure_is_retained_without_a_success_result(refresh, stage):
    def fail(argv):
        if (argv[:2] == ["cmake", "--build"]) == (stage == "build"):
            raise subprocess.CalledProcessError(7, argv)
    refresh["hook"] = fail
    with pytest.raises(subprocess.CalledProcessError):
        _run(refresh)
    failure = json.loads((refresh["root"] / "refreshed-native/failure.json").read_text())
    assert failure["status"] == "FAILED" and failure["stage"] == stage
    assert failure["board_started"] is False
    assert len(refresh["calls"]) == (1 if stage == "configure" else 2)
    assert not (refresh["root"] / "refreshed-native/result.json").exists()


@pytest.mark.parametrize("changed,stage", [
    ("current-source", "configure"), ("current-source", "build"), ("toolchain", "build"),
    ("header", "configure"), ("frozen-library", "build"), ("build-metadata", "configure"),
])
def test_mutation_during_build_stops_refresh_and_never_promotes_runner(refresh, changed, stage):
    def mutate(argv):
        if (argv[:2] == ["cmake", "--build"]) != (stage == "build"):
            return
        path = {"current-source": refresh["current_cpp"], "toolchain": refresh["toolchain"],
                "header": refresh["header"], "frozen-library": refresh["libraries"][0],
                "build-metadata": refresh["root"] / "refreshed-native/command.json"}[changed]
        path.write_bytes(path.read_bytes() + b" changed while building")
    refresh["hook"] = mutate
    with pytest.raises(ValueError, match="snapshotted source changed"):
        _run(refresh)
    folder = refresh["root"] / "refreshed-native"
    assert json.loads((folder / "failure.json").read_text())["stage"] == stage
    assert not (folder / "result.json").exists()
    assert len(refresh["calls"]) == (1 if stage == "configure" else 2)


@pytest.mark.parametrize("fault", ["missing", "empty", "symlink", "invalid-elf"])
def test_invalid_build_output_is_not_accepted_as_a_refreshed_runner(refresh, monkeypatch, fault):
    def damage(argv):
        if argv[:2] != ["cmake", "--build"]:
            return
        runner = Path(argv[2]) / "qwen35_hiai_op_probe"
        if fault == "missing":
            runner.unlink()
        elif fault == "empty":
            runner.write_bytes(b"")
        elif fault == "symlink":
            runner.unlink()
            runner.symlink_to(refresh["old_runner"])
    refresh["hook"] = damage
    if fault == "invalid-elf":
        def reject(path):
            raise ValueError("invalid ELF fixture")
        monkeypatch.setattr(build, "dynamic_info", reject)
    with pytest.raises(ValueError, match="nonempty regular runner|invalid ELF"):
        _run(refresh)
    folder = refresh["root"] / "refreshed-native"
    assert json.loads((folder / "failure.json").read_text())["stage"] == "validate-binary"
    assert not (folder / "result.json").exists()


def test_refresh_refuses_existing_native_directory_without_overwriting(refresh):
    folder = refresh["root"] / "refreshed-native"
    folder.mkdir()
    sentinel = folder / "command.json"
    sentinel.write_bytes(b"preserve previous evidence")
    with pytest.raises(FileExistsError):
        _run(refresh)
    assert sentinel.read_bytes() == b"preserve previous evidence"
    assert refresh["calls"] == []


def test_refresh_records_native_donor_command_separately_from_compile_only_models(refresh):
    donor = refresh["validated"]["source"]
    models = donor.parent / "compile-only-models"
    models.mkdir()
    refresh["validated"]["source"] = models
    refresh["validated"]["native_source"] = donor
    runner, record = _run(refresh)
    assert runner.is_file()
    assert record["build_command"]["source_command"] == str(donor / "native/command.json")
    assert record["build_command"]["source_command_sha256"] == sha256_file(donor / "native/command.json")
    assert record["historical_binary"]["path"] == str(refresh["old_runner"])
    assert not (models / "native").exists()

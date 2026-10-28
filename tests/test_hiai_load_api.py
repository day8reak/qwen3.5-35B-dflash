"""Selected-DDK API inventory must stay read-only and retain device evidence."""
import json
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.hiai_load_api import inspect_load_api


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    # An unrelated current shell must not redirect this recorded-DDK inspection.
    monkeypatch.setenv("HIAI_INCLUDE_DIR", str(tmp_path / "wrong-include"))
    monkeypatch.setenv("OHOS_NATIVE_SDK", str(tmp_path / "wrong-sdk"))
    monkeypatch.setenv("HIAI_LIBRARIES", str(tmp_path / "wrong-library.so"))
    job = tmp_path / "full-draft"
    write_json(job / "result.json", {"artifact_kind": "qwen35-full-draft-job", "components": {}})
    out = job / "load-diagnostic"
    write_json(out / ".owner.json", {"artifact_kind": "qwen35-hiai-descriptor-diagnostic"})
    include, sdk = tmp_path / "ddk/include", tmp_path / "sdk/native"
    include.mkdir(parents=True)
    service = include / "HiAiModelManagerService.h"
    service.write_text("class AiModelDescription {};\n")
    built = include / "built_model.h"
    built.write_text("namespace hiai {\nclass IBuiltModel {\npublic:\n"
                     "  virtual int RestoreFromFile(const char* path) = 0;\n};\n}\n")
    toolchain = sdk / "build/cmake/ohos.toolchain.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text("# fixture toolchain, never executed\n")
    library = tmp_path / "ddk/lib/libhiai.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"selected fixture library, never dlopened")
    cpu = library.with_name("libcpucl.so")
    cpu.write_bytes(b"unrelated library")
    command = {"argv": ["cmake", "-DHIAI_INCLUDE_DIR=" + str(include),
                        "-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain)],
               "header_sha256": sha256_file(service), "sdk_toolchain_sha256": sha256_file(toolchain)}
    command_path = out / "native/command.json"
    write_json(command_path, command)
    digest = sha256_file(command_path)
    write_json(out / "native/result.json", {"status": "LINKED", "command_sha256": digest})
    discovery = {"command_sha256": digest, "libraries": [
        {"name": p.name, "path": str(p), "sha256": sha256_file(p)} for p in (library, cpu)]}
    write_json(out / "result.json", {"status": "INSPECTED", "runtime_version": "fixture-runtime",
                                     "libraries": {"discovery": discovery}})
    (out / "probe.json").write_text('{"retained":"device observations"}\n')
    model = job / "compilation/model.omc"
    model.parent.mkdir()
    model.write_bytes(b"retained model must never be loaded")
    return {"job": job, "out": out, "include": include, "sdk": sdk,
            "library": library, "service": service, "toolchain": toolchain,
            "command": command_path, "model": model}


def fake_nm(selected, calls):
    def execute(argv, **kwargs):
        calls.append(argv)
        assert argv == [str(selected["sdk"] / "llvm/bin/llvm-nm"), "--dynamic",
                        "--defined-only", "--demangle", str(selected["library"])]
        assert kwargs["timeout"] == 30 and kwargs["check"] is False
        assert not any("hdc" in x or "omg" in x or "runner" in x for x in argv)
        return subprocess.CompletedProcess(argv, 0,
            "00000010 T hiai::CreateBuiltModel()\n"
            "00000020 T HIAI_MR_BuiltModel_RestoreFromFile\n"
            "00000030 T unrelated_function\n", "")
    return execute


def test_reads_recorded_headers_and_symbols_without_touching_prior_evidence(selected):
    s, calls = selected, []
    retained = [s["out"] / "probe.json", s["out"] / "result.json", s["command"], s["model"]]
    before = {p: (p.read_bytes(), p.stat().st_ino) for p in retained}
    report, text = inspect_load_api(s["job"], run_command=fake_nm(s, calls))
    assert report["status"] == "INSPECTED" and len(calls) == 1
    assert report["runtime_version_observed"] == "fixture-runtime"
    assert "RestoreFromFile(const char* path)" in text
    assert "HIAI_MR_BuiltModel_RestoreFromFile" in text
    assert "unrelated_function" not in text
    assert len(report["symbols"]) == 1
    assert json.loads((s["out"] / "api-report.json").read_text()) == report
    assert (s["out"] / "api-report.txt").read_text() == text
    # Repeat replaces only the two named inventory reports, not the probe/model.
    second, second_text = inspect_load_api(s["job"], run_command=fake_nm(s, calls))
    assert second == report and second_text == text and len(calls) == 2
    assert {p: (p.read_bytes(), p.stat().st_ino) for p in retained} == before


@pytest.mark.parametrize("fault", ["command", "service", "toolchain"])
def test_build_or_toolchain_identity_drift_stops_before_symbol_tool(selected, fault):
    path = selected[fault]
    path.write_text(path.read_text() + "\n ")
    calls = []
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        inspect_load_api(selected["job"], run_command=fake_nm(selected, calls))
    assert not calls
    assert not (selected["out"] / "api-report.json").exists()


def test_changed_library_is_incomplete_without_running_nm_on_it(selected):
    selected["library"].write_bytes(b"changed DDK")
    calls = []
    report, text = inspect_load_api(selected["job"], run_command=fake_nm(selected, calls))
    assert not calls
    assert report["status"] == "INCOMPLETE"
    assert "library changed since descriptor probe" in text


def test_missing_symbol_tool_is_incomplete_and_preserves_probe(selected):
    probe = selected["out"] / "probe.json"
    before = probe.read_bytes()

    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    report, text = inspect_load_api(selected["job"], run_command=missing)
    assert report["status"] == "INCOMPLETE" and len(report["errors"]) == 1
    assert "FileNotFoundError" in text and "RestoreFromFile(const char* path)" in text
    assert probe.read_bytes() == before


@pytest.mark.parametrize("name", ["api-report.json", "api-report.txt"])
def test_refuses_symlink_report_destinations(selected, tmp_path, name):
    external = tmp_path / "unrelated.txt"
    external.write_text("preserve me")
    (selected["out"] / name).symlink_to(external)
    with pytest.raises(ValueError, match="symlink|regular|report"):
        inspect_load_api(selected["job"], run_command=fake_nm(selected, []))
    assert external.read_text() == "preserve me"

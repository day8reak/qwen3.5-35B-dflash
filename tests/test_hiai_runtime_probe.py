"""Offline comparison orchestration tests; mocks provide no device evidence."""
import copy
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from qwen35_moe_dflash import hiai_runtime_probe as probe
from qwen35_moe_dflash import op_probe
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.op_probe_cases import cases


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _no_build(*args, **kwargs):
    raise AssertionError("runtime comparison must never compile or build")


@pytest.fixture
def previous(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "127.0.0.2:8790")
    for key in ("CPRO_DDK_ROOT", "CANN_ROOT", "HIAI_BUILD_DIR", "HIAI_RUNTIME_LIB_DIR", "OHOS_NATIVE_SDK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(op_probe, "compiler_resources", _no_build)
    monkeypatch.setattr(op_probe, "native_build", _no_build)
    monkeypatch.setattr(op_probe, "_logged", _no_build)
    source = tmp_path / "old-probe"
    records = []
    for case in cases():
        if case["name"] not in probe.CASE_NAMES:
            continue
        record = op_probe.prepare_case(case, source)
        Path(record["model"]).write_bytes(b"fixture model " + case["name"].encode())
        record["compile"] = {"status": "COMPILED", "model": op_probe._payload(Path(record["model"]))}
        records.append(record)
    libraries = []
    for name in ("libhiai.so", "libhiai_ir.so", "libhiai_ir_build.so", "libc++_shared.so", *sorted(probe.EXCLUDED)):
        path = tmp_path / "original-libraries" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture ELF " + name.encode())
        libraries.append({"name": name, "path": str(path), "sha256": sha256_file(path), "origin": "fixture"})
    runner = source / "native/build/qwen35_hiai_op_probe"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(b"fixture historical binary")
    command = {"argv": ["cmake", "historical command; never execute"],
               "libraries": [{"path": row["path"], "sha256": row["sha256"]} for row in libraries[:3]],
               "sources": [{"path": str(tmp_path / "old-checkout/removed.cpp"), "sha256": "a" * 64}],
               "library_discovery": {"libraries": libraries}}
    _json(source / "native/command.json", command)
    result = {"status": "LINKED", "command_sha256": sha256_file(source / "native/command.json"),
              "binary_sha256": sha256_file(runner), "device_validation": "NOT_RUN"}
    _json(source / "native/result.json", result)
    manifest = {"artifact_kind": "qwen35-cpro-op-probe", "status": "COLLECTED", "cases": records,
                "native": {"build": command, "result": result}}
    _json(source / "manifest.json", manifest)
    (source / "report.txt").write_text("original report must remain untouched\n")
    (source / "results.tar.gz").write_bytes(b"original archive must remain untouched")
    dynamic = {row["name"]: {"needed": ["libc.so"], "soname": row["name"], "rpath": None, "runpath": None}
               for row in libraries}
    dynamic[runner.name] = {"needed": [row["name"] for row in libraries[:3]], "soname": None,
                            "rpath": None, "runpath": None}
    monkeypatch.setattr(probe, "dynamic_info", lambda path: copy.deepcopy(dynamic[Path(path).name]))
    return {"source": source, "manifest": manifest, "runner": runner, "libraries": libraries,
            "dynamic": dynamic, "output": tmp_path / "comparison", "records": records}


def _args(previous, *extra):
    return probe.parser().parse_args(["--probe-run", str(previous["source"]),
                                     "--output", str(previous["output"]), *extra])


def _transport(monkeypatch, previous, *, mutation=None, fail=False):
    calls = []

    def run(output, runner, libraries, records, **kwargs):
        call = {"output": output, "runner": runner, "libraries": libraries, "records": records, **kwargs}
        calls.append(call)
        assert Path(runner).read_bytes() == previous.get("transport_runner_bytes", previous["runner"].read_bytes())
        assert [row["name"] for row in records] == list(probe.CASE_NAMES)
        result = {"status": "TRANSPORT_COMPLETE", "cases": {}, "board_cleanup": "REMOVED",
                  "serial": "fixture-device", "hdc_server": "127.0.0.2:8790"}
        for record in records:
            folder = output / "board" / record["name"]
            folder.mkdir(parents=True)
            baseline = output.name == "ddk_baseline"
            failed_load = baseline and record["name"] != "add_rank5"
            native = ({"status": "FAILED", "stage": "load", "error": "fixture Load failure"}
                      if failed_load else {"status": "EXECUTED", "runtime_version": "fixture-version"})
            _json(folder / "stdout", native)
            (folder / "stderr").write_text("")
            (folder / "exitcode").write_text("1" if failed_load else "0")
            (folder / "pid").write_text("123")
            if not failed_load:
                for tensor in record["outputs"]:
                    shutil.copyfile(Path(record["directory"]) / tensor["expected"], folder / tensor["path"])
            result["cases"][record["name"]] = {"directory": str(folder), "files": []}
            if fail:
                result.update(status="FAILED", stage="case:" + record["name"], error="fixture transport interruption")
                _json(output / "board/failure.json", result)
                raise RuntimeError("fixture transport interruption")
        if mutation:
            mutation(calls)
        return result

    monkeypatch.setattr(probe, "run_board", run)
    return calls


def test_reuses_exact_artifacts_two_arms_and_no_rebuild(previous, monkeypatch):
    from qwen35_moe_dflash import hiai_probe_build
    monkeypatch.setattr(hiai_probe_build, "build_probe", _no_build)
    before = {str(path): sha256_file(path) for path in previous["source"].rglob("*") if path.is_file()}
    calls = _transport(monkeypatch, previous)
    assert probe.run(_args(previous, "--keep-board", "--timeout-seconds", "41")) == 0
    assert len(calls) == 2
    baseline, candidate = calls
    assert baseline["output"] != candidate["output"]
    assert {path.name for path in baseline["libraries"]} == {row["name"] for row in previous["libraries"]}
    assert {path.name for path in candidate["libraries"]} == {
        row["name"] for row in previous["libraries"] if row["name"] not in probe.EXCLUDED}
    assert baseline["library_search_paths"] == ()
    assert candidate["library_search_paths"] == ("/vendor/lib64/chipsetsdk",)
    assert all(call["keep_board"] and call["timeout"] == 41 for call in calls)
    assert baseline["runner"] == candidate["runner"] != previous["runner"]
    assert baseline["records"] == candidate["records"]
    assert before == {str(path): sha256_file(path) for path in previous["source"].rglob("*") if path.is_file()}
    report = json.loads((previous["output"] / "runtime-comparison.json").read_text())
    assert report["status"] == "COMPARISON_COMPLETE"
    assert report["refresh_runner"] is False and report["runner"]["mode"] == "REUSED_NATIVE"
    assert report["runtime_placement"] == report["backend_selection"] == "UNVERIFIED"
    assert [row["load"] for row in report["arms"][0]["cases"]] == ["FAILED", "FAILED", "PASSED"]
    assert all(row["numerical"] == "PASSED" for row in report["arms"][1]["cases"])
    assert all(row["placement"] == "UNVERIFIED" for arm in report["arms"] for row in arm["cases"])
    snapshot = json.loads((previous["output"] / "source-snapshot.json").read_text())
    assert snapshot["historical_native_sources"][0]["sha256"] == "a" * 64
    assert len(snapshot["dependencies"]) == len(previous["libraries"]) + 1
    with tarfile.open(previous["output"] / "results.tar.gz") as archive:
        assert set(archive.getnames()) == {"runtime-comparison.json", "runtime-comparison.txt", "source-snapshot.json",
                                         *[f"arms/{arm}/results.tar.gz" for arm in probe.ARMS]}
        packaged = json.load(archive.extractfile("runtime-comparison.json"))
        assert packaged["arms"][1]["library_search_paths"] == ["/vendor/lib64/chipsetsdk"]
        assert packaged["arms"][1]["excluded_library_names"] == sorted(probe.EXCLUDED)
        for arm in probe.ARMS:
            with tarfile.open(fileobj=io.BytesIO(archive.extractfile(f"arms/{arm}/results.tar.gz").read())) as nested:
                assert not any(name.endswith((".bin", ".omc", ".so")) or "_evidence" in Path(name).parts
                               for name in nested.getnames())


@pytest.mark.parametrize("changed", ["runner", "command", "library", "model", "io", "input", "reference", "graph"])
def test_tampering_refused_before_upload(previous, monkeypatch, changed):
    case = previous["records"][0]
    folder = Path(case["directory"])
    paths = {"runner": previous["runner"], "command": previous["source"] / "native/command.json",
             "library": Path(previous["libraries"][0]["path"]), "model": Path(case["model"]),
             "io": Path(case["io"]), "input": Path(case["inputs"][0]),
             "reference": folder / case["outputs"][0]["expected"], "graph": folder / "graph.cpp"}
    path = paths[changed]
    path.write_bytes(path.read_bytes() + b" ")
    calls = _transport(monkeypatch, previous)
    with pytest.raises(ValueError, match="SHA256|source size"):
        probe.run(_args(previous))
    assert not calls and not previous["output"].exists()


@pytest.mark.parametrize("fault", ["needed-name", "needed-absolute", "needed-soname", "colliding-soname", "duplicate-bytes"])
def test_candidate_dependency_or_alias_blocks_only_candidate(previous, monkeypatch, fault):
    excluded = sorted(probe.EXCLUDED)[0]
    other = "libhiai_ir.so"
    if fault == "needed-name":
        previous["dynamic"][previous["runner"].name]["needed"].append(excluded)
    elif fault == "needed-absolute":
        previous["dynamic"][other]["needed"].append("/original/private/path/" + excluded)
    elif fault == "needed-soname":
        previous["dynamic"][excluded]["soname"] = "private-hcl-alias.so"
        previous["dynamic"][other]["needed"].append("private-hcl-alias.so")
    elif fault == "colliding-soname":
        previous["dynamic"][other]["soname"] = excluded
    else:
        # Exercise the guard directly: changing source bytes would correctly
        # invalidate the historical build before the comparison was started.
        value = probe.validate_source(previous["source"])["dependencies"]
        omitted = next(row for row in value if row["name"] == excluded)
        next(row for row in value if row["name"] == other)["sha256"] = omitted["sha256"]
        assert "duplicates excluded provider bytes" in " ".join(probe.candidate_blockers(value))
        return
    calls = _transport(monkeypatch, previous)
    assert probe.run(_args(previous)) == 1
    assert len(calls) == 1 and calls[0]["output"].name == "ddk_baseline"
    comparison = json.loads((previous["output"] / "runtime-comparison.json").read_text())
    assert comparison["status"] == "CANDIDATE_BLOCKED"
    assert comparison["arms"][0]["status"] == "COMPLETE"
    assert comparison["arms"][1]["status"] == "BLOCKED"
    assert comparison["arms"][1]["blocked_reasons"]


def test_original_mutation_after_baseline_stops_candidate_and_saves_report(previous, monkeypatch):
    def mutate(calls):
        path = Path(previous["records"][0]["model"])
        path.write_bytes(path.read_bytes() + b"changed")
    calls = _transport(monkeypatch, previous, mutation=mutate)
    assert probe.run(_args(previous)) == 1
    assert len(calls) == 1
    comparison = json.loads((previous["output"] / "runtime-comparison.json").read_text())
    assert comparison["status"] == "INFRASTRUCTURE_FAILED"
    assert comparison["arms"][0]["status"] == "INFRASTRUCTURE_FAILED"
    assert "snapshotted source changed" in comparison["arms"][0]["error"]
    assert comparison["arms"][1]["status"] == "NOT_RUN"
    assert (previous["output"] / "results.tar.gz").is_file()


def test_partial_transport_failure_is_reported_without_retrying_or_rebuilding(previous, monkeypatch):
    calls = _transport(monkeypatch, previous, fail=True)
    assert probe.run(_args(previous)) == 1
    assert len(calls) == 1
    comparison = json.loads((previous["output"] / "runtime-comparison.json").read_text())
    assert comparison["status"] == "INFRASTRUCTURE_FAILED"
    assert comparison["arms"][0]["board_transport_status"] == "FAILED"
    assert comparison["arms"][0]["cases"][0]["load"] == "FAILED"
    assert comparison["arms"][0]["cases"][1]["load"] == "NOT_RUN"
    assert (previous["output"] / "arms/ddk_baseline/report.json").is_file()
    assert (previous["output"] / "results.tar.gz").is_file()


def test_existing_output_and_original_directory_are_not_modified(previous, monkeypatch):
    _transport(monkeypatch, previous)
    previous["output"].mkdir()
    with pytest.raises(FileExistsError):
        probe.run(_args(previous))
    previous["output"] = previous["source"] / "nested-comparison"
    with pytest.raises(ValueError, match="separate"):
        probe.run(_args(previous))
    assert not previous["output"].exists()


def test_missing_case_and_missing_required_hash_are_rejected(previous, monkeypatch):
    manifest = previous["manifest"]
    manifest["cases"] = [row for row in manifest["cases"] if row["name"] != "add_rank5"]
    _json(previous["source"] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="missing required compiled cases"):
        probe.validate_source(previous["source"])
    result = manifest["native"]["result"]
    result.pop("binary_sha256")
    _json(previous["source"] / "native/result.json", result)
    _json(previous["source"] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="SHA256"):
        probe.validate_source(previous["source"])


def test_no_library_parameters_are_exposed():
    options = {option for action in probe.parser()._actions for option in action.option_strings}
    assert options == {"-h", "--help", "--probe-run", "--output", "--keep-board", "--timeout-seconds", "--refresh-runner"}


@pytest.mark.parametrize("field,value", [("sha256", None), ("sha256", "invalid"), ("bytes", None), ("bytes", True)])
def test_case_metadata_cannot_disable_historical_integrity(previous, field, value):
    previous["manifest"]["cases"][0]["compile"]["model"][field] = value
    _json(previous["source"] / "manifest.json", previous["manifest"])
    with pytest.raises(ValueError, match="SHA256|byte size"):
        probe.validate_source(previous["source"])


def _refresh(monkeypatch, previous, *, needed=(), failure=False):
    from qwen35_moe_dflash import hiai_probe_build
    calls = []
    previous["transport_runner_bytes"] = b"fixture refreshed diagnostic binary"

    def build(root, validated, libraries, frozen, timeout):
        calls.append({"root": root, "libraries": libraries, "timeout": timeout})
        assert root == previous["output"]
        assert all(root / "reused/lib" == path.parent for path in libraries)
        if failure:
            raise RuntimeError("fixture native build failed")
        runner = root / "refreshed-native/build/qwen35_hiai_op_probe"
        runner.parent.mkdir(parents=True)
        runner.write_bytes(previous["transport_runner_bytes"])
        digest = sha256_file(runner)
        frozen.add(runner, expected=digest, role="refreshed-native-runner")
        metadata = {"mode": "REFRESHED_NATIVE", "path": str(runner), "sha256": digest,
                    "historical_binary": {"path": str(validated["runner"]), "sha256": sha256_file(validated["runner"])},
                    "dependencies": {"needed": list(needed), "soname": None, "rpath": None, "runpath": None}}
        for filename in ("command.json", "build-command.json", "result.json"):
            path = root / "refreshed-native" / filename
            _json(path, {"status": "fixture", "binary_sha256": digest})
            frozen.add(path, role="refresh-build-metadata")
        return runner, metadata

    monkeypatch.setattr(hiai_probe_build, "build_probe", build)
    return calls


def test_refresh_builds_once_and_both_arms_share_new_runner(previous, monkeypatch):
    old_files = {str(path): sha256_file(path) for path in previous["source"].rglob("*") if path.is_file()}
    build_calls = _refresh(monkeypatch, previous)
    transport = _transport(monkeypatch, previous)
    assert probe.run(_args(previous, "--refresh-runner")) == 0
    assert len(build_calls) == 1 and len(transport) == 2
    assert transport[0]["runner"] == transport[1]["runner"]
    assert transport[0]["runner"].parent.parent.name == "refreshed-native"
    assert transport[0]["records"] == transport[1]["records"]
    assert old_files == {str(path): sha256_file(path) for path in previous["source"].rglob("*") if path.is_file()}
    report = json.loads((previous["output"] / "runtime-comparison.json").read_text())
    assert report["refresh_runner"] and report["runner"]["mode"] == "REFRESHED_NATIVE"
    assert report["runner"]["historical_binary"]["sha256"] == sha256_file(previous["runner"])
    assert report["runner"]["sha256"] != report["runner"]["historical_binary"]["sha256"]
    snapshot = json.loads((previous["output"] / "source-snapshot.json").read_text())
    assert any(row["sha256"] == report["runner"]["historical_binary"]["sha256"] for row in snapshot["frozen_files"])
    assert any(row["sha256"] == report["runner"]["sha256"] for row in snapshot["frozen_files"])
    assert next(row for row in snapshot["dependencies"] if row["kind"] == "runner")["sha256"] == report["runner"]["sha256"]
    text = (previous["output"] / "runtime-comparison.txt").read_text()
    assert "builds only the diagnostic runner once for both arms" in text
    assert "no compilation or build" not in text
    with tarfile.open(previous["output"] / "results.tar.gz") as archive:
        assert {"refreshed-native/command.json", "refreshed-native/build-command.json", "refreshed-native/result.json"} <= set(archive.getnames())
        assert not any("/build/" in name for name in archive.getnames())


def test_refreshed_runner_dependencies_recompute_candidate_guard(previous, monkeypatch):
    _refresh(monkeypatch, previous, needed=["/absolute/path/libhiai_hcl_model_runtime.so"])
    transport = _transport(monkeypatch, previous)
    assert probe.run(_args(previous, "--refresh-runner")) == 1
    assert len(transport) == 1
    report = json.loads((previous["output"] / "runtime-comparison.json").read_text())
    assert report["status"] == "CANDIDATE_BLOCKED"
    assert "DT_NEEDED" in report["arms"][1]["blocked_reasons"][0]


def test_refresh_failure_records_preflight_and_never_uploads(previous, monkeypatch):
    _refresh(monkeypatch, previous, failure=True)
    transport = _transport(monkeypatch, previous)
    with pytest.raises(RuntimeError, match="fixture native build failed"):
        probe.run(_args(previous, "--refresh-runner"))
    assert not transport
    failure = json.loads((previous["output"] / "failure.json").read_text())
    assert failure["status"] == "PREFLIGHT_FAILED" and failure["board_started"] is False
    assert not (previous["output"] / "arms").exists()


def test_refreshed_binary_is_verified_between_arms(previous, monkeypatch):
    _refresh(monkeypatch, previous)
    def mutate(calls):
        path = calls[0]["runner"]
        path.write_bytes(path.read_bytes() + b"changed")
    transport = _transport(monkeypatch, previous, mutation=mutate)
    assert probe.run(_args(previous, "--refresh-runner")) == 1
    assert len(transport) == 1
    report = json.loads((previous["output"] / "runtime-comparison.json").read_text())
    assert report["status"] == "INFRASTRUCTURE_FAILED"
    assert "snapshotted source changed" in report["arms"][0]["error"]
    assert report["arms"][1]["status"] == "NOT_RUN"

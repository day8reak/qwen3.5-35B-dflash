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


def _transport(monkeypatch, previous, *, mutation=None, fail=False, failed_cases=()):
    calls = []

    def run(output, runner, libraries, records, **kwargs):
        call = {"output": output, "runner": runner, "libraries": libraries, "records": records, **kwargs}
        calls.append(call)
        assert Path(runner).read_bytes() == previous.get("transport_runner_bytes", previous["runner"].read_bytes())
        assert [row["name"] for row in records] == previous.get("transport_cases", list(probe.CASE_NAMES))
        result = {"status": "TRANSPORT_COMPLETE", "cases": {}, "board_cleanup": "REMOVED",
                  "serial": "fixture-device", "hdc_server": "127.0.0.2:8790"}
        for record in records:
            folder = output / "board" / record["name"]
            folder.mkdir(parents=True)
            baseline = output.name == "ddk_baseline"
            failed_load = (baseline and record["name"] != "add_rank5") or record["name"] in failed_cases
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
    assert options == {"-h", "--help", "--probe-run", "--output", "--keep-board", "--timeout-seconds", "--refresh-runner", "--replay-all", "--native-from"}


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


@pytest.fixture
def replay(previous):
    extra = next(case for case in cases() if case["name"] == "neg_rank4")
    record = op_probe.prepare_case(extra, previous["source"])
    Path(record["model"]).write_bytes(b"fixture additional compiled model")
    record["compile"] = {"status": "COMPILED", "model": op_probe._payload(Path(record["model"]))}
    previous["manifest"]["cases"].append(record)
    for row in previous["manifest"]["cases"]:
        counts = {"NPU": int(row["name"] != "add_rank5"), "CPU": int(row["name"] == "add_rank5"), "GPU": 0, "ISP": 0}
        row["compile"]["evidence"] = {"main_partition_events": [{"scope": "main", "line": 12, "counts": counts}]}
    previous["transport_cases"] = [row["name"] for row in previous["manifest"]["cases"]]
    previous["manifest"]["cases"] += [
        {"name": "select_rank4", "compile": {"status": "FAILED", "error": "fixture kernel rejection"}},
        {"name": "unprepared", "compile": {"status": "NOT_RUN"}},
    ]
    _json(previous["source"] / "manifest.json", previous["manifest"])
    return previous


def test_replay_all_refreshes_once_runs_candidate_only_and_preserves_skipped(replay, monkeypatch):
    original_cases = replay["manifest"]["cases"]
    original_cases.insert(1, original_cases.pop(-2))
    _json(replay["source"] / "manifest.json", replay["manifest"])
    before = {str(path): sha256_file(path) for path in replay["source"].rglob("*") if path.is_file()}
    build_calls = _refresh(monkeypatch, replay)
    transport = _transport(monkeypatch, replay, failed_cases={"neg_rank4"})
    assert probe.run(_args(replay, "--replay-all")) == 0
    assert len(build_calls) == len(transport) == 1
    assert transport[0]["output"].name == "candidate_without_ddk_hcl"
    assert transport[0]["library_search_paths"] == ("/vendor/lib64/chipsetsdk",)
    assert not ({path.name for path in transport[0]["libraries"]} & probe.EXCLUDED)
    assert before == {str(path): sha256_file(path) for path in replay["source"].rglob("*") if path.is_file()}
    report = json.loads((replay["output"] / "runtime-comparison.json").read_text())
    assert report["status"] == "REPLAY_COMPLETE" and report["replay_all"] and report["refresh_runner"]
    assert report["counts"] == {"cases": 6, "compiled": 4, "skipped": 2}
    assert report["selected_cases"] == replay["transport_cases"]
    assert report["skipped_cases"] == ["select_rank4", "unprepared"]
    assert len(report["arms"]) == 1
    assert [row["name"] for row in report["arms"][0]["cases"]] == [row["name"] for row in original_cases]
    rows = {row["name"]: row for row in report["arms"][0]["cases"]}
    assert rows["neg_rank4"]["load"] == "FAILED"
    assert rows["add_rank4"]["numerical"] == "PASSED"
    for name, status in (("select_rank4", "FAILED"), ("unprepared", "NOT_RUN")):
        assert rows[name]["compile"] == status and rows[name]["retry"] == "NOT_RETRIED"
        assert rows[name]["load"] == rows[name]["run"] == rows[name]["numerical"] == "NOT_RUN"
        assert not (replay["output"] / "reused/cases" / name).exists()
    assert rows["add_rank5"]["compiler_partition"] == {
        "scope": "main", "line": 12, "counts": {"NPU": 0, "CPU": 1, "GPU": 0, "ISP": 0}}
    assert all(row["placement"] == "UNVERIFIED" for row in rows.values())
    text = (replay["output"] / "runtime-comparison.txt").read_text()
    assert "candidate deployment only" in text and "4 compiled, 2 skipped" in text
    assert "NPU=0,CPU=1,GPU=0,ISP=0 [main]" in text
    assert "both arms" not in text and "ddk_baseline" not in text and "three previous" not in text
    assert "select_rank4 | FAILED" in text and "not retried" in text
    with tarfile.open(replay["output"] / "results.tar.gz") as archive:
        assert "arms/ddk_baseline/results.tar.gz" not in archive.getnames()
        candidate = archive.extractfile("arms/candidate_without_ddk_hcl/results.tar.gz").read()
        with tarfile.open(fileobj=io.BytesIO(candidate)) as nested:
            detailed = json.load(nested.extractfile("report.json"))
            assert len(detailed["cases"]) == 6
            skipped = next(row for row in detailed["cases"] if row["name"] == "select_rank4")
            assert skipped["compile"]["error"] == "fixture kernel rejection" and skipped["load"] == "NOT_RUN"


def test_replay_all_new_dependency_blocker_never_runs_baseline_or_hdc(replay, monkeypatch):
    builds = _refresh(monkeypatch, replay, needed=["libhiai_hcl_model_runtime.so"])
    transport = _transport(monkeypatch, replay)
    assert probe.run(_args(replay, "--replay-all")) == 1
    assert len(builds) == 1 and not transport
    report = json.loads((replay["output"] / "runtime-comparison.json").read_text())
    assert report["status"] == "CANDIDATE_BLOCKED" and len(report["arms"]) == 1
    assert report["arms"][0]["status"] == "BLOCKED"
    assert len(report["arms"][0]["cases"]) == 6
    assert all(row["load"] == row["run"] == row["numerical"] == "NOT_RUN" for row in report["arms"][0]["cases"])


@pytest.mark.parametrize("fault", ["unsafe-compiled", "unsafe-skipped", "plus-name", "duplicate", "not-object", "invalid-status", "no-compiled", "missing-model", "tampered-extra"])
def test_replay_all_selection_rejects_bad_original_cases_before_build_or_upload(replay, monkeypatch, fault):
    records = replay["manifest"]["cases"]
    if fault == "unsafe-compiled":
        records[0]["name"] = "../escape"
    elif fault == "unsafe-skipped":
        records[-1]["name"] = "../unprepared"
    elif fault == "plus-name":
        records[-1]["name"] = "unsupported+report-name"
    elif fault == "duplicate":
        records[-1]["name"] = records[0]["name"]
    elif fault == "not-object":
        records.append("bad record")
    elif fault == "invalid-status":
        records[-1]["compile"]["status"] = None
    elif fault == "no-compiled":
        for record in records:
            record["compile"] = {"status": "FAILED"}
    elif fault == "missing-model":
        records[3]["compile"].pop("model")
    else:
        path = Path(records[3]["model"])
        path.write_bytes(path.read_bytes() + b"corrupt")
    _json(replay["source"] / "manifest.json", replay["manifest"])
    builds = _refresh(monkeypatch, replay)
    transport = _transport(monkeypatch, replay)
    with pytest.raises((ValueError, KeyError)):
        probe.run(_args(replay, "--replay-all"))
    assert not builds and not transport and not replay["output"].exists()


def test_partition_summary_uses_historical_main_or_unknown_events_only():
    constant = {"scope": "constant_folding", "line": 1, "counts": {"NPU": 0, "CPU": 9, "GPU": 0, "ISP": 0}}
    unknown = {"scope": "unknown", "line": 2, "counts": {"NPU": 1, "CPU": 2, "GPU": 0, "ISP": 0}}
    main = {"scope": "main", "line": 3, "counts": {"NPU": 2, "CPU": 1, "GPU": 0, "ISP": 0}}
    assert probe._partition_summary({"evidence": {"partition_events": [constant]}}) is None
    assert probe._partition_summary({"evidence": {"partition_events": [constant, unknown]}}) == unknown
    assert probe._partition_summary({"evidence": {"partition_events": [constant, unknown], "main_partition_events": [main]}}) == main
    assert probe._partition_summary({"evidence": {"partition_events": [main, unknown]}}) == main
    assert probe._partition_summary({"evidence": {"main_partition_events": [constant]}}) is None
    assert probe._partition_summary({"evidence": {"main_partition_events": [{"scope": "main", "counts": None}]}}) is None


def test_replay_all_has_no_dependency_on_the_original_three_case_names(replay, monkeypatch):
    replay["manifest"]["cases"] = [row for row in replay["manifest"]["cases"] if row["name"] not in probe.CASE_NAMES]
    replay["transport_cases"] = ["neg_rank4"]
    _json(replay["source"] / "manifest.json", replay["manifest"])
    builds = _refresh(monkeypatch, replay)
    transport = _transport(monkeypatch, replay)
    assert probe.run(_args(replay, "--replay-all")) == 0
    assert len(builds) == len(transport) == 1
    report = json.loads((replay["output"] / "runtime-comparison.json").read_text())
    assert report["selected_cases"] == ["neg_rank4"]
    assert report["counts"] == {"cases": 3, "compiled": 1, "skipped": 2}


def test_default_keeps_three_case_two_arm_behavior_with_other_cases_present(replay, monkeypatch):
    from qwen35_moe_dflash import hiai_probe_build
    monkeypatch.setattr(hiai_probe_build, "build_probe", _no_build)
    replay.pop("transport_cases")
    transport = _transport(monkeypatch, replay)
    assert probe.run(_args(replay)) == 0 and len(transport) == 2
    report = json.loads((replay["output"] / "runtime-comparison.json").read_text())
    assert not report["replay_all"] and not report["refresh_runner"]
    assert report["selected_cases"] == list(probe.CASE_NAMES)
    assert report["counts"] == {"cases": 3, "compiled": 3, "skipped": 0}


@pytest.fixture
def compile_only(previous):
    donor = previous["source"]
    identity = {"ddk_root": str(Path(previous["libraries"][0]["path"]).parent),
                "ddk_id": "fixture-ddk", "omg_sha256": "b" * 64}
    previous["manifest"]["tools"] = copy.deepcopy(identity)
    _json(donor / "manifest.json", previous["manifest"])
    previous["donor_manifest"] = previous["manifest"]
    previous["donor"] = donor
    source = donor.parent / "new-compile-only"
    records = []
    for case in cases():
        if case["name"] not in {"reduce_mean", "reduce_sum"}:
            continue
        record = op_probe.prepare_case(case, source)
        Path(record["model"]).write_bytes(b"new existing compiled bytes " + case["name"].encode())
        record["compile"] = {"status": "COMPILED", "model": op_probe._payload(Path(record["model"]))}
        records.append(record)
    manifest = {"artifact_kind": "qwen35-cpro-op-probe", "status": "COMPILE_COMPLETE",
                "board_requested": False, "tools": identity, "cases": records + [
                    {"name": "uncompiled_case", "compile": {"status": "FAILED", "error": "original compile failure"}}]}
    _json(source / "manifest.json", manifest)
    previous.update(source=source, manifest=manifest, records=records,
                    transport_cases=[record["name"] for record in records])
    return previous


def _donor_args(state, *extra):
    return _args(state, "--native-from", str(state["donor"]), "--refresh-runner", "--replay-all", *extra)


def test_compile_only_replay_uses_explicit_native_donor_and_never_recompiles_models(compile_only, monkeypatch):
    before = {str(path): sha256_file(path) for folder in (compile_only["source"], compile_only["donor"])
              for path in folder.rglob("*") if path.is_file()}
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    assert probe.run(_donor_args(compile_only)) == 0
    assert len(builds) == len(transport) == 1
    call, = transport
    assert call["output"].name == "candidate_without_ddk_hcl"
    assert call["library_search_paths"] == ("/vendor/lib64/chipsetsdk",)
    assert not {path.name for path in call["libraries"]} & probe.EXCLUDED
    for original, replayed in zip(compile_only["records"], call["records"]):
        assert sha256_file(Path(replayed["model"])) == original["compile"]["model"]["sha256"]
        for tensor in original["input_tensors"]:
            assert sha256_file(Path(replayed["directory"]) / tensor["path"]) == tensor["sha256"]
    assert before == {str(path): sha256_file(Path(path)) for path in before}
    assert not (compile_only["source"] / "native").exists()
    root = compile_only["output"]
    report = json.loads((root / "runtime-comparison.json").read_text())
    assert report["status"] == "REPLAY_COMPLETE"
    assert report["native_from"] == str(compile_only["donor"])
    assert report["source_manifest_sha256"] == sha256_file(compile_only["source"] / "manifest.json")
    assert report["native_source_manifest_sha256"] == sha256_file(compile_only["donor"] / "manifest.json")
    assert report["matched_ddk_identity"] == compile_only["manifest"]["tools"]
    assert report["counts"] == {"cases": 3, "compiled": 2, "skipped": 1}
    assert report["arms"][0]["cases"][-1]["run"] == "NOT_RUN"
    assert report["runtime_placement"] == "UNVERIFIED"
    snapshot = json.loads((root / "source-snapshot.json").read_text())
    assert snapshot["native_source_origin"] == "explicit-donor"
    assert snapshot["native_source_probe_run"] == str(compile_only["donor"])
    assert {str(folder / "manifest.json") for folder in (compile_only["source"], compile_only["donor"])} <= {
        item["path"] for item in snapshot["files"]}
    assert json.loads((root / "source-records/manifest.json").read_text()) == compile_only["manifest"]
    assert json.loads((root / "native-source-records/manifest.json").read_text()) == compile_only["donor_manifest"]
    with tarfile.open(root / "results.tar.gz") as archive:
        names = set(archive.getnames())
        assert {"source-records/manifest.json", "native-source-records/manifest.json",
                "native-source-records/native/command.json", "native-source-records/native/result.json"} <= names
        assert not any(name.endswith((".so", ".omc", ".bin")) or "/build/" in name for name in names)
    assert "Native build/library donor:" in (root / "runtime-comparison.txt").read_text()


@pytest.mark.parametrize("flags", [[], ["--replay-all"], ["--refresh-runner"]])
def test_native_donor_requires_both_explicit_flags(compile_only, monkeypatch, flags):
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError, match="requires both"):
        probe.run(_args(compile_only, "--native-from", str(compile_only["donor"]), *flags))
    assert not builds and not transport and not compile_only["output"].exists()


def test_compile_only_does_not_implicitly_rediscover_native(compile_only, monkeypatch):
    monkeypatch.setenv("HIAI_BUILD_DIR", str(compile_only["donor"] / "native"))
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError, match="required source file is missing"):
        probe.run(_args(compile_only, "--refresh-runner", "--replay-all"))
    assert not builds and not transport


@pytest.mark.parametrize("partial", ["manifest", "empty-directory", "command-only", "broken-symlink"])
def test_donor_cannot_replace_incomplete_original_native(compile_only, monkeypatch, partial):
    source = compile_only["source"]
    if partial == "manifest":
        compile_only["manifest"]["native"] = None
        _json(source / "manifest.json", compile_only["manifest"])
    elif partial == "empty-directory":
        (source / "native").mkdir()
    elif partial == "command-only":
        _json(source / "native/command.json", {})
    else:
        (source / "native").symlink_to(source / "missing-native")
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError, match="cannot replace"):
        probe.run(_donor_args(compile_only))
    assert not builds and not transport


def test_native_donor_cannot_override_a_complete_original_run(previous, monkeypatch):
    builds = _refresh(monkeypatch, previous)
    transport = _transport(monkeypatch, previous)
    with pytest.raises(ValueError, match="cannot replace"):
        probe.run(_args(previous, "--native-from", str(previous["source"]), "--refresh-runner", "--replay-all"))
    assert not builds and not transport


@pytest.mark.parametrize("field,value", [("status", "COLLECTED"), ("status", "PREPARED"),
                                         ("board_requested", True), ("board_requested", None)])
def test_native_donor_only_accepts_declared_compile_only_batches(compile_only, monkeypatch, field, value):
    compile_only["manifest"][field] = value
    _json(compile_only["source"] / "manifest.json", compile_only["manifest"])
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError, match="COMPILE_COMPLETE compile-only"):
        probe.run(_donor_args(compile_only))
    assert not builds and not transport


@pytest.mark.parametrize("field,value", [("ddk_root", "/other/ddk"), ("ddk_id", "other-version"),
                                         ("omg_sha256", "c" * 64), ("ddk_id", None),
                                         ("ddk_root", "relative/ddk"), ("omg_sha256", None)])
def test_native_donor_requires_matching_explicit_ddk_identity(compile_only, monkeypatch, field, value):
    compile_only["manifest"]["tools"][field] = value
    _json(compile_only["source"] / "manifest.json", compile_only["manifest"])
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError):
        probe.run(_donor_args(compile_only))
    assert not builds and not transport


def test_matching_manifest_ddk_cannot_mask_linking_another_ddk(compile_only, monkeypatch):
    for key in ("manifest", "donor_manifest"):
        compile_only[key]["tools"]["ddk_root"] = "/unrelated/declared-ddk"
    _json(compile_only["source"] / "manifest.json", compile_only["manifest"])
    _json(compile_only["donor"] / "manifest.json", compile_only["donor_manifest"])
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError, match="outside the matched compiler DDK"):
        probe.run(_donor_args(compile_only))
    assert not builds and not transport


@pytest.mark.parametrize("changed", ["donor-runner", "donor-command", "donor-result", "library", "omc", "input", "reference", "io", "recipe"])
def test_native_donor_replay_keeps_both_sources_payload_integrity(compile_only, monkeypatch, changed):
    record = compile_only["records"][0]
    paths = {"donor-runner": compile_only["runner"], "donor-command": compile_only["donor"] / "native/command.json",
             "donor-result": compile_only["donor"] / "native/result.json", "library": Path(compile_only["libraries"][0]["path"]),
             "omc": Path(record["model"]), "input": Path(record["inputs"][0]),
             "reference": Path(record["directory"]) / record["outputs"][0]["expected"], "io": Path(record["io"]),
             "recipe": Path(record["directory"]) / record["construction_files"][0]["path"]}
    path = paths[changed]
    path.write_bytes(path.read_bytes() + b"tampered")
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError):
        probe.run(_donor_args(compile_only))
    assert not builds and not transport and not compile_only["output"].exists()


@pytest.mark.parametrize("which", ["source", "donor"])
def test_native_donor_replay_output_cannot_be_inside_either_source(compile_only, monkeypatch, which):
    compile_only["output"] = compile_only[which] / "nested-output"
    builds = _refresh(monkeypatch, compile_only)
    transport = _transport(monkeypatch, compile_only)
    with pytest.raises(ValueError, match="separate from"):
        probe.run(_donor_args(compile_only))
    assert not builds and not transport and not compile_only["output"].exists()


def test_donor_refresh_rechecks_new_binary_dependencies_before_hdc(compile_only, monkeypatch):
    builds = _refresh(monkeypatch, compile_only, needed=["libhiai_hcl_model_runtime.so"])
    transport = _transport(monkeypatch, compile_only)
    assert probe.run(_donor_args(compile_only)) == 1
    assert len(builds) == 1 and not transport
    report = json.loads((compile_only["output"] / "runtime-comparison.json").read_text())
    assert report["status"] == "CANDIDATE_BLOCKED"


def test_native_donor_manifest_is_reverified_after_transport(compile_only, monkeypatch):
    _refresh(monkeypatch, compile_only)
    def mutate(calls):
        path = compile_only["donor"] / "manifest.json"
        path.write_text(path.read_text() + " ")
    transport = _transport(monkeypatch, compile_only, mutation=mutate)
    assert probe.run(_donor_args(compile_only)) == 1
    assert len(transport) == 1
    report = json.loads((compile_only["output"] / "runtime-comparison.json").read_text())
    assert report["status"] == "INFRASTRUCTURE_FAILED"
    assert "snapshotted source changed" in report["arms"][0]["error"]

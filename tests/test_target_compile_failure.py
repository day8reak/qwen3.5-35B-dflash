"""Failed compile inspection reads metadata/log/AIR, never weight payloads."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import target_compile_failure as inspect
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile


def _stat(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def case(tmp_path):
    job = tmp_path / "target-pytorch-compile"
    root = job / "compilation"
    work = root / "compile-work/generated"
    work.mkdir(parents=True)
    air = work / "export.air"
    air.write_bytes(b"OPAQUE AIR; only its small byte identity is tested here")
    weight = work / "glm_export_air_cpro_40block_64_quant/weight0"
    weight.parent.mkdir()
    weight.write_bytes(b"snapshot weight; do not reopen")
    partial = work / "model/SubGraph_0.weight"
    partial.parent.mkdir()
    partial.write_bytes(b"partial compiler weights; do not reopen")
    graph = work / "dump_graph/failed_graph.irpb"
    graph.parent.mkdir()
    graph.write_bytes(b"GRAPH; discovery must stat only")
    retained_graph = root / "diagnostics/failed_graph.irpb"
    retained_graph.parent.mkdir()
    retained_graph.write_bytes(graph.read_bytes())
    omg = tmp_path / "ddk/tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"\x7fELF mock compiler; do not reopen")
    snapshots = []
    for path, role in ((air, "air"), (weight, "neighbor_data_candidate")):
        snapshots.append({"path": str(path.relative_to(root)), "source": "/original/not-followed/" + path.name,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "snapshot_stat": _stat(path), "source_stat": [0, 0, 0, 0, 0], "snapshot_method": "COPY", "role": role})
    profile = get_existing_air_profile("cpro-40block-64-1")
    options = {"input_shape": profile["input_shape"], "dynamic_dims": profile["dynamic_dims"],
               "limit_partition_size": "false", "save_weights_as_external_data": "true"}
    report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "FAILED", "stage": "omg",
        "model_compilation": "FAILED", "returncode": 1, "error": "OMG exited with status 1",
        "requested_options": options, "input_snapshots": snapshots,
        "generated_artifacts": [{"path": str(partial.relative_to(root)), "bytes": partial.stat().st_size,
            "sha256": "a" * 64, "stat": _stat(partial), "role": "external-data-candidate"}],
        "compiler_work_artifacts": [{"path": str(graph.relative_to(root)), "bytes": graph.stat().st_size,
            "sha256": "b" * 64, "stat": _stat(graph), "role": "compiler-diagnostic"}],
        "source_preservation": {"status": "UNCHANGED_BY_STAT", "changed": []},
        "input_layout": {"mode": "air-directory", "compiler_cwd": "compile-work/generated"},
        "ddk_root": str(tmp_path / "ddk"), "ddk_id": "SELECTED_TEST_DDK", "omg_path": str(omg),
        "omg_sha256": hashlib.sha256(omg.read_bytes()).hexdigest(), "platform": "kirinx90"}
    _write(root / "report.json", report)
    outer = {"artifact_kind": "qwen35-original-target-job", "status": "FAILED", "model_compilation": "FAILED",
             "compilation_report_sha256": hashlib.sha256((root / "report.json").read_bytes()).hexdigest()}
    _write(job / "result.json", outer)
    _write(job / "compile-profile.json", profile)
    _write(root / "input-inventory.json", {"files": snapshots})
    _write(root / "compile-command.json", {"argv": [str(omg), "--framework=1", "--model=" + str(air),
        "--output=model", "--dynamic_dims=" + profile["dynamic_dims"]], "cwd": str(work)})
    _write(root / "omg-environment.json", {"ddk_root": str(tmp_path / "ddk"), "SOC_VERSION": "kirinx90"})
    _write(root / "tbe-compatibility-result.json", {"audit": {"status": "APPLIED"}, "compiler_invocation": "ATTEMPTED"})
    _write(root / "diagnostics/manifest.json", {"files": [{"name": retained_graph.name,
        "source": "dump_graph/" + graph.name, "role": "graph-dump", "status": "RETAINED",
        "bytes": retained_graph.stat().st_size, "sha256": hashlib.sha256(retained_graph.read_bytes()).hexdigest()}]})
    log = root / "compile.log"
    log.write_text("initial setup\nbefore branch\nbranches1_Reshape_849 dispatch\n"
                   "E/AI_FMK: Reshape_849 invalid shape input[1,1,2048] target[64,2048]\n"
                   "shape check failed\nOMG failed\n")
    return {"job": job, "root": root, "work": work, "air": air, "weight": weight, "partial": partial,
            "graph": graph, "retained_graph": retained_graph, "omg": omg, "log": log, "report": report}


def test_report_binds_small_evidence_and_keeps_weight_hash_claim_honest(case, monkeypatch):
    before = {path: _stat(path) for path in case["job"].rglob("*") if path.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("inspection cannot run tools"))
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["status"] == "INSPECTED", result
    assert result["failure"]["returncode"] == 1
    assert result["failure"]["compilation_report_binding"] == "MATCHED"
    assert result["snapshots"]["count"] == 2 and result["snapshots"]["snapshot_methods"] == {"COPY": 2}
    assert result["snapshots"]["hash_validation"] == "RECORDED_HASH_WITH_CURRENT_STAT"
    assert result["snapshots"]["current_stat_counts"] == {"MATCHED": 2}
    assert result["snapshots"]["air"]["hash_validation"] == "FRESH_SHA256_MATCHED"
    assert result["snapshot_air_path"] == str(case["air"])
    assert str(case["retained_graph"]) in result["compiler_graph_paths"] and len(result["compiler_graph_paths"]) <= 4
    assert all(row["content_read"] == "NOT_PERFORMED" for row in result["compiler_graph_candidates"])
    assert result["options"]["dynamic_dims"] == "64,64,64,64,64;1,1,1,1,1"
    assert result["profile"]["input_shape_matches_options"] is True
    assert result["ddk"]["omg_hash_validation"] == "RECORDED_NOT_REHASHED"
    assert result["ddk"]["tbe_audit"] == {"status": "APPLIED"}
    assert result["partial_artifacts"]["recorded_bytes"] == case["partial"].stat().st_size
    assert result["free_bytes"] > 0
    assert "Reshape_849" in result["log_evidence"]["selected_text"]
    assert "L4:" in result["log_evidence"]["selected_text"]
    assert result["log_evidence"]["sha256"] == hashlib.sha256(case["log"].read_bytes()).hexdigest()
    assert result["compiler_invocation"] == result["model_execution"] == "NOT_RUN"
    assert {path: _stat(path) for path in before} == before


def test_no_weights_partial_outputs_graphs_or_compiler_binary_are_opened(case, monkeypatch):
    original = Path.open
    forbidden = {case[key] for key in ("weight", "partial", "graph", "retained_graph", "omg")}
    def guarded(path, *args, **kwargs):
        assert path not in forbidden, "this file must be inspected by stat only: " + str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    original_os_open = inspect.os.open
    def os_guard(path, *args, **kwargs):
        assert Path(path) not in forbidden
        return original_os_open(path, *args, **kwargs)
    monkeypatch.setattr(inspect.os, "open", os_guard)
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["status"] == "INSPECTED" and result["weight_payload_access"] == "NONE"


def test_changed_snapshot_stats_are_reported_without_fresh_payload_hash(case):
    case["weight"].write_bytes(b"changed snapshot")
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["status"] == "INSPECTED"
    assert result["snapshots"]["current_stat_counts"]["DIFFERENT_OR_UNRECORDED"] == 1
    assert result["snapshots"]["hash_validation"] == "RECORDED_HASH_WITH_CURRENT_STAT"
    assert "glm_export" in result["snapshots"]["mismatches"][0]["path"]


def test_changed_small_air_gets_a_fresh_explicit_hash_difference(case):
    case["air"].write_bytes(b"different small AIR")
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["snapshots"]["air"]["hash_validation"] == "FRESH_SHA256_DIFFERENT"


@pytest.mark.parametrize("fault", ["outside_snapshot", "symlink_snapshot", "symlink_job", "unknown_kind", "success", "duplicate_json"])
def test_unsafe_or_nonfailed_evidence_is_not_promoted(case, fault):
    job = case["job"]
    if fault == "outside_snapshot":
        case["report"]["input_snapshots"][1]["path"] = "../outside/weight"
        _write(case["root"] / "report.json", case["report"])
    elif fault == "symlink_snapshot":
        case["weight"].unlink(); case["weight"].symlink_to(case["partial"])
    elif fault == "symlink_job":
        job = case["job"].with_name("alias-job"); job.symlink_to(case["job"], target_is_directory=True)
    elif fault == "unknown_kind": _write(job / "result.json", {"artifact_kind": "another-task", "status": "FAILED"})
    elif fault == "success": _write(job / "result.json", {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED"})
    elif fault == "duplicate_json": (job / "result.json").write_text('{"status":"FAILED","status":"COMPILED"}')
    result = inspect.inspect_target_compile_failure(job)
    if fault in {"outside_snapshot", "symlink_snapshot"}:
        assert result["snapshots"]["current_stat_counts"]["UNAVAILABLE"] == 1
    else:
        assert result["status"] == "BLOCKED"


def test_optional_metadata_absence_is_explicit_without_preventing_failure_context(case):
    (case["root"] / "omg-environment.json").unlink()
    (case["root"] / "tbe-compatibility-result.json").unlink()
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["status"] == "INSPECTED"
    assert result["metadata_evidence"]["omg_environment"]["status"] == "UNAVAILABLE"
    assert result["ddk"]["omg_hash_validation"] == "RECORDED_NOT_REHASHED"


def test_graph_discovery_is_fixed_directory_bounded_and_never_recursive(case, monkeypatch):
    unrelated = case["job"].parent / "huge-unrelated-tree"
    unrelated.mkdir()
    (unrelated / "fake.irpb").write_bytes(b"not a compiler graph")
    for index in range(8):
        (case["graph"].parent / f"extra_{index}.irpb").write_bytes(b"graph")
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: pytest.fail("no recursive tree scan is authorized"))
    result = inspect.inspect_target_compile_failure(case["job"])
    assert len(result["compiler_graph_paths"]) == 4
    assert str(case["retained_graph"]) == result["compiler_graph_paths"][0]
    assert not any("unrelated" in path for path in result["compiler_graph_paths"])


def test_log_selection_retains_line_numbers_and_never_exceeds_twelve_kib(case):
    lines = []
    for index in range(50):
        lines.extend([*("filler " + "x" * 3000 for _ in range(7)),
                      f"E/AI_FMK Reshape_{index} mismatch branches1_test"])
    case["log"].write_text("\n".join(lines) + "\n")
    result = inspect.inspect_target_compile_failure(case["job"])
    log = result["log_evidence"]
    assert 0 < len(log["selected_text"].encode()) <= 12 << 10
    assert log["omitted_selected_lines"] > 0
    assert f"L{len(lines)}:" in log["selected_text"]
    assert log["match_counts"]["reshape_error_candidates"] == 50
    assert "root cause are not inferred" in log["scope"]


def test_normal_e_level_shape_debug_does_not_hide_the_last_fatal_reshape(case):
    lines = [f"E/AI_FMK [lsy] Reshape_{index} input shape=[1,64,2048], output shape=[64,2048]"
             for index in range(1000)]
    lines.extend(["E/AI_FMK branches1/Reshape_849 [lsy] reshape input shape=[1,1,2048]",
        "E/AI_FMK ReshapeComputeOutputDim returns false: input[1,1,2048], target[64,2048]",
        "E/AI_FMK Reshape_849 compute output shape returnFAIL",
        "E/AI_FMK branches1 Optimize graph failed", "E/AI_FMK OMG process exit=1"])
    case["log"].write_text("\n".join(lines) + "\n")
    result = inspect.inspect_target_compile_failure(case["job"])
    log = result["log_evidence"]
    assert log["match_counts"]["reshape_error_candidates"] == 2
    assert log["match_counts"]["branch_error_candidates"] == 1
    assert "L1002: E/AI_FMK ReshapeComputeOutputDim returns false" in log["selected_text"]
    assert "Reshape_849 compute output shape returnFAIL" in log["selected_text"]
    assert "branches1 Optimize graph failed" in log["selected_text"]
    assert "Reshape_0 input shape" not in log["selected_text"]
    assert len(log["selected_text"].encode()) <= 12 << 10


def test_log_read_cap_never_labels_a_prefix_digest_as_full_sha(case, monkeypatch):
    case["log"].write_bytes(b"E/ Reshape mismatch branches1\n" + b"x" * 400)
    monkeypatch.setattr(inspect, "MAX_LOG_READ_BYTES", 100)
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["log_evidence"]["scanned_bytes"] == 100
    assert result["log_evidence"]["sha256"] is None
    assert result["log_evidence"]["hash_validation"] == "PARTIAL_OR_CHANGED_NOT_VERIFIED"


def test_missing_duplicate_air_rows_do_not_guess_a_source_graph(case):
    case["report"]["input_snapshots"][1]["role"] = "air"
    _write(case["root"] / "report.json", case["report"])
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["snapshot_air_path"] is None
    assert result["snapshots"]["air"]["recorded_air_rows"] == 2


def test_an_air_role_label_cannot_make_the_inspector_open_a_weight_payload(case, monkeypatch):
    case["report"]["input_snapshots"][0]["role"] = "neighbor_data_candidate"
    case["report"]["input_snapshots"][1]["role"] = "air"
    _write(case["root"] / "report.json", case["report"])
    original = inspect._read_small
    def guarded(path, maximum):
        assert Path(path) != case["weight"]
        return original(path, maximum)
    monkeypatch.setattr(inspect, "_read_small", guarded)
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["status"] == "INSPECTED" and result["snapshot_air_path"] is None
    assert "not an explicit .air file" in result["snapshots"]["air"]["reason"]


def test_large_graph_candidate_is_statted_but_not_sent_to_a_small_graph_parser(case):
    large = case["graph"].parent / "large.irpb"
    with large.open("wb") as stream:
        stream.truncate(inspect.MAX_AIR_BYTES + 1)
    result = inspect.inspect_target_compile_failure(case["job"])
    assert str(large) not in result["compiler_graph_paths"]
    assert any("64 MiB metadata parser budget" in error for error in result["compiler_graph_discovery_errors"])


def _regular_dump_graph(case):
    case["graph"].unlink()
    case["graph"].parent.rmdir()
    dump = case["work"] / "dump_graph"
    payload = b"Compiler graph candidate; no schema parsing in this inspector"
    dump.write_bytes(payload)
    case["retained_graph"].write_bytes(payload)
    case["report"]["compiler_work_artifacts"] = []
    row = {"path": "compile-work/generated/dump_graph", "bytes": len(payload),
           "sha256": hashlib.sha256(payload).hexdigest(), "stat": _stat(dump), "role": "external_data_candidate"}
    case["report"]["generated_artifacts"].append(row)
    _write(case["root"] / "report.json", case["report"])
    _write(case["root"] / "diagnostics/manifest.json", {"files": [{"name": case["retained_graph"].name,
        "role": "graph-dump", "status": "RETAINED", "bytes": len(payload), "sha256": row["sha256"]}]})
    outer = json.loads((case["job"] / "result.json").read_text())
    outer["compilation_report_sha256"] = hashlib.sha256((case["root"] / "report.json").read_bytes()).hexdigest()
    _write(case["job"] / "result.json", outer)
    return dump, row


def test_report_bound_regular_dump_graph_is_preferred_without_opening_or_duplicate_copy_parse(case, monkeypatch):
    dump, row = _regular_dump_graph(case)
    decoy = case["work"] / "model/ordinary_extensionless_weight"
    decoy.write_bytes(b"not a graph")
    case["report"]["generated_artifacts"].append({"path": str(decoy.relative_to(case["root"])),
        "bytes": decoy.stat().st_size, "sha256": "c" * 64, "stat": _stat(decoy), "role": "external_data_candidate"})
    _write(case["root"] / "report.json", case["report"])
    original_open, original_os_open = Path.open, inspect.os.open
    forbidden = {dump, case["retained_graph"], decoy}
    def guarded(path, *args, **kwargs):
        assert path not in forbidden, "graph discovery is metadata-only"
        return original_open(path, *args, **kwargs)
    def os_guard(path, *args, **kwargs):
        assert Path(path) not in forbidden, "graph discovery is metadata-only"
        return original_os_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(inspect.os, "open", os_guard)
    result = inspect.inspect_target_compile_failure(case["job"])
    assert result["status"] == "INSPECTED"
    assert result["compiler_graph_paths"] == [str(dump)]
    candidate = result["compiler_graph_candidates"][0]
    assert candidate["kind"] == "REPORT_BOUND_DUMP_GRAPH_FILE"
    assert candidate["recorded_sha256"] == row["sha256"]
    assert candidate["content_read"] == "NOT_PERFORMED"
    assert candidate["same_recorded_identity_aliases"] == [str(case["retained_graph"])]
    assert "not read or freshly verified" in candidate["deduplication_scope"]


@pytest.mark.parametrize("fault", ["missing_sha", "size", "stat", "missing_stat", "unrecorded"])
def test_regular_dump_graph_requires_the_exact_report_binding(case, fault):
    dump, row = _regular_dump_graph(case)
    if fault == "missing_sha": row.pop("sha256")
    elif fault == "size": row["bytes"] += 1
    elif fault == "stat": row["stat"][1] += 1
    elif fault == "missing_stat": row.pop("stat")
    else: case["report"]["generated_artifacts"].remove(row)
    _write(case["root"] / "report.json", case["report"])
    result = inspect.inspect_target_compile_failure(case["job"])
    assert str(dump) not in result["compiler_graph_paths"]
    # The ordinary .irpb retained copy remains a candidate. A malformed or
    # absent special-file receipt must never broaden extensionless scanning.
    assert str(case["retained_graph"]) in result["compiler_graph_paths"]

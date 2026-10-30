"""Retained output/log fixtures; no model, device or weight payloads exist."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import struct

import pytest

from qwen35_moe_dflash import target_prefill_output_inventory as inventory


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def refresh(case):
    save(case["board"] / "plan.json", case["plan"])
    case["report"]["plan_sha256"] = sha(case["board"] / "plan.json")
    save(case["board"] / "runtime-prefill-proof.json", case["plan"])
    save(case["execution"] / "plan.json", case["plan"])
    save(case["board"] / "result.json", case["report"])
    outer = dict(case["report"], artifact_kind="qwen35-target-prefill-smoke-job", mode="APPLY",
        board_result=str(case["board"] / "result.json"), plan=str(case["execution"] / "plan.json"))
    save(case["execution"] / "result.json", outer)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    execution = tmp_path / "prefill-execution"
    board = execution / "board"
    (board / "outputs").mkdir(parents=True)
    remote = "/data/local/tmp/fixture-target/run-fixture"
    descriptors = [{"index": 0, "shape": [1, 1, 1, 248320], "dtype": "FLOAT32", "bytes": 993280,
                    "tool_output_type": "FP32", "nd_view_policy": "EDGE_SINGLETON_ND_LINEAR_VIEW_ONLY"}]
    shapes = []
    for _ in range(10):
        shapes += [[1, 1, 8192, 4], [1, 32, 128, 128]] * 3 + [[64, 32, 32, 16]] * 2
    shapes += [[1, 1, 64, 16384]]
    for shape in shapes:
        size = 2
        for dim in shape:
            size *= dim
        descriptors.append({"index": len(descriptors), "shape": shape, "dtype": "FLOAT16", "bytes": size,
            "tool_output_type": "FP16", "nd_view_policy": "EDGE_SINGLETON_ND_LINEAR_VIEW_ONLY"})
    model_path = "/data/local/tmp/fixture-target/payload/model/model.omc"
    plan = {"artifact_kind": "qwen35-original-prefill-smoke-plan", "status": "PREFILL_SMOKE_PLANNED",
        "profile": "cpro-40block-prefill-64", "gear": 64, "inputs": [{}] * 89, "outputs": descriptors,
        "runtime_probe": str(tmp_path / "unopened-native-probe"), "input_dir": str(tmp_path / "original-inputs"),
        "execution_scope": "SINGLE_PREFILL_ONLY_IDENTICAL_STATE_GROUPS", "model_sha256": "a" * 64,
        "tool_binding": {"path": "/data/local/tmp/model_run_tool_internal"},
        "feature_runtime_output_index": 81, "logits_runtime_output_index": 0}
    report = {"artifact_kind": "qwen35-original-target-tool-execution", "status": "EXECUTED_UNVALIDATED",
        "device_execution": "EXECUTED_OUTPUTS_RETRIEVED", "remote_exitcode": 0, "tool_process_id": 8985,
        "execution_scope": "SINGLE_PREFILL_ONLY_IDENTICAL_STATE_GROUPS", "retrieval_errors": [],
        "lock_state": "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT",
        "source_preservation": "UNCHANGED_PREFILL_PLAN_METADATA_AND_ORIGINAL_INPUT_SHA256",
        "gear": 64, "runtime_prefill_probe": plan["runtime_probe"], "board_attempt": remote,
        "model_payloads": [{"path": model_path, "sha256": "a" * 64, "bytes": 1234}], "retrieved": {}}
    paths, log_lines = [], ["[WARNING] [debug]: from img skip realpath", "load model success"]
    for desc in descriptors:
        # Deliberately use reverse numerical filenames. Only explicit logged
        # indices, never filenames or file size, can establish the mapping.
        name = f"outputs/output_{81 - desc['index']:03d}"
        path = board / name
        with path.open("wb") as stream:
            stream.truncate(desc["bytes"])
        report["retrieved"][name] = record(path)
        paths.append(name)
        log_lines.append(f"output[{desc['index']}] :{remote}/{name}")
    log_lines.append("Execute result :Success")
    for name, raw in (("tool.stdout", "\n".join(log_lines).encode()), ("tool.stderr", b""),
                      ("tool.exitcode", b"0\n"), ("tool.pid", b"8985\n")):
        path = board / name
        path.write_bytes(raw)
        report["retrieved"][name] = record(path)
    save(board / "command.json", {"argv": [plan["tool_binding"]["path"], "--model=" + model_path,
        "--modelmanager_itf=V2", "--direct_from_file=true", "--times=1", "--weight_dir=" + str(Path(model_path).parent),
        "--output_dir=" + remote + "/outputs", "--output_type=" + ",".join(row["tool_output_type"] for row in descriptors)]})
    value = {"execution": execution, "board": board, "plan": plan, "report": report, "paths": paths,
             "out": tmp_path / "output-inventory"}
    refresh(value)
    return value


def inspect(case, **options):
    return inventory.inspect_target_prefill_outputs(case["execution"], case["out"], **options)


def replace_log(case, text):
    path = case["board"] / "tool.stdout"
    path.write_text(text)
    case["report"]["retrieved"]["tool.stdout"] = record(path)
    refresh(case)


def test_complete_explicit_log_binds_slots_without_trusting_reverse_numbered_filenames(case):
    result = inspect(case)
    assert result["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", result
    assert result["plan_output_count"] == result["actual_retrieved_output_file_count"] == 82
    assert result["bound_tensors"]["feature"]["path"] == "outputs/output_000"
    assert result["bound_tensors"]["logits"]["path"] == "outputs/output_081"
    assert result["bound_tensors"]["feature"]["statistics"]["elements"] == 1048576
    assert result["bound_tensors"]["feature"]["statistics"]["zero_count"] == 1048576
    assert result["bound_tensors"]["logits"]["statistics"]["argmax"]["token_id"] == 0
    assert result["raw_extent_multiset_check"] == "MATCHED_COUNTS_AND_EXTENTS_ONLY"
    assert result["tool_logs"]["tool.stdout"]["filtered_skip_realpath_lines"] == 1
    assert result["state_layer_binding"] == "UNVERIFIED" and result["numerical_validation"] == "NOT_RUN"
    assert result["model_payload_access"] == result["weight_payload_access"] == "NONE"
    assert all(row["slot_binding"] == "EXPLICIT_TOOL_LOG_PATH_AND_DESCRIPTOR_EXTENT" for row in result["output_files"])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_directory", "missing_file", "wrong_size", "malformed", "wrong_index"])
def test_ambiguous_or_incomplete_log_never_promotes_filename_or_size_candidates(case, fault):
    text = (case["board"] / "tool.stdout").read_text()
    feature_line = next(line for line in text.splitlines() if line.startswith("output[81]"))
    if fault == "missing": text = text.replace(feature_line, "no explicit feature log")
    elif fault == "duplicate": text += "\n" + feature_line
    elif fault == "wrong_directory": text = text.replace("run-fixture/outputs/output_000", "different-run/outputs/output_000")
    elif fault == "missing_file": text = text.replace("outputs/output_000", "outputs/not-retrieved")
    elif fault == "wrong_size":
        text = text.replace(feature_line, feature_line.replace("outputs/output_000", "outputs/output_081"))
    elif fault == "malformed": text = text.replace("output[81]", "output[81x]")
    else: text = text.replace("output[81]", "output[99]")
    replace_log(case, text)
    result = inspect(case)
    assert result["status"] == "OUTPUT_FILES_INSPECTED_SLOT_BINDING_UNVERIFIED", result
    assert result["logged_slot_mapping"]["status"] == "UNVERIFIED" and result["bound_tensors"] == {}
    assert result["raw_extent_multiset_check"] == "MATCHED_COUNTS_AND_EXTENTS_ONLY"
    assert len(next(row for row in result["output_files"] if row["path"] == case["paths"][81])["candidate_runtime_indices_by_bytes_only"]) == 21


def test_actual_count_is_separate_from_plan_count_and_extra_profile_is_unclassified(case):
    removed = case["paths"][81]
    case["report"]["retrieved"].pop(removed)
    path = case["board"] / "outputs/profiling.csv"
    path.write_bytes(b"retained profile metadata")
    case["report"]["retrieved"]["outputs/profiling.csv"] = record(path)
    refresh(case)
    result = inspect(case)
    assert result["status"] == "OUTPUT_FILES_INSPECTED_SLOT_BINDING_UNVERIFIED"
    assert result["actual_retrieved_output_file_count"] == result["plan_output_count"] == 82
    assert result["raw_extent_multiset_check"] == "DIFFERENT_COUNTS_OR_EXTENTS"
    assert result["unclassified_files"] == ["outputs/profiling.csv"]
    assert next(row for row in result["raw_extent_counts"] if row["bytes"] == 2097152)["missing"] == 1


@pytest.mark.parametrize("fault", ["content", "size", "symlink", "escape", "outer", "exit", "pid", "plan", "proof", "model", "format"])
def test_changed_or_unbound_evidence_fails_without_any_model_or_device_reads(case, fault):
    path = case["board"] / case["paths"][0]
    if fault == "content":
        with path.open("r+b") as stream: stream.write(b"x")
    elif fault == "size":
        with path.open("ab") as stream: stream.write(b"x")
    elif fault == "symlink":
        source = path.with_name("outside-copy")
        path.rename(source); path.symlink_to(source)
    elif fault == "escape":
        row = case["report"]["retrieved"].pop(case["paths"][0])
        case["report"]["retrieved"]["outputs/../../outside"] = row; refresh(case)
    elif fault == "outer":
        value = json.loads((case["execution"] / "result.json").read_text()); value["tool_process_id"] += 1
        save(case["execution"] / "result.json", value)
    elif fault in {"exit", "pid"}:
        name = "tool.exitcode" if fault == "exit" else "tool.pid"
        (case["board"] / name).write_bytes(b"7\n")
        case["report"]["retrieved"][name] = record(case["board"] / name); refresh(case)
    elif fault == "plan":
        with (case["board"] / "plan.json").open("a") as stream: stream.write("\n")
    elif fault == "proof":
        value = deepcopy(case["plan"]); value["outputs"][0]["shape"] = [248320]
        save(case["board"] / "runtime-prefill-proof.json", value)
    elif fault == "model":
        case["report"]["model_payloads"][0]["sha256"] = "b" * 64; refresh(case)
    else:
        value = json.loads((case["board"] / "command.json").read_text())
        value["argv"].append("--output_format=ND"); save(case["board"] / "command.json", value)
    result = inspect(case)
    assert result["status"] == "FAILED" and result["error"], result
    assert result["model_payload_access"] == result["weight_payload_access"] == "NONE"


def test_binding_metadata_changed_during_file_hash_is_rejected(case, monkeypatch):
    original = inventory._hash_retrieved
    changed = False
    def mutate(path, row, **options):
        nonlocal changed
        result = original(path, row, **options)
        if not changed:
            changed = True
            with (case["execution"] / "result.json").open("a") as stream: stream.write("\n")
        return result
    monkeypatch.setattr(inventory, "_hash_retrieved", mutate)
    result = inspect(case)
    assert result["status"] == "FAILED" and "changed during" in result["error"]


def test_nonfinite_bound_statistics_are_explicit_and_do_not_become_accuracy_acceptance(case):
    for index, data in ((81, struct.pack("<4e", float("nan"), float("inf"), -float("inf"), 0.5)),
                        (0, struct.pack("<3f", float("nan"), float("inf"), -float("inf")))):
        path = case["board"] / case["paths"][index]
        with path.open("r+b") as stream: stream.write(data)
        case["report"]["retrieved"][case["paths"][index]] = record(path)
    refresh(case)
    result = inspect(case)
    assert result["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    for label in ("feature", "logits"):
        stats = result["bound_tensors"][label]["statistics"]
        assert stats["nan_count"] == stats["positive_infinity_count"] == stats["negative_infinity_count"] == 1
    assert result["bound_tensors"]["logits"]["statistics"]["argmax"]["status"] == "NOT_REPORTED_NONFINITE_VALUES_PRESENT"
    assert result["numerical_validation"] == "NOT_RUN"


def test_optional_manifest_and_original_ids_read_only_without_following_model_records(case, tmp_path):
    bundle, inputs = tmp_path / "draft-bundle", tmp_path / "original-inputs"
    manifest = {"artifact_kind": "qwen35-standalone-draft-bundle", "status": "PREPARED", "sequence_length": 64,
                "inputs": [{"name": "features", "shape": [1, 64, 16384]}], "outputs": [{"name": "top1"}],
                "model": {"path": "/must-not-open/large-model.omc", "bytes": 3 << 30, "sha256": "f" * 64}}
    save(bundle / "manifest.json", manifest)
    inputs.mkdir(); (inputs / "input_ids.bin").write_bytes(struct.pack("<64i", *range(64)))
    result = inspect(case, draft_bundle=bundle, input_dir=inputs)
    assert result["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    assert result["draft_bundle"]["status"] == "MANIFEST_OBSERVED_ONLY"
    assert result["draft_bundle"]["manifest_status"] == "PREPARED"
    assert result["draft_bundle"]["sequence_length"] == 64 and result["draft_bundle"]["model"] == manifest["model"]
    assert result["original_input_ids"]["values"] == list(range(64))
    assert result["original_input_ids"]["matches_planned_input_dir"] is True


@pytest.mark.parametrize("fault", ["feature_index", "feature_shape", "logits_index"])
def test_same_extent_kv_cannot_be_selected_as_feature_by_tampered_saved_plan(case, fault):
    if fault == "feature_index":
        case["plan"]["feature_runtime_output_index"] = 7  # A2MiB KV descriptor.
    elif fault == "feature_shape":
        case["plan"]["outputs"][81]["shape"] = [64, 32, 32, 16]  # Same exact2MiB.
    else:
        case["plan"]["logits_runtime_output_index"] = 7
    refresh(case)  # Even coherent rewritten metadata cannot replace typed-view proof.
    result = inspect(case)
    assert result["status"] == "FAILED"
    assert "descriptor" in result["error"]


@pytest.mark.parametrize("raw", ['{"sequence_length": 1e999}', '{"status":"PREPARED","status":"FAILED"}'])
def test_malformed_optional_manifest_is_reported_separately_without_losing_bound_outputs(case, tmp_path, raw):
    bundle = tmp_path / "bad-bundle"; bundle.mkdir()
    (bundle / "manifest.json").write_text(raw)
    result = inspect(case, draft_bundle=bundle)
    assert result["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    assert result["draft_bundle"]["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("fault,expected", [("missing", "NOT_FOUND"), ("size", "DIFFERENT_EXTENT"),
    ("range", "OUT_OF_VOCABULARY"), ("symlink", "UNAVAILABLE")])
def test_optional_bad_ids_never_suppress_output_inventory(case, tmp_path, fault, expected):
    inputs = tmp_path / "original-inputs"; inputs.mkdir()
    path = inputs / "input_ids.bin"
    if fault == "size": path.write_bytes(b"\0")
    elif fault == "range": path.write_bytes(struct.pack("<64i", *([-1] + [0] * 63)))
    elif fault == "symlink": path.symlink_to(case["board"] / case["paths"][0])
    result = inspect(case, draft_bundle=tmp_path / "missing-bundle", input_dir=inputs)
    assert result["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    assert result["draft_bundle"]["status"] == "UNAVAILABLE"
    assert result["original_input_ids"]["status"] == expected


def test_existing_inspection_output_is_never_overwritten(case):
    case["out"].mkdir(); marker = case["out"] / "retained"; marker.write_text("evidence")
    with pytest.raises(FileExistsError): inspect(case)
    assert marker.read_text() == "evidence"


def test_cli_default_is_compact_and_full_report_retains_all_verified_files(case, capsys):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_prefill_outputs.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["--execution", str(case["execution"]), "--output", str(case["out"])]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
    assert "output_files" not in shown and "tool_logs" not in shown
    full = json.loads(Path(shown["report"]).read_text())
    assert len(full["output_files"]) == 82 and len(full["logged_slot_mapping"]["entries"]) == 82

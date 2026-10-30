"""Read-only fresh state-map validation preserves receipts and current IDs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import pytest

from qwen35_moe_dflash import target_decode_state_lineage as audit
from qwen35_moe_dflash.target_compile_failure import _stamp
from test_target_decode_state_lineage import case, inspect
from test_target_state_lineage import attach_torchair_source, save
from qwen35_moe_dflash.target_prefill_smoke_plan import _source


def prepare(case):
    report = inspect(case)
    assert report["status"] == "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED", report
    return report


def test_revalidation_returns_current_full_report_with_one_graph_parse_and_no_writes(case, monkeypatch):
    prepare(case)
    before = {str(path): _stamp(path.stat()) for path in case["tmp"].rglob("*") if path.is_file()}
    old_bytes = (case["out"] / "report.json").read_bytes()
    calls = []
    actual = audit.lineage.load_target_graph
    def count(*args, **kwargs):
        calls.append(args[0])
        return actual(*args, **kwargs)
    monkeypatch.setattr(audit.lineage, "load_target_graph", count)
    monkeypatch.setattr(audit, "save_json", lambda *a, **k: pytest.fail("validator wrote a receipt"))
    actual_open = os.open
    def guard(path, *args, **kwargs):
        if str(path).endswith(".weight"):
            pytest.fail("state validator opened a weight")
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", guard)
    current = audit.validate_target_decode_state_lineage(case["out"])
    assert calls == [str(case["decode_job"])]
    assert current["status"] == "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED"
    assert current["decode"]["state_file_mapping"]["bound_count"] == 80
    assert current["saved_report_identity"]["sha256"] == hashlib.sha256(old_bytes).hexdigest()
    assert current["saved_report_identity"]["stat"] == _stamp((case["out"] / "report.json").stat())
    assert current["feedback_ready"] is False and current["device_calls"] == current["weight_payload_access"] == "NONE"
    assert old_bytes == (case["out"] / "report.json").read_bytes()
    assert before == {str(path): _stamp(path.stat()) for path in case["tmp"].rglob("*") if path.is_file()}


def test_historical_output_ctimes_return_current80_identities_without_rewriting_old_report(case):
    old = prepare(case)
    old_bytes = (case["out"] / "report.json").read_bytes()
    time.sleep(0.02)
    for row in case["inspection"]["output_files"]:
        path = case["execution"] / "board" / row["path"]
        os.chmod(path, path.stat().st_mode & 0o7777)
    observations = []
    current = audit.validate_target_decode_state_lineage(case["out"], payload_revalidation_observations=observations)
    old_states = old["decode"]["state_file_mapping"]["states"]
    new_states = current["decode"]["state_file_mapping"]["states"]
    assert len(new_states) == 80 and observations
    for prior, fresh in zip(old_states, new_states):
        assert prior["source_refdata"] == fresh["source_refdata"]
        assert prior["output_file"]["stat"][:4] == fresh["output_file"]["stat"][:4]
        assert prior["output_file"]["stat"][4] != fresh["output_file"]["stat"][4]
        assert fresh["output_file"]["stat"] == _stamp(Path(fresh["output_file"]["path"]).stat())
    assert old_bytes == (case["out"] / "report.json").read_bytes()


def test_historical_metadata_ctimes_and_small_graph_ctimes_use_existing_role_bound_rules(case):
    old = prepare(case)
    old_bytes = (case["out"] / "report.json").read_bytes()
    paths = {Path(row["path"]) for row in old["source_bindings"] if Path(row["path"]).suffix == ".json"}
    paths.update((Path(old["decode"]["air"]["path"]), Path(old["decode"]["model"]["path"])))
    time.sleep(0.02)
    for path in paths: os.chmod(path, path.stat().st_mode & 0o7777)
    primary, metadata, payload = [], [], []
    current = audit.validate_target_decode_state_lineage(case["out"], revalidation_observations=primary,
        metadata_revalidation_observations=metadata, payload_revalidation_observations=payload)
    assert current["status"] == "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED"
    assert primary and metadata and any(row["role"] == "source-air" for row in payload if "role" in row)
    assert old_bytes == (case["out"] / "report.json").read_bytes()


@pytest.mark.parametrize("changed_mode", [False, True])
def test_source_input_current_full_identity_allows_only_ctime_after_fresh_input_hash(case, monkeypatch, changed_mode):
    folder = case["tmp"] / "provided-inputs"
    folder.mkdir()
    path = folder / "inputs_embeds.bin"
    path.write_bytes(b"12")
    source = _source(path, 2)
    case["plan"]["source_inputs"] = [source]
    case["plan"]["input_dir"] = str(folder)
    save(case["execution"] / "plan.json", case["plan"])
    original = case["validate"]
    def fresh_input_and_inventory(*args, **kwargs):
        fresh = _source(path, 2)
        assert fresh["sha256"] == source["sha256"]
        return original(*args, **kwargs)
    monkeypatch.setattr(audit, "validate_target_decode_output_inventory", fresh_input_and_inventory)
    old = prepare(case)
    time.sleep(0.02)
    os.chmod(path, 0o777 if changed_mode else path.stat().st_mode & 0o7777)
    if changed_mode:
        with pytest.raises(ValueError, match="claim changed"):
            audit.validate_target_decode_state_lineage(case["out"])
    else:
        current = audit.validate_target_decode_state_lineage(case["out"])
        before, after = old["source_input_current_identities"][0], current["source_input_current_identities"][0]
        assert before["identity"]["ctime_ns"] != after["identity"]["ctime_ns"]
        assert after["identity"] == _source(path, 2)["identity"]


@pytest.mark.parametrize("explicit_copy", [False, True])
def test_source_schema_selection_and_identical_explicit_pb2_copy_are_preserved(case, explicit_copy):
    producer = attach_torchair_source(case)
    report_sha = producer["outer"]["compilation_report_sha256"]
    case["probe"]["compilation_report_sha256"] = report_sha
    save(case["decode_probe"] / "result.json", case["probe"])
    case["plan"]["compilation_report_sha256"] = report_sha
    save(case["execution"] / "plan.json", case["plan"])
    if explicit_copy:
        copy = case["tmp"] / "copied-producer.py"
        copy.write_bytes(producer["pb2"].read_bytes())
        producer["pb2"].unlink()
        old = audit.inspect_target_decode_state_lineage(case["decode_job"], case["decode_probe"], case["inspection_dir"], case["out"],
            source_torchair_pb2=copy)
    else:
        old = prepare(case)
    current = audit.validate_target_decode_state_lineage(case["out"])
    assert current["decode"]["source_schema_binding"]["selection"] == old["decode"]["source_schema_binding"]["selection"]
    assert current["decode"]["source_schema_binding"]["selected_pb2"]["path"] == old["decode"]["source_schema_binding"]["selected_pb2"]["path"]
    assert current["decode"]["state_file_mapping"]["bound_count"] == 80


@pytest.mark.parametrize("fault", ["owner", "compiled_port", "output_file", "fx", "runtime_name", "missing_state", "feedback", "malformed_graph_ctime", "malformed_runtime_index"])
def test_saved_mapping_claims_cannot_be_changed_without_matching_fresh_graph_file_proof(case, fault):
    prepare(case)
    path = case["out"] / "report.json"
    saved = json.loads(path.read_text())
    state = saved["decode"]["state_file_mapping"]["states"][0]
    if fault == "owner": state["source_refdata"] = "RefData_0"
    elif fault == "compiled_port": state["compiled_output_index"] = (state["compiled_output_index"] + 1) % 82
    elif fault == "output_file": state["output_file"] = deepcopy(saved["decode"]["state_file_mapping"]["states"][1]["output_file"])
    elif fault == "fx": saved["decode"]["compiled_lineage"]["states"][0]["fx_tensor_identity"] = "guessed"
    elif fault == "runtime_name": state["runtime_output_name"] = "guessed"
    elif fault == "missing_state": saved["decode"]["state_file_mapping"]["states"].pop()
    elif fault == "malformed_graph_ctime": saved["decode"]["model"]["stat"][4] = True
    elif fault == "malformed_runtime_index":
        selected = next(row for row in saved["decode"]["state_file_mapping"]["states"] if row["runtime_input_index"] == 0)
        selected["runtime_input_index"] = False
    else: saved["feedback_ready"] = True
    save(path, saved)
    with pytest.raises(ValueError, match="claim|complete retained"):
        audit.validate_target_decode_state_lineage(case["out"])


@pytest.mark.parametrize("fault", ["output_bytes", "air_bytes", "model_bytes", "model_inode"])
def test_revalidation_rejects_changed_real_artifacts(case, fault):
    old = prepare(case)
    if fault == "output_bytes": path = Path(old["decode"]["state_file_mapping"]["states"][0]["output_file"]["path"])
    elif fault == "air_bytes": path = Path(old["decode"]["air"]["path"])
    else: path = Path(old["decode"]["model"]["path"])
    before, raw = path.stat(), path.read_bytes()
    if fault == "model_inode":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(raw)
        os.chmod(replacement, before.st_mode & 0o7777)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
    else:
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="fresh decode state lineage is incomplete"):
        audit.validate_target_decode_state_lineage(case["out"])


def test_active_change_after_recomputed_mapping_is_never_historical_ctime_recovery(case, monkeypatch):
    prepare(case)
    actual = audit._observe_target_decode_state_lineage
    def change_after_observation(*args, **kwargs):
        current = actual(*args, **kwargs)
        path = Path(current["decode"]["state_file_mapping"]["states"][0]["output_file"]["path"])
        time.sleep(0.02)
        os.chmod(path, path.stat().st_mode & 0o7777)
        return current
    monkeypatch.setattr(audit, "_observe_target_decode_state_lineage", change_after_observation)
    with pytest.raises(ValueError, match="state output changed during read-only"):
        audit.validate_target_decode_state_lineage(case["out"])


@pytest.mark.parametrize("target", ["input", "output", "log"])
@pytest.mark.parametrize("operation", ["inspect", "validate"])
def test_first_payload_validation_cannot_rebase_current_input_output_or_log_identity(case, monkeypatch, target, operation):
    folder = case["tmp"] / "provided-current-input"
    folder.mkdir()
    source = folder / "inputs_embeds.bin"
    source.write_bytes(b"12")
    case["plan"]["source_inputs"] = [_source(source, 2)]
    save(case["execution"] / "plan.json", case["plan"])
    logs = {}
    for name in ("tool.stdout", "tool.stderr", "tool.exitcode", "tool.pid"):
        path = case["execution"] / "board" / name
        path.write_bytes(b"retained log")
        logs[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "stat": _stamp(path.stat())}
    case["inspection"]["tool_logs"] = logs
    save(case["inspection_dir"] / "report.json", case["inspection"])
    prepare(case)
    old = (case["out"] / "report.json").read_bytes()
    original, calls = case["validate"], 0
    def mutate_after_first_fresh_payload_validation(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        _source(source, 2)
        for name in logs: hashlib.sha256((case["execution"] / "board" / name).read_bytes()).hexdigest()
        calls += 1
        if calls == 1:
            path = (source if target == "input" else case["execution"] / "board" / "tool.stdout" if target == "log" else
                case["execution"] / "board" / case["inspection"]["output_files"][3]["path"])
            time.sleep(0.02)
            os.chmod(path, path.stat().st_mode & 0o7777)
        return value
    monkeypatch.setattr(audit, "validate_target_decode_output_inventory", mutate_after_first_fresh_payload_validation)
    if operation == "validate":
        with pytest.raises(ValueError, match="fresh decode state lineage is incomplete"):
            audit.validate_target_decode_state_lineage(case["out"])
    else:
        current = audit.inspect_target_decode_state_lineage(case["decode_job"], case["decode_probe"], case["inspection_dir"], case["tmp"] / "active-reinspect")
        assert current["status"] == "FAILED" and "changed during state lineage audit" in current["error"]
    assert old == (case["out"] / "report.json").read_bytes()


def test_invalid_collector_is_refused_before_evidence_reads(monkeypatch):
    monkeypatch.setattr(audit, "_safe", lambda *a, **k: pytest.fail("invalid collector performed a read"))
    with pytest.raises(ValueError, match="observations must be lists"):
        audit.validate_target_decode_state_lineage("/unused", payload_revalidation_observations={})

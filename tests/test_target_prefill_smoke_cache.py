"""A failed tool preflight retains exact compiled bytes for a fresh attempt.

Small fake payloads exercise checksum decisions; no vendor model, large weight
or real target execution occurs.
"""
import json
from pathlib import Path

from qwen35_moe_dflash.existing_air import _stat
from test_target_prefill_smoke_execution import case, FakeHdc, execute
from test_target_prefill_smoke_plan import binding, save, sha


def _add_small_external_weight(case):
    weight = case["model"].parent / "SubGraph_0.weight"
    weight.write_bytes(b"small external-weight fixture; never a real checkpoint")
    case["compiled"]["generated_artifacts"].append({
        "path": weight.relative_to(case["compilation"]).as_posix(),
        "bytes": weight.stat().st_size, "sha256": sha(weight), "stat": _stat(weight),
        "role": "external_data_candidate"})
    save(case["compilation"] / "report.json", case["compiled"])
    compiled_sha = sha(case["compilation"] / "report.json")
    for path in (case["job"] / "result.json", case["job"] / "target-contract/report.json"):
        value = json.loads(path.read_text())
        value["compilation_report_sha256"] = compiled_sha
        save(path, value)
    outer = case["outer"]
    outer["compilation_report_sha256"] = compiled_sha
    outer["source_bindings"] = [binding(Path(row["path"])) for row in outer["source_bindings"]]
    save(case["probe"] / "result.json", outer)
    return weight


def _failed_attempt(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "status", 255)
    result = execute(case)
    assert result["status"] == "FAILED" and result["remote_exitcode"] == 255
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    assert {row["upload"] for row in result["model_payloads"]} == {"SEND"}
    assert len(result["model_payloads"]) == 2
    assert not result["retrieved"].keys() & {"outputs/vendor_000.bin"}
    monkeypatch.setattr(FakeHdc, "status", 0)
    return result


def test_fresh_attempt_reuses_checksum_matched_primary_and_external_weight_after_tool_failure(case, monkeypatch):
    weight = _add_small_external_weight(case)
    source_bytes = {path: path.read_bytes() for path in (*case["source"].iterdir(), case["model"], weight)}
    first = _failed_attempt(case, monkeypatch)
    retained_report = (case["out"] / "result.json").read_bytes()
    previous_sends = len(FakeHdc.sends)
    retry_case = dict(case, out=case["out"].with_name("fresh-prefill-retry"))
    retry = execute(retry_case)
    assert retry["status"] == "EXECUTED_UNVALIDATED", retry
    assert retry["board_model_cache"] == first["board_model_cache"]
    assert retry["board_attempt"] != first["board_attempt"]
    assert {row["upload"] for row in retry["model_payloads"]} == {"OK"}
    assert retry["board_disk_budget"]["missing_model_payload_bytes"] == 0
    sent = FakeHdc.sends[previous_sends:]
    assert not any(local.endswith((".omc", ".weight")) for local, _ in sent)
    assert (case["out"] / "result.json").read_bytes() == retained_report
    assert all(path.read_bytes() == raw for path, raw in source_bytes.items())


def test_fresh_attempt_repairs_only_checksum_mismatched_cached_external_weight(case, monkeypatch):
    weight = _add_small_external_weight(case)
    expected_weight = weight.read_bytes()
    first = _failed_attempt(case, monkeypatch)
    remote_weight = next(row["path"] for row in first["model_payloads"] if row["path"].endswith(".weight"))
    FakeHdc.remote[remote_weight] = b"x" * len(expected_weight)
    previous_sends = len(FakeHdc.sends)
    retry = execute(dict(case, out=case["out"].with_name("fresh-prefill-cache-repair")))
    assert retry["status"] == "EXECUTED_UNVALIDATED", retry
    decisions = {Path(row["path"]).suffix: row["upload"] for row in retry["model_payloads"]}
    assert decisions == {".omc": "OK", ".weight": "SEND"}
    assert retry["board_disk_budget"]["missing_model_payload_bytes"] == len(expected_weight)
    sent_payloads = [(local, remote) for local, remote in FakeHdc.sends[previous_sends:]
                     if local.endswith((".omc", ".weight"))]
    assert sent_payloads == [(str(weight), remote_weight)]
    assert FakeHdc.remote[remote_weight] == expected_weight == weight.read_bytes()

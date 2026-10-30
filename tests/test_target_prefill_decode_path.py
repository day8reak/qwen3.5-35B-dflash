"""A combined audit preserves independent gaps and one active source baseline."""
import json
import os
from pathlib import Path

import pytest

from qwen35_moe_dflash import target_prefill_decode_path as packet
from qwen35_moe_dflash import target_prefill_decode_bridge as files
from qwen35_moe_dflash import target_logit_selection as logits
from qwen35_moe_dflash import target_state_correspondence as states


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "retained-source.json"
    source.write_text('{"retained":true}\n')
    source, stamp = packet._safe(source)
    identity = packet._identity(source.stat())
    reports = {
        "prefill_files": {"status": "PREFILL_NAMED_SLOTS_AND_RETAINED_FILES_BOUND_BRIDGE_GAPS_REMAIN",
            "named_probe_witness": {"status": "POST_HOC_SAME_MODEL_ORDERED_DESCRIPTOR_NAMES_BOUND"},
            "retained_outputs": [{}] * 82, "executed_input_byte_binding": "ALL89_BOUND",
            "active_source_identities": [{"path": str(source), "identity": identity}]},
        "logit_selection": {"status": "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED",
            "source_bindings": [{"path": str(source), "stat": stamp}]},
        "state_correspondence": {"status": "CANDIDATES_WITH_GAPS", "unique_state_context_candidate_count": 80,
            "proven_cross_model_state_owner_count": 0, "feedback_ready": False,
            "source_bindings": [{"path": str(source), "stat": stamp}]},
    }
    calls = []
    def child(name):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return reports[name]
        return run
    monkeypatch.setattr(files, "inspect_prefill_decode_bridge", child("prefill_files"))
    monkeypatch.setattr(logits, "inspect_target_logit_selection", child("logit_selection"))
    monkeypatch.setattr(states, "inspect_target_state_correspondence", child("state_correspondence"))
    return {"source": source, "reports": reports, "calls": calls, "output": tmp_path / "packet"}


def inspect(case):
    return packet.inspect_prefill_decode_path("prefill-job", "decode-job", "old-prefill", "named-prefill",
        case["output"], source_torchair_pb2="copied-pb2.py", prefill_source_export_receipt="air-receipt.json")


def test_combined_evidence_keeps_out_of_range_and80_candidates_without_preparing_feedback(case):
    result = inspect(case)
    assert result["status"] == "PREFILL_DECODE_PATH_EVIDENCE_OBSERVED_WITH_GAPS"
    assert result["feedback_prepared"] is False and result["device_execution"] == "NOT_RUN"
    assert result["summary"]["bound_prefill_output_count"] == 82
    assert result["summary"]["unique_state_context_candidate_count"] == 80
    assert result["summary"]["proven_cross_model_state_owner_count"] == 0
    assert "not accepted as the next-token seed" in result["next"]
    assert result["active_source_identities"][0]["identity"] == packet._identity(case["source"].stat())
    calls = case["calls"]
    assert [row[0] for row in calls] == ["prefill_files", "logit_selection", "state_correspondence"]
    assert calls[0][1][:2] == ("old-prefill", "named-prefill")
    assert calls[1][2] == {"source_torchair_pb2": "copied-pb2.py", "source_export_receipt": "air-receipt.json"}
    assert calls[2][2] == {"prefill_source_torchair_pb2": "copied-pb2.py", "decode_source_torchair_pb2": "copied-pb2.py",
        "prefill_source_export_receipt": "air-receipt.json"}
    assert json.loads((case["output"] / "report.json").read_text()) == result


@pytest.mark.parametrize("raises", [False, True])
def test_one_failed_check_retains_the_other_independent_diagnostics(case, monkeypatch, raises):
    if raises:
        def failed(*args, **kwargs):
            raise ValueError("source schema unavailable")
        monkeypatch.setattr(logits, "inspect_target_logit_selection", failed)
    else:
        case["reports"]["logit_selection"] = {"status": "FAILED", "error": "source schema unavailable"}
    result = inspect(case)
    assert result["status"] == "PREFILL_DECODE_PATH_EVIDENCE_FAILED"
    assert result["failed_checks"] == ["logit_selection"]
    assert "schema unavailable" in result["checks"]["logit_selection"]["error"]
    assert result["checks"]["state_correspondence"]["unique_state_context_candidate_count"] == 80
    assert result["feedback_prepared"] is False


@pytest.mark.parametrize("mutation", ["chmod", "symlink"])
def test_source_cannot_change_between_independent_child_audits(case, monkeypatch, mutation):
    def changed(*args, **kwargs):
        source = case["source"]
        if mutation == "chmod":
            os.chmod(source, 0o600)
        else:
            target = source.with_suffix(".replacement")
            source.rename(target)
            source.symlink_to(target)
        return case["reports"]["logit_selection"]
    monkeypatch.setattr(logits, "inspect_target_logit_selection", changed)
    result = inspect(case)
    assert result["status"] == "FAILED"
    assert "changed" in result["error"] or "symlink" in result["error"]
    assert result["feedback_prepared"] is False
    assert not any(row[0] == "state_correspondence" for row in case["calls"])


def test_full_identity_capture_rejects_a_change_after_five_stat_comparison(case, monkeypatch):
    identity = packet._identity
    monkeypatch.setattr(packet, "_identity", lambda info: dict(identity(info), ctime_ns=info.st_ctime_ns + 1))
    with pytest.raises(ValueError, match="active identity was captured"):
        packet._capture(case["reports"]["logit_selection"], {})


def test_existing_packet_is_preserved(case):
    case["output"].mkdir()
    retained = case["output"] / "retained.json"
    retained.write_text("original evidence\n")
    with pytest.raises(FileExistsError):
        inspect(case)
    assert retained.read_text() == "original evidence\n" and not case["calls"]


def test_cli_prints_compact_check_statuses_and_never_calls_apply(case, capsys):
    import runpy
    cli = runpy.run_path(str(Path(__file__).parents[1] / "targets/cpro/scripts/inspect_target_prefill_decode_path.py"))
    assert cli["main"](["--prefill-job", "prefill-job", "--decode-job", "decode-job",
        "--prefill-execution", "old-prefill", "--prefill-runtime-probe", "named-prefill",
        "--source-torchair-pb2", "copied-pb2.py", "--prefill-source-export-receipt", "air-receipt.json",
        "--output", str(case["output"])]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["feedback_prepared"] is False and shown["device_execution"] == "NOT_RUN"
    assert shown["check_details"]["logit_selection"]["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED"
    assert "checks" not in shown and "active_source_identities" not in shown

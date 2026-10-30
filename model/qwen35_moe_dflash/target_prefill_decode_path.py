"""Collect the independent host evidence needed before a prefill/decode bridge."""
from __future__ import annotations

from pathlib import Path

from .artifacts import output_path, save_json
from .target_compile_failure import _safe
from .target_prefill_smoke_plan import _identity


def _capture(report, baselines):
    for row in report.get("source_bindings", []):
        path, stamp = _safe(row["path"])
        if stamp != row["stat"]:
            raise ValueError("child audit source changed before path evidence was collected: " + str(path))
        current = _identity(path.stat())
        if [current[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")] != stamp:
            raise ValueError("child audit source changed while its active identity was captured: " + str(path))
        previous = baselines.setdefault(str(path), current)
        if current != previous:
            raise ValueError("shared audit source identity changed between path checks: " + str(path))
    for row in report.get("active_source_identities", []):
        path, _ = _safe(row["path"])
        current = _identity(path.stat())
        if current != row["identity"] or baselines.setdefault(str(path), current) != current:
            raise ValueError("retained prefill source changed between path checks: " + str(path))


def _stable(baselines):
    for name, identity in baselines.items():
        path, _ = _safe(name)
        if _identity(path.stat()) != identity:
            raise ValueError("path evidence source changed during the combined audit: " + name)


def inspect_prefill_decode_path(prefill_job, decode_job, prefill_execution, prefill_runtime_probe, output, *,
                                source_torchair_pb2=None, prefill_source_export_receipt=None):
    from .target_prefill_decode_bridge import inspect_prefill_decode_bridge
    from .target_logit_selection import inspect_target_logit_selection
    from .target_state_correspondence import inspect_target_state_correspondence
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-prefill-decode-path-evidence",
        "status": "FAILED", "output": str(out), "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN",
        "weight_payload_access": "NONE", "numerical_validation": "NOT_RUN", "feedback_prepared": False,
        "state_continuity": "NOT_VALIDATED", "checks": {}}
    baselines = {}
    jobs = (
        ("prefill_files", lambda: inspect_prefill_decode_bridge(prefill_execution, prefill_runtime_probe, out / "prefill-files")),
        ("logit_selection", lambda: inspect_target_logit_selection(prefill_job, prefill_execution, out / "logit-selection",
            source_torchair_pb2=source_torchair_pb2, source_export_receipt=prefill_source_export_receipt)),
        ("state_correspondence", lambda: inspect_target_state_correspondence(prefill_job, decode_job, out / "state-correspondence",
            prefill_source_torchair_pb2=source_torchair_pb2, decode_source_torchair_pb2=source_torchair_pb2,
            prefill_source_export_receipt=prefill_source_export_receipt)),
    )
    try:
        for name, call in jobs:
            result["stage"] = name
            _stable(baselines)
            try:
                report = call()
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
                report = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
            result["checks"][name] = report
            _capture(report, baselines)
            _stable(baselines)
        files = result["checks"]["prefill_files"]
        logits = result["checks"]["logit_selection"]
        states = result["checks"]["state_correspondence"]
        failed = [name for name, report in result["checks"].items() if report.get("status") == "FAILED"]
        result.update(status="PREFILL_DECODE_PATH_EVIDENCE_FAILED" if failed else "PREFILL_DECODE_PATH_EVIDENCE_OBSERVED_WITH_GAPS",
            stage="complete", failed_checks=failed,
            summary={"prefill_named_probe_witness": files.get("named_probe_witness", {}).get("status"),
                "bound_prefill_output_count": len(files.get("retained_outputs", [])),
                "executed_prefill_input_binding": files.get("executed_input_byte_binding"),
                "logit_selection": logits.get("status"),
                "unique_state_context_candidate_count": states.get("unique_state_context_candidate_count", 0),
                "proven_cross_model_state_owner_count": states.get("proven_cross_model_state_owner_count", 0)},
            reports={name: str(out / folder / "report.json") for name, folder in
                (("prefill_files", "prefill-files"), ("logit_selection", "logit-selection"), ("state_correspondence", "state-correspondence"))},
            active_source_identities=[{"path": path, "identity": value} for path, value in sorted(baselines.items())])
        if failed:
            result["next"] = "Resolve the retained failing evidence checks before preparing bridge inputs"
        elif logits.get("status") == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED":
            result["next"] = "Review the exact source index path and prepare a separate last-row correction experiment; existing prefill argmax is not accepted as the next-token seed"
        else:
            result["next"] = "Review stable cross-export state ownership, the actual logit row, original prefix embeddings and next-token lookup before preparing a separate bridge experiment"
        _stable(baselines)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    save_json(out / "report.json", result)
    return result

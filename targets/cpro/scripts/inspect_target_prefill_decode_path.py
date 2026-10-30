#!/usr/bin/env python3
"""Collect read-only prefill naming, logit-row and cross-state evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-job", required=True)
    parser.add_argument("--decode-job", required=True)
    parser.add_argument("--prefill-execution", required=True)
    parser.add_argument("--prefill-runtime-probe", required=True)
    parser.add_argument("--source-torchair-pb2")
    parser.add_argument("--prefill-source-export-receipt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_prefill_decode_path import inspect_prefill_decode_path
    try:
        result = inspect_prefill_decode_path(args.prefill_job, args.decode_job, args.prefill_execution,
            args.prefill_runtime_probe, args.output, source_torchair_pb2=args.source_torchair_pb2,
            prefill_source_export_receipt=args.prefill_source_export_receipt)
    except Exception as error:
        result = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
    shown = {key: result[key] for key in ("status", "stage", "device_execution", "model_compilation", "weight_payload_access",
        "numerical_validation", "feedback_prepared", "state_continuity", "summary", "failed_checks", "reports", "error", "next") if key in result}
    shown["check_details"] = {name: {key: report[key] for key in ("status", "stage", "error", "next") if key in report}
        for name, report in result.get("checks", {}).items()}
    shown["report"] = str(Path(args.output).resolve() / "report.json")
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["status"] == "PREFILL_DECODE_PATH_EVIDENCE_OBSERVED_WITH_GAPS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

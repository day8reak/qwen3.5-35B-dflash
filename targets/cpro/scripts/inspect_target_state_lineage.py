#!/usr/bin/env python3
"""Audit retained original target state edges and names, without model execution."""
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
    parser.add_argument("--decode-runtime-probe", required=True)
    parser.add_argument("--prefill-runtime-probe", help="optional newer named probe for the same compiled prefill model")
    parser.add_argument("--output", required=True, help="new output below AI_RUN_DIR")
    parser.add_argument("--full", action="store_true", help="print every state row and bounded updater diagnostic")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_state_lineage import inspect_target_state_lineage
    try:
        report = inspect_target_state_lineage(args.prefill_job, args.decode_job, args.prefill_execution,
            args.decode_runtime_probe, args.output, prefill_runtime_probe=args.prefill_runtime_probe)
        shown = report if args.full else {key: report[key] for key in ("status", "error", "device_execution",
            "model_compilation", "weight_payload_access", "state_feedback", "state_continuity",
            "cross_model_state_identity", "feedback_ready", "numerical_validation", "next") if key in report}
        if not args.full:
            for name in ("prefill", "decode"):
                if name in report:
                    shown[name] = {part: {key: value for key, value in report[name][part].items()
                        if key not in {"states", "updater_diagnostics"}} for part in
                        ("source_lineage", "compiled_lineage", "runtime_bindings") if part in report[name]}
        shown["report"] = str(Path(args.output).resolve() / "report.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 1 if report["status"] == "FAILED" else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

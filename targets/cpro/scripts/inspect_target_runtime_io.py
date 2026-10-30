#!/usr/bin/env python3
"""Restore small target metadata and query descriptors before manager.Init."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="completed static prefill64 or decode1 original-target job")
    parser.add_argument("--tool-inspection", required=True, help="completed actual tool-help report directory or JSON")
    parser.add_argument("--output", required=True, help="new evidence directory under AI_RUN_DIR")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--lock-wait-seconds", type=int, default=300, help="bounded wait for another probe on the same device")
    parser.add_argument("--board-model-root", default="/data/local/tmp/qwen3_5_moe")
    parser.add_argument("--headers-only", action="store_true", help="build only the native descriptor runner and retain exact selected SDK external-weight declarations; no board calls or model transfers")
    parser.add_argument("--summary-only", action="store_true", help="print a compact summary; the complete report and all descriptor arrays remain in the output directory")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_model_io_probe import inspect_target_runtime_io
    try:
        report = inspect_target_runtime_io(args.job, args.tool_inspection, args.output,
            timeout_seconds=args.timeout_seconds, lock_wait_seconds=args.lock_wait_seconds,
            board_model_root=args.board_model_root, headers_only=args.headers_only)
        shown = {key: value for key, value in report.items() if key not in {"source_bindings", "model_upload_plan", "libraries"}}
        if args.summary_only:
            shown = {key: report[key] for key in ("status", "stage", "output", "model_restore", "runtime_tensor_names",
                "input_count", "output_count", "device_execution", "model_inference", "model_compilation",
                "weight_payload_access", "lock_state", "error") if key in report}
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if report["status"] in {"IO_METADATA_OBSERVED", "HEADERS_OBSERVED"} else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

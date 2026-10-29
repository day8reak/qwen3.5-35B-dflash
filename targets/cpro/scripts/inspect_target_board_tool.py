#!/usr/bin/env python3
"""Inspect the existing phone tool's SHA256, firmware and bounded --help only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", default="/data/local/tmp/tools_sysdbg/model_run_tool_internal")
    parser.add_argument("--timeout-seconds", type=int, choices=range(3, 11), default=10,
                        help="per HDC call; remote help timeout is two seconds shorter")
    parser.add_argument("--output", required=True, help="new report directory below AI_RUN_DIR")
    parser.add_argument("--writable-help-from", help="prior identity-matched output-directory help failure; retry in one new owned board scratch directory")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_board_tool_inspect import inspect_target_board_tool
    try:
        report = inspect_target_board_tool(args.output, tool=args.tool, timeout_seconds=args.timeout_seconds,
                                           writable_help_from=args.writable_help_from)
        shown = {key: report[key] for key in ("status", "stage", "tool", "tool_sha256_before", "tool_sha256_after",
                 "tool_identity", "model_upload", "model_load", "model_inference", "runtime_input_order",
                 "runtime_tensor_names", "hiai_runtime_version", "board_mutation", "model_arguments",
                 "previous_inspection", "writable_help_directory", "scope", "error", "report") if key in report}
        for key in ("firmware", "help"):
            if key in report:
                shown[key] = report[key]
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if report["status"] == "INSPECTED" else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

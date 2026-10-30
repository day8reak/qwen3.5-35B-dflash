#!/usr/bin/env python3
"""Audit one source-bound target logit-selection index without executing it."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="completed original static64 target job")
    parser.add_argument("--execution", required=True, help="successful original prefill execution")
    parser.add_argument("--source-torchair-pb2", help="exact producer pb2 literal or identical local copy; parsed as AST data")
    parser.add_argument("--source-export-receipt", help="optional canonical static64 producer receipt matching the compiled AIR snapshot")
    parser.add_argument("--output", required=True, help="new evidence directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_logit_selection import inspect_target_logit_selection
    try:
        report = inspect_target_logit_selection(args.job, args.execution, args.output,
            source_torchair_pb2=args.source_torchair_pb2, source_export_receipt=args.source_export_receipt)
        shown = {key: value for key, value in report.items() if key not in {"source_bindings", "job"}}
        shown["report"] = str(Path(args.output).absolute() / "report.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return int(report["status"] == "FAILED")
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

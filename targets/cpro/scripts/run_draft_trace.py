#!/usr/bin/env python3
"""Trace all six draft layers once on the retained inputs; never replace production artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="retained job; default AI_RUN_DIR/full-draft")
    parser.add_argument("--ge-root", default=os.environ.get("CANN_ROOT"))
    parser.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH"))
    parser.add_argument("--ddk-id", default=os.environ.get("CPRO_DDK_ID"))
    parser.add_argument("--tbe-diagnosis", help="reuse existing TBE diagnosis")
    parser.add_argument("--schema-library")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not os.environ.get("AI_RUN_DIR"):
            raise ValueError("AI_RUN_DIR is required; source the project env.sh")
        if args.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        from qwen35_moe_dflash.draft_trace_job import run_trace
        job = args.job or Path(os.environ["AI_RUN_DIR"]) / "full-draft"
        result = run_trace(job, **{key: value for key, value in vars(args).items() if key != "job"})
    except Exception as error:
        print(f"draft-trace: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    summary = dict(result)
    if summary.get("trace_diagnostic"):
        summary["trace_diagnostic"] = {key: value for key, value in summary["trace_diagnostic"].items() if key != "nodes"}
        summary["trace_diagnostic"]["node_count"] = len(result["trace_diagnostic"].get("nodes", []))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    if result.get("trace_diagnostic"):
        for row in result["trace_diagnostic"].get("nodes", []):
            compact = {key: row[key] for key in ("name", "exact_match", "relative_l2", "cosine", "max_abs_error", "nonfinite_mismatch_count") if key in row}
            print("draft-trace: " + json.dumps(compact, ensure_ascii=False, allow_nan=False), flush=True)
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

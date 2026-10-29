#!/usr/bin/env python3
"""Review retained packed draft observations; no compilation, HDC or writes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="retained full-draft job; default AI_RUN_DIR/full-draft")
    args = parser.parse_args(argv)
    try:
        if not args.job and not os.environ.get("AI_RUN_DIR"):
            raise ValueError("--job or AI_RUN_DIR is required")
        from qwen35_moe_dflash.draft_trace_integrity import review_trace_integrity
        report = review_trace_integrity(args.job or Path(os.environ["AI_RUN_DIR"]) / "full-draft")
    except Exception as error:
        print(f"draft-trace-review: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 1 if report["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

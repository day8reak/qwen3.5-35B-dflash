#!/usr/bin/env python3
"""Share verified readonly compiler input copies; default is a read-only plan."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="original full-draft job; default AI_RUN_DIR/full-draft")
    parser.add_argument("--apply", action="store_true", help="atomically hardlink byte-identical readonly snapshots under the draft lock")
    parser.add_argument("--full", action="store_true", help="include each path mapping and inode/stat change")
    args = parser.parse_args(argv)
    if not args.job and not os.environ.get("AI_RUN_DIR"):
        parser.error("--job or AI_RUN_DIR is required")
    from qwen35_moe_dflash.draft.storage import compact_draft_storage
    try:
        report = compact_draft_storage(args.job or Path(os.environ["AI_RUN_DIR"]) / "full-draft", apply=args.apply)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAILED", "reason": f"{type(error).__name__}: {str(error)[:1600]}"}), flush=True)
        return 1
    if not args.full:
        report = {key: value for key, value in report.items() if key not in {"changes", "skipped"}} | {
            "planned_changes": len(report["changes"]),
            "linked_changes": sum(row["status"] == "LINKED" for row in report["changes"]),
            "skipped_counts": dict(Counter(row["reason"] for row in report["skipped"])),
            "detail": "Use --full to include individual path mappings and stat changes"}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

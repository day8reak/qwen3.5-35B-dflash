#!/usr/bin/env python3
"""Plan exact old-sidecar reuse; --apply hardlinks verified files and copies AIR."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def main(argv=None):
    from qwen35_moe_dflash.target_export_reuse import stage_target_export_reuse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="origin SHA256_COMPLETE small export inventory JSON")
    parser.add_argument("--air", required=True, help="copied small new AIR; exact inventory SHA256 required")
    parser.add_argument("--existing-export", required=True, help="existing old export directory containing direct sidecar files")
    parser.add_argument("--output", required=True, help="new staging directory below AI_RUN_DIR; never overwritten")
    parser.add_argument("--apply", action="store_true", help="hardlink only if all sidecars match and filesystem/space checks pass")
    parser.add_argument("--full", action="store_true", help="print the full per-file report; applied receipts always retain it")
    args = parser.parse_args(argv)
    report = stage_target_export_reuse(args.inventory, args.air, args.existing_export, args.output, apply=args.apply)
    shown = report
    if not args.full:
        shown = {key: value for key, value in report.items() if key not in {"files", "transfer_required", "created_links"}}
        shown.update(file_count=len(report.get("files", [])),
                     file_status_counts=dict(Counter(row["status"] for row in report.get("files", []))),
                     transfer_required_count=len(report.get("transfer_required", [])),
                     transfer_required=report.get("transfer_required", [])[:8],
                     created_link_count=len(report.get("created_links", [])))
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] in {"PLAN_READY", "STAGED_HARDLINKS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

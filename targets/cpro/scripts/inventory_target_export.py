#!/usr/bin/env python3
"""Inventory A target AIR/FileConstant identities without copying payloads."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", required=True, help="new A raw export.air, read only")
    parser.add_argument("--torchair-pb2", help="explicit generated pb2; AST literal only, never imported")
    parser.add_argument("--discover-torchair-schema", action="store_true",
                        help="select smallest literal pb2 only when all discovered descriptors share one SHA256")
    parser.add_argument("--reference-root", help="explicit base for prefix/basename references; default AIR.parent.parent")
    parser.add_argument("--sequence-length", type=int, default=64, help="declared feature sequence length; A default64, B may explicitly use55")
    parser.add_argument("--hash-weights", action="store_true", help="explicitly stream SHA256 of all unique payloads; no copies")
    parser.add_argument("--output", help="new small manifest directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    run = os.environ.get("AI_RUN_DIR")
    if not run:
        parser.error("AI_RUN_DIR is required")
    try:
        from qwen35_moe_dflash.target_export_inventory import inventory_target_export
        last_progress = [None]
        def progress(row):
            now = time.monotonic()
            if last_progress[0] is None or now - last_progress[0] >= 10 or row["completed"] == row["total"]:
                print(f"target-export-inventory: {'hashed' if row['payload_hashed'] else 'stat'} "
                      f"{row['completed']}/{row['total']} files; {row['basename']}", file=sys.stderr, flush=True)
                last_progress[0] = now
        result = inventory_target_export(args.air, args.output or Path(run) / "target-export-inventory",
            torchair_pb2=args.torchair_pb2, discover_schema=args.discover_torchair_schema,
            hash_weights=args.hash_weights, reference_root=args.reference_root,
            sequence_length=args.sequence_length, progress=progress)
        summary = {key: result.get(key) for key in
            ("status", "weight_identity", "unique_reference_count", "total_weight_bytes",
             "source_relative_prefix", "air_preservation", "air", "error")}
        summary["manifest"] = str(Path(args.output or Path(run) / "target-export-inventory") / "inventory.json")
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if result["status"] == "INVENTORIED" else 1
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

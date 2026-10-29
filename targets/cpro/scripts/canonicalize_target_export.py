#!/usr/bin/env python3
"""Move confirmed feature output1 to output81, then inventory original weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", required=True)
    parser.add_argument("--torchair-pb2")
    parser.add_argument("--discover-torchair-schema", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--reference-root")
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse only this task's unchanged canonical AIR and complete inventory")
    parser.add_argument("--output", help="new small task directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    if not os.environ.get("AI_RUN_DIR"):
        parser.error("AI_RUN_DIR is required")
    try:
        from qwen35_moe_dflash.canonical_target_export import canonicalize_target_export
        last = [None]
        def progress(row):
            now = time.monotonic()
            if last[0] is None or now - last[0] >= 10 or row["completed"] == row["total"]:
                print(f"canonical-target: {'hashed' if row['payload_hashed'] else 'stat'} "
                      f"{row['completed']}/{row['total']} files; {row['basename']}", file=sys.stderr, flush=True)
                last[0] = now
        result = canonicalize_target_export(args.air,
            args.output or Path(os.environ["AI_RUN_DIR"]) / "target-pytorch/canonical-export",
            torchair_pb2=args.torchair_pb2, discover_schema=args.discover_torchair_schema,
            sequence_length=args.sequence_length, reference_root=args.reference_root,
            hash_weights=args.hash_weights, resume=args.resume, progress=progress)
        print(json.dumps({key: result.get(key) for key in ("status", "output", "original_air", "air", "inventory",
            "weight_identity", "unique_reference_count", "total_weight_bytes", "feature_output_index",
            "original_preservation", "validation", "error")}, ensure_ascii=False))
        return 0 if result["status"] == "EXPORT_READY" else 1
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

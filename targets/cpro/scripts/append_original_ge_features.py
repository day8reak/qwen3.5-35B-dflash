#!/usr/bin/env python3
"""Append real source-anchored features; preserve original target and weights."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", required=True)
    parser.add_argument("--torchair-pb2", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root")
    source.add_argument("--source-receipt")
    parser.add_argument("--gamma-template", required=True, help="exact FileConstant reference with one {layer}")
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--ge-root", default=os.environ.get("CANN_ROOT"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output", help="new directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    if not os.environ.get("AI_RUN_DIR"):
        parser.error("AI_RUN_DIR is required")
    try:
        from qwen35_moe_dflash.original_ge_append import append_original_ge_features
        result = append_original_ge_features(args.air, args.output or Path(os.environ["AI_RUN_DIR"]) / "target-ge/features",
            torchair_pb2=args.torchair_pb2, gamma_template=args.gamma_template, sequence_length=args.sequence_length,
            source_root=args.source_root, source_receipt_path=args.source_receipt, ge_root=args.ge_root,
            cxx=args.cxx, cxx11_abi=args.cxx11_abi, timeout_seconds=args.timeout_seconds)
        compact = {key: result.get(key) for key in ("status", "stage", "output", "air", "feature_output",
            "original_preservation", "compilation_input", "model_compilation", "device_execution", "error")}
        plan = result.get("tap_plan", {})
        compact["tap_plan"] = {key: plan.get(key) for key in ("status", "reason", "pending", "taps", "anchor_count")}
        for key in ("append_validation", "roundtrip_validation", "ge_roundtrip"):
            if key in result:
                compact[key] = result[key]
        print(json.dumps(compact, ensure_ascii=False))
        return 0 if result["status"] == "FEATURE_AIR_EXPORTED" else 1
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

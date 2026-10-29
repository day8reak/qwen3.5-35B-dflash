#!/usr/bin/env python3
"""Probe isolated GE AIR LoadFromFile/SaveToFile; do not edit graphs or run models."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", required=True, help="original AIR, read only and copied alone to the new workspace")
    parser.add_argument("--torchair-pb2", required=True, help="explicit generated pb2 used only for AST descriptor extraction")
    parser.add_argument("--ge-root", default=os.environ.get("CANN_ROOT"), help="declared matching host GE headers/libgraph root")
    parser.add_argument("--output", help="new probe directory below AI_RUN_DIR")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--trace-files", action="store_true", help="retain strace file syscalls for independent review")
    args = parser.parse_args(argv)
    run = os.environ.get("AI_RUN_DIR")
    if not run:
        parser.error("AI_RUN_DIR is required")
    try:
        from qwen35_moe_dflash.original_air_roundtrip import probe_original_air_roundtrip
        result = probe_original_air_roundtrip(args.air, args.output or Path(run) / "original-air-roundtrip",
            ge_root=args.ge_root, torchair_pb2=args.torchair_pb2, cxx=args.cxx,
            cxx11_abi=args.cxx11_abi, timeout_seconds=args.timeout_seconds, trace_files=args.trace_files)
        print(json.dumps({key: result.get(key) for key in
            ("status", "output", "stage", "error", "native", "fidelity", "source_preservation", "feature_extension_gate")}))
        return 0 if result["status"] == "ROUNDTRIP_METADATA_PASSED" else 1
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


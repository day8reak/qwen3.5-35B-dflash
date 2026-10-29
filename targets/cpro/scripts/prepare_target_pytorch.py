#!/usr/bin/env python3
"""Prepare the original-machine PyTorch/TorchAir route without exporting AIR."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="original private Cpro export project on the weight machine")
    parser.add_argument("--config", required=True, help="original YAML with all four original resources")
    parser.add_argument("--output", help="new small preparation directory below AI_RUN_DIR")
    parser.add_argument("--sequence-length", type=int, default=55, help="original export shape; default 55")
    args = parser.parse_args(argv)
    run = os.environ.get("AI_RUN_DIR")
    if not run:
        parser.error("AI_RUN_DIR is required")
    try:
        from qwen35_moe_dflash.original_target_origin import prepare_target_pytorch
        result = prepare_target_pytorch(args.source_root, args.config,
            args.output or Path(run) / "target-pytorch", sequence_length=args.sequence_length)
        print(json.dumps({key: result.get(key) for key in
            ("status", "export", "output", "on_origin_export_script", "stage", "error")}, ensure_ascii=False))
        return 0 if result["status"] == "SOURCE_READY" else 1
    except Exception as error:
        print(json.dumps({"status": "SOURCE_BLOCKED", "export": "EXPORT_NOT_RUN",
                          "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


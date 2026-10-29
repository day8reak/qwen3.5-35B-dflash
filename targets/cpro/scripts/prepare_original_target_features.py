#!/usr/bin/env python3
"""Prepare a derived original Cpro export source with real eight-layer features."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))


def main():
    from qwen35_moe_dflash.original_target_source import prepare_original_target_source

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="original private PyTorch export project")
    parser.add_argument("--config", help="original YAML; copied unchanged; weight paths are not changed")
    parser.add_argument("--output", help="small derived source directory below AI_RUN_DIR")
    parser.add_argument("--sequence-length", type=int, default=55, help="original export shape, default 55; compilation uses 64/1 gears")
    args = parser.parse_args()
    run = os.environ.get("AI_RUN_DIR")
    if not run:
        parser.error("AI_RUN_DIR is required")
    try:
        result = prepare_original_target_source(
            args.source_root, args.output or Path(run) / "original-target-features",
            config=args.config, sequence_length=args.sequence_length)
    except (OSError, SyntaxError, ValueError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "target_layers", "original_output_count", "output_count", "feature_layer_ids",
        "feature_shape", "feature_dtype", "export_script", "expected_air", "scope")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

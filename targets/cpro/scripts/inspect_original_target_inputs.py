#!/usr/bin/env python3
"""Print original target control values as JSON; no compilation, HDC or writes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def main(argv=None):
    from qwen35_moe_dflash.original_target_inputs import inspect_original_target_inputs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="existing chunksize_64_data or one-token input directory")
    parser.add_argument("--sequence-length", type=int, choices=(64, 1), default=64)
    parser.add_argument("--skip-mask", action="store_true", help="read only the seven small control files")
    args = parser.parse_args(argv)
    try:
        report = inspect_original_target_inputs(args.input_dir, sequence_length=args.sequence_length,
                                                include_mask=not args.skip_mask)
    except (OSError, ValueError) as error:
        report = {"status": "FAILED", "purpose": "NOT_A_CONTROL_GENERATOR",
                  "device_execution": "NOT_EXECUTED", "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] == "INSPECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

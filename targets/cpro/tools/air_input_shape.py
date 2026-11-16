#!/usr/bin/env python3
"""Print a static OMG --input_shape value from actual root AIR Data/RefData nodes."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))

from qwen35_moe_dflash.toolchain.air_input_shape import inspect_input_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", required=True, help="raw TorchAir/GE export.air (ModelDef or GraphDef)")
    parser.add_argument("--torchair-pb2", help="optional exporter GE pb2 with a literal descriptor; never executed")
    parser.add_argument("--report", help="optional JSON file listing the source inputs")
    args = parser.parse_args()
    try:
        report = inspect_input_shape(args.air, torchair_pb2=args.torchair_pb2)
        if args.report:
            dest = Path(args.report).expanduser()
            # A typo must not overwrite the source AIR or a previously saved report.
            with dest.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(f"air-input-shape: {exc}", file=sys.stderr)
        return 1
    print(f"air-input-shape: {report['input_count']} root inputs {report['input_types']}", file=sys.stderr)
    # Only the argument value goes to stdout, so shell command substitution is safe.
    print(report["input_shape"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

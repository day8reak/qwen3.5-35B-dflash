#!/usr/bin/env python3
"""Request SIGTERM for one exact job's OMG tree, preserving its Python driver."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="current compile_full_target.py job below AI_RUN_DIR")
    parser.add_argument("--apply", action="store_true", help="send only SIGTERM using pinned Linux process handles")
    parser.add_argument("--output", help="new stop-receipt directory outside job but below AI_RUN_DIR; required for apply")
    args = parser.parse_args(argv)
    if args.apply != bool(args.output):
        parser.error("--apply and --output must be supplied together; default plan writes nothing")
    from qwen35_moe_dflash.target_compile_stop import stop_target_compile
    report = stop_target_compile(args.job, apply=args.apply, output=args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] in {"PLAN_READY", "STOP_REQUESTED", "NOT_RUNNING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

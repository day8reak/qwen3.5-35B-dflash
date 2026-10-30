#!/usr/bin/env python3
"""Stop only an exact already-paused compiler whose owned job was deleted."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int, help="observed exact OMG PID; identity is revalidated")
    parser.add_argument("--job", required=True, help="already absent old job directly under AI_RUN_DIR")
    parser.add_argument("--apply", action="store_true", help="queue TERM then CONT for the exact stopped tree and its bound obsolete driver")
    parser.add_argument("--legacy-stopped-tree", action="store_true", help="explicit numeric-PID mode with held proc directories; residual PID race remains")
    parser.add_argument("--output", help="new receipt directory under AI_RUN_DIR, required with apply")
    args = parser.parse_args(argv)
    if args.apply != bool(args.output):
        parser.error("--apply and --output must be supplied together; default plan writes nothing")
    from qwen35_moe_dflash.deleted_target_compile_stop import stop_deleted_target_compile
    report = stop_deleted_target_compile(args.pid, args.job, apply=args.apply, output=args.output,
                                        legacy_stopped_tree=args.legacy_stopped_tree)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] in {"PLAN_READY", "STOP_REQUESTED", "NOT_RUNNING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

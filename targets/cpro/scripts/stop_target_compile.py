#!/usr/bin/env python3
"""Stop one exact OMG tree; continue stopped, bound processes for termination/audit."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="current compile_full_target.py job below AI_RUN_DIR")
    parser.add_argument("--apply", action="store_true", help="request SIGTERM; stopped bound targets may also receive SIGCONT")
    parser.add_argument("--output", help="new stop-receipt directory outside job but below AI_RUN_DIR; required for apply")
    parser.add_argument("--legacy-stopped-tree", action="store_true", help="explicit numeric-PID mode only for an already paused exact tree; continue its bound paused driver, with a residual PID race")
    parser.add_argument("--continue-driver", action="store_true", help="also continue an exactly bound paused driver; never send it SIGTERM")
    args = parser.parse_args(argv)
    if args.apply != bool(args.output):
        parser.error("--apply and --output must be supplied together; default plan writes nothing")
    from qwen35_moe_dflash.target_compile_stop import stop_target_compile
    report = stop_target_compile(args.job, apply=args.apply, output=args.output,
        legacy_stopped_tree=args.legacy_stopped_tree, continue_driver=args.continue_driver)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] in {"PLAN_READY", "STOP_REQUESTED", "NOT_RUNNING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

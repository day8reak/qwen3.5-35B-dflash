#!/usr/bin/env python3
"""Prepare/compile/compare installed-DDK six-input GDN; never accesses a board."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))
from qwen35_moe_dflash.target import gdn_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="probe directory below AI_RUN_DIR")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true", help="prepare four fixtures, golden, plans and two GE sources")
    mode.add_argument("--compile", action="store_true", help="compile existing prepared sources using selected GE/DDK")
    mode.add_argument("--compare", metavar="RETRIEVED_ROOT", help="read per-case runner outputs; print metrics without PASS")
    mode.add_argument("--runner-commands", action="store_true", help="print exact runner argv templates; no execution")
    parser.add_argument("--report", help="optional comparison report below AI_RUN_DIR; never overwrites")
    parser.add_argument("--ge-root", default=os.environ.get("CANN_ROOT"))
    parser.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT"))
    parser.add_argument("--ddk-id", default=os.environ.get("CPRO_DDK_ID"))
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or (args.report and not args.compare):
        parser.error("timeout must be positive; --report requires --compare")
    try:
        if args.report and Path(args.report).exists():
            raise FileExistsError(args.report)
        if args.plan_only:
            prepared = gdn_probe.prepare(args.output)
            result = {key: prepared[key] for key in (
                "status", "model_compilation", "device_execution", "numerical_validation", "scope")}
            result.update(models=len(prepared["models"]), cases=len(prepared["cases"]),
                          manifest=str(Path(args.output).resolve() / "manifest.json"))
        elif args.compile:
            result = gdn_probe.compile_prepared(args.output, ge_root=args.ge_root, ddk_root=args.ddk_root,
                ddk_id=args.ddk_id, cxx=args.cxx, cxx11_abi=args.cxx11_abi, timeout_seconds=args.timeout_seconds)
        elif args.runner_commands:
            result = gdn_probe.runner_commands(args.output)
        else:
            result = gdn_probe.compare(args.output, args.compare)
            if args.report:
                gdn_probe.write_json(args.report, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get("status") in ("FAILED", "PARTIAL") else 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}",
                          "device_execution": "NOT_RUN_BY_THIS_TOOL"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

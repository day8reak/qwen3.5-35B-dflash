#!/usr/bin/env python3
"""Prepare/compile/compare whole-cache CacheUpdate diagnostics; no board access."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))
from qwen35_moe_dflash.target import cache_probe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="probe directory below AI_RUN_DIR")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--compile", action="store_true")
    mode.add_argument("--compare", metavar="RETRIEVED_ROOT")
    mode.add_argument("--runner-commands", action="store_true")
    parser.add_argument("--report", help="optional new comparison report below AI_RUN_DIR")
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
        if args.report and Path(args.report).exists(): raise FileExistsError(args.report)
        if args.plan_only:
            prepared = cache_probe.prepare(args.output)
            result = {key: prepared[key] for key in ("status", "scope", "model_compilation", "device_execution", "numerical_validation")}
            result.update(models=len(prepared["models"]), cases=len(prepared["cases"]),
                          manifest=str(Path(args.output).resolve() / "manifest.json"))
        elif args.compile:
            result = cache_probe.compile_prepared(args.output, ge_root=args.ge_root, ddk_root=args.ddk_root,
                ddk_id=args.ddk_id, cxx=args.cxx, cxx11_abi=args.cxx11_abi, timeout_seconds=args.timeout_seconds)
        elif args.runner_commands:
            result = cache_probe.runner_commands(args.output)
        else:
            result = cache_probe.compare(args.output, args.compare)
            if args.report: cache_probe.write_json(args.report, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get("status") in ("FAILED", "PARTIAL", "NUMERICAL_MISMATCH") else 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}",
                          "device_execution": "NOT_RUN_BY_THIS_TOOL"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

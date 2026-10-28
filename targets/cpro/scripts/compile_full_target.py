#!/usr/bin/env python3
"""Compile the original target AIR using the supplied Cpro 40-block 64/1 profile."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))


def _print_result(result, *, result_path=None, log_paths=()):
    """Show the tool's diagnostics without repeating the entire input-shape argv.

    The on-disk result retains the full exception. This is only a bounded
    console presentation, and does not infer the cause of a compiler failure.
    """
    from qwen35_moe_dflash.tool_logs import log_excerpt

    console = dict(result)
    if result.get("status") == "FAILED":
        if "returncode" in result:
            console["error"] = f"OMG exited with status {result['returncode']}"
        elif "timeout_seconds" in result:
            console["error"] = f"OMG timed out after {result['timeout_seconds']} seconds"
        elif len(str(result.get("error", ""))) > 512:
            console["error"] = str(result["error"])[:512] + " [full exception in result file]"
    if result_path is not None:
        console["result_json"] = str(result_path)
    print(json.dumps(console, ensure_ascii=False), flush=True)
    if result.get("status") == "FAILED":
        for path in log_paths:
            excerpt = log_excerpt(path)
            if excerpt:
                print(f"full-target: compiler diagnostics from {path}", flush=True)
                print(excerpt, flush=True)
                break


def main():
    from qwen35_moe_dflash.existing_air_profile import (
        EXISTING_AIR_PROFILE_NAMES, get_existing_air_profile,
    )
    from qwen35_moe_dflash.existing_air import INPUT_LAYOUTS, compile_existing_air

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", required=True, help="original exported AIR; never modified")
    parser.add_argument("--input-layout", choices=INPUT_LAYOUTS, default="export-directory",
                        help="relative weight paths: default retains original export folder prefix; "
                             "air-directory stages data directly beside AIR")
    parser.add_argument("--profile", choices=EXISTING_AIR_PROFILE_NAMES,
                        default="cpro-40block-64-1", help="user-provided input names and dynamic shapes")
    parser.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH"))
    parser.add_argument("--ddk-id", default=os.environ.get("CPRO_DDK_ID"))
    parser.add_argument("--tbe-diagnosis", help="reuse an existing check-tbe-platform --diagnose directory")
    parser.add_argument("--omg", help="optional actual ELF binary in selected DDK; default tools/tools_omg/master/omg")
    parser.add_argument("--output", help="new job directory under AI_RUN_DIR")
    parser.add_argument("--timeout-seconds", type=int, default=7200, help="OMG compilation timeout")
    parser.add_argument("--skip-inspection", action="store_true", help="skip post-compile OMC-to-JSON inspection")
    args = parser.parse_args()

    from qwen35_moe_dflash.artifacts import output_path, save_json

    root = None
    owned_root = False
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-target-job",
              "status": "PREPARING", "device_execution": "NOT_RUN",
              "numerical_validation": "NOT_RUN", "device_ready": False}
    try:
        run = os.environ.get("AI_RUN_DIR")
        if not run or not args.ddk_root or not args.ddk_id:
            raise ValueError("AI_RUN_DIR, CPRO_DDK_ROOT/DDK_PATH and CPRO_DDK_ID are required; source the DDK and project env.sh")
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")
        air = Path(args.air).resolve()
        if not air.is_file():
            raise ValueError(f"AIR not found: {air}")
        root = output_path(args.output or Path(run) / ("full-target-compile-" + uuid.uuid4().hex[:12]))
        root.mkdir(parents=True, exist_ok=False)
        owned_root = True
        profile = get_existing_air_profile(args.profile)
        save_json(root / "compile-profile.json", profile)
        result.update(output=str(root), air=str(air), profile="compile-profile.json",
                      input_layout=args.input_layout,
                      compilation="compilation/report.json")
        print(f"full-target: output {root}", flush=True)
        print("full-target: original AIR; 64/1 dynamic shapes; external weights; kirinx90", flush=True)

        def progress(row):
            if (row["stage"] != "input-snapshot" or "completed" not in row
                    or row["completed"] in (1, row.get("total")) or row["completed"] % 25 == 0):
                print("full-target: " + row["message"], flush=True)

        compiled = compile_existing_air(
            air, root / "compilation", args.ddk_root, ddk_id=args.ddk_id,
            tbe_diagnosis=args.tbe_diagnosis, timeout_seconds=args.timeout_seconds,
            omg_path=args.omg, input_shape=profile["input_shape"], dynamic_dims=profile["dynamic_dims"],
            limit_partition_size=profile["limit_partition_size"],
            save_weights_as_external_data=profile["save_weights_as_external_data"], progress=progress,
            input_layout=args.input_layout,
        )
        result["status"] = compiled["status"]
        result["model_compilation"] = compiled["model_compilation"]
        result["compile_log"] = str(root / "compilation/compile.log")
        if compiled["status"] == "COMPILED":
            result["omc"] = str(root / "compilation" / compiled["omc"]["path"])
            print("full-target: OMC compiled; retained generated external files", flush=True)
            if not args.skip_inspection:
                print("full-target: inspecting compiled graph interface (no board execution)", flush=True)
                try:
                    from qwen35_moe_dflash.existing_omc_inspect import inspect_existing_omc
                    inspected = inspect_existing_omc(root / "compilation", root / "interface", timeout_seconds=120)
                    result["interface_inspection"] = inspected["status"]
                    result["interface_report"] = str(root / "interface/report.txt")
                except Exception as error:
                    result["interface_inspection"] = "UNAVAILABLE"
                    result["interface_error"] = f"{type(error).__name__}: {error}"
            else:
                result["interface_inspection"] = "NOT_RUN"
            result["next"] = "Confirm original graph I/O, external weights, dynamic gear and RefData state contract before board execution"
        else:
            result["error"] = compiled.get("error", "compilation failed")
            result["stage"] = compiled.get("stage", "compilation")
            for field in ("returncode", "timeout_seconds"):
                if field in compiled:
                    result[field] = compiled[field]
            result["next"] = "Inspect compilation/report.txt and retained compiler diagnostics"
        save_json(root / "result.json", result)
        _print_result(result, result_path=root / "result.json",
                      log_paths=(root / "compilation/compile.log", root / "compilation/help.log"))
        return 0 if result["status"] == "COMPILED" else 1
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        if owned_root:
            save_json(root / "failure.json", result)
        _print_result(result, result_path=root / "failure.json" if owned_root else None)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

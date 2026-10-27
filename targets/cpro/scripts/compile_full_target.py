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


def main():
    from qwen35_moe_dflash.existing_air_profile import (
        EXISTING_AIR_PROFILE_NAMES, get_existing_air_profile,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", required=True, help="original exported AIR; never modified")
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
    from qwen35_moe_dflash.existing_air import compile_existing_air

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
            result["next"] = "Inspect compilation/report.txt and retained compiler diagnostics"
        save_json(root / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["status"] == "COMPILED" else 1
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        if owned_root:
            save_json(root / "failure.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

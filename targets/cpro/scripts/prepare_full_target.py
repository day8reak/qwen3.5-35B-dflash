#!/usr/bin/env python3
"""One entry for full target weight preparation and selected-DDK interface evidence."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-weights", default=os.environ.get("QWEN35_TARGET_WEIGHTS"),
                        help="original locked target checkpoint; otherwise use AI_RUN_DIR/weights/Qwen3.5-35B-A3B")
    parser.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH"))
    parser.add_argument("--ge-root", default=os.environ.get("CANN_ROOT"))
    parser.add_argument("--hiai-include", default=os.environ.get("HIAI_INCLUDE_DIR"))
    parser.add_argument("--output", help="new directory under AI_RUN_DIR; defaults to full-target-<unique ID>")
    parser.add_argument("--resume", help="previous full-target job (or its weights subdirectory); verify and reuse completed tensors")
    parser.add_argument("--inspect-only", action="store_true", help="only full weight plan and selected-DDK source evidence; no weight conversion")
    args = parser.parse_args()
    from qwen35_moe_dflash.common.artifacts import output_path, save_json
    from qwen35_moe_dflash.graph.quant_interface import inspect_quant_interfaces
    from qwen35_moe_dflash.graph.quant_plan import target_quantization_plan
    from qwen35_moe_dflash.graph.quant_prepare import prepare_target_weights

    root = None
    owned_root = False
    result = {"schema_version": 1, "artifact_kind": "qwen35-full-target-preparation",
              "status": "PREPARING", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
              "device_ready": False}
    stage = "configuration"
    try:
        run = os.environ.get("AI_RUN_DIR")
        if not run:
            raise ValueError("AI_RUN_DIR is required; source targets/cpro/scripts/env.sh first")
        if not args.ddk_root or not Path(args.ddk_root).is_dir():
            raise ValueError("selected CPRO_DDK_ROOT/DDK_PATH is missing; source the DDK and project environment")
        checkpoint = args.target_weights or str(Path(run) / "weights/Qwen3.5-35B-A3B")
        old = None
        if args.resume:
            old = Path(args.resume).resolve()
            if (old / "weights/plan.json").is_file():
                old = old / "weights"
            if args.target_weights is None and (old / "plan.json").is_file():
                checkpoint = json.loads((old / "plan.json").read_text())["checkpoint_root"]
        root = output_path(args.output or Path(run) / ("full-target-" + uuid.uuid4().hex[:12]))
        root.mkdir(parents=True, exist_ok=False)
        owned_root = True
        stage = "target-plan"
        print("full-target: planning all 40 layers and real target weight shapes", flush=True)
        plan = target_quantization_plan()
        save_json(root / "quantization-plan.json", plan)
        result.update(output=str(root), checkpoint_root=str(Path(checkpoint).resolve()),
                      weight_plan="quantization-plan.json", summary=plan["summary"])
        stage = "installed-interfaces"
        print("full-target: reading selected-DDK quantization interfaces (no compilation)", flush=True)
        inspect_quant_interfaces(args.ddk_root, root / "interfaces", ge_root=args.ge_root,
                                 hiai_include=args.hiai_include)
        result["interface_report"] = "interfaces/report.txt"
        print((root / "interfaces/report.txt").read_text(), end="", flush=True)
        if not args.inspect_only:
            stage = "target-weights"
            print("full-target: streaming BF16 target to logical INT4/INT8/FP16 weight cache", flush=True)

            def progress(row):
                if row["completed"] == 1 or row["completed"] % 25 == 0 or row["completed"] == row["total"]:
                    print(json.dumps(row, ensure_ascii=False), flush=True)

            weights = prepare_target_weights(checkpoint, root / "weights", resume=old, progress=progress)
            result.update(status=weights["status"], weights="weights/manifest.json")
        else:
            result["status"] = "INSPECTED"
        result["next"] = "Resolve I4ZN/QPARAM and quantized GE ABI from selected-DDK or original AIR evidence; reuse this weight cache for packing"
        save_json(root / "report.json", result)
        lines = ["Full 40-layer target preparation", f"status={result['status']}",
                 "Logical quantized cache is not I4ZN/QPARAM and is not a deployable OMC.",
                 "No OMG compilation, platform matrix, or board execution.",
                 "Weight plan: quantization-plan.json", "DDK interface evidence: interfaces/report.txt"]
        if "weights" in result:
            lines.append("Completed weight cache: weights/manifest.json")
        if "gdn_report" in result:
            lines.append("Retained GDN summary: gdn-retained/report.txt")
        (root / "report.txt").write_text("\n".join(lines) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        result.update(status="FAILED", stage=stage, error=f"{type(error).__name__}: {error}")
        if owned_root:
            save_json(root / "failure.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

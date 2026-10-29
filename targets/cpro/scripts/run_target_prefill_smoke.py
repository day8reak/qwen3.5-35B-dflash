#!/usr/bin/env python3
"""Plan one exact original prefill64; --apply invokes the existing board tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    from qwen35_moe_dflash.artifacts import output_path, save_json
    from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
    from qwen35_moe_dflash.target_prefill_smoke_plan import prepare_prefill_smoke_plan, validate_prefill_smoke_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="successful full original static64 target job")
    parser.add_argument("--runtime-probe", required=True, help="completed restore-only runtime descriptor evidence")
    parser.add_argument("--input-dir", required=True, help="original chunksize64 input files; never synthesized or modified")
    parser.add_argument("--output", required=True, help="new evidence directory under AI_RUN_DIR")
    parser.add_argument("--apply", action="store_true", help="after the exact first-prefill plan passes, invoke the existing HIAI tool once")
    parser.add_argument("--board-model-root", default="/data/local/tmp/qwen3_5_moe")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--lock-wait-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    out = None
    owned = False
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-prefill-smoke-job",
        "status": "BLOCKED", "mode": "APPLY" if args.apply else "PLAN_ONLY",
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN_NO_REFERENCE",
        "state_continuity": "NOT_RUN", "state_output_layer_identity": "UNVERIFIED"}
    try:
        out = output_path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        owned = True
        job = Path(args.job).resolve()
        plan = prepare_prefill_smoke_plan(job / "compilation", args.runtime_probe, args.input_dir)
        validate_prefill_smoke_plan(plan)
        save_json(out / "plan.json", plan)
        result.update(status="PREFILL_SMOKE_PLANNED", plan=str(out / "plan.json"),
            input_count=len(plan["inputs"]), output_count=len(plan["outputs"]),
            execution_scope=plan["execution_scope"], feature_output_evidence=plan["feature_output_evidence"],
            next="Run a separate new output with --apply for this one prefill only; state feedback and draft input mapping remain unestablished by this plan")
        if args.apply:
            from qwen35_moe_dflash.original_target_board import execute_target_tool
            observed = execute_target_tool(job / "compilation", job / "target-contract/report.json", out / "board",
                profile=get_existing_air_profile("cpro-40block-prefill-64"), gear=64,
                host_input_dir=args.input_dir, target_features=True, runtime_prefill_probe=args.runtime_probe,
                tool=plan["tool_binding"]["path"], board_model_root=args.board_model_root,
                timeout_seconds=args.timeout_seconds, lock_wait_seconds=args.lock_wait_seconds)
            result.update(observed, artifact_kind="qwen35-target-prefill-smoke-job", mode="APPLY",
                board_result=str(out / "board/result.json"), plan=str(out / "plan.json"))
            if observed.get("stage") == "wait-board-model-lock" and observed.get("device_execution") == "NOT_RUN":
                result["next"] = "Inspect the retained lock request and current board owner before a new-output retry; no model execution or automatic lock cleanup occurred"
            elif observed.get("device_execution") == "NOT_RUN":
                result["next"] = "Inspect board/result.json and retained HDC calls at the reported failure stage before retrying; model inference was not reached"
            else:
                result["next"] = "Retain all actual tool outputs and PID/logs; map real filenames to runtime indices before using features or feeding back80states"
    except Exception as error:
        result.update(status="BLOCKED", error=f"{type(error).__name__}: {error}")
    if owned:
        save_json(out / "result.json", result)
    shown = {key: result[key] for key in ("status", "stage", "mode", "device_execution", "model_compilation",
        "numerical_validation", "state_continuity", "execution_scope", "input_count", "output_count",
        "tool_process_id", "remote_exitcode", "lock_state", "error", "evidence_error", "plan", "board_result", "next") if key in result}
    if owned:
        shown["result"] = str(out / "result.json")
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
    if args.apply and result["status"] != "EXECUTED_UNVALIDATED" and out is not None:
        from qwen35_moe_dflash.tool_logs import log_excerpt
        for name in ("tool.stderr", "tool.stdout"):
            path = out / "board" / name
            if path.is_file():
                selected = log_excerpt(path)
                if selected:
                    print(f"target-prefill-smoke: {name}\n{selected}", flush=True)
    return 0 if result["status"] in {"PREFILL_SMOKE_PLANNED", "EXECUTED_UNVALIDATED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

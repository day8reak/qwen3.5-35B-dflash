#!/usr/bin/env python3
"""Plan or run one original decode1 using the supplied input directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="completed original static decode1 compilation")
    parser.add_argument("--runtime-probe", required=True, help="named runtime descriptors for this exact decode1 model")
    parser.add_argument("--input-dir", required=True, help="user-supplied decode inputs; files remain unchanged")
    parser.add_argument("--output", required=True, help="new evidence directory below AI_RUN_DIR")
    parser.add_argument("--apply", action="store_true", help="after input checks, invoke the original V2 tool once")
    parser.add_argument("--board-model-root", default="/data/local/tmp/qwen3_5_moe")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--lock-wait-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.artifacts import output_path, save_json
    from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
    from qwen35_moe_dflash.target_decode_smoke_plan import prepare_decode_smoke_plan, validate_decode_smoke_plan
    out, owned = None, False
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-decode-smoke-job",
        "status": "BLOCKED", "mode": "APPLY" if args.apply else "PLAN_ONLY",
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "numerical_validation": "NOT_RUN_NO_REFERENCE", "state_continuity": "NOT_VALIDATED",
        "execution_scope": "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_NO_PREFILL_FEEDBACK"}
    try:
        out = output_path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        owned = True
        job = Path(args.job).resolve()
        plan = prepare_decode_smoke_plan(job / "compilation", args.runtime_probe, args.input_dir)
        validate_decode_smoke_plan(plan)
        save_json(out / "plan.json", plan)
        result.update(status="DECODE_SMOKE_PLANNED", plan=str(out / "plan.json"),
            input_count=len(plan["inputs"]), output_count=len(plan["outputs"]),
            control_observations=plan.get("control_observations", []),
            next="After DECODE_SMOKE_PLANNED, use a separate new output with --apply for one user-input decode only")
        if args.apply:
            from qwen35_moe_dflash.original_target_board import execute_target_tool
            observed = execute_target_tool(job / "compilation", job / "target-contract/report.json", out / "board",
                profile=get_existing_air_profile("cpro-40block-decode-1"), gear=1,
                host_input_dir=args.input_dir, target_features=True, runtime_decode_probe=args.runtime_probe,
                tool=plan["tool_binding"]["path"], board_model_root=args.board_model_root,
                timeout_seconds=args.timeout_seconds, lock_wait_seconds=args.lock_wait_seconds)
            result.update(observed, artifact_kind="qwen35-target-decode-smoke-job", mode="APPLY",
                board_result=str(out / "board/result.json"), plan=str(out / "plan.json"))
            result["next"] = (
                "Retain actual output files and tool logs; inspect this standalone decode result before establishing prefill-state feedback or numerical acceptance"
                if result.get("status") == "EXECUTED_UNVALIDATED" else
                "Inspect the recorded stage and retained logs/lock ownership before any new-output retry; no automatic retry")
    except Exception as error:
        result.update(status="BLOCKED", error=f"{type(error).__name__}: {error}")
    if owned:
        save_json(out / "result.json", result)
    shown = {key: result[key] for key in ("status", "stage", "mode", "device_execution", "model_compilation",
        "numerical_validation", "state_continuity", "execution_scope", "input_count", "output_count",
        "tool_process_id", "remote_exitcode", "lock_state", "error", "evidence_error", "plan", "board_result", "next") if key in result}
    if not args.apply:
        shown["provided_control_values"] = {Path(row["path"]).name: {"dtype": row["dtype"], "values": row["values"]}
            for row in result.get("control_observations", [])}
    if owned:
        shown["result"] = str(out / "result.json")
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
    if args.apply and result.get("status") != "EXECUTED_UNVALIDATED" and out is not None:
        from qwen35_moe_dflash.tool_logs import log_excerpt
        for name in ("tool.stderr", "tool.stdout"):
            path = out / "board" / name
            if path.is_file():
                excerpt = log_excerpt(path)
                if excerpt:
                    print(f"target-decode-smoke: {name}\n{excerpt}", flush=True)
    return 0 if result.get("status") in {"DECODE_SMOKE_PLANNED", "EXECUTED_UNVALIDATED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

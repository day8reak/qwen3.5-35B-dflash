#!/usr/bin/env python3
"""Run one prepared original-target decode with all80 parent state files."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True, help="new directory below AI_RUN_DIR")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--board-model-root", default="/data/local/tmp/qwen3_5_moe")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--lock-wait-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_decode_continuation_run import run_decode_continuation
    try:
        result = run_decode_continuation(args.prepared, args.output, apply=args.apply,
            board_model_root=args.board_model_root, timeout_seconds=args.timeout_seconds,
            lock_wait_seconds=args.lock_wait_seconds)
    except Exception as error:
        result = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
    shown = {key: result[key] for key in ("status", "stage", "mode", "device_execution", "model_compilation",
        "numerical_validation", "state_continuity", "state_feedback", "execution_scope", "input_count", "output_count",
        "feedback_input_count", "bound_child_output_count", "tool_process_id", "remote_exitcode", "lock_state",
        "model_process_status", "output_retrieval_status",
        "error", "evidence_error", "source_error", "prepared", "plan", "board_result", "state_chain", "next") if key in result}
    if "output" in result:
        shown["result"] = str(Path(result["output"]) / "result.json")
    if "logits" in result:
        shown["next_token"] = result["logits"].get("argmax")
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["status"] in {"DECODE_CONTINUATION_PLANNED", "DECODE_CONTINUATION_EXECUTED_UNVALIDATED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

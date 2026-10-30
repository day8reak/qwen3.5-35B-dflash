#!/usr/bin/env python3
"""Run the complete original models in a bounded decode1 DFlash loop, without recompilation."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("decode-job", "runtime-probe", "state-lineage", "source-torchair-pb2", "draft-bundle",
                 "weight-table", "scale-table", "prompt-input-ids", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-proposals-per-round", type=int, default=2)
    parser.add_argument("--minimum-draft-rounds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--lock-wait-seconds", type=int, default=1800)
    parser.add_argument("--board-model-root", default="/data/local/tmp/qwen3_5_moe")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.serial_dflash_smoke import run_serial_dflash_smoke
    try:
        result = run_serial_dflash_smoke(**vars(args))
    except Exception as error:
        result = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
    shown = {key: result[key] for key in ("status", "stage", "mode", "device_execution", "model_compilation",
        "numerical_validation", "generated_token_ids", "counts", "counts_scope", "multi_round_observed", "acceptance_rate",
        "acceptance_rate_scope",
        "error", "plan", "loop", "next") if key in result}
    if "output" in result:
        shown["result"] = str(Path(result["output"]) / "result.json")
    if "native_executions" in result:
        shown["native_call_counts"] = {role: row.get("native_calls")
            for role, row in result["native_executions"].items()}
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["status"] in {"DFLASH_SERIAL_SMOKE_PLANNED", "DFLASH_SERIAL_LOOP_EXECUTED_UNVALIDATED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

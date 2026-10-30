#!/usr/bin/env python3
"""Check original selected embedding rows against an executed target prefix."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", required=True, help="completed explicitly bound target prefill output inspection")
    parser.add_argument("--config", required=True, help="original YAML containing embedding_weight_path and embedding_scale_path")
    parser.add_argument("--output", required=True, help="new evidence directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.original_embedding_probe import inspect_original_embedding_rows
    try:
        result = inspect_original_embedding_rows(args.inspection, args.config, args.output)
        keys = ("status", "stage", "output", "selected_next_token_id", "token_selection", "prefix_comparison",
            "tables", "missing_paths", "next_token_inputs", "observed_payload_reads", "model_compilation",
            "device_execution", "numerical_validation", "runtime_control_recipe", "state_feedback", "error", "next")
        shown = {key: result[key] for key in keys if key in result}
        shown["report"] = str(Path(args.output).resolve() / "report.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result["status"] == "ORIGINAL_EMBEDDING_ROWS_BOUND_NEXT_TOKEN_PREPARED" else 1
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

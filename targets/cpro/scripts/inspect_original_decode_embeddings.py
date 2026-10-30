#!/usr/bin/env python3
"""Compare an executed decode row with its original configured embedding tables."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", required=True, help="completed explicitly bound decode output inspection")
    parser.add_argument("--config-yaml", required=True, help="original YAML declaring the original INT8 and FP32 scale tables")
    parser.add_argument("--match-source-id-candidates", action="store_true",
        help="explicitly match at most 1024 original source IDs against both executed INT8 row and FP32 scale; no position guess")
    parser.add_argument("--output", required=True, help="new evidence directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.original_decode_embedding_probe import inspect_original_decode_embeddings
    try:
        result = inspect_original_decode_embeddings(args.inspection, args.config_yaml, args.output,
            match_source_id_candidates=args.match_source_id_candidates)
        keys = ("status", "stage", "output", "source_id_diagnosis", "selected_source_token_id", "selected_next_token_id", "token_selection",
            "source_row_comparison", "source_candidate_matching", "tables", "missing_paths", "next_token_inputs", "observed_payload_reads",
            "model_compilation", "device_execution", "numerical_validation", "logical_history",
            "runtime_control_recipe", "state_feedback", "error", "next")
        shown = {key: result[key] for key in keys if key in result}
        shown["report"] = str(Path(args.output).resolve() / "report.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result["status"] == "ORIGINAL_DECODE_EMBEDDING_ROWS_BOUND_NEXT_TOKEN_PREPARED" else 1
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

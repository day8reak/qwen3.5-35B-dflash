#!/usr/bin/env python3
"""Prepare one original same-model decode continuation without board execution."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    from qwen35_moe_dflash.target_decode_continuation import ASSUMPTION, prepare_decode_continuation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-lineage", required=True)
    parser.add_argument("--original-rows", required=True)
    parser.add_argument("--assumption", required=True, choices=[ASSUMPTION])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_decode_continuation(args.state_lineage, args.original_rows, args.output, assumption=args.assumption)
        keys = ("status", "output", "input_count", "output_count", "independent_state_file_count", "continuation_token_id",
            "next_absolute_position", "explicit_weight_copy_bytes", "device_execution", "state_continuity", "numerical_validation", "error")
        shown = {key: result[key] for key in keys if key in result}
        shown["manifest"] = str(Path(args.output).resolve() / "manifest.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result["status"] == "SAME_MODEL_DECODE_CONTINUATION_PREPARED" else 1
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare a separate decode trial with an explicit unverified control value."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="original user-supplied decode inputs; never modified")
    parser.add_argument("--output", required=True, help="new trial directory below AI_RUN_DIR")
    parser.add_argument("--experimental-conv-current-query-count", action="store_true",
        help="explicitly test INT32[128]=[1,0,...], assuming its first value is current query length; contract unverified")
    parser.add_argument("--experimental-single-token-controls", action="store_true",
        help="explicit trial: conv=[1,0,...], output_pos=0, effective_length=1, allQLen=1; original files preserved")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_decode_trial_inputs import prepare_decode_trial_inputs
    try:
        result = prepare_decode_trial_inputs(args.input_dir, args.output,
            experimental_conv_current_query_count=args.experimental_conv_current_query_count,
            experimental_single_token_controls=args.experimental_single_token_controls)
        shown = {key: result[key] for key in ("status", "input_dir", "assumption", "assumptions",
            "explicit_copy_bytes", "source_preservation", "device_execution", "model_compilation",
            "numerical_validation", "scope") if key in result}
        shown["manifest"] = str(Path(args.output).resolve() / "manifest.json")
        shown["next"] = "Run one independent decode with this input_dir and --trial-input-receipt pointing to this manifest; no correctness or state-continuity claim"
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result.get("status") == "TARGET_DECODE_TRIAL_INPUTS_PREPARED" else 1
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "device_execution": "NOT_RUN",
            "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

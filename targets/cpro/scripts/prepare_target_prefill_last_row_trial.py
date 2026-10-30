#!/usr/bin/env python3
"""Prepare independent prefill inputs with only the source-Python last row changed."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", required=True, help="retained successful original prefill execution")
    parser.add_argument("--named-runtime-probe", required=True, help="later same-model named metadata probe")
    parser.add_argument("--output", required=True, help="new independent trial directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_prefill_last_row_trial import prepare_prefill_last_row_trial
    try:
        result = prepare_prefill_last_row_trial(args.execution, args.named_runtime_probe, args.output)
        shown = {key: result[key] for key in ("status", "stage", "device_execution", "model_compilation",
            "input_dir", "target_input_count", "target_source_file_count", "auxiliary_source_file_count",
            "derived_file_count", "changed_target_source_file_count", "control_replacement", "error", "next") if key in result}
        shown["manifest"] = str(Path(args.output).absolute() / "manifest.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return int(result["status"] != "TARGET_PREFILL_LAST_ROW_TRIAL_INPUTS_PREPARED")
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

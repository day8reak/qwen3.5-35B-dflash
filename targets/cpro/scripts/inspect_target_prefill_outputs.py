#!/usr/bin/env python3
"""Hash-check retained target outputs and list files without guessing slot names."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", required=True, help="successful first-prefill smoke execution directory")
    parser.add_argument("--output", required=True, help="new evidence directory below AI_RUN_DIR")
    parser.add_argument("--draft-bundle", help="optional existing draft bundle; only small manifest.json is read")
    parser.add_argument("--input-dir", help="optional original chunksize64 input directory; only input_ids.bin is read")
    parser.add_argument("--full", action="store_true", help="also print all file rows, descriptor metadata and bounded tool logs")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_prefill_output_inventory import inspect_target_prefill_outputs
    try:
        report = inspect_target_prefill_outputs(args.execution, args.output,
            draft_bundle=args.draft_bundle, input_dir=args.input_dir)
        if args.full:
            shown = report
        else:
            shown = {key: report[key] for key in ("status", "error", "execution", "plan_output_count",
                "actual_retrieved_output_file_count", "raw_extent_multiset_check", "raw_extent_counts",
                "unclassified_files", "slot_binding", "state_layer_binding", "feature_selection", "bound_tensors",
                "draft_bundle", "original_input_ids", "numerical_validation", "scope") if key in report}
            if "logged_slot_mapping" in report:
                shown["logged_slot_mapping"] = {key: value for key, value in report["logged_slot_mapping"].items() if key != "entries"}
        shown["report"] = str(Path(report["output"]) / "report.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if report["status"] in {"OUTPUT_FILES_INSPECTED_SLOT_BINDING_UNVERIFIED", "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"} else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

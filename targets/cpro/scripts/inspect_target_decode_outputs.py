#!/usr/bin/env python3
"""Inspect retained decode1 outputs, explicit tool slots and experimental inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", required=True, help="successful independent decode1 smoke directory")
    parser.add_argument("--output", required=True, help="new evidence directory below AI_RUN_DIR")
    parser.add_argument("--skip-state-statistics", action="store_true", help="hash state outputs without describing values in all 80 state tensors")
    parser.add_argument("--full", action="store_true", help="print full verified file, log, state statistics and provenance records")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_decode_output_inventory import inspect_target_decode_outputs
    try:
        report = inspect_target_decode_outputs(args.execution, args.output,
            include_state_statistics=not args.skip_state_statistics)
        if args.full:
            shown = dict(report)
        else:
            shown = {key: report[key] for key in ("status", "error", "execution", "execution_scope", "plan_output_count",
                "actual_retrieved_output_file_count", "raw_extent_multiset_check", "raw_extent_counts", "slot_binding",
                "source_validation", "state_layer_binding", "state_feedback", "state_continuity", "bound_tensors",
                "numerical_validation", "scope") if key in report}
            if "logged_slot_mapping" in report:
                shown["logged_slot_mapping"] = {key: value for key, value in report["logged_slot_mapping"].items() if key != "entries"}
            if "state_statistics" in report:
                shown["state_statistics"] = {key: value for key, value in report["state_statistics"].items() if key != "entries"}
            if "trial_inputs" in report:
                trial = report["trial_inputs"]
                shown["trial_inputs"] = {key: value for key, value in trial.items() if key != "assumption"}
                if "assumption" in trial:
                    shown["trial_inputs"]["assumption"] = {key: value for key, value in trial["assumption"].items() if key != "control_replacements"}
                    shown["trial_inputs"]["assumption"]["control_replacements"] = [
                        {**{key: value for key, value in row.items() if key != "new_values"},
                         "new_values": row["new_values"] if len(row["new_values"]) <= 8 else row["new_values"][:4],
                         "new_value_count": len(row["new_values"])} for row in trial["assumption"].get("control_replacements", [])]
        shown["report"] = str(Path(report["output"]) / "report.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if report["status"] in {"DECODE_OUTPUT_FILES_INSPECTED_SLOT_BINDING_UNVERIFIED", "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"} else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

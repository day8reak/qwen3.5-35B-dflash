#!/usr/bin/env python3
"""Inspect retained state edges, logit selection and original embedding rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-job", required=True)
    parser.add_argument("--decode-job", required=True)
    parser.add_argument("--execution", required=True, help="successful retained target prefill execution")
    parser.add_argument("--output-inspection", required=True, help="completed prefill output file/slot inspection")
    parser.add_argument("--prefill-runtime-probe", required=True, help="named restore-only prefill descriptors")
    parser.add_argument("--decode-runtime-probe", required=True)
    parser.add_argument("--original-config", required=True, help="original model YAML; only embedding tables are sampled")
    parser.add_argument("--output", required=True, help="new directory below AI_RUN_DIR")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.artifacts import output_path, save_json
    try:
        out = output_path(args.output)
        out.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1

    def inspect(name, function):
        try:
            value = function(out / name)
        except Exception as error:
            value = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
            destination = out / name
            destination.mkdir(parents=True, exist_ok=True)
            if not (destination / "report.json").exists():
                save_json(destination / "report.json", value)
        shown = {key: value[key] for key in ("status", "error", "next", "missing_paths",
            "selection", "input_selection", "logit_selection", "output_pos", "next_token",
            "control", "source_preservation", "cross_model_state_identity", "feedback_ready",
            "selected_next_token_id", "greedy_observation",
            "next_token_inputs", "prefix_comparison", "observed_payload_reads") if key in value}
        if name == "states":
            for phase in ("prefill", "decode"):
                if phase in value:
                    shown[phase] = {key: {field: data for field, data in value[phase][key].items()
                        if field not in {"states", "updater_diagnostics"}} for key in
                        ("source_lineage", "compiled_lineage", "runtime_bindings") if key in value[phase]}
        shown["report"] = str(out / name / "report.json")
        return shown

    def states(destination):
        from qwen35_moe_dflash.target_state_lineage import inspect_target_state_lineage
        return inspect_target_state_lineage(args.prefill_job, args.decode_job, args.execution,
            args.decode_runtime_probe, destination, prefill_runtime_probe=args.prefill_runtime_probe)

    def selection(destination):
        from qwen35_moe_dflash.target_logit_selection import inspect_target_logit_selection
        return inspect_target_logit_selection(args.prefill_job, args.execution, destination)

    def embedding(destination):
        from qwen35_moe_dflash.original_embedding_probe import inspect_original_embedding_rows
        return inspect_original_embedding_rows(args.output_inspection, args.original_config, destination)

    reports = {name: inspect(name, function) for name, function in
        (("states", states), ("logit-selection", selection), ("embedding", embedding))}
    failed = any(row["status"] == "FAILED" for row in reports.values())
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-continuation-inspection",
        "status": "CONTINUATION_INSPECTION_INCOMPLETE" if failed else "CONTINUATION_INSPECTED",
        "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN", "state_feedback": "NOT_RUN",
        "numerical_validation": "NOT_RUN", "decode_execution_ready": False, "inspections": reports,
        "next": "Inspect the actual state, logit-index and original-table evidence; original decode control semantics remain a separate required gate"}
    save_json(out / "report.json", result)
    print(json.dumps({**result, "report": str(out / "report.json")}, ensure_ascii=False, allow_nan=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

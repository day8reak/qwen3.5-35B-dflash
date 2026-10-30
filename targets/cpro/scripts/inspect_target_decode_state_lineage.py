#!/usr/bin/env python3
"""Audit one decode model's state owners and previously retrieved output files."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="completed static decode1 compile job")
    parser.add_argument("--runtime-probe", required=True, help="saved named metadata probe for this same model")
    parser.add_argument("--output-inspection", required=True, help="successful inspect_target_decode_outputs output directory")
    parser.add_argument("--source-torchair-pb2", help="optional exact copy of the canonical receipt's producer pb2 literal")
    parser.add_argument("--output", required=True, help="new output below AI_RUN_DIR")
    parser.add_argument("--full", action="store_true", help="print all state mappings and graph evidence")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_decode_state_lineage import inspect_target_decode_state_lineage
    try:
        report = inspect_target_decode_state_lineage(args.job, args.runtime_probe, args.output_inspection, args.output,
            source_torchair_pb2=args.source_torchair_pb2)
        shown = report if args.full else {key: report[key] for key in ("status", "stage", "error",
            "device_execution", "model_compilation", "weight_payload_access", "state_feedback", "feedback_ready",
            "state_continuity", "numerical_validation", "same_model_state_mapping", "gap_summary", "next") if key in report}
        if not args.full:
            shown["graph_edge_policy"] = report["graph_edge_policy"]["policy"]
            decoded = report.get("decode", {})
            shown["counts"] = {"source_owners": decoded.get("source_lineage", {}).get("observed_state_owner_count", 0),
                "unique_source_owners": decoded.get("source_lineage", {}).get("unique_state_owner_count", 0),
                "compiled_identity_matches": decoded.get("compiled_lineage", {}).get("matched_count", 0),
                "runtime_name_matches": decoded.get("runtime_bindings", {}).get("matched_count", 0),
                "bound_state_files": decoded.get("state_file_mapping", {}).get("bound_count", 0)}
        shown["report"] = str(Path(args.output).resolve() / "report.json")
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 1 if report["status"] == "FAILED" else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

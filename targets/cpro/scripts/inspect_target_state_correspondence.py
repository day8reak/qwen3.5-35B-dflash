#!/usr/bin/env python3
"""Inspect cross-export state provenance and bounded weight-context candidates."""
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
    parser.add_argument("--prefill-source-torchair-pb2")
    parser.add_argument("--decode-source-torchair-pb2")
    parser.add_argument("--prefill-source-export-receipt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_state_correspondence import inspect_target_state_correspondence
    result = inspect_target_state_correspondence(args.prefill_job, args.decode_job, args.output,
        prefill_source_torchair_pb2=args.prefill_source_torchair_pb2, decode_source_torchair_pb2=args.decode_source_torchair_pb2,
        prefill_source_export_receipt=args.prefill_source_export_receipt)
    shown = {key: result[key] for key in ("status", "stage", "error", "common_parameter_content_count", "unique_common_parameter_content_count",
        "unique_state_context_candidate_count", "proven_cross_model_state_owner_count", "cross_model_state_identity", "feedback_ready", "next") if key in result}
    shown["report"] = str(Path(args.output).absolute() / "report.json")
    print(json.dumps(shown, ensure_ascii=False))
    return 1 if result["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Print retained direct-output producer metadata; no compilation, HDC or writes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="direct-trace directory; default AI_RUN_DIR/full-draft/direct-trace")
    parser.add_argument("--schema", help="explicit selected compiler schema library")
    parser.add_argument("--full", action="store_true", help="include raw descriptor/attribute and source-consumer metadata")
    parser.add_argument("--math", action="store_true", help="also review retained board outputs and the small norm gamma")
    args = parser.parse_args(argv)
    if not args.job and not os.environ.get("AI_RUN_DIR"):
        parser.error("--job or AI_RUN_DIR is required")
    from qwen35_moe_dflash.draft_output_provenance import inspect_draft_output_provenance
    job = args.job or Path(os.environ["AI_RUN_DIR"]) / "full-draft/direct-trace"
    report = inspect_draft_output_provenance(job, args.schema)
    if not args.full and report["status"] == "INSPECTED":
        report = {key: report[key] for key in ("status", "job", "omc", "source_binding", "resolution_counts", "scope")} | {
            "outputs": [{key: row[key] for key in ("output_index", "semantic_names", "source", "resolution")} | {
                "main_producer": {key: row["main_producer"].get(key) for key in ("name", "type", "selected_output_port")},
                "main_cl_names": row["main_cl_names"],
                "main_output_descriptor": row["main_producer"].get("selected_output_descriptor"),
                "netoutput_input_descriptor": row["netoutput_input_descriptor"],
                "main_consumers": row["main_consumers"],
                "path": [{"owner": step["owner"], "kind": step["kind"], "name": step["node"].get("name"),
                          "type": step["node"].get("type"), "port": step["node"].get("selected_output_port", step.get("port")),
                          "descriptor": step.get("selected_input_descriptor", step["node"].get("selected_output_descriptor"))}
                         for step in row["path"]],
                "source_consumers": [{"owner": item["owner"], "type": item["node"].get("type"),
                    "output_descriptor": item["node"].get("selected_output_descriptor"),
                    "consumers": item["consumers"]["edges"][:16],
                    "omitted_consumers": max(0, len(item["consumers"]["edges"]) - 16),
                    "unresolved_input_count": item["consumers"]["unresolved_input_count"]}
                    for item in row["source_occurrences"]]}
                for row in report["outputs"]]}
    if args.math:
        try:
            from qwen35_moe_dflash.draft_output_review import review_direct_output_math
            report["output_math"] = review_direct_output_math(job)
        except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
            report["output_math"] = {"status": "UNAVAILABLE", "reason": f"{type(error).__name__}: {str(error)[:800]}"}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] == "INSPECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inspect one retained target compile failure without rebuilding or reading weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def _descriptor_summary(value):
    return {key: value.get(key) for key in ("shape", "raw_dtype", "schema_dtype_names", "layout", "shape_semantics")
            if key in value}


def _edge_summary(value):
    shown = {key: value.get(key) for key in ("reference", "status", "reason", "producer_port") if key in value}
    producer = value.get("producer", {})
    shown["producer"] = {key: producer.get(key) for key in ("name", "type", "input", "file_path") if key in producer}
    for key in ("producer_descriptor", "consumer_descriptor"):
        if key in value:
            shown[key] = _descriptor_summary(value[key])
    return shown


def _shape_summary(result):
    shown = {key: result.get(key) for key in ("status", "input", "protobuf_decode", "schema",
                                             "graph_count", "matching_node_count", "error") if key in result}
    shown["matches"] = []
    for row in result.get("matches", [])[:4]:
        item = {key: row.get(key) for key in ("graph_path", "graph_name", "node_name", "node_type",
                                            "shape_constant") if key in row}
        for key in ("data_edge", "shape_edge"):
            item[key] = _edge_summary(row.get(key, {}))
        item["output_descriptors"] = [_descriptor_summary(value) for value in row.get("node", {}).get("output_desc", [])[:4]]
        item["file_constant_operands"] = [{key: value.get(key) for key in ("consumer", "input_index", "name", "file_path") if key in value}
                                          for value in row.get("file_constant_operands", [])[:16]]
        item["nearby_nodes"] = [{key: node.get(key) for key in ("name", "type", "input", "file_path") if key in node}
                                for node in row.get("neighborhood", {}).get("nodes", [])[:24]]
        shown["matches"].append(item)
    if not shown["matches"]:
        shown["attempts"] = result.get("attempts", [])[:2]
    return shown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="retained failed compile_full_target.py job")
    parser.add_argument("--node", default="Reshape", help="exact failed operation name, not a regex")
    parser.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH"),
                        help="selected DDK for read-only HIAI dump schema extraction")
    parser.add_argument("--torchair-pb2", help="optional actual source-AIR schema; no candidate substitution is inferred")
    parser.add_argument("--output", help="optional new full JSON report below AI_RUN_DIR; never overwritten")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.target_compile_failure import inspect_target_compile_failure
    from qwen35_moe_dflash.target_reshape_metadata import inspect_target_reshape_metadata

    try:
        report = inspect_target_compile_failure(args.job)
        report["reshape_metadata"] = []
        if args.torchair_pb2 and report.get("snapshot_air_path"):
            source = inspect_target_reshape_metadata(report["snapshot_air_path"],
                torchair_pb2=args.torchair_pb2, node_name=args.node)
            source["role"] = "ORIGINAL_SNAPSHOT_AIR"
            report["reshape_metadata"].append(source)
        else:
            report["source_air_schema"] = "NOT_PARSED_WITHOUT_EXPLICIT_ORIGIN_SCHEMA"
        ddk = args.ddk_root or report.get("ddk", {}).get("root") or report.get("ddk", {}).get("ddk_root")
        if ddk:
            for graph in report.get("compiler_graph_paths", [])[:2]:
                result = inspect_target_reshape_metadata(graph, ddk_root=ddk, node_name=args.node)
                result["role"] = "RETAINED_COMPILER_GRAPH"
                report["reshape_metadata"].append(result)
        else:
            report["compiler_graph_schema"] = "UNAVAILABLE_WITHOUT_SELECTED_DDK_ROOT"
        report["diagnostic_actions"] = "READ_ONLY; NO_WEIGHT_PAYLOAD_READS; NO_GRAPH_REWRITE; NO_COMPILER_OR_DEVICE_CALL"
        report["gear_interpretation"] = "A branch name alone does not establish its dynamic-gear identity or a successful prefill/decode path"
        if args.output:
            from qwen35_moe_dflash.artifacts import output_path
            path = output_path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(report, ensure_ascii=False, allow_nan=False)
            if len(raw.encode("utf-8")) > 16 << 20:
                raise ValueError("diagnostic report exceeds 16 MiB")
            with path.open("x", encoding="utf-8") as stream:
                stream.write(raw + "\n")
            report["full_report"] = str(path)
        shown = {key: report.get(key) for key in ("status", "job", "failure", "profile", "ddk", "snapshots",
            "snapshot_air_path", "compiler_graph_paths", "compiler_graph_discovery_errors", "partial_artifacts", "free_bytes", "limitations",
            "full_report", "source_air_schema", "compiler_graph_schema") if key in report}
        shown["options"] = {key: report.get("options", {}).get(key) for key in
                            ("dynamic_dims", "weight_data_type", "use_origin_format", "save_weights_as_external_data")
                            if key in report.get("options", {})}
        shown["log_evidence"] = report.get("log_evidence", {})
        shown["reshape_metadata"] = [_shape_summary(row) for row in report["reshape_metadata"]]
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if report.get("status") == "INSPECTED" else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}",
                          "diagnostic_actions": "NO_COMPILER_OR_DEVICE_CALL"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Print bounded original AIR metadata; no weights, compiler, device or graph changes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main():
    from qwen35_moe_dflash.original_air_inspect import inspect_original_air

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", help="existing original AIR, read only; at most 64 MiB")
    parser.add_argument("--torchair-pb2", help="explicit TorchAir generated pb2; AST literal only, never imported; takes precedence over HIAI")
    parser.add_argument("--find-torchair-schema", action="store_true", help="list bounded distribution/sys.path pb2 candidates without importing torchair; --air is optional")
    parser.add_argument("--ddk-root", default=os.environ.get("CPRO_DDK_ROOT") or os.environ.get("DDK_PATH"),
                        help="selected DDK used only for static candidate protobuf schema extraction")
    parser.add_argument("--full", action="store_true", help="include bounded complete root operation/descriptor and FileConstant metadata; no tensor blobs")
    parser.add_argument("--output", help="optional new JSON file under AI_RUN_DIR; default writes stdout only")
    args = parser.parse_args()
    if args.find_torchair_schema:
        from qwen35_moe_dflash.air_schema import discover_torchair_schemas
        print(json.dumps(discover_torchair_schemas(), ensure_ascii=False))
        return 0
    if not args.air:
        parser.error("--air is required unless --find-torchair-schema is selected")
    result = inspect_original_air(args.air, args.ddk_root, full=args.full, torchair_pb2=args.torchair_pb2)
    raw = json.dumps(result, ensure_ascii=False)
    if args.output:
        from qwen35_moe_dflash.artifacts import output_path
        path = output_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(raw + "\n")
    if args.output:
        summary = {key: result[key] for key in (
            "schema_version", "artifact_kind", "status", "air", "tap_resolution",
            "external_weights", "device_execution", "model_compilation") if key in result}
        summary["report"] = str(path)
        summary["candidates"] = [{"format": row["format"], "status": row["status"],
                                  "error": row.get("error"),
                                  "original_boundary_check": row.get("graph", {}).get("original_boundary_check")}
                                 for row in result.get("candidates", [])]
        for key in ("error", "schema", "source_preservation"):
            if key in result:
                summary[key] = result[key]
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(raw)
    return 1 if result["status"] == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())

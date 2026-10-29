#!/usr/bin/env python3
"""Review plan B's original Cpro GE bindings; never generate AIR or load weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main():
    from qwen35_moe_dflash.original_ge_target import prepare_contract, read_json, source_receipt

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-receipt", help="original six-file source receipt JSON; no weight paths are followed")
    source.add_argument("--source-root", help="rehash only the six audited original code/config files")
    parser.add_argument("--air-report", help="JSON stdout from inspect_original_target_air.py --full")
    parser.add_argument("--air", help="original AIR path used only in the concrete next inspection command")
    parser.add_argument("--gamma-template", help="explicit exact FileConstant path with one {layer}; no fuzzy name matching")
    parser.add_argument("--output", help="optional new directory under AI_RUN_DIR containing contract.json; default stdout JSON only")
    args = parser.parse_args()
    receipt, receipt_hash = (read_json(args.source_receipt, 1 << 20) if args.source_receipt else
                             (source_receipt(args.source_root), None))
    report, report_hash = read_json(args.air_report) if args.air_report else (None, None)
    result = prepare_contract(receipt, report, gamma_template=args.gamma_template, air=args.air)
    result["metadata_receipts"] = {"source_receipt_sha256": receipt_hash, "air_report_sha256": report_hash,
                                    "actual_source_bytes_rehashed": bool(args.source_root)}
    raw = json.dumps(result, ensure_ascii=False)
    if args.output:
        from qwen35_moe_dflash.artifacts import output_path
        path = output_path(args.output)
        path.mkdir(parents=True, exist_ok=False)
        with (path / "contract.json").open("x") as stream:
            stream.write(raw + "\n")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

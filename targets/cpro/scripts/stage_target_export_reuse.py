#!/usr/bin/env python3
"""Inspect transferred metadata or plan exact same-filesystem sidecar reuse."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def _export_profile_check(receipt_path, inventory_path, air_path, *, sequence_length=64, allow_profile_difference=False):
    """Bind three small transferred files before any sidecar access."""
    from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
    from qwen35_moe_dflash.target_export_reuse import _absolute, _read_small, _unique_object, _reject_constant

    def metadata(path):
        raw, identity = _read_small(_absolute(path), 32 << 20)
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError("transferred metadata root must be an object")
        return value, identity

    receipt, receipt_identity = metadata(receipt_path)
    inventory, inventory_identity = metadata(inventory_path)
    _, air_identity = _read_small(_absolute(air_path), 64 << 20)
    if receipt.get("artifact_kind") != "qwen35-canonical-target-export" or receipt.get("status") != "EXPORT_READY":
        raise ValueError("export receipt is not a completed canonical target export")
    if (inventory.get("artifact_kind") != "qwen35-target-export-inventory" or inventory.get("status") != "INVENTORIED"
            or inventory.get("weight_identity") != "SHA256_COMPLETE"):
        raise ValueError("transferred inventory is not SHA256_COMPLETE")
    for expected in (receipt.get("air", {}), inventory.get("air", {})):
        if any(expected.get(key) != air_identity[key] for key in ("bytes", "sha256")):
            raise ValueError("transferred AIR size/SHA256 differs from the export receipt/inventory")
    expected_inventory = receipt.get("inventory", {})
    if (expected_inventory.get("sha256") != inventory_identity["sha256"]
            or ("bytes" in expected_inventory and expected_inventory["bytes"] != inventory_identity["bytes"])):
        raise ValueError("transferred inventory SHA256/size differs from the export receipt")
    source_schema, target_schema = receipt.get("schema", {}), inventory.get("schema", {})
    digest = source_schema.get("descriptor_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("export receipt selected-schema identity is missing")
    if any(source_schema.get(key) != target_schema.get(key) for key in ("descriptor_sha256", "schema_package", "schema_file")):
        raise ValueError("export receipt and inventory selected-schema identities differ")
    normalization = receipt.get("normalization", {})
    if (normalization.get("status") != "OUTPUT_ORDER_CANONICALIZED"
            or normalization.get("patched_sha256") != air_identity["sha256"]
            or type(normalization.get("sequence_length")) is not int
            or normalization.get("sequence_length") != sequence_length):
        raise ValueError(f"canonical output-order receipt is not bound to this {sequence_length}-token AIR")
    schema = normalization.get("schema", {})
    if any(schema.get(key) != source_schema.get(key) for key in ("descriptor_sha256", "schema_package", "schema_file")):
        raise ValueError("normalization selected-schema identity differs from its export receipt")
    observed = normalization.get("input_validation", {})
    rows = observed.get("inputs")
    if observed.get("status") != "MATCHED_TYPED_INPUT_GEOMETRIES" or not isinstance(rows, list):
        raise ValueError("export receipt does not retain typed input metadata")
    data = [row for row in rows if isinstance(row, dict) and row.get("type") == "Data"]
    profile = get_existing_air_profile("cpro-40block-decode-1" if sequence_length == 1 else "cpro-40block-64-1")
    expected = {name: [int(dim) for dim in shape.split(",")] for name, shape in
        (item.split(":", 1) for item in profile["input_shape"].split(";")) if not name.startswith("RefData_")}
    names = [row.get("name") for row in data]
    differences = []
    if len(data) != 9 or any(not isinstance(name, str) for name in names) or len(set(names)) != 9 or set(names) != set(expected):
        missing = sorted(set(expected) - {name for name in names if isinstance(name, str)})
        extra = sorted({name for name in names if isinstance(name, str)} - set(expected))
        differences.append({"field": "data_names", "missing": missing, "unexpected": extra, "count": len(data)})
    for row in data:
        shape = row.get("shape")
        if (not isinstance(row.get("name"), str) or not row["name"]
                or not isinstance(shape, list) or any(type(dim) is not int for dim in shape)):
            raise ValueError("PROFILE_BINDING_REQUIRED: malformed Data name or non-integer input shape")
        if row.get("name") not in expected:
            continue
        template = expected[row["name"]]
        if (not isinstance(shape, list) or len(shape) != len(template) or any(type(dim) is not int for dim in shape)
                or any(dim != fixed if fixed != -1 else dim not in {-1, 64} for dim, fixed in zip(shape, template))):
            differences.append({"field": "input_shape", "name": row["name"], "expected": template, "observed": shape})
    if differences and not allow_profile_difference:
        raise ValueError("PROFILE_BINDING_REQUIRED: exact Data names or declared input shapes differ: " + json.dumps(differences))
    return {"status": "PROFILE_BINDING_REQUIRED" if differences else "MATCHED_EXACT_DATA_NAMES_AND_DECLARED_GEOMETRIES", "profile": profile["name"],
            "sequence_length": sequence_length, "observed_data_inputs": data, "profile_differences": differences,
            "export_receipt": receipt_identity, "air_sha256": air_identity["sha256"],
            "air": air_identity, "inventory": inventory_identity,
            "inventory_sha256": inventory_identity["sha256"], "schema_descriptor_sha256": digest,
            "scope": "Source-bound names/shapes only; equal-shaped control identity, source-to-AIR semantics, compiler schema acceptance and device behavior remain unverified"}


def _inspect_transfer(args):
    from qwen35_moe_dflash.target_export_reuse import _absolute, _inventory, _read_small

    check = _export_profile_check(args.export_receipt, args.inventory, args.air,
        sequence_length=args.sequence_length, allow_profile_difference=True)
    raw, identity = _read_small(_absolute(args.inventory), 32 << 20)
    if identity != check["inventory"]:
        raise ValueError("transferred inventory changed during inspection")
    manifest, rows, prefix = _inventory(raw, sequence_length=args.sequence_length)
    # Recheck all three transferred files, never follow origin payload paths.
    for path, limit, expected in ((args.export_receipt, 32 << 20, check["export_receipt"]),
            (args.inventory, 32 << 20, check["inventory"]), (args.air, 64 << 20, check["air"])):
        if _read_small(_absolute(path), limit)[1] != expected:
            raise ValueError("transferred metadata changed during inspection")
    return {"schema_version": 1, "artifact_kind": "qwen35-target-export-transfer-inspection",
        "status": "TRANSFER_INSPECTED", "mode": "METADATA_ONLY", "sequence_length": args.sequence_length,
        "export_profile_check": check, "declared_abi": manifest["declared_abi"],
        "source_relative_prefix": prefix, "unique_reference_count": len(rows),
        "total_weight_bytes": sum(row["bytes"] for row in rows), "weight_payload_access": "NONE",
        "weight_payload_validation": "ORIGIN_HASH_DECLARATIONS_ONLY_NOT_REHASHED_ON_DDK",
        "air_metadata_decode": "NOT_PERFORMED_RECEIPT_BOUND_DECLARATIONS_ONLY",
        "staging": "NOT_RUN", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "next": "Resolve actual profile names and retained-source ownership before staging or compiling",
        "scope": "Three small transferred files only; no sidecar access, import directory, hardlink or compiler invocation"}


def main(argv=None):
    from qwen35_moe_dflash.target_export_reuse import stage_target_export_reuse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="origin SHA256_COMPLETE small export inventory JSON")
    parser.add_argument("--air", required=True, help="copied small new AIR; exact inventory SHA256 required")
    parser.add_argument("--existing-export", help="existing old export directory containing direct sidecar files; required for staging")
    parser.add_argument("--output", help="new staging directory below AI_RUN_DIR; required for staging, never overwritten")
    parser.add_argument("--export-receipt", help="optional transferred canonical result.json; validate exact profile names before sidecar reads")
    parser.add_argument("--apply", action="store_true", help="hardlink only if all sidecars match and filesystem/space checks pass")
    parser.add_argument("--full", action="store_true", help="print the full per-file report; applied receipts always retain it")
    parser.add_argument("--inspect-transfer", action="store_true", help="read only three small transferred files; no sidecar access or staging")
    parser.add_argument("--sequence-length", type=int, choices=(1, 64), default=64, help="explicit transferred feature length; 1 currently supports metadata inspection only")
    args = parser.parse_args(argv)
    if args.inspect_transfer:
        if not args.export_receipt or args.apply or args.existing_export or args.output:
            parser.error("--inspect-transfer requires --export-receipt and cannot use --apply, --existing-export or --output")
        try:
            report = _inspect_transfer(args)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            report = {"status": "BLOCKED", "stage": "transfer-inspection", "weight_payload_access": "NONE",
                "staging": "NOT_RUN", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
                "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if report["status"] == "TRANSFER_INSPECTED" else 1
    if not args.existing_export or not args.output or args.sequence_length != 64:
        parser.error("staging requires --existing-export/--output; sequence length 1 currently requires --inspect-transfer")
    try:
        check = _export_profile_check(args.export_receipt, args.inventory, args.air) if args.export_receipt else None
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        print(json.dumps({"status": "BLOCKED", "stage": "export-profile-preflight",
            "export_profile_check": "BLOCKED", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1
    report = stage_target_export_reuse(args.inventory, args.air, args.existing_export, args.output, apply=args.apply)
    if check is not None:
        try:
            from qwen35_moe_dflash.target_export_reuse import _absolute, _read_small
            if _read_small(_absolute(args.export_receipt), 32 << 20)[1] != check["export_receipt"]:
                raise ValueError("export receipt changed while staging")
            if report["status"] in {"PLAN_READY", "STAGED_HARDLINKS", "TRANSFER_REQUIRED"} and (
                report.get("air", {}).get("sha256") != check["air_sha256"]
                or report.get("inventory", {}).get("sha256") != check["inventory_sha256"]):
                raise ValueError("staged AIR/inventory differs from the profile preflight")
        except (OSError, ValueError, TypeError, KeyError) as error:
            report.update(stage_result_status=report["status"], status="BLOCKED", stage="export-profile-postcheck",
                          error=f"{type(error).__name__}: {error}; stage evidence was retained")
            check["status"] = "BINDING_CHANGED_AFTER_STAGE"
        report["export_profile_check"] = check
    shown = report
    if not args.full:
        shown = {key: value for key, value in report.items() if key not in {"files", "transfer_required", "created_links"}}
        shown.update(file_count=len(report.get("files", [])),
                     file_status_counts=dict(Counter(row["status"] for row in report.get("files", []))),
                     transfer_required_count=len(report.get("transfer_required", [])),
                     transfer_required=report.get("transfer_required", [])[:8],
                     created_link_count=len(report.get("created_links", [])))
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] in {"PLAN_READY", "STAGED_HARDLINKS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

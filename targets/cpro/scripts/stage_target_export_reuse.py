#!/usr/bin/env python3
"""Plan exact old-sidecar reuse; --apply hardlinks verified files and copies AIR."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))


def _export_profile_check(receipt_path, inventory_path, air_path):
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
            or normalization.get("sequence_length") != 64):
        raise ValueError("canonical output-order receipt is not bound to this 64-token AIR")
    schema = normalization.get("schema", {})
    if any(schema.get(key) != source_schema.get(key) for key in ("descriptor_sha256", "schema_package", "schema_file")):
        raise ValueError("normalization selected-schema identity differs from its export receipt")
    observed = normalization.get("input_validation", {})
    rows = observed.get("inputs", [])
    if observed.get("status") != "MATCHED_TYPED_INPUT_GEOMETRIES" or not isinstance(rows, list):
        raise ValueError("export receipt does not retain typed input metadata")
    data = [row for row in rows if isinstance(row, dict) and row.get("type") == "Data"]
    profile = get_existing_air_profile("cpro-40block-64-1")
    expected = {name: [int(dim) for dim in shape.split(",")] for name, shape in
        (item.split(":", 1) for item in profile["input_shape"].split(";")) if not name.startswith("RefData_")}
    names = [row.get("name") for row in data]
    if len(data) != 9 or any(not isinstance(name, str) for name in names) or len(set(names)) != 9 or set(names) != set(expected):
        missing = sorted(set(expected) - {name for name in names if isinstance(name, str)})
        extra = sorted({name for name in names if isinstance(name, str)} - set(expected))
        raise ValueError(f"PROFILE_BINDING_REQUIRED: exact nine Data names differ; missing={missing}; unexpected={extra}; count={len(data)}")
    for row in data:
        shape, template = row.get("shape"), expected[row["name"]]
        if (not isinstance(shape, list) or len(shape) != len(template) or any(type(dim) is not int for dim in shape)
                or any(dim != fixed if fixed != -1 else dim not in {-1, 64} for dim, fixed in zip(shape, template))):
            raise ValueError("PROFILE_BINDING_REQUIRED: declared input shape differs for " + row["name"])
    return {"status": "MATCHED_EXACT_DATA_NAMES_AND_DECLARED_GEOMETRIES", "profile": profile["name"],
            "export_receipt": receipt_identity, "air_sha256": air_identity["sha256"],
            "inventory_sha256": inventory_identity["sha256"], "schema_descriptor_sha256": digest,
            "scope": "Source-bound names/shapes only; equal-shaped control identity, source-to-AIR semantics, compiler schema acceptance and device behavior remain unverified"}


def main(argv=None):
    from qwen35_moe_dflash.target_export_reuse import stage_target_export_reuse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="origin SHA256_COMPLETE small export inventory JSON")
    parser.add_argument("--air", required=True, help="copied small new AIR; exact inventory SHA256 required")
    parser.add_argument("--existing-export", required=True, help="existing old export directory containing direct sidecar files")
    parser.add_argument("--output", required=True, help="new staging directory below AI_RUN_DIR; never overwritten")
    parser.add_argument("--export-receipt", help="optional transferred canonical result.json; validate exact profile names before sidecar reads")
    parser.add_argument("--apply", action="store_true", help="hardlink only if all sidecars match and filesystem/space checks pass")
    parser.add_argument("--full", action="store_true", help="print the full per-file report; applied receipts always retain it")
    args = parser.parse_args(argv)
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

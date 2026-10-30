"""Bind a source AIR's producer schema through its saved canonical receipt.

No package discovery, generated-module execution or vendor import is used.
The compiled OMC schema is selected independently by the graph loader.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .air_schema import load_torchair_schema


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_KEYS = ("descriptor_sha256", "schema_package", "schema_file")


def tensor_enum_fields(schema):
    """Read enum domains from this schema's own TensorDescriptor fields."""
    fields = {}
    for field in schema._messages.get(schema.package + ".TensorDescriptor", {}).values():
        if field.kind == 14:
            values = schema.enums[field.type_name]
            if len(values) > 256:
                raise ValueError("source TensorDescriptor enum exceeds256 entries")
            fields[field.name] = {"enum_type": field.type_name, "values": values}
    if "dtype" not in fields or len(json.dumps(fields).encode()) > 32 << 10:
        raise ValueError("source TensorDescriptor requires a bounded explicit dtype enum")
    return fields


def _prefill_inventory_schema(report, air_identity, *, read_metadata, read_pb2, source_torchair_pb2):
    """Use the successful static64 staging inventory as the producer pin.

    This reads only metadata and one explicitly hash-pinned pb2. The compiler's
    completed payload-verification declaration is retained, never reinterpreted
    as a fresh weight read or a source-model correctness claim.
    """
    external = report.get("external_weights", {})
    if not isinstance(external, dict):
        raise ValueError("declared external-weight producer metadata must be an object")
    if "reuse_binding" not in external:
        return None
    reuse = external["reuse_binding"]
    if (not isinstance(reuse, dict) or reuse.get("status") != "RECEIPT_BOUND_SELECTION"
            or reuse.get("payload_verification") != "ALL_SELECTED_SNAPSHOT_SHA256_MATCHED_RECEIPT"):
        raise ValueError("declared prefill producer inventory lacks its completed receipt-bound selection")
    pin = reuse.get("inventory")
    if (not isinstance(pin, dict) or not isinstance(pin.get("path"), str) or not Path(pin["path"]).is_absolute()
            or not _SHA.fullmatch(str(pin.get("sha256", "")))
            or type(pin.get("bytes")) is not int or not 0 < pin["bytes"] <= 32 << 20):
        raise ValueError("prefill producer inventory requires an exact bounded metadata path/bytes/SHA256 pin")
    manifest, manifest_id = read_metadata(Path(pin["path"]), "source-prefill-export-inventory")
    if any(manifest_id[key] != pin[key] for key in ("bytes", "sha256")):
        raise ValueError("prefill producer inventory bytes/SHA256 changed: " + pin["path"])
    from .target_export_reuse import _inventory
    validated, _, prefix = _inventory(json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode(), sequence_length=64)
    if (reuse.get("source_relative_prefix") != prefix
            or any(validated["air"].get(key) != air_identity[key] for key in ("bytes", "sha256"))):
        raise ValueError("prefill producer inventory does not bind this exact static64 source AIR snapshot")
    declared = validated["schema"]
    if (declared.get("schema_package") != "ge.proto"
            or declared.get("schema_file") not in {"ge_ir.proto", "proto/ge_ir.proto"}
            or not _SHA.fullmatch(str(declared.get("descriptor_sha256", "")))
            or not _SHA.fullmatch(str(declared.get("pb2_sha256", "")))
            or type(declared.get("pb2_bytes")) is not int or not 0 < declared["pb2_bytes"] <= 8 << 20
            or not isinstance(declared.get("pb2"), str) or not Path(declared["pb2"]).is_absolute()):
        raise ValueError("prefill inventory producer schema lacks its complete explicit AST source identity")
    chosen = Path(source_torchair_pb2).expanduser() if source_torchair_pb2 else Path(declared["pb2"])
    pb2_id = read_pb2(chosen)
    if (pb2_id["sha256"] != declared["pb2_sha256"] or pb2_id["bytes"] != declared["pb2_bytes"]):
        raise ValueError("producer pb2 content differs from the prefill inventory: " + str(chosen))
    schema = load_torchair_schema(chosen)
    if (any(schema.evidence.get(key) != declared.get(key) for key in _SCHEMA_KEYS)
            or schema.evidence["pb2_sha256"] != pb2_id["sha256"]):
        raise ValueError("AST-extracted producer descriptor differs from the prefill inventory schema")
    binding = {"status": "PREFILL_INVENTORY_AND_AST_DESCRIPTOR_MATCHED", "inventory": manifest_id,
        "recorded_pb2_path": declared["pb2"], "selected_pb2": pb2_id,
        "selection": "EXPLICIT_IDENTICAL_PB2_COPY" if source_torchair_pb2 else "BOUND_RECORDED_PRODUCER_PB2_PATH",
        "descriptor_sha256": declared["descriptor_sha256"], "pb2_execution": "NOT_EXECUTED",
        "sequence_length": 64, "source_relative_prefix": prefix,
        "compiled_schema_selection": "UNCHANGED_SELECTED_DDK_LIBRARY",
        "payload_verification": "SAVED_COMPLETED_SNAPSHOT_DECLARATION_ONLY_NO_FRESH_WEIGHT_ACCESS",
        "scope": "Exact static64 source AIR/inventory/producer schema identity only; no source-model, weight-table authenticity, tensor accuracy or cross-model state-equivalence claim"}
    return schema, binding


def _explicit_prefill_schema(report, air_identity, *, receipt_path, read_metadata, read_pb2, source_torchair_pb2):
    """An explicit canonical64 receipt may add an exact same-AIR witness."""
    view = report.get("decode_input_view")
    if view is not None:
        raise ValueError("an explicit prefill export receipt cannot replace a declared decode producer binding")
    receipt, receipt_id = read_metadata(Path(receipt_path), "source-export-receipt")
    declared, normalized = receipt.get("schema", {}), receipt.get("normalization", {})
    if (receipt.get("artifact_kind") != "qwen35-canonical-target-export" or receipt.get("status") != "EXPORT_READY"
            or not isinstance(declared, dict) or not isinstance(normalized, dict)
            or normalized.get("status") != "OUTPUT_ORDER_CANONICALIZED"
            or type(normalized.get("sequence_length")) is not int or normalized["sequence_length"] != 64
            or normalized.get("patched_sha256") != air_identity["sha256"]
            or any(receipt.get("air", {}).get(key) != air_identity[key] for key in ("bytes", "sha256"))):
        raise ValueError("explicit canonical producer receipt does not bind this exact static64 source AIR")
    if (declared.get("schema_package") != "ge.proto" or declared.get("schema_file") not in {"ge_ir.proto", "proto/ge_ir.proto"}
            or not _SHA.fullmatch(str(declared.get("descriptor_sha256", "")))
            or not _SHA.fullmatch(str(declared.get("pb2_sha256", "")))
            or not isinstance(declared.get("pb2"), str) or not Path(declared["pb2"]).is_absolute()
            or not isinstance(normalized.get("schema"), dict)
            or any(normalized["schema"].get(key) != declared.get(key) for key in _SCHEMA_KEYS)):
        raise ValueError("explicit canonical64 producer schema identity is missing or inconsistent")
    chosen = Path(source_torchair_pb2).expanduser() if source_torchair_pb2 else Path(declared["pb2"])
    pb2_id = read_pb2(chosen)
    if (pb2_id["sha256"] != declared["pb2_sha256"] or
            ("pb2_bytes" in declared and (type(declared["pb2_bytes"]) is not int or pb2_id["bytes"] != declared["pb2_bytes"]))):
        raise ValueError("producer pb2 content differs from explicit canonical64 receipt")
    schema = load_torchair_schema(chosen)
    if (any(schema.evidence.get(key) != declared.get(key) for key in _SCHEMA_KEYS)
            or schema.evidence["pb2_sha256"] != pb2_id["sha256"]):
        raise ValueError("AST-extracted producer descriptor differs from explicit canonical64 receipt")
    implicit = _prefill_inventory_schema(report, air_identity, read_metadata=read_metadata,
        read_pb2=read_pb2, source_torchair_pb2=source_torchair_pb2)
    binding = {"status": "PREFILL_CANONICAL_EXPORT_RECEIPT_AND_AST_DESCRIPTOR_MATCHED", "receipt": receipt_id,
        "recorded_pb2_path": declared["pb2"], "selected_pb2": pb2_id,
        "selection": "EXPLICIT_IDENTICAL_PB2_COPY" if source_torchair_pb2 else "BOUND_RECORDED_PRODUCER_PB2_PATH",
        "descriptor_sha256": declared["descriptor_sha256"], "pb2_execution": "NOT_EXECUTED", "sequence_length": 64,
        "compiled_schema_selection": "UNCHANGED_SELECTED_DDK_LIBRARY",
        "scope": "Explicit same-AIR canonical64 producer schema witness only; no state feedback, numerical accuracy or cross-model state-equivalence claim"}
    if implicit is not None:
        other_schema, other = implicit
        if any(schema.evidence.get(key) != other_schema.evidence.get(key) for key in (*_SCHEMA_KEYS, "pb2_sha256", "pb2_bytes")):
            raise ValueError("explicit canonical64 and completed prefill inventory producer schemas disagree; no fallback")
        inventory_pin = receipt.get("inventory")
        if inventory_pin is not None and (not isinstance(inventory_pin, dict)
                or not _SHA.fullmatch(str(inventory_pin.get("sha256", "")))
                or inventory_pin["sha256"] != other["inventory"]["sha256"]
                or ("bytes" in inventory_pin and (type(inventory_pin["bytes"]) is not int or inventory_pin["bytes"] != other["inventory"]["bytes"]))):
            raise ValueError("explicit canonical64 inventory pin disagrees with the completed compilation reuse inventory")
        binding.update(status="PREFILL_INVENTORY_AND_CANONICAL_RECEIPT_AST_DESCRIPTOR_MATCHED", inventory=other["inventory"],
            source_relative_prefix=other["source_relative_prefix"], payload_verification=other["payload_verification"])
    return schema, binding


def load_bound_source_schema(report, air_identity, *, read_metadata, read_pb2,
                             source_torchair_pb2=None, source_export_receipt=None):
    """Return a receipt-bound TorchAir schema, or None for a strict legacy lane.

    Callbacks retain the graph audit's read attempts and fresh file identities.
    A declared producer receipt is authoritative: any failure is fatal and
    cannot fall back to another protobuf schema.
    """
    if source_export_receipt is not None:
        return _explicit_prefill_schema(report, air_identity, receipt_path=source_export_receipt,
            read_metadata=read_metadata, read_pb2=read_pb2, source_torchair_pb2=source_torchair_pb2)
    view = report.get("decode_input_view")
    transfer = view.get("transfer") if isinstance(view, dict) else None
    if transfer is None:
        if isinstance(view, dict) and ("transfer" in view or "decode_export_receipt" in view):
            raise ValueError("declared producer receipt is missing its exact transfer binding; no schema fallback is permitted")
        prefill = _prefill_inventory_schema(report, air_identity, read_metadata=read_metadata,
            read_pb2=read_pb2, source_torchair_pb2=source_torchair_pb2)
        if prefill is not None:
            return prefill
        if source_torchair_pb2 is not None:
            raise ValueError("an explicit source pb2 requires a hash-bound canonical producer export receipt")
        return None
    if not isinstance(transfer, dict):
        raise ValueError("declared decode producer transfer must be an object")
    pin = transfer.get("export_receipt")
    if (not isinstance(pin, dict) or not isinstance(pin.get("path"), str)
            or not _SHA.fullmatch(str(pin.get("sha256", "")))
            or type(pin.get("bytes")) is not int or pin["bytes"] <= 0):
        raise ValueError("declared producer transfer lacks its exact canonical export receipt identity")
    if view.get("decode_export_receipt") != pin["path"]:
        raise ValueError("decode producer receipt path differs from its bound transfer")
    receipt, receipt_id = read_metadata(Path(pin["path"]), "source-export-receipt")
    if any(receipt_id[key] != pin[key] for key in ("bytes", "sha256")):
        raise ValueError("canonical producer export receipt bytes/SHA256 changed: " + pin["path"])
    normalization, declared = receipt.get("normalization", {}), receipt.get("schema", {})
    if (not isinstance(normalization, dict) or not isinstance(normalization.get("schema"), dict)
            or not isinstance(declared, dict)
            or not isinstance(receipt.get("air"), dict) or not isinstance(transfer.get("air"), dict)):
        raise ValueError("canonical producer AIR, schema and normalization must be objects")
    if (receipt.get("artifact_kind") != "qwen35-canonical-target-export" or receipt.get("status") != "EXPORT_READY"
            or normalization.get("status") != "OUTPUT_ORDER_CANONICALIZED"
            or type(normalization.get("sequence_length")) is not int or normalization["sequence_length"] != 1
            or normalization.get("patched_sha256") != air_identity["sha256"]
            or any(receipt.get("air", {}).get(key) != air_identity[key]
                   or transfer.get("air", {}).get(key) != air_identity[key] for key in ("bytes", "sha256"))):
        raise ValueError("canonical producer receipt does not bind this exact static1 source AIR snapshot")
    if (declared.get("schema_package") != "ge.proto"
            or declared.get("schema_file") not in {"ge_ir.proto", "proto/ge_ir.proto"}
            or not _SHA.fullmatch(str(declared.get("descriptor_sha256", "")))
            or not _SHA.fullmatch(str(declared.get("pb2_sha256", "")))
            or not isinstance(declared.get("pb2"), str) or not Path(declared["pb2"]).is_absolute()
            or any(normalization.get("schema", {}).get(key) != declared.get(key) for key in _SCHEMA_KEYS)
            or transfer.get("schema_descriptor_sha256") != declared["descriptor_sha256"]):
        raise ValueError("canonical producer schema identity is missing or inconsistent across bound receipts")
    chosen = Path(source_torchair_pb2).expanduser() if source_torchair_pb2 else Path(declared["pb2"])
    pb2_id = read_pb2(chosen)
    if (pb2_id["sha256"] != declared["pb2_sha256"]
            or ("pb2_bytes" in declared and pb2_id["bytes"] != declared["pb2_bytes"])):
        raise ValueError("producer pb2 content differs from canonical export receipt: " + str(chosen))
    schema = load_torchair_schema(chosen)
    if (any(schema.evidence.get(key) != declared.get(key) for key in _SCHEMA_KEYS)
            or schema.evidence["pb2_sha256"] != pb2_id["sha256"]):
        raise ValueError("AST-extracted producer descriptor differs from the canonical source AIR schema")
    binding = {"status": "CANONICAL_EXPORT_RECEIPT_AND_AST_DESCRIPTOR_MATCHED", "receipt": receipt_id,
        "recorded_pb2_path": declared["pb2"], "selected_pb2": pb2_id,
        "selection": "EXPLICIT_IDENTICAL_PB2_COPY" if source_torchair_pb2 else "BOUND_RECORDED_PRODUCER_PB2_PATH",
        "descriptor_sha256": declared["descriptor_sha256"], "pb2_execution": "NOT_EXECUTED",
        "compiled_schema_selection": "UNCHANGED_SELECTED_DDK_LIBRARY",
        "scope": "Exact source AIR/producer schema identity only; no tensor payload or compiler equivalence claim"}
    return schema, binding

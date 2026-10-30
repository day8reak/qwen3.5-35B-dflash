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


def load_bound_source_schema(report, air_identity, *, read_metadata, read_pb2,
                             source_torchair_pb2=None):
    """Return a receipt-bound TorchAir schema, or None for a strict legacy lane.

    Callbacks retain the graph audit's read attempts and fresh file identities.
    A declared producer receipt is authoritative: any failure is fatal and
    cannot fall back to another protobuf schema.
    """
    view = report.get("decode_input_view")
    transfer = view.get("transfer") if isinstance(view, dict) else None
    if transfer is None:
        if isinstance(view, dict) and ("transfer" in view or "decode_export_receipt" in view):
            raise ValueError("declared producer receipt is missing its exact transfer binding; no schema fallback is permitted")
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

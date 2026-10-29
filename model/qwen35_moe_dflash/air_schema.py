"""Read a TorchAir GE descriptor as AST data, without importing its package.

Generated pb2 modules may import torchair/NPU code through a version-selection
shim. This module executes none of them. The explicitly selected literal must
declare the known GE file/package and have no external descriptor dependencies.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import os
from pathlib import Path
import re
import stat
import sys

from .omc_wire import Schema, _descriptor_fields, _many, _one, _text


MAX_PB2_BYTES = 8 << 20
MAX_DESCRIPTOR_BYTES = 8 << 20
_MAX_AST_NODES = 100_000
_NAME = re.compile(r"ge_ir(?:_by_protoc_[0-9_]+)?_pb2\.py\Z")
_KNOWN_FILES = ("ge_ir_pb2.py", "ge_ir_by_protoc_3_13_pb2.py", "ge_ir_by_protoc_3_19_pb2.py")
_KNOWN_RELATIVES = tuple(
    str(Path(folder) / name)
    for folder in ("torchair/_ge_concrete_graph", "torchair/proto",
                   "torchair/_ge_concrete_graph/proto", "proto",
                   "python/torchair/_ge_concrete_graph")
    for name in _KNOWN_FILES
)


def _stamp(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def extract_torchair_descriptor(pb2):
    """Return one literal FileDescriptorProto and source identity; never exec."""
    path = Path(pb2).expanduser().resolve(strict=True)
    before = _stamp(path)
    if not stat.S_ISREG(path.stat().st_mode) or not 0 < before[2] <= MAX_PB2_BYTES:
        raise ValueError("TorchAir pb2 must be a nonempty regular file at most 8 MiB")
    with path.open("rb") as stream:
        source = stream.read(MAX_PB2_BYTES + 1)
    if len(source) != before[2] or _stamp(path) != before:
        raise ValueError("TorchAir pb2 changed during bounded read")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("TorchAir pb2 cannot be parsed safely as Python AST") from error
    for index, _ in enumerate(ast.walk(tree), 1):
        if index > _MAX_AST_NODES:
            raise ValueError("TorchAir pb2 exceeds AST node budget")
    assignments = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(target, ast.Name) and target.id == "DESCRIPTOR" for target in targets):
            assignments.append(node.value)
    if len(assignments) != 1 or not isinstance(assignments[0], ast.Call):
        raise ValueError("pb2 has no unique literal DESCRIPTOR; a shim requires selecting ge_ir_by_protoc_*_pb2.py")
    call = assignments[0]
    name = call.func.attr if isinstance(call.func, ast.Attribute) else (call.func.id if isinstance(call.func, ast.Name) else None)
    if name == "AddSerializedFile" and len(call.args) == 1 and not call.keywords:
        literal, method = call.args[0], "AddSerializedFile"
    elif name == "FileDescriptor":
        values = [entry.value for entry in call.keywords if entry.arg == "serialized_pb"]
        if len(values) != 1:
            raise ValueError("FileDescriptor requires one serialized_pb literal")
        literal, method = values[0], "FileDescriptor.serialized_pb"
    else:
        raise ValueError("DESCRIPTOR must use AddSerializedFile or FileDescriptor(serialized_pb=...)")
    if not isinstance(literal, ast.Constant) or not isinstance(literal.value, bytes):
        raise ValueError("serialized descriptor must be a literal bytes value; expressions are not executed")
    descriptor = literal.value
    if not 0 < len(descriptor) <= MAX_DESCRIPTOR_BYTES:
        raise ValueError("serialized descriptor exceeds byte budget")
    fields = _descriptor_fields(memoryview(descriptor), [2_000_000])
    filename = _text(_one(fields, 1, 2, b""))
    package = _text(_one(fields, 2, 2, b""))
    if filename not in {"ge_ir.proto", "proto/ge_ir.proto"} or package != "ge.proto":
        raise ValueError("TorchAir descriptor must declare ge_ir.proto or proto/ge_ir.proto in package ge.proto")
    if _many(fields, 3, 2):
        raise ValueError("external descriptor dependencies are unsupported")
    evidence = {"source_kind": "TORCHAIR_PB2_AST_LITERAL", "pb2": str(path),
                "pb2_bytes": len(source), "pb2_sha256": hashlib.sha256(source).hexdigest(),
                "extraction": method, "descriptor_bytes": len(descriptor),
                "descriptor_sha256": hashlib.sha256(descriptor).hexdigest(),
                "schema_file": filename, "schema_package": package,
                "pb2_execution": "NOT_EXECUTED"}
    # Validate the entire descriptor before it can be selected for AIR decoding.
    Schema(descriptor, evidence, expected_file=filename, expected_package=package)
    if _stamp(path) != before:
        raise ValueError("TorchAir pb2 changed during AST descriptor extraction")
    return descriptor, evidence


def load_torchair_schema(pb2):
    descriptor, evidence = extract_torchair_descriptor(pb2)
    return Schema(descriptor, evidence, expected_file=evidence["schema_file"],
                  expected_package=evidence["schema_package"])


def discover_torchair_schemas():
    """List bounded distribution/sys.path candidates without package imports."""
    paths = {}
    errors = []
    total_files = 0
    for number, dist in enumerate(metadata.distributions(), 1):
        if number > 4096:
            errors.append("distribution count exceeded discovery budget")
            break
        for item in dist.files or ():
            total_files += 1
            if total_files > 500_000:
                errors.append("distribution file list exceeded discovery budget")
                break
            value = Path(str(item))
            if _NAME.fullmatch(value.name) and ("torchair" in value.parts or "proto" in value.parts):
                path = Path(dist.locate_file(item))
                paths.setdefault(str(path), set()).add("DISTRIBUTION_FILES")
        if total_files > 500_000:
            break
    for entry in sys.path[:512]:
        root = Path(entry or os.curdir)
        if not root.is_dir():
            continue
        for relative in _KNOWN_RELATIVES:
            path = root / relative
            if path.is_file():
                paths.setdefault(str(path), set()).add("SYS_PATH_FIXED_CANDIDATE")
    candidates = []
    for name, origins in sorted(paths.items()):
        if len(candidates) >= 64:
            errors.append("candidate count exceeded discovery budget")
            break
        row = {"path": name, "discovered_by": sorted(origins), "status": "UNSUPPORTED"}
        try:
            _, evidence = extract_torchair_descriptor(name)
            row.update(status="LITERAL_DESCRIPTOR", **evidence)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            row["error"] = f"{type(error).__name__}: {str(error)[:256]}"
        candidates.append(row)
    return {"schema_version": 1, "artifact_kind": "qwen35-torchair-schema-discovery",
            "status": "FOUND" if candidates else "NOT_FOUND", "candidates": candidates,
            "errors": errors, "package_import": "NOT_PERFORMED",
            "scope": "Read-only metadata and fixed-path AST inspection; schema-to-user-AIR agreement is not established"}


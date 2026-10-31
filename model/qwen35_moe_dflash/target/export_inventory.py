"""Read-only raw target AIR/FileConstant inventory for cross-machine reuse.

Only AIR/schema metadata and file stats are read by default. Payload SHA256
requires an explicit hash_weights request, streams at most 1 MiB per read, and
does not copy, convert or stage any payload.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from qwen35_moe_dflash.toolchain.air_schema import discover_torchair_schemas, load_torchair_schema
from qwen35_moe_dflash.common.artifacts import output_path, save_json


MAX_AIR_BYTES = 64 << 20
HASH_CHUNK_BYTES = 1 << 20


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _checked_path(path, *, directory=False):
    path = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:], 1):
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("symlink in source path is not allowed: " + str(current))
        last = index == len(path.parts) - 1
        expected = stat.S_ISDIR if not last or directory else stat.S_ISREG
        if not expected(info.st_mode):
            raise ValueError("source path is not a regular file/directory: " + str(current))
    return path, _identity(path.stat())


def _hash_file(path, expected):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != expected:
            raise ValueError("source file changed before hashing: " + path.name)
        digest = hashlib.sha256()
        remaining = expected[2]
        while remaining:
            block = stream.read(min(HASH_CHUNK_BYTES, remaining))
            if not block:
                raise ValueError("source file truncated during hashing: " + path.name)
            digest.update(block)
            remaining -= len(block)
        if stream.read(1):
            raise ValueError("source file grew during hashing: " + path.name)
    _, after = _checked_path(path)
    if after != expected:
        raise ValueError("source file changed during hashing: " + path.name)
    return digest.hexdigest()


def _select_schema(pb2, discover):
    if pb2:
        return load_torchair_schema(pb2)
    if not discover:
        raise ValueError("declare --torchair-pb2 or explicitly request --discover-torchair-schema")
    found = discover_torchair_schemas()
    candidates = [row for row in found.get("candidates", []) if row.get("status") == "LITERAL_DESCRIPTOR"]
    if not candidates:
        raise ValueError("no literal TorchAir schema found; select the generated ge_ir_by_protoc_*_pb2.py")
    hashes = {row["descriptor_sha256"] for row in candidates}
    if len(hashes) != 1:
        raise ValueError("multiple TorchAir descriptor hashes found; explicit --torchair-pb2 is required")
    selected = min(candidates, key=lambda row: (row["pb2_bytes"], row["path"]))
    schema = load_torchair_schema(selected["path"])
    if schema.evidence["descriptor_sha256"] not in hashes:
        raise ValueError("discovered descriptor changed before selection")
    schema.evidence["selection"] = "UNIQUE_DISCOVERED_DESCRIPTOR_HASH_SMALLEST_LITERAL_SOURCE"
    return schema


def _attrs(node):
    result = {}
    for row in node.get("attr", []):
        key = row.get("key")
        if not isinstance(key, str) or key in result:
            raise ValueError("missing or duplicate AIR attribute key")
        result[key] = row.get("value", {})
    return result


def _file_constants(value):
    if isinstance(value, dict):
        if value.get("type") == "FileConstant":
            yield value
        for item in value.values():
            yield from _file_constants(item)
    elif isinstance(value, list):
        for item in value:
            yield from _file_constants(item)


def _read_air(air):
    source, before = _checked_path(air)
    if not 0 < before[2] <= MAX_AIR_BYTES:
        raise ValueError("raw AIR must be nonempty and at most 64 MiB")
    with source.open("rb") as stream:
        raw = stream.read(MAX_AIR_BYTES + 1)
    if len(raw) != before[2] or _checked_path(source)[1] != before:
        raise ValueError("AIR changed during bounded read")
    return source, before, raw


def _decode_graph(raw, schema):
    model = schema.decode(raw, "ModelDef")
    graphs = model.get("graph")
    if not isinstance(graphs, list) or len(graphs) != 1:
        raise ValueError("one strict raw ModelDef graph is required")
    return model, graphs[0]


def _shape(descriptor):
    # An absent ShapeDef is unknown metadata. An explicitly present empty
    # ShapeDef has no dimensions; its producer may still be unmaterialized.
    shape = descriptor.get("shape")
    return shape.get("dim", []) if isinstance(shape, dict) else None


def _descriptor_observation(descriptor, types):
    raw_dtype = descriptor.get("dtype")
    return {"raw_dtype": raw_dtype,
            "dtype_names": sorted(name for name, value in types.items() if value == raw_dtype),
            "shape": _shape(descriptor), "layout": descriptor.get("layout")}


def _interface_observation(graph, schema, sequence_length):
    """Retain actual interface metadata before any expected-ABI gate can fail."""
    ops = graph.get("op", [])
    if not isinstance(ops, list) or not 1 <= len(ops) <= 32768:
        raise ValueError("AIR root operation list is missing or excessive")
    by_name = {}
    for node in ops:
        name = node.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("AIR operation name missing or duplicated")
        by_name[name] = node
    outputs = [node for node in ops if node.get("type") == "NetOutput"]
    package = schema.evidence["schema_package"]
    enum_name = package + ".DataType"
    types = schema.enums.get(enum_name, {})
    observation = {"input_count": sum(node.get("type") in {"Data", "RefData"} for node in ops),
                   "netoutput_count": len(outputs), "dtype_enum": enum_name,
                   "expected_feature": {"raw_dtype": types.get("DT_FLOAT16"),
                       "dtype_name": "DT_FLOAT16", "shape": [1, sequence_length, 16384], "layout": "ND"},
                   "output_columns": ["index", "producer", "raw_dtype", "shape", "layout"],
                   "outputs": []}
    if len(outputs) != 1:
        return observation
    net = outputs[0]
    references = [value for value in net.get("input", []) if not value.endswith(":-1")]
    descs = net.get("input_desc", [])
    observation.update(netoutput=net["name"], output_reference_count=len(references),
                       output_descriptor_count=len(descs))
    # Compact rows keep a normal 82-port report small without abbreviating or
    # truncating any producer reference or declared shape.
    for index in range(max(len(references), len(descs))):
        descriptor = descs[index] if index < len(descs) else {}
        observation["outputs"].append([index, references[index] if index < len(references) else None,
            descriptor.get("dtype"), _shape(descriptor), descriptor.get("layout")])
    index = 81
    reference = references[index] if index < len(references) else None
    descriptor = descs[index] if index < len(descs) else {}
    tail = {"index": index, "producer": reference,
            "declared": _descriptor_observation(descriptor, types), "differences": []}
    expected = observation["expected_feature"]
    for key in ("raw_dtype", "shape", "layout"):
        if tail["declared"][key] != expected[key]:
            tail["differences"].append({"field": key, "expected": expected[key],
                                        "observed": tail["declared"][key]})
    if isinstance(reference, str):
        name, separator, port = reference.rpartition(":")
        producer = by_name.get(name)
        if separator and port.isdecimal() and producer is not None:
            descriptors = producer.get("output_desc", [])
            if int(port) < len(descriptors):
                tail["producer_descriptor"] = _descriptor_observation(descriptors[int(port)], types)
    observation["feature_output"] = tail
    return observation


def _boundary(graph, schema, sequence_length):
    ops = graph.get("op", [])
    if not isinstance(ops, list) or not 1 <= len(ops) <= 32768:
        raise ValueError("AIR root operation list is missing or excessive")
    by_name = {}
    for node in ops:
        name = node.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("AIR operation name missing or duplicated")
        by_name[name] = node
    inputs = [node for node in ops if node.get("type") in {"Data", "RefData"}]
    outputs = [node for node in ops if node.get("type") == "NetOutput"]
    if len(outputs) != 1:
        raise ValueError("exactly one original NetOutput is required")
    net = outputs[0]
    references = [value for value in net.get("input", []) if not value.endswith(":-1")]
    descs = net.get("input_desc", [])
    if len(inputs) != 89 or len(references) != 82 or len(descs) != 82:
        raise ValueError("new A target must declare exactly 89 inputs and 82 outputs")
    for reference in references:
        name, separator, port = reference.rpartition(":")
        if not separator or not port.isdecimal() or name not in by_name:
            raise ValueError("NetOutput producer reference is not explicit")
        if int(port) >= len(by_name[name].get("output_desc", [])):
            raise ValueError("NetOutput producer port is missing")
    types = schema.enums[schema.evidence["schema_package"] + ".DataType"]
    expected_dtype = types.get("DT_FLOAT16")
    tail = descs[-1]
    if (expected_dtype is None or tail.get("dtype") != expected_dtype
            or tail.get("shape", {}).get("dim") != [1, sequence_length, 16384] or tail.get("layout") != "ND"):
        actual = _descriptor_observation(tail, types)
        raise ValueError(f"feature output 81 must declare DT_FLOAT16={expected_dtype} ND "
            f"[1,{sequence_length},16384]; observed raw_dtype={actual['raw_dtype']}, "
            f"dtype_names={actual['dtype_names']}, shape={actual['shape']}, layout={actual['layout']!r}")
    name, _, port = references[-1].rpartition(":")
    producer = by_name[name]["output_desc"][int(port)]
    shape = producer.get("shape", {}).get("dim", [])
    if shape and shape != tail["shape"]["dim"]:
        raise ValueError("known feature producer shape contradicts NetOutput")
    if producer.get("dtype") != expected_dtype or producer.get("layout") != tail.get("layout"):
        raise ValueError("feature producer dtype/layout contradicts NetOutput")
    return {"status": "MATCHED_INTERFACE_METADATA", "input_count": 89, "output_count": 82,
            "feature_output_index": 81, "feature_shape": [1, sequence_length, 16384], "feature_dtype": "DT_FLOAT16",
            "feature_raw_dtype": expected_dtype, "feature_producer": references[-1],
            "producer_shape_status": "NOT_MATERIALIZED_OR_SCALAR_AMBIGUOUS" if not shape else "MATCHED",
            "producer_descriptor": producer, "feature_descriptor": tail,
            "execution": "NOT_RUN", "scope": "Declared interface metadata only; no producer shape inference or numerical validation"}


def inspect_target_export_interface(air, torchair_pb2=None, discover_schema=False, sequence_length=64):
    """Return AIR boundary observations without output files or payload access.

    INSPECTED describes completion of this read-only inspection. The separate
    interface_validation field retains the strict 89/82/FP16-ND ABI result.
    """
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-export-interface-inspection",
              "status": "BLOCKED", "interface_validation": "NOT_EVALUATED",
              "weight_payload_access": "NONE", "model_execution": "NOT_RUN",
              "numerical_validation": "NOT_RUN"}
    source, before = None, None
    try:
        if type(sequence_length) is not int or not 1 <= sequence_length <= 64:
            raise ValueError("sequence_length must be an integer in 1..64")
        source, before, raw = _read_air(air)
        result["air"] = {"path": str(source), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        schema = _select_schema(torchair_pb2, discover_schema)
        result["schema"] = {key: schema.evidence[key] for key in
                            ("schema_package", "descriptor_sha256", "pb2")}
        _, graph = _decode_graph(raw, schema)
        result["interface_observation"] = _interface_observation(graph, schema, sequence_length)
        try:
            _boundary(graph, schema, sequence_length)
        except ValueError as error:
            result.update(interface_validation="DIFFERENT", reason=str(error))
        else:
            result["interface_validation"] = "MATCHED_INTERFACE_METADATA"
        result["status"] = "INSPECTED"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if source is not None and "air" in result:
            try:
                same = (_checked_path(source)[1] == before
                        and _hash_file(source, before) == result["air"]["sha256"])
            except (OSError, ValueError):
                same = False
            result["air_preservation"] = "UNCHANGED_STAT_AND_SHA256" if same else "CHANGED"
            if not same:
                result.update(status="BLOCKED", interface_validation="NOT_VERIFIED",
                              error="source AIR changed during inspection")
    return result


def inventory_target_export(air, out, torchair_pb2=None, discover_schema=False,
                            hash_weights=False, reference_root=None, sequence_length=64, progress=None):
    """Write one new small manifest; never copy AIR or external payloads."""
    out = output_path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-export-inventory",
              "status": "BLOCKED", "weight_identity": "NOT_VERIFIED", "files": [],
              "payload_copies": "NONE", "payload_reads": "EXPLICIT_STREAM_HASH_ONLY" if hash_weights else "NONE",
              "model_execution": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
              "scope": "Raw AIR and external-file identity inventory only; no reuse compatibility or model accuracy pass"}
    checked_files, source, source_before = {}, None, None
    try:
        if type(sequence_length) is not int or not 1 <= sequence_length <= 64:
            raise ValueError("sequence_length must be an integer in 1..64")
        source, source_before, raw = _read_air(air)
        result["air"] = {"path": str(source), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        schema = _select_schema(torchair_pb2, discover_schema)
        result["schema"] = schema.evidence
        model, graph = _decode_graph(raw, schema)
        result["interface_observation"] = _interface_observation(graph, schema, sequence_length)
        result["declared_abi"] = _boundary(graph, schema, sequence_length)
        base, base_before = _checked_path(reference_root or source.parent.parent, directory=True)
        result["reference_root"] = str(base)
        constants = list(_file_constants(model))
        if not constants:
            raise ValueError("raw AIR contains no FileConstant references")
        by_relative, prefixes, metadata = {}, set(), []
        for node in constants:
            attributes = _attrs(node)
            reference = attributes.get("file_path", {}).get("s")
            if not isinstance(reference, str) or not reference or "\\" in reference or "\x00" in reference:
                raise ValueError("FileConstant must expose an explicit POSIX file_path")
            relative = PurePosixPath(reference)
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
                raise ValueError("FileConstant reference must be relative prefix/basename without parent traversal")
            normalized = str(relative)
            prefixes.add(relative.parts[0])
            row = by_relative.setdefault(normalized, {"reference": reference, "references": [],
                "relative_path": normalized, "basename": relative.name, "nodes": []})
            if reference not in row["references"]:
                row["references"].append(reference)
            row["nodes"].append(node["name"])
            metadata.append({"name": node["name"], "reference": reference,
                             "raw_attributes": node.get("attr", []),
                             "raw_output_desc": node.get("output_desc", []),
                             "dtype_domains": "Attribute encoding retained independently of schema descriptor dtype"})
        if len(prefixes) != 1:
            raise ValueError("FileConstant references do not share one relative prefix")
        result["source_relative_prefix"] = prefixes.pop()
        result["file_constants"] = metadata
        result["file_constant_count"] = len(constants)
        hash_cache = {}
        for index, (relative, row) in enumerate(sorted(by_relative.items()), 1):
            path, before = _checked_path(base / relative)
            if base not in path.parents:
                raise ValueError("FileConstant escaped the explicit reference root")
            row.update(bytes=before[2], sha256=None, stat=list(before))
            checked_files[path] = before
            if hash_weights:
                if before not in hash_cache:
                    hash_cache[before] = _hash_file(path, before)
                row["sha256"] = hash_cache[before]
            result["files"].append(row)
            if progress:
                progress({"completed": index, "total": len(by_relative), "basename": row["basename"],
                          "bytes": row["bytes"], "payload_hashed": bool(hash_weights)})
        result.update(status="INVENTORIED", weight_identity="SHA256_COMPLETE" if hash_weights else "STAT_ONLY",
                      unique_reference_count=len(result["files"]),
                      total_weight_bytes=sum(row["bytes"] for row in result["files"]),
                      hashed_unique_physical_files=len(hash_cache), root_op_count=len(graph["op"]))
        # Directory mtime changes are recorded only as an observation. Every
        # referenced regular file and AIR must retain their own identities.
        result["reference_root_identity"] = {"stat_before": list(base_before),
                                            "stat_after": list(_checked_path(base, directory=True)[1])}
    except Exception as error:
        result.update(status="BLOCKED", error=f"{type(error).__name__}: {error}")
    finally:
        changed = []
        for path, before in checked_files.items():
            try:
                same = _checked_path(path)[1] == before
            except (OSError, ValueError):
                same = False
            if not same:
                changed.append(str(path.relative_to(Path(result["reference_root"]))))
        if source is not None and "air" in result:
            try:
                same = (_checked_path(source)[1] == source_before
                        and _hash_file(source, source_before) == result["air"]["sha256"])
            except (OSError, ValueError):
                same = False
            result["air_preservation"] = "UNCHANGED_STAT_AND_SHA256" if same else "CHANGED"
            if not same:
                changed.append("AIR")
        result["source_preservation"] = ("UNCHANGED_BY_STAT" if checked_files or "air" in result else
                                         "NOT_ESTABLISHED") if not changed else "CHANGED"
        if changed:
            result.update(status="BLOCKED", weight_identity="NOT_VERIFIED", changed_sources=changed,
                          error="source AIR or payload metadata changed during inventory")
        save_json(out / "inventory.json", result)
    return result

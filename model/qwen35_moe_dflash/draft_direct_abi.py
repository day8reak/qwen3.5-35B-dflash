"""Inspect the physical byte ABI of ordered direct diagnostic outputs.

The selected compiler schema defines dtype numbers. This reader inspects only
the small external OMC's main ModelDef and does not execute vendor code or
interpret opaque NPU tasks. The caller must bind the compiled AIR snapshot to
the declared direct specification before using the ordered output contract.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

from .draft_compile_precision import _identity, _observed_desc, _read, _weight_info_trailer
from .draft_ir_inspect import _enum_fields
from .omc_container import read_container
from .omc_wire import load_schema


KIND = "qwen35-draft-direct-output-abi-inspection"
CONTRACT = "cpro-draft-direct-ordered-output-byte-abi-v1"
_MAX_OUTPUT_BYTES = 64 << 20
_SOURCE_BINDING = "ORDERED_COMPILED_OUTPUTS_CALLER_MUST_BIND_AIR_AND_DECLARED_SPEC"
_SCOPE = ("Physical direct-output decoding only; SOURCE_DTYPE_CHANGED permits an explicitly reported "
          "extra FP16-to-FP32 diagnostic output, not source mixed-precision equivalence, "
          "snapshot correctness, device accuracy or production promotion")


def _declared(spec):
    if not isinstance(spec, list) or not 2 <= len(spec) <= 256:
        raise ValueError("direct output specification must contain 2..256 semantic rows")
    names, sources, outputs = set(), {}, {}
    for row in spec:
        if not isinstance(row, dict) or set(row) != {"name", "source", "shape", "dtype", "output_index"}:
            raise ValueError("direct output specification row fields differ")
        if (not isinstance(row["name"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", row["name"])
                or row["name"] in names or not isinstance(row["source"], str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./-]*", row["source"])
                or not isinstance(row["shape"], list) or not 1 <= len(row["shape"]) <= 8
                or any(type(dim) is not int or dim <= 0 for dim in row["shape"])
                or row["dtype"] not in {"float16", "float32", "int32"}
                or type(row["output_index"]) is not int or not 0 <= row["output_index"] < 16):
            raise ValueError("direct output specification has an invalid name, index or descriptor")
        names.add(row["name"])
        index = row["output_index"]
        descriptor = {key: row[key] for key in ("source", "shape", "dtype")}
        if index in outputs and outputs[index] != descriptor:
            raise ValueError("semantic aliases disagree about their shared direct output")
        if row["source"] in sources and sources[row["source"]] != index:
            raise ValueError("one original direct value cannot occupy two physical output indices")
        sources[row["source"]] = index
        outputs[index] = descriptor
    if set(outputs) != set(range(len(outputs))) or len(outputs) < 2:
        raise ValueError("direct physical output indices must be contiguous from zero")
    semantic = {row["name"]: row for row in spec}
    for name, index, dtype in (("draft_top1", 0, "int32"), ("draft_hidden", 1, "float16")):
        row = semantic.get(name)
        if row is None or (row["source"], row["output_index"], row["dtype"]) != (name, index, dtype):
            raise ValueError("direct Top1/hidden names, sources, indices and dtypes must stay fixed")
    return [outputs[index] for index in range(len(outputs))]


def _ordered_edges(node, operations, count, enums, descriptors):
    refs = node.get("input")
    if not isinstance(refs, list):
        raise ValueError("compiled NetOutput must have explicit ordered input references")
    edges = []
    for reference in refs:
        if not isinstance(reference, str) or ":" not in reference:
            raise ValueError("compiled NetOutput input reference is unresolved")
        name, raw = reference.rsplit(":", 1)
        if not name or not re.fullmatch(r"-?\d+", raw):
            raise ValueError("compiled NetOutput input reference is malformed")
        index = int(raw)
        if index == -1:
            continue
        if index < 0 or name not in operations:
            raise ValueError("compiled NetOutput producer is missing or has an invalid port")
        edges.append((name, index))
    if len(edges) != count:
        raise ValueError("compiled NetOutput ordered inputs differ from its descriptor count")
    result = []
    for index, (name, port) in enumerate(edges):
        source = operations[name]
        outputs = source.get("output_desc")
        if not isinstance(outputs, list) or port >= len(outputs):
            raise ValueError("compiled NetOutput producer output descriptor is missing")
        observed = _observed_desc(outputs[port], enums)
        if any(observed[key] != descriptors[index][key] for key in ("dtype", "shape")):
            raise ValueError("compiled NetOutput and its physical producer disagree on dtype/shape")
        result.append({"output_index": index, "producer": name, "producer_output_index": port,
                       "producer_type": source.get("type")})
    return result


def inspect_direct_output_abi(omc, schema, requested_spec):
    """Return PASSED, SOURCE_DTYPE_CHANGED, or FAILED with hash-bound evidence.

    Only additional FP16 outputs may be physically promoted to FP32. Shape,
    count, Top1/hidden dtype and all other dtype changes are hard failures.
    ``observed_spec`` changes dtype only and is never returned on failure.
    """
    report = {"schema_version": 1, "artifact_kind": KIND, "contract": CONTRACT,
              "status": "FAILED", "declared_spec": deepcopy(requested_spec),
              "observed_spec": None, "changes": [], "source_binding": _SOURCE_BINDING,
              "scope": _SCOPE}
    try:
        declared = _declared(requested_spec)
        raw, identity = _read(omc)
        path = Path(omc).absolute()
        digest = hashlib.sha256(raw).hexdigest()
        report["omc"] = {"path": path.name, "bytes": len(raw), "sha256": digest}
        container = read_container(path)
        if _identity(path.stat()) != identity:
            raise ValueError("direct OMC changed during container inspection")
        parts = container["partitions"]
        if [part["type"] for part in parts] not in ([0, 1, 2], [0, 1, 2, 8]) or parts[1]["bytes"] != 0:
            raise ValueError("direct ABI inspection requires a small external OMC container")
        cursor = 0
        for part in parts:
            if part["table_word"] != cursor:
                raise ValueError("direct OMC partition table has unsupported offset semantics")
            cursor += part["bytes"]
        if len(parts) == 4:
            trailer = parts[3]
            _weight_info_trailer(memoryview(raw)[trailer["offset"]:trailer["offset"] + trailer["bytes"]])
        selected = load_schema(schema)
        report["schema"] = selected.evidence
        enums = _enum_fields(selected)
        main = parts[0]
        model = selected.decode(memoryview(raw)[main["offset"]:main["offset"] + main["bytes"]], "ModelDef")
        graphs = model.get("graph")
        if not isinstance(graphs, list) or len(graphs) != 1:
            raise ValueError("direct OMC must contain exactly one main graph")
        nodes = graphs[0].get("op")
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 8192:
            raise ValueError("direct main operation count is unsupported")
        operations = {}
        for node in nodes:
            name = node.get("name") if isinstance(node, dict) else None
            if not isinstance(name, str) or not name or name in operations:
                raise ValueError("direct main operation names are missing or ambiguous")
            operations[name] = node
        netouts = [node for node in nodes if node.get("type") == "NetOutput"]
        if len(netouts) != 1:
            raise ValueError("direct main graph must have one unambiguous NetOutput")
        output = netouts[0]
        descs = output.get("input_desc")
        if not isinstance(descs, list) or len(descs) != len(declared):
            raise ValueError("compiled direct output count differs from declared unique outputs")
        descriptors = [_observed_desc(desc, enums) for desc in descs]
        connections = _ordered_edges(output, operations, len(declared), enums, descriptors)
        changes, observed_types, total = [], [], 0
        for index, (expected, actual) in enumerate(zip(declared, descriptors)):
            dtype = actual["dtype"]
            if dtype not in {"float16", "float32", "int32"}:
                raise ValueError("compiled direct output has an unknown or unsupported schema dtype")
            if actual["shape"] != expected["shape"]:
                raise ValueError("compiled direct output shape differs at index " + str(index))
            if dtype != expected["dtype"]:
                if index < 2 or expected["dtype"] != "float16" or dtype != "float32":
                    raise ValueError("compiled direct output has a forbidden dtype change at index " + str(index))
                changes.append({"kind": "DIAGNOSTIC_OUTPUT_PROMOTED_FP16_TO_FP32", "output_index": index,
                    "source": expected["source"], "semantic_names": [row["name"] for row in requested_spec if row["output_index"] == index],
                    "declared_dtype": expected["dtype"], "observed_dtype": dtype, "shape": list(expected["shape"])})
            total += math.prod(expected["shape"]) * (2 if dtype == "float16" else 4)
            if total > _MAX_OUTPUT_BYTES:
                raise ValueError("compiled direct outputs exceed the 64 MiB physical byte budget")
            observed_types.append(dtype)
        if _identity(path.stat()) != identity or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("direct OMC changed during output ABI inspection")
        if hashlib.sha256(Path(schema).read_bytes()).hexdigest() != selected.evidence["library_sha256"]:
            raise ValueError("selected direct-output schema changed during inspection")
        observed_spec = [{**row, "dtype": observed_types[row["output_index"]]} for row in deepcopy(requested_spec)]
        report.update(status="SOURCE_DTYPE_CHANGED" if changes else "PASSED", observed_spec=observed_spec,
                      changes=changes, physical_output_bytes=total, output_connections=connections,
                      ordered_output_count=len(declared), source_output_precision="SOURCE_DTYPE_CHANGED" if changes else "SOURCE_DTYPES_PRESERVED")
        return report
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        report.update(status="FAILED", observed_spec=None, changes=[], reason=f"{type(error).__name__}: {str(error)[:1600]}")
        return report

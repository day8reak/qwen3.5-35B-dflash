"""Compose static Reshape edges touched by the optional row-Mul lowering.

This pure record transform changes no arithmetic and does not access external
weights. Unknown or malformed reshape contracts remain for the normal graph
validator/compiler; they are never repaired or bypassed here.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral
import struct


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _shape(port):
    shape = port.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    if any(not _integer(dim) or dim <= 0 for dim in shape):
        return None
    return tuple(int(dim) for dim in shape)


def _source(records, port):
    outputs = records[port["src"]].get("outputs", [])
    matches = [output for output in outputs if output.get("port") == port.get("src_port")]
    return matches[0] if len(matches) == 1 else None


def _same_descriptor(left, right):
    return (right is not None and _shape(left) is not None
            and _shape(left) == _shape(right) and left.get("dtype") == right.get("dtype")
            and left.get("format", "ND") == right.get("format", "ND"))


def _reshape_input(records, index):
    """Return a verified data port, or None without crossing an unknown view."""
    record = records[index]
    if record.get("type") != "Reshape":
        return None
    attrs = record.get("attrs", {})
    if not isinstance(attrs, dict) or set(attrs) != {"axis", "num_axes"}:
        return None
    for key, expected in (("axis", 0), ("num_axes", -1)):
        value = attrs[key]
        if (not isinstance(value, dict) or value.get("type") != "int"
                or not _integer(value.get("value")) or value["value"] != expected):
            return None
    inputs, outputs = record.get("inputs", []), record.get("outputs", [])
    if len(inputs) != 2 or len(outputs) != 1:
        return None
    ports = {port.get("port"): port for port in inputs}
    if set(ports) != {"x", "shape"}:
        return None
    data, shape_port, output = ports["x"], ports["shape"], outputs[0]
    # This transform only composes ordinary ND views. A format-changing view
    # can be a required boundary, e.g. the NCHW Conv2D result restored to ND
    # before RMSNorm; bypassing it would change the connected descriptor.
    if any(port.get("format", "ND") != "ND" for port in (data, shape_port, output)):
        return None
    source_desc, literal_desc = _source(records, data), _source(records, shape_port)
    if not _same_descriptor(data, source_desc) or not _same_descriptor(shape_port, literal_desc):
        return None
    output_shape = _shape(output)
    if (output_shape is None or data.get("dtype") not in {"float16", "int32", "int64", "bool"}
            or output.get("dtype") != data.get("dtype")
            or math.prod(output_shape) != math.prod(_shape(data))):
        return None
    literal = records[shape_port["src"]]
    if (literal.get("type") != "Const" or literal.get("inputs")
            or len(literal.get("outputs", [])) != 1 or shape_port.get("dtype") != "int32"
            or _shape(shape_port) != (len(output_shape),)):
        return None
    constant = literal.get("constant")
    if not isinstance(constant, dict) or set(constant) != {"data"}:
        return None
    raw = constant["data"]
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != 4 * len(output_shape):
        return None
    if struct.unpack("<" + "i" * len(output_shape), raw) != output_shape:
        return None
    return data


def compact_row_mul_reshape_chains(records, input_indices, output_pairs, row_mul_names):
    """Return copied records/boundaries and evidence for seed-scoped rewrites.

    ``row_mul_names`` contains original Mul output names, such as ``v83``.
    Its two added input views and output restoration are seeds; valid Reshape
    descendants inherit that scope. Parents with other users or graph outputs
    survive pruning. An outer view always keeps its name and output contract.
    """
    copied = deepcopy(records)
    inputs = list(input_indices)
    outputs = [tuple(pair) for pair in output_pairs]
    if isinstance(row_mul_names, (str, bytes)):
        raise ValueError("row_mul_names must be a sequence of original output names")
    names = list(row_mul_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("row Mul output names must be nonempty strings")
    # Only basic topology is required here. Descriptor/attribute failures are
    # intentionally left untouched rather than made valid by a rewrite.
    for index, record in enumerate(copied):
        for port in record.get("inputs", []):
            source = port.get("src")
            if not _integer(source) or not 0 <= source < index:
                raise ValueError("record inputs must reference earlier records")
    for index in inputs:
        if not _integer(index) or not 0 <= index < len(copied):
            raise ValueError("invalid graph input record index")
    for pair in outputs:
        if (len(pair) != 2 or not _integer(pair[0]) or not 0 <= pair[0] < len(copied)
                or not _integer(pair[1]) or not 0 <= pair[1] < len(copied[pair[0]].get("outputs", []))):
            raise ValueError("invalid graph output record/port index")
    seeds = {seed for name in names for seed in
             (name, f"__ge/{name}/row_mul/left", f"__ge/{name}/row_mul/right")}
    touched = set()
    rewrites = []
    for index, record in enumerate(copied):
        data = _reshape_input(copied, index)
        if data is None or (record.get("name") not in seeds and data["src"] not in touched):
            continue
        touched.add(index)
        old_source, bypassed = data["src"], []
        ancestor = _reshape_input(copied, data["src"])
        while ancestor is not None:
            bypassed.append(copied[data["src"]]["name"])
            for key in ("src", "src_port", "shape", "dtype"):
                data[key] = deepcopy(ancestor[key])
            ancestor = _reshape_input(copied, data["src"])
        if bypassed:
            rewrites.append({"name": record["name"], "outer": record["name"], "old_source": records[old_source]["name"],
                             "new_source": copied[data["src"]]["name"], "bypassed": bypassed})
    report = {"counts": {"rewrites": len(rewrites), "pruned": 0, "rewired_inputs": len(rewrites),
                          "pruned_reshapes": 0, "pruned_constants": 0},
              "rewrites": rewrites, "pruned": []}
    if not rewrites:
        return copied, inputs, outputs, report
    live, pending = set(), [*inputs, *(index for index, _ in outputs)]
    while pending:
        index = pending.pop()
        if index in live:
            continue
        live.add(index)
        pending.extend(port["src"] for port in copied[index].get("inputs", []))
    remap = {old: new for new, old in enumerate(index for index in range(len(copied)) if index in live)}
    kept = []
    for index, record in enumerate(copied):
        if index not in live:
            report["pruned"].append({"name": record["name"], "type": record["type"]})
            continue
        for port in record.get("inputs", []):
            port["src"] = remap[port["src"]]
        kept.append(record)
    report["counts"]["pruned"] = len(report["pruned"])
    report["counts"]["pruned_reshapes"] = sum(item["type"] == "Reshape" for item in report["pruned"])
    report["counts"]["pruned_constants"] = sum(item["type"] == "Const" for item in report["pruned"])
    return kept, [remap[index] for index in inputs], [(remap[index], port) for index, port in outputs], report

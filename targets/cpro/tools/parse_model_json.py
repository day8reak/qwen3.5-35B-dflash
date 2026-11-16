#!/usr/bin/env python3
"""Inspect OMG model JSON and print candidate I/O flags. Standard library only.

Graph descriptors/index attributes are not a query of HIAI runtime slots.
Numeric dtype values require an explicitly selected enum; no GE/HIAI guessing.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import sys


# ge_ir.proto TensorDescriptor.dtype enum; NOT ge::DataType or hiai::DataType.
# Source: repo specs/torchair-ge-ir.pb; symbolic names also work without this map.
GE_PROTO = {1: "FP32", 2: "FP16", 3: "INT8", 4: "UINT8", 5: "INT16",
            6: "UINT16", 7: "INT32", 8: "INT64", 9: "UINT32", 10: "UINT64",
            11: "BOOL", 12: "DOUBLE", 27: "BF16"}
WIDTHS = {"FP32": 4, "FP16": 2, "INT8": 1, "UINT8": 1, "INT16": 2,
          "UINT16": 2, "INT32": 4, "INT64": 8, "UINT32": 4, "UINT64": 8,
          "BOOL": 1, "DOUBLE": 8, "BF16": 2}
# Values supported by the user's model_run_tool_internal --help.
TOOL_TYPES = set(WIDTHS) - {"UINT16", "UINT64", "BF16"}
ALIASES = {"FLOAT": "FP32", "FLOAT32": "FP32", "FLOAT16": "FP16",
           "HALF": "FP16", "FLOAT64": "DOUBLE", "BFLOAT16": "BF16"}
NAME = re.compile(r"[A-Za-z_0-9./-]+\Z")
FIELDS = ("slot", "name", "node", "edge", "node_type", "serialized_index",
          "shape", "dtype", "dtype_raw", "layout", "logical_bytes", "issues")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 256 << 20:
        raise ValueError("JSON exceeds 256 MiB; select a metadata-only dump")
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def integer(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    raise ValueError("expected integer, got " + repr(value))


def attributes(node):
    attrs = node.get("attr", {})
    if isinstance(attrs, dict):
        return attrs
    if isinstance(attrs, list):
        return unique_object((entry["key"], entry["value"]) for entry in attrs)
    raise ValueError("node attr must be a map or key/value list")


def graphs(value, path="$"):
    """Stop at graph op arrays: their nested subgraphs are not host inputs."""
    if isinstance(value, dict):
        if isinstance(value.get("op"), list):
            yield path, value
            return
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from graphs(child, path + "/" + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from graphs(child, path + "/" + str(i))


def dtype(value, enum):
    if isinstance(value, str) and not re.fullmatch(r"-?[0-9]+", value):
        name = value.upper().removeprefix("DT_")
        name = ALIASES.get(name, name)
        return name if name in WIDTHS else None
    if value is not None:
        return enum.get(integer(value))
    return None


def descriptor(desc, enum):
    if not isinstance(desc, dict):
        raise ValueError("tensor descriptor must be an object")
    shape = desc.get("shape")
    dims = shape.get("dim") if isinstance(shape, dict) else shape
    dims = [integer(d) for d in dims] if isinstance(dims, list) and dims else None
    raw = desc.get("dtype", desc.get("dataType"))
    kind = dtype(raw, enum)
    issues = []
    if kind is None:
        issues.append("unresolved dtype; numeric values need --dtype-enum ge-proto or --dtype-map")
    elif kind not in TOOL_TYPES:
        issues.append("dtype absent from supplied model_run_tool --help: " + kind)
    if dims is None:
        issues.append("empty/missing shape; scalar versus unmaterialized shape is unknown")
    elif any(d < 0 for d in dims):
        issues.append("dynamic/unknown dimension; logical byte count is unavailable")
    return dict(shape=dims, dtype=kind, dtype_raw=raw, layout=desc.get("layout", desc.get("format")),
                logical_bytes=(math.prod(dims) * WIDTHS[kind]
                               if dims is not None and all(d >= 0 for d in dims) and kind else None),
                issues=issues)


def desc_list(node, direction):
    snake, camel = direction + "_desc", direction + "Desc"
    value = node.get(snake, node.get(camel, []))
    if not isinstance(value, list):
        raise ValueError(node["name"] + ": descriptors must be a list")
    return value


def edge_parts(edge):
    name, sep, port = edge.rpartition(":")
    if not sep:
        return edge, 0
    return name, integer(port)


def input_order(graph, rows, explicit, notes):
    names = {row["node"] for row in rows}
    if explicit is not None:
        order = explicit
        basis = "USER_SUPPLIED_NAMES"
    else:
        indices = [row["serialized_index"] for row in rows]
        indexed = (sorted(rows, key=lambda row: row["serialized_index"])
                   if all(i is not None for i in indices) and set(indices) == set(range(len(rows))) else None)
        graph_inputs = graph.get("input", [])
        ordered = []
        if isinstance(graph_inputs, list):
            for edge in graph_inputs:
                node, port = edge_parts(edge)
                if port != 0 or node not in names:
                    ordered = []
                    break
                ordered.append(node)
        valid = len(ordered) == len(names) and set(ordered) == names
        if indexed and valid and [row["node"] for row in indexed] != ordered:
            notes.append("Data.index conflicts with GraphDef.input; specify --input-order after checking runtime")
            return rows, "UNRESOLVED"
        if indexed:
            order, basis = [row["node"] for row in indexed], "CONTIGUOUS_DATA_INDEX"
        elif valid:
            order, basis = ordered, "GRAPH_INPUT_LIST"
        else:
            notes.append("input order unresolved; rows retain JSON order, and input_type is not emitted. "
                         "Use --input-order with actual ordered node names")
            return rows, "UNRESOLVED"
    if len(order) != len(names) or set(order) != names:
        raise ValueError("--input-order must contain every root Data/RefData name exactly once")
    by_name = {row["node"]: row for row in rows}
    return [by_name[name] for name in order], basis


def named_shapes(rows):
    names = [row["name"] for row in rows]
    if (any(not name or not NAME.fullmatch(name) for name in names)
            or len(set(names)) != len(names)
            or any(row["shape"] is None or any(d < -1 for d in row["shape"]) for row in rows)):
        return None
    return ";".join(row["name"] + ":" + ",".join(map(str, row["shape"])) for row in rows)


def type_list(rows):
    return ",".join(row["dtype"] for row in rows) if all(row["dtype"] in TOOL_TYPES for row in rows) else None


def inspect(document, *, enum=None, graph_name=None, ordered_inputs=None, output_names=None):
    enum = enum or {}
    choices = list(graphs(document))
    if graph_name:
        choices = [(path, graph) for path, graph in choices if graph_name in (path, graph.get("name"))]
    if len(choices) != 1:
        available = [(path, graph.get("name")) for path, graph in graphs(document)]
        raise ValueError("select one root graph with --graph; candidates: " + repr(available))
    path, graph = choices[0]
    nodes = graph["op"]
    by_name = unique_object((node["name"], node) for node in nodes)
    inputs, outputs, notes = [], [], []
    for node in nodes:
        if node.get("type") not in {"Data", "RefData"}:
            continue
        descs = desc_list(node, "output")
        if len(descs) != 1:
            raise ValueError(node["name"] + ": expected exactly one input output_desc")
        index = attributes(node).get("index")
        index = index.get("i") if isinstance(index, dict) else index
        inputs.append(dict(name=node["name"], node=node["name"], node_type=node["type"],
                           serialized_index=integer(index) if index is not None else None,
                           edge=node["name"] + ":0", **descriptor(descs[0], enum)))
    if not inputs:
        raise ValueError("no root Data/RefData inputs")
    inputs, basis = input_order(graph, inputs, ordered_inputs, notes)
    for i, row in enumerate(inputs):
        row["slot"] = i if basis != "UNRESOLVED" else None
    netoutputs = [node for node in nodes if node.get("type") == "NetOutput"]
    if len(netoutputs) != 1:
        raise ValueError("expected one NetOutput in selected root graph")
    net = netoutputs[0]
    descs = desc_list(net, "input")
    raw_edges = net.get("input", [])
    edges = [edge for edge in raw_edges if edge_parts(edge)[1] != -1]
    if not edges or (descs and len(descs) != len(edges)):
        raise ValueError("NetOutput data edges are missing or do not match input_desc count")
    if output_names is not None and len(output_names) != len(edges):
        raise ValueError("--output-names must contain one actual output tensor name per NetOutput port")
    for i, edge in enumerate(edges):
        if descs:
            desc, source = descs[i], "NETOUTPUT_INPUT_DESC"
        else:
            producer, port = edge_parts(edge)
            producer_descs = desc_list(by_name.get(producer, {"name": producer}), "output")
            if not 0 <= port < len(producer_descs):
                raise ValueError("NetOutput descriptor and producer descriptor unavailable: " + edge)
            desc, source = producer_descs[port], "PRODUCER_OUTPUT_DESC"
        name = output_names[i] if output_names is not None else desc.get("name")
        outputs.append(dict(slot=i, name=name, node=net["name"], node_type="NetOutput",
                            edge=edge, descriptor_source=source, **descriptor(desc, enum)))
    flags = dict(input_shape=named_shapes(inputs),
                 input_type=type_list(inputs) if basis != "UNRESOLVED" else None,
                 output_shape=named_shapes(outputs), output_type=type_list(outputs))
    if flags["output_shape"] is None:
        notes.append("output_shape unavailable: missing/duplicate/CLI-unsafe tensor names or shapes. "
                     "Shapes remain in output rows; --output-names accepts actual runtime names")
    notes.append("Graph input order and NetOutput port order are candidate CLI order, not queried HIAI runtime order. "
                 "Align input files with actual runtime slots before running")
    notes.append("logical_bytes excludes format padding, alignment and runtime buffer allocation")
    return dict(status="PARSED" if all(value is not None for value in flags.values()) else "PARTIAL",
                graph=graph.get("name"), graph_path=path, input_count=len(inputs), output_count=len(outputs),
                input_order_basis=basis, output_order_basis="NETOUTPUT_DATA_PORTS",
                runtime_order_verified=False, inputs=inputs, outputs=outputs, **flags, notes=notes)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row, shape=json.dumps(row["shape"]), issues="; ".join(row["issues"]))
            writer.writerow(row)


def render(report):
    lines = ["status=" + report["status"], f"graph={report['graph']} path={report['graph_path']}",
             "input_order_basis=" + report["input_order_basis"],
             "runtime_order_verified=false", ""]
    for kind in ("inputs", "outputs"):
        lines.append(kind.upper() + " (slot is a graph-derived candidate):")
        lines.append("slot\tname / edge\tdtype\tshape\tlogical_bytes\tData.index")
        for row in report[kind]:
            lines.append("\t".join(map(str, [row.get("slot"), row["name"] or row["edge"],
                          row["dtype"] or "UNRESOLVED(" + str(row["dtype_raw"]) + ")",
                          row["shape"], row["logical_bytes"], row.get("serialized_index")])))
            lines.extend("  " + issue for issue in row["issues"])
        lines.append("")
    for key in ("input_shape", "input_type", "output_shape", "output_type"):
        lines.append("--" + key + "=" + shlex.quote(report[key]) if report[key] is not None else key + "=UNRESOLVED")
    lines.extend("NOTE: " + note for note in report["notes"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_json", help="OMG --mode=1 JSON file")
    parser.add_argument("--output-dir", help="write io.json, inputs.csv, outputs.csv, summary.txt and args.sh")
    parser.add_argument("--graph", help="root graph name or printed JSON path when multiple graphs exist")
    parser.add_argument("--dtype-enum", choices=["ge-proto"], help="explicitly decode numeric TensorDescriptor dtype using ge_ir.proto")
    parser.add_argument("--dtype-map", help='explicit numeric dtype JSON map, e.g. {"1":"FP32","2":"FP16"}')
    parser.add_argument("--input-order", help="comma-separated actual ordered Data/RefData node names; no sorting")
    parser.add_argument("--output-names", help="comma-separated actual output tensor names in NetOutput port order")
    args = parser.parse_args()
    try:
        if args.dtype_enum and args.dtype_map:
            raise ValueError("choose --dtype-enum or --dtype-map, not both")
        enum = dict(GE_PROTO) if args.dtype_enum else {}
        if args.dtype_map:
            for raw, name in read_json(args.dtype_map).items():
                number, kind = integer(raw), dtype(name, {})
                if number in enum or kind is None:
                    raise ValueError("invalid/duplicate entry in dtype map: " + repr(raw))
                enum[number] = kind
        report = inspect(read_json(args.model_json), enum=enum, graph_name=args.graph,
                         ordered_inputs=args.input_order.split(",") if args.input_order else None,
                         output_names=args.output_names.split(",") if args.output_names else None)
        report["source_json"] = str(Path(args.model_json).resolve())
        report["numeric_dtype_mapping"] = args.dtype_map or args.dtype_enum or "UNSELECTED"
        summary = render(report)
        if args.output_dir:
            dest = Path(args.output_dir)
            dest.mkdir(parents=True, exist_ok=False)
            (dest / "io.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for key in ("inputs", "outputs"):
                write_csv(dest / (key + ".csv"), report[key])
            (dest / "summary.txt").write_text(summary, encoding="utf-8")
            shell = ["# Graph-derived candidates; verify runtime order and input files before use."]
            for key in ("input_shape", "input_type", "output_shape", "output_type"):
                shell.append(key.upper() + "=" + shlex.quote(report[key] or ""))
                if report[key] is None:
                    shell.append("# " + key.upper() + " unavailable; see summary.txt.")
            (dest / "args.sh").write_text("\n".join(shell) + "\n", encoding="utf-8")
        print(summary, end="")
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        print("parse-model-json: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

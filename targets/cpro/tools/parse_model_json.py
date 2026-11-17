#!/usr/bin/env python3
"""Inspect an OMC or OMG model JSON and print I/O flags. Standard library only.

Graph descriptors/index attributes are not a query of HIAI runtime slots.
OMC conversion uses the selected CPro DDK and its GE protobuf dtype enum.
Existing JSON input still requires --dtype-enum or --dtype-map for numbers.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


def omg_environment(ddk_root=None, omg_path=None):
    """Resolve the real OMG executable without importing the model package."""
    root = ddk_root
    if not root and not omg_path:
        root = next((os.environ[key] for key in ("CPRO_DDK_ROOT", "DDK_PATH", "TOOLCHAIN_HOME")
                     if os.environ.get(key)), None)
    binary = omg_path or (str(Path(root) / "tools/tools_omg/master/omg") if root else shutil.which("omg"))
    if not binary:
        raise ValueError("OMG not found; set CPRO_DDK_ROOT, pass --ddk-root, or pass --omg")
    binary = Path(binary).expanduser().resolve(strict=True)
    if not root:
        # A PATH entry may be the vendor wrapper; invoke master/omg directly.
        if binary.parent.name == "tools_omg" and (binary.parent / "master/omg").is_file():
            binary = (binary.parent / "master/omg").resolve()
        if binary.parent.name == "master" and binary.parent.parent.name == "tools_omg":
            root = binary.parents[3]
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("OMG is not an executable file: " + str(binary))
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if root:
        root = Path(root).expanduser().resolve(strict=True)
        master = root / "tools/tools_omg/master"
        # Do not let an explicitly declared server CANN override the CPro DDK.
        foreign_roots = [Path(env[key]).resolve() for key in
                         ("CANN_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if env.get(key)]
        def foreign(value):
            path = Path(value).resolve()
            return not path.is_relative_to(root) and any(path.is_relative_to(p) for p in foreign_roots)
        for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PATH"):
            if key in env:
                env[key] = os.pathsep.join(p for p in env[key].split(os.pathsep) if p and not foreign(p))
        for key in ("CANN_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH"):
            if env.get(key) and foreign(env[key]):
                env.pop(key)
        def prepend(key, paths):
            entries = [str(p) for p in paths if p.is_dir()] + env.get(key, "").split(os.pathsep)
            env[key] = os.pathsep.join(dict.fromkeys(p for p in entries if p))
        plugin = root / "tools/platform/kirinx90"
        prepend("LD_LIBRARY_PATH", [master / "lib64", plugin / "lib64", root / "ddk/tbe/lib64", root / "ddk/ascendcpp/lib64"])
        prepend("PYTHONPATH", [root / "ddk/ascendc/ops/impl", root / "tools/tools_ascendc/package/python",
                               root / "ddk/tbe/ops", plugin / "ops/impl"])
        prepend("PATH", [root / "ddk/ccec_compiler/bin", root / "tools/tools_ascendc/bisheng/bin"])
        env.update(DDK_PATH=str(root), TOOLCHAIN_HOME=str(root), HIAI_VERSION=str(master),
                   CCE_LIB_DIR=str(master / "lib64"), SOC_VERSION="kirinx90")
        if (root / "ddk/ascendcpp/conf").is_dir():
            env["KIRIN_CONF_JSON_PATH"] = str(root / "ddk/ascendcpp/conf")
    return binary, env


def read_omc(model, *, ddk_root=None, omg_path=None, timeout=120):
    """Convert in a disposable directory; link sidecars without copying weights."""
    model = Path(model).expanduser().resolve(strict=True)
    if not model.is_file():
        raise ValueError("OMC must be a file: " + str(model))
    omg, env = omg_environment(ddk_root, omg_path)
    temp_root = None
    if os.environ.get("AI_RUN_DIR"):
        temp_root = Path(os.environ["AI_RUN_DIR"]).expanduser() / "target/omc-inspection"
        temp_root.mkdir(parents=True, exist_ok=True)
    print("Converting OMC metadata with " + str(omg), file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix="omcio_", dir=temp_root) as directory:
        work = Path(directory).resolve()
        # Preserve names referenced by separated-weight bundles. Reserved names
        # belong to this conversion, so stale JSON/logs cannot become results.
        for entry in model.parent.iterdir():
            if entry.name in {"input.omc", "model.json", "omg.log"}:
                continue
            source = entry.resolve()
            if source == work or source in work.parents:
                continue
            (work / entry.name).symlink_to(source)
        (work / "input.omc").symlink_to(model)
        command = [str(omg), "--mode=1", "--om=input.omc", "--json=model.json"]
        log = work / "omg.log"
        problem = None
        with log.open("wb") as stream:
            try:
                result = subprocess.run(command, cwd=work, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
                if result.returncode:
                    problem = f"OMG exited with code {result.returncode}"
            except subprocess.TimeoutExpired:
                problem = f"OMG metadata conversion timed out after {timeout}s"
        converted = work / "model.json"
        if not problem and (converted.is_symlink() or not converted.is_file()):
            problem = "OMG succeeded but did not produce model.json"
        if problem:
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 12000))
                tail = stream.read().decode("utf-8", errors="replace")
            raise ValueError(problem + "\n" + tail)
        document = read_json(converted)
    return document, dict(source_omc=str(model), source_kind="OMC",
                          conversion_omg=str(omg), temporary_json_removed=True)


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
FIELDS = ("slot", "name", "name_source", "node", "edge", "node_type", "serialized_index", "graph_input_slot",
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


def input_order(graph, rows, explicit, source, notes):
    names = {row["node"] for row in rows}
    candidates = {"json-order": [row["node"] for row in rows]}
    indices = [row["serialized_index"] for row in rows]
    if all(i is not None for i in indices) and set(indices) == set(range(len(rows))):
        candidates["data-index"] = [row["node"] for row in sorted(rows, key=lambda row: row["serialized_index"])]
    graph_inputs = graph.get("input", [])
    ordered = []
    if isinstance(graph_inputs, list):
        for edge in graph_inputs:
            if not isinstance(edge, str):
                break
            node, port = edge_parts(edge)
            if port != 0 or node not in names:
                break
            ordered.append(node)
        if len(ordered) == len(graph_inputs) == len(names) and set(ordered) == names:
            candidates["graph-input"] = ordered
    graph_slots = {name: i for i, name in enumerate(candidates.get("graph-input", []))}
    for row in rows:
        row["graph_input_slot"] = graph_slots.get(row["node"])
    conflict = ("data-index" in candidates and "graph-input" in candidates
                and candidates["data-index"] != candidates["graph-input"])
    if conflict:
        notes.append("Data.index differs from GraphDef.input; Data.index may retain source-graph numbering. "
                     "Both orders are retained in input_order_candidates; selected order is shown below")
    if explicit is not None:
        if source != "auto":
            raise ValueError("choose --input-order or --input-order-source, not both")
        order = explicit
        basis = "USER_SUPPLIED_NAMES"
    else:
        if source == "auto":
            source = next((key for key in ("graph-input", "data-index") if key in candidates), None)
        if source is None:
            notes.append("input order unresolved; rows retain JSON order, and input_type is not emitted. "
                         "Use --input-order with actual ordered node names")
            return rows, "UNRESOLVED", candidates, conflict
        if source not in candidates:
            raise ValueError("--input-order-source " + source + " requires a complete unique input mapping")
        order = candidates[source]
        basis = {"graph-input": "GRAPH_INPUT_LIST", "data-index": "CONTIGUOUS_DATA_INDEX",
                 "json-order": "USER_SELECTED_JSON_ORDER"}[source]
    if len(order) != len(names) or set(order) != names:
        raise ValueError("--input-order must contain every root Data/RefData name exactly once")
    by_name = {row["node"]: row for row in rows}
    return [by_name[name] for name in order], basis, candidates, conflict


def output_labels(document, graph, net, descs, count, explicit, notes):
    """Read labels independently of output connectivity; never invent output0."""
    if explicit is not None:
        return explicit, "USER_SUPPLIED_NAMES"
    declared = [desc.get("name") for desc in descs]
    if len(declared) == count and all(isinstance(name, str) and name for name in declared):
        return declared, "NETOUTPUT_INPUT_DESC_NAME"
    attrs = attributes(net)
    keys = attrs.get("_input_name_key", {}).get("list", {}).get("s", [])
    values = attrs.get("_input_name_value", {}).get("list", {}).get("i", [])
    if keys or values:
        values = [integer(value) for value in values]
        if (len(keys) == len(values) == count and set(values) == set(range(count))
                and all(isinstance(name, str) and name for name in keys) and len(set(keys)) == count):
            return [name for _, name in sorted(zip(values, keys))], "NETOUTPUT_INPUT_NAME_MAP"
        notes.append("incomplete/invalid NetOutput input-name map; not used as output labels")
    # Reference GE GetOutputDescInfo label convention, not a HIAI runtime query:
    # https://github.com/Ascend/graphengine/blob/d2ab1c2819c90e61d98b5f88699bcd6b6de4423c/src/ge/graph/load/new_model_manager/davinci_model.cc
    src_names, src_ports = net.get("src_name", []), net.get("src_index", [])
    model_attrs = attributes(document) if isinstance(document, dict) else {}
    override = model_attrs.get("attr_model_out_nodes_name", attributes(graph).get("attr_model_out_nodes_name", {}))
    override = override.get("list", {}).get("s", [])
    ports_valid = isinstance(src_ports, list) and len(src_ports) == count
    ports = [integer(port) for port in src_ports] if ports_valid else []
    ports_valid = ports_valid and all(port >= 0 for port in ports)
    if ports_valid and len(override) == count and all(isinstance(name, str) and name for name in override):
        labels = [name if ":" in name else name + ":" + str(port) for name, port in zip(override, ports)]
        basis = "GE_MODEL_OUT_NODES_NAME_CANDIDATE"
    elif (ports_valid and isinstance(src_names, list) and len(src_names) == count
          and all(isinstance(name, str) and name for name in src_names)):
        labels = [f"output_{i}_{name}_{port}" for i, (name, port) in enumerate(zip(src_names, ports))]
        basis = "GE_LEGACY_SRC_LABEL_CANDIDATE"
    else:
        return [None] * count, "NOT_DECLARED"
    notes.append("Output labels follow the reference GE GetOutputDescInfo convention using serialized src metadata; "
                 "verify names against this DDK runtime before using --output_shape")
    return labels, basis


def named_shapes(rows):
    names = [row["name"] for row in rows]
    if (any(not name or not NAME.fullmatch(name) for name in names)
            or len(set(names)) != len(names)
            or any(row["shape"] is None or any(d < -1 for d in row["shape"]) for row in rows)):
        return None
    return ";".join(row["name"] + ":" + ",".join(map(str, row["shape"])) for row in rows)


def type_list(rows):
    return ",".join(row["dtype"] for row in rows) if all(row["dtype"] in TOOL_TYPES for row in rows) else None


def inspect(document, *, enum=None, graph_name=None, ordered_inputs=None, input_order_source="auto", output_names=None):
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
    inputs, basis, candidates, conflict = input_order(graph, inputs, ordered_inputs, input_order_source, notes)
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
    resolved_descs = []
    for i, edge in enumerate(edges):
        if descs:
            desc, source = descs[i], "NETOUTPUT_INPUT_DESC"
        else:
            producer, port = edge_parts(edge)
            producer_descs = desc_list(by_name.get(producer, {"name": producer}), "output")
            if not 0 <= port < len(producer_descs):
                raise ValueError("NetOutput descriptor and producer descriptor unavailable: " + edge)
            desc, source = producer_descs[port], "PRODUCER_OUTPUT_DESC"
        resolved_descs.append(desc)
        outputs.append(dict(slot=i, node=net["name"], node_type="NetOutput",
                            edge=edge, descriptor_source=source, **descriptor(desc, enum)))
    labels, label_basis = output_labels(document, graph, net, resolved_descs, len(edges), output_names, notes)
    for row, label in zip(outputs, labels):
        row.update(name=label, name_source=label_basis)
    flags = dict(input_shape=named_shapes(inputs),
                 input_type=type_list(inputs) if basis != "UNRESOLVED" else None,
                 output_shape=named_shapes(outputs), output_type=type_list(outputs))
    if flags["output_shape"] is None:
        notes.append("Named --output_shape unavailable: missing/duplicate/CLI-unsafe labels or missing shapes. "
                     "output_shapes contains parsed shapes by NetOutput slot; --output-names accepts actual runtime names")
    notes.append("Graph input order and NetOutput port order are candidate CLI order, not queried HIAI runtime order. "
                 "Align input files with actual runtime slots before running")
    notes.append("logical_bytes excludes format padding, alignment and runtime buffer allocation")
    complete = basis != "UNRESOLVED" and all(row["shape"] is not None and row["dtype"] is not None for row in inputs + outputs)
    return dict(status="PARSED" if complete else "PARTIAL",
                cli_status="READY" if all(value is not None for value in flags.values()) else "PARTIAL",
                graph=graph.get("name"), graph_path=path, input_count=len(inputs), output_count=len(outputs),
                input_order_basis=basis, output_order_basis="NETOUTPUT_DATA_PORTS",
                input_order_candidates=candidates, input_order_conflict=conflict, output_name_basis=label_basis,
                input_shapes=[row["shape"] for row in inputs], output_shapes=[row["shape"] for row in outputs],
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
             "input_order_conflict=" + str(report["input_order_conflict"]).lower(),
             "output_name_basis=" + report["output_name_basis"], "cli_status=" + report["cli_status"],
             f"input_count={report['input_count']} output_count={report['output_count']}",
             "runtime_order_verified=false", ""]
    for kind in ("inputs", "outputs"):
        lines.append(kind.upper() + " (slot is a graph-derived candidate):")
        lines.append("slot\tname / edge\tdtype\tshape\tlogical_bytes\tData.index\tGraph.input slot")
        for row in report[kind]:
            lines.append("\t".join(map(str, [row.get("slot"), row["name"] or row["edge"],
                          row["dtype"] or "UNRESOLVED(" + str(row["dtype_raw"]) + ")",
                          row["shape"], row["logical_bytes"], row.get("serialized_index"), row.get("graph_input_slot")])))
            lines.extend("  " + issue for issue in row["issues"])
        lines.append("")
    for key in ("input_shape", "input_type", "output_shape", "output_type"):
        if key == "output_shape" and report[key] is None:
            lines.append("output_shapes_by_slot=" + json.dumps(report["output_shapes"]))
            lines.append("output_shape_cli=UNAVAILABLE (see output labels and shapes above)")
        else:
            lines.append("--" + key + "=" + shlex.quote(report[key]) if report[key] is not None else key + "=UNRESOLVED")
    lines.extend("NOTE: " + note for note in report["notes"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_json", metavar="MODEL", help=".omc/.om model, or an existing OMG --mode=1 JSON file")
    parser.add_argument("--ddk-root", help="CPro DDK root; defaults to CPRO_DDK_ROOT, DDK_PATH or TOOLCHAIN_HOME")
    parser.add_argument("--omg", help="OMG executable; otherwise use the DDK master/omg or find omg on PATH")
    parser.add_argument("--timeout", type=int, default=120, help="OMC conversion timeout in seconds (default: 120)")
    parser.add_argument("--output-dir", help="write io.json, inputs.csv, outputs.csv, summary.txt and args.sh")
    parser.add_argument("--overwrite", action="store_true", help="replace those five report files in an existing --output-dir")
    parser.add_argument("--graph", help="root graph name or printed JSON path when multiple graphs exist")
    parser.add_argument("--dtype-enum", choices=["ge-proto"], help="explicitly decode numeric TensorDescriptor dtype using ge_ir.proto")
    parser.add_argument("--dtype-map", help='explicit numeric dtype JSON map, e.g. {"1":"FP32","2":"FP16"}')
    parser.add_argument("--input-order", help="comma-separated actual ordered Data/RefData node names; no sorting")
    parser.add_argument("--input-order-source", choices=["auto", "graph-input", "data-index", "json-order"], default="auto",
                        help="auto prefers complete GraphDef.input, then contiguous Data.index; json-order must be explicit")
    parser.add_argument("--output-names", help="comma-separated actual output tensor names in NetOutput port order")
    args = parser.parse_args()
    try:
        if args.overwrite and not args.output_dir:
            raise ValueError("--overwrite requires --output-dir")
        if args.dtype_enum and args.dtype_map:
            raise ValueError("choose --dtype-enum or --dtype-map, not both")
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        is_omc = Path(args.model_json).suffix.lower() in {".omc", ".om"}
        selected_enum = args.dtype_enum or ("ge-proto" if is_omc and not args.dtype_map else None)
        enum = dict(GE_PROTO) if selected_enum else {}
        if args.dtype_map:
            for raw, name in read_json(args.dtype_map).items():
                number, kind = integer(raw), dtype(name, {})
                if number in enum or kind is None:
                    raise ValueError("invalid/duplicate entry in dtype map: " + repr(raw))
                enum[number] = kind
        if is_omc:
            document, source = read_omc(args.model_json, ddk_root=args.ddk_root,
                                        omg_path=args.omg, timeout=args.timeout)
        else:
            document = read_json(args.model_json)
            source = dict(source_json=str(Path(args.model_json).resolve()), source_kind="JSON")
        report = inspect(document, enum=enum, graph_name=args.graph,
                         ordered_inputs=args.input_order.split(",") if args.input_order else None,
                         input_order_source=args.input_order_source,
                         output_names=args.output_names.split(",") if args.output_names else None)
        report.update(source)
        report["numeric_dtype_mapping"] = args.dtype_map or selected_enum or "UNSELECTED"
        summary = render(report)
        if args.output_dir:
            dest = Path(args.output_dir)
            dest.mkdir(parents=True, exist_ok=args.overwrite)
            sources = {Path(path).resolve() for path in (args.model_json, args.dtype_map) if path}
            for filename in ("io.json", "inputs.csv", "outputs.csv", "summary.txt", "args.sh"):
                target = dest / filename
                if target.is_symlink() or target.resolve() in sources or (target.exists() and not target.is_file()):
                    raise ValueError("refusing to replace source, symlink or non-file: " + str(target))
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

"""Batch inspection of retained draft precision descriptors; no vendor execution.

The construction recipe identifies semantic regions. Each retained ModelDef is
decoded once using the selected DDK schema. Exact names, explicit same-graph
edges, and uninterpreted attribute name references remain distinct evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

from .draft_compile_inspect import _bound, _json, _one, _read
from .draft_precision_log import inspect_precision_log
from .draft_ir_inspect import _DUMP_DIRECTORY, _enum_fields, _graphs, _inputs, _node
from .ge_recipe_read import _Reader, _read as _read_recipe
from .hashing import sha256_file
from .omc_inspect import _preview
from .omc_container import read_container
from .omc_cpu_inspect import _task
from .omc_wire import load_schema


_MAX_FILES = 16
_MAX_ENTRIES = 256
_MAX_FILE_BYTES = 64 << 20
_MAX_TOTAL_BYTES = 128 << 20
_MAX_RECIPE_BYTES = 32 << 20
_MAX_REPORT_BYTES = 2 << 20
_MAX_TARGETS = 160
_MAX_CAPTURED_NODES = 256
_MAX_ALIASES = 24
_MAX_OMC_BYTES = 64 << 20
_ATTRIBUTES = {"dtype", "dst_type", "src_type", "T", "shape", "axes", "axis", "keep_dims",
               "power", "scale", "shift", "epsilon", "eps", "half_to_float", "cl_name",
               "transpose_x1", "transpose_x2", "adj_x1", "adj_x2", "data_format", "src_dtype", "dst_dtype",
               "graphop_task_offset", "graphop_task_size"}
_DTYPE_NAMES = {"FLOAT": "float32", "DT_FLOAT": "float32", "FLOAT32": "float32", "FP32": "float32",
                "HALF": "float16", "DT_FLOAT16": "float16", "FLOAT16": "float16", "FP16": "float16",
                "INT32": "int32", "DT_INT32": "int32", "BOOL": "bool", "DT_BOOL": "bool"}


def _recipe_targets(recipe):
    records = recipe["records"]
    consumers = {}
    for index, record in enumerate(records):
        for port in record["inputs"]:
            consumers.setdefault(port["src"], set()).add(index)
    by_name = {r["name"]: i for i, r in enumerate(records)}
    groups, selected = [], {}

    def group(label, indices):
        indices = list(dict.fromkeys(indices))
        if len(selected.keys() | {records[i]["name"] for i in indices}) > _MAX_TARGETS:
            raise ValueError("semantic recipe selection exceeds target limit")
        groups.append({"semantic": label, "nodes": [records[i]["name"] for i in indices]})
        for i in indices:
            record = records[i]
            if (len(record["name"]) > 512 or len(record["type"]) > 128
                    or len(record["inputs"]) > 16 or len(record["outputs"]) > 16):
                raise ValueError("selected recipe node exceeds diagnostic metadata bounds")
            selected[record["name"]] = {k: record[k] for k in ("name", "type", "inputs", "outputs")}
            selected[record["name"]]["attrs"] = _preview(record["attrs"], budget=[256])

    def parents(index, depth):
        found, pending = [], [(index, 0)]
        while pending:
            i, level = pending.pop(0)
            if i in found:
                continue
            found.append(i)
            if level < depth and records[i]["type"] not in {"Data", "Const", "FileConstant"}:
                pending.extend((p["src"], level + 1) for p in records[i]["inputs"])
        return found

    # Follow the gamma edge, not version-dependent vNNN numbers. Include the
    # complete known RMS expression while avoiding the preceding layer tree.
    for weight in ("draft.norm.weight", "draft.hidden_norm.weight"):
        if weight not in by_name:
            groups.append({"semantic": weight, "nodes": [], "status": "NOT_FOUND"})
            continue
        wi = by_name[weight]
        roots = [i for i in consumers.get(wi, ()) if records[i]["type"] == "Mul"]
        if len(roots) != 1:
            groups.append({"semantic": weight, "nodes": [weight], "status": "AMBIGUOUS_GAMMA_CONSUMER"})
            group(weight + "/weight", [wi])
            continue
        root = roots[0]
        indices = [root, wi]
        normalized = [p["src"] for p in records[root]["inputs"] if p["src"] != wi]
        if len(normalized) == 1:
            # Cast -> Mul(x32, Rsqrt(...)). Stop x32 traversal after its input;
            # its producer belongs to the preceding layer, not this norm.
            ci = normalized[0]
            indices.append(ci)
            if records[ci]["type"] == "Cast" and len(records[ci]["inputs"]) == 1:
                product = records[ci]["inputs"][0]["src"]
                indices.append(product)
                for port in records[product]["inputs"]:
                    index = port["src"]
                    indices.extend(parents(index, 4 if records[index]["type"] == "Rsqrt" else 1))
        group(weight, indices)

    softmaxes = [i for i, r in enumerate(records) if r["type"] in {"Softmax", "SoftmaxV2"}]
    for i in softmaxes:
        group("attention/" + records[i]["name"], parents(i, 3) + sorted(consumers.get(i, ())))
    # Keep all norm reductions visible, including per-head Q/K norms. Their
    # exact gamma association is not inferred solely from being ReduceMean.
    group("all_reduce_mean", [i for i, r in enumerate(records) if r["type"] == "ReduceMean"])
    for key in ("draft.fc.weight",):
        if key in by_name:
            wi = by_name[key]
            first = sorted(consumers.get(wi, ()))
            second = [j for i in first for j in sorted(consumers.get(i, ())) if records[j]["type"] == "Conv2D"]
            group("feature_projection", [wi, *first, *second])
    return groups, selected


def _strings(value, depth=0):
    if depth > 5:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in list(value.values())[:32]:
            yield from _strings(item, depth + 1)
    elif isinstance(value, list):
        for item in value[:32]:
            yield from _strings(item, depth + 1)


def _capture(node, enum_fields):
    result = _node(node, enum_fields)
    attrs = node.get("attr", [])
    result["attributes"] = _preview([a for a in attrs if isinstance(a, dict) and a.get("key") in _ATTRIBUTES], budget=[512])
    result["other_attribute_count"] = len(attrs) - len(result["attributes"])
    # Absence remains explicit: the static decoder deliberately does not guess
    # proto2 defaults or equate public GE attribute IDs with schema enum IDs.
    for key in ("input_desc", "output_desc"):
        for desc in result[key]:
            desc["dtype_presence"] = "EXPLICIT" if "dtype" in desc["raw"] else "OMITTED_NOT_INTERPRETED"
    return result


def _dtype(descriptor):
    domain = descriptor.get("schema_enum_names", {}).get("dtype", {})
    names = domain.get("names", [])
    found = {_DTYPE_NAMES[name] for name in names if name in _DTYPE_NAMES}
    return next(iter(found)) if len(found) == 1 else None


def _differences(declared, actual):
    differences = []
    # Port ordering can change after optimization; each observation is stated
    # as an ordinal descriptor difference, never proof of equivalent edges.
    for direction, source_key in (("input_desc", "inputs"), ("output_desc", "outputs")):
        expected, observed = declared[source_key], actual[direction]
        if len(expected) != len(observed):
            differences.append({"field": direction, "kind": "PORT_COUNT_DIFFERS",
                                "declared": len(expected), "observed": len(observed)})
        for index, (source, target) in enumerate(zip(expected, observed)):
            dtype = _dtype(target)
            if dtype is not None and dtype != source["dtype"]:
                differences.append({"field": direction, "port": index, "kind": "DTYPE_DIFFERS",
                                    "declared": source["dtype"], "observed": dtype,
                                    "schema_enum": target["schema_enum_names"]["dtype"]})
            shape = target["raw"].get("shape", {}).get("dim")
            if shape is not None and shape != source["shape"]:
                differences.append({"field": direction, "port": index, "kind": "SHAPE_DIFFERS",
                                    "declared": source["shape"], "observed": shape})
    return differences


def _select_model(model, targets, enum_fields):
    matches, captured_count, aliases_count = [], 0, 0
    target_order = {name: i for i, name in enumerate(targets)}
    for graph, graph_path in _graphs(model):
        ops = graph["op"]
        by_name = {}
        for index, node in enumerate(ops):
            name = node.get("name")
            if isinstance(name, str):
                by_name.setdefault(name, []).append(index)
        roots, aliases = [], {}
        for index, node in enumerate(ops):
            if node.get("name") in targets:
                roots.append(index)
            refs = []
            for attr_index, attr in enumerate(node.get("attr", [])):
                if not isinstance(attr, dict) or attr.get("key") in {"value", "weights", "weight"}:
                    continue
                names = sorted(set(_strings(attr.get("value"))) & targets.keys())
                if names:
                    refs.append({"attribute_index": attr_index, "key": attr.get("key"), "names": names})
            if refs and node.get("name") not in targets and aliases_count < _MAX_ALIASES:
                aliases[index] = refs
                aliases_count += 1
        if not roots and not aliases:
            continue
        roots.sort(key=lambda i: target_order[ops[i]["name"]])
        capture, pending, edges = {}, [(i, 0) for i in dict.fromkeys([*roots, *aliases])], []
        capture_limited = False
        while pending:
            index, depth = pending.pop(0)
            if index in capture:
                continue
            if captured_count >= _MAX_CAPTURED_NODES:
                capture_limited = True
                break
            node = ops[index]
            name = node.get("name")
            actual = _capture(node, enum_fields)
            row = {"json_path": f"{graph_path}/op/{index}", "node": actual,
                   "exact_match_targets": [name] if name in targets else [],
                   "name_status": "UNIQUE" if len(by_name.get(name, [])) == 1 else "AMBIGUOUS",
                   "ancestry_depth": depth}
            if index in aliases:
                row["attribute_name_references"] = aliases[index]
                row["alias_scope"] = "Uninterpreted attribute text; not an established rename or equivalent operation"
            if name in targets and row["name_status"] == "UNIQUE":
                row["descriptor_differences"] = _differences(targets[name], actual)
            capture[index] = row
            captured_count += 1
            if depth >= 2 or node.get("type") in {"Data", "Const", "FileConstant"}:
                continue
            for reference in _inputs(node):
                found = by_name.get(reference.get("name"), [])
                edge = {"consumer_json_path": row["json_path"], **reference,
                        "status": "RESOLVED" if len(found) == 1 else "AMBIGUOUS" if found else "UNRESOLVED"}
                if len(found) == 1:
                    edge["producer_json_path"] = f"{graph_path}/op/{found[0]}"
                    pending.append((found[0], depth + 1))
                edges.append(edge)
        matches.append({"graph_path": graph_path, "graph_name": graph.get("name"),
                        "nodes": list(capture.values()), "ancestry": edges,
                        "capture_limited": capture_limited})
    return matches


def _finish(report):
    names, types, dtypes, differences = set(), Counter(), Counter(), []
    observations = {}
    norm_names = {name for group in report.get("source_targets", [])
                  if group["semantic"] in {"draft.norm.weight", "draft.hidden_norm.weight"}
                  for name in group["nodes"]}
    final_names = {name for group in report.get("source_targets", [])
                   if group["semantic"] == "draft.norm.weight" for name in group["nodes"]}

    def priority(row):
        return (row["kind"] != "DTYPE_DIFFERS", row["node"] not in norm_names,
                row["field"] != "output_desc", row["file"], row["node"])
    captured, final_conversions = 0, []
    for file in report["files"]:
        for graph in file.get("matches", []):
            ancestry = {}
            for edge in graph.get("ancestry", []):
                if edge["status"] == "RESOLVED":
                    ancestry.setdefault(edge["consumer_json_path"], set()).add(edge["producer_json_path"])
            frontier = {row["json_path"] for row in graph["nodes"] if final_names.intersection(row["exact_match_targets"])}
            related = set(frontier)
            for _ in range(2):
                frontier = {p for name in frontier for p in ancestry.get(name, ())}
                related.update(frontier)
            for row in graph["nodes"]:
                captured += 1
                node = row["node"]
                if (row["json_path"] in related and not row["exact_match_targets"]
                        and node.get("type") in {"Cast", "CastT"} and len(final_conversions) < 12):
                    final_conversions.append({"file": file["path"], "graph_path": graph["graph_path"],
                        "json_path": row["json_path"], "name": node["name"], "type": node["type"],
                        "name_status": row["name_status"], "input": node["input"],
                        "src_name": node["src_name"], "src_index": node["src_index"],
                        "attributes": node["attributes"],
                        "input_descriptors": [_descriptor_summary(d) for d in node["input_desc"]],
                        "output_descriptors": [_descriptor_summary(d) for d in node["output_desc"]]})
                names.update(row["exact_match_targets"])
                types[node.get("type", "")] += 1
                for desc in node.get("output_desc", []):
                    dtypes[_dtype(desc) or "NOT_INTERPRETED"] += 1
                for name in row["exact_match_targets"]:
                    observation = {"type": node.get("type"), "name_status": row["name_status"],
                                   "input_dtypes": [_dtype(d) or "NOT_INTERPRETED" for d in node["input_desc"]],
                                   "output_dtypes": [_dtype(d) or "NOT_INTERPRETED" for d in node["output_desc"]],
                                   "attributes": node.get("attributes", [])}
                    seen = observations.setdefault(name, [])
                    if observation not in seen and len(seen) < 8:
                        seen.append(observation)
                for difference in row.get("descriptor_differences", []):
                    differences.append({"file": file["path"], "graph_path": graph["graph_path"],
                                        "node": node["name"], **difference})
                    differences.sort(key=priority)
                    del differences[40:]
    targets = set(report.get("declared_nodes", {}))
    key_names = []
    for group in report.get("source_targets", []):
        semantic = group["semantic"]
        if semantic in {"draft.norm.weight", "draft.hidden_norm.weight", "all_reduce_mean"}:
            key_names.extend(group["nodes"])
        elif semantic.startswith("attention/"):
            key_names.extend(group["nodes"][:1])
    key_nodes = []
    for name in dict.fromkeys(key_names):
        declared = report["declared_nodes"].get(name)
        if declared is not None and (name in final_names or name in observations):
            key_nodes.append({"name": name, "declared_type": declared["type"],
                              "declared_output_dtypes": [p["dtype"] for p in declared["outputs"]],
                              "observed": observations.get(name, []),
                              "status": "OBSERVED" if name in names else "NOT_OBSERVED"})
    report["summary"] = {"status": report["status"], "reason": report.get("reason"),
                         "source_binding": report["source_binding"],
                         "executed_bundle_binding": report["executed_bundle_binding"],
                         "dump_files": sum(not f.get("evidence_kind", "").startswith("OMC_") for f in report["files"]),
                         "compiled_descriptor_slices": sum(f.get("evidence_kind", "").startswith("OMC_") for f in report["files"]),
                         "decoded_files": sum(f["status"] == "DECODED" for f in report["files"]),
                         "captured_nodes": captured, "observed_source_names": len(names),
                         "selected_source_names": len(targets), "not_observed": len(targets - names),
                         "captured_op_types": dict(types), "captured_output_dtypes": dict(dtypes),
                         "descriptor_differences": differences, "key_nodes": key_nodes,
                         "final_norm_conversion_ancestors": final_conversions,
                         "scope": "Retained IR descriptors across dumps; not runtime placement, measured tensor values, or root-cause proof"}
    report["not_observed_source_names"] = sorted(targets - names)
    if "compile_log_evidence" in report:
        report["summary"]["compile_log_evidence"] = report["compile_log_evidence"]
    if "omc_inspection" in report:
        omc = report["omc_inspection"]
        owners = {row["name"]: row for row in omc.get("graphop_boundaries", [])}
        needed = set(omc.get("final_norm_owners", []))
        frontier = {row.get("producer_name") for row in omc.get("output_connections", [])}
        # Exact main-graph predecessors provide context when the final NPU
        # task is opaque. No cross-graph name stripping or inferred fusion.
        for _ in range(2):
            needed.update(frontier)
            frontier = {edge.get("name") for name in frontier if name in owners
                        for edge in owners[name]["explicit_input_references"]}
        selected = [owners[name] for name in owners if name in needed][:8]
        report["summary"].update(evidence_route="BOUND_COMPILED_OMC", omc_status=omc["status"],
                                 omc_reason=omc.get("reason"), omc_binding=omc.get("binding"),
                                 graphop_counts=omc.get("graphop_counts", {}),
                                 graphop_boundaries=[{"name": row["name"], "serialized_cl_name": row["serialized_cl_name"],
                                     "task_status": row["task_status"], "input": row["explicit_input_references"],
                                     "input_descriptors": [_descriptor_summary(d) for d in row["node"]["input_desc"]],
                                     "output_descriptors": [_descriptor_summary(d) for d in row["node"]["output_desc"]]}
                                     for row in selected],
                                 graphop_boundaries_omitted=len(owners) - len(selected),
                                 output_connections=omc.get("output_connections", []),
                                 opaque_task_note=omc.get("opaque_task_note"))
        report["summary"]["scope"] = "Static compiled OMC descriptors and explicit GraphOp boundaries; not actual intermediate values or runtime placement"
    return report


def _descriptor_summary(descriptor):
    return {"dtype": _dtype(descriptor) or "NOT_INTERPRETED", "dtype_presence": descriptor.get("dtype_presence"),
            "shape": descriptor["raw"].get("shape", {}).get("dim"),
            "layout": descriptor["raw"].get("layout")}


def _output_connections(ops, enums):
    """Pair NetOutput input ports with exact producer outputs in the main graph."""
    by_name = {node["name"]: (index, node) for index, node in enumerate(ops)}
    rows = []
    for index, node in enumerate(ops):
        if node.get("type") != "NetOutput":
            continue
        for edge in _inputs(node):
            row = {"consumer_name": node["name"], "consumer_json_path": f"$/graph/0/op/{index}",
                   **edge, "status": "UNRESOLVED"}
            input_index = edge["input_index"]
            if input_index < len(node.get("input_desc", [])):
                desc = _capture({"input_desc": [node["input_desc"][input_index]]}, enums)["input_desc"][0]
                row["consumer_input_descriptor"] = _descriptor_summary(desc)
            producer = by_name.get(edge.get("name"))
            if producer is not None:
                pi, pn = producer
                row.update(producer_name=pn["name"], producer_type=pn.get("type"),
                           producer_json_path=f"$/graph/0/op/{pi}", status="PRODUCER_RESOLVED_PORT_UNRESOLVED")
                port = edge.get("output_port")
                if type(port) is int and 0 <= port < len(pn.get("output_desc", [])):
                    desc = _capture({"output_desc": [pn["output_desc"][port]]}, enums)["output_desc"][0]
                    row.update(producer_output_descriptor=_descriptor_summary(desc), status="RESOLVED")
            rows.append(row)
    return rows


def _dump_priority(path):
    name = path.name.lower()
    if re.search(r"after.*partition|connection.?point", name):
        category = 0
    elif "subgraph" in name:
        category = 1
    elif "final" in name or "optimized" in name:
        category = 2
    else:
        category = 3
    return category, name


def _inspect_small_omc(report, compiled, compile_root, bundle, bundle_root, schema, targets, enums):
    """Inspect the bound external source container, never its embedded weights.

    The already retained packing receipt supplies the correspondence between
    the external source and executed embedded model. Main/task byte identity,
    cumulative partition offsets and explicit task owners gate CPU decoding.
    Opaque NPU task bytes are hashed for binding but never interpreted.
    """
    result = {"status": "NOT_AVAILABLE", "binding": "NOT_ESTABLISHED", "graphop_boundaries": [],
              "scope": "Strict external IMOD variant; selected schema and explicit task owners only; no vendor code"}
    report["omc_inspection"] = result
    omc = compiled.get("omc")
    if not isinstance(omc, dict):
        result["reason"] = "Compiler report has no retained OMC record"
        return
    if bundle is None or bundle_root is None:
        result["reason"] = "No executed bundle identity for compiled OMC inspection"
        return
    try:
        source = _bound(compile_root, omc)
        before = source.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        raw = _read(source, _MAX_OMC_BYTES, "small external OMC", report["sources"], omc)
        container = read_container(source)
        after = source.stat()
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("source OMC changed during container inspection")
        parts = container["partitions"]
        if [p["type"] for p in parts] != [0, 1, 2] or parts[1]["bytes"] != 0:
            raise ValueError("fallback requires external OMC [0,1,2] with empty weight partition")
        cursor = 0
        for part in parts:
            if part["table_word"] != cursor:
                raise ValueError("OMC partition words are not cumulative payload offsets")
            cursor += part["bytes"]
        if hashlib.sha256(raw[:256]).hexdigest() != container["header_sha256"]:
            raise ValueError("OMC header differs from hash-bound bytes")

        def payload(part):
            return memoryview(raw)[part["offset"]:part["offset"] + part["bytes"]]

        hashes = {part["type"]: hashlib.sha256(payload(part)).hexdigest() for part in parts if part["type"] in (0, 2)}
        if bundle.get("weight_storage") == "embedded-stream":
            receipt = bundle.get("weight_packing")
            if not isinstance(receipt, dict):
                raise ValueError("embedded-stream bundle has no bound packing receipt")
            packing = _json(_bound(bundle_root, receipt), report["sources"], receipt)
            if (packing.get("artifact_kind") != "qwen35-omc-embedded-stream" or packing.get("status") != "EMBEDDED"
                    or any(packing.get("source", {}).get(k) != omc.get(k) for k in ("bytes", "sha256"))
                    or any(packing.get("output", {}).get(k) != bundle.get("model", {}).get(k) for k in ("bytes", "sha256"))):
                raise ValueError("packing source/output identity differs from compiler or executed bundle")
            if packing.get("schema", {}).get("library_sha256") != schema.evidence.get("library_sha256"):
                raise ValueError("selected schema differs from bound packing schema")
            for part in (parts[0], parts[2]):
                saved = _one(packing.get("partitions", []), lambda p: p.get("type") == part["type"], "packing partition")
                if (saved.get("bytes") != part["bytes"] or saved.get("sha256") != hashes[part["type"]]
                        or saved.get("preserved") != "BYTE_IDENTICAL"):
                    raise ValueError("main/task partition differs from executed packing receipt")
            result["binding"] = "PACKING_PRESERVED_MAIN_AND_TASK_BYTES"
        elif all(bundle.get("model", {}).get(k) == omc.get(k) for k in ("bytes", "sha256")):
            result["binding"] = "COMPILER_OMC_IDENTICAL_TO_EXECUTED_MODEL"
        else:
            raise ValueError("compiler OMC is not bound to the executed bundle")
        model = schema.decode(payload(parts[0]), "ModelDef")
        graphs = model.get("graph")
        if not isinstance(graphs, list) or len(graphs) != 1 or not isinstance(graphs[0], dict):
            raise ValueError("compiled OMC must contain exactly one main graph")
        ops = graphs[0].get("op")
        if not isinstance(ops, list) or not ops or len(ops) > 512:
            raise ValueError("main graph has unsupported operation count")
        names = [node.get("name") for node in ops]
        if any(not isinstance(name, str) or not name or len(name) > 512 for name in names) or len(set(names)) != len(names):
            raise ValueError("main graph has missing or duplicate names")
        owners, ranges = [], []
        for index, node in enumerate(ops):
            if node.get("type") != "GraphOp":
                continue
            task = _task(node)
            start, size = task["offset"], task["bytes"]
            if not size or start > parts[2]["bytes"] or size > parts[2]["bytes"] - start:
                raise ValueError("GraphOp task slice is outside the compiled task partition")
            end = start + size
            if any(start < right and left < end for left, right in ranges):
                raise ValueError("GraphOp task slices overlap")
            ranges.append((start, end))
            owners.append((index, node, task))
        if not owners or len(owners) > 128:
            raise ValueError("unsupported number of GraphOp task owners")
        # Validate every ownership bound before decoding any CPU slice.
        result.update(container=container, main_partition_sha256=hashes[0], task_partition_sha256=hashes[2],
                      source={"path": str(source), "bytes": len(raw), "sha256": omc["sha256"]},
                      graphop_counts=dict(Counter(task["cl_name"] for _, _, task in owners)),
                      model_io_boundaries=[{"json_path": f"$/graph/0/op/{i}", "node": _capture(n, enums),
                                            "explicit_input_references": _inputs(n)} for i, n in enumerate(ops)
                                           if n.get("type") in {"Data", "NetOutput"}],
                      output_connections=_output_connections(ops, enums), final_norm_owners=[],
                      main_graph_inputs=_preview(graphs[0].get("input", [])),
                      main_graph_outputs=_preview(graphs[0].get("output", [])))
        for index, node, task in owners:
            result["graphop_boundaries"].append({"name": node["name"], "json_path": f"$/graph/0/op/{index}",
                "serialized_cl_name": task["cl_name"], "task_bytes": task["bytes"], "task_offset": task["offset"],
                "node": _capture(node, enums), "explicit_input_references": _inputs(node),
                "task_status": "CPU_GRAPHDEF_CANDIDATE" if task["cl_name"] == "CPUCL" else "OPAQUE_TASK_NOT_DECODED"})
        main_row = {"path": omc["path"] + "#main", "absolute_path": str(source), "evidence_kind": "OMC_MODELDEF",
                    "status": "DECODED", "sha256": hashes[0], "bytes": parts[0]["bytes"],
                    "matches": _select_model(model, targets, enums)}
        rows = [main_row]
        for index, node, task in owners:
            if task["cl_name"] != "CPUCL":
                continue
            offset = parts[2]["offset"] + task["offset"]
            data = memoryview(raw)[offset:offset + task["bytes"]]
            row = {"path": omc["path"] + "#" + node["name"], "absolute_path": str(source),
                   "evidence_kind": "OMC_CPU_GRAPHDEF", "owner_json_path": f"$/graph/0/op/{index}",
                   "file_offset": offset, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                   "status": "UNSUPPORTED_CPU_SLICE"}
            rows.append(row)
            try:
                graph = schema.decode(data, "GraphDef")
                if graph.get("name") != node["name"]:
                    raise ValueError("CPU GraphDef name differs from its explicit GraphOp owner")
                row["matches"] = _select_model({"graph": [graph]}, targets, enums)
                row["status"] = "DECODED"
                final_names = {name for group in report["source_targets"]
                               if group["semantic"] == "draft.norm.weight" for name in group["nodes"]}
                if any(final_names.intersection(r["exact_match_targets"])
                       for match in row["matches"] for r in match["nodes"]):
                    result["final_norm_owners"].append(node["name"])
            except (OSError, ValueError, TypeError, RecursionError) as error:
                row["error"] = f"{type(error).__name__}: {str(error)[:512]}"
            if len(json.dumps({"result": result, "files": rows}, ensure_ascii=True, allow_nan=False).encode()) > _MAX_REPORT_BYTES:
                row.pop("matches", None)
                row.update(status="OUTPUT_LIMIT", reason="bounded OMC descriptor report")
                result["output_limited"] = True
                break
        report["files"].extend(rows)
        result["status"] = "INSPECTED" if all(row["status"] == "DECODED" for row in rows) else "PARTIAL"
        result["opaque_task_note"] = "Non-CPU task payloads were not decoded; GraphOp ports show serialized boundaries only. Missing final RMS internals still require intermediate-value evidence."
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status="NOT_AVAILABLE", reason=f"{type(error).__name__}: {str(error)[:512]}")



def inspect_draft_precision(job_dir, schema_library=None):
    """Return a bounded read-only batch report; never open model weight payloads."""
    job = Path(job_dir).absolute()
    report = {"schema_version": 1, "artifact_kind": "qwen35-draft-precision-inspection", "status": "NOT_AVAILABLE",
              "job": str(job), "source_binding": "NOT_ESTABLISHED", "sources": [], "files": [],
              "executed_bundle_binding": "NOT_AVAILABLE",
              "source_targets": [], "declared_nodes": {}, "new_compilation": "NOT_RUN", "new_device_execution": "NOT_RUN",
              "limitations": ["Recipe descriptors express export intent; retained dumps may precede optimization.",
                              "Missing or fused names do not prove precision fallback; attribute aliases are candidates only.",
                              "Descriptor enum names come from the selected schema; integer Cast attributes have a separate domain.",
                              "Omitted dtype defaults remain uninterpreted; capture limits and unsupported schemas remain explicit."]}
    sources = report["sources"]
    try:
        record = _json(job / ("result.json" if (job / "result.json").is_file() else "job.json"), sources)
        if record.get("artifact_kind") != "qwen35-full-draft-job":
            raise ValueError("expected standalone draft job")
        components = record.get("components", {})
        if not all(key in components for key in ("air", "compilation")):
            report["reason"] = "AIR or compilation component is unavailable"
            return _finish(report)
        air_root, compile_root = [Path(components[key]["directory"]).absolute() for key in ("air", "compilation")]
        air = _json(_bound(air_root, components["air"]["manifest"]), sources, components["air"]["manifest"])
        compiled = _json(_bound(compile_root, components["compilation"]["manifest"]), sources,
                         components["compilation"]["manifest"])
        if (air.get("artifact_kind") != "qwen35-cpro-draft-air" or air.get("status") != "EXPORTED"
                or compiled.get("artifact_kind") != "qwen35-existing-air-compilation"):
            raise ValueError("unexpected AIR or compiler manifest")
        bundle = bundle_root = None
        if "bundle" in components:
            bundle_record = components["bundle"]
            bundle_root = Path(bundle_record["directory"]).absolute()
            bundle = _json(_bound(bundle_root, bundle_record["manifest"]), sources, bundle_record["manifest"])
            if (bundle.get("compiler_report_sha256") != components["compilation"]["manifest"]["sha256"]
                    or bundle.get("export_manifest_sha256") != components["air"]["manifest"]["sha256"]):
                report["executed_bundle_binding"] = "MISMATCH"
                raise ValueError("bundle compiler/export identities differ from inspected components")
            report["executed_bundle_binding"] = "MATCHED"
        graph = _one(air.get("graphs", []), lambda row: row.get("name") == "draft", "draft graph")
        exported = _one(graph["files"], lambda row: row.get("path") == graph["air"], "exported AIR")
        snapshot = _one(compiled.get("input_snapshots", []), lambda row: row.get("role") == "air", "compiler AIR snapshot")
        if any(snapshot.get(k) != exported.get(k) for k in ("bytes", "sha256")):
            raise ValueError("compiled AIR identity differs from exported recipe manifest")
        if compiled.get("air") and any(compiled["air"].get(k) != exported.get(k) for k in ("bytes", "sha256")):
            raise ValueError("compiler source AIR identity differs from export")
        recipe_record = _one(graph.get("construction_files", []),
                             lambda row: row.get("path") == "draft/recipe/graph.recipe", "draft recipe")
        raw = _read(_bound(air_root, recipe_record), _MAX_RECIPE_BYTES, "recipe", sources, recipe_record)
        recipe = _read_recipe(_Reader(io.BytesIO(raw), len(raw)))
        if recipe["graph_name"] != "draft":
            raise ValueError("recipe graph must be draft")
        groups, targets = _recipe_targets(recipe)
        report.update(source_targets=groups, declared_nodes=targets, source_binding="RECORDED_AIR_HASH_MATCHED",
                      air_sha256=exported["sha256"], export_contract=air.get("export_contract"),
                      requested_options=compiled.get("requested_options", {}))
        final_names = [name for group in groups if group["semantic"] == "draft.norm.weight" for name in group["nodes"]]
        log = inspect_precision_log(compile_root, compiled, final_names)
        report["compile_log_evidence"] = log
        if log.get("source"):
            sources.append(log["source"])
        if len(json.dumps(report, ensure_ascii=True, allow_nan=False).encode()) > 512 << 10:
            report.update(source_targets=[], declared_nodes={})
            raise ValueError("source diagnostic metadata exceeds bounded report allowance")
        inventory = {}
        for row in compiled.get("generated_artifacts", []) + compiled.get("compiler_work_artifacts", []):
            if row.get("path") in inventory:
                raise ValueError("duplicate compiler artifact inventory path")
            inventory[row.get("path")] = row
        if schema_library is None and compiled.get("ddk_root"):
            schema_library = Path(compiled["ddk_root"]) / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
        if schema_library is None:
            report["reason"] = "No selected DDK schema library"
            return _finish(report)
        schema = load_schema(schema_library)
        enums = _enum_fields(schema)
        report.update(schema=schema.evidence, tensor_descriptor_enum_fields=enums)
        folder = compile_root / _DUMP_DIRECTORY
        if folder.is_dir() and folder.resolve() != folder:
            raise ValueError("dump directory contains symlink traversal")
        # Inventory paths are already bounded by the compiler JSON read. Start
        # with these names so a crowded directory cannot hide a later-stage
        # dump beyond a filesystem iteration prefix. Only direct children of
        # the fixed dump directory are considered.
        candidates = set()
        if folder.is_dir():
            candidates = {compile_root / relative for relative in inventory if isinstance(relative, str)
                          and Path(relative).parent == _DUMP_DIRECTORY and Path(relative).suffix == ".irpb"}
            for i, path in enumerate(folder.iterdir()):
                if i >= _MAX_ENTRIES:
                    report["directory_scan_limited"] = True
                    break
                if path.suffix == ".irpb":
                    candidates.add(path)
        candidates = sorted(candidates, key=_dump_priority)
        report.update(candidate_count=len(candidates), files_omitted=max(0, len(candidates) - _MAX_FILES),
                      selection_policy="Filename hints prioritize after-partition/connection-point, then subgraph, then final/optimized; filenames do not prove compilation order")
        total = 0
        for path in candidates[:_MAX_FILES]:
            relative = path.relative_to(compile_root).as_posix()
            row = {"path": relative, "absolute_path": str(path), "status": "UNSUPPORTED_DUMP"}
            report["files"].append(row)
            try:
                recorded = inventory.get(relative)
                if recorded is None:
                    row.update(status="UNBOUND_DUMP", inventory_binding="UNRECORDED")
                    continue
                raw = _read(path, min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - total), "IR dump", sources, recorded)
                total += len(raw)
                row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), inventory_binding="MATCHED")
                model = schema.decode(raw, "ModelDef")
                row["matches"] = _select_model(model, targets, enums)
                row["status"] = "DECODED"
                if len(json.dumps(report, ensure_ascii=True, allow_nan=False).encode()) > _MAX_REPORT_BYTES:
                    row.pop("matches", None)
                    row.update(status="OUTPUT_LIMIT", reason="bounded report byte limit")
                    report["output_limited"] = True
                    break
            except (OSError, ValueError, TypeError, RecursionError) as error:
                row.update(status="INTEGRITY_MISMATCH" if "SHA256/size mismatch" in str(error) else "UNSUPPORTED_DUMP",
                           error=f"{type(error).__name__}: {str(error)[:512]}")
        if not candidates:
            report["dump_availability"] = "NO_RETAINED_IRPB"
            _inspect_small_omc(report, compiled, compile_root, bundle, bundle_root, schema, targets, enums)
        report["read_bytes"] = total
        for source in sources:
            path = Path(source["path"])
            if path.stat().st_size != source["bytes"] or sha256_file(path) != source["sha256"]:
                raise ValueError("inspection source changed during analysis")
        complete = (not report.get("directory_scan_limited") and not report["files_omitted"]
                    and not report.get("output_limited") and all(r["status"] == "DECODED" for r in report["files"])
                    and not any(g.get("capture_limited") for r in report["files"] for g in r.get("matches", [])))
        report["status"] = "INSPECTED" if complete else "PARTIAL"
        if not candidates:
            report["status"] = report["omc_inspection"]["status"]
            report["reason"] = report["omc_inspection"].get("reason", "Inspected bound OMC because no retained IRPB dumps exist")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        report.update(status="NOT_AVAILABLE", reason=f"{type(error).__name__}: {str(error)[:512]}")
        # Never retain decoded conclusions after a late metadata integrity error.
        for file in report["files"]:
            file.pop("matches", None)
        if "omc_inspection" in report:
            report["omc_inspection"] = {"status": "NOT_AVAILABLE", "binding": "NOT_ESTABLISHED",
                                        "reason": report["reason"], "graphop_boundaries": []}
        if "compile_log_evidence" in report:
            report["compile_log_evidence"] = {"status": "NOT_AVAILABLE", "reason": report["reason"], "lines": []}
    return _finish(report)

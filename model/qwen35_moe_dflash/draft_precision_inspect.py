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
from .draft_ir_inspect import _DUMP_DIRECTORY, _enum_fields, _graphs, _inputs, _node
from .ge_recipe_read import _Reader, _read as _read_recipe
from .hashing import sha256_file
from .omc_inspect import _preview
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
_ATTRIBUTES = {"dtype", "dst_type", "src_type", "T", "shape", "axes", "axis", "keep_dims",
               "power", "scale", "shift", "epsilon", "eps", "half_to_float", "cl_name",
               "transpose_x1", "transpose_x2", "adj_x1", "adj_x2", "data_format"}
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

    def priority(row):
        return (row["kind"] != "DTYPE_DIFFERS", row["node"] not in norm_names,
                row["field"] != "output_desc", row["file"], row["node"])
    captured = 0
    for file in report["files"]:
        for graph in file.get("matches", []):
            for row in graph["nodes"]:
                captured += 1
                node = row["node"]
                names.update(row["exact_match_targets"])
                types[node.get("type", "")] += 1
                for desc in node.get("output_desc", []):
                    dtypes[_dtype(desc) or "NOT_INTERPRETED"] += 1
                for name in row["exact_match_targets"]:
                    observation = {"type": node.get("type"), "name_status": row["name_status"],
                                   "input_dtypes": [_dtype(d) or "NOT_INTERPRETED" for d in node["input_desc"]],
                                   "output_dtypes": [_dtype(d) or "NOT_INTERPRETED" for d in node["output_desc"]]}
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
        if declared is not None:
            key_nodes.append({"name": name, "declared_type": declared["type"],
                              "declared_output_dtypes": [p["dtype"] for p in declared["outputs"]],
                              "observed": observations.get(name, []),
                              "status": "OBSERVED" if name in names else "NOT_OBSERVED"})
    report["summary"] = {"status": report["status"], "reason": report.get("reason"),
                         "source_binding": report["source_binding"],
                         "executed_bundle_binding": report["executed_bundle_binding"],
                         "dump_files": len(report["files"]),
                         "decoded_files": sum(f["status"] == "DECODED" for f in report["files"]),
                         "captured_nodes": captured, "observed_source_names": len(names),
                         "selected_source_names": len(targets), "not_observed": sorted(targets - names),
                         "captured_op_types": dict(types), "captured_output_dtypes": dict(dtypes),
                         "descriptor_differences": differences, "key_nodes": key_nodes,
                         "scope": "Retained IR descriptors across dumps; not runtime placement, measured tensor values, or root-cause proof"}
    return report


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
        if len(json.dumps(report, ensure_ascii=True, allow_nan=False).encode()) > 512 << 10:
            report.update(source_targets=[], declared_nodes={})
            raise ValueError("source diagnostic metadata exceeds bounded report allowance")
        inventory = {}
        for row in compiled.get("generated_artifacts", []) + compiled.get("compiler_work_artifacts", []):
            if row.get("path") in inventory:
                raise ValueError("duplicate compiler artifact inventory path")
            inventory[row.get("path")] = row
        folder = compile_root / _DUMP_DIRECTORY
        if not folder.is_dir():
            report["reason"] = "No retained dump_graph directory"
            return _finish(report)
        if folder.resolve() != folder:
            raise ValueError("dump directory contains symlink traversal")
        # Inventory paths are already bounded by the compiler JSON read. Start
        # with these names so a crowded directory cannot hide a later-stage
        # dump beyond a filesystem iteration prefix. Only direct children of
        # the fixed dump directory are considered.
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
        if not candidates:
            report["reason"] = "No retained .irpb dumps"
            return _finish(report)
        if schema_library is None and compiled.get("ddk_root"):
            schema_library = Path(compiled["ddk_root"]) / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
        if schema_library is None:
            report["reason"] = "No selected DDK schema library"
            return _finish(report)
        schema = load_schema(schema_library)
        enums = _enum_fields(schema)
        report.update(schema=schema.evidence, tensor_descriptor_enum_fields=enums)
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
        report["read_bytes"] = total
        for source in sources:
            path = Path(source["path"])
            if path.stat().st_size != source["bytes"] or sha256_file(path) != source["sha256"]:
                raise ValueError("inspection source changed during analysis")
        complete = (not report.get("directory_scan_limited") and not report["files_omitted"]
                    and not report.get("output_limited") and all(r["status"] == "DECODED" for r in report["files"])
                    and not any(g.get("capture_limited") for r in report["files"] for g in r.get("matches", [])))
        report["status"] = "INSPECTED" if complete else "PARTIAL"
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        report.update(status="NOT_AVAILABLE", reason=f"{type(error).__name__}: {str(error)[:512]}")
        # Never retain decoded conclusions after a late metadata integrity error.
        for file in report["files"]:
            file.pop("matches", None)
    return _finish(report)

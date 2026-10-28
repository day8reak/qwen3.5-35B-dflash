"""Inspect one retained standalone-draft compiler failure without running tools.

Recipe metadata is the declared source graph, not a decoded optimized graph.
No model weights are opened and no preparation, compilation or board work runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .artifacts import output_path, save_json
from .compile_inspect import _failure_context
from .compiler_support import support_snapshot
from .ge_recipe_read import read_recipe
from .hashing import sha256_file


_MAX_JSON = 8 << 20
_MAX_LOG = 64 << 20
_MAX_RECIPE = 256 << 20


def _read(path, limit, label, sources, expected=None):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError(f"{label} must be a regular file without symlink traversal")
    size = path.stat().st_size
    if size > limit:
        raise ValueError(f"{label} exceeds {limit} byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or len(raw) != size:
        raise ValueError(f"{label} changed size while reading")
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and (size != expected.get("bytes") or digest != expected.get("sha256")):
        raise ValueError(f"{label} SHA256/size mismatch")
    sources.append({"path": str(path), "bytes": size, "sha256": digest})
    return raw


def _json(path, sources, expected=None):
    document = json.loads(_read(path, _MAX_JSON, "metadata", sources, expected))
    if not isinstance(document, dict):
        raise ValueError("metadata must be a JSON object")
    return document


def _bound(root, record):
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("metadata path must stay inside its declared bundle")
    path = root / relative
    if root not in path.resolve().parents:
        raise ValueError("metadata path escapes its declared bundle")
    return path


def _one(rows, predicate, label):
    found = [row for row in rows if predicate(row)]
    if len(found) != 1:
        raise ValueError(f"expected one {label}")
    return found[0]


def _node_metadata(recipe, name):
    records = recipe["records"]
    matches = [record for record in records if record["name"] == name]
    if not matches:
        return {"status": "NOT_FOUND", "requested_name": name}
    node = matches[0]
    producers = []
    for port in node["inputs"]:
        source = records[port["src"]]
        producers.append({"consumer_port": port["port"], "source_port": port["src_port"],
                          "node": source,
                          "input_producers": [{"consumer_port": p["port"], "source_port": p["src_port"],
                                               "name": records[p["src"]]["name"],
                                               "type": records[p["src"]]["type"],
                                               "outputs": records[p["src"]]["outputs"]}
                                              for p in source["inputs"][:8]]})
    return {"status": "EXACT_NAME_MATCHED", "node": node, "input_producers": producers,
            "scope": "Construction recipe descriptors before OMG optimization; weight payloads not read."}


def _contexts(text, name, producer_names):
    lines = text.splitlines()
    pattern = re.compile(r"(?<![A-Za-z0-9_./-])" + re.escape(name) + r"(?![A-Za-z0-9_./-])")
    hits = [index for index, line in enumerate(lines) if pattern.search(line)]
    rejections = [index for index in hits if re.search(r"don.t support|not support|unsupported|failed|reject", lines[index], re.I)]
    producers = [index for index, line in enumerate(lines) if any(p in line for p in producer_names)]
    selected = set()
    # Keep the last exact-node rejection and neighboring store diagnostics first.
    for anchor in [*reversed(rejections[-8:]), *reversed(hits[-16:]), *hits[:8], *reversed(producers[-4:])]:
        for index in [anchor, *range(max(0, anchor - 6), anchor),
                      *range(anchor + 1, min(len(lines), anchor + 7))]:
            if len(selected) >= 96:
                break
            selected.add(index)
    checks = []
    for index in rejections[-8:]:
        if index == 0:
            continue
        previous = index - 1
        wrappers = []
        if index >= 2 and re.search(
                r"\bbatchmatmul_cube_check_support\.cc\b.*\bIsSupport\b.*\bCheckBatchMatMulCubeSupport\s+failed\b",
                lines[previous]):
            wrappers = [previous + 1]
            previous -= 1
        limit = re.search(r"\bMatrix\s+k\s+greater\s+than\s+(\d+)\s+is\s+not\s+supported\b", lines[previous], re.I)
        if limit is not None:
            checks.append({"kind": "K_LIMIT_CHECK_REJECTION", "reported_limit": int(limit[1]),
                           "limit_line": previous + 1, "node_rejection_line": index + 1,
                           "support_wrapper_lines": wrappers,
                           "text": lines[previous][:2000],
                           "scope": "Limit message precedes this exact node's compiler rejection, directly or through one known support-check wrapper; not a runtime support claim."})
    return {"exact_node_occurrences": len(hits), "line_limit": 96,
            "adjacent_check_evidence": checks,
            "lines": [{"line": index + 1, "text": lines[index][:2000]} for index in sorted(selected)],
            "note": "Exact-node and neighboring per-store messages; individual store rejection is not global support proof."}


def _ir_node_summary(node):
    result = {"name": node.get("name"), "type": node.get("type")}
    for key in ("input_desc", "output_desc"):
        result[key] = []
        for descriptor in node.get(key, [])[:4]:
            raw = descriptor.get("raw", {})
            summary = {field: raw[field] for field in ("dtype", "shape", "layout", "format") if field in raw}
            summary["schema_enum_names"] = descriptor.get("schema_enum_names", {})
            result[key].append(summary)
    result["attributes"] = node.get("attributes", [])
    return json.dumps(result, ensure_ascii=False)


def _render(report):
    lines = ["Standalone draft: retained compilation inspection", f"status={report['status']}",
             "Read-only inspection; no compilation, reference execution or board execution.",
             f"Job: {report['job']}", f"Node: {report['node_name']}",
             "Compiler options: " + json.dumps(report["requested_options"], sort_keys=True),
             "AIR/log recipe binding: " + report["association"]["status"]]
    declared = report["declared_graph"]
    if declared["status"] == "EXACT_NAME_MATCHED":
        lines.append("Declared node: " + json.dumps(declared["node"], ensure_ascii=False))
        for producer in declared["input_producers"]:
            lines.append("Declared input producer: " + json.dumps(producer, ensure_ascii=False))
    else:
        lines.append("Requested node is absent from the construction recipe.")
    log = report["log"]
    lines.append("Compiler log: " + log["status"])
    for check in log.get("node_context", {}).get("adjacent_check_evidence", []):
        wrappers = check.get("support_wrapper_lines", [])
        relation = ("via support wrapper " + ", ".join(f"L{line}" for line in wrappers)
                    if wrappers else "immediately")
        lines.append(f"{check['kind']}: reported_limit={check['reported_limit']}; "
                     f"L{check['limit_line']} {relation} before node rejection L{check['node_rejection_line']}")
    for row in log.get("node_context", {}).get("lines", []):
        lines.append(f"L{row['line']}: {row['text']}")
    if not log.get("node_context", {}).get("lines"):
        for row in log.get("failure_context", {}).get("lines", []):
            lines.append(f"L{row['line']}: {row['text']}")
    lines.append("IR inspection: " + report["ir_inspection"].get("status", "UNAVAILABLE"))
    ir = report["ir_inspection"]
    for key in ("schema_error", "inventory_error", "directory_error"):
        if ir.get(key):
            lines.append(key + ": " + str(ir[key])[:500])
    for row in ir.get("files", [])[:16]:
        lines.append(f"IR dump: {row.get('path')} status={row.get('status')}" +
                     (" error=" + str(row["error"])[:300] if row.get("error") else ""))
    matched = [row for row in ir.get("files", []) if row.get("matches")]
    for row in matched[-4:]:
        for match in row["matches"][:2]:
            lines.append(f"IR exact node [{row.get('path')}; graph={match.get('graph_name')}]: " +
                         _ir_node_summary(match["node"]))
            for edge in match.get("input_producers", [])[:4]:
                if edge.get("producer"):
                    lines.append(f"IR direct producer [{edge.get('field')} input={edge.get('input_index')}]: " +
                                 _ir_node_summary(edge["producer"]))
            for weight in match.get("named_fc_weights", [])[:2]:
                lines.append("IR same-graph named weight (edge not inferred): " + _ir_node_summary(weight))
    lines.append("Retained dump inventory: " + json.dumps(report["dump_inventory"], ensure_ascii=False))
    lines.extend(report["limitations"])
    return "\n".join(lines) + "\n"


def inspect_draft_compile(job_dir, output_dir, *, node="__ge/v20/flat_output", schema_library=None):
    """Inspect recorded metadata/logs and optionally decode selected-DDK dumps.

    Input AIR identity is matched between recorded snapshot and export hashes;
    inspection does not reread AIR-contained weights or external weight files.
    """
    if (not isinstance(node, str) or not node or len(node) > 512
            or any(ord(c) < 32 or ord(c) == 127 for c in node)):
        raise ValueError("node must be nonempty bounded text without controls")
    job, output = Path(job_dir).resolve(), output_path(output_dir)
    if job == output or job in output.parents or output in job.parents:
        raise ValueError("inspection output must be outside the original job")
    sources = []
    job_path = job / ("result.json" if (job / "result.json").is_file() else "job.json")
    job_record = _json(job_path, sources)
    if job_record.get("artifact_kind") != "qwen35-full-draft-job":
        raise ValueError("expected standalone full-draft job")
    components = job_record.get("components", {})
    air_component = components.get("air")
    if air_component is None:
        air_component = _json(job / "completed-air.json", sources)
    air_root = Path(air_component["directory"]).resolve()
    air_manifest = _json(_bound(air_root, air_component["manifest"]), sources, air_component["manifest"])
    if air_manifest.get("artifact_kind") != "qwen35-cpro-draft-air" or air_manifest.get("status") != "EXPORTED":
        raise ValueError("expected exported standalone draft AIR metadata")
    graph = _one(air_manifest.get("graphs", []), lambda row: row.get("name") == "draft", "draft graph")
    recipe_record = _one(graph.get("construction_files", []),
                         lambda row: row.get("path") == "draft/recipe/graph.recipe", "hash-bound draft recipe")
    recipe_path = _bound(air_root, recipe_record)
    _read(recipe_path, _MAX_RECIPE, "recipe", sources, recipe_record)
    recipe = read_recipe(recipe_path)
    if recipe["graph_name"] != "draft":
        raise ValueError("recipe graph must be draft")
    declared = _node_metadata(recipe, node)
    compilation_component = components.get("compilation")
    compilation_root = (Path(compilation_component["directory"]).resolve()
                        if compilation_component is not None else job / "compilation")
    compilation_path = (_bound(compilation_root, compilation_component["manifest"])
                        if compilation_component is not None else compilation_root / "report.json")
    compiled = (_json(compilation_path, sources, compilation_component["manifest"] if compilation_component else None)
                if compilation_path.exists() else {})
    association = {"status": "UNAVAILABLE", "reason": "compiler input AIR snapshot identity unavailable"}
    if compiled:
        if compiled.get("artifact_kind") != "qwen35-existing-air-compilation":
            raise ValueError("unexpected compilation artifact kind")
        snapshots = [row for row in compiled.get("input_snapshots", []) if row.get("role") == "air"]
        if snapshots:
            snapshot = _one(snapshots, lambda row: True, "compiled AIR snapshot")
            exported = _one(graph.get("files", []), lambda row: row.get("path") == graph["air"], "exported AIR identity")
            if any(snapshot.get(key) != exported.get(key) for key in ("bytes", "sha256")):
                raise ValueError("compiled AIR snapshot identity differs from the recipe's export manifest")
            if compiled.get("air") and any(compiled["air"].get(key) != exported.get(key) for key in ("bytes", "sha256")):
                raise ValueError("compiler source AIR identity differs from the export manifest")
            association = {"status": "RECORDED_AIR_HASH_MATCHED", "air_sha256": exported["sha256"],
                           "scope": "Recorded input snapshot matches export metadata; AIR and weight payloads were not reread."}
    log_path = compilation_root / "compile.log"
    log = {"status": "UNAVAILABLE", "path": str(log_path), "reason": "retained compiler log not present"}
    if log_path.exists():
        raw = _read(log_path, _MAX_LOG, "compiler log", sources)
        log_hash = hashlib.sha256(raw).hexdigest()
        expected = compiled.get("compile_log_sha256")
        if expected is not None and expected != log_hash:
            raise ValueError("compiler log SHA256 mismatch")
        text = raw.decode("utf-8", errors="replace")
        support = support_snapshot(text)
        producers = [row["node"]["name"] for row in declared.get("input_producers", [])]
        log = {"status": "AVAILABLE", "path": str(log_path), "sha256": log_hash,
               "hash_status": "MATCHED_COMPILER_RECORD" if expected else "CAPTURED_WITHOUT_PRIOR_HASH",
               "node_context": _contexts(text, node, producers), "failure_context": _failure_context(text),
               "support_snapshot": {key: support[key] for key in ("status", "scope", "counts", "window")},
               "exact_node_support": [row for row in support["nodes"] if row["name"] == node]}
        if association["status"] == "UNAVAILABLE":
            log["exact_node_support"] = []
            log["association_note"] = "Raw text only; no compiler-record binding to this recipe."
    inventory = []
    for record in [*compiled.get("generated_artifacts", []), *compiled.get("compiler_work_artifacts", [])]:
        if str(record.get("path", "")).endswith(".irpb"):
            inventory.append({key: record[key] for key in ("path", "bytes", "sha256") if key in record})
    selected_library = schema_library
    if selected_library is None and compiled.get("ddk_root"):
        selected_library = (Path(compiled["ddk_root"]) / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so")
    ir = {"status": "UNAVAILABLE", "reason": "no selected DDK schema library recorded"}
    if selected_library is not None:
        from .draft_ir_inspect import inspect_draft_ir
        ir = inspect_draft_ir(compilation_root, selected_library, node_name=node)
    report = {"schema_version": 1, "artifact_kind": "qwen35-draft-compile-inspection",
              "status": "INSPECTED" if association["status"] != "UNAVAILABLE" and log["status"] == "AVAILABLE" else "PARTIAL",
              "job": str(job), "output": str(output), "comparison": str(output / "report.txt"),
              "node_name": node, "sources": sources, "association": association,
              "requested_options": compiled.get("requested_options", job_record.get("compile_options", {})),
              "declared_graph": declared, "log": log, "dump_inventory": inventory[:32],
              "dump_inventory_truncated": len(inventory) > 32, "ir_inspection": ir,
              "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN",
              "limitations": ["Recipe and log names do not establish optimized input descriptors.",
                              "Per-store support checks do not establish final placement or kernel correctness.",
                              "IR dumps are decoded only by the separate selected-schema inspector; no guessed schema fallback.",
                              "No AIR file or external weight file was read; prior compilation result is unchanged."]}
    for source in sources:
        path = Path(source["path"])
        if path.stat().st_size != source["bytes"] or sha256_file(path) != source["sha256"]:
            raise ValueError("inspection input changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", report)
    (output / "report.txt").write_text(_render(report), encoding="utf-8")
    return report

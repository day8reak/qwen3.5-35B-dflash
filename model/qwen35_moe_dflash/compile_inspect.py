"""Read-only, manifest-bound compiler support diagnostics; never infer placement."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .artifacts import checked_file, output_path, save_json
from .compiler_support import support_snapshot
from .ge_recipe_read import read_recipe
from .hashing import sha256_file
from .tool_logs import _is_error


_BOUNDARY_TYPES = {"Data", "Const", "FileConstant", "NetOutput"}
_LIMITATIONS = [
    "Support sets are compiler candidates before rollback, not final partitions or runtime placement.",
    "Recipe shapes and edges describe construction before OMG optimization; exact names do not prove unchanged compiled descriptors.",
    "Compiler-created or renamed nodes remain unmatched; no prefix stripping or fuzzy association is used.",
    "Candidate cross-device edges are diagnostic hints, not the number of NPU partitions.",
    "No compilation, board execution, weight payload loading, or precision comparison is performed.",
]


def _one(rows, name, label):
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label} record for {name}")
    return matches[0]


def _category(node):
    if node is None or node.get("device_conflict"):
        return "unknown"
    ids = node.get("device_ids", [])
    if 0 in ids:
        return "npu_candidate"
    if ids == [1]:
        return "cpu_only_candidate"
    return "other_or_unknown"


def _read_bound_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _compilation_binding(compilation, compiled, graph, log):
    """Accept old early failures without inventing a compiled graph record."""
    expected_status = "FAILED" if compilation.name == "failure.json" else "COMPILED"
    if (compiled.get("artifact_kind") != "qwen35-cpro-omc-pair"
            or compiled.get("status") != expected_status):
        raise ValueError("expected a matching OMC artifact kind and compilation status")
    rows = compiled.get("graphs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("expected a compiled graphs list")
    matches = [row for row in rows if row.get("name") == graph]
    if len(matches) > 1:
        raise ValueError(f"expected exactly one compiled graph record for {graph}")
    if matches:
        return matches[0], "compiled_graph_record"
    if (compilation.name != "failure.json" or compiled.get("failed_graph") != graph
            or compiled.get("stage") not in {f"{graph}-omg", f"{graph}-output-check", f"{graph}-collect"}):
        raise ValueError("missing compiled graph is not the recorded early failed graph/stage")
    if [row.get("name") for row in rows] != ([] if graph == "target" else ["target"]):
        raise ValueError("early failure compiled graphs do not match the completed graph prefix")
    # compile_air records the absolute path. Do not search sibling logs or
    # resolve a different path supplied by the failure manifest.
    if compiled.get("log_path") != str(log):
        raise ValueError("early failure log_path differs from the requested bundle graph log")
    return None, "early_failure_record"


def _early_snapshot(snapshot, text, graph):
    """Report unavailable support evidence without manufacturing a boundary."""
    names = re.findall(r"\bInferShape\s+for\s+Graph\s*(?:\[([^]]+)\]|([^\s\[\]\"']+))", text, re.I)
    observed = {a or b for a, b in names if "constnodegraph" not in (a or b).lower()
                and "subgraph" not in (a or b).lower()}
    if observed - {graph}:
        raise ValueError("compiler log graph name differs from requested recipe graph")
    if snapshot["status"] != "unknown":
        return snapshot
    has_unbound_rows = bool(re.search(r"\bopDeviceSupport\s*(?:size\s*:|:\s*op\s*=)", text, re.I))
    snapshot.update(status="partial" if has_unbound_rows else "missing",
                    reason="early_failure_support_window_unavailable" if has_unbound_rows else "early_failure_support_snapshot_missing",
                    association_status="NOT_ASSOCIATED_WITHOUT_BOUNDED_SUPPORT_WINDOW")
    # Unknown/constant-folding rows must not become main-graph candidates.
    snapshot["unbound_support_text_observed"] = has_unbound_rows
    return snapshot


def _failure_context(text):
    """Keep bounded failure-chain windows, rather than only cleanup at EOF."""
    lines = text.splitlines()
    details, errors = [], []
    for number, line in enumerate(lines, 1):
        if not _is_error(line.encode("utf-8")):
            continue
        if re.search(r"deinit|finaliz|unregister|cleanup|::~|~[A-Za-z]", line, re.I):
            continue
        if not re.search(r"fail|error|false\s*,\s*return|precheck|bypassnode", line, re.I):
            continue
        errors.append(number)
        if re.search(r"reshape.*fusion|bypassnode|precheck|afterinfershape|pass.*run.*fail", line, re.I):
            details.append(number)
    selected = set()
    for anchor in [*details[-4:], *errors[-8:]]:
        for number in (anchor, *range(max(1, anchor - 4), anchor), *range(anchor + 1, min(len(lines), anchor + 2) + 1)):
            if len(selected) == 60:
                break
            selected.add(number)
    return {"status": "AVAILABLE" if selected else "NO_ERROR_CONTEXT_FOUND",
            "note": "Bounded diagnostic failure context; these lines are not a proven root cause.",
            "lines": [{"line": number, "text": lines[number - 1][:1000]} for number in sorted(selected)]}


def correlate_support(snapshot, recipe):
    """Join exact names only; preserve separate log and construction types."""
    records = recipe["records"]
    declared = {row["name"]: row for row in records}
    supported = {row["name"]: row for row in snapshot["nodes"]}
    consumers = defaultdict(set)
    for row in records:
        for port in row["inputs"]:
            consumers[records[port["src"]]["name"]].add(row["name"])
    groups = defaultdict(list)
    details, excluded, unresolved = [], [], []
    for node in snapshot["nodes"]:
        if _category(node) != "cpu_only_candidate":
            continue
        name, log_type = node["name"], node.get("type", "unknown")
        record = declared.get(name)
        if log_type in _BOUNDARY_TYPES or (record and record["type"] in _BOUNDARY_TYPES):
            excluded.append(name)
            continue
        if node.get("type_conflict") or (record is None and log_type in {None, "?", "unknown"}):
            unresolved.append(node)
            continue
        detail = {"name": name, "log_type": log_type, "device_ids": node["device_ids"],
                  "support_line": node["support_line"], "recipe_match": "exact_name" if record else "unmatched"}
        signature = [log_type, record["type"] if record else "unmatched", []]
        if record:
            detail.update(recipe_type=record["type"],
                          inputs=[{k: p[k] for k in ("port", "shape", "dtype")} for p in record["inputs"]],
                          outputs=record["outputs"], attrs=record["attrs"])
            signature[2] = [(p["dtype"], len(p["shape"])) for p in record["inputs"]]
            parents = sorted({records[p["src"]]["name"] for p in record["inputs"]
                              if records[p["src"]]["type"] not in _BOUNDARY_TYPES})
            children = sorted(consumers[name])
            detail["candidate_neighbors"] = {
                direction: [{"name": other, "candidate": _category(supported.get(other))} for other in names]
                for direction, names in (("inputs", parents), ("outputs", children))}
            detail["has_npu_candidate_input"] = any(_category(supported.get(n)) == "npu_candidate" for n in parents)
            detail["has_npu_candidate_output"] = any(_category(supported.get(n)) == "npu_candidate" for n in children)
        details.append(detail)
        groups[json.dumps(signature)].append(detail)
    summaries = []
    for signature, rows in groups.items():
        log_type, recipe_type, inputs = json.loads(signature)
        summaries.append({"log_type": log_type, "recipe_type": recipe_type,
                          "declared_input_dtype_rank": inputs, "count": len(rows),
                          "with_npu_candidate_neighbors_on_both_sides": sum(
                              bool(r.get("has_npu_candidate_input") and r.get("has_npu_candidate_output")) for r in rows),
                          "examples": rows[:3]})
    summaries.sort(key=lambda r: (-r["count"], r["log_type"], r["recipe_type"], str(r["declared_input_dtype_rank"])))
    return {"groups": summaries, "cpu_only_compute_candidates": details,
            "excluded_boundary_candidates": excluded,
            "unresolved_cpu_candidates": unresolved,
            "candidate_counts": dict(Counter(_category(n) for n in snapshot["nodes"])),
            "exact_name_matches": sum(n["name"] in declared for n in snapshot["nodes"]),
            "unmatched_names": [n["name"] for n in snapshot["nodes"] if n["name"] not in declared]}


def _render(report):
    snap, joined = report["support_snapshot"], report["association"]
    lines = [f"Compiler support diagnostic: {report['graph']}",
             f"Support snapshot: {snap['status']}; scope={snap.get('scope')}; "
             f"log lines {snap.get('start_line')}..{snap.get('end_line')}",
             f"Checklist: {json.dumps(snap['counts'], sort_keys=True)}",
             f"Candidates: {json.dumps(joined['candidate_counts'], sort_keys=True)}",
             f"Exact recipe name matches: {joined['exact_name_matches']}; unmatched: {len(joined['unmatched_names'])}",
             f"Excluded Data/Const/FileConstant/NetOutput candidates: {len(joined['excluded_boundary_candidates'])}",
             f"CPU candidates with unresolved node role: {len(joined['unresolved_cpu_candidates'])}",
             "", "CPU-only compute candidates (construction shapes, not final placement):"]
    if report["compilation_record"]["kind"] == "early_failure_record":
        lines[1:1] = [f"Early compilation failure: stage={report['compilation_record']['stage']}",
                      "OMG log hash: captured for this diagnostic; no historical compiler log hash was recorded.",
                      f"Support evidence: {snap.get('reason')}; unbounded support text is not associated with recipe nodes."]
    for group in joined["groups"][:16]:
        input_types = group["declared_input_dtype_rank"]
        type_suffix = f" (+{len(input_types)-8} inputs)" if len(input_types) > 8 else ""
        lines.append(f"  {group['log_type']} / recipe={group['recipe_type']}: count={group['count']}, "
                     f"input(dtype,rank)={input_types[:8]}{type_suffix}, "
                     f"NPU-candidate neighbors both sides={group['with_npu_candidate_neighbors_on_both_sides']}")
        for row in group["examples"][:2]:
            shapes = [p["shape"] for p in row.get("inputs", [])]
            suffix = f" (+{len(shapes)-8} inputs)" if len(shapes) > 8 else ""
            lines.append(f"    {row['name']} @L{row['support_line']}: inputs={shapes[:8]}{suffix}, match={row['recipe_match']}")
    if len(joined["groups"]) > 16:
        lines.append(f"  {len(joined['groups']) - 16} further groups in report.json")
    lines.extend(["", "Representative compiler diagnostics (candidate checks may be non-fatal):"])
    for row in snap.get("diagnostics", [])[:32]:
        lines.append(f"  L{row['line']}: {row['text'][:800]}")
    if "failure_context" in report:
        lines.extend(["", "Terminal failure context (diagnostic candidates, not a proven root cause):"])
        for row in report["failure_context"]["lines"]:
            lines.append(f"  L{row['line']}: {row['text']}")
    lines.extend(["", *_LIMITATIONS])
    return "\n".join(lines) + "\n"


def inspect_compile(air_dir, omc_dir, output_dir, *, graph="target"):
    if graph not in {"target", "draft"}:
        raise ValueError("graph must be target or draft")
    air, omc = Path(air_dir).resolve(), Path(omc_dir).resolve()
    air_manifest = air / "manifest.json"
    air_doc, air_hash = _read_bound_json(air_manifest)
    if air_doc.get("artifact_kind") != "qwen35-cpro-air-pair" or air_doc.get("status") != "EXPORTED":
        raise ValueError("expected successfully exported AIR bundle")
    available = [p for p in (omc / "manifest.json", omc / "failure.json") if p.is_file()]
    if len(available) != 1:
        raise ValueError("expected exactly one OMC manifest.json or failure.json")
    compilation = available[0]
    compiled, compiled_hash = _read_bound_json(compilation)
    if compiled.get("air_manifest_sha256") != air_hash:
        raise ValueError("AIR manifest does not match the compilation record")
    ag = _one(air_doc.get("graphs", []), graph, "AIR graph")
    log = omc / graph / "omg.log"
    cg, binding = _compilation_binding(compilation, compiled, graph, log)
    rel = f"{graph}/recipe/graph.recipe"
    entries = [r for r in ag.get("construction_files", []) if r.get("path") == rel]
    if len(entries) != 1:
        raise ValueError("expected hash-locked graph.recipe in AIR construction_files")
    recipe_path = checked_file(air, entries[0])
    if log.is_symlink() or log.resolve() != log or not log.is_file() or log.stat().st_size > 256 << 20:
        raise ValueError("expected regular OMG log no larger than 256 MiB")
    log_bytes = log.read_bytes()
    if len(log_bytes) > 256 << 20:
        raise ValueError("expected regular OMG log no larger than 256 MiB")
    log_hash = hashlib.sha256(log_bytes).hexdigest()
    if cg is not None and cg.get("compiler_evidence", {}).get("log_sha256") != log_hash:
        raise ValueError("OMG log does not match recorded compiler evidence")
    recipe = read_recipe(recipe_path)
    if recipe["graph_name"] != graph:
        raise ValueError("recipe graph name differs from requested graph")
    log_text = log_bytes.decode("utf-8", errors="replace")
    snapshot = support_snapshot(log_text)
    if cg is None:
        snapshot = _early_snapshot(snapshot, log_text, graph)
    if snapshot.get("graph") not in {None, recipe["graph_name"]}:
        raise ValueError("compiler log graph name differs from requested recipe graph")
    report = {"schema_version": 1, "status": "DIAGNOSTIC_COMPLETE", "graph": graph,
              "device_execution": "NOT_RUN", "runtime_placement": "UNVERIFIED",
              "compilation_record": {"kind": binding, "status": compiled["status"],
                                     "failed_graph": compiled.get("failed_graph"),
                                     "stage": compiled.get("stage"),
                                     "requested_graph_has_compiled_record": cg is not None},
              "sources": {"air_manifest": {"path": str(air_manifest), "sha256": air_hash},
                          "compilation": {"path": str(compilation), "sha256": compiled_hash},
                          "recipe": {"path": str(recipe_path), "sha256": entries[0]["sha256"]},
                          "log": {"path": str(log), "sha256": log_hash,
                                  "hash_origin": "compiler_record" if cg is not None else "diagnostic_capture",
                                  "historical_hash_status": "VERIFIED" if cg is not None else "NOT_RECORDED"}},
              "device_id_legend": {"0": "NPU candidate", "1": "CPU candidate", "2": "GPU candidate", "3": "ISP candidate"},
              "device_id_source": "Declared blue DDK compatible/HiAiModelBuilderType.h ExecuteDevice; raw internal log IDs retained",
              "support_snapshot": snapshot, "association": correlate_support(snapshot, recipe),
              "limitations": _LIMITATIONS}
    if compilation.name == "failure.json" and compiled.get("failed_graph") == graph:
        report["failure_context"] = _failure_context(log_text)
    # Detect mutation during analysis without opening referenced weight data.
    for item in report["sources"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("diagnostic input changed during analysis")
    output = output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", report)
    (output / "report.txt").write_text(_render(report), encoding="utf-8")
    return report

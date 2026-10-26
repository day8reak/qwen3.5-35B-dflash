"""Read-only, manifest-bound compiler support diagnostics; never infer placement."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .artifacts import checked_file, output_path, save_json
from .compiler_support import support_snapshot
from .ge_recipe_read import read_recipe
from .hashing import sha256_file


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
    for group in joined["groups"][:16]:
        lines.append(f"  {group['log_type']} / recipe={group['recipe_type']}: count={group['count']}, "
                     f"input(dtype,rank)={group['declared_input_dtype_rank']}, "
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
    cg = _one(compiled.get("graphs", []), graph, "compiled graph")
    rel = f"{graph}/recipe/graph.recipe"
    entries = [r for r in ag.get("construction_files", []) if r.get("path") == rel]
    if len(entries) != 1:
        raise ValueError("expected hash-locked graph.recipe in AIR construction_files")
    recipe_path = checked_file(air, entries[0])
    log = omc / graph / "omg.log"
    if log.is_symlink() or not log.is_file() or log.stat().st_size > 256 << 20:
        raise ValueError("expected regular OMG log no larger than 256 MiB")
    log_hash = sha256_file(log)
    if cg.get("compiler_evidence", {}).get("log_sha256") != log_hash:
        raise ValueError("OMG log does not match recorded compiler evidence")
    recipe = read_recipe(recipe_path)
    if recipe["graph_name"] != graph:
        raise ValueError("recipe graph name differs from requested graph")
    snapshot = support_snapshot(log.read_text(encoding="utf-8", errors="replace"))
    if snapshot.get("graph") not in {None, recipe["graph_name"]}:
        raise ValueError("compiler log graph name differs from requested recipe graph")
    report = {"schema_version": 1, "status": "DIAGNOSTIC_COMPLETE", "graph": graph,
              "device_execution": "NOT_RUN", "runtime_placement": "UNVERIFIED",
              "sources": {"air_manifest": {"path": str(air_manifest), "sha256": air_hash},
                          "compilation": {"path": str(compilation), "sha256": compiled_hash},
                          "recipe": {"path": str(recipe_path), "sha256": entries[0]["sha256"]},
                          "log": {"path": str(log), "sha256": log_hash}},
              "device_id_legend": {"0": "NPU candidate", "1": "CPU candidate", "2": "GPU candidate", "3": "ISP candidate"},
              "device_id_source": "Declared blue DDK compatible/HiAiModelBuilderType.h ExecuteDevice; raw internal log IDs retained",
              "support_snapshot": snapshot, "association": correlate_support(snapshot, recipe),
              "limitations": _LIMITATIONS}
    # Detect mutation during analysis without opening referenced weight data.
    for item in report["sources"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("diagnostic input changed during analysis")
    output = output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", report)
    (output / "report.txt").write_text(_render(report), encoding="utf-8")
    return report

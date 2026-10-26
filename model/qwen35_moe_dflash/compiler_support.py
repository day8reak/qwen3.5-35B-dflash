"""Read-only, standard-library OMG support snapshots before a main rollback.

The logged device IDs are support candidates, not execution placement. A
rejection by one NPU store is not a rejection by all stores or by the compiler.
"""
from __future__ import annotations

import re

from .compiler_evidence import compile_evidence


_MAIN_BEGIN = re.compile(
    r"\bTCH\s+(?:iCompiler\s*->\s*Compile\s+begin|BuildOrigin2IRGraph\s+end)\b", re.I)
_GRAPH = re.compile(r"\bInferShape\s+for\s+Graph\s*(?:\[([^]]+)\]|([^\s\[\]\"']+))", re.I)
_SIZE = re.compile(r"\bopDeviceSupport\s+size\s*:\s*(\d+)\b", re.I)
_ENTRY = re.compile(r"\bopDeviceSupport\s*:\s*op\s*=\s*(.+?),\s*devices\s*=\s*\[([^]]*)\]", re.I)
_CHECK_TYPE = re.compile(
    r"\bCheckSupported\s*:\s*the\s+op\s+name\s*\[([^]]+)\]\s*type\s*\[([^]]+)\]", re.I)
_INFER_TYPE = re.compile(r"\[op:\s*(.*?)\s+type:\s*([^\]\s]+)\s*\]", re.I)
_STORE = re.compile(r"\bnot\s+supported\s+in\s+npucl\s+store\s*\[([^]]+)\]", re.I)


def _store_rounds(representatives):
    """Yield representatives by store rounds while preserving per-store order."""
    stores = {}
    for (store, _), number in representatives.items():
        stores.setdefault(store, []).append(number)
    index = 0
    while True:
        found = False
        for numbers in stores.values():
            if index < len(numbers):
                found = True
                yield numbers[index]
        if not found:
            return
        index += 1


def _contexts(lines, eligible, representatives, priority=None):
    """Prefer current CPU-only compute candidates, with a global 32-line cap.

    First give frequent compute types one context each, rotating stores. Then
    fill with the remaining candidate/store pairs and finally generic checks.
    This prevents Data or earlier multi-device candidates from hiding a large
    ReduceSum/ReduceMean population in the current support table.
    """
    priority = priority or {}
    stores = list(dict.fromkeys(store for store, _ in priority))
    types = list(dict.fromkeys(op_type for _, op_type in priority))
    first_per_type, cursor = [], 0
    for op_type in types:
        for offset in range(len(stores)):
            index = (cursor + offset) % len(stores)
            key = (stores[index], op_type)
            if key in priority:
                first_per_type.append(priority[key])
                cursor = (index + 1) % len(stores)
                break
    selected, anchors = set(), set()
    for group in (first_per_type, _store_rounds(priority), _store_rounds(representatives)):
        for number in group:
            if number in anchors:
                continue
            anchors.add(number)
            # Retain the actual rejection first even when only one slot remains.
            for candidate in (number, *range(number - 4, number), number + 1):
                if candidate in eligible:
                    selected.add(candidate)
                    if len(selected) == 32:
                        break
            if len(selected) == 32:
                break
        if len(selected) == 32:
            break
    return [{"line": number, "text": lines[number - 1][:1600]} for number in sorted(selected)]


def _cpu_representatives(nodes, node_rejections):
    """Select exact-name representatives using only the final table's raw IDs."""
    by_type = {}
    for node in nodes:
        if (node["device_ids"] == [1] and not node["device_conflict"]
                and not node["type_conflict"]
                and node["type"] not in {"unknown", "Data", "Const", "NetOutput", "Input", "Output"}):
            by_type.setdefault(node["type"], []).append(node)
    priority, summary = {}, []
    for op_type, candidates in sorted(by_type.items(), key=lambda pair: -len(pair[1])):
        for node in candidates:
            records = {key: line for key, line in node_rejections.get(node["name"], {}).items()
                       if key[1] == op_type}
            if records:
                priority.update(records)
                summary.append({"type": op_type, "count": len(candidates), "name": node["name"]})
                break
    return priority, summary


def support_snapshot(logtext: str) -> dict:
    """Extract the last support-size block in a clearly bounded compile window.

    Without an explicit main-pipeline begin marker before the latest main or
    unknown-scope rollback, no node table is inferred. Constant/subgraph windows
    are excluded. Missing nodes never mean that those operators are unsupported.
    """
    evidence = compile_evidence(logtext)
    events = [event for event in evidence["rollback_events"]
              if event["scope"] in {"main", "unknown"}]
    rollback = events[-1] if events else None
    result = {
        "status": "unknown", "scope": rollback["scope"] if rollback else "unknown",
        "graph": rollback["graph"] if rollback else None,
        "start_line": None, "end_line": rollback.get("end_line", rollback["line"]) if rollback else None,
        "window": {"status": "unknown", "start_line": None, "end_line": None},
        "rollback": rollback, "nodes": [],
        "counts": {"expected_count": None, "distinct_count": 0, "entry_count": 0,
                   "duplicate_count": 0, "conflicting_count": 0, "malformed_count": 0},
        "diagnostics": [], "diagnostic_cpu_only_representatives": [], "runtime_placement": "UNVERIFIED",
        "device_id_legend": {"0": "NPU candidate", "1": "CPU candidate"},
        "device_id_legend_source": "HiAiModelBuilderType.h (blue DDK); candidate labels only",
        "note": "Support candidates are not final partition or runtime placement. Missing nodes are unknown; per-store rejections do not prove overall rejection.",
    }
    if rollback is None:
        result["reason"] = "no_main_or_unknown_rollback"
        return result
    lines = logtext.splitlines()
    starts = [number for number, line in enumerate(lines[:rollback["line"] - 1], 1)
              if _MAIN_BEGIN.search(line)]
    if not starts:
        result["reason"] = "no_explicit_main_compile_boundary"
        return result
    start, end = starts[-1], rollback.get("end_line", rollback["line"])
    boundary_graph = None
    for line in lines[:start]:
        match = _GRAPH.search(line)
        if match:
            graph = match.group(1) or match.group(2)
            if "constnodegraph" not in graph.lower() and "subgraph" not in graph.lower():
                boundary_graph = graph
    if (boundary_graph is not None and rollback["graph"] is not None
            and boundary_graph != rollback["graph"]):
        # A previous graph's compile marker is not a boundary for a later
        # graph merely because both happen to share the same log file.
        result["reason"] = "main_compile_boundary_graph_mismatch"
        return result
    result.update(start_line=start, end_line=end)
    result["window"] = {"status": "bounded", "start_line": start, "end_line": end,
                        "start_text": lines[start - 1][:1600], "end_text": lines[end - 1][:1600]}

    active, eligible, types, representatives, block = True, set(), {}, {}, None
    node_rejections = {}
    for number in range(start, rollback["line"]):
        line = lines[number - 1]
        match = _GRAPH.search(line)
        if match:
            graph = match.group(1) or match.group(2)
            active = ("constnodegraph" not in graph.lower() and "subgraph" not in graph.lower()
                      and (rollback["graph"] is None or graph == rollback["graph"]))
            if not active and block is not None:
                block["interrupted"] = True
        if re.search(r"\bConstantFolding\s+(?:begin|end|success|finish)\b", line, re.I):
            active = False
            if block is not None:
                block["interrupted"] = True
        if _MAIN_BEGIN.search(line):
            active = True
        if not active:
            continue
        eligible.add(number)
        type_match = _CHECK_TYPE.search(line)
        type_source = "CheckSupported"
        if type_match is None and "RunInferShape" in line:
            type_match = _INFER_TYPE.search(line)
            type_source = "RunInferShape"
        if type_match is not None:
            name, op_type = (value.strip() for value in type_match.groups())
            by_type = types.setdefault(name, {})
            # Retain one line from each explicit source per distinct type;
            # repeated stores do not create unbounded per-node excerpts.
            by_type.setdefault(op_type, {}).setdefault(type_source, number)
            store_match = _STORE.search(line)
            if store_match:
                key = (store_match.group(1), op_type)
                representatives.setdefault(key, number)
                node_rejections.setdefault(name, {}).setdefault(key, number)

        size_match = _SIZE.search(line)
        if size_match:
            # Always replace, including a later truncated/empty round.
            block = {"line": number, "expected": int(size_match.group(1)), "nodes": {},
                     "entries": 0, "malformed": 0, "duplicates": 0, "interrupted": False}
            continue
        if "opdevicesupport" not in line.lower():
            continue
        if re.search(r"\bopDeviceSupport\s+size\s*:", line, re.I):
            block = {"line": number, "expected": None, "nodes": {},
                     "entries": 0, "malformed": 1, "duplicates": 0, "interrupted": False}
            continue
        if block is None:
            continue
        match = _ENTRY.search(line)
        if match is None:
            block["malformed"] += 1
            continue
        name, raw = match.groups()
        name = name.strip()
        fields = raw.split(",")
        if fields and not fields[-1].strip():
            fields.pop()
        if not name or any(not re.fullmatch(r"[+-]?\d+", item.strip()) for item in fields):
            block["malformed"] += 1
            continue
        devices = [int(item) for item in fields]
        block["entries"] += 1
        if name not in block["nodes"]:
            block["nodes"][name] = {"name": name, "device_ids": devices,
                                     "device_variants": [devices], "device_conflict": False,
                                     "support_line": number, "support_lines": [number],
                                     "support_occurrences": 1}
        else:
            node = block["nodes"][name]
            block["duplicates"] += 1
            node["support_occurrences"] += 1
            if len(node["support_lines"]) < 8:
                node["support_lines"].append(number)
            if devices not in node["device_variants"]:
                node["device_variants"].append(devices)
            if set(devices) != set(node["device_variants"][0]):
                node["device_conflict"] = True
                node["device_ids"] = []

    result["diagnostic_representatives"] = len(representatives)
    if block is None:
        result["diagnostics"] = _contexts(lines, eligible, representatives)
        result["reason"] = "no_main_support_size_block"
        return result
    nodes = list(block["nodes"].values())
    for node in nodes:
        by_type = types.get(node["name"], {})
        node["types"] = list(by_type)
        node["type"] = next(iter(by_type)) if len(by_type) == 1 else "unknown"
        node["type_conflict"] = len(by_type) > 1
        node["type_evidence_lines"] = sorted(number for sources in by_type.values() for number in sources.values())
        node["type_evidence"] = [{"type": op_type, "source": source, "line": number}
                                 for op_type, sources in by_type.items() for source, number in sources.items()]
    priority, summary = _cpu_representatives(nodes, node_rejections)
    result["diagnostics"] = _contexts(lines, eligible, representatives, priority)
    result["diagnostic_cpu_only_representatives"] = summary
    conflicts = sum(node["device_conflict"] for node in nodes)
    result["nodes"] = nodes
    result["support_block_line"] = block["line"]
    result["counts"] = {"expected_count": block["expected"], "distinct_count": len(nodes),
                        "entry_count": block["entries"], "duplicate_count": block["duplicates"],
                        "conflicting_count": conflicts, "malformed_count": block["malformed"]}
    complete = (block["expected"] == len(nodes) and not conflicts
                and not block["malformed"] and not block["interrupted"])
    result["status"] = "complete" if complete else "partial"
    result["reason"] = "support_count_matches" if complete else "incomplete_or_conflicting_support_block"
    result["support_block_interrupted"] = block["interrupted"]
    return result

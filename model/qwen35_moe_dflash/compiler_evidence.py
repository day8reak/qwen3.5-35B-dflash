"""Standard-library-only OMG diagnostics with explicit graph-scope boundaries.

Compiler partitions and rollback messages are compilation evidence. They never
establish runtime placement, execution, or numerical correctness.
"""
from __future__ import annotations

import re


_PARTITIONS = re.compile(r"partition\s+type\s+NPU\s*:\s*(\d+)\s*,\s*CPU\s*:\s*(\d+)"
                         r"\s*,\s*GPU\s*:\s*(\d+)\s*,\s*ISP\s*:\s*(\d+)", re.I)
_GRAPH = re.compile(r"\binfershape\s+for\s+graph\s*(?:\[([^]]+)\]|([^\s\[\]\"']+))", re.I)
_MAIN_PHASE = re.compile(r"\btch\s+(?:buildorigin2irgraph\s+end|icompiler\s*->\s*compile\s+begin)\b", re.I)
_ROLLBACK_LIMIT = re.compile(r"\bpartition\s+of\s+type\s*:\s*(NPUCL)\s+is\s+more\s+than\s+(\d+)\b", re.I)
_ROLLBACK_CPU = re.compile(r"\bwill\s+rollback\s+to\s+CPU\b", re.I)


def _diagnostic_priority(line):
    """Rank concrete rejection details above noisy per-store capability checks."""
    low = line.lower()
    if "return hiai::success" in low or "infershape success" in low:
        return None
    if re.search(r"kernelid\s+not\s+registered|castt\s+selectkernel\s+failed", low):
        return -1, "missing_kernel"
    if re.search(r"modulenotfounderror|importerror|attributeerror|cannot open shared|no such file|tbeinitialize failed|failed to initialize tbe", low):
        return 0, "environment"
    if re.search(r"does(?:n't| not) exist.*tbe.*ops.*store|(?:not found|missing).*tbe.*ops.*store", low):
        return 0, "missing_kernel"
    if (re.search(r"dtype|data\s*type|rank|dim(?:num|cnt|ension)|realdimcnt", low)
            and re.search(r"wrong|must|should|support|>|<|exceed|limit|failed", low)):
        return 1, "candidate_check" if re.search(r"checksupport|check support", low) else "compiler_diagnostic"
    if re.search(r"infershape.*failed|verify.*failed|check ir model compatibility failed|beforecompile.*fail|failed to generator|omg generate.*failed|\[fatal\]", low):
        return 2, "compiler_diagnostic"
    if re.search(r"don.t support|not support|unsupported", low):
        return (4, "candidate_check") if re.search(r"checksupport|check support|not supported in .*store", low) else (3, "compiler_diagnostic")
    return None


def compile_evidence(logtext: str) -> dict:
    """Retain partition and explicit NPUCL-limit rollback events by graph scope.

    A named graph establishes its scope. Main-pipeline phase markers only end
    an inherited constant-folding scope: without a previously named main graph,
    subsequent events remain unknown. Support checks are candidate queries.
    """
    partitions, subgraphs, rollbacks = [], [], []
    diagnostics, diagnostic_seen, diagnostic_total = [], {}, 0
    graph, scope, main_graph, candidates = None, "unknown", None, 0
    pending_rollback = None
    for number, line in enumerate(logtext.splitlines(), 1):
        low = line.lower()
        graph_match = _GRAPH.search(line)
        if graph_match:
            graph = graph_match.group(1) or graph_match.group(2)
            scope = ("constant_folding" if "constnodegraph" in graph.lower() else
                     "subgraph" if "subgraph" in graph.lower() else "main")
            if scope == "main":
                main_graph = graph
        if re.search(r"constantfolding\s+begin", low):
            scope, graph = "constant_folding", None
        elif re.search(r"constantfolding\s+(?:end|success|finish)", low):
            scope, graph = "unknown", None
        if _MAIN_PHASE.search(line) and scope == "constant_folding":
            graph, scope = main_graph, "main" if main_graph is not None else "unknown"

        limit_match = _ROLLBACK_LIMIT.search(line)
        rollback_to_cpu = bool(_ROLLBACK_CPU.search(line))
        if (pending_rollback is not None and rollback_to_cpu and limit_match is None
                and (scope, graph) == (pending_rollback["scope"], pending_rollback["graph"])):
            # Some logger layouts split the one diagnostic across two adjacent
            # lines. Never associate it with a later, unrelated fallback.
            pending_rollback.update(end_line=number, text=pending_rollback["text"] + "\n" + line)
            rollbacks.append(pending_rollback)
        pending_rollback = None
        if limit_match:
            event = {"line": number, "scope": scope, "graph": graph,
                     "device": limit_match.group(1).upper(), "limit": int(limit_match.group(2)),
                     "reason": "partition_limit_exceeded", "target_device": "CPU", "text": line}
            if rollback_to_cpu:
                rollbacks.append(event)
            else:
                pending_rollback = event

        if any(word in low for word in ("opdevicesupport", "checksupported", "checksupport", "check support")):
            candidates += 1
        diagnostic = _diagnostic_priority(line)
        if diagnostic is not None:
            diagnostic_total += 1
            text = line[:1600]
            key = re.sub(r"\(\d+\)", "(PID)", text)
            if key in diagnostic_seen:
                diagnostic_seen[key]["occurrences"] += 1
            else:
                priority, category = diagnostic
                diagnostic_seen[key] = {"line": number, "text": text, "occurrences": 1,
                                        "category": category, "priority": priority}
        match = _PARTITIONS.search(line)
        if match:
            partitions.append({"line": number, "scope": scope, "graph": graph,
                               "counts": dict(zip(("NPU", "CPU", "GPU", "ISP"), map(int, match.groups()))),
                               "text": line})
        if (re.search(r"run[_\s-]*by[_\s-]*cl\b", line, re.I)
                or ("subgraph" in low and re.search(r"(?:cpucl|npucl).*subgraph.*\.irpb", low))):
            subgraphs.append({"line": number, "scope": scope, "graph": graph,
                              "text": line, "kind": "run_by_cl" if "irpb" not in low else "partition_dump"})
    diagnostics = sorted(diagnostic_seen.values(), key=lambda item: (item["priority"], item["line"]))[:32]
    return {"partition_events": partitions, "subgraph_events": subgraphs, "rollback_events": rollbacks,
            "main_partition_events": [e for e in partitions if e["scope"] == "main"],
            "constant_folding_partition_events": [e for e in partitions if e["scope"] == "constant_folding"],
            "support_candidate_lines": candidates, "runtime_placement": "UNVERIFIED",
            "diagnostics": diagnostics, "diagnostic_matching_lines": diagnostic_total,
            "diagnostic_omitted_distinct": max(0, len(diagnostic_seen) - len(diagnostics)),
            "note": "Compiler partitions describe compilation only. Candidate support checks do not prove placement or execution."}

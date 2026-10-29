"""Read retained direct-output producer paths without vendor execution or weights."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .draft_compile_inspect import _bound, _json, _one
from .draft_compile_precision import _identity, _observed_desc, _read, _weight_info_trailer
from .draft_direct_abi import _declared
from .draft_ir_inspect import _enum_fields
from .draft_precision_edges import _edge, _ops
from .existing_omc_inspect import _preview
from .omc_container import read_container
from .omc_cpu_inspect import _task
from .omc_wire import load_schema

KIND = "qwen35-draft-direct-output-provenance"
_VIEWS = {"Cast", "CastT", "Reshape", "Identity", "Transpose", "Squeeze", "Unsqueeze", "Flatten"}


def _capture(node, port=None):
    # Preserve descriptor memory/offset/alias attributes as raw metadata. Their
    # meaning is not inferred from key spellings or numeric coincidences.
    result = _preview(node, items=32, budget=[4096])
    if port is not None:
        descs = node.get("output_desc", [])
        if not isinstance(descs, list) or not 0 <= port < len(descs):
            raise ValueError("producer output port lacks a descriptor")
        result["selected_output_port"] = port
        result["selected_output_descriptor"] = _preview(descs[port], items=64, budget=[2048])
    return result


def _consumers(nodes, name, port):
    result, unresolved = [], 0
    for node in nodes.values():
        if node.get("type") in {"Data", "RefData"}:
            continue
        for index in range(len(node.get("input_desc", []))):
            try:
                edge = _edge(node, index)
            except ValueError:
                unresolved += 1
                continue
            if edge == (name, port):
                result.append({"name": node["name"], "type": node.get("type"), "input_port": index,
                               "input_descriptor": _preview(node["input_desc"][index], budget=[512])})
                if len(result) > 128:
                    raise ValueError("selected output consumer count exceeds 128")
    return {"edges": result, "unresolved_input_count": unresolved}


def inspect_output_document(main_graph, cpu_graphs, spec, enums):
    """Describe explicit edges only; this pure helper establishes no binary binding."""
    physical = _declared(spec)
    main, cpu = _ops(main_graph), {}
    for item in cpu_graphs:
        owner, graph = item["owner"], item["graph"]
        if owner in cpu or graph.get("name") != owner or item.get("cl_name") != "CPUCL":
            raise ValueError("CPU GraphDef ownership is missing or ambiguous")
        cpu[owner] = _ops(graph)
    output = _one(list(main.values()), lambda n: n.get("type") == "NetOutput", "main NetOutput")
    if len(output.get("input_desc", [])) != len(physical):
        raise ValueError("main NetOutput count differs from direct specification")
    rows = []
    for index, declared in enumerate(physical):
        name, port = _edge(output, index)
        if name not in main:
            raise ValueError("main NetOutput producer is absent")
        producer = main[name]
        row = {"output_index": index, **declared,
               "semantic_names": [s["name"] for s in spec if s["output_index"] == index],
               "netoutput_input_descriptor": _preview(output["input_desc"][index], budget=[2048]),
               "main_producer": _capture(producer, port), "main_consumers": _consumers(main, name, port),
               "main_cl_names": [item.get("value", {}).get("s") for item in producer.get("attr", [])
                                 if isinstance(item, dict) and item.get("key") == "cl_name"],
               "path": [], "source_occurrences": []}
        actual = _observed_desc(producer["output_desc"][port], enums)
        target = _observed_desc(output["input_desc"][index], enums)
        if any(actual[key] != target[key] for key in ("dtype", "shape")):
            raise ValueError("main producer and NetOutput descriptor differ")
        for owner, nodes in cpu.items():
            if declared["source"] in nodes:
                found = nodes[declared["source"]]
                row["source_occurrences"].append({"owner": owner, "node": _capture(found, 0),
                                                  "consumers": _consumers(nodes, found["name"], 0)})
        owner, current, visited = None, producer, set()
        for _ in range(64):
            state = owner, current["name"], port
            if state in visited:
                row["resolution"] = "UNRESOLVED_CYCLE"
                break
            visited.add(state)
            if owner is None and current.get("type") == "GraphOp":
                owner = current["name"]
                if owner not in cpu:
                    row["resolution"] = "OPAQUE_NON_CPU_TASK"
                    break
                nodes = cpu[owner]
                netout = _one(list(nodes.values()), lambda n: n.get("type") == "NetOutput", "CPU NetOutput")
                if len(netout.get("input_desc", [])) != len(current.get("output_desc", [])):
                    row["resolution"] = "UNRESOLVED_CPU_INTERFACE_COUNT"
                    break
                row["path"].append({"owner": owner, "kind": "CPU_NETOUTPUT", "port": port,
                                    "selected_input_descriptor": _preview(netout["input_desc"][port], budget=[2048]),
                                    "node": _capture(netout)})
                name, port = _edge(netout, port)
                if name not in nodes:
                    row["resolution"] = "UNRESOLVED_CPU_PRODUCER"
                    break
                current = nodes[name]
            nodes = cpu[owner] if owner is not None else main
            row["path"].append({"owner": owner, "kind": "EXPLICIT_PRODUCER", "node": _capture(current, port),
                                "consumers": _consumers(nodes, current["name"], port)})
            if current["name"] == declared["source"] and port == 0:
                row["resolution"] = "EXACT_CPU_NAME_ON_PATH" if owner is not None else "EXACT_MAIN_NAME_ON_PATH"
                break
            if owner is not None and current.get("type") == "Data":
                boundary = main[owner]
                if len(current.get("output_desc", [])) != len(boundary.get("input_desc", [])):
                    row["resolution"] = "UNRESOLVED_CPU_INTERFACE_COUNT"
                    break
                name, port = _edge(boundary, port)
                if name not in main:
                    row["resolution"] = "UNRESOLVED_MAIN_PRODUCER"
                    break
                current, owner = main[name], None
                continue
            if current.get("type") in _VIEWS and len(current.get("input_desc", [])) == 1:
                name, port = _edge(current, 0)
                if name in nodes:
                    current = nodes[name]
                    continue
            row["resolution"] = "DIFFERENT_EXPLICIT_PRODUCER"
            break
        else:
            row["resolution"] = "UNRESOLVED_PATH_LIMIT"
        rows.append(row)
    return {"outputs": rows, "resolution_counts": dict(Counter(row["resolution"] for row in rows)),
            "opaque_tasks": "NOT_DECODED", "view_path_scope": "Cast/view edges are observations, not value-equivalence proof"}


def inspect_draft_output_provenance(job_dir, schema_library=None):
    """Read hash-bound retained metadata and small external OMC, never large weights."""
    root, sources = Path(job_dir).absolute(), []
    report = {"schema_version": 1, "artifact_kind": KIND, "status": "UNAVAILABLE", "job": str(root),
              "sources": sources, "new_compilation": "NOT_RUN", "new_device_execution": "NOT_RUN",
              "scope": "Static output/CPU-interface metadata only; no arithmetic, snapshot correctness or runtime-placement pass"}
    try:
        owner = _json(root / "owner.json", sources)
        if owner.get("artifact_kind") != "qwen35-draft-direct-trace-job-owner":
            raise ValueError("expected retained direct-trace owner")
        stages = {}
        for stage, filename in (("fixture", "report.json"), ("air", "manifest.json"), ("compilation", "report.json")):
            receipt = _json(root / ("completed-" + stage + ".json"), sources)
            if receipt.get("path") != filename:
                raise ValueError("unexpected completed-stage receipt path")
            stages[stage] = _json(_bound(root / stage, receipt), sources, receipt)
        fixture, air, compiled = (stages[key] for key in ("fixture", "air", "compilation"))
        spec = owner["trace_spec"]
        if fixture.get("direct_spec") != spec or air.get("direct_spec") != spec:
            raise ValueError("owner, fixture and exported direct specification differ")
        if compiled.get("artifact_kind") != "qwen35-existing-air-compilation" or compiled.get("status") != "COMPILED":
            raise ValueError("expected successful retained AIR compilation")
        graph = _one(air.get("graphs", []), lambda row: row.get("name") == "draft", "exported draft graph")
        exported = _one(graph["files"], lambda row: row.get("path") == graph["air"], "exported AIR identity")
        snapshot = _one(compiled.get("input_snapshots", []), lambda row: row.get("role") == "air", "compiler AIR snapshot")
        if any(snapshot.get(key) != exported.get(key) for key in ("bytes", "sha256")):
            raise ValueError("compiler AIR snapshot differs from export manifest")
        binding = _json(root / "compiled-output-abi.json", sources)
        if binding.get("status") not in {"PASSED", "SOURCE_DTYPE_CHANGED"} or binding.get("declared_spec") != spec:
            raise ValueError("compiled direct-output ABI is unavailable or differs from owner")
        omc = _bound(root / "compilation", compiled["omc"])
        raw, identity = _read(omc)
        digest = hashlib.sha256(raw).hexdigest()
        for record in (compiled["omc"], binding["omc"]):
            if record.get("bytes") != len(raw) or record.get("sha256") != digest:
                raise ValueError("retained OMC identity mismatch")
        selected = Path(schema_library or binding.get("schema", {}).get("library") or
                        Path(compiled["ddk_root"]) / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so").absolute()
        schema = load_schema(selected)
        if schema.evidence["library_sha256"] != binding.get("schema", {}).get("library_sha256"):
            raise ValueError("selected schema differs from retained compiled-output ABI")
        container = read_container(omc)
        parts = container["partitions"]
        if [p["type"] for p in parts] not in ([0, 1, 2], [0, 1, 2, 8]) or parts[1]["bytes"] != 0:
            raise ValueError("inspection requires small external OMC with empty weight partition")
        cursor = 0
        for part in parts:
            if part["table_word"] != cursor:
                raise ValueError("unsupported OMC offset semantics")
            cursor += part["bytes"]
        payload = lambda p: memoryview(raw)[p["offset"]:p["offset"] + p["bytes"]]
        if len(parts) == 4:
            _weight_info_trailer(payload(parts[3]))
        graphs = schema.decode(payload(parts[0]), "ModelDef").get("graph", [])
        if len(graphs) != 1:
            raise ValueError("expected one compiled main graph")
        main, cpu, spans, tasks = graphs[0], [], [], []
        for node in _ops(main).values():
            if node.get("type") != "GraphOp":
                continue
            task = _task(node)
            start, end = task["offset"], task["offset"] + task["bytes"]
            if not task["bytes"] or end > parts[2]["bytes"] or any(start < b and a < end for a, b in spans):
                raise ValueError("GraphOp task ownership outside bounds or overlapping")
            spans.append((start, end))
            tasks.append({"owner": node["name"], **task})
            if len(tasks) > 128:
                raise ValueError("GraphOp count exceeds 128")
            if task["cl_name"] == "CPUCL":
                offset = parts[2]["offset"] + start
                cpu.append({"owner": node["name"], "cl_name": "CPUCL",
                            "graph": schema.decode(memoryview(raw)[offset:offset + task["bytes"]], "GraphDef")})
        details = inspect_output_document(main, cpu, binding["observed_spec"], _enum_fields(schema))
        if _identity(omc.stat()) != identity or hashlib.sha256(omc.read_bytes()).hexdigest() != digest:
            raise ValueError("OMC changed during inspection")
        if hashlib.sha256(selected.read_bytes()).hexdigest() != schema.evidence["library_sha256"]:
            raise ValueError("selected schema changed during inspection")
        for source in sources:
            if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("retained metadata changed during inspection")
        report.update(status="INSPECTED", omc={"path": str(omc), "bytes": len(raw), "sha256": digest},
                      schema=schema.evidence, tasks=tasks, source_binding="RECEIPTS_AND_RECORDED_AIR_SNAPSHOT_MATCHED", **details)
        if len(json.dumps(report, ensure_ascii=True).encode()) > 8 << 20:
            raise ValueError("output provenance report exceeds 8 MiB metadata bound")
    except (OSError, ValueError, KeyError, TypeError, IndexError, RecursionError) as error:
        report.pop("outputs", None)
        report.update(status="UNAVAILABLE", reason=f"{type(error).__name__}: {str(error)[:1600]}")
    return report

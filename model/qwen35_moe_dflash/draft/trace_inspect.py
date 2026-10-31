"""Read a retained trace precision failure without rebuilding or changing it.

This inspection follows the original artifact receipts, not the current Python
source hashes. It reports the existing verifier's findings and never promotes a
failed precision result or starts a compiler, packer, or device runner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


_JSON_LIMIT = 16 << 20
_OMC_LIMIT = 64 << 20
_KIND = "qwen35-draft-layer-trace-job"


def _identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class _Reader:
    def __init__(self):
        self.identities = {}

    def read(self, path, *, limit):
        path = Path(path).absolute()
        if path.resolve() != path or any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("trace inspection requires real non-symlink paths: " + str(path))
        before = _identity(path)
        if not stat.S_ISREG(path.stat().st_mode) or not 0 < before[2] <= limit:
            raise ValueError("trace inspection input is not a bounded regular file: " + str(path))
        raw = path.read_bytes()
        if len(raw) != before[2] or _identity(path) != before:
            raise ValueError("trace inspection input changed while reading: " + str(path))
        self.identities[path] = before
        return raw

    def json(self, path):
        return json.loads(self.read(path, limit=_JSON_LIMIT))

    def bound(self, root, row, *, limit=_JSON_LIMIT):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("invalid trace inspection file receipt")
        relative = PurePosixPath(row["path"])
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or relative.as_posix() != row["path"]):
            raise ValueError("trace inspection file receipt escapes its artifact root")
        raw = self.read(root / relative, limit=limit)
        if (type(row.get("bytes")) is not int or row["bytes"] != len(raw)
                or hashlib.sha256(raw).hexdigest() != row.get("sha256")):
            raise ValueError("trace inspection SHA256/size mismatch: " + row["path"])
        return raw

    def unchanged(self):
        if any(_identity(path) != before for path, before in self.identities.items()):
            raise ValueError("retained trace evidence changed during inspection")


def _same_identity(left, right):
    return all(left.get(key) == right.get(key) for key in ("bytes", "sha256"))


def summarize_boundary_document(model, cpu_graphs, failed_names=()):
    """Retain exact decoded boundary references; do not guess port mappings.

    ``cpu_graphs`` items contain the explicit GraphOp owner and its decoded
    GraphDef. Numeric dtype enums and raw attributes stay in the report so the
    selected-schema verifier can interpret them independently.
    """
    from qwen35_moe_dflash.toolchain.omc_cpu_inspect import _task

    def node_record(node):
        if not isinstance(node, dict) or not isinstance(node.get("name"), str):
            raise ValueError("boundary operation is missing its exact name")
        row = {key: node.get(key, []) for key in ("input", "input_desc", "output_desc", "attr")}
        row.update(name=node["name"], type=node.get("type"))
        if any(not isinstance(row[key], list) or len(row[key]) > 2048
               for key in ("input", "input_desc", "output_desc", "attr")):
            raise ValueError("boundary operation metadata exceeds its port/attribute budget")
        if len(json.dumps(row, ensure_ascii=False)) > 512 << 10:
            raise ValueError("boundary operation metadata exceeds 512 KiB; no truncated mapping is inferred")
        if node.get("type") == "GraphOp":
            row["task"] = _task(node)
        return row

    graphs = model.get("graph")
    if not isinstance(graphs, list) or len(graphs) != 1:
        raise ValueError("boundary inspection requires one explicit main graph")
    main = graphs[0]
    ops = main.get("op")
    if not isinstance(ops, list) or not 0 < len(ops) <= 8192:
        raise ValueError("boundary inspection main operation count is unsupported")
    names = [node.get("name") for node in ops]
    if len(set(names)) != len(names):
        raise ValueError("boundary inspection main node names are ambiguous")
    main_records = [node_record(node) for node in ops if node.get("type") in {"GraphOp", "Data", "NetOutput"}]
    wanted = set(failed_names)
    subgraphs, owners = [], set()
    main_owners = {node["name"]: node for node in ops if node.get("type") == "GraphOp"}
    for item in cpu_graphs:
        owner, graph = item["owner"], item["graph"]
        if (owner in owners or owner not in main_owners or graph.get("name") != owner
                or _task(main_owners[owner])["cl_name"] != "CPUCL"):
            raise ValueError("boundary inspection CPU ownership is ambiguous")
        owners.add(owner)
        children = graph.get("op")
        if not isinstance(children, list) or not 0 < len(children) <= 8192:
            raise ValueError("boundary inspection CPU operation count is unsupported")
        child_names = [node.get("name") for node in children]
        if len(set(child_names)) != len(child_names):
            raise ValueError("boundary inspection CPU node names are ambiguous")
        subgraphs.append({"owner": owner, "task": _task(main_owners[owner]),
                          "boundary_nodes": [node_record(node) for node in children if node.get("type") in {"Data", "NetOutput"}],
                          "failed_nodes": [node_record(node) for node in children if node.get("name") in wanted]})
    result = {"main_graph": main.get("name"), "main_boundary_nodes": main_records,
              "cpu_subgraphs": subgraphs,
              "scope": "Exact decoded boundary metadata only; raw names and attributes do not by themselves prove a cross-partition port mapping."}
    if len(json.dumps(result, ensure_ascii=False)) > _JSON_LIMIT:
        raise ValueError("boundary inspection report exceeds 16 MiB")
    return result


def _decode_boundaries(read, root, compiled, precision, schema_library):
    from qwen35_moe_dflash.toolchain.omc_container import read_container
    from qwen35_moe_dflash.toolchain.omc_cpu_inspect import _task
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema

    selected = precision.get("schema", {})
    path = schema_library or selected.get("library")
    if not path:
        raise ValueError("boundary inspection requires the recorded DDK schema library")
    schema_path = Path(path).absolute()
    raw_library = read.read(schema_path, limit=512 << 20)
    digest = hashlib.sha256(raw_library).hexdigest()
    if digest != selected.get("library_sha256"):
        raise ValueError("boundary inspection schema differs from the retained precision report")
    del raw_library
    schema = load_schema(schema_path)
    if schema.evidence.get("library_sha256") != digest:
        raise ValueError("boundary inspection schema changed while loading")
    omc_path = root / "compilation" / compiled["omc"]["path"]
    raw = read.bound(root / "compilation", compiled["omc"], limit=_OMC_LIMIT)
    container = read_container(omc_path)
    parts = container["partitions"]
    if [part["type"] for part in parts] not in ([0, 1, 2], [0, 1, 2, 8]) or parts[1]["bytes"] != 0:
        raise ValueError("boundary inspection requires an external OMC with explicit main/task partitions")
    def payload(part):
        return memoryview(raw)[part["offset"]:part["offset"] + part["bytes"]]
    model = schema.decode(payload(parts[0]), "ModelDef")
    graphs = model.get("graph", [])
    if len(graphs) != 1:
        raise ValueError("boundary inspection requires one main graph")
    cpu, spans = [], []
    for node in graphs[0].get("op", []):
        if node.get("type") != "GraphOp":
            continue
        task = _task(node)
        start, size = task["offset"], task["bytes"]
        if not size or start > parts[2]["bytes"] or size > parts[2]["bytes"] - start:
            raise ValueError("boundary inspection task is outside its OMC partition")
        if any(start < end and begin < start + size for begin, end in spans):
            raise ValueError("boundary inspection task owners overlap")
        spans.append((start, start + size))
        if task["cl_name"] == "CPUCL":
            absolute = parts[2]["offset"] + start
            graph = schema.decode(memoryview(raw)[absolute:absolute + size], "GraphDef")
            cpu.append({"owner": node["name"], "graph": graph})
    result = summarize_boundary_document(model, cpu,
        [node["name"] for node in precision.get("nodes", []) if node.get("status") != "MATCHED"])
    result["schema"] = schema.evidence
    return result


def inspect_retained_trace_precision(job_dir, *, include_boundaries=False, schema_library=None):
    """Return the existing precision findings bound to recipe and small OMC.

    No AI_RUN_DIR, DDK installation, current-code match, or device is required.
    Every read is bounded; weight sidecars and packed multi-GB OMCs are not read.
    """
    job = Path(job_dir).absolute()
    if not job.is_dir() or job.resolve() != job:
        raise ValueError("trace inspection requires a real retained job directory")
    root = job / "layer-trace"
    read = _Reader()
    source_raw = read.read(job / "result.json", limit=_JSON_LIMIT)
    if json.loads(source_raw).get("artifact_kind") != "qwen35-full-draft-job":
        raise ValueError("trace inspection requires a retained full-draft job")
    owner = read.json(root / "owner.json")
    if (owner.get("artifact_kind") != _KIND + "-owner" or owner.get("job") != str(job)
            or owner.get("source_job_sha256") != hashlib.sha256(source_raw).hexdigest()):
        raise ValueError("trace owner does not bind the retained production job")
    # The original source-code hashes are evidence, not a requirement to install
    # an older checkout merely to view a failed compiler result.
    exported = json.loads(read.bound(root / "air", read.json(root / "completed-air.json")))
    if (exported.get("artifact_kind") != "qwen35-cpro-draft-air" or exported.get("status") != "EXPORTED"
            or exported.get("diagnostic_only") is not True
            or exported.get("trace_spec") != owner.get("trace_spec")
            or len(exported.get("graphs", [])) != 1):
        raise ValueError("trace AIR manifest differs from its original owner")
    graph = exported["graphs"][0]
    if graph.get("name") != "draft" or graph.get("outputs") != ["draft_top1", "draft_trace_values"]:
        raise ValueError("trace AIR does not declare the diagnostic output contract")
    recipes = [row for row in graph.get("construction_files", []) if row.get("path") == "draft/recipe/graph.recipe"]
    airs = [row for row in graph.get("files", []) if row.get("path") == graph.get("air")]
    if len(recipes) != 1 or len(airs) != 1:
        raise ValueError("trace AIR manifest must bind one AIR and one precision recipe")
    read.bound(root / "air", recipes[0])
    read.bound(root / "air", airs[0])
    compiled = json.loads(read.bound(root / "compilation", read.json(root / "completed-compilation.json")))
    if (compiled.get("artifact_kind") != "qwen35-existing-air-compilation"
            or compiled.get("status") != "COMPILED" or compiled.get("model_compilation") != "PASSED"):
        raise ValueError("trace inspection needs completed compilation evidence")
    snapshots = [row for row in compiled.get("input_snapshots", []) if row.get("role") == "air"]
    if len(snapshots) != 1 or not _same_identity(snapshots[0], airs[0]):
        raise ValueError("trace compilation belongs to a different AIR")
    read.bound(root / "compilation", snapshots[0])
    omc = compiled.get("omc", {})
    read.bound(root / "compilation", omc, limit=_OMC_LIMIT)
    precision_path = root / "precision-check.json"
    precision_raw = read.read(precision_path, limit=_JSON_LIMIT)
    precision = json.loads(precision_raw)
    if (precision.get("artifact_kind") != "qwen35-draft-compiled-precision-verification"
            or precision.get("contract") != "cpro-fp32-cpu-v1"
            or not _same_identity(precision.get("omc", {}), omc)
            or not _same_identity(precision.get("recipe", {}), recipes[0])):
        raise ValueError("retained precision findings do not bind the compiled OMC and recipe")
    nodes = precision.get("nodes", [])
    required, matched = precision.get("required_nodes"), precision.get("matched_nodes")
    if (not isinstance(nodes, list) or len(nodes) > 1024 or type(required) is not int
            or required != len(nodes) or type(matched) is not int
            or matched != sum(row.get("status") == "MATCHED" for row in nodes)):
        raise ValueError("retained precision node counts are inconsistent")
    failed = [{key: row[key] for key in ("name", "source_type", "type", "status", "errors") if key in row}
              for row in nodes if row.get("status") != "MATCHED"]
    if any(not re.fullmatch("[0-9a-f]{64}", item.get("sha256", "")) for item in (omc, recipes[0])):
        raise ValueError("retained precision identity is malformed")
    result = {"schema_version": 1, "artifact_kind": "qwen35-draft-trace-precision-inspection", "status": "INSPECTED",
            "output": str(root), "precision_report": str(precision_path),
            "precision_report_sha256": hashlib.sha256(precision_raw).hexdigest(),
            "compiled_precision": {key: precision.get(key) for key in ("status", "required_nodes", "matched_nodes", "error_count", "reason")},
            "failed_nodes": failed, "failed_node_count": len(failed),
            "original_code_hashes": owner.get("code", {}), "omc": precision["omc"], "recipe": precision["recipe"],
            "scope": "Read-only retained findings; no new verification, compilation, packing, upload or device execution; failed precision is not promoted."}
    if include_boundaries:
        result["boundaries"] = _decode_boundaries(read, root, compiled, precision, schema_library)
    read.unchanged()
    return result

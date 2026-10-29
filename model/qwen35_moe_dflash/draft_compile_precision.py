"""Plan and verify CPU FP32 draft regions without running vendor tools.

The plan is derived from the exported recipe. Verification requires explicit
CPUCL GraphDef task ownership and schema-defined tensor dtypes; source Cast
names and opaque NPU task payloads are never treated as precision evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import stat

from .draft_ir_inspect import _enum_fields
from .ge_recipe_read import _Reader, _read as _read_recipe
from .omc_container import read_container
from .omc_cpu_inspect import _task
from .omc_wire import load_schema

CONTRACT = "cpro-fp32-cpu-v1"
_MAX_BYTES = 64 << 20
_MAX_NODES = 8192
_MAX_REQUIREMENTS = 1024
# These CPro lowering names occur in the retained compiler/runtime evidence.
_TYPES = {"SelectV2": {"SelectV2", "Select"}, "SoftmaxV2": {"SoftmaxV2", "Softmax"}}
_DTYPES = {"FLOAT": "float32", "DT_FLOAT": "float32", "FLOAT32": "float32", "FP32": "float32",
           "HALF": "float16", "DT_FLOAT16": "float16", "FLOAT16": "float16", "FP16": "float16",
           "INT32": "int32", "DT_INT32": "int32", "INT64": "int64", "DT_INT64": "int64",
           "BOOL": "bool", "DT_BOOL": "bool"}
_SCOPE = ("Static CPUCL ownership, descriptor and FP32 edge checks only; "
          "not a device numerical result or proof of kernel internal arithmetic")


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path):
    path = Path(path).absolute()
    before = path.stat()
    if path.is_symlink() or path.resolve() != path or not stat.S_ISREG(before.st_mode):
        raise ValueError("precision input must be a regular non-symlink file")
    if before.st_size > _MAX_BYTES:
        raise ValueError("precision input exceeds metadata byte limit")
    raw = path.read_bytes()
    if len(raw) != before.st_size or _identity(before) != _identity(path.stat()):
        raise ValueError("precision input changed while reading")
    return raw, _identity(before)


def _weight_info_trailer(raw):
    """Recognize the observed public-DDK type-8 JSON, without reading weights.

    This metadata is not precision evidence. Its sole purpose here is to admit
    the observed container extension while keeping unknown trailers rejected.
    """
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key in weight-info trailer")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError("nonfinite JSON constant in weight-info trailer")

    if not 0 < len(raw) <= 4 << 20:
        raise ValueError("weight-info trailer exceeds its metadata limit")
    data = json.loads(bytes(raw).decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    if not isinstance(data, dict) or set(data) != {"WeightInfoLists"}:
        raise ValueError("unrecognized type-8 weight-info trailer")
    lists = data["WeightInfoLists"]
    if not isinstance(lists, list) or not 0 < len(lists) <= 128:
        raise ValueError("invalid weight-info subgraph list")
    names, total = set(), 0
    for row in lists:
        if not isinstance(row, dict) or set(row) != {"WeightBlockInfo", "subGraphName", "variableSkip"}:
            raise ValueError("unrecognized weight-info subgraph entry")
        name = row["subGraphName"]
        if not isinstance(name, str) or not name or name in names or type(row["variableSkip"]) is not bool:
            raise ValueError("invalid weight-info subgraph identity")
        names.add(name)
        blocks = row["WeightBlockInfo"]
        if not isinstance(blocks, list):
            raise ValueError("invalid weight-info block list")
        total += len(blocks)
        if total > _MAX_NODES:
            raise ValueError("weight-info block count exceeds metadata limit")
        for block in blocks:
            if not isinstance(block, dict) or set(block) != {"hash", "offset", "size", "type", "weightIdx"}:
                raise ValueError("unrecognized weight-info block entry")
            if any(type(value) is not int or not 0 <= value < (1 << 64) for value in block.values()):
                raise ValueError("invalid weight-info block integer")
    return {"type": 8, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "format": "WeightInfoLists JSON", "status": "RECOGNIZED_NOT_USED_FOR_PRECISION"}


def _descriptor(port):
    return {"dtype": port["dtype"], "shape": list(port["shape"])}


def collect_precision_plan(recipe_path):
    """Return a location-independent policy plan; never read weight payloads.

    All non-constant, non-Cast operations with any FP32 data port are required.
    Casts remain boundary metadata, because CPro may replace/eliminate them.
    """
    raw, _ = _read(recipe_path)
    recipe = _read_recipe(_Reader(io.BytesIO(raw), len(raw)))
    if recipe["graph_name"] != "draft":
        raise ValueError("FP32 CPU policy applies only to the standalone draft")
    requirements, casts, constants = [], [], []
    records = recipe["records"]
    for node in records:
        if not any(p["dtype"] == "float32" for p in node["inputs"] + node["outputs"]):
            continue
        row = {"name": node["name"], "type": node["type"],
               "inputs": [{"port": p["port"], **_descriptor(p),
                           "producer": records[p["src"]]["name"], "producer_type": records[p["src"]]["type"],
                           "producer_port": p["src_port"],
                           "producer_output_index": next(i for i, out in enumerate(records[p["src"]]["outputs"])
                                                         if out["port"] == p["src_port"])} for p in node["inputs"]],
               "outputs": [{"port": p["port"], **_descriptor(p)} for p in node["outputs"]]}
        if node["type"] in {"Const", "FileConstant", "Data"}:
            constants.append(row)
        elif node["type"] in {"Cast", "CastT"}:
            casts.append(row)
        else:
            row["attrs"] = node["attrs"]
            requirements.append(row)
    if not requirements or len(requirements) > _MAX_REQUIREMENTS:
        raise ValueError("recipe has no FP32 arithmetic requirements or exceeds policy limit")
    return {"schema_version": 1, "artifact_kind": "qwen35-draft-compile-precision-plan", "contract": CONTRACT,
            "policy": "fp32-cpu", "recipe": {"path": "graph.recipe", "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}, "graph": "draft", "required_nodes": requirements,
            "boundary_casts": casts, "fp32_constants": constants,
            "counts": {"required_nodes": len(requirements), "boundary_casts": len(casts),
                       "fp32_constants": len(constants), "types": dict(Counter(n["type"] for n in requirements))},
            "scope": "Requested CPUCL placement for all source FP32 arithmetic; compiled placement must be verified"}


def _check_plan(plan):
    if (plan.get("artifact_kind") != "qwen35-draft-compile-precision-plan" or plan.get("contract") != CONTRACT
            or plan.get("policy") != "fp32-cpu" or plan.get("graph") != "draft"):
        raise ValueError("unsupported draft precision plan")
    required = plan.get("required_nodes")
    if not isinstance(required, list) or not 0 < len(required) <= _MAX_REQUIREMENTS:
        raise ValueError("invalid FP32 requirement list")
    names = set()
    for node in required:
        name = node.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("missing/duplicate precision requirement name")
        names.add(name)
        if node.get("type") in {"Cast", "CastT", "Const", "FileConstant", "Data"}:
            raise ValueError("precision requirements must contain arithmetic/shape operations")
        if not any(p.get("dtype") == "float32" for p in node["inputs"] + node["outputs"]):
            raise ValueError("requirement has no FP32 data port")
    return required


def _dtype(desc, enums):
    if "dtype" not in desc:
        return None
    values = enums.get("dtype", {}).get("values", {})
    names = {_DTYPES[name] for name, number in values.items() if number == desc["dtype"] and name in _DTYPES}
    return next(iter(names)) if len(names) == 1 else None


def _observed_desc(desc, enums):
    if not isinstance(desc, dict):
        raise ValueError("compiled tensor descriptor must be an object")
    return {"dtype": _dtype(desc, enums), "shape": desc.get("shape", {}).get("dim"),
            "raw_dtype": desc.get("dtype")}


def _edges(node, port):
    # Compiled input references describe connectivity. src_name/src_index can
    # retain pre-rewrite names, so never merge them with explicit input edges.
    count = len(node.get("input_desc", []))
    if "input" in node:
        references = node["input"]
        if not isinstance(references, list):
            return set()
        edges = []
        for reference in references:
            if not isinstance(reference, str) or ":" not in reference:
                return set()
            name, index = reference.rsplit(":", 1)
            if not name or not re.fullmatch(r"-?\d+", index):
                return set()
            index = int(index)
            if index >= 0:
                edges.append((name, index))
            elif index != -1:
                return set()
        return {edges[port]} if len(edges) == count and port < count else set()
    names, indices = node.get("src_name", []), node.get("src_index", [])
    if (not isinstance(names, list) or not isinstance(indices, list)
            or len(names) != count or len(indices) != count or port >= count
            or not isinstance(names[port], str) or not names[port]
            or type(indices[port]) is not int or indices[port] < 0):
        return set()
    return {(names[port], indices[port])}


def verify_document(plan, cpu_graphs, *, main_graph=None):
    """Verify decoded CPU graphs using their selected-schema enum metadata.

    Each input item must contain ``owner``, ``cl_name='CPUCL'``, ``graph`` (an
    explicit GraphDef with that name), and ``descriptor_enum_fields`` from the
    selected schema. This pure helper does not establish binary provenance;
    ``verify_compiled_precision`` performs that preceding ownership check.
    """
    required = _check_plan(plan)
    resolver = None
    if main_graph is not None:
        from .draft_precision_edges import PartitionEdgeResolver
        resolver = PartitionEdgeResolver(main_graph, cpu_graphs, _observed_desc)
    by_name, owners = {}, set()
    for item in cpu_graphs:
        owner, graph, enums = item["owner"], item["graph"], item["descriptor_enum_fields"]
        if item.get("cl_name") != "CPUCL" or owner in owners or graph.get("name") != owner:
            raise ValueError("CPU GraphDef ownership is missing, repeated, or inconsistent")
        owners.add(owner)
        ops = graph.get("op")
        if not isinstance(ops, list) or not ops or len(ops) > _MAX_NODES:
            raise ValueError("unsupported CPU GraphDef operation list")
        local = {}
        for node in ops:
            if not isinstance(node, dict) or not isinstance(node.get("name"), str) or not node["name"]:
                raise ValueError("CPU operation name is missing")
            local.setdefault(node["name"], []).append(node)
        for name, nodes in local.items():
            for node in nodes:
                by_name.setdefault(name, []).append((owner, node, enums, local))
    required_names = {n["name"] for n in required}
    results, total_errors = [], 0
    for source in required:
        found = by_name.get(source["name"], [])
        row = {"name": source["name"], "source_type": source["type"], "status": "MISSING", "errors": []}
        results.append(row)
        if len(found) != 1:
            row["status"] = "MISSING" if not found else "AMBIGUOUS"
            row["errors"].append({"kind": "CPU_NODE_MISSING" if not found else "CPU_NODE_AMBIGUOUS", "matches": len(found)})
            total_errors += 1
            continue
        owner, actual, enums, local = found[0]
        row.update(owner=owner, compiled_type=actual.get("type"), inputs=[], outputs=[], fp32_input_edges=[])
        if actual.get("type") not in _TYPES.get(source["type"], {source["type"]}):
            row["errors"].append({"kind": "OP_TYPE_DIFFERS", "actual": actual.get("type")})
        for direction, key in (("inputs", "input_desc"), ("outputs", "output_desc")):
            descs = actual.get(key, [])
            if not isinstance(descs, list):
                raise ValueError("compiled port descriptors must be lists")
            if len(descs) != len(source[direction]):
                row["errors"].append({"kind": "PORT_COUNT_DIFFERS", "direction": direction,
                                      "expected": len(source[direction]), "actual": len(descs)})
            for index, (expected, desc) in enumerate(zip(source[direction], descs)):
                observed = _observed_desc(desc, enums)
                row[direction].append(observed)
                if any(observed[k] != expected[k] for k in ("dtype", "shape")):
                    row["errors"].append({"kind": "DESCRIPTOR_DIFFERS", "direction": direction, "port": index,
                                          "expected": _descriptor(expected), "actual": observed})
        for port, expected in enumerate(source["inputs"]):
            if expected["dtype"] != "float32":
                continue
            edges = _edges(actual, port)
            edge = {"input_port": port, "source_producer": expected["producer"], "status": "UNRESOLVED"}
            row["fp32_input_edges"].append(edge)
            if len(edges) != 1:
                row["errors"].append({"kind": "FP32_INPUT_EDGE_UNRESOLVED", "port": port})
                continue
            name, output = next(iter(edges))
            matches = local.get(name, [])
            edge.update(producer=name, output_port=output)
            if len(matches) != 1 or output >= len(matches[0].get("output_desc", [])):
                row["errors"].append({"kind": "FP32_PRODUCER_UNRESOLVED", "port": port})
                continue
            producer = _observed_desc(matches[0]["output_desc"][output], enums)
            edge["producer_descriptor"] = producer
            if producer["dtype"] != "float32" or producer["shape"] != expected["shape"]:
                row["errors"].append({"kind": "FP32_PRODUCER_DESCRIPTOR_DIFFERS", "port": port, "actual": producer})
                continue
            if expected["producer"] in required_names and (name != expected["producer"] or output != expected["producer_output_index"]):
                resolved = (resolver.resolve(owner, name, output, expected) if resolver is not None else None)
                if resolved is None or resolved.get("status") != "MATCHED":
                    error = {"kind": "FP32_REQUIRED_EDGE_CHANGED", "port": port,
                             "expected_producer": expected["producer"], "actual_producer": name}
                    if resolved is not None:
                        error["partition_resolution"] = resolved
                    row["errors"].append(error)
                    continue
                edge.update(partition_path=resolved["path"], resolved_producer=resolved["producer"],
                            resolved_owner=resolved["owner"], resolved_output_port=resolved["output_port"])
            edge["status"] = "MATCHED"
        row["status"] = "MATCHED" if not row["errors"] else "MISMATCH"
        total_errors += len(row["errors"])
    matched = sum(r["status"] == "MATCHED" for r in results)
    return {"schema_version": 1, "artifact_kind": "qwen35-draft-compiled-precision-verification", "contract": CONTRACT,
            "status": "PASSED" if matched == len(required) else "FAILED", "recipe": plan["recipe"],
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "required_nodes": len(required), "matched_nodes": matched, "error_count": total_errors,
            "cpu_graphs": len(owners), "nodes": results, "scope": _SCOPE,
            "cross_partition_edges": sum("partition_path" in edge for row in results for edge in row.get("fp32_input_edges", []))}


def verify_compiled_precision(omc, schema_library, plan):
    """Verify a bounded external compiled OMC before bundling or deployment.

    The caller binds returned OMC/plan identities to its compilation receipt.
    This function accepts the observed [main, empty weights, tasks] layout,
    optionally followed by the public DDK's type-8 WeightInfoLists JSON.
    """
    _check_plan(plan)
    report = {"schema_version": 1, "artifact_kind": "qwen35-draft-compiled-precision-verification", "contract": CONTRACT,
              "status": "NOT_VERIFIED", "recipe": plan["recipe"], "nodes": [], "scope": _SCOPE}
    try:
        raw, identity = _read(omc)
        omc = Path(omc).absolute()
        container = read_container(omc)
        if _identity(omc.stat()) != identity:
            raise ValueError("OMC changed during container inspection")
        parts = container["partitions"]
        if [p["type"] for p in parts] not in ([0, 1, 2], [0, 1, 2, 8]) or parts[1]["bytes"] != 0:
            raise ValueError("precision verification requires external OMC [0,1,2] with optional type-8 weight info")
        cursor = 0
        for part in parts:
            if part["table_word"] != cursor:
                raise ValueError("OMC table words are not cumulative payload offsets")
            cursor += part["bytes"]
        schema = load_schema(schema_library)
        enums = _enum_fields(schema)
        def payload(part):
            return memoryview(raw)[part["offset"]:part["offset"] + part["bytes"]]
        auxiliary = [_weight_info_trailer(payload(parts[3]))] if len(parts) == 4 else []
        model = schema.decode(payload(parts[0]), "ModelDef")
        graphs = model.get("graph")
        if not isinstance(graphs, list) or len(graphs) != 1:
            raise ValueError("compiled OMC requires one main graph")
        ops = graphs[0].get("op", [])
        if not ops or len(ops) > _MAX_NODES:
            raise ValueError("unsupported main operation count")
        names = [n.get("name") for n in ops]
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("main operation names are missing or repeated")
        owners, spans = [], []
        for node in ops:
            if node.get("type") != "GraphOp":
                continue
            task = _task(node)
            start, size = task["offset"], task["bytes"]
            end = start + size
            if not size or start > parts[2]["bytes"] or size > parts[2]["bytes"] - start:
                raise ValueError("GraphOp task is outside compiled task partition")
            if any(start < right and left < end for left, right in spans):
                raise ValueError("GraphOp task owners overlap")
            spans.append((start, end))
            owners.append((node["name"], task))
        if not 0 < len(owners) <= 128:
            raise ValueError("unsupported GraphOp owner count")
        cpu_graphs = []
        for name, task in owners:
            if task["cl_name"] != "CPUCL":
                continue
            start = parts[2]["offset"] + task["offset"]
            graph = schema.decode(memoryview(raw)[start:start + task["bytes"]], "GraphDef")
            cpu_graphs.append({"owner": name, "cl_name": "CPUCL", "graph": graph, "descriptor_enum_fields": enums})
        verified = verify_document(plan, cpu_graphs, main_graph=graphs[0])
        # Discard every decoded finding on a late identity failure.
        if _identity(omc.stat()) != identity or hashlib.sha256(omc.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
            raise ValueError("compiled OMC changed during precision verification")
        if hashlib.sha256(Path(schema_library).read_bytes()).hexdigest() != schema.evidence["library_sha256"]:
            raise ValueError("selected schema changed during precision verification")
        verified.update(omc={"path": omc.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                        schema=schema.evidence, graphop_counts=dict(Counter(t["cl_name"] for _, t in owners)),
                        opaque_tasks="NOT_DECODED", auxiliary_partitions=auxiliary)
        return verified
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        report["reason"] = f"{type(error).__name__}: {str(error)[:1024]}"
        return report

"""Read-only structural state provenance for the original target pair.

Named serialized operator ports and explicit tensor identities are evidence;
matching shapes, output_N spellings and equal numeric positions are not.
The audit never feeds a state back, imports vendor packages or opens weights.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .artifacts import output_path, save_json
from .air_schema import MAX_PB2_BYTES
from .draft_ir_inspect import _enum_fields
from .existing_air_profile import get_existing_air_profile
from .ge_fused.bindings import _source_record
from .omc_container import read_container
from .omc_wire import _MAX_LIBRARY, load_schema
from .original_target_contract import _attributes
from .target_compile_failure import _read_small, _relative, _safe
from .target_export_reuse import _reject_constant, _unique_object
from .target_prefill_smoke_plan import _nd_view_key, _runtime_keys
from .target_source_schema import load_bound_source_schema, tensor_enum_fields


_MAX_GRAPH_BYTES = 64 << 20
# The successful original decode compilation report is 10,306,827 bytes. Use
# the same bounded 32MiB JSON budget as its original smoke-plan validator.
_MAX_METADATA = 32 << 20
_STATES = {f"RefData_{i}" for i in range(80)}
_INPUT_NAMES = {item.split(":", 1)[0] for item in get_existing_air_profile("cpro-40block-prefill-64")["input_shape"].split(";")}
_PAIRS = {
    "CacheUpdate": ("x", "x"),
    "LinearAttentionConvstateUpdate": ("x2", "y1"),
    "ChunkGatedDeltaRule": ("initial_state", "last_recurrent_state"),
}
_DTYPES = {"DT_FLOAT16": "FLOAT16", "DT_FLOAT": "FLOAT32", "DT_FLOAT32": "FLOAT32",
           "DT_INT8": "INT8", "DT_INT16": "INT16", "DT_INT32": "INT32", "DT_INT64": "INT64"}


def _audit_read(path, maximum, role, reader, read_attempts=None, *, recorded_role=None):
    """Keep bounded read failures attributable without opening larger payloads."""
    attempt = {"role": role, "path": str(path), "observed_bytes": None,
        "maximum_bytes": maximum, "status": "PENDING"}
    if recorded_role is not None:
        attempt["recorded_role"] = recorded_role
    if read_attempts is not None:
        read_attempts.append(attempt)
    try:
        checked, stamp = _safe(path)
        attempt.update(path=str(checked), observed_bytes=stamp[2])
        if not 0 < stamp[2] <= maximum:
            raise ValueError("evidence must be nonempty and within its role-specific byte budget")
        value = reader(checked)
        attempt["status"] = "READ"
        return value
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        attempt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise ValueError(f"{role} read failed: path={attempt['path']}; observed_bytes={attempt['observed_bytes']}; "
            f"maximum_bytes={maximum}; cause={type(error).__name__}: {error}") from error


def _metadata(path, bindings, *, read_attempts=None, role="metadata"):
    def read(checked):
        raw, identity = _read_small(checked, _MAX_METADATA)
        def finite_float(text):
            value = float(text)
            if not math.isfinite(value):
                raise ValueError("nonfinite metadata number")
            return value
        value = json.loads(raw, object_pairs_hook=_unique_object,
            parse_constant=_reject_constant, parse_float=finite_float)
        if not isinstance(value, dict):
            raise ValueError("state lineage metadata must be an object")
        bindings.append(identity)
        return value
    return _audit_read(path, _MAX_METADATA, role, read, read_attempts)


def _bound_file(root, row, bindings, *, maximum=_MAX_GRAPH_BYTES, stat_key="stat",
                read_attempts=None, role="graph-payload"):
    if not isinstance(row, dict):
        raise ValueError("recorded metadata payload must be an object")
    path = _relative(root, row.get("path"))
    raw, identity = _audit_read(path, maximum, role, lambda checked: _read_small(checked, maximum),
        read_attempts, recorded_role=row.get("role"))
    if any(identity[key] != row.get(key) for key in ("bytes", "sha256")):
        raise ValueError("recorded payload hash/size changed: " + str(path))
    if stat_key in row and row[stat_key] != identity["stat"]:
        raise ValueError("recorded payload identity changed: " + str(path))
    bindings.append(identity)
    return path, raw, identity


def _graph(document):
    if not isinstance(document, dict):
        raise ValueError("decoded graph must be an object")
    graphs = document.get("graph")
    if graphs is not None:
        if not isinstance(graphs, list) or len(graphs) != 1:
            raise ValueError("one unambiguous root graph is required")
        document = graphs[0]
    if not isinstance(document, dict) or not isinstance(document.get("op"), list):
        raise ValueError("decoded root graph has no operation list")
    if not 1 <= len(document["op"]) <= 32768:
        raise ValueError("root graph exceeds operation budget")
    by_name = {}
    for node in document["op"]:
        name = node.get("name") if isinstance(node, dict) else None
        if (not isinstance(name, str) or not 0 < len(name) <= 1024 or name in by_name
                or not isinstance(node.get("type"), str)):
            raise ValueError("root graph has missing, duplicate or excessive operation names")
        by_name[name] = node
    net = [node for node in by_name.values() if node["type"] == "NetOutput"]
    if len(net) != 1:
        raise ValueError("root graph must have one explicit NetOutput")
    return document, by_name, net[0]


def _edge(text, by_name):
    if not isinstance(text, str) or len(text) > 1152 or ":" not in text:
        raise ValueError("edge must name an explicit same-graph producer port")
    name, port = text.rsplit(":", 1)
    if not re.fullmatch(r"0|[1-9][0-9]{0,3}", port):
        raise ValueError("unsupported data edge port")
    node, port = by_name.get(name), int(port)
    if node is None or port >= len(node.get("output_desc", [])):
        raise ValueError("edge producer or output port is absent")
    return node, port


def _inputs(node):
    values = node.get("input", [])
    if not isinstance(values, list) or len(values) > 1024:
        raise ValueError("unsupported operation input list")
    alternate = []
    if "src_name" in node or "src_index" in node:
        names, ports = node.get("src_name", []), node.get("src_index", [])
        if (not isinstance(names, list) or not isinstance(ports, list) or len(names) != len(ports)
                or len(names) > 1024 or any(not isinstance(name, str) or not name or len(name) > 1024 for name in names)
                or any(type(port) is not int or not -1 <= port <= 1023 for port in ports)):
            raise ValueError("unsupported alternate src_name/src_index input list")
        alternate = [name + ":" + str(port) for name, port in zip(names, ports)]
    if values and alternate and values != alternate:
        raise ValueError("conflicting input and src_name/src_index edge representations")
    values = values or alternate
    result = []
    for value in values:
        if isinstance(value, str) and value.endswith(":-1"):
            continue
        if not isinstance(value, str) or len(value) > 1152:
            raise ValueError("unsupported operation input reference")
        result.append(value)
    return result


def _port_map(node, direction):
    attrs = _attributes(node)
    keys, values = attrs.get(f"_{direction}_name_key"), attrs.get(f"_{direction}_name_value")
    if not isinstance(keys, dict) or not isinstance(values, dict):
        raise ValueError("missing explicit serialized " + direction + " port-name map")
    keys, values = keys.get("list", {}).get("s"), values.get("list", {}).get("i")
    limit = len(_inputs(node)) if direction == "input" else len(node.get("output_desc", []))
    if (not isinstance(keys, list) or not isinstance(values, list) or len(keys) != len(values)
            or not 1 <= len(keys) <= 1024 or any(not isinstance(k, str) or not k or len(k) > 128 for k in keys)
            or len(set(keys)) != len(keys) or len(set(values)) != len(values)
            or any(type(v) is not int or not 0 <= v < limit for v in values)):
        raise ValueError("ambiguous or unsupported serialized " + direction + " port-name map")
    return dict(zip(keys, values))


def _dtype(raw, enums):
    if not isinstance(raw, dict) or raw.get("layout") != "ND":
        raise ValueError("an explicit ND descriptor is required")
    dtype = raw.get("dtype")
    names = {name for name, number in enums.get("dtype", {}).get("values", {}).items()
             if type(dtype) is int and number == dtype}
    logical = {_DTYPES[name] for name in names if name in _DTYPES}
    if not names or any(name not in _DTYPES for name in names) or len(logical) != 1:
        raise ValueError("an explicit selected-schema dtype is required")
    return next(iter(logical))


def _view(raw, enums):
    dtype = _dtype(raw, enums)
    dims = raw.get("shape", {}).get("dim")
    if (not isinstance(dims, list) or not 1 <= len(dims) <= 16
            or any(type(dim) is not int or dim <= 0 for dim in dims)):
        raise ValueError("fixed shape and selected-schema dtype are required")
    return _nd_view_key((dtype, tuple(dims)))


def _identity(raw):
    value = _attributes(raw).get("_fx_tensor_name", {}).get("s")
    return value if isinstance(value, str) and 0 < len(value) <= 1024 else None


def _boundary(document, enums, *, allow_unmaterialized_producer=False):
    graph, by_name, net = _graph(document)
    edges, descs = _inputs(net), net.get("input_desc")
    if not isinstance(descs, list) or len(edges) != len(descs) or len(edges) != 82:
        raise ValueError("complete original feature target requires82 NetOutput descriptors and data edges")
    refs = {name: node for name, node in by_name.items() if name in _STATES}
    if set(refs) != _STATES or any(node["type"] not in {"Data", "RefData"} for node in refs.values()):
        raise ValueError("root must expose all80 exact RefData names as Data/RefData inputs")
    if {name for name, node in by_name.items() if node["type"] in {"Data", "RefData"}} != _INPUT_NAMES:
        raise ValueError("root must expose the exact89 original target input names")
    outputs = []
    for index, (edge, raw) in enumerate(zip(edges, descs)):
        node, port = _edge(edge, by_name)
        producer = node["output_desc"][port]
        view = _view(raw, enums)
        if _dtype(producer, enums) != view[0]:
            raise ValueError("NetOutput dtype differs from its actual producer")
        producer_shape = producer.get("shape", {}).get("dim")
        if allow_unmaterialized_producer and producer_shape in (None, []):
            producer_shape_status = "UNMATERIALIZED_OR_SCALAR_AMBIGUOUS"
        elif _view(producer, enums) != view:
            raise ValueError("NetOutput shape differs from its actual fixed producer")
        else:
            producer_shape_status = "FIXED_ND_VIEW_MATCHED"
        identities = {value for value in (_identity(raw), _identity(producer)) if value is not None}
        outputs.append({"index": index, "edge": edge, "descriptor": raw,
            "producer_descriptor": producer, "view": view,
            "producer_shape_status": producer_shape_status,
            "fx_tensor_identity": next(iter(identities)) if len(identities) == 1 else None,
            "fx_identity_status": "EXPLICIT_UNIQUE" if len(identities) == 1 else
                "CONFLICTING" if identities else "NOT_DECLARED"})
    return graph, by_name, refs, outputs


def _carrier(edge, by_name, *, allow_update):
    seen, path = set(), []
    for _ in range(32):
        node, port = _edge(edge, by_name)
        key = (node["name"], port)
        if key in seen:
            raise ValueError("cycle in state lineage")
        seen.add(key)
        path.append({"node": node["name"], "type": node["type"], "output_port": port})
        if node["type"] in {"Identity", "Cast"}:
            inputs = _inputs(node)
            if port != 0 or len(inputs) != 1:
                raise ValueError("state carrier Cast/Identity must have one explicit data edge")
            edge = inputs[0]
            continue
        if node["type"] in {"RefData", "Data"} and node["name"] in _STATES and port == 0:
            return node["name"], path, None
        if allow_update and node["type"] in _PAIRS:
            state_in, state_out = _PAIRS[node["type"]]
            inputs, outputs = _port_map(node, "input"), _port_map(node, "output")
            if state_in not in inputs or outputs.get(state_out) != port:
                raise ValueError("observed updater port is not its explicitly named state result")
            input_edge = _inputs(node)[inputs[state_in]]
            ref, carrier, nested = _carrier(input_edge, by_name, allow_update=False)
            if nested is not None:
                raise ValueError("nested state update is unsupported")
            return ref, path + carrier, {"node": node["name"], "type": node["type"],
                "input_port_name": state_in, "input_port_index": inputs[state_in],
                "input_edge": input_edge, "output_port_name": state_out, "output_port_index": port,
                "serialized_input_port_map": inputs, "serialized_output_port_map": outputs}
        raise ValueError("unsupported state lineage node/port: " + node["type"] + ":" + str(port))
    raise ValueError("state lineage exceeds32 carrier edges")


def analyze_state_lineage(document, enums):
    """Describe same-graph state owners; no output ordinal implies a RefData ID."""
    graph, by_name, refs, outputs = _boundary(document, enums, allow_unmaterialized_producer=True)
    keys = set()
    for node in refs.values():
        descs = node.get("output_desc", [])
        if len(descs) != 1:
            raise ValueError("each root state must have exactly one descriptor")
        keys.add(_view(descs[0], enums))
    rows = []
    for output in outputs:
        if output["view"] not in keys:
            continue
        row = {"source_output_index": output["index"], "source_output_edge": output["edge"],
               "producer_shape_status": output["producer_shape_status"],
               "fx_tensor_identity": output["fx_tensor_identity"],
               "fx_identity_status": output["fx_identity_status"], "status": "UNVERIFIED"}
        try:
            ref, path, update = _carrier(output["edge"], by_name, allow_update=True)
            if update is None:
                raise ValueError("direct state identity has no observed updater")
            row.update(status="NAMED_PORT_STATE_LINEAGE_OBSERVED", source_refdata=ref,
                       source_input_fx_tensor_identity=_identity(refs[ref]["output_desc"][0]),
                       carrier_path=path, updater=update)
        except (KeyError, TypeError, ValueError) as error:
            row["reason"] = str(error)
        rows.append(row)
    owners = [row["source_refdata"] for row in rows if "source_refdata" in row]
    materialized = sum(row["producer_shape_status"] == "FIXED_ND_VIEW_MATCHED" for row in rows)
    complete = (len(rows) == 80 and len(owners) == 80 and set(owners) == _STATES and len(set(owners)) == 80
                and materialized == 80)
    diagnostics = []
    for node in by_name.values():
        if node["type"] not in _PAIRS:
            continue
        entry = {"node": node["name"], "type": node["type"], "output_count": len(node.get("output_desc", []))}
        try:
            entry["input_edges"] = _inputs(node)
        except ValueError as error:
            entry["input_edges_error"] = str(error)
        for direction in ("input", "output"):
            try:
                entry[direction + "_port_map"] = _port_map(node, direction)
            except ValueError as error:
                entry[direction + "_port_map_error"] = str(error)
        diagnostics.append(entry)
        if len(diagnostics) > 128:
            raise ValueError("updater diagnostic count exceeds128")
    return {"status": "ALL80_NAMED_PORT_STATE_LINEAGES_OBSERVED" if complete else "STATE_LINEAGE_PARTIAL",
        "graph_name": graph.get("name"), "candidate_state_output_count": len(rows),
        "observed_state_owner_count": len(owners), "unique_state_owner_count": len(set(owners)),
        "fixed_state_output_producer_shape_count": materialized,
        "unmaterialized_output_producer_indices": [row["index"] for row in outputs
            if row["producer_shape_status"] == "UNMATERIALIZED_OR_SCALAR_AMBIGUOUS"],
        "producer_shape_inference": "NOT_PERFORMED_EMPTY_SHAPES_NEVER_ASSUMED_SCALAR_OR_FILLED_FROM_NETOUTPUT",
        "missing_state_owners": sorted(_STATES - set(owners)),
        "duplicate_state_owners": sorted(key for key, value in Counter(owners).items() if value > 1),
        "states": rows, "updater_diagnostics": diagnostics,
        "candidate_policy": "Typed shapes select candidates only; ownership uses actual edges and named ports",
        "state_semantics": "PINNED_DOCUMENT_PORT_CONTRACT_AND_SERIALIZED_GRAPH_ONLY_NOT_KERNEL_BEHAVIOR"}


def _join_compiled(source, document, enums):
    _, _, refs, outputs = _boundary(document, enums)
    by_identity = {}
    for row in outputs:
        if row["fx_tensor_identity"] is not None:
            by_identity.setdefault(row["fx_tensor_identity"], []).append(row)
    source_counts = Counter(row.get("fx_tensor_identity") for row in source["states"])
    rows = []
    for state in source["states"]:
        identity = state.get("fx_tensor_identity")
        matches = by_identity.get(identity, []) if identity is not None else []
        row = {"source_output_index": state["source_output_index"], "source_refdata": state.get("source_refdata"),
               "fx_tensor_identity": identity, "status": "UNVERIFIED"}
        if state["status"] != "NAMED_PORT_STATE_LINEAGE_OBSERVED":
            row["reason"] = "SOURCE_STATE_OWNER_UNVERIFIED"
        elif identity is None or source_counts[identity] != 1 or len(matches) != 1:
            row["reason"] = "MISSING_OR_NONUNIQUE_EXPLICIT_FX_TENSOR_IDENTITY"
        else:
            compiled = matches[0]
            row.update(status="EXPLICIT_FX_TENSOR_IDENTITY_MATCHED", compiled_output_index=compiled["index"],
                compiled_output_edge=compiled["edge"], compiled_descriptor=compiled["descriptor"],
                compiled_producer_descriptor=compiled["producer_descriptor"],
                compiled_input_name=state["source_refdata"],
                compiled_input_descriptor=refs[state["source_refdata"]]["output_desc"][0])
        rows.append(row)
    count = sum(row["status"] == "EXPLICIT_FX_TENSOR_IDENTITY_MATCHED" for row in rows)
    return {"status": "ALL80_EXPLICIT_FX_TENSOR_IDENTITIES_MATCHED" if count == 80 else "COMPILED_IDENTITY_PARTIAL",
        "matched_count": count, "states": rows,
        "scope": "Explicit declared tensor identity join only; no compiler arithmetic equivalence or task-byte interpretation"}


def _runtime_bindings(compiled, probe, enums):
    runtime_outputs = _runtime_keys(probe, "outputs")
    runtime_inputs = _runtime_keys(probe, "inputs")
    names = {}
    for direction in ("inputs", "outputs"):
        names[direction] = {}
        for row in probe[direction]:
            name = row.get("tensor_name")
            if isinstance(name, str) and name and row.get("tensor_name_status") == "OBSERVED":
                names[direction].setdefault(name, []).append(row)
    rows = []
    for state in compiled["states"]:
        row = {key: state.get(key) for key in ("source_output_index", "source_refdata", "compiled_output_index")}
        row["status"] = "UNVERIFIED"
        if state["status"] != "EXPLICIT_FX_TENSOR_IDENTITY_MATCHED":
            row["reason"] = "COMPILED_TENSOR_IDENTITY_UNVERIFIED"
            rows.append(row); continue
        declared = {desc.get("name") for desc in (state["compiled_descriptor"], state["compiled_producer_descriptor"])
                    if isinstance(desc.get("name"), str) and desc["name"]}
        matches = names["outputs"].get(next(iter(declared)), []) if len(declared) == 1 else []
        inputs = names["inputs"].get(state["compiled_input_name"], [])
        if len(matches) != 1 or len(inputs) != 1:
            row["reason"] = "MISSING_OR_NONUNIQUE_LITERAL_COMPILED_AND_RUNTIME_TENSOR_NAMES"
        elif (_nd_view_key(runtime_outputs[matches[0]["index"]]) != _view(state["compiled_descriptor"], enums)
                or _nd_view_key(runtime_inputs[inputs[0]["index"]]) != _view(state["compiled_input_descriptor"], enums)
                or _view(state["compiled_input_descriptor"], enums) != _view(state["compiled_descriptor"], enums)):
            row["reason"] = "RUNTIME_STATE_DESCRIPTOR_DIFFERS"
        else:
            row.update(status="EXPLICIT_TENSOR_NAME_AND_DESCRIPTOR_MATCHED",
                runtime_output_index=matches[0]["index"], runtime_output_name=matches[0]["tensor_name"],
                runtime_input_index=inputs[0]["index"], runtime_input_name=inputs[0]["tensor_name"])
        rows.append(row)
    count = sum(row["status"] == "EXPLICIT_TENSOR_NAME_AND_DESCRIPTOR_MATCHED" for row in rows)
    return {"status": "ALL80_EXPLICIT_RUNTIME_STATE_BINDINGS_OBSERVED" if count == 80 else "RUNTIME_STATE_BINDING_PARTIAL",
            "matched_count": count, "states": rows,
            "scope": "Literal compiled descriptor name equality and exact typed ND view; no output_N or ordinal fallback"}


def _decode_air(schema, raw):
    candidates, errors = [], []
    for kind in ("ModelDef", "GraphDef"):
        try:
            document = schema.decode(raw, kind)
            _, names, _ = _graph(document)
            if _STATES <= names.keys():
                candidates.append((kind, document))
            else:
                errors.append(kind + ": decoded root lacks80 exact state input names")
        except (KeyError, TypeError, ValueError, RecursionError) as error:
            errors.append(kind + ": " + str(error)[:384])
    if len(candidates) != 1:
        raise ValueError("AIR has no unique selected-schema root with80 exact state input names; " + "; ".join(errors))
    return candidates[0]


def load_target_graph(job, profile_name, bindings, *, source_capture=None, observations=None,
                      source_torchair_pb2=None):
    """Return bound job metadata, decoded AIR/OMC roots and selected enums.

    The returned graph objects stay in memory for bounded caller analysis;
    model tensor blobs have already been replaced with hashes by Schema.
    """
    job, _ = _safe(job, directory=True)
    compilation = job / "compilation"
    read_attempts = []
    if observations is not None:
        observations["evidence_read_attempts"] = read_attempts
    outer = _metadata(job / "result.json", bindings, read_attempts=read_attempts, role="job-result")
    outer_id = bindings[-1]
    report = _metadata(compilation / "report.json", bindings, read_attempts=read_attempts, role="compilation-report")
    report_id = bindings[-1]
    profile = _metadata(job / "compile-profile.json", bindings, read_attempts=read_attempts, role="compile-profile")
    inventory = _metadata(compilation / "input-inventory.json", bindings, read_attempts=read_attempts, role="input-inventory")
    if (outer.get("artifact_kind") != "qwen35-original-target-job" or outer.get("status") != "COMPILED"
            or outer.get("compilation_report_sha256") != report_id["sha256"] or outer.get("output") != str(job)
            or report.get("artifact_kind") != "qwen35-existing-air-compilation" or report.get("status") != "COMPILED"
            or report.get("model_compilation") != "PASSED" or report.get("returncode") != 0
            or report.get("platform") != "kirinx90" or profile != get_existing_air_profile(profile_name)
            or report.get("source_preservation", {}).get("status") != "UNCHANGED_BY_STAT"
            or any(report.get(key) for key in ("audit_error", "artifact_inventory_error", "decode_input_audit_error", "retained_input_audit_error"))
            or inventory.get("files") != report.get("input_snapshots")):
        raise ValueError("state audit requires the exact completed static target job and unchanged inventory")
    airs = [row for row in report["input_snapshots"] if isinstance(row, dict) and row.get("role") == "air"]
    if len(airs) != 1:
        raise ValueError("completed job must contain one bound small AIR snapshot")
    air_path, air_raw, air_id = _bound_file(compilation, airs[0], bindings, stat_key="snapshot_stat",
        read_attempts=read_attempts, role="source-air")
    omc_row = report.get("omc")
    if omc_row not in report.get("generated_artifacts", []):
        raise ValueError("primary OMC is not in the successful generated inventory")
    omc_path, omc_raw, omc_id = _bound_file(compilation, omc_row, bindings,
        read_attempts=read_attempts, role="primary-omc")
    library = Path(report["ddk_root"]) / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
    library, library_stat = _safe(library)
    schema = _audit_read(library, _MAX_LIBRARY, "selected-schema-library", load_schema, read_attempts)
    bindings.append({"path": str(library), "bytes": library_stat[2], "sha256": schema.evidence["library_sha256"], "stat": library_stat})
    metadata = {"job": str(job), "profile": profile_name, "source_result_sha256": outer_id["sha256"],
        "compilation_report_sha256": report_id["sha256"],
        "model_sha256": omc_id["sha256"], "model_bytes": omc_id["bytes"], "ddk_id": report.get("ddk_id"),
        "air": air_id, "model": omc_id, "schema": schema.evidence, "weight_payload_access": "NONE",
        "evidence_read_attempts": read_attempts}
    if report.get("decode_input_view") is not None:
        borrowed = _metadata(compilation / "decode-input-view.json", bindings,
            read_attempts=read_attempts, role="decode-input-view")
        if borrowed != report["decode_input_view"]:
            raise ValueError("decode input view differs from its successful compilation receipt")
        metadata["borrowed_prefill_inputs"] = {key: borrowed.get(key) for key in
            ("artifact_kind", "status", "source_job", "source_binding", "reference_method", "explicit_weight_copy_bytes", "audit_errors")}
    if observations is not None:
        observations.update(metadata)
    def source_metadata(path, role):
        value = _metadata(path, bindings, read_attempts=read_attempts, role=role)
        return value, bindings[-1]
    def source_pb2(path):
        _, identity = _audit_read(path, MAX_PB2_BYTES, "source-torchair-pb2",
            lambda checked: _read_small(checked, MAX_PB2_BYTES), read_attempts)
        bindings.append(identity)
        return identity
    producer = load_bound_source_schema(report, air_id, read_metadata=source_metadata,
        read_pb2=source_pb2, source_torchair_pb2=source_torchair_pb2)
    source_schema, source_binding = producer if producer is not None else (schema, {
        "status": "STRICT_SELECTED_DDK_SCHEMA_LEGACY_SOURCE_WITHOUT_PRODUCER_RECEIPT",
        "scope": "No producer declaration; same DDK schema is tried strictly, without cross-schema fallback"})
    enums = _enum_fields(schema)
    source_enums = tensor_enum_fields(source_schema) if producer is not None else enums
    metadata.update(source_schema=source_schema.evidence, source_schema_binding=source_binding,
        source_enum_fields=source_enums)
    if observations is not None:
        observations.update(metadata)
    air_payload, air_envelope = air_raw, "RAW_PROTOBUF"
    if air_raw.startswith(b"IMOD"):
        air_container = read_container(air_path)
        air_part = next(row for row in air_container["partitions"] if row["type"] == 0)
        air_payload = air_raw[air_part["offset"]:air_part["offset"] + air_part["bytes"]]
        air_envelope = "IMOD_MODEL_PARTITION"
    kind, source = _decode_air(source_schema, air_payload)
    container = read_container(omc_path)
    part = next(row for row in container["partitions"] if row["type"] == 0)
    model_raw = omc_raw[part["offset"]:part["offset"] + part["bytes"]]
    compiled = schema.decode(model_raw, "ModelDef")
    if source_capture is not None:
        source_capture.update(schema=source_schema, raw=air_payload, document=source, kind=kind)
    metadata.update(air_format=kind, air_envelope=air_envelope, air_payload_sha256=hashlib.sha256(air_payload).hexdigest())
    return metadata, source, compiled, enums


def load_bound_source_air(job, profile_name, bindings, *, source_torchair_pb2=None):
    """Expose the same verified payload for bounded raw scalar-cone readers."""
    capture = {}
    metadata, _, _, _ = load_target_graph(job, profile_name, bindings, source_capture=capture,
        source_torchair_pb2=source_torchair_pb2)
    return metadata, capture["schema"], capture["raw"], capture["document"], capture["kind"]


def _job(job, profile_name, bindings, *, observations=None):
    metadata, source, compiled, enums = load_target_graph(job, profile_name, bindings, observations=observations)
    metadata["source_lineage"] = analyze_state_lineage(source, metadata["source_enum_fields"])
    metadata["compiled_lineage"] = _join_compiled(metadata["source_lineage"], compiled, enums)
    return metadata, enums


def _probe(directory, job, bindings):
    directory, _ = _safe(directory, directory=True)
    report = _metadata(directory / "result.json", bindings)
    native = _metadata(directory / "model-io.json", bindings)
    native_id = bindings[-1]
    runtime = _metadata(directory / "runtime.json", bindings)
    runtime_id = bindings[-1]
    command = _metadata(directory / "model-io-command.json", bindings)
    if (report.get("artifact_kind") != "qwen35-target-model-io-probe" or report.get("status") != "IO_METADATA_OBSERVED"
            or report.get("job") != job["job"] or report.get("profile") != job["profile"]
            or report.get("compilation_report_sha256") != job["compilation_report_sha256"]
            or report.get("model_sha256") != job["model_sha256"] or report.get("model_bytes") != job["model_bytes"]
            or report.get("ddk_id") != job["ddk_id"] or report.get("remote_exitcode") != 0
            or report.get("retrieval_errors") != [] or report.get("lock_state") != "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
            or report.get("source_preservation") != "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED"
            or report.get("retrieved_sha256", {}).get("model-io.json") != native_id["sha256"]
            or native.get("artifact_kind") != "qwen35-hiai-model-io-probe" or native.get("evidence_kind") != "device-hiai"
            or native.get("status") != "IO_DESCRIPTORS_OBSERVED" or native.get("restore") != "PASSED"
            or native.get("descriptor_stage") != "restore-only" or native.get("manager_init") != "NOT_RUN"
            or native.get("model_sha256") != job["model_sha256"] or native.get("model_bytes") != job["model_bytes"]
            or native.get("model_path") != report.get("model_path") or native.get("model_load_api") != "file-v2"
            or native.get("model_load_scope") != "RESTORE_METADATA_ONLY" or native.get("restore_init") != "NOT_RUN_METADATA_ONLY"
            or native.get("external_weight_initialization") != "NOT_REQUESTED"
            or native.get("process_called") is not False or native.get("run_called") is not False
            or type(native.get("tensor_allocations_requested")) is not int or native["tensor_allocations_requested"] != 0
            or native.get("npu_supported") is not True or runtime.get("npu_supported") is not True
            or native.get("expected_runtime_version") != report.get("runtime_version")
            or runtime.get("runtime_version") != report.get("runtime_version")
            or report.get("retrieved_sha256", {}).get("runtime.json") != runtime_id["sha256"]
            or any(report.get(key) for key in ("error", "source_error", "lock_release_error", "native_evidence_error"))):
        raise ValueError("runtime descriptors do not bind this completed target artifact")
    for key in ("inputs", "outputs", "input_count", "output_count", "dtype_enum", "format_enum", "dtype_enum_values", "format_enum_values"):
        if native.get(key) != report.get(key):
            raise ValueError("saved native and outer runtime descriptors differ: " + key)
    for key in ("runtime_version", "firmware_id", "ddk_id"):
        if not isinstance(report.get(key), str) or not report[key] or native.get(key) != report[key]:
            raise ValueError("runtime identity missing or differs between native and outer receipt")
    for name in ("runtime.exitcode", "model-io.exitcode"):
        raw, identity = _read_small(directory / name, 16)
        if raw.strip() != b"0" or identity["sha256"] != report.get("retrieved_sha256", {}).get(name):
            raise ValueError("runtime probe exit hash/completion differs")
        bindings.append(identity)
    argv = command.get("argv")
    if not isinstance(argv, list) or len(argv) != 15 or any(not isinstance(value, str) for value in argv):
        raise ValueError("restore-only descriptor command is malformed")
    flags = dict(zip(argv[1::2], argv[2::2]))
    expected = {"--model-io-only": native["model_path"], "--model-sha256": job["model_sha256"],
        "--descriptor-stage": "restore-only", "--expected-runtime-version": report["runtime_version"],
        "--ddk-id": job["ddk_id"], "--firmware-id": report["firmware_id"], "--output": flags.get("--output")}
    if (flags != expected or not isinstance(flags.get("--output"), str) or not flags["--output"].endswith("/model-io.json")
            or command.get("cwd") != str(Path(native["model_path"]).parent)):
        raise ValueError("restore-only command does not bind these native descriptor identities")
    _runtime_keys(report, "inputs"); _runtime_keys(report, "outputs")
    return report


def inspect_target_state_lineage(prefill_job, decode_job, prefill_execution, decode_runtime_probe,
                                 output_dir, *, prefill_runtime_probe=None):
    """Retain one bounded state-graph audit. No feedback map is executed."""
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    bindings = []
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-state-lineage-audit",
        "status": "FAILED", "output": str(out), "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN",
        "weight_payload_access": "NONE", "state_feedback": "NOT_RUN", "state_continuity": "NOT_VALIDATED",
        "cross_model_state_identity": "UNVERIFIED", "feedback_ready": False,
        "numerical_validation": "NOT_RUN", "source_bindings": bindings,
        "scope": "Structural graph and declared tensor identities only. No state feedback, native inference, kernel correctness or numerical acceptance."}
    try:
        result["operator_port_contract"] = {"source": _source_record(), "pairs": {
            key: {"input": value[0], "output": value[1]} for key, value in _PAIRS.items()},
            "binding_policy": "Actual serialized named maps required; no fixed ordinal fallback"}
        result["prefill"], result["decode"] = {}, {}
        prefill, prefill_enums = _job(prefill_job, "cpro-40block-prefill-64", bindings, observations=result["prefill"])
        result["prefill"] = prefill
        decode, decode_enums = _job(decode_job, "cpro-40block-decode-1", bindings, observations=result["decode"])
        result.update(prefill=prefill, decode=decode)
        if prefill["ddk_id"] != decode["ddk_id"] or prefill["schema"]["descriptor_sha256"] != decode["schema"]["descriptor_sha256"]:
            raise ValueError("target pair must use the same selected DDK/schema")
        execution, _ = _safe(prefill_execution, directory=True)
        outer = _metadata(execution / "result.json", bindings)
        proof = _metadata(execution / "board/runtime-prefill-proof.json", bindings)
        if (outer.get("status") != "EXECUTED_UNVALIDATED" or outer.get("remote_exitcode") != 0
                or outer.get("device_execution") != "EXECUTED_OUTPUTS_RETRIEVED"
                or proof.get("compilation_report_sha256") != prefill["compilation_report_sha256"]
                or proof.get("model_sha256") != prefill["model_sha256"]
                or not isinstance(proof.get("runtime_probe"), str)):
            raise ValueError("prefill execution does not bind the completed static64 artifact")
        prefill_probe = _probe(prefill_runtime_probe or proof["runtime_probe"], prefill, bindings)
        decode_probe = _probe(decode_runtime_probe, decode, bindings)
        if any(prefill_probe[key] != decode_probe[key] for key in ("runtime_version", "firmware_id", "ddk_id")):
            raise ValueError("target pair descriptors came from different runtime identities")
        prefill["runtime_bindings"] = _runtime_bindings(prefill["compiled_lineage"], prefill_probe, prefill_enums)
        decode["runtime_bindings"] = _runtime_bindings(decode["compiled_lineage"], decode_probe, decode_enums)
        result["runtime_identity"] = {key: prefill_probe[key] for key in ("runtime_version", "firmware_id", "ddk_id")}
        borrowed = decode.get("borrowed_prefill_inputs", {})
        pair_bound = (borrowed.get("artifact_kind") == "qwen35-decode-input-reference-view"
            and borrowed.get("status") == "PASSED" and borrowed.get("source_job") == prefill["job"]
            and borrowed.get("reference_method") == "READONLY_SOURCE_REFERENCE_SYMLINK"
            and borrowed.get("explicit_weight_copy_bytes") == 0 and borrowed.get("audit_errors") == []
            and borrowed.get("source_binding", {}).get("source_compilation_report_sha256") == prefill["compilation_report_sha256"]
            and borrowed.get("source_binding", {}).get("source_result_sha256") == prefill["source_result_sha256"])
        result["target_pair_binding"] = {"status": "MATCHED_SAVED_PREFILL_INPUT_BORROW_DECLARATION" if pair_bound else "UNVERIFIED",
            "scope": "Saved successful input-view metadata only; no fresh weight payload access or cross-model numerical/state equivalence claim"}
        ready = pair_bound and all(job["source_lineage"]["status"] == "ALL80_NAMED_PORT_STATE_LINEAGES_OBSERVED"
            and job["source_lineage"]["observed_state_owner_count"] == 80
            and job["source_lineage"]["unique_state_owner_count"] == 80
            and job["compiled_lineage"]["matched_count"] == 80
            and job["runtime_bindings"]["matched_count"] == 80 for job in (prefill, decode))
        result["status"] = "ALL80_STRUCTURAL_STATE_BINDINGS_OBSERVED" if ready else "STATE_LINEAGE_OBSERVED_WITH_GAPS"
        result["next"] = ("Establish stable source-layer ownership across the two independent exports and original decode control/embedding inputs before deriving any feedback plan"
            if ready else "Review missing named ports, explicit FX identities or literal runtime tensor names in retained state rows; no ordinal or equal-shape fallback is permitted")
        for row in bindings:
            if _safe(row["path"])[1] != row["stat"]:
                raise ValueError("source evidence identity changed during lineage audit")
        result["source_preservation"] = "UNCHANGED_BOUNDED_METADATA_AIR_OMC_AND_SCHEMA_IDENTITIES"
    except (OSError, ValueError, TypeError, KeyError, StopIteration, RecursionError) as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(raw.encode()) > 4 << 20:
        raise ValueError("state lineage report exceeds4MiB budget")
    save_json(out / "report.json", result)
    return result

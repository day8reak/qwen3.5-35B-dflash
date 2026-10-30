"""Observe cross-export state context without inventing layer identities.

Source state owners follow actual data edges and named updater ports. External
parameter identities come only from hash-bound completed export inventories;
no weight is opened. A unique common weight context is a candidate, never by
itself a proven cross-model state owner or permission to feed a state back.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .artifacts import output_path, save_json
from . import target_state_lineage as lineage
from .target_compile_failure import _read_small, _safe
from .target_export_reuse import _inventory, _reference


_KIND = "qwen35-target-state-correspondence-evidence"
_MAX_DEPTH, _MAX_NODES, _MAX_PATHS, _MAX_ENQUEUED = 8, 1024, 16, 4096
_MAX_REPORT = 4 << 20


def _metadata(path, bindings):
    return lineage._metadata(path, bindings, role="cross-export-source-metadata")


def _same_pin(pin, identity):
    return (isinstance(pin, dict) and pin.get("path") == identity["path"]
        and pin.get("sha256") == identity["sha256"]
        and ("bytes" not in pin or pin["bytes"] == identity["bytes"]))


def _bound_inventory(metadata, bindings, sequence):
    binding = metadata.get("source_schema_binding", {})
    pin = binding.get("inventory")
    producer_pin, receipt_identity = pin, None
    if isinstance(binding.get("receipt"), dict):
        receipt = _metadata(binding["receipt"]["path"], bindings)
        receipt_identity = bindings[-1]
        if not _same_pin(binding["receipt"], receipt_identity):
            raise ValueError("source export receipt changed before cross-state inventory binding")
        producer_pin = receipt.get("inventory")
        if pin is None:
            pin = producer_pin
    # Read only the exact current compiler report before selecting a relocated
    # metadata copy. Its explicit transfer pin, never a rebased origin path,
    # is the same local inventory already validated during decode compilation.
    report = _metadata(Path(metadata["job"]) / "compilation/report.json", bindings)
    if bindings[-1]["sha256"] != metadata["compilation_report_sha256"]:
        raise ValueError("compilation report changed during cross-state metadata binding")
    view = report.get("decode_input_view")
    transfer = view.get("transfer") if isinstance(view, dict) else None
    if isinstance(view, dict) and "transfer" in view and not isinstance(transfer, dict):
        raise ValueError("declared decode inventory transfer must be an object; no origin-path fallback")
    if isinstance(transfer, dict) and "inventory" not in transfer:
        raise ValueError("declared decode inventory transfer lacks inventory; no origin-path fallback")
    local_transfer = isinstance(transfer, dict) and "inventory" in transfer
    if local_transfer:
        local_pin = transfer["inventory"]
        if (sequence != 1 or receipt_identity is None
                or not isinstance(local_pin, dict) or not isinstance(local_pin.get("path"), str)
                or not Path(local_pin["path"]).is_absolute()
                or type(local_pin.get("bytes")) is not int or not 0 < local_pin["bytes"] <= 32 << 20
                or re.fullmatch(r"[0-9a-f]{64}", str(local_pin.get("sha256", ""))) is None
                or not isinstance(producer_pin, dict) or not isinstance(producer_pin.get("path"), str)
                or not Path(producer_pin["path"]).is_absolute()
                or local_pin["sha256"] != producer_pin.get("sha256")
                or ("bytes" in producer_pin and (type(producer_pin["bytes"]) is not int
                    or producer_pin["bytes"] != local_pin["bytes"]))
                or ("export_receipt" in transfer and not _same_pin(transfer["export_receipt"], receipt_identity))):
            raise ValueError("declared local decode inventory transfer differs from its canonical producer pin; no origin-path fallback")
        if pin is not None and (not isinstance(pin, dict)
                or pin.get("sha256") != local_pin["sha256"]
                or ("bytes" in pin and pin["bytes"] != local_pin["bytes"])):
            raise ValueError("source-schema inventory and explicit local transfer inventory disagree")
        pin = local_pin
    if not isinstance(pin, dict) or not isinstance(pin.get("path"), str):
        raise ValueError("cross-state parameter context requires a producer-bound complete export inventory")
    raw, identity = _read_small(pin["path"], 32 << 20)
    if not _same_pin(pin, identity):
        raise ValueError("source inventory bytes/SHA differ from their producer binding")
    bindings.append(identity)
    manifest, rows, _ = _inventory(raw, sequence_length=sequence)
    if (any(manifest["air"].get(key) != metadata["air"][key] for key in ("bytes", "sha256"))
            or manifest["schema"].get("descriptor_sha256") != metadata["source_schema"]["descriptor_sha256"]):
        raise ValueError("inventory does not bind this exact source AIR and producer schema")
    snapshots = defaultdict(list)
    for row in report.get("input_snapshots", []):
        if isinstance(row, dict) and isinstance(row.get("source_relative"), str):
            snapshots[row["source_relative"]].append(row)
    for row in rows:
        selected = snapshots[row["relative_path"]]
        if len(selected) != 1 or any(selected[0].get(key) != row[key] for key in ("bytes", "sha256")):
            raise ValueError("inventory parameter reference is not in the bound compiler input snapshots: " + row["relative_path"])
    if local_transfer:
        metadata["inventory_resolution"] = {"status": "PINNED_LOCAL_TRANSFER_CONTENT_MATCH",
            "original_producer_inventory_pin": deepcopy(producer_pin), "selected_transfer_inventory_pin": deepcopy(pin),
            "current_inventory": identity, "canonical_receipt": receipt_identity,
            "compilation_report_sha256": metadata["compilation_report_sha256"],
            "origin_path_access": "NOT_ATTEMPTED", "root_rebasing": "NOT_PERFORMED",
            "source_receipt_mutation": "NONE", "weight_payload_access": "NONE"}
    return manifest, identity


def _provenance(node):
    attrs = lineage._attributes(node)
    keys = sorted(attrs)
    selected = [key for key in keys if re.search(r"meta|fqn|module|layer|scope|source|stack|fx", key, re.I)]
    values, used, truncated = {}, 0, len(keys) > 32 or len(selected) > 16
    for key in selected[:16]:
        value = lineage._provenance_value(attrs[key])
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        if len(raw) > 768:
            value = {"bounded_value_json_bytes": len(raw), "bounded_value_sha256": hashlib.sha256(raw).hexdigest(),
                "bounded_value_json_prefix": raw[:512].decode('utf-8', errors='replace'), "truncated": True}
            truncated = True
        size = len(json.dumps(value, ensure_ascii=False).encode())
        if used + size > 3072:
            truncated = True
            break
        values[key] = value
        used += size
    return {"attribute_keys": [key[:128] for key in keys[:32]], "attribute_key_count": len(keys), "selected": values,
        "selected_attribute_count": len(selected), "truncated": truncated,
        "acceptance": "OBSERVED_LITERAL_METADATA_ONLY_UNREVIEWED_FQN_SEMANTICS_NOT_PROMOTED"}


def _constants(document, manifest):
    _, nodes, _ = lineage._graph(document)
    files = {"/".join(_reference(row["reference"])): row for row in manifest["files"]}
    result = {}
    for node in nodes.values():
        if node["type"] != "FileConstant":
            continue
        attrs = lineage._attributes(node)
        reference = attrs.get("file_path", {}).get("s")
        try:
            normalized = "/".join(_reference(reference))
        except (ValueError, TypeError):
            raise ValueError("actual FileConstant has no exact bound inventory reference: " + node["name"])
        row = files.get(normalized)
        if row is None or reference not in row.get("references", [row["reference"]]):
            raise ValueError("actual FileConstant raw reference is not retained by its inventory aliases: " + node["name"])
        if node["name"] not in row.get("nodes", []):
            raise ValueError("actual FileConstant node is not retained by its inventory reference: " + node["name"])
        literal = lineage._provenance_value({"attributes": {key: value for key, value in attrs.items() if key != "file_path"},
            "output_descriptors": node.get("output_desc", [])})
        raw_literal = json.dumps(literal, ensure_ascii=False, sort_keys=True).encode()
        if len(raw_literal) > 2048:
            literal = {"bounded_json_bytes": len(raw_literal), "bounded_json_sha256": hashlib.sha256(raw_literal).hexdigest(),
                "bounded_json_prefix": raw_literal[:1536].decode('utf-8', errors='replace'), "truncated": True}
        result[node["name"]] = {"node": node["name"], "reference": reference, "normalized_relative_path": normalized,
            "bytes": row["bytes"], "sha256": row["sha256"], "provenance": _provenance(node),
            "literal_tensor_metadata": literal, "tensor_metadata_acceptance": "DIAGNOSTIC_ONLY_NOT_A_VALIDATED_TENSOR_EXTENT_OR_OWNER_CONTRACT"}
    if not result:
        raise ValueError("source AIR has no inventory-bound root FileConstant parameter anchors")
    return nodes, result


def _anchor_key(row):
    return row["sha256"], row["bytes"]


def _trace_context(state, nodes, constants, eligible_keys):
    update = state["updater"]
    node = nodes[update["node"]]
    edges = lineage._inputs(node)
    roles = update["serialized_input_port_map"]
    queue = deque((edges[index], role, []) for role, index in roles.items()
        if role != update["input_port_name"] and edges[index])
    seen, paths, terminals = set(), [], Counter()
    repeats, truncated, nearest, enqueued = 0, False, {}, len(queue)
    while queue:
        edge, role, path = queue.popleft()
        if role in nearest and len(path) + 1 > nearest[role]:
            continue
        identity = (role, edge)
        if identity in seen:
            repeats += 1
            continue
        if len(seen) >= _MAX_NODES or len(paths) >= _MAX_PATHS:
            truncated = True
            break
        seen.add(identity)
        producer, port = lineage._edge(edge, nodes)
        step = {"node": producer["name"], "type": producer["type"], "output_port": port}
        if producer["type"] == "FileConstant":
            anchor = constants[producer["name"]]
            if _anchor_key(anchor) in eligible_keys:
                nearest[role] = len(path) + 1
                paths.append({"context_input_port_name": role, "edges_from_updater_input": path + [step],
                    "distance": len(path) + 1, "parameter": anchor})
            else:
                terminals["NONUNIQUE_OR_NONCOMMON_PARAMETER_CONTENT"] += 1
            continue
        if producer["type"] in {"Data", "RefData", *lineage._PAIRS}:
            terminals["ROOT_OR_OTHER_STATE_UPDATER"] += 1
            continue
        if len(path) >= _MAX_DEPTH:
            terminals["DEPTH_LIMIT"] += 1
            truncated = True
            continue
        try:
            named = {index: name for name, index in lineage._port_map(producer, "input").items()}
        except ValueError:
            named = {}
        for index, source in enumerate(lineage._inputs(producer)):
            if source:
                if enqueued >= _MAX_ENQUEUED:
                    truncated = True
                    terminals["ENQUEUE_LIMIT"] += 1
                    break
                queue.append((source, role, path + [dict(step, input_port=index, input_port_name=named.get(index))]))
                enqueued += 1
        if truncated:
            break
    return {"paths": paths, "visited_edge_count": len(seen), "repeated_path_count": repeats,
        "enqueued_edge_count": enqueued, "terminals": dict(terminals), "truncated": truncated,
        "budgets": {"maximum_depth": _MAX_DEPTH, "maximum_edges": _MAX_NODES,
            "maximum_parameter_paths": _MAX_PATHS, "maximum_total_enqueued_edges": _MAX_ENQUEUED}}


def _context_signature(state, paths):
    # This structural signature deliberately ignores node names, RefData IDs,
    # FX counters, output ordinals and shape classes. It is a candidate only.
    role = state["updater"]
    entries = []
    for path in paths:
        edges = [(row["type"], row["output_port"], row.get("input_port"), row.get("input_port_name"))
            for row in path["edges_from_updater_input"]]
        entries.append((path["context_input_port_name"], _anchor_key(path["parameter"]), edges))
    return hashlib.sha256(json.dumps([role["type"], role["input_port_name"], role["output_port_name"],
        sorted(entries, key=lambda row: json.dumps(row, sort_keys=True))], sort_keys=True).encode()).hexdigest()


def analyze_state_correspondence(prefill, decode, prefill_inventory, decode_inventory, prefill_enums, decode_enums):
    """Pure bounded analysis; callers must first bind both graphs/inventories."""
    documents = {"prefill": prefill, "decode": decode}
    inventories = {"prefill": prefill_inventory, "decode": decode_inventory}
    enums = {"prefill": prefill_enums, "decode": decode_enums}
    models, constants, by_content, model_nodes = {}, {}, {}, {}
    for name in documents:
        nodes, selected = _constants(documents[name], inventories[name])
        constants[name], model_nodes[name] = selected, nodes
        refs = defaultdict(set)
        for row in selected.values(): refs[_anchor_key(row)].add(row["normalized_relative_path"])
        by_content[name] = refs
    common = by_content["prefill"].keys() & by_content["decode"].keys()
    unique = {key for key in common if len(by_content["prefill"][key]) == len(by_content["decode"][key]) == 1}
    for name in documents:
        nodes, selected = model_nodes[name], constants[name]
        _, _, boundary = lineage._graph(documents[name])
        source = lineage.analyze_state_lineage(documents[name], enums[name])
        states = []
        for state in source["states"]:
            row = {key: state.get(key) for key in ("source_refdata", "source_output_index", "source_output_edge", "status",
                "source_boundary_fx_tensor_identity", "source_producer_fx_tensor_identity", "source_fx_namespace_relation",
                "producer_shape_status", "source_netoutput_view")}
            if state["status"] != "NAMED_PORT_STATE_LINEAGE_OBSERVED":
                row["reason"] = state.get("reason", "SOURCE_STATE_OWNER_UNVERIFIED")
            else:
                ref = nodes[state["source_refdata"]]
                updater = nodes[state["updater"]["node"]]
                row.update(updater=state["updater"], source_input_fx_tensor_identity=state["source_input_fx_tensor_identity"],
                    state_input_provenance=_provenance(ref), state_input_descriptor_provenance=_provenance(ref["output_desc"][0]),
                    state_boundary_descriptor_provenance=_provenance(boundary["input_desc"][state["source_output_index"]]),
                    updater_output_descriptor_provenance=_provenance(updater["output_desc"][state["updater"]["output_port_index"]]),
                    updater_provenance=_provenance(updater), context=_trace_context(state, nodes, selected, unique))
            states.append(row)
        models[name] = {"source_owner_count": source["observed_state_owner_count"], "unique_source_owner_count": source["unique_state_owner_count"],
            "missing_source_owners": source["missing_state_owners"], "duplicate_source_owners": source["duplicate_state_owners"],
            "root_file_constant_count": len(selected), "states": states}
    signatures = {name: defaultdict(list) for name in models}
    for name, model in models.items():
        for state in model["states"]:
            if "context" not in state:
                continue
            eligible = [row for row in state["context"]["paths"] if _anchor_key(row["parameter"]) in unique]
            state["nearest_unique_common_parameter_context_count"] = len(eligible)
            if eligible and not state["context"]["truncated"]:
                signature = _context_signature(state, eligible)
                state["candidate_signature"] = signature
                signatures[name][signature].append(state)
    candidates = []
    for signature in signatures["prefill"].keys() & signatures["decode"].keys():
        left, right = signatures["prefill"][signature], signatures["decode"][signature]
        if len(left) == len(right) == 1:
            candidates.append({"prefill_refdata": left[0]["source_refdata"], "decode_refdata": right[0]["source_refdata"],
                "signature": signature, "status": "UNIQUE_SHARED_PARAMETER_CONTEXT_CANDIDATE",
                "proof_limit": "Identical whole-sidecar bytes and bounded role/edge context do not establish tensor values, stable source-layer ownership or cache roles across exports"})
    candidates.sort(key=lambda row: (row["prefill_refdata"], row["decode_refdata"]))
    complete = (len(candidates) == 80 and all(row["source_owner_count"] == row["unique_source_owner_count"] == 80 for row in models.values()))
    gaps = {name: {"source_owner_unverified": 80 - model["source_owner_count"],
        "context_budget_exhausted": sum(bool(row.get("context", {}).get("truncated")) for row in model["states"]),
        "no_unique_common_parameter_context": sum("context" in row and not row["context"]["paths"] for row in model["states"]),
        "ambiguous_context_signatures": sum(len(rows) > 1 for rows in signatures[name].values())}
        for name, model in models.items()}
    return {"status": "ALL80_UNIQUE_WEIGHT_CONTEXT_CANDIDATES_OBSERVED" if complete else "STATE_CORRESPONDENCE_EVIDENCE_OBSERVED_WITH_GAPS",
        "models": models, "common_parameter_content_count": len(common), "unique_common_parameter_content_count": len(unique),
        "gap_summary": gaps,
        "unique_state_context_candidate_count": len(candidates), "state_context_candidates": candidates,
        "proven_cross_model_state_owner_count": 0, "cross_model_state_identity": "UNVERIFIED_WEIGHT_CONTEXT_CANDIDATES_ONLY",
        "feedback_ready": False, "state_feedback": "NOT_RUN", "weight_payload_access": "NONE",
        "parameter_identity_scope": "WHOLE_SIDECAR_SHA256_AND_BYTES_FROM_BOUND_COMPLETED_EXPORT_INVENTORIES_NOT_TENSOR_CONTENT_OR_FRESH_WEIGHT_READ",
        "next": "Review exact provenance tags and parameter edge contexts; establish a supported stable source-layer/cache-role contract before promoting any candidate or preparing prefill feedback"}


def _pair_binding(prefill, decode):
    borrowed = decode.get("borrowed_prefill_inputs", {})
    matched = (borrowed.get("artifact_kind") == "qwen35-decode-input-reference-view"
        and borrowed.get("status") == "PASSED" and borrowed.get("source_job") == prefill["job"]
        and borrowed.get("reference_method") == "READONLY_SOURCE_REFERENCE_SYMLINK"
        and borrowed.get("explicit_weight_copy_bytes") == 0 and borrowed.get("audit_errors") == []
        and borrowed.get("source_binding", {}).get("source_compilation_report_sha256") == prefill["compilation_report_sha256"]
        and borrowed.get("source_binding", {}).get("source_result_sha256") == prefill["source_result_sha256"])
    return {"status": "MATCHED_SAVED_PREFILL_INPUT_BORROW_DECLARATION" if matched else "UNVERIFIED",
        "scope": "Saved successful compiler input-view metadata only; no fresh parameter read or cross-export state-owner equivalence claim"}


def inspect_target_state_correspondence(prefill_job, decode_job, output, *, prefill_source_torchair_pb2=None,
        decode_source_torchair_pb2=None, prefill_source_export_receipt=None):
    out = output_path(output); out.mkdir(parents=True, exist_ok=False)
    bindings = []
    result = {"schema_version": 1, "artifact_kind": _KIND, "status": "FAILED", "stage": "read-bound-models",
        "output": str(out), "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN", "weight_payload_access": "NONE",
        "feedback_ready": False, "state_feedback": "NOT_RUN", "cross_model_state_identity": "UNVERIFIED", "source_bindings": bindings}
    try:
        metadata, documents, inventories = {}, {}, {}
        for name, job, sequence, pb2 in (("prefill", prefill_job, 64, prefill_source_torchair_pb2),
                                       ("decode", decode_job, 1, decode_source_torchair_pb2)):
            meta, source, _, _ = lineage.load_target_graph(job, "cpro-40block-prefill-64" if sequence == 64 else "cpro-40block-decode-1", bindings,
                source_torchair_pb2=pb2, source_export_receipt=prefill_source_export_receipt if sequence == 64 else None)
            metadata[name], documents[name] = meta, source
            inventories[name], pin = _bound_inventory(meta, bindings, sequence)
            result[name] = {"job": meta["job"], "air": meta["air"], "model": meta["model"],
                "compilation_report_sha256": meta["compilation_report_sha256"], "source_schema_binding": meta["source_schema_binding"], "inventory": pin}
            if "inventory_resolution" in meta:
                result[name]["inventory_resolution"] = meta["inventory_resolution"]
        result["stage"] = "trace-source-state-parameter-contexts"
        result["target_pair_binding"] = _pair_binding(metadata["prefill"], metadata["decode"])
        result.update(analyze_state_correspondence(documents["prefill"], documents["decode"], inventories["prefill"], inventories["decode"],
            metadata["prefill"]["source_enum_fields"], metadata["decode"]["source_enum_fields"]))
        result["stage"] = "verify-active-preservation"
        for row in bindings:
            if _safe(row["path"])[1] != row["stat"]:
                raise ValueError("cross-state source evidence changed within this invocation: " + row["path"])
        result["source_preservation"] = "FRESH_SMALL_GRAPH_SCHEMA_METADATA_HASHES_AND_STRICT_ACTIVE_IDENTITIES"
        result["stage"] = "complete"
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > _MAX_REPORT:
        raise ValueError("cross-state correspondence evidence exceeds 4 MiB; reduce bounded diagnostic limits")
    save_json(out / "report.json", result)
    return result

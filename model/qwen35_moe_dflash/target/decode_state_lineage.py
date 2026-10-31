"""Audit same-model decode state ownership against already retrieved files.

This reads the small AIR/OMC graph, saved native descriptors and a freshly
revalidated output inventory. It never runs a model or prepares feedback.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.target import state_lineage as lineage
from qwen35_moe_dflash.common.file_io import _read_small, _relative, _safe
from qwen35_moe_dflash.target.decode_output_inventory import validate_target_decode_output_inventory
from qwen35_moe_dflash.target.prefill_output_inventory import _hash_retrieved, _json
from qwen35_moe_dflash.target.prefill_smoke_plan import _identity, _nd_view_key, _runtime_keys
from qwen35_moe_dflash.common.payload_revalidation import payload_identity_matches
from qwen35_moe_dflash.common.metadata_revalidation import metadata_identity_matches


_PROFILE = "cpro-40block-decode-1"
_KIND = "qwen35-target-decode-state-lineage-audit"
_MAX_REPORT = 4 << 20


def _decode_job(job, bindings, observations, *, source_torchair_pb2=None):
    metadata, source, compiled, enums = lineage.load_target_graph(job, _PROFILE, bindings, observations=observations,
        source_torchair_pb2=source_torchair_pb2)
    source_enums = metadata["source_enum_fields"]
    metadata["edge_representation_observations"] = {
        "source_air": lineage.observe_edge_representations(source, "source-air"),
        "compiled_omc": lineage.observe_edge_representations(compiled, "compiled-omc")}
    observations.update(metadata)
    metadata["source_lineage"] = lineage.analyze_state_lineage(source, source_enums)
    metadata["compiled_provenance_observation"] = lineage.observe_compiled_state_provenance(
        source, compiled, source_enums, enums)
    joined = lineage._join_compiled(metadata["source_lineage"], compiled, enums)
    _, _, refs, outputs = lineage._boundary(source, source_enums, allow_unmaterialized_producer=True,
        graph_context="source-air", source_boundary_identity=True)
    for row in joined["states"]:
        if row["status"] != "EXPLICIT_FX_TENSOR_IDENTITY_MATCHED":
            continue
        view = outputs[row["source_output_index"]]["view"]
        source_input = lineage._view(refs[row["source_refdata"]]["output_desc"][0], source_enums)
        compiled_views = [lineage._view(row[key], enums) for key in
            ("compiled_descriptor", "compiled_producer_descriptor", "compiled_input_descriptor")]
        row["source_netoutput_view"] = {"dtype": view[0], "shape_without_edge_singletons": list(view[1])}
        row["source_producer_shape_status"] = outputs[row["source_output_index"]]["producer_shape_status"]
        if source_input != view or any(candidate != view for candidate in compiled_views):
            row.update(status="UNVERIFIED", reason="SOURCE_COMPILED_STATE_TYPED_VIEW_DIFFERS")
        else:
            row["typed_view_binding"] = "EXPLICIT_SOURCE_NETOUTPUT_OWNER_AND_COMPILED_DESCRIPTORS_MATCHED"
    joined["matched_count"] = sum(row["status"] == "EXPLICIT_FX_TENSOR_IDENTITY_MATCHED" for row in joined["states"])
    joined["status"] = "ALL80_EXPLICIT_FX_TENSOR_IDENTITIES_MATCHED" if joined["matched_count"] == 80 else "COMPILED_IDENTITY_PARTIAL"
    metadata["compiled_lineage"] = joined
    return metadata, enums


def _bind_execution(inspection, job, probe_dir, probe, bindings):
    execution, _ = _safe(inspection["execution"], directory=True)
    plan = _json(execution / "plan.json", bindings)
    if (plan.get("profile") != _PROFILE or plan.get("gear") != 1 or plan.get("sequence_length") != 1
            or plan.get("compilation") != str(Path(job["job"]) / "compilation")
            or plan.get("runtime_probe") != str(probe_dir)
            or plan.get("compilation_report_sha256") != job["compilation_report_sha256"]
            or plan.get("model_sha256") != job["model_sha256"]
            or any(plan.get(key) != probe.get(key) for key in ("runtime_version", "firmware_id", "ddk_id"))):
        raise ValueError("decode output execution does not bind this exact job, model and named runtime probe")
    inputs, outputs = _runtime_keys(probe, "inputs"), _runtime_keys(probe, "outputs")
    for direction, runtime in (("inputs", inputs), ("outputs", outputs)):
        rows = plan.get(direction)
        if not isinstance(rows, list) or len(rows) != len(runtime):
            raise ValueError("executed decode plan lacks the complete named runtime " + direction)
        for index, (row, key) in enumerate(zip(rows, runtime)):
            if (type(row.get("index")) is not int or row["index"] != index
                    or row.get("runtime_tensor_name") != probe[direction][index].get("tensor_name")
                    or _nd_view_key((row.get("dtype"), tuple(row.get("shape", [])))) != _nd_view_key(key)):
                raise ValueError("executed decode descriptor differs from the actual named runtime " + direction)
    expected = [{key: row[key] for key in ("index", "shape", "dtype", "bytes")} for row in plan["outputs"]]
    if expected != inspection.get("expected_runtime_outputs"):
        raise ValueError("decode output inventory descriptors differ from the exact executed runtime plan")
    return execution, plan


def _output_rows(inspection):
    rows = inspection.get("output_files")
    if (not isinstance(rows, list) or len(rows) != 82 or any(not isinstance(row, dict) for row in rows)
            or any(type(row.get("runtime_index")) is not int for row in rows)
            or {row["runtime_index"] for row in rows} != set(range(82))
            or len({row.get("path") for row in rows}) != 82):
        raise ValueError("output inventory must bind82 distinct runtime slots and actual file paths")
    return rows


def _current_file_identity(path):
    path, stamp = _safe(path)
    identity = _identity(path.stat())
    if [identity[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")] != stamp:
        raise ValueError("retained decode identity changed before its first current snapshot")
    return {"path": str(path), "identity": identity}


def _bind_files(inspection, execution, *, payload_revalidation_observations=None, expected_current=None):
    """Use the validator's explicit log binding; never parse output_N names."""
    rows = _output_rows(inspection)
    current = {}
    for row in rows:
        if row.get("slot_binding") != "EXPLICIT_TOOL_LOG_PATH_AND_DESCRIPTOR_EXTENT":
            raise ValueError("state output file has no complete explicit tool log binding")
        path = _relative(execution / "board", row["path"])
        stamp = _safe(path)[1]
        if expected_current is not None:
            if stamp != expected_current[row["runtime_index"]]["stat"]:
                raise ValueError("decode output file changed during state lineage audit")
            current[row["runtime_index"]] = expected_current[row["runtime_index"]]
        elif stamp == row.get("stat"):
            current[row["runtime_index"]] = dict(row)
        else:
            try:
                identity, _ = _hash_retrieved(path, row)
            except ValueError as error:
                raise ValueError("decode output file changed after fresh inventory validation: " + str(error)) from error
            old = {"path": str(path), **{key: row[key] for key in ("bytes", "sha256", "stat")}}
            fresh = {"path": str(path), **identity}
            if not payload_identity_matches(old, fresh, role="decode-output",
                    observations=payload_revalidation_observations):
                raise ValueError("decode output file changed after fresh inventory validation")
            current[row["runtime_index"]] = dict(row, **identity)
    return current


def _file_mapping(job, inspection, execution, plan, files):
    rows = []
    for state in job["runtime_bindings"]["states"]:
        row = {key: state.get(key) for key in ("source_refdata", "source_output_index", "compiled_output_index")}
        row["status"] = "UNVERIFIED"
        if state["status"] != "EXPLICIT_TENSOR_NAME_AND_DESCRIPTOR_MATCHED":
            row["reason"] = state.get("reason", "RUNTIME_STATE_BINDING_UNVERIFIED")
        else:
            output_index, input_index = state["runtime_output_index"], state["runtime_input_index"]
            actual, target = files[output_index], plan["inputs"][input_index]
            source = plan["outputs"][output_index]
            if (target.get("name") != state["source_refdata"] or target.get("runtime_tensor_name") != state["runtime_input_name"]
                    or source.get("runtime_tensor_name") != state["runtime_output_name"]
                    or actual["bytes"] != source["bytes"] or actual["bytes"] != target["bytes"]
                    or _nd_view_key((source["dtype"], tuple(source["shape"]))) !=
                       _nd_view_key((target["dtype"], tuple(target["shape"])))):
                raise ValueError("proven runtime state mapping differs from the executed input/output descriptors")
            row.update(status="SAME_MODEL_STATE_FILE_BOUND", runtime_output_index=output_index,
                runtime_output_name=state["runtime_output_name"], runtime_input_index=input_index,
                runtime_input_name=state["runtime_input_name"], dtype=source["dtype"],
                output_shape=source["shape"], input_shape=target["shape"],
                output_file={"path": str(_relative(execution / "board", actual["path"])),
                    "relative_path": actual["path"], **{key: actual[key] for key in ("bytes", "sha256", "stat")}})
        rows.append(row)
    bound = [row for row in rows if row["status"] == "SAME_MODEL_STATE_FILE_BOUND"]
    complete = (len(bound) == 80 and {row["source_refdata"] for row in bound} == lineage._STATES
        and len({row["runtime_input_index"] for row in bound}) == 80
        and len({row["runtime_output_index"] for row in bound}) == 80)
    return {"status": "ALL80_SAME_MODEL_STATE_FILES_BOUND" if complete else "STATE_FILE_BINDING_PARTIAL",
        "bound_count": len(bound), "states": rows,
        "scope": "Structural output-file to named input mapping for this identical compiled model only; no feedback prepared or executed"}


def _gaps(job):
    result = []
    source = job.get("source_lineage", {})
    for key, reason in (("duplicate_state_owners", "DUPLICATE_SOURCE_STATE_OWNERS"),
                        ("missing_state_owners", "MISSING_SOURCE_STATE_OWNERS")):
        owners = source.get(key, [])
        if owners:
            result.append({"stage": "source_lineage", "reason": reason, "count": len(owners),
                "blocks_state_mapping": True, "examples": [{"source_refdata": name} for name in owners[:3]]})
    for stage, success in (("source_lineage", "NAMED_PORT_STATE_LINEAGE_OBSERVED"),
                           ("compiled_lineage", "EXPLICIT_FX_TENSOR_IDENTITY_MATCHED"),
                           ("runtime_bindings", "EXPLICIT_TENSOR_NAME_AND_DESCRIPTOR_MATCHED"),
                           ("state_file_mapping", "SAME_MODEL_STATE_FILE_BOUND")):
        groups = {}
        for row in job.get(stage, {}).get("states", []):
            reason = row.get("reason") if row.get("status") != success else None
            if stage == "source_lineage" and row.get("producer_shape_status") != "FIXED_ND_VIEW_MATCHED":
                reason = reason or "SOURCE_PRODUCER_SHAPE_UNMATERIALIZED"
            if reason:
                groups.setdefault(reason, []).append({key: row[key] for key in
                    ("source_refdata", "source_output_index", "compiled_output_index") if key in row})
        for reason, rows in sorted(groups.items()):
            result.append({"stage": stage, "reason": reason, "count": len(rows),
                "blocks_state_mapping": reason != "SOURCE_PRODUCER_SHAPE_UNMATERIALIZED", "examples": rows[:3]})
    return result


def _observe_target_decode_state_lineage(job, runtime_probe, output_inspection, output, *, source_torchair_pb2=None):
    """Recompute one complete observation in memory, without creating files."""
    out = Path(output)
    bindings, revalidations, metadata_revalidations, payload_revalidations = [], [], [], []
    result = {"schema_version": 1, "artifact_kind": _KIND, "status": "FAILED", "output": str(out),
        "stage": "validate-output-inspection", "device_execution": "NOT_RUN", "device_calls": "NONE",
        "model_compilation": "NOT_RUN", "model_payload_access": "SMALL_BOUND_AIR_AND_PRIMARY_OMC_GRAPH_ONLY",
        "weight_payload_access": "NONE", "state_feedback": "NOT_RUN", "feedback_ready": False,
        "state_continuity": "NOT_VALIDATED", "numerical_validation": "NOT_RUN", "source_bindings": bindings,
        "cross_model_state_identity": "NOT_EVALUATED_SAME_MODEL_ONLY", "same_model_state_mapping": "UNVERIFIED",
        "graph_edge_policy": lineage.graph_edge_policy(),
        "primary_omc_revalidation_observations": revalidations,
        "metadata_revalidation_observations": metadata_revalidations,
        "payload_revalidation_observations": payload_revalidations,
        "scope": "Same-model structural state identity and actual logged file bytes only; no feedback inputs, inference, control contract, greedy generation or accuracy acceptance"}
    try:
        inspection_dir, _ = _safe(output_inspection, directory=True)
        saved = _json(inspection_dir / "report.json", bindings)
        inspection_identity = bindings[-1]
        execution_root, _ = _safe(saved["execution"], directory=True)
        initial_plan = _json(execution_root / "plan.json", bindings)
        input_baselines = [_current_file_identity(row["path"]) for row in initial_plan.get("source_inputs", [])]
        retained_baselines, initial_files = [], {}
        for row in _output_rows(saved):
            baseline = _current_file_identity(_relative(execution_root / "board", row["path"]))
            retained_baselines.append(baseline)
            initial_files[row["runtime_index"]] = dict(row, stat=[baseline["identity"][key]
                for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")])
        for name in ("tool.stdout", "tool.stderr", "tool.exitcode", "tool.pid"):
            if name in saved.get("tool_logs", {}):
                retained_baselines.append(_current_file_identity(_relative(execution_root / "board", name)))
        inspection = validate_target_decode_output_inventory(inspection_dir, revalidation_observations=revalidations,
            metadata_revalidation_observations=metadata_revalidations,
            payload_revalidation_observations=payload_revalidations)
        if inspection != saved:
            raise ValueError("output inspection changed during fresh validation")
        result["output_inspection"] = inspection_identity
        result["execution"] = inspection["execution"]
        result["trial_inputs"] = inspection["trial_inputs"]
        result["source_input_current_identities"] = input_baselines
        result["stage"] = "read-decode-graphs"
        result["decode"] = {}
        decoded, enums = _decode_job(job, bindings, result["decode"], source_torchair_pb2=source_torchair_pb2)
        result["decode"] = decoded
        result["operator_port_contract"] = {"source": lineage._source_record(),
            "pairs": {key: {"input": value[0], "output": value[1]} for key, value in lineage._PAIRS.items()},
            "binding_policy": "Actual source edges and named updater ports; unique source NetOutput FX identities join coherent compiled boundary FX. Literal declared names or a documented IO-label protocol observed across all82 native names/views binds runtime slots. No text normalization, producer-FX fallback, native-order assumption or ordinal state-owner inference"}
        result["stage"] = "bind-named-runtime-descriptors"
        probe_dir, _ = _safe(runtime_probe, directory=True)
        probe = lineage._probe(probe_dir, decoded, bindings)
        decoded["runtime_bindings"] = lineage._runtime_bindings(decoded["compiled_lineage"], probe, enums)
        result["runtime_identity"] = {key: probe[key] for key in ("runtime_version", "firmware_id", "ddk_id")}
        result["stage"] = "bind-executed-state-files"
        execution, plan = _bind_execution(inspection, decoded, probe_dir, probe, bindings)
        files = _bind_files(inspection, execution, expected_current=initial_files)
        decoded["state_file_mapping"] = _file_mapping(decoded, inspection, execution, plan, files)
        result["gap_summary"] = _gaps(decoded)
        result["same_model_state_mapping"] = decoded["state_file_mapping"]["status"]
        source_rows = decoded["source_lineage"]["states"]
        complete = (len(source_rows) == 80
            and all(row["status"] == "NAMED_PORT_STATE_LINEAGE_OBSERVED" for row in source_rows)
            and {row["source_refdata"] for row in source_rows} == lineage._STATES
            and decoded["source_lineage"]["unique_state_owner_count"] == 80
            and decoded["source_lineage"]["observed_state_owner_count"] == 80
            and decoded["compiled_lineage"]["matched_count"] == 80
            and decoded["runtime_bindings"]["matched_count"] == 80
            and decoded["state_file_mapping"]["status"] == "ALL80_SAME_MODEL_STATE_FILES_BOUND")
        result["status"] = "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED" if complete else "DECODE_STATE_LINEAGE_OBSERVED_WITH_GAPS"
        result["next"] = ("Prepare a separate same-model continuation with all 80 individually bound states, proven original embedding bytes and declared control assumptions; then verify the actual state chain"
            if complete else "Review retained gap reasons and examples; obtain missing literal graph/runtime identities without ordinal or equal-shape substitutions")
        result["stage"] = "verify-source-preservation"
        if validate_target_decode_output_inventory(inspection_dir, revalidation_observations=revalidations,
                metadata_revalidation_observations=metadata_revalidations,
                payload_revalidation_observations=payload_revalidations) != inspection:
            raise ValueError("output inspection changed during state lineage audit")
        _bind_files(inspection, execution, expected_current=files)
        for row in input_baselines:
            path, _ = _safe(row["path"])
            if _identity(path.stat()) != row["identity"]:
                raise ValueError("decode source input changed during state lineage audit: " + str(path))
        for row in retained_baselines:
            path, _ = _safe(row["path"])
            if _identity(path.stat()) != row["identity"]:
                raise ValueError("retained decode output/log changed during state lineage audit: " + str(path))
        for identity in bindings:
            if _safe(identity["path"])[1] != identity["stat"]:
                raise ValueError("bound graph/runtime evidence changed during state lineage audit")
        result["source_preservation"] = "FRESH_BOUND_GRAPH_AND_OUTPUT_HASHES_UNCHANGED_FINAL_STATS"
        result["stage"] = "complete"
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result.update(status="FAILED", same_model_state_mapping="UNVERIFIED_EVIDENCE_FAILURE",
            error=f"{type(error).__name__}: {error}")
        result["gap_summary"] = _gaps(result.get("decode", {}))
        result["next"] = "Review the retained failing stage and exact evidence error; no feedback mapping is executable"
    for attempt in result.get("decode", {}).get("evidence_read_attempts", []):
        observed = attempt.get("source_air_ctime_revalidation")
        if observed is not None and observed not in payload_revalidations:
            payload_revalidations.append(observed)
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > _MAX_REPORT:
        raise ValueError("decode state lineage report exceeds4MiB budget")
    return result


def inspect_target_decode_state_lineage(job, runtime_probe, output_inspection, output, *, source_torchair_pb2=None):
    """Retain one same-artifact audit in a separate exclusively created output."""
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = _observe_target_decode_state_lineage(job, runtime_probe, output_inspection, out,
        source_torchair_pb2=source_torchair_pb2)
    save_json(out / "report.json", result)
    return result


def _project_revalidated_identities(saved, current, context, path=()):
    """Compare saved claims, projecting only freshly proved historical ctimes."""
    observations = {(): {"primary_omc_revalidation_observations", "metadata_revalidation_observations", "payload_revalidation_observations"},
        ("decode",): {"primary_omc_revalidation_observations", "source_air_revalidation_observations", "evidence_read_attempts"}}
    if isinstance(saved, dict) and isinstance(current, dict):
        ignored = observations.get(path, set())
        old, new = ({key: row for key, row in value.items() if key not in ignored} for value in (saved, current))
        if old.keys() != new.keys():
            raise ValueError("saved decode lineage claim keys changed at " + "/".join(map(str, path)))
        result = deepcopy(new)
        identity_keys = {"path", "bytes", "sha256", "stat"}
        if identity_keys <= old.keys() and identity_keys <= new.keys() and old != new:
            selected = new.get("path")
            prior = {key: old[key] for key in identity_keys}
            fresh = {key: new[key] for key in identity_keys}
            accepted = False
            if selected in context["outputs"]:
                accepted = payload_identity_matches(prior, fresh, role="decode-output", observations=context["payload"])
            elif selected == context["air"]:
                accepted = payload_identity_matches(prior, fresh, role="source-air", observations=context["payload"])
            elif selected == context["model"]:
                accepted = (prior["path"] == fresh["path"] and prior["bytes"] == fresh["bytes"]
                    and prior["sha256"] == fresh["sha256"] and isinstance(prior["stat"], list) and len(prior["stat"]) == 5
                    and all(type(value) is int and value >= 0 for value in prior["stat"])
                    and prior["stat"][:4] == fresh["stat"][:4] and 0 < fresh["bytes"] <= 64 << 20)
                if accepted and prior["stat"] != fresh["stat"]:
                    context["primary"].append({"kind": "PRIMARY_OMC_CTIME_ONLY_REVALIDATED", "status": "SHA256_VERIFIED",
                        "path": selected, "bytes": fresh["bytes"], "sha256": fresh["sha256"], "recorded_stat": prior["stat"],
                        "current_stat": fresh["stat"], "scope": "SMALL_PRIMARY_ALREADY_FRESHLY_HASHED_BY_SHARED_STATE_AUDIT", "receipt_mutation": "NONE"})
            elif selected in context["bound"]:
                accepted = metadata_identity_matches(prior, fresh, role="decode-state-lineage-metadata", observations=context["metadata"])
            if accepted:
                result["stat"] = deepcopy(old["stat"])
        if {"path", "identity"} <= old.keys() and old.get("identity") != new.get("identity"):
            source = context["inputs"].get(new.get("path"))
            if source is not None:
                prior = {key: source[key] for key in ("path", "bytes", "sha256")}
                fresh = dict(prior, identity=new["identity"])
                prior["identity"] = old["identity"]
                if payload_identity_matches(prior, fresh, role="decode-input", observations=context["payload"]):
                    result["identity"] = deepcopy(old["identity"])
        for key in result:
            result[key] = _project_revalidated_identities(old[key], result[key], context, (*path, key))
        return result
    if isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            raise ValueError("saved decode lineage claim length changed at " + "/".join(map(str, path)))
        return [_project_revalidated_identities(old, new, context, (*path, index)) for index, (old, new) in enumerate(zip(saved, current))]
    if type(saved) is not type(current) or saved != current:
        raise ValueError("saved decode lineage claim changed at " + "/".join(map(str, path)))
    return current


def validate_target_decode_state_lineage(directory, *, revalidation_observations=None,
        metadata_revalidation_observations=None, payload_revalidation_observations=None):
    """Recompute a saved complete audit once and return CURRENT file identities.

    No receipt or output is written. Historical read observations are not proof
    inputs; mapping, graph, native descriptors and all80 actual file bindings
    are freshly recomputed and compared. Active invocation identities stay exact.
    """
    collectors = (revalidation_observations, metadata_revalidation_observations, payload_revalidation_observations)
    if any(value is not None and not isinstance(value, list) for value in collectors):
        raise ValueError("state lineage revalidation observations must be lists or None")
    root, _ = _safe(directory, directory=True)
    bindings = []
    saved = _json(root / "report.json", bindings)
    saved_identity = bindings[-1]
    if (saved.get("artifact_kind") != _KIND or saved.get("status") != "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED"
            or saved.get("stage") != "complete" or saved.get("output") != str(root)
            or saved.get("same_model_state_mapping") != "ALL80_SAME_MODEL_STATE_FILES_BOUND"
            or saved.get("feedback_ready") is not False or saved.get("state_feedback") != "NOT_RUN"
            or not isinstance(saved.get("execution"), str)
            or not isinstance(saved.get("output_inspection", {}).get("path"), str)
            or not isinstance(saved.get("decode", {}).get("job"), str)):
        raise ValueError("expected a complete retained same-model decode state-lineage audit")
    plan = _json(Path(saved["execution"]) / "plan.json", bindings)
    job, probe = saved["decode"]["job"], plan.get("runtime_probe")
    if (not isinstance(probe, str) or plan.get("compilation") != str(Path(job) / "compilation")
            or Path(saved["output_inspection"]["path"]).name != "report.json"):
        raise ValueError("saved state lineage job/probe/inspection paths differ from its executed plan")
    schema = saved["decode"].get("source_schema_binding", {})
    pb2 = schema.get("selected_pb2", {}).get("path") if schema.get("selection") == "EXPLICIT_IDENTICAL_PB2_COPY" else None
    current = None
    try:
        current = _observe_target_decode_state_lineage(job, probe, Path(saved["output_inspection"]["path"]).parent, root,
            source_torchair_pb2=pb2)
        if current.get("status") != "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED":
            raise ValueError("fresh decode state lineage is incomplete: " + current.get("stage", "unknown") + "; " + current.get("error", str(current.get("gap_summary", []))[:1024]))
        # Compare JSON artifact semantics: integer mapping keys and tuples have
        # the same representation as the exclusively written original receipt.
        current = json.loads(json.dumps(current, ensure_ascii=False, allow_nan=False))
        mapped = current["decode"]["state_file_mapping"]["states"]
        context = {"bound": {row["path"] for row in current["source_bindings"]},
            "outputs": {row["output_file"]["path"] for row in mapped},
            "air": current["decode"]["air"]["path"], "model": current["decode"]["model"]["path"],
            "inputs": {row["path"]: row for row in plan.get("source_inputs", [])},
            "primary": current["primary_omc_revalidation_observations"],
            "metadata": current["metadata_revalidation_observations"], "payload": current["payload_revalidation_observations"]}
        _project_revalidated_identities(saved, current, context)
        for row in [*bindings, *current["source_bindings"]]:
            if _safe(row["path"])[1] != row["stat"]:
                raise ValueError("state lineage evidence changed during read-only revalidation: " + row["path"])
        for row in current["source_input_current_identities"]:
            path, _ = _safe(row["path"])
            if _identity(path.stat()) != row["identity"]:
                raise ValueError("decode input changed during read-only lineage revalidation")
        for row in mapped:
            if _safe(row["output_file"]["path"])[1] != row["output_file"]["stat"]:
                raise ValueError("state output changed during read-only lineage revalidation")
        current["saved_report_identity"] = saved_identity
        current["validation_mode"] = "READ_ONLY_RECOMPUTED_CURRENT_GRAPH_RUNTIME_AND_80_STATE_FILE_BINDINGS"
        return current
    finally:
        if current is not None:
            for target, key in zip(collectors, ("primary_omc_revalidation_observations", "metadata_revalidation_observations", "payload_revalidation_observations")):
                if target is not None:
                    target.extend(current.get(key, []))

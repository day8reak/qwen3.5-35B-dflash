"""Audit same-model decode state ownership against already retrieved files.

This reads the small AIR/OMC graph, saved native descriptors and a freshly
revalidated output inventory. It never runs a model or prepares feedback.
"""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import output_path, save_json
from . import target_state_lineage as lineage
from .target_compile_failure import _relative, _safe
from .target_decode_output_inventory import validate_target_decode_output_inventory
from .target_prefill_output_inventory import _json
from .target_prefill_smoke_plan import _nd_view_key, _runtime_keys


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
    joined = lineage._join_compiled(metadata["source_lineage"], compiled, enums)
    _, _, refs, outputs = lineage._boundary(source, source_enums, allow_unmaterialized_producer=True, graph_context="source-air")
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


def _bind_files(inspection, execution):
    """Use the validator's explicit log binding; never parse output_N names."""
    rows = inspection.get("output_files")
    if (not isinstance(rows, list) or len(rows) != 82 or any(not isinstance(row, dict) for row in rows)
            or any(type(row.get("runtime_index")) is not int for row in rows)
            or {row["runtime_index"] for row in rows} != set(range(82))
            or len({row.get("path") for row in rows}) != 82):
        raise ValueError("output inventory must bind82 distinct runtime slots and actual file paths")
    for row in rows:
        if row.get("slot_binding") != "EXPLICIT_TOOL_LOG_PATH_AND_DESCRIPTOR_EXTENT":
            raise ValueError("state output file has no complete explicit tool log binding")
        if _safe(_relative(execution / "board", row["path"]))[1] != row.get("stat"):
            raise ValueError("decode output file changed after fresh inventory validation")
    return {row["runtime_index"]: row for row in rows}


def _file_mapping(job, inspection, execution, plan):
    files = _bind_files(inspection, execution)
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


def inspect_target_decode_state_lineage(job, runtime_probe, output_inspection, output, *, source_torchair_pb2=None):
    """Retain one read-only same-artifact decode ownership audit."""
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    bindings, revalidations = [], []
    result = {"schema_version": 1, "artifact_kind": _KIND, "status": "FAILED", "output": str(out),
        "stage": "validate-output-inspection", "device_execution": "NOT_RUN", "device_calls": "NONE",
        "model_compilation": "NOT_RUN", "model_payload_access": "SMALL_BOUND_AIR_AND_PRIMARY_OMC_GRAPH_ONLY",
        "weight_payload_access": "NONE", "state_feedback": "NOT_RUN", "feedback_ready": False,
        "state_continuity": "NOT_VALIDATED", "numerical_validation": "NOT_RUN", "source_bindings": bindings,
        "cross_model_state_identity": "NOT_EVALUATED_SAME_MODEL_ONLY", "same_model_state_mapping": "UNVERIFIED",
        "graph_edge_policy": lineage.graph_edge_policy(),
        "primary_omc_revalidation_observations": revalidations,
        "scope": "Same-model structural state identity and actual logged file bytes only; no feedback inputs, inference, control contract, greedy generation or accuracy acceptance"}
    try:
        inspection_dir, _ = _safe(output_inspection, directory=True)
        saved = _json(inspection_dir / "report.json", bindings)
        inspection = validate_target_decode_output_inventory(inspection_dir, revalidation_observations=revalidations)
        if inspection != saved:
            raise ValueError("output inspection changed during fresh validation")
        result["output_inspection"] = bindings[-1]
        result["execution"] = inspection["execution"]
        result["trial_inputs"] = inspection["trial_inputs"]
        result["stage"] = "read-decode-graphs"
        result["decode"] = {}
        decoded, enums = _decode_job(job, bindings, result["decode"], source_torchair_pb2=source_torchair_pb2)
        result["decode"] = decoded
        result["operator_port_contract"] = {"source": lineage._source_record(),
            "pairs": {key: {"input": value[0], "output": value[1]} for key, value in lineage._PAIRS.items()},
            "binding_policy": "Actual serialized named maps, unique FX identities and literal runtime names; no ordinal fallback"}
        result["stage"] = "bind-named-runtime-descriptors"
        probe_dir, _ = _safe(runtime_probe, directory=True)
        probe = lineage._probe(probe_dir, decoded, bindings)
        decoded["runtime_bindings"] = lineage._runtime_bindings(decoded["compiled_lineage"], probe, enums)
        result["runtime_identity"] = {key: probe[key] for key in ("runtime_version", "firmware_id", "ddk_id")}
        result["stage"] = "bind-executed-state-files"
        execution, plan = _bind_execution(inspection, decoded, probe_dir, probe, bindings)
        decoded["state_file_mapping"] = _file_mapping(decoded, inspection, execution, plan)
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
        if validate_target_decode_output_inventory(inspection_dir, revalidation_observations=revalidations) != inspection:
            raise ValueError("output inspection changed during state lineage audit")
        _bind_files(inspection, execution)
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
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > _MAX_REPORT:
        raise ValueError("decode state lineage report exceeds4MiB budget")
    save_json(out / "report.json", result)
    return result

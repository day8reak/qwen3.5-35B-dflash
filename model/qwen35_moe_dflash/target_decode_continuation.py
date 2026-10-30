"""Prepare one same-model decode continuation with 80 independent state files.

This performs host preparation and evidence validation only. Position/mask
advancement and carried local-query controls remain explicit experiments.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct

from .artifacts import output_path, save_json
from . import original_decode_row_transfer as rows
from .metadata_revalidation import metadata_identity_matches, metadata_sequence_matches
from .payload_revalidation import payload_identity_matches
from .original_decode_embedding_probe import _input
from .target_compile_failure import _relative, _safe
from .target_decode_smoke_plan import (validate_decode_smoke_plan, _PRIMARY_STAT_SCOPE,
    _PRIMARY_REVALIDATED_SCOPE, _PRIMARY_REVALIDATED_ACCESS)
from .target_decode_state_lineage import validate_target_decode_state_lineage
from .target_prefill_smoke_plan import _identity, _nd_view_key, _source


ASSUMPTION = "single-query-absolute-position-plus-one"
_KIND = "qwen35-target-decode-continuation-preparation"
_PLAN_KIND = "qwen35-original-decode-continuation-plan"
_READY = "SAME_MODEL_DECODE_CONTINUATION_PREPARED"
_SCOPE = "SINGLE_SAME_MODEL_DECODE1_CONTINUATION_EXPERIMENTAL_CONTROLS"
_STATES = {f"RefData_{index}" for index in range(80)}
_CONTROL = {"arg107_1": ("position_ids", "INT32", 4), "arg2_1": ("new_kv_cache_pos", "INT32", 4),
    "arg1238_1": ("output_pos", "INT32", 4), "arg114_1": ("allQLen", "INT64", 8),
    "arg14_1": ("conv_state_pos", "INT32", 512), "arg18_1": ("effective_length", "INT16", 2),
    "arg113_1": ("attention_mask", "FLOAT32", 8192)}


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _meta(path, snapshot, maximum=32 << 20):
    value, identity = rows._json(path, maximum)
    rows._remember_current_metadata(identity, snapshot)
    return value, identity


def _payload(path, snapshot, expected=None):
    path, stamp = _safe(path)
    current = _identity(path.stat())
    if [current[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")] != stamp:
        raise ValueError("continuation payload changed before its current snapshot: " + str(path))
    if expected is not None:
        valid = (stamp == expected if isinstance(expected, list) else current == expected)
        if not valid:
            raise ValueError("continuation payload differs from its freshly read identity: " + str(path))
    if str(path) in snapshot and snapshot[str(path)] != current:
        raise ValueError("continuation source payload changed within this invocation: " + str(path))
    snapshot[str(path)] = current


def _stable(metadata, payloads):
    rows._assert_current_metadata(metadata)
    for path, identity in payloads.items():
        checked, _ = _safe(path)
        if _identity(checked.stat()) != identity:
            raise ValueError("continuation source payload changed within this invocation: " + path)


def _collectors():
    return {"revalidation_observations": [], "metadata_revalidation_observations": [], "payload_revalidation_observations": []}


def _bound_rows(directory, metadata, payloads, observations):
    directory, _ = _safe(directory, directory=True)
    report, report_id = _meta(directory / "report.json", metadata)
    if (report.get("artifact_kind") != "qwen35-original-decode-row-transfer-import"
            or report.get("status") != "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED"
            or report.get("output") != str(directory) or report.get("error")
            or report.get("schema_version") != 2 or report.get("binding_method") != rows._QUERY_METHOD):
        raise ValueError("expected the successful explicit-query original row import")
    prepared, _ = _safe(report["prepared"], directory=True)
    manifest, _ = _meta(prepared / "manifest.json", metadata)
    request, request_id = _meta(prepared / "request.json", metadata, rows._REQUEST_MAX)
    audit, audit_id = _meta(prepared / "source-audit/report.json", metadata)
    response, response_id = _meta(report["response"]["path"], metadata, rows._RESPONSE_MAX)
    for identity in audit.get("source_bindings", []):
        _payload(identity["path"], payloads)
    for identity in [audit["original_input_ids"], *audit.get("original_trial_inputs", [])]:
        _payload(identity["path"], payloads, identity["stat"])
    for expected, current, role in ((report["request"], request_id, "row-transfer-request-metadata"),
            (manifest["request"], request_id, "row-transfer-request-metadata"),
            (report["source_audit"], audit_id, "embedding-audit-metadata"),
            (manifest["source_audit"], audit_id, "embedding-audit-metadata"),
            (report["response"], response_id, "row-transfer-response-metadata")):
        if not metadata_identity_matches(expected, current, role=role, observations=observations["metadata_revalidation_observations"]):
            raise ValueError("original row import metadata differs beyond freshly verified historical ctime")
    if (manifest.get("status") != "ORIGINAL_DECODE_ROW_TRANSFER_PREPARED" or manifest.get("schema_version") != 2
            or manifest.get("query_token_id") != request.get("query_token_id") or manifest.get("binding_method") != rows._QUERY_METHOD
            or request.get("source_audit_sha256") != audit_id["sha256"] or request.get("binding_digest") != manifest.get("binding_digest")):
        raise ValueError("row import no longer binds the exact prepared request and source audit")
    code_path = prepared / "extract_rows.py"
    _payload(code_path, payloads)
    _, code_id = rows._read(code_path, 1 << 20)
    recovery = report.get("copied_extractor_revalidation_observations", [])
    expected_code = manifest["extraction_script"]
    if recovery:
        if (len(recovery) != 1 or report.get("copied_extractor_revalidation_count") != 1
                or recovery[0].get("kind") != "COPIED_EXTRACTOR_CTIME_ONLY_REVALIDATED"
                or recovery[0].get("status") != "FULL_SOURCE_SHA256_VERIFIED"
                or recovery[0].get("recorded_identity") != expected_code):
            raise ValueError("original row import copied-source recovery receipt is inconsistent")
        expected_code = recovery[0]["current_identity"]
        if rows._full_file_identity(code_path) != recovery[0]["current_full_identity"]:
            raise ValueError("copied extraction source changed since the successful row import")
    if code_id != expected_code or code_id["sha256"] != manifest["extraction_script"]["sha256"]:
        raise ValueError("copied extraction source changed since the successful row import")
    _payload(code_path, payloads, code_id["stat"])
    raw_ids, executed_row, executed_scale, next_id = rows._fresh_local(audit, Path(manifest["inspection"]), Path(manifest["config_yaml"]),
        **observations, active_metadata_identities=metadata)
    requested_ids, requested_row, requested_scale, _ = rows._request(request)
    if (raw_ids, executed_row, executed_scale, next_id) != (requested_ids, requested_row, requested_scale, request["next_token_id"]):
        raise ValueError("row request no longer binds actual source inputs and fresh greedy logits")
    next_row, next_scale, query, positions = rows._response(response, request, request_id)
    if (response.get("extractor_sha256") != manifest["extraction_script"]["sha256"]
            or report.get("selected_source_token_id") != query or report.get("selected_next_token_id") != next_id
            or report.get("matching_source_positions") != positions):
        raise ValueError("original row import source/next token receipt differs from its actual response")
    expected_files = {"inputs_embeds.bin": (next_row, "INT8", [1, 1, 2048]),
        "embed_scale.bin": (next_scale, "FLOAT32", [1, 1, 1]), "input_ids.bin": (struct.pack("<i", next_id), "INT32", [1, 1])}
    files = report.get("next_token_inputs")
    if not isinstance(files, list) or len(files) != 3 or {Path(row["path"]).name for row in files} != set(expected_files):
        raise ValueError("original row import must bind exactly its three next-token files")
    for row in files:
        path = directory / "next-token" / Path(row["path"]).name
        wanted, dtype, shape = expected_files[path.name]
        _payload(path, payloads)
        actual, identity = rows._read(path, len(wanted))
        if (row.get("path") != str(path) or row.get("dtype") != dtype or row.get("shape") != shape
                or row.get("byte_order") != "little" or identity != {key: row[key] for key in ("path", "bytes", "sha256", "stat")}
                or actual != wanted):
            raise ValueError("imported original next-token bytes changed or lost their typed source binding")
        _payload(path, payloads, identity["stat"])
    _stable(metadata, payloads)
    return {"report": report, "identity": report_id, "audit": audit, "next_id": next_id, "query_id": query,
        "embedding": next_row, "scale": next_scale, "inspection": manifest["inspection"]}


def _sources(state_lineage, original_rows, metadata, payloads, observations):
    state_lineage, _ = _safe(state_lineage, directory=True)
    saved, saved_id = _meta(state_lineage / "report.json", metadata, 4 << 20)
    # Establish current outer baselines before either nested proof can replay
    # historical ctimes. These are not historical acceptance checks.
    base, _ = _meta(Path(saved["execution"]) / "plan.json", metadata)
    for identity in [*saved.get("source_bindings", []), saved["decode"]["air"], saved["decode"]["model"]]:
        _payload(identity["path"], payloads)
    inventory, _ = _meta(saved["output_inspection"]["path"], metadata)
    board = Path(saved["execution"]) / "board"
    for output in inventory.get("output_files", []):
        _payload(_relative(board, output["path"]), payloads)
    for filename in ("tool.stdout", "tool.stderr", "tool.exitcode", "tool.pid"):
        _payload(board / filename, payloads)
    for source in base.get("source_inputs", []):
        _payload(source["path"], payloads)
    for state in saved.get("decode", {}).get("state_file_mapping", {}).get("states", []):
        _payload(state["output_file"]["path"], payloads)
    current = validate_target_decode_state_lineage(state_lineage, **observations)
    if (saved.get("status") != "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED"
            or current.get("status") != "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED"
            or current.get("same_model_state_mapping") != "ALL80_SAME_MODEL_STATE_FILES_BOUND"):
        raise ValueError("all80 same-model source/compiled/runtime/file state bindings are required")
    mapping = current["decode"]["state_file_mapping"]["states"]
    if (len(mapping) != 80 or {row.get("source_refdata") for row in mapping} != _STATES
            or len({row.get("runtime_input_index") for row in mapping}) != 80
            or len({row.get("runtime_output_index") for row in mapping}) != 80
            or any(row.get("status") != "SAME_MODEL_STATE_FILE_BOUND" for row in mapping)):
        raise ValueError("continuation requires80 independent proven state owner mappings")
    original = _bound_rows(original_rows, metadata, payloads, observations)
    if original["audit"]["execution"] != current["execution"] or original["inspection"] != str(Path(current["output_inspection"]["path"]).parent):
        raise ValueError("state lineage and original next-token rows must come from the identical actual decode execution")
    _meta(Path(current["execution"]) / "plan.json", metadata)
    base = validate_decode_smoke_plan(base, **observations)
    if (base.get("profile") != "cpro-40block-decode-1" or base.get("gear") != 1
            or len(base.get("inputs", [])) != 89 or len(base.get("outputs", [])) != 82
            or base["model_sha256"] != current["decode"]["model_sha256"]):
        raise ValueError("continuation must retain the identical complete40-layer static1 model and ABI")
    by_index = {row["index"]: row for row in base["inputs"]}
    for state in mapping:
        target = by_index[state["runtime_input_index"]]
        source = base["outputs"][state["runtime_output_index"]]
        if (target["name"] != state["source_refdata"] or target["runtime_tensor_name"] != state["runtime_input_name"]
                or source["runtime_tensor_name"] != state["runtime_output_name"]
                or target["bytes"] != state["output_file"]["bytes"]
                or _nd_view_key((target["dtype"], tuple(target["shape"]))) != _nd_view_key((source["dtype"], tuple(source["shape"])))):
            raise ValueError("proven parent state mapping differs from the identical named runtime ABI")
        _payload(state["output_file"]["path"], payloads, state["output_file"]["stat"])
    controls = {}
    for name, (semantic, dtype, size) in _CONTROL.items():
        raw, identity = _input(base, Path(base["input_dir"]), name, semantic, dtype, size,
            payload_revalidation_observations=observations["payload_revalidation_observations"])
        controls[name] = raw
        _payload(identity["path"], payloads, identity["stat"])
    _stable(metadata, payloads)
    return current, saved_id, original, base, controls


def _advance(controls):
    position, = struct.unpack("<i", controls["arg107_1"])
    cache, = struct.unpack("<i", controls["arg2_1"])
    if position != cache or not 0 <= position < 2047:
        raise ValueError("experimental source position_ids/new_kv_cache_pos must agree and leave one cache position")
    expected = {"arg1238_1": struct.pack("<i", 0), "arg114_1": struct.pack("<q", 1),
        "arg18_1": struct.pack("<h", 1), "arg14_1": struct.pack("<128i", 1, *([0] * 127))}
    if any(controls[key] != value for key, value in expected.items()):
        raise ValueError("source local-query controls differ from the explicit original single-query experiment")
    mask = controls["arg113_1"]
    if mask != bytes(4 * (position + 1)) + struct.pack("<f", float("-inf")) * (2048 - position - 1):
        raise ValueError("source mask must contain exactly the contiguous absolute-position causal prefix")
    changed = dict(controls)
    changed["arg107_1"] = changed["arg2_1"] = struct.pack("<i", position + 1)
    offset = 4 * (position + 1)
    changed["arg113_1"] = mask[:offset] + bytes(4) + mask[offset + 4:]
    return changed, {"id": ASSUMPTION, "contract_status": "EXPERIMENTAL_NOT_A_VALIDATED_KERNEL_RECIPE",
        "source_absolute_position": position, "next_absolute_position": position + 1,
        "source_visible_mask_columns": position + 1, "next_visible_mask_columns": position + 2,
        "changed_mask_column": position + 1, "local_query_count": 1,
        "carried_controls": {"output_pos": [0], "allQLen": [1], "effective_length": [1], "conv_state_pos": [1] + [0] * 127},
        "logical_history": "UNVERIFIED_USER_SUPPLIED_INITIAL_STATE", "numerical_reference": "NONE"}


def _copy_state(state, destination):
    expected = state["output_file"]
    path, before = _safe(expected["path"])
    if before != expected["stat"] or before[2] != expected["bytes"]:
        raise ValueError("bound parent state changed before independent continuation copy")
    full = _identity(path.stat())
    digest, count = hashlib.sha256(), 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source, destination.open("xb") as target:
        if _identity(os.fstat(source.fileno())) != full:
            raise ValueError("bound parent state changed before copy read")
        while block := source.read(64 << 10):
            count += len(block); digest.update(block); target.write(block)
        if _identity(os.fstat(source.fileno())) != full or _identity(path.stat()) != full:
            raise ValueError("parent state changed during independent continuation copy")
    if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("copied parent state differs from its freshly bound SHA256")
    result = _source(destination, count)
    if result["sha256"] != expected["sha256"] or result["identity"]["nlink"] != 1:
        raise ValueError("prepared state must be a byte-identical independent ordinary file")
    return result


def _plan(base, directory, sources, mapping, experimental, original, parent):
    plan = deepcopy(base)
    inputs, feedback = [], []
    by_name = {row["source_refdata"]: row for row in mapping}
    for old in base["inputs"]:
        name = old["name"]
        source = sources[name]
        row = dict(old, source=source["path"], source_filename=Path(source["path"]).name,
            sha256=source["sha256"], source_semantics="EXPERIMENTAL_SAME_MODEL_CONTINUATION")
        inputs.append(row)
        if name in by_name:
            state = by_name[name]
            feedback.append({key: state[key] for key in ("source_refdata", "runtime_input_index", "runtime_output_index", "output_file")})
            feedback[-1]["prepared_input_identity"] = source
    plan.update(artifact_kind=_PLAN_KIND, status="DECODE_CONTINUATION_PLANNED", input_dir=str(directory), inputs=inputs,
        source_inputs=sorted(sources.values(), key=lambda row: row["path"]), state_feedback_bindings=feedback,
        execution_scope=_SCOPE, source_semantics="EXPERIMENTAL_SAME_MODEL_CONTINUATION",
        state_feedback="PREPARED_ALL80_INDIVIDUAL_STATES_NOT_EXECUTED", state_continuity="NOT_VALIDATED",
        numerical_validation="NOT_RUN_NO_REFERENCE", continuation_parent_execution=parent["execution"],
        continuation_token_id=original["next_id"], source_query_token_id=original["query_id"],
        experimental_input_assumption=experimental, unique_source_input_bytes=sum(row["bytes"] for row in sources.values()),
        logical_input_bytes=sum(row["bytes"] for row in inputs), model_compilation="NOT_RUN", device_execution="NOT_RUN",
        state_input_layer_identity="ALL80_SAME_MODEL_PARENT_OUTPUT_BYTES_BOUND", state_output_layer_identity="ALL80_SAME_MODEL_STRUCTURAL_OWNERS_BOUND",
        host_model_graph_access="SMALL_BOUND_AIR_AND_PRIMARY_OMC_GRAPH_ONLY", weight_payload_access="NONE",
        scope="One original complete40-layer same-model decode step with80 separate feedback files and exact original greedy-token embedding; control/history/accuracy semantics remain unverified")
    plan["control_observations"] = experimental
    return plan


def prepare_decode_continuation(state_lineage, original_rows, output, *, assumption):
    out = output_path(output); out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "artifact_kind": _KIND, "status": "FAILED", "output": str(out),
        "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN", "numerical_validation": "NOT_RUN_NO_REFERENCE",
        "state_continuity": "NOT_VALIDATED", "execution_scope": _SCOPE}
    metadata, payloads, observations = {}, {}, _collectors()
    try:
        if assumption != ASSUMPTION:
            raise ValueError("the explicit single-query absolute-position continuation assumption is required")
        parent, lineage_id, original, base, controls = _sources(state_lineage, original_rows, metadata, payloads, observations)
        changed, experimental = _advance(controls)
        directory = out / "inputs"; directory.mkdir()
        sources = {}
        mapping = parent["decode"]["state_file_mapping"]["states"]
        for state in mapping:
            name = state["source_refdata"]
            sources[name] = _copy_state(state, directory / ("state_" + name + ".bin"))
        for name, raw in {"arg1_1": original["embedding"], "arg0_1": original["scale"], **changed}.items():
            semantic = "inputs_embeds" if name == "arg1_1" else "embed_scale" if name == "arg0_1" else _CONTROL[name][0]
            path = directory / (semantic + ".bin")
            with path.open("xb") as stream: stream.write(raw)
            sources[name] = _source(path, len(raw))
        if len(sources) != 89 or len({(row["identity"]["device"], row["identity"]["inode"]) for row in sources.values()}) != 89:
            raise ValueError("continuation requires89 distinct ordinary prepared input files")
        for source in sources.values():
            _payload(source["path"], payloads, source["identity"])
        plan = _plan(base, directory, sources, mapping, experimental, original, parent)
        _stable(metadata, payloads)
        save_json(out / "plan.json", plan)
        _, plan_identity = _meta(out / "plan.json", metadata)
        _stable(metadata, payloads)
        manifest.update(status=_READY, state_lineage=str(Path(state_lineage).absolute()), original_rows=str(Path(original_rows).absolute()),
            lineage_receipt=lineage_id, original_rows_receipt=original["identity"], assumption=ASSUMPTION,
            state_feedback_bindings=plan["state_feedback_bindings"], experimental_input_assumption=experimental,
            source_metadata_current=list(metadata.values()), source_payload_current=payloads,
            plan=plan_identity, input_count=89, output_count=82,
            continuation_token_id=original["next_id"], next_absolute_position=experimental["next_absolute_position"],
            independent_state_file_count=80, explicit_weight_copy_bytes=0)
        _stable(metadata, payloads)
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, struct.error) as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    manifest.update(observations)
    # The exclusive descriptor stays owned by this invocation until the final
    # preservation check. A failed candidate is finalized as FAILED on that same
    # descriptor; no previously completed receipt is reopened or overwritten.
    with (out / "manifest.json").open("x+") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush()
        if manifest["status"] == _READY:
            try:
                _stable(metadata, payloads)
            except (OSError, ValueError, TypeError, KeyError) as error:
                manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                stream.seek(0)
                json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n"); stream.truncate(); stream.flush()
    return manifest


def validate_decode_continuation(prepared):
    prepared, _ = _safe(prepared, directory=True)
    metadata, payloads, observations = {}, {}, _collectors()
    manifest, _ = _meta(prepared / "manifest.json", metadata)
    plan, plan_id = _meta(prepared / "plan.json", metadata)
    if (manifest.get("artifact_kind") != _KIND or manifest.get("status") != _READY or manifest.get("output") != str(prepared)
            or manifest.get("assumption") != ASSUMPTION or plan.get("artifact_kind") != _PLAN_KIND
            or plan_id != manifest.get("plan")
            or not _same(manifest.get("state_feedback_bindings"), plan.get("state_feedback_bindings"))):
        raise ValueError("expected an unchanged successful same-model continuation preparation")
    parent, lineage_id, original, base, controls = _sources(manifest["state_lineage"], manifest["original_rows"], metadata, payloads, observations)
    for old, fresh in ((manifest["lineage_receipt"], lineage_id), (manifest["original_rows_receipt"], original["identity"])):
        if not metadata_identity_matches(old, fresh, role="decode-continuation-source-metadata", observations=observations["metadata_revalidation_observations"]):
            raise ValueError("continuation parent metadata changed beyond freshly verified historical ctime")
    changed, experimental = _advance(controls)
    mapping = parent["decode"]["state_file_mapping"]["states"]
    by_state = {row["source_refdata"]: row for row in mapping}
    old_feedback = {row["source_refdata"]: row for row in manifest["state_feedback_bindings"]}
    for name, row in by_state.items():
        old = old_feedback[name]
        if any(old[key] != row[key] for key in ("runtime_input_index", "runtime_output_index")):
            raise ValueError("continuation state owner/slot mapping changed")
        if not payload_identity_matches(old["output_file"], row["output_file"], role="decode-output", observations=observations["payload_revalidation_observations"]):
            raise ValueError("bound parent state bytes or nonctime identity changed")
    # Bind read extents and all ABI fields before any prepared input hash.
    abi_excluded = {"source", "source_filename", "sha256", "source_semantics"}
    if (not isinstance(plan.get("inputs"), list) or len(plan["inputs"]) != 89
            or not _same(plan.get("outputs"), base["outputs"])
            or not _same([{key: value for key, value in row.items() if key not in abi_excluded} for row in plan["inputs"]],
                [{key: value for key, value in row.items() if key not in abi_excluded} for row in base["inputs"]])):
        raise ValueError("saved continuation ABI/order/read extents differ from the fresh complete model proof")
    directory = prepared / "inputs"
    expected_bytes = {"arg1_1": original["embedding"], "arg0_1": original["scale"], **changed}
    sources = {}
    saved_sources = {row["path"]: row for row in plan["source_inputs"]}
    for row in plan["inputs"]:
        name = row["name"]
        filename = "state_" + name + ".bin" if name in _STATES else (
            "inputs_embeds.bin" if name == "arg1_1" else "embed_scale.bin" if name == "arg0_1" else _CONTROL[name][0] + ".bin")
        path = directory / filename
        if row["source"] != str(path) or row["source_filename"] != filename:
            raise ValueError("prepared input path/basename differs from its distinct owner")
        _payload(path, payloads)
        current = _source(path, row["bytes"])
        if current != saved_sources.get(str(path)) or current["identity"]["nlink"] != 1:
            raise ValueError("prepared continuation input changed or is no longer an independent ordinary file")
        expected_sha = by_state[name]["output_file"]["sha256"] if name in _STATES else hashlib.sha256(expected_bytes[name]).hexdigest()
        if current["sha256"] != expected_sha:
            raise ValueError("prepared continuation bytes differ from the freshly bound parent/token/control source")
        sources[name] = current
        _payload(path, payloads, current["identity"])
    if (len(sources) != 89 or len({(row["identity"]["device"], row["identity"]["inode"]) for row in sources.values()}) != 89
            or {path.name for path in directory.iterdir()} != {Path(row["path"]).name for row in sources.values()}):
        raise ValueError("prepared inputs must be exactly89 independent named ordinary files")
    expected = _plan(base, directory, sources, mapping, experimental, original, parent)
    compared = deepcopy(expected)
    # Only purpose-scoped, freshly hashed historical ctimes can be projected.
    # Keep every other saved annotation, path, SHA, owner, mode and slot exact.
    if metadata_sequence_matches(plan.get("metadata_bindings"), expected.get("metadata_bindings"),
            role="decode-plan-metadata", observations=observations["metadata_revalidation_observations"]):
        compared["metadata_bindings"] = deepcopy(plan["metadata_bindings"])
    if metadata_identity_matches(plan.get("tool_binding", {}).get("inspection"),
            expected.get("tool_binding", {}).get("inspection"), role="runtime-probe-source-metadata",
            observations=observations["metadata_revalidation_observations"]):
        compared["tool_binding"]["inspection"] = deepcopy(plan["tool_binding"]["inspection"])
    old_primary, fresh_primary = plan["primary_model_binding"], expected["primary_model_binding"]
    if (any(row.get("status") == "SHA256_VERIFIED" and row.get("path") == fresh_primary["path"]
            for row in observations["revalidation_observations"])
            and fresh_primary.get("scope") == _PRIMARY_REVALIDATED_SCOPE
            and old_primary.get("scope") in {_PRIMARY_STAT_SCOPE, _PRIMARY_REVALIDATED_SCOPE}
            and plan.get("model_payload_access") in {"NONE", _PRIMARY_REVALIDATED_ACCESS}):
        compared["primary_model_binding"]["identity"]["ctime_ns"] = old_primary["identity"]["ctime_ns"]
        compared["primary_model_binding"]["scope"] = old_primary["scope"]
        compared["model_payload_access"] = plan["model_payload_access"]
    for fresh in compared["state_feedback_bindings"]:
        fresh["output_file"]["stat"] = deepcopy(old_feedback[fresh["source_refdata"]]["output_file"]["stat"])
    if not _same(compared, plan) or not _same(manifest["experimental_input_assumption"], experimental):
        raise ValueError("continuation plan differs from its fresh complete model/state/token/control proof")
    _stable(metadata, payloads)
    expected["continuation_validation_observations"] = observations
    expected["continuation_prepared"] = str(prepared)
    return expected

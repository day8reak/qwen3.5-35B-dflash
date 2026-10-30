"""Bind retained independent decode1 output files to explicitly logged slots.

Only saved metadata, decode input provenance and retrieved outputs are read.
No model/weight payload, device call, state feedback or numerical reference is
used. A successful inventory is not a decode continuation or accuracy pass.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

from .artifacts import output_path, save_json
from .original_target_board import _board_path
from .target_compile_failure import _read_small, _relative, _safe
from .target_decode_smoke_plan import validate_decode_smoke_plan
from .target_decode_trial_inputs import validate_decode_trial_inputs
from .target_prefill_output_inventory import (
    _LOGS, _MAX_METADATA, _MAX_TOTAL, _WIDTHS, _hash_retrieved, _json,
    _log_display, _logged_slots, _numerical,
)


_KIND = "qwen35-target-decode-output-inventory"
_SCOPE = "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_NO_PREFILL_FEEDBACK"
_PRESERVED = "UNCHANGED_DECODE_PLAN_METADATA_AND_USER_INPUT_SHA256"


def _bound_execution(execution, bindings):
    outer = _json(execution / "result.json", bindings)
    board = _json(execution / "board/result.json", bindings)
    if (outer.get("artifact_kind") != "qwen35-target-decode-smoke-job" or outer.get("mode") != "APPLY"
            or board.get("artifact_kind") != "qwen35-original-target-tool-execution"
            or outer.get("board_result") != str(execution / "board/result.json")
            or outer.get("plan") != str(execution / "plan.json")):
        raise ValueError("expected the independent decode1 APPLY execution and its exact board receipt")
    for report in (outer, board):
        if (report.get("status") != "EXECUTED_UNVALIDATED"
                or report.get("device_execution") != "EXECUTED_OUTPUTS_RETRIEVED"
                or type(report.get("remote_exitcode")) is not int or report["remote_exitcode"] != 0
                or type(report.get("tool_process_id")) is not int or report["tool_process_id"] <= 0
                or report.get("execution_scope") != _SCOPE or report.get("retrieval_errors") != []
                or report.get("lock_state") != "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
                or report.get("source_preservation") != _PRESERVED
                or any(report.get(key) for key in ("error", "evidence_error", "source_error", "lock_release_error", "trial_input_error"))):
            raise ValueError("successful decode1 exit, retrieval, source audit and released locks are required")
    for key in ("status", "device_execution", "remote_exitcode", "tool_process_id", "execution_scope",
                "retrieved", "retrieval_errors", "lock_state", "source_preservation", "runtime_decode_probe"):
        if outer.get(key) != board.get(key):
            raise ValueError("outer and board decode execution receipts differ: " + key)
    plan = _json(execution / "board/plan.json", bindings)
    if bindings[-1]["sha256"] != board.get("plan_sha256"):
        raise ValueError("board plan SHA256 differs from the successful execution receipt")
    proof = _json(execution / "board/runtime-decode-proof.json", bindings)
    if proof != _json(execution / "plan.json", bindings):
        raise ValueError("original decode plan differs from the saved runtime proof")
    if (plan.get("artifact_kind") != "qwen35-original-decode-smoke-plan"
            or plan.get("status") != "DECODE_SMOKE_PLANNED" or plan.get("profile") != "cpro-40block-decode-1"
            or type(plan.get("gear")) is not int or plan["gear"] != 1 or board.get("gear") != 1
            or plan.get("execution_scope") != _SCOPE
            or plan.get("runtime_probe") != board.get("runtime_decode_probe")
            or not isinstance(plan.get("inputs"), list) or len(plan["inputs"]) != 89
            or not isinstance(plan.get("outputs"), list) or len(plan["outputs"]) != 82):
        raise ValueError("saved plan must describe the frozen independent static1 decode scope")
    for key in proof:
        if key != "inputs" and plan.get(key) != proof[key]:
            raise ValueError("board plan differs from its saved decode proof: " + key)
    attempt = _board_path(board.get("board_attempt", ""))
    if not isinstance(proof.get("inputs"), list) or len(proof["inputs"]) != 89:
        raise ValueError("saved decode proof does not contain 89 ordered inputs")
    for row, original in zip(plan["inputs"], proof["inputs"]):
        if (not isinstance(original, dict) or not isinstance(original.get("source_filename"), str)
                or row != dict(original, host_source=original["source"],
                    source=str(attempt / "source-inputs" / original["source_filename"]))):
            raise ValueError("board input snapshot differs from its saved decode proof")
    command = _json(execution / "board/command.json", bindings)
    argv = command.get("argv")
    if (not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or argv[0] != plan.get("tool_binding", {}).get("path")):
        raise ValueError("saved decode command does not bind the planned binary")
    flags = {}
    for arg in argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError("unexpected saved tool argument")
        name, value = arg[2:].split("=", 1)
        if name in flags:
            raise ValueError("repeated saved tool argument")
        flags[name] = value
    if (flags.get("modelmanager_itf") != "V2" or flags.get("direct_from_file") != "true"
            or flags.get("times") != "1" or any(key in flags for key in ("input_format", "output_format", "input_shape", "output_shape"))
            or flags.get("output_dir") != str(attempt / "outputs")
            or flags.get("input") != ",".join(str(attempt / row["file"]) for row in plan["inputs"])
            or flags.get("input_type") != ",".join(row["tool_input_type"] for row in plan["inputs"])
            or flags.get("output_type") != ",".join(row["tool_output_type"] for row in plan["outputs"])):
        raise ValueError("saved command is not the observed V2/model-format-default decode1 invocation")
    models = [row for row in board.get("model_payloads", []) if isinstance(row, dict)
              and row.get("path") == flags.get("model") and row.get("sha256") == plan.get("model_sha256")]
    if len(models) != 1 or flags.get("weight_dir") != str(Path(flags.get("model", "")).parent):
        raise ValueError("saved decode command model/weight directory is not bound to the recorded compiled payload")
    validate_decode_smoke_plan(proof)
    return outer, board, plan, proof


def _view(shape):
    shape = tuple(shape)
    while shape and shape[0] == 1:
        shape = shape[1:]
    while shape and shape[-1] == 1:
        shape = shape[:-1]
    return shape


def _descriptors(plan):
    result = []
    for index, row in enumerate(plan["outputs"]):
        if not isinstance(row, dict):
            raise ValueError("saved decode output descriptor must be an object")
        shape, dtype = row.get("shape"), row.get("dtype")
        if (type(row.get("index")) is not int or row["index"] != index or dtype not in _WIDTHS
                or not isinstance(shape, list) or not 1 <= len(shape) <= 8
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or type(row.get("bytes")) is not int or row["bytes"] != math.prod(shape) * _WIDTHS[dtype]
                or row.get("tool_output_type") != {"FLOAT16": "FP16", "FLOAT32": "FP32"}[dtype]
                or row.get("nd_view_policy") != "EDGE_SINGLETON_ND_LINEAR_VIEW_ONLY"):
            raise ValueError("saved decode output descriptor is malformed or not a fixed ND view")
        result.append({key: row[key] for key in ("index", "shape", "dtype", "bytes")})
    if Counter((row["dtype"], _view(row["shape"])) for row in result) != Counter({
            ("FLOAT32", (248320,)): 1, ("FLOAT16", (8192, 4)): 30,
            ("FLOAT16", (32, 128, 128)): 30, ("FLOAT16", (64, 32, 32, 16)): 20,
            ("FLOAT16", (16384,)): 1}):
        raise ValueError("saved descriptor multiset differs from the complete original decode1 target")
    for key, dtype, shape in (("feature_runtime_output_index", "FLOAT16", (16384,)),
                              ("logits_runtime_output_index", "FLOAT32", (248320,))):
        index = plan.get(key)
        if (type(index) is not int or not 0 <= index < 82 or result[index]["dtype"] != dtype
                or _view(result[index]["shape"]) != shape):
            raise ValueError("saved decode feature/logits index differs from its unique typed descriptor")
    return result


def _trial(outer, plan, bindings):
    receipt = outer.get("trial_input_receipt")
    if receipt is None:
        if any(key in outer for key in ("experimental_input_assumption", "trial_input_preservation")):
            raise ValueError("experimental decode assumptions require their explicit trial receipt")
        return {"status": "NOT_RECORDED", "source_semantics": "UNVERIFIED_USER_INPUTS"}
    if (not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str)
            or Path(receipt["path"]).name != "manifest.json"
            or outer.get("trial_input_preservation") != "UNCHANGED_ORIGINALS_DERIVED_INPUTS_AND_EXPLICIT_ASSUMPTION"):
        raise ValueError("decode trial receipt or preservation binding is incomplete")
    _json(receipt["path"], bindings)
    if bindings[-1] != receipt:
        raise ValueError("decode trial manifest identity differs from the execution receipt")
    trial = validate_decode_trial_inputs(Path(receipt["path"]).parent)
    if trial["input_dir"] != plan.get("input_dir") or trial["assumption"] != outer.get("experimental_input_assumption"):
        raise ValueError("decode trial directory or explicit assumptions differ from the execution")
    return {"status": "FRESHLY_REVALIDATED_UNVERIFIED_EXPERIMENT", "manifest": receipt,
            "assumption": trial["assumption"], "trial_input_preservation": outer["trial_input_preservation"],
            "source_file_count": len(trial["source_bindings"]), "copied_input_count": len(trial["copied_inputs"]),
            "generated_control_count": len(trial["generated_controls"]),
            "scope": "Originals, derived inputs and explicit hypothesis freshly rehashed; kernel meanings and continuation remain unverified"}


def _states(execution, indexed, observations, descriptors, excluded):
    counts = {key: 0 for key in ("elements", "finite_count", "nan_count", "positive_infinity_count", "negative_infinity_count", "zero_count")}
    entries, low, high, maximum = [], None, None, None
    for desc in descriptors:
        index = desc["index"]
        if index in excluded:
            continue
        name = indexed[index]["path"]
        stats = _numerical(_relative(execution / "board", name), desc, observations[name])
        for key in counts:
            counts[key] += stats[key]
        if stats["finite_min"] is not None:
            low = stats["finite_min"] if low is None else min(low, stats["finite_min"])
            high = stats["finite_max"] if high is None else max(high, stats["finite_max"])
            maximum = stats["finite_max_abs"] if maximum is None else max(maximum, stats["finite_max_abs"])
        entries.append({"runtime_index": index, "path": name, "statistics": stats})
    return {"status": "DESCRIBED_UNVALIDATED", "tensor_count": len(entries), "dtype": "FLOAT16", **counts,
            "nonfinite_count": counts["elements"] - counts["finite_count"], "finite_min": low,
            "finite_max": high, "finite_max_abs": maximum, "entries": entries,
            "state_layer_binding": "UNVERIFIED", "state_feedback": "NOT_PERFORMED",
            "scope": "Typed statistics of 80 explicitly logged runtime state slots; no layer ownership, source history, accuracy or continuity proof"}


def inspect_target_decode_outputs(execution, output, *, include_state_statistics=True):
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": _KIND, "status": "FAILED", "output": str(out),
              "model_execution": "NOT_RUN", "device_calls": "NONE", "weight_payload_access": "NONE",
              "model_payload_access": "NONE", "slot_binding": "UNVERIFIED", "feature_selection": "NOT_PERFORMED",
              "state_layer_binding": "UNVERIFIED", "state_feedback": "NOT_PERFORMED",
              "state_continuity": "NOT_VALIDATED", "numerical_validation": "NOT_RUN_NO_REFERENCE"}
    try:
        if type(include_state_statistics) is not bool:
            raise ValueError("include_state_statistics must be an explicit boolean")
        execution, _ = _safe(execution, directory=True)
        bindings = []
        outer, board, plan, proof = _bound_execution(execution, bindings)
        trial = _trial(outer, plan, bindings)
        descriptors = _descriptors(plan)
        retrieved = board.get("retrieved")
        if not isinstance(retrieved, dict) or not 5 <= len(retrieved) <= 1028:
            raise ValueError("decode execution has no bounded complete retrieved file inventory")
        names = sorted(name for name in retrieved if isinstance(name, str) and name.startswith("outputs/"))
        if set(retrieved) != {*names, *_LOGS}:
            raise ValueError("unexpected or missing retrieved decode output/log records")
        if any(not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] < 0 for row in retrieved.values()):
            raise ValueError("retrieved decode records have invalid byte extents")
        if sum(row["bytes"] for row in retrieved.values()) > _MAX_TOTAL:
            raise ValueError("decode output/log inventory exceeds 512 MiB inspection budget")
        rows, observations, logs, raw_logs = [], {}, {}, {}
        for name in [*names, *_LOGS]:
            identity, raw = _hash_retrieved(_relative(execution / "board", name), retrieved[name], capture=name in _LOGS)
            observations[name] = identity
            if name in _LOGS:
                logs[name], raw_logs[name] = _log_display(raw, identity), raw
                if name == "tool.exitcode" and raw.strip() != b"0":
                    raise ValueError("retrieved decode exitcode differs from the successful execution")
                if name == "tool.pid" and raw.strip() != str(board["tool_process_id"]).encode():
                    raise ValueError("retrieved decode PID differs from the successful execution")
            else:
                candidates = [row["index"] for row in descriptors if row["bytes"] == identity["bytes"]]
                rows.append({"path": name, **identity, "candidate_runtime_indices_by_bytes_only": candidates,
                             "slot_binding": "UNVERIFIED"})
        mapping = _logged_slots(raw_logs["tool.stdout"], board, descriptors, rows)
        if mapping["status"] == "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG" and mapping["unlogged_retrieved_files"]:
            mapping.update(status="UNVERIFIED", reason="UNLOGGED_RETRIEVED_OUTPUT_FILES")
        tensors, state_stats = {}, {"status": "NOT_PERFORMED_SLOT_BINDING_UNVERIFIED"}
        if mapping["status"] == "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG":
            indexed = {row["runtime_index"]: row for row in mapping["entries"]}
            for label, index in (("feature", plan["feature_runtime_output_index"]), ("logits", plan["logits_runtime_output_index"])):
                name = indexed[index]["path"]
                tensors[label] = {"runtime_index": index, "path": name, **observations[name],
                    "statistics": _numerical(_relative(execution / "board", name), descriptors[index], observations[name])}
            state_stats = (_states(execution, indexed, observations, descriptors,
                {plan["feature_runtime_output_index"], plan["logits_runtime_output_index"]}) if include_state_statistics
                else {"status": "NOT_REQUESTED", "state_layer_binding": "UNVERIFIED"})
            by_path = {row["path"]: row for row in mapping["entries"]}
            for row in rows:
                row.update(runtime_index=by_path[row["path"]]["runtime_index"], slot_binding="EXPLICIT_TOOL_LOG_PATH_AND_DESCRIPTOR_EXTENT")
        validate_decode_smoke_plan(proof)
        if _trial(outer, plan, []) != trial:
            raise ValueError("decode trial provenance changed during output inspection")
        for identity in bindings:
            if _read_small(identity["path"], _MAX_METADATA)[1] != identity:
                raise ValueError("decode execution evidence changed during output inspection")
        for name, identity in observations.items():
            if _safe(_relative(execution / "board", name))[1] != identity["stat"]:
                raise ValueError("retrieved decode file changed after hash verification")
        expected, observed = Counter(row["bytes"] for row in descriptors), Counter(row["bytes"] for row in rows)
        result.update(status="DECODE_OUTPUT_FILES_INSPECTED_SLOT_BINDING_UNVERIFIED", execution=str(execution),
            execution_scope=_SCOPE, evidence_bindings=bindings, plan_output_count=82,
            actual_retrieved_output_file_count=len(rows), tool_process_id=board["tool_process_id"], remote_exitcode=0,
            source_validation="FRESH_DECODE_PLAN_METADATA_AND_USER_INPUT_SHA256_NO_MODEL_READ",
            expected_runtime_outputs=descriptors, output_files=rows,
            raw_extent_multiset_check="MATCHED_COUNTS_AND_EXTENTS_ONLY" if expected == observed else "DIFFERENT_COUNTS_OR_EXTENTS",
            raw_extent_counts=[{"bytes": size, "expected": expected[size], "observed": observed[size],
                "missing": max(0, expected[size] - observed[size]), "excess": max(0, observed[size] - expected[size])}
                for size in sorted(expected.keys() | observed.keys())],
            logged_slot_mapping=mapping, bound_tensors=tensors, state_statistics=state_stats,
            tool_logs=logs, trial_inputs=trial, source_semantics="UNVERIFIED_USER_INPUTS",
            scope="Fresh source/output hashes and saved decode evidence only. Runtime slot binding, control semantics, state lineage, continuation and numerical accuracy are separate claims; no model/weight reads or device calls.")
        if mapping["status"] == "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG":
            result.update(status="DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", slot_binding="EXPLICIT_COMPLETE_TOOL_LOG_AND_VERIFIED_BYTES",
                feature_selection="BOUND_FILE_DESCRIBED_ONLY_NO_COPY_OR_EXECUTION")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    save_json(out / "report.json", result)
    return result


def validate_target_decode_output_inventory(inspection):
    """Rebind a successful saved inventory without writing or repeating stats.

    This freshly verifies its source plan/trial, metadata, all 82 output hashes,
    logs and explicit index/path mapping. Descriptive numerical summaries are
    not an accuracy contract and are not recomputed by this read-only helper.
    """
    root, _ = _safe(inspection, directory=True)
    report_bindings = []
    report = _json(root / "report.json", report_bindings)
    if (report.get("artifact_kind") != _KIND or report.get("status") != "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
            or report.get("output") != str(root) or report.get("execution_scope") != _SCOPE
            or report.get("slot_binding") != "EXPLICIT_COMPLETE_TOOL_LOG_AND_VERIFIED_BYTES"
            or report.get("source_validation") != "FRESH_DECODE_PLAN_METADATA_AND_USER_INPUT_SHA256_NO_MODEL_READ"
            or report.get("plan_output_count") != 82 or report.get("actual_retrieved_output_file_count") != 82
            or report.get("state_layer_binding") != "UNVERIFIED" or report.get("state_feedback") != "NOT_PERFORMED"
            or report.get("state_continuity") != "NOT_VALIDATED"
            or report.get("numerical_validation") != "NOT_RUN_NO_REFERENCE"
            or report.get("model_payload_access") != "NONE" or report.get("weight_payload_access") != "NONE"
            or report.get("device_calls") != "NONE" or report.get("model_execution") != "NOT_RUN"
            or report.get("error")):
        raise ValueError("expected a successful untouched independent decode output inventory")
    execution, _ = _safe(report["execution"], directory=True)
    bindings = []
    outer, board, plan, proof = _bound_execution(execution, bindings)
    trial = _trial(outer, plan, bindings)
    if (trial != report.get("trial_inputs") or bindings != report.get("evidence_bindings")
            or report.get("tool_process_id") != board["tool_process_id"] or report.get("remote_exitcode") != 0):
        raise ValueError("decode inventory metadata or explicit trial evidence changed")
    descriptors = _descriptors(plan)
    if descriptors != report.get("expected_runtime_outputs"):
        raise ValueError("decode inventory descriptors differ from the current bound proof")
    rows = report.get("output_files")
    if (not isinstance(rows, list) or len(rows) != 82 or any(not isinstance(row, dict) for row in rows)
            or any(not isinstance(row.get("path"), str) or not row["path"].startswith("outputs/") for row in rows)
            or len({row["path"] for row in rows}) != 82):
        raise ValueError("decode inventory requires 82 unique output file bindings")
    retrieved = board.get("retrieved")
    if not isinstance(retrieved, dict) or set(retrieved) != {*[row["path"] for row in rows], *_LOGS}:
        raise ValueError("decode inventory differs from actual retrieved output/log records")
    observations = {}
    for row in rows:
        if any(row.get(key) != retrieved[row["path"]].get(key) for key in ("bytes", "sha256")):
            raise ValueError("decode inventory output extent/hash differs from its execution")
        identity, _ = _hash_retrieved(_relative(execution / "board", row["path"]), row)
        if identity != {key: row.get(key) for key in ("bytes", "sha256", "stat")}:
            raise ValueError("decode inventory output identity changed")
        observations[row["path"]] = identity
    raw_stdout = None
    for name in _LOGS:
        identity, raw = _hash_retrieved(_relative(execution / "board", name), retrieved[name], capture=True)
        observations[name] = identity
        if _log_display(raw, identity) != report.get("tool_logs", {}).get(name):
            raise ValueError("decode inventory retained tool log identity or display changed")
        if name == "tool.exitcode" and raw.strip() != b"0":
            raise ValueError("decode inventory tool exitcode differs from the successful execution")
        if name == "tool.pid" and raw.strip() != str(board["tool_process_id"]).encode():
            raise ValueError("decode inventory tool PID differs from the successful execution")
        if name == "tool.stdout":
            raw_stdout = raw
    mapping = _logged_slots(raw_stdout, board, descriptors, rows)
    if (mapping.get("status") != "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG" or mapping.get("unlogged_retrieved_files")
            or mapping != report.get("logged_slot_mapping")):
        raise ValueError("decode inventory explicit output slot mapping differs from its verified log")
    by_index = {row["runtime_index"]: row for row in mapping["entries"]}
    by_path = {row["path"]: row for row in rows}
    for index, entry in by_index.items():
        row = by_path[entry["path"]]
        if row.get("runtime_index") != index or row.get("slot_binding") != "EXPLICIT_TOOL_LOG_PATH_AND_DESCRIPTOR_EXTENT":
            raise ValueError("decode inventory output file runtime index changed")
    for label, index in (("feature", plan["feature_runtime_output_index"]), ("logits", plan["logits_runtime_output_index"])):
        bound = report.get("bound_tensors", {}).get(label, {})
        name = by_index[index]["path"]
        if (bound.get("runtime_index") != index or bound.get("path") != name
                or any(bound.get(key) != observations[name][key] for key in ("bytes", "sha256", "stat"))):
            raise ValueError("decode inventory feature/logits file binding changed")
    validate_decode_smoke_plan(proof)
    if _trial(outer, plan, []) != trial:
        raise ValueError("decode trial provenance changed during inventory validation")
    for identity in [*report_bindings, *bindings]:
        if _read_small(identity["path"], _MAX_METADATA)[1] != identity:
            raise ValueError("decode inventory evidence changed during validation")
    for name, identity in observations.items():
        if _safe(_relative(execution / "board", name))[1] != identity["stat"]:
            raise ValueError("decode output changed during inventory validation")
    return report

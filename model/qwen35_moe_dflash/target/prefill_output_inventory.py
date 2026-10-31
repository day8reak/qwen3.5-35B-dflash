"""Inspect retained first-prefill bytes and explicitly logged runtime slots.

Only saved execution metadata, retrieved output files/logs, an optional small
draft manifest and optional original input_ids are read. No HDC, model/weight
payloads, model execution, feature copies or state feedback is requested.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.target.original_target_board import _board_path
from qwen35_moe_dflash.common.file_io import _read_small, _relative, _safe, _stamp
from qwen35_moe_dflash.target.source_inventory import _reject_constant, _unique_object


_KIND = "qwen35-target-prefill-output-inventory"
_MAX_METADATA = 8 << 20
_MAX_FILE = 256 << 20
_MAX_TOTAL = 512 << 20
_MAX_DISPLAY = 64 << 10
_LOGS = ("tool.stdout", "tool.stderr", "tool.exitcode", "tool.pid")
_WIDTHS = {"FLOAT32": 4, "FLOAT16": 2}


def _json(path, bindings):
    raw, identity = _read_small(path, _MAX_METADATA)
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite metadata number")
        return value
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("output inspection metadata must be an object")
    bindings.append(identity)
    return value


def _hash_retrieved(path, row, *, capture=False):
    path, before = _safe(path)
    if (not isinstance(row, dict) or type(row.get("bytes")) is not int
            or not 0 <= row["bytes"] <= _MAX_FILE or row["bytes"] != before[2]
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))):
        raise ValueError("invalid retrieved file extent/hash: " + str(path))
    if capture and before[2] > _MAX_METADATA:
        raise ValueError("retrieved text log exceeds the bounded display/read budget")
    digest, chunks = hashlib.sha256(), []
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("retrieved file changed before read")
        while block := stream.read(1 << 20):
            digest.update(block)
            if capture:
                chunks.append(block)
        if _stamp(os.fstat(stream.fileno())) != before or _safe(path)[1] != before:
            raise ValueError("retrieved file changed during read")
    if digest.hexdigest() != row["sha256"]:
        raise ValueError("retrieved file SHA256 differs from the execution receipt: " + str(path))
    return {"bytes": before[2], "sha256": digest.hexdigest(), "stat": before}, b"".join(chunks)


def _bound_execution(execution, bindings):
    outer = _json(execution / "result.json", bindings)
    board = _json(execution / "board/result.json", bindings)
    if (outer.get("artifact_kind") != "qwen35-target-prefill-smoke-job" or outer.get("mode") != "APPLY"
            or board.get("artifact_kind") != "qwen35-original-target-tool-execution"
            or outer.get("board_result") != str(execution / "board/result.json")
            or outer.get("plan") != str(execution / "plan.json")):
        raise ValueError("expected the original first-prefill APPLY execution and its exact board receipt")
    for report in (outer, board):
        if (report.get("status") != "EXECUTED_UNVALIDATED"
                or report.get("device_execution") != "EXECUTED_OUTPUTS_RETRIEVED"
                or type(report.get("remote_exitcode")) is not int or report["remote_exitcode"] != 0
                or type(report.get("tool_process_id")) is not int or report["tool_process_id"] <= 0
                or report.get("execution_scope") != "SINGLE_PREFILL_ONLY_IDENTICAL_STATE_GROUPS"
                or report.get("retrieval_errors") != []
                or report.get("lock_state") != "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
                or report.get("source_preservation") != "UNCHANGED_PREFILL_PLAN_METADATA_AND_ORIGINAL_INPUT_SHA256"
                or any(report.get(key) for key in ("error", "evidence_error", "source_error", "lock_release_error"))):
            raise ValueError("successful first-prefill exit, retrieval, source audit and released locks are required")
    for key in ("status", "device_execution", "remote_exitcode", "tool_process_id", "execution_scope",
                "retrieved", "retrieval_errors", "lock_state", "source_preservation", "runtime_prefill_probe"):
        if outer.get(key) != board.get(key):
            raise ValueError("outer and board execution receipts differ: " + key)
    plan = _json(execution / "board/plan.json", bindings)
    if bindings[-1]["sha256"] != board.get("plan_sha256"):
        raise ValueError("board plan SHA256 differs from the successful execution receipt")
    proof = _json(execution / "board/runtime-prefill-proof.json", bindings)
    original_plan = _json(execution / "plan.json", bindings)
    if proof != original_plan:
        raise ValueError("original plan differs from the saved runtime proof")
    if (plan.get("artifact_kind") != "qwen35-original-prefill-smoke-plan"
            or plan.get("status") != "PREFILL_SMOKE_PLANNED" or plan.get("profile") != "cpro-40block-prefill-64"
            or type(plan.get("gear")) is not int or plan["gear"] != 64 or board.get("gear") != 64
            or plan.get("execution_scope") != "SINGLE_PREFILL_ONLY_IDENTICAL_STATE_GROUPS"
            or plan.get("runtime_probe") != board.get("runtime_prefill_probe")
            or not isinstance(plan.get("inputs"), list) or len(plan["inputs"]) != 89
            or not isinstance(plan.get("outputs"), list) or len(plan["outputs"]) != 82
            or any(not isinstance(row, dict) for row in plan["outputs"])):
        raise ValueError("saved plan must describe the frozen original static64 first-prefill scope")
    for key in ("artifact_kind", "status", "profile", "gear", "outputs", "compilation_report_sha256",
                "model_sha256", "runtime_version", "ddk_id", "firmware_id", "metadata_bindings",
                "primary_model_binding", "tool_binding", "logits_runtime_output_index", "feature_runtime_output_index"):
        if plan.get(key) != proof.get(key):
            raise ValueError("board plan differs from its saved first-prefill proof: " + key)
    command = _json(execution / "board/command.json", bindings)
    argv = command.get("argv")
    if (not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or argv[0] != plan.get("tool_binding", {}).get("path")):
        raise ValueError("saved tool command does not bind the planned binary")
    flags = {}
    for arg in argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError("unexpected saved tool argument")
        name, value = arg[2:].split("=", 1)
        if name in flags:
            raise ValueError("repeated saved tool argument")
        flags[name] = value
    if (flags.get("modelmanager_itf") != "V2" or flags.get("direct_from_file") != "true"
            or flags.get("times") != "1" or "input_format" in flags or "output_format" in flags
            or flags.get("output_dir") != str(Path(board.get("board_attempt", "")) / "outputs")
            or flags.get("output_type") != ",".join(row.get("tool_output_type", "") for row in plan.get("outputs", []))):
        raise ValueError("saved command is not the observed V2/model-format-default first-prefill invocation")
    model_rows = [row for row in board.get("model_payloads", []) if isinstance(row, dict)
                  and row.get("path") == flags.get("model") and row.get("sha256") == plan.get("model_sha256")]
    if len(model_rows) != 1 or flags.get("weight_dir") != str(Path(flags.get("model", "")).parent):
        raise ValueError("saved command model/weight directory is not bound to its recorded compiled payload")
    return outer, board, plan


def _descriptors(plan):
    rows = plan.get("outputs")
    if not isinstance(rows, list) or len(rows) != 82:
        raise ValueError("the saved runtime descriptor plan must contain exactly82outputs")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("saved runtime output descriptor must be an object")
        shape, dtype = row.get("shape"), row.get("dtype")
        if (not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index
                or dtype not in _WIDTHS or not isinstance(shape, list) or not 1 <= len(shape) <= 8
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or row.get("bytes") != math.prod(shape) * _WIDTHS[dtype]
                or row.get("tool_output_type") != {"FLOAT16": "FP16", "FLOAT32": "FP32"}[dtype]
                or row.get("nd_view_policy") != "EDGE_SINGLETON_ND_LINEAR_VIEW_ONLY"):
            raise ValueError("saved runtime output descriptor is malformed or not a fixed ND view")
        result.append({key: row[key] for key in ("index", "shape", "dtype", "bytes")})
    def view(shape):
        shape = tuple(shape)
        while shape and shape[0] == 1:
            shape = shape[1:]
        while shape and shape[-1] == 1:
            shape = shape[:-1]
        return shape
    if Counter((row["dtype"], view(row["shape"])) for row in result) != Counter({
            ("FLOAT32", (248320,)): 1, ("FLOAT16", (8192, 4)): 30,
            ("FLOAT16", (32, 128, 128)): 30, ("FLOAT16", (64, 32, 32, 16)): 20,
            ("FLOAT16", (64, 16384)): 1}):
        raise ValueError("saved descriptor multiset differs from the complete original prefill64 target")
    for key, dtype, shape in (("feature_runtime_output_index", "FLOAT16", (64, 16384)),
                              ("logits_runtime_output_index", "FLOAT32", (248320,))):
        index = plan.get(key)
        if (type(index) is not int or not 0 <= index < 82 or result[index]["dtype"] != dtype
                or view(result[index]["shape"]) != shape):
            raise ValueError("saved feature/logits runtime index differs from its unique typed descriptor")
    return result


def _log_display(raw, identity):
    selected, used, filtered, omitted = [], 0, 0, 0
    for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        if line.strip() == "[WARNING] [debug]: from img skip realpath":
            filtered += 1
            continue
        encoded = line.encode("utf-8")
        if used + len(encoded) + 1 > _MAX_DISPLAY or len(selected) >= 1024:
            omitted += 1
            continue
        selected.append({"line": number, "text": line})
        used += len(encoded) + 1
    return {**identity, "display_lines": selected, "filtered_skip_realpath_lines": filtered,
            "omitted_display_lines": omitted, "display_truncated": bool(omitted),
            "scope": "Raw file SHA verified; only exact repetitive skip-realpath lines filtered for display"}


def _logged_slots(raw, board, descriptors, files):
    """Use only explicit observed tool output[index]:absolute/path statements."""
    result = {"status": "UNVERIFIED", "entries": [],
              "scope": "No numeric filename or equal-byte-size convention is inferred"}
    by_file = {row["path"]: row for row in files}
    seen_indices, seen_paths = set(), set()
    root = _board_path(board["board_attempt"]) / "outputs"
    for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        if not re.match(r"^\s*output\[", line):
            continue
        match = re.fullmatch(r"\s*output\[(0|[1-9][0-9]*)\]\s*:\s*(/[^\r\n]+?)\s*", line)
        if match is None:
            result["reason"] = "MALFORMED_OUTPUT_LOG_STATEMENT"
            return result
        index = int(match[1])
        try:
            remote = _board_path(match[2])
            relative = remote.relative_to(root)
            if not relative.parts:
                raise ValueError("output log names a directory")
            name = "outputs/" + relative.as_posix()
        except ValueError:
            result["reason"] = "OUTPUT_LOG_PATH_OUTSIDE_THIS_ATTEMPT"
            return result
        if index in seen_indices or name in seen_paths:
            result["reason"] = "DUPLICATE_OUTPUT_LOG_INDEX_OR_PATH"
            return result
        if not 0 <= index < len(descriptors) or name not in by_file:
            result["reason"] = "OUTPUT_LOG_INDEX_OR_FILE_NOT_IN_BOUND_INVENTORY"
            return result
        if by_file[name]["bytes"] != descriptors[index]["bytes"]:
            result["reason"] = "LOGGED_SLOT_EXTENT_DIFFERS_FROM_DESCRIPTOR"
            return result
        seen_indices.add(index)
        seen_paths.add(name)
        result["entries"].append({"runtime_index": index, "path": name,
                                  "logged_absolute_path": str(remote), "log_line": number})
    if seen_indices != set(range(len(descriptors))):
        result["reason"] = "EXPLICIT_COMPLETE_OUTPUT_LOG_MAPPING_NOT_OBSERVED"
        return result
    result.update(status="BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG", entries=sorted(result["entries"], key=lambda row: row["runtime_index"]),
                  unlogged_retrieved_files=sorted(set(by_file) - seen_paths),
                  scope="Each runtime index is explicitly paired with this attempt's absolute file path in SHA-verified raw tool stdout; file SHA and descriptor byte extent verified. This binds output slots, not state layers or numerical correctness.")
    return result


def _numerical(path, desc, expected):
    """Describe one bound tensor, never treat finite values as accuracy proof."""
    path, before = _safe(path)
    if before != expected["stat"]:
        raise ValueError("bound tensor changed before numerical inspection")
    code, width = ("<e", 2) if desc["dtype"] == "FLOAT16" else ("<f", 4)
    counts = {"elements": 0, "finite_count": 0, "nan_count": 0, "positive_infinity_count": 0,
              "negative_infinity_count": 0, "zero_count": 0}
    digest, maximum, minimum, max_abs = hashlib.sha256(), None, None, None
    top_index, top_value = None, None
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("bound tensor changed before numerical read")
        while block := stream.read(1 << 20):
            if len(block) % width:
                raise ValueError("bound tensor byte extent is not divisible by its actual dtype")
            digest.update(block)
            for (value,) in struct.iter_unpack(code, block):
                index = counts["elements"]
                counts["elements"] += 1
                if math.isnan(value):
                    counts["nan_count"] += 1
                elif math.isinf(value):
                    counts["positive_infinity_count" if value > 0 else "negative_infinity_count"] += 1
                else:
                    counts["finite_count"] += 1
                    counts["zero_count"] += int(value == 0)
                    maximum = value if maximum is None else max(maximum, value)
                    minimum = value if minimum is None else min(minimum, value)
                    max_abs = abs(value) if max_abs is None else max(max_abs, abs(value))
                    if top_value is None or value > top_value:
                        top_index, top_value = index, value
        if (_stamp(os.fstat(stream.fileno())) != before or _safe(path)[1] != before
                or digest.hexdigest() != expected["sha256"] or counts["elements"] * width != expected["bytes"]):
            raise ValueError("bound tensor changed during numerical inspection")
    result = {"status": "DESCRIBED_UNVALIDATED", "dtype": desc["dtype"], "byte_order": "little",
              "shape": desc["shape"], **counts, "finite_min": minimum, "finite_max": maximum,
              "finite_max_abs": max_abs,
              "scope": "Actual bound output statistics only; no reference/accuracy, state or language-quality acceptance"}
    if desc["dtype"] == "FLOAT32":
        if counts["finite_count"] == counts["elements"]:
            result["argmax"] = {"status": "OBSERVED_ALL_FINITE", "token_id": top_index,
                                "value": top_value, "tie_policy": "FIRST_INDEX_STRICT_GREATER"}
        else:
            result["argmax"] = {"status": "NOT_REPORTED_NONFINITE_VALUES_PRESENT"}
    return result


def _optional_draft(path):
    if path is None:
        return {"status": "NOT_REQUESTED"}
    try:
        root, _ = _safe(path, directory=True)
        bindings = []
        value = _json(root / "manifest.json", bindings)
        return {"status": "MANIFEST_OBSERVED_ONLY", "manifest": bindings[-1],
                **{key: value.get(key) for key in ("artifact_kind", "sequence_length", "weight_storage",
                    "platform", "ddk_id", "vocab_size", "draft_config", "model", "inputs", "outputs")},
                "manifest_status": value.get("status"),
                "scope": "Only manifest bytes read; model/input/output payload identities and execution are not revalidated"}
    except (OSError, ValueError, TypeError) as error:
        return {"status": "UNAVAILABLE", "error": f"{type(error).__name__}: {error}"}


def _optional_input_ids(path):
    if path is None:
        return {"status": "NOT_REQUESTED"}
    target = Path(path) / "input_ids.bin"
    try:
        root, _ = _safe(path, directory=True)
        target, info = _safe(root / "input_ids.bin")
        if info[2] != 256:
            return {"status": "DIFFERENT_EXTENT", "path": str(target), "bytes": info[2], "expected_bytes": 256}
        raw, identity = _read_small(target, 256)
        values = list(struct.unpack("<64i", raw))
        return {"status": "OBSERVED" if all(0 <= value < 248320 for value in values) else "OUT_OF_VOCABULARY",
                **identity, "dtype": "int32", "byte_order": "little", "shape": [1, 64], "values": values,
                "scope": "Original file values only; no tokenizer provenance, input embedding equivalence or generation proof"}
    except FileNotFoundError:
        return {"status": "NOT_FOUND", "path": str(target)}
    except (OSError, ValueError, TypeError) as error:
        return {"status": "UNAVAILABLE", "path": str(target), "error": f"{type(error).__name__}: {error}"}


def inspect_target_prefill_outputs(execution, output, *, draft_bundle=None, input_dir=None):
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": _KIND, "status": "FAILED", "output": str(out),
              "model_execution": "NOT_RUN", "device_calls": "NONE", "weight_payload_access": "NONE",
              "model_payload_access": "NONE", "slot_binding": "UNVERIFIED", "feature_selection": "NOT_PERFORMED",
              "state_feedback": "NOT_PERFORMED", "numerical_validation": "NOT_RUN"}
    try:
        execution, _ = _safe(execution, directory=True)
        bindings = []
        outer, board, plan = _bound_execution(execution, bindings)
        descriptors = _descriptors(plan)
        retrieved = board.get("retrieved")
        if not isinstance(retrieved, dict) or not 5 <= len(retrieved) <= 1028:
            raise ValueError("execution has no bounded complete retrieved file inventory")
        output_names = sorted(name for name in retrieved if isinstance(name, str) and name.startswith("outputs/"))
        if set(retrieved) != {*output_names, *_LOGS}:
            raise ValueError("unexpected or missing retrieved output/log records")
        if any(not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] < 0 for row in retrieved.values()):
            raise ValueError("retrieved file records have invalid byte extents")
        if sum(row["bytes"] for row in retrieved.values()) > _MAX_TOTAL:
            raise ValueError("retrieved output/log inventory exceeds512MiB inspection budget")
        rows, observations, logs, log_bytes = [], {}, {}, {}
        for name in [*output_names, *_LOGS]:
            identity, raw = _hash_retrieved(_relative(execution / "board", name), retrieved[name], capture=name in _LOGS)
            observations[name] = identity
            if name in _LOGS:
                logs[name] = _log_display(raw, identity)
                log_bytes[name] = raw
                if name == "tool.exitcode" and raw.strip() != b"0":
                    raise ValueError("retrieved tool exitcode differs from the successful execution")
                if name == "tool.pid" and raw.strip() != str(board["tool_process_id"]).encode():
                    raise ValueError("retrieved tool PID differs from the successful execution")
            else:
                candidates = [row["index"] for row in descriptors if row["bytes"] == identity["bytes"]]
                rows.append({"path": name, **identity, "candidate_runtime_indices_by_bytes_only": candidates,
                             "slot_binding": "UNVERIFIED", "classification": "SIZE_ONLY_CANDIDATES" if candidates else "NO_EXPECTED_RAW_EXTENT"})
        mapping = _logged_slots(log_bytes["tool.stdout"], board, descriptors, rows)
        tensors = {}
        if mapping["status"] == "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG":
            indexed = {row["runtime_index"]: row for row in mapping["entries"]}
            for label, index in (("feature", plan["feature_runtime_output_index"]),
                                 ("logits", plan["logits_runtime_output_index"])):
                name = indexed[index]["path"]
                tensors[label] = {"runtime_index": index, "path": name, **observations[name],
                                  "statistics": _numerical(_relative(execution / "board", name), descriptors[index], observations[name])}
            by_path = {row["path"]: row for row in mapping["entries"]}
            for row in rows:
                if row["path"] in by_path:
                    row.update(runtime_index=by_path[row["path"]]["runtime_index"],
                               slot_binding="EXPLICIT_TOOL_LOG_PATH_AND_DESCRIPTOR_EXTENT")
        for identity in bindings:
            if _read_small(identity["path"], _MAX_METADATA)[1] != identity:
                raise ValueError("execution evidence changed during output inspection")
        for name, identity in observations.items():
            if _safe(_relative(execution / "board", name))[1] != identity["stat"]:
                raise ValueError("retrieved file changed after hash verification")
        expected, observed = Counter(row["bytes"] for row in descriptors), Counter(row["bytes"] for row in rows)
        all_extents = sorted(expected.keys() | observed.keys())
        result.update(status="OUTPUT_FILES_INSPECTED_SLOT_BINDING_UNVERIFIED", execution=str(execution),
            evidence_bindings=bindings, plan_output_count=len(descriptors), actual_retrieved_output_file_count=len(rows),
            tool_process_id=board["tool_process_id"], remote_exitcode=0,
            expected_runtime_outputs=descriptors, output_files=rows,
            raw_extent_multiset_check="MATCHED_COUNTS_AND_EXTENTS_ONLY" if expected == observed else "DIFFERENT_COUNTS_OR_EXTENTS",
            raw_extent_counts=[{"bytes": size, "expected": expected[size], "observed": observed[size],
                "missing": max(0, expected[size] - observed[size]), "excess": max(0, observed[size] - expected[size])} for size in all_extents],
            unclassified_files=[row["path"] for row in rows if not row["candidate_runtime_indices_by_bytes_only"]],
            all_output_files_slot_unclassified=True,
            planned_feature_runtime_index=plan["feature_runtime_output_index"],
            planned_logits_runtime_index=plan["logits_runtime_output_index"],
            logged_slot_mapping=mapping, bound_tensors=tensors,
            tool_logs=logs, draft_bundle=_optional_draft(draft_bundle), original_input_ids=_optional_input_ids(input_dir),
            scope="Fresh retrieved file SHA/size and saved execution/descriptor-plan checks only. File count/byte extents, numeric filenames and statistical appearance establish no runtime slot identity. Feature and KV can share2MiB; no feature selected, model/input weights read, device call, feedback, accuracy or DFlash completion claim.")
        if mapping["status"] == "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG":
            result.update(status="OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", slot_binding="EXPLICIT_COMPLETE_TOOL_LOG_AND_VERIFIED_BYTES",
                feature_selection="BOUND_FILE_DESCRIBED_ONLY_NO_COPY_OR_EXECUTION", all_output_files_slot_unclassified=False,
                state_layer_binding="UNVERIFIED", scope="Explicit SHA-verified tool log establishes runtime output index-to-file paths, checked against verified file bytes and saved runtime descriptors. Bound feature/logits statistics establish no numerical accuracy; state layer identities and feedback remain unverified. No model/weight payloads read or device calls.")
        if input_dir is not None:
            result["original_input_ids"]["matches_planned_input_dir"] = (os.path.abspath(Path(input_dir).expanduser()) == plan.get("input_dir"))
            result["original_input_ids"]["path_comparison_scope"] = "Directory path comparison only; no token-to-embedding or tokenizer provenance proof"
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    save_json(out / "report.json", result)
    return result

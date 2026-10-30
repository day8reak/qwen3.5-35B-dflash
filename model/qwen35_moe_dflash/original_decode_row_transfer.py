"""Bounded original-row handoff; the extract command uses only the stdlib.

The prepared directory contains this exact module as extract_rows.py, which can
run on the resource host without this package, YAML, Torch or a model load.
"""
from __future__ import annotations

from contextlib import ExitStack
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


_VOCAB, _HIDDEN, _MAX_IDS = 248320, 2048, 1024
_REQUEST_MAX, _RESPONSE_MAX, _METADATA_MAX = 64 << 10, 1 << 20, 32 << 20
_REQUEST_KIND = "qwen35-original-decode-row-request"
_RESPONSE_KIND = "qwen35-original-decode-row-response"
_PREPARED_KIND = "qwen35-original-decode-row-transfer-prepared"
_EXTRACTED = "ORIGINAL_DECODE_ROWS_EXTRACTED"
_LIMITS = {"source_ids_maximum_bytes": 4096, "source_candidate_limit": 1024,
           "table_payload_maximum_bytes": 2103300}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _stamp(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _safe(path, *, directory=False):
    path = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:], 1):
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("symlink is unsupported: " + str(current))
        last = index == len(path.parts) - 1
        if not (stat.S_ISDIR(info.st_mode) if directory or not last else stat.S_ISREG(info.st_mode)):
            raise ValueError("expected regular file/directory: " + str(current))
    return path, _stamp(path.stat())


def _read(path, maximum):
    path, before = _safe(path)
    if not 0 < before[2] <= maximum:
        raise ValueError(f"small transfer file extent unsupported: path={path}; bytes={before[2]}; maximum_bytes={maximum}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("transfer file changed before read: " + str(path))
        raw = stream.read(maximum + 1)
        if len(raw) != before[2] or _stamp(os.fstat(stream.fileno())) != before or _safe(path)[1] != before:
            raise ValueError("transfer file changed during read: " + str(path))
    return raw, {"path": str(path), "bytes": len(raw), "sha256": _sha(raw), "stat": before}


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("duplicate transfer JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite transfer JSON constant: " + value)


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite transfer JSON number: " + value)
    return result


def _json(path, maximum):
    raw, identity = _read(path, maximum)
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError("transfer JSON must be an object")
    return value, identity


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def _new_output(path, *, require_run=True):
    if require_run and not os.environ.get("AI_RUN_DIR"):
        raise ValueError("AI_RUN_DIR is required for a new owned transfer output")
    path = Path(path).resolve()
    if require_run:
        root = Path(os.environ["AI_RUN_DIR"]).resolve()
        if path == root or root not in path.parents:
            raise ValueError("transfer output must be below AI_RUN_DIR")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write(root, name, raw):
    path = root / name
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": _sha(raw), "stat": _safe(path)[1]}


def _save(root, name, value):
    return _write(root, name, _encoded(value))


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("transfer SHA256 must be lowercase hexadecimal")
    return value


def _blob(raw, dtype, shape):
    return {"dtype": dtype, "shape": shape, "byte_order": "little", "bytes": len(raw),
            "sha256": _sha(raw), "base64": base64.b64encode(raw).decode("ascii")}


def _unblob(value, dtype, shape, size):
    if (not isinstance(value, dict) or set(value) != {"dtype", "shape", "byte_order", "bytes", "sha256", "base64"}
            or value["dtype"] != dtype or value["shape"] != shape or not isinstance(value["shape"], list)
            or any(type(item) is not int for item in value["shape"]) or value["byte_order"] != "little"
            or type(value["bytes"]) is not int or value["bytes"] != size
            or not isinstance(value["base64"], str) or len(value["base64"]) != 4 * ((size + 2) // 3)):
        raise ValueError("transfer payload dtype/shape/extent differs from the frozen row ABI")
    raw = base64.b64decode(value["base64"], validate=True)
    if len(raw) != size or _sha(raw) != _digest(value["sha256"]):
        raise ValueError("transfer payload SHA256 or byte extent differs")
    return raw


def _binding(request):
    return _sha(_encoded({"source_audit_sha256": request["source_audit_sha256"],
        "source_ids_sha256": request["source_token_ids"]["sha256"],
        "executed_embedding_sha256": request["executed_embedding"]["sha256"],
        "executed_scale_sha256": request["executed_scale"]["sha256"],
        "bound_logits_sha256": request["bound_logits_sha256"], "next_token_id": request["next_token_id"],
        "table_paths": request["table_paths"]}))


def _request(value):
    if (set(value) != {"schema_version", "artifact_kind", "vocab_size", "hidden_size", "limits", "source_audit_sha256",
            "source_token_ids", "executed_embedding", "executed_scale", "next_token_id", "bound_logits_sha256", "binding_digest", "table_paths"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["artifact_kind"] != _REQUEST_KIND
            or type(value["vocab_size"]) is not int or value["vocab_size"] != _VOCAB
            or type(value["hidden_size"]) is not int or value["hidden_size"] != _HIDDEN or value["limits"] != _LIMITS):
        raise ValueError("unexpected original row request schema or frozen limits")
    paths = value["table_paths"]
    if (not isinstance(paths, dict) or set(paths) != {"weight", "scale"}
            or any(not isinstance(path, str) or not path or "\0" in path or not Path(path).is_absolute() for path in paths.values())
            or paths["weight"] == paths["scale"]):
        raise ValueError("request must bind the distinct original configured table paths")
    ids_blob = value["source_token_ids"]
    shape = ids_blob.get("shape") if isinstance(ids_blob, dict) else None
    if (not isinstance(shape, list) or len(shape) != 2 or type(shape[0]) is not int or shape[0] != 1
            or type(shape[1]) is not int or not 1 <= shape[1] <= _MAX_IDS):
        raise ValueError("source candidates require one bounded INT32 vector")
    raw_ids = _unblob(ids_blob, "INT32", shape, 4 * shape[1])
    ids = list(struct.unpack("<" + "i" * shape[1], raw_ids))
    if any(not 0 <= token < _VOCAB for token in ids):
        raise ValueError("source candidate ID outside original vocabulary")
    row = _unblob(value["executed_embedding"], "INT8", [1, 1, _HIDDEN], _HIDDEN)
    scale = _unblob(value["executed_scale"], "FLOAT32", [1, 1, 1], 4)
    factor, = struct.unpack("<f", scale)
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("executed original scale must be finite and nonnegative")
    if type(value["next_token_id"]) is not int or not 0 <= value["next_token_id"] < _VOCAB:
        raise ValueError("next token is outside original vocabulary")
    for key in ("source_audit_sha256", "bound_logits_sha256", "binding_digest"):
        _digest(value[key])
    if value["binding_digest"] != _binding(value):
        raise ValueError("request local evidence binding digest differs")
    positions = {}
    for index, token in enumerate(ids):
        positions.setdefault(token, []).append(index)
    return raw_ids, row, scale, positions


def _fresh_local(audit, inspection, config_yaml):
    # These imports are never executed by the portable extract command.
    from .original_decode_embedding_probe import _input
    from .original_embedding_probe import _bound_bytes
    from .target_decode_output_inventory import validate_target_decode_output_inventory
    from .target_prefill_output_inventory import _numerical

    if (audit.get("artifact_kind") != "qwen35-original-decode-embedding-row-probe"
            or audit.get("status") != "ORIGINAL_EMBEDDING_TABLES_UNAVAILABLE" or audit.get("inspection") != str(inspection)
            or audit.get("logical_history") != "UNVERIFIED" or audit.get("runtime_control_recipe") != "UNVERIFIED"
            or audit.get("next_token_inputs") != [] or audit.get("source_preservation") != "UNCHANGED_DECODE_METADATA_SOURCE_IDS_INPUT_ROWS_LOGITS_AND_SELECTED_TABLE_IDENTITIES"):
        raise ValueError("portable preparation requires an untouched bound source audit with original tables unavailable")
    current = validate_target_decode_output_inventory(inspection)
    report_identity = _read(inspection / "report.json", _METADATA_MAX)[1]
    config_identity = _read(config_yaml, 1 << 20)[1]
    allowed = [*current["evidence_bindings"], report_identity, config_identity]
    for identity in audit["source_bindings"]:
        if identity not in allowed or _read(identity["path"], _METADATA_MAX)[1] != identity:
            raise ValueError("local source audit metadata changed or is not in the fresh decode proof")
    if config_identity != audit["original_config"]:
        raise ValueError("original table configuration changed after source audit")
    execution = Path(current["execution"])
    plan, _ = _json(execution / "plan.json", _METADATA_MAX)
    trial_binding = current["trial_inputs"]["manifest"]
    trial, trial_identity = _json(trial_binding["path"], _METADATA_MAX)
    if trial_identity != trial_binding or str(execution) != audit["execution"]:
        raise ValueError("executed trial no longer matches the local source audit")
    ids_path = Path(trial["source_input_dir"]) / "input_ids.bin"
    raw_ids, ids_identity = _bound_bytes(ids_path, audit["original_input_ids"], 4096)
    embedding, embedding_identity = _input(plan, Path(plan["input_dir"]), "arg1_1", "inputs_embeds", "INT8", _HIDDEN)
    scale, scale_identity = _input(plan, Path(plan["input_dir"]), "arg0_1", "embed_scale", "FLOAT32", 4)
    if embedding_identity != audit["executed_embedding"] or scale_identity != audit["executed_scale"]:
        raise ValueError("actual executed row/scale identity changed after local source audit")
    logits = current["bound_tensors"]["logits"]
    logit_path = execution / "board" / logits["path"]
    logit_identity = {"path": str(logit_path), **{key: logits[key] for key in ("bytes", "sha256", "stat")}}
    if logit_identity != audit["bound_logits"]:
        raise ValueError("actual decode logits binding changed after local source audit")
    stats = _numerical(logit_path, current["expected_runtime_outputs"][logits["runtime_index"]], logits)
    if (stats != logits["statistics"] or stats.get("argmax", {}).get("status") != "OBSERVED_ALL_FINITE"
            or stats["argmax"]["token_id"] != audit["selected_next_token_id"]):
        raise ValueError("fresh decode logits no longer bind the selected greedy token")
    return raw_ids, embedding, scale, stats["argmax"]["token_id"]


def prepare_rows(inspection, config_yaml, output):
    from .original_decode_embedding_probe import inspect_original_decode_embeddings

    out = _new_output(output)
    result = {"schema_version": 1, "artifact_kind": _PREPARED_KIND, "status": "FAILED", "output": str(out)}
    try:
        inspection, _ = _safe(inspection, directory=True)
        config_yaml, _ = _safe(config_yaml)
        audit = inspect_original_decode_embeddings(inspection, config_yaml, out / "source-audit", match_source_id_candidates=True)
        raw_ids, embedding, scale, next_token = _fresh_local(audit, inspection, config_yaml)
        audit_identity = _read(out / "source-audit/report.json", _METADATA_MAX)[1]
        request = {"schema_version": 1, "artifact_kind": _REQUEST_KIND, "vocab_size": _VOCAB, "hidden_size": _HIDDEN,
            "limits": dict(_LIMITS), "source_audit_sha256": audit_identity["sha256"],
            "source_token_ids": _blob(raw_ids, "INT32", [1, len(raw_ids) // 4]),
            "executed_embedding": _blob(embedding, "INT8", [1, 1, _HIDDEN]),
            "executed_scale": _blob(scale, "FLOAT32", [1, 1, 1]), "next_token_id": next_token,
            "bound_logits_sha256": audit["bound_logits"]["sha256"],
            "table_paths": {table["role"]: table["path"] for table in audit["tables"]}}
        request["binding_digest"] = _binding(request)
        _request(request)
        request_raw = _encoded(request)
        if len(request_raw) > _REQUEST_MAX:
            raise ValueError("portable request exceeds64KiB budget")
        request_identity = _write(out, "request.json", request_raw)
        script_raw, _ = _read(__file__, 1 << 20)
        script_identity = _write(out, "extract_rows.py", script_raw)
        result.update(status="ORIGINAL_DECODE_ROW_TRANSFER_PREPARED", inspection=str(inspection), config_yaml=str(config_yaml),
            source_audit=audit_identity, request=request_identity, extraction_script=script_identity,
            binding_digest=request["binding_digest"], next_token_id=next_token,
            logical_history="UNVERIFIED", runtime_control_recipe="UNVERIFIED", state_feedback="UNVERIFIED",
            model_execution="NOT_RUN", model_payload_access="NONE", whole_table_copy="NOT_PERFORMED",
            next="Transfer only request.json and extract_rows.py to the original table host; return its response.json")
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, struct.error) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    _save(out, "manifest.json", result)
    return result


def extract_rows(request_path, weight_table, scale_table, output):
    out = _new_output(output, require_run=False)
    result = {"schema_version": 1, "artifact_kind": _RESPONSE_KIND, "status": "FAILED",
        "logical_history": "UNVERIFIED", "runtime_control_recipe": "UNVERIFIED", "state_feedback": "UNVERIFIED",
        "full_table_hash": "NOT_PERFORMED", "table_authenticity": "EXPLICIT_RESOURCE_PATHS_SELECTED_ROWS_ONLY",
        "selected_rows": [], "source_match": None, "next_token": None, "table_read_bytes": 0}
    try:
        _, extractor_identity = _read(__file__, 1 << 20)
        result["extractor_sha256"] = extractor_identity["sha256"]
        request, identity = _json(request_path, _REQUEST_MAX)
        raw_ids, executed_row, executed_scale, positions = _request(request)
        result.update(request_sha256=identity["sha256"], binding_digest=request["binding_digest"],
            source_id_sha256=_sha(raw_ids), candidate_count=len(raw_ids) // 4, unique_candidate_count=len(positions),
            maximum_table_read_bytes=(len(positions) + 1) * (_HIDDEN + 4),
            uniqueness_scope="EXPLICIT_SOURCE_ID_CANDIDATES_ONLY", position_index_base=0)
        tables, rows = [], {}
        result["tables"] = tables
        with ExitStack() as stack:
            for role, path, width in (("weight", weight_table, _HIDDEN), ("scale", scale_table, 4)):
                path, before = _safe(path)
                if str(path) != request["table_paths"][role]:
                    raise ValueError("resource table path differs from the explicit original configuration: " + str(path))
                if before[2] != _VOCAB * width:
                    raise ValueError("original table extent differs from frozen full-vocabulary ABI: " + str(path))
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                stack.callback(os.close, fd)
                if _stamp(os.fstat(fd)) != before:
                    raise ValueError("original table changed before selected-row read")
                tables.append({"role": role, "path": str(path), "bytes": before[2], "stat": before, "row_bytes": width,
                    "full_content_validation": "NOT_PERFORMED", "fd": fd})
            def read_row(token):
                payloads = []
                for table in tables:
                    if result["table_read_bytes"] + table["row_bytes"] > result["maximum_table_read_bytes"]:
                        raise ValueError("original table row budget exceeded")
                    raw = os.pread(table["fd"], table["row_bytes"], token * table["row_bytes"])
                    if len(raw) != table["row_bytes"]:
                        raise ValueError("selected original table row was truncated")
                    result["table_read_bytes"] += len(raw)
                    payloads.append(raw)
                factor, = struct.unpack("<f", payloads[1])
                if not math.isfinite(factor) or factor < 0:
                    raise ValueError("selected original scale must be finite and nonnegative")
                rows[token] = payloads
            for token in sorted(positions):
                read_row(token)
            matches = [token for token in sorted(positions) if rows[token] == [executed_row, executed_scale]]
            result["matching_token_ids"] = matches
            result["matching_positions"] = [{"token_id": token, "positions": positions[token]} for token in matches]
            if len(matches) == 1:
                source = matches[0]
                next_token = request["next_token_id"]
                if next_token not in rows:
                    read_row(next_token)
                result["source_match"] = {"token_id": source, "positions": positions[source],
                    "embedding": _blob(rows[source][0], "INT8", [1, 1, _HIDDEN]),
                    "scale": _blob(rows[source][1], "FLOAT32", [1, 1, 1])}
                result["next_token"] = {"token_id": next_token, "embedding": _blob(rows[next_token][0], "INT8", [1, 1, _HIDDEN]),
                    "scale": _blob(rows[next_token][1], "FLOAT32", [1, 1, 1])}
                result["status"] = _EXTRACTED
            else:
                result["status"] = "ORIGINAL_DECODE_ROWS_NO_MATCH" if not matches else "ORIGINAL_DECODE_ROWS_AMBIGUOUS"
            result["selected_rows"] = [{"token_id": token, "weight_sha256": _sha(values[0]), "scale_sha256": _sha(values[1])}
                for token, values in sorted(rows.items())]
            for table in tables:
                if _stamp(os.fstat(table["fd"])) != table["stat"] or _safe(table["path"])[1] != table["stat"]:
                    raise ValueError("original table changed during selected-row extraction")
            if _read(identity["path"], _REQUEST_MAX)[1] != identity:
                raise ValueError("portable request changed during selected-row extraction")
            if _read(extractor_identity["path"], 1 << 20)[1] != extractor_identity:
                raise ValueError("portable extraction source changed during selected-row extraction")
        result["source_preservation"] = "UNCHANGED_REQUEST_AND_SELECTED_TABLE_IDENTITIES"
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, struct.error) as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", source_match=None, next_token=None)
    for table in result.get("tables", []):
        table.pop("fd", None)
    if len(_encoded(result)) > _RESPONSE_MAX:
        raise ValueError("portable response exceeds1MiB budget")
    _save(out, "response.json", result)
    return result


def _response(value, request, request_identity):
    raw_ids, row, scale, positions = _request(request)
    if (value.get("artifact_kind") != _RESPONSE_KIND or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("status") != _EXTRACTED
            or value.get("request_sha256") != request_identity["sha256"] or value.get("binding_digest") != request["binding_digest"]
            or value.get("source_id_sha256") != _sha(raw_ids) or value.get("source_preservation") != "UNCHANGED_REQUEST_AND_SELECTED_TABLE_IDENTITIES"
            or any(value.get(key) != "UNVERIFIED" for key in ("logical_history", "runtime_control_recipe", "state_feedback"))
            or value.get("full_table_hash") != "NOT_PERFORMED" or value.get("table_authenticity") != "EXPLICIT_RESOURCE_PATHS_SELECTED_ROWS_ONLY"
            or value.get("uniqueness_scope") != "EXPLICIT_SOURCE_ID_CANDIDATES_ONLY" or value.get("position_index_base") != 0):
        raise ValueError("response is not the successful exact-request selected-row extraction")
    expected_tokens = sorted(set(positions) | {request["next_token_id"]})
    rows = value.get("selected_rows")
    if (not isinstance(rows, list) or len(rows) != len(expected_tokens)
            or any(not isinstance(item, dict) or set(item) != {"token_id", "weight_sha256", "scale_sha256"} for item in rows)
            or [item["token_id"] for item in rows] != expected_tokens
            or any(type(item["token_id"]) is not int for item in rows)):
        raise ValueError("response row set differs from the exact requested candidate/winner set")
    by_token = {item["token_id"]: item for item in rows}
    for item in rows:
        _digest(item["weight_sha256"]); _digest(item["scale_sha256"])
    matches = [token for token in sorted(positions) if (by_token[token]["weight_sha256"], by_token[token]["scale_sha256"]) == (_sha(row), _sha(scale))]
    if len(matches) != 1 or value.get("matching_token_ids") != matches:
        raise ValueError("response does not establish exactly one distinct candidate row/scale match")
    matched = matches[0]
    expected_positions = [{"token_id": matched, "positions": positions[matched]}]
    if (value.get("matching_positions") != expected_positions or type(value.get("candidate_count")) is not int
            or type(value.get("unique_candidate_count")) is not int or type(value.get("maximum_table_read_bytes")) is not int
            or value.get("candidate_count") != len(raw_ids) // 4
            or value.get("unique_candidate_count") != len(positions)
            or value.get("maximum_table_read_bytes") != (len(positions) + 1) * (_HIDDEN + 4)
            or type(value.get("table_read_bytes")) is not int or value["table_read_bytes"] != len(rows) * (_HIDDEN + 4)):
        raise ValueError("response candidate positions/counts or bounded table read extent differs")
    source, next_row = value.get("source_match"), value.get("next_token")
    if (not isinstance(source, dict) or set(source) != {"token_id", "positions", "embedding", "scale"}
            or source["token_id"] != matched or source["positions"] != positions[matched]
            or not isinstance(next_row, dict) or set(next_row) != {"token_id", "embedding", "scale"}
            or type(source["token_id"]) is not int or type(next_row["token_id"]) is not int
            or next_row["token_id"] != request["next_token_id"]):
        raise ValueError("response selected source/next token identities differ")
    if _unblob(source["embedding"], "INT8", [1, 1, _HIDDEN], _HIDDEN) != row or _unblob(source["scale"], "FLOAT32", [1, 1, 1], 4) != scale:
        raise ValueError("resource source row/scale is not byte-identical to the current executed inputs")
    next_bytes = _unblob(next_row["embedding"], "INT8", [1, 1, _HIDDEN], _HIDDEN)
    next_scale = _unblob(next_row["scale"], "FLOAT32", [1, 1, 1], 4)
    next_hashes = by_token[request["next_token_id"]]
    factor, = struct.unpack("<f", next_scale)
    if (not math.isfinite(factor) or factor < 0 or next_hashes["weight_sha256"] != _sha(next_bytes)
            or next_hashes["scale_sha256"] != _sha(next_scale)):
        raise ValueError("resource next row hashes/scale differ from its selected-row provenance")
    tables = value.get("tables")
    if not isinstance(tables, list) or len(tables) != 2:
        raise ValueError("resource table provenance must contain exactly original weight and scale tables")
    for table, role, width in zip(tables, ("weight", "scale"), (_HIDDEN, 4)):
        if (not isinstance(table, dict) or table.get("role") != role or table.get("row_bytes") != width
                or table.get("bytes") != _VOCAB * width or table.get("full_content_validation") != "NOT_PERFORMED"
                or not isinstance(table.get("path"), str) or not Path(table["path"]).is_absolute()
                or table["path"] != request["table_paths"][role]
                or not isinstance(table.get("stat"), list) or len(table["stat"]) != 5
                or any(type(item) is not int or item < 0 for item in table["stat"]) or table["stat"][2] != _VOCAB * width):
            raise ValueError("resource table path/stat/extent provenance differs from the frozen original table ABI")
    return next_bytes, next_scale, matched, positions[matched]


def import_rows(prepared, response_path, output):
    out = _new_output(output)
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-decode-row-transfer-import", "status": "FAILED",
        "output": str(out), "next_token_inputs": [], "logical_history": "UNVERIFIED", "runtime_control_recipe": "UNVERIFIED",
        "state_feedback": "UNVERIFIED", "numerical_validation": "NOT_RUN_NO_REFERENCE", "device_execution": "NOT_RUN"}
    try:
        prepared, _ = _safe(prepared, directory=True)
        manifest, manifest_identity = _json(prepared / "manifest.json", _METADATA_MAX)
        if (manifest.get("artifact_kind") != _PREPARED_KIND or manifest.get("status") != "ORIGINAL_DECODE_ROW_TRANSFER_PREPARED"
                or manifest.get("output") != str(prepared)):
            raise ValueError("expected the successful owned original row transfer preparation")
        request, request_identity = _json(prepared / "request.json", _REQUEST_MAX)
        audit, audit_identity = _json(prepared / "source-audit/report.json", _METADATA_MAX)
        script_identity = _read(prepared / "extract_rows.py", 1 << 20)[1]
        if request_identity != manifest["request"] or audit_identity != manifest["source_audit"] or request["source_audit_sha256"] != audit_identity["sha256"]:
            raise ValueError("prepared request/source audit identity changed")
        if script_identity != manifest["extraction_script"]:
            raise ValueError("prepared portable extraction source identity changed")
        if request.get("table_paths") != {table["role"]: table["path"] for table in audit["tables"]}:
            raise ValueError("portable request table paths differ from the original configured source audit")
        payloads = _fresh_local(audit, Path(manifest["inspection"]), Path(manifest["config_yaml"]))
        raw_ids, row, scale, _ = _request(request)
        if payloads != (raw_ids, row, scale, request["next_token_id"]) or request["binding_digest"] != manifest["binding_digest"]:
            raise ValueError("portable request no longer matches current actual decode input/logits bytes")
        response, response_identity = _json(response_path, _RESPONSE_MAX)
        if response.get("extractor_sha256") != manifest["extraction_script"]["sha256"]:
            raise ValueError("resource extractor SHA256 differs from the exact prepared extraction source")
        next_bytes, next_scale, source, positions = _response(response, request, request_identity)
        # Rebind again before any next-token file exists; source rows and logits
        # are small and the complete output inventory remains read-only.
        if _fresh_local(audit, Path(manifest["inspection"]), Path(manifest["config_yaml"])) != payloads:
            raise ValueError("local decode source changed during response verification")
        for identity, maximum in ((manifest_identity, _METADATA_MAX), (request_identity, _REQUEST_MAX),
                                  (audit_identity, _METADATA_MAX), (script_identity, 1 << 20), (response_identity, _RESPONSE_MAX)):
            if _read(identity["path"], maximum)[1] != identity:
                raise ValueError("row transfer evidence changed during import")
        next_dir = out / "next-token"
        next_dir.mkdir()
        files = [_write(next_dir, "inputs_embeds.bin", next_bytes), _write(next_dir, "embed_scale.bin", next_scale),
                 _write(next_dir, "input_ids.bin", struct.pack("<i", request["next_token_id"]))]
        for entry, dtype, shape in zip(files, ("INT8", "FLOAT32", "INT32"), ([1, 1, _HIDDEN], [1, 1, 1], [1, 1])):
            entry.update(dtype=dtype, shape=shape, byte_order="little")
        result.update(status="ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED", prepared=str(prepared),
            request=request_identity, response=response_identity, source_audit=audit_identity, next_token_inputs=files,
            selected_source_token_id=source, matching_source_positions=positions, selected_next_token_id=request["next_token_id"],
            uniqueness_scope="EXPLICIT_SOURCE_ID_CANDIDATES_ONLY", resource_table_provenance=response["tables"],
            selected_row_provenance=response["selected_rows"], table_authenticity="DELEGATED_RESOURCE_READER_SELECTED_ROWS_ONLY_NO_FULL_TABLE_AUTHENTICATION",
            source_preservation="FRESH_LOCAL_DECODE_EVIDENCE_AND_EXACT_REQUEST_RESPONSE_SHA256_BOUND",
            next="Retain the exact original INT8 row and separate FP32 scale; logical history, controls and all80 state edges remain unverified")
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, struct.error) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    _save(out, "report.json", result)
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="bind local decode evidence and write only a small portable request")
    prepare.add_argument("--inspection", required=True); prepare.add_argument("--config-yaml", required=True)
    prepare.add_argument("--output", required=True)
    extract = commands.add_parser("extract", help="stdlib-only selected row extraction on the original table host")
    extract.add_argument("--request", required=True); extract.add_argument("--weight-table", required=True)
    extract.add_argument("--scale-table", required=True); extract.add_argument("--output", required=True)
    consume = commands.add_parser("import", help="freshly rebind local evidence and verify a returned response")
    consume.add_argument("--prepared", required=True); consume.add_argument("--response", required=True)
    consume.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare": result = prepare_rows(args.inspection, args.config_yaml, args.output)
        elif args.command == "extract": result = extract_rows(args.request, args.weight_table, args.scale_table, args.output)
        else: result = import_rows(args.prepared, args.response, args.output)
        keys = ("status", "output", "request", "extraction_script", "request_sha256", "binding_digest", "candidate_count",
                "unique_candidate_count", "matching_token_ids", "matching_positions", "table_read_bytes", "tables",
                "selected_source_token_id", "selected_next_token_id", "matching_source_positions", "next_token_inputs", "error", "next")
        print(json.dumps({key: result[key] for key in keys if key in result}, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result["status"] in {"ORIGINAL_DECODE_ROW_TRANSFER_PREPARED", _EXTRACTED,
            "ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED"} else 1
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bind selected original INT8 embedding rows to an executed target prefix.

Only selected table rows are read. This never constructs embeddings from a
floating model, multiplies scales, hashes a complete table or runs a model.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import math
import os
from pathlib import Path
import struct

from .artifacts import output_path, save_json
from .target_compile_failure import _read_small, _relative, _safe, _stamp
from .target_prefill_output_inventory import _bound_execution, _descriptors, _json, _numerical


_VOCAB, _HIDDEN, _PREFIX = 248320, 2048, 64
_TABLES = (("weight", "embedding_weight_path", _VOCAB * _HIDDEN, _HIDDEN, "INT8"),
           ("scale", "embedding_scale_path", _VOCAB * 4, 4, "FLOAT32"))
_READY = "ORIGINAL_EMBEDDING_ROWS_BOUND_NEXT_TOKEN_PREPARED"


def _metadata(path, bindings):
    return _json(path, bindings)


def _bound_bytes(path, expected, maximum):
    raw, observed = _read_small(path, maximum)
    if any(observed[key] != expected.get(key) for key in ("bytes", "sha256")):
        raise ValueError("original prefix input SHA256/extent changed: " + str(path))
    if "stat" in expected and observed["stat"] != expected["stat"]:
        raise ValueError("original prefix input identity changed: " + str(path))
    original = expected.get("identity")
    if original is not None and observed["stat"] != [original.get(key)
            for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]:
        raise ValueError("original prefix input source-plan identity changed: " + str(path))
    return raw, observed


def _prefix_source(plan, root, *, name, semantic, dtype, size):
    expected = root / (semantic + ".bin")
    rows = [row for row in plan["inputs"] if isinstance(row, dict)
        and row.get("semantic") == semantic and row.get("compiled_candidates") == [name]]
    sources = [row for row in plan.get("source_inputs", []) if isinstance(row, dict)
        and row.get("path") == str(expected)]
    if (len(rows) != 1 or len(sources) != 1 or rows[0].get("source") != str(expected)
            or rows[0].get("dtype") != dtype or rows[0].get("bytes") != size
            or rows[0].get("sha256") != sources[0].get("sha256")):
        raise ValueError("missing unique successful-prefill embedding source: " + semantic)
    return _bound_bytes(expected, sources[0], size)


def _config(config_yaml, bindings):
    import yaml

    raw, identity = _read_small(config_yaml, 1 << 20)
    bindings.append(identity)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError("invalid original embedding YAML: " + str(error)) from error
    if not isinstance(value, dict):
        raise ValueError("original embedding YAML must be a mapping")
    paths = {}
    for _, field, _, _, _ in _TABLES:
        text = value.get(field)
        if not isinstance(text, str) or not text.strip() or "\0" in text:
            raise ValueError("original YAML requires " + field)
        path = Path(text).expanduser()
        paths[field] = Path(os.path.abspath(path if path.is_absolute() else Path(identity["path"]).parent / path))
    if paths[_TABLES[0][1]] == paths[_TABLES[1][1]]:
        raise ValueError("original weight and scale tables require distinct files")
    return paths, identity


def _write(root, relative, raw, *, dtype, shape):
    path = _relative(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": dtype, "shape": shape, "byte_order": "little"}


def inspect_original_embedding_rows(inspection_dir, config_yaml, output_dir):
    """Compare all64 original rows, then retain only the bound next-token row.

    A successful prefix match validates selected row lookup only. The complete
    table contents and original source-to-AIR revision remain unverified.
    """
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "artifact_kind": "qwen35-original-embedding-row-probe",
        "status": "FAILED", "stage": "bind-executed-prefix", "output": str(out),
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "vendor_import": "NOT_PERFORMED",
        "table_payload_access": "SELECTED_ROWS_ONLY", "full_table_hash": "NOT_PERFORMED",
        "numerical_validation": "NOT_RUN", "runtime_control_recipe": "UNVERIFIED",
        "state_feedback": "UNVERIFIED", "next_token_inputs": [], "source_preservation": "NOT_CHECKED",
        "read_budgets": {"table_payload_maximum_bytes": 65 * (_HIDDEN + 4),
            "original_prefix_payload_maximum_bytes": _PREFIX * (_HIDDEN + 8),
            "bound_logits_maximum_bytes": _VOCAB * 4},
        "scope": "Original selected INT8 rows and separate FP32 scales only; no requantization, multiplication, model load or decode controls/state construction."}
    bindings, originals, tables = [], [], []
    try:
        inspection, _ = _safe(inspection_dir, directory=True)
        prior = _metadata(inspection / "report.json", bindings)
        if (prior.get("artifact_kind") != "qwen35-target-prefill-output-inventory"
                or prior.get("status") != "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
                or prior.get("slot_binding") != "EXPLICIT_COMPLETE_TOOL_LOG_AND_VERIFIED_BYTES"
                or prior.get("logged_slot_mapping", {}).get("status") != "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG"
                or prior.get("actual_retrieved_output_file_count") != 82):
            raise ValueError("expected the completed explicitly bound target prefill output inspection")
        execution, _ = _safe(prior["execution"], directory=True)
        current = []
        _, board, plan = _bound_execution(execution, current)
        if current != prior.get("evidence_bindings"):
            raise ValueError("successful target execution metadata differs from the selected output inspection")
        bindings.extend(current)
        root, _ = _safe(plan["input_dir"], directory=True)
        ids_row = prior.get("original_input_ids", {})
        if (ids_row.get("status") != "OBSERVED" or ids_row.get("matches_planned_input_dir") is not True
                or ids_row.get("path") != str(root / "input_ids.bin") or ids_row.get("dtype") != "int32"
                or ids_row.get("shape") != [1, 64] or ids_row.get("byte_order") != "little"):
            raise ValueError("original64 token IDs must match the successful target input directory")
        ids_raw, ids_identity = _bound_bytes(root / "input_ids.bin", ids_row, 256)
        ids = list(struct.unpack("<64i", ids_raw))
        if ids != ids_row.get("values") or any(not 0 <= token < _VOCAB for token in ids):
            raise ValueError("original prefix token IDs differ or lie outside the vocabulary")
        originals.append(ids_identity)
        logits = prior.get("bound_tensors", {}).get("logits", {})
        descs = _descriptors(plan)
        index = plan["logits_runtime_output_index"]
        if (logits.get("runtime_index") != index or logits.get("bytes") != _VOCAB * 4
                or descs[index]["dtype"] != "FLOAT32"):
            raise ValueError("bound logits descriptor differs from successful target output")
        logged = [row for row in prior["logged_slot_mapping"].get("entries", [])
            if row.get("runtime_index") == index]
        received = board.get("retrieved", {}).get(logits.get("path"), {})
        if (len(logged) != 1 or logged[0].get("path") != logits.get("path")
                or any(received.get(key) != logits.get(key) for key in ("bytes", "sha256"))):
            raise ValueError("selected logits file is not bound to the explicit successful tool slot")
        logit_path = _relative(execution / "board", logits["path"])
        observed = _numerical(logit_path, descs[index], logits)
        greedy = observed.get("argmax", {})
        if observed != logits.get("statistics") or greedy.get("status") != "OBSERVED_ALL_FINITE":
            raise ValueError("fresh finite bound logits and its greedy token must match the prior inspection")
        token = greedy.get("token_id")
        if type(token) is not int or not 0 <= token < _VOCAB:
            raise ValueError("bound greedy token is outside the original embedding vocabulary")
        originals.append({"path": str(logit_path), **{key: logits[key] for key in ("bytes", "sha256", "stat")}})
        embedding, embedding_id = _prefix_source(plan, root, name="arg1_1", semantic="inputs_embeds", dtype="INT8", size=64 * _HIDDEN)
        scales, scales_id = _prefix_source(plan, root, name="arg0_1", semantic="embed_scale", dtype="FLOAT32", size=64 * 4)
        originals.extend((embedding_id, scales_id))
        report.update(inspection=str(inspection), execution=str(execution), selected_next_token_id=token,
            token_selection="FRESH_SHA_BOUND_ALL_FINITE_TARGET_LOGITS_ARGMAX_FIRST_INDEX",
            greedy_observation=greedy, bound_logits=originals[1], original_input_ids=ids_identity,
            original_prefix_embedding=embedding_id, original_prefix_scale=scales_id,
            feature_diagnostics=prior.get("bound_tensors", {}).get("feature", {}).get("statistics"),
            observed_payload_reads={"bound_logits_bytes": _VOCAB * 4,
                "original_prefix_input_bytes": len(ids_raw) + len(embedding) + len(scales), "table_bytes": 0})
        report["stage"] = "open-original-embedding-tables"
        paths, yaml_identity = _config(config_yaml, bindings)
        report["original_config"] = yaml_identity
        report["tables"] = tables
        missing = []
        for label, field, size, width, dtype in _TABLES:
            row = {"role": label, "path": str(paths[field]), "expected_bytes": size,
                   "dtype": dtype, "row_bytes": width, "full_content_validation": "NOT_PERFORMED"}
            tables.append(row)
            try:
                _, before = _safe(paths[field])
                row["stat"] = before
                if before[2] != size:
                    raise ValueError("original embedding table extent differs from frozen INT8/FP32 layout: " + str(paths[field]))
                row["status"] = "REGULAR_EXTENT_OBSERVED"
            except FileNotFoundError:
                row["status"] = "MISSING"
                missing.append(str(paths[field]))
        if missing:
            report.update(status="ORIGINAL_EMBEDDING_TABLES_UNAVAILABLE", missing_paths=missing,
                next="Use these original YAML resources on the resource host; transfer only selected original rows and the receipt, without exporting or copying a whole model")
        else:
            report["stage"] = "compare-selected-rows"
            selected, table_data = sorted(set(ids) | {token}), {}
            with ExitStack() as stack:
                descriptors = []
                for row in tables:
                    fd = os.open(row["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    stack.callback(os.close, fd)
                    if _stamp(os.fstat(fd)) != row["stat"]:
                        raise ValueError("original embedding table changed before row read")
                    descriptors.append((fd, row))
                for selected_token in selected:
                    values = []
                    for fd, row in descriptors:
                        raw = os.pread(fd, row["row_bytes"], selected_token * row["row_bytes"])
                        if len(raw) != row["row_bytes"]:
                            raise ValueError("selected original embedding row was truncated")
                        report["observed_payload_reads"]["table_bytes"] += len(raw)
                        values.append(raw)
                    factor, = struct.unpack("<f", values[1])
                    if not math.isfinite(factor) or factor < 0:
                        raise ValueError("selected original embedding scale must be finite and nonnegative")
                    table_data[selected_token] = values
                report["selected_rows"] = [{"token_id": item, "used_by_prefix": item in ids,
                    "used_by_next_token": item == token,
                    "weight_sha256": hashlib.sha256(table_data[item][0]).hexdigest(),
                    "scale_sha256": hashlib.sha256(table_data[item][1]).hexdigest(),
                    "scale_value": struct.unpack("<f", table_data[item][1])[0]} for item in selected]
                mismatches = []
                for position, prefix_token in enumerate(ids):
                    for field, raw, width, table_index in (("inputs_embeds", embedding, _HIDDEN, 0), ("embed_scale", scales, 4, 1)):
                        if raw[position * width:(position + 1) * width] != table_data[prefix_token][table_index]:
                            mismatches.append({"prefix_position": position, "token_id": prefix_token, "field": field})
                report["prefix_comparison"] = {"positions": 64, "weight_bytes": len(embedding), "scale_bytes": len(scales),
                    "status": "BYTE_IDENTICAL" if not mismatches else "DIFFERENT", "mismatches": mismatches}
                for fd, row in descriptors:
                    if _stamp(os.fstat(fd)) != row["stat"] or _safe(row["path"])[1] != row["stat"]:
                        raise ValueError("original embedding table changed during selected-row reads")
                for row in originals:
                    if _safe(row["path"])[1] != row["stat"]:
                        raise ValueError("successful target input/output changed during embedding inspection")
                for row in bindings:
                    if _read_small(row["path"], 8 << 20)[1] != row:
                        raise ValueError("bound embedding inspection metadata changed during row validation")
                if mismatches:
                    report.update(status="ORIGINAL_EMBEDDING_ROWS_DIFFER",
                        next="Resolve the original table/input-token pairing; no next-token embedding files were created")
                else:
                    report["next_token_inputs"] = [
                        _write(out, "next-token/inputs_embeds.bin", table_data[token][0], dtype="INT8", shape=[1, 1, _HIDDEN]),
                        _write(out, "next-token/embed_scale.bin", table_data[token][1], dtype="FLOAT32", shape=[1, 1, 1]),
                        _write(out, "next-token/input_ids.bin", struct.pack("<i", token), dtype="INT32", shape=[1, 1])]
                    report.update(status=_READY, next="Keep these exact original INT8 row/FP32 scale bytes; establish decode controls and all80 state edges before model execution")
                report["source_preservation"] = "UNCHANGED_METADATA_PREFIX_INPUTS_LOGITS_AND_SELECTED_TABLE_IDENTITIES"
        report["stage"] = "complete"
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, struct.error) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["source_bindings"] = bindings
    save_json(out / "report.json", report)
    return report

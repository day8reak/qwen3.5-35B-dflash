"""Bind original selected embedding rows to one retained decode1 execution.

Only the original source token row and the bound logits winner row are read.
This does not multiply scales, reconstruct embeddings, load a model or establish
the trial's control meanings, logical history, state feedback or accuracy.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import math
import os
from pathlib import Path
import struct

from .artifacts import output_path, save_json
from .original_embedding_probe import _bound_bytes, _config, _HIDDEN, _TABLES, _VOCAB, _write
from .target_compile_failure import _read_small, _relative, _safe, _stamp
from .target_decode_output_inventory import validate_target_decode_output_inventory
from .target_prefill_output_inventory import _json, _numerical


_READY = "ORIGINAL_DECODE_EMBEDDING_ROWS_BOUND_NEXT_TOKEN_PREPARED"
_PRESERVED = "UNCHANGED_DECODE_METADATA_SOURCE_IDS_INPUT_ROWS_LOGITS_AND_SELECTED_TABLE_IDENTITIES"


def _input(plan, root, name, semantic, dtype, size):
    path = root / (semantic + ".bin")
    rows = [row for row in plan["inputs"] if isinstance(row, dict) and row.get("name") == name]
    sources = [row for row in plan.get("source_inputs", []) if isinstance(row, dict)
               and row.get("path") == str(path)]
    if (len(rows) != 1 or len(sources) != 1 or rows[0].get("runtime_tensor_name") != name
            or rows[0].get("semantic") != semantic or rows[0].get("dtype") != dtype
            or rows[0].get("source") != str(path) or rows[0].get("source_filename") != path.name
            or rows[0].get("bytes") != size or rows[0].get("sha256") != sources[0].get("sha256")):
        raise ValueError("missing unique executed decode input binding: " + semantic)
    return _bound_bytes(path, sources[0], size)


def _unchanged(bindings, payloads, tables):
    for row in payloads:
        if _safe(row["path"])[1] != row["stat"]:
            raise ValueError("executed decode input/output changed during selected-row inspection")
    for row in bindings:
        if _read_small(row["path"], 32 << 20)[1] != row:
            raise ValueError("bound decode embedding metadata changed during selected-row inspection")
    for row in tables:
        if row.get("status") == "REGULAR_EXTENT_OBSERVED" and _safe(row["path"])[1] != row["stat"]:
            raise ValueError("original embedding table changed during selected-row inspection")


def inspect_original_decode_embeddings(inspection_dir, config_yaml, output_dir):
    """Compare one executed source row, then retain an exact original next row.

    The source token ID is freshly read from the trial manifest's original
    directory. It was not part of the trial inputs. Exact table-row comparison
    establishes this selected token/embedding pairing only, without proving the
    original table's entire contents or the trial's state and control history.
    """
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "artifact_kind": "qwen35-original-decode-embedding-row-probe",
        "status": "FAILED", "stage": "bind-executed-decode", "output": str(out),
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "vendor_import": "NOT_PERFORMED",
        "model_payload_access": "NONE", "table_payload_access": "SELECTED_ROWS_ONLY",
        "full_table_hash": "NOT_PERFORMED", "numerical_validation": "NOT_RUN_NO_REFERENCE",
        "logical_history": "UNVERIFIED", "runtime_control_recipe": "UNVERIFIED", "state_feedback": "UNVERIFIED",
        "state_continuity": "NOT_VALIDATED", "next_token_inputs": [], "source_preservation": "NOT_CHECKED",
        "read_budgets": {"table_payload_maximum_bytes": 2 * (_HIDDEN + 4),
            "source_input_payload_maximum_bytes": 2 * _HIDDEN + 12, "bound_logits_maximum_bytes": _VOCAB * 4,
            "scope": "Direct row-proof reads; the inventory validator separately rehashes its retained input/output evidence"},
        "scope": "Selected original INT8 row and separate FP32 scale lookup only. No requantization, multiplication, controls, state construction, history or generation-readiness proof."}
    bindings, payloads, tables = [], [], []
    try:
        inspection, _ = _safe(inspection_dir, directory=True)
        prior = _json(inspection / "report.json", bindings)
        if validate_target_decode_output_inventory(inspection) != prior:
            raise ValueError("selected decode inspection changed during fresh validation")
        bindings.extend(prior["evidence_bindings"])
        execution, _ = _safe(prior["execution"], directory=True)
        plan = _json(execution / "plan.json", bindings)
        if bindings[-1] not in prior["evidence_bindings"]:
            raise ValueError("decode plan changed after fresh inventory validation")
        input_root, _ = _safe(plan["input_dir"], directory=True)
        trial_binding = prior.get("trial_inputs", {})
        receipt = trial_binding.get("manifest")
        if (trial_binding.get("status") != "FRESHLY_REVALIDATED_UNVERIFIED_EXPERIMENT"
                or not isinstance(receipt, dict) or Path(receipt.get("path", "")).name != "manifest.json"):
            raise ValueError("retained decode trial manifest is required to bind the original source token directory")
        trial = _json(receipt["path"], bindings)
        if bindings[-1] != receipt or trial.get("input_dir") != str(input_root):
            raise ValueError("decode trial manifest/input directory differs from the executed source")
        source_root, _ = _safe(trial["source_input_dir"], directory=True)
        ids_raw, ids_identity = _read_small(source_root / "input_ids.bin", 4)
        if len(ids_raw) != 4:
            raise ValueError("original source input_ids.bin must contain exactly one INT32 token")
        source_token, = struct.unpack("<i", ids_raw)
        if not 0 <= source_token < _VOCAB:
            raise ValueError("original decode source token is outside the original embedding vocabulary")
        embedding, embedding_identity = _input(plan, input_root, "arg1_1", "inputs_embeds", "INT8", _HIDDEN)
        scales, scale_identity = _input(plan, input_root, "arg0_1", "embed_scale", "FLOAT32", 4)
        payloads.extend((ids_identity, embedding_identity, scale_identity))
        # These are the unchanged original files copied into the actual trial.
        original_inputs = []
        for semantic, identity, raw in (("inputs_embeds", embedding_identity, embedding),
                                         ("embed_scale", scale_identity, scales)):
            originals = [row for row in trial["source_bindings"] if row.get("file") == semantic + ".bin"]
            copies = [row for row in trial["copied_inputs"] if row.get("file") == semantic + ".bin"]
            if (len(originals) != 1 or len(copies) != 1 or copies[0].get("source") != originals[0].get("path")
                    or copies[0].get("path") != "inputs/" + semantic + ".bin"
                    or originals[0].get("path") != str(source_root / (semantic + ".bin"))
                    or any(copies[0].get(key) != identity[key] or originals[0].get(key) != identity[key]
                           for key in ("bytes", "sha256"))):
                raise ValueError("executed decode embedding source is not an unchanged original trial copy")
            source_bytes, original_identity = _bound_bytes(originals[0]["path"], originals[0], len(raw))
            if source_bytes != raw:
                raise ValueError("original decode embedding differs from the actual executed trial bytes")
            payloads.append(original_identity)
            original_inputs.append(original_identity)
        logits = prior["bound_tensors"]["logits"]
        index = plan["logits_runtime_output_index"]
        desc = prior["expected_runtime_outputs"][index]
        if (logits.get("runtime_index") != index or logits.get("bytes") != _VOCAB * 4
                or desc.get("dtype") != "FLOAT32" or desc.get("bytes") != _VOCAB * 4):
            raise ValueError("selected bound decode logits do not match the executed output descriptor")
        logits_path = _relative(execution / "board", logits["path"])
        observed = _numerical(logits_path, desc, logits)
        greedy = observed.get("argmax", {})
        if observed != logits.get("statistics") or greedy.get("status") != "OBSERVED_ALL_FINITE":
            raise ValueError("fresh finite bound decode logits and argmax differ from the selected inspection")
        next_token = greedy.get("token_id")
        if type(next_token) is not int or not 0 <= next_token < _VOCAB:
            raise ValueError("bound decode greedy token is outside the original embedding vocabulary")
        payloads.append({"path": str(logits_path), **{key: logits[key] for key in ("bytes", "sha256", "stat")}})
        report.update(inspection=str(inspection), execution=str(execution), trial_input_receipt=receipt,
            experimental_input_assumption=trial_binding["assumption"], selected_source_token_id=source_token,
            selected_next_token_id=next_token, token_selection="FRESH_SHA_BOUND_ALL_FINITE_DECODE_LOGITS_ARGMAX_FIRST_INDEX",
            greedy_observation=greedy, original_input_ids={**ids_identity, "dtype": "INT32", "shape": [1, 1],
                "byte_order": "little", "values": [source_token], "prior_trial_binding": "EXTRA_FILE_NOT_PREVIOUSLY_READ"},
            executed_embedding=embedding_identity, executed_scale=scale_identity, original_trial_inputs=original_inputs,
            bound_logits=payloads[-1], observed_payload_reads={"bound_logits_bytes": _VOCAB * 4,
                "source_input_bytes": len(ids_raw) + 2 * (len(embedding) + len(scales)), "table_bytes": 0})
        report["stage"] = "open-original-embedding-tables"
        paths, config_identity = _config(config_yaml, bindings)
        report.update(original_config=config_identity, tables=tables)
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
            _unchanged(bindings, payloads, tables)
            report.update(status="ORIGINAL_EMBEDDING_TABLES_UNAVAILABLE", missing_paths=missing,
                source_preservation=_PRESERVED,
                next="Resolve these exact configured original tables on their resource host; no fallback paths or reconstructed embeddings were used")
        else:
            report["stage"] = "compare-selected-original-rows"
            table_data = {}
            with ExitStack() as stack:
                descriptors = []
                for row in tables:
                    fd = os.open(row["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    stack.callback(os.close, fd)
                    if _stamp(os.fstat(fd)) != row["stat"]:
                        raise ValueError("original embedding table changed before selected-row read")
                    descriptors.append((fd, row))
                for token in sorted({source_token, next_token}):
                    values = []
                    for fd, row in descriptors:
                        raw = os.pread(fd, row["row_bytes"], token * row["row_bytes"])
                        if len(raw) != row["row_bytes"]:
                            raise ValueError("selected original embedding row was truncated")
                        values.append(raw)
                        report["observed_payload_reads"]["table_bytes"] += len(raw)
                    factor, = struct.unpack("<f", values[1])
                    if not math.isfinite(factor) or factor < 0:
                        raise ValueError("selected original embedding scale must be finite and nonnegative")
                    table_data[token] = values
                report["selected_rows"] = [{"token_id": token, "used_by_source": token == source_token,
                    "used_by_next_token": token == next_token,
                    "weight_sha256": hashlib.sha256(values[0]).hexdigest(),
                    "scale_sha256": hashlib.sha256(values[1]).hexdigest(),
                    "scale_value": struct.unpack("<f", values[1])[0]} for token, values in sorted(table_data.items())]
                mismatches = [semantic for semantic, raw, original in (
                    ("inputs_embeds", embedding, table_data[source_token][0]),
                    ("embed_scale", scales, table_data[source_token][1])) if raw != original]
                report["source_row_comparison"] = {"positions": 1, "weight_bytes": len(embedding),
                    "scale_bytes": len(scales), "status": "DIFFERENT" if mismatches else "BYTE_IDENTICAL", "mismatches": mismatches}
                for fd, row in descriptors:
                    if _stamp(os.fstat(fd)) != row["stat"]:
                        raise ValueError("original embedding table changed during selected-row read")
                _unchanged(bindings, payloads, tables)
                if mismatches:
                    report.update(status="ORIGINAL_DECODE_EMBEDDING_ROWS_DIFFER",
                        next="Resolve the original source token/table/input pairing; no next-token embedding files were created")
                else:
                    report["next_token_inputs"] = [
                        _write(out, "next-token/inputs_embeds.bin", table_data[next_token][0], dtype="INT8", shape=[1, 1, _HIDDEN]),
                        _write(out, "next-token/embed_scale.bin", table_data[next_token][1], dtype="FLOAT32", shape=[1, 1, 1]),
                        _write(out, "next-token/input_ids.bin", struct.pack("<i", next_token), dtype="INT32", shape=[1, 1])]
                    report.update(status=_READY,
                        next="Retain these exact original row/scale bytes; establish logical history, decode controls and all80 state edges separately before continuation")
                report["source_preservation"] = _PRESERVED
        report["stage"] = "complete"
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, struct.error) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["source_bindings"] = bindings
    save_json(out / "report.json", report)
    return report

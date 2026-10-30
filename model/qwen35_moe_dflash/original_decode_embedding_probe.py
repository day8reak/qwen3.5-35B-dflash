"""Bind original selected embedding rows to one retained decode1 execution.

Only an explicit bounded source-ID candidate set and the logits winner row are read.
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
from .metadata_revalidation import metadata_identity_matches
from .payload_revalidation import payload_identity_matches
from .original_embedding_probe import _bound_bytes, _config, _HIDDEN, _TABLES, _VOCAB, _write
from .target_compile_failure import _read_small, _relative, _safe, _stamp
from .target_decode_output_inventory import validate_target_decode_output_inventory
from .target_prefill_output_inventory import _hash_retrieved, _json, _numerical
from .target_prefill_smoke_plan import _identity


_READY = "ORIGINAL_DECODE_EMBEDDING_ROWS_BOUND_NEXT_TOKEN_PREPARED"
_PRESERVED = "UNCHANGED_DECODE_METADATA_SOURCE_IDS_INPUT_ROWS_LOGITS_AND_SELECTED_TABLE_IDENTITIES"
_MAX_CANDIDATE_IDS = 1024


def _input(plan, root, name, semantic, dtype, size, *, payload_revalidation_observations=None):
    path = root / (semantic + ".bin")
    rows = [row for row in plan["inputs"] if isinstance(row, dict) and row.get("name") == name]
    sources = [row for row in plan.get("source_inputs", []) if isinstance(row, dict)
               and row.get("path") == str(path)]
    if (len(rows) != 1 or len(sources) != 1 or rows[0].get("runtime_tensor_name") != name
            or rows[0].get("semantic") != semantic or rows[0].get("dtype") != dtype
            or rows[0].get("source") != str(path) or rows[0].get("source_filename") != path.name
            or rows[0].get("bytes") != size or rows[0].get("sha256") != sources[0].get("sha256")):
        raise ValueError("missing unique executed decode input binding: " + semantic)
    before = _identity(path.stat())
    raw, identity = _read_small(path, size)
    if _identity(path.stat()) != before:
        raise ValueError("executed decode input identity changed during bounded read")
    current = {"path": str(path), "bytes": len(raw), "sha256": identity["sha256"], "identity": before}
    if not payload_identity_matches(sources[0], current, role="decode-input", observations=payload_revalidation_observations):
        raise ValueError("executed decode input differs in content or nonctime identity fields")
    return raw, identity


def _check_current_metadata_bindings(bindings):
    current = {}
    for row in bindings:
        if row["path"] in current and current[row["path"]] != row:
            raise ValueError("metadata changed after its first current read: " + row["path"])
        current[row["path"]] = row


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


def inspect_original_decode_embeddings(inspection_dir, config_yaml, output_dir, *, match_source_id_candidates=False,
                                       revalidation_observations=None, metadata_revalidation_observations=None,
                                       payload_revalidation_observations=None):
    """Compare one executed source row, then retain an exact original next row.

    Source IDs are freshly read from the trial manifest's original directory.
    An explicitly enabled bounded vector is a candidate set, not an ordering
    recipe: exactly one distinct ID must match the executed row and scale.
    Neither mode proves logical history, position semantics or accuracy.
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
    observations = [] if revalidation_observations is None else revalidation_observations
    observation_start = len(observations) if isinstance(observations, list) else 0
    metadata_observations = [] if metadata_revalidation_observations is None else metadata_revalidation_observations
    metadata_start = len(metadata_observations) if isinstance(metadata_observations, list) else 0
    payload_observations = [] if payload_revalidation_observations is None else payload_revalidation_observations
    payload_start = len(payload_observations) if isinstance(payload_observations, list) else 0
    try:
        if not isinstance(observations, list):
            raise ValueError("revalidation_observations must be a list or None")
        if not isinstance(metadata_observations, list):
            raise ValueError("metadata_revalidation_observations must be a list or None")
        if not isinstance(payload_observations, list):
            raise ValueError("payload_revalidation_observations must be a list or None")
        if type(match_source_id_candidates) is not bool:
            raise ValueError("match_source_id_candidates must be an explicit boolean")
        maximum_ids = _MAX_CANDIDATE_IDS if match_source_id_candidates else 1
        report["match_source_id_candidates"] = match_source_id_candidates
        report["read_budgets"].update(table_payload_maximum_bytes=(maximum_ids + 1) * (_HIDDEN + 4),
            configured_table_payload_cap_bytes=(maximum_ids + 1) * (_HIDDEN + 4),
            source_input_payload_maximum_bytes=maximum_ids * 4 + 2 * (_HIDDEN + 4))
        report["stage"] = "read-decode-output-inspection"
        inspection, _ = _safe(inspection_dir, directory=True)
        prior = _json(inspection / "report.json", bindings)
        report["stage"] = "validate-decode-output-inspection"
        if validate_target_decode_output_inventory(inspection, revalidation_observations=observations,
                metadata_revalidation_observations=metadata_observations,
                payload_revalidation_observations=payload_observations) != prior:
            raise ValueError("selected decode inspection changed during fresh validation")
        for recorded in prior["evidence_bindings"]:
            fresh = _read_small(recorded["path"], 32 << 20)[1]
            if not metadata_identity_matches(recorded, fresh, role="embedding-audit-metadata", observations=metadata_observations):
                raise ValueError("historical decode metadata differs beyond freshly verified ctime")
            bindings.append(fresh)
        _check_current_metadata_bindings(bindings)
        report["stage"] = "read-executed-decode-plan"
        execution, _ = _safe(prior["execution"], directory=True)
        plan = _json(execution / "plan.json", bindings)
        _check_current_metadata_bindings(bindings)
        if not any(metadata_identity_matches(row, bindings[-1], role="decode-execution-metadata", observations=metadata_observations)
                for row in prior["evidence_bindings"] if row.get("path") == bindings[-1]["path"]):
            raise ValueError("decode plan changed after fresh inventory validation")
        input_root, _ = _safe(plan["input_dir"], directory=True)
        trial_binding = prior.get("trial_inputs", {})
        receipt = trial_binding.get("manifest")
        if (trial_binding.get("status") != "FRESHLY_REVALIDATED_UNVERIFIED_EXPERIMENT"
                or not isinstance(receipt, dict) or Path(receipt.get("path", "")).name != "manifest.json"):
            raise ValueError("retained decode trial manifest is required to bind the original source token directory")
        report["stage"] = "read-executed-trial-manifest"
        trial = _json(receipt["path"], bindings)
        _check_current_metadata_bindings(bindings)
        if (not metadata_identity_matches(receipt, bindings[-1], role="decode-trial-manifest", observations=metadata_observations)
                or trial.get("input_dir") != str(input_root)):
            raise ValueError("decode trial manifest/input directory differs from the executed source")
        report["stage"] = "inspect-original-source-id-extent"
        source_root, _ = _safe(trial["source_input_dir"], directory=True)
        ids_path = source_root / "input_ids.bin"
        diagnosis = {"path": str(ids_path), "status": "PATH_NOT_YET_OBSERVED",
            "expected_bytes": None if match_source_id_candidates else 4,
            "maximum_read_bytes": maximum_ids * 4, "required_multiple_bytes": 4,
            "declared_encoding": "BOUNDED_INT32_LITTLE_ENDIAN_CANDIDATES" if match_source_id_candidates else "ONE_INT32_LITTLE_ENDIAN",
            "source_path_binding": "EXECUTED_TRIAL_MANIFEST_SOURCE_DIRECTORY",
            "payload_access": "NOT_READ", "token_selection": "NONE"}
        report["source_id_diagnosis"] = diagnosis
        _, ids_stat = _safe(ids_path)
        diagnosis.update(bytes=ids_stat[2], stat=ids_stat,
            candidate_int32_elements=ids_stat[2] // 4 if ids_stat[2] % 4 == 0 else None)
        if not 0 < ids_stat[2] <= maximum_ids * 4 or ids_stat[2] % 4:
            diagnosis["status"] = "UNSUPPORTED_SOURCE_ID_CANDIDATE_EXTENT" if match_source_id_candidates else "UNSUPPORTED_SINGLE_DECODE_ID_EXTENT"
            report["next"] = "Retain this stat-only source-ID diagnosis; establish its token-to-executed-embedding correspondence before selecting any vector element"
            raise ValueError(f"original decode source ID extent is unsupported: path={ids_path}; "
                f"observed_bytes={ids_stat[2]}; expected_bytes={diagnosis['expected_bytes']}; "
                f"required_multiple_bytes=4; maximum_read_bytes={maximum_ids * 4}; payload_not_read")
        diagnosis["status"] = "BOUNDED_CANDIDATE_EXTENT_OBSERVED" if ids_stat[2] > 4 else "SINGLE_TOKEN_EXTENT_OBSERVED"
        report["stage"] = "read-original-source-id-candidates" if ids_stat[2] > 4 else "read-original-single-source-id"
        ids_raw, ids_identity = _read_small(ids_path, maximum_ids * 4)
        if ids_identity["stat"] != ids_stat:
            raise ValueError("original source ID file changed after stat-only extent diagnosis: " + str(ids_path))
        ids = list(struct.unpack("<" + "i" * (len(ids_raw) // 4), ids_raw))
        diagnosis["payload_access"] = "READ_BOUNDED_SOURCE_ID_CANDIDATES" if len(ids) > 1 else "READ_SINGLE_INT32_ONLY"
        diagnosis["sha256"] = ids_identity["sha256"]
        if any(not 0 <= token < _VOCAB for token in ids):
            raise ValueError("original decode source token is outside the original embedding vocabulary")
        positions = {}
        for position, token in enumerate(ids):
            positions.setdefault(token, []).append(position)
        report["read_budgets"]["table_payload_maximum_bytes"] = (len(positions) + 1) * (_HIDDEN + 4)
        source_token = ids[0] if len(ids) == 1 else None
        report["stage"] = "read-executed-embedding-row"
        embedding, embedding_identity = _input(plan, input_root, "arg1_1", "inputs_embeds", "INT8", _HIDDEN,
            payload_revalidation_observations=payload_observations)
        report["stage"] = "read-executed-embedding-scale"
        scales, scale_identity = _input(plan, input_root, "arg0_1", "embed_scale", "FLOAT32", 4,
            payload_revalidation_observations=payload_observations)
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
            report["stage"] = "read-original-trial-copy-" + semantic
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
        report["stage"] = "read-bound-decode-logits"
        historical_logits = {"path": str(logits_path), **{key: logits[key] for key in ("bytes", "sha256", "stat")}}
        current_logits, logit_passes = historical_logits, 1
        if _safe(logits_path)[1] != logits["stat"]:
            fresh, _ = _hash_retrieved(logits_path, logits)
            current_logits = {"path": str(logits_path), **fresh}
            if not payload_identity_matches(historical_logits, current_logits, role="decode-output", observations=payload_observations):
                raise ValueError("bound decode logits differ beyond freshly verified historical ctime")
            logit_passes = 2
        observed = _numerical(logits_path, desc, current_logits)
        greedy = observed.get("argmax", {})
        if observed != logits.get("statistics") or greedy.get("status") != "OBSERVED_ALL_FINITE":
            raise ValueError("fresh finite bound decode logits and argmax differ from the selected inspection")
        next_token = greedy.get("token_id")
        if type(next_token) is not int or not 0 <= next_token < _VOCAB:
            raise ValueError("bound decode greedy token is outside the original embedding vocabulary")
        payloads.append(current_logits)
        report.update(inspection=str(inspection), execution=str(execution), trial_input_receipt=receipt,
            experimental_input_assumption=trial_binding["assumption"],
            selected_next_token_id=next_token, token_selection="FRESH_SHA_BOUND_ALL_FINITE_DECODE_LOGITS_ARGMAX_FIRST_INDEX",
            greedy_observation=greedy, original_input_ids={**ids_identity, "dtype": "INT32", "shape": [1, len(ids)],
                "byte_order": "little", "values": ids, "prior_trial_binding": "EXTRA_FILE_NOT_PREVIOUSLY_READ"},
            source_candidate_matching={"mode": "EXPLICIT_BOUNDED_CANDIDATES" if len(ids) > 1 else "SINGLE_SOURCE_ID",
                "method": "EXACT_ORIGINAL_INT8_ROW_AND_SEPARATE_FP32_SCALE_BYTES",
                "candidate_count": len(ids), "unique_candidate_count": len(positions),
                "matching_token_ids": [], "matching_positions": [], "status": "NOT_EVALUATED",
                "uniqueness_scope": "EXPLICIT_SOURCE_ID_CANDIDATES_ONLY", "position_index_base": 0,
                "logical_position_semantics": "UNVERIFIED"},
            executed_embedding=embedding_identity, executed_scale=scale_identity, original_trial_inputs=original_inputs,
            bound_logits=historical_logits, bound_logits_current_read=current_logits,
            observed_payload_reads={"bound_logits_bytes": logit_passes * _VOCAB * 4,
                "source_input_bytes": len(ids_raw) + 2 * (len(embedding) + len(scales)), "table_bytes": 0})
        report["read_budgets"]["bound_logits_maximum_bytes"] = logit_passes * _VOCAB * 4
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
                def read_row(token):
                    values = []
                    for fd, row in descriptors:
                        if report["observed_payload_reads"]["table_bytes"] + row["row_bytes"] > report["read_budgets"]["table_payload_maximum_bytes"]:
                            raise ValueError("selected original row read exceeds the explicit candidate budget")
                        raw = os.pread(fd, row["row_bytes"], token * row["row_bytes"])
                        if len(raw) != row["row_bytes"]:
                            raise ValueError("selected original embedding row was truncated")
                        values.append(raw)
                        report["observed_payload_reads"]["table_bytes"] += len(raw)
                    factor, = struct.unpack("<f", values[1])
                    if not math.isfinite(factor) or factor < 0:
                        raise ValueError("selected original embedding scale must be finite and nonnegative")
                    table_data[token] = values
                for token in sorted(positions):
                    read_row(token)
                matching_ids = [token for token in sorted(positions) if table_data[token] == [embedding, scales]]
                candidate = report["source_candidate_matching"]
                candidate.update(matching_token_ids=matching_ids,
                    matching_positions=[{"token_id": token, "positions": positions[token]} for token in matching_ids],
                    status="UNIQUE_TOKEN_ID_MATCH" if len(matching_ids) == 1 else "NO_MATCH" if not matching_ids else "AMBIGUOUS_TOKEN_IDS")
                if len(matching_ids) == 1:
                    source_token = matching_ids[0]
                    report["selected_source_token_id"] = source_token
                    diagnosis["token_selection"] = "UNIQUE_TOKEN_ID_BY_EXECUTED_ROW_AND_SCALE"
                    if next_token not in table_data:
                        read_row(next_token)
                report["selected_rows"] = [{"token_id": token, "used_by_source": token == source_token,
                    "source_candidate_positions": positions.get(token, []), "used_by_next_token": token == next_token,
                    "weight_sha256": hashlib.sha256(values[0]).hexdigest(),
                    "scale_sha256": hashlib.sha256(values[1]).hexdigest(),
                    "scale_value": struct.unpack("<f", values[1])[0]} for token, values in sorted(table_data.items())]
                if len(ids) == 1:
                    mismatches = [semantic for semantic, raw, original in (
                        ("inputs_embeds", embedding, table_data[ids[0]][0]),
                        ("embed_scale", scales, table_data[ids[0]][1])) if raw != original]
                    report["source_row_comparison"] = {"positions": 1, "weight_bytes": len(embedding),
                        "scale_bytes": len(scales), "status": "DIFFERENT" if mismatches else "BYTE_IDENTICAL", "mismatches": mismatches}
                else:
                    report["source_row_comparison"] = {"positions": len(positions[source_token]) if len(matching_ids) == 1 else 0,
                        "matching_source_positions": positions[source_token] if len(matching_ids) == 1 else [],
                        "weight_bytes": len(embedding), "scale_bytes": len(scales),
                        "status": "BYTE_IDENTICAL" if len(matching_ids) == 1 else candidate["status"],
                        "position_semantics": "CANDIDATE_OCCURRENCES_ONLY_NOT_LOGICAL_HISTORY"}
                for fd, row in descriptors:
                    if _stamp(os.fstat(fd)) != row["stat"]:
                        raise ValueError("original embedding table changed during selected-row read")
                _unchanged(bindings, payloads, tables)
                if len(matching_ids) != 1:
                    status = "ORIGINAL_DECODE_EMBEDDING_ROWS_DIFFER" if len(ids) == 1 else (
                        "ORIGINAL_DECODE_EMBEDDING_CANDIDATES_NO_MATCH" if not matching_ids else "ORIGINAL_DECODE_EMBEDDING_CANDIDATES_AMBIGUOUS")
                    report.update(status=status,
                        next="Resolve the original source token/table/input pairing; no unique executed row/scale match or next-token embedding files were established")
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
    current_observations = observations[observation_start:] if isinstance(observations, list) else []
    report["primary_omc_revalidation_observations"] = current_observations
    if current_observations:
        report["model_payload_access"] = "SMALL_PRIMARY_OMC_REHASHED_FOR_CTIME_ONLY"
    current_metadata = metadata_observations[metadata_start:] if isinstance(metadata_observations, list) else []
    report["metadata_revalidation_observations"] = current_metadata
    if current_metadata:
        report["metadata_identity_revalidation"] = "CTIME_ONLY_FRESH_METADATA_SHA256_OLD_RECEIPTS_UNCHANGED"
    current_payload = payload_observations[payload_start:] if isinstance(payload_observations, list) else []
    report["payload_revalidation_observations"] = current_payload
    if current_payload:
        report["payload_identity_revalidation"] = "CTIME_ONLY_FRESH_PAYLOAD_SHA256_OLD_RECEIPTS_UNCHANGED"
    report["source_bindings"] = bindings
    save_json(out / "report.json", report)
    return report

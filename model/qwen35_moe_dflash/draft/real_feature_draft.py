"""Prepare one complete draft call from retained, explicitly bound target bytes.

The existing embedded model stays in its original bundle.  This directory owns
only three small inputs, the unchanged ABI and a native step plan.  The model
path in that plan describes the later remote deployment, not a local bundle.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.draft.board import _direct_output_binding, _file, _io, _wire
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.common.file_io import _read_small, _relative, _safe
from qwen35_moe_dflash.target.prefill_output_inventory import _json, inspect_target_prefill_outputs


_KIND = "qwen35-real-feature-draft-step"
_STATUS = "REAL_FEATURE_DRAFT_STEP_PREPARED"
_MAX_METADATA = 8 << 20
_FEATURE_BYTES = 64 * 16384 * 2
_CONTROLS = (("effective_length.bin", "<h"), ("allQLen.bin", "<q"), ("output_pos.bin", "<i"))


def _metadata(path):
    bindings = []
    value = _json(path, bindings)
    return value, bindings[0]


def _same_file(path, record, *, maximum=_MAX_METADATA):
    path, before = _safe(path)
    if before[2] == 0 and record.get("bytes") == 0:
        if sha256_file(path) != hashlib.sha256(b"").hexdigest() or _safe(path)[1] != before:
            raise ValueError("empty source identity changed: " + str(path))
        raw, identity = b"", {"path": str(path), "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(), "stat": before}
    else:
        raw, identity = _read_small(path, maximum)
    if any(identity[key] != record.get(key) for key in ("bytes", "sha256")):
        raise ValueError("source SHA256/size changed: " + str(path))
    if "stat" in record and identity["stat"] != record["stat"]:
        raise ValueError("source identity changed: " + str(path))
    if "identity" in record and identity["stat"] != [record["identity"].get(key)
            for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]:
        raise ValueError("recorded original control identity changed: " + str(path))
    return raw, identity


def _facts(report):
    return {key: value for key, value in report.items() if key not in {"output", "report"}}


def _inspection_contract(report):
    if (report.get("artifact_kind") != "qwen35-target-prefill-output-inventory"
            or report.get("status") != "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS"
            or report.get("slot_binding") != "EXPLICIT_COMPLETE_TOOL_LOG_AND_VERIFIED_BYTES"
            or report.get("plan_output_count") != 82 or report.get("actual_retrieved_output_file_count") != 82
            or report.get("raw_extent_multiset_check") != "MATCHED_COUNTS_AND_EXTENTS_ONLY"
            or report.get("logged_slot_mapping", {}).get("status") != "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG"):
        raise ValueError("real draft requires all82 explicitly logged and SHA-bound target outputs")
    feature = report.get("bound_tensors", {}).get("feature", {})
    stats = feature.get("statistics", {})
    if (feature.get("runtime_index") != 81 or feature.get("bytes") != _FEATURE_BYTES
            or stats.get("dtype") != "FLOAT16" or stats.get("byte_order") != "little"
            or stats.get("shape") != [1, 1, 64, 16384] or stats.get("elements") != 64 * 16384
            or stats.get("finite_count") != 64 * 16384
            or any(stats.get(key) != 0 for key in ("nan_count", "positive_infinity_count", "negative_infinity_count"))):
        raise ValueError("actual bound target feature must be finite FP16 [1,1,64,16384]")
    ids = report.get("original_input_ids", {})
    if (ids.get("status") != "OBSERVED" or ids.get("dtype") != "int32"
            or ids.get("byte_order") != "little" or ids.get("shape") != [1, 64]
            or ids.get("bytes") != 256 or ids.get("matches_planned_input_dir") is not True
            or not isinstance(ids.get("values"), list) or len(ids["values"]) != 64
            or any(type(value) is not int or not 0 <= value < 248320 for value in ids["values"])):
        raise ValueError("actual original64 INT32 IDs from the successful prefill input directory are required")
    return feature, ids


def _bundle(root, observed):
    """Validate the existing direct model without copying it or reading goldens."""
    root, _ = _safe(root, directory=True)
    manifest, identity = _metadata(root / "manifest.json")
    observed_manifest = observed.get("manifest", {})
    if (observed.get("status") != "MANIFEST_OBSERVED_ONLY"
            or any(identity[key] != observed_manifest.get(key) for key in ("path", "bytes", "sha256", "stat"))):
        raise ValueError("draft source manifest differs from the target-output inspection")
    if (manifest.get("artifact_kind") != "qwen35-standalone-draft-direct-bundle"
            or manifest.get("status") != "PREPARED" or manifest.get("sequence_length") != 64
            or manifest.get("platform") != "kirinx90" or manifest.get("weight_storage") != "embedded-stream"
            or manifest.get("source_identity") != "MATCHED_EXPORT_AND_REFERENCE"
            or not isinstance(manifest.get("direct_spec"), list)
            or not isinstance(manifest.get("ddk_id"), str) or not manifest["ddk_id"]):
        raise ValueError("existing model must be the prepared embedded-stream direct draft with capacity64")
    cfg = Qwen35DFlashConfig.from_dict(manifest["draft_config"])
    if (cfg.num_hidden_layers, cfg.feature_size, cfg.hidden_size, cfg.block_size, cfg.vocab_size,
            cfg.num_target_layers, cfg.target_layer_ids) != (6, 16384, 2048, 16, 248320, 40, (1, 6, 11, 16, 22, 27, 32, 37)):
        raise ValueError("real features require the locked full six-layer draft and all eight target taps")
    inputs, outputs = _io(64, direct_spec=manifest["direct_spec"])
    if len(outputs) != 2 or manifest.get("outputs") != outputs or len(manifest.get("inputs", [])) != 3:
        raise ValueError("existing direct draft must expose exactly three inputs and two outputs")
    if any(any(row.get(key) != value for key, value in desc.items())
           for row, desc in zip(manifest["inputs"], inputs)):
        raise ValueError("existing draft input descriptor changed")
    bindings = [identity]
    abi_path = _file(root, manifest["draft_abi"])
    abi_raw, abi_identity = _read_small(abi_path, 1 << 20)
    abi, _ = _metadata(abi_path)
    if (abi.get("artifact_kind") != "qwen35-standalone-draft-direct-abi"
            or abi.get("inputs") != inputs or abi.get("outputs") != outputs
            or abi.get("sequence_length") != 64 or abi.get("state_policy") != "prefix-recompute"
            or abi.get("draft_config") != manifest["draft_config"]
            or abi.get("direct_spec") != manifest["direct_spec"]
            or abi.get("direct_output_binding") != manifest.get("direct_output_binding")
            or abi.get("source_compiled_omc") != manifest.get("source_compiled_omc")):
        raise ValueError("existing direct ABI differs from its source manifest")
    binding = manifest.get("direct_output_binding")
    precision = ("DECLARED_SOURCE_DTYPES" if binding is None else _direct_output_binding(
        binding.get("declared_spec"), manifest["direct_spec"], binding, manifest.get("source_compiled_omc")))
    if manifest.get("source_output_precision") != precision or abi.get("source_output_precision") != precision:
        raise ValueError("existing direct ABI precision receipt changed")
    bindings.append(abi_identity)
    model = manifest.get("model", {})
    if (model.get("path") != "model/model.omc" or type(model.get("bytes")) is not int
            or not 0 < model["bytes"] <= (1 << 32) - 1 or manifest.get("model_payloads") != [model]):
        raise ValueError("real draft requires one embedded primary OMC and no external model sidecars")
    model_path, before = _safe(_relative(root, model["path"]))
    if before[2] != model["bytes"] or sha256_file(model_path) != model.get("sha256") or _safe(model_path)[1] != before:
        raise ValueError("existing draft model SHA256/size/identity changed")
    model_source = {"path": str(model_path), "bytes": model["bytes"], "sha256": model["sha256"], "stat": before}
    packing_path = _file(root, manifest["weight_packing"])
    packing, packing_identity = _metadata(packing_path)
    if (packing.get("status") != "EMBEDDED"
            or any(packing.get("output", {}).get(key) != model[key] for key in ("bytes", "sha256"))):
        raise ValueError("existing embedded model is not bound to its packing receipt")
    bindings.append(packing_identity)
    step_path = _file(root, manifest["step_plan"])
    step, step_identity = _read_small(step_path, 1 << 20)
    if step != _wire(manifest):
        raise ValueError("existing source draft step differs from its manifest")
    bindings.append(step_identity)
    return manifest, model_source, abi_raw, bindings


def _controls(plan, input_root):
    records = plan.get("source_inputs")
    if not isinstance(records, list):
        raise ValueError("successful target plan must bind original control SHA256 records")
    bindings, result = [], []
    for filename, dtype in _CONTROLS:
        path = input_root / filename
        matches = [row for row in records if isinstance(row, dict) and row.get("path") == str(path)]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous original control binding: " + filename)
        raw, identity = _same_file(path, matches[0], maximum=8)
        if len(raw) != struct.calcsize(dtype) or struct.unpack(dtype, raw) != (64,):
            raise ValueError("all64 original tokens must be valid: " + filename)
        bindings.append(identity)
        result.append({**identity, "encoding": dtype, "value": 64})
    return result, bindings


def _source_bindings(fresh, execution, records):
    for row in fresh["output_files"]:
        records.append({"path": str(_relative(execution / "board", row["path"])),
            **{key: row[key] for key in ("bytes", "sha256", "stat")}})
    for filename, row in fresh["tool_logs"].items():
        records.append({"path": str(execution / "board" / filename),
            **{key: row[key] for key in ("bytes", "sha256", "stat")}})
    unique = {}
    for row in records:
        if row["path"] in unique and unique[row["path"]] != row:
            raise ValueError("conflicting source identities in real-feature preparation")
        unique[row["path"]] = row
    return list(unique.values())


def _target_plan(execution, input_root, source):
    plan, _ = _metadata(execution / "plan.json")
    proof, _ = _metadata(execution / "board/runtime-prefill-proof.json")
    if (plan != proof or plan.get("input_dir") != str(input_root)
            or plan.get("ddk_id") != source["ddk_id"]
            or any(not isinstance(plan.get(key), str) or not plan[key] for key in ("runtime_version", "firmware_id"))):
        raise ValueError("original IDs, DDK and runtime must match the successful target execution proof")
    return plan


def _write(out, relative, raw, role):
    path = _relative(out, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    _, identity = _read_small(path, max(_FEATURE_BYTES, _MAX_METADATA))
    return {"path": relative, "bytes": identity["bytes"], "sha256": identity["sha256"]}, {
        **identity, "path": relative, "local_path": str(path), "role": role}


def prepare_real_feature_draft(inspection_dir, bundle_dir, output_dir):
    """Prepare byte-identical real inputs and a remote-only native step plan."""
    prior_root, _ = _safe(inspection_dir, directory=True)
    prior, prior_identity = _metadata(prior_root / "report.json")
    _, ids = _inspection_contract(prior)
    input_root, _ = _safe(Path(ids["path"]).parent, directory=True)
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    fresh = inspect_target_prefill_outputs(prior["execution"], out / "source-validation",
        draft_bundle=bundle_dir, input_dir=input_root)
    feature, ids = _inspection_contract(fresh)
    if _facts(prior) != _facts(fresh):
        raise ValueError("fresh target-output inspection differs from the selected prior inspection")
    source, model_source, abi_raw, source_bindings = _bundle(bundle_dir, fresh["draft_bundle"])
    execution, _ = _safe(fresh["execution"], directory=True)
    plan = _target_plan(execution, input_root, source)
    controls, control_bindings = _controls(plan, input_root)
    feature_raw, feature_identity = _same_file(_relative(execution / "board", feature["path"]), feature,
        maximum=_FEATURE_BYTES)
    if len(feature_raw) != _FEATURE_BYTES or not bool(np.isfinite(np.frombuffer(feature_raw, dtype="<f2")).all()):
        raise ValueError("real target feature contains nonfinite values")
    ids_raw, ids_identity = _same_file(ids["path"], ids, maximum=256)
    values = list(struct.unpack("<64i", ids_raw))
    if values != ids["values"]:
        raise ValueError("original IDs changed after source inspection")
    fresh_identity = _read_small(out / "source-validation/report.json", _MAX_METADATA)[1]
    source_bindings = _source_bindings(fresh, execution, source_bindings + [prior_identity,
        fresh_identity, *fresh["evidence_bindings"], *control_bindings, feature_identity, ids_identity])
    manifest = {"schema_version": 1, "artifact_kind": _KIND, "status": _STATUS, "output": str(out),
        "sequence_length": 64, "past_tokens": 0, "platform": source["platform"], "ddk_id": source["ddk_id"],
        "runtime_version": plan.get("runtime_version"), "firmware_id": plan.get("firmware_id"),
        "draft_config": source["draft_config"], "direct_spec": source["direct_spec"],
        "weight_storage": "embedded-stream", "model": source["model"], "model_source": model_source,
        "model_upload_source": str(model_source["path"]), "source_bundle": str(Path(model_source["path"]).parents[1]),
        "source_inspection": prior_identity, "source_bindings": source_bindings, "inputs": [], "outputs": source["outputs"],
        "upload_payloads": [], "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "numerical_validation": "NOT_RUN_NO_REAL_FEATURE_REFERENCE", "synthetic_golden_comparison": "NOT_REQUESTED",
        "state_feedback": "NOT_RUN", "model_host_copy": "NONE_EXTERNAL_ORIGINAL_BUNDLE_SOURCE",
        "feature_provenance": {"source": feature_identity, "runtime_index": feature["runtime_index"],
            "source_shape": [1, 1, 64, 16384], "draft_shape": [1, 64, 16384],
            "transformation": "DROP_ONE_LEADING_SINGLETON_ND_AXIS_BYTES_UNCHANGED",
            "statistics": feature["statistics"], "finite_extremes_policy": "PRESERVED_WITHOUT_CLIPPING_OR_NORMALIZATION"},
        "token_provenance": {"source": ids_identity, "valid_length": 64, "anchor_position": 63,
            "anchor_token_id": values[63], "anchor_source": "ORIGINAL_LAST_INPUT_ID",
            "prefix_policy": "EXACT_ORIGINAL64_NO_TRUNCATION_OR_REBASE", "target_greedy_token_used_as_anchor": False,
            "token_to_target_embedding_equivalence": "UNVERIFIED"},
        "mask_provenance": {"kind": "GENERATED_INT32_ONES", "valid_tokens": 64, "controls": controls,
            "basis": "ORIGINAL_EFFECTIVE_LENGTH_ALLQLEN_OUTPUT_POS_ALL_EQUAL64_WITH_RECORDED_SOURCE_SHA256"},
        "target_greedy_observation": fresh.get("bound_tensors", {}).get("logits", {}).get("statistics", {}).get("argmax"),
        "scope": "One full six-layer draft step from actual target features and original64 IDs. Finite extreme values are preserved. No target verification, state continuity, accuracy, acceptance-rate or speedup claim. The model path in step.tsv is remote-relative; this directory is not a self-contained draft bundle."}
    for desc, raw in zip(_io(64)[0], (feature_raw, ids_raw, struct.pack("<64i", *([1] * 64)))):
        record, upload = _write(out, f"inputs/input_{desc['index']:04d}.bin", raw, desc["name"])
        manifest["inputs"].append({**desc, **record})
        manifest["upload_payloads"].append(upload)
    manifest["draft_abi"], upload = _write(out, "draft-abi.json", abi_raw, "draft_abi")
    manifest["upload_payloads"].append(upload)
    manifest["step_plan"], upload = _write(out, "step.tsv", _wire(manifest), "step_plan")
    manifest["upload_payloads"].append(upload)
    save_json(out / "manifest.json", manifest)
    validate_real_feature_draft(out)
    return manifest


def validate_real_feature_draft(prepared_dir):
    """Rehash prepared payloads and retained evidence immediately before upload.

    This does not rewrite evidence, construct a new fixture, copy a model or
    invoke a device.  Any changed original source requires a fresh preparation.
    """
    out, _ = _safe(prepared_dir, directory=True)
    manifest, _ = _metadata(out / "manifest.json")
    if (manifest.get("artifact_kind") != _KIND or manifest.get("status") != _STATUS
            or manifest.get("output") != str(out) or manifest.get("sequence_length") != 64
            or manifest.get("past_tokens") != 0):
        raise ValueError("expected one prepared real-feature draft step")
    if (out / "model").exists() or (out / "model").is_symlink():
        raise ValueError("real-feature preparation must not contain a copied or linked local model")
    for row in manifest["source_bindings"]:
        _same_file(row["path"], row)
    fresh, fresh_identity = _metadata(out / "source-validation/report.json")
    feature, ids = _inspection_contract(fresh)
    _, prior_identity = _same_file(manifest["source_inspection"]["path"], manifest["source_inspection"])
    prior, _ = _metadata(prior_identity["path"])
    if _facts(prior) != _facts(fresh):
        raise ValueError("selected prior inspection differs from the saved fresh inspection")
    source, model_source, abi_raw, bundle_bindings = _bundle(manifest["source_bundle"], fresh["draft_bundle"])
    if (model_source != manifest.get("model_source") or manifest.get("model") != source["model"]
            or manifest.get("model_upload_source") != model_source["path"]
            or any(manifest.get(key) != source[key] for key in ("draft_config", "direct_spec", "outputs", "ddk_id", "platform", "weight_storage"))):
        raise ValueError("prepared real-feature step differs from its unchanged original model/ABI")
    execution, _ = _safe(fresh["execution"], directory=True)
    input_root, _ = _safe(Path(ids["path"]).parent, directory=True)
    plan = _target_plan(execution, input_root, source)
    if any(manifest.get(key) != plan.get(key) for key in ("runtime_version", "firmware_id")):
        raise ValueError("prepared runtime identity differs from successful target proof")
    controls, control_bindings = _controls(plan, input_root)
    _, feature_identity = _same_file(_relative(execution / "board", feature["path"]), feature,
        maximum=_FEATURE_BYTES)
    ids_raw, ids_identity = _same_file(ids["path"], ids, maximum=256)
    values = list(struct.unpack("<64i", ids_raw))
    if values != ids["values"]:
        raise ValueError("original IDs differ from the saved fresh inspection")
    expected_feature = {"source": feature_identity, "runtime_index": feature["runtime_index"],
        "source_shape": [1, 1, 64, 16384], "draft_shape": [1, 64, 16384],
        "transformation": "DROP_ONE_LEADING_SINGLETON_ND_AXIS_BYTES_UNCHANGED",
        "statistics": feature["statistics"], "finite_extremes_policy": "PRESERVED_WITHOUT_CLIPPING_OR_NORMALIZATION"}
    expected_tokens = {"source": ids_identity, "valid_length": 64, "anchor_position": 63,
        "anchor_token_id": values[63], "anchor_source": "ORIGINAL_LAST_INPUT_ID",
        "prefix_policy": "EXACT_ORIGINAL64_NO_TRUNCATION_OR_REBASE", "target_greedy_token_used_as_anchor": False,
        "token_to_target_embedding_equivalence": "UNVERIFIED"}
    expected_mask = {"kind": "GENERATED_INT32_ONES", "valid_tokens": 64, "controls": controls,
        "basis": "ORIGINAL_EFFECTIVE_LENGTH_ALLQLEN_OUTPUT_POS_ALL_EQUAL64_WITH_RECORDED_SOURCE_SHA256"}
    if (manifest.get("feature_provenance") != expected_feature
            or manifest.get("token_provenance") != expected_tokens or manifest.get("mask_provenance") != expected_mask
            or manifest.get("target_greedy_observation") != fresh.get("bound_tensors", {}).get("logits", {}).get("statistics", {}).get("argmax")):
        raise ValueError("prepared input provenance differs from actual target features, original IDs or verified controls")
    expected_bindings = _source_bindings(fresh, execution, bundle_bindings + [prior_identity,
        fresh_identity, *fresh["evidence_bindings"], *control_bindings, feature_identity, ids_identity])
    if manifest["source_bindings"] != expected_bindings:
        raise ValueError("prepared source inventory is not the complete target inspection and original model evidence")
    inputs, _ = _io(64)
    if len(manifest.get("inputs", [])) != 3:
        raise ValueError("real-feature step requires exactly three inputs")
    expected_payloads = []
    for row, desc in zip(manifest["inputs"], inputs):
        if any(row.get(key) != value for key, value in desc.items()):
            raise ValueError("prepared real input descriptor changed")
        raw, identity = _same_file(_relative(out, row["path"]), row)
        role = desc["name"]
        expected_payloads.append({**identity, "path": row["path"], "local_path": str(_relative(out, row["path"])), "role": role})
        if role == "target_features" and hashlib.sha256(raw).hexdigest() != manifest["feature_provenance"]["source"]["sha256"]:
            raise ValueError("prepared feature is not the exact bound original bytes")
        if role == "input_ids" and hashlib.sha256(raw).hexdigest() != manifest["token_provenance"]["source"]["sha256"]:
            raise ValueError("prepared IDs are not the exact original64 bytes")
        if role == "attention_mask" and raw != struct.pack("<64i", *([1] * 64)):
            raise ValueError("prepared valid-token mask changed")
    for field, raw, role in (("draft_abi", abi_raw, "draft_abi"), ("step_plan", _wire(manifest), "step_plan")):
        row = manifest[field]
        actual, identity = _same_file(_relative(out, row["path"]), row)
        if actual != raw:
            raise ValueError("prepared native ABI/step bytes changed")
        expected_payloads.append({**identity, "path": row["path"], "local_path": str(_relative(out, row["path"])), "role": role})
    if manifest.get("upload_payloads") != expected_payloads:
        raise ValueError("prepared upload inventory differs from the five small owned payloads")
    return manifest

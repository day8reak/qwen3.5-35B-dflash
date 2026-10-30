"""Real feature preparation fixtures use tiny fake models and no device calls."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from qwen35_moe_dflash import draft_board
from qwen35_moe_dflash import real_feature_draft as real
from qwen35_moe_dflash.resources import repository_root
from qwen35_moe_dflash.target_prefill_smoke_plan import _source
from test_target_prefill_output_inventory import case as retained_case, refresh


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(path, root=None):
    return {"path": str(path.relative_to(root)) if root else str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def fixture(retained_case, tmp_path):
    case = retained_case
    inputs = Path(case["plan"]["input_dir"])
    inputs.mkdir()
    (inputs / "input_ids.bin").write_bytes(struct.pack("<64i", *([111] * 63 + [220])))
    controls = []
    for name, encoding in real._CONTROLS:
        path = inputs / name
        path.write_bytes(struct.pack(encoding, 64))
        controls.append(_source(path, struct.calcsize(encoding)))
    case["plan"].update(source_inputs=controls, ddk_id="fixture-ddk", runtime_version="fixture-runtime",
        firmware_id="fixture-firmware")
    # Include legitimate FP16 extrema without clipping or rejecting them.
    feature = case["board"] / case["paths"][81]
    values = np.ones(64 * 16384, dtype="<f2")
    values[:2] = [-65504, 65504]
    feature.write_bytes(values.tobytes())
    case["report"]["retrieved"][case["paths"][81]] = {key: value for key, value in record(feature).items() if key != "path"}
    refresh(case)
    bundle = tmp_path / "source-bundle"
    (bundle / "model").mkdir(parents=True)
    model = bundle / "model/model.omc"
    model.write_bytes(b"tiny-fake-embedded-model-for-host-preparation-tests")
    cfg = json.loads((repository_root() / "specs/draft-config.json").read_text())
    descriptors, outputs = draft_board._io(64, direct_spec=[])
    abi = {"artifact_kind": "qwen35-standalone-draft-direct-abi", "draft_config": cfg,
        "inputs": descriptors, "outputs": outputs, "sequence_length": 64, "state_policy": "prefix-recompute",
        "direct_spec": [], "source_output_precision": "DECLARED_SOURCE_DTYPES"}
    save(bundle / "draft-abi.json", abi)
    manifest = {"artifact_kind": "qwen35-standalone-draft-direct-bundle", "status": "PREPARED",
        "sequence_length": 64, "platform": "kirinx90", "weight_storage": "embedded-stream",
        "ddk_id": "fixture-ddk", "vocab_size": 248320, "draft_config": cfg, "direct_spec": [],
        "source_output_precision": "DECLARED_SOURCE_DTYPES", "source_identity": "MATCHED_EXPORT_AND_REFERENCE",
        "draft_abi": record(bundle / "draft-abi.json", bundle), "model": record(model, bundle),
        "inputs": [], "outputs": outputs}
    manifest["model_payloads"] = [manifest["model"]]
    for desc, size in zip(descriptors, (64 * 16384 * 2, 256, 256)):
        path = bundle / f"synthetic_{desc['index']}.bin"
        path.write_bytes(bytes(size))
        manifest["inputs"].append({**desc, **record(path, bundle)})
    manifest["expected"] = []
    for desc, size in zip(outputs, (60, 15 * 2048 * 2)):
        path = bundle / f"golden_{desc['index']}.bin"
        path.write_bytes(bytes(size))
        manifest["expected"].append({**desc, **record(path, bundle)})
    save(bundle / "weight-packing.json", {"status": "EMBEDDED", "output": manifest["model"]})
    manifest["weight_packing"] = record(bundle / "weight-packing.json", bundle)
    (bundle / "step.tsv").write_bytes(draft_board._wire(manifest))
    manifest["step_plan"] = record(bundle / "step.tsv", bundle)
    save(bundle / "manifest.json", manifest)
    inspection = tmp_path / "original-inspection"
    report = real.inspect_target_prefill_outputs(case["execution"], inspection, draft_bundle=bundle, input_dir=inputs)
    assert report["status"] == "OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS", report
    return {"case": case, "bundle": bundle, "bundle_manifest": manifest, "inspection": inspection,
            "report": report, "out": tmp_path / "real-inputs", "input_dir": inputs}


def prepare(fixture):
    return real.prepare_real_feature_draft(fixture["inspection"], fixture["bundle"], fixture["out"])


def test_real_step_preserves_extreme_feature_bytes_original_anchor_and_existing_bundle(fixture):
    before = {path: (path.stat(), path.read_bytes()) for path in fixture["bundle"].rglob("*") if path.is_file()}
    manifest = prepare(fixture)
    out = fixture["out"]
    assert manifest["status"] == "REAL_FEATURE_DRAFT_STEP_PREPARED"
    assert manifest["artifact_kind"] == "qwen35-real-feature-draft-step"
    assert (out / "source-validation/report.json").is_file()
    assert not (out / "model").exists()
    assert not any(path.is_symlink() for path in out.rglob("*"))
    assert len(manifest["upload_payloads"]) == 5
    assert sum(row["bytes"] for row in manifest["upload_payloads"]) < (3 << 20)
    assert not any("model.omc" in row["path"] for row in manifest["upload_payloads"])
    original_feature = fixture["case"]["board"] / fixture["case"]["paths"][81]
    assert (out / manifest["inputs"][0]["path"]).read_bytes() == original_feature.read_bytes()
    assert manifest["inputs"][0]["sha256"] == fixture["report"]["bound_tensors"]["feature"]["sha256"]
    assert manifest["inputs"][0]["shape"] == [1, 64, 16384]
    assert (out / manifest["inputs"][1]["path"]).read_bytes() == (fixture["input_dir"] / "input_ids.bin").read_bytes()
    assert struct.unpack("<64i", (out / manifest["inputs"][2]["path"]).read_bytes()) == (1,) * 64
    assert manifest["token_provenance"]["anchor_position"] == 63
    assert manifest["token_provenance"]["anchor_token_id"] == 220
    assert manifest["target_greedy_observation"]["token_id"] == 0
    assert manifest["token_provenance"]["target_greedy_token_used_as_anchor"] is False
    assert manifest["token_provenance"]["token_to_target_embedding_equivalence"] == "UNVERIFIED"
    assert (out / "step.tsv").read_bytes().startswith(b"QWEN35_DRAFT_DIRECT_STEP\t1\nmodel\tmodel/model.omc\t")
    assert (out / "draft-abi.json").read_bytes() == (fixture["bundle"] / "draft-abi.json").read_bytes()
    assert real.validate_real_feature_draft(out) == manifest
    for path, (info, raw) in before.items():
        assert path.read_bytes() == raw
        current = path.stat()
        assert (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns, current.st_nlink) == (
            info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


@pytest.mark.parametrize("name,encoding", real._CONTROLS)
def test_control_value_must_match_successful_plan_hash_and_all64_tokens(fixture, name, encoding):
    path = fixture["input_dir"] / name
    path.write_bytes(struct.pack(encoding, 63))
    with pytest.raises(ValueError, match="source SHA256|original control identity"):
        prepare(fixture)


def test_control64_value_without_original_plan_binding_is_rejected(fixture):
    fixture["case"]["plan"]["source_inputs"] = []
    refresh(fixture["case"])
    # Re-inspect the changed execution so this tests control provenance itself.
    alternate = fixture["out"].parent / "second-inspection"
    real.inspect_target_prefill_outputs(fixture["case"]["execution"], alternate,
        draft_bundle=fixture["bundle"], input_dir=fixture["input_dir"])
    fixture["inspection"] = alternate
    with pytest.raises(ValueError, match="missing or ambiguous original control"):
        prepare(fixture)


@pytest.mark.parametrize("fault", ["feature", "ids", "model", "generated_mask", "source_log", "anchor", "provenance", "bindings", "runtime"])
def test_immediate_validation_rejects_changed_source_or_payload(fixture, fault):
    manifest = prepare(fixture)
    if fault == "feature":
        path = fixture["case"]["board"] / fixture["case"]["paths"][81]
    elif fault == "ids": path = fixture["input_dir"] / "input_ids.bin"
    elif fault == "model": path = fixture["bundle"] / "model/model.omc"
    elif fault == "generated_mask": path = fixture["out"] / manifest["inputs"][2]["path"]
    elif fault == "source_log": path = fixture["case"]["board"] / "tool.stdout"
    else:
        path = None
        if fault == "anchor": manifest["token_provenance"]["anchor_token_id"] = 0
        elif fault == "provenance": manifest["feature_provenance"]["source"]["sha256"] = "b" * 64
        elif fault == "bindings": manifest["source_bindings"].pop()
        elif fault == "runtime": manifest["runtime_version"] = "different-runtime"
        save(fixture["out"] / "manifest.json", manifest)
    if path:
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
    with pytest.raises(ValueError):
        real.validate_real_feature_draft(fixture["out"])


def test_preparation_rejects_stale_inspection_and_nonfinite_bound_features(fixture):
    path = fixture["input_dir"] / "input_ids.bin"
    path.write_bytes(struct.pack("<64i", *([111] * 64)))
    with pytest.raises(ValueError, match="fresh target-output inspection differs"):
        prepare(fixture)


def test_nonfinite_features_are_rejected_without_modifying_original_bytes(fixture):
    case = fixture["case"]
    path = case["board"] / case["paths"][81]
    raw = bytearray(path.read_bytes())
    raw[:2] = struct.pack("<e", float("nan"))
    path.write_bytes(raw)
    case["report"]["retrieved"][case["paths"][81]] = {key: value for key, value in record(path).items() if key != "path"}
    refresh(case)
    alternate = fixture["out"].parent / "nan-inspection"
    real.inspect_target_prefill_outputs(case["execution"], alternate, draft_bundle=fixture["bundle"], input_dir=fixture["input_dir"])
    fixture["inspection"] = alternate
    with pytest.raises(ValueError, match="must be finite"):
        prepare(fixture)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ["capacity", "source_identity", "extra_output"])
def test_source_bundle_requires_complete_matching_direct_capacity64_contract(fixture, fault):
    manifest = fixture["bundle_manifest"]
    if fault == "capacity": manifest["sequence_length"] = 63
    elif fault == "source_identity": manifest["source_identity"] = "UNVERIFIED"
    elif fault == "extra_output": manifest["outputs"].append({"index": 2, "shape": [1], "dtype": "float16", "name": "extra"})
    save(fixture["bundle"] / "manifest.json", manifest)
    alternate = fixture["out"].parent / "changed-source-inspection"
    real.inspect_target_prefill_outputs(fixture["case"]["execution"], alternate,
        draft_bundle=fixture["bundle"], input_dir=fixture["input_dir"])
    fixture["inspection"] = alternate
    with pytest.raises(ValueError, match="capacity64|exactly three inputs and two outputs"):
        prepare(fixture)

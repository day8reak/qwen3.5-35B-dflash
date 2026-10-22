"""Prepared direct-GE weights must be the weights of the CPU board oracle."""
import json
import os

import numpy as np
import pytest

from qwen35_moe_dflash.ge_export import prepare_pair, weight_identity
from qwen35_moe_dflash.ge_reference import reference_tiny
from qwen35_moe_dflash.hashing import sha256_file


def test_ge_reference_uses_prepared_tiny_weights(tmp_path, monkeypatch):
    from qwen35_moe_dflash import backends, precision
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    # The CLI reference policy intentionally persists for its whole process.
    # Record every field it changes so pytest restores the original process
    # state before subsequent AIR backend tests, even if this test fails.
    monkeypatch.setattr(backends, "_REFERENCE_DEVICE", backends._REFERENCE_DEVICE)
    monkeypatch.setattr(utils, "is_torch_npu_available", utils.is_torch_npu_available)
    monkeypatch.setattr(imports, "is_torch_npu_available", imports.is_torch_npu_available)
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD", ""))
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, sequence_length=8, tiny=True)
    original_lower = precision.prepare_fp16_pair
    inspected = []

    def assert_loaded(target, draft, sequence):
        assert draft.embedding.weight is target.language_model.embed_tokens.weight
        assert draft.lm_head.weight is target.lm_head.weight
        for key, record in plan["weights"].items():
            if key == "target.lm_head.weight":
                actual = target.lm_head.weight
            elif key.startswith("target."):
                actual = target.language_model.get_parameter(key.removeprefix("target."))
            else:
                actual = draft.draft.get_parameter(key.removeprefix("draft."))
            expected = np.fromfile(prepared / record["path"], dtype="<f2").reshape(record["shape"])
            np.testing.assert_array_equal(actual.detach().numpy(), expected)
            inspected.append(key)
        return original_lower(target, draft, sequence)

    monkeypatch.setattr(precision, "prepare_fp16_pair", assert_loaded)
    output = tmp_path / "reference.json"
    result = reference_tiny(prepared, output)
    assert len(inspected) == len(plan["weights"])
    assert result == json.loads(output.read_text())
    assert result["status"] == "PASS" and result["evidence_kind"] == "framework-reference"
    assert result["source"] == plan["source"]
    assert result["source"]["fp16_payload_sha256"] == weight_identity(plan["weights"])
    assert result["prepared_manifest_sha256"] == sha256_file(prepared / "prepared.json")
    assert result["precision"] == "torch.float16" and result["quantization"] is None
    assert result["sequence_length"] == 8 and result["max_new_tokens"] == 4
    assert result["prompt_token_ids"] == [1, 2, 3]
    assert result["ordinary"]["token_ids"] == result["dflash"]["token_ids"]
    assert len(result["ordinary"]["token_ids"]) == 4
    assert all(audit["status"] == "PASS" for audit in result["precision_audit"].values())


def test_ge_reference_rejects_corrupted_payload_before_output(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, sequence_length=8, tiny=True)
    entry = next(iter(plan["weights"].values()))
    path = prepared / entry["path"]
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)
    output = tmp_path / "reference.json"
    with pytest.raises(ValueError, match="SHA256/size mismatch"):
        reference_tiny(prepared, output)
    assert not output.exists()


def test_ge_reference_rejects_full_or_changed_model_config(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, sequence_length=8, tiny=True)
    plan["tiny"] = False
    (prepared / "prepared.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="prepared GE tiny pair"):
        reference_tiny(prepared, tmp_path / "reference-full.json")
    plan["tiny"] = True
    plan["draft_config"]["rope_parameters"]["rope_theta"] = 99.0
    (prepared / "prepared.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="configuration differs"):
        reference_tiny(prepared, tmp_path / "reference-config.json")


def test_ge_reference_rejects_stale_weight_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, sequence_length=8, tiny=True)
    plan["source"]["fp16_payload_sha256"] = "0" * 64
    (prepared / "prepared.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="source identity differs"):
        reference_tiny(prepared, tmp_path / "reference-stale.json")

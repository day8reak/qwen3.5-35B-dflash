import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
import yaml

from qwen35_moe_dflash.ge_weights import checkpoint_key
from qwen35_moe_dflash.original_quant import OriginalQuantWeights, canonicalkeys
from qwen35_moe_dflash.original_quant_prepare import prepare_original_target
from qwen35_moe_dflash.target_weight_shapes import required_weight_shapes


@pytest.fixture(autouse=True)
def preparation_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))


def _shard(path, tensors):
    header, payload, offset = {}, [], 0
    for key, value in tensors.items():
        raw = value.tobytes()
        header[key] = {"dtype": {"i1": "I8", "f4": "F32"}[value.dtype.str[1:]],
                       "shape": list(value.shape), "data_offsets": [offset, offset + len(raw)]}
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))


@pytest.fixture
def original(tmp_path):
    hf, quant = tmp_path / "hf", tmp_path / "quant"
    hf.mkdir()
    quant.mkdir()
    config = {"hidden_size": 256, "head_dim": 64, "vocab_size": 32,
              "num_attention_heads": 4, "num_key_value_heads": 1,
              "num_hidden_layers": 1, "layer_types": ["linear_attention"],
              "linear_num_key_heads": 2, "linear_key_head_dim": 16,
              "linear_num_value_heads": 16, "linear_value_head_dim": 16,
              "linear_conv_kernel_dim": 4, "num_experts": 16,
              "num_experts_per_tok": 2, "moe_intermediate_size": 256,
              "shared_expert_intermediate_size": 256, "rms_norm_eps": 1e-6}
    (hf / "config.json").write_text(json.dumps({"text_config": config}))
    qweights, base = {}, {}
    for key, shape in required_weight_shapes(config).items():
        if key == "target.embed_tokens.weight":
            continue
        try:
            wkey, skey = canonicalkeys(key)
        except ValueError:
            base[checkpoint_key(key)[1]] = np.full(shape, 0.25, dtype="<f4")
            continue
        qweights[wkey] = (np.arange(np.prod(shape)).reshape(shape) % 16 - 8).astype("i1")
        scale_shape = (*shape[:-1], shape[-1] // 256) if len(shape) == 3 else (shape[0], shape[1] // 256, 1)
        qweights[skey] = np.full(scale_shape, 0.125, dtype="<f4")
    _shard(hf / "model.safetensors", base)
    _shard(quant / "data00001.safetensors", qweights)
    np.ones((32, 256), dtype="i1").tofile(tmp_path / "embedding.bin")
    np.full((32, 1), 0.25, dtype="<f4").tofile(tmp_path / "embedding_scale.bin")
    settings = {"model_file": "hf", "quanted_pth": "quant",
                "embedding_weight_path": "embedding.bin", "embedding_scale_path": "embedding_scale.bin",
                "kv_cache_max_len": 2048}
    path = tmp_path / "original.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path, config, qweights, base


def test_inspection_is_header_only(original, tmp_path, monkeypatch):
    import qwen35_moe_dflash.original_quant_prepare as module
    def unexpected(*args, **kwargs):
        raise AssertionError("inspection must not touch tensor payload")
    monkeypatch.setattr(module.OriginalQuantWeights, "packed", unexpected)
    monkeypatch.setattr(module.SafeTensorFiles, "write_fp16", unexpected)
    monkeypatch.setattr(module, "_hash_region", unexpected)
    monkeypatch.setattr(module, "load_embedding", unexpected)
    result = prepare_original_target(original[0], tmp_path / "inspect", inspect_only=True)
    assert result["status"] == "INSPECTED"
    assert result["schema_version"] == 1
    assert result["artifact_kind"] == "qwen35-original-quant-preparation"
    assert result["output"] == str(tmp_path / "inspect")
    assert result["model_compilation"] == result["device_execution"] == "NOT_RUN"
    assert not result["export_ready"] and not result["device_ready"]
    assert result["weights"] == {}
    assert result["quant_descriptors"] and result["hf_descriptors"]
    assert result["embedding"]["weight"]["sha256"] is None
    assert result["source_hash_scope"] == "HEADERS_AND_CONFIG_ONLY"
    assert not (tmp_path / "inspect/weights").exists()


def test_prepare_preserves_raw_norms_and_original_quantized_values(original, tmp_path):
    path, config, tensors, _ = original
    output = tmp_path / "prepared"
    progress = []
    result = prepare_original_target(path, output, progress=progress.append)
    assert result["status"] == "QPARAM_PENDING"
    assert not result["export_ready"] and not result["device_ready"]
    assert len(result["pending_expert_qparam"]) == 2
    assert result["sources"]["model_file"] == str(tmp_path / "hf")
    assert result == json.loads((output / "manifest.json").read_text())
    assert len(progress) == len(required_weight_shapes(config))
    assert progress[-1]["completed"] == progress[-1]["total"]
    assert len(list((output / "tensor_records").glob("*.json"))) == len(progress)
    assert "target.embed_tokens.weight" not in result["weights"]
    assert result["embedding"]["weight"]["path"] == str(tmp_path / "embedding.bin")
    assert result["embedding"]["weight"]["sha256"] == hashlib.sha256((tmp_path / "embedding.bin").read_bytes()).hexdigest()
    for key, weight in result["weights"].items():
        payload = (output / weight["path"]).read_bytes()
        assert len(payload) == weight["bytes"]
        assert hashlib.sha256(payload).hexdigest() == weight["sha256"]
        assert len(payload) == np.prod(weight["shape"]) * np.dtype(weight["dtype"]).itemsize
        if "norm.weight" in key or "layernorm.weight" in key:
            assert np.all(np.frombuffer(payload, dtype="<f2") == 0.25), key
    key = "target.layers.0.linear_attn.in_proj_qkv.weight"
    desc = result["quant_descriptors"][key]
    expected = OriginalQuantWeights(tmp_path / "quant").packed(key, (320, 256))
    assert (output / result["weights"][key]["path"]).read_bytes() == expected["weight"].tobytes()
    assert (output / result["weights"][key + ".quant_scale"]["path"]).read_bytes() == expected["scale"].tobytes()
    for field, source_key in zip(("source_weight", "source_scale"), canonicalkeys(key)):
        assert desc[field]["sha256"] == hashlib.sha256(tensors[source_key].tobytes()).hexdigest()
    for key in result["pending_expert_qparam"]:
        assert result["weights"][key + ".quant_scale"]["dtype"] == "float32"
    for desc in result["hf_descriptors"].values():
        assert desc["sha256"] == hashlib.sha256(original[3][desc["tensor"]].tobytes()).hexdigest()


def test_explicit_converter_records_int64_and_no_device_claim(original, tmp_path):
    seen = []
    def converter(array):
        assert array.dtype == np.dtype("<f4") and array.flags.c_contiguous
        seen.append(array.shape)
        # Interface sentinel only; this is not a QPARAM encoding implementation.
        return np.full(array.shape, 77, dtype="<i8")
    output = tmp_path / "converted"
    result = prepare_original_target(original[0], output, qparam_converter=converter)
    assert result["status"] == "PREPARED" and result["export_ready"]
    assert not result["device_ready"]
    assert result["pending_expert_qparam"] == [] and len(seen) == 2
    for key, desc in result["quant_descriptors"].items():
        if desc["kind"] == "expert":
            assert desc["scale_encoding"] == "QPARAM_VENDOR_CONVERTED"
            scale = result["weights"][key + ".quant_scale"]
            assert scale["dtype"] == "int64"
            assert np.all(np.fromfile(output / scale["path"], dtype="<i8") == 77)


def test_failed_converter_retains_failure_and_completed_tensor_records(original, tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="return int64"):
        prepare_original_target(original[0], output, qparam_converter=lambda value: value)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["status"] == "FAILED"
    assert ".experts." in failure["tensor"]
    assert failure["completed_tensors"] > 0
    assert len(list((output / "tensor_records").glob("*.json"))) == failure["completed_tensors"]
    assert not (output / "manifest.json").exists()


def test_missing_quantized_key_fails_without_float_fallback(original, tmp_path):
    tensors = original[2]
    tensors.pop("model.layers.0.linear_attn.in_proj_b_quant_scale")
    _shard(tmp_path / "quant/data00001.safetensors", tensors)
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="original quantized tensor missing"):
        prepare_original_target(original[0], output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["tensor"] == "target.layers.0.linear_attn.in_proj_b.weight"


def test_output_must_be_exclusive(original, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "weights.bin"
    sentinel.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        prepare_original_target(original[0], output)
    assert sentinel.read_bytes() == b"preserved"
    assert not (output / "failure.json").exists()


def test_configuration_failure_is_retained(tmp_path):
    output = tmp_path / "config-failed"
    with pytest.raises(FileNotFoundError):
        prepare_original_target(tmp_path / "missing.yaml", output)
    assert json.loads((output / "failure.json").read_text())["stage"] == "configuration"


def test_inspection_rejects_wrong_embedding_size(original, tmp_path):
    (tmp_path / "embedding_scale.bin").write_bytes(b"\x00")
    output = tmp_path / "bad-embedding"
    with pytest.raises(ValueError, match="embedding scale file size"):
        prepare_original_target(original[0], output, inspect_only=True)
    assert json.loads((output / "failure.json").read_text())["stage"] == "embedding"


def test_prepare_output_must_stay_under_declared_run(original, tmp_path):
    output = tmp_path.parent / (tmp_path.name + "-outside")
    with pytest.raises(ValueError, match="below AI_RUN_DIR"):
        prepare_original_target(original[0], output, inspect_only=True)
    assert not output.exists()


def test_prepare_requires_run_variable(original, tmp_path, monkeypatch):
    monkeypatch.delenv("AI_RUN_DIR")
    with pytest.raises(RuntimeError, match="AI_RUN_DIR is required"):
        prepare_original_target(original[0], tmp_path / "missing-run", inspect_only=True)
    assert not (tmp_path / "missing-run").exists()


@pytest.mark.parametrize("stage,expected", [("inspect", "INSPECTED"), ("prepare", "QPARAM_PENDING")])
def test_cli_yaml_inspect_and_prepare(original, tmp_path, stage, expected):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/adapt_original_target.py"
    output = tmp_path / ("cli-" + stage)
    run = subprocess.run([sys.executable, str(script), "--config", str(original[0]),
                          "--stage", stage, "--output", str(output)],
                         capture_output=True, text=True, env=dict(os.environ), check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    summary = json.loads(run.stdout.splitlines()[-1])
    assert summary == {"status": expected, "output": str(output), "device_ready": False,
                       "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN"}
    assert json.loads((output / "manifest.json").read_text())["status"] == expected

"""Original YAML paths and external embedding ABI, using local fixture bytes."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from qwen35_moe_dflash.original_model_config import OriginalModelConfig


PATH_FIELDS = ("model_file", "quanted_pth", "embedding_weight_path", "embedding_scale_path")


@pytest.fixture
def config_case(tmp_path):
    folder = tmp_path / "configuration"
    folder.mkdir()
    hf, quant = folder / "hf", folder / "quant"
    hf.mkdir(); quant.mkdir()
    (hf / "config.json").write_text(json.dumps({"text_config": {
        "vocab_size": 5, "hidden_size": 4, "num_attention_heads": 2,
        "kv_cache_max_len": 17, "rope_parameters": None}}))
    (quant / "data0.safetensors").write_bytes(b"NOT READ BY CONFIG")
    weights = np.arange(-10, 10, dtype=np.int8).reshape(5, 4)
    scales = np.array([.1, .2, .3, .4, .5], dtype=np.float32).reshape(5, 1)
    weights.tofile(folder / "embedding.bin")
    scales.tofile(folder / "scale.bin")
    values = dict(zip(PATH_FIELDS, ("hf", "quant", "embedding.bin", "scale.bin")))
    source = folder / "model.yaml"
    source.write_text(yaml.safe_dump(values))
    return source, values, weights, scales


def test_yaml_preserves_all_four_original_absolute_paths(tmp_path):
    values = {
        "model_file": "/home/l00886453/models/Qwen3.5-35B-A3B",
        "quanted_pth": "/home/l00886453/models/Qwen3.5-35B-A3B_quant_4bit",
        "embedding_weight_path": "/home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_weight.bin",
        "embedding_scale_path": "/home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_scale.bin",
    }
    source = tmp_path / "original.yaml"
    source.write_text(yaml.safe_dump(values))
    config = OriginalModelConfig.from_yaml(source)
    assert {field: str(getattr(config, field)) for field in PATH_FIELDS} == values
    assert config.record()["quantization"] == "REUSE_ORIGINAL_NO_REQUANTIZATION"


def test_relative_paths_resolve_against_yaml_and_kv_default_override(config_case, monkeypatch, tmp_path):
    source, values, *_ = config_case
    monkeypatch.chdir(tmp_path)
    config = OriginalModelConfig.from_yaml(source)
    assert all(getattr(config, field) == (source.parent / values[field]).resolve() for field in PATH_FIELDS)
    config.validate_paths()
    assert config.kv_cache_max_len == 2048
    assert config.text_config()["kv_cache_max_len"] == 2048
    assert config.text_config()["head_dim"] == 2
    assert config.text_config()["rope_parameters"] == {}
    source.write_text(yaml.safe_dump(dict(values, kv_cache_max_len=4096)))
    assert OriginalModelConfig.from_yaml(source).text_config()["kv_cache_max_len"] == 4096


@pytest.mark.parametrize("length", [True, 0, -1, 2.5, "2048", None])
def test_kv_capacity_must_be_a_positive_integer(config_case, length):
    source, values, *_ = config_case
    source.write_text(yaml.safe_dump(dict(values, kv_cache_max_len=length)))
    with pytest.raises(ValueError, match="positive integer"):
        OriginalModelConfig.from_yaml(source)


def test_embedding_rows_keep_original_int8_and_fp32_and_do_not_modify_files(config_case):
    source, _, weights, scales = config_case
    config = OriginalModelConfig.from_yaml(source)
    ids = np.array([[4, 0, 4, 2]], dtype=np.int32)
    rows, factors = config.embedding_rows(ids)
    assert rows.dtype == np.int8 and rows.shape == (1, 4, 4)
    assert factors.dtype == np.float32 and factors.shape == (1, 4, 1)
    np.testing.assert_array_equal(rows, weights[ids])
    np.testing.assert_array_equal(factors, scales[ids])
    rows[:] = 0; factors[:] = 0
    np.testing.assert_array_equal(np.fromfile(config.embedding_weight_path, dtype=np.int8).reshape(weights.shape), weights)
    np.testing.assert_array_equal(np.fromfile(config.embedding_scale_path, dtype=np.float32).reshape(scales.shape), scales)


@pytest.mark.parametrize("ids", [np.array([[1.0]]), np.array([[True]]), np.array([1]),
                                np.array([[1], [2]]), np.empty((1, 0), dtype=np.int32),
                                np.array([[-1]]), np.array([[5]]), np.array([[2**63]], dtype=np.uint64)])
def test_embedding_rejects_wrong_token_shape_dtype_range(config_case, ids):
    config = OriginalModelConfig.from_yaml(config_case[0])
    with pytest.raises(ValueError, match="token_ids"):
        config.embedding_rows(ids)


@pytest.mark.parametrize("field,dtype", [("embedding_weight_path", np.float16), ("embedding_scale_path", np.float16)])
def test_embedding_rejects_wrong_raw_file_size_or_element_width(config_case, field, dtype):
    config = OriginalModelConfig.from_yaml(config_case[0])
    count = 20 if field == "embedding_weight_path" else 5
    np.zeros(count, dtype=dtype).tofile(getattr(config, field))
    with pytest.raises(ValueError, match="must contain"):
        config.embedding_rows(np.array([[1]], dtype=np.int64))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -.1])
def test_embedding_rejects_invalid_selected_scales(config_case, bad):
    source, _, _, scales = config_case
    config = OriginalModelConfig.from_yaml(source)
    scales[2] = bad
    scales.tofile(config.embedding_scale_path)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        config.embedding_rows(np.array([[2]], dtype=np.int32))


def test_load_pytorch_passes_the_actual_four_paths_and_config(config_case, monkeypatch):
    from qwen35_moe_dflash.fused import quantization
    source = config_case[0]
    config = OriginalModelConfig.from_yaml(source)
    actual = {}
    sentinel = object()
    def load(text_config, **kwargs):
        actual.update(kwargs, config=text_config)
        return sentinel
    monkeypatch.setattr(quantization, "load_original_quantized_target", load)
    ops, quant_ops = object(), object()
    result = config.load_pytorch(feature_layers=(0, 2), ops=ops, quant_ops=quant_ops,
                                 device="npu:0", recurrent_dtype="sentinel-dtype")
    assert result is sentinel
    assert {field: actual[field] for field in PATH_FIELDS} == {field: getattr(config, field) for field in PATH_FIELDS}
    assert actual["config"]["kv_cache_max_len"] == 2048
    assert actual["feature_layers"] == (0, 2) and actual["device"] == "npu:0"
    assert actual["ops"] is ops and actual["quant_ops"] is quant_ops
    assert actual["recurrent_dtype"] == "sentinel-dtype"


def test_text_config_rejects_non_mapping_root(config_case):
    config = OriginalModelConfig.from_yaml(config_case[0])
    (config.model_file / "config.json").write_text("[]")
    with pytest.raises(ValueError, match="mapping"):
        config.text_config()

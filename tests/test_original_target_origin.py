"""Header-only origin preparation cannot accept missing random-initialized weights."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import yaml

from qwen35_moe_dflash.original_quant import OriginalQuantWeights, canonicalkeys
from qwen35_moe_dflash.original_target_origin import (
    inspect_target_pytorch_resources, prepare_target_pytorch,
)
from qwen35_moe_dflash.original_target_source import EXPORTER, SOURCE_FILES, WRAPPER
from qwen35_moe_dflash.target_weight_shapes import required_weight_shapes


REPO = Path(__file__).resolve().parents[1]
# Independent stand-in for the source patcher's structural contract.
SOURCE = """import torch
class Qwen3_5MoeForCausalLMCproWrapper:
    def forward(self, hidden_states, past_key_values, use_cache=True):
        kv_caches_out = []
        for idx, decoder_layer in enumerate(self.model.model.layers):
            layer_outputs = decoder_layer(hidden_states, past_key_values=past_key_values[idx],
                layer_idx=idx, new_kv_cache_pos=None, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            kv_caches_out.extend(layer_outputs[1])
        lm_logits = hidden_states
        return lm_logits, *kv_caches_out
"""


def sparse_shard(path, specs):
    header, offset = {}, 0
    for name, (shape, dtype) in specs.items():
        size = math.prod(shape) * {"BF16": 2, "I8": 1, "F32": 4}[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)) + raw)
        stream.truncate(8 + len(raw) + offset)
    return 8 + len(raw)


@pytest.fixture
def origin(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "run"))
    hf, quant, source = (tmp_path / name for name in ("hf", "quant", "private-source"))
    hf.mkdir()
    quant.mkdir()
    text = json.loads((REPO / "specs/target-config.json").read_text())["text_config"]
    (hf / "config.json").write_text(json.dumps({"text_config": text}))
    base, quantized = {}, {}
    for name, shape in required_weight_shapes(text).items():
        if ".mlp.experts." not in name:
            clean = name[len("target."):]
            raw = clean if clean == "lm_head.weight" else "model.language_model." + clean
            base[raw] = (shape, "BF16")
        try:
            weight, scale = canonicalkeys(name)
        except ValueError:
            continue
        quantized[weight] = (shape, "I8")
        scale_shape = (*shape[:-1], shape[-1] // 256) if len(shape) == 3 else (shape[0], shape[1] // 256, 1)
        quantized[scale] = (scale_shape, "F32")
    limits = {}
    limits[hf / "model-00001.safetensors"] = sparse_shard(hf / "model-00001.safetensors", base)
    limits[quant / "data00001.safetensors"] = sparse_shard(quant / "data00001.safetensors", quantized)
    # The original exporter ignores this index; an index-based preflight would be wrong.
    (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"wrong": "unrelated.safetensors"}}))
    embedding, scale = tmp_path / "embedding.bin", tmp_path / "embedding-scale.bin"
    for path, size in ((embedding, 248320 * 2048), (scale, 248320 * 4)):
        with path.open("wb") as stream:
            stream.truncate(size)
    settings = {"model_file": "hf", "quanted_pth": "quant",
                "embedding_weight_path": "embedding.bin", "embedding_scale_path": "embedding-scale.bin",
                "kv_cache_max_len": 2048, "model_name": "qwen3_5_moe"}
    config = tmp_path / "original.yaml"
    config.write_text(yaml.safe_dump(settings))
    for name in SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SOURCE if name == WRAPPER else "value = 1\n")
    (source / EXPORTER).write_text("torchair.dynamo_export(model=model_wrapper, dynamic=False)\n")
    return {"config": config, "text": text, "hf": hf, "quant": quant, "source": source,
            "base": base, "quantized": quantized, "limits": limits, "embedding": embedding, "scale": scale,
            "output": tmp_path / "run/target-pytorch"}


def test_full_header_inventory_prepares_only_source_without_reading_payloads(origin, monkeypatch):
    original_open = Path.open
    reads = []

    class HeaderOnly:
        def __init__(self, file, path):
            self.file, self.path = file, path
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.file.__exit__(*args)
        def read(self, size=-1):
            assert size >= 0 and self.file.tell() + size <= origin["limits"][self.path]
            reads.append((self.path.name, size))
            return self.file.read(size)
        def __getattr__(self, name):
            return getattr(self.file, name)

    def guarded_open(path, mode="r", *args, **kwargs):
        if path in (origin["embedding"], origin["scale"]):
            raise AssertionError("embedding payload must not be opened")
        file = original_open(path, mode, *args, **kwargs)
        return HeaderOnly(file, path) if path in origin["limits"] else file

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(OriginalQuantWeights, "packed", lambda *a, **k: pytest.fail("no quant conversion"))
    result = prepare_target_pytorch(origin["source"], origin["config"], origin["output"])
    assert result["status"] == "SOURCE_READY" and result["export"] == "EXPORT_NOT_RUN"
    assert reads and result["resources"]["target_layers"] == 40
    assert result["resources"]["hf_loader_policy"].endswith("INDEX_IGNORED")
    assert len(result["resources"]["quant_required_tensors"]) > 400
    assert result["resources"]["embedding"]["weight"]["bytes"] == 508559360
    assert result["resources"]["embedding"]["scale"]["bytes"] == 993280
    assert result["payload_validation"] == "NOT_READ"
    assert result["model_execution"] == result["air_export"] == result["device_execution"] == "NOT_RUN"
    assert not list(origin["output"].rglob("*.safetensors"))
    assert not list(origin["output"].rglob("*.bin"))
    assert not list(origin["output"].rglob("*.air"))
    resolved = yaml.safe_load((origin["output"] / "resolved-original.yaml").read_text())
    assert all(Path(resolved[key]).is_absolute() for key in ("model_file", "quanted_pth", "embedding_weight_path", "embedding_scale_path"))
    assert yaml.safe_load(origin["config"].read_text())["model_file"] == "hf"
    assert Path(result["on_origin_export_script"]).is_file()
    assert "exec bash" in Path(result["on_origin_export_script"]).read_text()
    assert (origin["output"] / "result.json").is_file()
    with pytest.raises(ValueError, match="preserve"):
        prepare_target_pytorch(origin["source"], origin["config"], origin["output"])


@pytest.mark.parametrize("fault", ["hf_nonquant", "hf_dense", "quant_weight", "quant_scale", "embedding", "scale", "layers", "experts", "vocab", "source"])
def test_missing_or_incomplete_resources_block_without_export(origin, fault):
    if fault.startswith("hf_"):
        suffix = "layers.0.input_layernorm.weight" if fault == "hf_nonquant" else "layers.0.linear_attn.in_proj_qkv.weight"
        origin["base"].pop("model.language_model." + suffix)
        sparse_shard(origin["hf"] / "model-00001.safetensors", origin["base"])
    elif fault.startswith("quant_"):
        name = "model.layers.0.linear_attn.in_proj_qkv_quant_" + ("weight" if fault == "quant_weight" else "scale")
        origin["quantized"].pop(name)
        sparse_shard(origin["quant"] / "data00001.safetensors", origin["quantized"])
    elif fault in {"embedding", "scale"}:
        with origin[fault].open("r+b") as stream:
            stream.truncate(1)
    elif fault in {"layers", "experts", "vocab"}:
        key = {"layers": "num_hidden_layers", "experts": "num_experts", "vocab": "vocab_size"}[fault]
        origin["text"][key] -= 1
        (origin["hf"] / "config.json").write_text(json.dumps({"text_config": origin["text"]}))
    else:
        (origin["source"] / WRAPPER).write_text("class Wrong: pass\n")
    result = prepare_target_pytorch(origin["source"], origin["config"], origin["output"])
    assert result["status"] == "SOURCE_BLOCKED" and result["export"] == "EXPORT_NOT_RUN"
    assert result["error"]
    assert not (origin["output"] / "export.sh").exists()
    assert json.loads((origin["output"] / "result.json").read_text())["status"] == "SOURCE_BLOCKED"


def test_original_loader_scan_cannot_use_only_index_referenced_nonmodel_shard(origin):
    model = origin["hf"] / "model-00001.safetensors"
    model.rename(origin["hf"] / "unrelated.safetensors")
    with pytest.raises(ValueError, match=r"model\*"):
        inspect_target_pytorch_resources(origin["config"])


def test_duplicate_clean_hf_names_are_rejected(origin):
    name, value = next(iter(origin["base"].items()))
    sparse_shard(origin["hf"] / "model-00002.safetensors", {name: value})
    with pytest.raises(ValueError, match="ambiguous"):
        inspect_target_pytorch_resources(origin["config"])


def test_cli_reports_source_blocked_without_a_model_import(origin):
    result = subprocess.run([sys.executable, str(REPO / "targets/cpro/scripts/prepare_target_pytorch.py"),
        "--source-root", str(origin["source"]), "--config", str(origin["config"]) + ".missing",
        "--output", str(origin["output"])], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "SOURCE_BLOCKED"


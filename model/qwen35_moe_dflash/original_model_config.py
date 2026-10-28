"""Read the user's original YAML without executing its export project.

Paths belong to the configuration, not the shell or a repository default.
Original HF tensors, quantized safetensors, and embedding files are distinct
sources. No quantization, weight copying or remote access happens here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OriginalModelConfig:
    source: Path
    model_file: Path
    quanted_pth: Path
    embedding_weight_path: Path
    embedding_scale_path: Path
    kv_cache_max_len: int

    @classmethod
    def from_yaml(cls, path):
        import yaml

        path = Path(path).expanduser().resolve()
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("model YAML must be a mapping")
        resolved = {}
        for key in ("model_file", "quanted_pth", "embedding_weight_path", "embedding_scale_path"):
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"model YAML requires {key}")
            value = Path(value).expanduser()
            resolved[key] = (value if value.is_absolute() else path.parent / value).resolve()
        length = raw.get("kv_cache_max_len", 2048)
        if type(length) is not int or length <= 0:
            raise ValueError("kv_cache_max_len must be a positive integer")
        return cls(source=path, kv_cache_max_len=length, **resolved)

    def validate_paths(self):
        for key in ("model_file", "quanted_pth"):
            if not getattr(self, key).is_dir():
                raise ValueError(f"YAML {key} directory missing: {getattr(self, key)}")
        for key in ("embedding_weight_path", "embedding_scale_path"):
            if not getattr(self, key).is_file():
                raise ValueError(f"YAML {key} file missing: {getattr(self, key)}")
        if not (self.model_file / "config.json").is_file():
            raise ValueError("model_file/config.json is missing")
        if not any(self.quanted_pth.glob("data*.safetensors")):
            raise ValueError("quanted_pth contains no data*.safetensors")

    def text_config(self):
        raw = json.loads((self.model_file / "config.json").read_text())
        if not isinstance(raw, dict):
            raise ValueError("checkpoint config.json must be a mapping")
        config = raw.get("text_config", raw)
        if not isinstance(config, dict):
            raise ValueError("checkpoint text_config must be a mapping")
        config = dict(config)
        config["kv_cache_max_len"] = self.kv_cache_max_len
        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]
        # Preserve the checkpoint values; no RMS +1 is applied during loading.
        if config.get("rope_parameters") is None:
            config["rope_parameters"] = dict(config.get("rope_scaling") or {})
        return config

    def embedding_rows(self, token_ids):
        """Return original INT8 rows and FP32 per-token scales for GE Data.

        Keep the two tensors separate. The target applies FP32 multiplication
        then its declared FP16 activation boundary, matching the PyTorch path.
        """
        config = self.text_config()
        vocab, hidden = config["vocab_size"], config["hidden_size"]
        if any(type(value) is not int or value <= 0 for value in (vocab, hidden)):
            raise ValueError("embedding vocab_size and hidden_size must be positive integers")
        ids = np.asarray(token_ids)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype.kind not in "iu":
            raise ValueError("token_ids must be nonempty integer [1, sequence]")
        if np.any(ids < 0) or np.any(ids >= vocab):
            raise ValueError("token_ids outside vocabulary")
        if self.embedding_weight_path.stat().st_size != vocab * hidden:
            raise ValueError("embedding_weight_path must contain INT8 [vocab_size, hidden_size]")
        if self.embedding_scale_path.stat().st_size != vocab * 4:
            raise ValueError("embedding_scale_path must contain FP32 [vocab_size, 1]")
        weights = np.memmap(self.embedding_weight_path, dtype=np.int8, mode="r", shape=(vocab, hidden))
        scales = np.memmap(self.embedding_scale_path, dtype="<f4", mode="r", shape=(vocab, 1))
        rows, factors = np.array(weights[ids], copy=True), np.array(scales[ids], copy=True)
        if not np.isfinite(factors).all() or np.any(factors < 0):
            raise ValueError("selected embedding scales must be finite and nonnegative")
        return rows, factors

    def load_pytorch(self, *, feature_layers, ops, quant_ops, device="cpu", recurrent_dtype=None):
        """Create the quantized target using exactly these four source paths."""
        from .fused.quantization import load_original_quantized_target

        self.validate_paths()
        options = {} if recurrent_dtype is None else {"recurrent_dtype": recurrent_dtype}
        return load_original_quantized_target(
            self.text_config(), model_file=self.model_file, quanted_pth=self.quanted_pth,
            embedding_weight_path=self.embedding_weight_path,
            embedding_scale_path=self.embedding_scale_path,
            feature_layers=feature_layers, ops=ops, quant_ops=quant_ops, device=device, **options)

    def record(self):
        return {"yaml": str(self.source), "model_file": str(self.model_file),
                "quanted_pth": str(self.quanted_pth),
                "embedding_weight_path": str(self.embedding_weight_path),
                "embedding_scale_path": str(self.embedding_scale_path),
                "kv_cache_max_len": self.kv_cache_max_len,
                "quantization": "REUSE_ORIGINAL_NO_REQUANTIZATION",
                "norm_weights": "RAW_HF_MAIN_OFFSET_ONCE_GATED_NO_OFFSET"}

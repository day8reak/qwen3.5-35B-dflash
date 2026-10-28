"""Stream original safetensors into GE FP16 payloads without importing PyTorch.

This is a floating conversion diagnostic, not INT4 compression. The existing
DOPT/TorchAir quantized path is retained separately.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np

from .hashing import sha256_file


class SafeTensorFiles:
    """Read the documented safetensors header and bounded contiguous slices."""

    MAX_DISCOVERY_ENTRIES = 4096
    MAX_DISCOVERY_SHARDS = 512
    MAX_DISCOVERY_HEADER_BYTES = 64 << 20

    def __init__(self, root, *, allow_indexless_shards=False, required_keys=None):
        self.root = Path(root).resolve()
        self.headers = {}
        self.header_records = {}
        if required_keys is not None:
            if isinstance(required_keys, (str, bytes)):
                raise ValueError("required_keys must contain tensor names, not one string")
            required_keys = tuple(required_keys)
            if any(not isinstance(key, str) or not key or key == "__metadata__" for key in required_keys):
                raise ValueError("required_keys must contain nonempty tensor names")
        self.required_keys = tuple(sorted(set(required_keys or ())))
        index = self.root / "model.safetensors.index.json"
        self.discovery = {"mode": None, "index_present": index.is_file(), "files": [],
                          "required_keys": list(self.required_keys), "selected_mapping": {},
                          "scope": "Local tensor locations only; not full checkpoint completeness or upstream weight identity."}
        if index.is_file():
            self.mapping = json.loads(index.read_text())["weight_map"]
            if not isinstance(self.mapping, dict) or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in self.mapping.items()):
                raise ValueError("invalid safetensors weight_map")
            self.discovery.update(mode="INDEX", index_sha256=sha256_file(index))
        elif allow_indexless_shards:
            self.mapping = self._discover_headers()
            self.discovery["mode"] = "HEADER_SCAN"
        else:
            path = self.root / "model.safetensors"
            header, _ = self._header(path)
            self.mapping = {key: path.name for key in header if key != "__metadata__"}
            self.discovery["mode"] = "MONOLITHIC"
        missing = sorted(set(self.required_keys) - self.mapping.keys())
        self.discovery["selected_mapping"] = {key: self.mapping[key] for key in self.required_keys if key in self.mapping}
        if missing:
            raise ValueError(f"Required checkpoint tensors missing from {self.root}: {', '.join(missing)}; "
                             f"mapping={self.discovery['mode']}, local headers inspected={len(self.headers)}. "
                             "Provide the safetensors files containing these exact tensors; other target decoder shards are not needed for standalone draft.")

    def _discover_headers(self):
        """Inspect bounded immediate shard headers without reading tensor payloads."""
        paths = []
        with os.scandir(self.root) as entries:
            for count, entry in enumerate(entries, 1):
                if count > self.MAX_DISCOVERY_ENTRIES:
                    raise ValueError("indexless safetensors discovery exceeds directory entry limit")
                if not entry.name.endswith(".safetensors"):
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"indexless checkpoint shard must be a regular file: {entry.name}")
                paths.append(Path(entry.path))
                if len(paths) > self.MAX_DISCOVERY_SHARDS:
                    raise ValueError("indexless safetensors discovery exceeds shard limit")
        if not paths:
            raise FileNotFoundError(f"No safetensors index or regular *.safetensors files found in {self.root}")
        mapping, total = {}, 0
        for path in sorted(paths):
            header, start = self._header(path, max_header_bytes=self.MAX_DISCOVERY_HEADER_BYTES - total)
            total += start - 8
            for key in header:
                if key == "__metadata__":
                    continue
                if key in mapping:
                    raise ValueError(f"Ambiguous checkpoint tensor {key}: present in both {mapping[key]} and {path.name}")
                mapping[key] = path.name
        self.discovery["header_bytes_read"] = total
        return mapping

    def _header(self, path, *, max_header_bytes=100_000_000):
        path = path.resolve()
        if self.root not in path.parents or not path.is_file():
            raise ValueError(f"checkpoint shard missing or outside checkpoint: {path.name}")
        if path not in self.headers:
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError("truncated safetensors length")
                size = struct.unpack("<Q", prefix)[0]
                if not 2 <= size <= min(100_000_000, max_header_bytes, path.stat().st_size - 8):
                    raise ValueError(f"invalid or over-budget safetensors header length: {path.name}")
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError(f"truncated safetensors header: {path.name}")
                def unique_object(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError(f"duplicate safetensors header key {key}: {path.name}")
                        result[key] = value
                    return result
                header = json.loads(raw, object_pairs_hook=unique_object)
                if not isinstance(header, dict):
                    raise ValueError(f"safetensors header must be an object: {path.name}")
            self.headers[path] = (header, size + 8)
            self.header_records[path] = {"file": path.relative_to(self.root).as_posix(),
                                         "bytes": path.stat().st_size, "header_bytes": size,
                                         "header_sha256": hashlib.sha256(raw).hexdigest()}
            self.discovery["files"] = list(self.header_records.values())
        return self.headers[path]

    def tensor(self, key, expected_shape):
        if key not in self.mapping:
            raise ValueError(f"checkpoint tensor missing: {key}")
        path = (self.root / self.mapping[key]).resolve()
        header, start = self._header(path)
        item = header[key]
        if tuple(item["shape"]) != tuple(expected_shape):
            raise ValueError(f"{key}: shape {item['shape']} != {list(expected_shape)}")
        sizes = {"BF16": 2, "F16": 2, "F32": 4}
        if item["dtype"] not in sizes:
            raise ValueError(f"{key}: unsupported source dtype {item['dtype']}")
        offset, end = item["data_offsets"]
        count = math.prod(expected_shape)
        if (not isinstance(offset, int) or not isinstance(end, int) or offset < 0
                or end - offset != count * sizes[item["dtype"]]
                or start + end > path.stat().st_size):
            raise ValueError(f"{key}: invalid safetensors data offsets")
        return path, start + offset, count, item["dtype"]

    def write_fp16(self, key, shape, destination, *, require_bf16=True, chunk_elements=1 << 20):
        path, offset, count, dtype = self.tensor(key, shape)
        if require_bf16 and dtype != "BF16":
            raise ValueError(f"{key}: original locked checkpoint must be BF16")
        if chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        digest = hashlib.sha256()
        source_digest = hashlib.sha256()
        with path.open("rb") as source, Path(destination).open("xb") as output:
            source.seek(offset)
            for begin in range(0, count, chunk_elements):
                n = min(chunk_elements, count - begin)
                width = 4 if dtype == "F32" else 2
                data = source.read(n * width)
                if len(data) != n * width:
                    raise ValueError(f"{key}: truncated tensor")
                source_digest.update(data)
                if dtype == "BF16":
                    values = (np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4")
                else:
                    values = np.frombuffer(data, dtype="<f4" if dtype == "F32" else "<f2")
                with np.errstate(over="ignore", invalid="ignore"):
                    half = values.astype("<f2")
                if not np.isfinite(half).all():
                    raise ValueError(f"{key}: nonfinite or FP16-overflowing source weights")
                payload = half.tobytes()
                output.write(payload)
                digest.update(payload)
        return {"bytes": count * 2, "sha256": digest.hexdigest(), "source_sha256": source_digest.hexdigest(), "source_tensor": key,
                "source_shard": path.name, "source_dtype": dtype}

    def source_files(self, names):
        return [{"file": name, "bytes": (self.root / name).stat().st_size,
                 "sha256": sha256_file(self.root / name)} for name in sorted(set(names))]


def checkpoint_key(key):
    if key.startswith("draft."):
        return "draft", key[len("draft."):]
    if not key.startswith("target."):
        raise ValueError(f"unknown weight namespace: {key}")
    suffix = key[len("target."):]
    return "target", suffix if suffix == "lm_head.weight" else "model.language_model." + suffix


def tiny_weights(specs, seed=17):
    """Deterministic small weights; independent of Torch and upstream RNG APIs."""
    rng = np.random.default_rng(seed)
    result = {}
    for key, shape in sorted(specs.items()):
        if key.endswith("A_log"):
            value = np.log(np.arange(1, math.prod(shape) + 1)).reshape(shape)
        elif key.endswith("dt_bias"):
            dt = np.exp(rng.uniform(math.log(.001), math.log(.1), shape))
            value = dt + np.log(-np.expm1(-dt))
        elif key.endswith("weight") and ("norm" in key or "layernorm" in key):
            # Target RMSNorm stores an offset; GDN gated norm and draft store
            # their effective scales. Do not share these initialization rules.
            direct_scale = key.startswith("draft.") or ".linear_attn.norm." in key
            value = np.full(shape, 1.0 if direct_scale else 0.0)
        else:
            value = rng.normal(0, .04, shape)
        result[key] = np.asarray(value, dtype="<f2")
    return result

"""Prepare original W4 carriers and untouched HF normalization parameters for GE.

Preparation is a local storage conversion. It neither requantizes weights nor
claims that the resulting graph or a vendor QPARAM encoding ran on the device.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import output_path
from .ge_weights import SafeTensorFiles, checkpoint_key
from .hashing import sha256_file
from .original_model_config import OriginalModelConfig
from .original_quant import OriginalQuantWeights, canonicalkeys, load_embedding
from .target_weight_shapes import required_weight_shapes


def _json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _hash_region(path, offset, count):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        remaining = count
        while remaining:
            data = stream.read(min(remaining, 8 << 20))
            if not data:
                raise ValueError(f"truncated source tensor while hashing: {path}")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def _reserve(path, shape, dtype):
    dtype = np.dtype(dtype)
    with path.open("xb") as stream:
        stream.truncate(math.prod(shape) * dtype.itemsize)
    return np.memmap(path, mode="r+", shape=tuple(shape), dtype=dtype)


def _weight_record(root, path, shape, dtype, **extra):
    size = math.prod(shape) * np.dtype(dtype).itemsize
    if path.stat().st_size != size:
        raise ValueError(f"prepared tensor size mismatch: {path.name}")
    return {"path": str(path.relative_to(root)), "shape": list(shape),
            "dtype": np.dtype(dtype).name, "bytes": size,
            "sha256": sha256_file(path), **extra}


def _embedding_record(config, text, *, inspect_only):
    shape = (text["vocab_size"], text["hidden_size"])
    if not inspect_only:
        # Maps data read-only; validates the original FP32 per-row scales.
        weights, scales = load_embedding(config.embedding_weight_path,
                                         config.embedding_scale_path, shape)
        del weights, scales
    result = {}
    for role, path, dims, dtype in (
        ("weight", config.embedding_weight_path, shape, "int8"),
        ("scale", config.embedding_scale_path, (shape[0], 1), "float32"),
    ):
        size = math.prod(dims) * np.dtype(dtype).itemsize
        if path.stat().st_size != size:
            raise ValueError(f"original embedding {role} file size mismatch")
        result[role] = {"path": str(path), "shape": list(dims), "dtype": dtype,
                        "bytes": size, "sha256": None if inspect_only else sha256_file(path)}
    result.update(storage="EXTERNAL_UNCHANGED", interpretation="INT8_ROWS_TIMES_FP32_ROW_SCALE",
                  payload_validation="NOT_READ_INSPECTION" if inspect_only else "SCALES_VALIDATED_AND_FILES_HASHED")
    return result


def prepare_original_target(config_yaml, output_dir, *, qparam_converter=None,
                            inspect_only=False, progress=None):
    """Create an exclusive preparation directory and return its manifest.

    ``progress`` receives one dictionary per completed canonical tensor.
    ``inspect_only`` reads model/quantization headers and checks embedding file
    sizes, but reads no weight payloads and performs no packing. Expert scales
    require a caller-supplied vendor converter before actual GE export.
    """
    root = output_path(Path(output_dir).expanduser())
    root.mkdir(parents=True, exist_ok=False)
    current, stage = None, "configuration"
    completed = 0
    try:
        config = OriginalModelConfig.from_yaml(config_yaml)
        config.validate_paths()
        text = config.text_config()
        inventory = required_weight_shapes(text)
        quant = OriginalQuantWeights(config.quanted_pth)
        hf = SafeTensorFiles(config.model_file)
        sources = config.record()
        sources.update(yaml_sha256=sha256_file(config.source),
                       text_config_sha256=sha256_file(config.model_file / "config.json"))
        manifest = {"schema_version": 1, "artifact_kind": "qwen35-original-quant-preparation",
                    "status": "PREPARING", "output": str(root), "config": text, "sources": sources,
                    "weights": {}, "quant_descriptors": {}, "hf_descriptors": {},
                    "device_ready": False, "export_ready": False,
                    "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
                    "inspection_only": bool(inspect_only),
                    "source_hash_scope": "TENSOR_PAYLOAD_BYTES" if not inspect_only else "HEADERS_AND_CONFIG_ONLY",
                    "norm_weights": "RAW_HF_NO_PRE_FOLD",
                    "quantization": "REUSE_ORIGINAL_NO_REQUANTIZATION"}
        stage = "embedding"
        manifest["embedding"] = _embedding_record(config, text, inspect_only=inspect_only)
        if not inspect_only:
            (root / "weights").mkdir()
        (root / "tensor_records").mkdir()
        for index, (key, shape) in enumerate(sorted(inventory.items())):
            current, stage = key, "tensor"
            if key == "target.embed_tokens.weight":
                record = {"key": key, "storage": "EXTERNAL", "embedding": manifest["embedding"]}
            else:
                try:
                    canonicalkeys(key)
                    is_quantized = True
                except ValueError:
                    is_quantized = False
                if is_quantized:
                    desc = quant.describe(key, shape)
                    if not inspect_only:
                        weight_path = root / "weights" / f"{index:04d}.weight.bin"
                        scale_path = root / "weights" / f"{index:04d}.scale.bin"
                        scale_dtype = "int64" if desc["kind"] == "expert" and qparam_converter is not None else "float32"
                        weight = _reserve(weight_path, desc["packed_shape"], "int32")
                        scale = _reserve(scale_path, desc["scale_shape"], scale_dtype)
                        packed = quant.packed(key, shape, out=weight, scale_out=scale,
                            qparam_converter=qparam_converter if desc["kind"] == "expert" else None)
                        weight.flush()
                        scale.flush()
                        desc = packed["metadata"]
                        for field in ("source_weight", "source_scale"):
                            source = desc[field]
                            source["sha256"] = _hash_region(config.quanted_pth / source["shard"],
                                                            source["offset"], source["bytes"])
                            source["header_sha256"] = quant.shards[source["shard"]]["header_sha256"]
                        quant._unchanged((desc["source_weight"]["shard"], desc["source_scale"]["shard"]))
                        manifest["weights"][key] = _weight_record(root, weight_path, desc["packed_shape"], "int32")
                        manifest["weights"][key + ".quant_scale"] = _weight_record(root, scale_path, desc["scale_shape"], scale_dtype)
                        del packed, weight, scale
                    manifest["quant_descriptors"][key] = desc
                    record = {"key": key, "quant_descriptor": desc,
                              "weight": manifest["weights"].get(key),
                              "scale": manifest["weights"].get(key + ".quant_scale")}
                else:
                    _, hf_key = checkpoint_key(key)
                    path, offset, count, dtype = hf.tensor(hf_key, shape)
                    source = {"tensor": hf_key, "shard": path.name, "shape": list(shape),
                              "dtype": dtype, "offset": offset,
                              "bytes": count * (4 if dtype == "F32" else 2)}
                    if not inspect_only:
                        output = root / "weights" / f"{index:04d}.fp16.bin"
                        # Raw HF offsets/scales are copied without any RMS +1 fold.
                        hf.write_fp16(hf_key, shape, output, require_bf16=False)
                        source["sha256"] = _hash_region(path, offset, source["bytes"])
                        manifest["weights"][key] = _weight_record(root, output, shape, "float16")
                    manifest["hf_descriptors"][key] = source
                    record = {"key": key, "hf_descriptor": source,
                              "weight": manifest["weights"].get(key)}
            _json_exclusive(root / "tensor_records" / f"{index:04d}.json", record)
            completed += 1
            if progress is not None:
                progress({"key": key, "completed": completed, "total": len(inventory),
                          "status": "INSPECTED" if inspect_only else "PREPARED"})
        pending = [key for key, desc in manifest["quant_descriptors"].items()
                   if desc["scale_encoding"] == "QPARAM_REQUIRED"]
        manifest["pending_expert_qparam"] = pending
        manifest["status"] = "INSPECTED" if inspect_only else ("QPARAM_PENDING" if pending else "PREPARED")
        manifest["export_ready"] = not inspect_only and not pending
        manifest["completed_tensors"] = completed
        stage = "manifest"
        _json_exclusive(root / "manifest.json", manifest)
        return manifest
    except Exception as error:
        _json_exclusive(root / "failure.json", {"status": "FAILED", "stage": stage,
                        "tensor": current, "completed_tensors": completed,
                        "error": f"{type(error).__name__}: {error}", "device_ready": False})
        raise

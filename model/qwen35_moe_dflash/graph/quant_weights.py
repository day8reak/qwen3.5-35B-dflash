"""Bounded, portable weight preparation; these files are not NPU packed payloads.

INT4 stores logical [N, K] (or [E, N, K]) order, two signed two's
complement values per byte, first value in the low nibble. In particular,
this is NOT I4ZN and does not establish a hardware ABI or numerical gate.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import math
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.graph.weights import SafeTensorFiles


INT4_FORMAT = "PORTABLE_LOGICAL_INT4_ROW_MAJOR_LOW_NIBBLE_FIRST"
INT8_FORMAT = "PORTABLE_LOGICAL_INT8_ROW_MAJOR"
FP16_FORMAT = "PORTABLE_LOGICAL_FP16_ROW_MAJOR_LE"
MAX_CHUNK_ELEMENTS = 1 << 24


def _shape(shape, ranks=None):
    shape = tuple(shape)
    if not shape or (ranks is not None and len(shape) not in ranks):
        raise ValueError(f"unsupported tensor rank: {len(shape)}")
    if any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("shape dimensions must be positive integers")
    return shape


def _chunk_size(chunk_elements, unit=1):
    if type(chunk_elements) is not int or not unit <= chunk_elements <= MAX_CHUNK_ELEMENTS:
        raise ValueError(f"chunk_elements must be an integer in [{unit}, {MAX_CHUNK_ELEMENTS}]")
    # Round down to complete groups/rows while respecting the memory bound.
    return chunk_elements // unit * unit


def _source(reader, key, shape, require_bf16):
    if not isinstance(reader, SafeTensorFiles):
        raise TypeError("reader must be SafeTensorFiles")
    path, offset, count, dtype = reader.tensor(key, shape)
    if require_bf16 and dtype != "BF16":
        raise ValueError(f"{key}: original locked checkpoint must be BF16")
    return path, offset, count, dtype


def _chunks(stream, count, dtype, chunk_elements, key, digest):
    width = 4 if dtype == "F32" else 2
    for begin in range(0, count, chunk_elements):
        n = min(chunk_elements, count - begin)
        data = stream.read(n * width)
        if len(data) != n * width:
            raise ValueError(f"{key}: truncated tensor")
        digest.update(data)
        if dtype == "BF16":
            original = (np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4")
        else:
            original = np.frombuffer(data, dtype="<f4" if dtype == "F32" else "<f2")
        with np.errstate(over="ignore", invalid="ignore"):
            half = original.astype("<f2")
        if not np.isfinite(half).all():
            raise ValueError(f"{key}: nonfinite or FP16-overflowing source weights")
        yield original, half


class _Output:
    def __init__(self, stack, path, shape, dtype, format_name):
        self.path = Path(path).resolve()
        self.stream = stack.enter_context(self.path.open("xb"))
        self.shape, self.dtype, self.format = list(shape), dtype, format_name
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, array):
        payload = array.tobytes(order="C")
        self.stream.write(payload)
        self.digest.update(payload)
        self.bytes += len(payload)

    def metadata(self):
        return {"path": str(self.path), "shape": self.shape, "dtype": self.dtype,
                "format": self.format, "bytes": self.bytes,
                "sha256": self.digest.hexdigest()}


class _Error:
    def __init__(self):
        self.maximum = self.squared_error = self.squared_reference = 0.0

    def update(self, reference, actual):
        reference = reference.astype(np.float64)
        error = actual.astype(np.float64) - reference
        self.maximum = max(self.maximum, float(np.max(np.abs(error))))
        self.squared_error += float(np.sum(error * error, dtype=np.float64))
        self.squared_reference += float(np.sum(reference * reference, dtype=np.float64))

    def metadata(self, reference):
        relative = math.sqrt(self.squared_error / self.squared_reference) if self.squared_reference else 0.0
        return {"reference": reference, "max_abs": self.maximum, "relative_l2": relative}


def _source_metadata(key, path, dtype, count, digest):
    return {"tensor": key, "shard": path.name, "dtype": dtype,
            "bytes": count * (4 if dtype == "F32" else 2), "sha256": digest.hexdigest()}


def _write_quantized(reader, key, shape, weights_path, scales_path, *, bits,
                     require_bf16, chunk_elements):
    shape = _shape(shape, (2, 3) if bits == 4 else (2,))
    group_size = 256 if bits == 4 else shape[-1]
    if shape[-1] % group_size:
        raise ValueError(f"{key}: K must be divisible by {group_size}")
    chunk_elements = _chunk_size(chunk_elements, group_size)
    path, offset, count, dtype = _source(reader, key, shape, require_bf16)
    weights_path, scales_path = Path(weights_path).resolve(), Path(scales_path).resolve()
    if weights_path == scales_path:
        raise ValueError("weights and scales destinations must differ")
    # Validate both names before creating either; open('xb') also protects races.
    if weights_path.exists() or scales_path.exists():
        raise FileExistsError("weight preparation requires new weight and scale files")
    denominator = 7 if bits == 4 else 127
    minimum = -8 if bits == 4 else -128
    format_name = INT4_FORMAT if bits == 4 else INT8_FORMAT
    scales_shape = (*shape[:-1], shape[-1] // group_size) if bits == 4 else shape[:-1]
    digest, half_digest = hashlib.sha256(), hashlib.sha256()
    error = _Error()
    zero_groups = 0
    with ExitStack() as stack:
        source = stack.enter_context(path.open("rb"))
        source.seek(offset)
        weights = _Output(stack, weights_path, shape, "int4" if bits == 4 else "int8", format_name)
        scales = _Output(stack, scales_path, scales_shape, "float32", "LOGICAL_SCALES_ROW_MAJOR_FP32_LE")
        for _, half in _chunks(source, count, dtype, chunk_elements, key, digest):
            half_digest.update(half.tobytes())
            values = half.astype(np.float32).reshape(-1, group_size)
            maxima = np.max(np.abs(values), axis=-1)
            zero = maxima == 0
            zero_groups += int(np.count_nonzero(zero))
            scale = maxima / np.float32(denominator)
            scale[zero] = np.float32(1)
            quant = np.clip(np.rint(values / scale[:, None]), minimum, denominator).astype(np.int8)
            error.update(values, quant.astype(np.float32) * scale[:, None])
            if bits == 4:
                nibbles = quant.reshape(-1).astype(np.uint8) & np.uint8(15)
                weights.write(nibbles[0::2] | (nibbles[1::2] << np.uint8(4)))
            else:
                weights.write(quant.reshape(-1))
            scales.write(scale.astype("<f4"))
    return {
        "format": format_name, "device_ready": False, "device_pack": "NOT_PERFORMED",
        "source": _source_metadata(key, path, dtype, count, digest),
        "fp16_source_sha256": half_digest.hexdigest(),
        "weights": weights.metadata(), "scales": scales.metadata(),
        "quantization": {"bits": bits, "group_size": group_size,
                         "axis": "last", "source_preprocessing": "CAST_TO_FP16",
                         "statistics_dtype": "float32", "rounding": "TIES_TO_EVEN",
                         "scale_formula": f"max(abs(group))/{denominator}",
                         "clip": [minimum, denominator], "zero_group_scale": 1.0,
                         "zero_groups": zero_groups, "chunk_elements": chunk_elements},
        "error": error.metadata("FP16_NORMALIZED_SOURCE"),
        "validation": "HOST_QUANTIZATION_DIAGNOSTIC_ONLY",
    }


def write_int4(reader, key, shape, weights_path, scales_path, *, group_size=256,
               require_bf16=True, chunk_elements=1 << 20):
    """Write logical INT4 and FP32 group scales from a matrix/expert bank.

    Chunks are rounded down to whole groups, never enlarged. Failed calls may
    retain partial newly created files; no metadata is returned for such files.
    """
    if type(group_size) is not int or group_size != 256:
        raise ValueError("only group_size=256 is supported")
    return _write_quantized(reader, key, shape, weights_path, scales_path, bits=4,
                            require_bf16=require_bf16, chunk_elements=chunk_elements)


def write_embedding_int8(reader, key, shape, weights_path, scales_path, *,
                         require_bf16=True, chunk_elements=1 << 20):
    """Write [N,K] logical INT8 embeddings and one FP32 scale per row."""
    return _write_quantized(reader, key, shape, weights_path, scales_path, bits=8,
                            require_bf16=require_bf16, chunk_elements=chunk_elements)


def write_fp16(reader, key, shape, destination, *, require_bf16=True,
               chunk_elements=1 << 20):
    """Write dense little-endian FP16, also hashing source tensor bytes."""
    shape = _shape(shape)
    chunk_elements = _chunk_size(chunk_elements)
    path, offset, count, dtype = _source(reader, key, shape, require_bf16)
    digest, error = hashlib.sha256(), _Error()
    with ExitStack() as stack:
        source = stack.enter_context(path.open("rb"))
        source.seek(offset)
        weights = _Output(stack, destination, shape, "float16", FP16_FORMAT)
        for original, half in _chunks(source, count, dtype, chunk_elements, key, digest):
            error.update(original, half)
            weights.write(half)
    return {"format": FP16_FORMAT, "device_ready": False, "device_pack": "NOT_PERFORMED",
            "source": _source_metadata(key, path, dtype, count, digest),
            "weights": weights.metadata(),
            "error": error.metadata("SOURCE_FLOAT32"),
            "validation": "HOST_CONVERSION_DIAGNOSTIC_ONLY"}

"""Read existing group-256 quantized weights without changing quantization.

I4ZN byte order follows the supplied original packing functions. This module
does not import torch_npu, invent QPARAM encodings, or establish device support.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np


GROUP_SIZE = 256
MAX_CHUNK_ELEMENTS = 1 << 20
_WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
           "I32": 4, "U32": 4, "I64": 8, "U64": 8,
           "F16": 2, "BF16": 2, "F32": 4, "F64": 8}
_NUMPY_DTYPES = {"I8": np.dtype("i1"), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
_LINEARS = frozenset({
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
    "linear_attn.in_proj_a", "linear_attn.out_proj", "self_attn.q_proj",
    "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
})
_EXPERTS = frozenset({"mlp.experts.gate_up_proj", "mlp.experts.down_proj"})


def canonicalkeys(key):
    """Map a recognized target weight name to the original quantized keys."""
    if key == "target.lm_head.weight":
        stem = "lm_head"
    else:
        match = re.fullmatch(r"target\.layers\.([0-9]+)\.(.+)", key)
        if match is None:
            raise ValueError(f"not an original target quantized weight: {key}")
        suffix = match[2]
        if suffix.endswith(".weight") and suffix[:-7] in _LINEARS:
            suffix = suffix[:-7]
        elif suffix not in _EXPERTS:
            raise ValueError(f"not an original target quantized weight: {key}")
        stem = f"model.layers.{match[1]}.{suffix}"
    return stem + "_quant_weight", stem + "_quant_scale"


def _shape(shape, ranks=(2, 3)):
    shape = tuple(shape)
    if len(shape) not in ranks or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("weight shape must have positive integer dimensions and rank two or three")
    return shape


def _chunk(chunk_elements):
    if type(chunk_elements) is not int or not 1024 <= chunk_elements <= 1 << 24:
        raise ValueError("chunk_elements must be an integer between 1024 and 2**24")
    return chunk_elements


def _output(out, shape, dtype):
    dtype = np.dtype(dtype)
    if out is None:
        return np.empty(shape, dtype=dtype)
    if (not isinstance(out, np.ndarray) or out.shape != tuple(shape) or out.dtype != dtype
            or not out.flags.c_contiguous or not out.flags.writeable):
        raise ValueError(f"output must be writable contiguous {dtype} with shape {tuple(shape)}")
    return out


def _overlap(source, output):
    if np.may_share_memory(source, output):
        raise ValueError("packing output cannot alias source data")
    if isinstance(source, np.memmap) and isinstance(output, np.memmap):
        if Path(source.filename).resolve() == Path(output.filename).resolve():
            raise ValueError("packing output cannot use the source file")


def pack_i4zn(logical, *, out=None, chunk_elements=MAX_CHUNK_ELEMENTS):
    """Pack source [N,K]/[E,N,K] signed I4 values into the original I4ZN carrier.

    Only K64 by N16 tiles are materialized. Output storage is int32
    [K,N/8]/[E,K,N/8]; each byte contains adjacent K, even low and odd high.
    An output memmap can avoid allocating a complete expert bank in RAM.
    """
    if not isinstance(logical, np.ndarray) or logical.dtype != np.dtype("i1"):
        raise ValueError("logical original quantized weights must have dtype int8")
    shape = _shape(logical.shape)
    chunk_elements = _chunk(chunk_elements)
    n, k = shape[-2:]
    if k % 64 or n % 16:
        raise ValueError("I4ZN requires K divisible by 64 and N divisible by 16")
    packed_shape = (*shape[:-2], k, n // 8)
    result = _output(out, packed_shape, "<i4")
    _overlap(logical, result)
    batches = 1 if len(shape) == 2 else shape[0]
    source = logical[None, ...] if len(shape) == 2 else logical
    storage = result.view("u1").reshape(batches, k // 64, n // 16, 16, 32)
    channels_per_chunk = max(16, chunk_elements // 1024 * 16)
    for batch in range(batches):
        for kb in range(k // 64):
            for begin in range(0, n, channels_per_chunk):
                end = min(n, begin + channels_per_chunk)
                slab = source[batch, begin:end, kb * 64:(kb + 1) * 64]
                if np.any(slab < -8) or np.any(slab > 7):
                    raise ValueError("original quantized weight contains values outside signed I4 [-8,7]")
                nibble = slab.astype("u1") & np.uint8(15)
                packed = nibble[:, 0::2] | (nibble[:, 1::2] << np.uint8(4))
                storage[batch, kb, begin // 16:end // 16] = packed.reshape(-1, 16, 32)
    return result


def _transpose_scale(source, *, expert, out=None, chunk_elements=MAX_CHUNK_ELEMENTS):
    if source.dtype not in (np.dtype("<f2"), np.dtype("<f4")):
        raise ValueError("original scales must be float16 or float32")
    shape = source.shape if expert else source.shape[:-1]
    batches = shape[0] if expert else 1
    n, groups = shape[-2:]
    destination_shape = (batches, groups, n) if expert else (groups, n)
    result = _output(out, destination_shape, "<f4")
    _overlap(source, result)
    data, target = source.reshape(batches, n, groups), result.reshape(batches, groups, n)
    chunk_elements = _chunk(chunk_elements)
    # Bound both axes, including unusual configurations with many K groups.
    group_block = min(groups, 1024)
    rows_per_chunk = max(1, chunk_elements // group_block)
    for batch in range(batches):
        for gb in range(0, groups, group_block):
            for nb in range(0, n, rows_per_chunk):
                values = data[batch, nb:nb + rows_per_chunk, gb:gb + group_block].astype("<f4")
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("original quantization scales must be finite and nonnegative")
                target[batch, gb:gb + group_block, nb:nb + rows_per_chunk] = values.T
    return result


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate safetensors JSON key: " + key)
        result[key] = value
    return result


def _stamp(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


class OriginalQuantWeights:
    """Lazy, read-only access to original data*.safetensors, with or without an index."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        paths = sorted(self.root.glob("data*.safetensors"))
        if not 0 < len(paths) <= 4096:
            raise ValueError("original quantized directory must contain 1..4096 data*.safetensors shards")
        self.mapping, self.shards = {}, {}
        for named in paths:
            path = named.resolve()
            if named.is_symlink() or self.root not in path.parents or not path.is_file():
                raise ValueError("quantized shard must be a regular file inside the supplied directory")
            stamp = _stamp(path)
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"{path.name}: truncated safetensors prefix")
                header_bytes = struct.unpack("<Q", prefix)[0]
                if not 2 <= header_bytes <= min(32 << 20, stamp[2] - 8):
                    raise ValueError(f"{path.name}: invalid safetensors header length")
                payload = stream.read(header_bytes)
            header = json.loads(payload, object_pairs_hook=_unique_json)
            if not isinstance(header, dict):
                raise ValueError(f"{path.name}: safetensors header must be an object")
            data_start, ranges = 8 + header_bytes, []
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                if key in self.mapping:
                    raise ValueError(f"duplicate quantized tensor across shards: {key}")
                if not isinstance(spec, dict):
                    raise ValueError(f"{key}: invalid safetensors tensor descriptor")
                dtype, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
                if (not isinstance(dtype, str) or dtype not in _WIDTHS or not isinstance(shape, list)
                        or any(type(x) is not int or x < 0 for x in shape)
                        or not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(x) is not int for x in offsets)):
                    raise ValueError(f"{key}: invalid dtype, shape or offsets")
                begin, end = offsets
                size = math.prod(shape) * _WIDTHS[dtype]
                if begin < 0 or end < begin or end - begin != size or data_start + end > stamp[2]:
                    raise ValueError(f"{key}: invalid safetensors data extent")
                if size:
                    ranges.append((begin, end, key))
                self.mapping[key] = {"tensor": key, "shard": path.name, "dtype": dtype,
                                     "shape": shape, "offset": data_start + begin, "bytes": size}
            previous = 0
            for begin, end, key in sorted(ranges):
                if begin < previous:
                    raise ValueError(f"{key}: overlapping safetensors data extents")
                previous = end
            if _stamp(path) != stamp:
                raise ValueError("quantized shard changed while reading its header")
            self.shards[path.name] = {"stat": stamp, "header_sha256": hashlib.sha256(payload).hexdigest()}

    def _unchanged(self, names):
        for name in set(names):
            if _stamp(self.root / name) != self.shards[name]["stat"]:
                raise ValueError(f"quantized shard changed since header validation: {name}")

    def describe(self, key, shape):
        """Validate names/shapes/dtypes from headers only; never read tensor payloads."""
        shape = _shape(shape)
        weight_key, scale_key = canonicalkeys(key)
        expert = key.endswith(tuple(_EXPERTS))
        if len(shape) != (3 if expert else 2):
            raise ValueError(f"{key}: rank does not match linear/expert role")
        n, k = shape[-2:]
        if k % GROUP_SIZE or n % 16:
            raise ValueError(f"{key}: group-256 I4ZN requires K%256=0 and N%16=0")
        scale_shape = (*shape[:-1], k // GROUP_SIZE) if expert else (n, k // GROUP_SIZE, 1)
        for name, expected_shape, types in ((weight_key, shape, {"I8"}),
                                             (scale_key, scale_shape, {"F16", "F32"})):
            record = self.mapping.get(name)
            if record is None:
                raise ValueError("original quantized tensor missing: " + name)
            if tuple(record["shape"]) != expected_shape or record["dtype"] not in types:
                raise ValueError(f"{name}: expected {list(expected_shape)} with dtype in {sorted(types)}")
        weight, scale = self.mapping[weight_key], self.mapping[scale_key]
        self._unchanged((weight["shard"], scale["shard"]))
        return {"key": key, "logical_shape": list(shape), "kind": "expert" if expert else "linear",
                "source_weight": dict(weight), "source_scale": dict(scale),
                "packed_shape": [*shape[:-2], k, n // 8], "packed_dtype": "int32",
                "scale_shape": [*shape[:-2], k // GROUP_SIZE, n], "scale_dtype": "float32",
                "group_size": GROUP_SIZE,
                "scale_encoding": "QPARAM_REQUIRED" if expert else "FP32_GROUP_SCALE",
                "quantization": "REUSED_ORIGINAL_VALUES_NO_REQUANTIZATION",
                "packing": "I4ZN_K64_N16_ADJACENT_K_LOW_NIBBLE_FIRST",
                "device_validation": "NOT_RUN"}

    def _views(self, metadata):
        arrays = []
        for field in ("source_weight", "source_scale"):
            spec = metadata[field]
            arrays.append(np.memmap(self.root / spec["shard"], mode="r",
                                    dtype=_NUMPY_DTYPES[spec["dtype"]], offset=spec["offset"],
                                    shape=tuple(spec["shape"])))
        return tuple(arrays)

    def logical(self, key, shape):
        """Return validated read-only original int8 weights and original float scales."""
        metadata = self.describe(key, shape)
        weight, scale = self._views(metadata)
        for source, is_scale in ((weight, False), (scale, True)):
            flat = source.reshape(-1)
            for start in range(0, flat.size, MAX_CHUNK_ELEMENTS):
                values = flat[start:start + MAX_CHUNK_ELEMENTS]
                if is_scale:
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError("original quantization scales must be finite and nonnegative")
                elif np.any(values < -8) or np.any(values > 7):
                    raise ValueError("original quantized weight contains values outside signed I4 [-8,7]")
        self._unchanged((metadata["source_weight"]["shard"], metadata["source_scale"]["shard"]))
        return weight, scale

    def packed(self, key, shape, *, out=None, scale_out=None, qparam_converter=None,
               chunk_elements=MAX_CHUNK_ELEMENTS):
        """Return packed arrays and metadata, optionally using an explicit vendor QPARAM converter.

        qparam_converter accepts a contiguous FP32 array [E,K/256,N] and
        must return an equally shaped int64 ndarray. With no converter,
        expert scales remain FP32 and metadata explicitly says QPARAM_REQUIRED.
        """
        metadata = self.describe(key, shape)
        if qparam_converter is not None and (not callable(qparam_converter) or metadata["kind"] != "expert"):
            raise ValueError("QPARAM converter must be callable and is only used for expert scales")
        weight, scale = self._views(metadata)
        for output in (out, scale_out):
            if output is not None:
                _overlap(weight, output)
                _overlap(scale, output)
        if out is not None and scale_out is not None:
            _overlap(out, scale_out)
        result = pack_i4zn(weight, out=out, chunk_elements=chunk_elements)
        floats = _transpose_scale(scale, expert=metadata["kind"] == "expert",
                                  out=scale_out if qparam_converter is None else None,
                                  chunk_elements=chunk_elements)
        if qparam_converter is not None:
            converted = qparam_converter(floats)
            if (not isinstance(converted, np.ndarray) or converted.dtype != np.dtype("<i8")
                    or converted.shape != floats.shape):
                raise ValueError("vendor QPARAM converter must return int64 with unchanged scale shape")
            final_scale = _output(scale_out, floats.shape, "<i8")
            final_scale[...] = converted
            metadata.update(scale_dtype="int64", scale_encoding="QPARAM_VENDOR_CONVERTED",
                            qparam_source="EXPLICIT_CALLER_SUPPLIED_CONVERTER_UNVERIFIED_DEVICE")
        else:
            final_scale = floats
        self._unchanged((metadata["source_weight"]["shard"], metadata["source_scale"]["shard"]))
        return {"weight": result, "scale": final_scale, "metadata": metadata}


def load_embedding(weight_path, scale_path, shape):
    """Map original external embedding int8[N,H] and FP32[N,1] files read-only."""
    shape = _shape(shape, ranks=(2,))
    weight_path, scale_path = Path(weight_path).resolve(), Path(scale_path).resolve()
    for path, size in ((weight_path, math.prod(shape)), (scale_path, shape[0] * 4)):
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"original embedding file size mismatch: {path}")
    weight = np.memmap(weight_path, mode="r", dtype="i1", shape=shape)
    scale = np.memmap(scale_path, mode="r", dtype="<f4", shape=(shape[0], 1))
    for begin in range(0, shape[0], MAX_CHUNK_ELEMENTS):
        values = scale[begin:begin + MAX_CHUNK_ELEMENTS]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("embedding scales must be finite and nonnegative")
    return weight, scale

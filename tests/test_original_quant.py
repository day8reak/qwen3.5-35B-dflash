"""Original quantized data stays logical I4; packing never requantizes it."""
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from qwen35_moe_dflash.original_quant import (
    OriginalQuantWeights, canonicalkeys, load_embedding, pack_i4zn,
)


LINEAR = "target.layers.0.linear_attn.in_proj_qkv.weight"
EXPERT = "target.layers.3.mlp.experts.gate_up_proj"
TYPES = {"i1": "I8", "f2": "F16", "f4": "F32", "i4": "I32"}


def shard(path, tensors):
    header, chunks, offset = {}, [], 0
    for key, data in tensors.items():
        data = np.asarray(data)
        raw = data.tobytes()
        header[key] = {"dtype": TYPES[data.dtype.str[1:]], "shape": list(data.shape),
                       "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    raw_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(chunks))


def data_for(key=LINEAR, shape=(32, 512), scale_dtype="f2"):
    weight = (np.arange(np.prod(shape)).reshape(shape) % 16 - 8).astype("i1")
    if len(shape) == 2:
        scale_shape = (shape[0], shape[1] // 256, 1)
    else:
        scale_shape = (*shape[:-1], shape[-1] // 256)
    scale = ((np.arange(np.prod(scale_shape)) + 1) / 512).astype(scale_dtype).reshape(scale_shape)
    weight_key, scale_key = canonicalkeys(key)
    return {weight_key: weight, scale_key: scale}


def unpack_original(packed, logical_shape):
    """Independent byte-index oracle for the published K64/N16 carrier."""
    shape = tuple(logical_shape)
    n, k = shape[-2:]
    batches = 1 if len(shape) == 2 else shape[0]
    raw = packed.tobytes()
    decoded = np.empty((batches, n, k), dtype="i1")
    for e in range(batches):
        for ni in range(n):
            for ki in range(k):
                tile = (e * (k // 64) + ki // 64) * (n // 16) + ni // 16
                offset = tile * 512 + (ni % 16) * 32 + (ki % 64) // 2
                nibble = (raw[offset] >> (4 * (ki % 2))) & 15
                decoded[e, ni, ki] = nibble - 16 if nibble >= 8 else nibble
    return decoded.reshape(shape)


@pytest.mark.parametrize("shape", [(32, 128), (2, 48, 256)])
def test_i4zn_all_signed_values_and_tile_boundaries(shape):
    rng = np.random.default_rng(91)
    logical = rng.integers(-8, 8, size=shape, dtype="i1")
    before = logical.copy()
    packed = pack_i4zn(logical, chunk_elements=1024)
    assert packed.dtype == np.dtype("<i4")
    assert packed.shape == (*shape[:-2], shape[-1], shape[-2] // 8)
    np.testing.assert_array_equal(unpack_original(packed, shape), logical)
    np.testing.assert_array_equal(logical, before)
    np.testing.assert_array_equal(packed, pack_i4zn(logical, chunk_elements=8192))


def test_noncontiguous_logical_weight_and_memmap_destination(tmp_path):
    logical = np.arange(128 * 32, dtype="i1").reshape(128, 32).T % 16 - 8
    assert not logical.flags.c_contiguous
    out = np.memmap(tmp_path / "packed.bin", mode="w+", dtype="<i4", shape=(128, 4))
    assert pack_i4zn(logical, out=out) is out
    np.testing.assert_array_equal(unpack_original(out, logical.shape), logical)


@pytest.mark.parametrize("suffix", [
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj",
])
def test_canonical_mapping(suffix):
    stem = suffix[:-7] if suffix.endswith(".weight") else suffix
    assert canonicalkeys("target.layers.39." + suffix) == (
        "model.layers.39." + stem + "_quant_weight", "model.layers.39." + stem + "_quant_scale")
    assert canonicalkeys("target.lm_head.weight") == ("lm_head_quant_weight", "lm_head_quant_scale")


@pytest.mark.parametrize("key", ["draft.fc.weight", "target.embed_tokens.weight",
                                 "target.norm.weight", "target.layers.0.input_layernorm.weight",
                                 "target.layers.0.mlp.shared_expert_gate.weight"])
def test_nonquantized_keys_never_silently_fall_back(key):
    with pytest.raises(ValueError, match="not an original"):
        canonicalkeys(key)


def test_indexless_shards_header_describe_and_original_values(tmp_path):
    tensors = data_for()
    wkey, skey = canonicalkeys(LINEAR)
    shard(tmp_path / "data-00001.safetensors", {wkey: tensors[wkey]})
    shard(tmp_path / "data-00002.safetensors", {skey: tensors[skey]})
    reader = OriginalQuantWeights(tmp_path)
    meta = reader.describe(LINEAR, (32, 512))
    assert meta["source_weight"]["shard"] == "data-00001.safetensors"
    assert meta["packed_shape"] == [512, 4]
    assert meta["scale_shape"] == [2, 32]
    assert meta["scale_encoding"] == "FP32_GROUP_SCALE"
    weight, scale = reader.logical(LINEAR, (32, 512))
    assert not weight.flags.writeable and not scale.flags.writeable
    np.testing.assert_array_equal(weight, tensors[wkey])
    result = reader.packed(LINEAR, (32, 512))
    np.testing.assert_array_equal(unpack_original(result["weight"], (32, 512)), tensors[wkey])
    np.testing.assert_array_equal(result["scale"], tensors[skey][..., 0].astype("f4").T)
    assert result["scale"].dtype == np.dtype("<f4")


def test_describe_does_not_read_payload(tmp_path, monkeypatch):
    shard(tmp_path / "data.safetensors", data_for())
    reader = OriginalQuantWeights(tmp_path)
    def fail(*args, **kwargs):
        raise AssertionError("describe must use only headers")
    monkeypatch.setattr(reader, "_views", fail)
    assert reader.describe(LINEAR, (32, 512))["kind"] == "linear"


def test_expert_scales_require_explicit_qparam_converter(tmp_path):
    shape = (2, 32, 512)
    tensors = data_for(EXPERT, shape, "f4")
    shard(tmp_path / "data.safetensors", tensors)
    reader = OriginalQuantWeights(tmp_path)
    original = reader.packed(EXPERT, shape)
    expected_scale = tensors[canonicalkeys(EXPERT)[1]].transpose(0, 2, 1)
    assert original["metadata"]["scale_encoding"] == "QPARAM_REQUIRED"
    np.testing.assert_array_equal(original["scale"], expected_scale)
    np.testing.assert_array_equal(unpack_original(original["weight"], shape),
                                  tensors[canonicalkeys(EXPERT)[0]])
    observed = []
    def explicit_test_converter(scales):
        assert scales.flags.c_contiguous and scales.dtype == np.dtype("<f4")
        observed.append(scales.copy())
        # Test interface sentinel, deliberately not a vendor QPARAM encoding.
        return np.full(scales.shape, 193, dtype="<i8")
    out = np.memmap(tmp_path / "scale.bin", mode="w+", dtype="<i8", shape=(2, 2, 32))
    converted = reader.packed(EXPERT, shape, qparam_converter=explicit_test_converter, scale_out=out)
    np.testing.assert_array_equal(observed[0], expected_scale)
    assert converted["scale"] is out
    assert np.all(out == 193)
    assert converted["metadata"]["scale_encoding"] == "QPARAM_VENDOR_CONVERTED"
    assert converted["metadata"]["device_validation"] == "NOT_RUN"
    for invalid in (lambda x: x, lambda x: np.zeros((1,), dtype="i8")):
        with pytest.raises(ValueError, match="return int64"):
            reader.packed(EXPERT, shape, qparam_converter=invalid)


@pytest.mark.parametrize("alter", ["missing", "weight_dtype", "weight_shape", "scale_dtype", "scale_shape"])
def test_incompatible_source_rejected_before_loading_payload(tmp_path, alter):
    tensors = data_for()
    wkey, skey = canonicalkeys(LINEAR)
    if alter == "missing":
        del tensors[skey]
    elif alter == "weight_dtype":
        tensors[wkey] = tensors[wkey].astype("i4")
    elif alter == "weight_shape":
        tensors[wkey] = tensors[wkey].T
    elif alter == "scale_dtype":
        tensors[skey] = tensors[skey].astype("i4")
    else:
        tensors[skey] = tensors[skey][..., 0]
    shard(tmp_path / "data.safetensors", tensors)
    with pytest.raises(ValueError):
        OriginalQuantWeights(tmp_path).describe(LINEAR, (32, 512))


@pytest.mark.parametrize("bad_value", [-9, 8])
def test_out_of_range_i4_is_rejected(tmp_path, bad_value):
    tensors = data_for()
    tensors[canonicalkeys(LINEAR)[0]][0, -1] = bad_value
    shard(tmp_path / "data.safetensors", tensors)
    reader = OriginalQuantWeights(tmp_path)
    for method in (reader.logical, reader.packed):
        with pytest.raises(ValueError, match="outside signed I4"):
            method(LINEAR, (32, 512))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1])
def test_invalid_scale_is_rejected(tmp_path, bad_value):
    tensors = data_for()
    tensors[canonicalkeys(LINEAR)[1]][-1, 0, 0] = bad_value
    shard(tmp_path / "data.safetensors", tensors)
    reader = OriginalQuantWeights(tmp_path)
    for method in (reader.logical, reader.packed):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            method(LINEAR, (32, 512))


def test_source_changed_after_header_validation_is_rejected(tmp_path):
    path = tmp_path / "data.safetensors"
    shard(path, data_for())
    reader = OriginalQuantWeights(tmp_path)
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    # Some workspace filesystems coarsen timestamp updates for immediate writes.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed since header"):
        reader.packed(LINEAR, (32, 512))


def test_duplicate_shard_tensors_are_rejected(tmp_path):
    shard(tmp_path / "data1.safetensors", data_for())
    shard(tmp_path / "data2.safetensors", data_for())
    with pytest.raises(ValueError, match="duplicate quantized"):
        OriginalQuantWeights(tmp_path)


@pytest.mark.parametrize("payload", [b"", struct.pack("<Q", 1000) + b"{}",
                                     struct.pack("<Q", 5) + b"[]   "])
def test_invalid_shard_header_is_rejected(tmp_path, payload):
    (tmp_path / "data.safetensors").write_bytes(payload)
    with pytest.raises(ValueError):
        OriginalQuantWeights(tmp_path)


def test_overlapping_extents_are_rejected(tmp_path):
    header = {key: {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}
              for key in ("a", "b")}
    encoded = json.dumps(header).encode()
    (tmp_path / "data.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00\x00")
    with pytest.raises(ValueError, match="overlapping"):
        OriginalQuantWeights(tmp_path)


def test_source_output_same_file_rejected(tmp_path):
    path = tmp_path / "data.safetensors"
    shard(path, data_for())
    reader = OriginalQuantWeights(tmp_path)
    # r+ mapping does not itself alter the original file.
    output = np.memmap(path, mode="r+", dtype="<i4", shape=(512, 4))
    with pytest.raises(ValueError, match="source file"):
        reader.packed(LINEAR, (32, 512), out=output)


def test_embedding_preserves_int8_weights_and_float32_scale(tmp_path):
    weight_path, scale_path = tmp_path / "embedding.bin", tmp_path / "scale.bin"
    expected = np.arange(128, dtype="i1").reshape(8, 16)
    scales = np.arange(8, dtype="f4").reshape(8, 1)
    expected.tofile(weight_path)
    scales.tofile(scale_path)
    weight, scale = load_embedding(weight_path, scale_path, (8, 16))
    np.testing.assert_array_equal(weight, expected)
    np.testing.assert_array_equal(scale, scales)
    assert not weight.flags.writeable and not scale.flags.writeable
    with pytest.raises(ValueError, match="file size mismatch"):
        load_embedding(weight_path, scale_path, (7, 16))
    scales[0] = np.nan
    scales.tofile(scale_path)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        load_embedding(weight_path, scale_path, (8, 16))


@pytest.mark.parametrize("shape", [(32, 128), (15, 256), (32, 257), (1, 32, 512)])
def test_quant_reader_requires_group256_and_matching_roles(tmp_path, shape):
    shard(tmp_path / "data.safetensors", data_for())
    with pytest.raises(ValueError):
        OriginalQuantWeights(tmp_path).describe(LINEAR, shape)

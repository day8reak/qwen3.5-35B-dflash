import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from qwen35_moe_dflash.ge_quant_weights import (
    INT4_FORMAT, MAX_CHUNK_ELEMENTS, write_embedding_int8, write_fp16, write_int4,
)
from qwen35_moe_dflash.ge_weights import SafeTensorFiles


def _checkpoint(root, values, dtype="BF16"):
    root.mkdir()
    values = np.asarray(values)
    if dtype == "BF16":
        raw = (values.astype("<f4").view("<u4") >> 16).astype("<u2").tobytes()
    else:
        raw = values.astype("<f2" if dtype == "F16" else "<f4").tobytes()
    header = json.dumps({"weight": {"dtype": dtype, "shape": list(values.shape),
                                     "data_offsets": [0, len(raw)]}}).encode()
    path = root / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    return SafeTensorFiles(root), raw, path


def _assert_hash(record):
    payload = Path(record["path"]).read_bytes()
    assert record["bytes"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
def test_int4_known_signed_nibbles_rounding_zero_group_and_hashes(tmp_path, dtype):
    values = np.zeros((2, 256), dtype=np.float32)
    values[0, :11] = [-7, -6, -3.5, -2.5, -.5, 0, .5, 1.5, 2.5, 3.5, 7]
    reader, raw, _ = _checkpoint(tmp_path / "source", values, dtype)
    result = write_int4(reader, "weight", values.shape, tmp_path / "q.bin", tmp_path / "s.bin",
                        require_bf16=dtype == "BF16", chunk_elements=256)
    packed = (tmp_path / "q.bin").read_bytes()
    assert packed[:6] == bytes([0xA9, 0xEC, 0, 0x20, 0x42, 0x07])
    assert packed[6:] == bytes(250)
    assert (tmp_path / "s.bin").read_bytes() == struct.pack("<ff", 1, 1)
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["source"]["bytes"] == len(raw)
    assert result["quantization"]["zero_groups"] == 1
    assert result["quantization"]["zero_group_scale"] == 1.0
    assert result["format"] == INT4_FORMAT
    assert result["device_ready"] is False
    assert result["device_pack"] == "NOT_PERFORMED"
    assert result["weights"]["shape"] == [2, 256]
    assert result["scales"]["shape"] == [2, 1]
    expected_ref = [-7, -6, -3.5, -2.5, -.5, 0, .5, 1.5, 2.5, 3.5, 7]
    expected_q = [-7, -6, -4, -2, 0, 0, 0, 2, 2, 4, 7]
    relative = math.sqrt(sum((a-b)**2 for a,b in zip(expected_ref, expected_q)) /
                         sum(a*a for a in expected_ref))
    assert result["error"]["max_abs"] == .5
    assert result["error"]["relative_l2"] == pytest.approx(relative)
    _assert_hash(result["weights"])
    _assert_hash(result["scales"])


def test_expert_bank_chunk_boundaries_use_bounded_reads_and_preserve_order(tmp_path, monkeypatch):
    values = np.concatenate([np.full(256, (-1 if i % 2 else 1) * 7 * (i+1), np.float32)
                             for i in range(12)]).reshape(2, 3, 512)
    reader, raw, source_path = _checkpoint(tmp_path / "source", values)
    reader.tensor("weight", values.shape)  # Cache the small header before measuring tensor reads.
    original_open = Path.open
    reads = []

    class MeasuredRead:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def seek(self, *args):
            return self.stream.seek(*args)

        def read(self, size=-1):
            assert 0 < size <= 768 * 2
            reads.append(size)
            return self.stream.read(size)

    def measured_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return MeasuredRead(stream) if path == source_path and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", measured_open)
    result = write_int4(reader, "weight", values.shape, tmp_path / "q.bin", tmp_path / "s.bin",
                        chunk_elements=768)
    assert reads == [1536] * 4
    assert (tmp_path / "q.bin").read_bytes() == b"".join(
        bytes([0x99 if i % 2 else 0x77]) * 128 for i in range(12))
    np.testing.assert_array_equal(np.fromfile(tmp_path / "s.bin", dtype="<f4"), np.arange(1, 13))
    assert result["scales"]["shape"] == [2, 3, 2]
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["error"] == {"reference": "FP16_NORMALIZED_SOURCE", "max_abs": 0., "relative_l2": 0.}
    # Another partition must produce identical output, including a partial final chunk.
    monkeypatch.setattr(Path, "open", original_open)
    other = write_int4(reader, "weight", values.shape, tmp_path / "q2.bin", tmp_path / "s2.bin",
                       chunk_elements=1280)
    assert other["weights"]["sha256"] == result["weights"]["sha256"]
    assert other["scales"]["sha256"] == result["scales"]["sha256"]


def test_embedding_per_row_known_values_and_fp16_preprocessing(tmp_path):
    values = np.array([[-127, 127, -.5, .5, 1.5, 2.5, 1.0002, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [-254, 254, -1, 1, 3, 5, 2, 0]], dtype=np.float32)
    reader, _, _ = _checkpoint(tmp_path / "source", values, "F32")
    result = write_embedding_int8(reader, "weight", values.shape, tmp_path / "q.bin", tmp_path / "s.bin",
                                  require_bf16=False, chunk_elements=16)
    expected = [-127,127,0,0,2,2,1,0] + [0]*8 + [-127,127,0,0,2,2,1,0]
    assert (tmp_path / "q.bin").read_bytes() == struct.pack("24b", *expected)
    assert (tmp_path / "s.bin").read_bytes() == struct.pack("<fff", 1, 1, 2)
    assert result["scales"]["shape"] == [3]
    assert result["quantization"]["zero_groups"] == 1
    assert result["fp16_source_sha256"] == hashlib.sha256(values.astype("<f2").tobytes()).hexdigest()
    _assert_hash(result["weights"])
    _assert_hash(result["scales"])


def test_fp16_writer_tracks_conversion_error_and_exclusive_files(tmp_path):
    values = np.array([1.0002, -2, 3.5], dtype=np.float32)
    reader, raw, _ = _checkpoint(tmp_path / "source", values, "F32")
    destination = tmp_path / "half.bin"
    result = write_fp16(reader, "weight", values.shape, destination, require_bf16=False, chunk_elements=2)
    assert destination.read_bytes() == struct.pack("<eee", 1, -2, 3.5)
    assert result["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["error"]["max_abs"] == pytest.approx(float(values[0]) - 1)
    _assert_hash(result["weights"])
    with pytest.raises(FileExistsError):
        write_fp16(reader, "weight", values.shape, destination, require_bf16=False)
    assert destination.read_bytes() == struct.pack("<eee", 1, -2, 3.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 70000.])
@pytest.mark.parametrize("writer", [write_int4, write_embedding_int8, write_fp16])
def test_rejects_nonfinite_and_fp16_overflow(tmp_path, bad, writer):
    values = np.zeros((1, 256), dtype=np.float32)
    values[0, -1] = bad
    reader, _, _ = _checkpoint(tmp_path / "source", values, "F32")
    paths = [tmp_path / "q.bin"] if writer is write_fp16 else [tmp_path / "q.bin", tmp_path / "s.bin"]
    with pytest.raises(ValueError, match="nonfinite|FP16-overflowing"):
        writer(reader, "weight", values.shape, *paths, require_bf16=False)


@pytest.mark.parametrize("chunk", [0, -1, 255, True, 256.5, MAX_CHUNK_ELEMENTS + 1])
def test_invalid_quant_chunk_rejected_before_output(tmp_path, chunk):
    reader, _, _ = _checkpoint(tmp_path / "source", np.ones((1, 256)))
    with pytest.raises(ValueError, match="chunk_elements"):
        write_int4(reader, "weight", (1, 256), tmp_path / "q.bin", tmp_path / "s.bin", chunk_elements=chunk)
    assert not (tmp_path / "q.bin").exists()


@pytest.mark.parametrize("shape", [(256,), (1, 255), (1, 1, 1, 256), (0, 256), (True, 256), (1.0, 256)])
def test_invalid_shape_rejected(tmp_path, shape):
    reader, _, _ = _checkpoint(tmp_path / "source", np.ones((1, 256)))
    with pytest.raises(ValueError):
        write_int4(reader, "weight", shape, tmp_path / "q.bin", tmp_path / "s.bin")
    assert not (tmp_path / "q.bin").exists()


def test_exclusive_scale_and_locked_dtype_and_shape_checks(tmp_path):
    reader, _, _ = _checkpoint(tmp_path / "source", np.ones((1, 256)), "F16")
    with pytest.raises(ValueError, match="must be BF16"):
        write_int4(reader, "weight", (1, 256), tmp_path / "q.bin", tmp_path / "s.bin")
    with pytest.raises(ValueError, match="shape"):
        write_int4(reader, "weight", (2, 256), tmp_path / "q.bin", tmp_path / "s.bin", require_bf16=False)
    with pytest.raises(ValueError, match="group_size"):
        write_int4(reader, "weight", (1, 256), tmp_path / "q.bin", tmp_path / "s.bin", group_size=128)
    (tmp_path / "s.bin").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        write_int4(reader, "weight", (1, 256), tmp_path / "q.bin", tmp_path / "s.bin", require_bf16=False)
    assert (tmp_path / "s.bin").read_bytes() == b"preserve"
    assert not (tmp_path / "q.bin").exists()


def test_all_zero_relative_error_is_zero(tmp_path):
    reader, _, _ = _checkpoint(tmp_path / "source", np.zeros((1, 256)))
    result = write_int4(reader, "weight", (1, 256), tmp_path / "q.bin", tmp_path / "s.bin")
    assert result["error"]["max_abs"] == result["error"]["relative_l2"] == 0.
    assert (tmp_path / "s.bin").read_bytes() == struct.pack("<f", 1)

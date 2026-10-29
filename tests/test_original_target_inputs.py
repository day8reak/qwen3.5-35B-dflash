"""Host file inspections only; these fixtures never execute a target model."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from qwen35_moe_dflash import original_target_inputs as inspector


def make_inputs(root, sequence=64):
    values = {
        "new_kv_cache_pos": ("i", list(range(sequence))),
        "output_pos": ("i", [sequence - 1]),
        "allQLen": ("q", [2**42 + sequence]),
        "conv_state_pos": ("i", list(range(-64, 64))),
        "effective_length": ("h", [sequence]),
        "position_ids": ("i", list(range(100, 100 + sequence))),
        "embed_scale": ("f", [0.25] * sequence),
    }
    root.mkdir()
    for name, (code, data) in values.items():
        (root / (name + ".bin")).write_bytes(struct.pack("<" + code * len(data), *data))
    (root / "attention_mask.bin").write_bytes(struct.pack("<f", -1000.0) * (sequence * 2048))
    # Real model files are outside this inspector's read allowlist.
    for name in ("inputs_embeds.bin", "past_conv_state.bin", "past_recurrent_state.bin", "past_k.bin", "model.weight"):
        (root / name).write_bytes(b"DO NOT READ")
    return values


def rows(report):
    return {row["semantic"]: row for row in report["controls"]}


def test_exact_exporter_dtypes_shapes_and_complete_control_values(tmp_path):
    root = tmp_path / "inputs"
    values = make_inputs(root)
    report = inspector.inspect_original_target_inputs(root)
    found = rows(report)
    assert report["status"] == "INSPECTED"
    assert report["purpose"] == "NOT_A_CONTROL_GENERATOR"
    assert report["device_execution"] == "NOT_EXECUTED"
    assert report["dtype_binding"] == "FROM_ORIGINAL_EXPORTER_NOT_COMPILED_ABI"
    assert found["allQLen"]["dtype"] == "int64" and found["allQLen"]["bytes"] == 8
    assert found["effective_length"]["dtype"] == "int16" and found["effective_length"]["bytes"] == 2
    assert found["conv_state_pos"]["shape"] == [128]
    assert found["position_ids"]["shape"] == [1, 64]
    assert found["embed_scale"]["shape"] == [1, 64, 1]
    for name, (_, expected) in values.items():
        row = found[name]
        assert row["values"] == expected and row["values_complete"]
        assert row["sha256"] == hashlib.sha256((root / row["file"]).read_bytes()).hexdigest()
        assert row["source_stability"] == "UNCHANGED_DURING_READ"
    assert report["attention_mask"]["shape"] == [1, 1, 64, 2048]
    assert report["attention_mask"]["statistics"]["finite_negative_count"] == 64 * 2048


def test_one_token_gear_keeps_128_conv_control_entries(tmp_path):
    root = tmp_path / "inputs"
    make_inputs(root, 1)
    report = inspector.inspect_original_target_inputs(root, sequence_length=1)
    assert report["status"] == "INSPECTED"
    found = rows(report)
    assert found["new_kv_cache_pos"]["shape"] == [1]
    assert found["position_ids"]["shape"] == [1, 1]
    assert found["embed_scale"]["shape"] == [1, 1, 1]
    assert found["conv_state_pos"]["elements"] == 128
    assert found["conv_state_pos"]["values"] == list(range(-64, 64))
    assert report["attention_mask"]["expected_bytes"] == 2048 * 4


def test_only_allowlisted_files_opened_read_only_and_no_files_created(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    make_inputs(root)
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in root.iterdir()}
    original_open = os.open
    opened = []
    allowed = {row[2] + ".bin" for row in inspector._CONTROLS} | {"attention_mask.bin"}

    def checked_open(path, flags, *args, **kwargs):
        assert Path(path).name in allowed
        assert flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) == 0
        opened.append(Path(path).name)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(inspector.os, "open", checked_open)
    assert inspector.inspect_original_target_inputs(root)["status"] == "INSPECTED"
    assert set(opened) == allowed
    assert {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in root.iterdir()} == before


def test_missing_and_size_mismatch_return_no_default_values_or_unbounded_hash(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    make_inputs(root)
    (root / "conv_state_pos.bin").unlink()
    (root / "allQLen.bin").write_bytes(struct.pack("<i", 64))
    original_open = os.open

    def checked_open(path, flags, *args, **kwargs):
        assert Path(path).name not in {"conv_state_pos.bin", "allQLen.bin"}
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(inspector.os, "open", checked_open)
    report = inspector.inspect_original_target_inputs(root)
    assert report["status"] == "INCOMPLETE"
    found = rows(report)
    assert found["conv_state_pos"]["status"] == "MISSING"
    assert found["allQLen"]["reason"] == "SIZE_MISMATCH"
    for name in ("conv_state_pos", "allQLen"):
        assert "values" not in found[name] and found[name]["sha256"] is None


def test_symlink_is_not_read(tmp_path):
    root = tmp_path / "inputs"
    make_inputs(root)
    (root / "output_pos.bin").unlink()
    (root / "output_pos.bin").symlink_to(root / "effective_length.bin")
    row = rows(inspector.inspect_original_target_inputs(root))["output_pos"]
    assert row["status"] == "NOT_READ" and row["reason"] == "SYMLINK_OR_NONREGULAR_FILE"
    assert "values" not in row


def test_mask_is_chunked_and_nonfinite_values_remain_visible_in_json(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    make_inputs(root)
    data = [0.0, -0.0, -1.0, 2.0, math.nan, math.inf, -math.inf] + [0.0] * (64 * 2048 - 7)
    (root / "attention_mask.bin").write_bytes(struct.pack("<" + "f" * len(data), *data))
    scales = [math.nan, math.inf, -math.inf] + [0.25] * 61
    (root / "embed_scale.bin").write_bytes(struct.pack("<" + "f" * 64, *scales))
    original_unpack = struct.iter_unpack
    chunk_sizes = []

    def checked_unpack(code, raw):
        chunk_sizes.append(len(raw))
        assert len(raw) <= inspector._CHUNK_BYTES
        return original_unpack(code, raw)

    monkeypatch.setattr(inspector.struct, "iter_unpack", checked_unpack)
    report = inspector.inspect_original_target_inputs(root)
    stats = report["attention_mask"]["statistics"]
    assert report["status"] == "INSPECTED"
    assert stats["elements"] == 64 * 2048
    assert stats["nan_count"] == stats["positive_infinity_count"] == stats["negative_infinity_count"] == 1
    assert stats["finite_negative_count"] == stats["finite_positive_count"] == 1
    assert stats["zero_count"] == 64 * 2048 - 5
    assert stats["finite_min"] == -1.0 and stats["finite_max"] == 2.0
    assert rows(report)["embed_scale"]["values"][:3] == ["NaN", "+Infinity", "-Infinity"]
    assert len(chunk_sizes) > 8
    json.dumps(report, allow_nan=False)


def test_changed_during_read_discards_values_and_hash(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    make_inputs(root)
    original = inspector._mask_stats

    def changing(chunks):
        result = original(chunks)
        (root / "attention_mask.bin").write_bytes(b"changed during inspection")
        return result

    monkeypatch.setattr(inspector, "_mask_stats", changing)
    report = inspector.inspect_original_target_inputs(root)
    row = report["attention_mask"]
    assert report["status"] == "INCOMPLETE"
    assert row["status"] == "UNAVAILABLE" and "changed during read" in row["reason"]
    assert row["sha256"] is None and "statistics" not in row


def test_skip_mask_does_not_open_mask(tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    make_inputs(root)
    (root / "attention_mask.bin").unlink()
    report = inspector.inspect_original_target_inputs(root, include_mask=False)
    assert report["status"] == "INSPECTED"
    assert report["attention_mask"]["status"] == "NOT_REQUESTED"


@pytest.mark.parametrize("length", [0, 2, 63, 65, True, "64"])
def test_undeclared_sequence_lengths_rejected(tmp_path, length):
    with pytest.raises(ValueError, match="64 or 1"):
        inspector.inspect_original_target_inputs(tmp_path, sequence_length=length)


def test_cli_standalone_json_with_missing_files_and_no_bytecode(tmp_path):
    root = tmp_path / "inputs"
    make_inputs(root)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_original_target_inputs.py"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(script), "--input-dir", str(root)],
                               capture_output=True, text=True, env=env)
    assert completed.returncode == 0 and completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "INSPECTED"
    (root / "effective_length.bin").unlink()
    completed = subprocess.run([sys.executable, str(script), "--input-dir", str(root), "--skip-mask"],
                               capture_output=True, text=True, env=env)
    assert completed.returncode == 1 and completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["status"] == "INCOMPLETE" and rows(report)["effective_length"]["status"] == "MISSING"

"""Stat-only missing-input diagnostics never create or read a decode value."""
import builtins
import os
from pathlib import Path

import pytest

from qwen35_moe_dflash import target_decode_smoke_plan as planner
from test_target_decode_smoke_plan import case as bound_case


EXPECTED = {
    "inputs_embeds.bin": 2048, "attention_mask.bin": 8192, "position_ids.bin": 4,
    "past_conv_state.bin": 65536, "past_recurrent_state.bin": 1048576,
    "past_k.bin": 2097152, "past_v.bin": 2097152, "new_kv_cache_pos.bin": 4,
    "output_pos.bin": 4, "embed_scale.bin": 4, "allQLen.bin": 8,
    "conv_state_pos.bin": 512, "effective_length.bin": 2,
}


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "decode-inputs"
    root.mkdir()
    for name, size in EXPECTED.items():
        with (root / name).open("wb") as stream:
            stream.truncate(size)
    return root


def index(report):
    return {row["file"]: row for row in report["files"]}


def test_complete_extents_group_all89_slots_without_reading_payloads(inputs, monkeypatch):
    # Even same-size data of an arbitrary encoding is only an extent match.
    (inputs / "golden_logits.bin").write_bytes(b"this extra file is not an input")
    (inputs / "golden_k_0.bin").symlink_to(inputs / "past_k.bin")
    before = {name: (inputs / name).stat() for name in EXPECTED}
    def forbidden(*args, **kwargs):
        pytest.fail("extent inspection must not open, hash or enumerate payloads")
    with monkeypatch.context() as guard:
        for obj, name in ((builtins, "open"), (os, "open"), (Path, "open"),
                          (Path, "iterdir"), (Path, "glob"), (planner, "_source"),
                          (planner, "_read_small")):
            guard.setattr(obj, name, forbidden)
        report = planner.inspect_decode_input_extents(inputs)
    assert report["status"] == "DECODE_INPUT_EXTENTS_MATCHED"
    assert report["payload_access"] == "NONE_STAT_ONLY"
    assert report["encoding_validation"] == "NOT_PERFORMED"
    assert report["readability_validation"] == "NOT_OPENED_NOT_TESTED"
    assert report["device_execution"] == report["model_compilation"] == "NOT_RUN"
    assert report["input_mutation"] == "NONE"
    assert report["required_file_count"] == report["matched_file_count"] == 13
    assert report["failed_files"] == report["missing_files"] == []
    rows = index(report)
    assert {name: row["expected_bytes"] for name, row in rows.items()} == EXPECTED
    assert all(row["status"] == "REGULAR_EXTENT_MATCHED" for row in rows.values())
    assert sum(len(row["input_names"]) for row in rows.values()) == 89
    assert rows["conv_state_pos.bin"]["expected_dtype"] == "INT32"
    assert rows["conv_state_pos.bin"]["expected_shape"] == [128]
    assert rows["past_conv_state.bin"]["expected_shape"] == [1, 8192, 4]
    assert rows["allQLen.bin"]["expected_dtype"] == "INT64"
    for name, info in before.items():
        after = (inputs / name).stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_nlink) == (
            info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def test_empty_directory_reports_all13_missing_in_one_observation(tmp_path):
    report = planner.inspect_decode_input_extents(tmp_path)
    assert report["status"] == "DECODE_INPUT_EXTENTS_INCOMPLETE"
    assert report["matched_file_count"] == 0 and report["failed_file_count"] == 13
    assert set(report["missing_files"]) == set(EXPECTED)
    assert all(row["status"] == "MISSING" and row["observed_bytes"] is None for row in report["files"])
    assert not list(tmp_path.iterdir())


def test_mixed_missing_and_wrong_extents_are_reported_together(inputs):
    (inputs / "conv_state_pos.bin").unlink()
    (inputs / "allQLen.bin").unlink()
    (inputs / "inputs_embeds.bin").write_bytes(bytes(4096))
    (inputs / "past_conv_state.bin").write_bytes(bytes(49152))
    report = planner.inspect_decode_input_extents(inputs)
    rows = index(report)
    assert report["status"] == "DECODE_INPUT_EXTENTS_INCOMPLETE"
    assert report["failed_file_count"] == 4 and report["matched_file_count"] == 9
    assert set(report["missing_files"]) == {"conv_state_pos.bin", "allQLen.bin"}
    assert rows["inputs_embeds.bin"]["status"] == "WRONG_SIZE"
    assert rows["inputs_embeds.bin"]["observed_bytes"] == 4096
    assert rows["past_conv_state.bin"]["status"] == "WRONG_SIZE"
    assert rows["past_conv_state.bin"]["expected_bytes"] == 65536
    assert rows["past_conv_state.bin"]["observed_bytes"] == 49152


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_nonregular_input_is_never_opened_or_followed(inputs, monkeypatch, kind):
    path = inputs / "conv_state_pos.bin"
    path.unlink()
    if kind == "symlink":
        other = inputs / "not-an-input.bin"
        other.write_bytes(bytes(512))
        path.symlink_to(other)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with monkeypatch.context() as guard:
        guard.setattr(os, "open", lambda *a, **k: pytest.fail("nonregular input was opened"))
        report = planner.inspect_decode_input_extents(inputs)
    row = index(report)[path.name]
    assert row["status"] == "UNSAFE_OR_UNREADABLE"
    assert row["reason"] == ("symlink" if kind == "symlink" else "nonregular file")
    assert report["failed_file_count"] == 1


def test_stat_permission_error_is_per_file_and_does_not_hide_other_failures(inputs, monkeypatch):
    original = Path.lstat
    def guarded(path, *args, **kwargs):
        if path == inputs / "allQLen.bin":
            raise PermissionError("host-only denied metadata fixture")
        return original(path, *args, **kwargs)
    (inputs / "conv_state_pos.bin").unlink()
    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", guarded)
        report = planner.inspect_decode_input_extents(inputs)
    rows = index(report)
    assert rows["allQLen.bin"]["status"] == "UNSAFE_OR_UNREADABLE"
    assert "PermissionError" in rows["allQLen.bin"]["reason"]
    assert rows["conv_state_pos.bin"]["status"] == "MISSING"
    assert report["failed_file_count"] == 2


def test_input_directory_symlink_is_not_followed(inputs, tmp_path):
    alias = tmp_path / "input-alias"
    alias.symlink_to(inputs, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        planner.inspect_decode_input_extents(alias)


def test_planner_reports_every_bad_filename_before_any_input_hash(tmp_path, monkeypatch):
    case = bound_case.__wrapped__(tmp_path)
    (case["source"] / "conv_state_pos.bin").unlink()
    (case["source"] / "allQLen.bin").unlink()
    (case["source"] / "past_conv_state.bin").write_bytes(bytes(49152))
    monkeypatch.setattr(planner, "_source", lambda *a, **k: pytest.fail("input hash preceded aggregate extent check"))
    with pytest.raises(ValueError) as captured:
        planner.prepare_decode_smoke_plan(case["compilation"], case["probe"], case["source"])
    text = str(captured.value)
    assert all(name in text for name in ("conv_state_pos.bin", "allQLen.bin", "past_conv_state.bin"))
    assert "expected=512 observed=missing" in text
    assert "expected=8 observed=missing" in text
    assert "expected=65536 observed=49152" in text


def test_metadata_binding_still_precedes_input_extent_inspection(tmp_path, monkeypatch):
    case = bound_case.__wrapped__(tmp_path)
    report = case["compilation"] / "report.json"
    report.write_text(report.read_text() + " ")
    monkeypatch.setattr(planner, "inspect_decode_input_extents",
        lambda *a: pytest.fail("extent inspection preceded stale job metadata rejection"))
    with pytest.raises(ValueError, match="successful static1"):
        planner.prepare_decode_smoke_plan(case["compilation"], case["probe"], case["source"])

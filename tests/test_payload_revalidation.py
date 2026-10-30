"""Fresh SHA identities reconcile only ctime for declared decode payload roles."""
from copy import deepcopy
import os
import time

import pytest

from qwen35_moe_dflash.payload_revalidation import payload_identity_matches, payload_sequence_matches
from qwen35_moe_dflash.target_compile_failure import _read_small
from qwen35_moe_dflash.target_prefill_smoke_plan import _source


@pytest.mark.parametrize("role,relative,raw", [
    ("decode-input", "inputs/inputs_embeds.bin", b"\x81\x00"),
    ("decode-trial-copied", "inputs/past_k.bin", b"\x00\x00"),
    ("decode-trial-generated", "inputs/conv_state_pos.bin", b"\x01\x00"),
    ("decode-trial-input", "inputs/allQLen.bin", b"\x01\x00"),
    ("decode-output", "outputs/output_81", b"\x00\x00"),
    ("decode-log", "tool.stdout", b"success\n"),
    ("source-air", "export.air", b"bound fixture source graph"),
])
def test_each_role_keeps_receipt_and_returns_fresh_ctime_audit(tmp_path, role, relative, raw):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    recorded = _source(path, len(raw)) if role == "decode-input" else _read_small(path, 64 << 20)[1]
    preserved = deepcopy(recorded)
    time.sleep(0.02)
    os.chmod(path, path.stat().st_mode & 0o7777)
    current = _source(path, len(raw)) if role == "decode-input" else _read_small(path, 64 << 20)[1]
    events = []
    assert payload_identity_matches(recorded, current, role=role, observations=events)
    assert recorded == preserved and path.read_bytes() == raw
    event, = events
    assert event["status"] == "SHA256_VERIFIED" and event["role"] == role
    assert event["recorded_stat"][:4] == event["current_stat"][:4]
    assert event["recorded_stat"][4] != event["current_stat"][4]


@pytest.mark.parametrize("fault", ["mode", "uid", "gid", "nlink", "inode", "mtime_ns", "sha256", "size"])
def test_decode_input_full_identity_rejects_every_non_ctime_difference(tmp_path, fault):
    path = tmp_path / "inputs_embeds.bin"
    path.write_bytes(b"12")
    recorded = _source(path, 2)
    current = deepcopy(recorded)
    current["identity"]["ctime_ns"] += 1
    if fault == "sha256": current["sha256"] = "0" * 64
    elif fault == "size":
        current["bytes"] += 1
        current["identity"]["bytes"] += 1
    else: current["identity"][fault] += 1
    events = []
    assert not payload_identity_matches(recorded, current, role="decode-input", observations=events)
    assert events == []


def test_real_changed_input_bytes_with_restored_mtime_cannot_pass_ctime_recovery(tmp_path):
    path = tmp_path / "inputs_embeds.bin"
    path.write_bytes(b"12")
    before = path.stat()
    recorded = _source(path, 2)
    path.write_bytes(b"13")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    current = _source(path, 2)
    assert not payload_identity_matches(recorded, current, role="decode-input")


@pytest.mark.parametrize("role,filename", [("decode-output", "output_82"), ("decode-output", "output_01"),
    ("decode-input", "input_ids.bin"), ("decode-trial-input", "golden_logits.bin"),
    ("decode-log", "runtime.json"), ("source-air", "model.omc"),
    ("source-air", "SubGraph_0.weight"), ("decode-log", "extract_rows.py")])
def test_unassigned_payload_paths_refuse_even_equal_identity(tmp_path, role, filename):
    path = tmp_path / filename
    path.write_bytes(b"12")
    identity = _source(path, 2) if role == "decode-input" else _read_small(path, 64 << 20)[1]
    assert not payload_identity_matches(identity, identity, role=role)


def test_small_payload_budget_and_bounded_source_air_role(tmp_path):
    source = {"path": str(tmp_path / "export.air"), "bytes": 9 << 20, "sha256": "a" * 64,
        "stat": [1, 2, 9 << 20, 3, 4]}
    current = deepcopy(source); current["stat"][4] += 1
    assert payload_identity_matches(source, current, role="source-air")
    output = deepcopy(source); output["path"] = str(tmp_path / "outputs/output_0")
    assert not payload_identity_matches(output, output, role="decode-output")
    source["stat"][2] = source["bytes"] = (64 << 20) + 1
    assert not payload_identity_matches(source, source, role="source-air")


@pytest.mark.parametrize("name", ["retrieved_081", "unrelated_033", "output_B_10"])
def test_logged_output_filename_is_preserved_without_inferring_its_slot(tmp_path, name):
    row = {"path": str(tmp_path / "outputs" / name), "bytes": 2, "sha256": "a" * 64, "stat": [1, 2, 2, 3, 4]}
    current = deepcopy(row); current["stat"][4] += 1
    assert payload_identity_matches(row, current, role="decode-output")


def test_ordered_sequence_checks_all_slots_and_emits_no_partial_acceptance(tmp_path):
    rows = [{"path": str(tmp_path / f"outputs/output_{index}"), "bytes": 2,
        "sha256": "a" * 64, "stat": [1, index + 1, 2, 3, 4]} for index in range(2)]
    current = deepcopy(rows)
    for row in current: row["stat"][4] += 1
    events = []
    assert payload_sequence_matches(rows, current, role="decode-output", observations=events)
    assert len(events) == 2
    events.clear()
    assert not payload_sequence_matches(rows, list(reversed(current)), role="decode-output", observations=events)
    assert events == []


def test_empty_verified_tool_stderr_is_allowed_but_raw_payload_is_not(tmp_path):
    empty = {"path": str(tmp_path / "tool.stderr"), "bytes": 0, "sha256": "a" * 64, "stat": [1, 2, 0, 3, 4]}
    assert payload_identity_matches(empty, empty, role="decode-log")
    empty["path"] = str(tmp_path / "outputs/output_1")
    assert not payload_identity_matches(empty, empty, role="decode-output")

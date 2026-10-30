"""Historical metadata reconciliation cannot admit raw payload or content drift."""
from copy import deepcopy
import os
import time

import pytest

from qwen35_moe_dflash.metadata_revalidation import (
    MAX_METADATA_BYTES, metadata_identity_matches, metadata_sequence_matches)
from qwen35_moe_dflash.target_compile_failure import _read_small


ROLES = {
    "runtime-probe-source-metadata": "report.json",
    "decode-plan-metadata": "model-io.exitcode",
    "decode-execution-metadata": "runtime-decode-proof.json",
    "decode-trial-manifest": "manifest.json",
    "decode-output-inventory-metadata": "report.json",
    "embedding-audit-metadata": "report.json",
    "embedding-config-metadata": "original.yaml",
    "row-transfer-request-metadata": "request.json",
    "row-transfer-response-metadata": "response.json",
    "row-transfer-prepared-manifest-metadata": "manifest.json",
    "decode-state-lineage-metadata": "input-inventory.json",
    "decode-continuation-source-metadata": "report.json",
}


@pytest.mark.parametrize("role,filename", ROLES.items())
def test_each_explicit_metadata_role_reconciles_only_verified_ctime(tmp_path, role, filename):
    path = tmp_path / filename
    path.write_bytes(b"0\n" if filename.endswith(".exitcode") else b"{}\n")
    original_bytes, recorded = _read_small(path, MAX_METADATA_BYTES)
    preserved = deepcopy(recorded)
    time.sleep(0.02)
    os.chmod(path, path.stat().st_mode & 0o7777)
    current_bytes, current = _read_small(path, MAX_METADATA_BYTES)
    assert recorded["stat"][:4] == current["stat"][:4]
    assert recorded["stat"][4] != current["stat"][4]
    events = []
    assert metadata_identity_matches(recorded, current, role=role, observations=events)
    assert len(events) == 1 and events[0]["status"] == "SHA256_VERIFIED"
    assert events[0]["recorded_stat"] == recorded["stat"]
    assert events[0]["current_stat"] == current["stat"]
    assert original_bytes == current_bytes == path.read_bytes() and recorded == preserved
    legacy = {key: value for key, value in recorded.items() if key != "bytes"}
    assert metadata_identity_matches(legacy, current, role=role)


@pytest.mark.parametrize("fault", ["path", "sha256", "device", "inode", "size", "mtime", "oversized", "extra", "role"])
def test_non_ctime_changes_cannot_be_normalized(tmp_path, fault):
    path = tmp_path / "report.json"
    path.write_text("{}")
    _, recorded = _read_small(path, MAX_METADATA_BYTES)
    current = deepcopy(recorded)
    current["stat"][4] += 1
    role = "decode-output-inventory-metadata"
    if fault == "path": current["path"] = str(tmp_path / "other" / "report.json")
    elif fault == "sha256": current["sha256"] = "0" * 64
    elif fault in {"device", "inode", "size", "mtime"}:
        index = {"device": 0, "inode": 1, "size": 2, "mtime": 3}[fault]
        current["stat"][index] += 1
        if fault == "size": current["bytes"] += 1
    elif fault == "oversized":
        current["stat"][2] = current["bytes"] = MAX_METADATA_BYTES + 1
        recorded["stat"][2] = recorded["bytes"] = MAX_METADATA_BYTES + 1
    elif fault == "extra": current["mode"] = 0o100777
    else: role = "raw-output"
    events = []
    assert not metadata_identity_matches(recorded, current, role=role, observations=events)
    assert events == []


@pytest.mark.parametrize("filename", ["input_000.bin", "output_1", "tool.stdout", "tool.exitcode",
    "model.omc", "SubGraph_0.weight", "export.air", "extract_rows.py", "arbitrary.json"])
def test_payloads_code_and_unassigned_metadata_paths_are_outside_policy(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(b"{}")
    _, identity = _read_small(path, MAX_METADATA_BYTES)
    for role in ROLES:
        assert not metadata_identity_matches(identity, identity, role=role)


def test_sequence_preserves_order_and_collects_only_complete_matches(tmp_path):
    ids = []
    for filename in ("result.json", "plan.json"):
        path = tmp_path / filename
        path.write_text("{}")
        ids.append(_read_small(path, MAX_METADATA_BYTES)[1])
    current = deepcopy(ids)
    for row in current: row["stat"][4] += 1
    original = deepcopy(ids)
    events = []
    assert metadata_sequence_matches(ids, current, role="decode-execution-metadata", observations=events)
    assert len(events) == 2 and ids == original
    events.clear()
    assert not metadata_sequence_matches(ids, list(reversed(current)), role="decode-execution-metadata", observations=events)
    assert events == []
    assert not metadata_sequence_matches(ids, current[:1], role="decode-execution-metadata")


def test_collector_is_validated_before_comparison():
    with pytest.raises(ValueError, match="observations must be a list"):
        metadata_identity_matches({}, {}, role="decode-plan-metadata", observations={})

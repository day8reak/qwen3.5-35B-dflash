"""Completed reuse receipts select exact nested sidecars for the compiler.

The real hardlink stager and snapshot code run on tiny files; OMG is mocked.
No real target compilation, weights or device execution are involved.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import existing_air as compiler
from qwen35_moe_dflash import target_export_reuse as reuse
from test_existing_air import _help
from test_target_export_reuse import case as reuse_fixture


@pytest.fixture
def case(tmp_path, monkeypatch):
    item = reuse_fixture.__wrapped__(tmp_path, monkeypatch)
    result = reuse.stage_target_export_reuse(item["inventory"], item["air"], item["old"], item["output"], apply=True)
    assert result["status"] == "STAGED_HARDLINKS"
    item["staged_air"] = item["output"] / "export.air"
    item["receipt"] = item["output"] / "reuse-receipt.json"
    item["compilation"] = item["run"] / "compiled"
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_text("#!/bin/sh\nexit 0\n")
    omg.chmod(0o755)
    item["ddk"] = ddk
    return item


def _compile(case, **kwargs):
    return compiler.compile_existing_air(case["staged_air"], case["compilation"], case["ddk"],
        ddk_id="MOCK_REUSE_ONLY", **kwargs)


def _receipt(case):
    return json.loads(case["receipt"].read_text())


def _replace_receipt(case, value):
    case["receipt"].write_text(json.dumps(value) + "\n")


def _snapshot(folder):
    return {str(path.relative_to(folder)): (path.read_bytes(), compiler._stat(path), path.stat().st_mode)
            for path in folder.rglob("*") if path.is_file() and not path.is_symlink()}


def test_real_staged_prefix_is_selected_without_extra_payload_hash_or_source_mutation(case, monkeypatch):
    # This unbound sibling directory must never be traversed or staged.
    outside = case["output"] / "outside"
    outside.mkdir()
    (outside / "ignored_weight").write_bytes(b"unselected")
    (case["output"] / "decoy.bin").write_bytes(b"not in the receipt")
    before = _snapshot(case["output"])
    original = reuse._hash_file
    small_reads = []
    def only_small_metadata(folder, name, **kwargs):
        assert name in {"reuse-receipt.json", "inventory.json", "export.air"}, "selection must not rehash payloads"
        small_reads.append(name)
        return original(folder, name, **kwargs)
    monkeypatch.setattr(reuse, "_hash_file", only_small_metadata)
    invoked = []
    def omg(command, **kwargs):
        if "--help" in command:
            return _help(command, **kwargs)
        invoked.append(command)
        cwd = kwargs["cwd"]
        assert (cwd / "export.air").read_bytes() == case["staged_air"].read_bytes()
        for row in case["manifest"]["files"]:
            staged = cwd / row["relative_path"]
            assert staged.read_bytes() == (case["old"] / row["basename"]).read_bytes()
            assert hashlib.sha256(staged.read_bytes()).hexdigest() == row["sha256"]
        assert not (cwd / "decoy.bin").exists() and not (cwd / "outside").exists()
        assert not (cwd / "inventory.json").exists()
        (cwd / "model.omc").write_bytes(b"MOCK OMC; no model execution")
    monkeypatch.setattr(subprocess, "run", omg)
    result = _compile(case, input_layout="air-directory")
    assert result["status"] == "COMPILED", result.get("error")
    assert len(invoked) == 1 and len(small_reads) == 3
    assert result["device_execution"] == "NOT_RUN" and result["device_ready"] is False
    binding = result["external_weights"]["reuse_binding"]
    assert binding["status"] == "RECEIPT_BOUND_SELECTION"
    assert binding["source_relative_prefix"] == case["prefix"]
    assert binding["payload_verification"] == "ALL_SELECTED_SNAPSHOT_SHA256_MATCHED_RECEIPT"
    assert binding["receipt"]["sha256"] == hashlib.sha256(case["receipt"].read_bytes()).hexdigest()
    expected = {"export.air", *[row["relative_path"] for row in case["manifest"]["files"]]}
    assert {row["source_relative"] for row in result["input_snapshots"]} == expected
    assert set(binding["expected_files"]) == expected
    assert result["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    assert before == _snapshot(case["output"])
    for row in result["input_snapshots"]:
        assert compiler._stat(Path(row["source"]))[:2] != compiler._stat(case["compilation"] / row["path"])[:2]


def test_input_inventory_two_value_api_remains_usable_by_target_resume(case):
    selected, skipped = compiler._input_inventory(case["staged_air"])
    assert selected[0] == case["staged_air"] and len(selected) == 3
    assert not skipped


@pytest.mark.parametrize("fault", ["kind", "status", "mode", "air_hash", "inventory_hash", "prefix", "row_hash",
    "row_stat", "duplicate_path", "missing_file", "extra_file", "symlink_file", "symlink_directory"])
def test_bad_or_unbound_reuse_inputs_fail_before_snapshots_and_compiler(case, monkeypatch, fault):
    receipt = _receipt(case)
    row = receipt["files"][0]
    payload = case["output"] / row["relative_path"]
    if fault == "kind": receipt["artifact_kind"] = "unowned-receipt"
    elif fault == "status": receipt["status"] = "APPLY_FAILED_PARTIAL_PRESERVED"
    elif fault == "mode": receipt["mode"] = "PLAN_ONLY"
    elif fault == "air_hash": receipt["air"]["sha256"] = "0" * 64
    elif fault == "inventory_hash": receipt["inventory"]["sha256"] = "0" * 64
    elif fault == "prefix": receipt["source_relative_prefix"] = "unrelated-directory"
    elif fault == "row_hash": row["sha256"] = "0" * 64
    elif fault == "row_stat": row["staged_stat"]["inode"] += 1
    elif fault == "duplicate_path": receipt["files"][1]["relative_path"] = row["relative_path"]
    elif fault == "missing_file": payload.unlink()
    elif fault == "extra_file": (payload.parent / "unrecorded").write_bytes(b"extra")
    elif fault == "symlink_file":
        payload.unlink(); payload.symlink_to(case["old"] / row["basename"])
    elif fault == "symlink_directory":
        folder = payload.parent
        moved = folder.with_name("moved-sidecars")
        folder.rename(moved); folder.symlink_to(moved, target_is_directory=True)
    _replace_receipt(case, receipt)
    def help_only(command, **kwargs):
        assert "--help" in command, "invalid reuse input must prevent OMG compilation"
        return _help(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", help_only)
    monkeypatch.setattr(compiler, "_snapshot", lambda *a, **k: pytest.fail("invalid receipt must prevent snapshots"))
    result = _compile(case)
    assert result["status"] == "FAILED" and result["model_compilation"] == "NOT_RUN"
    assert result["error"]


def test_export_directory_does_not_duplicate_the_receipt_bound_prefix(case, monkeypatch):
    def help_only(command, **kwargs):
        assert "--help" in command
        return _help(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", help_only)
    result = _compile(case, input_layout="export-directory")
    assert result["status"] == "FAILED" and "requires input_layout=air-directory" in result["error"]
    assert not result["input_snapshots"]


def test_snapshot_sha_is_compared_to_receipt_without_an_extra_source_hash(case, monkeypatch):
    original = compiler._snapshot
    def wrong_snapshot(source, destination):
        row = original(source, destination)
        if source != case["staged_air"]:
            row["sha256"] = "0" * 64
        return row
    monkeypatch.setattr(compiler, "_snapshot", wrong_snapshot)
    def help_only(command, **kwargs):
        assert "--help" in command
        return _help(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", help_only)
    result = _compile(case)
    assert result["status"] == "FAILED" and result["model_compilation"] == "NOT_RUN"
    assert "snapshot differs from receipt-bound SHA256" in result["error"]


@pytest.mark.parametrize("name", ["reuse-receipt.json", "inventory.json"])
def test_metadata_changes_during_snapshot_are_rejected_before_omg(case, monkeypatch, name):
    original = compiler._snapshot
    mutated = []
    def changed(source, destination):
        row = original(source, destination)
        if not mutated:
            path = case["output"] / name
            path.write_bytes(path.read_bytes() + b"\n")
            mutated.append(True)
        return row
    monkeypatch.setattr(compiler, "_snapshot", changed)
    def help_only(command, **kwargs):
        assert "--help" in command
        return _help(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", help_only)
    result = _compile(case)
    assert mutated and result["status"] == "FAILED"
    assert result["model_compilation"] == "NOT_RUN"
    assert "identity changed during input snapshot" in result["error"]
    assert name in result["source_preservation"]["changed"]

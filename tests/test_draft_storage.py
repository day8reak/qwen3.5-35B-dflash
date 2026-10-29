"""Storage compaction preserves every payload byte, receipt and access mode."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from qwen35_moe_dflash.draft_storage import compact_draft_storage
from qwen35_moe_dflash.draft_workspace import DraftWorkspace


def _row(path, root, **fields):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), **fields}


def _json(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    return _row(path, path.parent)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    job = tmp_path / "full-draft"
    prepared = job / "prepared"
    prepared.mkdir(parents=True)
    source = prepared / "weights/w00000.bin"
    source.parent.mkdir()
    source.write_bytes(bytes(range(256)) * 256)
    source.chmod(0o644)
    prepared_receipt = _json(prepared / "prepared.json", {"artifact_kind": "qwen35-draft-ge-prepared", "status": "PREPARED",
        "weights": {"draft.fc.weight": _row(source, prepared, dtype="float16", shape=[32768])}})
    folders, snapshots = [], []
    for lane in ("production", "layer-trace", "direct-trace", "residual-fix"):
        folder = job / "compilation" if lane == "production" else job / lane / "compilation"
        target = folder / "compile-work/generated/weights/w00000.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        target.chmod(0o444)
        omc = folder / "model.omc"
        omc.write_bytes(b"small unmodified OMC")
        report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "model_compilation": "PASSED",
            "input_snapshots": [_row(target, folder, role="neighbor_data_candidate", source_relative="weights/w00000.bin")],
            "omc": _row(omc, folder)}
        receipt = _json(folder / "report.json", report)
        folders.append((lane, folder, receipt))
        snapshots.append(target)
    _json(job / "result.json", {"artifact_kind": "qwen35-full-draft-job", "components": {
        "prepared": {"directory": str(prepared), "manifest": prepared_receipt},
        "compilation": {"directory": str(folders[0][1]), "manifest": folders[0][2]}}})
    source_sha = hashlib.sha256((job / "result.json").read_bytes()).hexdigest()
    for lane, folder, receipt in folders[1:]:
        root = job / lane
        _json(root / "completed-compilation.json", receipt)
        _json(root / "owner.json", {"artifact_kind": "qwen35-draft-direct-trace-job-owner", "job": str(job),
                                    "source_job_sha256": source_sha})
    # Establish the ordinary workspace lock/owner before measuring immutability.
    with DraftWorkspace(job):
        pass
    return job, source, snapshots


def _bytes(job):
    return {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}


def test_default_is_readonly_plan_and_does_not_read_projection_bytes(case, monkeypatch):
    job, source, snapshots = case
    before = _bytes(job)
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert self.suffix != ".bin", "readonly plan uses bound metadata, not large weight reads"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    plan = compact_draft_storage(job)
    assert plan["status"] == "PLANNED" and len(plan["changes"]) == 3
    assert plan["logical_duplicate_bytes"] == source.stat().st_size * 3
    monkeypatch.setattr(Path, "open", original)
    assert _bytes(job) == before
    assert len({p.stat().st_ino for p in snapshots}) == 4


def test_apply_links_only_snapshots_preserving_all_bytes_permissions_and_receipts(case):
    job, source, snapshots = case
    before, source_stat = _bytes(job), source.stat()
    modes = {p: (p.stat().st_mode, p.stat().st_uid, p.stat().st_gid) for p in snapshots}
    result = compact_draft_storage(job, apply=True)
    assert result["status"] == "COMPACTED" and all(r["status"] == "LINKED" for r in result["changes"])
    assert len({p.stat().st_ino for p in snapshots}) == 1
    assert source.stat() == source_stat
    assert source.stat().st_ino != snapshots[0].stat().st_ino
    assert _bytes(job) == before
    assert {p: (p.stat().st_mode, p.stat().st_uid, p.stat().st_gid) for p in snapshots} == modes
    repeated = compact_draft_storage(job, apply=True)
    assert repeated["changes"] == [] and repeated["logical_duplicate_bytes"] == 0
    assert not list(job.rglob(".draft-weight-dedup-*"))


@pytest.mark.parametrize("fault", ["snapshot", "prepared", "receipt", "owner", "symlink", "escape"])
def test_changed_or_unowned_data_is_rejected_before_any_link(case, fault):
    job, source, snapshots = case
    if fault in {"snapshot", "prepared"}:
        path = snapshots[-1] if fault == "snapshot" else source
        path.chmod(0o644)
        path.write_bytes(b"x" * path.stat().st_size)
        if fault == "snapshot":
            path.chmod(0o444)
    elif fault == "receipt":
        path = job / "direct-trace/compilation/report.json"
        path.write_text(path.read_text() + "\n")
    elif fault == "owner":
        path = job / "direct-trace/owner.json"
        doc = json.loads(path.read_text())
        doc["source_job_sha256"] = "0" * 64
        path.write_text(json.dumps(doc))
    elif fault == "symlink":
        snapshots[-1].unlink()
        snapshots[-1].symlink_to(source)
    else:
        path = job / "direct-trace/compilation/report.json"
        doc = json.loads(path.read_text())
        doc["input_snapshots"][0]["path"] = "../../prepared/weights/w00000.bin"
        receipt = _json(path, doc)
        _json(job / "direct-trace/completed-compilation.json", receipt)
    inode = snapshots[0].stat().st_ino
    with pytest.raises(ValueError):
        compact_draft_storage(job, apply=True)
    assert snapshots[0].stat().st_ino == inode
    assert not list(job.rglob(".draft-weight-dedup-*"))


def test_active_workspace_blocks_apply(case):
    job, _, snapshots = case
    before = [p.stat().st_ino for p in snapshots]
    with DraftWorkspace(job):
        with pytest.raises(RuntimeError, match="locked"):
            compact_draft_storage(job, apply=True)
    assert [p.stat().st_ino for p in snapshots] == before


def test_writable_snapshot_is_skipped_without_chmod(case):
    job, _, snapshots = case
    snapshots[-1].chmod(0o644)
    inode = snapshots[-1].stat().st_ino
    result = compact_draft_storage(job, apply=True)
    assert len(result["changes"]) == 2
    assert snapshots[-1].stat().st_ino == inode and snapshots[-1].stat().st_mode & 0o222
    assert any(r["reason"] == "WRITABLE_SNAPSHOT" for r in result["skipped"])


def test_incomplete_compilation_is_never_compacted(case):
    job, _, snapshots = case
    (job / "residual-fix/completed-compilation.json").unlink()
    inode = snapshots[-1].stat().st_ino
    result = compact_draft_storage(job, apply=True)
    assert len(result["changes"]) == 2 and snapshots[-1].stat().st_ino == inode
    assert any(r.get("lane") == "residual-fix" and r["reason"] == "NO_COMPLETED_COMPILATION" for r in result["skipped"])


def test_different_filesystem_is_skipped(case, monkeypatch):
    from qwen35_moe_dflash import draft_storage
    job, _, snapshots = case
    original = draft_storage._token

    def other_device(path):
        token = original(path)
        return (token[0] + 1, *token[1:]) if path == snapshots[-1] else token

    monkeypatch.setattr(draft_storage, "_token", other_device)
    plan = compact_draft_storage(job)
    assert len(plan["changes"]) == 2
    assert any(r["reason"] == "DIFFERENT_FILESYSTEM" for r in plan["skipped"])


def test_different_access_metadata_does_not_share_inode(case):
    job, _, snapshots = case
    snapshots[-1].chmod(0o440)
    inode = snapshots[-1].stat().st_ino
    result = compact_draft_storage(job, apply=True)
    assert len(result["changes"]) == 2
    assert snapshots[-1].stat().st_ino == inode and snapshots[-1].stat().st_mode & 0o777 == 0o440


def test_link_failure_preserves_destinations_and_removes_owned_temp(case, monkeypatch):
    job, _, snapshots = case
    before, inodes = _bytes(job), [p.stat().st_ino for p in snapshots]

    def fail(*args, **kwargs):
        raise OSError("simulated hardlink failure")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="simulated"):
        compact_draft_storage(job, apply=True)
    assert _bytes(job) == before
    assert [p.stat().st_ino for p in snapshots] == inodes
    assert not list(job.rglob(".draft-weight-dedup-*"))


def test_existing_successful_cache_content_validator_accepts_compacted_snapshots(case):
    from qwen35_moe_dflash.draft_board import _file
    job, _, snapshots = case
    compact_draft_storage(job, apply=True)
    for path in snapshots:
        folder = path.parents[3]
        report = json.loads((folder / "report.json").read_text())
        assert _file(folder, report["input_snapshots"][0]) == path


@pytest.mark.parametrize("full", [False, True])
def test_cli_defaults_to_small_summary_with_full_mapping_opt_in(case, capsys, full):
    import importlib.util
    job, _, _ = case
    path = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/compact_draft_storage.py"
    spec = importlib.util.spec_from_file_location("compact_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--job", str(job), *(["--full"] if full else [])]) == 0
    output = json.loads(capsys.readouterr().out)
    if full:
        assert len(output["changes"]) == 3 and "skipped" in output
    else:
        assert output["planned_changes"] == 3 and output["linked_changes"] == 0
        assert "changes" not in output and "skipped_counts" in output

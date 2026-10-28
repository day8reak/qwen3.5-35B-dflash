"""Host filesystem checks for fixed draft workspaces; no model/device tools."""
import copy
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash.artifacts import save_json
from qwen35_moe_dflash.hashing import sha256_file


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    root = tmp_path / "legacy"
    root.mkdir()
    state = {"schema_version": 1, "artifact_kind": "qwen35-full-draft-job",
             "status": "COMPILED", "stage": "complete", "output": str(root),
             "settings": {"sequence_length": 64, "valid_length": 6, "seed": 17},
             "compile_options": {"save_weights_as_external_data": "false"}, "components": {}}
    names = {
        "prepared": ("prepared.json", "qwen35-draft-ge-prepared", "PREPARED"),
        "fixture": ("report.json", "qwen35-draft-reference", "REFERENCE_COMPLETE"),
        "air": ("manifest.json", "qwen35-cpro-draft-air", "EXPORTED"),
        "compilation": ("report.json", "qwen35-existing-air-compilation", "COMPILED"),
        "bundle": ("manifest.json", "qwen35-standalone-draft-bundle", "PREPARED"),
    }
    for stage, (filename, kind, status) in names.items():
        folder = root / stage
        folder.mkdir()
        payload = folder / "payload.bin"
        payload.write_bytes(("payload for " + stage).encode())
        manifest = folder / filename
        save_json(manifest, {"artifact_kind": kind, "status": status, "payload": {
            "path": payload.name, "bytes": payload.stat().st_size, "sha256": sha256_file(payload)}})
        record = {"directory": str(folder), "manifest": {
            "path": filename, "bytes": manifest.stat().st_size, "sha256": sha256_file(manifest)}}
        state["components"][stage] = record
        save_json(root / f"completed-{stage}.json", record)
    save_json(root / "job.json", state)
    return root, state


def _files(root):
    return {path.relative_to(root): (path.read_bytes(), path.stat().st_ino)
            for path in root.rglob("*") if path.is_file()}


def test_adoption_moves_payloads_without_rewriting_hash_bound_manifests(legacy, tmp_path):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    source, state = legacy
    original = copy.deepcopy(state)
    before = {name: _files(source / name) for name in state["components"]}
    root = tmp_path / "full-draft"
    with DraftWorkspace(root) as workspace:
        assert workspace.load_pending() is None
        adopted = workspace.adopt(state)
        for name, record in adopted["components"].items():
            assert record["directory"] == str(root / name)
            assert record["manifest"] == original["components"][name]["manifest"]
            assert _files(root / name) == before[name]
            assert not (source / name).exists()
            assert json.loads((root / f"completed-{name}.json").read_text()) == record
        assert workspace.load_pending() is None


@pytest.mark.parametrize("interruption", ["before-air-move", "after-prepared-move"])
def test_interrupted_adoption_resumes_renames_and_preserves_payload_inodes(
        legacy, tmp_path, monkeypatch, interruption):
    from qwen35_moe_dflash import draft_workspace
    source, state = legacy
    before = {name: _files(source / name) for name in state["components"]}
    root = tmp_path / "full-draft"
    original_rename = draft_workspace.os.rename
    fired = False

    def interrupted_rename(src, dst, *args, **kwargs):
        nonlocal fired
        src, dst = Path(src), Path(dst)
        if not fired and src == source / ("air" if interruption == "before-air-move" else "prepared"):
            fired = True
            if interruption == "after-prepared-move":
                original_rename(src, dst, *args, **kwargs)
            raise OSError("injected migration interruption")
        return original_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(draft_workspace.os, "rename", interrupted_rename)
    with draft_workspace.DraftWorkspace(root) as workspace:
        with pytest.raises(OSError, match="injected migration interruption"):
            workspace.adopt(state)
    assert fired and (root / "prepared").is_dir() and not (source / "prepared").exists()
    monkeypatch.setattr(draft_workspace.os, "rename", original_rename)
    with draft_workspace.DraftWorkspace(root) as workspace:
        pending = workspace.load_pending()
        assert pending["components"]["prepared"]["directory"] == str(root / "prepared")
        assert pending["components"]["air"]["directory"] == str(source / "air")
        adopted = workspace.adopt(pending)
        assert workspace.load_pending() is None
        for name in adopted["components"]:
            assert _files(root / name) == before[name]
            assert not (source / name).exists()


def test_reset_only_removes_invalid_stages_receipts_and_previous_board_result(legacy, tmp_path):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    _, state = legacy
    root = tmp_path / "full-draft"
    with DraftWorkspace(root) as workspace:
        adopted = workspace.adopt(state)
        workspace.reset(adopted)
        preserved = {name: _files(root / name) for name in ("prepared", "fixture", "air")}
        (root / "board").mkdir()
        (root / "board/failure.log").write_text("stale board evidence")
        workspace.save("result.json", {**adopted, "status": "FAILED"})
        note = root / "user-note.txt"
        note.write_text("unrelated root file")
        selected = copy.deepcopy(adopted)
        for name in ("compilation", "bundle"):
            selected["components"].pop(name)
        workspace.reset(selected)
        assert not (root / "board").exists() and not (root / "result.json").exists()
        for name in ("compilation", "bundle"):
            assert not (root / name).exists() and not (root / f"completed-{name}.json").exists()
        for name, payloads in preserved.items():
            assert _files(root / name) == payloads
        assert note.read_text() == "unrelated root file"
        assert json.loads((root / "job.json").read_text())["components"] == selected["components"]


def test_workspace_lock_conflict_cannot_replace_lock_or_touch_active_files(legacy, tmp_path):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    _, state = legacy
    root = tmp_path / "full-draft"
    with DraftWorkspace(root) as first:
        adopted = first.adopt(state)
        first.reset(adopted)
        before = _files(root)
        with pytest.raises((OSError, ValueError, RuntimeError), match="(?i)(lock|active|busy|running)"):
            with DraftWorkspace(root):
                pytest.fail("a concurrent workspace acquired the active lock")
        assert _files(root) == before


def test_adoption_checks_manifest_before_moving_any_stage(legacy, tmp_path):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    source, state = legacy
    (source / "air/manifest.json").write_text("tampered manifest")
    before = _files(source)
    root = tmp_path / "full-draft"
    with DraftWorkspace(root) as workspace:
        with pytest.raises(ValueError, match="(?i)(hash|sha256|size|manifest)"):
            workspace.adopt(state)
    assert _files(source) == before
    assert not any((root / name).exists() for name in state["components"])


def test_reset_refuses_stage_symlink_without_deleting_target(legacy, tmp_path):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    _, state = legacy
    state = copy.deepcopy(state)
    state["components"].pop("bundle")
    root = tmp_path / "full-draft"
    protected = tmp_path / "unrelated"
    protected.mkdir()
    (protected / "keep.bin").write_bytes(b"unrelated payload")
    with DraftWorkspace(root) as workspace:
        adopted = workspace.adopt(state)
        (root / "bundle").symlink_to(protected, target_is_directory=True)
        with pytest.raises(ValueError, match="(?i)(symlink|non-directory)"):
            workspace.reset(adopted)
        assert (protected / "keep.bin").read_bytes() == b"unrelated payload"


@pytest.mark.parametrize("name", ["../outside.json", "prepared/prepared.json", "other.json"])
def test_atomic_save_is_limited_to_job_result_and_stage_receipts(tmp_path, monkeypatch, name):
    from qwen35_moe_dflash.draft_workspace import DraftWorkspace
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    root = tmp_path / "full-draft"
    with DraftWorkspace(root) as workspace:
        with pytest.raises(ValueError):
            workspace.save(name, {"status": "unexpected"})
    assert not (tmp_path / "outside.json").exists()

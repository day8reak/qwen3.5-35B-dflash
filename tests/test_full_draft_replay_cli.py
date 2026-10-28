"""Host replay CLI orchestration; fabricated outputs are never device evidence."""
import json
from pathlib import Path
import sys
import types

import pytest

from qwen35_moe_dflash.draft_workspace import DraftWorkspace
from qwen35_moe_dflash.hashing import sha256_file
from test_draft_review import _cli, _snapshot, bundle, retained


@pytest.fixture
def replay_cli(retained, monkeypatch):
    calls = []
    result = {"status": "INSPECTED", "numerical_validation": "TOP1_MATCHED",
              "new_device_execution": "NOT_RUN", "new_compilation": "NOT_RUN"}
    failure = []

    def replay(job, output, *, progress):
        assert Path(job) == retained["job"]
        assert Path(output) == retained["job"] / "reference-replay"
        # A second independent descriptor cannot acquire this flock while the
        # CLI is reading the retained inputs and publishing its replay report.
        with pytest.raises(RuntimeError, match="locked by another process"):
            with DraftWorkspace(job):
                pytest.fail("replay must run under the workspace lock")
        calls.append((job, output))
        if failure:
            raise ValueError(failure[0])
        output.mkdir(exist_ok=True)
        (output / "report.json").write_text(json.dumps(result, allow_nan=False))
        progress("fixture CPU replay complete")
        return dict(result)

    module = types.ModuleType("qwen35_moe_dflash.draft_reference_replay")
    module.replay_reference = replay
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def no_reset(*args, **kwargs):
        pytest.fail("CPU replay must not reset or migrate successful stages")

    monkeypatch.setattr(DraftWorkspace, "reset", no_reset)
    monkeypatch.setattr(DraftWorkspace, "adopt", no_reset)
    for name in ("CPRO_DDK_ROOT", "CPRO_DDK_ID", "CANN_ROOT", "DDK_PATH"):
        monkeypatch.delenv(name, raising=False)
    return _cli(), calls, result, failure


def _assert_originals_unchanged(root, before):
    assert all((root / name).is_file() and sha256_file(root / name) == digest
               for name, digest in before.items())


@pytest.mark.parametrize("selection", ["default", "resume", "output", "both"])
def test_replay_uses_one_retained_job_and_fixed_output_without_reset(
        retained, replay_cli, selection, capsys):
    main, calls, _, _ = replay_cli
    before = _snapshot(retained["job"])
    args = ["--replay-reference"]
    if selection in ("resume", "both"):
        args += ["--resume", str(retained["job"])]
    if selection in ("output", "both"):
        args += ["--output", str(retained["job"])]
    for _ in range(2):
        assert main(args) == 0
        _assert_originals_unchanged(retained["job"], before)
    assert len(calls) == 2
    created = set(_snapshot(retained["job"])) - set(before)
    assert created == {"reference-replay/report.json", ".draft-workspace.json",
                       ".draft-workspace.lock"}
    report = json.loads((retained["job"] / "reference-replay/report.json").read_text())
    assert report["new_device_execution"] == report["new_compilation"] == "NOT_RUN"
    stdout = capsys.readouterr().out
    assert "draft-reference: fixture CPU replay complete" in stdout
    assert str(retained["job"] / "reference-replay/report.json") in stdout


@pytest.mark.parametrize("numerical,expected_exit", [
    ("TOP1_MATCHED", 0), ("FAILED", 1), ("NOT_EVALUATED", 1), (None, 1),
])
def test_replay_exit_tracks_numeric_result_without_reexecuting_board(
        retained, replay_cli, numerical, expected_exit):
    main, calls, result, _ = replay_cli
    if numerical is None:
        result.pop("numerical_validation")
    else:
        result["numerical_validation"] = numerical
    before = _snapshot(retained["job"])
    assert main(["--replay-reference"]) == expected_exit
    assert len(calls) == 1
    _assert_originals_unchanged(retained["job"], before)


def test_replay_obeys_existing_workspace_lock(retained, replay_cli):
    main, calls, _, _ = replay_cli
    with DraftWorkspace(retained["job"]):
        before = _snapshot(retained["job"])
        assert main(["--replay-reference"]) == 1
        assert _snapshot(retained["job"]) == before
    assert calls == []
    assert not (retained["job"] / "reference-replay").exists()


@pytest.mark.parametrize("extra", [
    ["--target-weights", "different-target"],
    ["--draft-weights", "different-draft"],
    ["--sequence-length", "64"], ["--valid-length", "6"],
    ["--weight-storage", "embedded-stream"],
    ["--model-load-api", "file-v2"],
    ["--schema-library", "different-schema.so"],
])
def test_replay_rejects_changed_inputs_or_storage_before_any_job_mutation(
        retained, replay_cli, extra):
    main, calls, _, _ = replay_cli
    before = _snapshot(retained["job"])
    assert main(["--replay-reference", *extra]) == 1
    assert calls == []
    assert _snapshot(retained["job"]) == before


@pytest.mark.parametrize("mode", ["--prepare-only", "--compile-only", "--compare-only", "--inspect-only"])
def test_replay_rejects_conflicting_execution_modes(retained, replay_cli, mode):
    main, calls, _, _ = replay_cli
    before = _snapshot(retained["job"])
    with pytest.raises(SystemExit) as error:
        main(["--replay-reference", mode])
    assert error.value.code == 2
    assert calls == []
    assert _snapshot(retained["job"]) == before


def test_replay_does_not_migrate_to_another_output(retained, replay_cli, tmp_path):
    main, calls, _, _ = replay_cli
    before = _snapshot(retained["job"])
    other = tmp_path / "other-job"
    assert main(["--replay-reference", "--resume", str(retained["job"]),
                 "--output", str(other)]) == 1
    assert calls == [] and not other.exists()
    assert _snapshot(retained["job"]) == before


def test_replay_failure_preserves_original_job_and_releases_lock(retained, replay_cli):
    main, calls, _, failure = replay_cli
    failure.append("reference payload digest differs")
    before = _snapshot(retained["job"])
    assert main(["--replay-reference"]) == 1
    assert len(calls) == 1
    _assert_originals_unchanged(retained["job"], before)
    assert not (retained["job"] / "reference-replay").exists()
    with DraftWorkspace(retained["job"]):
        pass


def test_replay_refuses_pending_workspace_migration(retained, replay_cli, monkeypatch):
    main, calls, _, _ = replay_cli
    monkeypatch.setattr(DraftWorkspace, "load_pending", lambda self: {"components": {}})
    before = _snapshot(retained["job"])
    assert main(["--replay-reference"]) == 1
    assert calls == []
    _assert_originals_unchanged(retained["job"], before)
    assert not (retained["job"] / "reference-replay").exists()

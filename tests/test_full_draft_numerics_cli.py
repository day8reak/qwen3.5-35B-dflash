"""The numerical CLI inspects fabricated retained evidence without execution."""
import json

import numpy as np
import pytest

from qwen35_moe_dflash import draft_numerics, resources
from qwen35_moe_dflash.draft_workspace import DraftWorkspace
from qwen35_moe_dflash.hashing import sha256_file
from test_draft_numerics import inspection, _bind_norm
from test_draft_review import _cli, _snapshot, _rebind_output, bundle, retained


@pytest.fixture
def numerical_cli(retained, inspection, monkeypatch):
    _bind_norm(retained)
    calls = []
    actual = draft_numerics.diagnose_draft_numerics
    def inspect(job, schema_library=None):
        with pytest.raises(RuntimeError, match="locked by another process"):
            with DraftWorkspace(job):
                pytest.fail("numerical inspection must hold the workspace lock")
        calls.append((job, schema_library))
        return actual(job, schema_library=schema_library)
    monkeypatch.setattr(draft_numerics, "diagnose_draft_numerics", inspect)
    def forbidden(*args, **kwargs):
        pytest.fail("retained numerical inspection must not reset/adopt or resolve a toolchain")
    monkeypatch.setattr(DraftWorkspace, "reset", forbidden)
    monkeypatch.setattr(DraftWorkspace, "adopt", forbidden)
    monkeypatch.setattr(resources, "load_resources", forbidden)
    for name in ("CPRO_DDK_ROOT", "CPRO_DDK_ID", "CANN_ROOT", "DDK_PATH", "SN", "BIP"):
        monkeypatch.delenv(name, raising=False)
    return _cli(), calls


def _unchanged(root, before):
    assert all((root / name).is_file() and sha256_file(root / name) == digest
               for name, digest in before.items())


@pytest.mark.parametrize("selection", ["default", "resume", "output", "both"])
def test_numerical_cli_reuses_fixed_report_offline_and_keeps_bound_fixture_metadata(
        retained, numerical_cli, selection, capsys):
    main, calls = numerical_cli
    before = _snapshot(retained["job"])
    args = ["--diagnose-numerics"]
    if selection in ("resume", "both"):
        args += ["--resume", str(retained["job"])]
    if selection in ("output", "both"):
        args += ["--output", str(retained["job"])]
    for _ in range(2):
        assert main(args) == 0
        _unchanged(retained["job"], before)
    assert calls == [(retained["job"], None)] * 2
    report = json.loads((retained["job"] / "numerical-diagnostic/report.json").read_text())
    assert report["reference_policy"]["reference"] == "PYTORCH_FP16_AIR4B_PRECISION"
    assert report["reference_policy"]["reference_contract"] == "pytorch-fp16-air4b-precision-v3"
    assert report["reference_audit"]["manifest_binding"] == "MATCHED"
    assert report["unit_final_scale_constraint"]["unit_scale_verified"] is True
    assert report["new_device_execution"] == report["new_compilation"] == "NOT_RUN"
    assert set(_snapshot(retained["job"])) - set(before) == {
        "numerical-diagnostic/report.json", ".draft-workspace.json", ".draft-workspace.lock"}
    output = capsys.readouterr().out
    assert "Numerical diagnostic:" in output
    assert str(retained["job"] / "numerical-diagnostic/report.json") in output


def test_numerical_cli_passes_explicit_schema_without_toolchain_resolution(retained, numerical_cli, inspection):
    main, calls = numerical_cli
    before = _snapshot(retained["job"])
    assert main(["--diagnose-numerics", "--schema-library", "/selected/schema.so"]) == 0
    assert calls == [(retained["job"], "/selected/schema.so")]
    assert inspection[0] == calls
    _unchanged(retained["job"], before)


def test_numerical_failure_remains_exit_one_and_does_not_execute_again(retained, numerical_cli):
    main, calls = numerical_cli
    np.zeros((1, 15), dtype="<i4").tofile(retained["board"] / "results/output_0000.bin")
    _rebind_output(retained, 0)
    before = _snapshot(retained["job"])
    assert main(["--diagnose-numerics"]) == 1
    assert len(calls) == 1
    report = json.loads((retained["job"] / "numerical-diagnostic/report.json").read_text())
    assert report["status"] == "INSPECTED" and report["device_execution"] == "PASSED"
    assert report["numerical_validation"] == "FAILED"
    assert report["comparison"]["token_mismatches"] == 15
    assert report["threshold_changes"] == "NONE"
    _unchanged(retained["job"], before)


@pytest.mark.parametrize("extra", [
    ["--target-weights", "different-target"], ["--draft-weights", "different-draft"],
    ["--sequence-length", "64"], ["--valid-length", "6"],
    ["--weight-storage", "embedded-stream"], ["--model-load-api", "file-v2"],
])
def test_numerical_cli_rejects_changed_inputs_and_loader_before_writing(retained, numerical_cli, extra):
    main, calls = numerical_cli
    before = _snapshot(retained["job"])
    assert main(["--diagnose-numerics", *extra]) == 1
    assert calls == []
    assert _snapshot(retained["job"]) == before


@pytest.mark.parametrize("mode", ["--prepare-only", "--compile-only", "--compare-only", "--replay-reference", "--inspect-only"])
def test_numerical_cli_rejects_conflicting_modes(retained, numerical_cli, mode):
    main, calls = numerical_cli
    before = _snapshot(retained["job"])
    with pytest.raises(SystemExit) as error:
        main(["--diagnose-numerics", mode])
    assert error.value.code == 2 and calls == []
    assert _snapshot(retained["job"]) == before


def test_numerical_cli_respects_existing_workspace_lock(retained, numerical_cli):
    main, calls = numerical_cli
    with DraftWorkspace(retained["job"]):
        before = _snapshot(retained["job"])
        assert main(["--diagnose-numerics"]) == 1
        assert _snapshot(retained["job"]) == before
    assert calls == []


def test_numerical_cli_never_migrates_to_another_output(retained, numerical_cli, tmp_path):
    main, calls = numerical_cli
    before = _snapshot(retained["job"])
    other = tmp_path / "different-job"
    assert main(["--diagnose-numerics", "--resume", str(retained["job"]), "--output", str(other)]) == 1
    assert calls == [] and not other.exists()
    assert _snapshot(retained["job"]) == before


def test_numerical_cli_integrity_error_preserves_originals_and_releases_lock(retained, numerical_cli):
    main, calls = numerical_cli
    (retained["board"] / "results/output_0001.bin").write_bytes(b"bad retained output")
    before = _snapshot(retained["job"])
    assert main(["--diagnose-numerics"]) == 1
    assert len(calls) == 1
    assert not (retained["job"] / "numerical-diagnostic").exists()
    _unchanged(retained["job"], before)
    with DraftWorkspace(retained["job"]):
        pass


def test_numerical_cli_refuses_pending_migration(retained, numerical_cli, monkeypatch):
    main, calls = numerical_cli
    monkeypatch.setattr(DraftWorkspace, "load_pending", lambda self: {"components": {}})
    before = _snapshot(retained["job"])
    assert main(["--diagnose-numerics"]) == 1
    assert calls == []
    _unchanged(retained["job"], before)

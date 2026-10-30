"""Real preparation and validation joined to a host-only native/HDC simulator."""
import hashlib
import json
from pathlib import Path

from qwen35_moe_dflash import real_feature_draft as real
from qwen35_moe_dflash import real_feature_draft_board as board
from test_real_feature_draft import fixture, retained_case
from test_real_feature_draft_board import install_hdc


def test_actual_preparation_validates_and_deploys_one_real_feature_step(fixture, tmp_path, monkeypatch):
    """Exercise the manifest boundary without replacing its production validator."""
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "fixture:8790")
    monkeypatch.setenv("BOARD_HIAI_LIB_DIR", "/data/local/tmp/fixture_hiai")
    for name in ("CPRO_DDK_ID", "HIAI_RUNTIME_VERSION", "FIRMWARE_ID"):
        monkeypatch.delenv(name, raising=False)
    original = {path: (path.stat(), path.read_bytes())
                for path in fixture["bundle"].rglob("*") if path.is_file()}
    manifest = real.prepare_real_feature_draft(fixture["inspection"], fixture["bundle"], fixture["out"])
    assert real.validate_real_feature_draft(fixture["out"]) == manifest
    runner = tmp_path / "host-only-runner"
    runner.write_bytes(b"host-only-native-runner-fixture")
    monkeypatch.setattr(board, "_build_runner", lambda out, timeout: runner)
    fake = install_hdc(monkeypatch, manifest, cache_decision="MISSING")
    simulated_shell = fake.shell

    def check_deployed_inputs(self, command, label, **kwargs):
        if label == "execute-real-feature-draft":
            deployed = self.attempt / "bundle"
            assert self.files[str(deployed / "step.tsv")] == (fixture["out"] / "step.tsv").read_bytes()
            for row in manifest["inputs"]:
                raw = self.files[str(deployed / row["path"])]
                assert len(raw) == row["bytes"]
                assert hashlib.sha256(raw).hexdigest() == row["sha256"]
            self.actual_input_payloads_checked = True
        return simulated_shell(self, command, label, **kwargs)

    monkeypatch.setattr(fake, "shell", check_deployed_inputs)
    execution = tmp_path / "real-step-execution"
    result = board.execute_real_feature_draft(fixture["out"], execution)
    assert result["status"] == "REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED", result
    assert result["source_preservation"] == "UNCHANGED_BOUND_SOURCES_AND_PREPARED_INPUT_BYTES"
    assert result["host_input_readback"] == "MATCHED_REAL_SOURCE_BYTES"
    assert result["synthetic_golden_comparison"] == "NOT_RUN"
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["state_continuity"] == "NOT_VALIDATED"
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    assert result["draft_top1"] == [list(range(15))]
    remote = fake.instances[0]
    assert remote.actual_input_payloads_checked is True
    assert sum(label == "execute-real-feature-draft" for label, _ in remote.commands) == 1
    model_sends = [(local, target) for local, target in remote.sends
                   if local == manifest["model_source"]["path"]]
    assert len(model_sends) == 1 and model_sends[0][1].endswith("model.omc.partial")
    assert not (fixture["out"] / "model").exists() and not (execution / "model").exists()
    assert result["host_model_copy_bytes"] == 0
    assert list(tmp_path.rglob("model.omc")) == [Path(manifest["model_source"]["path"])]
    native = json.loads((execution / "results/result.json").read_text())
    assert native["model_sha256"] == manifest["model"]["sha256"]
    assert native["draft_abi_sha256"] == manifest["draft_abi"]["sha256"]
    assert native["step_plan_sha256"] == manifest["step_plan"]["sha256"]
    assert [row["sha256"] for row in native["inputs"]] == [row["sha256"] for row in manifest["inputs"]]
    assert manifest["token_provenance"]["anchor_token_id"] == 220
    assert manifest["feature_provenance"]["statistics"]["finite_max_abs"] == 65504.0
    assert real.validate_real_feature_draft(fixture["out"]) == manifest
    for path, (before, raw) in original.items():
        current = path.stat()
        assert path.read_bytes() == raw
        assert (current.st_ino, current.st_nlink, current.st_mtime_ns, current.st_ctime_ns) == (
            before.st_ino, before.st_nlink, before.st_mtime_ns, before.st_ctime_ns)

"""Restore-only probes rehash small ctime-drifted primaries, never weights."""
import os

import pytest

from qwen35_moe_dflash import target_model_io_probe as probe
from test_target_model_io_probe import case as io_case, execute, FakeHdc
from test_original_target_primary_revalidation import ctime_only


@pytest.fixture
def case(tmp_path, monkeypatch):
    return io_case.__wrapped__(tmp_path, monkeypatch)


def test_restore_probe_admits_only_freshly_hashed_historical_primary_ctime(case):
    report_path = case["job"] / "compilation/report.json"
    saved = report_path.read_bytes()
    ctime_only(case["primary"])
    result = execute(case)
    assert result["status"] == "IO_METADATA_OBSERVED", result
    assert result["model_inference"] == "NOT_RUN" and result["sidecar_upload"] == "NOT_RUN"
    assert len(result["primary_omc_revalidation_observations"]) == 1
    observation = result["primary_omc_revalidation_observations"][0]
    assert observation["status"] == "SHA256_VERIFIED" and observation["bytes"] == case["primary"].stat().st_size
    assert report_path.read_bytes() == saved
    assert not any(".weight" in local for local, remote in FakeHdc.sends)


@pytest.mark.parametrize("change", ["content", "inode", "mtime"])
def test_restore_probe_refuses_nonctime_primary_changes_before_any_device_call(case, change):
    path = case["primary"]
    before = path.stat()
    if change == "content":
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "inode":
        raw = path.read_bytes()
        path.rename(path.with_name("retained-old-primary"))
        path.write_bytes(raw)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
    result = execute(case)
    assert result["status"] == "FAILED" and result["model_inference"] == "NOT_RUN"
    assert not FakeHdc.instances and not FakeHdc.sends


def test_restore_probe_preserves_active_primary_identity_after_historical_revalidation(case, monkeypatch):
    ctime_only(case["primary"])
    build = probe._build_runner
    def mutate_after_baseline(out, timeout):
        runner = build(out, timeout)
        ctime_only(case["primary"])
        return runner
    monkeypatch.setattr(probe, "_build_runner", mutate_after_baseline)
    result = execute(case)
    assert result["status"] == "FAILED" and result["model_inference"] == "NOT_RUN"
    assert "changed" in str(result)
    assert not any(local.endswith("model.omc") for local, remote in FakeHdc.sends)


def test_header_only_does_not_rehash_or_admit_changed_primary_metadata(case):
    ctime_only(case["primary"])
    result = execute(case, headers_only=True)
    assert result["status"] == "FAILED" and result["primary_omc_revalidation_observations"] == []
    assert not FakeHdc.instances

"""File-recv artifact validation; no HDC/device/model process is executed."""
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
from types import SimpleNamespace

import pytest

from qwen35_moe_dflash import hdc_run


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    output = tmp_path / "receive-run"
    output.mkdir()
    state = {"calls": [], "timers": [], "action": None}
    def fake(argv, **options):
        assert argv[:5] == ["fixture-hdc", "-t", "selected-device", "-s", "selected-server:8790"]
        assert argv[5:7] == ["file", "recv"]
        state["calls"].append(list(argv))
        if state["action"] is not None:
            result = state["action"](Path(argv[-1]), len(state["calls"]))
            if result is not None:
                return result
        return SimpleNamespace(returncode=0, stdout="FileTransfer finish\n", stderr="original stderr\n")
    monkeypatch.setattr(hdc_run.subprocess, "run", fake)
    client = hdc_run.Hdc("fixture-hdc", "selected-device", "selected-server:8790", output, 3)
    state.update(client=client, output=output, local=output / "target.bin", remote="/data/local/tmp/retained/output_10")
    yield state
    for timer in state["timers"]:
        timer.join(timeout=2)


def receive(value):
    return value["client"].receive(value["remote"], value["local"], timeout=2)


def audit(value):
    paths = list((value["output"] / "receives").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


@pytest.mark.parametrize("raw", [b"", b"fresh received bytes\x00"])
def test_new_regular_file_is_published_without_overwrite_and_raw_hdc_response_is_preserved(transfer, raw):
    def write(candidate, attempt):
        candidate.write_bytes(raw)
    transfer["action"] = write
    assert receive(transfer) is None
    assert transfer["local"].read_bytes() == raw
    assert transfer["local"].stat().st_nlink == 1
    assert len(transfer["calls"]) == 1
    candidate = Path(transfer["calls"][0][-1])
    assert candidate != transfer["local"] and not candidate.exists()
    assert stat.S_IMODE(candidate.parent.stat().st_mode) == 0o700
    record = json.loads(next((transfer["output"] / "calls").glob("*.json")).read_text())
    assert record["stdout"] == "FileTransfer finish\n" and record["stderr"] == "original stderr\n"
    result = audit(transfer)
    assert result["status"] == "RECEIVED_NEW_REGULAR_FILE"
    assert result["operation"] == "READ_ONLY_FILE_RECV_ONLY"
    assert result["checksum_validation"] == "CALLER_MUST_VERIFY_INDEPENDENT_REMOTE_SHA256"


@pytest.mark.parametrize("channel", ["stdout", "stderr", "empty"])
def test_zero_exit_without_file_retries_only_recv_three_unique_destinations_and_retains_original_error(transfer, channel):
    raw = "[Fail]Device not founded or connected\nExecuteCommand need connect-key?\n"
    def disconnected(candidate, attempt):
        return SimpleNamespace(returncode=0, stdout=raw if channel == "stdout" else "",
            stderr=raw if channel == "stderr" else "")
    transfer["action"] = disconnected
    with pytest.raises(RuntimeError) as caught:
        receive(transfer)
    assert "three read-only attempts" in str(caught.value)
    if channel != "empty": assert raw.splitlines()[0] in str(caught.value)
    assert not transfer["local"].exists()
    assert len(transfer["calls"]) == 3
    assert len({argv[-1] for argv in transfer["calls"]}) == 3
    assert all(argv[5:8] == ["file", "recv", transfer["remote"]] for argv in transfer["calls"])
    result = audit(transfer)
    assert len(result["attempts"]) == 3 and result["status"] == "FAILED"
    assert all(row["status"] == "ZERO_EXIT_WITHOUT_LOCAL_ARTIFACT" for row in result["attempts"])
    records = [json.loads(path.read_text()) for path in sorted((transfer["output"] / "calls").glob("*.json"))]
    assert all(row["returncode"] == 0 for row in records)
    assert all(row["stdout"] == (raw if channel == "stdout" else "") for row in records)
    assert all(row["stderr"] == (raw if channel == "stderr" else "") for row in records)


def test_file_materializing_just_after_client_exit_can_be_observed_without_another_recv(transfer):
    def late(candidate, attempt):
        timer = threading.Timer(0.03, lambda: candidate.write_bytes(b"late complete artifact"))
        transfer["timers"].append(timer)
        timer.start()
    transfer["action"] = late
    receive(transfer)
    assert len(transfer["calls"]) == 1
    assert transfer["local"].read_bytes() == b"late complete artifact"


def test_late_first_attempt_artifact_never_substitutes_for_second_attempt(transfer):
    previous = []
    def action(candidate, attempt):
        if attempt == 1:
            previous.append(candidate)
        else:
            previous[0].write_bytes(b"late stale first attempt")
            candidate.write_bytes(b"fresh second attempt")
    transfer["action"] = action
    receive(transfer)
    assert len(transfer["calls"]) == 2
    assert transfer["local"].read_bytes() == b"fresh second attempt"
    assert previous[0].read_bytes() == b"late stale first attempt"
    assert audit(transfer)["attempts"][0]["status"] == "ZERO_EXIT_WITHOUT_LOCAL_ARTIFACT"


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo", "hardlink"])
def test_nonregular_or_shared_hdc_artifact_is_rejected_without_accepting_nested_files(transfer, kind):
    target = transfer["output"] / "unrelated-target"
    target.write_bytes(b"original target bytes")
    def create(candidate, attempt):
        if kind == "directory":
            candidate.mkdir()
            (candidate / "output_10").write_bytes(b"nested file is not the requested artifact")
        elif kind == "symlink": candidate.symlink_to(target)
        elif kind == "fifo": os.mkfifo(candidate)
        else: os.link(target, candidate)
    transfer["action"] = create
    with pytest.raises(RuntimeError, match="nonregular or shared") as caught:
        receive(transfer)
    assert "FileTransfer finish" in str(caught.value) and "original stderr" in str(caught.value)
    assert not transfer["local"].exists() and len(transfer["calls"]) == 1
    assert target.read_bytes() == b"original target bytes"
    assert audit(transfer)["status"] == "FAILED"


@pytest.mark.parametrize("kind", ["regular", "directory", "symlink", "dangling_symlink", "fifo"])
def test_existing_final_path_is_never_given_to_hdc_or_overwritten(transfer, kind):
    local = transfer["local"]
    target = transfer["output"] / "existing-target"
    target.write_bytes(b"leave unrelated bytes untouched")
    if kind == "regular": local.write_bytes(b"preexisting local bytes")
    elif kind == "directory": local.mkdir()
    elif kind == "symlink": local.symlink_to(target)
    elif kind == "dangling_symlink": local.symlink_to(target.with_name("absent"))
    else: os.mkfifo(local)
    before = local.lstat()
    with pytest.raises(FileExistsError, match="already exists"):
        receive(transfer)
    assert transfer["calls"] == [] and transfer["client"].count == 0
    after = local.lstat()
    assert (before.st_dev, before.st_ino, before.st_mode, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_dev, after.st_ino, after.st_mode, after.st_mtime_ns, after.st_ctime_ns)
    if kind == "regular": assert local.read_bytes() == b"preexisting local bytes"
    assert target.read_bytes() == b"leave unrelated bytes untouched"


def test_concurrent_final_file_creation_is_not_overwritten_during_publication(transfer):
    def create(candidate, attempt):
        transfer["local"].write_bytes(b"concurrently created final bytes")
        candidate.write_bytes(b"remote bytes")
    transfer["action"] = create
    with pytest.raises(RuntimeError):
        receive(transfer)
    assert transfer["local"].read_bytes() == b"concurrently created final bytes"
    assert Path(transfer["calls"][0][-1]).read_bytes() == b"remote bytes"
    assert audit(transfer)["status"] == "FAILED"


def test_symlink_destination_parent_is_rejected_before_any_transfer(transfer):
    real = transfer["output"] / "real-parent"
    real.mkdir()
    alias = transfer["output"] / "linked-parent"
    alias.symlink_to(real, target_is_directory=True)
    transfer["local"] = alias / "new.bin"
    with pytest.raises(ValueError, match="ordinary destination parents"):
        receive(transfer)
    assert not list(real.iterdir()) and not transfer["calls"]


@pytest.mark.parametrize("kind", ["nonzero", "timeout"])
def test_nonzero_or_timeout_preserves_raw_call_evidence_without_receive_retry(transfer, kind):
    def fail(candidate, attempt):
        if kind == "nonzero":
            return SimpleNamespace(returncode=9, stdout="raw transfer stdout\n", stderr="raw transfer stderr\n")
        raise subprocess.TimeoutExpired(["fixture-hdc"], 2, output=b"timeout stdout", stderr=b"timeout stderr")
    transfer["action"] = fail
    with pytest.raises(RuntimeError) as caught:
        receive(transfer)
    assert len(transfer["calls"]) == 1 and not transfer["local"].exists()
    assert "stdout" in str(caught.value) and "stderr" in str(caught.value)
    record = json.loads(next((transfer["output"] / "calls").glob("*.json")).read_text())
    assert record["stdout"] and record["stderr"]
    assert audit(transfer)["attempts"][0]["status"] == "HDC_CALL_FAILED"

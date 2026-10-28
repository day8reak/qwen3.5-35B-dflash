"""Host-only fake-HDC tests; these snapshots are not device execution evidence."""
import hashlib
from pathlib import Path, PurePosixPath
import shlex

import pytest

from qwen35_moe_dflash import draft_hilog as hilog
from qwen35_moe_dflash.draft_board import _receive
from qwen35_moe_dflash.hashing import sha256_file


SNAPSHOT = ("10-28 11:30:00.001 42 42 I A00000/qwen35_dflash_hiai: HIAI_Foundation_Init(3) init hiai foundation success\n"
            "10-28 11:30:00.002 99 99 E A00000/AI_FMK: Load unrelated model failed\n"
            "10-28 11:30:00.003 42 42 E A00000/AI_FMK: RestoreFromBuffer failed for draft\n")


class FakeHdc:
    def __init__(self, *, timeout=100, status=0, saved=None, snapshot=SNAPSHOT,
                 missing=None, corrupt=None, disconnect=False):
        self.timeout = timeout
        self.status = status
        self.saved = str(status) + "\n" if saved is None else saved
        self.snapshot = snapshot
        self.missing, self.corrupt, self.disconnect = missing, corrupt, disconnect
        self.calls = []
        self.files = {}

    def shell(self, command, label, *, allow_failure=False, timeout=None):
        self.calls.append((label, timeout, command))
        if label == "capture-draft-hilog":
            assert allow_failure
            self.files = {"draft-hilog.txt": self.snapshot.encode(),
                          "draft-hilog.stderr": b"fake diagnostic stderr\n",
                          "draft-hilog.exitcode": self.saved.encode()}
            self.files.pop(self.missing, None)
            if self.disconnect:
                raise RuntimeError("fake missing remote completion marker")
            return self.status, ""
        assert label == "hash-result"
        name = PurePosixPath(shlex.split(command)[3].rstrip(";")).name
        if name not in self.files:
            return 0, "__MISSING__"
        return 0, hashlib.sha256(self.files[name]).hexdigest() + "  /board/" + name

    def receive(self, remote, local, *, timeout=None):
        self.calls.append(("receive", timeout, str(remote)))
        contents = self.files[remote.name]
        if remote.name == self.corrupt:
            contents += b"corrupted during fake delivery"
        local.write_bytes(contents)


def capture(tmp_path, **kwargs):
    hdc = FakeHdc(**kwargs)
    result = hilog.capture_draft_hilog(hdc, PurePosixPath("/data/local/tmp/draft-test"), tmp_path,
                                      receive=_receive, recorded_pid=42)
    return hdc, result


@pytest.mark.parametrize("timeout", [100, .25])
def test_hash_verified_snapshot_selects_only_recorded_pid_with_bounded_calls(tmp_path, timeout):
    hdc, result = capture(tmp_path, timeout=timeout)
    assert result["status"] == "CAPTURED"
    assert result["returncode"] == result["recorded_exitcode"] == 0
    assert set(result["retrieved_files"]) == set(hilog._NAMES)
    assert hdc.timeout == timeout
    assert all(call[1] == min(timeout, 10) for call in hdc.calls)
    captures = [call for call in hdc.calls if call[0] == "capture-draft-hilog"]
    assert len(captures) == 1 and captures[0][2].count("hilog -x") == 1
    assert "-r" not in captures[0][2] and "-w" not in captures[0][2]
    assert "shared" in result["scope"] and "PID" in result["scope"]
    assert result["summary"]["recorded_pid"] == 42
    selected = "\n".join(result["summary_lines"])
    assert "RestoreFromBuffer failed for draft" in selected
    assert "unrelated" not in selected
    artifact = result["candidate_log"]
    assert artifact["sha256"] == sha256_file(Path(artifact["path"]))
    assert artifact["source_sha256"] == result["retrieved_sha256"]["draft-hilog.txt"]


def test_absent_recorded_pid_does_not_substitute_another_process(tmp_path):
    hdc, result = capture(tmp_path, snapshot=SNAPSHOT.replace(" 42 42 ", " 13 13 "))
    assert result["status"] == "CAPTURED" and result["summary_lines"] == []
    assert result["summary"]["status"] == "NO_MATCHING_RECORDS"
    assert "candidate_log" not in result


def test_missing_pid_is_explicitly_shared_candidate_selection(tmp_path):
    result = hilog.capture_draft_hilog(FakeHdc(), "/data/local/tmp/draft-test", tmp_path, receive=_receive)
    assert result["status"] == "CAPTURED"
    assert "not confirmed" in result["summary"]["scope"]
    assert result["process_identity"]["recorded_pid"] is None


@pytest.mark.parametrize("kwargs,expected", [
    ({"status": 127}, "UNAVAILABLE"),
    ({"snapshot": "Mutlti commands can't be used in combination [CODE: -31]\n"}, "UNAVAILABLE"),
    ({"saved": "1\n"}, "FAILED"),
    ({"saved": "oops\n"}, "FAILED"),
    ({"saved": "256\n"}, "FAILED"),
    ({"disconnect": True}, "FAILED"),
])
def test_unavailable_or_unverified_snapshot_never_shows_summary(tmp_path, kwargs, expected):
    _, result = capture(tmp_path, **kwargs)
    assert result["status"] == expected and result["summary_lines"] == []
    assert "candidate_log" not in result
    assert len(result["retrieved_files"]) == 3
    if kwargs.get("disconnect"):
        assert "missing remote completion marker" in result["error"]


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
@pytest.mark.parametrize("name", list(hilog._NAMES))
def test_partial_or_corrupt_transfer_is_retained_without_claiming_snapshot(tmp_path, kind, name):
    _, result = capture(tmp_path, **{kind: name})
    assert result["status"] == "FAILED" and result["summary_lines"] == []
    assert name not in result["retrieved_sha256"]
    assert result["retrieval_errors"][0]["path"] == name
    assert not (tmp_path / name).exists()


def test_large_snapshot_keeps_useful_failure_without_loading_whole_file(tmp_path, monkeypatch):
    snapshot = ("unrelated shared system log " + "x" * 1000 + "\n") * 17000 + SNAPSHOT
    assert len(snapshot) > 16 * 1024 * 1024
    original = Path.read_text
    def no_whole_snapshot(path, *args, **kwargs):
        assert path.name != "draft-hilog.txt", "snapshot selection must stream"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", no_whole_snapshot)
    _, result = capture(tmp_path, snapshot=snapshot)
    assert result["status"] == "CAPTURED"
    assert result["summary"]["snapshot_bytes"] == len(snapshot.encode())
    assert result["summary"]["selection_io"] == "streamed"
    assert "RestoreFromBuffer failed for draft" in "\n".join(result["summary_lines"])
    assert len(result["summary_lines"]) <= 100


def test_service_root_cause_around_explicit_runner_pid_is_included(tmp_path):
    snapshot = (
        "03-11 05:49:25.128 42 42 I C02154/runner/AI_FMK: load model begin\n"
        "03-11 05:49:25.174 1112 9 E C02151/hiaiserver/AI_NPUCL: unrelated thread error\n"
        "03-11 05:49:25.175 1112 1150 E C02151/hiaiserver/AI_NPUCL: Invalid weights memory size: weightsMemBaseSize=16 modelWeightMemorySize=1151486976\n"
        "03-11 05:49:25.175 1112 1150 E C02151/hiaiserver/AI_NPUCL: InitRuntimeParams failed\n"
        "03-11 05:49:25.175 1112 1150 E C02151/hiaiserver/AI_NPUCL: model loaded fail, Pid: 42\n"
        "03-11 05:49:25.176 42 42 E C02151/runner/AI_NPUCL: Init failed\n"
        "03-11 05:49:29.175 1112 1150 E C02151/hiaiserver/AI_NPUCL: outside runner lifetime error\n")
    _, result = capture(tmp_path, snapshot=snapshot)
    selected = "\n".join(result["summary_lines"])
    assert "weightsMemBaseSize=16" in selected and "model loaded fail" in selected
    assert "unrelated thread" not in selected and "outside runner lifetime" not in selected
    assert result["summary"]["service_context"]["server_tid"] == 1150
    assert result["summary"]["service_context"]["recorded_pid"] == 42


def test_service_errors_without_explicit_matching_pid_are_not_attributed(tmp_path):
    snapshot = SNAPSHOT.replace(
        "10-28 11:30:00.002 99 99 E A00000/AI_FMK: Load unrelated model failed",
        "10-28 11:30:00.002 1112 1150 E C02151/hiaiserver/AI_NPUCL: model loaded fail, Pid: 420")
    _, result = capture(tmp_path, snapshot=snapshot)
    assert "Pid: 420" not in "\n".join(result["summary_lines"])
    assert result["summary"]["service_context"]["status"] == "NO_EXPLICIT_PID_FAILURE"


def test_changed_snapshot_is_not_displayed(tmp_path, monkeypatch):
    original = hilog._hilog_summary_details
    def changed(path, **kwargs):
        result = original(path, **kwargs)
        path.write_text("changed after initial SHA verification")
        return result
    monkeypatch.setattr(hilog, "_hilog_summary_details", changed)
    _, result = capture(tmp_path)
    assert result["status"] == "FAILED" and result["summary_lines"] == []
    assert "changed during" in result["summary_error"]
    assert "candidate_log" not in result


@pytest.mark.parametrize("pid", [0, -1, True, "42"])
def test_invalid_pid_does_not_start_diagnostic(tmp_path, pid):
    hdc = FakeHdc()
    result = hilog.capture_draft_hilog(hdc, "/data/local/tmp/draft-test", tmp_path,
                                      receive=_receive, recorded_pid=pid)
    assert result["status"] == "FAILED" and hdc.calls == []


def test_existing_snapshot_is_not_overwritten(tmp_path):
    path = tmp_path / "draft-hilog.txt"
    path.write_text("retained earlier evidence")
    hdc, result = capture(tmp_path)
    assert result["status"] == "FAILED" and hdc.calls == []
    assert path.read_text() == "retained earlier evidence"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_receive_rejects_invalid_per_call_timeout(tmp_path, timeout):
    hdc = FakeHdc()
    with pytest.raises(ValueError, match="timeout"):
        _receive(hdc, PurePosixPath("/board"), tmp_path, "draft-hilog.txt", timeout_seconds=timeout)
    assert hdc.calls == []

"""Compiler wait observations never change the compiler's result or lifetime."""
from pathlib import Path
import subprocess
import threading

import pytest

from qwen35_moe_dflash.compile_progress import compiler_heartbeat


def test_disabled_monitor_does_not_read_the_log(tmp_path, monkeypatch):
    log = tmp_path / "missing.log"
    original_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        if path == log:
            pytest.fail("disabled heartbeat must not access the log")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    with compiler_heartbeat(log, None, interval_seconds=.01) as report:
        assert report["enabled"] is False
    assert report["samples"] == 0 and report["error"] is None


@pytest.mark.parametrize("last_line", ["\x1b[31mLATEST\x1b[0m\x00\x07", "X" * 1000])
def test_monitor_reads_only_bounded_tail_and_sanitizes_console_text(tmp_path, monkeypatch, last_line):
    log = tmp_path / "compile.log"
    payload = b"UNRELATED EARLY LOG\n" * 10000 + (last_line + "\n").encode()
    log.write_bytes(payload)
    original_open, original_read_text, original_read_bytes = Path.open, Path.read_text, Path.read_bytes
    read_sizes = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            assert 0 < size <= 4096, "heartbeat must not read the complete compiler log"
            read_sizes.append(size)
            return self.stream.read(size)

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return BoundedReader(stream) if path == log else stream

    def guarded_text(path, *args, **kwargs):
        if path == log:
            pytest.fail("heartbeat must not use whole-log read_text")
        return original_read_text(path, *args, **kwargs)

    def guarded_bytes(path, *args, **kwargs):
        if path == log:
            pytest.fail("heartbeat must not use whole-log read_bytes")
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    sampled, rows = threading.Event(), []

    def progress(row):
        rows.append(row)
        sampled.set()

    with compiler_heartbeat(log, progress, interval_seconds=.01) as report:
        assert sampled.wait(1), "monitor did not report within the bounded test wait"
    assert read_sizes and report["samples"] >= 1 and report["error"] is None
    assert report["monitor_stop_timeout"] is False
    row = rows[-1]
    assert row["stage"] == "omg-wait" and "OMG" in row["message"]
    assert row["log_bytes"] == len(payload)
    assert row["elapsed_seconds"] >= 0 and row["log_idle_seconds"] >= 0
    assert len(row["last_line"]) <= 260
    assert all(char.isprintable() for char in row["last_line"])
    assert "UNRELATED EARLY LOG" not in row["last_line"]
    assert "\x1b" not in row["last_line"] and "[31m" not in row["last_line"]
    if last_line.startswith("\x1b"):
        assert row["last_line"] == "LATEST"
    count = len(rows)
    threading.Event().wait(.04)
    assert len(rows) == count, "normal context exit must stop future progress callbacks"


def test_missing_log_is_observable_without_aborting_compile_wait(tmp_path):
    sampled = threading.Event()
    with compiler_heartbeat(tmp_path / "not-created.log", lambda row: sampled.set(),
                            interval_seconds=.01) as report:
        assert sampled.wait(1)
    assert report["samples"] >= 1 and report["error"] is None
    assert "FileNotFoundError" in report["last"]["log_error"]
    assert report["monitor_stop_timeout"] is False


def test_callback_failure_disables_observer_without_failing_compile_body(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text("compiler still running\n")
    entered, rows = threading.Event(), []

    def failed_progress(row):
        rows.append(row)
        entered.set()
        raise RuntimeError("fixture progress callback failed")

    with compiler_heartbeat(log, failed_progress, interval_seconds=.01) as report:
        assert entered.wait(1)
        compiler_result = "SUCCEEDED"
    assert compiler_result == "SUCCEEDED"
    assert report["samples"] == 1
    assert "fixture progress callback failed" in report["error"]
    assert report["monitor_stop_timeout"] is False
    threading.Event().wait(.04)
    assert len(rows) == 1


@pytest.mark.parametrize("error", [RuntimeError("compiler failed"),
                                    subprocess.CalledProcessError(7, ["fixture-omg"]),
                                    subprocess.TimeoutExpired(["fixture-omg"], 3)])
def test_compiler_errors_and_timeouts_are_preserved_and_stop_observer(tmp_path, error):
    log = tmp_path / "compile.log"
    log.write_text("last diagnostic\n")
    sampled, rows = threading.Event(), []

    def progress(row):
        rows.append(row)
        sampled.set()

    with pytest.raises(type(error)) as raised:
        with compiler_heartbeat(log, progress, interval_seconds=.01) as report:
            assert sampled.wait(1)
            raise error
    assert raised.value is error
    assert report["error"] is None and report["monitor_stop_timeout"] is False
    count = len(rows)
    threading.Event().wait(.04)
    assert len(rows) == count


def test_callback_failure_does_not_mask_compiler_timeout(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text("waiting\n")
    entered = threading.Event()
    timeout = subprocess.TimeoutExpired(["fixture-omg"], 3)

    def failed_progress(row):
        entered.set()
        raise ValueError("observer failed too")

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        with compiler_heartbeat(log, failed_progress, interval_seconds=.01) as report:
            assert entered.wait(1)
            raise timeout
    assert raised.value is timeout
    assert "observer failed too" in report["error"]


def test_blocked_callback_is_reported_without_unbounded_context_exit(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text("waiting\n")
    entered, release = threading.Event(), threading.Event()
    workers, calls = [], []

    def blocked_progress(row):
        workers.append(threading.current_thread())
        calls.append(row)
        entered.set()
        assert release.wait(3), "test must release its deliberately blocked callback"

    try:
        with compiler_heartbeat(log, blocked_progress, interval_seconds=.01) as report:
            assert entered.wait(1)
        assert report["monitor_stop_timeout"] is True
        saved = dict(report)
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=1)
            assert not worker.is_alive()
    assert len(calls) == 1 and report == saved

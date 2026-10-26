"""Bounded HiLog candidate windows, without claiming current-run attribution."""
import hashlib

import pytest

from qwen35_moe_dflash.hdc_run import _hilog_summary, _hilog_summary_details


def _line(message, *, pid=50375, date="03-06", level="I", tag=None):
    tag = tag or "A0FFFF/qwen35_dflash_hiai/HIAI_DDK_MSG"
    return f"{date} 06:54:14.142 {pid} {pid} {level} {tag}: {message}\n"


def _init(**kwargs):
    return _line('hiai_foundation_dl_helper.c HIAI_Foundation_Init(53)::"init hiai foundation success."', **kwargs)


def _snapshot(tmp_path, lines):
    path = tmp_path / "hiai-hilog.txt"
    path.write_text("".join(lines))
    return path


def test_first_error_and_restore_context_survive_many_cleanup_errors(tmp_path):
    lines = [_init(), _line("foundation ready"),
             _line("DDK dlopen attempt failed, subsequent fallback succeeds", level="E")]
    lines += [_line(f"setup progress {i}") for i in range(90)]
    lines += [_line("kernel compatibility check begins"),
              _line("actual model incompatibility detail", level="E"),
              _line('RestoreFromBuffer(267)::"restore model failed."', level="E")]
    lines += [_line(f"BinderInvoker cleanup failure {i}", level="E") for i in range(200)]
    path = _snapshot(tmp_path, lines)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    selected, matches, metadata = _hilog_summary_details(path)
    text = "\n".join(selected)
    assert "DDK dlopen attempt failed" in text
    assert "actual model incompatibility detail" in text
    assert "kernel compatibility check begins" in text
    assert 'restore model failed' in text
    assert "BinderInvoker cleanup failure 199" in text
    assert len(selected) <= 60 and matches == len(lines)
    numbers = [int(line.split(":", 1)[0][1:]) for line in selected]
    assert numbers == sorted(set(numbers))
    assert metadata["first_error_line"] == 3
    assert metadata["restore_failure_line"] == 96
    assert metadata["candidate_lines"] == len(lines)
    assert metadata["method"] == "latest-observed-app-session-candidate"
    assert "not confirmed as this run" in metadata["scope"]
    assert "not a root-cause verdict" in metadata["scope"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert _hilog_summary(path) == (selected, matches)


def test_latest_session_start_wins_over_old_process_late_cleanup(tmp_path):
    path = _snapshot(tmp_path, [
        _init(pid=100), _line("old process error", pid=100, level="E"),
        _line("new process starts", pid=200), _init(pid=200),
        _line("new process error", pid=200, level="E"),
        _line("old process late cleanup", pid=100, level="E"),
    ])
    selected, _, metadata = _hilog_summary_details(path)
    assert metadata["candidate_pid"] == 200 and metadata["candidate_start_line"] == 3
    assert metadata["candidate_foundation_init_line"] == 4
    assert "new process error" in "\n".join(selected)
    assert "old process" not in "\n".join(selected)


@pytest.mark.parametrize("reuse", ["next-date", "same-date-init"])
def test_reused_pid_does_not_merge_older_session(tmp_path, reuse):
    old_date = "03-05" if reuse == "next-date" else "03-06"
    path = _snapshot(tmp_path, [
        _init(date=old_date), _line("old session error", date=old_date, level="E"),
        _init(), _line("current candidate error", level="E"),
    ])
    selected, _, metadata = _hilog_summary_details(path)
    assert metadata["candidate_pid"] == 50375 and metadata["candidate_date"] == "03-06"
    assert metadata["candidate_start_line"] == metadata["candidate_foundation_init_line"] == 3
    assert metadata["candidate_lines"] == 2
    assert "old session" not in "\n".join(selected)
    assert "current candidate error" in "\n".join(selected)


def test_same_pid_loader_errors_included_but_other_services_and_transport_excluded(tmp_path):
    path = _snapshot(tmp_path, [
        _init(),
        _line("load libc++_shared.so failed", level="E", tag="C03F00/MUSL-LDSO"),
        _line("input dispatch failed", pid=927, level="E", tag="C02801/multimodalinput/MMIService"),
        _line("NPU service load failed", pid=1123, level="E", tag="C01234/service/NPU"),
        _line("transfer target.omc libhiai.so", tag="C02D13/hdcd/HDC_LOG"),
        _line("fake tagged transport", pid=60000, tag="C02D13/qwen35_dflash_hiai/HDC_LOG"),
        _line("RestoreFromBuffer return 1, load failed", level="E"),
    ])
    selected, matches, metadata = _hilog_summary_details(path)
    text = "\n".join(selected)
    assert "libc++_shared.so" in text and "RestoreFromBuffer" in text
    assert "input dispatch" not in text and "NPU service" not in text and "transport" not in text
    assert "HDC_LOG" not in text
    assert metadata["candidate_pid"] == 50375 and metadata["candidate_lines"] == 3
    assert metadata["first_error_line"] == 2
    assert matches == 3  # Global keywords include the unrelated NPU service, not MUSL.


def test_unparseable_app_lines_keep_exact_legacy_keyword_tail(tmp_path):
    lines = [f"NPU error {index}: " + "x" * 2000 + "\n" for index in range(80)]
    lines += ["03-06 06:32:41 I hdcd/HDC_LOG: send libhiai.so\n",
              _line("fake app transport", tag="C02D13/qwen35_dflash_hiai/HDC_LOG")]
    path = _snapshot(tmp_path, lines)
    selected, matches, metadata = _hilog_summary_details(path)
    assert matches == 80 and len(selected) == 60
    assert selected[0].startswith("L21: NPU error 20:")
    assert selected[-1].startswith("L80: NPU error 79:")
    assert max(map(len, selected)) <= 1000
    assert metadata["method"] == "keyword-tail" and "candidate_pid" not in metadata
    assert path.read_text() == "".join(lines)


def test_parseable_non_app_records_are_only_keyword_candidates(tmp_path):
    path = _snapshot(tmp_path, [
        _line("input noise", pid=927, tag="C02801/multimodalinput/MMIService"),
        _line("NPU failure unrelated to runner", pid=1123, level="E", tag="C01234/service/NPU"),
    ])
    selected, matches, metadata = _hilog_summary_details(path)
    assert len(selected) == matches == 1 and "NPU failure" in selected[0]
    assert metadata["method"] == "keyword-tail" and "not attributed" in metadata["scope"]


def test_candidate_lines_are_bounded_and_latest_without_init_is_labelled(tmp_path):
    path = _snapshot(tmp_path, [_init(pid=100), _line("old error", pid=100, level="E"),
                               _line("x" * 5000, pid=200), _line("still running", pid=200)])
    selected, _, metadata = _hilog_summary_details(path)
    assert len(selected) == 2 and max(map(len, selected)) <= 1000
    assert metadata["candidate_pid"] == 200
    assert metadata["candidate_foundation_init_line"] is None
    assert metadata["first_error_line"] is None and metadata["restore_failure_line"] is None

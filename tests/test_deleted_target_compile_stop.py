"""Exact deleted-job cleanup through fake proc identities and signal doubles."""
import json
import os
from pathlib import Path
import signal

import pytest

from qwen35_moe_dflash import deleted_target_compile_stop as stop


def process(case, pid, ppid, *, argv=None, cwd=None, start=None, state="T", env=None):
    folder = case["proc"] / str(pid)
    folder.mkdir(exist_ok=True)
    fields = [state, str(ppid)] + ["0"] * 17 + [str(start or pid * 10)]
    (folder / "stat").write_text(f"{pid} (worker with spaces) " + " ".join(fields))
    (folder / "cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in (argv or ["python3", "worker"])) + b"\0")
    link = folder / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd or case["cwd"])
    (folder / "environ").write_bytes(env if env is not None else b"AI_RUN_DIR=" + os.fsencode(str(case["run"])) + b"\0SECRET=never report this\0")
    (folder / "fd").mkdir(exist_ok=True)


@pytest.fixture
def case(tmp_path, monkeypatch):
    run, proc = tmp_path / "run", tmp_path / "proc"
    run.mkdir(); proc.mkdir()
    job = run / "full-target-compile-0491abfcd79b"
    cwd = str(job / "compilation/compile-work/generated") + " (deleted)"
    driver = ["python3", "/repo/targets/cpro/scripts/compile_full_target.py", "--air", "/source/export.air", "--output", str(job)]
    argv = ["/selected/ddk/tools/tools_omg/master/omg", "--framework=1",
            "--model=" + str(job / "compilation/compile-work/generated/export.air"),
            "--output=model", "--target=omc", "--platform=kirinx90"]
    result = {"run": run, "proc": proc, "job": job, "cwd": cwd, "argv": argv,
              "driver": driver, "output": run / "deleted-stop-receipt"}
    process(result, 50, 1, argv=driver)
    process(result, 100, 50, argv=argv)
    process(result, 110, 100)
    process(result, 111, 100)
    process(result, 112, 110)
    process(result, 200, 1, argv=["python3", "unrelated"])
    monkeypatch.setenv("AI_RUN_DIR", str(run))
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("unconfigured numeric signal"))
    monkeypatch.setattr(os, "killpg", lambda *a: pytest.fail("process groups forbidden"))
    return result


def invoke(case, **kwargs):
    return stop.stop_deleted_target_compile(100, case["job"], proc_root=case["proc"], **kwargs)


def signals(case, monkeypatch, *, on_signal=None):
    calls = []
    def send(pid, requested):
        assert (case["output"] / "plan.json").is_file()
        text = (case["output"] / "events.jsonl").read_text()
        assert '"SIGNAL_INTENT"' in text and f'"pid": {pid}' in text
        calls.append((pid, requested))
        if on_signal:
            on_signal(pid, requested)
    monkeypatch.setattr(os, "kill", send)
    return calls


def test_default_plan_writes_nothing_and_requires_no_deleted_job_receipt(case, monkeypatch):
    monkeypatch.setattr(os, "pidfd_open", lambda *a: pytest.fail("plan cannot open pidfds"))
    result = invoke(case)
    assert result["status"] == "PLAN_READY", result
    assert not case["job"].exists() and not case["output"].exists()
    assert [row["pid"] for row in result["selected_processes"]] == [112, 110, 111, 100, 50]
    assert result["driver_binding"]["job_binding"] == "EXACT_EXPLICIT_DRIVER_OUTPUT"
    assert result["exit_validation"] == "NOT_PERFORMED"


def test_implicit_generated_output_binds_only_run_env_and_child_without_leaking_env(case):
    process(case, 50, 1, argv=case["driver"][:-2])
    result = invoke(case)
    assert result["status"] == "PLAN_READY", result
    assert result["driver_binding"]["job_binding"] == "GENERATED_JOB_NAME_AND_DRIVER_RUN_ENV_AND_EXACT_CHILD"
    assert "never report this" not in json.dumps(result)


@pytest.mark.parametrize("fault", ["root_model", "root_cwd", "root_not_omg", "root_active", "worker_active", "parent_active",
    "parent_script", "parent_output", "duplicate_output", "implicit_env", "implicit_non_generated", "resume", "other_child", "job_exists", "job_symlink"])
def test_unknown_or_active_ownership_refuses_all_signals(case, monkeypatch, fault):
    if fault == "root_model": process(case, 100, 50, argv=[arg.replace("export.air", "other.air") for arg in case["argv"]])
    elif fault == "root_cwd": process(case, 100, 50, argv=case["argv"], cwd=str(case["run"]))
    elif fault == "root_not_omg": process(case, 100, 50, argv=["/bin/sh", *case["argv"][1:]])
    elif fault == "root_active": process(case, 100, 50, argv=case["argv"], state="S")
    elif fault == "worker_active": process(case, 112, 110, state="S")
    elif fault == "parent_active": process(case, 50, 1, argv=case["driver"], state="S")
    elif fault == "parent_script": process(case, 50, 1, argv=["python3", "/unrelated/script.py", *case["driver"][2:]])
    elif fault == "parent_output": process(case, 50, 1, argv=[*case["driver"][:-1], str(case["run"] / "other-job")])
    elif fault == "duplicate_output": process(case, 50, 1, argv=[*case["driver"], "--output", str(case["job"])])
    elif fault == "implicit_env": process(case, 50, 1, argv=case["driver"][:-2], env=b"AI_RUN_DIR=/other/run\0")
    elif fault == "implicit_non_generated":
        case["job"] = case["run"] / "unknown-job"
        process(case, 50, 1, argv=case["driver"][:-2])
    elif fault == "resume": process(case, 50, 1, argv=[*case["driver"], "--resume", str(case["job"])])
    elif fault == "other_child": process(case, 51, 50)
    elif fault == "job_exists": case["job"].mkdir()
    else: case["job"].symlink_to(case["run"] / "missing")
    calls = signals(case, monkeypatch)
    result = invoke(case, apply=True, output=case["output"], legacy_stopped_tree=True)
    assert result["status"] == "BLOCKED" and calls == [], result
    assert not case["output"].exists()


def test_legacy_queues_term_for_entire_tree_and_driver_before_any_continue(case, monkeypatch):
    calls = signals(case, monkeypatch)
    result = invoke(case, apply=True, output=case["output"], legacy_stopped_tree=True)
    pids = [112, 110, 111, 100, 50]
    assert result["status"] == "STOP_REQUESTED", result
    assert calls == [(pid, signal.SIGTERM) for pid in pids] + [(pid, signal.SIGCONT) for pid in pids]
    assert result["exit_validation"] == "NOT_PERFORMED" and result["reclaimed_bytes"] == "NOT_INFERRED"
    assert json.loads((case["output"] / "report.json").read_text()) == result
    assert not case["job"].exists()


def test_pid_replacement_during_pinning_refuses_before_any_signal(case, monkeypatch):
    original = stop._process_fd
    def changed(pid, fd):
        if pid == 111:
            process(case, 111, 100, start=999999)
        return original(pid, fd)
    monkeypatch.setattr(stop, "_process_fd", changed)
    calls = signals(case, monkeypatch)
    result = invoke(case, apply=True, output=case["output"], legacy_stopped_tree=True)
    assert result["status"] == "BLOCKED" and "identity/state changed" in result["error"]
    assert calls == [] and not case["output"].exists()


def test_pid_replacement_between_signals_is_logged_and_never_signaled(case, monkeypatch):
    def changed(pid, requested):
        if pid == 112 and requested == signal.SIGTERM:
            process(case, 110, 100, start=999999)
    calls = signals(case, monkeypatch, on_signal=changed)
    result = invoke(case, apply=True, output=case["output"], legacy_stopped_tree=True)
    assert result["status"] == "STOP_REQUESTED", result
    assert not any(pid == 110 for pid, requested in calls)
    assert any(row["event"] == "IDENTITY_CHANGED_NOT_SIGNALED" for row in result["events"])


def test_live_or_symlink_output_is_not_replaced(case, monkeypatch):
    case["output"].mkdir()
    (case["output"] / "retained").write_text("existing evidence")
    calls = signals(case, monkeypatch)
    result = invoke(case, apply=True, output=case["output"], legacy_stopped_tree=True)
    assert result["status"] == "BLOCKED" and not calls
    assert (case["output"] / "retained").read_text() == "existing evidence"


def test_missing_pidfd_support_does_not_silently_use_numeric_pid(case, monkeypatch):
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "BLOCKED" and "explicit --legacy-stopped-tree" in result["error"]
    assert not case["output"].exists()


def test_available_pidfds_use_only_pinned_handles_and_preserve_other_process(case, monkeypatch):
    pins, calls = {}, []
    def pin(pid, flags):
        descriptor = os.open("/dev/null", os.O_RDONLY)
        pins[descriptor] = pid
        return descriptor
    def send(descriptor, requested, info, flags):
        assert (case["output"] / "plan.json").exists()
        calls.append((pins[descriptor], requested))
    monkeypatch.setattr(os, "pidfd_open", pin)
    monkeypatch.setattr(signal, "pidfd_send_signal", send, raising=False)
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "STOP_REQUESTED", result
    expected = [112, 110, 111, 100, 50]
    assert calls == [(pid, signal.SIGTERM) for pid in expected] + [(pid, signal.SIGCONT) for pid in expected]
    assert all(pid != 200 for pid, _ in calls)


def test_recreated_job_between_signal_requests_prevents_further_signals(case, monkeypatch):
    def recreate(pid, requested):
        if pid == 112 and requested == signal.SIGTERM:
            case["job"].mkdir()
    calls = signals(case, monkeypatch, on_signal=recreate)
    result = invoke(case, apply=True, output=case["output"], legacy_stopped_tree=True)
    assert result["status"] == "BLOCKED" and "job directory already absent" in result["error"]
    assert calls == [(112, signal.SIGTERM)]


def test_exited_root_is_not_running_without_opening_other_processes(case):
    (case["proc"] / "100/stat").unlink()
    result = invoke(case)
    assert result["status"] == "NOT_RUNNING" and not case["output"].exists()


def test_deleted_log_fd_1_and_2_count_one_regular_inode_without_reading_payload(case, monkeypatch):
    path = case["run"] / "old-log"
    path.write_bytes(b"log" * 2048)
    fd = os.open(path, os.O_RDONLY)
    path.unlink()
    try:
        links = set()
        for number in (1, 2):
            link = case["proc"] / "100/fd" / str(number)
            link.symlink_to(f"/proc/self/fd/{fd}")
            links.add(str(link))
        original = os.readlink
        def observed(value, *args, **kwargs):
            return str(case["job"] / "compilation/compile.log") + " (deleted)" if str(value) in links else original(value, *args, **kwargs)
        monkeypatch.setattr(os, "readlink", observed)
        report = invoke(case)
        assert report["status"] == "PLAN_READY", report
        deleted = report["deleted_open_files"]
        assert deleted["unique_regular_inodes"] == 1 and deleted["logical_bytes"] == 6144
        assert [row["fd"] for row in deleted["files"][0]["holders"]] == [1, 2]
        assert deleted["allocated_bytes"] == os.fstat(fd).st_blocks * 512
    finally:
        os.close(fd)


def test_signal_failure_retains_journal_and_never_escalates(case, monkeypatch):
    def denied(pid, requested):
        raise PermissionError("synthetic denied signal")
    calls = signals(case, monkeypatch, on_signal=denied)
    result = invoke(case, apply=True, output=case["output"], legacy_stopped_tree=True)
    assert result["status"] == "BLOCKED" and len(calls) == 1
    assert "STOP_ERROR" in (case["output"] / "events.jsonl").read_text()

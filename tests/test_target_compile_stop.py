"""Only pinned, exact-job process handles receive controlled stop signals."""
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import time

import pytest

from qwen35_moe_dflash import target_compile_stop as stop


def process(case, pid, ppid, *, argv=None, cwd=None, start=None, state="S"):
    folder = case["proc"] / str(pid)
    folder.mkdir(exist_ok=True)
    fields = [state, str(ppid)] + ["0"] * 17 + [str(start or pid * 10)]
    (folder / "stat").write_text(f"{pid} (worker with spaces) " + " ".join(fields))
    (folder / "cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in (argv or ["python3", "worker"])) + b"\0")
    link = folder / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd or case["cwd"])


@pytest.fixture
def case(tmp_path, monkeypatch):
    run, proc = tmp_path / "run", tmp_path / "proc"
    run.mkdir(); proc.mkdir()
    job = run / "target-pytorch-prefill64-compile"
    cwd = job / "compilation/compile-work/generated"
    cwd.mkdir(parents=True)
    argv = ["/selected/ddk/tools/tools_omg/master/omg", "--framework=1", "--model=" + str(cwd / "export.air"),
            "--output=model", "--target=omc", "--platform=kirinx90"]
    command = job / "compilation/compile-command.json"
    command.write_text(json.dumps({"argv": argv, "cwd": str(cwd)}))
    result = {"run": run, "proc": proc, "job": job, "cwd": str(cwd), "argv": argv,
              "command": command, "output": run / "stop-receipt"}
    process(result, 50, 1, argv=["python3", "compile_full_target.py"])
    process(result, 100, 50, argv=argv)
    process(result, 110, 100)
    process(result, 111, 100)
    process(result, 112, 110)
    process(result, 200, 1, argv=["python3", "unrelated"])
    monkeypatch.setenv("AI_RUN_DIR", str(run))
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("numeric PID kill is forbidden"))
    monkeypatch.setattr(os, "killpg", lambda *a: pytest.fail("process group control is forbidden"))
    return result


def invoke(case, **kwargs):
    return stop.stop_target_compile(case["job"], proc_root=case["proc"], **kwargs)


def pinned_signals(case, monkeypatch, *, after_pin=None, on_signal=None):
    descriptors, calls = {}, []
    def pin(pid, flags):
        fd = os.open("/dev/null", os.O_RDONLY)
        descriptors[fd] = pid
        if after_pin:
            after_pin(pid)
        return fd
    def send(fd, sig, info, flags):
        pid = descriptors[fd]
        assert (case["output"] / "plan.json").is_file()
        journal = (case["output"] / "events.jsonl").read_text()
        assert '"SIGNAL_INTENT"' in journal and f'"pid": {pid}' in journal
        calls.append((pid, sig))
        if on_signal:
            on_signal(pid)
    monkeypatch.setattr(os, "pidfd_open", pin)
    monkeypatch.setattr(signal, "pidfd_send_signal", send)
    return calls


def test_plan_matches_exact_job_and_never_opens_process_handles_or_writes(case, monkeypatch):
    monkeypatch.setattr(os, "pidfd_open", lambda *a: pytest.fail("plan is read-only"))
    result = invoke(case)
    assert result["status"] == "PLAN_READY"
    assert [row["pid"] for row in result["selected_processes"]] == [112, 110, 111, 100]
    assert result["driver_pid"] == 50 and result["driver_process"] == "NOT_SIGNALED"
    assert not case["output"].exists()


def test_apply_journals_before_each_pidfd_signal_and_never_targets_wrapper_or_other_process(case, monkeypatch):
    calls = pinned_signals(case, monkeypatch)
    before = {p: p.read_bytes() for p in case["job"].rglob("*") if p.is_file()}
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "STOP_REQUESTED"
    assert calls == [(pid, signal.SIGTERM) for pid in [112, 110, 111, 100]]
    assert result["final_compiler_report"] == "NOT_VERIFIED"
    saved = json.loads((case["output"] / "report.json").read_text())
    assert saved == result
    events = [json.loads(line) for line in (case["output"] / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == ["SIGNAL_INTENT", "SIGTERM_SENT"] * 4
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("kind", ["argv", "cwd", "gone"])
def test_nonmatching_or_completed_job_is_not_signaled(case, monkeypatch, kind):
    if kind == "argv":
        process(case, 100, 50, argv=case["argv"] + ["--different=true"])
    elif kind == "cwd":
        process(case, 100, 50, argv=case["argv"], cwd=str(case["run"]))
    else:
        process(case, 100, 50, argv=case["argv"], state="Z")
    monkeypatch.setattr(os, "pidfd_open", lambda *a: pytest.fail("no match cannot pin any PID"))
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "NOT_RUNNING" and not case["output"].exists()


def test_multiple_exact_compiler_processes_refuse_all_signals(case, monkeypatch):
    process(case, 101, 50, argv=case["argv"])
    monkeypatch.setattr(os, "pidfd_open", lambda *a: pytest.fail("ambiguous cannot pin"))
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "BLOCKED" and "multiple exact" in result["error"]
    assert not case["output"].exists()


def test_pid_reuse_while_pinning_blocks_before_any_signal(case, monkeypatch):
    def changed(pid):
        if pid == 110:
            process(case, pid, 100, start=999999)
    calls = pinned_signals(case, monkeypatch, after_pin=changed)
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "BLOCKED" and "identity changed" in result["error"]
    assert not calls and not case["output"].exists()


def test_changed_command_receipt_blocks_before_any_signal(case, monkeypatch):
    def changed(pid):
        if pid == 100:
            case["command"].write_text(case["command"].read_text() + "\n")
    calls = pinned_signals(case, monkeypatch, after_pin=changed)
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "BLOCKED" and "command changed" in result["error"]
    assert not calls and not case["output"].exists()


def test_pid_reuse_after_first_signal_is_logged_and_not_signaled(case, monkeypatch):
    def changed(pid):
        if pid == 112:
            process(case, 110, 100, start=999999)
    calls = pinned_signals(case, monkeypatch, on_signal=changed)
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "STOP_REQUESTED"
    assert (110, signal.SIGTERM) not in calls
    assert any(row["event"] == "IDENTITY_CHANGED_NOT_SIGNALED" for row in result["events"])


def test_naturally_exited_omg_after_child_signal_is_logged_without_signaling_new_pid(case, monkeypatch):
    def changed(pid):
        if pid == 112:
            (case["proc"] / "100/stat").unlink()
    calls = pinned_signals(case, monkeypatch, on_signal=changed)
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "STOP_REQUESTED"
    assert (100, signal.SIGTERM) not in calls
    assert result["events"][-1]["event"] == "ALREADY_EXITED"


@pytest.mark.parametrize("kind", ["outside_run", "inside_job", "existing_output", "missing_output"])
def test_receipt_path_errors_do_not_signal_any_process(case, monkeypatch, kind):
    calls = pinned_signals(case, monkeypatch)
    if kind == "outside_run": target = case["run"].parent / "outside"
    elif kind == "inside_job": target = case["job"] / "stop"
    elif kind == "existing_output":
        target = case["output"]
        target.mkdir(); (target / "retained").write_text("must remain")
    else: target = None
    result = invoke(case, apply=True, output=target)
    assert result["status"] == "BLOCKED" and calls == []
    if kind == "existing_output": assert (target / "retained").read_text() == "must remain"


def test_missing_pidfd_support_never_falls_back_to_numeric_pid_signal(case, monkeypatch):
    monkeypatch.delattr(signal, "pidfd_send_signal")
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "BLOCKED" and "pidfd support" in result["error"]
    assert not case["output"].exists()


@pytest.mark.parametrize("kind", ["other_model", "other_cwd", "not_omg", "duplicate_key", "missing_target"])
def test_unbound_command_refuses_all_process_control(case, monkeypatch, kind):
    value = json.loads(case["command"].read_text())
    if kind == "other_model": value["argv"][2] = "--model=/another/job/export.air"
    elif kind == "other_cwd": value["cwd"] = str(case["run"])
    elif kind == "not_omg": value["argv"][0] = "/bin/sh"
    elif kind == "missing_target": value["argv"].remove("--target=omc")
    if kind == "duplicate_key": case["command"].write_text('{"argv":[],"argv":[]}')
    else: case["command"].write_text(json.dumps(value))
    monkeypatch.setattr(os, "pidfd_open", lambda *a: pytest.fail("invalid command cannot pin"))
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "BLOCKED" and not case["output"].exists()


def test_signal_failure_retains_plan_and_journal_without_escalating(case, monkeypatch):
    def denied(pid):
        raise PermissionError("synthetic denial")
    calls = pinned_signals(case, monkeypatch, on_signal=denied)
    result = invoke(case, apply=True, output=case["output"])
    assert result["status"] == "BLOCKED" and len(calls) == 1
    assert (case["output"] / "plan.json").is_file()
    assert '"STOP_ERROR"' in (case["output"] / "events.jsonl").read_text()
    assert json.loads((case["output"] / "report.json").read_text())["status"] == "BLOCKED"


def test_genuine_pidfds_stop_owned_host_tree_and_preserve_driver_and_input(case):
    """Real host SIGTERM, restricted to two processes created by this test."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("host Python has no Linux pidfd API")
    try:
        probe = os.pidfd_open(os.getpid(), 0)
    except OSError as error:
        pytest.skip(f"host kernel cannot open a pidfd: {error}")
    os.close(probe)
    child_record = case["run"] / "owned-child-pid"
    owned_input = Path(case["cwd"]) / "source-weight.bin"
    owned_input.write_bytes(b"source must remain unchanged")
    before = owned_input.stat()
    code = ("import subprocess,time\nfrom pathlib import Path\n"
        f"p=subprocess.Popen([{sys.executable!r},'-c','import time; time.sleep(60)'])\n"
        f"Path({str(child_record)!r}).write_text(str(p.pid))\n"
        "time.sleep(60)\n")
    argv = [case["argv"][0], "-c", code, *case["argv"][1:]]
    case["command"].write_text(json.dumps({"argv": argv, "cwd": case["cwd"]}))
    driver = os.getpid()
    root = subprocess.Popen(argv, executable=sys.executable, cwd=case["cwd"])
    handles = []
    try:
        handles.append(os.pidfd_open(root.pid, 0))
        deadline = time.monotonic() + 5
        while not child_record.exists() and root.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_record.exists(), f"owned mock OMG exited unexpectedly: {root.poll()}"
        child_pid = int(child_record.read_text())
        handles.append(os.pidfd_open(child_pid, 0))
        result = stop.stop_target_compile(case["job"], apply=True, output=case["output"])
        assert result["status"] == "STOP_REQUESTED", result
        assert result["driver_pid"] == driver == os.getpid()
        assert {row["pid"] for row in result["selected_processes"]} == {root.pid, child_pid}
        assert root.wait(timeout=5) == -signal.SIGTERM
        poller = select.poll()
        poller.register(handles[1], select.POLLIN)
        assert poller.poll(5000), "owned child did not exit after SIGTERM"
        assert owned_input.read_bytes() == b"source must remain unchanged"
        assert owned_input.stat() == before
        assert json.loads((case["output"] / "plan.json").read_text())["status"] == "APPLY_READY"
    finally:
        # Handle-pinned cleanup of this test's own two processes only.
        for handle in reversed(handles):
            try:
                signal.pidfd_send_signal(handle, signal.SIGTERM, None, 0)
            except ProcessLookupError:
                pass
            os.close(handle)
        root.wait(timeout=5)

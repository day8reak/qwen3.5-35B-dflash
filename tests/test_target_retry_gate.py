"""Synthetic small receipts test a read-only retry gate, not a model or build."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import target_retry_gate as gate
from qwen35_moe_dflash.tbe_tmpdir_probe import _CHILD, _SOURCE, _predicate
from qwen35_moe_dflash.compile import _OMG_IPC_MULTIPROCESSING
from qwen35_moe_dflash.target_export_reuse import _read_small
from test_tbe_tmpdir_probe import _VENDOR_SOURCE


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    run = tmp_path / "run"; run.mkdir()
    monkeypatch.setenv("AI_RUN_DIR", str(run))
    job, probe, stopped, proc = [run / name for name in ("static-failed", "probe", "stop", "fake-proc")]
    for path in (job, probe, stopped, proc): path.mkdir()
    comp = job / "compilation"; comp.mkdir()
    source = comp / _SOURCE; source.parent.mkdir(parents=True); source.write_text(_VENDOR_SOURCE)
    command = {"argv": ["/synthetic/ddk/omg"], "cwd": str(comp / "compile-work/generated")}
    write(comp / "compile-command.json", command)
    write(comp / "omg-environment.json", {"environment": {"TMPDIR": "/proc/old/fd/4/.ipc-old"}})
    audit = {"status": "PASSED", "scope": "synthetic test receipt; owner checks have separate tests"}
    compiled = {"status": "FAILED", "model_compilation": "FAILED", "retained_input_reuse": audit,
        "source_preservation": {"status": "UNCHANGED_BY_STAT", "changed": []}}
    write(comp / "retained-input-reuse.json", audit); write(comp / "report.json", compiled)
    outer = {"artifact_kind": "qwen35-original-target-job", "status": "FAILED",
        "compilation_report_sha256": hashlib.sha256((comp / "report.json").read_bytes()).hexdigest()}
    write(job / "result.json", outer)
    physical, short = str(probe / ".ipc-fixture"), "/proc/42424/fd/4/.ipc-fixture"
    overlay_sha = hashlib.sha256(_OMG_IPC_MULTIPROCESSING.replace("__QWEN35_IPC_PATHS__", repr((physical, short))).encode()).hexdigest()
    overlay = {"path": str(Path(physical) / "python-ipc/multiprocessing/__init__.py"), "sha256": overlay_sha,
        "stdlib_path": "/synthetic/stdlib/multiprocessing/__init__.py", "stdlib_sha256": "a" * 64,
        "physical_tmpdir": physical, "multiprocessing_tmpdir": short}
    child = {"status": "PROBE_PASSED", "old_alias_control": "REJECTED_AS_EXPECTED", "physical_tmpdir_predicate": "PASSED",
        "vendor_package_import": "NOT_PERFORMED", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "embedded_omg_startup": "NOT_TESTED", "native_cce_compilation": "NOT_RUN",
        "forkserver_socket": {"status": "PASSED", "path": short + "/listener-fixture", "bytes": len(short + "/listener-fixture")},
        "physical_tmpdir": physical, "multiprocessing_tmpdir": short,
        "multiprocessing_overlay": overlay, "multiprocessing_module": {"path": overlay["stdlib_path"],
            "spec_origin": overlay["stdlib_path"], "sha256": overlay["stdlib_sha256"]},
        "ipc_configuration": "BOUND_SHORT_ALIAS_OBSERVED_IN_ACTUAL_PROCESS_CONFIG",
        "controlled_build_flags": {"debug": False, "dump": False, "random_cce_file_location": True}}
    predicate = _predicate(source.read_bytes())
    (probe / "validator-source.py").write_text(predicate); (probe / "probe-child.py").write_text(_CHILD)
    write(probe / "child-result.json", child)
    write(probe / "command.json", {"argv": ["/synthetic/python", str(probe / "probe-child.py"), str(probe / "child-result.json"),
        str(probe / "validator-source.py"), short], "cwd": str(probe), "timeout_seconds": 30})
    report = {"artifact_kind": "qwen35-tbe-tmpdir-probe", "status": "PROBE_PASSED", "returncode": 0,
        "weight_payload_access": "NONE", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "embedded_omg_startup": "NOT_TESTED", "input_mutation": "NONE", "job": str(job), "child": child,
        "source": _read_small(source, 1 << 20)[1],
        "input_command": _read_small(comp / "compile-command.json", 1 << 20)[1],
        "input_environment": _read_small(comp / "omg-environment.json", 1 << 20)[1],
        "source_predicate_sha256": hashlib.sha256(predicate.encode()).hexdigest(),
        "probe_environment": {"path": short, "physical_path": physical, "compiler_tmpdir": physical,
            "multiprocessing_tmpdir": short, "multiprocessing_overlay": {"path": overlay["path"], "sha256": overlay_sha}}}
    write(probe / "report.json", report)
    selected = [{"pid": 101, "ppid": 100, "start_ticks": 1010, "state": "T", "depth": 1},
                {"pid": 100, "ppid": 50, "start_ticks": 1000, "state": "T", "depth": 0}]
    driver = {"pid": 50, "ppid": 1, "start_ticks": 500, "state": "T", "argv": ["python3", "/fixture/compile_full_target.py"]}
    events = [{"event": sig + "_SENT", "pid": row["pid"], "start_ticks": row["start_ticks"], "signal": sig}
              for sig, rows in (("SIGTERM", selected), ("SIGCONT", selected + [driver])) for row in rows]
    stop = {"artifact_kind": "qwen35-target-compile-stop", "status": "STOP_REQUESTED", "job": str(job),
        "command": report["input_command"], "mode": "APPLY", "signal_backend": "EXPLICIT_LEGACY_STOPPED_TREE",
        "selected_processes": selected, "bound_driver": driver, "omg_pid": 100, "driver_pid": 50, "events": events}
    data = {"job": job, "comp": comp, "probe": probe, "stopped": stopped, "proc": proc,
        "report": report, "child": child, "stop": stop, "compiled": compiled, "outer": outer}
    save_stop(data)
    # Deliberately unreadable sentinel path: metadata must not select payloads.
    (job / "large-weight-must-not-open").write_bytes(b"retained model payload sentinel")
    return data


def save_stop(case):
    stop = case["stop"]
    write(case["stopped"] / "report.json", stop)
    write(case["stopped"] / "plan.json", dict(stop, status="APPLY_READY", events=[]))
    journal = []
    for event in stop["events"]:
        journal += [dict(event, event="SIGNAL_INTENT"), event]
    (case["stopped"] / "events.jsonl").write_text("\n".join(json.dumps(row) for row in journal) + "\n")


def save_probe(case):
    case["report"]["child"] = deepcopy(case["child"])
    write(case["probe"] / "child-result.json", case["child"])
    write(case["probe"] / "report.json", case["report"])


def invoke(case):
    return gate.validate_static_retry_gate(case["job"], tmpdir_probe=case["probe"], stop_receipt=case["stopped"], proc_root=case["proc"])


def test_complete_evidence_is_checked_without_writes_payloads_or_process_control(case, monkeypatch):
    before = {p: p.read_bytes() for root in (case["job"], case["probe"], case["stopped"]) for p in root.rglob("*") if p.is_file()}
    original = os.open
    def readonly(path, flags, *args, **kwargs):
        assert "large-weight-must-not-open" not in str(path)
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", readonly)
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("preflight cannot signal a process"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("preflight cannot spawn a process"))
    result = invoke(case)
    assert result["status"] == "PREFLIGHT_PASSED" and result["weight_payload_access"] == "NONE"
    assert result["model_compilation"] == result["device_execution"] == "NOT_RUN"
    assert len(result["recorded_processes"]) == 3 and result["metadata_bindings"]
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("kind", ["outer_running", "compiled_success", "audit_failed", "probe_failed", "probe_job",
    "child_mismatch", "old_control", "new_control", "debug_relaxed", "overlay_sha", "overlay_path", "module_sha",
    "source_changed", "script_changed", "validator_changed", "probe_command", "input_command", "stop_failed",
    "stop_command", "stop_journal", "stop_plan", "driver_term", "unrelated_signal", "missing_target",
    "native_claim", "bad_environment", "bad_overlay", "bad_socket"])
def test_invalid_or_incomplete_evidence_blocks_before_payloads(case, kind):
    comp, probe, stopped = case["comp"], case["probe"], case["stopped"]
    if kind == "outer_running":
        case["outer"]["status"] = "PREPARING"; write(case["job"] / "result.json", case["outer"])
    elif kind == "compiled_success":
        case["compiled"]["status"] = "COMPILED"; write(comp / "report.json", case["compiled"])
    elif kind == "audit_failed": write(comp / "retained-input-reuse.json", {"status": "FAILED"})
    elif kind == "probe_failed": case["report"]["status"] = "BLOCKED"; save_probe(case)
    elif kind == "probe_job": case["report"]["job"] += "-wrong"; save_probe(case)
    elif kind == "child_mismatch": write(probe / "child-result.json", {"status": "PROBE_PASSED"})
    elif kind in {"old_control", "new_control", "debug_relaxed", "overlay_sha", "overlay_path", "module_sha"}:
        if kind == "old_control": case["child"]["old_alias_control"] = "NOT_RUN"
        elif kind == "new_control": case["child"]["physical_tmpdir_predicate"] = "FAILED"
        elif kind == "debug_relaxed": case["child"]["controlled_build_flags"]["debug"] = True
        elif kind == "overlay_sha": case["child"]["multiprocessing_overlay"]["sha256"] = "b" * 64
        elif kind == "overlay_path": case["child"]["multiprocessing_overlay"]["path"] = "/outside/not-opened"
        else: case["child"]["multiprocessing_module"]["sha256"] = "b" * 64
        save_probe(case)
    elif kind == "source_changed": (comp / _SOURCE).write_bytes((comp / _SOURCE).read_bytes() + b"\n")
    elif kind == "script_changed": (probe / "probe-child.py").write_bytes(_CHILD.encode() + b"\n")
    elif kind == "validator_changed": (probe / "validator-source.py").write_text("raise RuntimeError('changed')\n")
    elif kind == "probe_command": write(probe / "command.json", {"argv": ["/outside/interpreter"], "cwd": str(probe)})
    elif kind == "input_command": case["report"]["input_command"]["sha256"] = "b" * 64; save_probe(case)
    elif kind == "native_claim": case["child"]["native_cce_compilation"] = "PASSED"; save_probe(case)
    elif kind == "bad_environment": case["report"]["probe_environment"] = None; save_probe(case)
    elif kind == "bad_overlay": case["child"]["multiprocessing_overlay"] = None; save_probe(case)
    elif kind == "bad_socket": case["child"]["forkserver_socket"] = []; save_probe(case)
    elif kind in {"stop_failed", "stop_command", "driver_term", "unrelated_signal", "missing_target"}:
        if kind == "stop_failed": case["stop"]["status"] = "BLOCKED"
        elif kind == "stop_command": case["stop"]["command"] = {"sha256": "b" * 64}
        elif kind == "driver_term": case["stop"]["events"][-1].update(signal="SIGTERM", event="SIGTERM_SENT")
        elif kind == "unrelated_signal": case["stop"]["events"][-1]["pid"] = 999
        else: case["stop"]["events"] = [row for row in case["stop"]["events"] if row["pid"] != 101]
        save_stop(case)
    elif kind == "stop_journal": (stopped / "events.jsonl").write_text("{}\n")
    elif kind == "stop_plan": write(stopped / "plan.json", {"status": "APPLY_READY"})
    with pytest.raises((ValueError, OSError)):
        invoke(case)


def proc_stat(case, pid, start, state="S"):
    folder = case["proc"] / str(pid); folder.mkdir()
    fields = [state, "1"] + ["0"] * 17 + [str(start)]
    (folder / "stat").write_text(str(pid) + " (name with spaces) " + " ".join(fields))


@pytest.mark.parametrize("pid,start", [(100,1000), (101,1010), (50,500)])
def test_still_present_compiler_worker_or_driver_blocks(case, pid, start):
    proc_stat(case, pid, start)
    with pytest.raises(ValueError, match="still present"):
        invoke(case)


@pytest.mark.parametrize("state,start,expected", [("S", 99999, "PID_REUSED_OTHER_GENERATION"), ("Z", 1000, "EXITED_ZOMBIE")])
def test_reused_pids_and_exited_zombies_are_not_live_old_workers(case, state, start, expected):
    proc_stat(case, 100, start, state)
    result = invoke(case)
    assert result["status"] == "PREFLIGHT_PASSED"
    assert next(row for row in result["recorded_processes"] if row["pid"] == 100)["status"] == expected


def test_source_and_receipt_symlinks_cannot_escape_fixed_paths(case):
    target = case["probe"] / "probe-child.py"
    outside = case["probe"].parent / "outside.py"; outside.write_text(_CHILD)
    target.unlink(); target.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        invoke(case)


def test_final_failed_report_must_exist_before_any_other_check(case):
    (case["job"] / "result.json").unlink()
    with pytest.raises(FileNotFoundError):
        invoke(case)

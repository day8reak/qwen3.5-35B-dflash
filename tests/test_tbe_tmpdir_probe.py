"""Bounded source-only CCE predicate and real host IPC probe regressions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import tbe_tmpdir_probe as probe


_VENDOR_SOURCE = '''raise RuntimeError("whole vendor module must never execute")
def _check_tmpdir(dirpath):
    from tbe.common.buildcfg.buildcfg_mapping import tbe_debug_level, op_debug_config
    from tbe.common.buildcfg.buildcfg_mapping import random_cce_file_location
    if current_build_config().get(tbe_debug_level) or "dump" in current_build_config().get(op_debug_config) or not current_build_config().get(random_cce_file_location):
        return
    if sys.platform.startswith("linux"):
        tmp_dir = os.getenv("TMPDIR")
        if not (dirpath.startswith("/tmp") or dirpath.startswith("/var/tmp") or dirpath.startswith("/usr/tmp") or dirpath.startswith(os.path.realpath("/tmp")) or dirpath.startswith(os.path.realpath("/var/tmp")) or dirpath.startswith(os.path.realpath("/usr/tmp")) or (tmp_dir is not None and dirpath.startswith(tmp_dir))):
            raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "dir path is invalid")
    else:
        raise_tbe_python_err(TBE_DEFAULT_PYTHON_ERROR_CODE, "unsupported test platform")

def compile_cce(dirpath):
    dirpath = os.path.realpath(dirpath)
    _check_tmpdir(dirpath)
    raise RuntimeError("native build must never execute")
'''


@pytest.fixture
def case(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setenv("AI_RUN_DIR", str(run))
    job = run / "retained-static64"
    comp = job / "compilation"
    comp.mkdir(parents=True)
    (comp / "omg-environment.json").write_text(json.dumps({"environment": {"TMPDIR": "/proc/old/fd/4/.ipc-old"}}))
    (comp / "compile-command.json").write_text(json.dumps({"argv": ["/selected/ddk/omg"], "cwd": str(comp / "compile-work/generated")}))
    source = comp / probe._SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(_VENDOR_SOURCE)
    unrelated = job / "large-weight-must-not-open"
    unrelated.write_bytes(b"no payload reads")
    output = run / ("deep-owned-probe-" * 6) / "probe"
    return {"job": job, "source": source, "output": output,
            "hash": hashlib.sha256(source.read_bytes()).hexdigest(), "weight": unrelated}


def invoke(case, **kwargs):
    return probe.probe_tbe_tmpdir(case["job"], case["output"], expected_ccec_sha256=case["hash"], **kwargs)


def test_actual_child_runs_only_extracted_predicate_and_forkserver_ipc(case, monkeypatch):
    before = {path: path.read_bytes() for path in case["job"].rglob("*") if path.is_file()}
    original = os.open
    def guarded(path, *args, **kwargs):
        assert "large-weight-must-not-open" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", guarded)
    result = invoke(case)
    standard = ("/tmp", "/var/tmp", "/usr/tmp")
    if str(case["output"]).startswith(standard):
        # This vendor accepts standard temp roots independently of TMPDIR, so
        # it cannot reproduce the observed /home-prefix failure in that root.
        assert result["status"] == "BLOCKED"
        assert "old alias control did not reproduce" in result["child"]["error"]
        return
    assert result["status"] == "PROBE_PASSED", result
    assert result["weight_payload_access"] == "NONE"
    assert result["model_compilation"] == result["device_execution"] == "NOT_RUN"
    assert result["embedded_omg_startup"] == "NOT_TESTED"
    child = result["child"]
    assert child["old_alias_control"] == "REJECTED_AS_EXPECTED"
    assert child["physical_tmpdir_predicate"] == "PASSED"
    assert child["forkserver_socket"]["status"] == "PASSED"
    assert child["multiprocessing_module"]["path"]
    assert child["multiprocessing_overlay"]["sha256"] == result["probe_environment"]["multiprocessing_overlay"]["sha256"]
    assert child["ipc_configuration"] == "BOUND_SHORT_ALIAS_OBSERVED_IN_ACTUAL_PROCESS_CONFIG"
    assert not Path(child["physical_tmpdir"]).exists()
    assert {path: path.read_bytes() for path in before} == before
    assert (case["output"] / "report.json").is_file()


@pytest.mark.parametrize("digest", ["", "bad", "A" * 64, "0" * 64])
def test_expected_source_hash_is_required_before_any_child_or_output(case, monkeypatch, digest):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("bad source hash cannot launch child"))
    result = probe.probe_tbe_tmpdir(case["job"], case["output"], expected_ccec_sha256=digest)
    assert result["status"] == "BLOCKED"
    assert not case["output"].exists()


@pytest.mark.parametrize("change", ["duplicate", "decorator", "signature", "no_canonicalization", "reverse_order"])
def test_unrecognized_source_contract_is_not_executed(case, monkeypatch, change):
    text = _VENDOR_SOURCE
    if change == "duplicate": text += "\ndef _check_tmpdir(dirpath):\n    pass\n"
    elif change == "decorator": text = text.replace("def _check_tmpdir", "@arbitrary_code()\ndef _check_tmpdir")
    elif change == "signature": text = text.replace("def _check_tmpdir(dirpath):", "def _check_tmpdir(dirpath, mode):")
    elif change == "no_canonicalization": text = text.replace("    dirpath = os.path.realpath(dirpath)\n", "")
    else: text = text.replace("    dirpath = os.path.realpath(dirpath)\n    _check_tmpdir(dirpath)", "    _check_tmpdir(dirpath)\n    dirpath = os.path.realpath(dirpath)")
    case["source"].write_text(text)
    case["hash"] = hashlib.sha256(case["source"].read_bytes()).hexdigest()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unrecognized source cannot execute"))
    result = invoke(case)
    assert result["status"] == "BLOCKED"
    assert not case["output"].exists()


def test_existing_output_is_never_overwritten(case):
    case["output"].mkdir(parents=True)
    retained = case["output"] / "retained.json"
    retained.write_bytes(b"preserve this evidence")
    result = invoke(case)
    assert result["status"] == "BLOCKED"
    assert retained.read_bytes() == b"preserve this evidence"
    assert sorted(path.name for path in case["output"].iterdir()) == ["retained.json"]


def test_output_cannot_mutate_retained_job_or_escape_active_run(case):
    for output in (case["job"] / "probe", case["job"].parent.parent / "outside"):
        result = probe.probe_tbe_tmpdir(case["job"], output, expected_ccec_sha256=case["hash"])
        assert result["status"] == "BLOCKED"
        assert not output.exists()


def test_failed_child_result_is_retained_and_cannot_become_probe_pass(case, monkeypatch):
    def fails(argv, **kwargs):
        Path(argv[2]).write_text(json.dumps({"status": "FAILED", "error": "synthetic child failure"}))
        return subprocess.CompletedProcess(argv, 1)
    monkeypatch.setattr(subprocess, "run", fails)
    result = invoke(case)
    assert result["status"] == "BLOCKED"
    assert result["child"]["error"] == "synthetic child failure"
    assert json.loads((case["output"] / "report.json").read_text())["status"] == "BLOCKED"


def test_child_timeout_retains_probe_evidence(case, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    result = invoke(case, timeout_seconds=1)
    assert result["status"] == "BLOCKED" and "TimeoutExpired" in result["error"]
    assert (case["output"] / "command.json").is_file()
    assert (case["output"] / "report.json").is_file()

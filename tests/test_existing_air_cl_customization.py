"""Explicit compiler policy snapshots; mocked OMG only, no device claims."""
import json
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import existing_air as subject
from qwen35_moe_dflash.hashing import sha256_file
from test_existing_air import case, _help, _tool


def _policy(case):
    source = case[1].parent / "placement-source.json"
    source.write_text('{"op_name_list": {"norm_square": "CPUCL"}}\n')
    return source


def _advertised_help(command, **kwargs):
    kwargs["stdout"].write("OMG usage: --cl_customization --save_weights_as_external_data\n")
    return subprocess.CompletedProcess(command, 0)


def _option_path(command, cwd):
    flags = [value for value in command if value.startswith("--cl_customization=")]
    assert flags == ["--cl_customization=../options/cl-customization.json"]
    return (cwd / flags[0].split("=", 1)[1]).resolve()


def test_policy_uses_bound_independent_snapshot_and_stable_relative_option(case, monkeypatch):
    air, output, ddk = case
    policy = _policy(case)
    original = policy.read_bytes()
    invocations = []
    def tool(command, **kwargs):
        invocations.append(command)
        if "--help" in command:
            return _advertised_help(command, **kwargs)
        staged = _option_path(command, kwargs["cwd"])
        assert staged == output / "compile-work/options/cl-customization.json"
        assert staged.read_bytes() == original
        assert staged.stat().st_ino != policy.stat().st_ino
        assert not staged.is_symlink()
        return _tool(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", tool)
    report = subject.compile_existing_air(air, output, ddk, ddk_id="MOCK", cl_customization=policy)
    assert report["status"] == "COMPILED" and len(invocations) == 2
    receipt = report["cl_customization"]
    assert receipt == {"path": "compile-work/options/cl-customization.json",
                       "bytes": len(original), "sha256": sha256_file(policy)}
    assert report["requested_options"] == {"cl_customization": "../options/cl-customization.json"}
    assert report["help_check"]["advertised"]["cl_customization"]
    assert report["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    assert report["source_preservation"]["changed"] == []
    assert all(row["path"] != receipt["path"] for row in report["input_snapshots"])
    assert all(row["path"] != receipt["path"] for row in report["generated_artifacts"])
    assert all(row["path"] != receipt["path"] for row in report["external_data_candidates"])
    bound = next(row for row in report["compiler_work_artifacts"] if row["path"] == receipt["path"])
    assert bound["sha256"] == receipt["sha256"] and bound["role"] == "compiler_context"
    assert json.loads((output / "report.json").read_text())["cl_customization"] == receipt
    assert policy.read_bytes() == original
    assert report["device_execution"] == "NOT_RUN"


def test_policy_option_omitted_from_help_is_warned_and_passed_unchanged(case, monkeypatch):
    air, output, ddk = case
    policy = _policy(case)
    calls, progress = [], []
    def tool(command, **kwargs):
        calls.append(command)
        if "--help" in command:
            return _help(command, **kwargs)
        staged = _option_path(command, kwargs["cwd"])
        assert staged.read_bytes() == policy.read_bytes()
        return _tool(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", tool)
    report = subject.compile_existing_air(air, output, ddk, ddk_id="MOCK",
                                          cl_customization=policy, progress=progress.append)
    assert report["status"] == "COMPILED" and report["model_compilation"] == "PASSED"
    assert report["help_check"]["status"] == "OPTIONS_NOT_ADVERTISED"
    assert report["help_check"]["missing"] == ["cl_customization"]
    assert any(row.get("options") == ["cl_customization"] for row in progress)
    assert report["cl_customization"]["sha256"] == sha256_file(policy)
    assert len(calls) == 2


@pytest.mark.parametrize("data", [b"", b"[]", b"null", b"{broken", b'{"x":NaN}',
                                 b'{"x":Infinity}', b'\xff', b" " * ((1 << 20) + 1)])
def test_invalid_policy_rejected_before_tools_or_output(case, monkeypatch, data):
    air, output, ddk = case
    policy = _policy(case)
    policy.write_bytes(data)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("invalid policy must not invoke OMG"))
    with pytest.raises(ValueError):
        subject.compile_existing_air(air, output, ddk, ddk_id="MOCK", cl_customization=policy)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_nonregular_policy_rejected_before_tools_or_output(case, monkeypatch, kind):
    air, output, ddk = case
    policy = _policy(case)
    candidate = policy.with_name("invalid-policy")
    if kind == "symlink":
        candidate.symlink_to(policy)
    else:
        candidate.mkdir()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("invalid policy must not invoke OMG"))
    with pytest.raises(ValueError, match="regular non-symlink"):
        subject.compile_existing_air(air, output, ddk, ddk_id="MOCK", cl_customization=candidate)
    assert not output.exists()


def test_policy_changed_after_validation_fails_before_bulk_air_copy(case, monkeypatch):
    air, output, ddk = case
    policy = _policy(case)
    def tool(command, **kwargs):
        assert "--help" in command
        policy.write_text('{"different":true}')
        return _advertised_help(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", tool)
    report = subject.compile_existing_air(air, output, ddk, ddk_id="MOCK", cl_customization=policy)
    assert report["status"] == "FAILED" and report["stage"] == "cl-customization-snapshot"
    assert "changed since validation" in report["error"]
    assert report["input_snapshots"] == []
    assert report["model_compilation"] == "NOT_RUN"


@pytest.mark.parametrize("mutated", ["source", "snapshot"])
def test_policy_source_or_snapshot_mutation_rejects_successful_mock_compile(case, monkeypatch, mutated):
    air, output, ddk = case
    policy = _policy(case)
    def tool(command, **kwargs):
        if "--help" in command:
            return _advertised_help(command, **kwargs)
        staged = _option_path(command, kwargs["cwd"])
        result = _tool(command, **kwargs)
        changed = policy if mutated == "source" else staged
        changed.chmod(0o644)
        changed.write_text('{"changed":true}')
        return result
    monkeypatch.setattr(subprocess, "run", tool)
    report = subject.compile_existing_air(air, output, ddk, ddk_id="MOCK", cl_customization=policy)
    assert report["status"] == "FAILED" and report["stage"] == "input-recheck"
    assert report["source_preservation"]["status"] == "CHANGED"
    assert "cl_customization " + mutated in report["source_preservation"]["changed"]
    assert not (output / "manifest.json").exists()
    assert report["device_execution"] == "NOT_RUN"

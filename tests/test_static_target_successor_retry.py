"""Real tiny ownership receipts across two retries; OMG execution is mocked.

These checks exercise source inventory, staging, compiler orchestration and
failed-attempt audits. They establish host ownership only, never model accuracy,
native CCE compilation or an embedded Python startup result.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from qwen35_moe_dflash import existing_air as compiler
from qwen35_moe_dflash import full_target
from qwen35_moe_dflash.artifacts import save_json
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from test_static_target_retry import case as original_case_fixture


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


def _metadata_bytes(folder):
    return {path: path.read_bytes() for path in folder.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".txt"}}


@pytest.fixture
def chain(tmp_path, monkeypatch):
    original = original_case_fixture.__wrapped__(tmp_path, monkeypatch)
    failed_static = original["run"] / "target-pytorch-prefill64-compile"
    calls = []

    def terminated_static(argv, **kwargs):
        if "--help" in argv:
            kwargs["stdout"].write("OMG --input_shape --dynamic_dims --limit_partition_size --save_weights_as_external_data\n")
            return subprocess.CompletedProcess(argv, 0)
        calls.append(argv)
        assert not any(arg.startswith("--dynamic_dims") for arg in argv)
        assert "-1" not in next(arg for arg in argv if arg.startswith("--input_shape="))
        kwargs["stdout"].write("TBE _check_tmpdir: dir path is invalid; host simulation only\n")
        raise subprocess.CalledProcessError(-15, argv)

    monkeypatch.setattr(subprocess, "run", terminated_static)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/compile_full_target.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setattr(sys, "argv", [str(script),
        "--retry-static-prefill-from", str(original["failed"]),
        "--profile", "cpro-40block-prefill-64", "--input-layout", "air-directory",
        "--target-features", "--skip-inspection", "--output", str(failed_static)])
    assert main() == 1
    assert len(calls) == 1
    report = json.loads((failed_static / "compilation/report.json").read_text())
    outer = json.loads((failed_static / "result.json").read_text())
    audit = json.loads((failed_static / "compilation/retained-input-reuse.json").read_text())
    assert report["status"] == outer["status"] == "FAILED"
    assert report["returncode"] == -15 and report["stage"] == "omg"
    assert report["retained_input_reuse"] == audit and audit["status"] == "PASSED"
    assert outer["retained_failed_job"] == str(original["failed"])
    assert outer["input_snapshot_reuse"]["status"] == "PASSED"
    assert outer["compilation_report_sha256"] == _sha(failed_static / "compilation/report.json")
    assert report["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    weights = [row for row in report["input_snapshots"] if row["role"] != "air"]
    assert all(row["snapshot_method"] == "SHARED_READONLY_SNAPSHOT_INODE" for row in weights)
    assert all((failed_static / "compilation" / row["path"]).stat().st_nlink == 2 for row in weights)
    return dict(original, previous=failed_static, previous_report=report,
                previous_outer=outer, previous_audit=audit, weights=weights)


def _successor(chain, monkeypatch, *, profile_name="cpro-40block-prefill-64", air=None):
    profile = get_existing_air_profile(profile_name)
    job = chain["run"] / "target-pytorch-prefill64-tmpdirfix"
    job.mkdir()
    save_json(job / "compile-profile.json", profile)
    normal_snapshot = compiler._snapshot
    copied, invocations, observed = [], [], []

    def air_copy_only(source, destination):
        assert source == chain["air"], "successor must not make another weight payload copy"
        copied.append(str(source))
        return normal_snapshot(source, destination)

    monkeypatch.setattr(compiler, "_snapshot", air_copy_only)

    def current_static(argv, **kwargs):
        invocations.append(argv)
        if "--help" in argv:
            kwargs["stdout"].write("OMG --input_shape --dynamic_dims --limit_partition_size --save_weights_as_external_data\n")
            return subprocess.CompletedProcess(argv, 0)
        cwd, env = kwargs["cwd"], kwargs["env"]
        assert cwd == job / "compilation/compile-work/generated"
        assert not any(arg.startswith("--dynamic_dims") for arg in argv)
        assert "-1" not in next(arg for arg in argv if arg.startswith("--input_shape="))
        physical = Path(env["TMPDIR"])
        assert str(physical.resolve()) == env["TMPDIR"]
        assert env["TMP"] == env["TEMP"] == str(physical)
        overlay_root = Path(env["PYTHONPATH"].split(os.pathsep)[0])
        assert overlay_root.is_relative_to(physical)
        assert (overlay_root / "multiprocessing/__init__.py").is_file()
        for row in chain["weights"]:
            ancestor = chain["failed"] / "compilation" / row["path"]
            previous = chain["previous"] / "compilation" / row["path"]
            current = job / "compilation" / row["path"]
            assert ancestor.samefile(previous) and previous.samefile(current)
            assert current.stat().st_nlink == 3
            assert not current.stat().st_mode & 0o222
            assert current.stat().st_ino != Path(row["source"]).stat().st_ino
            assert _sha(current) == row["sha256"]
            observed.append(current)
        air_row = next(row for row in chain["previous_report"]["input_snapshots"] if row["role"] == "air")
        air_paths = [chain["air"], chain["failed"] / "compilation" / air_row["path"],
                     chain["previous"] / "compilation" / air_row["path"], cwd / "export.air"]
        assert len({(path.stat().st_dev, path.stat().st_ino) for path in air_paths}) == 4
        assert len({_sha(path) for path in air_paths}) == 1
        model = cwd / "model"
        model.mkdir()
        (model / "model.omc").write_bytes(b"mock corrected static64 OMC")
        (model / "SubGraph_0.weight").write_bytes(b"mock current compiled external data")
        kwargs["stdout"].write("OMG host simulation complete; no native compiler executed\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", current_static)
    report = compiler.compile_existing_air(air or chain["air"], job / "compilation", chain["ddk"],
        ddk_id="STATIC_RETRY_INTEGRATION_MOCK", input_shape=profile["input_shape"],
        dynamic_dims=profile["dynamic_dims"], limit_partition_size=profile["limit_partition_size"],
        save_weights_as_external_data=profile["save_weights_as_external_data"], input_layout="air-directory",
        reuse_failed_job=chain["previous"])
    return job, profile, report, copied, invocations, observed


def test_audited_static_successor_preserves_owner_receipts_and_only_copies_air(chain, monkeypatch):
    original_metadata = {**_metadata_bytes(chain["failed"]), **_metadata_bytes(chain["previous"])}
    source_before = {Path(row["source"]): _stat(Path(row["source"]))
                     for row in chain["previous_report"]["input_snapshots"]}
    source_bytes = {path: path.read_bytes() for path in source_before}
    job, profile, report, copied, calls, observed = _successor(chain, monkeypatch)
    assert report["status"] == "COMPILED", report.get("error")
    assert len(calls) == 2 and copied == [str(chain["air"])] and len(observed) == 2
    audit = report["retained_input_reuse"]
    assert audit["status"] == "PASSED" and audit["explicit_weight_copy_bytes"] == 0
    assert audit["reuse_lineage"] == "AUDITED_STATIC_SUCCESSOR"
    assert report["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    assert {path: path.read_bytes() for path in original_metadata} == original_metadata
    assert {path: _stat(path) for path in source_before} == source_before
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes
    events = [json.loads(line) for line in Path(audit["journal"]["path"]).read_text().splitlines()]
    links = [row for row in events if row["event"] == "LINK_CREATED"]
    assert len(links) == 2 and all(row["after"]["nlink"] == 3 for row in links)
    assert events[-1]["event"] == "AUDIT" and events[-1]["status"] == "PASSED"
    save_json(job / "result.json", {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED",
        "model_compilation": "PASSED", "air": str(chain["air"]), "output": str(job),
        "input_layout": "air-directory", "compilation": "compilation/report.json",
        "compilation_report_sha256": _sha(job / "compilation/report.json")})
    validated = full_target.validate_compilation(job, air=chain["air"], ddk_root=chain["ddk"],
        ddk_id="STATIC_RETRY_INTEGRATION_MOCK", profile=profile, input_layout="air-directory", target_features=True)
    assert validated["status"] == "COMPILED" and validated["device_execution"] == "NOT_RUN"


@pytest.mark.parametrize("kind", ["missing_final_result", "audit", "journal", "ancestor_report", "chmod", "extra_link", "payload_write"])
def test_unbound_prior_static_evidence_stops_before_omg_and_preserves_the_failure(chain, monkeypatch, kind):
    previous = chain["previous"] / "compilation"
    if kind == "missing_final_result":
        (chain["previous"] / "result.json").unlink()
    elif kind == "audit":
        path = previous / "retained-input-reuse.json"
        value = json.loads(path.read_text()); value["unbound_edit"] = True
        path.write_text(json.dumps(value))
    elif kind == "journal":
        with (previous / "retained-input-links.jsonl").open("a") as stream:
            stream.write('{"event":"UNBOUND_APPEND"}\n')
    elif kind == "ancestor_report":
        path = chain["failed"] / "compilation/report.json"
        value = json.loads(path.read_text()); value["unbound_edit"] = True
        path.write_text(json.dumps(value))
    else:
        target = previous / chain["weights"][0]["path"]
        if kind == "chmod":
            target.chmod(0o644)
        elif kind == "extra_link":
            os.link(target, chain["run"] / "unexpected-extra-alias")
        else:
            target.chmod(0o644)
            target.write_bytes(b"privileged external shared-inode rewrite")
    metadata_before = {**_metadata_bytes(chain["failed"]), **_metadata_bytes(chain["previous"])}
    sources = {Path(row["source"]): (Path(row["source"]).read_bytes(), _stat(Path(row["source"])))
               for row in chain["previous_report"]["input_snapshots"]}
    job, _, report, copied, calls, _ = _successor(chain, monkeypatch)
    assert report["status"] == "FAILED"
    assert calls == copied == []
    assert not report["input_snapshots"]
    assert not (job / "compilation/retained-input-links.jsonl").exists()
    assert {path: path.read_bytes() for path in metadata_before} == metadata_before
    assert {path: (path.read_bytes(), _stat(path)) for path in sources} == sources
    if kind in {"chmod", "payload_write"}:
        assert target.stat().st_mode & 0o222
        assert target.samefile(chain["failed"] / "compilation" / chain["weights"][0]["path"])
        if kind == "payload_write":
            assert target.read_bytes() == b"privileged external shared-inode rewrite"


@pytest.mark.parametrize("kind", ["profile", "ddk", "air"])
def test_successor_does_not_rebind_profile_tool_or_source_air(chain, monkeypatch, kind):
    if kind == "ddk":
        chain["omg"].write_bytes(b"\x7fELF changed selected compiler")
    air = None
    if kind == "air":
        air = chain["staged"] / "other.air"
        air.write_bytes(chain["air"].read_bytes())
    _, _, report, copied, calls, _ = _successor(chain, monkeypatch,
        profile_name="cpro-40block-64-1" if kind == "profile" else "cpro-40block-prefill-64", air=air)
    assert report["status"] == "FAILED" and calls == copied == []

"""Cross-component retry: real tiny receipts/files, mocked OMG arithmetic only.

The AIR is a synthetic encoded 89/82 interface. These tests establish host
ownership, input reuse and auditing, never model execution or correctness.
"""
import errno
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from qwen35_moe_dflash import existing_air as compiler
from qwen35_moe_dflash import full_target
from qwen35_moe_dflash.artifacts import save_json
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.target_export_inventory import inventory_target_export
from qwen35_moe_dflash.target_export_reuse import stage_target_export_reuse
from test_target_export_inventory import source as encoded_source_fixture


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def _bytes_under(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    source = encoded_source_fixture.__wrapped__(tmp_path, monkeypatch)
    run = source["out"].parent
    manifest = inventory_target_export(source["air"], source["out"],
        torchair_pb2=source["schema"], hash_weights=True)
    assert manifest["status"] == "INVENTORIED"
    staged = run / "target-pytorch-import"
    receipt = stage_target_export_reuse(source["out"] / "inventory.json", source["air"],
        source["folder"], staged, apply=True)
    assert receipt["status"] == "STAGED_HARDLINKS"
    air = staged / "export.air"
    ddk = tmp_path / "selected-ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"\x7fELF host-only selected OMG mock")
    omg.chmod(0o755)
    monkeypatch.setenv("CPRO_DDK_ROOT", str(ddk))
    monkeypatch.setenv("CPRO_DDK_ID", "STATIC_RETRY_INTEGRATION_MOCK")
    monkeypatch.delenv("HIAI_INCLUDE_DIR", raising=False)
    def no_reflink(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "fixture forces the actual COPY fallback")
    monkeypatch.setattr(compiler.fcntl, "ioctl", no_reflink)
    def failed_dual_compile(argv, **kwargs):
        if "--help" in argv:
            kwargs["stdout"].write("OMG usage: --input_shape --dynamic_dims --limit_partition_size --save_weights_as_external_data\n")
            return subprocess.CompletedProcess(argv, 0)
        assert "--dynamic_dims=64,64,64,64,64;1,1,1,1,1" in argv
        assert "-1" in next(value for value in argv if value.startswith("--input_shape="))
        (kwargs["cwd"] / "dump_graph").write_bytes(b"retained failed compiler graph metadata")
        kwargs["stdout"].write("E/AI_FMK branches1 ReshapeComputeOutputDim false, input[1,1,2048] target[64,2048]\n")
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(subprocess, "run", failed_dual_compile)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/compile_full_target.py"
    main = runpy.run_path(str(script))["main"]
    failed_job = run / "target-pytorch-compile"
    monkeypatch.setattr(sys, "argv", [str(script), "--air", str(air), "--input-layout", "air-directory",
        "--target-features", "--output", str(failed_job), "--skip-inspection"])
    assert main() == 1
    old = json.loads((failed_job / "compilation/report.json").read_text())
    outer = json.loads((failed_job / "result.json").read_text())
    assert outer["status"] == old["status"] == "FAILED" and old["returncode"] == 1
    assert all(row["snapshot_method"] == "COPY" for row in old["input_snapshots"])
    assert outer["compilation_report_sha256"] == _sha(failed_job / "compilation/report.json")
    return {"run": run, "air": air, "staged": staged, "ddk": ddk, "omg": omg,
        "failed": failed_job, "old_report": old, "source": source}


def _retry(case, monkeypatch, *, tamper=None):
    profile = get_existing_air_profile("cpro-40block-prefill-64")
    job = case["run"] / "target-pytorch-prefill-64"
    job.mkdir()
    save_json(job / "compile-profile.json", profile)
    sources = {row["source"]: row for row in case["old_report"]["input_snapshots"]}
    weight_rows = [row for row in sources.values() if row["role"] != "air"]
    original_snapshot = compiler._snapshot
    copied = []
    def independent_air_only(source, destination):
        assert source == case["air"], "retry must not copy the large payload a second time"
        copied.append(str(source))
        return original_snapshot(source, destination)
    monkeypatch.setattr(compiler, "_snapshot", independent_air_only)
    calls, observations = [], []
    def static_compile(argv, **kwargs):
        if "--help" in argv:
            kwargs["stdout"].write("OMG usage: --input_shape --dynamic_dims --limit_partition_size --save_weights_as_external_data\n")
            return subprocess.CompletedProcess(argv, 0)
        calls.append(argv)
        assert not any(value.startswith("--dynamic_dims") for value in argv)
        assert "-1" not in next(value for value in argv if value.startswith("--input_shape="))
        cwd = kwargs["cwd"]
        assert cwd == job / "compilation/compile-work/generated"
        for row in weight_rows:
            old_path = case["failed"] / "compilation" / row["path"]
            new_path = job / "compilation" / row["path"]
            assert (old_path.stat().st_dev, old_path.stat().st_ino) == (new_path.stat().st_dev, new_path.stat().st_ino)
            assert not old_path.stat().st_mode & 0o222 and not new_path.stat().st_mode & 0o222
            assert old_path.read_bytes() == new_path.read_bytes()
            assert new_path.stat().st_ino != Path(row["source"]).stat().st_ino
            observations.append(new_path)
        air_row = next(row for row in sources.values() if row["role"] == "air")
        old_air = case["failed"] / "compilation" / air_row["path"]
        new_air = cwd / "export.air"
        assert len({case["air"].stat().st_ino, old_air.stat().st_ino, new_air.stat().st_ino}) == 3
        assert new_air.read_bytes() == old_air.read_bytes() == case["air"].read_bytes()
        if tamper:
            target = observations[0]
            target.chmod(0o644)
            if tamper == "write": target.write_bytes(b"MOCK OMG unexpectedly modified a shared snapshot")
        model = cwd / "model"
        model.mkdir()
        (model / "model.omc").write_bytes(b"mock static64 OMC")
        (model / "SubGraph_0.weight").write_bytes(b"mock new compiler external weight")
        kwargs["stdout"].write("OMG generate offline model success; host simulation only\n")
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", static_compile)
    result = compiler.compile_existing_air(case["air"], job / "compilation", case["ddk"],
        ddk_id="STATIC_RETRY_INTEGRATION_MOCK", input_shape=profile["input_shape"],
        dynamic_dims=profile["dynamic_dims"], limit_partition_size=profile["limit_partition_size"],
        save_weights_as_external_data=profile["save_weights_as_external_data"], input_layout="air-directory",
        reuse_failed_job=case["failed"])
    return job, profile, result, copied, calls, observations


def test_static_retry_reuses_only_owned_snapshot_inodes_and_new_job_can_be_validated(case, monkeypatch):
    before_old = _bytes_under(case["failed"])
    before_staged = {path: _stat(path) for path in case["staged"].rglob("*") if path.is_file()}
    job, profile, report, copied, calls, _ = _retry(case, monkeypatch)
    assert report["status"] == "COMPILED", report.get("error")
    assert len(calls) == 1 and copied == [str(case["air"])]
    audit = report["retained_input_reuse"]
    assert audit["status"] == "PASSED" and audit["explicit_weight_copy_bytes"] == 0
    assert report["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    assert all(Path(row["source"]) in before_staged for row in report["input_snapshots"])
    journal = Path(audit["journal"]["path"])
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert sum(row["event"] == "LINK_INTENT" for row in events) == 2
    assert sum(row["event"] == "LINK_CREATED" for row in events) == 2
    assert events[-1]["event"] == "AUDIT" and events[-1]["status"] == "PASSED"
    assert all(row["snapshot_method"] == "SHARED_READONLY_SNAPSHOT_INODE"
               for row in report["input_snapshots"] if row["role"] != "air")
    assert before_old == _bytes_under(case["failed"])
    assert before_staged == {path: _stat(path) for path in before_staged}
    save_json(job / "result.json", {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED",
        "model_compilation": "PASSED", "air": str(case["air"]), "output": str(job),
        "input_layout": "air-directory", "compilation": "compilation/report.json",
        "compilation_report_sha256": _sha(job / "compilation/report.json")})
    validated = full_target.validate_compilation(job, air=case["air"], ddk_root=case["ddk"],
        ddk_id="STATIC_RETRY_INTEGRATION_MOCK", profile=profile, input_layout="air-directory", target_features=True)
    assert validated["status"] == "COMPILED" and validated["device_execution"] == "NOT_RUN"
    audit_file = job / "compilation/retained-input-reuse.json"
    changed = json.loads(audit_file.read_text())
    changed["unbound_edit"] = True
    audit_file.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="retained-input audit receipt"):
        full_target.validate_compilation(job, air=case["air"], ddk_root=case["ddk"],
            ddk_id="STATIC_RETRY_INTEGRATION_MOCK", profile=profile, input_layout="air-directory", target_features=True)


@pytest.mark.parametrize("tamper", ["chmod", "write"])
def test_privileged_mock_compiler_mutation_is_rejected_preserved_and_never_repaired(case, monkeypatch, tamper):
    old_metadata = {path: path.read_bytes() for path in case["failed"].rglob("*")
                    if path.is_file() and path.suffix in {".json", ".log", ".txt"}}
    staged_before = _bytes_under(case["staged"])
    job, _, report, copied, calls, observed = _retry(case, monkeypatch, tamper=tamper)
    assert len(calls) == 1 and copied == [str(case["air"])]
    assert report["status"] == "FAILED"
    assert report["retained_input_reuse"]["status"] == "FAILED"
    assert report["retained_input_reuse"]["audit_errors"]
    assert (job / "compilation/failure.json").is_file()
    assert (job / "compilation/retained-input-reuse.json").is_file()
    assert (job / "compilation/retained-input-links.jsonl").is_file()
    assert (job / "compilation/compile-work/generated/model/model.omc").read_bytes() == b"mock static64 OMC"
    # The audit observes and rejects privileged writes. It must not chmod,
    # restore old bytes, unlink evidence or claim an independent snapshot.
    assert observed[0].stat().st_mode & 0o222
    relative = observed[0].relative_to(job / "compilation")
    old_shared = case["failed"] / "compilation" / relative
    assert old_shared.samefile(observed[0]) and old_shared.stat().st_mode & 0o222
    if tamper == "write":
        assert observed[0].read_bytes() == b"MOCK OMG unexpectedly modified a shared snapshot"
        assert old_shared.read_bytes() == observed[0].read_bytes()
    assert old_metadata == {path: path.read_bytes() for path in old_metadata}
    assert staged_before == _bytes_under(case["staged"])

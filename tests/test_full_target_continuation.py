"""Original-target continuation using simulated OMG, never target hardware."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from qwen35_moe_dflash import full_target
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("CPRO_DDK_ID", "MOCK-INTERNAL")
    monkeypatch.delenv("HIAI_INCLUDE_DIR", raising=False)
    source = tmp_path / "original/export"
    source.mkdir(parents=True)
    air = source / "export.air"
    air.write_bytes(b"OPAQUE ORIGINAL AIR; NO MODEL SEMANTICS IN THIS TEST")
    weight = source / "model_layers_0_weight"
    weight.write_bytes(b"ORIGINAL QUANTIZED BYTES")
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"\x7fELF MOCK OMG")
    omg.chmod(0o755)
    monkeypatch.setenv("CPRO_DDK_ROOT", str(ddk))
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/compile_full_target.py"
    main = runpy.run_path(str(script))["main"]
    job = tmp_path / "full-target"
    calls = []
    def tool(argv, **kwargs):
        calls.append(argv)
        if argv[1:] == ["--help"]:
            kwargs["stdout"].write("OMG usage: --input_shape --dynamic_dims --limit_partition_size --save_weights_as_external_data\n")
        else:
            assert "--platform=kirinx90" in argv
            assert "--save_weights_as_external_data=true" in argv
            output = kwargs["cwd"] / "model"
            output.mkdir()
            (output / "model.omc").write_bytes(b"MOCK EXTERNAL OMC")
            (output / "SubGraph_0.weight").write_bytes(b"COMPILED EXTERNAL WEIGHTS")
            kwargs["stdout"].write("OMG generate offline model success\n")
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", tool)
    monkeypatch.setattr(sys, "argv", [str(script), "--air", str(air), "--output", str(job), "--skip-inspection"])
    assert main() == 0
    options = dict(air=air, ddk_root=ddk, ddk_id="MOCK-INTERNAL",
                   profile=get_existing_air_profile("cpro-40block-64-1"), input_layout="export-directory")
    return dict(job=job, main=main, script=script, calls=calls, options=options, weight=weight, omg=omg)


def test_repeated_cli_reuses_the_complete_external_artifact_without_a_tool_call(target, monkeypatch):
    before = {p.relative_to(target["job"]).as_posix(): p.read_bytes()
              for p in target["job"].rglob("*") if p.is_file()}
    count = len(target["calls"])
    monkeypatch.setattr(sys, "argv", [str(target["script"]), "--resume", str(target["job"]), "--skip-inspection"])
    assert target["main"]() == 0
    assert len(target["calls"]) == count
    for name, raw in before.items():
        assert (target["job"] / name).read_bytes() == raw
    result = full_target.read_metadata(next((target["job"] / "continuations").glob("*/result.json")))
    assert result["compiler_invocation"] == "NOT_RUN_REUSED_COMPLETE_COMPILE"
    assert result["status"] == "COMPILED" and result["device_execution"] == "NOT_RUN"
    assert result["device_ready"] is False


@pytest.mark.parametrize("change", ["source", "staged", "generated", "added", "missing", "compiler", "command", "log", "report"])
def test_modified_source_tool_or_any_external_artifact_rejects_reuse(target, change):
    job, root = target["job"], target["job"] / "compilation"
    report = full_target.read_metadata(root / "report.json")
    generated = root / next(r["path"] for r in report["generated_artifacts"] if r["path"].endswith(".weight"))
    if change == "source":
        target["weight"].write_bytes(b"ALTERED QUANTIZED BYTES!")
    elif change == "staged":
        path = root / report["input_snapshots"][1]["path"]
        path.chmod(0o644)
        before = path.stat()
        path.write_bytes(b"X" * path.stat().st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "generated":
        generated.write_bytes(b"X" * generated.stat().st_size)
    elif change == "added":
        (generated.parent / "unexpected.weight").write_bytes(b"UNBOUND")
    elif change == "missing":
        generated.unlink()
    elif change == "compiler":
        target["omg"].write_bytes(b"\x7fELF OTHER OMG")
    elif change == "command":
        (root / "compile-command.json").write_text("{}")
    elif change == "log":
        (root / "compile.log").write_text("DIFFERENT LOG")
    else:
        report["ddk_id"] = "ALTERED"
        (root / "report.json").write_text(json.dumps(report))
    count = len(target["calls"])
    with pytest.raises(ValueError):
        full_target.validate_compilation(job, **target["options"])
    assert len(target["calls"]) == count


def test_profile_changes_and_failed_jobs_are_never_recompiled_automatically(target, monkeypatch):
    profile = dict(target["options"]["profile"], dynamic_dims="32,32,32,32,32;1,1,1,1,1")
    with pytest.raises(ValueError, match="differs"):
        full_target.validate_compilation(target["job"], **dict(target["options"], profile=profile))
    result_path = target["job"] / "result.json"
    result = full_target.read_metadata(result_path)
    result["status"] = "FAILED"
    result_path.write_text(json.dumps(result))
    count = len(target["calls"])
    monkeypatch.setattr(sys, "argv", [str(target["script"]), "--resume", str(target["job"]), "--skip-inspection"])
    assert target["main"]() == 1
    assert len(target["calls"]) == count
    assert full_target.read_metadata(result_path)["status"] == "FAILED"
    assert not (target["job"] / "continuations").exists()


def test_source_lock_spans_different_job_directories(target):
    with full_target.target_source_lock(target["options"]["air"]):
        with pytest.raises(RuntimeError, match="active"):
            with full_target.target_source_lock(target["options"]["air"]):
                pytest.fail("must not acquire a second source lock")


def test_active_entry_and_orphan_omg_are_found_without_blocking_draft(target, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    air = target["options"]["air"]
    report = full_target.read_metadata(target["job"] / "compilation/report.json")
    staged = target["job"] / "compilation" / next(r["path"] for r in report["input_snapshots"] if r["role"] == "air")
    jobs = [
        ["python3", str(target["script"]), "--air", str(air)],
        [str(target["omg"]), "--model=" + str(staged), "--platform=kirinx90"],
        ["python3", "run_full_draft.py", "--air", str(air)],
        [str(target["omg"]), "--model=" + str(tmp_path / "draft/export.air"), "--platform=kirinx90"],
        [str(target["omg"]), "--model=" + str(air), "--platform=kirinx90"],
    ]
    for i, argv in enumerate(jobs, 90000):
        folder = proc / str(i)
        folder.mkdir()
        (folder / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (folder / "cwd").symlink_to(tmp_path, target_is_directory=True)
    rows = full_target.active_target_jobs(air, proc_root=proc)
    assert {r["pid"] for r in rows} == {90000, 90001, 90004}

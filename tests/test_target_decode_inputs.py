"""Host-only decode reference ownership tests; no vendor/model execution."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from qwen35_moe_dflash import target_decode_inputs as view
from qwen35_moe_dflash.existing_air import _snapshot
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from test_target_export_transfer_inspection import incoming as incoming_fixture, save as save_transfer


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _five(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _full(path):
    info = path.lstat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid, info.st_nlink)


def _save(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def case(tmp_path, monkeypatch):
    transfer = tmp_path / "incoming"
    transfer.mkdir()
    incoming = incoming_fixture.__wrapped__(transfer, monkeypatch)
    job, source = tmp_path / "successful-prefill", tmp_path / "original-prefill"
    source.mkdir()
    compilation = job / "compilation"
    generated = compilation / "compile-work/generated"
    generated.mkdir(parents=True)
    old_air = source / "export.air"
    old_air.write_bytes(b"independent host-only old prefill AIR")
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"\x7fELFhost-only compiler identity, never executed")
    omg.chmod(0o755)
    originals = [old_air]
    prefix = "glm_export_air_cpro_40block_64_quant"
    (source / prefix).mkdir()
    for row in incoming["manifest"]["files"]:
        path = source / prefix / row["basename"]
        path.write_bytes((incoming["origin"] / row["basename"]).read_bytes())
        originals.append(path)
    snapshots = []
    for original in originals:
        relative = original.relative_to(source)
        target = generated / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(original.read_bytes())
        target.chmod(0o444)
        snapshots.append({"source": str(original), "source_stat": _five(original),
            "bytes": original.stat().st_size, "sha256": _sha(original),
            "snapshot_stat": _five(target), "snapshot_method": "COPY",
            "path": target.relative_to(compilation).as_posix(), "source_relative": relative.as_posix(),
            "role": "air" if original == old_air else "neighbor_data_candidate"})
    options = view._options("cpro-40block-prefill-64")
    (compilation / "compile.log").write_bytes(b"Host-only source compiler success, never native/model execution")
    report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED",
        "model_compilation": "PASSED", "returncode": 0, "output": str(compilation),
        "air_source": str(old_air), "air": {"path": str(old_air), "bytes": old_air.stat().st_size,
            "sha256": _sha(old_air)}, "requested_options": options,
        "ddk_root": str(ddk), "ddk_id": "HOST_TEST_ONLY", "omg_path": str(omg),
        "omg_sha256": _sha(omg), "platform": "kirinx90",
        "compile_log": "compile.log", "compile_log_sha256": _sha(compilation / "compile.log"),
        "source_preservation": {"status": "UNCHANGED_BY_STAT", "changed": []},
        "input_layout": {"mode": "air-directory", "sidecar_prefix": ".",
            "compiler_cwd": "compile-work/generated", "air_path": "compile-work/generated/export.air"},
        "input_snapshots": snapshots}
    _save(job / "compile-profile.json", get_existing_air_profile("cpro-40block-prefill-64"))
    _save(compilation / "report.json", report)
    _save(compilation / "input-inventory.json", {"files": snapshots})
    _save(compilation / "compile-command.json", {"argv": [str(omg), "--framework=1",
        "--model=" + str(generated / "export.air"), "--output=model", "--target=omc", "--platform=kirinx90",
        *("--" + key + "=" + value for key, value in options.items())],
        "cwd": str(generated), "weight_dtype_override": None})
    outer = {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED",
        "model_compilation": "PASSED", "output": str(job), "air": str(old_air),
        "input_layout": "air-directory", "compilation": "compilation/report.json",
        "compilation_report_sha256": _sha(compilation / "report.json")}
    _save(job / "result.json", outer)
    output = tmp_path / "new-decode-job/compilation"
    (output / "compile-work/generated").mkdir(parents=True)
    return dict(incoming=incoming, source_job=job, source_report=report, source_outer=outer,
        output=output, ddk_root=ddk, omg_path=omg, ddk_id="HOST_TEST_ONLY",
        omg_sha256=_sha(omg), options=view._options("cpro-40block-decode-1"), input_layout="air-directory")


def _prepare(case):
    incoming = case["incoming"]
    return view.prepare_decode_input_view(case["source_job"], air=incoming["air"],
        inventory=incoming["inventory"], export_receipt=incoming["receipt"],
        **{key: case[key] for key in ("output", "ddk_root", "ddk_id", "omg_path", "omg_sha256", "options", "input_layout")})


def _stage(case, monkeypatch=None):
    plan = _prepare(case)
    _save(case["output"] / "decode-input-view-plan.json", plan.evidence)
    snapshots, copied = [], []
    def small_air_only(source, destination):
        assert source == case["incoming"]["air"]
        copied.append(source)
        return _snapshot(source, destination)
    for source in plan.sources:
        relative = plan.source_relative(source)
        destination = case["output"] / "compile-work/generated" / relative
        row = plan.snapshot(source, destination, small_air_only)
        row.update(path=destination.relative_to(case["output"]).as_posix(),
            source_relative=relative.as_posix(), role="air" if source == plan.air else "neighbor_data_candidate")
        snapshots.append(row)
    return plan, snapshots, copied


def _complete(case, plan, snapshots):
    audit = plan.audit()
    _save(case["output"] / "decode-input-view.json", audit)
    report = {key: case[key] for key in ("ddk_root", "ddk_id", "omg_path", "omg_sha256")}
    report.update(ddk_root=str(report["ddk_root"]), omg_path=str(report["omg_path"]),
        decode_input_view=audit, requested_options=case["options"],
        input_layout={"mode": "air-directory"}, input_snapshots=snapshots)
    return audit, report


def test_only_air_copies_and_original_prefill_metadata_and_receipts_are_unchanged(case):
    old_files = [path for path in case["source_job"].rglob("*") if path.is_file()]
    old_bytes = {path: path.read_bytes() for path in old_files}
    old_stats = {path: _full(path) for path in old_files}
    source_stats = {Path(row["source"]): _full(Path(row["source"])) for row in case["source_report"]["input_snapshots"]}
    plan, snapshots, copied = _stage(case)
    assert copied == [case["incoming"]["air"]]
    assert len(list((case["output"] / "compile-work/generated").rglob("*"))) == 4
    for row in snapshots:
        alias = case["output"] / row["path"]
        if row["role"] == "air":
            assert not alias.is_symlink() and alias.read_bytes() == case["incoming"]["air"].read_bytes()
            assert not alias.samefile(case["incoming"]["air"])
        else:
            assert alias.parent.name == "glm_export_air_cpro_40block_1_quant"
            assert alias.is_symlink() and os.readlink(alias) == row["source"]
            assert alias.samefile(row["source"])
            assert not Path(row["source"]).stat().st_mode & 0o222
            assert row["snapshot_method"] == view.REFERENCE_METHOD
    audit, report = _complete(case, plan, snapshots)
    assert audit["status"] == "PASSED", audit.get("audit_errors")
    assert audit["explicit_weight_copy_bytes"] == 0
    assert audit["transfer"]["data_dtype_validation"] == "DATA_DTYPE_SEMANTICS_UNVERIFIED"
    aliases = view.validate_decode_input_view(case["output"], report)
    assert len(aliases) == 2
    assert {path: path.read_bytes() for path in old_files} == old_bytes
    assert {path: _full(path) for path in old_files} == old_stats
    assert {path: _full(path) for path in source_stats} == source_stats


@pytest.mark.parametrize("fault", ["failed_source", "wrong_profile", "unbound_report", "recursive_source", "writable_snapshot", "source_symlink", "declared_weight_hash", "decode64", "renamed_data", "missing_input", "schema", "known_input_dtype", "inventory_sha", "parent_traversal", "compile_log"])
def test_invalid_source_or_transfer_stops_before_any_alias(case, fault):
    source_report, incoming = case["source_report"], case["incoming"]
    source_comp = case["source_job"] / "compilation"
    if fault == "failed_source":
        source_report["status"] = "FAILED"
    elif fault == "wrong_profile":
        _save(case["source_job"] / "compile-profile.json", get_existing_air_profile("cpro-40block-decode-1"))
    elif fault == "unbound_report":
        source_report["unbound_edit"] = True
        _save(source_comp / "report.json", source_report)
    elif fault == "recursive_source":
        source_report["decode_input_view"] = {"status": "PASSED"}
    elif fault in {"writable_snapshot", "source_symlink"}:
        target = source_comp / source_report["input_snapshots"][1]["path"]
        if fault == "writable_snapshot": target.chmod(0o644)
        else:
            target.unlink()
            target.symlink_to(source_report["input_snapshots"][1]["source"])
    elif fault == "declared_weight_hash": incoming["manifest"]["files"][0]["sha256"] = "f" * 64
    elif fault == "decode64": incoming["result"]["normalization"]["sequence_length"] = 64
    elif fault == "renamed_data": incoming["result"]["normalization"]["input_validation"]["inputs"][0]["name"] = "arg999_1"
    elif fault == "missing_input": incoming["result"]["normalization"]["input_validation"]["inputs"].pop()
    elif fault == "schema": incoming["result"]["normalization"]["schema"]["descriptor_sha256"] = "f" * 64
    elif fault == "known_input_dtype":
        for schema in (incoming["manifest"]["schema"], incoming["result"]["schema"],
                       incoming["result"]["normalization"]["schema"]):
            schema["descriptor_sha256"] = view._ORIGIN_SCHEMA
        for row in incoming["result"]["normalization"]["input_validation"]["inputs"]:
            row["raw_dtype"] = 2 if row["type"] == "RefData" else view._ORIGIN_DATA_DTYPES[row["name"]]
        incoming["result"]["normalization"]["input_validation"]["inputs"][0]["raw_dtype"] = 999
    elif fault == "compile_log": (source_comp / "compile.log").write_bytes(b"unbound compiler log")
    elif fault == "parent_traversal":
        incoming["manifest"]["files"][0].update(reference="./../escape", relative_path="../escape", basename="escape")
    else: incoming["result"]["inventory"]["sha256"] = "f" * 64
    if fault in {"failed_source", "recursive_source"}:
        _save(source_comp / "report.json", source_report)
        case["source_outer"]["compilation_report_sha256"] = _sha(source_comp / "report.json")
        _save(case["source_job"] / "result.json", case["source_outer"])
    if fault not in {"failed_source", "wrong_profile", "unbound_report", "recursive_source", "writable_snapshot", "source_symlink", "inventory_sha"}:
        save_transfer(incoming)
    elif fault == "inventory_sha":
        _save(incoming["receipt"], incoming["result"])
    with pytest.raises((ValueError, OSError)):
        _prepare(case)
    assert list((case["output"] / "compile-work/generated").iterdir()) == []


@pytest.mark.parametrize("fault", ["repoint", "recreate_alias", "extra_alias", "target_write", "target_chmod", "target_hardlink", "old_receipt", "new_air"])
def test_post_compile_mutations_fail_and_are_retained(case, fault):
    plan, rows, _ = _stage(case)
    weight = rows[1]
    alias, target = case["output"] / weight["path"], Path(weight["source"])
    if fault in {"repoint", "recreate_alias"}:
        replacement = alias.with_name("replacement")
        replacement.symlink_to(rows[2]["source"] if fault == "repoint" else target)
        os.replace(replacement, alias)
    elif fault == "extra_alias": (alias.parent / "extra-weight").symlink_to(target)
    elif fault == "target_write":
        target.chmod(0o644)
        target.write_bytes(b"privileged writer changed read-only shared snapshot")
    elif fault == "target_chmod": target.chmod(0o644)
    elif fault == "target_hardlink": os.link(target, case["output"] / "unexpected-extra-hardlink")
    elif fault == "old_receipt":
        path = case["source_job"] / "result.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        air = case["output"] / rows[0]["path"]
        air.chmod(0o644)
        air.write_bytes(b"modified staged AIR")
    audit = plan.audit()
    assert audit["status"] == "FAILED" and audit["audit_errors"]
    assert alias.is_symlink(), "failed audit preserves its caller-owned evidence"


def test_live_target_hash_not_just_prior_receipt_is_checked_before_creating_alias(case):
    plan = _prepare(case)
    source = plan.sources[1]
    source.chmod(0o644)
    source.write_bytes(b"replacement payload after metadata preflight")
    destination = case["output"] / "compile-work/generated" / plan.source_relative(source)
    with pytest.raises(ValueError, match="identity changed"):
        plan.snapshot(source, destination, lambda *args: pytest.fail("must never copy a weight"))
    assert not destination.exists() and not destination.is_symlink()


def test_declared_hashes_and_current_stats_never_replace_a_fresh_payload_hash(case):
    comp = case["source_job"] / "compilation"
    row = case["source_report"]["input_snapshots"][1]
    target = comp / row["path"]
    target.chmod(0o644)
    target.write_bytes(b"X" * row["bytes"])
    target.chmod(0o444)
    # Simulate stale content declarations in otherwise internally consistent
    # current source receipts. Only reading the actual target can reject this.
    row["snapshot_stat"] = _five(target)
    _save(comp / "report.json", case["source_report"])
    _save(comp / "input-inventory.json", {"files": case["source_report"]["input_snapshots"]})
    case["source_outer"]["compilation_report_sha256"] = _sha(comp / "report.json")
    _save(case["source_job"] / "result.json", case["source_outer"])
    plan = _prepare(case)
    destination = case["output"] / "compile-work/generated" / plan.source_relative(target)
    with pytest.raises(ValueError, match="SHA256 differs"):
        plan.snapshot(target, destination, lambda *args: pytest.fail("weight copy is forbidden"))
    assert not destination.exists() and not destination.is_symlink()


@pytest.mark.parametrize("fault", ["forged_target", "missing_row", "duplicate_row", "wrong_path", "audit_changed", "extra_alias", "protected_metadata"])
def test_resume_requires_complete_bound_alias_and_original_source_evidence(case, fault):
    plan, rows, _ = _stage(case)
    audit, report = _complete(case, plan, rows)
    assert audit["status"] == "PASSED"
    if fault == "forged_target":
        report["input_snapshots"][1]["reference"]["target"] = str(case["incoming"]["air"])
    elif fault == "missing_row": report["input_snapshots"].pop()
    elif fault == "duplicate_row": report["input_snapshots"][2] = deepcopy(report["input_snapshots"][1])
    elif fault == "wrong_path": report["input_snapshots"][1]["path"] = "compile-work/generated/escape"
    elif fault == "audit_changed":
        audit["extra_unbound_note"] = "changed after report"
        _save(case["output"] / "decode-input-view.json", audit)
        report["decode_input_view"] = deepcopy(audit)
        report["decode_input_view"].pop("extra_unbound_note")
    elif fault == "extra_alias":
        alias = case["output"] / rows[1]["path"]
        (alias.parent / "extra-weight").symlink_to(rows[1]["source"])
    else:
        old = case["source_job"] / "compile-profile.json"
        old.write_bytes(old.read_bytes() + b" ")
    with pytest.raises((ValueError, OSError)):
        view.validate_decode_input_view(case["output"], report)

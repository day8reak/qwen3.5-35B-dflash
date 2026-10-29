"""Host-only fixtures for a single failed dual-gear to static64 retry.

Tiny opaque bytes do not represent an AIR, model accuracy or device execution.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from qwen35_moe_dflash import target_snapshot_retry as retry
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _stat(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _normal(source, destination):
    before = _stat(source)
    with Path(source).open("rb") as src, Path(destination).open("xb") as dst:
        shutil.copyfileobj(src, dst)
    destination.chmod(0o444)
    return {"source": str(source), "source_stat": before, "bytes": source.stat().st_size,
            "sha256": _sha(destination), "snapshot_stat": _stat(destination), "snapshot_method": "COPY"}


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "staged"
    source.mkdir()
    air = source / "export.air"
    air.write_bytes(b"opaque host-only AIR")
    weights = [source / "prefix64" / name for name in ("weight-a", "weight-b")]
    weights[0].parent.mkdir()
    weights[0].write_bytes(b"unchanged quantized weight A")
    weights[1].write_bytes(b"unchanged quantized weight B")
    selected = [air, *weights]
    job = tmp_path / "failed"
    compilation = job / "compilation"
    generated = compilation / "compile-work/generated"
    generated.mkdir(parents=True)
    rows = []
    for path in selected:
        relative = path.relative_to(source)
        snapshot = generated / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        row = _normal(path, snapshot)
        row.update(path=snapshot.relative_to(compilation).as_posix(), source_relative=relative.as_posix(),
                   role="air" if path == air else "neighbor_data_candidate")
        rows.append(row)
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"\x7fELF host-only mock compiler")
    omg.chmod(0o755)
    diagnosis = tmp_path / "diagnosis"
    _write(diagnosis / "report.json", {"host_fixture": "retained TBE diagnosis"})
    compatibility = {"diagnosis": str(diagnosis), "compiler_invocation": "ATTEMPTED", "compile": "FAILED",
        "audit": {"status": "APPLIED"}, "metadata": {"diagnosis_report": {
            "path": str(diagnosis / "report.json"), "sha256": _sha(diagnosis / "report.json")}}}
    profile = get_existing_air_profile("cpro-40block-64-1")
    old_options = retry._options(profile, static=False)
    report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "FAILED", "stage": "omg",
        "model_compilation": "FAILED", "returncode": 1, "output": str(compilation),
        "air_source": str(air), "air": {"path": str(air), "bytes": air.stat().st_size, "sha256": _sha(air)},
        "requested_options": old_options, "input_snapshots": rows,
        "source_preservation": {"status": "UNCHANGED_BY_STAT", "changed": []},
        "input_layout": {"mode": "air-directory", "compiler_cwd": "compile-work/generated",
                         "air_path": rows[0]["path"], "sidecar_prefix": "."},
        "ddk_root": str(ddk), "ddk_id": "HOST_TEST_DDK", "omg_path": str(omg),
        "omg_sha256": _sha(omg), "platform": "kirinx90", "tbe_compatibility": compatibility}
    outer = {"artifact_kind": "qwen35-original-target-job", "status": "FAILED", "model_compilation": "FAILED",
        "air": str(air), "output": str(job), "compilation": "compilation/report.json", "input_layout": "air-directory"}
    _write(job / "compile-profile.json", profile)
    _write(compilation / "input-inventory.json", {"files": rows})
    _write(compilation / "compile-command.json", {"argv": [str(omg), "--framework=1", "--model=" + str(generated / air.name),
        "--output=model", "--target=omc", "--platform=kirinx90", *(f"--{k}={v}" for k, v in old_options.items())],
        "cwd": str(generated), "weight_dtype_override": None})
    _write(compilation / "tbe-compatibility-result.json", compatibility)
    output = tmp_path / "new-static-job/compilation"
    output.mkdir(parents=True)
    c = dict(tmp=tmp_path, source=source, air=air, weights=weights, selected=selected,
             job=job, compilation=compilation, generated=generated, rows=rows, report=report,
             outer=outer, ddk=ddk, omg=omg, diagnosis=diagnosis, output=output, profile=profile)
    _save(c)
    return c


def _save(c):
    _write(c["compilation"] / "report.json", c["report"])
    c["outer"]["compilation_report_sha256"] = _sha(c["compilation"] / "report.json")
    _write(c["job"] / "result.json", c["outer"])


def _prepare(c, **overrides):
    options = dict(air=c["air"], ddk_root=c["ddk"], ddk_id="HOST_TEST_DDK", omg_path=c["omg"],
        omg_sha256=_sha(c["omg"]), tbe_diagnosis=c["diagnosis"], input_layout="air-directory",
        options=retry._options(c["profile"], static=True), selected=c["selected"], output=c["output"])
    options.update(overrides)
    return retry.prepare_failed_prefill_reuse(c["job"], **options)


def _snapshots(c, plan):
    return [plan.snapshot(source, c["output"] / row["path"], _normal)
            for source, row in zip(c["selected"], c["rows"])]


def test_full_reuse_shares_only_readonly_payloads_keeps_sources_and_old_receipts(case, monkeypatch):
    old_metadata = {p: p.read_bytes() for p in case["job"].glob("*.json")}
    old_metadata.update({p: p.read_bytes() for p in case["compilation"].glob("*.json")})
    staged_before = {p: retry._identity(p) for p in case["selected"]}
    original_modes = {p: p.stat().st_mode for p in case["selected"]}
    plan = _prepare(case)
    # A new attempt may not chmod either the original files or old shared inode.
    monkeypatch.setattr(Path, "chmod", lambda *a, **k: pytest.fail("payload sharing must never chmod"))
    def independent_air(source, destination):
        # Fixture callback creates a separate read-only inode without chmod.
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source.read_bytes())
        return {"source": str(source), "source_stat": _stat(source), "bytes": source.stat().st_size,
                "sha256": _sha(destination), "snapshot_stat": _stat(destination), "snapshot_method": "COPY"}
    rows = [plan.snapshot(src, case["output"] / old["path"], independent_air)
            for src, old in zip(case["selected"], case["rows"])]
    result = plan.audit()
    assert result["status"] == "PASSED", result
    assert result["ownership"] == "SHARED_READONLY_INODE"
    assert result["explicit_weight_copy_bytes"] == 0
    assert result["device_execution"] == result["numerical_validation"] == "NOT_RUN"
    assert {p: p.read_bytes() for p in old_metadata} == old_metadata
    assert {p: retry._identity(p) for p in staged_before} == staged_before
    assert {p: p.stat().st_mode for p in original_modes} == original_modes
    assert rows[0]["snapshot_method"] == "COPY"
    assert rows[0]["snapshot_stat"][1] != case["rows"][0]["snapshot_stat"][1]
    for row, old in zip(rows[1:], case["rows"][1:]):
        assert row["source"] == old["source"] and row["source_stat"] == old["source_stat"]
        assert row["snapshot_method"] == "SHARED_READONLY_SNAPSHOT_INODE"
        assert (case["output"] / old["path"]).stat().st_ino == (case["compilation"] / old["path"]).stat().st_ino
        assert (case["compilation"] / old["path"]).stat().st_nlink == 2
        assert row["snapshot_stat"] == _stat(case["output"] / old["path"])
    events = [json.loads(line) for line in Path(result["journal"]["path"]).read_text().splitlines()]
    assert [e["event"] for e in events] == ["PREPARED", "LINK_INTENT", "LINK_CREATED", "LINK_INTENT", "LINK_CREATED", "AUDIT"]
    assert events[1]["before"]["nlink"] == 1 and events[2]["after"]["nlink"] == 2
    assert all(len(json.dumps(e).encode()) < 64 << 10 for e in events)
    assert result["journal"]["sha256"] == _sha(Path(result["journal"]["path"]))


def test_prepare_does_not_open_old_or_original_weight_payloads(case, monkeypatch):
    original = retry._hash_file
    def guarded(path):
        assert Path(path) == case["omg"], "prepare may only fully hash the selected compiler"
        return original(path)
    monkeypatch.setattr(retry, "_hash_file", guarded)
    original_read = retry._read_small
    def small_guard(path, maximum):
        assert Path(path) not in case["weights"]
        assert Path(path) not in [case["compilation"] / r["path"] for r in case["rows"][1:]]
        return original_read(path, maximum)
    monkeypatch.setattr(retry, "_read_small", small_guard)
    assert _prepare(case).evidence["status"] == "PREPARED"


@pytest.mark.parametrize("fault", ["success", "missing_report_sha", "wrong_report_sha", "timeout", "zero_exit",
    "bool_exit", "wrong_stage", "wrong_kind", "source_changed", "audit_error", "artifact_error",
    "skipped_nonregular", "ddk", "omg_sha", "source_air", "old_options", "inventory_missing",
    "inventory_different", "snapshot_method", "invalid_hash", "unsafe_path", "wrong_snapshot_path",
    "source_relative", "role", "duplicate_source", "duplicate_path", "source_stat", "snapshot_stat",
    "tbe_diagnosis", "tbe_audit", "tbe_report_hash"])
def test_rejects_unbound_or_incompatible_old_evidence(case, fault):
    report, outer = case["report"], case["outer"]
    if fault == "success": outer["status"] = "COMPILED"
    elif fault == "missing_report_sha": pass
    elif fault == "wrong_report_sha": pass
    elif fault == "timeout": report.pop("returncode")
    elif fault == "zero_exit": report["returncode"] = 0
    elif fault == "bool_exit": report["returncode"] = True
    elif fault == "wrong_stage": report["stage"] = "artifact-inventory"
    elif fault == "wrong_kind": report["artifact_kind"] = "other"
    elif fault == "source_changed": report["source_preservation"]["status"] = "CHANGED"
    elif fault == "audit_error": report["audit_error"] = {"status": "REJECTED"}
    elif fault == "artifact_error": report["artifact_inventory_error"] = "incomplete"
    elif fault == "skipped_nonregular": report["skipped_nonregular_artifacts"] = ["fifo"]
    elif fault == "ddk": report["ddk_id"] = "OTHER"
    elif fault == "omg_sha": report["omg_sha256"] = "a" * 64
    elif fault == "source_air": report["air_source"] = "/other/export.air"
    elif fault == "old_options": report["requested_options"]["weight_data_type"] = "FP16"
    elif fault == "inventory_missing": report.pop("input_snapshots")
    elif fault == "inventory_different": pass
    elif fault == "snapshot_method": report["input_snapshots"][1]["snapshot_method"] = "REFLINK"
    elif fault == "invalid_hash": report["input_snapshots"][1]["sha256"] = "not-a-digest"
    elif fault == "unsafe_path": report["input_snapshots"][1]["path"] = "../outside"
    elif fault == "wrong_snapshot_path": report["input_snapshots"][1]["path"] = report["input_snapshots"][2]["path"]
    elif fault == "source_relative": report["input_snapshots"][1]["source_relative"] = "other-name"
    elif fault == "role": report["input_snapshots"][1]["role"] = "primary_omc"
    elif fault == "duplicate_source": report["input_snapshots"].append(dict(report["input_snapshots"][1]))
    elif fault == "duplicate_path": report["input_snapshots"][1]["path"] = report["input_snapshots"][0]["path"]
    elif fault == "source_stat": report["input_snapshots"][1]["source_stat"][-1] += 1
    elif fault == "snapshot_stat": report["input_snapshots"][1]["snapshot_stat"][-1] += 1
    elif fault == "tbe_diagnosis": report["tbe_compatibility"]["diagnosis"] = "/other/diagnosis"
    elif fault == "tbe_audit": report["tbe_compatibility"]["audit"]["status"] = "REJECTED"
    elif fault == "tbe_report_hash": report["tbe_compatibility"]["metadata"]["diagnosis_report"]["sha256"] = "f" * 64
    if "input_snapshots" in report and fault != "inventory_different":
        _write(case["compilation"] / "input-inventory.json", {"files": report["input_snapshots"]})
    elif fault == "inventory_different":
        _write(case["compilation"] / "input-inventory.json", {"files": []})
    _write(case["compilation"] / "tbe-compatibility-result.json", report["tbe_compatibility"])
    _save(case)
    if fault in {"missing_report_sha", "wrong_report_sha"}:
        outer.pop("compilation_report_sha256")
        if fault == "wrong_report_sha": outer["compilation_report_sha256"] = "a" * 64
        _write(case["job"] / "result.json", outer)
    with pytest.raises((ValueError, OSError)):
        _prepare(case)
    assert not (case["output"] / "retained-input-links.jsonl").exists()
    assert all((case["compilation"] / row["path"]).stat().st_nlink == 1
               for row in case["rows"] if (case["compilation"] / row["path"]).is_file())


@pytest.mark.parametrize("fault", ["new_dynamic", "new_precision", "new_customization", "new_partition",
    "missing_selected", "duplicate_selected", "wrong_argv", "wrong_cwd", "bad_profile", "writable_snapshot",
    "already_linked", "shared_source_inode", "duplicate_snapshot_inode", "source_symlink", "snapshot_symlink",
    "job_symlink", "output_oldjob", "output_source", "output_ddk", "existing_journal"])
def test_rejects_unsafe_mutations_or_new_option_variants(case, fault):
    overrides = {}
    weight = case["compilation"] / case["rows"][1]["path"]
    if fault.startswith("new_"):
        options = retry._options(case["profile"], static=True)
        if fault == "new_dynamic": options["dynamic_dims"] = "64,64,64,64,64"
        elif fault == "new_precision": options["weight_data_type"] = "FP16"
        elif fault == "new_customization": options["cl_customization"] = "config.json"
        else: options["limit_partition_size"] = "true"
        overrides["options"] = options
    elif fault == "missing_selected": overrides["selected"] = case["selected"][:-1]
    elif fault == "duplicate_selected": overrides["selected"] = case["selected"] + [case["air"]]
    elif fault in {"wrong_argv", "wrong_cwd"}:
        path = case["compilation"] / "compile-command.json"
        command = json.loads(path.read_text())
        if fault == "wrong_argv": command["argv"].append("--weight_data_type=FP16")
        else: command["cwd"] = str(case["generated"].parent)
        _write(path, command)
    elif fault == "bad_profile": _write(case["job"] / "compile-profile.json", {})
    elif fault == "writable_snapshot": weight.chmod(0o644)
    elif fault == "already_linked": os.link(weight, case["tmp"] / "prior-alias")
    elif fault == "shared_source_inode":
        weight.unlink(); os.link(case["weights"][0], weight); weight.chmod(0o444)
    elif fault == "duplicate_snapshot_inode":
        other = case["compilation"] / case["rows"][2]["path"]
        other.unlink(); os.link(weight, other)
    elif fault == "source_symlink":
        source = case["weights"][0]; source.unlink(); source.symlink_to(case["weights"][1])
    elif fault == "snapshot_symlink": weight.unlink(); weight.symlink_to(case["weights"][0])
    elif fault == "job_symlink":
        alias = case["tmp"] / "alias"; alias.symlink_to(case["job"], target_is_directory=True)
        case["job"] = alias
    elif fault == "output_oldjob": overrides["output"] = case["job"] / "new"
    elif fault == "output_source": overrides["output"] = case["source"] / "new"
    elif fault == "output_ddk": overrides["output"] = case["ddk"] / "new"
    elif fault == "existing_journal": (case["output"] / "retained-input-links.jsonl").write_text("old partial\n")
    with pytest.raises((ValueError, OSError)):
        _prepare(case, **overrides)


def test_weight_payload_hash_is_freshly_checked_before_link_even_with_matching_stats(case, monkeypatch):
    plan = _prepare(case)
    source, old = case["weights"][0], case["rows"][1]
    original = retry._hash_file
    def mismatch(path):
        digest, identity = original(path)
        return ("f" * 64 if Path(path) == case["compilation"] / old["path"] else digest), identity
    monkeypatch.setattr(retry, "_hash_file", mismatch)
    with pytest.raises(ValueError, match="SHA256"):
        plan.snapshot(source, case["output"] / old["path"], _normal)
    assert not (case["output"] / old["path"]).exists()
    assert (case["compilation"] / old["path"]).stat().st_nlink == 1
    assert plan.audit()["status"] == "FAILED"


def test_cross_filesystem_link_failure_has_no_copy_fallback_or_cleanup(case, monkeypatch):
    plan = _prepare(case)
    def no_link(*a, **k): raise OSError(errno.EXDEV, "cross-device fixture")
    monkeypatch.setattr(retry.os, "link", no_link)
    with pytest.raises(OSError, match="cross-device"):
        plan.snapshot(case["weights"][0], case["output"] / case["rows"][1]["path"],
                      lambda *a: pytest.fail("payload must not copy"))
    events = [json.loads(line) for line in plan.journal.read_text().splitlines()]
    assert events[-2]["event"] == "LINK_INTENT" and events[-1]["event"] == "SNAPSHOT_FAILED"
    assert plan.audit()["status"] == "FAILED"


def test_post_link_exception_preserves_link_and_flushed_intent(case, monkeypatch):
    plan = _prepare(case)
    source, row = case["weights"][0], case["rows"][1]
    destination = case["output"] / row["path"]
    original = retry._identity
    def changed(path):
        if Path(path) == destination and destination.exists():
            raise ValueError("injected post-link observation failure")
        return original(path)
    monkeypatch.setattr(retry, "_identity", changed)
    with pytest.raises(ValueError, match="post-link"):
        plan.snapshot(source, destination, _normal)
    assert destination.exists() and destination.stat().st_nlink == 2
    assert json.loads(plan.journal.read_text().splitlines()[1])["event"] == "LINK_INTENT"
    assert plan.evidence["snapshot_attempts"][0]["link_created"] is True
    assert plan.audit()["status"] == "FAILED"


@pytest.mark.parametrize("change", ["shared_payload", "chmod", "unlink_alias", "new_alias", "source",
                                   "old_report", "diagnosis", "omg", "air_copy", "journal"])
def test_final_audit_detects_mutations_and_keeps_all_existing_evidence(case, change):
    plan = _prepare(case)
    _snapshots(case, plan)
    old_weight = case["compilation"] / case["rows"][1]["path"]
    alias = case["output"] / case["rows"][1]["path"]
    if change == "shared_payload":
        old_weight.chmod(0o644)
        old_weight.write_bytes(b"corrupted shared payload by privileged writer")
    elif change == "chmod": alias.chmod(0o644)
    elif change == "unlink_alias": alias.unlink()
    elif change == "new_alias": os.link(alias, case["tmp"] / "external-link")
    elif change == "source": case["weights"][0].write_bytes(b"changed staged source")
    elif change == "old_report": (case["compilation"] / "report.json").write_text("{}")
    elif change == "diagnosis": (case["diagnosis"] / "report.json").write_text("{}")
    elif change == "omg": case["omg"].write_bytes(b"changed tool")
    elif change == "air_copy":
        copied_air = case["output"] / case["rows"][0]["path"]
        copied_air.chmod(0o644)
        copied_air.write_bytes(b"changed AIR")
    elif change == "journal": plan.journal.write_text("tampered journal\n")
    result = plan.audit()
    assert result["status"] == "FAILED" and result["audit_errors"]
    assert old_weight.exists()
    if change != "unlink_alias": assert alias.exists()
    assert (case["compilation"] / "report.json").exists()
    assert result["device_execution"] == "NOT_RUN"
    if change == "shared_payload":
        assert result["audit_checks"][1]["payload_verification"] == "FRESH_SHA256_DIFFERENT"


def test_second_retry_is_blocked_after_first_attempt_created_payload_aliases(case):
    plan = _prepare(case)
    _snapshots(case, plan)
    another = case["tmp"] / "another-job/compilation"
    another.mkdir(parents=True)
    with pytest.raises(ValueError, match="stat changed|shared|independent"):
        _prepare(case, output=another)


def test_duplicate_source_destination_override_and_existing_partial_are_rejected(case):
    plan = _prepare(case)
    source, row = case["weights"][0], case["rows"][1]
    with pytest.raises(ValueError, match="relative path"):
        plan.snapshot(source, case["output"] / "wrong-name", _normal)
    destination = case["output"] / row["path"]
    plan.snapshot(source, destination, _normal)
    with pytest.raises(ValueError, match="already attempted"):
        plan.snapshot(source, destination, _normal)


def test_no_tbe_is_allowed_only_when_both_old_and_new_have_none(case):
    case["report"].pop("tbe_compatibility")
    _save(case)
    plan = _prepare(case, tbe_diagnosis=None)
    _snapshots(case, plan)
    assert plan.audit()["status"] == "PASSED"


def test_air_callback_cannot_return_a_shared_old_snapshot(case):
    plan = _prepare(case)
    source, old = case["air"], case["rows"][0]
    def shared(source, destination):
        os.link(case["compilation"] / old["path"], destination)
        return {"source": str(source), "source_stat": _stat(source), "bytes": old["bytes"],
                "sha256": old["sha256"], "snapshot_stat": _stat(destination), "snapshot_method": "COPY"}
    with pytest.raises(ValueError, match="independent"):
        plan.snapshot(source, case["output"] / old["path"], shared)
    assert plan.audit()["status"] == "FAILED"


def test_link_intent_is_fsynced_before_os_link_and_failed_increment_is_recorded(case, monkeypatch):
    plan = _prepare(case)
    original_link = retry.os.link
    real_fsync = retry.os.fsync
    synced = []
    def fsync(descriptor):
        synced.append(True)
        return real_fsync(descriptor)
    def unexpected_alias(source, destination, **kwargs):
        assert synced, "intent must be fsynced before creating the link"
        assert json.loads(plan.journal.read_text().splitlines()[-1])["event"] == "LINK_INTENT"
        original_link(source, destination, **kwargs)
        original_link(case["compilation"] / case["rows"][1]["path"], case["tmp"] / "concurrent-alias")
    monkeypatch.setattr(retry.os, "fsync", fsync)
    monkeypatch.setattr(retry.os, "link", unexpected_alias)
    destination = case["output"] / case["rows"][1]["path"]
    with pytest.raises(ValueError, match="unexpected"):
        plan.snapshot(case["weights"][0], destination, _normal)
    events = [json.loads(line) for line in plan.journal.read_text().splitlines()]
    assert events[-2]["event"] == "LINK_CREATED" and events[-2]["after"]["nlink"] == 3
    assert events[-1]["event"] == "SNAPSHOT_FAILED"
    assert destination.exists() and plan.audit()["status"] == "FAILED"


def test_existing_partial_destination_is_preserved_without_source_mutation(case):
    plan = _prepare(case)
    destination = case["output"] / case["rows"][1]["path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"retained partial output")
    with pytest.raises(ValueError, match="already exists"):
        plan.snapshot(case["weights"][0], destination, _normal)
    assert destination.read_bytes() == b"retained partial output"
    assert (case["compilation"] / case["rows"][1]["path"]).stat().st_nlink == 1


def test_journal_write_failure_prevents_link_and_preserves_failure_evidence(case, monkeypatch):
    plan = _prepare(case)
    def full(*a): raise OSError(errno.ENOSPC, "journal disk full fixture")
    monkeypatch.setattr(retry.os, "fsync", full)
    destination = case["output"] / case["rows"][1]["path"]
    with pytest.raises(OSError, match="journal disk full"):
        plan.snapshot(case["weights"][0], destination, _normal)
    assert not destination.exists()
    assert plan.evidence["snapshot_attempts"][0]["status"] == "FAILED"
    assert plan.audit()["status"] == "FAILED"


@pytest.fixture
def static_case(case):
    """A complete first-static failure built through the actual sharing API."""
    plan = _prepare(case)
    copied = _snapshots(case, plan)
    for row, original in zip(copied, case["rows"]):
        row.update({key: original[key] for key in ("path", "source_relative", "role")})
    audit = json.loads(json.dumps(plan.audit()))
    assert audit["status"] == "PASSED"
    original_job = case["job"]
    current_job, current_compilation = case["output"].parent, case["output"]
    report = json.loads(json.dumps(case["report"]))
    report.update(output=str(current_compilation), input_snapshots=copied,
                  requested_options=retry._options(case["profile"], static=True),
                  retained_input_reuse=audit, returncode=-15)
    outer = dict(case["outer"], output=str(current_job), retained_failed_job=str(original_job),
        input_snapshot_reuse={"status": "PASSED", "explicit_weight_copy_bytes": 0,
            "report": str(current_compilation / "retained-input-reuse.json")})
    _write(current_job / "compile-profile.json", get_existing_air_profile("cpro-40block-prefill-64"))
    _write(current_compilation / "retained-input-reuse.json", audit)
    _write(current_compilation / "input-inventory.json", {"files": copied})
    command = json.loads((case["compilation"] / "compile-command.json").read_text())
    command["argv"] = command["argv"][:6]
    command["argv"][2] = "--model=" + str(current_compilation / copied[0]["path"])
    command["argv"].extend(f"--{key}={value}" for key, value in report["requested_options"].items())
    command["cwd"] = str(current_compilation / "compile-work/generated")
    _write(current_compilation / "compile-command.json", command)
    _write(current_compilation / "tbe-compatibility-result.json", report["tbe_compatibility"])
    output = case["tmp"] / "successor-job/compilation"
    output.mkdir(parents=True)
    current = dict(case, ancestor_job=original_job, ancestor_compilation=case["compilation"],
        job=current_job, compilation=current_compilation, generated=current_compilation / "compile-work/generated",
        rows=copied, report=report, outer=outer, audit=audit, output=output)
    _save(current)
    return current


def _save_static_audit(case):
    case["report"]["retained_input_reuse"] = case["audit"]
    _write(case["compilation"] / "retained-input-reuse.json", case["audit"])
    _save(case)


def test_audited_static_successor_preserves_both_prior_aliases_and_all_receipt_bytes(static_case):
    c = static_case
    old = {path: path.read_bytes() for job in (c["job"], c["ancestor_job"])
           for path in job.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl"}}
    sources = {path: retry._identity(path) for path in c["selected"]}
    plan = _prepare(c)
    rows = _snapshots(c, plan)
    audit = plan.audit()
    assert audit["status"] == "PASSED", audit
    assert audit["reuse_lineage"] == "AUDITED_STATIC_SUCCESSOR"
    assert audit["owner_chain"]["original_failed_job"] == str(c["ancestor_job"])
    assert audit["explicit_weight_copy_bytes"] == 0
    assert {path: path.read_bytes() for path in old} == old
    assert {path: retry._identity(path) for path in sources} == sources
    for current in rows[1:]:
        name = Path(current["source"]).relative_to(c["source"])
        paths = [job / "compilation/compile-work/generated" / name
                 for job in (c["ancestor_job"], c["job"], c["output"].parent)]
        assert len({path.stat().st_ino for path in paths}) == 1
        assert all(path.stat().st_nlink == 3 for path in paths)
    assert audit["owner_chain"]["prior_link_journal"]["sha256"] == _sha(c["compilation"] / "retained-input-links.jsonl")


@pytest.mark.parametrize("fault", ["audit_failed", "audit_missing", "audit_mismatch", "chain_depth",
    "wrong_lineage", "missing_completed", "duplicate_attempt", "bad_check", "wrong_owner_path",
    "wrong_before_nlink", "bool_before_nlink", "journal_missing", "journal_changed", "journal_rebound_but_wrong_events", "journal_rebound_duplicate_key",
    "original_changed", "original_result_changed", "original_snapshot_missing", "third_alias",
    "writable_alias", "outer_missing_binding", "old_tool_binding", "wrong_audit_options"])
def test_static_successor_rejects_any_gap_in_the_first_ownership_transition(static_case, fault):
    c, audit = static_case, static_case["audit"]
    journal = c["compilation"] / "retained-input-links.jsonl"
    if fault == "audit_failed": audit["status"] = "FAILED"
    elif fault == "audit_missing": pass
    elif fault == "audit_mismatch": pass
    elif fault == "chain_depth": audit["owner_chain"] = {"unexpected": "a second successor"}
    elif fault == "wrong_lineage": audit["reuse_lineage"] = "AUDITED_STATIC_SUCCESSOR"
    elif fault == "missing_completed": audit["completed_inputs"] -= 1
    elif fault == "duplicate_attempt": audit["snapshot_attempts"][1] = dict(audit["snapshot_attempts"][0])
    elif fault == "bad_check": audit["audit_checks"][1]["observed_sha256"] = "f" * 64
    elif fault == "wrong_owner_path": audit["snapshot_attempts"][1]["old_snapshot"] = str(c["weights"][0])
    elif fault == "wrong_before_nlink": audit["snapshot_attempts"][1]["before"]["nlink"] = 2
    elif fault == "bool_before_nlink": audit["snapshot_attempts"][1]["before"]["nlink"] = True
    elif fault == "journal_missing": journal.unlink()
    elif fault == "journal_changed": journal.write_text(journal.read_text() + "{}\n")
    elif fault == "journal_rebound_but_wrong_events":
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        events[1]["expected_sha256"] = "f" * 64
        journal.write_text("".join(json.dumps(event) + "\n" for event in events))
        _, identity = retry._read_small(journal, retry.MAX_METADATA_BYTES)
        audit["journal"].update(identity)
    elif fault == "journal_rebound_duplicate_key":
        data = journal.read_text().replace('{"completed_inputs":', '{"event": "ignored_duplicate", "completed_inputs":', 1)
        journal.write_text(data)
        _, identity = retry._read_small(journal, retry.MAX_METADATA_BYTES)
        audit["journal"].update(identity)
    elif fault == "original_changed": (c["ancestor_compilation"] / "report.json").write_text("{}\n")
    elif fault == "original_result_changed": (c["ancestor_job"] / "result.json").write_text("{}\n")
    elif fault == "original_snapshot_missing": (c["ancestor_compilation"] / c["rows"][1]["path"]).unlink()
    elif fault == "third_alias": os.link(c["compilation"] / c["rows"][1]["path"], c["tmp"] / "unowned-third-alias")
    elif fault == "writable_alias": (c["compilation"] / c["rows"][1]["path"]).chmod(0o644)
    elif fault == "outer_missing_binding": c["outer"].pop("retained_failed_job")
    elif fault == "old_tool_binding": audit["tool_binding"]["sha256"] = "f" * 64
    elif fault == "wrong_audit_options": audit["old_options"]["input_shape"] = "other"
    _save_static_audit(c)
    if fault == "audit_missing": (c["compilation"] / "retained-input-reuse.json").unlink()
    elif fault == "audit_mismatch": (c["compilation"] / "retained-input-reuse.json").write_text("{}\n")
    with pytest.raises((OSError, ValueError, KeyError, TypeError)):
        _prepare(c)
    assert not (c["output"] / "retained-input-links.jsonl").exists()


def test_successor_audit_detects_an_earlier_owner_alias_replaced_after_prepare(static_case):
    c = static_case
    plan = _prepare(c)
    _snapshots(c, plan)
    alias = c["ancestor_compilation"] / c["rows"][1]["path"]
    alias.unlink()
    alias.write_bytes(b"different earlier alias")
    result = plan.audit()
    assert result["status"] == "FAILED"
    assert any(error.get("path") == str(alias) for error in result["audit_errors"])

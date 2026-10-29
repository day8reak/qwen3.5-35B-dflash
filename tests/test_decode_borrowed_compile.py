"""Real tiny export/ownership receipts with only the selected OMG mocked.

These tests prove host input identity and retained-prefill resumability. They do
not execute a target model, native CCE compiler or device runtime.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import existing_air as compiler, full_target
from qwen35_moe_dflash.artifacts import save_json
from qwen35_moe_dflash.canonical_target_export import canonicalize_target_export
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from test_static_target_successor_retry import chain as chain_fixture, _successor


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_nlink)


def _outer(job, air):
    save_json(job / "result.json", {
        "artifact_kind": "qwen35-original-target-job", "status": "COMPILED",
        "model_compilation": "PASSED", "air": str(air), "output": str(job),
        "input_layout": "air-directory", "compilation": "compilation/report.json",
        "compilation_report_sha256": _sha(job / "compilation/report.json")})


def _validate(case, job, air, profile):
    return full_target.validate_compilation(job, air=air, ddk_root=case["ddk"],
        ddk_id="STATIC_RETRY_INTEGRATION_MOCK", profile=profile,
        input_layout="air-directory", target_features=True)


def _decode_transfer(case, tmp_path):
    """Generate encoded T1 AIR and real canonical/inventory receipts."""
    original = case["source"]
    model = original["model_class"].FromString(original["air"].read_bytes())
    profile = get_existing_air_profile("cpro-40block-decode-1")
    shapes = {name: list(map(int, shape.split(","))) for name, shape in
              (entry.split(":") for entry in profile["input_shape"].split(";"))}
    source = tmp_path / "decode-origin/glm_export_air_cpro_40block_1_quant"
    source.mkdir(parents=True)
    graph = model.graph[0]
    for node in graph.op:
        if node.type in {"Data", "RefData"}:
            node.output_desc[0].shape.dim[:] = shapes[node.name]
        if node.type == "FileConstant":
            attribute = next(row for row in node.attr if row.key == "file_path")
            name = Path(attribute.value.s.decode()).name
            attribute.value.s = ("./" + source.name + "/" + name).encode()
            destination = source / name
            if not destination.exists():
                destination.write_bytes(original["weights"][name].read_bytes())
    # Source TorchAir export places features at1; canonicalization moves them
    # behind80 state outputs without changing any computation statement.
    net = next(node for node in graph.op if node.type == "NetOutput")
    references = list(net.input)
    descriptors = [row.SerializeToString() for row in net.input_desc]
    permutation = [0, 81, *range(1, 81)]
    net.input[:] = [references[index] for index in permutation]
    del net.input_desc[:]
    for index in permutation:
        net.input_desc.add().ParseFromString(descriptors[index])
    net.input_desc[1].shape.dim[1] = 1
    air = source / "export.air"
    air.write_bytes(model.SerializeToString())
    transfer = case["run"] / "target-pytorch-decode1-transfer"
    result = canonicalize_target_export(air, transfer, torchair_pb2=original["schema"],
        sequence_length=1, hash_weights=True)
    assert result["status"] == "EXPORT_READY", result.get("error")
    return {"air": transfer / "export.air", "inventory": transfer / "inventory/inventory.json",
            "receipt": transfer / "result.json", "profile": profile}


@pytest.fixture
def completed(tmp_path, monkeypatch):
    normal_snapshot = compiler._snapshot
    case = chain_fixture.__wrapped__(tmp_path, monkeypatch)
    job, profile, report, _, _, _ = _successor(case, monkeypatch)
    assert report["status"] == "COMPILED", report.get("error")
    _outer(job, case["air"])
    monkeypatch.setattr(compiler, "_snapshot", normal_snapshot)
    assert _validate(case, job, case["air"], profile)["status"] == "COMPILED"
    transfer = _decode_transfer(case, tmp_path)
    return dict(case, prefill=job, prefill_profile=profile,
                prefill_report=report, transfer=transfer)


def _compile(case, monkeypatch, *, tamper=None, profile_name="cpro-40block-decode-1"):
    transfer = case["transfer"]
    profile = get_existing_air_profile(profile_name)
    job = case["run"] / "target-pytorch-decode1-compile"
    job.mkdir()
    save_json(job / "compile-profile.json", profile)
    normal_snapshot = compiler._snapshot
    copied, calls, aliases = [], [], []

    def only_air(source, destination):
        assert source == transfer["air"], "decode must not copy a weight payload"
        copied.append(source)
        return normal_snapshot(source, destination)

    def no_link(*args, **kwargs):
        raise AssertionError("decode must not mutate prefill source inode link counts")

    monkeypatch.setattr(compiler, "_snapshot", only_air)
    monkeypatch.setattr(os, "link", no_link)
    if tamper == "alias_create":
        normal_symlink = os.symlink
        created = []
        def fail_second_alias(*args, **kwargs):
            if len(created) == 1:
                raise OSError("injected second decode alias creation failure")
            normal_symlink(*args, **kwargs)
            created.append(args[0])
        monkeypatch.setattr(os, "symlink", fail_second_alias)

    def omg(argv, **kwargs):
        calls.append(argv)
        if "--help" in argv:
            kwargs["stdout"].write("OMG --input_shape --limit_partition_size --save_weights_as_external_data\n")
            return subprocess.CompletedProcess(argv, 0)
        cwd = kwargs["cwd"]
        assert not any(value.startswith("--dynamic_dims") for value in argv)
        prefix = cwd / "glm_export_air_cpro_40block_1_quant"
        assert not prefix.is_symlink()
        for row in case["prefill_report"]["input_snapshots"]:
            if row["role"] == "air":
                continue
            target = case["prefill"] / "compilation" / row["path"]
            alias = prefix / target.name
            assert alias.is_symlink() and alias.resolve() == target
            assert not target.stat().st_mode & 0o222
            assert _sha(alias) == row["sha256"]
            aliases.append(alias)
        assert len(aliases) == 2
        if tamper == "alias":
            aliases[0].unlink()
            aliases[0].symlink_to(aliases[1].resolve())
        elif tamper in {"source_mode", "source_write"}:
            target = aliases[0].resolve()
            target.chmod(0o644)
            if tamper == "source_write":
                target.write_bytes(b"unexpected privileged compiler write")
        elif tamper == "extra_alias":
            (cwd / "unexpected-input-link").symlink_to(aliases[0].resolve())
        model = cwd / "model"
        model.mkdir()
        (model / "model.omc").write_bytes(b"mock static decode1 OMC")
        (model / "SubGraph_0.weight").write_bytes(b"mock independent decode compiled external data")
        kwargs["stdout"].write("Host simulation complete; no native compiler or model execution\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", omg)
    report = compiler.compile_existing_air(transfer["air"], job / "compilation", case["ddk"],
        ddk_id="STATIC_RETRY_INTEGRATION_MOCK", input_shape=profile["input_shape"],
        dynamic_dims=profile["dynamic_dims"], limit_partition_size=profile["limit_partition_size"],
        save_weights_as_external_data=profile["save_weights_as_external_data"],
        input_layout="air-directory", borrowed_decode_job=case["prefill"],
        decode_inventory=transfer["inventory"], decode_export_receipt=transfer["receipt"])
    return job, profile, report, copied, calls, aliases


def test_decode_view_compiles_without_payload_copy_or_prefill_inode_changes(completed, monkeypatch):
    case = completed
    old_paths = [Path(row["source"]) for row in case["prefill_report"]["input_snapshots"]]
    old_paths += [case["prefill"] / "compilation" / row["path"]
                  for row in case["prefill_report"]["input_snapshots"]]
    stamps = {path: _identity(path) for path in old_paths}
    metadata = {path: path.read_bytes() for folder in (case["failed"], case["previous"], case["prefill"])
                for path in folder.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl", ".txt", ".log"}}
    job, profile, report, copied, calls, aliases = _compile(case, monkeypatch)
    assert report["status"] == "COMPILED", report.get("error")
    assert copied == [case["transfer"]["air"]] and len(calls) == 2
    assert report["decode_input_view"]["status"] == "PASSED"
    assert report["decode_input_view"]["explicit_weight_copy_bytes"] == 0
    assert all(row["snapshot_method"] == "READONLY_SOURCE_REFERENCE_SYMLINK"
               for row in report["input_snapshots"] if row["role"] != "air")
    assert {path: _identity(path) for path in stamps} == stamps
    assert {path: path.read_bytes() for path in metadata} == metadata
    assert _validate(case, case["prefill"], case["air"], case["prefill_profile"])["status"] == "COMPILED"
    _outer(job, case["transfer"]["air"])
    assert _validate(case, job, case["transfer"]["air"], profile)["status"] == "COMPILED"
    with pytest.raises(ValueError, match="symlink"):
        full_target._recorded_file(job / "compilation", next(row for row in report["input_snapshots"]
            if row["role"] != "air"), "snapshot_stat")
    aliases[0].unlink()
    aliases[0].symlink_to(aliases[1].resolve())
    with pytest.raises(ValueError):
        _validate(case, job, case["transfer"]["air"], profile)


@pytest.mark.parametrize("tamper", ["alias", "source_mode", "source_write", "extra_alias"])
def test_compiler_alias_or_source_mutation_fails_and_preserves_every_artifact(completed, monkeypatch, tamper):
    job, _, report, copied, calls, aliases = _compile(completed, monkeypatch, tamper=tamper)
    assert len(calls) == 2 and copied == [completed["transfer"]["air"]]
    assert report["status"] == "FAILED"
    assert (job / "compilation/failure.json").is_file()
    assert (job / "compilation/decode-input-view.json").is_file()
    assert (job / "compilation/compile-work/generated/model/model.omc").read_bytes() == b"mock static decode1 OMC"
    assert all(alias.is_symlink() for alias in aliases)
    if tamper in {"source_mode", "source_write"}:
        assert aliases[0].stat().st_mode & 0o222
    if tamper == "source_write":
        assert aliases[0].read_bytes() == b"unexpected privileged compiler write"


@pytest.mark.parametrize("fault", ["decode_profile", "missing_prefill_result", "changed_triplet", "source_writable"])
def test_decode_preflight_rejects_unbound_inputs_before_any_omg_help(completed, monkeypatch, fault):
    if fault == "missing_prefill_result":
        (completed["prefill"] / "result.json").unlink()
    elif fault == "changed_triplet":
        completed["transfer"]["receipt"].write_text("{}")
    elif fault == "source_writable":
        row = next(row for row in completed["prefill_report"]["input_snapshots"] if row["role"] != "air")
        (completed["prefill"] / "compilation" / row["path"]).chmod(0o644)
    job, _, report, copied, calls, _ = _compile(completed, monkeypatch,
        profile_name="cpro-40block-prefill-64" if fault == "decode_profile" else "cpro-40block-decode-1")
    assert report["status"] == "FAILED" and report["stage"] == "decode-input-validation"
    assert calls == copied == [] and not report["input_snapshots"]
    assert not (job / "compilation/help-command.json").exists()


def test_partial_alias_failure_retains_owned_aliases_and_preserves_prefill(completed, monkeypatch):
    job, _, report, copied, calls, _ = _compile(completed, monkeypatch, tamper="alias_create")
    assert len(calls) == 1 and "--help" in calls[0]
    assert copied == [completed["transfer"]["air"]]
    assert report["status"] == "FAILED" and report["stage"] == "input-snapshot"
    assert report["model_compilation"] == "NOT_RUN"
    assert report["decode_input_view"]["status"] == "FAILED"
    assert report["decode_input_view"]["audit_errors"]
    prefix = job / "compilation/compile-work/generated/glm_export_air_cpro_40block_1_quant"
    assert len(list(prefix.iterdir())) == 1 and next(prefix.iterdir()).is_symlink()
    assert (job / "compilation/compile-work/generated/export.air").is_file()
    assert (job / "compilation/decode-input-view-plan.json").is_file()
    assert (job / "compilation/decode-input-view.json").is_file()
    assert _validate(completed, completed["prefill"], completed["air"],
                     completed["prefill_profile"])["status"] == "COMPILED"


@pytest.mark.parametrize("fault", ["audit", "plan", "generated_symlink", "same_target_alias_replacement"])
def test_decode_continuation_rejects_changed_owned_proofs_or_unexpected_links(completed, monkeypatch, fault):
    job, profile, report, _, _, aliases = _compile(completed, monkeypatch)
    assert report["status"] == "COMPILED", report.get("error")
    _outer(job, completed["transfer"]["air"])
    assert _validate(completed, job, completed["transfer"]["air"], profile)["status"] == "COMPILED"
    if fault in {"audit", "plan"}:
        path = job / "compilation" / ("decode-input-view.json" if fault == "audit" else "decode-input-view-plan.json")
        value = json.loads(path.read_text())
        value["unbound_edit"] = True
        path.write_text(json.dumps(value))
    elif fault == "generated_symlink":
        (job / "compilation/compile-work/generated/rogue-output").symlink_to(aliases[0].resolve())
    else:
        target = aliases[0].resolve()
        aliases[0].unlink()
        aliases[0].symlink_to(target)
    with pytest.raises(ValueError):
        _validate(completed, job, completed["transfer"]["air"], profile)
    assert _validate(completed, completed["prefill"], completed["air"],
                     completed["prefill_profile"])["status"] == "COMPILED"

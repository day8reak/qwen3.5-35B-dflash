"""Descriptor diagnostics must never turn into a full-model retry."""
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash.hashing import sha256_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def file_record(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def bind_compilation(job, report):
    folder = job / "compilation"
    write_json(folder / "report.json", report)
    record = {"directory": str(folder), "manifest": file_record(folder / "report.json", folder)}
    state = {"schema_version": 1, "artifact_kind": "qwen35-full-draft-job",
             "status": "PREPARED", "stage": "prepared", "output": str(job),
             "components": {"compilation": record},
             "settings": {"sequence_length": 64, "valid_length": 6, "seed": 17}}
    write_json(job / "result.json", state)
    write_json(job / "job.json", state)
    write_json(job / "completed-compilation.json", record)


@pytest.fixture
def draft_job(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "127.0.0.1:8710")
    monkeypatch.delenv("BOARD_HIAI_LIB_DIR", raising=False)
    monkeypatch.delenv("HIAI_RUNTIME_VERSION", raising=False)
    monkeypatch.delenv("CPRO_DDK_ID", raising=False)
    job = tmp_path / "full-draft"
    compiler = job / "compilation"
    model = compiler / "compile-work/generated/model/model.omc"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"small external-data model; no actual HIAI execution")
    report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED",
              "model_compilation": "PASSED", "ddk_id": "fixture-ddk",
              "requested_options": {"save_weights_as_external_data": "true"},
              "omc": file_record(model, compiler),
              "external_data_candidates": [{"path": "compile-work/generated/model/SubGraph_1.weight",
                                            "bytes": 1151486976, "sha256": "1" * 64}]}
    # The giant sidecar deliberately does not exist: this probe needs only OMC bytes.
    report["generated_artifacts"] = [report["omc"], *report["external_data_candidates"]]
    bind_compilation(job, report)
    return job, model, report


def test_selects_bound_small_main_omc_without_opening_sidecars(draft_job):
    from qwen35_moe_dflash.descriptor_probe import select_model
    job, model, _ = draft_job
    selected, identity = select_model(job)
    assert selected == model
    assert sha256_file(selected) in json.dumps(identity)
    assert not model.with_name("SubGraph_1.weight").exists()


@pytest.mark.parametrize("fault", ["manifest", "model", "embedded", "failed"])
def test_rejects_changed_identity_or_nonexternal_compilation(draft_job, fault):
    from qwen35_moe_dflash.descriptor_probe import select_model
    job, model, report = draft_job
    if fault == "manifest":
        (job / "compilation/report.json").write_text("{}")
    elif fault == "model":
        model.write_bytes(b"different model bytes")
    else:
        if fault == "embedded":
            report["requested_options"]["save_weights_as_external_data"] = "false"
        else:
            report["status"] = "FAILED"
        bind_compilation(job, report)
    with pytest.raises((ValueError, RuntimeError)):
        select_model(job)


def test_rejects_large_main_omc_from_metadata_before_hashing(draft_job, monkeypatch):
    from qwen35_moe_dflash import descriptor_probe as probe
    job, model, report = draft_job
    report["omc"]["bytes"] = (64 << 20) + 1
    bind_compilation(job, report)
    original = probe.sha256_file

    def no_large_hash(path):
        if Path(path) == model:
            pytest.fail("oversized OMC was opened before the size limit check")
        return original(path)

    monkeypatch.setattr(probe, "sha256_file", no_large_hash)
    with pytest.raises((ValueError, RuntimeError)):
        probe.select_model(job)


def fake_transport(monkeypatch, tmp_path, *, native_change=None, remote_exitcode=0,
                   missing=None, transport_failure=False):
    from qwen35_moe_dflash import descriptor_probe as probe
    calls, uploaded = [], []
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    library_names = ("libhiai.so", "libhiai_hcl_model_runtime.so",
                     "libhiai_hcl_model_runtime_impl.so", "libc++_shared.so")
    for name in library_names:
        (libraries / name).write_bytes(name.encode())

    def build_runner(out, timeout):
        target = out / "native/build/qwen35_dflash_hiai"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"mock native descriptor probe")
        return target

    class FakeHdc:
        def __init__(self, executable, sn, bip, out, timeout):
            self.out, self.timeout, self.count = out, timeout, 0
            self.board = None

        def shell(self, command, label, **kwargs):
            self.count += 1
            calls.append((label, command))
            assert "--draft-step" not in command and "--stateful-step" not in command
            assert "omg" not in command
            if label == "observe-descriptor":
                assert "--probe-model-description" in command
                if transport_failure:
                    raise RuntimeError("injected HDC disconnect")
                return remote_exitcode, ""
            return 0, ""

    def upload(hdc, board, entries, checksum, **kwargs):
        hdc.board = board
        uploaded.extend(entries)
        for row in entries:
            assert sha256_file(Path(row["local"])) == row["sha256"]

    def receive(hdc, board, out, name):
        if name == missing:
            return None
        path = out / name
        if name == "probe.json":
            model = next(row for row in uploaded if row["remote"] == "model.omc")
            remote_path = str(board / "model.omc")

            def step(operation, model_path, buffer):
                return {"operation": operation, "setter_status": 0, "model_path": model_path,
                        "model_bytes": model["bytes"] if buffer else 0,
                        "buffer_present": buffer, "buffer_matches_input": buffer}

            native = {"schema_version": 1, "artifact_kind": "qwen35-hiai-model-description-probe",
                      "status": "INSPECTED", "model_sha256": model["sha256"], "model_bytes": model["bytes"],
                      "model_path": remote_path, "runtime_version": "fake-runtime", "process_id": 42,
                      "evidence_kind": "device-hiai",
                      "load_called": False, "process_called": False,
                      "observations": [
                          {"order": "path-then-buffer", "steps": [step("SetModelPath", remote_path, False),
                              step("SetModelBuffer", "", True)]},
                          {"order": "buffer-then-path", "steps": [step("SetModelBuffer", "", True),
                              step("SetModelPath", remote_path, False)]}]}
            if native_change:
                native_change(native)
            write_json(path, native)
        elif name == "probe.exitcode":
            path.write_text(str(remote_exitcode) + "\n")
        else:
            path.write_text("mock descriptor stderr\n")
        return sha256_file(path)

    monkeypatch.setattr(probe, "Hdc", FakeHdc)
    monkeypatch.setattr(probe, "_build_runner", build_runner)
    monkeypatch.setattr(probe, "_upload", upload)
    monkeypatch.setattr(probe, "_receive", receive)
    monkeypatch.setattr(probe, "discover_libraries", lambda *a: {
        "libraries": [libraries / name for name in library_names], "metadata": {"mode": "host-test"}})
    monkeypatch.setattr(probe, "dynamic_info", lambda *a: {
        "needed": [], "soname": None, "rpath": None, "runpath": None})
    return calls, uploaded


def test_probe_observes_cleared_properties_without_loading_or_uploading_weights(draft_job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import descriptor_probe as probe
    job, model, _ = draft_job
    before = (model.read_bytes(), model.stat().st_ino)
    calls, uploads = fake_transport(monkeypatch, tmp_path)
    result = probe.run_probe(job)
    assert result["status"] == "INSPECTED"
    assert result["model_loading"] == "NOT_RUN" and result["model_execution"] == "NOT_RUN"
    assert result["observations"][0]["steps"][1]["model_path"] == ""
    assert result["observations"][1]["steps"][1]["buffer_present"] is False
    assert result["board_cleanup"] == "REMOVED"
    assert {r["remote"] for r in uploads} == {"model.omc", "runner", "lib/libhiai.so", "lib/libc++_shared.so"}
    assert sum(label == "observe-descriptor" for label, _ in calls) == 1
    assert all("SubGraph_" not in command and "bundle" not in command for _, command in calls)
    assert (model.read_bytes(), model.stat().st_ino) == before
    assert json.loads((job / "load-diagnostic/result.json").read_text()) == result


@pytest.mark.parametrize("fault", ["hash", "size", "load", "process", "missing-step", "runtime", "host-fake"])
def test_native_mismatch_never_becomes_inspected(draft_job, tmp_path, monkeypatch, fault):
    from qwen35_moe_dflash import descriptor_probe as probe
    job, model, _ = draft_job

    def change(native):
        if fault == "hash":
            native["model_sha256"] = "0" * 64
        elif fault == "size":
            native["model_bytes"] += 1
        elif fault == "load":
            native["load_called"] = True
        elif fault == "process":
            native["process_called"] = True
        elif fault == "missing-step":
            native["observations"][1]["steps"].pop()
        elif fault == "host-fake":
            native["evidence_kind"] = "host-fake-hiai"
        else:
            native["runtime_version"] = "wrong-runtime"

    if fault == "runtime":
        monkeypatch.setenv("HIAI_RUNTIME_VERSION", "fake-runtime")
    calls, _ = fake_transport(monkeypatch, tmp_path, native_change=change)
    result = probe.run_probe(job)
    assert result["status"] == "FAILED" and result["validation_error"]
    assert result["board_cleanup"] == "RETAINED"
    assert not any(label == "cleanup-board" for label, _ in calls)
    assert model.exists()


@pytest.mark.parametrize("fault", ["nonzero", "missing-json", "disconnect"])
def test_transport_failure_retains_evidence_and_does_not_retry_model(draft_job, tmp_path, monkeypatch, fault):
    from qwen35_moe_dflash import descriptor_probe as probe
    job, _, _ = draft_job
    calls, _ = fake_transport(monkeypatch, tmp_path, remote_exitcode=1 if fault == "nonzero" else 0,
                             missing="probe.json" if fault == "missing-json" else None,
                             transport_failure=fault == "disconnect")
    result = probe.run_probe(job)
    assert result["status"] == "FAILED"
    assert result["board_cleanup"] == "RETAINED"
    assert result["validation_error"]
    assert sum(label == "observe-descriptor" for label, _ in calls) == 1
    assert (job / "load-diagnostic/probe.stderr").read_text() == "mock descriptor stderr\n"
    if fault == "disconnect":
        assert "disconnect" in result["error"]
    if fault == "missing-json":
        assert result["retrieval_errors"][0]["path"] == "probe.json"


def test_rerun_replaces_only_owned_diagnostic_and_preserves_compilation(draft_job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import descriptor_probe as probe
    job, model, _ = draft_job
    calls, _ = fake_transport(monkeypatch, tmp_path)
    first = probe.run_probe(job)
    stale = job / "load-diagnostic/stale.log"
    stale.write_text("previous failed diagnostic")
    old = (model.read_bytes(), model.stat().st_ino, (job / "result.json").read_bytes())
    second = probe.run_probe(job)
    assert first["status"] == second["status"] == "INSPECTED"
    assert not stale.exists()
    assert (model.read_bytes(), model.stat().st_ino, (job / "result.json").read_bytes()) == old
    assert sum(label == "observe-descriptor" for label, _ in calls) == 2


def test_unrecognized_diagnostic_directory_is_not_removed(draft_job):
    from qwen35_moe_dflash import descriptor_probe as probe
    job, _, _ = draft_job
    out = job / "load-diagnostic"
    out.mkdir()
    note = out / "user-note"
    note.write_text("keep")
    with pytest.raises(ValueError, match="unrecognized"):
        probe.run_probe(job)
    assert note.read_text() == "keep"

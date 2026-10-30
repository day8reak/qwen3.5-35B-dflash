"""Host-only protocol tests; simulated HDC results are not board evidence."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import numpy as np
import pytest

from qwen35_moe_dflash import real_feature_draft_board as board
from qwen35_moe_dflash.hashing import sha256_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(source, **extra):
    return {**extra, "bytes": source.stat().st_size, "sha256": sha256_file(source)}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "fixture:8790")
    monkeypatch.setenv("BOARD_HIAI_LIB_DIR", "/data/local/tmp/fixture_hiai")
    for name in ("CPRO_DDK_ID", "HIAI_RUNTIME_VERSION", "FIRMWARE_ID"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "prepared"
    root.mkdir()
    model = tmp_path / "original/model.omc"
    model.parent.mkdir()
    model.write_bytes(b"host-only-small-model-fixture")
    inputs = []
    for index, (name, dtype, shape, value) in enumerate([
        ("target_features", "float16", [1, 64, 16384], .25),
        ("input_ids", "int32", [1, 64], 220),
        ("attention_mask", "int32", [1, 64], 1)]):
        path = root / f"inputs/input_{index:04d}.bin"
        path.parent.mkdir(exist_ok=True)
        np.full(shape, value, dtype="<f2" if dtype == "float16" else "<i4").tofile(path)
        inputs.append(record(path, index=index, name=name, dtype=dtype, shape=shape,
                             path=path.relative_to(root).as_posix()))
    abi = root / "draft-abi.json"
    abi.write_bytes(b'{"test_only":true}')
    step = root / "step.tsv"
    step.write_bytes(b"test-only-remote-wire\n")
    manifest = {"schema_version": 1, "artifact_kind": "qwen35-real-feature-draft-step",
        "status": "REAL_FEATURE_DRAFT_STEP_PREPARED", "sequence_length": 64,
        "ddk_id": "fixture-ddk", "runtime_version": "fixture-runtime", "firmware_id": "fixture-firmware",
        "draft_config": {"vocab_size": 248320}, "model": record(model, path="model/model.omc"),
        "model_source": record(model, path=str(model)),
        "inputs": inputs, "outputs": [
            {"index": 0, "name": "draft_top1", "dtype": "int32", "shape": [1, 15]},
            {"index": 1, "name": "draft_hidden", "dtype": "float16", "shape": [1, 15, 2048]}],
        "draft_abi": record(abi, path="draft-abi.json"), "step_plan": record(step, path="step.tsv")}
    manifest["upload_payloads"] = [{**row, "local_path": str(root / row["path"]), "role": "small-derived"}
        for row in [*inputs, manifest["draft_abi"], manifest["step_plan"]]]
    write_json(root / "manifest.json", manifest)

    def validate(path):
        observed = json.loads((Path(path) / "manifest.json").read_text())
        for row in [observed["model_source"], *observed["upload_payloads"]]:
            source = Path(row.get("local_path", row["path"]))
            if source.stat().st_size != row["bytes"] or sha256_file(source) != row["sha256"]:
                raise ValueError("test source mutated")
        return observed

    from qwen35_moe_dflash import real_feature_draft
    monkeypatch.setattr(real_feature_draft, "validate_real_feature_draft", validate)
    runner = tmp_path / "runner"
    runner.write_bytes(b"host-only-runner-fixture")
    monkeypatch.setattr(board, "_build_runner", lambda out, timeout: runner)
    from qwen35_moe_dflash import draft_hilog
    monkeypatch.setattr(draft_hilog, "capture_draft_hilog", lambda *a, **kw: {"status": "HOST_FAKE_CAPTURE"})
    return root, manifest, validate


def native_result(manifest, *, pid=1234):
    return {"schema_version": 1, "scope": "standalone-draft-step", "status": "EXECUTED",
        "process_id": pid, "evidence_kind": "device-hiai", "model_load_api": "legacy-buffer",
        "runtime_version": manifest["runtime_version"], "firmware_id": manifest["firmware_id"],
        "ddk_id": manifest["ddk_id"], "target_calls": 0, "draft_calls": 1,
        "model_sha256": manifest["model"]["sha256"], "draft_abi_sha256": manifest["draft_abi"]["sha256"],
        "step_plan_sha256": manifest["step_plan"]["sha256"], "sequence_length": 64,
        "state_policy": "prefix-recompute", "inputs": copy.deepcopy(manifest["inputs"]),
        "outputs": [], "io_diagnostic": {"enabled": True, "process_called": True,
            "output_fill_byte": 0, "input_readback_point": "AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS",
            "inputs": [{"index": row["index"], "readback_matches_source": True,
                        "readback_sha256": row["sha256"]} for row in manifest["inputs"]],
            "outputs": [{"index": index, "checked_after_process": True,
                         "object_same": True, "buffer_same": True} for index in range(2)]}}


class FakeHdc:
    """Tiny remote protocol simulator, never executes a model or shell."""
    cache_decision = "MATCHED"
    fail_label = None
    mutate_native = None
    exitcode = 0
    hidden_nonfinite = False
    invalid_top1 = False
    sent = []
    instances = []

    def __init__(self, executable, sn, bip, output, timeout):
        self.output, self.timeout, self.count = output, timeout, 0
        self.commands, self.files, self.sends = [], {}, []
        self.instances.append(self)

    def shell(self, command, label, **kwargs):
        self.count += 1
        self.commands.append((label, command))
        if label == self.fail_label:
            raise subprocess.TimeoutExpired(label, 1)
        if label == "firmware":
            return 0, "fixture-firmware"
        if label == "acquire-global-model-io-lock":
            return 0, "ACQUIRED"
        if label == "inspect-draft-model-cache":
            return 0, self.cache_decision
        if label == "draft-board-disk":
            return 0, "Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/fake 50000000 1000 49000000 1% /data"
        if label == "create-real-draft-attempt":
            self.attempt = PurePosixPath(shlex.split(command)[1])
        if label == "runtime-probe":
            self.files[str(self.attempt / "runtime.json")] = json.dumps({"npu_supported": True,
                "runtime_version": "fixture-runtime"}).encode()
            self.files[str(self.attempt / "runtime.stderr")] = b""
            self.files[str(self.attempt / "runtime.exitcode")] = b"0\n"
        if label == "execute-real-feature-draft":
            native = native_result(self.manifest)
            values = [np.arange(15, dtype="<i4").reshape(1, 15), np.full((1, 15, 2048), .5, dtype="<f2")]
            if self.hidden_nonfinite:
                values[1][0, 0, 0] = np.nan
            if self.invalid_top1:
                values[0][0, 0] = self.manifest["draft_config"]["vocab_size"]
            for descriptor, value in zip(self.manifest["outputs"], values):
                path = f"output_{descriptor['index']:04d}.bin"
                raw = value.tobytes()
                self.files[str(self.attempt / "results" / path)] = raw
                native["outputs"].append({**descriptor, "path": path, "bytes": len(raw),
                                          "sha256": hashlib.sha256(raw).hexdigest()})
            if self.mutate_native:
                self.mutate_native(native)
            self.files[str(self.attempt / "results/result.json")] = json.dumps(native).encode()
            for name, raw in {"draft.stdout": b"HOST ONLY simulated response\n", "draft.stderr": b"",
                              "draft.exitcode": f"{self.exitcode}\n".encode(), "draft.pid": b"1234\n"}.items():
                self.files[str(self.attempt / name)] = raw
            return self.exitcode, ""
        if label == "hash-result":
            path = shlex.split(command)[3].rstrip(";")
            if path not in self.files:
                return 0, "__MISSING__"
            return 0, hashlib.sha256(self.files[path]).hexdigest() + "  " + path
        return 0, ""

    def send(self, local, remote, label="send"):
        self.count += 1
        self.sends.append((str(local), str(remote)))
        self.files[str(remote)] = Path(local).read_bytes()

    def receive(self, remote, local, timeout=None):
        self.count += 1
        Path(local).write_bytes(self.files[str(remote)])


def install_hdc(monkeypatch, manifest, **options):
    class CurrentFake(FakeHdc):
        pass
    CurrentFake.instances = []
    CurrentFake.manifest = manifest
    for key, value in options.items():
        # Functions should be plain data, not bound methods.
        setattr(CurrentFake, key, staticmethod(value) if callable(value) else value)
    monkeypatch.setattr(board, "Hdc", CurrentFake)
    return CurrentFake


def test_exact_real_features_use_one_full_draft_and_no_golden_comparison(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    hdc = install_hdc(monkeypatch, manifest)
    result = board.execute_real_feature_draft(root, tmp_path / "execution")
    assert result["status"] == "REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED"
    assert result["synthetic_golden_comparison"] == "NOT_RUN"
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["draft_top1"] == [list(range(15))]
    assert result["host_model_copy_bytes"] == 0
    assert result["remote_model_preservation"] == "SAME_REGULAR_INODE_EXTENT_AND_SHA256"
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    remote = hdc.instances[0]
    execution = [command for label, command in remote.commands if label == "execute-real-feature-draft"]
    assert len(execution) == 1 and "--model-load-api legacy-buffer" in execution[0]
    assert "--output-fill-byte" not in execution[0]
    assert not any(local == manifest["model_source"]["path"] for local, _ in remote.sends)
    assert sum(label == "release-global-model-io-lock" for label, _ in remote.commands) == 2
    assert not (tmp_path / "execution/model/model.omc").exists()


def test_missing_model_streams_original_once_and_never_copies_it_on_host(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    hdc = install_hdc(monkeypatch, manifest, cache_decision="MISSING")
    result = board.execute_real_feature_draft(root, tmp_path / "upload-execution")
    assert result["status"] == "REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED"
    assert result["model_cache"]["missing_model_upload_bytes"] == manifest["model"]["bytes"]
    model_sends = [(local, remote) for local, remote in hdc.instances[0].sends
                   if local == manifest["model_source"]["path"]]
    assert len(model_sends) == 1 and model_sends[0][1].endswith("model.omc.partial")


@pytest.mark.parametrize("damage", ["model", "abi", "plan", "runtime", "firmware", "pid", "evidence",
                                   "input", "readback", "buffer", "fill", "output-name", "output-hash"])
def test_native_provenance_and_actual_output_bytes_are_required(prepared, tmp_path, monkeypatch, damage):
    root, manifest, _ = prepared
    def change(native):
        keys = {"model": "model_sha256", "abi": "draft_abi_sha256", "plan": "step_plan_sha256",
                "runtime": "runtime_version", "firmware": "firmware_id", "evidence": "evidence_kind"}
        if damage in keys: native[keys[damage]] = "wrong"
        elif damage == "pid": native["process_id"] = 999
        elif damage == "input": native["inputs"][0]["sha256"] = "0" * 64
        elif damage == "readback": native["io_diagnostic"]["inputs"][0]["readback_matches_source"] = False
        elif damage == "buffer": native["io_diagnostic"]["outputs"][0]["buffer_same"] = False
        elif damage == "fill": native["io_diagnostic"]["output_fill_byte"] = 165
        elif damage == "output-name": native["outputs"][0]["path"] = "../output.bin"
        else: native["outputs"][0]["sha256"] = "0" * 64
    install_hdc(monkeypatch, manifest, mutate_native=change)
    result = board.execute_real_feature_draft(root, tmp_path / ("bad-" + damage))
    assert result["status"] == "FAILED" and result["evidence_error"]
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"


@pytest.mark.parametrize("label", ["execute-real-feature-draft", "runtime-probe", "verify-draft-model-upload"])
def test_uncertain_remote_work_retains_locks_and_never_retries(prepared, tmp_path, monkeypatch, label):
    root, manifest, _ = prepared
    hdc = install_hdc(monkeypatch, manifest, fail_label=label, cache_decision="MISSING" if "upload" in label else "MATCHED")
    result = board.execute_real_feature_draft(root, tmp_path / ("timeout-" + label))
    assert result["status"] == "FAILED" and result["lock_state"] == "RETAINED_OR_UNCERTAIN"
    commands = hdc.instances[0].commands
    assert sum(found == label for found, _ in commands) == 1
    assert not any(found == "release-global-model-io-lock" for found, _ in commands)


def test_known_cache_mismatch_releases_locks_without_overwrite_or_model_call(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    hdc = install_hdc(monkeypatch, manifest, cache_decision="MISMATCH")
    result = board.execute_real_feature_draft(root, tmp_path / "mismatch")
    assert result["status"] == "FAILED" and result["device_execution"] == "NOT_RUN"
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    assert not any(local == manifest["model_source"]["path"] for local, _ in hdc.instances[0].sends)
    assert not any(label == "execute-real-feature-draft" for label, _ in hdc.instances[0].commands)


def test_nonfinite_hidden_is_reported_without_inventing_accuracy_acceptance(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    install_hdc(monkeypatch, manifest, hidden_nonfinite=True)
    result = board.execute_real_feature_draft(root, tmp_path / "nonfinite")
    assert result["status"] == "REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED"
    assert result["output_numerics"] == "NONFINITE_OBSERVED"
    assert result["hidden_statistics"]["nan_count"] == 1
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"


def test_nonzero_exit_cannot_be_promoted_by_executed_native_json(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    install_hdc(monkeypatch, manifest, exitcode=255)
    result = board.execute_real_feature_draft(root, tmp_path / "nonzero")
    assert result["status"] == "FAILED" and "nonzero" in result["evidence_error"]


def test_lock_release_failure_is_not_a_terminal_pass(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    install_hdc(monkeypatch, manifest, fail_label="release-global-model-io-lock")
    result = board.execute_real_feature_draft(root, tmp_path / "release-failure")
    assert result["status"] == "FAILED" and result["lock_state"] == "RETAINED_OR_UNCERTAIN"
    assert result["device_execution"] == "EXECUTED_OUTPUTS_RETRIEVED"


def test_mutated_real_input_fails_before_hdc_or_runner(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    hdc = install_hdc(monkeypatch, manifest)
    (root / manifest["inputs"][0]["path"]).write_bytes(b"changed")
    result = board.execute_real_feature_draft(root, tmp_path / "mutated")
    assert result["status"] == "FAILED" and result["hdc_invocations"] == 0
    assert hdc.instances == []


def test_out_of_range_top1_is_rejected_even_with_matching_output_hash(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    install_hdc(monkeypatch, manifest, invalid_top1=True)
    result = board.execute_real_feature_draft(root, tmp_path / "invalid-top1")
    assert result["status"] == "FAILED" and "vocabulary" in result["evidence_error"]


def test_runtime_failure_is_bound_and_preserves_native_cause(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    def fail(native):
        native.update(status="RUNTIME_FAILED", draft_calls=0, outputs=[], failure_stage="PROCESS",
                      reason="host-only simulated Process failure", runtime_stage="PROCESS",
                      expected_runtime_version=native.pop("runtime_version"))
    install_hdc(monkeypatch, manifest, mutate_native=fail, exitcode=1)
    result = board.execute_real_feature_draft(root, tmp_path / "native-failed")
    assert result["status"] == "FAILED" and result["native_status"] == "RUNTIME_FAILED"
    assert result["failure_stage"] == "PROCESS" and "simulated" in result["error"]
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"


def test_source_mutation_during_execution_demotes_observed_device_success(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    def mutate_source(native):
        (root / manifest["inputs"][0]["path"]).write_bytes(b"changed after upload")
    install_hdc(monkeypatch, manifest, mutate_native=mutate_source)
    result = board.execute_real_feature_draft(root, tmp_path / "source-race")
    assert result["status"] == "FAILED" and result["source_preservation"] == "FAILED"
    assert result["device_execution"] == "EXECUTED_OUTPUTS_RETRIEVED"


def test_remote_cached_model_postcheck_failure_demotes_success(prepared, tmp_path, monkeypatch):
    root, manifest, _ = prepared
    install_hdc(monkeypatch, manifest, fail_label="verify-cached-model-after-execution")
    result = board.execute_real_feature_draft(root, tmp_path / "cache-race")
    assert result["status"] == "FAILED" and result["remote_model_preservation"] == "FAILED"
    assert result["device_execution"] == "EXECUTED_OUTPUTS_RETRIEVED"

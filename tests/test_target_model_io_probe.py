"""Host transport fixtures; no real HIAI, target model or board is executed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import shlex
import subprocess

import pytest

from qwen35_moe_dflash import target_model_io_probe as probe
from qwen35_moe_dflash.existing_air import _stat, _conversion_options
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.hdc_run import Hdc


TOOL = "/data/local/tmp/tools_sysdbg/model_run_tool_internal"
TOOL_HASH = "a" * 64
FIRMWARE = "OpenHarmony-fixture"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeHdc:
    instances = []
    remote = {}
    fail_label = None
    timeout_label = None
    native_fault = None
    disk_available_kb = 500000
    busy = False
    busy_responses = []
    sends = []
    on_native = None
    corrupt_receive = False

    def __init__(self, executable, sn, bip, out, timeout):
        self.out, self.timeout, self.calls = out, timeout, []
        self.instances.append(self)

    def send(self, local, remote, label="send"):
        local, remote = Path(local), str(remote)
        assert local.suffix != ".weight", "restore-only must never send an external weight"
        self.sends.append((str(local), remote))
        self.remote[remote] = local.read_bytes()
        if remote.endswith("/model-files.tsv"):
            self.board = str(PurePosixPath(remote).parent)

    def receive(self, remote, local, *, timeout=None):
        value = self.remote[str(remote)]
        if self.corrupt_receive and str(remote).endswith("/model-io.json"):
            value += b"tampered"
        Path(local).write_bytes(value)

    def shell(self, command, label, *, timeout=None, allow_failure=False):
        self.calls.append((command, label, timeout))
        if label == self.timeout_label:
            raise subprocess.TimeoutExpired(["fixture-hdc"], 1)
        if label == self.fail_label:
            raise RuntimeError("fixture remote command failed")
        if label in {"device-preflight", "device-identity"}:
            return 0, TOOL_HASH + "  " + TOOL + ("\n" + FIRMWARE if label == "device-identity" else "")
        if label == "acquire-global-model-io-lock":
            if self.busy_responses:
                return 0, self.busy_responses.pop(0)
            return 0, "BUSY" if self.busy else "ACQUIRED"
        if label == "inspect-model-cache":
            manifest = self.remote[self.board + "/model-files.tsv"].decode()
            return 0, "\n".join(("OK" if path in self.remote and digest(self.remote[path]) == expected else "SEND") + " " + path
                for expected, path in (line.split(" ", 1) for line in manifest.splitlines()))
        if label == "board-disk":
            return 0, f"Filesystem 1024-blocks Used Available Capacity Mounted on\nfixture 999999 1 {self.disk_available_kb} 1% /data"
        if label == "board-memory":
            return 0, "MemAvailable: 1000000 kB\nMemTotal: 2000000 kB"
        if label == "verify-model-payload":
            path = shlex.split(command)[1]
            return 0, digest(self.remote[path]) + "  " + path
        if label == "isolate-primary-metadata":
            args = shlex.split(command)
            source, target = args[args.index("cp") + 1:args.index("cp") + 3]
            assert source.endswith(".omc") and "/metadata/" in target
            self.remote[target] = self.remote[source]
            return 0, digest(self.remote[target]) + "  " + target
        if label == "verify-checksums":
            return 0, "fixture checksums matched"
        if label == "runtime-probe":
            self.remote[self.board + "/runtime.json"] = json.dumps({"runtime_version": "fixture-runtime", "npu_supported": True}).encode()
            self.remote[self.board + "/runtime.stderr"] = b""
            self.remote[self.board + "/runtime.exitcode"] = b"0\n"
            return 0, ""
        if label == "model-io-load":
            args = shlex.split(command)
            def arg(name):
                return args[args.index(name) + 1]
            assert arg("--descriptor-stage") == "restore-only"
            assert "--input" not in args and "--draft-step" not in args
            model = arg("--model-io-only")
            assert all(not path.endswith(".weight") for path in self.remote if path.startswith(str(PurePosixPath(model).parent) + "/"))
            row = {"schema_version": 1, "artifact_kind": "qwen35-hiai-model-io-probe", "evidence_kind": "device-hiai",
                "status": "IO_DESCRIPTORS_OBSERVED", "restore": "PASSED", "restore_init": "NOT_RUN_METADATA_ONLY",
                "descriptor_stage": "restore-only", "manager_init": "NOT_RUN", "external_weight_initialization": "NOT_REQUESTED",
                "model_load_scope": "RESTORE_METADATA_ONLY", "model_load_api": "file-v2", "model_path": model,
                "model_sha256": arg("--model-sha256"), "model_bytes": len(self.remote[model]),
                "expected_runtime_version": arg("--expected-runtime-version"), "runtime_version": "fixture-runtime",
                "firmware_id": arg("--firmware-id"), "ddk_id": arg("--ddk-id"), "process_called": False,
                "run_called": False, "tensor_allocations_requested": 0, "runtime_stage": "TENSOR_DESCRIPTORS",
                "dtype_enum": "hiai::DataType", "format_enum": "hiai::Format", "dtype_enum_values": {"FLOAT16": 1, "INT32": 3},
                "format_enum_values": {"ND": 2}, "runtime_input_order": "ORDERED_DESCRIPTOR_INDICES_ONLY",
                "runtime_tensor_names": "NOT_EXPOSED_BY_SELECTED_ADAPTER", "input_count": 89, "output_count": 82,
                "inputs": [{"index": i, "shape": [1], "raw_dtype": 3, "raw_format": 2} for i in range(89)],
                "outputs": [{"index": i, "shape": [1], "raw_dtype": 1, "raw_format": 2} for i in range(82)]}
            status = 0
            if self.native_fault == "manager_init": row["manager_init"] = "PASSED"
            elif self.native_fault == "ge_enum": row["dtype_enum"] = "ge.proto.DataType"
            elif self.native_fault == "hash": row["model_sha256"] = "b" * 64
            elif self.native_fault == "count":
                row["inputs"] = row["inputs"][:88]
                row["input_count"] = 88
            elif self.native_fault == "restore":
                row.update(status="MODEL_IO_FAILED", restore="FAILED_OR_NOT_REACHED", reason="fixture missing external dependency", runtime_stage="RESTORE_FROM_FILE")
                status = 1
            self.remote[self.board + "/model-io.json"] = json.dumps(row).encode()
            self.remote[self.board + "/model-io.stdout"] = b"fixture raw descriptor observation only"
            self.remote[self.board + "/model-io.stderr"] = b"fixture failure" if status else b""
            self.remote[self.board + "/model-io.exitcode"] = str(status).encode()
            if self.on_native is not None:
                self.on_native()
            return status, ""
        if label == "hash-result":
            path = shlex.split(command)[3].rstrip(";")
            if path not in self.remote:
                return 0, "__MISSING__"
            return 0, digest(self.remote[path]) + "  " + path
        return 0, ""


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SN", "fixture-device")
    monkeypatch.setenv("BIP", "fixture-server:8710")
    monkeypatch.setenv("CPRO_DDK_ID", "FIXTURE-DDK")
    for key in ("HIAI_RUNTIME_VERSION", "FIRMWARE_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(probe, "Hdc", FakeHdc)
    FakeHdc.instances, FakeHdc.remote, FakeHdc.sends = [], {}, []
    FakeHdc.busy_responses = []
    for key, value in (("fail_label", None), ("timeout_label", None), ("native_fault", None), ("disk_available_kb", 500000), ("busy", False)):
        monkeypatch.setattr(FakeHdc, key, value)
    monkeypatch.setattr(FakeHdc, "on_native", None)
    monkeypatch.setattr(FakeHdc, "corrupt_receive", False)
    job = tmp_path / "prefill-job"
    compilation = job / "compilation"
    primary = compilation / "compile-work/generated/model/model.omc"
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"small fixture metadata OMC")
    primary_row = {"path": primary.relative_to(compilation).as_posix(), "bytes": primary.stat().st_size,
        "sha256": digest(primary.read_bytes()), "stat": _stat(primary), "role": "primary_omc"}
    # This missing fifteen-GiB sidecar must never be opened or required by the
    # deliberately small restore-only metadata experiment.
    weight_row = {"path": "compile-work/generated/model/SubGraph_0.weight", "bytes": 15 << 30,
        "sha256": "f" * 64, "stat": [1, 2, 15 << 30, 0, 0], "role": "external_data_candidate"}
    profile = get_existing_air_profile("cpro-40block-prefill-64")
    report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "model_compilation": "PASSED",
        "returncode": 0, "platform": "kirinx90", "ddk_id": "FIXTURE-DDK",
        "output": str(compilation), "air_source": str(tmp_path / "original.air"),
        "input_layout": {"mode": "air-directory"},
        "requested_options": _conversion_options(profile["input_shape"], profile["dynamic_dims"], profile["limit_partition_size"], profile["save_weights_as_external_data"]),
        "source_preservation": {"status": "UNCHANGED_BY_STAT"}, "generated_artifacts": [primary_row, weight_row], "omc": primary_row}
    save(compilation / "report.json", report)
    compiled_sha = digest((compilation / "report.json").read_bytes())
    outer = {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED", "model_compilation": "PASSED",
        "output": str(job), "compilation_report_sha256": compiled_sha, "air": report["air_source"],
        "compilation": "compilation/report.json", "input_layout": "air-directory"}
    save(job / "result.json", outer)
    save(job / "compile-profile.json", get_existing_air_profile("cpro-40block-prefill-64"))
    contract = {"artifact_kind": "qwen35-original-target-contract", "status": "OBSERVED",
        "compilation_report_sha256": compiled_sha, "omc": primary_row,
        "boundaries": {"inputs_in_serialized_op_order": [{}] * 89, "outputs": [{}] * 82}}
    save(job / "target-contract/report.json", contract)
    help_dir = tmp_path / "help"
    tool = {"artifact_kind": "qwen35-target-board-tool-inspection", "status": "INSPECTED", "tool": TOOL,
        "tool_identity": "SHA256_MATCHED_BEFORE_AND_AFTER", "tool_sha256_before": TOOL_HASH, "tool_sha256_after": TOOL_HASH,
        "help": {"observation": "HELP_TEXT_OBSERVED"}, "firmware": {"returncode": 0, "text": FIRMWARE},
        "model_upload": "NOT_RUN", "model_load": "NOT_RUN", "model_inference": "NOT_RUN"}
    save(help_dir / "report.json", tool)
    def build(out, timeout):
        runner = out / "native/build/qwen35_dflash_hiai"
        runner.parent.mkdir(parents=True)
        runner.write_bytes(b"fixture native executable")
        include = out / "selected-headers"
        records = []
        for name in ("model/built_model.h", "model_manager/model_manager.h", "model_manager/model_manager_types.h", "tensor/nd_tensor_desc.h"):
            path = include / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// fixture declaration only\nvoid InitWeights(void* memory, unsigned long size);\n")
            records.append({"path": name, "status": "PRESENT", "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())})
        save(out / "native/command.json", {"argv": ["cmake", "-DHIAI_INCLUDE_DIR=" + str(include)], "v2_headers": records})
        return runner
    monkeypatch.setattr(probe, "_build_runner", build)
    monkeypatch.setattr(probe, "discover_libraries", lambda *a: {"libraries": [], "metadata": {"scope": "fixture only"}})
    monkeypatch.setattr(probe, "dynamic_info", lambda *a: {"needed": [], "soname": None})
    original_hash = probe.sha256_file
    def no_weight_hash(path):
        assert Path(path).suffix != ".weight", "restore-only must never open/hash sidecar payloads"
        return original_hash(path)
    monkeypatch.setattr(probe, "sha256_file", no_weight_hash)
    return {"job": job, "inspection": help_dir, "output": tmp_path / "io-probe", "primary": primary,
            "report": report, "outer": outer, "contract": contract, "tool": tool}


def execute(case, **kwargs):
    options = {"lock_wait_seconds": 0, **kwargs}
    return probe.inspect_target_runtime_io(case["job"], case["inspection"], case["output"], **options)


def test_restore_only_reads_and_uploads_small_primary_but_no_weight_or_input(case):
    result = execute(case)
    assert result["status"] == "IO_METADATA_OBSERVED", result
    assert result["manager_init"] == "NOT_RUN" and result["external_weight_initialization"] == "NOT_REQUESTED"
    assert result["model_inference"] == "NOT_RUN" and result["semantic_slot_identity"] == "UNVERIFIED"
    assert result["compiler_sidecar_declared_bytes"] == 15 << 30
    assert len(result["model_upload_plan"]) == 1 and result["model_bytes"] < 64 << 20
    assert not any(".weight" in local or "/inputs/" in remote for local, remote in FakeHdc.sends)
    command = json.loads((case["output"] / "model-io-command.json").read_text())
    assert command["argv"][command["argv"].index("--descriptor-stage") + 1] == "restore-only"
    assert command["cwd"] == str(PurePosixPath(result["model_path"]).parent)
    assert result["source_preservation"].startswith("UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256")
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    assert result["compiled_boundary_count_check"] == "MATCHED_89_INPUT_82_OUTPUT_COUNTS_ONLY"
    assert result["dtype_enum"] == "hiai::DataType"
    assert result["dtype_enum_values"]["INT32"] == 3
    assert "/metadata/" in result["model_path"] and result["model_path"] != result["cached_primary_model_path"]
    assert result["isolation"]["weight_files_copied"] == 0 and result["sidecar_upload"] == "NOT_RUN"
    assert (case["output"] / "selected-model-io-headers.json").is_file()


@pytest.mark.parametrize("profile", ["cpro-40block-prefill-64", "cpro-40block-decode-1"])
def test_both_static_profiles_share_one_phone_global_lock(case, profile):
    profile = get_existing_air_profile(profile)
    save(case["job"] / "compile-profile.json", profile)
    report = deepcopy(case["report"])
    report["requested_options"] = _conversion_options(profile["input_shape"], profile["dynamic_dims"], profile["limit_partition_size"], profile["save_weights_as_external_data"])
    save(case["job"] / "compilation/report.json", report)
    sha = digest((case["job"] / "compilation/report.json").read_bytes())
    outer = deepcopy(case["outer"]); outer["compilation_report_sha256"] = sha
    contract = deepcopy(case["contract"]); contract["compilation_report_sha256"] = sha
    save(case["job"] / "result.json", outer)
    save(case["job"] / "target-contract/report.json", contract)
    result = execute(case)
    assert result["status"] == "IO_METADATA_OBSERVED"
    assert result["board_global_lock"] == "/data/local/tmp/.qwen35-target-model-io-lock"


@pytest.mark.parametrize("fault", ["manager_init", "ge_enum", "hash", "restore"])
def test_invalid_or_failed_native_result_never_promotes_model_load_or_inference(case, monkeypatch, fault):
    monkeypatch.setattr(FakeHdc, "native_fault", fault)
    result = execute(case)
    assert result["status"] == "FAILED" and result["model_inference"] == "NOT_RUN"
    assert all(name in result["retrieved_sha256"] for name in ("model-io.json", "model-io.stderr", "model-io.exitcode"))
    assert result["lock_state"] == "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
    if fault == "restore":
        assert result["runtime_stage"] == "RESTORE_FROM_FILE" and "missing external dependency" in result["error"]


def test_runtime_count_difference_is_retained_without_relabeling_restore_failure(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "native_fault", "count")
    result = execute(case)
    assert result["status"] == "IO_METADATA_OBSERVED" and result["input_count"] == 88
    assert result["compiled_boundary_count_check"] == "DIFFERENT_COUNTS_RAW_METADATA_RETAINED"


def test_shared_lock_busy_stops_before_model_or_runner_upload(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "busy", True)
    result = execute(case)
    assert result["status"] == "FAILED" and "BOARD_MODEL_IO_BUSY" in result["error"]
    assert FakeHdc.sends == []
    assert all(label != "model-io-load" for _, label, _ in FakeHdc.instances[0].calls)


def test_disk_budget_counts_only_small_primary_and_native_payloads(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "disk_available_kb", 1)
    result = execute(case)
    assert result["status"] == "FAILED" and "disk budget insufficient" in result["error"]
    assert result["board_resources"]["missing_model_payload_bytes"] == case["primary"].stat().st_size
    assert all(not remote.endswith(".omc") for _, remote in FakeHdc.sends)


def test_existing_primary_cache_is_reused_but_neighbor_weight_never_enters_restore_directory(case):
    sha = digest((case["job"] / "compilation/report.json").read_bytes())
    cache = "/data/local/tmp/qwen3_5_moe/codex-target-" + sha[:20]
    primary = cache + "/payload/model/model.omc"
    weight = cache + "/payload/model/SubGraph_0.weight"
    FakeHdc.remote[primary] = case["primary"].read_bytes()
    FakeHdc.remote[weight] = b"unrelated previously retained sidecar, never touched"
    result = execute(case)
    assert result["status"] == "IO_METADATA_OBSERVED"
    assert result["board_resources"]["missing_model_payload_bytes"] == 0
    assert result["board_resources"]["isolated_primary_copy_bytes"] == result["model_bytes"]
    assert all(not remote.endswith(".omc") for _, remote in FakeHdc.sends)
    assert FakeHdc.remote[weight] == b"unrelated previously retained sidecar, never touched"


def test_busy_board_query_is_queued_before_upload_then_releases_after_known_exit(case, monkeypatch):
    FakeHdc.busy_responses = ["BUSY", "ACQUIRED"]
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    result = execute(case, lock_wait_seconds=2)
    assert result["status"] == "IO_METADATA_OBSERVED"
    labels = [label for _, label, _ in FakeHdc.instances[0].calls]
    assert labels.count("acquire-global-model-io-lock") == 2
    assert labels.index("acquire-global-model-io-lock") < labels.index("inspect-model-cache")
    assert labels.index("model-io-load") < labels.index("release-global-model-io-lock")


def test_unverified_native_result_bytes_are_not_promoted(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "corrupt_receive", True)
    result = execute(case)
    assert result["status"] == "FAILED"
    assert "model-io.json" not in result["retrieved_sha256"] and result["retrieval_errors"]


def test_original_small_omc_change_during_query_is_preserved_and_rejected(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "on_native", staticmethod(lambda: case["primary"].write_bytes(b"changed primary artifact")))
    result = execute(case)
    assert result["status"] == "FAILED" and result["source_preservation"] == "FAILED"
    assert case["primary"].read_bytes() == b"changed primary artifact"


def test_custom_cache_root_cannot_bypass_phone_global_serialization(case):
    result = execute(case, board_model_root="/data/local/tmp/different-owned-root")
    assert result["status"] == "IO_METADATA_OBSERVED"
    assert result["board_global_lock"] == "/data/local/tmp/.qwen35-target-model-io-lock"


def test_uncertain_native_wait_retains_both_locks_and_does_not_retry(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, "timeout_label", "model-io-load")
    result = execute(case)
    assert result["status"] == "FAILED" and result["lock_state"] == "RETAINED_OR_UNCERTAIN"
    labels = [label for _, label, _ in FakeHdc.instances[0].calls]
    assert labels.count("model-io-load") == 1 and "release-global-model-io-lock" not in labels


@pytest.mark.parametrize("fault", ["outer", "contract", "profile", "primary", "tool"])
def test_invalid_local_bindings_prevent_runner_build_and_hdc(case, monkeypatch, fault):
    if fault == "outer":
        value = deepcopy(case["outer"]); value["status"] = "FAILED"; save(case["job"] / "result.json", value)
    elif fault == "contract":
        value = deepcopy(case["contract"]); value["compilation_report_sha256"] = "0" * 64; save(case["job"] / "target-contract/report.json", value)
    elif fault == "profile": save(case["job"] / "compile-profile.json", get_existing_air_profile("cpro-40block-64-1"))
    elif fault == "primary": case["primary"].write_bytes(b"changed")
    else:
        value = deepcopy(case["tool"]); value["help"]["observation"] = "HELP_NOT_OBSERVED"; save(case["inspection"] / "report.json", value)
    monkeypatch.setattr(probe, "_build_runner", lambda *a: pytest.fail("invalid bindings cannot build"))
    result = execute(case)
    assert result["status"] == "FAILED" and not FakeHdc.instances


def test_headers_only_never_contacts_device_or_reads_weight_payload(case, monkeypatch):
    monkeypatch.delenv("SN")
    monkeypatch.delenv("BIP")
    result = execute(case, headers_only=True)
    assert result["status"] == "HEADERS_OBSERVED" and not FakeHdc.instances
    assert result["model_upload"] == result["model_load"] == "NOT_RUN"
    assert len(result["selected_v2_headers"]["files"]) == 4


def test_cli_uses_selected_successful_job_and_returns_raw_metadata_only(case, capsys):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_runtime_io.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["--job", str(case["job"]), "--tool-inspection", str(case["inspection"]), "--output", str(case["output"]), "--lock-wait-seconds", "0"]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["status"] == "IO_METADATA_OBSERVED" and result["model_inference"] == "NOT_RUN"


def test_global_lock_owner_write_failure_cannot_announce_acquisition(tmp_path):
    class OwnedShell:
        timeout = 10

        def shell(self, command, label, **options):
            # Exercise the generated shell text locally. The injected mkdir
            # creates an owner directory so printf fails even for uid0.
            prelude = 'mkdir() { command mkdir "$1" && command mkdir "$1/owner"; }; '
            result = subprocess.run(["sh", "-c", prelude + command], capture_output=True, text=True)
            assert "ACQUIRED" not in result.stdout
            if result.returncode:
                raise RuntimeError("owned lock marker could not be written")
            return 0, result.stdout.strip()

    lock = tmp_path / "owned-model-lock"
    with pytest.raises(RuntimeError, match="marker could not be written"):
        probe._acquire_global(OwnedShell(), lock, "owned-test-token", 0)
    assert lock.is_dir() and (lock / "owner").is_dir()


class MergedPosixHdc(Hdc):
    """Exercise real Hdc.shell framing with board-style merged output."""

    def __init__(self, output):
        super().__init__("fixture-hdc", "fixture-device", "fixture:1", output, 10)
        self.responses = []

    def call(self, arguments, label, *, timeout=None):
        assert arguments[0] == "shell"
        self.count += 1
        result = subprocess.run(["sh", "-c", arguments[1]], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout)
        self.last_call = {"label": label, "returncode": 0, "stdout": result.stdout, "stderr": ""}
        self.responses.append(result.stdout)
        # HDC can exit zero even when the remote command fails. Hdc.shell's
        # unique remote status marker remains the actual success gate.
        return result.stdout


def test_posix_merged_transport_acquires_and_releases_owned_global_lock(tmp_path):
    hdc = MergedPosixHdc(tmp_path)
    lock = tmp_path / "model-lock"
    probe._acquire_global(hdc, lock, "new-owner", 0)
    assert (lock / "owner").read_text() == "new-owner\n"
    probe._release_global(hdc, lock, "new-owner")
    assert not lock.exists()


def test_posix_existing_lock_is_busy_despite_merged_mkdir_stderr(tmp_path):
    hdc = MergedPosixHdc(tmp_path)
    lock = tmp_path / "model-lock"
    lock.mkdir()
    owner = lock / "owner"
    owner.write_text("other-active-owner\n")
    with pytest.raises(ValueError, match="BOARD_MODEL_IO_BUSY"):
        probe._acquire_global(hdc, lock, "waiting-owner", 0)
    assert owner.read_text() == "other-active-owner\n"
    assert hdc.responses[0].startswith("BUSY\n")
    assert "mkdir:" not in hdc.responses[0]


def test_posix_busy_wait_acquires_after_other_owner_releases(tmp_path, monkeypatch):
    hdc = MergedPosixHdc(tmp_path)
    lock = tmp_path / "model-lock"
    lock.mkdir()
    (lock / "owner").write_text("other-active-owner\n")
    waits = []

    def release_fixture_lock(seconds):
        waits.append(seconds)
        assert (lock / "owner").read_text() == "other-active-owner\n"
        (lock / "owner").unlink()
        lock.rmdir()

    monkeypatch.setattr(probe.time, "sleep", release_fixture_lock)
    probe._acquire_global(hdc, lock, "waiting-owner", 2)
    assert len(waits) == 1 and len(hdc.responses) == 2
    assert (lock / "owner").read_text() == "waiting-owner\n"
    probe._release_global(hdc, lock, "waiting-owner")


@pytest.mark.parametrize("kind", ["file", "symlink", "missing-parent"])
def test_posix_non_directory_or_symlink_lock_never_counts_as_contention(tmp_path, kind):
    hdc = MergedPosixHdc(tmp_path)
    lock = tmp_path / "model-lock"
    if kind == "file":
        lock.write_text("unowned file")
    elif kind == "symlink":
        target = tmp_path / "unowned-lock"
        target.mkdir()
        (target / "owner").write_text("other-owner\n")
        lock.symlink_to(target, target_is_directory=True)
    else:
        lock = tmp_path / "missing-parent" / "model-lock"
    with pytest.raises(RuntimeError, match="remote acquire-global-model-io-lock failed"):
        probe._acquire_global(hdc, lock, "candidate-owner", 0)
    if kind == "file":
        assert lock.read_text() == "unowned file"
    elif kind == "symlink":
        assert lock.is_symlink() and (target / "owner").read_text() == "other-owner\n"
    else:
        assert not lock.parent.exists()


def test_unrecognized_global_lock_reply_retains_raw_reply_and_transport_diagnostics(tmp_path):
    class UnexpectedHdc(MergedPosixHdc):
        def shell(self, command, label, **options):
            self.count = 1
            self.last_call = {"label": label, "returncode": 0, "stdout": "unexpected reply\n", "stderr": "transport detail"}
            return 0, "unexpected reply"

    with pytest.raises(ValueError) as observed:
        probe._acquire_global(UnexpectedHdc(tmp_path), tmp_path / "lock", "candidate", 0)
    assert "'unexpected reply'" in str(observed.value)
    assert "transport detail" in str(observed.value) and "Full log:" in str(observed.value)

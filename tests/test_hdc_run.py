"""Independent HDC protocol tests; never device or model-execution evidence.

The fake HDC executes real host shell, transfer, and checksum operations inside
the pytest run directory. A separate Python double replaces ONLY execution of
the uploaded AArch64 ELF. Its ``device-hiai`` string is deliberately synthetic:
it exercises the production comparison gate and establishes no device result.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "targets/cpro/scripts/run_hdc.py"


FAKE_HDC = r'''
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_HDC_CALLS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\n")
if args[:4] != ["-t", os.environ["SN"], "-s", os.environ["BIP"]]:
    print("SIMULATED HDC: incorrect serial/server selection", file=sys.stderr)
    sys.exit(72)
args = args[4:]
remote = os.environ["FAKE_BOARD_ROOT"]
local = os.environ["FAKE_BOARD_SANDBOX"]

def translated(path):
    if path != remote and not path.startswith(remote + "/"):
        raise RuntimeError("SIMULATED HDC: path outside isolated board root: " + path)
    return Path(local + path[len(remote):])

if args[:2] == ["file", "send"]:
    if len(args) != 4:
        raise RuntimeError("unsupported file send arguments")
    # A missing destination parent is a real transfer failure.
    shutil.copyfile(args[2], translated(args[3]))
    transferred = translated(args[3]).read_bytes()
    with open(os.environ["FAKE_HDC_UPLOADS"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"remote": args[3], "bytes": len(transferred),
                                 "sha256": hashlib.sha256(transferred).hexdigest()}) + "\n")
elif args[:2] == ["file", "recv"]:
    if len(args) != 4:
        raise RuntimeError("unsupported file recv arguments")
    shutil.copyfile(translated(args[2]), args[3])
    if os.environ.get("FAKE_HDC_FAILURE") == "receive-corruption" and args[2].endswith("/benchmark.json"):
        # Semantically identical JSON still MUST fail the transport hash gate.
        with open(args[3], "ab") as stream:
            stream.write(b" ")
    if os.environ.get("FAKE_HILOG_FAILURE") == "receive-corruption" and args[2].endswith("/hiai-hilog.txt"):
        with open(args[3], "ab") as stream:
            stream.write(b"SIMULATED HIAI corrupt transfer must not print\n")
    if os.environ.get("FAKE_HDC_FAILURE") == "owner-mismatch" and args[2].endswith("/benchmark.exitcode"):
        owner = Path(local) / ".qwen35-hdc-owner"
        if not owner.is_file():
            raise RuntimeError("test requires an existing uploaded ownership marker")
        owner.write_text("SIMULATED-FOREIGN-OWNER\n")
    if os.environ.get("FAKE_HDC_FAILURE") == "root-symlink" and args[2].endswith("/benchmark.exitcode"):
        board = Path(local)
        replaced = Path(local + "-replaced")
        board.rename(replaced)
        board.symlink_to(replaced, target_is_directory=True)
elif args[:1] == ["shell"]:
    failure = os.environ.get("FAKE_HDC_FAILURE", "")
    if failure in ("missing-marker-stdout", "missing-marker-stderr", "missing-marker-empty"):
        # Match the real disconnected-device response: transport reports zero
        # but executes no remote command and supplies no remote exit marker.
        if failure != "missing-marker-empty":
            print("[Fail]Device not founded or connected",
                  file=sys.stderr if failure == "missing-marker-stderr" else sys.stdout)
        sys.exit(0)
    if len(args) != 2:
        raise RuntimeError("shell must be one argument")
    command = shlex.split(args[1])
    if command[:2] != ["sh", "-c"] or len(command) != 3:
        raise RuntimeError("expected sh -c wrapper")
    payload = command[2].replace(remote, local)
    if "hilog -x" in payload:
        # Substitute only the diagnostic executable, preserving actual shell
        # redirection, remote status marker, transfer, and checksum operations.
        # No host hilog or board access is involved in this protocol test.
        mode = os.environ.get("FAKE_HILOG_FAILURE", "")
        argument_error = "Mutlti commands can't be used in combination [CODE: -31]"
        # Match the board CLI rather than accepting every requested argument.
        # It rejects combined actions even when the process reports success.
        guard = ('if [ "$#" -ne 1 ] || [ "$1" != "-x" ]; then '
                 + shlex.join(["printf", "%s\n", argument_error]) + '; return 0; fi')
        if mode == "argument-error-zero":
            statements = [shlex.join(["printf", "%s\n", argument_error]), "return 0"]
        elif mode == "unavailable":
            statements = [shlex.join(["printf", "%s\n", "SIMULATED hilog unavailable"])
                          + " >&2", "return 127"]
        else:
            statements = [
                shlex.join(["printf", "%s\n", "SIMULATED unrelated network telemetry"]),
                shlex.join(["printf", "%s\n", "SIMULATED HIAI Load model failure"]),
                shlex.join(["printf", "%s\n", "SIMULATED hilog reader diagnostic"]) + " >&2",
                "return 0"]
        payload = "hilog() {\n" + guard + ";\n" + ";\n".join(statements) + ";\n};\n" + payload
    if os.environ.get("FAKE_HDC_FAILURE") == "cleanup-rm":
        # Preserve the real shell, ownership guards, and remote exit marker;
        # only the deletion operation fails. The directory remains untouched.
        payload = "rm() { printf '%s\\n' 'SIMULATED rm failure' >&2; return 13; };\n" + payload
    if "param get const.ohos.fullname" in payload:
        firmware = ("Get parameter const.ohos.fullname fail! errNum is: 100"
                    if os.environ.get("FAKE_HDC_FAILURE") == "firmware-error"
                    else "OpenHarmony-6.0.fake")
        payload = payload.replace("param get const.ohos.fullname",
                                  shlex.join(["printf", "%s\\n", firmware]))
    # Preserve uploaded ELF and its SHA256. Only actual execution is replaced;
    # chmod, upload, and checksum still operate on the original test payload.
    for basename in ("probe.sh", "run.sh"):
        script = Path(local) / basename
        invocation = "sh " + shlex.quote(str(script))
        if invocation in payload and script.is_file():
            content = script.read_text().replace(remote, local)
            content = content.replace(local + "/bin/qwen35_dflash_hiai",
                                      os.environ["FAKE_RUNNER"])
            # Do not alter the transferred script: checksum checks must verify
            # exactly its original payload. Substitute only this invocation.
            payload = payload.replace(invocation, "sh -c " + shlex.quote(content))
    result = subprocess.run(["sh", "-c", payload], capture_output=True)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    # Deliberately mask the remote exit status, as an HDC transport can do.
    # Production must inspect its shell marker and pulled exitcode instead.
    sys.exit(0)
else:
    raise RuntimeError("unsupported HDC invocation: " + repr(args))
'''


FAKE_RUNNER = r'''
import copy
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_RUNNER_CALLS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({"argv": args, "cwd": os.getcwd(),
                             "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}) + "\n")
failure = os.environ.get("FAKE_HDC_FAILURE", "")
if args == ["--probe"]:
    if failure == "probe-exit":
        print("SIMULATED probe failure retained on stderr", file=sys.stderr)
        sys.exit(19)
    print(json.dumps({"runtime_version": "SIMULATED-HIAI-PROTOCOL-ONLY",
                      "npu_supported": failure != "no-npu"}))
    # npu_supported=false must be rejected independently of return code.
    sys.exit(0)
if failure == "remote-exit":
    print("SIMULATED benchmark failure retained on stderr", file=sys.stderr)
    print("SIMULATED benchmark partial stdout")
    sys.exit(17)
flags = dict(zip(args[::2], args[1::2]))
if flags.get("--mode") != "benchmark":
    raise RuntimeError("expected paired benchmark mode")
source = json.loads(Path(os.environ["FAKE_EXPECTED_REPORT"]).read_text())
source["target_sha256"] = flags["--target-sha256"]
source["draft_sha256"] = flags["--draft-sha256"]
source["runtime_version"] = flags["--expected-runtime-version"]
if failure == "measurement-mismatch":
    source["dflash"]["measurements"][7]["generated_token_ids"] = [3, 5]
if failure == "fake-evidence":
    source["evidence_kind"] = "host-fake-hiai"
Path(flags["--output"]).write_text(json.dumps(source))
print("SIMULATED benchmark completed; no device execution")
'''


def _json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _executable(path, body):
    path.write_text("#!" + sys.executable + "\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _record(root, path):
    payload = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def case(tmp_path):
    bundle = tmp_path / "omc bundle"
    bundle.mkdir()
    graphs = []
    for name in ("target", "draft"):
        folder = bundle / name
        folder.mkdir()
        omc = folder / (name + ".omc")
        omc.write_bytes(b"SIMULATED-OMC-" + name.encode())
        sidecar = folder / "weights" / "data.bin"
        sidecar.parent.mkdir()
        sidecar.write_bytes(b"SIMULATED-SIDECAR-" + name.encode())
        record = _record(bundle, omc)
        graphs.append({"name": name, "omc": record.pop("path"), **record,
                       "payloads": [_record(bundle, omc), _record(bundle, sidecar)]})
    source = {"kind": "FAKE-HDC-PROTOCOL-ONLY", "not_device_evidence": True}
    manifest = {"schema_version": 2, "status": "COMPILED",
                "artifact_kind": "qwen35-cpro-omc-pair", "runtime": "hiai",
                "source": source, "sequence_length": 8, "feature_width": 32,
                "vocab_size": 64, "draft_width": 3, "graphs": graphs,
                "pad_token_id": 0, "eos_token_ids": [10], "ddk_id": "SIMULATED-DDK",
                "abi_version": 2, "feature_dtype": "torch.float16",
                "precision": "torch.float16", "quantization": None}
    _json(bundle / "manifest.json", manifest)
    reference = tmp_path / "reference.json"
    _json(reference, {"status": "PASS", "source": source, "prompt_token_ids": [1, 2],
                      "max_new_tokens": 2, "sequence_length": 8,
                      "ordinary": {"token_ids": [3, 4], "stop_reason": "length"},
                      "precision": "torch.float16", "quantization": None})
    sample = {"generated_token_ids": [3, 4], "stop_reason": "length"}
    mode = {"stable_generated_token_ids": [3, 4], "stable_stop_reason": "length",
            "warmup": 3, "repetitions": 10,
            "measurements": [copy.deepcopy(sample) for _ in range(10)]}
    report = tmp_path / "simulated-report.json"
    _json(report, {"status": "PASS", "evidence_kind": "device-hiai",
                   "simulation_boundary": "Protocol fixture only; no device execution",
                   "source": source, "prompt_token_ids": [1, 2], "max_new_tokens": 2,
                   "sequence_length": 8, "ordinary": mode, "dflash": copy.deepcopy(mode),
                   "abi_version": 2, "feature_dtype": "torch.float16"})
    runner = tmp_path / "runner.elf"
    elf = bytearray(64)
    elf[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", elf, 16, 2, 183, 1)
    struct.pack_into("<H", elf, 52, 64)
    runner.write_bytes(elf)
    libs = tmp_path / "runtime libs"
    libs.mkdir()
    (libs / "libhiai-fixture.so").write_bytes(b"SIMULATED-LIBRARY")
    hdc = tmp_path / "fake-hdc"
    fake_runner = tmp_path / "simulated-runner"
    _executable(hdc, FAKE_HDC)
    _executable(fake_runner, FAKE_RUNNER)
    board_root = "/data/local/tmp/qwen35-hdc-protocol-test"
    board = tmp_path / "isolated-board"
    out = tmp_path / "host-output"
    env = dict(os.environ, AI_RUN_DIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(REPO / "model"), SN="SIMULATED-SERIAL-001", BIP="127.0.0.1:8710",
               FAKE_HDC_CALLS=str(tmp_path / "hdc-calls.jsonl"),
               FAKE_HDC_UPLOADS=str(tmp_path / "hdc-uploads.jsonl"),
               FAKE_RUNNER_CALLS=str(tmp_path / "runner-calls.jsonl"),
               FAKE_EXPECTED_REPORT=str(report), FAKE_HDC_FAILURE="", FAKE_HILOG_FAILURE="",
               FAKE_BOARD_ROOT=board_root, FAKE_BOARD_SANDBOX=str(board),
               FAKE_RUNNER=str(fake_runner))
    argv = [sys.executable, str(SCRIPT), "--omc", str(bundle), "--runner", str(runner),
            "--reference", str(reference), "--lib-dir", str(libs), "--output", str(out),
            "--board-root", board_root, "--firmware-id", "SIMULATED-FIRMWARE",
            "--hdc", str(hdc), "--timeout-seconds", "20"]
    return {"root": tmp_path, "bundle": bundle, "manifest": manifest, "reference": reference,
            "runner": runner, "libs": libs, "board": board, "output": out, "env": env, "argv": argv}


def _run(case, failure=""):
    return subprocess.run(case["argv"], env=dict(case["env"], FAKE_HDC_FAILURE=failure),
                          text=True, capture_output=True, timeout=90)


def _calls(case, runner=False):
    path = Path(case["env"]["FAKE_RUNNER_CALLS" if runner else "FAKE_HDC_CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _cleanup(case):
    return json.loads((case["output"] / "result.json").read_text())["board_cleanup"]["status"]


def _build_record(case, *, legacy=False, missing_library=False):
    """Make build metadata with intentionally unavailable historical SDK paths."""
    folder = case["root"] / "recorded-hiai-build"
    folder.mkdir()
    library = case["libs"] / "libhiai-fixture.so"
    libraries = [{"path": "/simulated-historical-sdk/lib/libhiai-fixture.so",
                  "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}]
    if missing_library:
        libraries.append({"path": "/simulated-historical-sdk/lib/libhiai-not-uploaded.so",
                          "sha256": hashlib.sha256(b"SIMULATED-OTHER-LINK-INPUT").hexdigest()})
    command = {"headers_only": False, "argv": ["cmake", "SIMULATED-CONFIGURE"],
               "libraries": libraries}
    _json(folder / "command.json", command)
    result = {"status": "LINKED", "device_validation": "PENDING",
              "binary_sha256": hashlib.sha256(case["runner"].read_bytes()).hexdigest()}
    if not legacy:
        result["command_sha256"] = hashlib.sha256((folder / "command.json").read_bytes()).hexdigest()
    _json(folder / "result.json", result)
    case["argv"].extend(["--hiai-build-dir", str(folder)])
    return folder, command, result


def test_hdc_protocol_upload_environment_and_strict_comparison(case):
    result = _run(case)
    assert result.returncode == 0, result.stdout + result.stderr
    comparison = json.loads((case["output"] / "comparison.json").read_text())
    assert comparison["status"] == "PASS" and all(comparison["checks"].values())
    assert comparison["checks"]["ten_stable_runs"] and comparison["checks"]["real_hiai"]
    uploads = {entry["remote"]: entry for entry in (
        json.loads(line) for line in Path(case["env"]["FAKE_HDC_UPLOADS"]).read_text().splitlines())}
    board_root = case["env"]["FAKE_BOARD_ROOT"]
    for graph in case["manifest"]["graphs"]:
        for record in graph["payloads"]:
            uploaded = uploads[board_root + "/models/" + record["path"]]
            assert (uploaded["bytes"], uploaded["sha256"]) == (record["bytes"], record["sha256"])
    for relative, local in (("models/manifest.json", case["bundle"] / "manifest.json"),
                            ("bin/qwen35_dflash_hiai", case["runner"]),
                            ("lib/libhiai-fixture.so", case["libs"] / "libhiai-fixture.so")):
        assert uploads[board_root + "/" + relative]["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    calls = _calls(case)
    assert calls and all(args[:4] == ["-t", case["env"]["SN"], "-s", case["env"]["BIP"]]
                         for args in calls)
    assert not any("hilog -x" in " ".join(args) for args in calls)
    assert not (case["output"] / "hiai-hilog.txt").exists()
    assert any("sha256sum" in " ".join(args) for args in calls)
    cleanup_calls = [index for index, args in enumerate(calls)
                     if args[4] == "shell" and "rm " in args[5]]
    receive_calls = [index for index, args in enumerate(calls) if args[4:6] == ["file", "recv"]]
    assert cleanup_calls and receive_calls and min(cleanup_calls) > max(receive_calls)
    assert cleanup_calls[-1] == len(calls) - 1, "cleanup must be the final HDC operation"
    executions = _calls(case, runner=True)
    assert len(executions) == 2 and executions[0]["argv"] == ["--probe"]
    for execution in executions:
        assert str(case["board"] / "lib") in execution["LD_LIBRARY_PATH"].split(":")
    flags = dict(zip(executions[1]["argv"][::2], executions[1]["argv"][1::2]))
    assert flags["--mode"] == "benchmark"
    assert flags["--expected-runtime-version"] == "SIMULATED-HIAI-PROTOCOL-ONLY"
    assert flags["--firmware-id"] == "SIMULATED-FIRMWARE"
    assert (case["output"] / "runtime-probe.json").is_file()
    assert (case["output"] / "benchmark.json").is_file()
    assert (case["output"] / "benchmark.exitcode").read_text().strip() == "0"
    assert not case["board"].exists()
    assert _cleanup(case) == "REMOVED"


def test_hdc_zero_transport_status_does_not_hide_remote_failure(case):
    result = _run(case, "remote-exit")
    assert result.returncode != 0
    assert "SIMULATED benchmark failure retained on stderr" in result.stderr
    assert "SIMULATED benchmark partial stdout" in result.stderr
    assert str(case["output"] / "benchmark.stderr") in result.stderr
    assert str(case["output"] / "benchmark.stdout") in result.stderr
    assert "SIMULATED benchmark failure" in (case["output"] / "benchmark.stderr").read_text()
    assert "partial stdout" in (case["output"] / "benchmark.stdout").read_text()
    assert (case["output"] / "benchmark.exitcode").read_text().strip() == "17"
    assert (case["output"] / "failure.json").is_file()
    assert (case["output"] / "result.json").is_file()
    assert (case["board"] / "models/target/target.omc").is_file()
    assert not (case["output"] / "comparison.json").exists()
    assert _cleanup(case) == "RETAINED"


@pytest.mark.parametrize("channel", ["stdout", "stderr", "empty"])
def test_missing_remote_marker_shows_hdc_response_and_preserves_local_only_state(case, channel):
    result = _run(case, "missing-marker-" + channel)
    assert result.returncode != 0
    assert "marker" in result.stderr.lower()
    records = list((case["output"] / "calls").glob("*.json"))
    assert len(records) == 1
    assert str(records[0]) in result.stderr
    record = json.loads(records[0].read_text())
    assert record["returncode"] == 0
    original = "[Fail]Device not founded or connected\n"
    if channel == "empty":
        assert record["stdout"] == record["stderr"] == ""
    else:
        assert record[channel] == original
        assert record["stderr" if channel == "stdout" else "stdout"] == ""
        assert original.strip() in result.stderr
    assert "stdout" in result.stderr and "stderr" in result.stderr
    assert "list targets -v" in result.stderr and case["env"]["BIP"] in result.stderr
    calls = _calls(case)
    assert len(calls) == 1 and calls[0][4] == "shell"
    assert not any("rm " in " ".join(args) for args in calls)
    assert not Path(case["env"]["FAKE_HDC_UPLOADS"]).exists()
    assert not _calls(case, runner=True)
    assert not case["board"].exists()
    assert _cleanup(case) == "NOT_CREATED"
    assert (case["output"] / "failure.json").is_file()


def test_probe_must_report_npu_even_with_successful_remote_exit(case):
    result = _run(case, "no-npu")
    assert result.returncode != 0
    probe = json.loads((case["output"] / "runtime-probe.json").read_text())
    assert probe["npu_supported"] is False
    assert len(_calls(case, runner=True)) == 1
    assert (case["output"] / "probe.exitcode").read_text().strip() == "0"
    assert (case["output"] / "failure.json").is_file()
    assert case["board"].is_dir() and _cleanup(case) == "RETAINED"


def test_automatic_firmware_identity_runs_full_protocol(case):
    index = case["argv"].index("--firmware-id")
    del case["argv"][index:index + 2]
    result = _run(case)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _calls(case, runner=True)
    flags = dict(zip(calls[1]["argv"][::2], calls[1]["argv"][1::2]))
    assert flags["--firmware-id"] == "OpenHarmony-6.0.fake"
    assert json.loads((case["output"] / "comparison.json").read_text())["status"] == "PASS"
    assert not case["board"].exists() and _cleanup(case) == "REMOVED"


def test_firmware_query_error_text_is_rejected_despite_zero_exit(case):
    index = case["argv"].index("--firmware-id")
    del case["argv"][index:index + 2]
    result = _run(case, "firmware-error")
    assert result.returncode != 0
    assert "--firmware-id" in result.stderr + (case["output"] / "failure.json").read_text()
    assert len(_calls(case, runner=True)) == 1
    assert not (case["output"] / "benchmark.json").exists()
    assert case["board"].is_dir() and _cleanup(case) == "RETAINED"


def test_hdc_server_requires_explicit_port_before_connecting(case):
    case["env"]["BIP"] = "127.0.0.1"
    result = _run(case)
    assert result.returncode != 0
    assert not _calls(case)
    assert not case["board"].exists()
    assert _cleanup(case) == "NOT_CREATED"


def test_received_report_hash_is_required_even_when_json_is_equivalent(case):
    result = _run(case, "receive-corruption")
    assert result.returncode != 0
    assert not (case["output"] / "benchmark.json").exists()
    assert (case["board"] / "benchmark.json").is_file()
    retained = list((case["output"] / "received").glob("*benchmark.json"))
    assert retained, "retain the corrupt transfer for diagnosis"
    original = (case["board"] / "benchmark.json").read_bytes()
    assert all(path.read_bytes() != original for path in retained)
    assert all(json.loads(path.read_bytes()) == json.loads(original) for path in retained)
    assert (case["output"] / "failure.json").is_file()
    assert not (case["output"] / "comparison.json").exists()
    assert _cleanup(case) == "RETAINED"


@pytest.mark.parametrize("failure,gate", [("measurement-mismatch", "ten_stable_runs"),
                                         ("fake-evidence", "real_hiai")])
def test_existing_comparison_rejects_bad_measurements_and_fake_device_evidence(case, failure, gate):
    result = _run(case, failure)
    assert result.returncode != 0
    comparison = json.loads((case["output"] / "comparison.json").read_text())
    assert comparison["status"] == "FAIL" and comparison["checks"][gate] is False
    assert (case["output"] / "benchmark.json").is_file()
    assert (case["output"] / "failure.json").is_file()
    assert (case["board"] / "benchmark.json").is_file()
    assert _cleanup(case) == "RETAINED"


@pytest.mark.parametrize("bad", ["graph-hash", "sidecar-hash", "reference", "runner-architecture"])
def test_invalid_local_inputs_are_rejected_before_connecting(case, bad):
    if bad == "graph-hash":
        (case["bundle"] / "target/target.omc").write_bytes(b"tampered")
    elif bad == "sidecar-hash":
        (case["bundle"] / "draft/weights/data.bin").write_bytes(b"tampered sidecar")
    elif bad == "reference":
        reference = json.loads(case["reference"].read_text())
        reference["source"] = {"kind": "different weights"}
        _json(case["reference"], reference)
    else:
        elf = bytearray(case["runner"].read_bytes())
        struct.pack_into("<H", elf, 18, 62)  # x86-64, intentionally not AArch64.
        case["runner"].write_bytes(elf)
    result = _run(case)
    assert result.returncode != 0
    assert not _calls(case)
    assert not case["board"].exists()
    assert _cleanup(case) == "NOT_CREATED"


def test_existing_remote_directory_is_preserved_and_not_reused(case):
    case["board"].mkdir()
    sentinel = case["board"] / "preexisting-user-data"
    sentinel.write_bytes(b"preserve")
    result = _run(case)
    assert result.returncode != 0
    assert sentinel.read_bytes() == b"preserve"
    assert list(case["board"].iterdir()) == [sentinel]
    assert not _calls(case, runner=True)
    assert (case["output"] / "failure.json").is_file()
    assert _cleanup(case) == "NOT_CREATED"


@pytest.mark.parametrize("failure", ["owner-mismatch", "cleanup-rm", "root-symlink"])
def test_cleanup_failure_retains_remote_data_and_successful_device_evidence(case, failure):
    result = _run(case, failure)
    assert result.returncode != 0
    assert case["board"].is_dir()
    assert (case["board"] / "models/target/target.omc").is_file()
    assert (case["board"] / "benchmark.json").is_file()
    assert (case["output"] / "benchmark.json").is_file()
    assert json.loads((case["output"] / "comparison.json").read_text())["status"] == "PASS"
    status = json.loads((case["output"] / "result.json").read_text())
    assert status["device_validation"] == "PASS"
    assert status["status"] != "PASS" and status["board_cleanup"]["status"] == "FAILED"
    assert (case["output"] / "failure.json").is_file()
    if failure == "owner-mismatch":
        assert (case["board"] / ".qwen35-hdc-owner").read_text() == "SIMULATED-FOREIGN-OWNER\n"
    if failure == "root-symlink":
        protected = Path(str(case["board"]) + "-replaced")
        assert case["board"].is_symlink()
        assert (protected / "models/target/target.omc").read_bytes() == (
            case["bundle"] / "target/target.omc").read_bytes()


def test_omc_cleanup_removes_only_omcs_and_preserves_weight_sidecars(case):
    case["argv"].extend(["--board-cleanup", "omc"])
    result = _run(case)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _cleanup(case) == "OMC_FILES_REMOVED"
    assert case["board"].is_dir()
    for graph in case["manifest"]["graphs"]:
        for record in graph["payloads"]:
            remote = case["board"] / "models" / record["path"]
            if record["path"] == graph["omc"]:
                assert not remote.exists()
            else:
                assert remote.is_file()
                assert hashlib.sha256(remote.read_bytes()).hexdigest() == record["sha256"]
                assert remote.read_bytes() == (case["bundle"] / record["path"]).read_bytes()
            assert remote.parent.is_dir(), "model directories must be retained"
    assert (case["board"] / "models/manifest.json").read_bytes() == (
        case["bundle"] / "manifest.json").read_bytes()
    assert (case["board"] / "bin/qwen35_dflash_hiai").read_bytes() == case["runner"].read_bytes()
    assert (case["board"] / "lib/libhiai-fixture.so").read_bytes() == b"SIMULATED-LIBRARY"
    assert (case["board"] / ".qwen35-hdc-owner").is_file()
    assert (case["board"] / "benchmark.json").is_file()
    assert (case["board"] / "benchmark.stderr").is_file()
    assert json.loads((case["output"] / "comparison.json").read_text())["status"] == "PASS"


def test_none_cleanup_preserves_complete_remote_run(case):
    case["argv"].extend(["--board-cleanup", "none"])
    result = _run(case)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _cleanup(case) == "RETAINED"
    for graph in case["manifest"]["graphs"]:
        for record in graph["payloads"]:
            assert (case["board"] / "models" / record["path"]).read_bytes() == (
                case["bundle"] / record["path"]).read_bytes()
    assert (case["board"] / "bin/qwen35_dflash_hiai").is_file()
    assert (case["board"] / "lib/libhiai-fixture.so").is_file()
    assert (case["board"] / "models/manifest.json").is_file()
    assert (case["board"] / ".qwen35-hdc-owner").is_file()
    assert (case["board"] / "benchmark.json").is_file()
    assert not any(args[4] == "shell" and "rm " in args[5] for args in _calls(case))
    assert json.loads((case["output"] / "comparison.json").read_text())["status"] == "PASS"


def test_matching_hiai_build_and_uploaded_libraries_are_bound_in_evidence(case):
    folder, command, build_result = _build_record(case)
    result = _run(case)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads((case["output"] / name).read_text())["hiai_build"]
               for name in ("preflight.json", "result.json")]
    assert records[0] == records[1]
    audit = records[0]
    assert audit["command_binding"] == "sha256-verified"
    assert audit["command_sha256"] == build_result["command_sha256"]
    assert audit["result_sha256"] == hashlib.sha256((folder / "result.json").read_bytes()).hexdigest()
    assert audit["binary_sha256"] == hashlib.sha256(case["runner"].read_bytes()).hexdigest()
    assert audit["missing_from_uploads"] == []
    assert len(audit["link_libraries"]) == 1
    library = audit["link_libraries"][0]
    assert library["name"] == "libhiai-fixture.so" and library["status"] == "matched"
    assert library["path"] == command["libraries"][0]["path"]
    assert library["sha256"] == library["upload_sha256"] == command["libraries"][0]["sha256"]
    assert json.loads((case["output"] / "comparison.json").read_text())["status"] == "PASS"


@pytest.mark.parametrize("mismatch", ["same-name-library", "runner-binary", "command-record"])
def test_build_identity_mismatches_are_rejected_before_hdc(case, mismatch):
    folder, command, build_result = _build_record(case)
    if mismatch == "same-name-library":
        command["libraries"][0]["sha256"] = hashlib.sha256(b"DIFFERENT-LINK-LIBRARY").hexdigest()
        _json(folder / "command.json", command)
        build_result["command_sha256"] = hashlib.sha256((folder / "command.json").read_bytes()).hexdigest()
        _json(folder / "result.json", build_result)
    elif mismatch == "runner-binary":
        build_result["binary_sha256"] = hashlib.sha256(b"DIFFERENT-RUNNER").hexdigest()
        _json(folder / "result.json", build_result)
    else:
        command["argv"].append("SIMULATED-UNRECORDED-CHANGE")
        _json(folder / "command.json", command)
    result = _run(case)
    assert result.returncode != 0
    assert not _calls(case) and not _calls(case, runner=True)
    assert not case["board"].exists() and _cleanup(case) == "NOT_CREATED"
    assert (case["output"] / "failure.json").is_file()
    if mismatch == "same-name-library":
        assert command["libraries"][0]["path"] in result.stderr
        assert command["libraries"][0]["sha256"] in result.stderr
        uploaded = case["libs"] / "libhiai-fixture.so"
        assert str(uploaded) in result.stderr
        assert hashlib.sha256(uploaded.read_bytes()).hexdigest() in result.stderr


def test_legacy_build_and_unuploaded_link_input_are_explicit_without_board_assumptions(case):
    folder, command, _ = _build_record(case, legacy=True, missing_library=True)
    result = _run(case)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("preflight.json", "result.json"):
        audit = json.loads((case["output"] / name).read_text())["hiai_build"]
        assert audit["command_binding"] == "legacy-unbound"
        assert audit["legacy_note"]
        assert audit["command_sha256"] == hashlib.sha256((folder / "command.json").read_bytes()).hexdigest()
        assert len(audit["missing_from_uploads"]) == 1
        absent = audit["missing_from_uploads"][0]
        assert absent["name"] == "libhiai-not-uploaded.so"
        assert absent["path"] == command["libraries"][1]["path"]
        assert absent["sha256"] == command["libraries"][1]["sha256"]
        assert {entry["name"]: entry["status"] for entry in audit["link_libraries"]} == {
            "libhiai-fixture.so": "matched", "libhiai-not-uploaded.so": "not_uploaded"}
    assert json.loads((case["output"] / "comparison.json").read_text())["status"] == "PASS"


def _shared_elf(path, marker, *, machine=183):
    """Minimal ELF header for selection tests, never a loadable runtime library."""
    elf = bytearray(64)
    elf[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", elf, 16, 3, machine, 1)
    struct.pack_into("<H", elf, 52, 64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(elf + marker.encode())
    return path


def _automatic_build(case):
    """Reproduce the user's original build layout without extra CLI/env inputs."""
    build = case["root"] / "original hiai build"
    runner = build / "build/qwen35_dflash_hiai"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(case["runner"].read_bytes())
    case["runner"] = runner
    case["argv"][case["argv"].index("--runner") + 1] = str(runner)
    index = case["argv"].index("--lib-dir")
    del case["argv"][index:index + 2]
    for name in ("HIAI_RUNTIME_LIB_DIR", "HIAI_BUILD_DIR", "OHOS_NATIVE_SDK",
                 "CPRO_DDK_ROOT", "OHOS_SDK_ROOT", "HIAI_LIB_DIR"):
        case["env"].pop(name, None)
    ddk = case["root"] / "internal ddk/ddk/ai_ddk_lib/lib64"
    linked = _shared_elf(ddk / "libhiai.so", "SIMULATED-LINK-LIBRARY")
    adjacent = _shared_elf(ddk / "libhiai_ir.so", "SIMULATED-ADJACENT-LIBRARY")
    native = case["root"] / "ohos native sdk"
    toolchain = native / "build/cmake/ohos.toolchain.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text("# SIMULATED SDK toolchain marker; no compilation\n")
    cxx = _shared_elf(native / "llvm/lib/aarch64-linux-ohos/libc++_shared.so",
                      "SIMULATED-SDK-CXX-LIBRARY")
    command = {"argv": ["cmake", "-S", "SIMULATED-SOURCE", "-B", str(build / "build"),
                        "-DCMAKE_TOOLCHAIN_FILE=" + str(toolchain),
                        "-DOHOS_ARCH=arm64-v8a", "-DOHOS_STL=c++_shared"],
               "headers_only": False,
               "libraries": [{"path": str(linked),
                              "sha256": hashlib.sha256(linked.read_bytes()).hexdigest()}]}
    result = {"status": "LINKED", "device_validation": "PENDING",
              "binary_sha256": hashlib.sha256(runner.read_bytes()).hexdigest()}
    _json(build / "command.json", command)
    _json(build / "result.json", result)
    return {"build": build, "command": command, "result": result,
            "linked": linked, "adjacent": adjacent, "cxx": cxx}


def test_automatic_runtime_discovery_uses_recorded_ddk_and_sdk_without_library_flags(case):
    automatic = _automatic_build(case)
    assert "--lib-dir" not in case["argv"] and "--hiai-build-dir" not in case["argv"]
    result = _run(case)
    assert result.returncode == 0, result.stdout + result.stderr
    uploads = {entry["remote"]: entry for entry in (
        json.loads(line) for line in Path(case["env"]["FAKE_HDC_UPLOADS"]).read_text().splitlines())}
    board_root = case["env"]["FAKE_BOARD_ROOT"]
    for key in ("linked", "adjacent", "cxx"):
        local = automatic[key]
        record = uploads[board_root + "/lib/" + local.name]
        assert record["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert board_root + "/lib/libhiai-fixture.so" not in uploads
    for name in ("preflight.json", "result.json"):
        association = json.loads((case["output"] / name).read_text())["hiai_build"]
        assert association["build_dir"] == str(automatic["build"])
        assert association["command_binding"] == "legacy-unbound"
        assert association["binary_sha256"] == automatic["result"]["binary_sha256"]
        assert association["missing_from_uploads"] == []
        assert association["link_libraries"][0]["status"] == "matched"
    assert json.loads((case["output"] / "comparison.json").read_text())["status"] == "PASS"


@pytest.mark.parametrize("bad", ["changed-link-library", "wrong-binary-record",
                                  "wrong-library-architecture", "conflicting-library-names",
                                  "missing-command", "missing-result"])
def test_automatic_runtime_discovery_rejects_inconsistent_inputs_before_hdc(case, bad):
    automatic = _automatic_build(case)
    build, command, result = automatic["build"], automatic["command"], automatic["result"]
    if bad == "changed-link-library":
        with automatic["linked"].open("ab") as stream:
            stream.write(b"CHANGED-AFTER-BUILD")
    elif bad == "wrong-binary-record":
        result["binary_sha256"] = hashlib.sha256(b"OTHER-BINARY").hexdigest()
        _json(build / "result.json", result)
    elif bad == "wrong-library-architecture":
        _shared_elf(automatic["linked"], "SIMULATED-WRONG-X86-LIBRARY", machine=62)
        command["libraries"][0]["sha256"] = hashlib.sha256(automatic["linked"].read_bytes()).hexdigest()
        _json(build / "command.json", command)
    elif bad == "conflicting-library-names":
        duplicate = _shared_elf(case["root"] / "other library family/libhiai.so",
                                "SIMULATED-INCOMPATIBLE-LIBRARY")
        command["libraries"].append({"path": str(duplicate),
                                     "sha256": hashlib.sha256(duplicate.read_bytes()).hexdigest()})
        _json(build / "command.json", command)
    elif bad == "missing-command":
        (build / "command.json").unlink()
    else:
        (build / "result.json").unlink()
    execution = _run(case)
    assert execution.returncode != 0
    assert not _calls(case) and not _calls(case, runner=True)
    assert not case["board"].exists()
    assert _cleanup(case) == "NOT_CREATED"
    assert (case["output"] / "failure.json").is_file()


@pytest.mark.parametrize("failure,stage,exitcode", [("remote-exit", "benchmark", 17),
                                                    ("probe-exit", "probe", 19)])
def test_failed_runtime_captures_verified_hilog_without_extra_cli_arguments(case, failure, stage, exitcode):
    _automatic_build(case)
    result = _run(case, failure)
    assert result.returncode != 0
    assert f"remote {stage} failed with exit code {exitcode}" in result.stderr
    assert "SIMULATED HIAI Load model failure" in result.stderr
    evidence = json.loads((case["output"] / "result.json").read_text())
    assert evidence["hilog"]["status"] == "CAPTURED"
    assert evidence["hilog"]["returncode"] == 0
    assert evidence["stage"] == stage
    assert f"remote {stage} failed with exit code {exitcode}" in evidence["error"]
    names = {"hiai-hilog.txt", "hiai-hilog.stderr", "hiai-hilog.exitcode"}
    assert names <= set(evidence["retrieved_files"])
    for name in names:
        local = case["output"] / name
        assert local.read_bytes() == (case["board"] / name).read_bytes()
        assert evidence["retrieved_sha256"][name] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert (case["output"] / "hiai-hilog.stderr").read_text() == "SIMULATED hilog reader diagnostic\n"
    assert (case["output"] / "hiai-hilog.exitcode").read_text().strip() == "0"
    hilog_calls = [args for args in _calls(case) if "hilog -x" in " ".join(args)]
    assert len(hilog_calls) == 1
    assert evidence["hilog"]["command"] == ["hilog", "-x"]
    assert "--lib-dir" not in case["argv"] and "--hiai-build-dir" not in case["argv"]
    assert case["board"].is_dir() and _cleanup(case) == "RETAINED"
    assert len(_calls(case, runner=True)) == (1 if stage == "probe" else 2)


def test_unavailable_hilog_preserves_original_benchmark_failure(case):
    _automatic_build(case)
    case["env"]["FAKE_HILOG_FAILURE"] = "unavailable"
    result = _run(case, "remote-exit")
    assert result.returncode != 0
    assert "remote benchmark failed with exit code 17" in result.stderr
    evidence = json.loads((case["output"] / "result.json").read_text())
    assert evidence["hilog"]["status"] == "UNAVAILABLE"
    assert evidence["hilog"]["returncode"] == 127
    assert "remote benchmark failed with exit code 17" in evidence["error"]
    assert (case["output"] / "benchmark.exitcode").read_text().strip() == "17"
    assert (case["output"] / "hiai-hilog.exitcode").read_text().strip() == "127"
    assert (case["output"] / "hiai-hilog.stderr").read_text() == "SIMULATED hilog unavailable\n"
    assert case["board"].is_dir() and _cleanup(case) == "RETAINED"


def test_corrupt_hilog_transfer_is_not_printed_or_claimed_captured(case):
    _automatic_build(case)
    case["env"]["FAKE_HILOG_FAILURE"] = "receive-corruption"
    result = _run(case, "remote-exit")
    assert result.returncode != 0
    assert "remote benchmark failed with exit code 17" in result.stderr
    assert "SIMULATED HIAI Load model failure" not in result.stderr
    assert "SIMULATED HIAI corrupt transfer must not print" not in result.stderr
    evidence = json.loads((case["output"] / "result.json").read_text())
    assert evidence["hilog"]["status"] == "FAILED"
    assert "remote benchmark failed with exit code 17" in evidence["error"]
    assert "hiai-hilog.txt" not in evidence["retrieved_files"]
    assert "hiai-hilog.txt" not in evidence["retrieved_sha256"]
    assert not (case["output"] / "hiai-hilog.txt").exists()
    assert (case["board"] / "hiai-hilog.txt").is_file()
    retained = list((case["output"] / "received").glob("*hiai-hilog.txt"))
    assert retained and all(path.read_bytes() != (case["board"] / "hiai-hilog.txt").read_bytes()
                            for path in retained)
    assert case["board"].is_dir() and _cleanup(case) == "RETAINED"


def test_hilog_cli_argument_error_is_unavailable_despite_zero_exit(case):
    _automatic_build(case)
    case["env"]["FAKE_HILOG_FAILURE"] = "argument-error-zero"
    result = _run(case, "remote-exit")
    assert result.returncode != 0
    assert "remote benchmark failed with exit code 17" in result.stderr
    evidence = json.loads((case["output"] / "result.json").read_text())
    assert evidence["hilog"]["status"] == "UNAVAILABLE"
    assert evidence["hilog"]["returncode"] == 0
    assert evidence["hilog"].get("reason")
    assert "remote benchmark failed with exit code 17" in evidence["error"]
    assert (case["output"] / "benchmark.exitcode").read_text().strip() == "17"
    assert (case["output"] / "hiai-hilog.exitcode").read_text().strip() == "0"
    text = (case["output"] / "hiai-hilog.txt").read_text()
    assert text == "Mutlti commands can't be used in combination [CODE: -31]\n"
    assert evidence["retrieved_sha256"]["hiai-hilog.txt"] == hashlib.sha256(text.encode()).hexdigest()
    assert case["board"].is_dir() and _cleanup(case) == "RETAINED"

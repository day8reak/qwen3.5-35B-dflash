"""Hash-checked HDC deployment and paired HIAI benchmark orchestration.

This module deliberately uses only the Python standard library and lightweight
project helpers. It neither builds a runner nor changes board system settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import subprocess
import sys
import uuid

from .artifacts import checked_file, output_path, save_json
from .hashing import sha256_file


def _positive(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omc", required=True, help="compiled OMC pair directory")
    p.add_argument("--runner", required=True, help="linked HarmonyOS ARM64 HIAI ELF executable")
    p.add_argument("--reference", required=True, help="matching successful reference.json")
    p.add_argument("--lib-dir", action="append", default=[], help="directory of required *.so* libraries; repeatable")
    p.add_argument("--output", required=True, help="new output directory below AI_RUN_DIR")
    p.add_argument("--board-root", help="new remote directory; existing directories are never reused")
    p.add_argument("--board-cleanup", choices=("all", "omc", "none"), default="all",
                   help="after successful retrieval/comparison: remove the run directory (all, default), only OMC files (omc), or retain everything (none)")
    p.add_argument("--firmware-id", help="otherwise query param get const.ohos.fullname")
    p.add_argument("--hdc", default=os.environ.get("HDC", "hdc"))
    p.add_argument("--timeout-seconds", type=_positive, default=3600, help="timeout of each HDC invocation")
    return p


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{label} must be nonempty text without control characters")
    return value


def _relative(value):
    _text(value, "payload path")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"payload path must be a normalized relative path: {value!r}")
    return path


def _record(root, record):
    path = _relative(record["path"])
    if type(record.get("bytes")) is not int or record["bytes"] < 0:
        raise ValueError(f"invalid byte count for {path}")
    if not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError(f"invalid SHA256 for {path}")
    return checked_file(root, record)


def _file_record(path, remote):
    return {"local": str(path), "remote": str(remote), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _preflight(args):
    sn, bip = _text(os.environ.get("SN"), "SN"), _text(os.environ.get("BIP"), "BIP")
    if sn.startswith("-") or bip.startswith("-"):
        raise ValueError("SN and BIP cannot start with an option prefix")
    host, separator, port = bip.rpartition(":")
    if (not separator or not host or any(c.isspace() for c in host)
            or not re.fullmatch(r"[0-9]+", port) or not 1 <= int(port) <= 65535):
        raise ValueError("BIP must be the remote HDC server address including its actual port (IP:port, port 1..65535)")
    _text(args.hdc, "HDC executable")
    root = Path(args.omc).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_kind") != "qwen35-cpro-omc-pair" or manifest.get("status") != "COMPILED":
        raise ValueError("expected a successfully COMPILED OMC pair")
    if (manifest.get("abi_version") != 2 or manifest.get("feature_dtype") != "torch.float16"
            or manifest.get("precision") != "torch.float16"):
        raise ValueError("HIAI runner requires ABI v2 and FP16 features/precision")
    for field in ("sequence_length", "vocab_size", "feature_width", "draft_width"):
        if type(manifest.get(field)) is not int or not 0 < manifest[field] <= 2147483647:
            raise ValueError(f"invalid manifest {field}")
    _text(manifest.get("ddk_id"), "manifest DDK identity")
    vocab = manifest["vocab_size"]
    special = manifest.get("eos_token_ids")
    if not isinstance(special, list) or any(type(v) is not int or not 0 <= v < vocab for v in special):
        raise ValueError("invalid manifest EOS token IDs")
    if type(manifest.get("pad_token_id")) is not int or not 0 <= manifest["pad_token_id"] < vocab:
        raise ValueError("invalid manifest PAD token ID")
    graphs = manifest.get("graphs")
    if (not isinstance(graphs, list) or len(graphs) != 2
            or {g.get("name") for g in graphs} != {"target", "draft"}):
        raise ValueError("OMC bundle requires exactly target and draft graphs")
    files = {"manifest.json": _file_record(manifest_path, "models/manifest.json")}
    for graph in graphs:
        omc_record = {"path": graph["omc"], "bytes": graph["bytes"], "sha256": graph["sha256"]}
        _record(root, omc_record)
        payloads = graph.get("payloads", [omc_record])
        if not isinstance(payloads, list) or not payloads:
            raise ValueError(f"empty or invalid payload list: {graph['name']}")
        if not any(record == omc_record or all(record.get(k) == v for k, v in omc_record.items()) for record in payloads):
            raise ValueError(f"OMC missing from matching hash-locked payloads: {graph['name']}")
        for record in payloads:
            path = _record(root, record)
            rel = record["path"]
            if rel == "manifest.json":
                raise ValueError("graph payload cannot replace manifest.json")
            entry = _file_record(path, PurePosixPath("models") / rel)
            if rel in files and files[rel] != entry:
                raise ValueError(f"conflicting payload records: {rel}")
            files[rel] = entry
    reference_path = Path(args.reference).expanduser().resolve()
    reference = json.loads(reference_path.read_text())
    if reference.get("status") != "PASS":
        raise ValueError("reference must have status PASS")
    for key in ("source", "precision", "quantization", "sequence_length"):
        if reference.get(key) != manifest.get(key) or (key == "source" and reference.get(key) is None):
            raise ValueError(f"reference/OMC {key} mismatch")
    prompt, limit = reference.get("prompt_token_ids"), reference.get("max_new_tokens")
    if not isinstance(prompt, list) or not prompt or any(type(v) is not int or not 0 <= v < vocab for v in prompt):
        raise ValueError("reference prompt token IDs are empty or outside the vocabulary")
    if type(limit) is not int or limit <= 0 or len(prompt) + limit - 1 > manifest["sequence_length"]:
        raise ValueError("reference generation limit exceeds OMC sequence gear")
    runner = Path(args.runner).expanduser().resolve()
    with runner.open("rb") as stream:
        header = stream.read(64)
    if (len(header) < 64 or header[:7] != b"\x7fELF\x02\x01\x01"
            or struct.unpack_from("<H", header, 18)[0] != 183
            or struct.unpack_from("<H", header, 16)[0] not in (2, 3)):
        raise ValueError("runner must be a linked ELF64 little-endian AArch64 executable (machine 183)")
    uploads = list(files.values()) + [_file_record(runner, "bin/qwen35_dflash_hiai")]
    library_names = set()
    for directory in args.lib_dir:
        folder = Path(directory).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"library directory does not exist: {folder}")
        for library in sorted(folder.glob("*.so*")):
            _relative(library.name)
            if not library.is_file():
                raise ValueError(f"library is not a file (or has a broken link): {library}")
            if library.name in library_names:
                raise ValueError(f"conflicting library name: {library.name}")
            library_names.add(library.name)
            uploads.append(_file_record(library.resolve(), PurePosixPath("lib") / library.name))
    board = args.board_root or ("/data/local/tmp/qwen35-dflash-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex)
    _text(board, "board root")
    bp = PurePosixPath(board)
    if not bp.is_absolute() or bp == PurePosixPath("/") or str(bp) != board or ".." in bp.parts or "\\" in board:
        raise ValueError("board root must be a normalized absolute path other than /")
    if args.firmware_id is not None:
        _text(args.firmware_id, "firmware identity")
    return {"sn": sn, "bip": bip, "root": root, "board": bp, "manifest": manifest,
            "reference": reference, "reference_path": reference_path, "uploads": uploads,
            "manifest_sha256": sha256_file(manifest_path), "reference_sha256": sha256_file(reference_path)}


class Hdc:
    def __init__(self, executable, sn, bip, output, timeout):
        self.prefix = [executable, "-t", sn, "-s", bip]
        self.output, self.timeout, self.count = output, timeout, 0

    def call(self, arguments, label, *, timeout=None):
        self.count += 1
        argv = self.prefix + list(arguments)
        record = {"argv": argv, "label": label, "returncode": None, "stdout": "", "stderr": ""}
        try:
            result = subprocess.run(argv, capture_output=True, text=True, errors="replace",
                                    timeout=self.timeout if timeout is None else timeout, check=False)
            record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            for key in ("stdout", "stderr"):
                value = getattr(error, key, None)
                if value is not None:
                    record[key] = value.decode(errors="replace") if isinstance(value, bytes) else value
            raise
        finally:
            save_json(self.output / "calls" / f"{self.count:04d}-{label}.json", record)
        if result.returncode != 0:
            raise RuntimeError(f"HDC {label} failed with exit code {result.returncode}; see calls/{self.count:04d}-{label}.json")
        return result.stdout

    def shell(self, command, label, *, allow_failure=False, timeout=None):
        marker = "__QWEN35_HDC_" + uuid.uuid4().hex + "__"
        payload = "( " + command + "\n); rc=$?; printf '\\n" + marker + "=%s\\n' \"$rc\"; exit \"$rc\""
        stdout = self.call(["shell", "sh -c " + shlex.quote(payload)], label, timeout=timeout)
        matches = list(re.finditer(r"^" + re.escape(marker) + r"=([0-9]+)\r?$", stdout, re.MULTILINE))
        if len(matches) != 1:
            raise RuntimeError(f"HDC {label} missing unique remote exit marker; remote success is unverified")
        status = int(matches[0].group(1))
        if status and not allow_failure:
            raise RuntimeError(f"remote {label} failed with exit code {status}; HDC process exit was zero")
        return status, stdout[:matches[0].start()].rstrip("\r\n")

    def send(self, local, remote, label="send"):
        return self.call(["file", "send", str(local), str(remote)], label)

    def receive(self, remote, local, *, timeout=None):
        self.call(["file", "recv", str(remote), str(local)], "receive", timeout=timeout)
        if not local.is_file():
            raise RuntimeError(f"HDC reported success without retrieving {remote}")


def _script(board, argv, stdout, stderr, exitcode):
    return ("#!/system/bin/sh\ncd " + shlex.quote(str(board)) + " || exit 1\n"
            "export LD_LIBRARY_PATH=" + shlex.quote(str(board / "lib")) + '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            + shlex.join(argv) + " > " + shlex.quote(stdout) + " 2> " + shlex.quote(stderr) + "\n"
            "rc=$?\nprintf '%s\\n' \"$rc\" > " + shlex.quote(exitcode) + "\nexit \"$rc\"\n")


def _check_local(entry):
    path = Path(entry["local"])
    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"upload source changed since preflight: {path}")


def _upload(hdc, board, entries, checksum):
    for entry in entries:
        _check_local(entry)
        hdc.send(entry["local"], board / entry["remote"])
    checksum.write_text("".join(entry["sha256"] + "  " + entry["remote"] + "\n" for entry in entries))
    hdc.send(checksum, board / checksum.name, "send-checksums")
    hdc.shell("cd " + shlex.quote(str(board)) + " && sha256sum -c " + shlex.quote(checksum.name), "verify-checksums")


def _cleanup_remote(hdc, board, owner_token, *, mode="all", omc_files=()):
    # Delete only the freshly created directory whose run marker still matches.
    # Never follow a substituted root/marker symlink or retry after a failure.
    root = shlex.quote(str(board))
    marker = shlex.quote(str(board / ".qwen35-hdc-owner"))
    guard = (f"test -d {root} && test ! -L {root} && "
             f"test -f {marker} && test ! -L {marker} && "
             f'test "$(cat {marker})" = {shlex.quote(owner_token)}')
    if mode == "all":
        hdc.shell(guard + f" && rm -rf -- {root} && test ! -e {root} && test ! -L {root}", "cleanup-board")
    elif mode == "omc":
        # Enumerate only graph OMC entries, not every payload: external weights,
        # manifests, logs, libraries and directory structure must remain intact.
        paths = [board / _relative(path) for path in omc_files]
        parents = {parent for path in paths for parent in path.parents if parent != board and board in parent.parents}
        checks = [guard]
        checks += [f"test -d {shlex.quote(str(p))} && test ! -L {shlex.quote(str(p))}" for p in sorted(parents)]
        checks += [f"test ! -L {shlex.quote(str(p))}" for p in paths]
        if not paths:
            raise ValueError("OMC cleanup requires manifest graph files")
        checks.append("rm -f -- " + " ".join(shlex.quote(str(p)) for p in paths))
        checks += [f"test ! -e {shlex.quote(str(p))} && test ! -L {shlex.quote(str(p))}" for p in paths]
        hdc.shell(" && ".join(checks), "cleanup-omc")
    else:
        raise ValueError(f"unsupported cleanup mode: {mode}")


def execute(args):
    out = output_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    stage, context, hdc, remote_created = "preflight", None, None, False
    collected, recovered_hashes, recovery_errors = set(), {}, []
    result = {"status": "FAILED", "device_validation": "PENDING",
              "board_cleanup": {"mode": args.board_cleanup,
                                "policy": "after-successful-retrieval-and-comparison", "status": "NOT_CREATED"},
              "scope": "HIAI paired benchmark with token-and-stop reference comparison",
              "limitations": ["Probe and runner evidence do not independently prove per-operator NPU placement.",
                              "A host timeout does not prove the remote process stopped; the remote directory is retained."]}

    def collect(names, *, recovery=False):
        for name in names:
            if name in collected:
                continue
            try:
                timeout = min(30, args.timeout_seconds) if recovery else None
                status, _ = hdc.shell("test -f " + shlex.quote(str(context["board"] / name)), "check-result", allow_failure=True, timeout=timeout)
                if status:
                    continue
                _, checksum = hdc.shell("sha256sum " + shlex.quote(str(context["board"] / name)),
                                        "hash-result", timeout=timeout)
                checksum_lines = checksum.splitlines()
                if len(checksum_lines) != 1 or not re.match(r"^[0-9a-f]{64} [ *]", checksum_lines[0]):
                    raise RuntimeError(f"remote SHA256 response is invalid for {name}")
                expected_hash = checksum_lines[0][:64]
                temporary = out / "received" / f"{hdc.count + 1:04d}-{name}"
                temporary.parent.mkdir(exist_ok=True)
                hdc.receive(context["board"] / name, temporary, timeout=timeout)
                if sha256_file(temporary) != expected_hash:
                    raise RuntimeError(f"retrieved SHA256 differs from remote file: {name}")
                if (out / name).exists():
                    raise FileExistsError(out / name)
                temporary.rename(out / name)
                recovered_hashes[name] = expected_hash
                collected.add(name)
            except Exception as error:
                recovery_errors.append({"file": name, "error": f"{type(error).__name__}: {error}"})

    try:
        context = _preflight(args)
        board, manifest, reference = context["board"], context["manifest"], context["reference"]
        result.update(serial=context["sn"], hdc_server=context["bip"], board_root=str(board),
                      manifest_sha256=context["manifest_sha256"], reference_sha256=context["reference_sha256"],
                      source=manifest["source"], ddk_id=manifest["ddk_id"],
                      precision=manifest["precision"], quantization=manifest.get("quantization"))
        uploads = context["uploads"]
        result["runner_sha256"] = next(r["sha256"] for r in uploads if r["remote"] == "bin/qwen35_dflash_hiai")
        save_json(out / "preflight.json", {**result, "status": "PASS", "uploads": uploads})
        save_json(out / "prompt.json", {"prompt_token_ids": reference["prompt_token_ids"]})
        owner_token = uuid.uuid4().hex
        owner_file = out / ".qwen35-hdc-owner"
        owner_file.write_text(owner_token + "\n")
        uploads.append(_file_record(owner_file, owner_file.name))
        probe_script = out / "probe.sh"
        probe_script.write_text(_script(board, [str(board / "bin/qwen35_dflash_hiai"), "--probe"],
                                        "runtime-probe.json", "probe.stderr", "probe.exitcode"))
        uploads.append(_file_record(probe_script, "probe.sh"))
        save_json(out / "initial-upload-plan.json", {"board_root": str(board), "files": uploads})
        hdc = Hdc(args.hdc, context["sn"], context["bip"], out, args.timeout_seconds)
        stage = "create-remote-directory"
        hdc.shell("mkdir " + shlex.quote(str(board)), "create-root")
        remote_created = True
        result["board_cleanup"]["status"] = "RETAINED"
        directories = {board / "models", board / "bin", board / "lib"}
        directories.update((board / e["remote"]).parent for e in uploads)
        hdc.shell("mkdir -p " + " ".join(shlex.quote(str(p)) for p in sorted(directories)), "create-subdirs")
        stage = "upload-and-verify"
        _upload(hdc, board, uploads, out / "payloads.sha256")
        hdc.shell("chmod 755 " + shlex.quote(str(board / "bin/qwen35_dflash_hiai")), "runner-permissions")
        stage = "probe"
        try:
            hdc.shell("sh " + shlex.quote(str(board / "probe.sh")), "probe")
        finally:
            collect(("runtime-probe.json", "probe.stderr", "probe.exitcode"))
        if not {"runtime-probe.json", "probe.stderr", "probe.exitcode"} <= collected:
            raise RuntimeError("could not retrieve and hash-verify all HIAI probe outputs; see retrieval_errors")
        if (out / "probe.exitcode").read_text().strip() != "0":
            raise RuntimeError("HIAI probe exited nonzero")
        probe = json.loads((out / "runtime-probe.json").read_text())
        if probe.get("npu_supported") is not True:
            raise RuntimeError("HIAI probe did not report npu_supported=true")
        runtime = _text(probe.get("runtime_version"), "probed runtime version")
        result["runtime_version"] = runtime
        stage = "firmware-identity"
        firmware = args.firmware_id
        if firmware is None:
            _, firmware = hdc.shell("param get const.ohos.fullname", "firmware")
            firmware = firmware.strip()
            if not firmware or "\n" in firmware or re.search(r"errNum|Get parameter.*fail|error:", firmware, re.IGNORECASE):
                raise RuntimeError("param get did not return a valid firmware identity; supply the actual --firmware-id")
        result["firmware_id"] = _text(firmware, "firmware identity")
        stage = "benchmark-command"
        from .cli import parser as cli_parser, execute as cli_execute
        cli_execute(cli_parser().parse_args(["board-command", "--omc", str(context["root"]),
            "--prompt", str(out / "prompt.json"), "--output", str(out / "board-command.json"),
            "--board-model-dir", str(board / "models"), "--board-output", str(board / "benchmark.json"),
            "--runner", str(board / "bin/qwen35_dflash_hiai"), "--runtime-version", runtime,
            "--firmware-id", firmware, "--max-new-tokens", str(reference["max_new_tokens"]), "--mode", "benchmark"]))
        command = json.loads((out / "board-command.json").read_text())
        run_script = out / "run.sh"
        run_script.write_text(_script(board, command["argv"], "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode"))
        run_record = _file_record(run_script, "run.sh")
        _upload(hdc, board, [run_record], out / "run.sha256")
        save_json(out / "upload-plan.json", {"board_root": str(board), "files": uploads + [run_record]})
        stage = "benchmark"
        try:
            hdc.shell("sh " + shlex.quote(str(board / "run.sh")), "benchmark")
        finally:
            collect(("benchmark.json", "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode"))
        if not {"benchmark.json", "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode"} <= collected:
            raise RuntimeError("could not retrieve and hash-verify all benchmark outputs; see retrieval_errors")
        if (out / "benchmark.exitcode").read_text().strip() != "0":
            raise RuntimeError("HIAI benchmark exited nonzero")
        stage = "compare"
        if (sha256_file(context["root"] / "manifest.json") != context["manifest_sha256"]
                or sha256_file(context["reference_path"]) != context["reference_sha256"]):
            raise ValueError("manifest or reference changed since preflight")
        cli_execute(cli_parser().parse_args(["compare", "--omc", str(context["root"]),
            "--reference", str(context["reference_path"]), "--report", str(out / "benchmark.json"),
            "--output", str(out / "comparison.json")]))
        result.update(device_validation="PASS", comparison_sha256=sha256_file(out / "comparison.json"),
                      benchmark_sha256=sha256_file(out / "benchmark.json"),
                      probe_sha256=sha256_file(out / "runtime-probe.json"),
                      upload_plan_sha256=sha256_file(out / "upload-plan.json"))
        stage = "cleanup"
        if args.board_cleanup != "none":
            omc_files = [str(PurePosixPath("models") / graph["omc"]) for graph in manifest["graphs"]]
            _cleanup_remote(hdc, board, owner_token, mode=args.board_cleanup, omc_files=omc_files)
            result["board_cleanup"]["status"] = "REMOVED" if args.board_cleanup == "all" else "OMC_FILES_REMOVED"
            if args.board_cleanup == "omc":
                result["board_cleanup"]["removed_files"] = omc_files
        result.update(status="PASS", stage="complete")
    except Exception as error:
        if stage == "cleanup":
            # The command may have partly completed or the connection may have
            # failed after removal. Preserve local evidence; do not claim removal.
            result["board_cleanup"]["status"] = "FAILED"
        if remote_created:
            collect(("runtime-probe.json", "probe.stderr", "probe.exitcode", "benchmark.json",
                     "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode"), recovery=True)
        result.update(stage=stage, error=f"{type(error).__name__}: {error}")
        if stage in ("probe", "benchmark", "compare"):
            result["device_validation"] = "FAILED"
        save_json(out / "failure.json", {**result, "retrieval_errors": recovery_errors})
        raise
    finally:
        result.update(hdc_invocations=hdc.count if hdc else 0, retrieved_files=sorted(collected),
                      retrieved_sha256=recovered_hashes, retrieval_errors=recovery_errors)
        save_json(out / "result.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as error:
        print(f"run-hdc: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

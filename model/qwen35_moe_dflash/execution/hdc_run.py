"""Hash-checked HDC deployment and paired HIAI benchmark orchestration.

This module deliberately uses only the Python standard library and lightweight
project helpers. It neither builds a runner nor changes board system settings.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import struct
import subprocess
import sys
import time
import uuid

from qwen35_moe_dflash.common.artifacts import checked_file, output_path, save_json
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.common.elf_dynamic import dynamic_info
from qwen35_moe_dflash.toolchain.hiai_deployment import BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers



_HILOG_RECORD = re.compile(
    r"^(?P<date>\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2}\.\d{3,})\s+"
    r"(?P<pid>[1-9]\d*)\s+\d+\s+(?P<level>[DIWEF])\s+(?P<tag>[^\s:]+):(?P<message>.*)$")
_HILOG_APP = re.compile(r"(?:^|/)qwen35_dflash_hiai(?:/|$)")
_HILOG_INIT = re.compile(r"HIAI_Foundation_Init\(\d+\).*init hiai foundation success", re.IGNORECASE)
_HILOG_RESTORE_FAILURE = re.compile(r"(?:RestoreFromBuffer|RestoreFromFile|restore model).*(?:fail|false|null)", re.IGNORECASE)
_HILOG_RUNTIME_OPERATION = re.compile(
    r"\b(?:Init|ComputeInit|GenerateOp|ReshapeUpdate|ParseModel|Prepare\w*|Load\w*|RestoreFromBuffer|RestoreFromFile|Run)\b|restore model",
    re.IGNORECASE)
_HILOG_FAILURE = re.compile(r"\b(?:fail\w*|error\w*|false|null|reject\w*)\b|!=", re.IGNORECASE)


def _hilog_summary_details(path, *, recorded_pid=None, candidate_output=None):
    """Bounded candidate-session diagnostics, without attributing a PID to this run.

    A recorded PID restricts selection to that process number, without falling
    back to another process when its records are absent. Otherwise snapshot
    order determines the latest observed app session. PID/date and
    repeated foundation initialization separate candidates; their identity is
    not independently tied to the HDC invocation. The first E can be a handled
    fallback, so also retain the most recent initialization/preparation/loading
    failure window. These are diagnostic candidates, not a root-cause verdict.
    """
    if recorded_pid is not None and (type(recorded_pid) is not int or recorded_pid <= 0):
        raise ValueError("recorded PID must be a positive integer")
    related = re.compile(
        r"hiai|ai_fmk|ai_infra|cpucl|npucl|te_fusion|\bomc\b|"
        r"(?<![A-Za-z0-9_])npu(?![A-Za-z0-9_])|"
        r"qwen35_dflash_hiai|CheckModelCompatibility", re.IGNORECASE)
    transport = re.compile(r"(?:^|[/\s])HDC_LOG(?::|\s)", re.IGNORECASE)
    fallback = deque(maxlen=60)
    matches = 0
    sessions = {}
    candidate = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if transport.search(line):
                continue
            if related.search(line):
                matches += 1
                fallback.append(f"L{number}: {line.rstrip()}"[:1000])
            parsed = _HILOG_RECORD.match(line)
            if parsed is None:
                continue
            if recorded_pid is not None and int(parsed["pid"]) != recorded_pid:
                continue
            if recorded_pid is None and not _HILOG_APP.search(parsed["tag"]):
                continue
            key = (parsed["date"], int(parsed["pid"]))
            initialized = bool(_HILOG_INIT.search(parsed["message"]))
            previous = sessions.get(key)
            if previous is None or (initialized and previous["init_line"] is not None):
                candidate = {"date": key[0], "pid": key[1], "start_line": number,
                             "start_time": parsed["time"], "init_line": number if initialized else None}
                sessions[key] = candidate
            elif initialized:
                previous["init_line"] = number
    if candidate is None:
        if recorded_pid is not None:
            return [], matches, {"method": "recorded-pid-session-candidate", "recorded_pid": recorded_pid,
                "candidate_lines": 0, "status": "NO_MATCHING_RECORDS",
                "scope": "No records match the recorded PID; no other process was substituted. Snapshot rotation may omit the run."}
        return list(fallback), matches, {
            "method": "keyword-tail", "scope": "Entire verified snapshot; keyword candidates, not attributed to this run."}

    head, first_error, failure = [], [], []
    before, tail = deque(maxlen=20), deque(maxlen=15)
    after_error = after_failure = 0
    count = 0
    first_error_line = restore_line = failure_line = None
    end_line, end_time = candidate["start_line"], candidate["start_time"]
    failure_count = 0
    destination = candidate_output.open("x", encoding="utf-8") if candidate_output is not None else nullcontext()
    with destination as complete, path.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if number < candidate["start_line"] or transport.search(line):
                continue
            parsed = _HILOG_RECORD.match(line)
            if (parsed is None or parsed["date"] != candidate["date"]
                    or int(parsed["pid"]) != candidate["pid"]):
                continue
            # Include all tags in this PID, especially MUSL-LDSO loader errors.
            if complete is not None:
                complete.write(f"L{number}: " + line.rstrip("\r\n") + "\n")
            item = (number, f"L{number}: {line.rstrip()}"[:1000])
            end_line, end_time = number, parsed["time"]
            count += 1
            if len(head) < 6:
                head.append(item)
            tail.append(item)
            if after_error:
                first_error.append(item)
                after_error -= 1
            if after_failure:
                failure.append(item)
                after_failure -= 1
            if first_error_line is None and parsed["level"] in ("E", "F"):
                first_error_line = number
                first_error = list(before)[-4:] + [item]
                after_error = 8
            if restore_line is None and _HILOG_RESTORE_FAILURE.search(parsed["message"]):
                restore_line = number
            if (_HILOG_RUNTIME_OPERATION.search(parsed["message"])
                    and _HILOG_FAILURE.search(parsed["message"])):
                failure_line = number
                failure_count += 1
                failure = list(before) + [item]
                after_failure = 5
            before.append(item)
    # 6 head + 13 first-error + 26 latest-runtime-failure + 15 tail <= 60 lines;
    # overlapping windows are deduplicated and displayed in snapshot order.
    selected = dict(head + first_error + failure + list(tail))
    metadata = {"method": "latest-observed-app-session-candidate",
                "candidate_pid": candidate["pid"], "candidate_date": candidate["date"],
                "candidate_start_time": candidate["start_time"],
                "candidate_start_line": candidate["start_line"],
                "candidate_end_line": end_line, "candidate_end_time": end_time,
                "candidate_foundation_init_line": candidate["init_line"],
                "candidate_lines": count, "first_error_line": first_error_line,
                "restore_failure_line": restore_line,
                "runtime_failure_line": failure_line,
                "runtime_failure_candidates": failure_count,
                "scope": "PID/date/init-based session candidate, not confirmed as this run; error windows are not a root-cause verdict."}
    if recorded_pid is not None:
        metadata.update(method="recorded-pid-session-candidate", recorded_pid=recorded_pid,
                        scope="Recorded process PID matched; PID/date/init session is still a candidate because PID reuse is possible. Error windows are not a root-cause verdict.")
    return [selected[number] for number in sorted(selected)], matches, metadata


def _hilog_summary(path):
    """Preserve the legacy (display lines, whole-snapshot keyword count) API."""
    selected, matches, _ = _hilog_summary_details(path)
    return selected, matches


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
    p.add_argument("--hiai-build-dir", help="optional override for build records; otherwise inferred beside --runner")
    p.add_argument("--reference", required=True, help="matching reference.json; prompt/config only with --execution-only, otherwise must be successful")
    p.add_argument("--execution-only", action="store_true",
                   help="execute target and draft once using reference prompt/config; skip accuracy comparison and performance benchmark")
    p.add_argument("--capture-target-features", action="store_true",
                   help="requires --execution-only; capture at most 65536 bytes of target FP16 features for offline diagnosis")
    p.add_argument("--lib-dir", action="append", default=[], help="legacy manual library override; omit to use matching build libraries automatically")
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


def _build_association(build_dir, runner, uploads):
    """Compare recorded link inputs without assuming they are DT_NEEDED entries.

    Original linker input files may have been moved or removed. Their recorded
    digests remain the comparison source; this never resolves or opens them.
    """
    folder = Path(build_dir).expanduser().resolve()
    result_bytes = (folder / "result.json").read_bytes()
    command_bytes = (folder / "command.json").read_bytes()
    result, command = json.loads(result_bytes), json.loads(command_bytes)
    if not isinstance(result, dict) or not isinstance(command, dict):
        raise ValueError("HIAI build result.json and command.json must be JSON objects")
    command_hash = hashlib.sha256(command_bytes).hexdigest()
    if result.get("status") != "LINKED":
        raise ValueError("--hiai-build-dir must contain a LINKED build result")
    if result.get("binary_sha256") != runner["sha256"]:
        raise ValueError(f"HIAI build runner SHA256 mismatch: {folder / 'result.json'} "
                         f"records {result.get('binary_sha256')!r}; runner {runner['local']} "
                         f"has {runner['sha256']}")
    bound = "command_sha256" in result
    if bound and result["command_sha256"] != command_hash:
        raise ValueError(f"HIAI build command SHA256 mismatch: {folder / 'result.json'} "
                         f"records {result['command_sha256']!r}; {folder / 'command.json'} "
                         f"has {command_hash}")
    libraries = command.get("libraries")
    if not isinstance(libraries, list):
        raise ValueError("HIAI build command.json must contain a libraries list")
    deployment = {PurePosixPath(entry["remote"]).name: entry for entry in uploads
                  if PurePosixPath(entry["remote"]).parent == PurePosixPath("lib")}
    association = {"build_dir": str(folder), "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                   "command_sha256": command_hash,
                   "command_binding": "sha256-verified" if bound else "legacy-unbound",
                   "binary_sha256": runner["sha256"], "runner_path": runner["local"],
                   "link_libraries": [], "missing_from_uploads": [],
                   "scope": "Recorded link-input comparison only; DT_NEEDED and board library availability are not verified."}
    for library in libraries:
        if not isinstance(library, dict):
            raise ValueError("HIAI build library records must be JSON objects")
        path = _text(library.get("path"), "recorded link-input path")
        name = Path(path).name
        _relative(name)
        digest = library.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid recorded link-input SHA256: {path}")
        item = {"name": name, "path": path, "sha256": digest}
        upload = deployment.get(name)
        if upload is None:
            association["missing_from_uploads"].append(dict(item))
            item["status"] = "not_uploaded"
        else:
            if digest != upload["sha256"]:
                raise ValueError(f"HIAI library SHA256 mismatch for {name}: link input {path} "
                                 f"has recorded SHA256 {digest}; upload source {upload['local']} "
                                 f"has SHA256 {upload['sha256']}")
            item.update(status="matched", upload_path=upload["local"], upload_sha256=upload["sha256"])
        association["link_libraries"].append(item)
    if not bound:
        association["legacy_note"] = ("Legacy build result has no command_sha256: runner digest matches, "
                                      "but the build result does not hash-bind command.json.")
        print("run-hdc: " + association["legacy_note"], file=sys.stderr)
    if association["missing_from_uploads"]:
        names = ", ".join(item["name"] for item in association["missing_from_uploads"])
        print(f"run-hdc: recorded link inputs not packaged in --lib-dir uploads: {names}. "
              "Their DT_NEEDED status and availability on the board have not been verified; "
              "no libraries are added automatically.", file=sys.stderr)
    return association


def _preflight(args):
    if args.capture_target_features and not args.execution_only:
        raise ValueError("--capture-target-features requires --execution-only")
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
    if args.capture_target_features and 2 * manifest["sequence_length"] * manifest["feature_width"] > 65536:
        raise ValueError("target feature capture exceeds 65536-byte limit")
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
    if not args.execution_only and reference.get("status") != "PASS":
        raise ValueError("reference must have status PASS")
    for key in ("source", "precision", "quantization", "sequence_length"):
        if reference.get(key) != manifest.get(key) or (key == "source" and reference.get(key) is None):
            raise ValueError(f"reference/OMC {key} mismatch")
    prompt, limit = reference.get("prompt_token_ids"), reference.get("max_new_tokens")
    if not isinstance(prompt, list) or not prompt or any(type(v) is not int or not 0 <= v < vocab for v in prompt):
        raise ValueError("reference prompt token IDs are empty or outside the vocabulary")
    if (type(limit) is not int or limit <= 0 or len(prompt) > manifest["sequence_length"]
            or (not args.execution_only and len(prompt) + limit - 1 > manifest["sequence_length"])):
        raise ValueError("reference generation limit exceeds OMC sequence gear")
    runner = Path(args.runner).expanduser().resolve()
    with runner.open("rb") as stream:
        header = stream.read(64)
    if (len(header) < 64 or header[:7] != b"\x7fELF\x02\x01\x01"
            or struct.unpack_from("<H", header, 18)[0] != 183
            or struct.unpack_from("<H", header, 16)[0] not in (2, 3)):
        raise ValueError("runner must be a linked ELF64 little-endian AArch64 executable (machine 183)")
    uploads = list(files.values()) + [_file_record(runner, "bin/qwen35_dflash_hiai")]
    build_dir = getattr(args, "hiai_build_dir", None)
    discovery = None
    library_paths = []
    if not args.lib_dir:
        from qwen35_moe_dflash.toolchain.hiai_deployment import discover_libraries
        automatic = discover_libraries(runner, build_dir)
        build_dir = automatic["build_dir"]
        library_paths = automatic["libraries"]
        discovery = automatic["metadata"]
    for directory in args.lib_dir:
        folder = Path(directory).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"library directory does not exist: {folder}")
        for library in sorted(folder.glob("*.so*")):
            _relative(library.name)
            if not library.is_file():
                raise ValueError(f"library is not a file (or has a broken link): {library}")
            library_paths.append(library)
    library_names = set()
    for library in library_paths:
        _relative(library.name)
        if library.name in library_names:
            raise ValueError(f"conflicting library name: {library.name}")
        library_names.add(library.name)
        uploads.append(_file_record(library.resolve(), PurePosixPath("lib") / library.name))
    runtime_records = [entry for entry in uploads
                       if PurePosixPath(entry["remote"]).parent in (PurePosixPath("lib"), PurePosixPath("bin"))]
    dependencies = [{"name": PurePosixPath(entry["remote"]).name, "path": entry["local"],
                     "sha256": entry["sha256"], "bytes": entry["bytes"],
                     "kind": "runner" if entry["remote"].startswith("bin/") else "library",
                     **dynamic_info(entry["local"])} for entry in runtime_records]
    for entry in runtime_records:
        _check_local(entry)
    blockers = candidate_blockers(dependencies, require_pair=False)
    if blockers:
        raise ValueError("cannot select board HCL deployment: " + "; ".join(blockers))
    excluded = [entry for entry in runtime_records if PurePosixPath(entry["remote"]).parent == PurePosixPath("lib")
                and PurePosixPath(entry["remote"]).name in DDK_HCL_EXCLUDED]
    uploads = [entry for entry in uploads if entry not in excluded]
    deployment = {"mode": "board-hcl", "excluded_library_names": sorted(DDK_HCL_EXCLUDED),
                  "excluded_libraries": excluded, "elf_dependencies": dependencies,
                  "library_search_paths": list(BOARD_LIBRARY_SEARCH_PATHS),
                  "backend_selection": "UNVERIFIED", "runtime_placement": "UNVERIFIED",
                  "scope": "Private deployment omits the DDK HCL pair and adds the board vendor directory to this process's search path. Explicit ELF dependencies are checked; dlopen selection, board ABI and NPU placement require runtime evidence."}
    build = None
    if build_dir:
        runner_record = next(entry for entry in uploads if entry["remote"] == "bin/qwen35_dflash_hiai")
        build = _build_association(build_dir, runner_record, uploads)
    board = args.board_root or ("/data/local/tmp/qwen35-dflash-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex)
    _text(board, "board root")
    bp = PurePosixPath(board)
    if not bp.is_absolute() or bp == PurePosixPath("/") or str(bp) != board or ".." in bp.parts or "\\" in board:
        raise ValueError("board root must be a normalized absolute path other than /")
    if args.firmware_id is not None:
        _text(args.firmware_id, "firmware identity")
    return {"sn": sn, "bip": bip, "root": root, "board": bp, "manifest": manifest,
            "reference": reference, "reference_path": reference_path, "uploads": uploads,
            "hiai_build": build, "runtime_libraries": discovery, "runtime_deployment": deployment,
            "manifest_sha256": sha256_file(manifest_path), "reference_sha256": sha256_file(reference_path)}


class Hdc:
    def __init__(self, executable, sn, bip, output, timeout):
        self.prefix = [executable, "-t", sn, "-s", bip]
        self.output, self.timeout, self.count = output, timeout, 0
        self.last_call = None

    def _diagnostics(self):
        record = self.last_call
        details = []
        for channel in ("stdout", "stderr"):
            value = record[channel].strip()
            if len(value) > 2000:
                value = value[:2000] + " ... [truncated; see full log]"
            details.append(f"HDC {channel}: {value!r}")
        log_path = self.output / "calls" / f"{self.count:04d}-{record['label']}.json"
        details.append(f"Full log: {log_path}")
        return "\n" + "\n".join(details)

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
            self.last_call = record
        if result.returncode != 0:
            raise RuntimeError(f"HDC {label} failed with exit code {result.returncode}" + self._diagnostics())
        return result.stdout

    def shell(self, command, label, *, allow_failure=False, timeout=None):
        marker = "__QWEN35_HDC_" + uuid.uuid4().hex + "__"
        payload = "( " + command + "\n); rc=$?; printf '\\n" + marker + "=%s\\n' \"$rc\"; exit \"$rc\""
        stdout = self.call(["shell", "sh -c " + shlex.quote(payload)], label, timeout=timeout)
        matches = list(re.finditer(r"^" + re.escape(marker) + r"=([0-9]+)\r?$", stdout, re.MULTILINE))
        if len(matches) != 1:
            # Some HDC versions return zero for connection errors. Keep the
            # completion marker as the gate, but show the actual HDC response;
            # do not classify success/failure by version-dependent error text.
            query = shlex.join([self.prefix[0], "-s", self.prefix[4], "list", "targets", "-v"])
            raise RuntimeError(f"HDC {label} missing unique remote exit marker; remote success is unverified"
                               + self._diagnostics() + f"\nInspect devices on the selected HDC server: {query}")
        status = int(matches[0].group(1))
        if status and not allow_failure:
            raise RuntimeError(f"remote {label} failed with exit code {status}; HDC process exit was zero")
        return status, stdout[:matches[0].start()].rstrip("\r\n")

    def send(self, local, remote, label="send"):
        return self.call(["file", "send", str(local), str(remote)], label)

    def receive(self, remote, local, *, timeout=None):
        """Retrieve one new file; a zero client exit alone proves no transfer.

        Only a missing artifact after a zero exit is retried, at most three
        times. Every attempt has a private destination. Existing final paths
        are never passed to HDC or overwritten; callers still verify the
        independently obtained remote checksum after this method returns.
        """
        local = Path(os.path.abspath(os.fspath(local)))
        token = uuid.uuid4().hex
        receipt_path = self.output / "receives" / (token + ".json")
        receipt = {"schema_version": 1, "artifact_kind": "qwen35-hdc-file-receive",
            "status": "FAILED", "remote": str(remote), "local": str(local), "attempts": [],
            "maximum_attempts": 3, "operation": "READ_ONLY_FILE_RECV_ONLY",
            "model_execution": "NOT_RUN", "remote_mutation": "NONE_REQUESTED",
            "checksum_validation": "CALLER_MUST_VERIFY_INDEPENDENT_REMOTE_SHA256",
            "staging_retention": "FAILED_AND_LATE_ATTEMPT_ARTIFACTS_RETAINED"}
        initial_count = self.count
        try:
            _receive_parent(local)
            if os.path.lexists(local):
                raise FileExistsError("HDC receive destination already exists and will not be changed: " + str(local))
            staging = local.parent / (".hdc-receive-" + token)
            staging.mkdir(mode=0o700)
            receipt["staging"] = str(staging)
            for number in range(1, 4):
                directory = staging / ("attempt-" + str(number))
                directory.mkdir(mode=0o700)
                candidate = directory / "payload"
                attempt = {"attempt": number, "destination": str(candidate), "status": "REQUESTED"}
                receipt["attempts"].append(attempt)
                try:
                    self.call(["file", "recv", str(remote), str(candidate)], "receive", timeout=timeout)
                except Exception as error:
                    attempt.update(status="HDC_CALL_FAILED", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    attempt["call_log"] = str(self.output / "calls" / f"{self.count:04d}-receive.json")
                before = _wait_received_file(candidate)
                if before is None:
                    attempt["status"] = "ZERO_EXIT_WITHOUT_LOCAL_ARTIFACT"
                    if number < 3:
                        continue
                    raise RuntimeError("HDC file recv returned zero without creating a local file after three read-only attempts"
                        + self._diagnostics() + "\nReceive audit: " + str(receipt_path))
                attempt.update(status="STABLE_REGULAR_FILE_OBSERVED", file_identity=before)
                _receive_parent(local)
                _receive_parent(candidate)
                # link() is an atomic no-clobber publication on the same
                # filesystem. A concurrent final destination causes EEXIST.
                fd = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    if _receive_identity(os.fstat(stream.fileno())) != before:
                        raise RuntimeError("HDC receive artifact changed before publication")
                    os.link(candidate, local, follow_symlinks=False)
                    current = _receive_identity(local.lstat())
                    if (not stat.S_ISREG(current["mode"]) or current["nlink"] != 2
                            or any(current[key] != before[key] for key in
                                ("device", "inode", "bytes", "mtime_ns", "mode", "uid", "gid"))
                            or _receive_identity(candidate.lstat()) != current
                            or _receive_identity(os.fstat(stream.fileno())) != current):
                        raise RuntimeError("HDC receive artifact changed during no-clobber publication")
                    candidate.unlink()
                    _receive_parent(local)
                    published = _receive_identity(local.lstat())
                    if (not stat.S_ISREG(published["mode"]) or published["nlink"] != 1
                            or any(published[key] != before[key] for key in
                                ("device", "inode", "bytes", "mtime_ns", "mode", "uid", "gid"))):
                        raise RuntimeError("HDC receive final artifact changed during publication")
                receipt.update(status="RECEIVED_NEW_REGULAR_FILE", local_file_identity=published)
                attempt["status"] = "PUBLISHED_WITHOUT_OVERWRITE"
                break
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            if receipt["attempts"] and receipt["attempts"][-1]["status"] == "REQUESTED":
                receipt["attempts"][-1].update(status="ARTIFACT_VALIDATION_FAILED", error=receipt["error"])
            try:
                save_json(receipt_path, receipt)
            except Exception as audit_error:
                raise RuntimeError(f"HDC receive failed: {error}; receive audit could not be saved: {audit_error}") from error
            if self.count > initial_count and "HDC stdout:" not in str(error):
                raise RuntimeError("HDC receive failed: " + str(error) + self._diagnostics()
                    + "\nReceive audit: " + str(receipt_path)) from error
            raise
        save_json(receipt_path, receipt)


def _receive_identity(info):
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns, "mode": info.st_mode,
        "uid": info.st_uid, "gid": info.st_gid, "nlink": info.st_nlink}


def _receive_parent(local):
    """Require existing ordinary parents without following a symlink."""
    current = Path(local.anchor)
    for part in local.parent.parts[1:]:
        current /= part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError("HDC receive requires ordinary destination parents: " + str(current))


def _wait_received_file(path):
    """Allow bounded late materialization, then require an unchanged file."""
    deadline, previous = time.monotonic() + 0.5, None
    while True:
        _receive_parent(path)
        try:
            current = _receive_identity(path.lstat())
        except FileNotFoundError:
            current = None
        if current is not None:
            if not stat.S_ISREG(current["mode"]) or current["nlink"] != 1:
                raise RuntimeError("HDC receive produced a nonregular or shared artifact: " + str(path))
            if current == previous:
                return current
        if time.monotonic() >= deadline:
            if current is not None:
                raise RuntimeError("HDC receive artifact did not become stable within the bounded observation window")
            return None
        previous = current
        time.sleep(0.05)


def _script(board, argv, stdout, stderr, exitcode, *, pidfile=None):
    invocation = shlex.join(argv)
    if pidfile is not None:
        # exec replaces this inner shell, so its $$ is the native process PID.
        # PID capture is auxiliary: a failed write must not prevent execution.
        native = "printf '%s\\n' \"$$\" > " + shlex.quote(pidfile) + "; exec " + invocation
        invocation = "sh -c " + shlex.quote(native)
    return ("#!/system/bin/sh\ncd " + shlex.quote(str(board)) + " || exit 1\n"
            "export LD_LIBRARY_PATH=" + shlex.quote(":".join((str(board / "lib"), *BOARD_LIBRARY_SEARCH_PATHS)))
            + '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            + invocation + " > " + shlex.quote(stdout) + " 2> " + shlex.quote(stderr) + "\n"
            "rc=$?\nprintf '%s\\n' \"$rc\" > " + shlex.quote(exitcode) + "\nexit \"$rc\"\n")


def _check_local(entry):
    path = Path(entry["local"])
    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"upload source changed since preflight: {path}")


def _upload(hdc, board, entries, checksum, *, checksum_connection_retries=0):
    if type(checksum_connection_retries) is not int or not 0 <= checksum_connection_retries <= 2:
        raise ValueError("checksum_connection_retries must be an integer in 0..2")
    for entry in entries:
        _check_local(entry)
        hdc.send(entry["local"], board / entry["remote"])
    checksum.write_text("".join(entry["sha256"] + "  " + entry["remote"] + "\n" for entry in entries))
    hdc.send(checksum, board / checksum.name, "send-checksums")
    command = "cd " + shlex.quote(str(board)) + " && sha256sum -c " + shlex.quote(checksum.name)
    # Repeat only the read-only checksum query after this observed HDC transport
    # failure. Never replay uploads or Process, accept a missing exit marker, or
    # treat an actual remote checksum failure as a retryable connection error.
    # Opt-in bounded probe queries use short attempts. Large-model uploads keep
    # their existing timeout and single-query behavior unless explicitly chosen.
    options = {"timeout": min(30, hdc.timeout)} if checksum_connection_retries else {}
    for attempt in range(checksum_connection_retries + 1):
        try:
            hdc.shell(command, "verify-checksums", **options)
            break
        except RuntimeError as error:
            if attempt == checksum_connection_retries:
                raise
            record = hdc.last_call or {}
            channels = [record.get(name, "").strip() for name in ("stdout", "stderr")]
            daemon_failure = [value for value in channels if value] == ["[Fail]Failed to communicate with daemon"]
            transport_failure = (str(error).startswith("HDC verify-checksums missing unique remote exit marker;")
                                 or str(error).startswith("HDC verify-checksums failed with exit code "))
            if record.get("label") != "verify-checksums" or not daemon_failure or not transport_failure:
                raise
            print(f"hdc-upload: checksum query lost daemon connection; retry {attempt + 2}/{checksum_connection_retries + 1}", flush=True)
            time.sleep(1)


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


def _smoke_checks(report, manifest, reference, runtime, firmware, *, capture_target_features=False):
    """Check completed target/draft execution identity, never numerical accuracy."""
    if not isinstance(report, dict):
        raise ValueError("smoke report must be a JSON object")
    graphs = {graph["name"]: graph for graph in manifest["graphs"]}
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("smoke report must contain execution results")
    expected = {"schema_version": 2, "abi_version": 2, "feature_dtype": "torch.float16",
                "status": "EXECUTION_COMPLETE", "mode": "smoke", "scope": "target-draft-execution",
                "evidence_kind": "device-hiai", "device_execution": "PASS",
                "accuracy": "NOT_EVALUATED", "performance": "NOT_EVALUATED",
                "runtime_version": runtime, "ddk_id": manifest["ddk_id"], "firmware_id": firmware,
                "target_sha256": graphs["target"]["sha256"], "draft_sha256": graphs["draft"]["sha256"],
                "sequence_length": manifest["sequence_length"], "pad_token_id": manifest["pad_token_id"],
                "prompt_token_ids": reference["prompt_token_ids"], "eos_token_ids": manifest["eos_token_ids"]}
    checks = {key: type(report.get(key)) is type(value) and report[key] == value
              for key, value in expected.items()}
    for key in ("prompt_token_ids", "eos_token_ids"):
        checks[key] = checks[key] and all(type(token) is int for token in report[key])
    for name, length in (("target", manifest["sequence_length"]), ("draft", manifest["draft_width"])):
        calls, tokens = execution.get(name + "_calls"), execution.get(name + "_top1")
        checks[name + "_calls"] = type(calls) is int and calls == 1
        checks[name + "_top1"] = (isinstance(tokens, list) and len(tokens) == length
                                  and all(type(token) is int and 0 <= token < manifest["vocab_size"] for token in tokens))
    checks["execution_only"] = not any(key in report for key in ("ordinary", "dflash", "generation", "timing_scope"))
    if capture_target_features:
        payload = execution.get("target_features")
        checks["target_features_present"] = isinstance(payload, dict)
        if isinstance(payload, dict):
            shape = [1, manifest["sequence_length"], manifest["feature_width"]]
            size = 2 * shape[1] * shape[2]
            raw_shape, data = payload.get("shape"), payload.get("data")
            checks.update(
                target_features_dtype=payload.get("dtype") == "float16",
                target_features_encoding=payload.get("encoding") == "hex-le",
                target_features_shape=(isinstance(raw_shape, list) and raw_shape == shape
                                       and all(type(dim) is int for dim in raw_shape)),
                target_features_bytes=(type(payload.get("bytes")) is int and payload["bytes"] == size
                                       and 0 < size <= 65536),
                target_features_data=(0 < size <= 65536 and isinstance(data, str)
                                      and len(data) == 2 * size and re.fullmatch(r"[0-9a-f]+", data) is not None))
    return checks


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
    if args.execution_only:
        result.update(scope="HIAI target-and-draft execution only", device_execution="PENDING",
                      device_validation="NOT_EVALUATED", accuracy="NOT_EVALUATED", performance="NOT_EVALUATED")
        result["board_cleanup"]["policy"] = "after-successful-retrieval-and-execution-report-validation"
    if args.capture_target_features:
        result["capture_target_features"] = True

    def collect(names, *, recovery=False, timeout=None):
        for name in names:
            if name in collected:
                continue
            try:
                call_timeout = timeout if timeout is not None else (min(30, args.timeout_seconds) if recovery else None)
                status, _ = hdc.shell("test -f " + shlex.quote(str(context["board"] / name)), "check-result", allow_failure=True, timeout=call_timeout)
                if status:
                    continue
                _, checksum = hdc.shell("sha256sum " + shlex.quote(str(context["board"] / name)),
                                        "hash-result", timeout=call_timeout)
                checksum_lines = checksum.splitlines()
                if len(checksum_lines) != 1 or not re.match(r"^[0-9a-f]{64} [ *]", checksum_lines[0]):
                    raise RuntimeError(f"remote SHA256 response is invalid for {name}")
                expected_hash = checksum_lines[0][:64]
                temporary = out / "received" / f"{hdc.count + 1:04d}-{name}"
                temporary.parent.mkdir(exist_ok=True)
                hdc.receive(context["board"] / name, temporary, timeout=call_timeout)
                if sha256_file(temporary) != expected_hash:
                    raise RuntimeError(f"retrieved SHA256 differs from remote file: {name}")
                if (out / name).exists():
                    raise FileExistsError(out / name)
                temporary.rename(out / name)
                recovered_hashes[name] = expected_hash
                collected.add(name)
            except Exception as error:
                recovery_errors.append({"file": name, "error": f"{type(error).__name__}: {error}"})

    def collect_pid(operation):
        name = operation + ".pid"
        identity = {"status": "UNAVAILABLE", "file": name,
                    "scope": "Auxiliary exec PID; PID reuse and shared-log session boundaries remain possible."}
        result.setdefault("process_ids", {})[operation] = identity
        collect((name,), recovery=True, timeout=min(10, args.timeout_seconds))
        if name not in collected:
            identity["reason"] = "PID file was not retrieved and hash-verified"
            return
        try:
            with (out / name).open(encoding="ascii") as stream:
                value = stream.read(64).strip()
                if not re.fullmatch(r"[1-9][0-9]*", value) or stream.read(1):
                    raise ValueError("invalid recorded process PID")
            identity.update(status="VERIFIED", pid=int(value), sha256=recovered_hashes[name])
        except (OSError, ValueError, UnicodeError) as error:
            identity["reason"] = f"{type(error).__name__}: {error}"

    def capture_hilog():
        # Diagnostic-only: do not clear logs, change log settings, or start a
        # persistent reader. A snapshot can include unrelated processes/runs.
        if "hilog" in result:
            return
        names = ("hiai-hilog.txt", "hiai-hilog.stderr", "hiai-hilog.exitcode")
        diagnostic = {"status": "FAILED", "returncode": None, "retrieved_files": [],
                      "command": ["hilog", "-x"],
                      "scope": "Recent shared system-log snapshot; may include other processes and runs. Not proof of this run's cause."}
        result["hilog"] = diagnostic
        timeout = min(10, args.timeout_seconds)
        board = context["board"]
        diagnostic["paths"] = {name: {"local": str(out / name), "remote": str(board / name)} for name in names}
        command = ("hilog -x > " + shlex.quote(str(board / names[0]))
                   + " 2> " + shlex.quote(str(board / names[1]))
                   + "\nrc=$?\nprintf '%s\\n' \"$rc\" > " + shlex.quote(str(board / names[2]))
                   + '\nexit "$rc"')
        try:
            status, _ = hdc.shell(command, "capture-hilog", allow_failure=True, timeout=timeout)
            diagnostic["returncode"] = status
        except Exception as capture_error:
            diagnostic["error"] = f"{type(capture_error).__name__}: {capture_error}"
        # All reads, including their remote checksum queries, use the short
        # diagnostic timeout. Partial or hash-mismatched content is never shown.
        try:
            collect(names, recovery=True, timeout=timeout)
            diagnostic["retrieved_files"] = [name for name in names if name in collected]
            if not set(names) <= collected:
                raise RuntimeError("could not retrieve and hash-verify all hilog outputs; see retrieval_errors")
            saved_status = (out / names[2]).read_text().strip()
            if not re.fullmatch(r"[0-9]+", saved_status):
                raise RuntimeError("invalid hiai-hilog.exitcode")
            diagnostic["recorded_exitcode"] = int(saved_status)
            if diagnostic["returncode"] != int(saved_status):
                raise RuntimeError("hilog completion marker and exitcode file do not agree")
            diagnostic["status"] = "CAPTURED" if int(saved_status) == 0 else "UNAVAILABLE"
            # Some hilog versions report argument errors on stdout with exit 0.
            # This exact CLI response is not a HIAI runtime log or a valid snapshot.
            for name in names[:2]:
                with (out / name).open("rb") as stream:
                    header = stream.read(8192).decode("utf-8", errors="replace")
                if any(re.fullmatch(r"(?:Mutlti|Multi) commands can't be used in combination\s*\[CODE:\s*-31\]",
                                    line.strip()) for line in header.splitlines()):
                    diagnostic["status"] = "UNAVAILABLE"
                    diagnostic["reason"] = "hilog rejected the command combination (CODE -31)"
                    break
            print(f"run-hdc: hilog snapshot: {out / names[0]} (shared system logs; may include other runs)", file=sys.stderr)
            if diagnostic["status"] == "CAPTURED":
                operation = "probe" if stage == "probe" else "benchmark"
                identity = result.get("process_ids", {}).get(operation, {})
                recorded_pid = identity.get("pid") if identity.get("status") == "VERIFIED" else None
                diagnostic["process_identity"] = {"operation": operation, **identity}
                candidate_path = out / "hiai-hilog-candidate.txt"
                related, matches, summary_metadata = _hilog_summary_details(
                    out / names[0], recorded_pid=recorded_pid, candidate_output=candidate_path)
                if sha256_file(out / names[0]) != recovered_hashes[names[0]]:
                    raise RuntimeError("HiLog snapshot changed during diagnostic selection")
                diagnostic["summary"] = {"matching_lines": matches, "shown_lines": len(related), **summary_metadata}
                if candidate_path.is_file():
                    diagnostic["candidate_log"] = {"path": str(candidate_path), "sha256": sha256_file(candidate_path),
                        "bytes": candidate_path.stat().st_size, "source_path": str(out / names[0]),
                        "source_sha256": recovered_hashes[names[0]], "selection": summary_metadata,
                        "scope": "All selected candidate records with original snapshot line numbers; may include PID reuse."}
                    print(f"run-hdc: full hilog candidate records: {candidate_path}", file=sys.stderr)
                if related:
                    if "candidate_pid" in summary_metadata:
                        identity_scope = ("recorded process PID matched; PID reuse remains possible"
                                          if recorded_pid is not None else "not confirmed as this run")
                        print(f"run-hdc: hilog diagnostic candidate: PID {summary_metadata['candidate_pid']} "
                              f"on {summary_metadata['candidate_date']}; {identity_scope}, "
                              "and the first error is not necessarily the cause", file=sys.stderr)
                    print("\n".join(related), file=sys.stderr)
                else:
                    message = (f"no records for recorded PID {recorded_pid}; no other process substituted"
                               if recorded_pid is not None else "no matching HIAI/NPU diagnostic lines in the snapshot")
                    print(f"run-hdc: {message}; "
                          "the original runtime failure remains unresolved", file=sys.stderr)
            else:
                reason = diagnostic.get("reason", f"exit {saved_status}")
                print(f"run-hdc: hilog unavailable ({reason}); details: {out / names[0]}, {out / names[1]}", file=sys.stderr)
        except Exception as capture_error:
            diagnostic["status"] = "FAILED"
            diagnostic.setdefault("errors", []).append(f"{type(capture_error).__name__}: {capture_error}")

    try:
        context = _preflight(args)
        board, manifest, reference = context["board"], context["manifest"], context["reference"]
        result.update(serial=context["sn"], hdc_server=context["bip"], board_root=str(board),
                      manifest_sha256=context["manifest_sha256"], reference_sha256=context["reference_sha256"],
                      source=manifest["source"], ddk_id=manifest["ddk_id"],
                      precision=manifest["precision"], quantization=manifest.get("quantization"))
        if context["hiai_build"] is not None:
            result["hiai_build"] = context["hiai_build"]
        if context["runtime_libraries"] is not None:
            result["runtime_libraries"] = context["runtime_libraries"]
        result["runtime_deployment"] = context["runtime_deployment"]
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
                                        "runtime-probe.json", "probe.stderr", "probe.exitcode", pidfile="probe.pid"))
        uploads.append(_file_record(probe_script, "probe.sh"))
        save_json(out / "initial-upload-plan.json", {"board_root": str(board), "files": uploads})
        hdc = Hdc(args.hdc, context["sn"], context["bip"], out, args.timeout_seconds)
        stage = "library-search-paths"
        for index, path in enumerate(BOARD_LIBRARY_SEARCH_PATHS, 1):
            quoted = shlex.quote(path)
            hdc.shell(f"test -d {quoted} && test -r {quoted} && test -x {quoted}",
                      f"library-search-path-{index}")
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
            collect_pid("probe")
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
        from qwen35_moe_dflash.cli import parser as cli_parser, execute as cli_execute
        command_args = ["board-command", "--omc", str(context["root"]),
            "--prompt", str(out / "prompt.json"), "--output", str(out / "board-command.json"),
            "--board-model-dir", str(board / "models"), "--board-output", str(board / "benchmark.json"),
            "--runner", str(board / "bin/qwen35_dflash_hiai"), "--runtime-version", runtime,
            "--firmware-id", firmware, "--max-new-tokens", str(reference["max_new_tokens"]),
            "--mode", "smoke" if args.execution_only else "benchmark"]
        if args.capture_target_features:
            command_args.append("--capture-target-features")
        cli_execute(cli_parser().parse_args(command_args))
        command = json.loads((out / "board-command.json").read_text())
        run_script = out / "run.sh"
        run_script.write_text(_script(board, command["argv"], "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode",
                                      pidfile="benchmark.pid"))
        run_record = _file_record(run_script, "run.sh")
        _upload(hdc, board, [run_record], out / "run.sha256")
        save_json(out / "upload-plan.json", {"board_root": str(board), "files": uploads + [run_record]})
        stage = "benchmark"
        try:
            hdc.shell("sh " + shlex.quote(str(board / "run.sh")), "benchmark")
        finally:
            collect(("benchmark.json", "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode"))
            collect_pid("benchmark")
        if not {"benchmark.json", "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode"} <= collected:
            raise RuntimeError("could not retrieve and hash-verify all benchmark outputs; see retrieval_errors")
        if (out / "benchmark.exitcode").read_text().strip() != "0":
            raise RuntimeError("HIAI benchmark exited nonzero")
        stage = "execution-report" if args.execution_only else "compare"
        if (sha256_file(context["root"] / "manifest.json") != context["manifest_sha256"]
                or sha256_file(context["reference_path"]) != context["reference_sha256"]):
            raise ValueError("manifest or reference changed since preflight")
        if args.execution_only:
            report = json.loads((out / "benchmark.json").read_text())
            result["execution_checks"] = _smoke_checks(report, manifest, reference, runtime, firmware,
                                                       capture_target_features=args.capture_target_features)
            if not all(result["execution_checks"].values()):
                failed = ", ".join(key for key, value in result["execution_checks"].items() if not value)
                raise ValueError("smoke execution report failed checks: " + failed)
            result["device_execution"] = "PASS"
            try:
                capture_hilog()
            except Exception as capture_error:
                result.setdefault("hilog", {})["status"] = "FAILED"
                result["hilog"].setdefault("errors", []).append(f"{type(capture_error).__name__}: {capture_error}")
        else:
            cli_execute(cli_parser().parse_args(["compare", "--omc", str(context["root"]),
                "--reference", str(context["reference_path"]), "--report", str(out / "benchmark.json"),
                "--output", str(out / "comparison.json")]))
            result.update(device_validation="PASS", comparison_sha256=sha256_file(out / "comparison.json"))
        result.update(benchmark_sha256=sha256_file(out / "benchmark.json"),
                      probe_sha256=sha256_file(out / "runtime-probe.json"),
                      upload_plan_sha256=sha256_file(out / "upload-plan.json"))
        stage = "cleanup"
        if args.board_cleanup != "none":
            omc_files = [str(PurePosixPath("models") / graph["omc"]) for graph in manifest["graphs"]]
            _cleanup_remote(hdc, board, owner_token, mode=args.board_cleanup, omc_files=omc_files)
            result["board_cleanup"]["status"] = "REMOVED" if args.board_cleanup == "all" else "OMC_FILES_REMOVED"
            if args.board_cleanup == "omc":
                result["board_cleanup"]["removed_files"] = omc_files
        result.update(status="EXECUTION_COMPLETE" if args.execution_only else "PASS", stage="complete")
    except Exception as error:
        result.update(stage=stage, error=f"{type(error).__name__}: {error}")
        if remote_created and stage in ("probe", "benchmark", "execution-report"):
            # Never replace the failed operation with an optional log-capture
            # failure (including local output errors).
            try:
                capture_hilog()
            except Exception as capture_error:
                result.setdefault("hilog", {})["status"] = "FAILED"
                result["hilog"].setdefault("errors", []).append(f"{type(capture_error).__name__}: {capture_error}")
        if stage == "cleanup":
            # The command may have partly completed or the connection may have
            # failed after removal. Preserve local evidence; do not claim removal.
            result["board_cleanup"]["status"] = "FAILED"
        if remote_created:
            collect(("runtime-probe.json", "probe.stderr", "probe.exitcode", "benchmark.json",
                     "benchmark.stdout", "benchmark.stderr", "benchmark.exitcode"), recovery=True)
        if args.execution_only and stage in ("probe", "benchmark", "execution-report"):
            result["device_execution"] = "FAILED"
        elif not args.execution_only and stage in ("probe", "benchmark", "compare"):
            result["device_validation"] = "FAILED"
        save_json(out / "failure.json", {**result, "retrieval_errors": recovery_errors})
        if stage in ("probe", "benchmark"):
            names = ("probe.stderr", "runtime-probe.json") if stage == "probe" else ("benchmark.stderr", "benchmark.stdout")
            for name in names:
                if name not in collected:
                    continue
                log_path = out / name
                try:
                    with log_path.open("rb") as stream:
                        stream.seek(max(0, log_path.stat().st_size - 6000))
                        tail = stream.read().decode("utf-8", errors="replace").strip()
                    print(f"run-hdc: {stage} log: {log_path}", file=sys.stderr)
                    if tail:
                        print("\n".join(tail.splitlines()[-60:]), file=sys.stderr)
                except Exception as display_error:
                    result.setdefault("diagnostic_display_errors", []).append(
                        {"file": name, "error": f"{type(display_error).__name__}: {display_error}"})
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

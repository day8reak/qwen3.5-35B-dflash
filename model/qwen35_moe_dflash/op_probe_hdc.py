"""Hash-checked HDC transport for independently compiled HIAI operator cases."""
from contextlib import ExitStack
from collections.abc import Sequence
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import uuid

from .artifacts import output_path, save_json
from .hashing import sha256_file
from .hdc_run import Hdc, _cleanup_remote, _file_record, _upload, _text


def _basename(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_+.-]+", value) or value in (".", ".."):
        raise ValueError(f"expected a safe basename: {value!r}")
    return value


def _case_files(record):
    name = _basename(record["name"])
    if name in {"calls", "_evidence", "deployment.json", "failure.json", "result.json", "payloads.sha256", ".qwen35-hdc-owner"}:
        raise ValueError(f"case name conflicts with transport metadata: {name}")
    directory = Path(record["directory"])
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError(f"case directory must be an existing absolute directory: {directory}")
    directory = directory.resolve()

    def file(value):
        path = Path(value)
        if not path.is_absolute() or not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError(f"case file must exist within its directory: {path}")
        return path.resolve()

    model, io = file(record["model"]), file(record["io"])
    inputs = [file(path) for path in record["inputs"]]
    if len({path.name for path in inputs}) != len(inputs):
        raise ValueError(f"duplicate input basename for case {name}")
    rows = {"input": [], "output": []}
    for line in io.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4 or parts[0] not in rows or parts[1] not in ("float16", "float32", "int32"):
            raise ValueError(f"invalid IO row in {io}: {line!r}")
        if not re.fullmatch(r"[1-9][0-9]*(?:,[1-9][0-9]*)*", parts[2]):
            raise ValueError(f"IO dimensions must be positive integers: {line!r}")
        rows[parts[0]].append(_basename(parts[3]))
    for direction in rows:
        if not rows[direction] or len(set(rows[direction])) != len(rows[direction]):
            raise ValueError(f"case {name} needs unique nonempty {direction} rows")
    if set(rows["input"]) != {path.name for path in inputs}:
        raise ValueError(f"case {name} input files disagree with io.tsv")
    reserved = {"model.omc", "io.tsv", "run.sh", "stdout", "stderr", "exitcode", "pid", "results", "hilog.txt"}
    if reserved.intersection(rows["input"] + rows["output"]):
        raise ValueError(f"case {name} IO filename conflicts with transport metadata")
    return name, model, io, inputs, rows["output"]


def _receive(hdc, remote, local, *, timeout=None):
    if local.exists():
        raise FileExistsError(local)
    _, text = hdc.shell("sha256sum " + shlex.quote(str(remote)), "hash-result", timeout=timeout)
    lines = text.splitlines()
    if len(lines) != 1 or not re.match(r"^[0-9a-f]{64} [ *]", lines[0]):
        raise RuntimeError(f"invalid remote SHA256 response: {remote}")
    expected = lines[0][:64]
    partial = local.with_name(local.name + ".received")
    local.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        raise FileExistsError(partial)
    hdc.receive(remote, partial, timeout=timeout)
    if sha256_file(partial) != expected:
        raise RuntimeError(f"retrieved SHA256 differs from remote file: {remote}")
    partial.rename(local)
    return {"path": str(local), "remote": str(remote), "sha256": expected, "bytes": local.stat().st_size}


def _integer(path, label, maximum=None):
    text = path.read_text().strip()
    if not re.fullmatch(r"[0-9]+", text) or (maximum is not None and int(text) > maximum):
        raise RuntimeError(f"invalid {label} file: {path}")
    return int(text)


def _library_search_paths(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("library_search_paths must be a sequence of absolute POSIX paths")
    paths = []
    for value in values:
        if (not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_+./-]*", value)
                or value.startswith("//") or ".." in PurePosixPath(value).parts
                or str(PurePosixPath(value)) != value):
            raise ValueError("library search path must be an absolute canonical POSIX path "
                             "without whitespace, shell metacharacters, colon or parent traversal")
        if value in paths:
            raise ValueError("duplicate library search path")
        paths.append(value)
    return tuple(paths)


def _run_script(board, remote_case, *, library_search_paths=()):
    library_search_paths = _library_search_paths(library_search_paths)
    # The inner shell is replaced by exec, so $$ is the native process PID.
    native = 'printf \'%s\\n\' "$$" > pid; exec ' + shlex.join([
        str(board / "bin/op_probe"), "--model", "model.omc", "--io", "io.tsv", "--output", "results"])
    return ("#!/system/bin/sh\ncd " + shlex.quote(str(remote_case)) + " || exit 1\n"
        "export LD_LIBRARY_PATH=" + shlex.quote(":".join((str(board / "lib"), *library_search_paths)))
        + '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        "sh -c " + shlex.quote(native) + " > stdout 2> stderr\n"
        'rc=$?\nprintf \'%s\\n\' "$rc" > exitcode\nexit "$rc"\n')


def _hilog(hdc, board, local, cases, timeout):
    evidence = local / "_evidence"
    evidence.mkdir()
    command = ("hilog -x > " + shlex.quote(str(board / "hilog.txt")) + " 2> "
        + shlex.quote(str(board / "hilog.stderr")) + '\nrc=$?\nprintf \'%s\\n\' "$rc" > '
        + shlex.quote(str(board / "hilog.exitcode")) + '\nexit "$rc"')
    status, _ = hdc.shell(command, "capture-hilog", allow_failure=True, timeout=timeout)
    files = [_receive(hdc, board / name, evidence / name, timeout=timeout)
             for name in ("hilog.txt", "hilog.stderr", "hilog.exitcode")]
    saved = _integer(evidence / "hilog.exitcode", "hilog exitcode", 255)
    if saved != status:
        raise RuntimeError("HiLog remote status and exitcode file disagree")
    if status:
        raise RuntimeError(f"HiLog snapshot unavailable: exit {status}; see {evidence}")
    argument_error = re.compile(r"(?:Mutlti|Multi) commands can't be used in combination\s*\[CODE:\s*-31\]")
    for name in ("hilog.txt", "hilog.stderr"):
        with (evidence / name).open(errors="replace") as stream:
            if any(argument_error.fullmatch(line.strip()) for line in stream):
                raise RuntimeError(f"HiLog rejected its command (CODE -31); see {evidence}")
    pid_line = re.compile(r"^(?:[0-9]{4}-)?[0-9]{2}-[0-9]{2}\s+\S+\s+([0-9]+)\s+[0-9]+\s")
    scope = ("Shared snapshot filtered by native PID only; PID reuse is possible. The ring buffer may "
             "omit earlier cases. Missing log lines do not prove success, failure or device placement.")
    with ExitStack() as stack:
        streams = {}
        for name, case in cases.items():
            path = local / name / "hilog.txt"
            streams[name] = stack.enter_context(path.open("x"))
            case.update(hilog=str(path), hilog_lines=0, hilog_scope=scope)
        with (evidence / "hilog.txt").open(errors="replace") as stream:
            for line in stream:
                match = pid_line.match(line)
                if not match:
                    continue
                for name, case in cases.items():
                    if int(match[1]) == case["pid"]:
                        streams[name].write(line)
                        case["hilog_lines"] += 1
    for case in cases.values():
        path = Path(case["hilog"])
        case["files"].append({"path": str(path), "remote": None, "sha256": sha256_file(path),
                              "bytes": path.stat().st_size, "kind": "pid-filtered-log"})
    return {"status": "CAPTURED", "command": ["hilog", "-x"], "scope": scope,
            "local_evidence_only": files, "deliverable_policy": "Include case PID logs; exclude _evidence and raw HDC calls."}


def run_board(output: Path, runner: Path, libraries: list[Path], records: list[dict],
              hdc_executable="hdc", timeout=120, keep_board=False, *, library_search_paths=()) -> dict:
    """Run independent cases; required transport/cleanup failures raise.

    Native failures are results. Final HiLog failure is a diagnostic warning
    after all required case files are verified, and retains the remote directory.
    Files live in output/board/<name>/{stdout,stderr,exitcode,pid,output_N.bin,hilog.txt}.
    Returned cases map contains exitcode/PID integers, paths, and hash records.
    The full shared snapshot and HDC calls are local evidence, not deliverables.
    Additional library search paths must already exist on the board and are
    inspected read-only; they follow uploaded libraries and precede inherited paths.
    """
    library_search_paths = _library_search_paths(library_search_paths)
    if not records or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("nonempty cases and a positive timeout are required")
    sn, bip = _text(os.environ.get("SN"), "SN"), _text(os.environ.get("BIP"), "BIP")
    host, separator, port = bip.rpartition(":")
    if (sn.startswith("-") or bip.startswith("-") or not host or not separator or any(c.isspace() for c in host)
            or not re.fullmatch(r"[0-9]+", port) or not 1 <= int(port) <= 65535):
        raise ValueError("SN must name a device and BIP must include the remote HDC server port")
    _text(hdc_executable, "HDC executable")
    prepared = [_case_files(record) for record in records]
    if len({row[0] for row in prepared}) != len(prepared):
        raise ValueError("operator case names must be unique")
    runner = Path(runner).resolve(strict=True)
    if not runner.is_file():
        raise ValueError("runner must be a file")
    names = [_basename(Path(path).name) for path in libraries]
    if len(set(names)) != len(names):
        raise ValueError("runtime library basenames must be unique")
    for path in libraries:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    local = output_path(Path(output) / "board")
    local.mkdir(parents=True, exist_ok=False)
    board = PurePosixPath("/data/local/tmp/qwen35-op-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
                         + uuid.uuid4().hex)
    owner = uuid.uuid4().hex
    marker = local / ".qwen35-hdc-owner"
    marker.write_text(owner + "\n")
    uploads = [_file_record(marker, marker.name), _file_record(runner, "bin/op_probe")]
    uploads.extend(_file_record(Path(path).resolve(), "lib/" + name) for path, name in zip(libraries, names))
    for name, model, io, inputs, _ in prepared:
        case = local / name
        case.mkdir()
        remote_case = board / "cases" / name
        script = case / "run.sh"
        script.write_text(_run_script(board, remote_case, library_search_paths=library_search_paths))
        for path, relative in [(model, "model.omc"), (io, "io.tsv"), (script, "run.sh")]:
            uploads.append(_file_record(path, str(PurePosixPath("cases") / name / relative)))
        uploads.extend(_file_record(path, str(PurePosixPath("cases") / name / path.name)) for path in inputs)
    result = {"status": "FAILED", "board_root": str(board), "serial": sn, "hdc_server": bip,
              "library_search_paths": list(library_search_paths),
              "board_cleanup": "NOT_CREATED", "cases": {}, "scope": "HIAI operator diagnostic execution; no placement or numerical verdict."}
    save_json(local / "deployment.json", {**result, "uploads": uploads, "keep_board": keep_board})
    hdc = Hdc(hdc_executable, sn, bip, local, timeout)
    stage = "firmware"
    try:
        firmware_status, firmware = hdc.shell("param get const.ohos.fullname", "firmware", allow_failure=True)
        result["firmware"] = {"value": firmware.strip(), "returncode": firmware_status}
        if library_search_paths:
            stage = "library-search-paths"
            for index, path in enumerate(library_search_paths, 1):
                quoted = shlex.quote(path)
                hdc.shell(f"test -d {quoted} && test -r {quoted} && test -x {quoted}",
                          f"library-search-path-{index}")
        stage = "create-root"
        result["board_cleanup"] = "UNVERIFIED_CREATION_RETAINED"
        hdc.shell("mkdir " + shlex.quote(str(board)), "create-root")
        result["board_cleanup"] = "RETAINED"
        stage = "upload"
        directories = sorted({str((board / entry["remote"]).parent) for entry in uploads})
        hdc.shell("mkdir -p " + " ".join(shlex.quote(path) for path in directories), "create-directories")
        _upload(hdc, board, uploads, local / "payloads.sha256")
        hdc.shell("chmod 755 " + shlex.quote(str(board / "bin/op_probe")), "runner-permissions")
        for index, (name, _, _, _, outputs) in enumerate(prepared, 1):
            print(f"op-probe: [{index}/{len(prepared)}] {name}: running", file=sys.stderr, flush=True)
            stage = "case:" + name
            remote_case, case = board / "cases" / name, local / name
            status, _ = hdc.shell("sh " + shlex.quote(str(remote_case / "run.sh")), "case-" + name, allow_failure=True)
            files = [_receive(hdc, remote_case / file, case / file) for file in ("stdout", "stderr", "exitcode", "pid")]
            saved = _integer(case / "exitcode", "native exitcode", 255)
            pid = _integer(case / "pid", "native PID")
            if status != saved or pid <= 0:
                raise RuntimeError(f"case {name} has inconsistent native status or invalid PID")
            case_result = {"status": "EXECUTED" if status == 0 else "NATIVE_FAILED", "exitcode": saved,
                "pid": pid, "directory": str(case), "stdout": str(case / "stdout"), "stderr": str(case / "stderr"),
                "files": files, "output_files": []}
            result["cases"][name] = case_result
            if status == 0:
                for file in outputs:
                    downloaded = _receive(hdc, remote_case / "results" / file, case / file)
                    case_result["files"].append(downloaded)
                    case_result["output_files"].append(downloaded)
            print(f"op-probe: [{index}/{len(prepared)}] {name}: native exit {saved}; "
                  f"logs: {case}", file=sys.stderr, flush=True)
        stage = "hilog"
        try:
            result["hilog"] = _hilog(hdc, board, local, result["cases"], min(timeout, 30))
        except Exception as error:
            # Every required case file has already been retrieved and verified.
            # Retain incomplete diagnostic evidence for inspection, without
            # replacing it or changing the execution/numerical evidence.
            result["hilog"] = {"status": "INCOMPLETE", "command": ["hilog", "-x"],
                               "error": f"{type(error).__name__}: {error}",
                               "scope": "Final optional HiLog capture/retrieval failed; remote evidence retained.",
                               "deliverable_policy": "Include case PID logs; exclude _evidence and raw HDC calls."}
            print(f"op-probe: diagnostic warning: {result['hilog']['error']}; "
                  f"remote evidence retained: {board}", file=sys.stderr, flush=True)
        if not keep_board and result["hilog"]["status"] == "CAPTURED":
            stage = "cleanup"
            _cleanup_remote(hdc, board, owner)
            result["board_cleanup"] = "REMOVED"
        result.update(status="TRANSPORT_COMPLETE", stage="complete",
                      native_failures=sum(case["exitcode"] != 0 for case in result["cases"].values()))
        save_json(local / "result.json", result)
        return result
    except Exception as error:
        result.update(stage=stage, error=f"{type(error).__name__}: {error}")
        if stage == "cleanup":
            result["board_cleanup"] = "FAILED"
        save_json(local / "failure.json", result)
        raise

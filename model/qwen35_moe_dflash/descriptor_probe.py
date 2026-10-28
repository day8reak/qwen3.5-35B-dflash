"""Observe selected HIAI descriptor setters without loading or executing a model."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import uuid

from .artifacts import checked_file, save_json
from .draft_board import _build_runner, _file, _receive
from .draft_workspace import DraftWorkspace
from .elf_dynamic import dynamic_info
from .hashing import sha256_file
from .hdc_run import Hdc, _cleanup_remote, _file_record, _text, _upload
from .hiai_deployment import (
    BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers, discover_libraries,
)

MAX_MODEL_BYTES = 64 << 20
OWNER = "qwen35-hiai-descriptor-diagnostic"


def select_model(job):
    """Hash metadata and the small external main OMC, never the weight sidecars."""
    job = Path(job)
    if job.is_symlink() or job.resolve() != job.absolute():
        raise ValueError("draft job must not traverse symlinks")
    report = json.loads((job / "result.json").read_text())
    if report.get("artifact_kind") != "qwen35-full-draft-job":
        raise ValueError("expected a full-draft job")
    record = report.get("components", {}).get("compilation")
    receipt = job / "completed-compilation.json"
    if receipt.is_file():
        completed = json.loads(receipt.read_text())
        if record is not None and completed != record:
            raise ValueError("compilation receipt differs from latest job result")
        record = completed
    if not record:
        raise ValueError("diagnostic needs completed external OMC compilation")
    folder = Path(record["directory"])
    manifest_path = checked_file(folder, record["manifest"])
    compiled = json.loads(manifest_path.read_text())
    if (compiled.get("artifact_kind") != "qwen35-existing-air-compilation"
            or compiled.get("status") != "COMPILED"
            or compiled.get("model_compilation") != "PASSED"
            or compiled.get("requested_options", {}).get("save_weights_as_external_data") != "true"):
        raise ValueError("diagnostic requires successful external OMC compilation")
    model = compiled["omc"]
    if type(model.get("bytes")) is not int or not 0 < model["bytes"] <= MAX_MODEL_BYTES:
        raise ValueError("descriptor diagnostic main OMC must be at most 64 MiB; do not use embedded weights")
    path = _file(folder, model)
    return path, {"job": str(job), "compilation_manifest": record["manifest"],
                  "model_sha256": model["sha256"], "model_bytes": model["bytes"],
                  "ddk_id": compiled["ddk_id"]}


def _reset_output(out):
    if out.is_symlink():
        raise ValueError("diagnostic output must not be a symlink")
    if out.exists():
        marker = out / ".owner.json"
        if (not out.is_dir() or not marker.is_file() or marker.is_symlink()
                or json.loads(marker.read_text()) != {"artifact_kind": OWNER}):
            raise ValueError("refusing to replace an unrecognized load-diagnostic directory")
        shutil.rmtree(out)
    out.mkdir()
    save_json(out / ".owner.json", {"artifact_kind": OWNER})


def _validate_native(native, identity, remote_model):
    if (native.get("schema_version") != 1
            or native.get("artifact_kind") != "qwen35-hiai-model-description-probe"
            or native.get("status") != "INSPECTED"
            or native.get("evidence_kind") != "device-hiai"
            or native.get("load_called") is not False or native.get("process_called") is not False
            or native.get("model_sha256") != identity["model_sha256"]
            or native.get("model_bytes") != identity["model_bytes"]
            or native.get("model_path") != str(remote_model)):
        raise ValueError("native descriptor probe identity or no-execution contract differs")
    if type(native.get("process_id")) is not int or not 0 < native["process_id"] <= 2**31 - 1:
        raise ValueError("native descriptor probe requires its process ID")
    _text(native.get("runtime_version"), "native runtime version")
    if os.environ.get("HIAI_RUNTIME_VERSION") and native["runtime_version"] != os.environ["HIAI_RUNTIME_VERSION"]:
        raise ValueError("descriptor probe runtime differs from HIAI_RUNTIME_VERSION")
    orders = native.get("observations", [])
    expected = (("path-then-buffer", ("SetModelPath", "SetModelBuffer")),
                ("buffer-then-path", ("SetModelBuffer", "SetModelPath")))
    if len(orders) != 2:
        raise ValueError("descriptor probe must observe both setter orders")
    for row, (order, operations) in zip(orders, expected):
        if row.get("order") != order or [x.get("operation") for x in row.get("steps", [])] != list(operations):
            raise ValueError("descriptor probe setter sequence differs")
        for step in row["steps"]:
            if (type(step.get("setter_status")) is not int or type(step.get("model_bytes")) is not int
                    or step["model_bytes"] < 0 or not isinstance(step.get("model_path"), str)
                    or any(type(step.get(key)) is not bool for key in ("buffer_present", "buffer_matches_input"))):
                raise ValueError("descriptor probe has incomplete setter observations")


def run_probe(job, *, timeout_seconds=300):
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout must be positive and finite")
    job = Path(job).absolute()
    out = job / "load-diagnostic"
    with DraftWorkspace(job):
        model, identity = select_model(job)
        if os.environ.get("CPRO_DDK_ID") and os.environ["CPRO_DDK_ID"] != identity["ddk_id"]:
            raise ValueError("CPRO_DDK_ID differs from the compiled draft")
        sn, bip = _text(os.environ.get("SN"), "SN"), _text(os.environ.get("BIP"), "BIP")
        host, separator, port = bip.rpartition(":")
        if (sn.startswith("-") or bip.startswith("-") or not separator or not host
                or any(c.isspace() for c in host) or not port.isdigit() or not 1 <= int(port) <= 65535):
            raise ValueError("SN must be a device ID; BIP must be an HDC server host:port")
        _reset_output(out)
        result = {"schema_version": 1, "artifact_kind": OWNER, "status": "FAILED",
                  "stage": "preflight", "source": identity, "output": str(out),
                  "model_loading": "NOT_RUN", "model_execution": "NOT_RUN",
                  "retrieved_sha256": {}, "retrieval_errors": [],
                  "scope": "Descriptor setter observations only; not external-weight loading or numerical validation."}
        hdc = Hdc(_text(os.environ.get("HDC", "hdc"), "HDC"), sn, bip, out, timeout_seconds)
        board = PurePosixPath("/data/local/tmp/qwen35-descriptor-" + uuid.uuid4().hex)
        owner_token = uuid.uuid4().hex
        result.update(board_root=str(board), serial=sn, hdc_server=bip, board_cleanup="NOT_CREATED")
        attempted = False
        try:
            paths = list(BOARD_LIBRARY_SEARCH_PATHS)
            private = os.environ.get("BOARD_HIAI_LIB_DIR")
            if private:
                _text(private, "BOARD_HIAI_LIB_DIR")
                if not private.startswith("/") or ":" in private or ".." in PurePosixPath(private).parts:
                    raise ValueError("BOARD_HIAI_LIB_DIR must be an absolute board path")
                paths.insert(0, private)
            print("draft-load-probe: checking HDC and building the descriptor probe", flush=True)
            hdc.shell("command -v sha256sum >/dev/null && " + " && ".join(
                "test -d " + shlex.quote(p) + " && test -r " + shlex.quote(p) for p in paths), "preflight")
            result["stage"] = "build-runner"
            runner = _build_runner(out, timeout_seconds)
            uploads = [_file_record(model, "model.omc"), _file_record(runner, "runner")]
            if private:
                result["libraries"] = {"mode": "existing-board-directory", "path": private,
                                       "compatibility": "UNVERIFIED"}
            else:
                discovery = discover_libraries(runner, out / "native")
                libraries = [_file_record(p.resolve(), "lib/" + p.name) for p in discovery["libraries"]]
                dependencies = [{"name": PurePosixPath(row["remote"]).name,
                                 "kind": "runner" if row["remote"] == "runner" else "library",
                                 "sha256": row["sha256"], **dynamic_info(row["local"])}
                                for row in [uploads[-1], *libraries]]
                blockers = candidate_blockers(dependencies, require_pair=False)
                if blockers:
                    raise ValueError("cannot select deployment: " + "; ".join(blockers))
                uploads.extend(row for row in libraries if PurePosixPath(row["remote"]).name not in DDK_HCL_EXCLUDED)
                paths.insert(0, str(board / "lib"))
                result["libraries"] = {"mode": "candidate_without_ddk_hcl",
                                       "discovery": discovery["metadata"], "dependencies": dependencies}
            result["stage"] = "upload"
            save_json(out / "upload-plan.json", {"board_root": str(board), "files": uploads})
            hdc.shell("mkdir " + shlex.quote(str(board)), "create-root")
            result["board_cleanup"] = "RETAINED"
            hdc.shell("printf '%s\\n' " + shlex.quote(owner_token) + " > " +
                      shlex.quote(str(board / ".qwen35-hdc-owner")), "mark-owner")
            hdc.shell("mkdir -p " + shlex.quote(str(board / "lib")), "create-subdirs")
            _upload(hdc, board, uploads, out / "payloads.sha256", checksum_connection_retries=2)
            hdc.shell("chmod 755 " + shlex.quote(str(board / "runner")), "runner-permissions")
            environment = "LD_LIBRARY_PATH=" + shlex.quote(":".join(paths)) + " "
            command = environment + shlex.join([str(board / "runner"), "--probe-model-description", str(board / "model.omc")])
            result["stage"] = "observe-descriptor"
            print("draft-load-probe: observing both setter orders; no model Load or Process", flush=True)
            attempted = True
            code, _ = hdc.shell("cd " + shlex.quote(str(board)) + " && " + command +
                                " > probe.json 2> probe.stderr; rc=$?; printf '%s\\n' \"$rc\" > " +
                                shlex.quote(str(board / "probe.exitcode")) + "; exit \"$rc\"",
                                "observe-descriptor", allow_failure=True)
            result["remote_exitcode"] = code
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        if attempted:
            for name in ("probe.json", "probe.stderr", "probe.exitcode"):
                try:
                    digest = _receive(hdc, board, out, name)
                    if not digest:
                        raise ValueError("remote diagnostic file missing")
                    result["retrieved_sha256"][name] = digest
                except Exception as error:
                    result["retrieval_errors"].append({"path": name, "error": str(error)})
            try:
                if result.get("error") or result["retrieval_errors"] or result.get("remote_exitcode") != 0:
                    raise ValueError("descriptor command/transport incomplete; see retained logs")
                if (out / "probe.exitcode").read_text().strip() != "0":
                    raise ValueError("native descriptor probe returned nonzero")
                native = json.loads((out / "probe.json").read_text())
                _validate_native(native, identity, board / "model.omc")
                if sha256_file(model) != identity["model_sha256"]:
                    raise ValueError("main OMC changed during diagnostic")
                result.update(status="INSPECTED", stage="complete", observations=native["observations"],
                              runtime_version=native["runtime_version"], native_process_id=native.get("process_id"))
            except Exception as error:
                result["validation_error"] = str(error)
        # Only remove this invocation's newly created remote diagnostic after
        # successful evidence retrieval; never touch the deployed full draft.
        if result["status"] == "INSPECTED":
            try:
                _cleanup_remote(hdc, board, owner_token)
                result["board_cleanup"] = "REMOVED"
            except Exception as error:
                result["cleanup_error"] = str(error)
        save_json(out / "result.json", result)
        return result

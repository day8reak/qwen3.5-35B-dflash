"""Execute one complete draft using bound real target features, without a golden.

The existing full draft model is streamed from its original host path into a
SHA-owned phone cache. Only small derived inputs are new host artifacts. This
diagnostic does not establish target state feedback, accuracy or speedup.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import uuid

import numpy as np

from .artifacts import output_path, save_json
from .draft_board import _build_runner, _file, _hidden_summary, _receive
from .elf_dynamic import dynamic_info
from .hashing import sha256_file
from .hdc_run import Hdc, _file_record, _relative, _text, _upload
from .hiai_deployment import (BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED,
                              candidate_blockers, discover_libraries)


def _json(path):
    return json.loads(Path(path).read_text())


def _observe_native(manifest, native, root, *, runtime, firmware, pid):
    """Check the actual invocation and bytes; never compare synthetic goldens."""
    identity = {"schema_version": 1, "scope": "standalone-draft-step",
        "model_sha256": manifest["model"]["sha256"],
        "draft_abi_sha256": manifest["draft_abi"]["sha256"],
        "step_plan_sha256": manifest["step_plan"]["sha256"],
        "ddk_id": manifest["ddk_id"], "firmware_id": firmware,
        "model_load_api": "legacy-buffer", "target_calls": 0, "process_id": pid,
        "evidence_kind": "device-hiai"}
    if (any(native.get(key) != value for key, value in identity.items())
            or any(type(native.get(key)) is not int for key in
                   ("schema_version", "target_calls", "draft_calls", "process_id"))):
        raise ValueError("native result does not bind this real-feature draft invocation")
    runtime_key = "runtime_version" if native.get("status") == "EXECUTED" else "expected_runtime_version"
    if native.get(runtime_key) != runtime:
        raise ValueError("native runtime differs from this invocation")
    if native.get("status") == "RUNTIME_FAILED":
        if (native.get("draft_calls") != 0 or native.get("outputs") != []
                or native.get("failure_stage") not in {"LOAD_OR_SETUP", "INPUT_BINDING", "PROCESS"}
                or not isinstance(native.get("reason"), str) or not native["reason"].strip()):
            raise ValueError("invalid native runtime failure record")
        return {"native_status": "RUNTIME_FAILED", "device_execution": "FAILED",
                "failure_stage": native["failure_stage"], "error": native["reason"],
                "runtime_stage": native.get("runtime_stage")}
    if (native.get("status") != "EXECUTED" or native.get("draft_calls") != 1
            or native.get("sequence_length") != manifest["sequence_length"]
            or native.get("state_policy") != "prefix-recompute"
            or len(native.get("inputs", [])) != 3 or len(native.get("outputs", [])) != 2):
        raise ValueError("native result must describe exactly one complete draft call")
    for row, expected in zip(native["inputs"], manifest["inputs"]):
        if any(row.get(key) != expected[key] for key in ("index", "name", "dtype", "shape", "bytes", "sha256")):
            raise ValueError("native input differs from prepared real target bytes")
    diagnostic = native.get("io_diagnostic", {})
    if (diagnostic.get("enabled") is not True or diagnostic.get("process_called") is not True
            or type(diagnostic.get("output_fill_byte")) is not int or diagnostic["output_fill_byte"] != 0
            or diagnostic.get("input_readback_point") != "AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS"
            or len(diagnostic.get("inputs", [])) != 3 or len(diagnostic.get("outputs", [])) != 2):
        raise ValueError("native buffer readback evidence is incomplete")
    for row, expected in zip(diagnostic["inputs"], manifest["inputs"]):
        if (row.get("index") != expected["index"] or row.get("readback_matches_source") is not True
                or row.get("readback_sha256") != expected["sha256"]):
            raise ValueError("native input readback differs from real-feature sources")
    for index, row in enumerate(diagnostic["outputs"]):
        if (row.get("index") != index or row.get("checked_after_process") is not True
                or row.get("object_same") is not True or row.get("buffer_same") is not True):
            raise ValueError("native output buffer identity changed through Process")
    values, outputs = [], []
    for row, expected in zip(native["outputs"], manifest["outputs"]):
        if (any(row.get(key) != expected[key] for key in ("index", "name", "dtype", "shape"))
                or row.get("path") != f"output_{expected['index']:04d}.bin"):
            raise ValueError("native draft output ABI differs from the prepared step")
        dtype = np.dtype("<i4" if expected["dtype"] == "int32" else "<f2")
        if row.get("bytes") != math.prod(expected["shape"]) * dtype.itemsize:
            raise ValueError("native output extent differs from its ABI")
        path = _file(root, row)
        values.append(np.fromfile(path, dtype=dtype).reshape(expected["shape"]))
        outputs.append({key: row[key] for key in ("index", "name", "dtype", "shape", "path", "bytes", "sha256")})
    top1 = values[0]
    vocab = manifest["draft_config"]["vocab_size"]
    if np.any(top1 < 0) or np.any(top1 >= vocab):
        raise ValueError("draft output token IDs are outside its vocabulary")
    hidden = _hidden_summary(values[1].astype(np.float64))
    return {"native_status": "EXECUTED", "runtime_stage": native.get("runtime_stage"),
        "device_execution": "EXECUTED_OUTPUTS_RETRIEVED", "host_input_readback": "MATCHED_REAL_SOURCE_BYTES",
        "outputs": outputs, "draft_top1": top1.tolist(), "token_id_range": "WITHIN_DECLARED_VOCABULARY",
        "hidden_statistics": hidden,
        "output_numerics": "NONFINITE_OBSERVED" if hidden["nonfinite_count"] else "ALL_FINITE_UNVALIDATED",
        "synthetic_golden_comparison": "NOT_RUN", "numerical_validation": "NOT_RUN_NO_REFERENCE"}


def _cache_model(hdc, cache, source, out, *, extra_bytes=0):
    """Reuse only a regular SHA-matching cached model, with no host snapshot."""
    model = cache / "model/model.omc"
    quoted = shlex.quote(str(model))
    hdc.shell("mkdir -p " + shlex.quote(str(model.parent))
              + " && test ! -L " + shlex.quote(str(model.parent)), "model-cache-directories")
    _, decision = hdc.shell("if test -L " + quoted + "; then printf 'UNSAFE\\n'; elif test -f "
        + quoted + " && test \"$(wc -c < " + quoted + ")\" -eq " + str(source["bytes"])
        + " && test \"$(sha256sum " + quoted + " | cut -d ' ' -f 1)\" = "
        + shlex.quote(source["sha256"]) + "; then printf 'MATCHED\\n'; elif test -e "
        + quoted + "; then printf 'MISMATCH\\n'; else printf 'MISSING\\n'; fi", "inspect-draft-model-cache")
    if decision not in {"MATCHED", "MISSING"}:
        raise ValueError("draft model cache is unsafe or contains different bytes; it was retained")
    from .target_model_io_probe import _space
    _, disk = hdc.shell("df -Pk " + shlex.quote(str(cache)), "draft-board-disk", timeout=min(60, hdc.timeout))
    free = _space(disk)
    missing = source["bytes"] if decision == "MISSING" else 0
    required = missing + extra_bytes + (64 << 20)
    if free < required:
        raise ValueError(f"draft board disk insufficient: need {required}, available {free}")
    print(f"real-draft: cached model {decision.lower()}; missing upload={missing} bytes", flush=True)
    if decision == "MISSING":
        # An interrupted transfer remains .partial; it can never be accepted
        # as the final model or overwritten by this attempt automatically.
        partial = model.with_name("model.omc.partial")
        hdc.shell("test ! -e " + shlex.quote(str(partial)) + " && test ! -L "
                  + shlex.quote(str(partial)), "check-draft-partial-absent")
        hdc.send(Path(source["path"]), partial)
        hdc.shell("test -f " + shlex.quote(str(partial)) + " && test ! -L " + shlex.quote(str(partial))
            + " && test \"$(wc -c < " + shlex.quote(str(partial)) + ")\" -eq " + str(source["bytes"])
            + " && test \"$(sha256sum " + shlex.quote(str(partial)) + " | cut -d ' ' -f 1)\" = "
            + shlex.quote(source["sha256"]) + " && test ! -e " + quoted + " && mv "
            + shlex.quote(str(partial)) + " " + quoted, "verify-draft-model-upload")
    return model, {"cache_status": decision, "missing_model_upload_bytes": missing,
                   "board_free_bytes": free, "required_bytes": required,
                   "host_model_copy_bytes": 0, "model_source": source["path"]}


def execute_real_feature_draft(prepared_dir, output_dir, *, timeout_seconds=1800, lock_wait_seconds=1800):
    """Build the current runner and execute one real-feature full-draft step."""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds < .001
            or type(lock_wait_seconds) is not int or lock_wait_seconds < 0):
        raise ValueError("execution timeout must be finite/positive and lock wait a nonnegative integer")
    from .real_feature_draft import validate_real_feature_draft
    root, out = Path(prepared_dir).resolve(), output_path(output_dir)
    if root == out or root in out.parents:
        raise ValueError("execution output must be outside the immutable prepared step")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-real-feature-draft-execution",
        "status": "FAILED", "stage": "preflight", "device_execution": "NOT_RUN",
        "numerical_validation": "NOT_RUN_NO_REFERENCE", "state_continuity": "NOT_VALIDATED",
        "model_compilation": "NOT_RUN", "runtime_placement": "UNVERIFIED",
        "model_load_api": "legacy-buffer", "board_cleanup": "OWNED_CACHE_AND_ATTEMPT_RETAINED",
        "prepared": str(root), "output": str(out), "retrieved_sha256": {}, "retrieval_errors": [],
        "synthetic_golden_comparison": "NOT_RUN", "host_model_copy_bytes": 0}
    hdc = board = manifest = cache = None
    attempted = global_held = cache_held = uncertain = exit_known = False
    global_lock = PurePosixPath("/data/local/tmp/.qwen35-target-model-io-lock")
    token = uuid.uuid4().hex

    def collect(names):
        for name in names:
            if name in result["retrieved_sha256"]:
                continue
            try:
                digest = _receive(hdc, board, out, name)
                if digest:
                    result["retrieved_sha256"][name] = digest
            except Exception as error:
                result["retrieval_errors"].append({"path": name, "error": f"{type(error).__name__}: {error}"})

    try:
        manifest = validate_real_feature_draft(root)
        result["prepared_manifest_sha256"] = sha256_file(root / "manifest.json")
        result.update(sequence_length=manifest["sequence_length"],
                      execution_scope="ONE_FULL_DRAFT_STEP_FROM_BOUND_TARGET_PREFILL",
                      token_provenance=manifest.get("token_provenance", {}),
                      feature_provenance=manifest.get("feature_provenance", {}))
        ddk, runtime, firmware = (manifest[key] for key in ("ddk_id", "runtime_version", "firmware_id"))
        if os.environ.get("CPRO_DDK_ID") and os.environ["CPRO_DDK_ID"] != ddk:
            raise ValueError("CPRO_DDK_ID differs from prepared complete draft")
        for key, expected in (("HIAI_RUNTIME_VERSION", runtime), ("FIRMWARE_ID", firmware)):
            if os.environ.get(key) and os.environ[key].strip() != expected.strip():
                raise ValueError(key + " differs from bound target execution")
        sn, bip = _text(os.environ.get("SN"), "SN"), _text(os.environ.get("BIP"), "BIP")
        host, separator, port = bip.rpartition(":")
        if (sn.startswith("-") or bip.startswith("-") or not separator or not host
                or any(c.isspace() for c in host) or not port.isdigit() or not 1 <= int(port) <= 65535):
            raise ValueError("SN must name a device and BIP must be an HDC server host:port")
        hdc = Hdc(_text(os.environ.get("HDC", "hdc"), "HDC"), sn, bip, out, timeout_seconds)
        cache = PurePosixPath("/data/local/tmp/qwen35_real_draft_" + manifest["model"]["sha256"])
        board = cache / ("run_" + uuid.uuid4().hex)
        paths = list(BOARD_LIBRARY_SEARCH_PATHS)
        private = os.environ.get("BOARD_HIAI_LIB_DIR")
        if private:
            _text(private, "BOARD_HIAI_LIB_DIR")
            if not private.startswith("/") or ":" in private or ".." in PurePosixPath(private).parts:
                raise ValueError("BOARD_HIAI_LIB_DIR must be an absolute board path")
            paths.insert(0, private)
        hdc.shell("command -v sha256sum >/dev/null && " + " && ".join(
            "test -d " + shlex.quote(path) + " && test -r " + shlex.quote(path) for path in paths), "preflight")
        _, actual_firmware = hdc.shell("param get const.ohos.fullname", "firmware")
        if actual_firmware.strip() != firmware.strip():
            raise ValueError("actual firmware differs from the source target execution")
        result.update(serial=sn, hdc_server=bip, ddk_id=ddk, runtime_version=runtime, firmware_id=firmware,
                      board_model_cache=str(cache), board_attempt=str(board), board_global_lock=str(global_lock))
        result["stage"] = "build-runner"
        print("real-draft: building current runner; existing full draft OMC is reused", flush=True)
        runner = _build_runner(out, timeout_seconds)
        uploads = [_file_record(runner, "runner")]
        if private:
            result["libraries"] = {"mode": "existing-board-directory", "path": private, "compatibility": "UNVERIFIED"}
        else:
            discovery = discover_libraries(runner, out / "native")
            libraries = [_file_record(path.resolve(), "lib/" + str(_relative(path.name))) for path in discovery["libraries"]]
            dependencies = [{"name": PurePosixPath(row["remote"]).name,
                "kind": "runner" if row["remote"] == "runner" else "library",
                "sha256": row["sha256"], **dynamic_info(row["local"])} for row in [*uploads, *libraries]]
            blockers = candidate_blockers(dependencies, require_pair=False)
            if blockers:
                raise ValueError("cannot select board libraries: " + "; ".join(blockers))
            uploads.extend(row for row in libraries if PurePosixPath(row["remote"]).name not in DDK_HCL_EXCLUDED)
            paths.insert(0, str(board / "lib"))
            result["libraries"] = {"mode": "candidate_without_ddk_hcl", "discovery": discovery["metadata"],
                "dependencies": dependencies, "backend_selection": "UNVERIFIED"}
        uploads.extend({"local": row["local_path"], "remote": "bundle/" + str(_relative(row["path"])),
                        "bytes": row["bytes"], "sha256": row["sha256"]} for row in manifest["upload_payloads"])
        validate_real_feature_draft(root)
        from .target_model_io_probe import _acquire_global, _release_global
        result["stage"] = "wait-board-model-lock"
        uncertain = True
        _acquire_global(hdc, global_lock, token, lock_wait_seconds)
        global_held, uncertain = True, False
        qcache, owner = shlex.quote(str(cache)), shlex.quote(str(cache / ".model-sha256"))
        result["stage"] = "model-cache"
        uncertain = True
        hdc.shell("if mkdir " + qcache + "; then printf '%s\\n' " + shlex.quote(manifest["model"]["sha256"])
            + " > " + owner + "; else test -d " + qcache + " && test ! -L " + qcache
            + " && test ! -L " + owner + " && read -r draft_cache_owner < " + owner
            + " && test \"$draft_cache_owner\" = " + shlex.quote(manifest["model"]["sha256"]) + "; fi", "own-draft-model-cache")
        uncertain = False
        cache_lock = cache / ".execution-lock"
        uncertain = True
        _acquire_global(hdc, cache_lock, token, 0)
        cache_held, uncertain = True, False
        uncertain = True
        try:
            model, budget = _cache_model(hdc, cache, manifest["model_source"], out,
                                       extra_bytes=sum(row["bytes"] for row in uploads))
        except ValueError:
            # A parsed cache/disk rejection is known to precede inference.
            # Transport exceptions instead retain the uncertain lock state.
            uncertain = False
            raise
        result["model_cache"] = budget
        hdc.shell("mkdir " + shlex.quote(str(board)), "create-real-draft-attempt")
        hdc.shell("mkdir -p " + " ".join(shlex.quote(path) for path in sorted({
            str((board / row["remote"]).parent) for row in uploads} | {str(board / "bundle/model")})), "create-subdirs")
        linked = board / "bundle/model/model.omc"
        hdc.shell("test ! -L " + shlex.quote(str(model)) + " && ln " + shlex.quote(str(model)) + " "
            + shlex.quote(str(linked)) + " && test ! -L " + shlex.quote(str(linked))
            + " && test " + shlex.quote(str(model)) + " -ef " + shlex.quote(str(linked))
            + " && test \"$(sha256sum " + shlex.quote(str(linked)) + " | cut -d ' ' -f 1)\" = "
            + shlex.quote(manifest["model"]["sha256"]), "link-verified-cached-model")
        result["remote_model_storage"] = {"mode": "OWNED_REGULAR_FILE_HARDLINK",
            "cache": str(model), "attempt": str(linked), "host_copy_or_link": "NONE"}
        save_json(out / "upload-plan.json", {"model_source": manifest["model_source"],
                  "board_cache_model": str(model), "board_attempt": str(board), "files": uploads})
        _upload(hdc, board, uploads, out / "payloads.sha256")
        hdc.shell("chmod 755 " + shlex.quote(str(board / "runner")), "runner-permissions")
        uncertain = False
        environment = "LD_LIBRARY_PATH=" + shlex.quote(":".join(paths)) + " "
        result["stage"] = "runtime-probe"
        uncertain = True
        try:
            probe_status, _ = hdc.shell("cd " + shlex.quote(str(board)) + " && " + environment
                + "./runner --probe > runtime.json 2> runtime.stderr; rc=$?; printf '%s\\n' \"$rc\" > "
                + shlex.quote(str(board / "runtime.exitcode")) + "; exit \"$rc\"", "runtime-probe", allow_failure=True)
            uncertain = False
        finally:
            collect(("runtime.json", "runtime.stderr", "runtime.exitcode"))
        if any(name not in result["retrieved_sha256"] for name in ("runtime.json", "runtime.stderr", "runtime.exitcode")):
            raise ValueError("runtime probe evidence is incomplete")
        probe = _json(out / "runtime.json")
        if (probe_status != 0 or (out / "runtime.exitcode").read_text().strip() != "0"
                or probe.get("npu_supported") is not True or probe.get("runtime_version") != runtime):
            raise ValueError("actual runtime capability/version differs from the source target")
        validate_real_feature_draft(root)
        native_timeout = min(int(timeout_seconds * 1000), 2**32 - 1)
        argv = [str(board / "runner"), "--draft-step", str(board / "bundle/step.tsv"),
            "--output", str(board / "results"), "--expected-runtime-version", runtime,
            "--ddk-id", ddk, "--firmware-id", firmware, "--timeout-ms", str(native_timeout),
            "--model-load-api", "legacy-buffer"]
        save_json(out / "command.json", {"argv": argv, "library_paths": paths, "native_timeout_ms": native_timeout})
        command = ("cd " + shlex.quote(str(board)) + " && { " + environment + shlex.join(argv)
            + " > draft.stdout 2> draft.stderr & draft_pid=$!; printf '%s\\n' \"$draft_pid\" > draft.pid; "
            + "wait \"$draft_pid\"; rc=$?; printf '%s\\n' \"$rc\" > draft.exitcode; exit \"$rc\"; }")
        result.update(stage="execute", device_execution="ATTEMPTED", native_timeout_ms=native_timeout)
        attempted, uncertain = True, True
        print("real-draft: full six-layer draft, exact real target features, one native invocation", flush=True)
        status, _ = hdc.shell(command, "execute-real-feature-draft", allow_failure=True)
        result["remote_exitcode"] = status
        exit_known, uncertain = True, False
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if attempted:
            collect(("draft.stdout", "draft.stderr", "draft.exitcode", "draft.pid", "results/result.json"))
            try:
                required = ("draft.stdout", "draft.stderr", "draft.exitcode", "draft.pid", "results/result.json")
                if any(name not in result["retrieved_sha256"] for name in required):
                    raise ValueError("hash-verified native execution evidence is incomplete")
                pid_text = (out / "draft.pid").read_text().strip()
                exit_text = (out / "draft.exitcode").read_text().strip()
                if not re.fullmatch(r"[1-9][0-9]{0,9}", pid_text) or int(pid_text) > 2**31 - 1:
                    raise ValueError("invalid native process PID")
                result["native_process_id"] = int(pid_text)
                if (not re.fullmatch(r"0|[1-9][0-9]{0,2}", exit_text) or int(exit_text) > 255
                        or type(result.get("remote_exitcode")) is not int
                        or int(exit_text) != result["remote_exitcode"]):
                    raise ValueError("retained native exit code differs from verified remote completion")
                native = _json(out / "results/result.json")
                if native.get("status") == "EXECUTED":
                    collect(("results/output_0000.bin", "results/output_0001.bin"))
                observed = _observe_native(manifest, native, out / "results", runtime=result["runtime_version"],
                                           firmware=result["firmware_id"], pid=int(pid_text))
                if result["retrieval_errors"]:
                    raise ValueError("native evidence retrieval reported failures")
                result.update(observed)
                if observed["native_status"] == "EXECUTED":
                    if result["remote_exitcode"] != 0:
                        raise ValueError("native EXECUTED conflicts with nonzero process exit")
                    result.update(status="REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED", stage="complete")
                elif result["remote_exitcode"] == 0:
                    raise ValueError("native failure conflicts with zero process exit")
            except Exception as error:
                result.update(status="FAILED", device_execution="FAILED",
                              evidence_error=f"{type(error).__name__}: {error}")
            if result["status"] == "FAILED":
                try:
                    from .draft_hilog import capture_draft_hilog
                    result["hilog"] = capture_draft_hilog(hdc, board, out, receive=_receive,
                                                          recorded_pid=result.get("native_process_id"))
                except Exception as error:
                    result["hilog"] = {"status": "UNAVAILABLE", "error": f"{type(error).__name__}: {error}"}
        if manifest is not None:
            try:
                validate_real_feature_draft(root)
                if sha256_file(root / "manifest.json") != result["prepared_manifest_sha256"]:
                    raise ValueError("prepared manifest changed during execution")
                result["source_preservation"] = "UNCHANGED_BOUND_SOURCES_AND_PREPARED_INPUT_BYTES"
            except Exception as error:
                result.update(status="FAILED", source_preservation="FAILED", source_error=f"{type(error).__name__}: {error}")
        if result.get("remote_model_storage") and not uncertain and (not attempted or exit_known):
            try:
                model = result["remote_model_storage"]["cache"]
                linked = result["remote_model_storage"]["attempt"]
                hdc.shell("test -f " + shlex.quote(model) + " && test ! -L " + shlex.quote(model)
                    + " && test -f " + shlex.quote(linked) + " && test ! -L " + shlex.quote(linked)
                    + " && test " + shlex.quote(model) + " -ef " + shlex.quote(linked)
                    + " && test \"$(wc -c < " + shlex.quote(linked) + ")\" -eq " + str(manifest["model"]["bytes"])
                    + " && test \"$(sha256sum " + shlex.quote(linked) + " | cut -d ' ' -f 1)\" = "
                    + shlex.quote(manifest["model"]["sha256"]), "verify-cached-model-after-execution")
                result["remote_model_preservation"] = "SAME_REGULAR_INODE_EXTENT_AND_SHA256"
            except Exception as error:
                result.update(status="FAILED", remote_model_preservation="FAILED",
                              cache_error=f"{type(error).__name__}: {error}")
        if global_held and not uncertain and (not attempted or exit_known):
            try:
                from .target_model_io_probe import _release_global
                if cache_held:
                    _release_global(hdc, cache / ".execution-lock", token)
                    cache_held = False
                _release_global(hdc, global_lock, token)
                global_held = False
            except Exception as error:
                result.update(status="FAILED", lock_release_error=f"{type(error).__name__}: {error}")
        result["lock_state"] = "RETAINED_OR_UNCERTAIN" if global_held or cache_held or uncertain else "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
        result["remote_cancellation"] = "NOT_REQUESTED; uncertain waits retain locks and never trigger automatic model retry"
        result["next"] = (
            "Preserve this real-feature draft observation; establish target decode state-slot lineage, feedback and original embedding inputs before a multi-round loop"
            if result["status"] == "REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED" else
            "Retain reports, PID and logs; inspect uncertain remote work and lock ownership before any manual retry; no automatic model retry or lock removal"
            if result["lock_state"] == "RETAINED_OR_UNCERTAIN" else
            "Inspect the recorded failing stage and native logs before preparing a separate corrected attempt")
        result["hdc_invocations"] = hdc.count if hdc else 0
        save_json(out / "result.json", result)
    return result

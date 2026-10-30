"""Restore small target metadata to observe raw HIAI tensor descriptors only.

No manager.Init, weight initialization, tensor buffers, Process or Run calls
are requested. Model-cache
files preserve the exact compiler hierarchy and are reusable by the original
target executor. Equal descriptors do not establish semantic input identities.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shlex
import time
import uuid

from .artifacts import output_path, save_json
from .draft_board import _build_runner, _receive
from .elf_dynamic import dynamic_info
from .existing_air import _stat, _conversion_options
from .existing_air_profile import get_existing_air_profile
from .full_target import read_metadata, _recorded_file, _file
from .hashing import sha256_file
from .hdc_run import Hdc, _file_record, _upload
from .hiai_deployment import BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers, discover_libraries
from .original_target_board import _board_path
from .target_prefill_smoke_plan import _identity


_KIND = "qwen35-target-model-io-probe"
_RESERVE = 64 << 20


def _metadata(path, bindings):
    before = _stat(path)
    value = read_metadata(path)
    digest = sha256_file(path)
    if _stat(path) != before:
        raise ValueError("target probe metadata changed while reading")
    bindings.append({"path": str(path), "stat": before, "sha256": digest})
    return value


def _unchanged(bindings, payloads):
    for row in bindings:
        path = Path(row["path"])
        if _stat(path) != row["stat"] or sha256_file(path) != row["sha256"] or _stat(path) != row["stat"]:
            raise ValueError("target probe bound metadata changed: " + str(path))
    for row, path in payloads:
        if (_stat(path) != row.get("stat") or sha256_file(path) != row["sha256"] or _stat(path) != row.get("stat")):
            raise ValueError("target compiler payload metadata changed during probe: " + str(path))
        if row.get("active_identity") is not None and _identity(path.stat()) != row["active_identity"]:
            raise ValueError("target primary full identity changed during probe: " + str(path))


def _bound_inputs(job, inspection, *, verify_primary=True, primary_revalidation_observations=None):
    job = Path(job).resolve()
    root = job / "compilation"
    bindings = []
    outer = _metadata(job / "result.json", bindings)
    report = _metadata(root / "report.json", bindings)
    profile = _metadata(job / "compile-profile.json", bindings)
    contract = _metadata(job / "target-contract/report.json", bindings)
    inspection = Path(inspection).resolve()
    if inspection.is_dir():
        inspection = inspection / "report.json"
    tool = _metadata(inspection, bindings)
    digest = sha256_file(root / "report.json")
    profiles = [get_existing_air_profile(name) for name in ("cpro-40block-prefill-64", "cpro-40block-decode-1")]
    if profile not in profiles:
        raise ValueError("model-I/O probe requires one frozen static64 or static1 profile")
    expected_options = _conversion_options(profile["input_shape"], profile["dynamic_dims"],
        profile["limit_partition_size"], profile["save_weights_as_external_data"])
    if (outer.get("artifact_kind") != "qwen35-original-target-job" or outer.get("status") != "COMPILED"
            or outer.get("model_compilation") != "PASSED" or outer.get("output") != str(job)
            or outer.get("compilation_report_sha256") != digest
            or report.get("artifact_kind") != "qwen35-existing-air-compilation"
            or report.get("status") != "COMPILED" or report.get("model_compilation") != "PASSED"
            or type(report.get("returncode")) is not int or report["returncode"] != 0 or report.get("platform") != "kirinx90"
            or report.get("requested_options") != expected_options or report.get("output") != str(root)
            or outer.get("air") != report.get("air_source")
            or outer.get("compilation") != "compilation/report.json" or outer.get("input_layout") != "air-directory"
            or report.get("input_layout", {}).get("mode") != "air-directory"
            or contract.get("artifact_kind") != "qwen35-original-target-contract" or contract.get("status") != "OBSERVED"
            or contract.get("compilation_report_sha256") != digest
            or any(contract.get("omc", {}).get(k) != report.get("omc", {}).get(k) for k in ("bytes", "sha256"))
            or report.get("source_preservation", {}).get("status") != "UNCHANGED_BY_STAT"
            or report["source_preservation"].get("changed")
            or any(report.get(key) for key in ("audit_error", "artifact_inventory_error", "decode_input_audit_error", "retained_input_audit_error", "skipped_nonregular_artifacts"))):
        raise ValueError("model-I/O probe requires an unchanged successful frozen static target compilation and bound contract")
    if (tool.get("artifact_kind") != "qwen35-target-board-tool-inspection" or tool.get("status") != "INSPECTED"
            or tool.get("tool_identity") != "SHA256_MATCHED_BEFORE_AND_AFTER"
            or tool.get("tool_sha256_before") != tool.get("tool_sha256_after")
            or not re.fullmatch(r"[0-9a-f]{64}", str(tool.get("tool_sha256_before", "")))
            or tool.get("help", {}).get("observation") != "HELP_TEXT_OBSERVED"
            or any(tool.get(key) != "NOT_RUN" for key in ("model_upload", "model_load", "model_inference"))
            or tool.get("firmware", {}).get("returncode") != 0):
        raise ValueError("completed tool help and identity inspection required before model loading")
    records = report.get("generated_artifacts")
    if not isinstance(records, list) or not 1 <= len(records) <= 8192 or report.get("omc") not in records:
        raise ValueError("completed target has no complete generated payload inventory")
    paths, payloads = set(), []
    for row in records:
        if (not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))):
            raise ValueError("invalid compiler artifact declaration")
        relative = PurePosixPath(row["path"])
        try:
            relative = relative.relative_to("compile-work/generated")
        except ValueError as error:
            raise ValueError("compiled payload must remain inside the generated hierarchy") from error
        _board_path("/payload/" + relative.as_posix())
        if not relative.parts or relative in paths:
            raise ValueError("duplicate generated payload path")
        paths.add(relative)
        if row == report["omc"]:
            if not 0 < row["bytes"] <= 64 << 20:
                raise ValueError("restore metadata probe accepts only a small primary OMC at most64MiB; no large-model fallback")
            if verify_primary:
                # A prior permission change may alter historical ctime. Reuse
                # the bounded primary-only binder; no sidecar is opened and
                # the compiler receipt remains unchanged.
                from .target_decode_smoke_plan import bind_primary_omc
                path = _file(root, row.get("path"))
                current = bind_primary_omc(path, row,
                    revalidation_observations=primary_revalidation_observations)
                # The binder hashes historical ctime drift. A stat-identical
                # primary still needs its original complete content check.
                if current["scope"] == "DECLARED_SHA_AND_CURRENT_STAT_ONLY_MODEL_NOT_OPENED":
                    _recorded_file(root, row, "stat")
                active = dict(row, stat=_stat(path), active_identity=_identity(path.stat()))
                if active["active_identity"] != current["identity"]:
                    raise ValueError("primary changed between historical revalidation and probe baseline")
            else:
                path = _file(root, row.get("path"))
                active = row
            if not verify_primary and (_stat(path) != row.get("stat") or path.stat().st_size != row.get("bytes")):
                raise ValueError("primary OMC metadata differs before header inspection")
            payloads.append((active, path))
    boundary = contract.get("boundaries", {})
    if (len(boundary.get("inputs_in_serialized_op_order", [])) != 89 or len(boundary.get("outputs", [])) != 82):
        raise ValueError("target probe requires the complete89-input82-output feature target")
    firmware = tool["firmware"].get("text", "").strip()
    if not firmware or any(ord(c) < 32 for c in firmware):
        raise ValueError("tool inspection did not retain a usable firmware identity")
    return job, root, outer, report, profile, tool, firmware, bindings, payloads


def _cache_decisions(text, uploads):
    decisions = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(OK|SEND) (/[A-Za-z0-9_./-]+)", line.strip())
        if match is None or match[2] in decisions:
            raise ValueError("incomplete or ambiguous target cache checksum response")
        decisions[match[2]] = match[1]
    if set(decisions) != {str(remote) for _, _, remote in uploads}:
        raise ValueError("target cache response does not cover every compiler payload")
    return decisions


def _space(text):
    rows = [line.split() for line in text.splitlines() if line.strip()]
    candidates = [int(row[3]) * 1024 for row in rows if len(row) >= 6 and all(v.isdigit() for v in row[1:4])]
    if len(candidates) != 1:
        raise ValueError("board df -Pk did not return one unambiguous available-byte observation")
    return candidates[0]


def _acquire_global(hdc, lock, token, wait_seconds):
    started, last_update = time.monotonic(), -30.0
    while True:
        # HDC remote shells merge stderr into their returned stream. An
        # existing lock is expected contention, so its mkdir diagnostic must
        # not precede the exact BUSY protocol response.
        command = ("if mkdir " + shlex.quote(str(lock)) + " 2>/dev/null; then printf '%s\\n' " + shlex.quote(token)
                   + " > " + shlex.quote(str(lock / "owner")) + " && printf 'ACQUIRED\\n'; else test -d "
                   + shlex.quote(str(lock)) + " && test ! -L " + shlex.quote(str(lock)) + " && printf 'BUSY\\n'; fi")
        # A busy shared HDC server can take longer than ten seconds even for
        # this small command. A timeout still remains uncertain and is never
        # replayed automatically; the overall queue budget is separate.
        _, text = hdc.shell(command, "acquire-global-model-io-lock", timeout=min(60, hdc.timeout))
        if text == "ACQUIRED":
            return
        if text != "BUSY":
            detail = ""
            if callable(getattr(hdc, "_diagnostics", None)) and getattr(hdc, "last_call", None) is not None:
                detail = hdc._diagnostics()
            excerpt = repr(text[:2000]) + (" [truncated]" if len(text) > 2000 else "")
            raise ValueError("unrecognized global target model-I/O lock response: " + excerpt + detail)
        elapsed = time.monotonic() - started
        if elapsed >= wait_seconds:
            raise ValueError("BOARD_MODEL_IO_BUSY: another target probe owns the board; no model upload or loading attempted")
        if elapsed - last_update >= 30:
            print(f"target-model-io: waiting for the other board model probe; elapsed={elapsed:.0f}s", flush=True)
            last_update = elapsed
        time.sleep(min(5, wait_seconds - elapsed))


def _release_global(hdc, lock, token):
    hdc.shell("test -d " + shlex.quote(str(lock)) + " && test ! -L " + shlex.quote(str(lock))
              + " && read -r probe_lock_owner < " + shlex.quote(str(lock / "owner"))
              + " && test \"$probe_lock_owner\" = " + shlex.quote(token)
              + " && rm " + shlex.quote(str(lock / "owner")) + " && rmdir " + shlex.quote(str(lock)),
              "release-global-model-io-lock", timeout=min(60, hdc.timeout))


def _selected_headers(out):
    command = read_metadata(out / "native/command.json")
    prefixes = [value[len("-DHIAI_INCLUDE_DIR="):] for value in command.get("argv", []) if value.startswith("-DHIAI_INCLUDE_DIR=")]
    if len(prefixes) != 1 or not Path(prefixes[0]).is_absolute():
        raise ValueError("new runner build did not retain one selected HIAI include directory")
    include = Path(prefixes[0])
    bound = {row.get("path"): row for row in command.get("v2_headers", []) if isinstance(row, dict)}
    rows = []
    for relative in ("model/built_model.h", "model_manager/model_manager.h", "model_manager/model_manager_types.h", "tensor/nd_tensor_desc.h"):
        path = include / relative
        record = bound.get(relative, {})
        if not path.is_file() or record.get("status") != "PRESENT" or sha256_file(path) != record.get("sha256"):
            raise ValueError("selected V2 header differs from the new runner build: " + relative)
        with path.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("selected model/descriptor header exceeds small declaration read budget")
        rows.append({"path": relative, "bytes": len(raw), "sha256": record["sha256"], "text": raw.decode("utf-8", errors="replace")})
    external, descriptor_api = [], []
    for row in rows:
        lines = row["text"].splitlines()
        matches = [index for index, line in enumerate(lines) if re.search(r"InitWeights|ExternalWeight|Weight.*(?:Path|File|Buffer|Memory)|(?:Set|Get).*Weight", line, re.I)]
        chosen = sorted({index for match in matches for index in range(max(0, match - 3), min(len(lines), match + 7))})
        external.append({"path": row["path"], "lines": [{"line": index + 1, "text": lines[index]} for index in chosen[:240]],
            "truncated": len(chosen) > 240})
        matches = [index for index, line in enumerate(lines) if re.search(r"TensorDesc|Get.*Name|Tensor.*Name", line)]
        chosen = sorted({index for match in matches for index in range(max(0, match - 2), min(len(lines), match + 5))})
        descriptor_api.append({"path": row["path"], "lines": [{"line": index + 1, "text": lines[index]} for index in chosen[:240]],
            "truncated": len(chosen) > 240})
    save_json(out / "selected-model-io-headers.json", {"headers": rows, "external_weight_declarations": external,
        "descriptor_api_declarations": descriptor_api,
        "scope": "Exact four small declarations used for this native build; no SDK scan or inferred external-weight API semantics"})
    return {"files": [{key: row[key] for key in ("path", "bytes", "sha256")} for row in rows],
            "external_weight_declarations": external, "descriptor_api_declarations": descriptor_api,
            "name_scope": "Raw declarations only; a built-model GetName is not inferred to be a tensor-name API",
            "full_declarations": str(out / "selected-model-io-headers.json")}


def inspect_target_runtime_io(job, tool_inspection, output_dir, *, timeout_seconds=600,
                              lock_wait_seconds=300, board_model_root="/data/local/tmp/qwen3_5_moe",
                              headers_only=False):
    """Upload/cache small primary metadata and perform one restore-only query."""
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": _KIND, "status": "FAILED", "stage": "binding",
        "device_execution": "NOT_RUN", "model_inference": "NOT_RUN", "model_compilation": "NOT_RUN",
        "runtime_input_order": "NOT_QUERIED", "semantic_slot_identity": "UNVERIFIED", "numerical_validation": "NOT_RUN",
        "runtime_tensor_names": "NOT_QUERIED", "state_continuity": "NOT_RUN", "retrieved_sha256": {},
        "model_upload": "NOT_RUN", "model_load": "NOT_RUN",
        "descriptor_stage": "restore-only", "manager_init": "NOT_RUN", "external_weight_initialization": "NOT_REQUESTED",
        "weight_payload_access": "NONE; compiler sidecar declarations only, no fresh weight hash or upload",
        "primary_omc_revalidation_observations": [],
        "retrieval_errors": [], "board_cleanup": "OWNED_CACHE_AND_ATTEMPT_RETAINED", "output": str(out)}
    hdc = board = global_lock = cache = None
    global_held = cache_held = attempted = exit_known = uncertain = False
    bindings, payloads = [], []
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
        if (type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200
                or type(lock_wait_seconds) is not int or not 0 <= lock_wait_seconds <= 7200):
            raise ValueError("probe timeout must be1..7200seconds and lock wait0..7200seconds")
        if type(headers_only) is not bool:
            raise ValueError("headers_only must be boolean")
        job, root, outer, compiled, profile, tool, firmware, bindings, payloads = _bound_inputs(job, tool_inspection,
            verify_primary=not headers_only, primary_revalidation_observations=result["primary_omc_revalidation_observations"])
        digest, ddk = sha256_file(root / "report.json"), compiled.get("ddk_id")
        if not isinstance(ddk, str) or not ddk or (os.environ.get("CPRO_DDK_ID") and os.environ["CPRO_DDK_ID"] != ddk):
            raise ValueError("selected DDK differs from this compiled target")
        if headers_only:
            result.update(stage="build-runner-header-preflight", job=str(job), profile=profile["name"], ddk_id=ddk,
                compilation_report_sha256=digest, source_bindings=bindings, generated_payload_access="STAT_ONLY_NO_WEIGHT_CONTENT_READ")
            print("target-model-io: building the descriptor-only runner and capturing exact selected SDK weight declarations; no HDC/model upload or load", flush=True)
            _build_runner(out, timeout_seconds)
            result["selected_v2_headers"] = _selected_headers(out)
            result.update(status="HEADERS_OBSERVED", stage="complete", external_weight_contract="UNVERIFIED_REQUIRES_REVIEW",
                next="Review exact selected InitWeights/external-weight declarations before any large target transfer or manager.Init")
            return result
        sn, bip = os.environ.get("SN", ""), os.environ.get("BIP", "")
        if not sn or sn.startswith("-") or not re.fullmatch(r"[^\s:]+:[0-9]+", bip) or not 1 <= int(bip.rsplit(":", 1)[1]) <= 65535:
            raise ValueError("selected SN and HDC server BIP are required")
        base = _board_path(board_model_root)
        cache = base / ("codex-target-" + digest[:20])
        board = cache / ("io-" + token)
        global_lock = PurePosixPath("/data/local/tmp/.qwen35-target-model-io-lock")
        result.update(job=str(job), compilation_report_sha256=digest, profile=profile["name"],
            sequence_length=profile["declared_model_facts"]["fixed_sequence_length"],
            ddk_id=ddk, firmware_id=firmware, board_model_cache=str(cache), board_attempt=str(board),
            source_bindings=bindings, board_global_lock=str(global_lock), global_lock_wait_seconds=lock_wait_seconds,
            compiler_sidecar_declaration_count=len(compiled["generated_artifacts"]) - 1,
            compiler_sidecar_declared_bytes=sum(row["bytes"] for row in compiled["generated_artifacts"] if row != compiled["omc"]))
        hdc = Hdc(os.environ.get("HDC", "hdc"), sn, bip, out, timeout_seconds)
        hdc.shell("command -v sha256sum >/dev/null && command -v df >/dev/null && test -x "
            + shlex.quote(tool["tool"]) + " && sha256sum " + shlex.quote(tool["tool"]), "device-preflight", timeout=min(10, timeout_seconds))
        _, text = hdc.shell("sha256sum " + shlex.quote(tool["tool"]) + " && param get const.ohos.fullname", "device-identity", timeout=min(10, timeout_seconds))
        expected_line = tool["tool_sha256_before"] + "  " + tool["tool"]
        if expected_line not in text.splitlines() or text.splitlines()[-1].strip() != firmware:
            raise ValueError("selected board tool/firmware differs from the completed help inspection")
        result["stage"] = "build-runner"
        print("target-model-io: building current native descriptor-only runner; no model compilation", flush=True)
        runner = _build_runner(out, timeout_seconds)
        result["selected_v2_headers"] = _selected_headers(out)
        uploads = [_file_record(runner, "runner")]
        discovery = discover_libraries(runner, out / "native")
        libraries = [_file_record(path.resolve(), "lib/" + path.name) for path in discovery["libraries"]]
        dependencies = [{"name": PurePosixPath(row["remote"]).name, "kind": "runner" if row["remote"] == "runner" else "library",
            "sha256": row["sha256"], **dynamic_info(row["local"])} for row in [*uploads, *libraries]]
        blockers = candidate_blockers(dependencies, require_pair=False)
        if blockers:
            raise ValueError("cannot select native descriptor deployment: " + "; ".join(blockers))
        uploads.extend(row for row in libraries if PurePosixPath(row["remote"]).name not in DDK_HCL_EXCLUDED)
        result["libraries"] = {"discovery": discovery["metadata"], "dependencies": dependencies, "mode": "candidate_without_ddk_hcl"}
        hdc.shell("mkdir -p " + shlex.quote(str(base)), "board-root")
        result["stage"] = "wait-board-model-lock"
        uncertain = True
        _acquire_global(hdc, global_lock, token, lock_wait_seconds)
        global_held, uncertain = True, False
        owner = cache / ".compilation-sha256"
        command = ("if mkdir " + shlex.quote(str(cache)) + "; then printf '%s\\n' " + shlex.quote(digest) + " > " + shlex.quote(str(owner))
            + "; else test -d " + shlex.quote(str(cache)) + " && test ! -L " + shlex.quote(str(cache))
            + " && read -r target_cache_owner < " + shlex.quote(str(owner)) + " && test \"$target_cache_owner\" = " + shlex.quote(digest) + "; fi")
        hdc.shell(command, "own-model-cache")
        uncertain = True
        hdc.shell("mkdir " + shlex.quote(str(cache / ".execution-lock")), "acquire-cache-execution-lock")
        cache_held, uncertain = True, False
        result["stage"] = "model-cache"
        model_uploads = [(row, path, cache / "payload" / PurePosixPath(row["path"]).relative_to("compile-work/generated")) for row, path in payloads]
        directories = {remote.parent for _, _, remote in model_uploads}
        hdc.shell("mkdir " + shlex.quote(str(board)) + " && mkdir " + shlex.quote(str(board / "lib")), "own-probe-attempt")
        hdc.shell("mkdir -p " + " ".join(shlex.quote(str(value)) for value in sorted(directories)), "probe-directories")
        manifest = out / "model-files.tsv"
        manifest.write_text("".join(row["sha256"] + " " + str(remote) + "\n" for row, _, remote in model_uploads))
        hdc.send(manifest, board / "model-files.tsv")
        command = ("while read -r expected path; do if test -f \"$path\" && test ! -L \"$path\" && "
            "sha256sum \"$path\" | grep -q \"^$expected  \"; then printf 'OK %s\\n' \"$path\"; else printf 'SEND %s\\n' \"$path\"; fi; done < "
            + shlex.quote(str(board / "model-files.tsv")))
        _, text = hdc.shell(command, "inspect-model-cache")
        decisions = _cache_decisions(text, model_uploads)
        missing_bytes = sum(row["bytes"] for row, _, remote in model_uploads if decisions[str(remote)] == "SEND")
        _, disk = hdc.shell("df -Pk " + shlex.quote(str(cache)), "board-disk", timeout=min(10, timeout_seconds))
        _, memory = hdc.shell("cat /proc/meminfo", "board-memory", timeout=min(10, timeout_seconds))
        free = _space(disk)
        available = re.findall(r"(?m)^MemAvailable:\s+([0-9]+)\s+kB\s*$", memory)
        result["board_resources"] = {"disk_free_bytes": free, "missing_model_payload_bytes": missing_bytes,
            "runtime_upload_bytes": sum(row["bytes"] for row in uploads), "reserve_bytes": _RESERVE,
            "mem_available_bytes": int(available[0]) * 1024 if len(available) == 1 else None,
            "memory_scope": "MemAvailable observation only; compiled weight bytes are not inferred resident RAM", "df_text": disk}
        primary_bytes = compiled["omc"]["bytes"]
        result["board_resources"]["isolated_primary_copy_bytes"] = primary_bytes
        required = missing_bytes + primary_bytes + sum(row["bytes"] for row in uploads) + _RESERVE
        if free < required:
            raise ValueError(f"board disk budget insufficient: need{required}bytes, available{free}; no payload uploaded")
        primary = cache / "payload" / PurePosixPath(compiled["omc"]["path"]).relative_to("compile-work/generated")
        result.update(cached_primary_model_path=str(primary), model_sha256=compiled["omc"]["sha256"], model_bytes=compiled["omc"]["bytes"], model_upload_plan=[
            {"path": str(remote), "bytes": row["bytes"], "sha256": row["sha256"], "action": decisions[str(remote)]} for row, _, remote in model_uploads])
        save_json(out / "upload-plan.json", result["model_upload_plan"])
        print(f"target-model-io: model={primary}; missing payload upload={missing_bytes}bytes; board free={free}bytes", flush=True)
        _unchanged(bindings, payloads)
        for row, path, remote in model_uploads:
            if decisions[str(remote)] == "SEND":
                result["stage"] = "model-upload"
                print(f"target-model-io: uploading {remote.name} ({row['bytes']}bytes)", flush=True)
                uncertain = True
                hdc.send(path, remote)
                uncertain = False
            uncertain = True
            _, text = hdc.shell("sha256sum " + shlex.quote(str(remote)), "verify-model-payload")
            uncertain = False
            if text.strip() != row["sha256"] + "  " + str(remote):
                raise ValueError("board model payload SHA256 mismatch")
        # Restore in a fresh metadata-only directory. A previous full-model
        # cache may contain weights; those files are never deleted or placed
        # beside the actual restore experiment's primary OMC.
        isolated = board / "metadata" / PurePosixPath(compiled["omc"]["path"]).relative_to("compile-work/generated")
        metadata_root = board / "metadata"
        uncertain = True
        _, text = hdc.shell("mkdir " + shlex.quote(str(metadata_root)) + " && mkdir -p " + shlex.quote(str(isolated.parent))
            + " && cp " + shlex.quote(str(primary)) + " " + shlex.quote(str(isolated))
            + " && sha256sum " + shlex.quote(str(isolated)), "isolate-primary-metadata")
        uncertain = False
        if text.strip() != compiled["omc"]["sha256"] + "  " + str(isolated):
            raise ValueError("isolated small primary metadata copy SHA256 mismatch")
        result.update(model_path=str(isolated), sidecar_upload="NOT_RUN", restore_directory_scope="FRESH_ATTEMPT_METADATA_ONLY_SMALL_PRIMARY",
            isolation={"cache_primary": str(primary), "restored_primary": str(isolated), "sha256": compiled["omc"]["sha256"],
                "weight_files_copied": 0, "links_created": 0, "existing_cache_sidecars": "NOT_INSPECTED_NOT_DELETED_NOT_ASSUMED"})
        result["stage"] = "runner-upload"
        uncertain = True
        _upload(hdc, board, uploads, out / "native-files.sha256")
        hdc.shell("chmod 755 " + shlex.quote(str(board / "runner")), "runner-permissions")
        uncertain = False
        environment = "LD_LIBRARY_PATH=" + shlex.quote(":".join([str(board / "lib"), *BOARD_LIBRARY_SEARCH_PATHS])) + " "
        result["stage"] = "runtime-probe"
        uncertain = True
        try:
            runtime_status, _ = hdc.shell("cd " + shlex.quote(str(board)) + " && " + environment + "./runner --probe > runtime.json 2> runtime.stderr; "
                "rc=$?; printf '%s\\n' \"$rc\" > " + shlex.quote(str(board / "runtime.exitcode")) + "; exit \"$rc\"", "runtime-probe", allow_failure=True)
            uncertain = False
        finally:
            collect(("runtime.json", "runtime.stderr", "runtime.exitcode"))
        if any(name not in result["retrieved_sha256"] for name in ("runtime.json", "runtime.stderr", "runtime.exitcode")):
            raise ValueError("runtime capability probe evidence incomplete")
        runtime = read_metadata(out / "runtime.json")
        version = runtime.get("runtime_version")
        if (runtime_status != 0 or (out / "runtime.exitcode").read_text().strip() != "0" or runtime.get("npu_supported") is not True
                or not isinstance(version, str) or not version
                or (os.environ.get("HIAI_RUNTIME_VERSION") and os.environ["HIAI_RUNTIME_VERSION"] != version)):
            raise ValueError("actual HIAI runtime capability/version probe failed")
        result.update(runtime_version=version, stage="restore-metadata-query", device_execution="RESTORE_METADATA_ATTEMPTED",
            model_upload="SMALL_PRIMARY_OMC_AND_NATIVE_RUNTIME_ONLY", model_load="RESTORE_METADATA_ONLY")
        command = [str(board / "runner"), "--model-io-only", str(primary), "--model-sha256", compiled["omc"]["sha256"],
            "--descriptor-stage", "restore-only", "--expected-runtime-version", version, "--ddk-id", ddk,
            "--firmware-id", firmware, "--output", str(board / "model-io.json")]
        command[command.index("--model-io-only") + 1] = str(isolated)
        save_json(out / "model-io-command.json", {"argv": command, "cwd": str(isolated.parent),
            "scope": "One RestoreFromFile/descriptor query before manager.Init; no weight initialization, input files, Process or Run"})
        attempted, uncertain = True, True
        status, _ = hdc.shell("cd " + shlex.quote(str(isolated.parent)) + " && " + environment + shlex.join(command)
            + " > " + shlex.quote(str(board / "model-io.stdout")) + " 2> " + shlex.quote(str(board / "model-io.stderr"))
            + "; rc=$?; printf '%s\\n' \"$rc\" > " + shlex.quote(str(board / "model-io.exitcode")) + "; exit \"$rc\"",
            "model-io-load", allow_failure=True)
        result["remote_exitcode"] = status
        exit_known, uncertain = True, False
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if attempted:
            collect(("model-io.json", "model-io.stdout", "model-io.stderr", "model-io.exitcode"))
            try:
                if any(name not in result["retrieved_sha256"] for name in ("model-io.json", "model-io.stdout", "model-io.stderr", "model-io.exitcode")):
                    raise ValueError("hash-verified native model-I/O evidence incomplete")
                native = read_metadata(out / "model-io.json")
                if (native.get("artifact_kind") != "qwen35-hiai-model-io-probe" or native.get("evidence_kind") != "device-hiai"
                        or native.get("model_path") != result["model_path"] or native.get("model_sha256") != result["model_sha256"]
                        or native.get("model_bytes") != result["model_bytes"]
                        or native.get("expected_runtime_version") != result["runtime_version"] or native.get("firmware_id") != result["firmware_id"]
                        or native.get("ddk_id") != result["ddk_id"] or native.get("process_called") is not False
                        or native.get("run_called") is not False or native.get("tensor_allocations_requested") != 0
                        or native.get("model_load_api") != "file-v2" or native.get("descriptor_stage") != "restore-only"
                        or native.get("manager_init") != "NOT_RUN" or native.get("model_load_scope") != "RESTORE_METADATA_ONLY"
                        or native.get("external_weight_initialization") != "NOT_REQUESTED"
                        or native.get("dtype_enum") != "hiai::DataType" or native.get("format_enum") != "hiai::Format"):
                    raise ValueError("native model-I/O result differs from this exact descriptor-only invocation")
                result.update(native_status=native.get("status"), runtime_stage=native.get("runtime_stage"), native_report=str(out / "model-io.json"),
                    model_restore=native.get("restore"), runtime_input_order=native.get("runtime_input_order"), runtime_tensor_names=native.get("runtime_tensor_names"),
                    input_count=native.get("input_count"), output_count=native.get("output_count"), inputs=native.get("inputs"), outputs=native.get("outputs"))
                if native.get("status") == "MODEL_IO_FAILED":
                    result.update(device_execution="RESTORE_METADATA_FAILED", error=native.get("reason", "native model restore metadata query failed"))
                elif (native.get("status") == "IO_DESCRIPTORS_OBSERVED" and result.get("remote_exitcode") == 0
                        and (out / "model-io.exitcode").read_text().strip() == "0"
                        and native.get("runtime_version") == result["runtime_version"] and native.get("restore") == "PASSED"
                        and native.get("restore_init") == "NOT_RUN_METADATA_ONLY"):
                    for key in ("inputs", "outputs"):
                        rows = native.get(key)
                        if (not isinstance(rows, list) or not 1 <= len(rows) <= 4096 or native.get(key[:-1] + "_count") != len(rows)
                                or any(not isinstance(row, dict) or row.get("index") != index or not isinstance(row.get("shape"), list)
                                    or any(type(dim) is not int for dim in row["shape"]) or type(row.get("raw_dtype")) is not int
                                    or type(row.get("raw_format")) is not int for index, row in enumerate(rows))):
                            raise ValueError("native target raw descriptor records are malformed")
                    result.update(status="IO_METADATA_OBSERVED", stage="complete", device_execution="RESTORE_METADATA_QUERIED_NO_MANAGER_INIT", model_restore="PASSED",
                        compiled_boundary_count_check=("MATCHED_89_INPUT_82_OUTPUT_COUNTS_ONLY" if native["input_count"] == 89 and native["output_count"] == 82 else "DIFFERENT_COUNTS_RAW_METADATA_RETAINED"),
                        dtype_enum=native.get("dtype_enum"), format_enum=native.get("format_enum"),
                        dtype_enum_values=native.get("dtype_enum_values"), format_enum_values=native.get("format_enum_values"),
                        inputs=native["inputs"], outputs=native["outputs"])
                else:
                    raise ValueError("native model-I/O status/exit did not establish completed descriptor observation")
            except Exception as error:
                result["native_evidence_error"] = f"{type(error).__name__}: {error}"
        if bindings:
            try:
                _unchanged(bindings, payloads)
                result["source_preservation"] = "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED"
            except Exception as error:
                result.update(status="FAILED", source_preservation="FAILED", source_error=str(error))
        result["remote_cancellation"] = "NOT_REQUESTED; uncertain waits retain locks and never trigger automatic retry"
        if global_held and not uncertain and (not attempted or exit_known):
            try:
                if cache_held:
                    hdc.shell("rmdir " + shlex.quote(str(cache / ".execution-lock")), "release-cache-execution-lock", timeout=min(10, hdc.timeout))
                    cache_held = False
                _release_global(hdc, global_lock, token)
                global_held = False
            except Exception as error:
                result["lock_release_error"] = f"{type(error).__name__}: {error}"
        result["lock_state"] = "RETAINED_OR_UNCERTAIN" if global_held or cache_held or uncertain else "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
        result["scope"] = ("Selected native build/header declarations only; no HDC or model restore." if headers_only else
            "Restore-only metadata experiment before manager.Init or weight initialization. Raw ordered HIAI descriptor enum domains differ from GE. Missing external dependencies fail without large-weight fallback. Equal-shaped state/scalar slots remain semantically unverified; no full model load, inference, state continuity or accuracy pass.")
        save_json(out / "result.json", result)
    return result

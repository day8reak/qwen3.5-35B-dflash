"""Compile an existing AIR with its own, still unverified model/runtime ABI.

Inputs are copied or reflinked into an isolated input directory. No source
directory links, FileConstant rewrites, FP16 coercion, graph reconstruction,
platform diagnosis matrix, or device execution is performed.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from .artifacts import output_path, save_json
from .compile import (omg_environment, _environment_record, _omg_ipc_environment,
                      _retain_compiler_diagnostics, prepare_tbe_compatibility,
                      verify_tbe_compatibility)
from .compiler_evidence import compile_evidence
from .hashing import sha256_file
from .tool_logs import log_excerpt


_DATA_SUFFIXES = frozenset({".bin", ".data", ".dat", ".weights", ".weight",
                            ".npy", ".npz", ".safetensors"})
_DATA_DIRECTORIES = frozenset({"weights", "weight", "data", "constants", "external_data"})
_MAX_ENTRIES, _MAX_FILES, _MAX_DEPTH = 8192, 4096, 4
_COPY_RESERVE_BYTES = 64 << 20


def _stat(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _snapshot(source, destination):
    """Make an independent snapshot; reflink avoids duplicating large files."""
    before = _stat(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    method = "REFLINK"
    with source.open("rb") as src, destination.open("xb+") as dst:
        try:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
        except OSError as clone_error:
            method = "COPY"
            dst.seek(0)
            dst.truncate()
            space = os.fstatvfs(dst.fileno())
            available = space.f_bavail * space.f_frsize
            required = before[2] + _COPY_RESERVE_BYTES
            if available < required:
                raise OSError(errno.ENOSPC,
                    f"Cannot snapshot {source.name}: reflink is unavailable and physical copy "
                    f"needs {required} bytes ({before[2]} source + {_COPY_RESERVE_BYTES} reserve), "
                    f"but destination filesystem has {available} bytes available. "
                    "Free disk space or choose AI_RUN_DIR on a filesystem with sufficient space "
                    "or reflink support. Source is unchanged; the empty destination is retained.") from clone_error
            while data := src.read(8 << 20):
                dst.write(data)
                digest.update(data)
        dst.flush()
    if _stat(source) != before or destination.stat().st_size != before[2]:
        raise ValueError(f"source changed during input snapshot: {source.name}")
    digest_text = sha256_file(destination) if method == "REFLINK" else digest.hexdigest()
    destination.chmod(0o444)
    return {"source": str(source), "bytes": before[2], "sha256": digest_text,
            "snapshot_method": method, "source_stat": before,
            "snapshot_stat": _stat(destination)}


def _input_inventory(air):
    """Bound traversal to data-like siblings and explicitly named data roots."""
    parent = air.parent
    files, skipped, visited = [air], [], 0
    directories = _DATA_DIRECTORIES | {air.name + ".data", air.stem + "_weights", air.stem + "_data"}

    def walk(folder, depth):
        nonlocal visited
        with os.scandir(folder) as entries:
            for entry in entries:
                visited += 1
                if visited > _MAX_ENTRIES:
                    raise ValueError("AIR sidecar inventory exceeds bounded entry limit")
                path = Path(entry.path)
                relative = str(path.relative_to(parent))
                if path == air:
                    continue
                if entry.is_symlink():
                    skipped.append({"path": relative, "reason": "SYMLINK_NOT_FOLLOWED"})
                elif entry.is_dir(follow_symlinks=False):
                    if depth == 0 and entry.name not in directories:
                        skipped.append({"path": relative, "reason": "UNSELECTED_DIRECTORY"})
                    elif depth >= _MAX_DEPTH:
                        skipped.append({"path": relative, "reason": "DEPTH_LIMIT"})
                    else:
                        walk(path, depth + 1)
                elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in _DATA_SUFFIXES:
                    files.append(path)
                    if len(files) > _MAX_FILES:
                        raise ValueError("AIR sidecar inventory exceeds bounded file limit")
                else:
                    skipped.append({"path": relative, "reason": "NOT_SELECTED_DATA_FILE"})

    walk(parent, 0)
    return [air, *sorted(path for path in files if path != air)], skipped


def _conversion_options(input_shape, dynamic_dims, limit_partition_size, save_weights_as_external_data):
    values = {}
    for name, value in (("input_shape", input_shape), ("dynamic_dims", dynamic_dims)):
        if value is not None:
            if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError(f"{name} must be nonempty text without control characters")
            values[name] = value
    if input_shape is not None:
        names = []
        for entry in input_shape.split(";"):
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_./-]*):(-1|[1-9][0-9]*)(,(-1|[1-9][0-9]*))*", entry)
            if match is None:
                raise ValueError("input_shape requires unique name:dimension[,dimension] entries separated by semicolons")
            names.append(match.group(1))
        if len(set(names)) != len(names):
            raise ValueError("input_shape names must be unique")
    if dynamic_dims is not None:
        if input_shape is None:
            raise ValueError("dynamic_dims requires input_shape")
        if re.fullmatch(r"[1-9][0-9]*(,[1-9][0-9]*)*(;[1-9][0-9]*(,[1-9][0-9]*)*)*", dynamic_dims) is None:
            raise ValueError("dynamic_dims requires positive comma-separated dimensions and semicolon-separated gears")
    for name, value in (("limit_partition_size", limit_partition_size),
                        ("save_weights_as_external_data", save_weights_as_external_data)):
        if value is not None:
            if type(value) is not bool:
                raise ValueError(f"{name} must be bool or None")
            values[name] = "true" if value else "false"
    return values


def _check_help(omg, env, output, options, timeout_seconds):
    folder = output / "compile-work/help"
    folder.mkdir()
    command = [str(omg), "--help"]
    save_json(output / "help-command.json", {"argv": command, "cwd": str(folder)})
    log = output / "help.log"
    with _omg_ipc_environment(folder, env) as (local_env, ipc), log.open("x") as stream:
        record = _environment_record(env["DDK_PATH"], local_env)
        record["ipc_temporary_directory"] = ipc
        save_json(output / "help-environment.json", record)
        result = subprocess.run(command, cwd=folder, env=local_env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False, timeout=min(timeout_seconds, 60))
    text = log.read_text(errors="replace")
    advertised = {name: re.search(r"(?<![A-Za-z0-9_])--" + re.escape(name) + r"(?![A-Za-z0-9_])", text) is not None
                  for name in options}
    missing = [name for name, present in advertised.items() if not present]
    recognizable_help = bool(options and not missing) or bool(
        re.search(r"(?i)\b(?:usage|options|parameters|flags)\b", text)
        and re.search(r"(?<![A-Za-z0-9_])--[a-z][a-z0-9_]*", text))
    status = ("EMPTY_HELP" if not text.strip() else "INVALID_HELP_RESPONSE" if not recognizable_help else
              "OPTIONS_NOT_ADVERTISED" if missing else "OPTIONS_ADVERTISED")
    record = {"status": status,
              "returncode": result.returncode, "requested": options, "advertised": advertised,
              "missing": missing, "log": "help.log", "sha256": sha256_file(log),
              "scope": "Help may omit internal options. Requested arguments are preserved for actual compiler validation; help absence is not proof of unsupported options. Some tools return nonzero for help."}
    save_json(output / "help-check.json", record)
    # An empty/failing help response is not evidence even when no extra options
    # were requested. Nonzero usage exits with actual help text remain recorded.
    if not text.strip():
        raise ValueError("selected OMG --help emitted no help text")
    if not recognizable_help:
        raise ValueError("selected OMG --help did not emit recognizable option help; inspect help.log for loader/runtime errors")
    return record


def _generated_inventory(output, snapshots):
    """Hash all retained compiler files, excluding independently staged inputs."""
    excluded = {row["path"] for row in snapshots}
    generated, context, skipped = [], [], []
    work = output / "compile-work"
    for directory, subdirs, names in os.walk(work, followlinks=False):
        base = Path(directory)
        for name in list(subdirs):
            path = base / name
            if path.is_symlink():
                skipped.append(str(path.relative_to(output)))
                subdirs.remove(name)
        for name in names:
            path = base / name
            relative = str(path.relative_to(output))
            if relative in excluded:
                continue
            if path.is_symlink() or not path.is_file():
                skipped.append(relative)
                continue
            before = _stat(path)
            digest = sha256_file(path)
            if _stat(path) != before:
                raise ValueError(f"compiler artifact changed while hashing: {relative}")
            in_generated = (work / "generated") in path.parents
            role = ("primary_omc" if relative == "compile-work/generated/model.omc" else
                    "external_data_candidate" if in_generated else "compiler_context")
            row = {"path": relative, "bytes": before[2], "sha256": digest,
                   "stat": before, "role": role}
            (generated if in_generated else context).append(row)
    return sorted(generated, key=lambda row: row["path"]), sorted(context, key=lambda row: row["path"]), skipped


def compile_existing_air(air_path, output_dir, ddk_root, *, ddk_id,
                         tbe_diagnosis=None, timeout_seconds=7200, omg_path=None,
                         input_shape=None, dynamic_dims=None, limit_partition_size=None,
                         save_weights_as_external_data=None, progress=None):
    """Return a host compilation report, retaining failures without deployment.

    Only inventoried local snapshots are hash bound. Absolute or out-of-tree
    FileConstant references are not discovered or rewritten, so their identity
    and the original model's tensor ABI remain explicitly unverified.
    """
    if (not isinstance(ddk_id, str) or not ddk_id.strip()
            or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("declare a nonempty ddk_id and finite positive timeout_seconds")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    options = _conversion_options(input_shape, dynamic_dims, limit_partition_size, save_weights_as_external_data)

    def notify(stage, message, **details):
        if progress is not None:
            progress({"stage": stage, "message": message, **details})

    air, ddk = Path(air_path).resolve(), Path(ddk_root).resolve()
    if not air.is_file() or not air.stat().st_size:
        raise ValueError("existing AIR must be a nonempty regular file")
    output = output_path(output_dir)
    if output == air.parent or air.parent in output.parents or output == ddk or ddk in output.parents:
        raise ValueError("compilation output must be outside the source and DDK directories")
    omg, env = omg_environment(ddk)
    if omg_path is not None:
        selected = Path(omg_path).resolve()
        if ddk not in selected.parents or not selected.is_file():
            raise ValueError("explicit OMG must be a regular executable within the selected DDK")
        with selected.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError("explicit OMG must be an ELF tool; vendor shell/Python wrappers can write shared DDK files. Use tools/tools_omg/master/omg")
        omg = selected
    if not os.access(omg, os.X_OK):
        raise PermissionError(f"selected OMG is not executable: {omg}")
    output.mkdir(parents=True, exist_ok=False)
    scratch, inputs, generated = output / "compile-work", output / "compile-work/input", output / "compile-work/generated"
    inputs.mkdir(parents=True)
    generated.mkdir()
    log = output / "compile.log"
    report = {
        "schema_version": 1, "artifact_kind": "qwen35-existing-air-compilation",
        "status": "COMPILING", "output": str(output), "comparison": str(output / "report.txt"),
        "platform": "kirinx90", "ddk_id": ddk_id, "ddk_root": str(ddk),
        "omg_path": str(omg), "omg_sha256": sha256_file(omg), "air_source": str(air),
        "requested_options": options,
        "source_contract": {key: "UNVERIFIED" for key in
                            ("checkpoint", "quantization", "layer_count", "shape", "input_output_abi")},
        "legacy_hiai_abi_compatible": "UNVERIFIED", "stateful_hiai_abi_compatible": "UNVERIFIED",
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
        "runtime_placement": "UNVERIFIED", "device_ready": False,
        "compile_work": "compile-work", "compile_log": "compile.log",
        "input_snapshots": [],
        "external_weights": {
            "status": "UNVERIFIED", "scope": "Only selected neighboring data files are inventoried and hash bound.",
            "limitations": "AIR is not parsed. Absolute/out-of-tree, symlinked, unselected names or deeper FileConstant references remain unbound; no path is rewritten.",
            "max_entries": _MAX_ENTRIES, "max_files": _MAX_FILES, "max_depth": _MAX_DEPTH,
            "data_suffixes": sorted(_DATA_SUFFIXES), "skipped": [],
        },
    }
    compatibility = None
    if tbe_diagnosis is not None:
        compatibility = {"diagnosis": str(Path(tbe_diagnosis).resolve()),
                         "compiler_invocation": "NOT_ATTEMPTED", "compile": "NOT_RUN",
                         "audit": {"status": "NOT_PREPARED"}, "metadata": None,
                         "scope": "Compile outcome is separate from adaptation; NOT_OBSERVED does not prove a repair."}
        report["tbe_compatibility"] = compatibility
    stage = "omg-help"
    attempted = False
    error = None
    try:
        notify(stage, "Checking selected OMG help before copying AIR or sidecars")
        report["help_check"] = _check_help(omg, env, output, options, timeout_seconds)
        if report["help_check"]["missing"]:
            notify(stage, "Requested options not listed in help; compiler will validate unchanged arguments",
                   options=report["help_check"]["missing"])
        if compatibility is not None:
            stage = "tbe-compatibility-prepare"
            env, metadata = prepare_tbe_compatibility(ddk, tbe_diagnosis,
                                                       output / "tbe-compat", inherited=env)
            compatibility.update(metadata=metadata, audit={"status": "NOT_OBSERVED"})
        stage = "input-snapshot"
        selected, skipped = _input_inventory(air)
        report["external_weights"]["skipped"] = skipped
        selection = {"file_count": len(selected),
                     "total_bytes": sum(source.stat().st_size for source in selected),
                     "physical_copy_reserve_bytes": _COPY_RESERVE_BYTES,
                     "scope": "Selected AIR and neighboring data candidates; not verified AIR dependencies. "
                              "Free space is checked per file only if reflink fails."}
        report["input_selection"] = selection
        save_json(output / "input-selection.json", selection)
        notify(stage, f"Snapshotting {selection['file_count']} files, {selection['total_bytes']} bytes selected",
               total=selection["file_count"], total_bytes=selection["total_bytes"])
        for index, source in enumerate(selected):
            relative = source.relative_to(air.parent)
            snapshot = inputs / relative
            notify(stage, "Snapshotting input", completed=index, total=len(selected),
                   source_relative=str(relative), bytes=source.stat().st_size)
            row = _snapshot(source, snapshot)
            row.update(path=str(snapshot.relative_to(output)), source_relative=str(relative),
                       role="air" if source == air else "neighbor_data_candidate")
            report["input_snapshots"].append(row)
            if source == air:
                report["air"] = {"path": str(air), "bytes": row["bytes"], "sha256": row["sha256"]}
        save_json(output / "input-inventory.json", {"files": report["input_snapshots"],
                                                    "external_weights": report["external_weights"]})
        command = [str(omg), "--framework=1", f"--model={inputs / air.name}",
                   f"--output={generated / 'model'}", "--target=omc", "--platform=kirinx90"]
        command.extend(f"--{name}={value}" for name, value in options.items())
        save_json(output / "compile-command.json", {"argv": command, "cwd": str(inputs),
                                                    "weight_dtype_override": None})
        stage = "omg"
        notify(stage, "Compiling existing AIR with the selected DDK; full output is in compile.log")
        with _omg_ipc_environment(inputs, env) as (local_env, ipc), log.open("x") as stream:
            environment = _environment_record(ddk, local_env)
            environment["ipc_temporary_directory"] = ipc
            save_json(output / "omg-environment.json", environment)
            attempted = True
            report["model_compilation"] = "FAILED"
            if compatibility is not None:
                compatibility.update(compiler_invocation="ATTEMPTED", compile="FAILED")
            subprocess.run(command, cwd=inputs, env=local_env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout_seconds)
        stage = "output-check"
        omc = generated / "model.omc"
        if omc.is_symlink() or not omc.is_file() or not omc.stat().st_size:
            raise RuntimeError("OMG succeeded without a nonempty regular model.omc")
        report["model_compilation"] = "PASSED"
        if compatibility is not None:
            compatibility["compile"] = "PASSED"
    except Exception as exc:
        error = exc
        report.update(stage=stage, error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, subprocess.CalledProcessError):
            report["returncode"] = exc.returncode
        if isinstance(exc, subprocess.TimeoutExpired):
            report["timeout_seconds"] = exc.timeout
    finally:
        if "help_check" not in report and (output / "help-check.json").is_file():
            report["help_check"] = json.loads((output / "help-check.json").read_text())
        if compatibility is not None:
            if compatibility["metadata"] is not None and attempted:
                try:
                    audit = verify_tbe_compatibility(compatibility["metadata"])
                    if not isinstance(audit, dict) or audit.get("status") not in {"APPLIED", "NOT_OBSERVED", "REJECTED"}:
                        raise ValueError("invalid TBE audit result")
                    compatibility["audit"] = audit
                except Exception as exc:
                    compatibility["audit"] = {"status": "AUDIT_FAILED", "error": f"{type(exc).__name__}: {exc}"}
                if compatibility["audit"]["status"] not in {"APPLIED", "NOT_OBSERVED"}:
                    report["audit_error"] = compatibility["audit"]
                    if error is None:
                        error = RuntimeError("TBE compatibility audit rejected compilation")
                        report.update(stage="tbe-compatibility-audit", error=str(error))
            save_json(output / "tbe-compatibility-result.json", compatibility)
        try:
            report["compiler_diagnostics"] = _retain_compiler_diagnostics(inputs, output, failure=error is not None)
        except Exception as exc:
            report["compiler_diagnostics"] = {"status": "RETENTION_FAILED", "error": str(exc)}
        try:
            notify("artifact-inventory", "Hashing retained compiler outputs, including external-data candidates")
            artifacts, context, skipped = _generated_inventory(output, report["input_snapshots"])
            report["generated_artifacts"], report["compiler_work_artifacts"] = artifacts, context
            report["skipped_nonregular_artifacts"] = skipped
            report["external_data_candidates"] = [row for row in artifacts if row["role"] != "primary_omc"]
            report["external_data_scope"] = "Preserve every generated file and its relative path. Candidate classification does not prove model dependency completeness."
            primary = next((row for row in artifacts if row["role"] == "primary_omc"), None)
            if primary is not None:
                report["omc"] = primary
            if any(path == "compile-work/generated" or path.startswith("compile-work/generated/") for path in skipped):
                raise ValueError("generated model directory contains nonregular artifacts that could not be hash bound")
        except Exception as exc:
            report["artifact_inventory_error"] = f"{type(exc).__name__}: {exc}"
            if error is None:
                error = exc
                report.update(stage="artifact-inventory", error=report["artifact_inventory_error"])
        changed = []
        for row in report["input_snapshots"]:
            try:
                if (_stat(Path(row["source"])) != row["source_stat"]
                        or _stat(output / row["path"]) != row["snapshot_stat"]):
                    changed.append(row["source_relative"])
            except OSError:
                changed.append(row["source_relative"])
        preservation = "CHANGED" if changed else ("UNCHANGED_BY_STAT" if report["input_snapshots"] else "NOT_CHECKED")
        report["source_preservation"] = {"status": preservation,
                                         "changed": changed,
                                         "scope": "Hash-bound snapshot plus before/after metadata checks; no original file writes or source directory links."}
        if changed and error is None:
            error = ValueError("source or staged input changed during compilation")
            report.update(stage="input-recheck", error=str(error))
        if log.is_file():
            report["compiler_evidence"] = compile_evidence(log.read_text(errors="replace"))
            report["compile_log_sha256"] = sha256_file(log)
    report["status"] = "FAILED" if error is not None else "COMPILED"
    if error is not None:
        save_json(output / "failure.json", report)
    else:
        save_json(output / "manifest.json", report)
    save_json(output / "report.json", report)
    lines = ["Existing AIR -> selected-DDK OMG compilation", f"status={report['status']}",
             "Target: kirinx90; AIR weight dtype retained without FP16 override.",
             "Original model checkpoint, quantization, layer count, shapes and runtime ABI remain UNVERIFIED.",
             "No board execution; this artifact is not validated against this project's runner ABI.",
             "Tool help: help.log; option verification: help-check.json.",
             f"Source AIR: {air}", "Input inventory: input-inventory.json",
             "Full compiler log: compile.log; compiler work: compile-work/"]
    lines.append(f"Generated files: {len(report.get('generated_artifacts', []))}; external-data candidates: {len(report.get('external_data_candidates', []))}")
    if compatibility is not None:
        lines.append("TBE compatibility: " + json.dumps({key: compatibility[key] for key in
                                                       ("compiler_invocation", "compile", "audit")}))
    if error is not None:
        lines.append(report["error"])
        if log.is_file():
            lines.append(log_excerpt(log))
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    return report

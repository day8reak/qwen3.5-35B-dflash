"""One narrowly bound static-prefill retry using a failed job's read-only inputs.

Weights share old snapshot inodes.  This saves an explicit second payload copy;
it is not an independent snapshot or protection against privileged writers.
No old receipt is changed, and no failure cleanup removes created links.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .existing_air_profile import get_existing_air_profile
from .target_compile_failure import MAX_AIR_BYTES, MAX_METADATA_BYTES, _read_small, _relative, _safe


_HASH = re.compile(r"[0-9a-f]{64}")
_MAX_FILES = 4096


def _json(path, bindings):
    raw, identity = _read_small(path, MAX_METADATA_BYTES)
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate retry metadata key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite retry metadata: " + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("retry metadata must be an object")
    bindings.append(identity)
    return value, identity


def _identity(path):
    path, _ = _safe(path)
    info = path.stat(follow_symlinks=False)
    return {"dev": info.st_dev, "ino": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "mode": info.st_mode, "uid": info.st_uid, "gid": info.st_gid,
            "nlink": info.st_nlink}


def _five(identity):
    return [identity[k] for k in ("dev", "ino", "bytes", "mtime_ns", "ctime_ns")]


def _inode(identity):
    return identity["dev"], identity["ino"]


def _open_directory(path):
    """Walk directory components using no-follow directory descriptors."""
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _hash_file(path):
    path, _ = _safe(path)
    before = _identity(path)
    parent = _open_directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
    finally:
        os.close(parent)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        if _five(before) != [getattr(os.fstat(stream.fileno()), key) for key in
                            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")]:
            raise ValueError("retry input changed before hashing: " + str(path))
        while block := stream.read(8 << 20):
            digest.update(block)
        info = os.fstat(stream.fileno())
        if _five(before) != [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]:
            raise ValueError("retry input changed during hashing: " + str(path))
    if _identity(path) != before:
        raise ValueError("retry input identity changed during hashing: " + str(path))
    return digest.hexdigest(), before


def _new_path(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    parent = path
    while not parent.exists() and not parent.is_symlink():
        parent = parent.parent
    _safe(parent, directory=True)
    return path


def _options(profile, *, static):
    shape = profile["input_shape"]
    if static:
        shape = ";".join(name + ":" + ",".join("64" if size == "-1" else size
                           for size in dims.split(","))
                         for name, dims in (entry.split(":", 1) for entry in shape.split(";")))
    result = {"input_shape": shape,
              "limit_partition_size": "false", "save_weights_as_external_data": "true"}
    if not static:
        result["dynamic_dims"] = profile["dynamic_dims"]
    return result


class _ReusePlan:
    def __init__(self, *, output, rows, identities, metadata, evidence):
        self.output, self.rows, self.identities = output, rows, identities
        self.metadata = metadata
        self.evidence = evidence
        self.attempted = set()
        self.completed = set()
        self.journal = output / "retained-input-links.jsonl"
        descriptor = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
        self.journal_identity = _identity(self.journal)
        self.evidence["journal"] = {"path": str(self.journal), "scope": "Intent/results only; no interrupted-attempt recovery"}
        self._journal("PREPARED", failed_job=evidence["failed_job"], input_count=len(rows))

    def _journal(self, event, **values):
        current = _identity(self.journal)
        if current != self.journal_identity:
            raise ValueError("retry journal identity changed outside this attempt")
        raw = (json.dumps({"event": event, **values}, sort_keys=True) + "\n").encode()
        if len(raw) > 64 << 10:
            raise ValueError("retry journal event exceeds its small-record budget")
        descriptor = os.open(self.journal, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if _five(current) != [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]:
                raise ValueError("retry journal changed before append")
            remaining = memoryview(raw)
            while remaining:
                count = os.write(descriptor, remaining)
                if count <= 0:
                    raise OSError("retry journal short write")
                remaining = remaining[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        after = _identity(self.journal)
        if (any(after[key] != current[key] for key in ("dev", "ino", "mode", "uid", "gid", "nlink"))
                or after["bytes"] != current["bytes"] + len(raw)):
            raise ValueError("retry journal changed during append")
        self.journal_identity = after

    def snapshot(self, source, destination, normal_snapshot):
        source = Path(os.path.abspath(source))
        destination = Path(os.path.abspath(destination))
        key = str(source)
        if key not in self.rows or key in self.attempted:
            raise ValueError("retry snapshot source is unknown or already attempted")
        row = self.rows[key]
        if destination != self.output / row["path"]:
            raise ValueError("retry destination must preserve the old input snapshot relative path")
        self.attempted.add(key)
        observation = {"source": key, "old_snapshot": str(row["old_snapshot"]),
                       "destination": str(destination), "role": row["role"],
                       "status": "PREPARING", "link_created": False}
        self.evidence["snapshot_attempts"].append(observation)
        try:
            if _identity(source) != self.identities[key]["source"]:
                raise ValueError("original staged source changed before retry snapshot")
            before = _identity(row["old_snapshot"])
            if before != self.identities[key]["snapshot"]:
                raise ValueError("old snapshot changed before retry snapshot")
            if destination.exists() or destination.is_symlink():
                raise ValueError("retry destination already exists; partial evidence is preserved")
            _new_path(destination.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _safe(destination.parent, directory=True)
            if row["role"] == "air":
                result = normal_snapshot(source, destination)
                current = _identity(destination)
                if (result.get("source") != key or result.get("source_stat") != row["source_stat"]
                        or result.get("sha256") != row["sha256"] or result.get("bytes") != row["bytes"]
                        or result.get("snapshot_stat") != _five(current)
                        or _inode(current) in {_inode(before), _inode(self.identities[key]["source"])}
                        or current["mode"] & 0o222 or current["nlink"] != 1):
                    raise ValueError("retry AIR normal snapshot is not independent and hash bound")
                observation.update(status="INDEPENDENT_AIR_SNAPSHOT", after=current)
            else:
                digest, hashed = _hash_file(row["old_snapshot"])
                if hashed != before or digest != row["sha256"]:
                    raise ValueError("old read-only payload SHA256 differs from failed-job receipt")
                observation.update(before=before, sha256=digest)
                parent = _identity_directory(destination.parent)
                if parent["dev"] != before["dev"]:
                    raise ValueError("retry hardlink requires the same filesystem; no copy fallback")
                self._journal("LINK_INTENT", source=key, old_snapshot=str(row["old_snapshot"]),
                              destination=str(destination), expected_sha256=digest, before=before)
                old_fd, new_fd = _open_directory(row["old_snapshot"].parent), _open_directory(destination.parent)
                try:
                    os.link(row["old_snapshot"].name, destination.name,
                            src_dir_fd=old_fd, dst_dir_fd=new_fd, follow_symlinks=False)
                    observation["link_created"] = True
                finally:
                    os.close(old_fd); os.close(new_fd)
                after, alias = _identity(row["old_snapshot"]), _identity(destination)
                observation.update(after=after, alias_identity=alias)
                self._journal("LINK_CREATED", source=key, old_snapshot=str(row["old_snapshot"]),
                              destination=str(destination), after=after, alias_identity=alias,
                              expected_nlink_increment=1,
                              observed_nlink_increment=after["nlink"] - before["nlink"])
                stable = {k: v for k, v in before.items() if k not in {"nlink", "ctime_ns"}}
                if (alias != after or any(after[k] != v for k, v in stable.items())
                        or after["nlink"] != before["nlink"] + 1):
                    raise ValueError("retry hardlink changed unexpected old snapshot metadata")
                observation.update(status="SHARED_READONLY_SNAPSHOT_INODE", after=after,
                                   own_nlink_increment=1, ctime_change="EXPECTED_FROM_OWN_LINK")
                result = {"source": key, "source_stat": row["source_stat"],
                          "bytes": row["bytes"], "sha256": digest,
                          "snapshot_stat": _five(alias),
                          "snapshot_method": "SHARED_READONLY_SNAPSHOT_INODE"}
            self.completed.add(key)
            return result
        except BaseException as error:
            observation.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            try:
                self._journal("SNAPSHOT_FAILED", source=key, destination=str(destination),
                              link_created=observation["link_created"], error=str(error)[:500])
            except (OSError, ValueError) as journal_error:
                observation["journal_error"] = str(journal_error)
            raise

    def audit(self):
        """Freshly verify shared payloads and retain every failure without repair."""
        errors, checks = [], []
        for identity in self.metadata:
            try:
                _, current = _read_small(identity["path"], MAX_AIR_BYTES if identity["path"].endswith(".air")
                                          else MAX_METADATA_BYTES)
                if current != identity:
                    raise ValueError("bound evidence changed")
            except (OSError, ValueError) as error:
                errors.append({"path": identity["path"], "error": str(error)})
        attempts = {entry["source"]: entry for entry in self.evidence["snapshot_attempts"]}
        for source, row in self.rows.items():
            check = {"source": source, "status": "PASSED"}
            failures = []
            try:
                if _identity(source) != self.identities[source]["source"]:
                    failures.append("original staged source metadata changed")
            except (OSError, ValueError) as error:
                failures.append(str(error))
            try:
                attempt = attempts.get(source)
                expected = (attempt.get("after") if attempt and attempt.get("link_created")
                            else self.identities[source]["snapshot"])
                if attempt and attempt.get("link_created"):
                    digest, observed = _hash_file(row["old_snapshot"])
                    if (digest != row["sha256"] or observed != expected
                            or _identity(attempt["destination"]) != expected):
                        failures.append("shared payload SHA256/identity changed after retry")
                    check["payload_verification"] = ("FRESH_SHA256_MATCHED_AFTER_RETRY"
                                                       if digest == row["sha256"] else "FRESH_SHA256_DIFFERENT")
                    check["observed_sha256"] = digest
                elif _identity(row["old_snapshot"]) != expected:
                    failures.append("old snapshot metadata differs from own-link expectation")
                if attempt and attempt["status"] == "INDEPENDENT_AIR_SNAPSHOT":
                    digest, observed = _hash_file(attempt["destination"])
                    if digest != row["sha256"] or observed != attempt["after"]:
                        failures.append("independent AIR snapshot changed after retry")
                if not attempt or attempt["status"] == "FAILED":
                    failures.append("retry snapshot was not completed")
            except (OSError, ValueError) as error:
                failures.append(str(error))
            if failures:
                check.update(status="FAILED", error="; ".join(failures))
                errors.append(check.copy())
            checks.append(check)
        tool = self.evidence["tool_binding"]
        try:
            digest, current = _hash_file(tool["path"])
            if digest != tool["sha256"] or current != tool["identity"]:
                raise ValueError("selected OMG changed during retry")
        except (OSError, ValueError) as error:
            errors.append({"path": tool["path"], "error": str(error)})
        self.evidence.update(status="FAILED" if errors else "PASSED", audit_checks=checks,
                             audit_errors=errors, completed_inputs=len(self.completed))
        try:
            self._journal("AUDIT", status=self.evidence["status"], completed_inputs=len(self.completed),
                          failure_count=len(errors), first_errors=errors[:4])
            _, journal_identity = _read_small(self.journal, MAX_METADATA_BYTES)
            if journal_identity["stat"] != _five(self.journal_identity):
                raise ValueError("retry journal changed after final audit append")
            self.evidence["journal"].update(journal_identity)
        except (OSError, ValueError) as error:
            errors.append({"path": str(self.journal), "error": str(error)})
            self.evidence["status"] = "FAILED"
        return self.evidence


def _identity_directory(path):
    path, _ = _safe(path, directory=True)
    return {"dev": path.stat().st_dev}


def prepare_failed_prefill_reuse(failed_job, *, air, ddk_root, ddk_id, omg_path,
                                omg_sha256, tbe_diagnosis, input_layout, options,
                                selected, output):
    """Bind exactly a failed dual-gear target job to one new static64 attempt."""
    job, _ = _safe(failed_job, directory=True)
    compilation, _ = _safe(job / "compilation", directory=True)
    air, _ = _safe(air)
    ddk, _ = _safe(ddk_root, directory=True)
    omg, _ = _safe(omg_path)
    output = _new_path(output)
    if any(output == root or root in output.parents or output in root.parents
           for root in (job, air.parent, ddk)):
        raise ValueError("retry output must be separate from failed job, source and DDK trees")
    _safe(output, directory=True)
    metadata = []
    outer, outer_id = _json(job / "result.json", metadata)
    report, report_id = _json(compilation / "report.json", metadata)
    profile, _ = _json(job / "compile-profile.json", metadata)
    inventory, _ = _json(compilation / "input-inventory.json", metadata)
    command, _ = _json(compilation / "compile-command.json", metadata)
    frozen = get_existing_air_profile("cpro-40block-64-1")
    if (outer.get("artifact_kind") != "qwen35-original-target-job" or outer.get("status") != "FAILED"
            or outer.get("model_compilation") != "FAILED" or outer.get("air") != str(air)
            or outer.get("output") != str(job) or outer.get("compilation") != "compilation/report.json"
            or outer.get("compilation_report_sha256") != report_id["sha256"]
            or report.get("artifact_kind") != "qwen35-existing-air-compilation"
            or report.get("status") != "FAILED" or report.get("stage") != "omg"
            or report.get("model_compilation") != "FAILED"
            or type(report.get("returncode")) is not int or report["returncode"] == 0):
        raise ValueError("retry requires a hash-bound original target FAILED job with a real nonzero OMG exit")
    if (profile != frozen or report.get("requested_options") != _options(frozen, static=False)
            or options != _options(frozen, static=True)):
        raise ValueError("retry permits only the frozen dual64/1 to static-prefill64 option change")
    layout = report.get("input_layout", {})
    if (input_layout not in {"air-directory", "export-directory"} or layout.get("mode") != input_layout
            or outer.get("input_layout") != input_layout or layout.get("compiler_cwd") != "compile-work/generated"
            or report.get("output") != str(compilation) or report.get("air_source") != str(air)
            or report.get("ddk_root") != str(ddk) or report.get("ddk_id") != ddk_id
            or report.get("platform") != "kirinx90" or report.get("omg_path") != str(omg)
            or report.get("omg_sha256") != omg_sha256 or ddk not in omg.parents
            or not os.access(omg, os.X_OK)):
        raise ValueError("retry source/layout/DDK/OMG differs from failed job")
    digest, omg_identity = _hash_file(omg)
    if digest != omg_sha256:
        raise ValueError("selected OMG content identity changed")
    preservation = report.get("source_preservation", {})
    if (preservation.get("status") != "UNCHANGED_BY_STAT" or preservation.get("changed") != []
            or report.get("audit_error") or report.get("artifact_inventory_error")
            or report.get("skipped_nonregular_artifacts")):
        raise ValueError("failed compilation has incomplete source/artifact/compatibility evidence")
    compatibility = report.get("tbe_compatibility")
    if tbe_diagnosis is None:
        if compatibility is not None:
            raise ValueError("retry must retain the failed job's TBE diagnosis")
    else:
        diagnosis, _ = _safe(tbe_diagnosis, directory=True)
        tbe, _ = _json(compilation / "tbe-compatibility-result.json", metadata)
        if (not isinstance(compatibility, dict) or tbe != compatibility
                or compatibility.get("diagnosis") != str(diagnosis)
                or compatibility.get("compiler_invocation") != "ATTEMPTED"
                or compatibility.get("compile") != "FAILED"
                or compatibility.get("audit", {}).get("status") not in {"APPLIED", "NOT_OBSERVED"}):
            raise ValueError("retry TBE diagnosis/audit differs from the failed invocation")
        bound_diagnosis = compatibility.get("metadata", {}).get("diagnosis_report", {})
        _, diagnosis_id = _read_small(diagnosis / "report.json", MAX_METADATA_BYTES)
        if (bound_diagnosis.get("path") != diagnosis_id["path"]
                or bound_diagnosis.get("sha256") != diagnosis_id["sha256"]):
            raise ValueError("retry TBE diagnosis report SHA256 changed")
        metadata.append(diagnosis_id)
    snapshots = report.get("input_snapshots")
    if (not isinstance(snapshots, list) or not 1 <= len(snapshots) <= _MAX_FILES
            or inventory.get("files") != snapshots):
        raise ValueError("retry requires complete identical input inventories")
    chosen = []
    for path in selected:
        chosen.append(str(_safe(path)[0]))
        if len(chosen) > _MAX_FILES:
            raise ValueError("retry selected sources exceed the file count bound")
    if len(chosen) != len(set(chosen)):
        raise ValueError("retry selected sources contain duplicates")
    rows, identities, paths, inodes = {}, {}, set(), set()
    air_rows = []
    for row in snapshots:
        if (not isinstance(row, dict) or row.get("snapshot_method") != "COPY"
                or type(row.get("bytes")) is not int or row["bytes"] < 0
                or not isinstance(row.get("sha256"), str) or not _HASH.fullmatch(row["sha256"])
                or row.get("role") not in {"air", "neighbor_data_candidate"}):
            raise ValueError("retry accepts only complete COPY input snapshots with valid hashes")
        source, _ = _safe(row.get("source"))
        snapshot, _ = _safe(_relative(compilation, row.get("path")))
        key = str(source)
        expected_relative = source.relative_to(air.parent).as_posix()
        if (key in rows or row["path"] in paths or row.get("source_relative") != expected_relative
                or not row["path"].startswith("compile-work/generated/")
                or (source == air) != (row["role"] == "air")):
            raise ValueError("retry input source/path/role is duplicate or not bound to source layout")
        source_id, snapshot_id = _identity(source), _identity(snapshot)
        prefix = Path(air.parent.name) if input_layout == "export-directory" and source != air else Path()
        expected_path = (Path("compile-work/generated") / prefix / expected_relative).as_posix()
        if (_five(source_id) != row.get("source_stat") or _five(snapshot_id) != row.get("snapshot_stat")
                or source_id["bytes"] != row["bytes"] or snapshot_id["bytes"] != row["bytes"]
                or snapshot_id["mode"] & 0o222 or snapshot_id["nlink"] != 1
                or _inode(snapshot_id) == _inode(source_id) or _inode(snapshot_id) in inodes
                or row["path"] != expected_path):
            raise ValueError("retry snapshot/source stat changed or snapshot is writable/shared/not independent")
        rows[key] = dict(row, old_snapshot=snapshot)
        identities[key] = {"source": source_id, "snapshot": snapshot_id}
        paths.add(row["path"]); inodes.add(_inode(snapshot_id))
        if row["role"] == "air":
            air_rows.append(row)
            for path in (source, snapshot):
                _, identity = _read_small(path, MAX_AIR_BYTES)
                if identity["sha256"] != row["sha256"] or identity["bytes"] != row["bytes"]:
                    raise ValueError("retry original/snapshot AIR SHA256 differs from the failed job")
                metadata.append(identity)
    if len(air_rows) != 1 or set(rows) != set(chosen):
        raise ValueError("retry selected source set differs from failed input inventory or has no unique AIR")
    air_row = air_rows[0]
    if (layout.get("air_path") != air_row["path"]
            or report.get("air") != {"path": str(air), "bytes": air_row["bytes"], "sha256": air_row["sha256"]}):
        raise ValueError("retry AIR layout/content receipt is not exact")
    argv = [str(omg), "--framework=1", "--model=" + str(compilation / air_row["path"]),
            "--output=model", "--target=omc", "--platform=kirinx90",
            *(f"--{key}={value}" for key, value in _options(frozen, static=False).items())]
    # JSON map ordering is not an option semantic; require exactly the fixed
    # prefix plus one occurrence of every expected old option.
    actual = command.get("argv")
    if (not isinstance(actual, list) or actual[:6] != argv[:6] or len(actual) != len(argv)
            or sorted(actual[6:]) != sorted(argv[6:])
            or command.get("cwd") != str(compilation / "compile-work/generated")
            or command.get("weight_dtype_override") is not None):
        raise ValueError("retry failed compiler argv/cwd differs from frozen options")
    evidence = {"schema_version": 1, "artifact_kind": "qwen35-static-prefill-snapshot-reuse",
                "status": "PREPARED", "failed_job": str(job), "output": str(output),
                "original_air": str(air), "failed_result_sha256": outer_id["sha256"],
                "failed_report_sha256": report_id["sha256"], "input_count": len(rows),
                "tool_binding": {"path": str(omg), "sha256": omg_sha256, "identity": omg_identity},
                "old_options": _options(frozen, static=False), "new_options": dict(options),
                "ownership": "SHARED_READONLY_INODE", "explicit_weight_copy_bytes": 0,
                "snapshot_attempts": [], "metadata_bindings": metadata,
                "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
                "scope": "One static64 compiler retry only. Shared read-only inode content is verified before/after; privileged writes remain possible. Old receipts are unchanged; own links change old snapshot nlink/ctime. Physical allocation and compiler space are not inferred."}
    return _ReusePlan(output=output, rows=rows, identities=identities, metadata=metadata, evidence=evidence)

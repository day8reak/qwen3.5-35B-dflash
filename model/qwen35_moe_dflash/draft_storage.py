"""Compact verified, readonly compiler weight snapshots without changing bytes.

Prepared weights provide the SHA/extent whitelist. Only completed compiler
INPUT snapshots are linked to another readonly snapshot with matching access
metadata, retaining their independence from potentially writable source files.
Completed-stage manifests and all paths remain unchanged. Inode/ctime values
recorded during compilation describe that past invocation, not the compacted
storage layout; successful reuse checks content receipts rather than those
in-flight stat guards. Failed/incomplete compilations are never compacted.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from .draft_trace_integrity import _Reads
from .draft_workspace import DraftWorkspace


_LANES = ("layer-trace", "direct-trace", "residual-fix")
_MAX_SNAPSHOTS = 16384


def _path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("storage receipt escapes its artifact root")
    path = root / relative
    if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("storage paths must not traverse symlinks")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("storage payload must be a regular file")
    return path


def _record(row):
    if (not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] <= 0
            or not isinstance(row.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
        raise ValueError("invalid storage payload receipt")
    return row["bytes"], row["sha256"]


def _token(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("storage payload became a nonregular file")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
            stat.S_IMODE(s.st_mode), s.st_uid, s.st_gid)


def _access(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("storage access metadata requires a regular file")
    digest = hashlib.sha256()
    try:
        for key in sorted(os.listxattr(path, follow_symlinks=False)):
            value = os.getxattr(path, key, follow_symlinks=False)
            digest.update(key.encode() + b"\0" + len(value).to_bytes(8, "little") + value)
    except OSError as error:
        raise ValueError("cannot verify snapshot extended access metadata") from error
    return stat.S_IMODE(s.st_mode), s.st_uid, s.st_gid, digest.hexdigest()


def _hash_bound(path, row, cache):
    expected = _record(row)
    before = _token(path)
    if before[2] != expected[0]:
        raise ValueError("storage payload size differs: " + str(path))
    if before not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if before != _token(path) or digest.hexdigest() != expected[1]:
            raise ValueError("storage payload SHA256 or identity differs: " + str(path))
        cache[before] = expected[1]
    elif cache[before] != expected[1]:
        raise ValueError("ambiguous storage inode receipt")
    return before


def _inventory(job, read):
    source_raw = read.bytes(job / "result.json")
    source = json.loads(source_raw)
    if source.get("artifact_kind") != "qwen35-full-draft-job":
        raise ValueError("storage compaction requires the original full-draft job")
    component = source.get("components", {}).get("prepared", {})
    prepared_root = Path(component.get("directory", "")).absolute()
    if prepared_root != job / "prepared":
        raise ValueError("prepared weights must belong to this fixed draft workspace")
    prepared = json.loads(read.bound(prepared_root, component["manifest"]))
    if prepared.get("artifact_kind") != "qwen35-draft-ge-prepared" or prepared.get("status") != "PREPARED":
        raise ValueError("storage compaction requires completed prepared weights")
    whitelist = {}
    for row in prepared.get("weights", {}).values():
        if row.get("dtype") != "float16":
            raise ValueError("draft storage whitelist requires FP16 weights")
        identity = _record(row)
        path = _path(prepared_root, row["path"])
        if path.stat().st_size != identity[0]:
            raise ValueError("prepared storage extent differs")
        whitelist.setdefault(identity, (path, row))
    if not 0 < len(whitelist) <= 4096:
        raise ValueError("prepared storage whitelist is empty or too large")
    reports, skipped = [], []
    production = source.get("components", {}).get("compilation")
    if production:
        folder = Path(production["directory"]).absolute()
        if folder != job / "compilation":
            raise ValueError("production compilation must belong to this fixed workspace")
        reports.append(("production", folder, json.loads(read.bound(folder, production["manifest"]))))
    source_sha = hashlib.sha256(source_raw).hexdigest()
    for lane in _LANES:
        root = job / lane
        receipt_path = root / "completed-compilation.json"
        if not receipt_path.exists():
            skipped.append({"lane": lane, "reason": "NO_COMPLETED_COMPILATION"})
            continue
        owner = read.json(root / "owner.json")
        if (owner.get("artifact_kind") not in {"qwen35-draft-layer-trace-job-owner", "qwen35-draft-direct-trace-job-owner"}
                or owner.get("job") != str(job) or owner.get("source_job_sha256") != source_sha):
            raise ValueError("diagnostic compilation owner differs from the source job")
        receipt = read.json(receipt_path)
        if receipt.get("path") != "report.json":
            raise ValueError("unexpected completed compilation receipt")
        folder = root / "compilation"
        reports.append((lane, folder, json.loads(read.bound(folder, receipt))))
    snapshots, seen = [], set()
    for lane, folder, report in reports:
        if (report.get("artifact_kind") != "qwen35-existing-air-compilation"
                or report.get("status") != "COMPILED" or report.get("model_compilation") != "PASSED"):
            skipped.append({"lane": lane, "reason": "NOT_SUCCESSFULLY_COMPILED"})
            continue
        for row in report.get("input_snapshots", []):
            if row.get("role") != "neighbor_data_candidate" or _record(row) not in whitelist:
                continue
            relative = Path(row.get("path", ""))
            if relative.parts[:2] != ("compile-work", "generated"):
                raise ValueError("compiler input snapshot is outside its declared input tree")
            path = _path(folder, row["path"])
            if path in seen:
                raise ValueError("duplicate compiler input snapshot path")
            seen.add(path)
            if len(seen) > _MAX_SNAPSHOTS:
                raise ValueError("compiler snapshot count exceeds storage limit")
            if path.stat().st_size != row["bytes"]:
                raise ValueError("compiler input snapshot extent differs")
            snapshots.append({"lane": lane, "path": path, "record": row, "token": _token(path), "access": _access(path)})
    return whitelist, snapshots, skipped


def compact_draft_storage(job_dir, *, apply=False):
    """Print-only plan by default; explicit apply preserves payload bytes/paths.

    No chmod, source-weight links, report rewrites, compiler or board calls.
    Copy-on-write files can already share physical extents, so logical duplicate
    bytes are not a promise of reclaimed space. Filesystem free-space changes
    may also include unrelated concurrent writes.
    """
    job = Path(job_dir).absolute()
    if job.resolve() != job:
        raise ValueError("storage workspace must not traverse symlinks")
    report = {"schema_version": 1, "artifact_kind": "qwen35-draft-storage-compaction",
              "status": "PLANNED", "job": str(job), "apply": bool(apply),
              "identity_scope": "Recorded weight SHA/extent whitelist; apply rehashes all selected files before mutation",
              "changes": [], "skipped": [], "logical_duplicate_bytes": 0,
              "space_scope": "Logical bytes are not physical savings; reflinks and concurrent allocations can affect free-space change"}
    with DraftWorkspace(job) if apply else nullcontext():
        read = _Reads()
        whitelist, snapshots, skipped = _inventory(job, read)
        report["skipped"].extend(skipped)
        canonical, actions = {}, []
        for item in snapshots:
            path, record, access = item["path"], item["record"], item["access"]
            if access[0] & 0o222:
                report["skipped"].append({"path": str(path.relative_to(job)), "reason": "WRITABLE_SNAPSHOT"})
                continue
            key = (*_record(record), *access)
            if key not in canonical:
                canonical[key] = item
                continue
            first = canonical[key]
            if item["token"][:2] == first["token"][:2]:
                report["skipped"].append({"path": str(path.relative_to(job)), "reason": "ALREADY_SHARED_INODE"})
                continue
            if item["token"][0] != first["token"][0]:
                report["skipped"].append({"path": str(path.relative_to(job)), "reason": "DIFFERENT_FILESYSTEM"})
                continue
            if path.stat().st_nlink != 1:
                # Replacing one of several noncanonical hardlinks would not
                # release its blocks and would change the other aliases' ctime.
                report["skipped"].append({"path": str(path.relative_to(job)), "reason": "OTHER_SHARED_INODE"})
                continue
            actions.append((first, item))
            report["changes"].append({"canonical": str(first["path"].relative_to(job)), "path": str(path.relative_to(job)),
                "bytes": record["bytes"], "sha256": record["sha256"], "mode": oct(access[0]),
                "uid": access[1], "gid": access[2], "status": "PLANNED"})
            report["logical_duplicate_bytes"] += record["bytes"]
        read.unchanged()
        report["free_bytes_before"] = shutil.disk_usage(job).free
        if apply:
            cache = {}
            # Validate every planned pair and its prepared whitelist before any
            # replacement, so a bad later receipt cannot cause a partial apply.
            for first, item in actions:
                prepared, row = whitelist[_record(item["record"])]
                _hash_bound(prepared, row, cache)
                for selected in (first, item):
                    selected["token"] = _hash_bound(selected["path"], selected["record"], cache)
                    if _access(selected["path"]) != selected["access"]:
                        raise ValueError("snapshot access metadata changed")
            read.unchanged()
            for (first, item), change in zip(actions, report["changes"]):
                source, destination = first["path"], item["path"]
                if _token(source) != first["token"] or _token(destination) != item["token"]:
                    raise ValueError("snapshot changed before compaction")
                temporary = destination.with_name(".draft-weight-dedup-" + uuid.uuid4().hex)
                try:
                    os.link(source, temporary, follow_symlinks=False)
                    if _token(temporary)[:2] != first["token"][:2] or _token(destination) != item["token"]:
                        raise ValueError("snapshot identity changed before atomic replacement")
                    os.replace(temporary, destination)
                    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                    first["token"] = _token(source)
                    change.update(status="LINKED", before=list(item["token"]), after=list(_token(destination)))
                finally:
                    temporary.unlink(missing_ok=True)
            read.unchanged()
            report["status"] = "COMPACTED"
            report["identity_scope"] = "Every replaced snapshot and prepared whitelist weight SHA/extent verified before mutation; all access modes/uid/gid/xattrs preserved"
        report["free_bytes_after"] = shutil.disk_usage(job).free
        report["observed_free_byte_change"] = report["free_bytes_after"] - report["free_bytes_before"]
        report["compilations_considered"] = sorted({item["lane"] for item in snapshots})
    return report

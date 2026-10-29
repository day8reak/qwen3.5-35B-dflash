"""Original-target job continuation; never substitute the explicit-state ABI.

Compiler outputs remain in their original directory tree. A completed compile
can be inspected again without copying weights or invoking model compilation.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from .artifacts import output_path
from .existing_air import _conversion_options, _input_inventory, _stat
from .hashing import sha256_file


MAX_METADATA_BYTES = 32 << 20


def read_metadata(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("missing, symlinked or oversized target metadata: " + str(path))
    before = _stat(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES or _stat(path) != before:
        raise ValueError("target metadata changed or exceeded its read budget")
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate target metadata key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite target metadata value: " + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("target metadata must be an object")
    return result


def _file(root, relative):
    if not isinstance(relative, str):
        raise ValueError("target artifact requires a relative path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in {"", ".", ".."} for p in relative.split("/")):
        raise ValueError("invalid target artifact relative path: " + relative)
    path = root
    for part in parts.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("target artifact path contains a symlink: " + relative)
    if not path.is_file():
        raise ValueError("target artifact is missing: " + relative)
    return path


def _recorded_file(root, row, stat_key):
    if not isinstance(row, dict):
        raise ValueError("invalid target artifact record")
    path = _file(root, row.get("path"))
    before = _stat(path)
    digest = row.get("sha256")
    if (type(row.get("bytes")) is not int or before[2] != row["bytes"]
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("invalid target artifact size/hash: " + str(path))
    # Stat is an early rejection, never a substitute for content identity.
    # Some filesystems can retain the same timestamps across fast rewrites.
    if stat_key in row:
        if row[stat_key] != before:
            raise ValueError("target artifact changed since compilation: " + str(path))
    if sha256_file(path) != digest:
        raise ValueError("target artifact hash mismatch: " + str(path))
    if _stat(path) != before:
        raise ValueError("target artifact changed during validation: " + str(path))
    return path


def validate_compilation(job, *, air, ddk_root, ddk_id, profile, input_layout,
                         omg_path=None, tbe_diagnosis=None, target_features=False):
    """Reuse only a complete, unchanged original-target compiler invocation.

    This does not recover interrupted or failed compilation and never promotes
    a generated OMC found beside a failed report. Payload hashes are streamed
    again without copying weights; stat checks alone cannot establish identity.
    target_features selects a later board boundary check, not a compiler option;
    changing that selection never changes or bypasses these compile receipts.
    """
    if type(target_features) is not bool:
        raise ValueError("target_features must be an explicit boolean")
    job = Path(job).resolve()
    root = job / "compilation"
    result = read_metadata(job / "result.json")
    report = read_metadata(root / "report.json")
    if (result.get("artifact_kind") != "qwen35-original-target-job" or result.get("status") != "COMPILED"
            or report.get("artifact_kind") != "qwen35-existing-air-compilation"
            or report.get("status") != "COMPILED" or report.get("model_compilation") != "PASSED"
            or report.get("returncode") != 0):
        raise ValueError("resume requires a completed original target job; failed/interrupted jobs are preserved, never recompiled automatically")
    if (result.get("compilation_report_sha256") is not None
            and sha256_file(root / "report.json") != result["compilation_report_sha256"]):
        raise ValueError("target compilation report differs from completed job receipt")
    expected = _conversion_options(profile["input_shape"], profile["dynamic_dims"],
                                   profile["limit_partition_size"], profile["save_weights_as_external_data"])
    air, ddk = Path(air).resolve(), Path(ddk_root).resolve()
    if (report.get("air_source") != str(air) or report.get("ddk_root") != str(ddk)
            or report.get("ddk_id") != ddk_id or report.get("platform") != "kirinx90"
            or report.get("requested_options") != expected
            or report.get("input_layout", {}).get("mode") != input_layout
            or read_metadata(job / "compile-profile.json") != profile):
        raise ValueError("target resume AIR/DDK/profile/options/layout differs from completed compilation")
    if (report.get("audit_error") or report.get("artifact_inventory_error")
            or report.get("skipped_nonregular_artifacts")
            or report.get("source_preservation", {}).get("status") != "UNCHANGED_BY_STAT"
            or report["source_preservation"].get("changed")):
        raise ValueError("completed target has failed source/artifact/compatibility audit")
    omg = Path(report["omg_path"])
    if (ddk not in omg.resolve().parents or not os.access(omg, os.X_OK)
            or (omg_path is not None and omg.resolve() != Path(omg_path).resolve())
            or sha256_file(omg) != report.get("omg_sha256")):
        raise ValueError("target compiler identity changed")
    if tbe_diagnosis is not None and report.get("tbe_compatibility", {}).get("diagnosis") != str(Path(tbe_diagnosis).resolve()):
        raise ValueError("target resume TBE diagnosis differs from completed compilation")
    snapshots = report.get("input_snapshots")
    generated = report.get("generated_artifacts")
    if (not isinstance(snapshots, list) or not 1 <= len(snapshots) <= 4096
            or not isinstance(generated, list) or not 1 <= len(generated) <= 8192):
        raise ValueError("target resume requires complete source and generated inventories")
    selected, _ = _input_inventory(air)
    sources = set()
    for row in snapshots:
        source = Path(row["source"])
        if str(source) in sources or source not in selected or _stat(source) != row.get("source_stat"):
            raise ValueError("target source weights/AIR changed since compilation")
        before = _stat(source)
        if sha256_file(source) != row.get("sha256") or _stat(source) != before:
            raise ValueError("target source weights/AIR hash changed since compilation")
        sources.add(str(source))
        _recorded_file(root, row, "snapshot_stat")
    if sources != {str(p) for p in selected}:
        raise ValueError("target source inventory differs from completed compilation")
    air_rows = [r for r in snapshots if r.get("role") == "air"]
    if len(air_rows) != 1 or air_rows[0]["source"] != str(air):
        raise ValueError("target source inventory has no unique original AIR")
    paths = set()
    for row in generated:
        if row.get("path") in paths:
            raise ValueError("duplicate target generated artifact")
        paths.add(row.get("path"))
        _recorded_file(root, row, "stat")
    primary = report.get("omc")
    if not isinstance(primary, dict) or primary not in generated:
        raise ValueError("target primary OMC is not bound to generated inventory")
    observed = set()
    generated_root = root / "compile-work/generated"
    if any(p.is_symlink() for p in (root / "compile-work", generated_root)):
        raise ValueError("target generated directory contains a symlink")
    for folder, dirs, names in os.walk(generated_root, followlinks=False):
        for name in dirs + names:
            path = Path(folder) / name
            if path.is_symlink():
                raise ValueError("target generated directory contains a symlink")
        for name in names:
            path = Path(folder) / name
            if not path.is_file():
                raise ValueError("target generated directory contains a nonregular artifact")
            observed.add(path.relative_to(root).as_posix())
            if len(observed) > 16384:
                raise ValueError("target generated directory exceeds file count bound")
    staged = {r["path"] for r in snapshots}
    if observed != paths | staged:
        raise ValueError("target generated file set changed since compilation")
    command = read_metadata(root / "compile-command.json")
    staged_air = root / air_rows[0]["path"]
    argv = [str(omg), "--framework=1", "--model=" + str(staged_air), "--output=model",
            "--target=omc", "--platform=kirinx90", *(f"--{k}={v}" for k, v in expected.items())]
    if (command.get("argv") != argv
            or command.get("cwd") != str(root / report["input_layout"]["compiler_cwd"])
            or sha256_file(root / "compile.log") != report.get("compile_log_sha256")):
        raise ValueError("target compiler command/log changed")
    return report


def active_target_jobs(air, *, proc_root=Path("/proc")):
    """Find earlier CLI invocations for this exact original source on Linux.

    Other draft/OMG invocations are deliberately excluded. This is process
    observation, not a device probe or proof that an orphan compiler is absent.
    """
    air = str(Path(air).resolve())
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().split(b"\0")
            args = [part.decode(errors="replace") for part in parts if part]
            models = [a[len("--model="):] for a in args if a.startswith("--model=")]
            if models and "--platform=kirinx90" in args and any(Path(a).name == "omg" for a in args):
                cwd = (entry / "cwd").resolve()
                for value in models:
                    model = (cwd / value).resolve()
                    if str(model) == air:
                        found.append({"pid": int(entry.name), "air": air, "compiler_model": str(model)})
                        continue
                    if len(model.parents) < 3:
                        continue
                    inventory = model.parents[2] / "input-inventory.json"
                    if not inventory.is_file():
                        continue
                    rows = read_metadata(inventory).get("files", [])
                    if any(isinstance(r, dict) and r.get("role") == "air" and r.get("source") == air
                           and (inventory.parent / r.get("path", "")).resolve() == model for r in rows):
                        found.append({"pid": int(entry.name), "air": air, "compiler_model": str(model)})
            if not any(Path(arg).name == "compile_full_target.py" for arg in args):
                continue
            values = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "--air"]
            values += [a[len("--air="):] for a in args if a.startswith("--air=")]
            cwd = (entry / "cwd").resolve()
            if any(str((cwd / value).resolve()) == air for value in values):
                found.append({"pid": int(entry.name), "air": air})
        except (OSError, ValueError):
            continue
    return found


@contextmanager
def target_source_lock(air):
    """Serialize this AIR across job directories while allowing draft work."""
    key = hashlib.sha256(str(Path(air).resolve()).encode()).hexdigest()[:24]
    lock = output_path(Path(os.environ["AI_RUN_DIR"]) / (".original-target-" + key + ".lock"))
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("original target lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this original AIR already has an active target job") from error
        yield
    finally:
        os.close(fd)

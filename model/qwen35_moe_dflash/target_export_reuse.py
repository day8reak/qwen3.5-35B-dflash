"""Verify old target sidecars against a new export inventory before hardlinking.

Default operation only reads files and prints a plan. Applying creates one new
owned directory, links byte-verified sidecars on the same filesystem, and copies
only the small AIR/inventory. It never follows inventory source paths, changes
permissions, overwrites retained evidence, deletes files, or compiles a model.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+\Z")
_MAX_INVENTORY = 32 << 20
_MAX_AIR = 64 << 20
_COPY_RESERVE = 16 << 20
_CHUNK = 8 << 20
_RESERVED = {"export.air", "inventory.json", "reuse-receipt.json", "failed-reuse-receipt.json"}
_CORE = ("device", "inode", "bytes", "mtime_ns", "mode", "uid", "gid")


def _absolute(value):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("parent traversal is not supported")
    return path if path.is_absolute() else Path.cwd() / path


@contextmanager
def _directory(path, *, create=False):
    """Walk every component through directory fds without following symlinks."""
    path = _absolute(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, dir_fd=descriptor)
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _stamp(info):
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "mode": info.st_mode, "uid": info.st_uid, "gid": info.st_gid, "nlink": info.st_nlink}


def _named_stat(folder, name):
    info = os.stat(name, dir_fd=folder, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("source must be a direct regular non-symlink file: " + name)
    return _stamp(info)


def _same_directory(path, descriptor):
    with _directory(path) as current:
        before, after = os.fstat(descriptor), os.fstat(current)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("directory identity changed while inspecting/staging: " + str(path))


def _hash_file(folder, name, *, maximum=None, capture=False):
    before = _named_stat(folder, name)
    if maximum is not None and not 0 < before["bytes"] <= maximum:
        raise ValueError("file exceeds the bounded small-file limit: " + name)
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder)
    digest, chunks = hashlib.sha256(), []
    with os.fdopen(descriptor, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("source changed before hash: " + name)
        remaining = before["bytes"]
        while remaining:
            raw = stream.read(min(_CHUNK, remaining))
            if not raw:
                raise ValueError("source truncated while hashing: " + name)
            remaining -= len(raw)
            digest.update(raw)
            if capture:
                chunks.append(raw)
        if _stamp(os.fstat(stream.fileno())) != before or _named_stat(folder, name) != before:
            raise ValueError("source changed while hashing: " + name)
    return digest.hexdigest(), before, b"".join(chunks) if capture else None


def _read_small(path, maximum):
    with _directory(path.parent) as folder:
        digest, identity, raw = _hash_file(folder, path.name, maximum=maximum, capture=True)
    return raw, {"path": str(path), "bytes": len(raw), "sha256": digest, "stat": identity}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate inventory JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite inventory JSON constant is unsupported: " + value)


def _reference(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("sidecar reference must be an explicit relative POSIX path")
    normalized = value[2:] if value.startswith("./") else value
    parts = normalized.split("/")
    if (len(parts) != 2 or any(part in {"", ".", ".."} or not _COMPONENT.fullmatch(part) for part in parts)
            or parts[0] in _RESERVED):
        raise ValueError("only one relative sidecar directory and direct basename are supported: " + value)
    return parts[0], parts[1]


def _inventory(raw, *, sequence_length=64):
    if type(sequence_length) is not int or sequence_length not in {1, 64}:
        raise ValueError("inventory inspection requires an explicit sequence length of 1 or 64")
    manifest = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if (not isinstance(manifest, dict) or manifest.get("artifact_kind") != "qwen35-target-export-inventory"
            or manifest.get("status") != "INVENTORIED" or manifest.get("weight_identity") != "SHA256_COMPLETE"):
        raise ValueError("a complete SHA256 target-export inventory is required")
    air = manifest.get("air", {})
    if type(air.get("bytes")) is not int or not 0 < air["bytes"] <= _MAX_AIR or not _SHA.fullmatch(str(air.get("sha256", ""))):
        raise ValueError("inventory AIR size/SHA256 is missing or invalid")
    abi = manifest.get("declared_abi", {})
    expected = {"status": "MATCHED_INTERFACE_METADATA", "input_count": 89, "output_count": 82,
                "feature_output_index": 81, "feature_shape": [1, sequence_length, 16384], "feature_dtype": "DT_FLOAT16"}
    if any(json.dumps(abi.get(key), sort_keys=True) != json.dumps(value, sort_keys=True) for key, value in expected.items()):
        raise ValueError("inventory must declare the full original 89-input/82-output target and real feature interface")
    schema = manifest.get("schema", {})
    if not isinstance(schema, dict) or not _SHA.fullmatch(str(schema.get("descriptor_sha256", ""))):
        raise ValueError("inventory selected-schema identity is missing")
    files = manifest.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 8192:
        raise ValueError("inventory must contain a bounded nonempty sidecar list")
    names, prefixes, rows = set(), set(), []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("invalid sidecar inventory entry")
        prefix, name = _reference(item.get("reference"))
        if item.get("basename") != name or name in names:
            raise ValueError("duplicate/ambiguous sidecar basename or reference mapping")
        if "relative_path" in item and item["relative_path"] != prefix + "/" + name:
            raise ValueError("sidecar relative_path contradicts its reference")
        aliases = item.get("references", [item["reference"]])
        if (not isinstance(aliases, list) or not aliases or item["reference"] not in aliases
                or len(set(aliases)) != len(aliases) or any(_reference(alias) != (prefix, name) for alias in aliases)):
            raise ValueError("sidecar aliases must denote the same unambiguous relative file")
        size, digest = item.get("bytes"), item.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("every sidecar requires an exact byte size and SHA256")
        names.add(name)
        prefixes.add(prefix)
        rows.append({"reference": item["reference"], "references": aliases, "basename": name,
                     "relative_path": prefix + "/" + name, "bytes": size, "sha256": digest})
    if len(prefixes) != 1:
        raise ValueError("all sidecars must share one explicit relative directory prefix")
    prefix = next(iter(prefixes))
    if manifest.get("source_relative_prefix") != prefix:
        raise ValueError("inventory source_relative_prefix contradicts its file references")
    return manifest, rows, prefix


def _destination_parent(output):
    candidate = output.parent
    while True:
        try:
            with _directory(candidate) as folder:
                info, space = os.fstat(folder), os.fstatvfs(folder)
                return {"existing_parent": str(candidate), "device": info.st_dev,
                        "free_bytes": space.f_bavail * space.f_frsize}
        except FileNotFoundError:
            if candidate == candidate.parent:
                raise
            candidate = candidate.parent


def _write(folder, name, raw):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o644, dir_fd=folder)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def stage_target_export_reuse(inventory, air, existing_export, output, *, apply=False):
    """Plan/perform exact same-filesystem sidecar reuse; never copy a sidecar."""
    report = {"schema_version": 1, "artifact_kind": "qwen35-target-export-reuse",
              "status": "BLOCKED", "mode": "APPLY" if apply else "PLAN_ONLY",
              "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
              "numerical_validation": "NOT_EVALUATED", "replicated_weight_data_bytes": 0,
              "explicit_sidecar_copy_bytes": 0,
              "physical_weight_allocation": "NOT_INFERRED_HARDLINK_FILESYSTEM_COPY_UP_POSSIBLE",
              "required_compile_input_layout": "air-directory",
              "compiler_budget": "PENDING_INPUT_SNAPSHOT_AND_COMPILER_OUTPUT_BUDGET",
              "scope": "Exact sidecar byte reuse only; zero replicated bytes counts explicit tool copies, not physical allocation. Filesystem hardlinks may trigger copy-up. No model, feature correctness or later compilation-space pass."}
    created, output_fd = False, None
    try:
        if type(apply) is not bool:
            raise ValueError("apply must be an explicit boolean")
        inventory, air, existing, output = map(_absolute, (inventory, air, existing_export, output))
        run = os.environ.get("AI_RUN_DIR")
        if not run:
            raise ValueError("AI_RUN_DIR is required")
        run = _absolute(run)
        if output == run or run not in output.parents:
            raise ValueError("new staging output must be below AI_RUN_DIR")
        destination = _destination_parent(output)
        with _directory(destination["existing_parent"]):
            pass
        if os.path.lexists(output):
            raise ValueError("output already exists; preserve partial/retained evidence and choose a new output")
        destination["free_bytes_before"] = destination["free_bytes"]
        report.update(output=str(output), existing_export=str(existing), disk=destination)
        raw_inventory, inventory_identity = _read_small(inventory, _MAX_INVENTORY)
        manifest, rows, prefix = _inventory(raw_inventory)
        raw_air, air_identity = _read_small(air, _MAX_AIR)
        if any(air_identity[key] != manifest["air"][key] for key in ("bytes", "sha256")):
            raise ValueError("explicit AIR file does not match the inventory size/SHA256")
        report.update(inventory=inventory_identity, air=air_identity, source_relative_prefix=prefix,
                      declared_abi=manifest["declared_abi"], schema=manifest["schema"],
                      source_bytes=sum(row["bytes"] for row in rows), files=[], transfer_required=[])
        report_reserve = max(_COPY_RESERVE, 2 * len(raw_inventory) + len(rows) * 1024)
        required = len(raw_air) + len(raw_inventory) + report_reserve
        report["disk"].update(small_copy_required_bytes=required, reserve_bytes=report_reserve,
                              later_input_snapshot_bytes=report["source_bytes"], later_compiler_output_bytes="UNKNOWN")
        with _directory(existing) as source_fd:
            report["source_directory_stat"] = _stamp(os.fstat(source_fd))
            same_fs = os.fstat(source_fd).st_dev == destination["device"]
            report["same_filesystem"] = same_fs
            for planned in rows:
                row = dict(planned)
                try:
                    observed = _named_stat(source_fd, row["basename"])
                    row["source_stat"] = observed
                    if observed["bytes"] != row["bytes"]:
                        row.update(status="SIZE_MISMATCH", observed_bytes=observed["bytes"])
                    else:
                        digest, identity, _ = _hash_file(source_fd, row["basename"])
                        row.update(source_stat=identity, observed_sha256=digest,
                                   status="MATCHED" if digest == row["sha256"] else "HASH_MISMATCH")
                except FileNotFoundError:
                    row["status"] = "MISSING"
                except (OSError, ValueError) as error:
                    row.update(status="UNSAFE_OR_CHANGED_SOURCE", error=f"{type(error).__name__}: {error}")
                report["files"].append(row)
                if row["status"] != "MATCHED":
                    report["transfer_required"].append({key: row[key] for key in ("reference", "basename", "bytes", "sha256", "status")})
            for row in report["files"]:
                if row["status"] == "MATCHED":
                    try:
                        if _named_stat(source_fd, row["basename"]) != row["source_stat"]:
                            raise ValueError("source changed before the inventory scan completed")
                    except (OSError, ValueError) as error:
                        row.update(status="SOURCE_CHANGED", error=f"{type(error).__name__}: {error}")
                        report["transfer_required"].append({key: row[key] for key in ("reference", "basename", "bytes", "sha256", "status")})
            if report["transfer_required"]:
                report["status"] = "TRANSFER_REQUIRED"
                if apply:
                    report["error"] = "apply requires every sidecar to match; no output was created"
                return report
            if not same_fs:
                raise ValueError("cross-filesystem hardlinks are unavailable; no sidecar-copy fallback is permitted")
            if destination["free_bytes"] < required:
                raise ValueError("insufficient space for the small AIR/inventory copy plus reserve; no output was created")
            _same_directory(existing, source_fd)
            report["status"] = "PLAN_READY"
            if not apply:
                return report
            # Re-read small identities before creating anything. The captured
            # bytes, not an untrusted inventory path, are the actual staged data.
            if _read_small(inventory, _MAX_INVENTORY)[1] != inventory_identity or _read_small(air, _MAX_AIR)[1] != air_identity:
                raise ValueError("AIR/inventory changed after planning")
            if any(_named_stat(source_fd, row["basename"]) != row["source_stat"] for row in report["files"]):
                raise ValueError("source changed after planning; no output was created")
            _same_directory(existing, source_fd)
            with _directory(output.parent, create=True) as parent:
                if os.fstat(parent).st_dev != os.fstat(source_fd).st_dev:
                    raise ValueError("destination filesystem changed before apply")
                os.mkdir(output.name, dir_fd=parent)
                output_fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                created = True
            _write(output_fd, "export.air", raw_air)
            _write(output_fd, "inventory.json", raw_inventory)
            os.mkdir(prefix, dir_fd=output_fd)
            weights_fd = os.open(prefix, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=output_fd)
            try:
                inode_states = {}
                report["created_links"] = []
                for row in report["files"]:
                    initial = row["source_stat"]
                    key = (initial["device"], initial["inode"])
                    state = inode_states.setdefault(key, {"initial": initial, "current": initial, "created": 0})
                    before = _named_stat(source_fd, row["basename"])
                    if before != state["current"]:
                        raise ValueError("source identity changed before hardlink: " + row["basename"])
                    os.link(row["basename"], row["basename"], src_dir_fd=source_fd,
                            dst_dir_fd=weights_fd, follow_symlinks=False)
                    report["created_links"].append(row["relative_path"])
                    after = _named_stat(source_fd, row["basename"])
                    if (any(after[key] != before[key] for key in _CORE)
                            or after["nlink"] != before["nlink"] + 1 or after["ctime_ns"] < before["ctime_ns"]
                            or _named_stat(weights_fd, row["basename"]) != after):
                        raise ValueError("unexpected source mutation during hardlink: " + row["basename"])
                    state.update(current=after, created=state["created"] + 1)
                final_hashes = {}
                for row in report["files"]:
                    initial = row["source_stat"]
                    key = (initial["device"], initial["inode"])
                    state = inode_states[key]
                    current = _named_stat(source_fd, row["basename"])
                    if (current != state["current"] or any(current[key] != initial[key] for key in _CORE)
                            or current["nlink"] != initial["nlink"] + state["created"]):
                        raise ValueError("source changed after hardlink creation: " + row["basename"])
                    if key not in final_hashes:
                        final_hashes[key] = _hash_file(weights_fd, row["basename"])[0]
                    if final_hashes[key] != row["sha256"] or _named_stat(weights_fd, row["basename"]) != current:
                        raise ValueError("final hardlink payload SHA256/identity differs: " + row["basename"])
                    row.update(staged_stat=current, final_sha256=final_hashes[key],
                               own_hardlink_count=state["created"], status="HARDLINKED_AND_REHASHED")
                for row in report["files"]:
                    key = (row["source_stat"]["device"], row["source_stat"]["inode"])
                    if (_named_stat(source_fd, row["basename"]) != inode_states[key]["current"]
                            or _named_stat(weights_fd, row["basename"]) != inode_states[key]["current"]):
                        raise ValueError("source changed before final receipt: " + row["basename"])
                current_weights = os.open(prefix, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=output_fd)
                try:
                    current, expected = os.fstat(current_weights), os.fstat(weights_fd)
                    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
                        raise ValueError("staged sidecar directory identity changed")
                finally:
                    os.close(current_weights)
            finally:
                os.close(weights_fd)
            if _hash_file(output_fd, "export.air")[0] != air_identity["sha256"]:
                raise ValueError("staged AIR hash differs")
            _same_directory(existing, source_fd)
            _same_directory(output, output_fd)
            report.update(status="STAGED_HARDLINKS", copied_air_bytes=len(raw_air),
                          copied_inventory_bytes=len(raw_inventory), staged_air=str(output / "export.air"),
                          required_compile_input_layout="air-directory",
                          source_metadata_change="ONLY_OWN_HARDLINK_NLINK_INCREMENT_AND_CTIME_UPDATE_ALLOWED",
                          weight_payload_validation="INITIAL_AND_FINAL_STREAMED_SHA256_MATCHED")
            space = os.fstatvfs(output_fd)
            report["disk"].update(free_bytes_after=space.f_bavail * space.f_frsize,
                                  after_observation_point="AFTER_LINKS_FINAL_HASH_AND_SMALL_COPIES_BEFORE_RECEIPT_WRITE",
                                  delta_scope="Observed shared filesystem free space; concurrent usage and physical copy-up are not isolated")
            receipt_raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            _write(output_fd, "reuse-receipt.json", receipt_raw)
            report["receipt"] = {"path": str(output / "reuse-receipt.json"), "bytes": len(receipt_raw),
                                 "sha256": hashlib.sha256(receipt_raw).hexdigest()}
            return report
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        report.update(status="APPLY_FAILED_PARTIAL_PRESERVED" if created else "BLOCKED",
                      error=f"{type(error).__name__}: {str(error)[:1200]}")
        if created:
            report["partial_output_preserved"] = True
            try:
                space = os.fstatvfs(output_fd)
                report["disk"]["free_bytes_after"] = space.f_bavail * space.f_frsize
                _write(output_fd, "failed-reuse-receipt.json", (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
            except OSError:
                pass
        return report
    finally:
        if output_fd is not None:
            os.close(output_fd)

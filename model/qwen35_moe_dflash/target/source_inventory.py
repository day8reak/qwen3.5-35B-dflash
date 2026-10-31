"""Shared source inventory utilities used by model preparation and execution."""
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

_MAX_AIR = 64 << 20

_CHUNK = 8 << 20

_RESERVED = {"export.air", "inventory.json", "reuse-receipt.json", "failed-reuse-receipt.json"}

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

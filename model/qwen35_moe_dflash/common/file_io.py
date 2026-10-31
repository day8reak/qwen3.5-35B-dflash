"""Shared file io utilities used by model preparation and execution."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat

MAX_METADATA_BYTES = 32 << 20

MAX_AIR_BYTES = 64 << 20

def _stamp(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]

def _safe(path, *, directory=False):
    path = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:], 1):
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("symlink is unsupported: " + str(current))
        last = index == len(path.parts) - 1
        if not (stat.S_ISDIR(info.st_mode) if not last or directory else stat.S_ISREG(info.st_mode)):
            raise ValueError("expected regular file/directory: " + str(current))
    return path, _stamp(path.stat())

def _relative(root, text):
    if not isinstance(text, str) or "\\" in text or "\x00" in text:
        raise ValueError("artifact path must be explicit relative POSIX text")
    parts = PurePosixPath(text)
    if parts.is_absolute() or not text or any(piece in {"", ".", ".."} for piece in text.split("/")):
        raise ValueError("artifact path escapes the failed compilation")
    return root.joinpath(*parts.parts)

def _read_small(path, maximum):
    path, before = _safe(path)
    if not 0 < before[2] <= maximum:
        raise ValueError("small evidence file is empty or exceeds its byte budget: "
                         f"path={path}; bytes={before[2]}; maximum_bytes={maximum}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, chunks, remaining = hashlib.sha256(), [], before[2]
    with os.fdopen(fd, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("evidence identity changed before read")
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise ValueError("evidence truncated during read")
            remaining -= len(block); digest.update(block); chunks.append(block)
        if stream.read(1) or _stamp(os.fstat(stream.fileno())) != before or _safe(path)[1] != before:
            raise ValueError("evidence identity changed during read")
    return b"".join(chunks), {"path": str(path), "bytes": before[2], "sha256": digest.hexdigest(), "stat": before}

"""Shared embedding tables utilities used by model preparation and execution."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from qwen35_moe_dflash.common.file_io import _read_small, _relative, _safe, _stamp

_VOCAB, _HIDDEN, _PREFIX = 248320, 2048, 64

_TABLES = (("weight", "embedding_weight_path", _VOCAB * _HIDDEN, _HIDDEN, "INT8"),
           ("scale", "embedding_scale_path", _VOCAB * 4, 4, "FLOAT32"))

def _bound_bytes(path, expected, maximum):
    raw, observed = _read_small(path, maximum)
    if any(observed[key] != expected.get(key) for key in ("bytes", "sha256")):
        raise ValueError("original prefix input SHA256/extent changed: " + str(path))
    if "stat" in expected and observed["stat"] != expected["stat"]:
        raise ValueError("original prefix input identity changed: " + str(path))
    original = expected.get("identity")
    if original is not None and observed["stat"] != [original.get(key)
            for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")]:
        raise ValueError("original prefix input source-plan identity changed: " + str(path))
    return raw, observed

def _config(config_yaml, bindings):
    import yaml

    raw, identity = _read_small(config_yaml, 1 << 20)
    bindings.append(identity)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError("invalid original embedding YAML: " + str(error)) from error
    if not isinstance(value, dict):
        raise ValueError("original embedding YAML must be a mapping")
    paths = {}
    for _, field, _, _, _ in _TABLES:
        text = value.get(field)
        if not isinstance(text, str) or not text.strip() or "\0" in text:
            raise ValueError("original YAML requires " + field)
        path = Path(text).expanduser()
        paths[field] = Path(os.path.abspath(path if path.is_absolute() else Path(identity["path"]).parent / path))
    if paths[_TABLES[0][1]] == paths[_TABLES[1][1]]:
        raise ValueError("original weight and scale tables require distinct files")
    return paths, identity

def _write(root, relative, raw, *, dtype, shape):
    path = _relative(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": dtype, "shape": shape, "byte_order": "little"}

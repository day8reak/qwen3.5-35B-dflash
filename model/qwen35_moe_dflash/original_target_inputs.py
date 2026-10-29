"""Bounded, read-only inspection of the original target's control input files.

Shapes come from the frozen 89-port profile. Dtypes are the original exporter's
declarations, not an observation of a compiled model. These values are not an
incremental-input recipe: in particular conv_state_pos is left uninterpreted.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat
import struct

from .existing_air_profile import get_existing_air_profile


_PROFILE = "cpro-40block-64-1"
_DTYPE_SOURCE = "FROM_ORIGINAL_EXPORTER_NOT_COMPILED_ABI"
# Exact positional filename mapping used by original_target_board.py. Never
# visit the embedding, any RefData/state file, or any weight file.
_CONTROLS = (
    (83, "arg2_1", "new_kv_cache_pos", "int32", "i"),
    (84, "arg1238_1", "output_pos", "int32", "i"),
    (86, "arg114_1", "allQLen", "int64", "q"),
    (87, "arg14_1", "conv_state_pos", "int32", "i"),
    (88, "arg18_1", "effective_length", "int16", "h"),
    (2, "arg107_1", "position_ids", "int32", "i"),
    (85, "arg0_1", "embed_scale", "float32", "f"),
)
_MASK = (1, "arg113_1", "attention_mask", "float32", "f")
_CHUNK_BYTES = 64 * 1024


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
    return value


def _descriptor(entry, shapes, sequence_length):
    index, name, semantic, dtype, code = entry
    actual_name, declared_shape = shapes[index]
    if actual_name != name:
        raise ValueError("frozen original input mapping changed at port " + str(index))
    shape = [sequence_length if size == -1 else size for size in declared_shape]
    return {"index": index, "name": name, "semantic": semantic,
            "file": semantic + ".bin", "dtype": dtype, "byte_order": "little",
            "dtype_binding": _DTYPE_SOURCE, "shape": shape,
            "elements": math.prod(shape), "expected_bytes": math.prod(shape) * struct.calcsize("<" + code)}


def _mask_stats(chunks):
    result = {"elements": 0, "finite_count": 0, "nan_count": 0,
              "positive_infinity_count": 0, "negative_infinity_count": 0,
              "zero_count": 0, "finite_negative_count": 0, "finite_positive_count": 0,
              "finite_min": None, "finite_max": None}
    for chunk in chunks:
        for (value,) in struct.iter_unpack("<f", chunk):
            result["elements"] += 1
            if math.isnan(value):
                result["nan_count"] += 1
            elif math.isinf(value):
                result["positive_infinity_count" if value > 0 else "negative_infinity_count"] += 1
            else:
                result["finite_count"] += 1
                result["zero_count" if value == 0 else "finite_positive_count" if value > 0 else "finite_negative_count"] += 1
                result["finite_min"] = value if result["finite_min"] is None else min(value, result["finite_min"])
                result["finite_max"] = value if result["finite_max"] is None else max(value, result["finite_max"])
    return result


def _inspect_file(root, descriptor, code, *, mask=False):
    row = dict(descriptor)
    path = root / row["file"]
    row.update(status="UNAVAILABLE", bytes=None, sha256=None)
    try:
        before = path.lstat()
        row["bytes"] = before.st_size
        if not stat.S_ISREG(before.st_mode):
            row.update(status="NOT_READ", reason="SYMLINK_OR_NONREGULAR_FILE")
            return row
        if before.st_size != row["expected_bytes"]:
            row.update(status="NOT_READ", reason="SIZE_MISMATCH")
            return row
        digest = hashlib.sha256()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                raise ValueError("input changed before read")

            def chunks():
                remaining = row["expected_bytes"]
                while remaining:
                    chunk = stream.read(min(_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError("input was truncated during read")
                    if mask and len(chunk) % 4:
                        raise ValueError("mask read did not end on an FP32 element")
                    remaining -= len(chunk)
                    digest.update(chunk)
                    yield chunk

            if mask:
                row["statistics"] = _mask_stats(chunks())
                row["read_chunk_limit_bytes"] = _CHUNK_BYTES
            else:
                if row["elements"] > 128:
                    raise ValueError("control vector exceeds the 128-element inspection bound")
                raw = b"".join(chunks())
                row["values"] = [_json_value(values[0]) for values in struct.iter_unpack("<" + code, raw)]
                row["values_complete"] = True
            if _identity(os.fstat(stream.fileno())) != _identity(before) or _identity(path.lstat()) != _identity(before):
                raise ValueError("input changed during read")
        row.update(status="READ", sha256=digest.hexdigest(), source_stability="UNCHANGED_DURING_READ")
    except FileNotFoundError:
        row.update(status="MISSING", reason="INPUT_FILE_NOT_FOUND")
    except (OSError, ValueError) as error:
        row.update(status="UNAVAILABLE", reason=f"{type(error).__name__}: {error}")
    if row["status"] != "READ":
        # A partial or unstable observation must never look like a usable input.
        row.pop("values", None)
        row.pop("values_complete", None)
        row.pop("statistics", None)
        row["sha256"] = None
    return row


def inspect_original_target_inputs(input_dir, *, sequence_length=64, include_mask=True):
    """Inspect only seven small controls and optionally a bounded FP32 mask.

    Missing/malformed files are reported individually. No default values,
    semantic interpretation, files, model invocations or device calls are made.
    """
    if type(sequence_length) is not int or sequence_length not in (1, 64):
        raise ValueError("sequence_length must be 64 or 1")
    if type(include_mask) is not bool:
        raise ValueError("include_mask must be boolean")
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("original input directory does not exist: " + str(root))
    profile = get_existing_air_profile(_PROFILE)
    shapes = [(name, [int(size) for size in shape.split(",")])
              for name, shape in (value.split(":", 1) for value in profile["input_shape"].split(";"))]
    if len(shapes) != 89:
        raise ValueError("original target profile must retain 89 input ports")
    controls = [_inspect_file(root, _descriptor(entry, shapes, sequence_length), entry[4])
                for entry in _CONTROLS]
    mask = (_inspect_file(root, _descriptor(_MASK, shapes, sequence_length), _MASK[4], mask=True)
            if include_mask else {"status": "NOT_REQUESTED", "file": "attention_mask.bin"})
    complete = all(row["status"] == "READ" for row in controls) and (not include_mask or mask["status"] == "READ")
    return {"schema_version": 1, "artifact_kind": "qwen35-original-target-input-inspection",
            "status": "INSPECTED" if complete else "INCOMPLETE", "input_dir": str(root),
            "sequence_length": sequence_length, "profile": _PROFILE,
            "purpose": "NOT_A_CONTROL_GENERATOR", "device_execution": "NOT_EXECUTED",
            "numerical_validation": "NOT_EVALUATED", "dtype_binding": _DTYPE_SOURCE,
            "shape_binding": "FROZEN_USER_DECLARED_PROFILE_NOT_COMPILED_ABI",
            "control_semantics": "UNINTERPRETED_VALUES_CONV_STATE_POS_MEANING_NOT_ESTABLISHED",
            "controls": controls, "attention_mask": mask,
            "scope": "Read-only original control values and mask statistics; no embeddings, states or weights read; no generated controls or runtime validation."}

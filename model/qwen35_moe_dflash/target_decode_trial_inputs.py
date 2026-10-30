"""Own a small independent decode trial with an explicit unverified control.

An explicit mode may generate the absent convolution control alone or select
four controls for a single local query. Nothing here establishes the actual
kernel contract or a prefill/decode state recipe. Other inputs are copied
byte-for-byte; every original file stays unchanged.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import struct

from .artifacts import output_path, save_json
from .target_compile_failure import _read_small, _safe, _stamp
from .target_prefill_output_inventory import _json


_KIND = "qwen35-target-decode-trial-inputs"
_STATUS = "TARGET_DECODE_TRIAL_INPUTS_PREPARED"
_CONTROL = "conv_state_pos.bin"
_CHUNK = 64 << 10
_MAX_TOTAL = 8 << 20
_CONTROL_BYTES = struct.pack("<128i", 1, *([0] * 127))
_ASSUMPTION = {
    "id": "CONV_FIRST_WORD_CURRENT_QUERY_COUNT",
    "contract_status": "NOT_A_VALIDATED_KERNEL_CONTRACT",
    "current_query_count": 1,
    "remaining_words": "127_EXPERIMENTAL_ZEROS",
    "statement": "Experimental x3[0]=current query token count; no runtime contract is inferred from this choice",
    "numerical_reference": "NONE",
}


def _recipe(single_token):
    rows = [{"file": _CONTROL, "raw": _CONTROL_BYTES, "dtype": "INT32", "shape": [128],
        "values": [1] + [0] * 127,
        "reason": "Experimental hypothesis that convolution x3[0] is current query count, with127 experimental zeros"}]
    if single_token:
        rows.extend([
            {"file": "output_pos.bin", "raw": struct.pack("<i", 0), "dtype": "INT32", "shape": [1],
                "values": [0], "encoding": "<i", "reason": "Select local query row0 in the declared single-query graph"},
            {"file": "effective_length.bin", "raw": struct.pack("<h", 1), "dtype": "INT16", "shape": [1],
                "values": [1], "encoding": "<h", "reason": "Experimental current effective query count of one"},
            {"file": "allQLen.bin", "raw": struct.pack("<q", 1), "dtype": "INT64", "shape": [1],
                "values": [1], "encoding": "<q", "reason": "Experimental current Q length of one; actual decode kernel semantics remain unverified"},
        ])
    return rows


def _assumption(single_token, source, originals):
    if not single_token:
        return dict(_ASSUMPTION)
    previous = {row["file"]: row for row in originals}
    replacements = []
    for row in _recipe(True):
        original = previous.get(row["file"])
        values = None
        if original is not None:
            raw, identity = _read_small(source / row["file"], len(row["raw"]))
            if any(identity[key] != original[key] for key in ("bytes", "sha256", "stat")):
                raise ValueError("original single-token trial control changed during preparation")
            values = list(struct.unpack(row["encoding"], raw))
        replacements.append({"file": row["file"], "old_values": values,
            "old_sha256": original["sha256"] if original else None,
            "new_values": row["values"], "reason": row["reason"]})
    return {**_ASSUMPTION, "id": "SINGLE_TOKEN_QUERY_CONTROLS_WITH_EXPERIMENTAL_CONV",
        "statement": "Explicit single-local-query trial controls, not a validated incremental kernel recipe",
        "control_replacements": replacements}


def _missing_control(source):
    try:
        (source / _CONTROL).lstat()
    except FileNotFoundError:
        return
    raise ValueError("experimental preparation requires the original conv_state_pos.bin to be absent; existing files or links are never replaced")


def _preflight(input_dir):
    from .target_decode_smoke_plan import inspect_decode_input_extents

    source, info = _safe(input_dir, directory=True)
    _missing_control(source)
    observed = inspect_decode_input_extents(source)
    rows = observed.get("files", [])
    if (observed.get("profile") != "cpro-40block-decode-1" or len(rows) != 13
            or len({row.get("file") for row in rows}) != 13):
        raise ValueError("expected the frozen13-file decode1 input extent inspection")
    controls = [row for row in rows if row.get("file") == _CONTROL]
    if (len(controls) != 1 or controls[0].get("status") != "MISSING"
            or controls[0].get("expected_bytes") != len(_CONTROL_BYTES)):
        raise ValueError("only the absent512-byte conv_state_pos.bin can be generated experimentally")
    sources = []
    for row in rows:
        name = row.get("file")
        if name == _CONTROL:
            continue
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.bin", name)
                or row.get("path") != str(source / name)
                or row.get("status") != "REGULAR_EXTENT_MATCHED"
                or type(row.get("expected_bytes")) is not int
                or not 0 < row["expected_bytes"] <= _MAX_TOTAL):
            raise ValueError("all other supplied decode files must be regular and have their exact declared extent: " + str(name))
        path, before = _safe(source / name)
        if before[2] != row["expected_bytes"]:
            raise ValueError("supplied decode extent changed before hashing: " + name)
        sources.append({"path": str(path), "file": name, "bytes": before[2], "stat": before})
    if len(sources) != 12 or sum(row["bytes"] for row in sources) > _MAX_TOTAL:
        raise ValueError("experimental decode copies must contain only12 bounded original input files")
    return source, {"device": info[0], "inode": info[1]}, sorted(sources, key=lambda row: row["file"])


def _hash(path, *, expected_bytes, expected_stat=None, destination=None):
    """Hash a bounded regular file, optionally copying with exclusive creation."""
    path, before = _safe(path)
    if (before[2] != expected_bytes or not 0 < expected_bytes <= _MAX_TOTAL
            or (expected_stat is not None and before != expected_stat)):
        raise ValueError("decode trial source/payload extent or identity changed: " + str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, count = hashlib.sha256(), 0
    with os.fdopen(descriptor, "rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            raise ValueError("decode trial source changed before read")
        target = destination.open("xb") if destination is not None else None
        try:
            while block := stream.read(_CHUNK):
                count += len(block)
                if count > expected_bytes:
                    raise ValueError("decode trial source grew during read")
                digest.update(block)
                if target is not None:
                    target.write(block)
            if target is not None:
                target.flush()
            if (count != expected_bytes or _stamp(os.fstat(stream.fileno())) != before
                    or _safe(path)[1] != before):
                raise ValueError("decode trial source changed during read")
        finally:
            if target is not None:
                target.close()
    return {"path": str(path), "bytes": expected_bytes, "sha256": digest.hexdigest(), "stat": before}


def _source_rows(source, directory_identity, expected):
    current_source, current_identity, required = _preflight(source)
    if current_source != source or current_identity != directory_identity:
        raise ValueError("original decode input directory identity changed")
    if [{key: row[key] for key in ("file", "path", "bytes", "stat")} for row in required] != [
            {key: row[key] for key in ("file", "path", "bytes", "stat")} for row in expected]:
        raise ValueError("required original decode input identities changed")
    rows = []
    for row in expected:
        actual = _hash(row["path"], expected_bytes=row["bytes"], expected_stat=row["stat"])
        if row.get("sha256") is not None and actual["sha256"] != row["sha256"]:
            raise ValueError("required original decode input SHA256 changed")
        rows.append({**actual, "file": row["file"]})
    return rows


def prepare_decode_trial_inputs(input_dir, output_dir, *, experimental_conv_current_query_count=False,
                               experimental_single_token_controls=False):
    """Create a new owned input view only after an explicit hypothesis choice."""
    if type(experimental_single_token_controls) is not bool or type(experimental_conv_current_query_count) is not bool:
        raise ValueError("experimental input recipe flags must be explicit booleans")
    if experimental_conv_current_query_count is not True and experimental_single_token_controls is not True:
        raise ValueError("explicit experimental_conv_current_query_count=True or experimental_single_token_controls=True is required; no automatic missing-control fallback")
    source, directory_identity, requirements = _preflight(input_dir)
    out = output_path(output_dir)
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("experimental output must be separate from the original input directory")
    if out.exists() or out.is_symlink():
        raise FileExistsError("experimental output already exists: " + str(out))
    originals = _source_rows(source, directory_identity, requirements)
    recipe = _recipe(experimental_single_token_controls)
    assumption = _assumption(experimental_single_token_controls, source, originals)
    replaced = {row["file"] for row in recipe}
    # Complete every original-size/hash check before producing any new file.
    _missing_control(source)
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    copied = []
    for row in originals:
        if row["file"] in replaced:
            continue
        _safe(inputs, directory=True)
        destination = inputs / row["file"]
        copied_source = _hash(row["path"], expected_bytes=row["bytes"], expected_stat=row["stat"],
            destination=destination)
        if copied_source["sha256"] != row["sha256"]:
            raise ValueError("original decode bytes changed between hash and copy")
        current = _hash(destination, expected_bytes=row["bytes"])
        if current["sha256"] != row["sha256"]:
            raise ValueError("experimental copied input SHA256 differs from original")
        copied.append({**current, "path": "inputs/" + row["file"], "file": row["file"], "source": row["path"]})
    generated_controls = []
    for selected in recipe:
        _safe(inputs, directory=True)
        control = inputs / selected["file"]
        with control.open("xb") as stream:
            stream.write(selected["raw"])
        generated = _hash(control, expected_bytes=len(selected["raw"]))
        generated.update(path="inputs/" + selected["file"], file=selected["file"],
            dtype=selected["dtype"], shape=selected["shape"], byte_order="little",
            values=selected["values"], source="EXPLICIT_EXPERIMENTAL_HYPOTHESIS")
        generated_controls.append(generated)
    _source_rows(source, directory_identity, originals)
    manifest = {
        "schema_version": 1, "artifact_kind": _KIND, "status": _STATUS, "output": str(out),
        "input_dir": str(inputs), "source_input_dir": str(source), "source_directory_identity": directory_identity,
        "profile": "cpro-40block-decode-1", "experimental_conv_current_query_count": experimental_conv_current_query_count is True,
        "experimental_single_token_controls": experimental_single_token_controls,
        "assumption": assumption, "source_bindings": originals, "copied_inputs": copied,
        "generated_control": generated_controls[0], "generated_controls": generated_controls,
        "explicit_copy_bytes": sum(row["bytes"] for row in copied),
        "generated_control_bytes": sum(len(row["raw"]) for row in recipe),
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "numerical_validation": "NOT_RUN_NO_REFERENCE", "conv_control_semantics": "UNVERIFIED_EXPERIMENTAL_HYPOTHESIS",
        "state_continuity": "NOT_VALIDATED", "prefill_state_feedback": "NOT_RUN",
        "execution_scope": "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_WITH_EXPERIMENTAL_CONTROL",
        "source_preservation": "UNCHANGED_REQUIRED_ORIGINAL_INPUT_SHA256_AND_FILE_IDENTITIES",
        "source_control": "ABSENT_ORIGINAL_NOT_MODIFIED", "original_extra_files": "NOT_OPENED_OR_COPIED",
        "model_payload_access": "NONE", "weight_payload_access": "NONE", "links_created": 0,
        "scope": f"Owned byte-exact copies of{len(copied)} required inputs plus {len(recipe)} explicitly experimental controls. This is not a validated kernel input contract, a numerical reference or a continuation recipe.",
    }
    save_json(out / "manifest.json", manifest)
    validate_decode_trial_inputs(out)
    return manifest


def validate_decode_trial_inputs(prepared_dir):
    """Read-only validation before/after using an explicitly selected trial."""
    root, _ = _safe(prepared_dir, directory=True)
    bindings = []
    manifest = _json(root / "manifest.json", bindings)
    single_token = manifest.get("experimental_single_token_controls")
    if type(single_token) is not bool:
        raise ValueError("experimental single-token mode must remain an explicit boolean")
    if (manifest.get("schema_version") != 1 or manifest.get("artifact_kind") != _KIND
            or manifest.get("status") != _STATUS or manifest.get("output") != str(root)
            or manifest.get("input_dir") != str(root / "inputs")
            or type(manifest.get("experimental_conv_current_query_count")) is not bool
            or (manifest.get("experimental_conv_current_query_count") is not True and single_token is not True)
            or manifest.get("profile") != "cpro-40block-decode-1"
            or manifest.get("conv_control_semantics") != "UNVERIFIED_EXPERIMENTAL_HYPOTHESIS"
            or manifest.get("execution_scope") != "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_WITH_EXPERIMENTAL_CONTROL"
            or manifest.get("source_control") != "ABSENT_ORIGINAL_NOT_MODIFIED"
            or manifest.get("prefill_state_feedback") != "NOT_RUN"
            or manifest.get("model_compilation") != "NOT_RUN" or manifest.get("device_execution") != "NOT_RUN"
            or manifest.get("original_extra_files") != "NOT_OPENED_OR_COPIED"
            or manifest.get("model_payload_access") != "NONE" or manifest.get("weight_payload_access") != "NONE"
            or manifest.get("links_created") != 0):
        raise ValueError("expected an explicit untouched independent decode experimental input manifest")
    source, _ = _safe(manifest["source_input_dir"], directory=True)
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("experimental input view overlaps its original source directory")
    inputs, _ = _safe(root / "inputs", directory=True)
    originals = manifest.get("source_bindings", [])
    if not isinstance(originals, list) or len(originals) != 12:
        raise ValueError("experimental input manifest requires12 complete source bindings")
    observed = _source_rows(source, manifest["source_directory_identity"], originals)
    if observed != originals:
        raise ValueError("original source binding inventory changed")
    recipe = _recipe(single_token)
    if manifest.get("assumption") != _assumption(single_token, source, originals):
        raise ValueError("experimental control assumptions differ from the current original values and explicit recipe")
    replaced = {row["file"] for row in recipe}
    expected_copies = [row for row in originals if row["file"] not in replaced]
    copies = manifest.get("copied_inputs", [])
    if not isinstance(copies, list) or len(copies) != len(expected_copies):
        raise ValueError("experimental input manifest copied-file inventory differs from its explicit recipe")
    if {path.name for path in inputs.iterdir()} != {row["file"] for row in originals} | {_CONTROL}:
        raise ValueError("experimental inputs contain unlisted or missing files")
    for row, original in zip(copies, expected_copies):
        if (row.get("path") != "inputs/" + original["file"] or row.get("file") != original["file"]
                or row.get("source") != original["path"] or row.get("bytes") != original["bytes"]
                or row.get("sha256") != original["sha256"] or not isinstance(row.get("stat"), list)
                or len(row["stat"]) != 5 or any(type(value) is not int for value in row["stat"])):
            raise ValueError("experimental copied input does not match its exact original source")
        path = inputs / row["file"]
        actual = _hash(path, expected_bytes=row["bytes"], expected_stat=row["stat"])
        info = path.stat()
        if (actual["sha256"] != row["sha256"] or info.st_nlink != 1
                or (info.st_dev, info.st_ino) == tuple(original["stat"][:2])):
            raise ValueError("experimental copied bytes changed or are linked to another file")
    generated_controls = manifest.get("generated_controls", [])
    if (not isinstance(generated_controls, list) or len(generated_controls) != len(recipe)
            or manifest.get("generated_control") != generated_controls[0]):
        raise ValueError("experimental generated-control inventory differs from its explicit recipe")
    for generated, selected in zip(generated_controls, recipe):
        if (generated.get("path") != "inputs/" + selected["file"] or generated.get("file") != selected["file"]
                or generated.get("dtype") != selected["dtype"] or generated.get("shape") != selected["shape"]
                or generated.get("byte_order") != "little" or generated.get("values") != selected["values"]
                or generated.get("source") != "EXPLICIT_EXPERIMENTAL_HYPOTHESIS"
                or generated.get("bytes") != len(selected["raw"])
                or generated.get("sha256") != hashlib.sha256(selected["raw"]).hexdigest()):
            raise ValueError("generated control differs from its explicit experimental single-query recipe")
        raw, identity = _read_small(inputs / selected["file"], len(selected["raw"]))
        if (raw != selected["raw"] or identity["stat"] != generated.get("stat")
                or (inputs / selected["file"]).stat().st_nlink != 1):
            raise ValueError("experimental control bytes or identity changed")
    if (manifest.get("explicit_copy_bytes") != sum(row["bytes"] for row in copies)
            or manifest.get("generated_control_bytes") != sum(len(row["raw"]) for row in recipe)
            or manifest.get("source_preservation") != "UNCHANGED_REQUIRED_ORIGINAL_INPUT_SHA256_AND_FILE_IDENTITIES"
            or manifest.get("numerical_validation") != "NOT_RUN_NO_REFERENCE"
            or manifest.get("state_continuity") != "NOT_VALIDATED"):
        raise ValueError("experimental preservation or execution-scope declarations changed")
    if _read_small(root / "manifest.json", 8 << 20)[1] != bindings[0]:
        raise ValueError("experimental manifest changed during validation")
    return manifest

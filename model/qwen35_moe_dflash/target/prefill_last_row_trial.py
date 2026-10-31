"""Prepare one independent source-Python last-row prefill experiment.

The old successful input/output receipt is validated read-only. Only the new
output_pos file changes; original inputs and receipts remain untouched. This
does not define private IndexByTensor bounds or explain numerical output.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.target.original_target_board import original_input_files
from qwen35_moe_dflash.common.file_io import _read_small, _safe
from qwen35_moe_dflash.target.prefill_decode_bridge import inspect_prefill_decode_bridge
from qwen35_moe_dflash.target.prefill_output_inventory import _json
from qwen35_moe_dflash.target.prefill_smoke_plan import _identity, _source


_KIND = "qwen35-target-prefill-last-row-trial-inputs"
_STATUS = "TARGET_PREFILL_LAST_ROW_TRIAL_INPUTS_PREPARED"
_CONTROL = "output_pos.bin"
_MAX_TOTAL = 16 << 20
_SOURCE_RULE = {
    "source_manifest_sha256": "9e731ea3ceb6a6eb6a9b011bd8e12dbac7d46f399bb704818921601b075557c5",
    "wrapper": "models/export_model_wrapper_qwen3_5_moe.py",
    "wrapper_sha256": "fbd152783ac93d1fb0e525a24eb1d73db3ab2fbaa100cd7806d8eda890991e97",
    "source_lines": [383, 385], "source_read_this_tool": "NOT_REQUESTED_PREVIOUSLY_REVIEWED_SNAPSHOT_DECLARATION",
    "observation": "Original Cpro Python selects the current-call sequence row before norm and LM head",
    "rule": "CURRENT_CALL_LAST_ROW_ZERO_BASED_SEQUENCE_LENGTH_MINUS_ONE",
    "sequence_length": 64, "trial_output_pos": 63, "source_to_current_air_revision": "UNVERIFIED",
    "selected_kernel_index_semantics": "UNVERIFIED", "old_index_out_of_range": "NOT_CLAIMED",
    "numerical_root_cause": "NOT_INFERRED"}


def _stable(snapshots):
    for name, expected in snapshots.items():
        path, _ = _safe(name)
        if _identity(path.stat()) != expected:
            raise ValueError("prefill trial source or derived input changed within this preparation: " + name)


def _pin(snapshots, path, expected=None):
    path, stamp = _safe(path)
    current = _identity(path.stat())
    if [current[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")] != stamp:
        raise ValueError("prefill trial source changed while its active identity was captured")
    if expected is not None and current != expected:
        raise ValueError("prefill trial source changed after its validated read: " + str(path))
    if snapshots.setdefault(str(path), current) != current:
        raise ValueError("prefill trial source was replaced within this preparation: " + str(path))
    return path, current


def _copy(source, destination, expected, replacement=None):
    """Copy one independently owned file under current full-identity checks."""
    path, current = _pin({}, source, expected["identity"])
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, count = hashlib.sha256(), 0
    with os.fdopen(descriptor, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != current:
            raise ValueError("original prefill input changed before copy")
        target_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(target_fd, "wb") as target:
            while block := stream.read(64 << 10):
                count += len(block)
                if count > expected["bytes"]:
                    raise ValueError("original prefill input grew during copy")
                digest.update(block)
                if replacement is None:
                    target.write(block)
            if replacement is not None:
                target.write(replacement)
            target.flush()
        if (count != expected["bytes"] or digest.hexdigest() != expected["sha256"]
                or _identity(os.fstat(stream.fileno())) != current):
            raise ValueError("original prefill input bytes or identity changed during copy")
    _pin({}, path, current)
    copied = _source(destination, expected["bytes"])
    expected_sha = expected["sha256"] if replacement is None else hashlib.sha256(replacement).hexdigest()
    if (copied["sha256"] != expected_sha or copied["identity"]["nlink"] != 1
            or (copied["identity"]["device"], copied["identity"]["inode"]) == (current["device"], current["inode"])):
        raise ValueError("prefill trial copy is not an independent exact derived file")
    return copied


def prepare_prefill_last_row_trial(execution, named_runtime_probe, output):
    """Copy 13 target sources and one auxiliary ID file; change only one word."""
    execution, _ = _safe(execution, directory=True)
    named_runtime_probe, _ = _safe(named_runtime_probe, directory=True)
    out = output_path(output)
    initial_bindings, snapshots, derived = [], {}, {}
    initial_plan = _json(execution / "plan.json", initial_bindings)
    _, initial_plan_id = _pin(snapshots, execution / "plan.json")
    if [initial_plan_id[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")] != initial_bindings[-1]["stat"]:
        raise ValueError("retained prefill plan changed before trial output selection")
    initial_root, _ = _safe(initial_plan["input_dir"], directory=True)
    if any(out == source or source in out.parents or out in source.parents for source in
           (execution, named_runtime_probe, initial_root)):
        raise ValueError("trial output must be separate from retained evidence and original inputs")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": _KIND, "status": "BLOCKED", "stage": "validate-retained-prefill",
        "output": str(out), "execution": str(execution), "named_runtime_probe": str(named_runtime_probe),
        "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN", "model_payload_access": "NONE",
        "weight_payload_access": "NONE", "numerical_validation": "NOT_RUN", "state_feedback": "NOT_RUN",
        "source_python_last_row_trial": dict(_SOURCE_RULE), "original_mutation": "NONE", "source_bindings": [],
        "derived_inputs": []}
    published = False
    try:
        proof = inspect_prefill_decode_bridge(execution, named_runtime_probe, out / "source-validation")
        if (proof.get("status") != "PREFILL_NAMED_SLOTS_AND_RETAINED_FILES_BOUND_BRIDGE_GAPS_REMAIN"
                or proof.get("input_count") != 89 or proof.get("output_count") != 82
                or proof.get("executed_input_byte_binding") != "ALL89_SAVED_PRIVATE_SLOT_SHA256_MATCHED_CURRENT13_ORIGINAL_SOURCES"
                or proof.get("named_probe_witness", {}).get("status") != "POST_HOC_SAME_MODEL_ORDERED_DESCRIPTOR_NAMES_BOUND"):
            raise ValueError("last-row trial requires the successful retained prefill and its same-model named metadata witness")
        for row in proof.get("active_source_identities", []):
            _pin(snapshots, row["path"], row["identity"])
        if not snapshots:
            raise ValueError("retained prefill validation did not expose current active source identities")
        sources = proof.get("original_source_inputs")
        required = {filename for _, filename in original_input_files()}
        if (not isinstance(sources, list) or len(sources) != 13
                or {Path(row["path"]).name for row in sources} != required):
            raise ValueError("last-row trial requires exactly the 13 validated original target source files")
        bindings = []
        plan = _json(execution / "plan.json", bindings)
        for row in bindings:
            path, current = _pin(snapshots, row["path"])
            if [current[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")] != row["stat"]:
                raise ValueError("retained prefill plan changed after its current read")
        source_root, source_dir = _safe(plan["input_dir"], directory=True)
        if (out == source_root or source_root in out.parents or out in source_root.parents
                or any(Path(row["path"]).parent != source_root for row in sources)):
            raise ValueError("trial must use the exact original input directory and a separate output")
        for row in sources:
            _pin(snapshots, row["path"], row["identity"])
        if sum(row["bytes"] for row in sources) + 256 > _MAX_TOTAL:
            raise ValueError("prefill trial source files exceed the 16 MiB copy budget")
        result["stage"] = "bind-auxiliary-original-ids"
        ids = _source(source_root / "input_ids.bin", 256)
        _pin(snapshots, ids["path"], ids["identity"])
        ids_raw, ids_id = _read_small(ids["path"], 256)
        if ids_id["sha256"] != ids["sha256"]:
            raise ValueError("auxiliary original input IDs changed after hashing")
        _pin(snapshots, ids["path"], ids["identity"])
        control = next(row for row in sources if Path(row["path"]).name == _CONTROL)
        control_raw, control_id = _read_small(control["path"], 4)
        if control_id["sha256"] != control["sha256"] or control["bytes"] != 4:
            raise ValueError("original output_pos is not its validated INT32 word")
        old_value, = struct.unpack("<i", control_raw)
        if old_value == 63:
            raise ValueError("original output_pos already selects row 63; no distinct trial is needed")
        _stable(snapshots)
        result["stage"] = "copy-independent-trial-inputs"
        inputs = out / "inputs"; inputs.mkdir()
        input_dir_id = _safe(inputs, directory=True)[1][:2]
        for source in [*sources, ids]:
            _safe(inputs, directory=True)
            filename = Path(source["path"]).name
            copied = _copy(source["path"], inputs / filename, source,
                replacement=struct.pack("<i", 63) if filename == _CONTROL else None)
            _pin(derived, copied["path"], copied["identity"])
            result["derived_inputs"].append({**copied, "file": filename, "source": source,
                "role": "AUXILIARY_INPUT_IDS_NOT_A_TARGET_MODEL_INPUT" if filename == "input_ids.bin" else "TARGET_SOURCE_FILE",
                "transformation": "REPLACE_SINGLE_INT32_WORD_WITH_SOURCE_PYTHON_LAST_ROW63" if filename == _CONTROL else "BYTE_IDENTICAL_INDEPENDENT_COPY"})
        _stable(snapshots); _stable(derived)
        if (_safe(source_root, directory=True)[1][:2] != source_dir[:2]
                or _safe(inputs, directory=True)[1][:2] != input_dir_id):
            raise ValueError("original or derived input directory was replaced")
        validation = _source(out / "source-validation/report.json", (out / "source-validation/report.json").stat().st_size)
        _pin(snapshots, validation["path"], validation["identity"])
        result.update(status=_STATUS, stage="complete", input_dir=str(inputs), source_input_dir=str(source_root),
            target_input_count=89, target_output_count=82, target_source_file_count=13, auxiliary_source_file_count=1,
            derived_file_count=14, changed_target_source_file_count=1, byte_identical_target_source_file_count=12,
            original_input_ids={**ids, "values": list(struct.unpack("<64i", ids_raw)),
                "scope": "Auxiliary original IDs copied unchanged; not a 90th target input or proof of token-to-embedding equivalence"},
            control_replacement={"file": _CONTROL, "dtype": "INT32", "byte_order": "little", "shape": [1],
                "old_values": [old_value], "new_values": [63], "old_sha256": control["sha256"],
                "new_sha256": hashlib.sha256(struct.pack("<i", 63)).hexdigest()},
            source_validation=validation, compilation=plan["compilation"], model_sha256=plan["model_sha256"],
            compilation_report_sha256=plan["compilation_report_sha256"],
            source_preservation="CURRENT_FULL_IDENTITIES_AND_SOURCE_SHA256_UNCHANGED; ORIGINAL_RECEIPTS_NOT_REWRITTEN",
            source_bindings=[{"path": name, "identity": identity} for name, identity in sorted(snapshots.items())],
            scope="One controlled source-Python last-row prefill experiment; no private IndexByTensor out-of-range contract, numerical root cause, accuracy or feedback acceptance",
            next="Use the original static64 job, this named runtime probe and this new input_dir with run_target_prefill_smoke.py in a separate output; it creates a fresh plan for the changed input")
        # Write the new metadata payload before the final source checks, then
        # publish without replacing any existing path. No source/input inode
        # is linked; this staging inode belongs exclusively to this new task.
        pending = out / ".manifest.pending.json"
        save_json(pending, result)
        _stable(snapshots); _stable(derived)
        os.link(pending, out / "manifest.json", follow_symlinks=False)
        published = True
        pending.unlink()
    except (OSError, ValueError, TypeError, KeyError, StopIteration, struct.error) as error:
        result.update(status="BLOCKED", error=f"{type(error).__name__}: {error}",
            source_preservation="NOT_ESTABLISHED_FOR_COMPLETED_PREPARATION")
    if not published:
        save_json(out / "manifest.json", result)
    return result

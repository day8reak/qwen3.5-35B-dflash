"""Plan one decode1 trial from provided files and observed named HIAI inputs.

This does not turn the files into a continuation recipe. Common state files
are used exactly as supplied for each named input; their history and values,
the controls, and token/embedding correspondence remain unverified.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct

from .existing_air import _conversion_options
from .existing_air_profile import get_existing_air_profile
from .original_target_board import original_input_files
from .target_compile_failure import _read_small, _safe
from .target_prefill_smoke_plan import (_GE_TYPES, _INPUT_TYPES, _ND_VIEW_POLICY, _TOOL_TYPES, _WIDTHS,
    _ge_key, _identity, _json, _nd_view_key, _runtime_keys, _source, _tool_binding)


_KIND = "qwen35-original-decode-smoke-plan"
_PROFILE = "cpro-40block-decode-1"
_SCOPE = "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_NO_PREFILL_FEEDBACK"
_PRIMARY_STAT_SCOPE = "DECLARED_SHA_AND_CURRENT_STAT_ONLY_MODEL_NOT_OPENED"
_PRIMARY_REVALIDATED_SCOPE = "FRESH_SHA256_VERIFIED_CTIME_ONLY_PRIMARY_OMC_RECONCILIATION"
_PRIMARY_REVALIDATED_ACCESS = "SMALL_PRIMARY_OMC_REHASHED_FOR_CTIME_ONLY_RECONCILIATION"
_PLAN_SCOPE = "One independent decode1 invocation with user-supplied input values. Exact runtime input names bind named compiled slots and byte extents; common state-file history, scalar meanings, numerical accuracy and continuation are unverified. No controls generated, token repair, prefill-state feedback, draft invocation, model/weight read, device call or compiler invocation."
_PLAN_REVALIDATED_SCOPE = _PLAN_SCOPE.replace("model/weight read", "weight read") + " The small primary OMC was freshly hashed only for ctime-only reconciliation."


def bind_primary_omc(path, primary, *, expected_identity=None, revalidation_observations=None):
    """Bind primary metadata; fresh-hash only an otherwise identical ctime drift."""
    if revalidation_observations is not None and not isinstance(revalidation_observations, list):
        raise ValueError("primary revalidation observations must be a list")
    if Path(path).name != "model.omc" or not isinstance(primary, dict):
        raise ValueError("ctime-only reconciliation requires the selected primary model.omc record")
    model_path, stamp = _safe(path)
    current_identity = _identity(model_path.stat())
    if [current_identity[key] for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")] != stamp:
        raise ValueError("primary OMC identity changed before metadata validation")
    reasons = []
    recorded_bytes, recorded_stat = primary.get("bytes"), primary.get("stat")
    if type(recorded_bytes) is not int:
        reasons.append("INVALID_RECORDED_BYTE_COUNT")
    elif not 0 < recorded_bytes <= 64 << 20:
        reasons.append("RECORDED_BYTE_COUNT_OUTSIDE_64MIB_BUDGET")
    if stamp != recorded_stat:
        reasons.append("FIVE_STAT_IDENTITY_DIFFERS")
    if stamp[2] != recorded_bytes:
        reasons.append("BYTE_COUNT_DIFFERS")
    if not re.fullmatch(r"[0-9a-f]{64}", str(primary.get("sha256", ""))):
        reasons.append("INVALID_RECORDED_SHA256_DECLARATION")
    saved_differences = {}
    if expected_identity is not None:
        if not isinstance(expected_identity, dict) or set(expected_identity) != set(current_identity):
            reasons.append("SAVED_PLAN_PRIMARY_IDENTITY_FIELDS_DIFFER")
        else:
            saved_differences = {key: {"recorded": expected_identity[key], "current": value}
                for key, value in current_identity.items() if key != "ctime_ns" and expected_identity[key] != value}
            if saved_differences:
                reasons.append("SAVED_PLAN_PRIMARY_IDENTITY_DIFFERS_BEYOND_CTIME")
    ctime_only = (reasons in ([], ["FIVE_STAT_IDENTITY_DIFFERS"])
        and isinstance(recorded_stat, list) and len(recorded_stat) == 5
        and recorded_stat[:4] == stamp[:4]
        and (recorded_stat[4] != stamp[4] or isinstance(expected_identity, dict)
             and expected_identity.get("ctime_ns") != current_identity["ctime_ns"]))
    if reasons and not ctime_only:
        fields = ("device", "inode", "bytes", "mtime_ns", "ctime_ns")
        differences = ({name: {"recorded": old, "current": current}
            for name, old, current in zip(fields, recorded_stat, stamp) if old != current}
            if isinstance(recorded_stat, list) and len(recorded_stat) == 5 else {})
        diagnostic = {"path": str(model_path), "recorded_bytes": recorded_bytes, "current_bytes": stamp[2],
            "recorded_stat": recorded_stat, "current_stat": stamp, "stat_fields": list(fields),
            "stat_differences": differences, "saved_plan_identity_differences": saved_differences,
            "reasons": reasons, "payload_read": "NOT_OPENED", "content_sha256_comparison": "NOT_PERFORMED"}
        raise ValueError("recorded primary OMC extent or current identity changed: "
            + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True))
    scope = _PRIMARY_STAT_SCOPE
    if ctime_only:
        digest, remaining = hashlib.sha256(), recorded_bytes
        observation = {"kind": "PRIMARY_OMC_CTIME_ONLY_REVALIDATION", "status": "READ_REQUESTED",
            "path": str(model_path), "recorded_stat": recorded_stat, "current_stat": stamp,
            "current_identity": current_identity, "saved_plan_identity": expected_identity,
            "bytes": recorded_bytes, "expected_sha256": primary["sha256"], "read_bytes": 0,
            "hash_scope": "BOUNDED_PRIMARY_ONLY_MAXIMUM_64MIB_FULL_IDENTITY_STABLE_BEFORE_AFTER",
            "receipt_mutation": "NONE", "weight_payload_access": "NONE"}
        if revalidation_observations is not None:
            revalidation_observations.append(observation)
        try:
            fd = os.open(model_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                if _identity(os.fstat(stream.fileno())) != current_identity:
                    raise ValueError("primary OMC changed before ctime-only hash revalidation")
                while remaining:
                    raw = stream.read(min(1 << 20, remaining))
                    if not raw:
                        raise ValueError("primary OMC truncated during ctime-only hash revalidation")
                    remaining -= len(raw)
                    observation["read_bytes"] += len(raw)
                    digest.update(raw)
                if (stream.read(1) or _identity(os.fstat(stream.fileno())) != current_identity
                        or _identity(model_path.stat()) != current_identity):
                    raise ValueError("primary OMC changed during ctime-only hash revalidation")
            if digest.hexdigest() != primary["sha256"]:
                observation["observed_sha256"] = digest.hexdigest()
                raise ValueError("primary OMC ctime-only revalidation SHA256 differs from the compiled record: " + str(model_path))
            observation.update(kind="PRIMARY_OMC_CTIME_ONLY_REVALIDATED", status="SHA256_VERIFIED", sha256=digest.hexdigest())
        except Exception as error:
            observation.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        scope = _PRIMARY_REVALIDATED_SCOPE
    return {"path": str(model_path), "bytes": recorded_bytes, "sha256": primary["sha256"],
        "identity": current_identity, "scope": scope}


def _decode_expected_files():
    """Group the frozen declared profile's89 slots into13 source filenames."""
    profile = get_existing_air_profile(_PROFILE)
    entries = [entry.split(":", 1) for entry in profile["input_shape"].split(";")]
    filenames = original_input_files()
    if len(entries) != 89 or len(filenames) != 89:
        raise ValueError("decode file preflight requires the frozen89-input profile")
    grouped = {}
    for (name, raw_shape), (_, filename) in zip(entries, filenames):
        shape = list(map(int, raw_shape.split(",")))
        dtype = "FLOAT16" if name.startswith("RefData_") else _INPUT_TYPES[name]
        if (not shape or any(dim <= 0 for dim in shape)
                or not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.bin", filename)):
            raise ValueError("invalid frozen decode source file descriptor")
        expected = {"file": filename, "expected_dtype": dtype, "expected_shape": shape,
                    "expected_bytes": math.prod(shape) * _WIDTHS[dtype]}
        if filename in grouped:
            if any(grouped[filename][key] != expected[key] for key in expected):
                raise ValueError("one frozen decode source filename has conflicting descriptors: " + filename)
        else:
            grouped[filename] = {**expected, "input_names": []}
        grouped[filename]["input_names"].append(name)
    if len(grouped) != 13:
        raise ValueError("decode file preflight requires exactly13 distinct source filenames")
    return list(grouped.values())


def inspect_decode_input_extents(input_dir):
    """Observe all required file types and extents without opening payloads.

    This is a declared-profile diagnostic, not runtime-slot, byte encoding,
    readability or input-value validation. The planner still binds the actual
    named runtime descriptors and freshly hashes every supplied source file.
    Extra files, including goldens, are neither enumerated nor opened.
    """
    root, _ = _safe(input_dir, directory=True)
    files = []
    for expected in _decode_expected_files():
        path = root / expected["file"]
        row = {**expected, "path": str(path), "observed_bytes": None}
        try:
            info = path.lstat()
            row["observed_bytes"] = info.st_size
            if not stat.S_ISREG(info.st_mode):
                row.update(status="UNSAFE_OR_UNREADABLE",
                           reason="symlink" if stat.S_ISLNK(info.st_mode) else "nonregular file")
            elif info.st_size != expected["expected_bytes"]:
                row.update(status="WRONG_SIZE")
            else:
                row.update(status="REGULAR_EXTENT_MATCHED")
        except FileNotFoundError:
            row.update(status="MISSING", reason="required input file not found")
        except OSError as error:
            row.update(status="UNSAFE_OR_UNREADABLE", reason=f"{type(error).__name__}: {error}")
        files.append(row)
    failed = [row for row in files if row["status"] != "REGULAR_EXTENT_MATCHED"]
    return {"schema_version": 1, "artifact_kind": "qwen35-decode-input-extent-inspection",
        "status": "DECODE_INPUT_EXTENTS_INCOMPLETE" if failed else "DECODE_INPUT_EXTENTS_MATCHED",
        "input_dir": str(root), "profile": _PROFILE, "sequence_length": 1,
        "required_file_count": len(files), "matched_file_count": len(files) - len(failed),
        "failed_file_count": len(failed), "files": files, "failed_files": failed,
        "missing_files": [row["file"] for row in files if row["status"] == "MISSING"],
        "payload_access": "NONE_STAT_ONLY", "encoding_validation": "NOT_PERFORMED",
        "dtype_binding": "FROZEN_PROFILE_AND_ORIGINAL_EXPORTER_DECLARATIONS_NOT_FILE_ENCODING",
        "shape_binding": "FROZEN_CPRO_40BLOCK_DECODE1_PROFILE_NOT_RUNTIME_SLOT_VALIDATION",
        "readability_validation": "NOT_OPENED_NOT_TESTED", "device_execution": "NOT_RUN",
        "model_compilation": "NOT_RUN", "input_mutation": "NONE",
        "scope": "lstat of13 required input filenames only; no payload, golden, model or weight reads, default generation or device calls"}


def _input_extent_error(report):
    rows = []
    for row in report["failed_files"]:
        observed = "missing" if row["observed_bytes"] is None else str(row["observed_bytes"])
        reason = " (" + row["reason"] + ")" if row.get("reason") else ""
        rows.append(f"{row['file']}: {row['status']} expected={row['expected_bytes']} observed={observed}" + reason)
    return "decode input preflight found missing or invalid files: " + "; ".join(rows)


def _bound_probe(compilation, probe_dir, bindings, *, expected_primary_identity=None, revalidation_observations=None):
    """Bind metadata; permit only hash-verified primary OMC ctime drift."""
    job = compilation.parent
    outer = _json(job / "result.json", bindings)
    compiled = _json(compilation / "report.json", bindings)
    profile = _json(job / "compile-profile.json", bindings)
    contract = _json(job / "target-contract/report.json", bindings)
    probe = _json(probe_dir / "result.json", bindings)
    native = _json(probe_dir / "model-io.json", bindings)
    runtime = _json(probe_dir / "runtime.json", bindings)
    command = _json(probe_dir / "model-io-command.json", bindings)
    compiled_sha = next(row["sha256"] for row in bindings if row["path"] == str(compilation / "report.json"))
    frozen = get_existing_air_profile(_PROFILE)
    options = _conversion_options(frozen["input_shape"], frozen["dynamic_dims"],
        frozen["limit_partition_size"], frozen["save_weights_as_external_data"])
    if (outer.get("artifact_kind") != "qwen35-original-target-job" or outer.get("status") != "COMPILED"
            or outer.get("model_compilation") != "PASSED" or outer.get("output") != str(job)
            or outer.get("compilation_report_sha256") != compiled_sha
            or compiled.get("artifact_kind") != "qwen35-existing-air-compilation"
            or compiled.get("status") != "COMPILED" or compiled.get("model_compilation") != "PASSED"
            or compiled.get("returncode") != 0 or compiled.get("platform") != "kirinx90"
            or compiled.get("requested_options") != options or profile != frozen
            or compiled.get("source_preservation", {}).get("status") != "UNCHANGED_BY_STAT"
            or any(compiled.get(key) for key in ("audit_error", "artifact_inventory_error", "decode_input_audit_error", "retained_input_audit_error"))
            or contract.get("artifact_kind") != "qwen35-original-target-contract" or contract.get("status") != "OBSERVED"
            or contract.get("compilation_report_sha256") != compiled_sha
            or any(contract.get("omc", {}).get(key) != compiled.get("omc", {}).get(key) for key in ("bytes", "sha256"))):
        raise ValueError("decode smoke requires the exact successful static1 feature target and bound compiled contract")
    primary = compiled.get("omc", {})
    if not isinstance(primary, dict) or primary not in compiled.get("generated_artifacts", []):
        raise ValueError("primary OMC is absent from successful generated inventory")
    relative = primary.get("path")
    if relative not in {"compile-work/generated/model.omc", "compile-work/generated/model/model.omc"}:
        raise ValueError("unsupported primary OMC path")
    model = bind_primary_omc(compilation / relative, primary, expected_identity=expected_primary_identity,
        revalidation_observations=revalidation_observations)
    if (probe.get("artifact_kind") != "qwen35-target-model-io-probe" or probe.get("status") != "IO_METADATA_OBSERVED"
            or probe.get("job") != str(job) or probe.get("profile") != _PROFILE or probe.get("sequence_length") != 1
            or probe.get("compilation_report_sha256") != compiled_sha
            or probe.get("lock_state") != "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
            or probe.get("source_preservation") != "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED"
            or probe.get("compiled_boundary_count_check") != "MATCHED_89_INPUT_82_OUTPUT_COUNTS_ONLY"
            or probe.get("retrieval_errors") != []
            or any(probe.get(key) for key in ("lock_release_error", "native_evidence_error", "source_error"))
            or probe.get("remote_exitcode") != 0 or probe.get("model_sha256") != model["sha256"]
            or probe.get("model_bytes") != model["bytes"]
            or not isinstance(probe.get("model_path"), str)
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", probe["model_path"]) or ".." in Path(probe["model_path"]).parts
            or native.get("artifact_kind") != "qwen35-hiai-model-io-probe" or native.get("evidence_kind") != "device-hiai"
            or native.get("status") != "IO_DESCRIPTORS_OBSERVED" or native.get("restore") != "PASSED"
            or native.get("model_load_scope") != "RESTORE_METADATA_ONLY" or native.get("restore_init") != "NOT_RUN_METADATA_ONLY"
            or native.get("descriptor_stage") != "restore-only" or native.get("manager_init") != "NOT_RUN"
            or native.get("external_weight_initialization") != "NOT_REQUESTED"
            or native.get("process_called") is not False or native.get("run_called") is not False
            or type(native.get("tensor_allocations_requested")) is not int or native["tensor_allocations_requested"] != 0
            or native.get("model_sha256") != model["sha256"] or native.get("model_bytes") != model["bytes"]
            or native.get("model_path") != probe.get("model_path") or native.get("model_load_api") != "file-v2"
            or native.get("npu_supported") is not True or runtime.get("npu_supported") is not True
            or not isinstance(probe.get("runtime_version"), str) or not probe["runtime_version"]
            or runtime.get("runtime_version") != probe["runtime_version"]
            or native.get("runtime_version") != probe["runtime_version"]
            or native.get("expected_runtime_version") != probe["runtime_version"]
            or native.get("ddk_id") != compiled.get("ddk_id") or probe.get("ddk_id") != compiled.get("ddk_id")
            or not isinstance(probe.get("firmware_id"), str) or not probe["firmware_id"]
            or native.get("firmware_id") != probe["firmware_id"]):
        raise ValueError("actual restore-only runtime probe is not bound to this successful static1 OMC")
    for field in ("inputs", "outputs", "input_count", "output_count", "dtype_enum", "format_enum", "dtype_enum_values", "format_enum_values"):
        if probe.get(field) != native.get(field):
            raise ValueError("outer runtime observation differs from its native evidence: " + field)
    retrieved = probe.get("retrieved_sha256", {})
    for filename in ("model-io.json", "runtime.json", "model-io.exitcode", "runtime.exitcode"):
        raw, identity = _read_small(probe_dir / filename, 32 << 20)
        if identity["sha256"] != retrieved.get(filename) or (filename.endswith("exitcode") and raw.strip() != b"0"):
            raise ValueError("runtime probe retrieved hash/exit binding failed: " + filename)
        if filename.endswith("exitcode"):
            bindings.append(identity)
    argv = command.get("argv")
    expected = {"--model-io-only": native["model_path"], "--model-sha256": model["sha256"],
        "--descriptor-stage": "restore-only", "--expected-runtime-version": probe["runtime_version"],
        "--ddk-id": compiled["ddk_id"], "--firmware-id": probe["firmware_id"],
        "--output": str(Path(native["model_path"]).parent)}
    if not isinstance(argv, list) or len(argv) != 15 or not isinstance(argv[0], str):
        raise ValueError("native model-I/O command is not the bounded restore-only invocation")
    observed = dict(zip(argv[1::2], argv[2::2]))
    if (len(observed) != 7 or set(observed) != set(expected)
            or any(observed.get(key) != value for key, value in expected.items() if key != "--output")
            or not isinstance(observed.get("--output"), str) or not observed["--output"].endswith("/model-io.json")
            or command.get("cwd") != str(Path(native["model_path"]).parent)):
        raise ValueError("native model-I/O command/profile binding changed")
    return compiled, contract, probe, native, model


def _names(native, direction):
    names = []
    for row in native[direction]:
        name = row.get("tensor_name")
        if (row.get("tensor_name_status") != "OBSERVED" or not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,255}", name)):
            raise ValueError("all actual runtime tensor names must be literally observed: " + direction)
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("runtime tensor names are duplicated: " + direction)
    return names


def _control_observations(input_root, sources):
    controls, warnings = [], []
    for filename, dtype, code in (("new_kv_cache_pos.bin", "INT32", "i"), ("output_pos.bin", "INT32", "i"),
            ("allQLen.bin", "INT64", "q"), ("conv_state_pos.bin", "INT32", "i"),
            ("effective_length.bin", "INT16", "h"), ("position_ids.bin", "INT32", "i"),
            ("embed_scale.bin", "FLOAT32", "f")):
        identity = sources[str(input_root / filename)]
        raw, now = _read_small(input_root / filename, 1024)
        if now["sha256"] != identity["sha256"] or now["bytes"] != identity["bytes"]:
            raise ValueError("provided decode control changed during observation")
        values = [value[0] for value in struct.iter_unpack("<" + code, raw)]
        values = [value if not isinstance(value, float) or math.isfinite(value)
                  else "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity" for value in values]
        controls.append({"path": identity["path"], "bytes": identity["bytes"], "sha256": identity["sha256"],
            "dtype": dtype, "byte_order": "little", "values": values, "interpretation": "UNINTERPRETED_USER_VALUES"})
    return controls, warnings


def prepare_decode_smoke_plan(compilation_dir, runtime_probe, input_dir, *, revalidation_observations=None,
                               _saved_primary_identity=None):
    """Return one independently supplied decode trial plan; no device calls."""
    if revalidation_observations is not None and not isinstance(revalidation_observations, list):
        raise ValueError("primary revalidation observations must be a list")
    compilation, _ = _safe(compilation_dir, directory=True)
    probe_dir, _ = _safe(runtime_probe, directory=True)
    input_root, _ = _safe(input_dir, directory=True)
    bindings = []
    compiled, contract, probe, native, model = _bound_probe(compilation, probe_dir, bindings,
        expected_primary_identity=_saved_primary_identity, revalidation_observations=revalidation_observations)
    tool = _tool_binding(probe, bindings)
    extents = inspect_decode_input_extents(input_root)
    if extents["status"] != "DECODE_INPUT_EXTENTS_MATCHED":
        raise ValueError(_input_extent_error(extents))
    profile = get_existing_air_profile(_PROFILE)
    entries = [entry.split(":", 1) for entry in profile["input_shape"].split(";")]
    input_rows = contract.get("boundaries", {}).get("inputs_in_serialized_op_order")
    output_rows = contract.get("boundaries", {}).get("outputs")
    if (not isinstance(input_rows, list) or len(input_rows) != 89 or not isinstance(output_rows, list)
            or len(output_rows) != 82 or any(not isinstance(row, dict) for row in [*input_rows, *output_rows])):
        raise ValueError("compiled decode target requires89inputs/82outputs")
    by_name = {row.get("name"): row for row in input_rows}
    if len(by_name) != 89 or set(by_name) != {name for name, _ in entries}:
        raise ValueError("compiled decode input names are missing or duplicated")
    runtime_inputs = _runtime_keys(native, "inputs")
    runtime_outputs = _runtime_keys(native, "outputs")
    input_names, output_names = _names(native, "inputs"), _names(native, "outputs")
    if set(input_names) != set(by_name):
        raise ValueError("literal runtime input names do not exactly match compiled decode inputs")
    sources, source_by_name, descriptors = {}, {}, {}
    for (name, raw_shape), (semantic, filename) in zip(entries, original_input_files()):
        row = by_name[name]
        dtype = "FLOAT16" if name.startswith("RefData_") else _INPUT_TYPES[name]
        key = _ge_key(row)
        if key != (dtype, tuple(map(int, raw_shape.split(",")))) or row.get("type") not in (
                {"Data", "RefData"} if name.startswith("RefData_") else {"Data"}):
            raise ValueError("compiled decode input geometry/type differs: " + name)
        width = math.prod(key[1]) * _WIDTHS[dtype]
        path = input_root / filename
        if str(path) not in sources:
            sources[str(path)] = _source(path, width)
        elif sources[str(path)]["bytes"] != width:
            raise ValueError("one supplied decode file has conflicting exact extents")
        source_by_name[name], descriptors[name] = (semantic, filename, sources[str(path)]), key
    inputs = []
    for index, (name, key) in enumerate(zip(input_names, runtime_inputs)):
        if _nd_view_key(key) != _nd_view_key(descriptors[name]):
            raise ValueError("named runtime decode input differs in dtype or exact ND view: " + name)
        semantic, filename, source = source_by_name[name]
        inputs.append({"index": index, "name": name, "runtime_tensor_name": name, "semantic": semantic,
            "shape": list(key[1]), "source_shape": list(descriptors[name][1]), "dtype": key[0],
            "nd_view_policy": _ND_VIEW_POLICY, "tool_input_type": _TOOL_TYPES[key[0]],
            "schema_dtype_names": by_name[name]["schema_dtype_names"], "bytes": source["bytes"],
            "source": source["path"], "source_filename": filename, "sha256": source["sha256"],
            "file": f"inputs/input_{index:03d}.bin", "serialized_data_index": by_name[name].get("serialized_data_index"),
            "semantic_slot_identity": "RUNTIME_NAME_MATCHED_COMPILED_INPUT_NAME",
            "source_semantics": "UNVERIFIED_USER_INPUTS"})
    by_port = {row.get("output_index"): row for row in output_rows}
    if set(by_port) != set(range(82)) or len({row.get("netoutput") for row in output_rows}) != 1:
        raise ValueError("compiled decode output ports are missing or ambiguous")
    expected = []
    for port in range(82):
        key = _ge_key(by_port[port])
        frozen = ("FLOAT32", (1, 1, 248320)) if port == 0 else (
            ("FLOAT16", (1, 1, 16384)) if port == 81 else descriptors[f"RefData_{port - 1}"])
        if key != frozen:
            raise ValueError("compiled decode logits/state/feature output differs")
        expected.append(_nd_view_key(key))
    observed = [_nd_view_key(key) for key in runtime_outputs]
    if Counter(observed) != Counter(expected):
        raise ValueError("actual runtime decode output typed ND views differ from complete compiled outputs")
    logits = [index for index, key in enumerate(observed) if key == expected[0]]
    features = [index for index, key in enumerate(observed) if key == expected[81]]
    if len(logits) != 1 or len(features) != 1:
        raise ValueError("decode logits/feature descriptors are not unique")
    outputs = [{"index": index, "name": "logits" if index == logits[0] else "target_features" if index == features[0]
                else f"state_output_unmapped_{index}", "runtime_tensor_name": output_names[index],
        "shape": list(key[1]), "dtype": key[0], "tool_output_type": _TOOL_TYPES[key[0]],
        "schema_dtype_names": ["DT_FLOAT" if key[0] == "FLOAT32" else "DT_FLOAT16"],
        "bytes": math.prod(key[1]) * _WIDTHS[key[0]], "nd_view_policy": _ND_VIEW_POLICY,
        "semantic_slot_identity": "UNIQUE_TYPED_ND_VIEW" if index in {logits[0], features[0]} else "UNVERIFIED"}
        for index, key in enumerate(runtime_outputs)]
    if sum(row["bytes"] for row in inputs + outputs) > 512 << 20:
        raise ValueError("decode trial raw input/output extents exceed512MiB budget")
    controls, warnings = _control_observations(input_root, sources)
    for source in sources.values():
        if _source(source["path"], source["bytes"]) != source:
            raise ValueError("provided decode input changed while constructing plan")
    for identity in bindings:
        if _read_small(identity["path"], 32 << 20)[1] != identity:
            raise ValueError("bound decode/runtime metadata changed while constructing plan")
    if _identity(Path(model["path"]).stat()) != model["identity"]:
        raise ValueError("compiled decode primary identity changed while constructing plan")
    return {"schema_version": 1, "artifact_kind": _KIND, "status": "DECODE_SMOKE_PLANNED", "gear": 1,
        "sequence_length": 1, "profile": _PROFILE, "compilation": str(compilation), "runtime_probe": str(probe_dir),
        "input_dir": str(input_root), "compilation_report_sha256": probe["compilation_report_sha256"],
        "model_sha256": model["sha256"], "runtime_version": probe["runtime_version"], "firmware_id": probe["firmware_id"],
        "ddk_id": compiled["ddk_id"], "tool_binding": tool, "metadata_bindings": bindings,
        "primary_model_binding": model, "source_inputs": sorted(sources.values(), key=lambda row: row["path"]),
        "inputs": inputs, "outputs": outputs, "input_shape": None, "output_shape": None,
        "input_type": ",".join(row["tool_input_type"] for row in inputs),
        "output_type": ",".join(row["tool_output_type"] for row in outputs), "input_format": None, "output_format": None,
        "format_policy": "RESTORED_MODEL_ND_DESCRIPTORS_NO_TOOL_OVERRIDE", "nd_view_policy": _ND_VIEW_POLICY,
        "input_mapping_source": "ACTUAL_HIAI_TENSOR_NAMES_AND_TYPED_ND_VIEWS",
        "runtime_input_order": "RUNTIME_DESCRIPTOR_INDICES_BOUND_BY_LITERAL_INPUT_NAMES",
        "source_semantics": "UNVERIFIED_USER_INPUTS", "state_input_layer_identity": "NAMES_OBSERVED_SOURCE_HISTORY_UNVERIFIED",
        "state_output_layer_identity": "UNVERIFIED", "execution_scope": _SCOPE,
        "state_continuity": "NOT_VALIDATED", "state_feedback": "NOT_RUN", "numerical_validation": "NOT_RUN_NO_REFERENCE",
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "model_payload_access": _PRIMARY_REVALIDATED_ACCESS if model["scope"] == _PRIMARY_REVALIDATED_SCOPE else "NONE",
        "weight_payload_access": "NONE", "target_features_requested": True, "target_features": "EXPOSED_NOT_VALIDATED",
        "feature_output_evidence": {"status": "EXPOSED_NOT_VALIDATED", "compiled_output_index": 81,
            "runtime_output_index": features[0], "shape": outputs[features[0]]["shape"], "dtype": "FLOAT16",
            "nd_view_policy": _ND_VIEW_POLICY, "output_mapping": "RUNTIME_DESCRIPTOR_INDEX_ONLY_TOOL_FILENAMES_UNVERIFIED",
            "semantic_validation": "NOT_RUN"}, "feature_runtime_output_index": features[0],
        "logits_runtime_output_index": logits[0], "control_observations": controls, "warnings": warnings,
        "logical_input_bytes": sum(row["bytes"] for row in inputs), "output_bytes": sum(row["bytes"] for row in outputs),
        "unique_source_input_bytes": sum(row["bytes"] for row in sources.values()),
        "scope": _PLAN_REVALIDATED_SCOPE if model["scope"] == _PRIMARY_REVALIDATED_SCOPE else _PLAN_SCOPE}


def validate_decode_smoke_plan(plan, *, revalidation_observations=None):
    """Rehash all supplied input files and bound metadata without writing them."""
    if revalidation_observations is not None and not isinstance(revalidation_observations, list):
        raise ValueError("primary revalidation observations must be a list")
    if (not isinstance(plan, dict) or plan.get("artifact_kind") != _KIND
            or plan.get("status") != "DECODE_SMOKE_PLANNED" or plan.get("execution_scope") != _SCOPE
            or any(not isinstance(plan.get(key), str) for key in ("compilation", "runtime_probe", "input_dir"))):
        raise ValueError("expected an exact independent decode1 smoke plan")
    old_binding = plan.get("primary_model_binding", {})
    if not isinstance(old_binding, dict) or not isinstance(old_binding.get("identity"), dict):
        raise ValueError("saved decode plan requires its complete primary identity")
    observations = []
    try:
        current = prepare_decode_smoke_plan(plan["compilation"], plan["runtime_probe"], plan["input_dir"],
            _saved_primary_identity=old_binding["identity"], revalidation_observations=observations)
        if current != plan:
            compared = deepcopy(current)
            allowed_annotations = (any(row.get("status") == "SHA256_VERIFIED" for row in observations)
                and current["primary_model_binding"]["scope"] == _PRIMARY_REVALIDATED_SCOPE
                and old_binding.get("scope") in {_PRIMARY_STAT_SCOPE, _PRIMARY_REVALIDATED_SCOPE}
                and plan.get("model_payload_access") in {"NONE", _PRIMARY_REVALIDATED_ACCESS}
                and plan.get("scope") in {_PLAN_SCOPE, _PLAN_REVALIDATED_SCOPE})
            if allowed_annotations:
                compared["primary_model_binding"]["identity"]["ctime_ns"] = old_binding["identity"]["ctime_ns"]
                compared["primary_model_binding"]["scope"] = old_binding["scope"]
                compared["model_payload_access"] = plan["model_payload_access"]
                compared["scope"] = plan["scope"]
            if compared != plan:
                raise ValueError("decode smoke metadata or supplied input identity changed after planning")
        return current
    finally:
        if revalidation_observations is not None:
            revalidation_observations.extend(observations)

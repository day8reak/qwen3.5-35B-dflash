"""Plan one decode1 trial from provided files and observed named HIAI inputs.

This does not turn the files into a continuation recipe. Common state files
are used exactly as supplied for each named input; their history and values,
the controls, and token/embedding correspondence remain unverified.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
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


def _bound_probe(compilation, probe_dir, bindings):
    """Bind saved metadata and model stats without opening any OMC or weight."""
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
    model_path, stamp = _safe(compilation / relative)
    if (type(primary.get("bytes")) is not int or not 0 < primary["bytes"] <= 64 << 20
            or stamp != primary.get("stat") or stamp[2] != primary["bytes"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(primary.get("sha256", "")))):
        raise ValueError("recorded primary OMC extent or current identity changed")
    model = {"path": str(model_path), "bytes": primary["bytes"], "sha256": primary["sha256"],
        "identity": _identity(model_path.stat()), "scope": "DECLARED_SHA_AND_CURRENT_STAT_ONLY_MODEL_NOT_OPENED"}
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


def prepare_decode_smoke_plan(compilation_dir, runtime_probe, input_dir):
    """Return one independently supplied decode trial plan; no device calls."""
    compilation, _ = _safe(compilation_dir, directory=True)
    probe_dir, _ = _safe(runtime_probe, directory=True)
    input_root, _ = _safe(input_dir, directory=True)
    bindings = []
    compiled, contract, probe, native, model = _bound_probe(compilation, probe_dir, bindings)
    tool = _tool_binding(probe, bindings)
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
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "model_payload_access": "NONE",
        "weight_payload_access": "NONE", "target_features_requested": True, "target_features": "EXPOSED_NOT_VALIDATED",
        "feature_output_evidence": {"status": "EXPOSED_NOT_VALIDATED", "compiled_output_index": 81,
            "runtime_output_index": features[0], "shape": outputs[features[0]]["shape"], "dtype": "FLOAT16",
            "nd_view_policy": _ND_VIEW_POLICY, "output_mapping": "RUNTIME_DESCRIPTOR_INDEX_ONLY_TOOL_FILENAMES_UNVERIFIED",
            "semantic_validation": "NOT_RUN"}, "feature_runtime_output_index": features[0],
        "logits_runtime_output_index": logits[0], "control_observations": controls, "warnings": warnings,
        "logical_input_bytes": sum(row["bytes"] for row in inputs), "output_bytes": sum(row["bytes"] for row in outputs),
        "unique_source_input_bytes": sum(row["bytes"] for row in sources.values()),
        "scope": "One independent decode1 invocation with user-supplied input values. Exact runtime input names bind named compiled slots and byte extents; common state-file history, scalar meanings, numerical accuracy and continuation are unverified. No controls generated, token repair, prefill-state feedback, draft invocation, model/weight read, device call or compiler invocation."}


def validate_decode_smoke_plan(plan):
    """Rehash all supplied input files and bound metadata without writing them."""
    if (not isinstance(plan, dict) or plan.get("artifact_kind") != _KIND
            or plan.get("status") != "DECODE_SMOKE_PLANNED" or plan.get("execution_scope") != _SCOPE
            or any(not isinstance(plan.get(key), str) for key in ("compilation", "runtime_probe", "input_dir"))):
        raise ValueError("expected an exact independent decode1 smoke plan")
    current = prepare_decode_smoke_plan(plan["compilation"], plan["runtime_probe"], plan["input_dir"])
    if current != plan:
        raise ValueError("decode smoke metadata or supplied input identity changed after planning")
    return current

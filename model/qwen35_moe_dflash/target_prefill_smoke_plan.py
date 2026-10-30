"""Plan one original prefill64 call from observed runtime descriptors.

This deliberately does not assign equal-shaped state tensors to layers. A
permutation is harmless for this one invocation only when every source value
in that typed shape class is byte-identical. Subsequent state feedback needs a
separate semantic mapping and cannot reuse this certificate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from .existing_air import _conversion_options
from .existing_air_profile import get_existing_air_profile
from .original_target_board import original_input_files
from .metadata_revalidation import metadata_identity_matches
from .target_compile_failure import _read_small, _safe
from .target_export_reuse import _reject_constant, _unique_object


_KIND = "qwen35-original-prefill-smoke-plan"
_PROFILE = "cpro-40block-prefill-64"
_ND_VIEW_POLICY = "EDGE_SINGLETON_ND_LINEAR_VIEW_ONLY"
_WIDTHS = {"FLOAT32": 4, "FLOAT16": 2, "INT8": 1, "INT16": 2, "INT32": 4, "INT64": 8}
_GE_TYPES = {"DT_FLOAT": "FLOAT32", "DT_FLOAT32": "FLOAT32", "DT_FLOAT16": "FLOAT16",
             "DT_INT8": "INT8", "DT_INT16": "INT16", "DT_INT32": "INT32", "DT_INT64": "INT64"}
_TOOL_TYPES = {name: {"FLOAT32": "FP32", "FLOAT16": "FP16"}.get(name, name) for name in _WIDTHS}
_INPUT_TYPES = {"arg0_1": "FLOAT32", "arg1_1": "INT8", "arg2_1": "INT32", "arg14_1": "INT32",
                "arg18_1": "INT16", "arg107_1": "INT32", "arg113_1": "FLOAT32",
                "arg114_1": "INT64", "arg1238_1": "INT32"}


def _json(path, bindings):
    raw, identity = _read_small(path, 32 << 20)
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("prefill smoke metadata must be an object")
    bindings.append(identity)
    return value


def _identity(info):
    return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "mode": info.st_mode, "uid": info.st_uid, "gid": info.st_gid, "nlink": info.st_nlink}


def _source(path, expected_bytes):
    path, _ = _safe(path)
    before = _identity(path.stat())
    if (not stat.S_ISREG(before["mode"]) or before["bytes"] != expected_bytes
            or not 0 < expected_bytes <= 64 << 20):
        raise ValueError("original prefill input size/type differs from its exact descriptor: " + str(path))
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != before:
            raise ValueError("original prefill input changed before hashing")
        while block := stream.read(1 << 20):
            digest.update(block)
        if _identity(os.fstat(stream.fileno())) != before or _identity(path.stat()) != before:
            raise ValueError("original prefill input changed during hashing")
    return {"path": str(path), "bytes": expected_bytes, "sha256": digest.hexdigest(), "identity": before}


def _shape(row):
    shape = row.get("shape")
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 8
            or any(type(dim) is not int or dim <= 0 for dim in shape)):
        raise ValueError("prefill smoke requires exact positive fixed-rank descriptors")
    return tuple(shape)


def _ge_key(row):
    names = row.get("schema_dtype_names")
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or name not in _GE_TYPES for name in names)
            or len({_GE_TYPES[name] for name in names}) != 1):
        raise ValueError("unsupported or ambiguous compiled dtype")
    if row.get("layout") != "ND":
        raise ValueError("prefill smoke requires the compiled ND layout")
    return _GE_TYPES[names[0]], _shape(row)


def _nd_view_key(key):
    """Drop only outer size-one axes; ND byte order and count are unchanged.

    The input to this helper is an explicit nonempty positive descriptor.
    Thus an empty normalized shape denotes a known one-element view, never an
    unknown or missing raw scalar descriptor.
    """
    dtype, shape = key
    first, end = 0, len(shape)
    while first < end and shape[first] == 1:
        first += 1
    while end > first and shape[end - 1] == 1:
        end -= 1
    return dtype, shape[first:end]


def _runtime_keys(native, direction):
    values, formats = native.get("dtype_enum_values"), native.get("format_enum_values")
    if (native.get("dtype_enum") != "hiai::DataType" or native.get("format_enum") != "hiai::Format"
            or not isinstance(values, dict) or not set(_WIDTHS) <= set(values)
            or any(type(values[name]) is not int for name in _WIDTHS)
            or len({values[name] for name in _WIDTHS}) != len(_WIDTHS)
            or not isinstance(formats, dict) or type(formats.get("ND")) is not int):
        raise ValueError("actual selected SDK enum tables must explicitly include INT8 and the original target types")
    reverse = {values[name]: name for name in _WIDTHS}
    rows = native.get(direction)
    count = 89 if direction == "inputs" else 82
    if (not isinstance(rows, list) or len(rows) != count or native.get(direction[:-1] + "_count") != count):
        raise ValueError("prefill smoke requires actual runtime counts89/82")
    result = []
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index
                or type(row.get("raw_dtype")) is not int or row["raw_dtype"] not in reverse
                or type(row.get("raw_format")) is not int or row["raw_format"] != formats["ND"]):
            raise ValueError("runtime descriptors require ordered unique indices and known ND dtype values")
        result.append((reverse[row["raw_dtype"]], _shape(row)))
    return result


def _tool_binding(probe, bindings, metadata_revalidation_observations=None):
    if metadata_revalidation_observations is not None and not isinstance(metadata_revalidation_observations, list):
        raise ValueError("metadata revalidation observations must be a list")
    rows = probe.get("source_bindings")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("runtime probe must retain its exact source/help metadata bindings")
    known = {row["path"]: row for row in bindings}
    tools, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in seen:
            raise ValueError("invalid repeated runtime probe source binding")
        seen.add(row["path"])
        if row["path"] in known:
            identity = known[row["path"]]
        else:
            value = _json(Path(row["path"]), bindings)
            identity = bindings[-1]
            if value.get("artifact_kind") != "qwen35-target-board-tool-inspection":
                raise ValueError("unexpected runtime probe source outside the compiled job/help receipt")
            tools.append((value, identity))
        if not metadata_identity_matches(row, identity, role="runtime-probe-source-metadata",
                observations=metadata_revalidation_observations):
            raise ValueError("runtime probe source/help identity changed")
    if len(tools) != 1 or not {row["path"] for row in bindings[:4]} <= seen:
        raise ValueError("one exact previously observed tool-help receipt is required")
    tool, identity = tools[0]
    help_row = tool.get("help", {})
    text = help_row.get("text")
    required = ("model", "input", "output_dir", "modelmanager_itf", "direct_from_file", "weight_dir", "times", "input_type", "output_type")
    if (tool.get("status") != "INSPECTED" or tool.get("tool_identity") != "SHA256_MATCHED_BEFORE_AND_AFTER"
            or not re.fullmatch(r"[0-9a-f]{64}", str(tool.get("tool_sha256_before", "")))
            or tool.get("tool_sha256_before") != tool.get("tool_sha256_after")
            or not isinstance(tool.get("tool"), str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", tool["tool"])
            or ".." in Path(tool["tool"]).parts
            or any(tool.get(key) != "NOT_RUN" for key in ("model_upload", "model_load", "model_inference"))
            or tool.get("firmware", {}).get("returncode") != 0
            or not isinstance(tool["firmware"].get("text"), str)
            or tool["firmware"]["text"].strip() != probe["firmware_id"]
            or help_row.get("observation") != "HELP_TEXT_OBSERVED" or not isinstance(text, str)
            or help_row.get("truncated") is not False or help_row.get("utf8_bytes") != len(text.encode())
            or help_row.get("utf8_sha256") != hashlib.sha256(text.encode()).hexdigest()
            or any(not re.search(r"(?<![A-Za-z0-9_])--" + flag + r"(?![A-Za-z0-9_])", text) for flag in required)):
        raise ValueError("actual tool-help/firmware does not establish the explicit first-prefill options")
    return {"path": tool["tool"], "sha256": tool["tool_sha256_before"], "inspection": identity}


def _bound_probe(compilation, probe_dir, bindings, *, primary_revalidations=None):
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
        raise ValueError("prefill smoke requires the exact successful static64 feature target and bound compiled contract")
    primary = compiled.get("omc", {})
    if not isinstance(primary, dict) or primary not in compiled.get("generated_artifacts", []):
        raise ValueError("primary OMC is absent from successful generated inventory")
    relative = primary.get("path")
    if not isinstance(relative, str) or relative not in {"compile-work/generated/model.omc", "compile-work/generated/model/model.omc"}:
        raise ValueError("unsupported primary OMC path")
    from .target_decode_smoke_plan import bind_primary_omc
    current_primary = bind_primary_omc(compilation / relative, primary,
        revalidation_observations=primary_revalidations)
    model = _source(compilation / relative, primary.get("bytes"))
    if model["sha256"] != primary.get("sha256") or model["identity"] != current_primary["identity"]:
        raise ValueError("original compiled OMC content changed")
    if (probe.get("artifact_kind") != "qwen35-target-model-io-probe" or probe.get("status") != "IO_METADATA_OBSERVED"
            or probe.get("job") != str(job) or probe.get("profile") != _PROFILE or probe.get("sequence_length") != 64
            or probe.get("compilation_report_sha256") != compiled_sha
            or probe.get("lock_state") != "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
            or probe.get("source_preservation") != "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED"
            or probe.get("compiled_boundary_count_check") != "MATCHED_89_INPUT_82_OUTPUT_COUNTS_ONLY"
            or probe.get("retrieval_errors") != []
            or any(probe.get(key) for key in ("lock_release_error", "native_evidence_error", "source_error"))
            or probe.get("remote_exitcode") != 0 or probe.get("model_sha256") != model["sha256"]
            or probe.get("model_bytes") != model["bytes"]
            or not isinstance(probe.get("model_path"), str)
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", probe["model_path"])
            or ".." in Path(probe["model_path"]).parts
            or native.get("artifact_kind") != "qwen35-hiai-model-io-probe" or native.get("evidence_kind") != "device-hiai"
            or native.get("status") != "IO_DESCRIPTORS_OBSERVED" or native.get("restore") != "PASSED"
            or native.get("model_load_scope") != "RESTORE_METADATA_ONLY" or native.get("restore_init") != "NOT_RUN_METADATA_ONLY"
            or native.get("descriptor_stage") != "restore-only" or native.get("manager_init") != "NOT_RUN"
            or native.get("external_weight_initialization") != "NOT_REQUESTED"
            or native.get("process_called") is not False or native.get("run_called") is not False
            or type(native.get("tensor_allocations_requested")) is not int or native["tensor_allocations_requested"] != 0
            or native.get("model_sha256") != model["sha256"] or native.get("model_bytes") != model["bytes"]
            or native.get("model_path") != probe.get("model_path")
            or native.get("model_load_api") != "file-v2"
            or native.get("npu_supported") is not True
            or runtime.get("npu_supported") is not True
            or not isinstance(probe.get("runtime_version"), str) or not probe["runtime_version"]
            or runtime.get("runtime_version") != probe["runtime_version"]
            or native.get("runtime_version") != probe["runtime_version"]
            or native.get("expected_runtime_version") != probe["runtime_version"]
            or native.get("ddk_id") != compiled.get("ddk_id") or probe.get("ddk_id") != compiled.get("ddk_id")
            or not isinstance(probe.get("firmware_id"), str) or not probe["firmware_id"]
            or native.get("firmware_id") != probe["firmware_id"]):
        raise ValueError("actual restore-only runtime probe is not bound to this successful static64 OMC")
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


def prepare_prefill_smoke_plan(compilation_dir, runtime_probe_dir, input_dir):
    """Return a first-call plan or reject unresolved exact descriptor bindings.

    No network calls, model execution, synthesized controls or weight payload
    reads occur. Only the small primary OMC and original input files are hashed.
    """
    compilation, _ = _safe(compilation_dir, directory=True)
    probe_dir, _ = _safe(runtime_probe_dir, directory=True)
    input_root, _ = _safe(input_dir, directory=True)
    bindings = []
    primary_revalidations = []
    compiled, contract, probe, native, model = _bound_probe(compilation, probe_dir, bindings,
        primary_revalidations=primary_revalidations)
    tool = _tool_binding(probe, bindings)
    profile = get_existing_air_profile(_PROFILE)
    entries = [entry.split(":", 1) for entry in profile["input_shape"].split(";")]
    observed = contract.get("boundaries", {})
    input_rows, output_rows = observed.get("inputs_in_serialized_op_order"), observed.get("outputs")
    if (not isinstance(input_rows, list) or len(input_rows) != 89
            or any(not isinstance(row, dict) or not isinstance(row.get("name"), str) for row in input_rows)
            or not isinstance(output_rows, list) or len(output_rows) != 82
            or any(not isinstance(row, dict) or type(row.get("output_index")) is not int
                or not isinstance(row.get("netoutput"), str) or not row["netoutput"] for row in output_rows)):
        raise ValueError("compiled feature target requires89inputs/82outputs")
    by_name = {row.get("name"): row for row in input_rows if isinstance(row, dict)}
    if len(by_name) != 89 or set(by_name) != {name for name, _ in entries}:
        raise ValueError("compiled original input names are incomplete or repeated")
    groups, sources = defaultdict(list), {}
    for (name, raw_shape), (semantic, filename) in zip(entries, original_input_files()):
        row = by_name[name]
        physical_key = _ge_key(row)
        expected = ("FLOAT16" if name.startswith("RefData_") else _INPUT_TYPES[name], tuple(map(int, raw_shape.split(","))))
        if physical_key != expected or row.get("type") not in ({"Data", "RefData"} if name.startswith("RefData_") else {"Data"}):
            raise ValueError("compiled original input geometry/type differs: " + name)
        key = _nd_view_key(physical_key)
        width = math.prod(physical_key[1]) * _WIDTHS[physical_key[0]]
        source = input_root / filename
        if str(source) not in sources:
            sources[str(source)] = _source(source, width)
        elif sources[str(source)]["bytes"] != width:
            raise ValueError("one original input file has conflicting exact byte extents")
        groups[key].append({"name": name, "semantic": semantic, "source": str(source),
            "schema_dtype_names": row["schema_dtype_names"], "source_shape": list(physical_key[1]),
            "serialized_data_index": row.get("serialized_data_index")})
    runtime_input_shapes = _runtime_keys(native, "inputs")
    runtime_inputs = [_nd_view_key(key) for key in runtime_input_shapes]
    if Counter(runtime_inputs) != Counter({key: len(rows) for key, rows in groups.items()}):
        raise ValueError("runtime input typed ND views differ; only edge singleton axes are byte-equivalent")
    certificates = []
    for key, rows in groups.items():
        identities = {(sources[row["source"]]["bytes"], sources[row["source"]]["sha256"]) for row in rows}
        if len(identities) != 1:
            raise ValueError("same ND-view input class has different actual bytes; first-call permutation is unresolved")
        certificates.append({"dtype": key[0], "canonical_shape": list(key[1]), "nd_view_policy": _ND_VIEW_POLICY,
            "source_shapes": [{"name": row["name"], "shape": row["source_shape"]} for row in rows],
            "runtime_shapes": [{"index": index, "shape": list(runtime_input_shapes[index][1])}
                for index, actual in enumerate(runtime_inputs) if actual == key],
            "compiled_candidates": [row["name"] for row in rows],
            "runtime_indices": [index for index, actual in enumerate(runtime_inputs) if actual == key],
            "source_files": sorted({row["source"] for row in rows}), "sha256": next(iter(identities))[1],
            "basis": "UNIQUE_TYPED_ND_VIEW" if len(rows) == 1 else "IDENTICAL_SOURCE_BYTES_WITHIN_TYPED_ND_VIEW_CLASS"})
    inputs = []
    for index, key in enumerate(runtime_inputs):
        rows = groups[key]
        representative = rows[0]
        inputs.append({"index": index, "name": representative["name"],
            "semantic": representative["semantic"] if len(rows) == 1 else "BYTE_EQUIVALENT_INPUT_VALUE_CLASS",
            "shape": list(runtime_input_shapes[index][1]), "source_shape": representative["source_shape"],
            "source_candidate_shapes": [{"name": row["name"], "shape": row["source_shape"]} for row in rows],
            "nd_view_policy": _ND_VIEW_POLICY, "dtype": key[0], "tool_input_type": _TOOL_TYPES[key[0]],
            "schema_dtype_names": representative["schema_dtype_names"], "bytes": sources[representative["source"]]["bytes"],
            "source": representative["source"], "source_filename": Path(representative["source"]).name,
            "sha256": sources[representative["source"]]["sha256"], "file": f"inputs/input_{index:03d}.bin",
            "compiled_candidates": [row["name"] for row in rows], "semantic_slot_identity": "UNVERIFIED" if len(rows) > 1 else "UNIQUE_TYPED_ND_VIEW"})
    by_port = {row.get("output_index"): row for row in output_rows if isinstance(row, dict)}
    if set(by_port) != set(range(82)) or len({row.get("netoutput") for row in output_rows}) != 1:
        raise ValueError("compiled output ports are missing or ambiguous")
    expected_outputs = []
    for index in range(82):
        key = _ge_key(by_port[index])
        expected = ("FLOAT32", (1, 1, 248320)) if index == 0 else (
            ("FLOAT16", (1, 64, 16384)) if index == 81 else _ge_key(by_name[f"RefData_{index - 1}"]))
        if key != expected:
            raise ValueError("compiled canonical logits/state/feature output differs")
        expected_outputs.append(key)
    expected_output_views = [_nd_view_key(key) for key in expected_outputs]
    runtime_output_shapes = _runtime_keys(native, "outputs")
    runtime_outputs = [_nd_view_key(key) for key in runtime_output_shapes]
    if Counter(runtime_outputs) != Counter(expected_output_views):
        raise ValueError("runtime output typed ND views differ from complete feature target")
    logits = [index for index, key in enumerate(runtime_outputs) if key == expected_output_views[0]]
    features = [index for index, key in enumerate(runtime_outputs) if key == expected_output_views[81]]
    if len(logits) != 1 or len(features) != 1:
        raise ValueError("logits/feature runtime descriptors are not unique")
    outputs = [{"index": index, "name": "logits" if index == logits[0] else "target_features" if index == features[0] else f"state_output_unmapped_{index}",
        "shape": list(runtime_output_shapes[index][1]), "nd_view_policy": _ND_VIEW_POLICY,
        "compiled_candidate_shapes": [{"output_index": port, "shape": list(expected_outputs[port][1])}
            for port, expected in enumerate(expected_output_views) if expected == key],
        "dtype": key[0], "tool_output_type": _TOOL_TYPES[key[0]],
        "schema_dtype_names": [{"FLOAT32": "DT_FLOAT", "FLOAT16": "DT_FLOAT16"}[key[0]]],
        "bytes": math.prod(runtime_output_shapes[index][1]) * _WIDTHS[key[0]], "semantic_slot_identity": "UNVERIFIED" if index not in {logits[0], features[0]} else "UNIQUE_TYPED_ND_VIEW"}
        for index, key in enumerate(runtime_outputs)]
    # Recheck every original file after preparing all descriptor classes.
    for source in sources.values():
        if _source(source["path"], source["bytes"]) != source:
            raise ValueError("original prefill input changed while constructing the plan")
    for identity in bindings:
        if _read_small(Path(identity["path"]), 32 << 20)[1] != identity:
            raise ValueError("bound prefill/runtime metadata changed while constructing the plan")
    if _source(model["path"], model["bytes"]) != model:
        raise ValueError("compiled primary OMC changed while constructing the plan")
    plan = {"schema_version": 1, "artifact_kind": _KIND, "status": "PREFILL_SMOKE_PLANNED", "gear": 64,
        "profile": _PROFILE, "compilation": str(compilation), "runtime_probe": str(probe_dir), "input_dir": str(input_root),
        "compilation_report_sha256": probe["compilation_report_sha256"], "model_sha256": model["sha256"],
        "runtime_version": probe["runtime_version"], "firmware_id": probe["firmware_id"], "ddk_id": compiled["ddk_id"],
        "tool_binding": tool,
        "metadata_bindings": bindings, "primary_model_binding": model, "source_inputs": sorted(sources.values(), key=lambda row: row["path"]),
        "inputs": inputs, "outputs": outputs, "input_shape": None, "output_shape": None,
        "input_type": ",".join(row["tool_input_type"] for row in inputs),
        "output_type": ",".join(row["tool_output_type"] for row in outputs),
        "input_format": None, "output_format": None,
        "format_policy": "RESTORED_MODEL_ND_DESCRIPTORS_NO_TOOL_OVERRIDE",
        "permutation_equivalence": certificates, "input_mapping_source": "RUNTIME_TYPED_ND_VIEWS_AND_IDENTICAL_SOURCE_BYTES",
        "runtime_input_order": "UNIQUE_TYPED_ND_VIEWS_AND_IDENTICAL_INPUT_VALUE_CLASSES", "nd_view_policy": _ND_VIEW_POLICY,
        "execution_scope": "SINGLE_PREFILL_ONLY_IDENTICAL_STATE_GROUPS", "state_continuity": "NOT_RUN",
        "state_input_layer_identity": "UNVERIFIED", "state_output_layer_identity": "UNVERIFIED",
        "numerical_validation": "NOT_RUN_NO_REFERENCE", "target_features_requested": True, "target_features": "EXPOSED_NOT_VALIDATED",
        "feature_output_evidence": {"status": "EXPOSED_NOT_VALIDATED", "compiled_output_index": 81,
            "runtime_output_index": features[0], "shape": list(runtime_output_shapes[features[0]][1]),
            "compiled_shape": [1, 64, 16384], "nd_view_policy": _ND_VIEW_POLICY, "dtype": "FLOAT16",
            "output_mapping": "RUNTIME_DESCRIPTOR_INDEX_ONLY_TOOL_FILENAMES_UNVERIFIED",
            "semantic_validation": "NOT_RUN", "source_provenance": "COMPILED_BRANCH_SOURCE_RECEIPT_NOT_REESTABLISHED_BY_THIS_PLANNER"},
        "logits_runtime_output_index": logits[0], "feature_runtime_output_index": features[0],
        "scope": "One prefill call only. Only leading/trailing singleton ND axes are byte-equivalent views; actual runtime shapes are retained without changing files or graphs. Every ambiguous class requires identical source bytes. This never proves semantic slot/layer mapping, feedback, accuracy or feature filenames; serialized indices are not runtime-order evidence."}
    if primary_revalidations:
        plan["primary_omc_revalidation_observations"] = primary_revalidations
    return plan


def validate_prefill_smoke_plan(plan):
    """Recompute metadata and input hashes immediately before execution."""
    if not isinstance(plan, dict) or plan.get("artifact_kind") != _KIND or plan.get("status") != "PREFILL_SMOKE_PLANNED":
        raise ValueError("expected an exact first-prefill smoke plan")
    if any(not isinstance(plan.get(key), str) for key in ("compilation", "runtime_probe", "input_dir")):
        raise ValueError("prefill smoke plan requires its original explicit source directories")
    current = prepare_prefill_smoke_plan(plan.get("compilation"), plan.get("runtime_probe"), plan.get("input_dir"))
    if current != plan:
        raise ValueError("prefill smoke evidence or original input identity changed after planning")
    return current

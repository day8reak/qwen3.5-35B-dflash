"""Typed runtime fixtures prove first-call planning only, not device execution."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from qwen35_moe_dflash import target_prefill_smoke_plan as planner
from qwen35_moe_dflash.existing_air import _conversion_options, _stat
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.original_target_board import original_input_files


ENUMS = {"FLOAT32": 103, "FLOAT16": 107, "INT8": 111, "INT16": 113, "INT32": 127, "INT64": 131}
GE = {"FLOAT32": "DT_FLOAT", "FLOAT16": "DT_FLOAT16", "INT8": "DT_INT8", "INT16": "DT_INT16", "INT32": "DT_INT32", "INT64": "DT_INT64"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def binding(path):
    return {"path": str(path), "stat": _stat(path), "sha256": sha(path)}


@pytest.fixture
def case(tmp_path):
    job, probe, source = tmp_path / "compiled-job", tmp_path / "runtime-probe", tmp_path / "original-inputs"
    compilation = job / "compilation"
    source.mkdir()
    profile = get_existing_air_profile("cpro-40block-prefill-64")
    inputs = []
    for index, ((name, raw_shape), (semantic, filename)) in enumerate(zip(
            [row.split(":") for row in profile["input_shape"].split(";")], original_input_files())):
        dtype = "FLOAT16" if name.startswith("RefData_") else planner._INPUT_TYPES[name]
        shape = list(map(int, raw_shape.split(",")))
        inputs.append({"name": name, "type": "Data", "shape": shape, "layout": "ND",
            "schema_dtype_names": [GE[dtype]], "serialized_data_index": (index + 17) % 89})
        path = source / filename
        if not path.exists():
            path.write_bytes(b"\0" * (math.prod(shape) * planner._WIDTHS[dtype]))
    outputs = [{"output_index": 0, "netoutput": "NetOutput", "shape": [1, 1, 248320], "layout": "ND", "schema_dtype_names": ["DT_FLOAT"]}]
    by_name = {row["name"]: row for row in inputs}
    outputs += [{"output_index": index + 1, "netoutput": "NetOutput", "shape": by_name[f"RefData_{index}"]["shape"],
        "layout": "ND", "schema_dtype_names": ["DT_FLOAT16"]} for index in range(80)]
    outputs += [{"output_index": 81, "netoutput": "NetOutput", "shape": [1, 64, 16384], "layout": "ND", "schema_dtype_names": ["DT_FLOAT16"]}]
    model = compilation / "compile-work/generated/model/model.omc"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"opaque host-fixture compiled metadata; no vendor model")
    primary = {"path": model.relative_to(compilation).as_posix(), "bytes": model.stat().st_size,
        "sha256": sha(model), "stat": _stat(model), "role": "primary_omc"}
    compiled = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "model_compilation": "PASSED",
        "returncode": 0, "platform": "kirinx90", "ddk_id": "FIXTURE-DDK", "omc": primary, "generated_artifacts": [primary],
        "source_preservation": {"status": "UNCHANGED_BY_STAT", "changed": []},
        "requested_options": _conversion_options(profile["input_shape"], None, False, True)}
    save(compilation / "report.json", compiled)
    report_sha = sha(compilation / "report.json")
    save(job / "result.json", {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED", "model_compilation": "PASSED",
        "output": str(job), "compilation_report_sha256": report_sha})
    save(job / "compile-profile.json", profile)
    contract = {"artifact_kind": "qwen35-original-target-contract", "status": "OBSERVED", "compilation_report_sha256": report_sha,
        "omc": primary, "boundaries": {"inputs_in_serialized_op_order": inputs, "outputs": outputs,
            "declared_profile_check": {"status": "DIFFERENT_VISIBLE_BOUNDARIES"}}}
    save(job / "target-contract/report.json", contract)
    probe.mkdir()
    def runtime_rows(rows):
        return [{"index": index, "shape": row["shape"], "raw_dtype": ENUMS[planner._GE_TYPES[row["schema_dtype_names"][0]]], "raw_format": 211}
            for index, row in enumerate(reversed(rows))]
    native = {"artifact_kind": "qwen35-hiai-model-io-probe", "evidence_kind": "device-hiai", "status": "IO_DESCRIPTORS_OBSERVED",
        "restore": "PASSED", "restore_init": "NOT_RUN_METADATA_ONLY", "model_load_scope": "RESTORE_METADATA_ONLY",
        "descriptor_stage": "restore-only", "manager_init": "NOT_RUN", "external_weight_initialization": "NOT_REQUESTED",
        "process_called": False, "run_called": False, "tensor_allocations_requested": 0, "model_load_api": "file-v2",
        "model_path": "/data/local/tmp/fixture/metadata/model/model.omc", "model_sha256": primary["sha256"], "model_bytes": primary["bytes"],
        "npu_supported": True, "expected_runtime_version": "fixture-runtime", "runtime_version": "fixture-runtime",
        "firmware_id": "fixture-firmware", "ddk_id": "FIXTURE-DDK", "dtype_enum": "hiai::DataType", "format_enum": "hiai::Format",
        "dtype_enum_values": dict(ENUMS), "format_enum_values": {"ND": 211}, "input_count": 89, "output_count": 82,
        "inputs": runtime_rows(inputs), "outputs": runtime_rows(outputs)}
    save(probe / "model-io.json", native)
    save(probe / "runtime.json", {"runtime_version": "fixture-runtime", "npu_supported": True})
    for filename in ("model-io.exitcode", "runtime.exitcode"):
        (probe / filename).write_text("0\n")
    argv = ["/data/local/tmp/fixture/runner", "--model-io-only", native["model_path"], "--model-sha256", primary["sha256"],
        "--descriptor-stage", "restore-only", "--expected-runtime-version", "fixture-runtime", "--ddk-id", "FIXTURE-DDK",
        "--firmware-id", "fixture-firmware", "--output", "/data/local/tmp/fixture/model-io.json"]
    save(probe / "model-io-command.json", {"argv": argv, "cwd": str(Path(native["model_path"]).parent)})
    help_text = "Usage: tool " + " ".join("--" + name for name in ("model", "input", "output_dir", "modelmanager_itf", "direct_from_file", "weight_dir", "times", "input_type", "output_type", "input_format", "output_format"))
    help_report = tmp_path / "tool-help/report.json"
    save(help_report, {"artifact_kind": "qwen35-target-board-tool-inspection", "status": "INSPECTED",
        "tool": "/data/local/tmp/tools_sysdbg/model_run_tool_internal", "tool_identity": "SHA256_MATCHED_BEFORE_AND_AFTER",
        "tool_sha256_before": "a" * 64, "tool_sha256_after": "a" * 64, "model_upload": "NOT_RUN", "model_load": "NOT_RUN", "model_inference": "NOT_RUN",
        "firmware": {"returncode": 0, "text": "fixture-firmware"}, "help": {"observation": "HELP_TEXT_OBSERVED", "text": help_text,
            "utf8_bytes": len(help_text.encode()), "utf8_sha256": hashlib.sha256(help_text.encode()).hexdigest(), "truncated": False}})
    outer = {"artifact_kind": "qwen35-target-model-io-probe", "status": "IO_METADATA_OBSERVED", "job": str(job), "profile": profile["name"],
        "sequence_length": 64, "compilation_report_sha256": report_sha, "remote_exitcode": 0,
        "lock_state": "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT", "retrieval_errors": [],
        "source_preservation": "UNCHANGED_METADATA_AND_SMALL_PRIMARY_SHA256; WEIGHTS_NOT_OPENED",
        "compiled_boundary_count_check": "MATCHED_89_INPUT_82_OUTPUT_COUNTS_ONLY",
        "model_path": native["model_path"], "model_sha256": primary["sha256"], "model_bytes": primary["bytes"],
        "runtime_version": "fixture-runtime", "firmware_id": "fixture-firmware", "ddk_id": "FIXTURE-DDK",
        "source_bindings": [binding(path) for path in (job / "result.json", compilation / "report.json", job / "compile-profile.json", job / "target-contract/report.json", help_report)],
        "retrieved_sha256": {name: sha(probe / name) for name in ("model-io.json", "runtime.json", "model-io.exitcode", "runtime.exitcode")}}
    for key in ("inputs", "outputs", "input_count", "output_count", "dtype_enum", "format_enum", "dtype_enum_values", "format_enum_values"):
        outer[key] = deepcopy(native[key])
    save(probe / "result.json", outer)
    return {"job": job, "compilation": compilation, "probe": probe, "source": source, "compiled": compiled,
        "contract": contract, "native": native, "outer": outer, "help_report": help_report, "model": model}


def prepare(case):
    return planner.prepare_prefill_smoke_plan(case["compilation"], case["probe"], case["source"])


def rewrite_native(case):
    save(case["probe"] / "model-io.json", case["native"])
    case["outer"]["retrieved_sha256"]["model-io.json"] = sha(case["probe"] / "model-io.json")
    for key in ("inputs", "outputs", "input_count", "output_count", "dtype_enum", "format_enum", "dtype_enum_values", "format_enum_values"):
        case["outer"][key] = deepcopy(case["native"][key])
    save(case["probe"] / "result.json", case["outer"])


def test_permuted_runtime_indices_and_same_byte_states_plan_only_one_call(case):
    plan = prepare(case)
    assert plan["status"] == "PREFILL_SMOKE_PLANNED" and len(plan["inputs"]) == 89 and len(plan["outputs"]) == 82
    assert plan["inputs"][0]["name"] == "arg18_1" and plan["inputs"][0]["tool_input_type"] == "INT16"
    assert plan["logits_runtime_output_index"] == 81 and plan["feature_runtime_output_index"] == 0
    assert plan["outputs"][0]["bytes"] == 2097152  # Same byte count as KV; exact rank distinguishes it.
    assert plan["state_input_layer_identity"] == plan["state_output_layer_identity"] == "UNVERIFIED"
    assert plan["state_continuity"] == "NOT_RUN" and plan["input_shape"] is plan["output_shape"] is None
    # KV positions[64] and position_ids[1,64] are the same ND view and contain
    # identical original bytes. Their semantic identities remain unclaimed.
    assert len(plan["source_inputs"]) == 13 and len(plan["permutation_equivalence"]) == 11
    assert plan["nd_view_policy"] == "EDGE_SINGLETON_ND_LINEAR_VIEW_ONLY"
    assert plan["tool_binding"]["sha256"] == "a" * 64
    assert len(plan["input_type"].split(",")) == 89 and len(plan["output_type"].split(",")) == 82
    assert planner.validate_prefill_smoke_plan(plan) == plan


def test_distinct_kv_initial_bytes_reject_ambiguous_same_shape_state_class(case):
    path = case["source"] / "past_v.bin"
    with path.open("r+b") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="different actual bytes"):
        prepare(case)


@pytest.mark.parametrize("field,value", [
    ("lock_state", "RETAINED_OR_UNCERTAIN"), ("lock_release_error", "fixture owner mismatch"),
    ("native_evidence_error", "fixture native hash mismatch"), ("source_error", "fixture source mutation"),
    ("retrieval_errors", [{"path": "runtime.stderr", "error": "fixture missing evidence"}]),
    ("source_preservation", "FAILED"), ("compiled_boundary_count_check", "DIFFERENT_COUNTS_RAW_METADATA_RETAINED"),
])
def test_observed_probe_is_not_ready_while_lock_or_final_audit_is_unresolved(case, field, value):
    case["outer"][field] = value
    save(case["probe"] / "result.json", case["outer"])
    with pytest.raises(ValueError, match="restore-only runtime probe"):
        prepare(case)


@pytest.mark.parametrize("fault", ["missing_int8", "duplicate_enum", "ge_domain", "format", "count", "padding", "runtime_index", "feature_rank", "dtype_bool", "shape_bool", "manager_init", "inference", "fake", "model_hash", "runtime_version"])
def test_native_typed_or_scope_disagreement_is_never_inferred_from_indices(case, fault):
    native = case["native"]
    if fault == "missing_int8": native["dtype_enum_values"].pop("INT8")
    elif fault == "duplicate_enum": native["dtype_enum_values"]["INT8"] = native["dtype_enum_values"]["INT32"]
    elif fault == "ge_domain": native["dtype_enum"] = "ge.proto.DataType"
    elif fault == "format": native["inputs"][0]["raw_format"] = 999
    elif fault == "count": native["input_count"] = 88
    elif fault == "padding": native["inputs"][0]["shape"] = [1, 2]
    elif fault == "runtime_index": native["inputs"][0]["index"] = 1
    elif fault == "feature_rank": native["outputs"][0]["shape"] = [64, 32, 32, 16]
    elif fault == "dtype_bool": native["inputs"][0]["raw_dtype"] = True
    elif fault == "shape_bool": native["inputs"][0]["shape"] = [True]
    elif fault == "manager_init": native["manager_init"] = "PASSED"
    elif fault == "inference": native["run_called"] = True
    elif fault == "fake": native["evidence_kind"] = "host-fake-hiai"
    elif fault == "model_hash": native["model_sha256"] = "b" * 64
    else: native["runtime_version"] = "other-runtime"
    rewrite_native(case)
    with pytest.raises(ValueError):
        prepare(case)


@pytest.mark.parametrize("direction,side", [("inputs", "leading"), ("inputs", "trailing"),
    ("inputs", "both"), ("outputs", "leading"), ("outputs", "trailing"), ("outputs", "both")])
def test_edge_singleton_nd_views_keep_actual_runtime_shape_and_unchanged_input_files(case, direction, side):
    original_inputs = {path.name: (path.stat().st_size, sha(path), _stat(path)) for path in case["source"].iterdir()}
    for row in case["native"][direction]:
        row["shape"] = ([1] if side in {"leading", "both"} else []) + row["shape"] + ([1] if side in {"trailing", "both"} else [])
    rewrite_native(case)
    plan = prepare(case)
    assert [row["shape"] for row in plan[direction]] == [row["shape"] for row in case["native"][direction]]
    assert {path.name: (path.stat().st_size, sha(path), _stat(path)) for path in case["source"].iterdir()} == original_inputs
    assert planner.validate_prefill_smoke_plan(plan) == plan


def test_ordinary_arguments_with_same_actual_bytes_can_share_a_singleton_nd_view_class(case):
    matches = [row for row in case["native"]["inputs"] if row["raw_dtype"] == ENUMS["INT32"] and row["shape"] in ([64], [1, 64])]
    assert len(matches) == 2
    for row in matches:
        row["shape"] = [1, 1, 64, 1]
    rewrite_native(case)
    plan = prepare(case)
    certificate = next(row for row in plan["permutation_equivalence"] if set(row["compiled_candidates"]) == {"arg2_1", "arg107_1"})
    assert len(certificate["runtime_indices"]) == 2 and certificate["canonical_shape"] == [64]
    assert {tuple(row["shape"]) for row in certificate["source_shapes"]} == {(64,), (1, 64)}
    assert all(plan["inputs"][index]["shape"] == [1, 1, 64, 1] for index in certificate["runtime_indices"])
    assert all(plan["inputs"][index]["semantic_slot_identity"] == "UNVERIFIED" for index in certificate["runtime_indices"])


def test_ordinary_view_equivalence_rejects_different_values_even_when_exact_ranks_were_distinct(case):
    path = case["source"] / "position_ids.bin"
    with path.open("r+b") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="different actual bytes"):
        prepare(case)


@pytest.mark.parametrize("fault", ["inner_one", "axis_order", "nonunit_padding", "raw_empty"])
def test_nd_view_policy_never_removes_inner_axes_reorders_data_or_infers_empty_descriptors(case, fault):
    conv = next(row for row in case["native"]["inputs"] if row["shape"] == [1, 8192, 4])
    if fault == "inner_one": conv["shape"] = [1, 8192, 1, 4]
    elif fault == "axis_order": conv["shape"] = [1, 4, 8192]
    elif fault == "nonunit_padding": conv["shape"] = [2, 8192, 4]
    else: case["native"]["inputs"][0]["shape"] = []
    rewrite_native(case)
    with pytest.raises(ValueError):
        prepare(case)


def test_unique_feature_and_logits_match_nd_views_by_typed_dimensions_not_equal_byte_counts(case):
    case["native"]["outputs"][0]["shape"] = [1, 1, 64, 16384, 1]
    case["native"]["outputs"][-1]["shape"] = [248320]
    rewrite_native(case)
    plan = prepare(case)
    assert plan["feature_runtime_output_index"] == 0 and plan["logits_runtime_output_index"] == 81
    assert plan["outputs"][0]["shape"] == [1, 1, 64, 16384, 1]
    assert plan["outputs"][-1]["shape"] == [248320]
    assert plan["feature_output_evidence"]["compiled_shape"] == [1, 64, 16384]
    assert plan["feature_output_evidence"]["shape"] == [1, 1, 64, 16384, 1]


@pytest.mark.parametrize("fault", ["native_hash", "outer_arrays", "exitcode", "command", "source_binding", "help_hash", "help_flags", "profile", "model_bytes", "input_size", "input_symlink"])
def test_bound_receipts_and_original_inputs_are_required(case, fault):
    if fault == "native_hash":
        (case["probe"] / "model-io.json").write_bytes((case["probe"] / "model-io.json").read_bytes() + b" ")
    elif fault == "outer_arrays":
        case["outer"]["inputs"][0]["raw_dtype"] = ENUMS["FLOAT16"]
        save(case["probe"] / "result.json", case["outer"])
    elif fault == "exitcode": (case["probe"] / "model-io.exitcode").write_text("1\n")
    elif fault == "command":
        path = case["probe"] / "model-io-command.json"
        value = json.loads(path.read_text()); value["argv"] += ["--input", "dummy"]
        save(path, value)
    elif fault == "source_binding":
        case["outer"]["source_bindings"][0]["sha256"] = "b" * 64
        save(case["probe"] / "result.json", case["outer"])
    elif fault in {"help_hash", "help_flags"}:
        path = case["help_report"]
        value = json.loads(path.read_text())
        if fault == "help_hash": value["tool_sha256_after"] = "b" * 64
        else: value["help"]["text"] = "Usage: unrelated --help"
        save(path, value)
    elif fault == "profile": save(case["job"] / "compile-profile.json", get_existing_air_profile("cpro-40block-decode-1"))
    elif fault == "model_bytes": case["model"].write_bytes(case["model"].read_bytes() + b"x")
    elif fault == "input_size": (case["source"] / "effective_length.bin").write_bytes(b"x")
    else:
        path = case["source"] / "past_k.bin"
        path.unlink(); path.symlink_to(case["source"] / "past_v.bin")
    with pytest.raises((ValueError, OSError)):
        prepare(case)


def test_pre_execution_validation_rehashes_inputs_and_preserves_changed_evidence(case):
    plan = prepare(case)
    path = case["source"] / "inputs_embeds.bin"
    with path.open("r+b") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="changed after planning"):
        planner.validate_prefill_smoke_plan(plan)
    assert path.read_bytes()[0:1] == b"x"


def test_plan_never_opens_external_weight_payload(case, monkeypatch):
    huge = case["compilation"] / "compile-work/generated/model/SubGraph_0.weight"
    huge.write_bytes(b"small fixture representing unaccessed source")
    case["compiled"]["generated_artifacts"].append({"path": huge.relative_to(case["compilation"]).as_posix(),
        "bytes": 19_000_000_000, "sha256": "f" * 64, "stat": [0, 0, 19_000_000_000, 0, 0]})
    save(case["compilation"] / "report.json", case["compiled"])
    new_sha = sha(case["compilation"] / "report.json")
    for path in (case["job"] / "result.json", case["job"] / "target-contract/report.json", case["probe"] / "result.json"):
        value = json.loads(path.read_text()); value["compilation_report_sha256"] = new_sha
        save(path, value)
    case["outer"] = json.loads((case["probe"] / "result.json").read_text())
    case["outer"]["source_bindings"] = [binding(Path(row["path"])) for row in case["outer"]["source_bindings"]]
    save(case["probe"] / "result.json", case["outer"])
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path != huge, "planner must never read source weights"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    assert prepare(case)["status"] == "PREFILL_SMOKE_PLANNED"

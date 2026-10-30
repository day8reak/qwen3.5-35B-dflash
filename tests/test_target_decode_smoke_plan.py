"""Named decode1 planning uses tiny metadata models and no vendor/device calls."""
import math
from pathlib import Path
import struct

import pytest

from qwen35_moe_dflash import target_decode_smoke_plan as planner
from qwen35_moe_dflash.existing_air import _conversion_options, _stat
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.original_target_board import original_input_files
from test_target_prefill_smoke_plan import (ENUMS, binding, case as prefill_case, save, sha,
    rewrite_native)


@pytest.fixture
def case(tmp_path):
    value = prefill_case.__wrapped__(tmp_path)
    profile = get_existing_air_profile("cpro-40block-decode-1")
    entries = [row.split(":") for row in profile["input_shape"].split(";")]
    shapes = {name: list(map(int, raw.split(","))) for name, raw in entries}
    inputs = value["contract"]["boundaries"]["inputs_in_serialized_op_order"]
    seen = set()
    for row, ((name, _), (_, filename)) in zip(inputs, zip(entries, original_input_files())):
        row["shape"] = shapes[name]
        path = value["source"] / filename
        if filename not in seen:
            # Distinct common K/V payloads are intentionally allowed. Literal
            # runtime input names disambiguate them even when geometry agrees.
            dtype = planner._GE_TYPES[row["schema_dtype_names"][0]]
            marker = {"past_k.bin": 1, "past_v.bin": 2, "past_conv_state.bin": 3,
                      "past_recurrent_state.bin": 4}.get(filename, 0)
            path.write_bytes(bytes([marker]) * (math.prod(row["shape"]) * planner._WIDTHS[dtype]))
            seen.add(filename)
    (value["source"] / "new_kv_cache_pos.bin").write_bytes(struct.pack("<i", 64))
    (value["source"] / "position_ids.bin").write_bytes(struct.pack("<i", 63))
    (value["source"] / "output_pos.bin").write_bytes(struct.pack("<i", 0))
    (value["source"] / "allQLen.bin").write_bytes(struct.pack("<q", 65))
    (value["source"] / "effective_length.bin").write_bytes(struct.pack("<h", 1))
    outputs = value["contract"]["boundaries"]["outputs"]
    outputs[81]["shape"] = [1, 1, 16384]
    value["compiled"]["requested_options"] = _conversion_options(profile["input_shape"], None, False, True)
    save(value["compilation"] / "report.json", value["compiled"])
    report_sha = sha(value["compilation"] / "report.json")
    save(value["job"] / "result.json", {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED",
        "model_compilation": "PASSED", "output": str(value["job"]), "compilation_report_sha256": report_sha})
    save(value["job"] / "compile-profile.json", profile)
    value["contract"]["compilation_report_sha256"] = report_sha
    save(value["job"] / "target-contract/report.json", value["contract"])
    def runtime_rows(rows, direction):
        return [{"index": index, "shape": [1, *row["shape"]],
            "raw_dtype": ENUMS[planner._GE_TYPES[row["schema_dtype_names"][0]]], "raw_format": 211,
            "tensor_name": row["name"] if direction == "inputs" else f"output_{row['output_index']}_fixture",
            "tensor_name_status": "OBSERVED"} for index, row in enumerate(reversed(rows))]
    value["native"]["inputs"] = runtime_rows(inputs, "inputs")
    value["native"]["outputs"] = runtime_rows(outputs, "outputs")
    value["outer"].update(profile=profile["name"], sequence_length=1, compilation_report_sha256=report_sha,
        runtime_tensor_names="OBSERVED_ALL")
    value["outer"]["source_bindings"] = [binding(path) for path in (value["job"] / "result.json",
        value["compilation"] / "report.json", value["job"] / "compile-profile.json",
        value["job"] / "target-contract/report.json", value["help_report"])]
    rewrite_native(value)
    return value


def prepare(case):
    return planner.prepare_decode_smoke_plan(case["compilation"], case["probe"], case["source"])


def test_literal_named_runtime_inputs_bind_distinct_common_states_without_prefill_feedback(case):
    before = {path.name: (sha(path), _stat(path)) for path in case["source"].iterdir()}
    plan = prepare(case)
    assert plan["status"] == "DECODE_SMOKE_PLANNED" and plan["gear"] == 1
    assert plan["execution_scope"] == "SINGLE_DECODE1_USER_SUPPLIED_INPUTS_NO_PREFILL_FEEDBACK"
    assert plan["source_semantics"] == "UNVERIFIED_USER_INPUTS" and plan["state_continuity"] == "NOT_VALIDATED"
    assert plan["model_payload_access"] == plan["weight_payload_access"] == "NONE"
    assert len(plan["inputs"]) == 89 and len(plan["outputs"]) == 82
    assert plan["inputs"][0]["name"] == "arg18_1"
    assert plan["inputs"][0]["source_filename"] == "effective_length.bin"
    by_name = {row["name"]: row for row in plan["inputs"]}
    assert by_name["RefData_6"]["source_filename"] == "past_k.bin"
    assert by_name["RefData_7"]["source_filename"] == "past_v.bin"
    assert by_name["RefData_6"]["sha256"] != by_name["RefData_7"]["sha256"]
    assert by_name["arg107_1"]["sha256"] != by_name["arg2_1"]["sha256"]
    assert len(plan["source_inputs"]) == 13 and "permutation_equivalence" not in plan
    assert plan["feature_runtime_output_index"] == 0 and plan["logits_runtime_output_index"] == 81
    assert plan["outputs"][0]["shape"] == [1, 1, 1, 16384]
    assert plan["input_format"] is plan["output_format"] is plan["input_shape"] is plan["output_shape"] is None
    assert len(plan["input_type"].split(",")) == 89 and len(plan["output_type"].split(",")) == 82
    assert plan["warnings"] == []
    controls = {Path(row["path"]).name: row for row in plan["control_observations"]}
    assert controls["output_pos.bin"]["values"] == [0]
    assert controls["allQLen.bin"]["values"] == [65]
    assert controls["effective_length.bin"]["values"] == [1]
    assert all(row["interpretation"] == "UNINTERPRETED_USER_VALUES" for row in controls.values())
    assert planner.validate_decode_smoke_plan(plan) == plan
    assert {path.name: (sha(path), _stat(path)) for path in case["source"].iterdir()} == before


def test_planning_does_not_open_omc_or_weight_payloads(case, monkeypatch):
    import os
    original = os.open
    def guarded(path, *args, **kwargs):
        if str(path).endswith((".omc", ".weight")):
            raise AssertionError("planner opened model or weight payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", guarded)
    original_path_open = Path.open
    def guarded_path(path, *args, **kwargs):
        if path.suffix in {".omc", ".weight"}:
            raise AssertionError("planner opened model or weight payload")
        return original_path_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_path)
    assert prepare(case)["status"] == "DECODE_SMOKE_PLANNED"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_name", "unobserved", "anonymous_output", "type", "shape", "interior_singleton"])
def test_no_positional_or_same_extent_fallback_when_named_runtime_contract_disagrees(case, fault):
    native = case["native"]
    if fault == "missing": native["inputs"][0].pop("tensor_name")
    elif fault == "duplicate": native["inputs"][0]["tensor_name"] = native["inputs"][1]["tensor_name"]
    elif fault == "wrong_name": native["inputs"][0]["tensor_name"] = "other_name"
    elif fault == "unobserved": native["inputs"][0]["tensor_name_status"] = "UNAVAILABLE"
    elif fault == "anonymous_output": native["outputs"][0].pop("tensor_name")
    elif fault == "type": native["inputs"][0]["raw_dtype"] = ENUMS["INT32"]
    elif fault == "shape": native["inputs"][0]["shape"] = [2]
    elif fault == "interior_singleton":
        row = next(row for row in native["inputs"] if row["tensor_name"] == "RefData_1")
        row["shape"] = [1, 32, 1, 128, 128]
    rewrite_native(case)
    with pytest.raises(ValueError):
        prepare(case)


@pytest.mark.parametrize("fault", ["profile", "unreleased", "fake", "runtime", "enum", "compiled_type", "feature"])
def test_failed_or_wrong_profile_runtime_evidence_is_not_accepted(case, fault):
    if fault == "profile":
        case["outer"]["profile"] = "cpro-40block-prefill-64"
        save(case["probe"] / "result.json", case["outer"])
    elif fault == "unreleased":
        case["outer"]["lock_state"] = "RETAINED_OR_UNCERTAIN"
        save(case["probe"] / "result.json", case["outer"])
    elif fault == "fake":
        case["native"]["evidence_kind"] = "host-fake-hiai"
        rewrite_native(case)
    elif fault == "runtime":
        case["native"]["runtime_version"] = "different"
        rewrite_native(case)
    elif fault == "enum":
        case["native"]["dtype_enum"] = "ge.proto.DataType"
        rewrite_native(case)
    elif fault == "compiled_type":
        case["contract"]["boundaries"]["inputs_in_serialized_op_order"][0]["type"] = "RefData"
        save(case["job"] / "target-contract/report.json", case["contract"])
        case["outer"]["source_bindings"] = [binding(Path(row["path"])) for row in case["outer"]["source_bindings"]]
        save(case["probe"] / "result.json", case["outer"])
    else:
        case["native"]["outputs"][0]["shape"] = [1, 64, 32, 32, 16]
        rewrite_native(case)
    with pytest.raises(ValueError):
        prepare(case)


@pytest.mark.parametrize("fault", ["input", "source_metadata", "model_stat", "plan_control", "scope"])
def test_immediate_validation_rejects_source_plan_or_model_identity_changes(case, fault):
    plan = prepare(case)
    if fault == "input":
        path = case["source"] / "output_pos.bin"
        path.write_bytes(struct.pack("<i", 1))
    elif fault == "source_metadata":
        case["outer"]["firmware_id"] = "changed"
        save(case["probe"] / "result.json", case["outer"])
    elif fault == "model_stat": case["model"].write_bytes(b"changed model")
    elif fault == "plan_control": plan["control_observations"][0]["values"] = [999]
    else: plan["execution_scope"] = "PREFILL_STATE_FEEDBACK"
    with pytest.raises(ValueError):
        planner.validate_decode_smoke_plan(plan)


def test_symlink_supplied_input_is_never_followed(case):
    source = case["source"] / "past_k.bin"
    backup = case["source"] / "original-past-k.bin"
    source.rename(backup)
    source.symlink_to(backup)
    with pytest.raises(ValueError, match="symlink"):
        prepare(case)


def test_changed_decode_control_values_are_observed_without_interpretation_or_repair(case):
    path = case["source"] / "output_pos.bin"
    path.write_bytes(struct.pack("<i", -7))
    plan = prepare(case)
    assert plan["warnings"] == []
    observed = next(row for row in plan["control_observations"] if row["path"] == str(path))
    assert observed["values"] == [-7]
    assert path.read_bytes() == struct.pack("<i", -7)

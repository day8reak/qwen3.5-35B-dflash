"""Same-model graph/file joins reject ordinal guesses and stale evidence.

Graph decoding and named runtime validation use the real existing audit
primitives. The separately tested output-inventory validator is isolated in
these fixtures; its returned output files are freshly hashed on every call.
"""
import hashlib
import json
import math
from pathlib import Path
import runpy

import pytest

from qwen35_moe_dflash import target_decode_state_lineage as audit
from qwen35_moe_dflash import target_state_lineage as lineage
from qwen35_moe_dflash.target_compile_failure import _stamp
from qwen35_moe_dflash.target_prefill_smoke_plan import _runtime_keys
from test_target_state_lineage import jobs as graph_jobs, save


@pytest.fixture
def case(tmp_path, monkeypatch):
    jobs = graph_jobs.__wrapped__(tmp_path, monkeypatch)
    probe = json.loads((jobs["decode_probe"] / "result.json").read_text())
    execution, inspection_dir = tmp_path / "decode-execution", tmp_path / "decode-output-inspection"
    files = []
    for index, (dtype, shape) in enumerate(_runtime_keys(probe, "outputs")):
        # Unrelated numeric spellings ensure filenames never establish indices.
        path = execution / "board/outputs" / f"unrelated_{(index * 37) % 83:03d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        width = math.prod(shape) * {"FLOAT16": 2, "FLOAT32": 4}[dtype]
        with path.open("wb") as stream:
            stream.truncate(width)
        files.append({"runtime_index": index, "path": path.relative_to(execution / "board").as_posix(),
            "bytes": width, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "stat": _stamp(path.stat()),
            "slot_binding": "EXPLICIT_TOOL_LOG_PATH_AND_DESCRIPTOR_EXTENT"})
    plan = {"profile": "cpro-40block-decode-1", "gear": 1, "sequence_length": 1,
        "compilation": str(jobs["decode_job"] / "compilation"), "runtime_probe": str(jobs["decode_probe"]),
        **{key: probe[key] for key in ("compilation_report_sha256", "model_sha256", "runtime_version", "firmware_id", "ddk_id")}}
    for direction in ("inputs", "outputs"):
        plan[direction] = []
        for index, ((dtype, shape), row) in enumerate(zip(_runtime_keys(probe, direction), probe[direction])):
            plan[direction].append({"index": index, "name": row["tensor_name"], "runtime_tensor_name": row["tensor_name"],
                "dtype": dtype, "shape": row["shape"], "bytes": math.prod(shape) * {"FLOAT16": 2, "FLOAT32": 4}[dtype]})
    save(execution / "plan.json", plan)
    report = {"artifact_kind": "qwen35-target-decode-output-inventory", "status": "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS",
        "output": str(inspection_dir), "execution": str(execution), "output_files": files,
        "trial_inputs": {"status": "FRESHLY_REVALIDATED_UNVERIFIED_EXPERIMENT", "assumption": {"contract_status": "UNVERIFIED"}},
        "expected_runtime_outputs": [{key: row[key] for key in ("index", "shape", "dtype", "bytes")} for row in plan["outputs"]]}
    save(inspection_dir / "report.json", report)

    def validate(directory):
        value = json.loads((Path(directory) / "report.json").read_text())
        if value["status"] != "DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS":
            raise ValueError("saved output inventory is not bound")
        for row in value["output_files"]:
            path = execution / "board" / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"] or _stamp(path.stat()) != row["stat"]:
                raise ValueError("fresh output inventory hash or stat changed")
        return value

    monkeypatch.setattr(audit, "validate_target_decode_output_inventory", validate)
    return {**jobs, "execution": execution, "inspection_dir": inspection_dir, "inspection": report,
        "plan": plan, "probe": probe, "out": tmp_path / "decode-lineage", "validate": validate}


def inspect(case):
    return audit.inspect_target_decode_state_lineage(case["decode_job"], case["decode_probe"], case["inspection_dir"], case["out"])


def test_all80_proven_owner_file_input_joins_follow_reordered_graphs_and_names(case):
    result = inspect(case)
    assert result["status"] == "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED", result
    assert result["same_model_state_mapping"] == "ALL80_SAME_MODEL_STATE_FILES_BOUND"
    mapping = result["decode"]["state_file_mapping"]
    assert mapping["bound_count"] == 80 and len(mapping["states"]) == 80
    assert {row["source_refdata"] for row in mapping["states"]} == lineage._STATES
    first = mapping["states"][0]
    assert first["source_refdata"] == "RefData_79" and first["source_output_index"] == 1
    assert first["runtime_output_index"] != first["source_output_index"]
    assert first["runtime_input_name"] == first["source_refdata"]
    actual = case["inspection"]["output_files"][first["runtime_output_index"]]
    assert first["output_file"]["relative_path"] == actual["path"]
    assert first["output_file"]["sha256"] == actual["sha256"]
    assert result["trial_inputs"]["assumption"]["contract_status"] == "UNVERIFIED"
    assert result["feedback_ready"] is False and result["state_feedback"] == "NOT_RUN"
    assert result["state_continuity"] == "NOT_VALIDATED" and result["numerical_validation"] == "NOT_RUN"
    assert result["cross_model_state_identity"] == "NOT_EVALUATED_SAME_MODEL_ONLY"
    assert result["gap_summary"] == []


@pytest.mark.parametrize("fault,reason", [
    ("fx", "MISSING_OR_NONUNIQUE_EXPLICIT_FX_TENSOR_IDENTITY"),
    ("descriptor_name", "MISSING_OR_NONUNIQUE_LITERAL_COMPILED_AND_RUNTIME_TENSOR_NAMES"),
    ("typed_shape", "SOURCE_COMPILED_STATE_TYPED_VIEW_DIFFERS"),
    ("source_shape", "SOURCE_COMPILED_STATE_TYPED_VIEW_DIFFERS"),
    ("producer_shape", "SOURCE_PRODUCER_SHAPE_UNMATERIALIZED"),
])
def test_missing_identity_or_geometry_reports_readable_gaps_without_ordinal_fallback(case, fault, reason):
    source = case["documents"][b"decode SOURCE"]["graph"][0]["op"]
    compiled = case["documents"][b"decode COMPILED"]["graph"][0]["op"]
    if fault == "fx":
        next(row for row in compiled if row["name"] == "NetOutput")["input_desc"][0].pop("attr", None)
        next(row for row in compiled if row["name"] == "opaque-task")["output_desc"][0].pop("attr", None)
    elif fault == "descriptor_name":
        for node, field in (("NetOutput", "input_desc"), ("opaque-task", "output_desc")):
            next(row for row in compiled if row["name"] == node)[field][0].pop("name")
    elif fault == "typed_shape":
        for node, field in (("NetOutput", "input_desc"), ("opaque-task", "output_desc")):
            next(row for row in compiled if row["name"] == node)[field][0]["shape"] = {"dim": [2, 8192, 4]}
    elif fault == "source_shape":
        net = next(row for row in source if row["name"] == "NetOutput")
        net["input_desc"][1]["shape"] = {"dim": [1, 8192, 4]}
        node = net["input"][1].rsplit(":", 1)[0]
        next(row for row in source if row["name"] == node)["output_desc"][0]["shape"] = {"dim": [1, 8192, 4]}
    else:
        next(row for row in source if row["name"] == "state_cast_0")["output_desc"][0]["shape"] = {}
    result = inspect(case)
    assert result["status"] == ("ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED" if fault == "producer_shape"
        else "DECODE_STATE_LINEAGE_OBSERVED_WITH_GAPS"), result
    assert any(row["reason"] == reason for row in result["gap_summary"])
    assert all(0 < len(row["examples"]) <= 3 for row in result["gap_summary"])
    assert result["feedback_ready"] is False
    if fault == "producer_shape":
        assert result["decode"]["source_lineage"]["observed_state_owner_count"] == 80
        assert result["decode"]["source_lineage"]["fixed_state_output_producer_shape_count"] == 79
        assert result["decode"]["source_lineage"]["status"] == "STATE_LINEAGE_PARTIAL"
        assert result["decode"]["state_file_mapping"]["bound_count"] == 80
        assert next(row for row in result["gap_summary"] if row["reason"] == reason)["blocks_state_mapping"] is False
        assert next(row for row in source if row["name"] == "state_cast_0")["output_desc"][0]["shape"] == {}
    else:
        assert result["decode"]["state_file_mapping"]["bound_count"] == 79


def test_duplicate_same_shaped_owner_never_promotes_all80(case):
    source = case["documents"][b"decode SOURCE"]["graph"][0]["op"]
    next(row for row in source if row["name"] == "update_2")["input"][1] = "RefData_0:0"
    result = inspect(case)
    assert result["status"] == "DECODE_STATE_LINEAGE_OBSERVED_WITH_GAPS"
    assert result["decode"]["source_lineage"]["unique_state_owner_count"] == 79
    assert result["same_model_state_mapping"] == "STATE_FILE_BINDING_PARTIAL"
    assert result["feedback_ready"] is False
    assert any(row["reason"] == "DUPLICATE_SOURCE_STATE_OWNERS" for row in result["gap_summary"])


@pytest.mark.parametrize("fault", ["model_sha256", "compilation", "runtime_probe", "runtime_name", "descriptor_shape"])
def test_actual_output_execution_must_bind_same_job_probe_and_typed_descriptors(case, fault):
    plan = case["plan"]
    if fault in {"model_sha256", "compilation", "runtime_probe"}:
        plan[fault] = "different-model-or-artifact"
    elif fault == "runtime_name": plan["inputs"][0]["runtime_tensor_name"] = "another-name"
    else: plan["outputs"][0]["shape"] = [2, 1, 16384]
    save(case["execution"] / "plan.json", plan)
    result = inspect(case)
    assert result["status"] == "FAILED" and result["stage"] == "bind-executed-state-files"
    assert result["same_model_state_mapping"] == "UNVERIFIED_EVIDENCE_FAILURE"
    assert result["feedback_ready"] is False


def test_revalidation_detects_state_output_mutation_during_graph_read(case, monkeypatch):
    original = audit._decode_job
    def mutate(*args):
        result = original(*args)
        path = case["execution"] / "board" / case["inspection"]["output_files"][3]["path"]
        with path.open("r+b") as stream:
            stream.write(b"changed")
        return result
    monkeypatch.setattr(audit, "_decode_job", mutate)
    result = inspect(case)
    assert result["status"] == "FAILED" and "changed" in result["error"]


def test_weights_are_never_opened_and_no_feedback_or_device_call_is_requested(case, monkeypatch):
    original = Path.open
    def guard(path, *args, **kwargs):
        if path.suffix == ".weight":
            raise AssertionError("state audit opened a weight")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("state audit invoked a subprocess"))
    result = inspect(case)
    assert result["status"] == "ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED"
    assert result["device_calls"] == result["weight_payload_access"] == "NONE"
    assert result["model_payload_access"] == "SMALL_BOUND_AIR_AND_PRIMARY_OMC_GRAPH_ONLY"


def test_cli_is_compact_and_reports_counts_without_dumping80states(case, capsys):
    path = Path(__file__).parents[1] / "targets/cpro/scripts/inspect_target_decode_state_lineage.py"
    cli = runpy.run_path(str(path))
    assert cli["main"](["--job", str(case["decode_job"]), "--runtime-probe", str(case["decode_probe"]),
        "--output-inspection", str(case["inspection_dir"]), "--output", str(case["out"])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["bound_state_files"] == result["counts"]["unique_source_owners"] == 80
    assert "decode" not in result and result["gap_summary"] == []
    with pytest.raises(FileExistsError):
        inspect(case)

"""Source-bound direct reference reuse and explicitly promoted wire fixtures."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash import draft_direct_job as subject
from qwen35_moe_dflash.draft_board import _record as _file_record
from qwen35_moe_dflash.draft_reference import DRAFT_REFERENCE_CONTRACT
from qwen35_moe_dflash.hashing import sha256_file


def _record(path, root, **fields):
    return {**fields, **_file_record(path, root)}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return _record(path, path.parent)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    job, prepared, fixture = tmp_path / "job", tmp_path / "job/prepared", tmp_path / "job/fixture"
    prior = job / "layer-trace"
    for folder in (prepared, fixture, prior / "reference"):
        folder.mkdir(parents=True, exist_ok=True)
    _write_json(job / "result.json", {"artifact_kind": "qwen35-full-draft-job", "note": "host fixture"})
    _write_json(prepared / "prepared.json", {"artifact_kind": "qwen35-draft-ge-prepared", "status": "PREPARED"})
    inputs = []
    for index, (name, dtype, shape) in enumerate((("target_features", "float16", [1, 6, 8]),
            ("input_ids", "int32", [1, 6]), ("attention_mask", "int32", [1, 6]))):
        path = fixture / (name + ".bin")
        np.zeros(shape, dtype=dtype).tofile(path)
        inputs.append(_record(path, fixture, index=index, name=name, dtype=dtype, shape=shape))
    values = {"draft_top1": np.arange(15, dtype="<i4").reshape(1, 15),
              "draft_hidden": np.arange(120, dtype="<f2").reshape(1, 15, 8),
              "layer_0_output": np.arange(128, dtype="<f2").reshape(1, 16, 8)}
    values["layer_1_input"] = values["layer_0_output"].copy()
    tensors = []
    for index, (name, value) in enumerate(values.items()):
        path = prior / "reference" / (name + ".bin")
        value.tofile(path)
        tensors.append(_record(path, prior / "reference", index=index, name=name, shape=list(value.shape), dtype=value.dtype.name))
    expected = []
    for index, name in enumerate(("draft_top1", "draft_hidden")):
        path = fixture / (name + ".bin")
        values[name].tofile(path)
        expected.append(_record(path, fixture, index=index, name=name, shape=list(values[name].shape), dtype=values[name].dtype.name))
    original = {"artifact_kind": "qwen35-draft-reference", "status": "REFERENCE_COMPLETE", "inputs": inputs,
                "reference_contract": DRAFT_REFERENCE_CONTRACT, "expected_top1": expected[0], "expected_hidden": expected[1],
                "prepared_manifest_sha256": sha256_file(prepared / "prepared.json")}
    _write_json(fixture / "report.json", original)
    report = {"artifact_kind": "qwen35-draft-layer-reference-trace", "status": "REFERENCE_TRACE_COMPLETE",
              "reference_contract": DRAFT_REFERENCE_CONTRACT, "prepared_manifest_sha256": sha256_file(prepared / "prepared.json"),
              "fixture_report_sha256": sha256_file(fixture / "report.json"), "inputs": inputs,
              "golden_reproduction": [{"name": name, "status": "EXACT_MATCH", "sha256": row["sha256"]}
                                      for name, row in zip(("draft_top1", "draft_hidden"), expected)],
              "tensors": tensors}
    receipt = _write_json(prior / "reference/report.json", report)
    _write_json(prior / "completed-reference.json", receipt)
    _write_json(prior / "owner.json", {"source_job_sha256": sha256_file(job / "result.json"), "code": {
        name: sha256_file(Path(subject.__file__).with_name(name)) for name in ("draft_layer_trace.py", "draft_reference.py")}})
    spec = [{"name": name, "source": name if index < 3 else "layer_0_output", "dtype": value.dtype.name,
             "shape": list(value.shape), "output_index": min(index, 2)} for index, (name, value) in enumerate(values.items())]
    return {"job": job, "prepared": prepared, "fixture": fixture, "prior": prior, "values": values,
            "report": report, "spec": spec, "tmp": tmp_path}


def test_reuses_bound_cpu_observations_without_replay(case, monkeypatch):
    from qwen35_moe_dflash import draft_layer_trace
    monkeypatch.setattr(draft_layer_trace, "trace_prepared_draft", lambda *a, **k: pytest.fail("must reuse completed CPU observations"))
    names = ["draft_top1", "draft_hidden", "layer_0_output", "layer_1_input"]
    output = case["tmp"] / "direct-reference"
    result = subject.prepare_direct_reference(case["job"], case["prepared"], case["fixture"], output, names=names)
    assert [row["name"] for row in result["tensors"]] == names
    assert result["reused_report_sha256"] == sha256_file(case["prior"] / "reference/report.json")
    assert result["golden_reproduction"] == case["report"]["golden_reproduction"]
    for row in result["tensors"]:
        assert (output / row["path"]).read_bytes() == case["values"][row["name"]].tobytes()


@pytest.mark.parametrize("fault", ["source_job", "reference_code", "prepared", "fixture", "golden", "probe", "missing_probe"])
def test_reuse_rejects_changed_source_or_unreproduced_goldens(case, monkeypatch, fault):
    from qwen35_moe_dflash import draft_layer_trace
    monkeypatch.setattr(draft_layer_trace, "trace_prepared_draft", lambda *a, **k: pytest.fail("must not silently replay changed state"))
    if fault in {"source_job", "prepared", "fixture"}:
        path = {"source_job": case["job"] / "result.json", "prepared": case["prepared"] / "prepared.json", "fixture": case["fixture"] / "report.json"}[fault]
        path.write_text(path.read_text() + "\n")
    elif fault == "reference_code":
        path = case["prior"] / "owner.json"
        value = json.loads(path.read_text())
        value["code"]["draft_reference.py"] = "0" * 64
        _write_json(path, value)
    elif fault == "probe":
        (case["prior"] / "reference/layer_0_output.bin").write_bytes(b"changed")
    else:
        report = deepcopy(case["report"])
        if fault == "golden":
            report["golden_reproduction"][0]["status"] = "DIFFERENT"
        else:
            report["tensors"] = [row for row in report["tensors"] if row["name"] != "layer_0_output"]
        receipt = _write_json(case["prior"] / "reference/report.json", report)
        _write_json(case["prior"] / "completed-reference.json", receipt)
    with pytest.raises(ValueError):
        subject.prepare_direct_reference(case["job"], case["prepared"], case["fixture"], case["tmp"] / "bad-reference",
                                         names=["draft_top1", "draft_hidden", "layer_0_output"])


def _fixture(case):
    path = case["tmp"] / "direct-fixture"
    report = subject.prepare_direct_fixture(case["prepared"], case["fixture"], case["report"],
                                             case["prior"] / "reference", case["spec"], path)
    return path, report


def test_direct_fixture_deduplicates_equal_aliases_and_preserves_terminal_goldens(case):
    path, report = _fixture(case)
    assert len(report["expected_direct"]) == 3
    assert report["expected_top1"]["dtype"] == "int32" and report["expected_hidden"]["dtype"] == "float16"
    assert report["expected_hidden"]["sha256"] == json.loads((case["fixture"] / "report.json").read_text())["expected_hidden"]["sha256"]
    outputs = [{**row, "path": row["path"]} for row in report["expected_direct"]]
    values = subject.read_direct_values(path, {"outputs": outputs}, case["spec"])
    assert values["layer_0_output"] is values["layer_1_input"]
    assert values["layer_0_output"].dtype == np.float16


def test_direct_fixture_rejects_nonidentical_reference_aliases(case):
    reference = deepcopy(case["report"])
    row = reference["tensors"][-1]
    path = case["prior"] / "reference" / row["path"]
    data = case["values"]["layer_1_input"].copy()
    data.flat[0] += 1
    data.tofile(path)
    row.update(_record(path, path.parent))
    with pytest.raises(ValueError, match="aliases differ"):
        subject.prepare_direct_fixture(case["prepared"], case["fixture"], reference,
                                       case["prior"] / "reference", case["spec"], case["tmp"] / "bad-fixture")


def _binding(spec, promote=True):
    observed = deepcopy(spec)
    if promote:
        for row in observed:
            if row["output_index"] == 2:
                row["dtype"] = "float32"
    changes = [] if not promote else [{"kind": "DIAGNOSTIC_OUTPUT_PROMOTED_FP16_TO_FP32", "output_index": 2,
        "source": spec[2]["source"], "semantic_names": [row["name"] for row in spec if row["output_index"] == 2],
        "declared_dtype": "float16", "observed_dtype": "float32", "shape": spec[2]["shape"]}]
    return {"artifact_kind": "qwen35-draft-direct-output-abi-inspection", "status": "SOURCE_DTYPE_CHANGED" if promote else "PASSED",
            "declared_spec": spec, "observed_spec": observed, "changes": changes,
            "omc": {"path": "model.omc", "bytes": 123, "sha256": "a" * 64}, "schema": {"library_sha256": "b" * 64}}


@pytest.mark.parametrize("promote", [False, True])
def test_observed_fixture_records_wire_dtype_without_overwriting_source(case, promote):
    source, original = _fixture(case)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    binding = _binding(case["spec"], promote)
    output = case["tmp"] / "observed-fixture"
    report = subject.prepare_observed_direct_fixture(source, output, binding)
    assert report["direct_spec"] == binding["observed_spec"] and report["direct_output_binding"] == binding
    assert report["declared_direct_spec"] == case["spec"]
    assert report["expected_top1"]["dtype"] == "int32" and report["expected_hidden"]["dtype"] == "float16"
    extra = report["expected_direct"][2]
    assert extra["declared_dtype"] == "float16" and extra["dtype"] == ("float32" if promote else "float16")
    values = subject.read_direct_values(output, {"outputs": report["expected_direct"]}, report["direct_spec"])
    np.testing.assert_array_equal(values["layer_0_output"], case["values"]["layer_0_output"])
    assert values["layer_0_output"].dtype == (np.float32 if promote else np.float16)
    assert values["layer_0_output"] is values["layer_1_input"]
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("fault", ["terminal_dtype", "shape", "alias_dtype", "changed_source", "bad_changes", "bad_identity", "failed_status"])
def test_observed_fixture_refuses_unbound_or_unsupported_changes_before_writing(case, fault):
    source, _ = _fixture(case)
    binding = _binding(case["spec"])
    if fault == "terminal_dtype":
        binding["observed_spec"][1]["dtype"] = "float32"
    elif fault == "shape":
        binding["observed_spec"][2]["shape"] = [1, 8, 16]
    elif fault == "alias_dtype":
        binding["observed_spec"][-1]["dtype"] = "float16"
    elif fault == "changed_source":
        binding["observed_spec"][2]["source"] = "different_source"
    elif fault == "bad_changes":
        binding["changes"] = []
    elif fault == "bad_identity":
        binding["omc"]["sha256"] = "invalid"
    else:
        binding["status"] = "FAILED"
    output = case["tmp"] / "rejected-observed"
    with pytest.raises(ValueError):
        subject.prepare_observed_direct_fixture(source, output, binding)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["count", "dtype", "bytes", "hash"])
def test_direct_value_reader_refuses_wrong_native_wire_bytes(case, fault):
    root, report = _fixture(case)
    native = {"outputs": deepcopy(report["expected_direct"])}
    if fault == "count":
        native["outputs"].pop()
    elif fault == "dtype":
        native["outputs"][2]["dtype"] = "float32"
    elif fault == "bytes":
        native["outputs"][2]["bytes"] += 2
    else:
        native["outputs"][2]["sha256"] = "c" * 64
    with pytest.raises(ValueError):
        subject.read_direct_values(root, native, case["spec"])

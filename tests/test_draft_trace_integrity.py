"""Observation consistency and small, source-bound read-only trace reviews."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from qwen35_moe_dflash import draft_trace_integrity as subject
from qwen35_moe_dflash.draft_reference import prepare_draft_reference, load_draft_reference
from qwen35_moe_dflash.draft_layer_trace import trace_draft_graph
from test_draft_reference import prepared


@pytest.fixture
def observations(prepared, tmp_path):
    root, _, config, _, _ = prepared
    fixture = tmp_path / "reference"
    report = prepare_draft_reference(root, fixture, valid_length=3)
    inputs = {row["name"]: np.fromfile(fixture / row["path"], dtype=row["dtype"]).reshape(row["shape"])
              for row in report["inputs"]}
    values, _ = trace_draft_graph(load_draft_reference(root), inputs)
    return values, config


def test_original_six_layer_reference_has_no_internal_contradiction(observations):
    values, config = observations
    report = subject.evaluate_trace_integrity(values, config, native_top1=values["draft_top1"])
    assert report["status"] == "PASSED", report["contradictions"]
    assert report["contradiction_count"] == 0
    assert report["unavailable_count"] == 0
    assert all(row["status"] == "EXACT" for row in report["checks"] if row["name"].startswith("same_source:"))
    json.dumps(report, allow_nan=False)


def test_same_graph_value_disagreement_invalidates_trace_before_root_cause_attribution(observations):
    values, config = observations
    values = {name: array.copy() for name, array in values.items()}
    values["layer_1_input"].flat[0] = np.nan
    report = subject.evaluate_trace_integrity(values, config)
    assert report["status"] == "FAILED"
    assert report["root_cause_attribution"] == "UNAVAILABLE_WHILE_TRACE_CONTRADICTORY"
    row = next(row for row in report["checks"] if row["name"] == "same_source:layer_0_output=layer_1_input")
    assert "SAME_SOURCE_BYTES_DIFFER" in row["contradictions"]


def test_zero_fc_followed_by_finite_nonzero_norm_is_a_hard_contradiction(observations):
    values, config = observations
    values = dict(values, context_fc=np.zeros_like(values["context_fc"]), projected_context=np.ones_like(values["projected_context"]))
    report = subject.evaluate_trace_integrity(values, config)
    row = next(row for row in report["checks"] if row["name"] == "zero_norm:context_fc")
    assert report["status"] == "FAILED" and row["contradictory_rows"] > 0


def test_packed_top1_must_match_independent_native_port(observations):
    values, config = observations
    native = values["draft_top1"].copy()
    native.flat[0] += 1
    report = subject.evaluate_trace_integrity(values, config, native_top1=native)
    assert report["status"] == "FAILED"
    assert any("INDEPENDENT_TOP1_PORT_DIFFERS" in row["reasons"] for row in report["contradictions"])


def test_one_ulp_score_rounding_is_reported_without_a_new_model_threshold(observations):
    values, config = observations
    values = {name: array.copy() for name, array in values.items()}
    score = values["layer_0_scores_fp32"]
    valid = np.isfinite(score) & (score != 0)
    index = tuple(np.argwhere(valid)[0])
    score[index] = np.nextafter(score[index], np.float32(np.inf))
    report = subject.evaluate_trace_integrity(values, config)
    row = next(row for row in report["checks"] if row["name"] == "layer_0_masked_score")
    assert report["status"] == "PASSED", report["contradictions"]
    assert row["status"] == "DIFFERENT_DIAGNOSTIC" and row["max_abs_error"] > 0
    assert report["arithmetic_acceptance_threshold"] == "NOT_DEFINED"


def test_explicit_mask_infinity_is_not_an_arithmetic_tolerance(observations):
    values, config = observations
    values = {name: array.copy() for name, array in values.items()}
    score = values["layer_0_scores_fp32"]
    index = tuple(np.argwhere(np.isneginf(score))[0])
    score[index] = 0
    report = subject.evaluate_trace_integrity(values, config)
    assert report["status"] == "FAILED"
    assert any("EXPLICIT_MASK_NEGATIVE_INFINITY_DIFFERS" in row["reasons"] for row in report["contradictions"])


def test_direct_subset_marks_unavailable_and_deduplicated_alias_is_not_independent(observations):
    values, config = observations
    subset = {name: values[name] for name in ("noise_embedding", "layer_0_input", "draft_hidden")}
    report = subject.evaluate_trace_integrity(subset, config, alias_deduplicated=True)
    assert report["status"] == "PASSED" and report["unavailable_count"] > 0
    row = next(row for row in report["checks"] if row["name"].startswith("same_source:noise_embedding"))
    assert row["status"] == "EXACT" and row["independent_observations"] is False


def _record(path, root, **fields):
    data = path.read_bytes()
    return {**fields, "path": str(path.relative_to(root)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def retained(tmp_path, monkeypatch, observations):
    # This fixture records mock execution metadata only. The pure observation
    # arrays come from an actual tiny reference, not target-device inference.
    from qwen35_moe_dflash import draft_trace_inspect
    from qwen35_moe_dflash.draft_board import _wire
    values, cfg = observations
    job, trace = tmp_path / "full-draft", tmp_path / "full-draft/layer-trace"
    for name in ("fixture", "air", "compilation", "bundle", "board/results"):
        (trace / name).mkdir(parents=True, exist_ok=True)
    source_prepared, source_fixture = job / "prepared", job / "fixture"
    source_prepared.mkdir()
    source_fixture.mkdir()
    # Keep the locked JSON form used by the real reference fixture.
    from qwen35_moe_dflash.ge_export import configurations
    config = deepcopy(configurations(tiny=True)[1])
    # observations uses the six-layer small config from test_draft_reference.
    config.update(num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"])
    config["dflash_config"]["block_size"] = 16

    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")
        return _record(path, path.parent)

    prepared_record = write(source_prepared / "prepared.json", {"draft_config": config})
    inputs = []
    for index, (name, dtype, shape) in enumerate((("target_features", "float16", [1, 6, cfg.feature_size]),
            ("input_ids", "int32", [1, 6]), ("attention_mask", "int32", [1, 6]))):
        path = trace / "bundle" / (name + ".bin")
        np.zeros(shape, dtype=dtype).tofile(path)
        inputs.append(_record(path, trace / "bundle", index=index, name=name, dtype=dtype, shape=shape))
    original_top = {"index": 0, "name": "draft_top1", "dtype": "int32", "shape": [1, cfg.block_size - 1],
                    "path": "original_top1.bin", "bytes": values["draft_top1"].nbytes,
                    "sha256": hashlib.sha256(values["draft_top1"].tobytes()).hexdigest()}
    source_fixture_record = write(source_fixture / "report.json", {"draft_config": config, "inputs": inputs, "expected_top1": original_top})
    source = {"artifact_kind": "qwen35-full-draft-job", "components": {
        "prepared": {"directory": str(source_prepared), "manifest": prepared_record},
        "fixture": {"directory": str(source_fixture), "manifest": source_fixture_record}}}
    write(job / "result.json", source)
    spec, arrays, offset = [], [], 0
    for name, array in values.items():
        spec.append({"name": name, "shape": list(array.shape), "dtype": array.dtype.name,
                     "offset": offset, "elements": array.size})
        arrays.append(array.astype("<f4").reshape(-1))
        offset += array.size
    owner = {"source_job_sha256": _record(job / "result.json", job)["sha256"], "trace_spec": spec}
    write(trace / "owner.json", owner)
    write(trace / "result.json", {"artifact_kind": "qwen35-draft-layer-trace-job", "production_equivalence": {"status": "DIFFERENT"}})
    fixture = {"status": "REFERENCE_COMPLETE", "draft_config": config, "inputs": inputs, "trace_spec": spec,
               "original_fixture_sha256": source_fixture_record["sha256"], "prepared_manifest_sha256": prepared_record["sha256"],
               "expected_top1": original_top}
    air = {"trace_spec": spec, "prepared_manifest_sha256": prepared_record["sha256"]}
    stage_records = {}
    compiler_omc = {"path": "mock.omc", "bytes": 3, "sha256": "c" * 64}
    for stage, report in (("fixture", fixture), ("air", air), ("compilation", {"omc": compiler_omc})):
        filename = "manifest.json" if stage == "air" else "report.json"
        stage_records[stage] = write(trace / stage / filename, report)
        write(trace / ("completed-" + stage + ".json"), stage_records[stage])
    outputs = [{"index": 0, "name": "draft_top1", "dtype": "int32", "shape": [1, cfg.block_size - 1]},
               {"index": 1, "name": "draft_trace_values", "dtype": "float32", "shape": [1, 1, offset]}]
    bundle = {"artifact_kind": "qwen35-standalone-draft-trace-bundle", "status": "PREPARED", "source_identity": "MATCHED_EXPORT_AND_REFERENCE",
              "trace_spec": spec, "draft_config": config, "inputs": inputs, "outputs": outputs, "ddk_id": "mock-ddk", "sequence_length": 6,
              "fixture_report_sha256": stage_records["fixture"]["sha256"], "export_manifest_sha256": stage_records["air"]["sha256"],
              "compiler_report_sha256": stage_records["compilation"]["sha256"],
              "model": {"path": "model.omc", "bytes": 2_800_000_000, "sha256": "a" * 64}}
    abi = {"artifact_kind": "qwen35-standalone-draft-trace-abi", "trace_spec": spec, "draft_config": config,
           "inputs": [{key: row[key] for key in ("index", "name", "dtype", "shape")} for row in inputs], "outputs": outputs}
    bundle["draft_abi"] = write(trace / "bundle/draft-abi.json", abi)
    (trace / "bundle/step.tsv").write_bytes(_wire(bundle))
    bundle["step_plan"] = _record(trace / "bundle/step.tsv", trace / "bundle")
    values["draft_top1"].tofile(trace / "bundle/expected_top1.bin")
    bundle["expected"] = [_record(trace / "bundle/expected_top1.bin", trace / "bundle", **outputs[0]), None]
    bundle_record = write(trace / "bundle/manifest.json", bundle)
    write(trace / "completed-bundle.json", bundle_record)
    board_root = trace / "board"
    values["draft_top1"].astype("<i4").tofile(board_root / "results/output_0000.bin")
    np.concatenate(arrays).astype("<f4").tofile(board_root / "results/output_0001.bin")
    native = {"schema_version": 1, "scope": "standalone-draft-step", "status": "EXECUTED", "evidence_kind": "host-fake-hiai",
              "target_calls": 0, "draft_calls": 1, "model_sha256": bundle["model"]["sha256"], "draft_abi_sha256": bundle["draft_abi"]["sha256"],
              "step_plan_sha256": bundle["step_plan"]["sha256"], "ddk_id": "mock-ddk", "runtime_version": "mock-runtime", "firmware_id": "mock-firmware",
              "inputs": inputs, "outputs": [_record(board_root / f"results/output_{i:04d}.bin", board_root / "results", **output) for i, output in enumerate(outputs)],
              "io_diagnostic": {"enabled": True, "process_called": True,
                  "inputs": [{"index": row["index"], "readback_matches_source": True, "readback_sha256": row["sha256"]} for row in inputs],
                  "outputs": [{"index": i, "checked_after_process": True, "object_same": True, "buffer_same": True} for i in range(2)]}}
    write(board_root / "results/result.json", native)
    write(board_root / "runtime.json", {"npu_supported": True, "runtime_version": "mock-runtime"})
    for name in ("draft.stdout", "draft.stderr", "runtime.stderr"):
        (board_root / name).write_bytes(b"")
    for name in ("draft.exitcode", "runtime.exitcode"):
        (board_root / name).write_bytes(b"0\n")
    board = {"artifact_kind": "qwen35-standalone-draft-execution", "device_execution": "PASSED", "remote_exitcode": 0,
             "model_load_api": "legacy-buffer", "bundle_manifest_sha256": bundle_record["sha256"], "ddk_id": "mock-ddk",
             "runtime_version": "mock-runtime", "firmware_id": "mock-firmware", "retrieval_errors": [],
             "retrieved_sha256": {str(path.relative_to(board_root)): _record(path, board_root)["sha256"] for path in board_root.rglob("*") if path.is_file()}}
    board_record = write(board_root / "result.json", board)
    write(trace / "completed-board.json", board_record)
    precision_record = write(trace / "precision-check.json", {"status": "PASSED"})
    monkeypatch.setattr(draft_trace_inspect, "inspect_retained_trace_precision", lambda _: {
        "compiled_precision": {"status": "PASSED"}, "precision_report_sha256": precision_record["sha256"], "omc": compiler_omc})
    return job, trace, values


def test_read_only_review_validates_outputs_without_touching_large_model(retained):
    job, trace, _ = retained
    before = {path: path.read_bytes() for path in job.rglob("*") if path.is_file()}
    report = subject.review_trace_integrity(job, allow_host_fake=True)
    assert report["status"] == "PASSED", report["contradictions"]
    assert report["production_equivalence"]["status"] == "DIFFERENT"
    assert "not rehashed" in report["model_identity_scope"]
    assert not (trace / "bundle/model.omc").exists()
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("fault", ["output", "bundle_receipt", "missing_hash", "native_model", "native_inputs", "wire", "source_job", "native_fake"])
def test_review_rejects_changed_or_unbound_evidence(retained, fault):
    job, trace, _ = retained
    board_root = trace / "board"
    if fault == "output":
        (board_root / "results/output_0001.bin").write_bytes(b"changed")
    elif fault == "bundle_receipt":
        (trace / "completed-bundle.json").write_text('{}')
    elif fault == "missing_hash":
        p = board_root / "result.json"
        report = json.loads(p.read_text())
        report["retrieved_sha256"].pop("runtime.exitcode")
        p.write_text(json.dumps(report))
        (trace / "completed-board.json").write_text(json.dumps(_record(p, p.parent)))
    elif fault in {"native_model", "native_inputs"}:
        p = board_root / "results/result.json"
        native = json.loads(p.read_text())
        if fault == "native_model":
            native["model_sha256"] = "b" * 64
        else:
            native["inputs"][0]["sha256"] = "b" * 64
        p.write_text(json.dumps(native))
        b = board_root / "result.json"
        report = json.loads(b.read_text())
        report["retrieved_sha256"]["results/result.json"] = _record(p, p.parent)["sha256"]
        b.write_text(json.dumps(report))
        (trace / "completed-board.json").write_text(json.dumps(_record(b, b.parent)))
    elif fault == "wire":
        (trace / "bundle/step.tsv").write_bytes(b"changed")
    elif fault == "source_job":
        with (job / "result.json").open("a") as out:
            out.write("\n")
    with pytest.raises((ValueError, KeyError)):
        subject.review_trace_integrity(job, allow_host_fake=fault != "native_fake")

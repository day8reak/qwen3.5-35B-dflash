"""Retained direct arithmetic review reads small, receipt-bound evidence only."""
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.draft_board import _record
from qwen35_moe_dflash.draft_output_review import review_direct_output_math
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.hashing import sha256_file


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return _record(path, path.parent)


@pytest.fixture
def retained(tmp_path):
    source, root = tmp_path / "full-draft", tmp_path / "full-draft/direct-trace"
    prepared = source / "prepared"
    prepared.mkdir(parents=True)
    config = configurations(tiny=True)[1]
    from qwen35_moe_dflash.config import Qwen35DFlashConfig
    cfg = Qwen35DFlashConfig.from_dict(config)
    gamma = prepared / "gamma.bin"
    np.ones(cfg.hidden_size, dtype="<f2").tofile(gamma)
    prepared_manifest = {"artifact_kind": "qwen35-draft-ge-prepared", "status": "PREPARED",
                         "draft_config": config, "weights": {
        "draft.layers.1.input_layernorm.weight": {**_record(gamma, prepared), "shape": [cfg.hidden_size], "dtype": "float16"},
        "unused.giant.projection": {"path": "must-not-be-opened.bin", "bytes": 10**10}}}
    manifest_receipt = _json(prepared / "prepared.json", prepared_manifest)
    _json(source / "result.json", {"components": {"prepared": {"directory": str(prepared), "manifest": manifest_receipt}}})
    shape = (1, cfg.block_size, cfg.hidden_size)
    arrays = {"draft_top1": np.zeros((1, cfg.block_size - 1), dtype="<i4"),
              "draft_hidden": np.zeros((1, cfg.block_size - 1, cfg.hidden_size), dtype="<f2")}
    for name, fill in (("layer_0_input", 1), ("layer_0_attention", 2),
                       ("layer_0_post_attention", 0), ("layer_0_mlp_down", 3),
                       ("layer_0_output", 0), ("layer_1_input_norm", 1)):
        arrays[name] = np.full(shape, fill, dtype="<f2")
    spec = [{"name": name, "source": name, "shape": list(value.shape), "dtype": value.dtype.name,
             "output_index": index} for index, (name, value) in enumerate(arrays.items())]
    _json(root / "owner.json", {"artifact_kind": "qwen35-draft-direct-trace-job-owner", "job": str(source),
        "source_job_sha256": sha256_file(source / "result.json"), "trace_spec": spec})

    def stage(name, file, value):
        receipt = _json(root / name / file, value)
        _json(root / ("completed-" + name + ".json"), receipt)
        return receipt

    original = stage("fixture", "report.json", {"direct_spec": spec,
                     "prepared_manifest_sha256": sha256_file(prepared / "prepared.json")})
    observed = stage("observed-fixture", "report.json", {"declared_direct_spec": spec, "direct_spec": spec,
                     "source_direct_fixture_sha256": original["sha256"]})
    inputs = [{"index": i, "name": name, "shape": [1], "dtype": dtype, "bytes": size, "sha256": str(i) * 64}
              for i, (name, dtype, size) in enumerate((("target_features", "float16", 2), ("input_ids", "int32", 4), ("attention_mask", "int32", 4)))]
    bundle = {"draft_config": config, "direct_spec": spec, "fixture_report_sha256": observed["sha256"], "inputs": inputs,
              "model": {"path": "model/must-not-be-opened.omc", "sha256": "a" * 64},
              "draft_abi": {"sha256": "b" * 64}, "step_plan": {"sha256": "c" * 64}, "ddk_id": "fake"}
    bundle_receipt = stage("bundle", "manifest.json", bundle)
    results = root / "board/results"
    results.mkdir(parents=True)
    outputs = []
    for index, (name, value) in enumerate(arrays.items()):
        path = results / ("output_%04d.bin" % index)
        value.tofile(path)
        outputs.append({"index": index, "name": name, "dtype": value.dtype.name, "shape": list(value.shape), **_record(path, results)})
    native = {"scope": "standalone-draft-step", "status": "EXECUTED", "target_calls": 0, "draft_calls": 1,
              "evidence_kind": "host-fake-hiai", "ddk_id": "fake", "model_sha256": "a" * 64,
              "draft_abi_sha256": "b" * 64, "step_plan_sha256": "c" * 64, "inputs": inputs, "outputs": outputs,
              "io_diagnostic": {"process_called": True,
                  "inputs": [{"index": r["index"], "readback_matches_source": True, "readback_sha256": r["sha256"]} for r in inputs],
                  "outputs": [{"index": i, "checked_after_process": True, "object_same": True, "buffer_same": True} for i in range(len(outputs))]}}
    native_receipt = _json(results / "result.json", native)
    transport = {"results/result.json": native_receipt["sha256"], **{"results/" + r["path"]: r["sha256"] for r in outputs}}
    stage("board", "result.json", {"device_execution": "PASSED", "bundle_manifest_sha256": bundle_receipt["sha256"],
                                    "retrieved_sha256": transport})
    return root


def test_retained_math_never_opens_model_or_projection_weights(retained, monkeypatch):
    before = {p: p.read_bytes() for p in retained.parent.rglob("*") if p.is_file()}
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert "must-not-be-opened" not in self.name
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    report = review_direct_output_math(retained, allow_host_fake=True)
    assert report["status"] == "DIAGNOSTIC_COMPLETE"
    assert report["root_cause_attribution"] == "NOT_ESTABLISHED"
    assert report["norm_weight"]["bytes"] < 65536
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("fault", ["source", "gamma", "output", "transport", "bundle", "readback"])
def test_changed_retained_evidence_is_rejected(retained, fault):
    if fault in {"source", "gamma", "output"}:
        path = {"source": retained.parent / "result.json", "gamma": retained.parent / "prepared/gamma.bin",
                "output": retained / "board/results/output_0000.bin"}[fault]
        path.write_bytes(path.read_bytes() + b"x")
    else:
        if fault == "transport":
            path = retained / "board/result.json"
            row = json.loads(path.read_text())
            row["retrieved_sha256"]["results/result.json"] = "f" * 64
            _json(path, row)
            _json(retained / "completed-board.json", _record(path, path.parent))
        elif fault == "bundle":
            path = retained / "bundle/manifest.json"
            row = json.loads(path.read_text())
            row["model"]["sha256"] = "f" * 64
            _json(path, row)
        else:
            path = retained / "board/results/result.json"
            row = json.loads(path.read_text())
            row["io_diagnostic"]["inputs"][0]["readback_matches_source"] = False
            _json(path, row)
            board_path = retained / "board/result.json"
            board = json.loads(board_path.read_text())
            board["retrieved_sha256"]["results/result.json"] = sha256_file(path)
            _json(board_path, board)
            _json(retained / "completed-board.json", _record(board_path, board_path.parent))
    with pytest.raises(ValueError):
        review_direct_output_math(retained, allow_host_fake=True)


def test_host_fake_is_not_device_evidence(retained):
    with pytest.raises(ValueError, match="device HIAI"):
        review_direct_output_math(retained)

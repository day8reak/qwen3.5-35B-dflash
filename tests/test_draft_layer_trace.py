"""Small six-layer v3 probes; these tests provide no board accuracy evidence."""
import json

import numpy as np
import pytest
import torch

from qwen35_moe_dflash import draft_layer_trace as subject, draft_reference
from qwen35_moe_dflash.hashing import sha256_file
from test_draft_reference import prepared


def fixture_case(prepared, tmp_path):
    root = prepared[0]
    fixture = tmp_path / "reference"
    report = draft_reference.prepare_draft_reference(root, fixture, valid_length=3)
    arrays = {row["name"]: np.fromfile(fixture / row["path"], dtype=row["dtype"]).reshape(row["shape"])
              for row in report["inputs"]}
    return root, fixture, report, arrays


def hooks(graph):
    return {name: (len(module._forward_hooks), len(module._forward_pre_hooks))
            for name, module in graph.named_modules()}


def test_six_layer_trace_preserves_v3_outputs_and_all_hooks(prepared, tmp_path):
    root, fixture, golden, inputs = fixture_case(prepared, tmp_path)
    graph = draft_reference.load_draft_reference(root)
    dispatch = {name: module.ops for name, module in graph.named_modules() if hasattr(module, "ops")}
    before = hooks(graph)
    input_copies = {name: array.copy() for name, array in inputs.items()}
    events = []
    arrays, report = subject.trace_draft_graph(graph, inputs, progress=events.append)
    assert tuple(arrays) == subject.trace_names(graph.draft.config)
    assert [row["name"] for row in report["tensors"]] == list(arrays)
    assert len(arrays) == 166
    assert len(events) == 166
    assert report["total_bytes"] == sum(value.nbytes for value in arrays.values())
    assert arrays["block_ids"].dtype == np.int32
    assert arrays["positions"].dtype == np.int32
    np.testing.assert_array_equal(arrays["block_ids"][0, :1], inputs["input_ids"][0, 2:3])
    np.testing.assert_array_equal(arrays["context_valid"], [[True, True, False, False, False, False]])
    assert arrays["layer_0_scores_fp32"].dtype == np.float32
    assert np.isneginf(arrays["layer_0_scores_fp32"]).any()
    assert arrays["layer_5_probability_fp32"].dtype == np.float32
    assert arrays["layer_5_pv_fp16"].dtype == np.float16
    for name, key in (("draft_hidden", "expected_hidden"), ("draft_top1", "expected_top1")):
        assert arrays[name].tobytes() == (fixture / golden[key]["path"]).read_bytes()
    for name, old in input_copies.items():
        assert inputs[name].tobytes() == old.tobytes()
    assert hooks(graph) == before
    assert all(graph.get_submodule(name).ops is original for name, original in dispatch.items())
    json.dumps(report, allow_nan=False)


def test_arbitrary_selection_order_is_preserved_without_large_logits(prepared, tmp_path):
    root, _, _, inputs = fixture_case(prepared, tmp_path)
    graph = draft_reference.load_draft_reference(root)
    names = ["layer_5_output", "context_fc", "layer_0_qk_fp16", "draft_hidden"]
    values, report = subject.trace_draft_graph(graph, inputs, names=names)
    assert list(values) == names
    assert [row["name"] for row in report["tensors"]] == names
    assert [row["index"] for row in report["tensors"]] == list(range(len(names)))
    assert not any("logit" in name for name in values)


@pytest.mark.parametrize("valid_length", [1, 3, 6])
def test_complete_six_layer_ge_trace_matches_independent_cpu_observer(prepared, tmp_path, valid_length):
    from qwen35_moe_dflash.draft_ge_trace import instrument_draft_trace
    from reference.ge_torch import evaluate_torch
    root, _, config, graph, weights = prepared
    fixture = tmp_path / "reference"
    golden = draft_reference.prepare_draft_reference(root, fixture, valid_length=valid_length)
    inputs = {row["name"]: np.fromfile(fixture / row["path"], dtype=row["dtype"]).reshape(row["shape"])
              for row in golden["inputs"]}
    diagnostic, spec = instrument_draft_trace(graph)
    top1, packed_tensor = evaluate_torch(diagnostic, inputs, weights)
    observed, _ = subject.trace_draft_graph(draft_reference.load_draft_reference(root), inputs)
    assert len(spec) == len(subject.trace_names(config)) == 166
    np.testing.assert_array_equal(top1.numpy(), observed["draft_top1"])
    packed = packed_tensor.numpy().reshape(-1)
    for row in spec:
        actual = packed[row["offset"]:row["offset"] + row["elements"]].reshape(row["shape"])
        expected = observed[row["name"]]
        assert list(expected.shape) == row["shape"]
        assert expected.dtype.name == row["dtype"]
        if row["dtype"] in {"bool", "int32"}:
            np.testing.assert_array_equal(actual, expected, err_msg=row["name"])
        else:
            np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=2.5e-2, err_msg=row["name"])


@pytest.mark.parametrize("fault", ["legacy", "training", "mask", "dtype", "empty", "unknown", "duplicate"])
def test_trace_rejects_bad_contract_before_installing_hooks(prepared, tmp_path, fault):
    root, _, _, inputs = fixture_case(prepared, tmp_path)
    graph = draft_reference.load_draft_reference(root)
    names = ["draft_hidden"]
    if fault == "legacy":
        graph.reference_rms_mode = "legacy-fp16"
    elif fault == "training":
        graph.train()
    elif fault == "mask":
        inputs["attention_mask"][0, :3] = [1, 0, 1]
    elif fault == "dtype":
        inputs["target_features"] = inputs["target_features"].astype(np.float32)
    elif fault == "empty":
        names = []
    elif fault == "unknown":
        names = ["fake_output"]
    elif fault == "duplicate":
        names = ["draft_hidden", "draft_hidden"]
    before = hooks(graph)
    with pytest.raises(ValueError):
        subject.trace_draft_graph(graph, inputs, names=names)
    assert hooks(graph) == before


@pytest.mark.parametrize("fault", ["progress", "bytes", "forward"])
def test_trace_restores_hooks_and_dispatch_after_failure(prepared, tmp_path, monkeypatch, fault):
    root, _, _, inputs = fixture_case(prepared, tmp_path)
    graph = draft_reference.load_draft_reference(root)
    dispatch = {name: module.ops for name, module in graph.named_modules() if hasattr(module, "ops")}
    old_hook = graph.draft.norm.register_forward_hook(lambda *args: None)
    before = hooks(graph)
    def callback(message):
        if fault == "progress" and "layer_0_q_proj" in message:
            raise RuntimeError("intentional progress error")
    if fault == "bytes":
        monkeypatch.setattr(subject, "_MAX_TOTAL_BYTES", 30)
    elif fault == "forward":
        def broken(*args):
            raise RuntimeError("intentional forward error")
        monkeypatch.setattr(graph.lm_head, "forward", broken)
    with pytest.raises((ValueError, RuntimeError)):
        subject.trace_draft_graph(graph, inputs, progress=callback)
    assert hooks(graph) == before
    assert all(graph.get_submodule(name).ops is original for name, original in dispatch.items())
    old_hook.remove()


def test_prepared_trace_binds_inputs_keeps_goldens_and_reproduces_them(prepared, tmp_path):
    root, fixture, golden, _ = fixture_case(prepared, tmp_path)
    original_files = {path: sha256_file(path) for path in fixture.iterdir()}
    names = ["layer_5_output", "projected_context", "positions"]
    output = tmp_path / "trace"
    result = subject.trace_prepared_draft(root, fixture, output, names=names)
    assert result["status"] == "REFERENCE_TRACE_COMPLETE"
    assert result["reference_contract"] == draft_reference.DRAFT_REFERENCE_CONTRACT
    assert result["fixture_report_sha256"] == sha256_file(fixture / "report.json")
    assert result["prepared_manifest_sha256"] == sha256_file(root / "prepared.json")
    assert [row["name"] for row in result["tensors"]] == names
    assert [row["index"] for row in result["tensors"]] == [0, 1, 2]
    assert [row["status"] for row in result["golden_reproduction"]] == ["EXACT_MATCH", "EXACT_MATCH"]
    assert result["inputs"] == golden["inputs"]
    for row in result["tensors"]:
        assert row["sha256"] == sha256_file(output / row["path"])
    assert json.loads((output / "report.json").read_text()) == result
    assert all(sha256_file(path) == digest for path, digest in original_files.items())
    assert not (output / "draft_hidden.bin").exists()
    assert not (output / "draft_top1.bin").exists()


@pytest.mark.parametrize("fault", ["binding", "legacy", "inputs", "golden", "weight", "changed"])
def test_prepared_trace_rejects_bad_provenance_or_nonreproduction(prepared, tmp_path, monkeypatch, fault):
    root, fixture, golden, _ = fixture_case(prepared, tmp_path)
    if fault == "binding":
        golden["prepared_manifest_sha256"] = "f" * 64
    elif fault == "legacy":
        golden["reference_contract"] = "pytorch-fp16-primitives-v1"
    elif fault == "inputs":
        path = fixture / golden["inputs"][0]["path"]
        path.write_bytes(b"\x00" * path.stat().st_size)
    elif fault == "golden":
        row = golden["expected_top1"]
        path = fixture / row["path"]
        path.write_bytes(b"\x00" * path.stat().st_size)
        row["sha256"] = sha256_file(path)
    elif fault == "weight":
        plan = prepared[1]
        path = root / next(iter(plan["weights"].values()))["path"]
        path.write_bytes(b"\x00" * path.stat().st_size)
    elif fault == "changed":
        trace = subject.trace_draft_graph
        def modify(*args, **kwargs):
            result = trace(*args, **kwargs)
            (fixture / "report.json").write_text("{}")
            return result
        monkeypatch.setattr(subject, "trace_draft_graph", modify)
    (fixture / "report.json").write_text(json.dumps(golden))
    output = tmp_path / "trace"
    with pytest.raises(ValueError):
        subject.trace_prepared_draft(root, fixture, output, names=["draft_hidden"])
    if output.exists():
        result = json.loads((output / "report.json").read_text())
        assert result["status"] == "FAILED"

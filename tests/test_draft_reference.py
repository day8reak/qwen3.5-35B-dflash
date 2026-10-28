"""Full layer-stack draft reference without allocating any target decoder."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from qwen35_moe_dflash import draft_reference as subject
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_export import configurations, weight_identity
from qwen35_moe_dflash.ge_weights import tiny_weights
from qwen35_moe_dflash.hashing import sha256_file
from reference.ge_torch import evaluate_torch


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    torch.set_num_threads(1)
    raw = deepcopy(configurations(tiny=True)[1])
    # Exercise all six layers and all 15 proposal rows at small hidden/vocab
    # widths. This is intentionally not labelled a full-weight checkpoint.
    raw.update(num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"])
    raw["dflash_config"]["block_size"] = 16
    monkeypatch.setattr(subject, "configurations", lambda **kwargs: ({}, raw))
    config = Qwen35DFlashConfig.from_dict(raw)
    graph = build_draft(config, 6)
    weights = tiny_weights({k: v.shape for k, v in graph.weights.items()})
    root = tmp_path / "prepared"
    (root / "weights").mkdir(parents=True)
    records = {}
    for index, (name, value) in enumerate(sorted(weights.items())):
        path = root / "weights" / f"w{index:03}.bin"
        value.tofile(path)
        records[name] = {"path": str(path.relative_to(root)), "bytes": value.nbytes,
                         "dtype": "float16", "shape": list(value.shape), "sha256": sha256_file(path)}
    manifest = {"artifact_kind": "qwen35-draft-ge-prepared", "status": "PREPARED", "tiny": True,
                "precision": "float16", "quantization": None, "draft_config": raw,
                "sequence_length": 6, "weights": records,
                "source": {"kind": "test-six-layer-small-width", "fp16_payload_sha256": weight_identity(records)}}
    (root / "prepared.json").write_text(json.dumps(manifest))
    return root, manifest, config, graph, weights


@pytest.mark.parametrize("valid_length", [1, 3, 6])
def test_complete_six_layer_reference_matches_independent_ge_graph(prepared, tmp_path, monkeypatch, valid_length):
    root, manifest, config, graph, weights = prepared
    from qwen35_moe_dflash import graphs, ge_export

    def forbidden(*args, **kwargs):
        pytest.fail("standalone draft must not construct/load target or a target/draft pair")

    monkeypatch.setattr(graphs, "TargetGraph", forbidden)
    monkeypatch.setattr(graphs, "load_graphs", forbidden)
    monkeypatch.setattr(ge_export, "build_pair", forbidden)
    output = tmp_path / f"reference-{valid_length}"
    result = subject.prepare_draft_reference(root, output, valid_length=valid_length)
    assert result["status"] == "REFERENCE_COMPLETE"
    assert result["draft_layers_executed"] == 6 and result["full_vocabulary_rows"] == config.vocab_size
    assert result["target_features_source"] == "SYNTHETIC_DIAGNOSTIC_NOT_TARGET_EXECUTION"
    assert result["numerical_acceptance"]["hidden_threshold"] == "NOT_DEFINED"
    inputs = {record["name"]: np.fromfile(output / record["path"], dtype=record["dtype"]).reshape(record["shape"])
              for record in result["inputs"]}
    values = {node.output.name: node.output for node in graph.nodes}
    graph.outputs.append(values["draft_hidden"])
    expected_top1, expected_hidden = evaluate_torch(graph, inputs, weights)
    actual_top1 = np.fromfile(output / result["expected_top1"]["path"], dtype="<i4").reshape(1, 15)
    actual_hidden = np.fromfile(output / result["expected_hidden"]["path"], dtype="<f2").reshape(1, 15, config.hidden_size)
    np.testing.assert_array_equal(actual_top1, expected_top1.numpy())
    torch.testing.assert_close(torch.from_numpy(actual_hidden), expected_hidden)
    assert np.count_nonzero(actual_hidden) > 0
    for descriptor in result["inputs"] + [result["expected_top1"], result["expected_hidden"]]:
        assert descriptor["sha256"] == sha256_file(output / descriptor["path"])
    assert json.loads((output / "report.json").read_text()) == result


def test_reference_is_reproducible_and_mapped_parameters_cannot_modify_payload(prepared, tmp_path):
    root, manifest, _, _, _ = prepared
    first = subject.prepare_draft_reference(root, tmp_path / "first")
    second = subject.prepare_draft_reference(root, tmp_path / "second")
    assert first["inputs"] == second["inputs"]
    assert first["expected_top1"] == second["expected_top1"]
    assert first["expected_hidden"] == second["expected_hidden"]
    graph = subject.load_draft_reference(root)
    record = manifest["weights"]["target.embed_tokens.weight"]
    with torch.no_grad():
        graph.embedding.weight[0, 0] += 1
    assert sha256_file(root / record["path"]) == record["sha256"]
    assert all(not value.is_meta and value.dtype == torch.float16 for value in graph.parameters())
    assert not any("target" in name for name, _ in graph.named_modules())


def test_padding_features_and_anchor_feature_are_excluded(prepared):
    root, _, config, _, _ = prepared
    graph = subject.load_draft_reference(root)
    torch.manual_seed(91)
    features = torch.randn(1, 6, config.feature_size, dtype=torch.float16) * .2
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.int32)
    changed = features.clone()
    changed[:, 2:] = 1
    with torch.inference_mode():
        before, after = graph(features, ids, mask), graph(changed, ids, mask)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_input_only_mode_never_loads_reference(prepared, tmp_path, monkeypatch):
    root, _, _, _, _ = prepared
    monkeypatch.setattr(subject, "_load_graph", lambda *args: pytest.fail("reference unexpectedly loaded"))
    result = subject.prepare_draft_reference(root, tmp_path / "inputs", run_reference=False)
    assert result["status"] == "INPUTS_PREPARED"
    assert result["expected_top1"] is None and result["expected_hidden"] is None


@pytest.mark.parametrize("invalid_length", [0, 7, True, 1.5])
def test_reject_invalid_prefix_before_writing(prepared, tmp_path, invalid_length):
    root, _, _, _, _ = prepared
    with pytest.raises(ValueError, match="valid_length"):
        subject.prepare_draft_reference(root, tmp_path / "bad", valid_length=invalid_length)
    assert not (tmp_path / "bad").exists()


def test_corrupted_weight_is_rejected_before_reference_load(prepared, tmp_path):
    root, manifest, _, _, _ = prepared
    record = manifest["weights"]["target.lm_head.weight"]
    with (root / record["path"]).open("r+b") as stream:
        stream.write(b"\x00\x00")
    with pytest.raises(ValueError, match="SHA256"):
        subject.prepare_draft_reference(root, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_target_decoder_weight_is_rejected(prepared):
    root, manifest, _, _, _ = prepared
    manifest["weights"]["target.layers.0.weight"] = manifest["weights"]["target.lm_head.weight"]
    (root / "prepared.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly"):
        subject.load_draft_reference(root)


def test_failure_retains_input_identity_and_error(prepared, tmp_path, monkeypatch):
    root, _, _, _, _ = prepared
    def fail(*args):
        raise RuntimeError("reference fixture failure")
    monkeypatch.setattr(subject, "_load_graph", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        subject.prepare_draft_reference(root, tmp_path / "failed")
    result = json.loads((tmp_path / "failed" / "report.json").read_text())
    assert result["status"] == "FAILED" and len(result["inputs"]) == 3
    assert result["expected_top1"] is None and "fixture failure" in result["error"]


def test_finite_all_zero_hidden_is_not_accepted_as_reference(prepared, tmp_path, monkeypatch):
    root, _, _, _, _ = prepared
    graph = subject.load_draft_reference(root)
    # Simulate finite collapse at the final norm. The head remains finite and
    # argmax returns zeros, so the former finite-only guard would accept it.
    graph.draft.norm.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
    monkeypatch.setattr(subject, "_load_graph", lambda *args: graph)
    folder = tmp_path/"collapsed-reference"
    with pytest.raises(FloatingPointError, match="degenerate all-zero"):
        subject.prepare_draft_reference(root, folder)
    report = json.loads((folder/"report.json").read_text())
    assert report["status"] == "FAILED"
    assert report["expected_top1"] is report["expected_hidden"] is None
    assert len(report["inputs"]) == 3

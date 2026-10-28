"""Standalone full draft boundaries; target decoder construction is forbidden."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from qwen35_moe_dflash import draft_standalone as ds
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_weights import checkpoint_key, tiny_weights


def fake_ge(root):
    ge = root / "ge"
    (ge / "include/graph").mkdir(parents=True)
    (ge / "lib64").mkdir()
    for name in ("graph.h", "operator.h"):
        (ge / "include/graph" / name).write_text("mock header")
    (ge / "lib64/libgraph.so").write_bytes(b"mock GE library")
    return ge


def test_full_plan_preserves_all_six_layers_full_vocabulary_and_only_shared_target_tensors(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    plan = ds.plan_draft(tmp_path / "plan", 64)
    assert not plan["tiny"] and plan["draft_layers"] == 6 and plan["draft_block_size"] == 16
    assert plan["vocab_size"] == 248320 and plan["feature_width"] == 16384
    assert len(plan["graphs"]) == 1 and plan["graphs"][0]["name"] == "draft"
    graph = ds._graph(64)[2]
    assert {key.split(".")[2] for key in graph.weights if key.startswith("draft.layers.")} == set(map(str, range(6)))
    assert {key for key in graph.weights if key.startswith("target.")} == {
        "target.embed_tokens.weight", "target.lm_head.weight"}
    assert graph.weights["target.lm_head.weight"].shape == (248320, 2048)
    assert [(v.name, v.shape, v.dtype) for v in graph.outputs] == [
        ("draft_top1", (1, 15), "int32"), ("draft_hidden", (1, 15, 2048), "float16")]
    assert plan["target_decoder"] == "NOT_CONSTRUCTED" and plan["device_validation"] == "NOT_RUN"
    assert not list((tmp_path / "plan").rglob("*.bin"))


def test_preparation_does_not_import_torch_or_target_builder(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.mkdir()
    for module in ("torch", "torch_npu", "torchair", "transformers"):
        (blocker / (module + ".py")).write_text("raise AssertionError('unexpected framework import')\n")
    repo = Path(__file__).resolve().parents[1]
    code = """
import sys
from qwen35_moe_dflash.draft_standalone import prepare_draft
prepare_draft(sys.argv[1],sequence_length=8,tiny=True)
assert 'qwen35_moe_dflash.ge_target' not in sys.modules
assert 'qwen35_moe_dflash.ge_fused.target' not in sys.modules
"""
    env = dict(os.environ, AI_RUN_DIR=str(tmp_path), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join((str(blocker), str(repo / "model"))))
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "prepared")], env=env,
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _safetensors(root, values, *, unavailable_decoder=False):
    root.mkdir()
    header, chunks, offset = {}, [], 0
    for key, value in values.items():
        data = (value.astype("<f4").view("<u4") >> 16).astype("<u2").tobytes()
        header[key] = {"dtype": "BF16", "shape": list(value.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        chunks.append(data)
    raw = json.dumps(header).encode()
    (root / "selected.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))
    index = {key: "selected.safetensors" for key in values}
    if unavailable_decoder:
        index["model.language_model.layers.0.mlp.experts.gate_up_proj"] = "DO_NOT_READ_DECODER.safetensors"
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


def test_checkpoint_preparation_reads_only_selected_tensors_and_retains_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    configs = configurations(tiny=True)
    monkeypatch.setattr(ds, "configurations", lambda **kwargs: configs)
    monkeypatch.setattr(ds, "_standalone_checkpoint_metadata", lambda *args: [
        {"owner": "target", "repo": "fixture-target", "expected_revision": "fixture"},
        {"owner": "draft", "repo": "fixture-draft", "expected_revision": "fixture"}])
    graph = ds._graph(8)[2]
    values = tiny_weights({key: v.shape for key, v in graph.weights.items()})
    selected = {"target": {}, "draft": {}}
    for key, value in values.items():
        owner, source_key = checkpoint_key(key)
        selected[owner][source_key] = value
    _safetensors(tmp_path / "target", selected["target"], unavailable_decoder=True)
    _safetensors(tmp_path / "draft", selected["draft"])
    rows = []
    plan = ds.prepare_draft(tmp_path / "prepared", 8, target_path=tmp_path / "target",
                            draft_path=tmp_path / "draft", progress=rows.append)
    assert plan["status"] == "PREPARED" and plan["external_weights"]
    assert len(rows) == len(values)
    assert set(plan["weights"]) == set(values)
    target_source = plan["source"]["checkpoints"][0]
    assert {item["tensor"] for item in target_source["selected_tensors"]} == {
        "model.language_model.embed_tokens.weight", "lm_head.weight"}
    assert all(len(item["sha256"]) == 64 for source in plan["source"]["checkpoints"]
               for item in source["selected_tensors"])
    for source in plan["source"]["checkpoints"]:
        for tensor in source["selected_tensors"]:
            raw = (tmp_path / source["owner"] / tensor["shard"]).read_bytes()
            payload = raw[tensor["offset"]:tensor["offset"] + tensor["bytes"]]
            assert tensor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not (tmp_path / "target/DO_NOT_READ_DECODER.safetensors").exists()


def test_full_preparation_requires_matching_checkpoint_metadata_before_output(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="original draft and target"):
        ds.prepare_draft(tmp_path / "missing")
    def mismatch(*args):
        raise ValueError("config differs from locked revision")
    monkeypatch.setattr(ds, "_standalone_checkpoint_metadata", mismatch)
    with pytest.raises(ValueError, match="locked revision"):
        ds.prepare_draft(tmp_path / "bad", target_path="target", draft_path="draft")
    assert not (tmp_path / "bad").exists()


def _locked_configs(target, draft):
    specs = Path(__file__).resolve().parents[1] / "specs"
    for root, name in ((target, "target-config.json"), (draft, "draft-config.json")):
        root.mkdir(exist_ok=True)
        (root / "config.json").write_bytes((specs / name).read_bytes())


def test_standalone_metadata_requires_config_but_not_pair_index_or_generation_settings(tmp_path):
    target, draft = tmp_path / "target", tmp_path / "draft"
    _locked_configs(target, draft)
    identities = ds._standalone_checkpoint_metadata(target, draft)
    for item in identities:
        assert item["metadata"]["config.json"]["status"] == "MATCHED"
        assert item["metadata"]["model.safetensors.index.json"]["status"] == "ABSENT_ALLOWED"
        assert item["metadata"]["generation_config.json"]["status"] == "ABSENT_ALLOWED"
        assert item["revision_verification"] == "NOT_ESTABLISHED" and "revision" not in item
    (target / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    (target / "generation_config.json").write_text('{"temperature":0.7}')
    identities = ds._standalone_checkpoint_metadata(target, draft)
    assert identities[0]["metadata"]["model.safetensors.index.json"]["status"] == "DIFFERS_FROM_LOCK"
    assert identities[0]["metadata"]["generation_config.json"]["status"] == "DIFFERS_FROM_LOCK"
    (draft / "config.json").write_text('{}')
    with pytest.raises(ValueError, match="config.json differs from locked configuration"):
        ds._standalone_checkpoint_metadata(target, draft)


def _indexless_fixture(tmp_path, monkeypatch, *, omit_head=False, bad_last_shape=False):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    configs = configurations(tiny=True)
    monkeypatch.setattr(ds, "configurations", lambda **kwargs: configs)
    graph = ds._graph(8)[2]
    values = tiny_weights({key: value.shape for key, value in graph.weights.items()})
    selected = {"target": {}, "draft": {}}
    for key, value in values.items():
        owner, source_key = checkpoint_key(key)
        if omit_head and key == "target.lm_head.weight":
            continue
        selected[owner][source_key] = value
    if bad_last_shape:
        selected["draft"]["norm.weight"] = np.zeros((1,), dtype="<f2")
    target, draft = tmp_path / "target", tmp_path / "draft"
    _safetensors(target, selected["target"])
    _safetensors(draft, selected["draft"])
    _locked_configs(target, draft)
    (target / "model.safetensors.index.json").unlink()
    (target / "selected.safetensors").rename(target / "model.safetensors-00001-of-00014.safetensors")
    (draft / "model.safetensors.index.json").unlink()
    (draft / "selected.safetensors").rename(draft / "model.safetensors")
    return target, draft


def test_indexless_partial_target_shards_suffice_when_both_shared_tensors_exist(tmp_path, monkeypatch):
    target, draft = _indexless_fixture(tmp_path, monkeypatch)
    result = ds.prepare_draft(tmp_path / "prepared", 8, target_path=target, draft_path=draft)
    assert result["status"] == "PREPARED"
    identity = result["source"]["checkpoints"][0]
    assert identity["metadata"]["model.safetensors.index.json"]["status"] == "ABSENT_ALLOWED"
    assert identity["revision_verification"] == "NOT_ESTABLISHED"
    assert identity["discovery"]
    assert {item["tensor"] for item in identity["selected_tensors"]} == {
        "model.language_model.embed_tokens.weight", "lm_head.weight"}
    assert not (target / "model.safetensors.index.json").exists(), "reader must not synthesize an upstream index"


@pytest.mark.parametrize("problem", ["missing_head", "late_draft_shape"])
def test_all_required_tensors_preflight_before_any_conversion(tmp_path, monkeypatch, problem):
    from qwen35_moe_dflash.ge_weights import SafeTensorFiles
    target, draft = _indexless_fixture(tmp_path, monkeypatch, omit_head=problem == "missing_head",
                                       bad_last_shape=problem == "late_draft_shape")
    def no_convert(*args, **kwargs):
        pytest.fail("conversion began before every required tensor passed preflight")
    monkeypatch.setattr(SafeTensorFiles, "write_fp16", no_convert)
    with pytest.raises(ValueError, match="lm_head.weight" if problem == "missing_head" else "norm.weight"):
        ds.prepare_draft(tmp_path / "prepared", 8, target_path=target, draft_path=draft)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("include_hidden", [True, False])
def test_export_only_builds_draft_and_regenerates_sources_with_external_payloads(tmp_path, monkeypatch, include_hidden):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared, output = tmp_path / "prepared", tmp_path / "air"
    plan = ds.prepare_draft(prepared, 8, tiny=True, include_hidden=include_hidden)
    (prepared / "draft/draft.cpp").write_text("STALE_CODE")
    original = (prepared / "prepared.json").read_bytes()
    calls = []
    def tool(command, **kwargs):
        calls.append(command)
        if "-o" in command:
            source = next(Path(x) for x in command if x.endswith(".cpp"))
            assert "STALE_CODE" not in source.read_text()
            Path(command[command.index("-o") + 1]).write_bytes(b"fixture exporter")
        else:
            Path(command[1]).write_bytes(b"fixture AIR")
    monkeypatch.setattr(subprocess, "run", tool)
    manifest = ds.export_draft(prepared, output, fake_ge(tmp_path))
    assert manifest["status"] == "EXPORTED" and len(calls) == 2
    assert manifest["artifact_kind"] == "qwen35-cpro-draft-air"
    assert manifest["graphs"][0]["air"] == "draft.air"
    assert (output / "draft.air").is_file()
    assert len(manifest["graphs"]) == 1 and manifest["graphs"][0]["name"] == "draft"
    assert manifest["graphs"][0]["outputs"] == (["draft_top1", "draft_hidden"] if include_hidden else ["draft_top1"])
    assert len(manifest["graphs"][0]["files"]) == len(plan["weights"]) + 1
    assert (prepared / "prepared.json").read_bytes() == original
    assert manifest["device_validation"] == manifest["numerical_validation"] == "NOT_RUN"


@pytest.mark.parametrize("corruption", ["payload", "config", "weight_set", "identity"])
def test_export_rejects_corruption_before_output_or_tool_execution(tmp_path, monkeypatch, corruption):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = ds.prepare_draft(prepared, 8, tiny=True)
    if corruption == "payload":
        (prepared / next(iter(plan["weights"].values()))["path"]).write_bytes(b"bad")
    elif corruption == "config":
        plan["draft_config"]["num_hidden_layers"] = 1
    elif corruption == "weight_set":
        plan["weights"].pop(next(iter(plan["weights"])))
    else:
        plan["source"]["fp16_payload_sha256"] = "0" * 64
    (prepared / "prepared.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        ds.export_draft(prepared, tmp_path / "air", tmp_path / "unneeded-ge")
    assert not (tmp_path / "air").exists()


def test_failed_serialize_retains_evidence_without_success_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ds.prepare_draft(tmp_path / "prepared", 8, tiny=True)
    def tool(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"fixture exporter")
        else:
            Path(command[1]).write_bytes(b"partial AIR")
            kwargs["stdout"].write("ERROR: fixture graph serialization failed\n")
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", tool)
    with pytest.raises(RuntimeError, match="serialization failed"):
        ds.export_draft(tmp_path / "prepared", tmp_path / "air", fake_ge(tmp_path))
    assert not (tmp_path / "air/manifest.json").exists()
    failure = json.loads((tmp_path / "air/failure.json").read_text())
    assert failure["stage"] == "draft-serialize" and failure["device_validation"] == "NOT_RUN"

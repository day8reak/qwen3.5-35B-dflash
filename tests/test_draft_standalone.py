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
    monkeypatch.setattr(ds, "_locked_checkpoints", lambda *args: [
        {"repo": "fixture-target", "revision": "fixture"}, {"repo": "fixture-draft", "revision": "fixture"}])
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
    monkeypatch.setattr(ds, "_locked_checkpoints", mismatch)
    with pytest.raises(ValueError, match="locked revision"):
        ds.prepare_draft(tmp_path / "bad", target_path="target", draft_path="draft")
    assert not (tmp_path / "bad").exists()


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

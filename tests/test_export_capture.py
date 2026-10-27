"""Real Dynamo graph capture and changed-input checks, independent of TorchAir."""
import copy
import torch
import pytest
from reference.tiny import create
from qwen35_moe_dflash.export_moe import prepare_export_target, StaticExportExperts


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_explicit_export_loop_matches_upstream(dtype):
    torch.set_num_threads(1)
    target, draft = create()
    target.to(dtype); draft.to(dtype)
    oracle = copy.deepcopy(target)
    candidate = prepare_export_target(target)
    for tokens in [[4], [3, 24, 9], [4, 7, 22, 11, 5, 3, 18, 42]]:
        ids = torch.zeros((1, 8), dtype=torch.int32); mask = torch.zeros_like(ids)
        ids[0, :len(tokens)] = torch.tensor(tokens); mask[0, :len(tokens)] = 1
        with torch.inference_mode():
            expected = oracle(ids, mask)
            actual = candidate(ids, mask)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-4 if dtype==torch.float32 else 1e-2,
                                   atol=1e-5 if dtype==torch.float32 else 1e-3)


def test_real_dynamo_fullgraph_and_dynamic_routes():
    torch.set_num_threads(1)
    target, draft = create()
    oracle = copy.deepcopy(target)
    candidate = prepare_export_target(target)
    ids = torch.tensor([[1, 2, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
    mask = (ids != 0).int()
    with torch.inference_mode():
        target_graph = torch.export.export(candidate, (ids, mask), strict=True).module()
        draft_graph = torch.export.export(draft, (torch.zeros(1,8,32), ids, mask), strict=True).module()
        for tokens in [[4], [8, 20, 16, 3, 2], list(range(1,9))]:
            ids.zero_(); mask.zero_()
            ids[0, :len(tokens)] = torch.tensor(tokens); mask[0, :len(tokens)] = 1
            top1, features = oracle(ids, mask)
            captured = target_graph(ids, mask)
            torch.testing.assert_close(captured[0], top1, rtol=0, atol=0)
            torch.testing.assert_close(captured[1], features, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(draft_graph(captured[1], ids, mask), draft(features, ids, mask), rtol=0, atol=0)


def test_hierarchical_fx_matches_existing_target_and_draft_on_changed_inputs():
    import io
    from qwen35_moe_dflash.export import _hierarchical_pytorch_graphs
    torch.set_num_threads(1)
    target, draft = create()
    target = prepare_export_target(target, mode="static")
    target_fx, draft_fx = _hierarchical_pytorch_graphs(target, draft, 8)
    assert target_fx.language_model.embed_tokens is draft_fx.embedding
    assert target_fx.lm_head is draft_fx.lm_head
    archive = io.BytesIO()
    torch.save((target_fx, draft_fx), archive)
    archive.seek(0)
    reloaded = torch.load(archive, weights_only=False)
    for tokens in ([4], [8, 20, 16, 3, 2], list(range(1, 9))):
        ids = torch.zeros((1, 8), dtype=torch.int32)
        mask = torch.zeros_like(ids)
        ids[0, :len(tokens)] = torch.tensor(tokens)
        mask[0, :len(tokens)] = 1
        with torch.inference_mode():
            expected = target(ids, mask)
            for target_graph, draft_graph in ((target_fx, draft_fx), reloaded):
                actual = target_graph(ids, mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(draft_graph(actual[1], ids, mask), draft(expected[1], ids, mask), rtol=0, atol=0)


def test_full_pytorch_cli_builds_real_meta_graphs_and_complete_parameter_contract(tmp_path, monkeypatch):
    import json
    from qwen35_moe_dflash.cli import execute, parser
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    output = tmp_path / "full"
    execute(parser().parse_args(["build-full-pytorch", "--sequence-length", "128", "--output", str(output)]))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "META_STRUCTURE_READY" and manifest["tiny"] is False
    assert (manifest["target_layers"], manifest["draft_layers"]) == (40, 6)
    assert manifest["feature_layers"] == [1, 6, 11, 16, 22, 27, 32, 37]
    assert manifest["feature_width"] == 16384 and manifest["draft_block_size"] == 16
    assert manifest["checkpoint_loaded"] is False
    assert manifest["air_export"] == manifest["device_execution"] == manifest["aten_lowering"] == "NOT_RUN"
    assert manifest["stateful_torchair_export"] == "NOT_IMPLEMENTED"
    parameters = json.loads((output / "parameters.json").read_text())
    assert parameters["target"]["language_model.layers.39.self_attn.q_proj.weight"]["shape"] == [8192, 2048]
    assert parameters["target"]["language_model.layers.0.mlp.experts.gate_up_proj"]["shape"] == [256, 1024, 2048]
    assert parameters["draft"]["draft.fc.weight"]["shape"] == [2048, 16384]
    assert parameters["target"]["lm_head.weight"]["storage_id"] == parameters["draft"]["lm_head.weight"]["storage_id"]
    assert sum(layer["layer_type"] == "linear_attention" for layer in json.loads((output / "target/layers.json").read_text())) == 30
    models = torch.load(output / "models.meta.fx.pt", weights_only=False)
    assert models["target"].lm_head is models["draft"].lm_head
    assert models["target"].language_model.embed_tokens is models["draft"].embedding
    for name, expected_layers, prefix in (("target", 40, "language_model.layers."), ("draft", 6, "draft.layers.")):
        graph = models[name]
        graph.graph.lint()
        assert sum(node.op == "call_module" and str(node.target).startswith(prefix) for node in graph.graph.nodes) == expected_layers
        assert all(tensor.device.type == "meta" for tensor in graph.parameters())
        assert all(tensor.device.type == "meta" for tensor in graph.buffers())
        # FX's pickle reader retraces generated Python, so harmless node names
        # and operator-vs-method spelling can differ after a round trip.
        compile((output / name / "forward.py").read_text(), f"{name}/forward.py", "exec")
    assert (output / "models.meta.fx.pt").stat().st_size < 10 << 20

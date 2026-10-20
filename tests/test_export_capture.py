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

import pytest
import torch
from qwen35_moe_dflash.runtime import TorchExecutor, generate
from reference.tiny import create


class PatternExecutor:
    sequence_length, draft_width, vocab_size = 64, 3, 256

    def __init__(self, corrupt=None):
        self.corrupt = corrupt
        self.prefix = []
        self.calls = []
        self.drafts = 0

    def target(self, prefix):
        self.prefix = prefix[:]
        self.calls.append(prefix[:])
        return [t + 1 for t in prefix] + [0] * (self.sequence_length - len(prefix))

    def draft(self):
        self.drafts += 1
        result = [self.prefix[-1] + i + 1 for i in range(self.draft_width)]
        if self.corrupt is not None:
            result[self.corrupt] += 100
        return result


@pytest.mark.parametrize("corrupt", [None, 0, 1, 2])
@pytest.mark.parametrize("eos", [(), (4,), (6,), (10,)])
@pytest.mark.parametrize("budget", [1, 2, 3, 8])
def test_greedy_branches_and_budget(corrupt, eos, budget):
    backend = PatternExecutor(corrupt)
    expected = generate(backend, [1, 2], ordinary=True, max_new_tokens=budget, eos_token_ids=eos)
    actual = generate(backend, [1, 2], max_new_tokens=budget, eos_token_ids=eos)
    assert (actual.token_ids, actual.stop_reason) == (expected.token_ids, expected.stop_reason)
    assert len(actual.token_ids) <= budget
    assert all(token < 100 for token in actual.token_ids)


def test_ordinary_never_calls_draft():
    backend = PatternExecutor()
    generate(backend, [1], ordinary=True, max_new_tokens=4, eos_token_ids=())
    assert backend.drafts == 0


@pytest.mark.parametrize("prompt,budget", [([], 3), ([1], 0), ([1] * 64, 2), ([-1], 4), ([256], 1)])
def test_invalid_requests(prompt, budget):
    with pytest.raises(ValueError):
        generate(PatternExecutor(), prompt, max_new_tokens=budget, eos_token_ids=())


@pytest.mark.parametrize("prompt", [[1], [1, 2, 3], [8, 4, 7, 3, 2]])
def test_real_moe_gdn_ordinary_dflash_and_independent_target(prompt):
    torch.set_num_threads(1)
    target, draft = create()
    backend = TorchExecutor(target, draft, 16, pad_token_id=0)
    ordinary = generate(backend, prompt, ordinary=True, max_new_tokens=6, eos_token_ids=())
    dflash = generate(backend, prompt, max_new_tokens=6, eos_token_ids=())
    assert ordinary.token_ids == dflash.token_ids
    # Independent unpadded target-only path, without the graph wrapper,
    # feature selection or DFlash scheduler.
    prefix = prompt[:]
    with torch.inference_mode():
        for token in ordinary.token_ids:
            output = target.language_model(input_ids=torch.tensor([prefix]), use_cache=False)
            expected = target.lm_head(output.last_hidden_state[:, -1:]).argmax(-1).item()
            assert token == expected
            prefix.append(token)


def test_draft_ignores_padded_features_and_excludes_anchor():
    target, draft = create()
    ids = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]], dtype=torch.int32)
    mask = (ids != 0).int()
    with torch.inference_mode():
        _, features = target(ids, mask)
        expected = draft(features, ids, mask)
        changed = features.clone()
        changed[:, 2:] = 10000  # anchor and padding must never be context
        assert torch.equal(expected, draft(changed, ids, mask))


def test_export_experts_unseen_routes_match_upstream():
    from qwen35_moe_dflash.export_moe import ExportExperts
    target, _ = create()
    upstream = target.language_model.layers[0].mlp.experts
    candidate = ExportExperts(upstream)
    for seed in range(4):
        torch.manual_seed(seed)
        hidden = torch.randn(7, 16)
        # Includes experts absent from another seed, and an entirely idle expert.
        selected = torch.stack([torch.randperm(3)[:2] for _ in range(7)])
        weights = torch.softmax(torch.randn(7, 2), -1)
        with torch.no_grad():
            torch.testing.assert_close(candidate(hidden, selected, weights),
                                       upstream(hidden, selected, weights), rtol=1e-6, atol=1e-7)

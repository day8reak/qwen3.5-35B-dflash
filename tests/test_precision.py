import torch
from reference.tiny import create
from qwen35_moe_dflash.precision import prepare_fp16_pair, floating_dtype_audit
from qwen35_moe_dflash.runtime import TorchExecutor, generate


def test_fp16_graph_internals_and_greedy_equivalence():
    torch.set_num_threads(1)
    target, draft = create()
    target, draft, audit = prepare_fp16_pair(target, draft, 8)
    for report in audit.values():
        assert report["status"] == "PASS"
        assert set(report["floating_node_values"]) == {"torch.float16"}
        assert report["explicit_dtype_rewrites"] > 0
    executor = TorchExecutor(target, draft, 8, 0)
    for prompt in [[1], [1, 2, 3], [7, 8]]:
        ordinary = generate(executor, prompt, ordinary=True, max_new_tokens=4, eos_token_ids=())
        speculative = generate(executor, prompt, max_new_tokens=4, eos_token_ids=())
        assert (ordinary.token_ids, ordinary.stop_reason) == (speculative.token_ids, speculative.stop_reason)
        assert executor.features.dtype == torch.float16


def test_dtype_audit_catches_hidden_float32_computation():
    class Bad(torch.nn.Module):
        def forward(self, x):
            return x.float().square().half()
    with torch.inference_mode():
        program = torch.export.export(Bad(), (torch.ones(2, dtype=torch.float16),))
    report = floating_dtype_audit(program)
    assert report["status"] == "FAIL"
    assert "torch.float32" in report["floating_node_values"]

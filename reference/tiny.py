"""Reduced random-weight fixture using real Qwen3.5 MoE and Gated DeltaNet."""
import json
from pathlib import Path
import torch
from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.model import DFlashDraftModel
from qwen35_moe_dflash.graphs import TargetGraph, DraftGraph


def create(seed=17, *, quantization_aligned=False):
    torch.manual_seed(seed)
    hidden = 128 if quantization_aligned else 16
    head = hidden // 2
    config = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=hidden, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=head,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=head, linear_value_head_dim=head,
        linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=hidden, shared_expert_intermediate_size=hidden,
        max_position_embeddings=128, pad_token_id=0,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    target = Qwen3_5MoeForCausalLM(config).eval()
    raw = json.loads((Path(__file__).resolve().parents[1] / "specs/draft-config.json").read_text())
    raw.update(hidden_size=hidden, intermediate_size=hidden*2, vocab_size=64,
               num_hidden_layers=2, num_target_layers=4, num_attention_heads=2,
               num_key_value_heads=1, head_dim=head, max_position_embeddings=128,
               layer_types=["sliding_attention", "full_attention"], sliding_window=8)
    raw["dflash_config"] = {"block_size": 4, "mask_token_id": 63, "target_layer_ids": [0, 2]}
    draft = DFlashDraftModel(Qwen35DFlashConfig.from_dict(raw), dtype=torch.float32).eval()
    with torch.no_grad():
        for name, parameter in draft.named_parameters():
            if "norm.weight" in name:
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.04)
    return (TargetGraph(target.model, target.lm_head, draft.config, head_chunk_rows=4).eval(),
            DraftGraph(draft, target.model.embed_tokens, target.lm_head).eval())

"""Separate target and draft graphs for the CPro HIAI recompute baseline."""
from __future__ import annotations

import torch
from torch import nn

from qwen35_moe_dflash.model import extract_context_feature


class TargetGraph(nn.Module):
    """Preserve Qwen3.5 MoE/Gated DeltaNet math in the upstream text decoder.

    The unmodified reference returns FP32 features. Deployment lowering makes
    all floating tensors, including the HIAI feature boundary, FP16.
    Input ABI: int32 [1,S] IDs and a right-padded 0/1 mask.
    Output ABI: int32 [1,S] Top1; float32 [1,S,F] features.
    """

    def __init__(self, language_model, lm_head, draft_config, head_chunk_rows=16):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.config = draft_config
        self.head_chunk_rows = int(head_chunk_rows)
        tc = language_model.config
        for key, expected in {
            "hidden_size": draft_config.hidden_size,
            "num_hidden_layers": draft_config.num_target_layers,
            "vocab_size": draft_config.vocab_size,
        }.items():
            if getattr(tc, key) != expected:
                raise ValueError(f"target/draft {key} mismatch")
        if self.head_chunk_rows < 1:
            raise ValueError("head_chunk_rows must be positive")
        expected = (tc.vocab_size, tc.hidden_size)
        if tuple(language_model.embed_tokens.weight.shape) != expected:
            raise ValueError("target embedding shape mismatch")
        if tuple(lm_head.weight.shape) != expected:
            raise ValueError("target LM head shape mismatch")

    def forward(self, input_ids, attention_mask):
        output = self.language_model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        features = extract_context_feature(
            output.hidden_states, self.config.target_layer_ids
        )
        # Bound the otherwise S*vocabulary output allocation. Each row still
        # covers the complete vocabulary and uses the actual untied target head.
        top1 = []
        for start in range(0, input_ids.shape[1], self.head_chunk_rows):
            logits = self.lm_head(
                output.last_hidden_state[:, start : start + self.head_chunk_rows]
            )
            top1.append(logits.argmax(-1).to(torch.int32))
        return torch.cat(top1, dim=1), features.float()


class DraftGraph(nn.Module):
    """One clean anchor followed by masks; exclude anchor from target context."""

    def __init__(self, draft, embedding, lm_head):
        super().__init__()
        self.draft = draft
        self.embedding = embedding
        self.lm_head = lm_head
        c = draft.config
        expected = (c.vocab_size, c.hidden_size)
        if tuple(embedding.weight.shape) != expected or tuple(lm_head.weight.shape) != expected:
            raise ValueError("draft must share the target embedding and target LM head")

    def forward(self, target_features, input_ids, attention_mask):
        c = self.draft.config
        valid = attention_mask.bool()
        length = valid.long().sum(-1, keepdim=True)
        anchor = length - 1
        anchor_ids = input_ids.long().gather(1, anchor)
        block_ids = torch.cat((
            anchor_ids,
            torch.full((1, c.block_size - 1), c.mask_token_id,
                       dtype=torch.long, device=input_ids.device),
        ), dim=-1)
        rows = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
        context_valid = valid & (rows < anchor)
        context_positions = (context_valid.long().cumsum(-1) - 1).clamp_min(0)
        block_positions = anchor + torch.arange(c.block_size, device=input_ids.device).view(1, -1)
        positions = torch.cat((context_positions, block_positions), dim=-1)
        noise = self.draft.embed_block(block_ids, self.embedding.weight)
        hidden = self.draft.draft_hidden(
            target_features.to(noise.dtype), noise, positions,
            context_attention_mask=context_valid,
        )
        return self.lm_head(hidden).argmax(-1).to(torch.int32)


def load_graphs(target_path, draft_path, device="cpu", *, dtype=torch.float16):
    """Load original checkpoints locally, with explicitly selected precision.

    No remote code or model downloads are enabled. The vision tower and MTP
    are outside this text-only DFlash contract.
    """
    from transformers import Qwen3_5MoeForConditionalGeneration
    from qwen35_moe_dflash.config import Qwen35DFlashConfig, audit_official_35b_dflash_config
    from qwen35_moe_dflash.model import DFlashDraftModel
    from qwen35_moe_dflash.reference.weights import audit_dflash_checkpoint, load_dflash_weights
    from qwen35_moe_dflash.reference.checkpoints import audit_checkpoints
    audit = audit_checkpoints(target_path, draft_path, hash_weights=False)
    if audit["status"] != "PASS":
        raise ValueError(f"checkpoint contract failed: {audit['errors']}")
    c = Qwen35DFlashConfig.from_pretrained(draft_path)
    errors = audit_official_35b_dflash_config(c)
    if errors:
        raise ValueError(f"incompatible draft: {errors}")
    audit = audit_dflash_checkpoint(draft_path)
    if audit["status"] != "PASS":
        raise ValueError(f"draft checkpoint audit failed: {audit['errors']}")
    owner = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        target_path, dtype=dtype, attn_implementation="eager",
        local_files_only=True,
    ).eval()
    owner.model.visual = None
    owner = owner.to(device)
    language = owner.model.language_model
    language.config._experts_implementation = "eager"
    head = owner.lm_head
    owner.model.visual = None
    draft = DFlashDraftModel(c, dtype=dtype).to(device).eval()
    load_dflash_weights(draft, draft_path)
    return TargetGraph(language, head, c).eval(), DraftGraph(
        draft, language.embed_tokens, head
    ).eval()

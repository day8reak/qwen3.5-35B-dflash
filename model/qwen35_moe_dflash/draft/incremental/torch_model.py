"""PyTorch incremental Draft with the same committed-context KV contract as GE.

Only the RMS callable changes between CPU reference and TorchAir export. The
export uses the installed npu_rms_norm converter, without custom registration.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.graph.incremental_draft import _validate, incremental_draft_abi
from qwen35_moe_dflash.model import DFlashDraftModel
from qwen35_moe_dflash.ops import _rotate_half


def reference_rms(x, gamma, epsilon):
    """Fused RMS formula with FP32 statistics and direct, non-zero-centered gamma."""
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    return x * rstd * gamma, rstd


class IncrementalDraft(nn.Module):
    def __init__(self, draft, embedding, head, *, context_rows=16, cache_capacity=192,
                 max_draft_tokens=15, rms=reference_rms):
        super().__init__()
        self.config = c = draft.config
        _validate(c, context_rows, cache_capacity, True, max_draft_tokens)
        self.draft, self.embedding, self.head = draft, embedding, head
        self.context_rows, self.capacity, self.k = context_rows, cache_capacity, max_draft_tokens
        self.rms = rms
        self.register_buffer('context_offsets', torch.arange(context_rows, dtype=torch.int32).view(1, -1))
        self.register_buffer('block_offsets', torch.arange(max_draft_tokens + 1, dtype=torch.int32).view(1, -1))
        self.register_buffer('cache_positions', torch.arange(cache_capacity, dtype=torch.int32).view(1, -1))
        self.register_buffer('mask_tokens', torch.full((1, max_draft_tokens), c.mask_token_id, dtype=torch.int32))
        self.register_buffer('query_repeat', torch.zeros(max_draft_tokens + 1, dtype=torch.int32))
        self.register_buffer('head_repeat', torch.zeros(c.num_attention_heads, dtype=torch.int32))
        self.register_buffer('kv_heads', torch.arange(c.num_key_value_heads, dtype=torch.int32)
                             .repeat_interleave(c.num_key_value_groups))
        witness = torch.zeros((1, 1, 1, cache_capacity + max_draft_tokens + 1), dtype=torch.bool)
        witness[..., cache_capacity] = True
        self.register_buffer('anchor_witness', witness)

    def norm(self, x, norm):
        # Request FP32 normalization through the same RMS API as Target CPro.
        # Draft weights are gamma; never apply the Target's 1+w convention.
        return self.rms(x.float(), norm.weight.float(), norm.eps)[0].to(x.dtype)

    def rope(self, x, positions):
        cosine, sine = self.draft.rotary(positions, x.dtype)
        return x * cosine.unsqueeze(1) + _rotate_half(x) * sine.unsqueeze(1)

    def forward(self, target_features, start_position, valid_rows, anchor, proposal_count,
                past_k_0, past_v_0, past_k_1, past_v_1, past_k_2, past_v_2,
                past_k_3, past_v_3, past_k_4, past_v_4, past_k_5, past_v_5):
        c, rows, b = self.config, self.context_rows, self.k + 1
        end = start_position + valid_rows
        context_positions = start_position + self.context_offsets
        block_positions = end + self.block_offsets
        positions = torch.cat((self.cache_positions, block_positions), dim=1)
        key_valid = torch.cat((self.cache_positions < end, self.block_offsets <= proposal_count), dim=1)
        # Keep the GE workaround for failed BOOL query/head BroadcastTo:
        # integer Gather copies singleton axes exactly, before casting to BOOL.
        key_valid = torch.index_select(key_valid.reshape(1, 1, 1, self.capacity + b).int(),
                                       2, self.query_repeat).bool()
        witness = (self.block_offsets > proposal_count).reshape(1, 1, b, 1) & self.anchor_witness
        local = self.cache_positions.reshape(-1) - start_position
        # Match the GE integer Select expression, without tensor->Python scalars.
        local = torch.where(local < 0, torch.zeros_like(local), local)
        local = torch.where(local < rows, local, torch.full_like(local, rows - 1))
        # X90 GatherV2D rejects INT64 indices. These bounded row indices and
        # token IDs already fit the INT32 control ABI; do not promote to long.
        local = local.int()
        write = ((self.cache_positions >= start_position) & (self.cache_positions < end)).reshape(1, 1, self.capacity, 1)
        write = write.int().expand(1, c.num_key_value_heads, self.capacity, c.head_dim).bool()
        ids = torch.cat((anchor.reshape(1, 1), self.mask_tokens), dim=1)
        hidden = self.embedding(ids.int()) * c.input_embedding_scale
        # Preserve the proven full-K projection expression. A 16384-wide MatMul
        # was rejected on the selected OMG; changing frontend is not a fix.
        channels = self.draft.fc.in_features // 16
        projected = F.conv2d(target_features.reshape(rows, channels, 4, 4),
                             self.draft.fc.weight.reshape(c.hidden_size, channels, 4, 4))
        context = self.norm(projected.reshape(1, rows, c.hidden_size), self.draft.hidden_norm)
        past = ((past_k_0, past_v_0), (past_k_1, past_v_1), (past_k_2, past_v_2),
                (past_k_3, past_v_3), (past_k_4, past_v_4), (past_k_5, past_v_5))
        updated = []
        for index, layer in enumerate(self.draft.layers):
            attn = layer.self_attn
            x = self.norm(hidden, layer.input_layernorm)
            query = attn.q_proj(x).reshape(1, b, c.num_attention_heads, c.head_dim)
            query = self.rope(self.norm(query, attn.q_norm).transpose(1, 2), block_positions)
            new_k = attn.k_proj(context).reshape(1, rows, c.num_key_value_heads, c.head_dim)
            new_k = self.rope(self.norm(new_k, attn.k_norm).transpose(1, 2), context_positions)
            new_v = attn.v_proj(context).reshape(1, rows, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            present_k = torch.where(write, torch.index_select(new_k, 2, local), past[index][0])
            present_v = torch.where(write, torch.index_select(new_v, 2, local), past[index][1])
            updated.extend((present_k, present_v))
            block_k = attn.k_proj(x).reshape(1, b, c.num_key_value_heads, c.head_dim)
            block_k = self.rope(self.norm(block_k, attn.k_norm).transpose(1, 2), block_positions)
            block_v = attn.v_proj(x).reshape(1, b, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            key = torch.index_select(torch.cat((present_k, block_k), dim=2), 1, self.kv_heads)
            value = torch.index_select(torch.cat((present_v, block_v), dim=2), 1, self.kv_heads)
            visible = key_valid
            if c.layer_types[index] == 'sliding_attention':
                distance = block_positions.reshape(1, 1, b, 1) - positions.reshape(1, 1, 1, -1)
                visible = visible & (distance >= 0)
                if c.use_sliding_window:
                    visible = visible & (distance < c.sliding_window)
            visible = visible | witness
            visible = torch.index_select(visible.int(), 1, self.head_repeat).bool()
            scores = torch.matmul(query, key.transpose(-1, -2)).float() * (c.head_dim ** -0.5)
            probabilities = torch.softmax(torch.where(visible, scores, torch.full_like(scores, -float('inf'))), dim=-1).half()
            mixed = torch.matmul(probabilities, value).transpose(1, 2).reshape(1, b, c.query_width)
            hidden = (hidden.float() + attn.o_proj(mixed).float()).half()
            mlp = layer.mlp(self.norm(hidden, layer.post_attention_layernorm))
            hidden = (hidden.float() + mlp.float()).half()
        hidden = self.norm(hidden, self.draft.norm)[:, 1:, :]
        top1 = self.head(hidden).argmax(dim=-1).to(torch.int32)
        return (top1, hidden, *updated)

    def abi(self):
        contract = incremental_draft_abi(self.config, self.context_rows, self.capacity, max_draft_tokens=self.k)
        return {direction: contract['graph'][direction] for direction in ('inputs', 'outputs')}

    def example_inputs(self):
        # Controls are tensor inputs, including zero-row catchup/proposal modes.
        abi = self.abi()
        values = [torch.zeros(row['shape'], dtype=getattr(torch, row['dtype'])) for row in abi['inputs']]
        values[2].fill_(min(2, self.context_rows))
        values[3].fill_(1)
        values[4].fill_(self.k)
        return tuple(values)


def load_prepared(prepared, *, context_rows=16, cache_capacity=192, max_draft_tokens=15, rms=reference_rms):
    """Map existing standalone FP16 files; do not load the Target decoder."""
    import json
    root = Path(prepared).resolve(strict=True)
    plan = json.loads((root / 'prepared.json').read_text())
    if (plan.get('artifact_kind') != 'qwen35-draft-ge-prepared' or plan.get('status') != 'PREPARED'
            or plan.get('precision') != 'float16' or plan.get('quantization') is not None):
        raise ValueError('expected prepared standalone FP16 Draft weights')
    c = Qwen35DFlashConfig.from_dict(plan['draft_config'])
    _validate(c, context_rows, cache_capacity, True, max_draft_tokens)
    draft = DFlashDraftModel(c, device='meta', dtype=torch.float16)
    embedding = nn.Embedding(c.vocab_size, c.hidden_size, device='meta', dtype=torch.float16)
    head = nn.Linear(c.hidden_size, c.vocab_size, bias=False, device='meta', dtype=torch.float16)
    parameters = {'draft.' + name: (draft, name, p.shape) for name, p in draft.named_parameters()}
    parameters.update({'target.embed_tokens.weight': (embedding, 'weight', embedding.weight.shape),
                       'target.lm_head.weight': (head, 'weight', head.weight.shape)})
    if set(plan['weights']) != set(parameters):
        raise ValueError('prepared weights must cover exactly Draft plus target embedding/head')
    for key, (owner, name, shape) in parameters.items():
        record = plan['weights'][key]
        path = root / record['path']
        if (Path(record['path']).is_absolute() or root not in path.resolve().parents
                or record.get('shape') != list(shape) or record.get('dtype') != 'float16'
                or record.get('bytes') != math.prod(shape) * 2 or path.stat().st_size != record['bytes']):
            raise ValueError('prepared weight extent/descriptor differs: ' + key)
        tensor = torch.from_file(str(path), shared=False, size=math.prod(shape), dtype=torch.float16).reshape(shape)
        parent, _, attribute = name.rpartition('.')
        setattr(owner.get_submodule(parent) if parent else owner, attribute, nn.Parameter(tensor, requires_grad=False))
    return IncrementalDraft(draft, embedding, head, context_rows=context_rows,
                            cache_capacity=cache_capacity, max_draft_tokens=max_draft_tokens, rms=rms).eval()

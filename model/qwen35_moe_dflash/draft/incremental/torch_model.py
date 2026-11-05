"""PyTorch incremental Draft with the same committed-context KV contract as GE.

Export uses the installed npu_rms_norm converter and explicit attention
primitives. The optional Target kernel is retained for isolated diagnostics.
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
from .optimization import optimization_contract


def reference_rms(x, gamma, epsilon):
    """Fused RMS formula with FP32 statistics and direct, non-zero-centered gamma."""
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    return x * rstd * gamma, rstd


class IncrementalDraft(nn.Module):
    def __init__(self, draft, embedding, head, *, context_rows=16, cache_capacity=192,
                 max_draft_tokens=15, rms=reference_rms, attention_kernel=None,
                 optimization='baseline', top1_kernel=None):
        super().__init__()
        self.config = c = draft.config
        _validate(c, context_rows, cache_capacity, True, max_draft_tokens)
        self.host_embedding = optimization == 'packed-spatial-host-embedding'
        if self.host_embedding and c.input_embedding_scale != 1.0:
            raise ValueError('host embedding candidate requires the original unit embedding scale')
        self.draft, self.embedding, self.head = draft, None if self.host_embedding else embedding, head
        self.context_rows, self.capacity, self.k = context_rows, cache_capacity, max_draft_tokens
        self.rms = rms
        self.top1_kernel = top1_kernel
        self.optimization = optimization_contract(optimization)
        self.packed = optimization != 'baseline'
        self.spatial_projection = self.host_embedding or optimization in ('packed-spatial', 'packed-spatial-4d', 'packed-spatial-controls', 'packed-spatial-rope-table')
        self.rank4_spatial = self.host_embedding or optimization in ('packed-spatial-4d', 'packed-spatial-controls', 'packed-spatial-rope-table')
        self.shared_masks = self.rank4_spatial
        self.integer_controls = self.host_embedding or optimization in ('packed-spatial-controls', 'packed-spatial-rope-table')
        self.phase_table = self.host_embedding or optimization == 'packed-spatial-rope-table'
        if self.packed:
            if attention_kernel is not None:
                raise ValueError('packed optimization requires composed attention')
            # Concatenate output channels only: each output retains its full
            # dot product and the original FP16 weight bytes.
            for layer in draft.layers:
                self._pack(layer.self_attn, 'k_proj', 'v_proj', 'kv_proj')
                self._pack(layer.mlp, 'gate_proj', 'up_proj', 'gate_up_proj')
        from .torch_target import TargetAttention
        self.target_attention = (TargetAttention(c, cache_capacity, max_draft_tokens + 1, attention_kernel)
                                 if attention_kernel is not None else None)
        self.register_buffer('context_offsets', torch.arange(context_rows, dtype=torch.int32).view(1, -1))
        self.register_buffer('block_offsets', torch.arange(max_draft_tokens + 1, dtype=torch.int32).view(1, -1))
        self.register_buffer('cache_positions', torch.arange(cache_capacity, dtype=torch.int32).view(1, -1))
        self.register_buffer('cache_zeros', torch.zeros(cache_capacity, dtype=torch.int32))
        self.register_buffer('mask_tokens', torch.full((1, max_draft_tokens), c.mask_token_id, dtype=torch.int32))
        self.register_buffer('query_repeat', torch.zeros(max_draft_tokens + 1, dtype=torch.int32))
        self.register_buffer('head_repeat', torch.zeros(c.num_attention_heads, dtype=torch.int32))
        self.register_buffer('kv_heads', torch.arange(c.num_key_value_heads, dtype=torch.int32)
                             .repeat_interleave(c.num_key_value_groups))
        if self.phase_table:
            # Include every computed row, even padded context at a full cache.
            # The existing control contract bounds start/end by capacity.
            phase_rows = cache_capacity + max(context_rows, max_draft_tokens + 1)
            positions = torch.arange(phase_rows, dtype=torch.int32,
                                     device=draft.rotary.inv_freq.device)
            phases = positions.float().unsqueeze(-1) * draft.rotary.inv_freq.view(1, -1)
            self.register_buffer('rope_phases', phases, persistent=False)
        if attention_kernel is None:
            # A literal buffer avoids a Shape(scores)->Fill dependency on
            # un-inferred MatMul dimensions in the source TorchAir graph.
            score_heads = c.num_key_value_heads if self.packed else c.num_attention_heads
            score_rows = (max_draft_tokens + 1) * (c.num_key_value_groups if self.packed else 1)
            self.register_buffer('attention_masked_scores', torch.full(
                (1, score_heads, score_rows, cache_capacity + max_draft_tokens + 1),
                -float('inf'), dtype=torch.float32))
        witness = torch.zeros((1, 1, 1, cache_capacity + max_draft_tokens + 1), dtype=torch.bool)
        witness[..., cache_capacity] = True
        self.register_buffer('anchor_witness', witness)

    @staticmethod
    def _pack(owner, first, second, destination):
        a, b = getattr(owner, first).weight, getattr(owner, second).weight
        linear = nn.Linear(a.shape[1], a.shape[0] + b.shape[0], bias=False,
                           device='meta', dtype=a.dtype)
        linear.weight = nn.Parameter(torch.cat((a, b), dim=0), requires_grad=False)
        setattr(owner, destination, linear)
        delattr(owner, first)
        delattr(owner, second)

    def norm(self, x, norm):
        # Request FP32 normalization through the same RMS API as Target CPro.
        # Draft weights are gamma; never apply the Target's 1+w convention.
        return self.rms(x.float(), norm.weight.float(), norm.eps)[0].to(x.dtype)

    def rope(self, x, positions):
        cosine, sine = self.draft.rotary(positions, x.dtype)
        return self.apply_rope(x, cosine, sine)

    def control_rope(self, positions, dtype):
        if not self.integer_controls:
            return self.draft.rotary(positions, dtype)
        if self.phase_table:
            # Replace only position*inv_freq, retaining source FP32 phases and
            # the existing trigonometry. Compiled placement/precision still
            # require inspection and device comparison.
            frequencies = F.embedding(positions.int(), self.rope_phases)
        else:
            # This expression still compiled with two CPU frequency Muls and
            # floating NPU inputs; retained as an explicit comparison policy.
            frequencies = positions.unsqueeze(-1).float() * self.draft.rotary.inv_freq.view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)

    @staticmethod
    def apply_rope(x, cosine, sine):
        return x * cosine.unsqueeze(1) + _rotate_half(x) * sine.unsqueeze(1)

    def project_features(self, features):
        c, rows = self.config, self.context_rows
        channels = self.draft.fc.in_features // 16
        weight = self.draft.fc.weight.reshape(c.hidden_size, channels, 4, 4)
        if not self.spatial_projection:
            return F.conv2d(features.reshape(rows, channels, 4, 4), weight).reshape(1, rows, c.hidden_size)
        # Place each token in a distinct 4x4 patch. A plain reshape would
        # interleave channels/tokens and change the projection.
        grid = math.isqrt(rows)
        if self.rank4_spatial:
            # [i,j,c,h,w] -> [c,i,j,h,w] -> [c,i,h,j,w], expressed
            # with rank-4 permutations. The previous rank-5 Permute was
            # compiled on CPU with FP32 interfaces on the selected DDK.
            tiles = features.reshape(rows, channels, 4, 4).permute(1, 0, 2, 3)
            packed = tiles.reshape(channels * grid, grid, 4, 4).permute(0, 2, 1, 3)
            packed = packed.reshape(1, channels, grid * 4, grid * 4)
        else:
            tiles = features.reshape(grid, grid, channels, 4, 4)
            packed = tiles.permute(2, 0, 3, 1, 4).reshape(1, channels, grid * 4, grid * 4)
        projected = F.conv2d(packed, weight, stride=4)
        return projected.permute(0, 2, 3, 1).reshape(1, rows, c.hidden_size)

    def expand_visibility(self, visible):
        visible = torch.index_select(visible.int(), 1, self.head_repeat).bool()
        if self.packed:
            c, b = self.config, self.k + 1
            visible = visible.reshape(1, c.num_key_value_heads, c.num_key_value_groups * b, self.capacity + b)
        return visible

    def forward(self, target_features, start_position, valid_rows, anchor, proposal_count,
                past_k_0, past_v_0, past_k_1, past_v_1, past_k_2, past_v_2,
                past_k_3, past_v_3, past_k_4, past_v_4, past_k_5, past_v_5):
        c, rows, b = self.config, self.context_rows, self.k + 1
        end = start_position + valid_rows
        context_positions = start_position + self.context_offsets
        block_positions = end + self.block_offsets
        positions = torch.cat((self.cache_positions, block_positions), dim=1)
        cache_valid, block_valid = self.cache_positions < end, self.block_offsets <= proposal_count
        # This BOOL validity concat matches the NPU output consumed by the
        # intermediate CPU mask graph in the observed partition boundaries.
        # Concatenate exact 0/1 INT32 values, keeping the existing integer
        # Gather expansion. No boolean BroadcastTo or arithmetic mask change.
        if self.integer_controls:
            cache_valid, block_valid = cache_valid.int(), block_valid.int()
        key_valid = torch.cat((cache_valid, block_valid), dim=1)
        # Keep the GE workaround for failed BOOL query/head BroadcastTo:
        # integer Gather copies singleton axes exactly, before casting to BOOL.
        key_valid = torch.index_select(key_valid.reshape(1, 1, 1, self.capacity + b).int(),
                                       2, self.query_repeat).bool()
        witness = (self.block_offsets > proposal_count).reshape(1, 1, b, 1) & self.anchor_witness
        local = self.cache_positions.reshape(-1) - start_position
        # Match the GE integer Select expression, without tensor->Python scalars.
        local = torch.where(local < 0, self.cache_zeros, local)
        local = torch.where(local < rows, local, torch.full_like(local, rows - 1))
        # X90 GatherV2D rejects INT64 indices. These bounded row indices and
        # token IDs already fit the INT32 control ABI; do not promote to long.
        local = local.int()
        write = ((self.cache_positions >= start_position) & (self.cache_positions < end)).reshape(1, 1, self.capacity, 1)
        write = write.int().expand(1, c.num_key_value_heads, self.capacity, c.head_dim).bool()
        # This optional ABI uses the fourth argument for byte-preserved rows
        # [anchor, MASK, ...], not an anchor ID. The complete table stays out
        # of the compiled graph; the baseline token-ID ABI remains available.
        if self.host_embedding:
            hidden = anchor
        else:
            ids = torch.cat((anchor.reshape(1, 1), self.mask_tokens), dim=1)
            hidden = self.embedding(ids.int()) * c.input_embedding_scale
        context = self.norm(self.project_features(target_features), self.draft.hidden_norm)
        if self.packed:
            context_rope = self.control_rope(context_positions, context.dtype)
            block_rope = self.control_rope(block_positions, hidden.dtype)
        if self.shared_masks:
            # Layer visibility depends only on controls and the attention
            # type. Preserve causal/window rules and the inactive-query
            # anchor witness, while expanding each distinct mask once.
            full_visible = self.expand_visibility(key_valid | witness)
            distance = block_positions.reshape(1, 1, b, 1) - positions.reshape(1, 1, 1, -1)
            sliding_visible = key_valid & (distance >= 0)
            if c.use_sliding_window:
                sliding_visible = sliding_visible & (distance < c.sliding_window)
            sliding_visible = self.expand_visibility(sliding_visible | witness)
        past = ((past_k_0, past_v_0), (past_k_1, past_v_1), (past_k_2, past_v_2),
                (past_k_3, past_v_3), (past_k_4, past_v_4), (past_k_5, past_v_5))
        updated = []
        for index, layer in enumerate(self.draft.layers):
            attn = layer.self_attn
            x = self.norm(hidden, layer.input_layernorm)
            query = attn.q_proj(x).reshape(1, b, c.num_attention_heads, c.head_dim)
            query = self.norm(query, attn.q_norm).transpose(1, 2)
            query = self.apply_rope(query, *block_rope) if self.packed else self.rope(query, block_positions)
            if self.packed:
                kv = attn.kv_proj(torch.cat((context, x), dim=1))
                keys, values = kv.split(c.key_value_width, dim=-1)
                new_k, block_k = keys.split((rows, b), dim=1)
                new_v, block_v = values.split((rows, b), dim=1)
            else:
                new_k, new_v = attn.k_proj(context), attn.v_proj(context)
                block_k, block_v = attn.k_proj(x), attn.v_proj(x)
            new_k = self.norm(new_k.reshape(1, rows, c.num_key_value_heads, c.head_dim), attn.k_norm).transpose(1, 2)
            new_k = self.apply_rope(new_k, *context_rope) if self.packed else self.rope(new_k, context_positions)
            new_v = new_v.reshape(1, rows, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            present_k = torch.where(write, torch.index_select(new_k, 2, local), past[index][0])
            present_v = torch.where(write, torch.index_select(new_v, 2, local), past[index][1])
            updated.extend((present_k, present_v))
            block_k = self.norm(block_k.reshape(1, b, c.num_key_value_heads, c.head_dim), attn.k_norm).transpose(1, 2)
            block_k = self.apply_rope(block_k, *block_rope) if self.packed else self.rope(block_k, block_positions)
            block_v = block_v.reshape(1, b, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            key = torch.cat((present_k, block_k), dim=2)
            value = torch.cat((present_v, block_v), dim=2)
            if self.shared_masks:
                visible = sliding_visible if c.layer_types[index] == 'sliding_attention' else full_visible
            else:
                visible = key_valid
                if c.layer_types[index] == 'sliding_attention':
                    distance = block_positions.reshape(1, 1, b, 1) - positions.reshape(1, 1, 1, -1)
                    visible = visible & (distance >= 0)
                    if c.use_sliding_window:
                        visible = visible & (distance < c.sliding_window)
                visible = visible | witness
            if self.target_attention is not None:
                mixed = self.target_attention(query, key, value, visible)
            else:
                if self.packed:
                    query = query.reshape(1, c.num_key_value_heads, c.num_key_value_groups * b, c.head_dim)
                else:
                    key = torch.index_select(key, 1, self.kv_heads)
                    value = torch.index_select(value, 1, self.kv_heads)
                if not self.shared_masks:
                    visible = self.expand_visibility(visible)
                scores = torch.matmul(query, key.transpose(-1, -2)).float() * (c.head_dim ** -0.5)
                probabilities = torch.softmax(torch.where(visible, scores, self.attention_masked_scores), dim=-1).half()
                mixed = torch.matmul(probabilities, value)
                mixed = mixed.reshape(1, c.num_attention_heads, b, c.head_dim)
            mixed = mixed.transpose(1, 2).reshape(1, b, c.query_width)
            hidden = (hidden.float() + attn.o_proj(mixed).float()).half()
            mlp_input = self.norm(hidden, layer.post_attention_layernorm)
            if self.packed:
                gate, up = layer.mlp.gate_up_proj(mlp_input).chunk(2, dim=-1)
                mlp = layer.mlp.down_proj(F.silu(gate) * up)
            else:
                mlp = layer.mlp(mlp_input)
            hidden = (hidden.float() + mlp.float()).half()
        hidden = self.norm(hidden, self.draft.norm)[:, 1:, :]
        logits = self.head(hidden)
        top1 = (logits.argmax(dim=-1).to(torch.int32) if self.top1_kernel is None
                else self.top1_kernel(logits))
        return (top1, hidden, *updated)

    def attention_contract(self):
        if self.target_attention is not None:
            return self.target_attention.contract()
        c, b = self.config, self.k + 1
        return dict(policy='composed-attention-v1', layers=c.num_hidden_layers,
            num_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads, head_dim=c.head_dim,
            query_shape=[1, c.num_attention_heads, b, c.head_dim],
            kv_shape=[1, c.num_key_value_heads, self.capacity + b, c.head_dim],
            score_shape=([1, c.num_key_value_heads, c.num_key_value_groups * b, self.capacity + b]
                         if self.packed else [1, c.num_attention_heads, b, self.capacity + b]),
            scale_value=c.head_dim ** -0.5, logical_kv_rows=self.capacity + b,
            operations=('GROUPED_QUERY_RESHAPE; QK_MATMUL; SCALE; SELECT_MASK; SOFTMAX; PV_MATMUL'
                        if self.packed else 'INT32_GQA_GATHER; QK_MATMUL; SCALE; SELECT_MASK; SOFTMAX; PV_MATMUL'),
            mask='DRAFT_VISIBILITY_WITH_INACTIVE_QUERY_ANCHOR_WITNESS',
            cache='PUBLIC_ND_COMMITTED_CONTEXT_ONLY; CANDIDATE_KV_TEMPORARY',
            precision='FP16_QK_RESULT; FP32_SCALE_MASK_SOFTMAX; FP16_PROBABILITIES_AND_PV; DEVICE_INTERNAL_PRECISION_UNVERIFIED')

    def abi(self):
        contract = incremental_draft_abi(self.config, self.context_rows, self.capacity, max_draft_tokens=self.k)
        if self.host_embedding:
            contract['graph']['inputs'][3] = dict(name='block_embeddings', dtype='float16',
                                                 shape=[1, self.k + 1, self.config.hidden_size])
        return {direction: contract['graph'][direction] for direction in ('inputs', 'outputs')}

    def example_inputs(self):
        # Controls are tensor inputs, including zero-row catchup/proposal modes.
        abi = self.abi()
        values = [torch.zeros(row['shape'], dtype=getattr(torch, row['dtype'])) for row in abi['inputs']]
        values[2].fill_(min(2, self.context_rows))
        values[3].fill_(1)
        values[4].fill_(self.k)
        return tuple(values)


def load_prepared(prepared, *, context_rows=16, cache_capacity=192, max_draft_tokens=15,
                  rms=reference_rms, attention_kernel=None, optimization='baseline', top1_kernel=None):
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
                            cache_capacity=cache_capacity, max_draft_tokens=max_draft_tokens,
                            rms=rms, attention_kernel=attention_kernel, optimization=optimization,
                            top1_kernel=top1_kernel).eval()

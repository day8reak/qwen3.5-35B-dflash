"""Six-layer 35B DFlash context-KV graph, using the original draft primitives.

Only newly committed target features before the latest anchor are appended.
Persistent keys already carry absolute RoPE; temporary anchor/MASK KV never
enter the caches. This is a host construction IR, not selected-DDK support,
compiled ABI, device execution, accuracy or acceleration evidence.
"""
from __future__ import annotations

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.graph.draft import _DraftBuilder, _fp32


_INT32_MAX = (1 << 31) - 1
_LAYERS = ('sliding_attention',) * 5 + ('full_attention',)
MAX_DRAFT_TOKEN_CHOICES = (15, 7, 3, 1)


def _validate(config, context_rows, cache_capacity, include_hidden, max_draft_tokens=15):
    if not isinstance(config, Qwen35DFlashConfig):
        raise TypeError('an explicit Qwen35DFlashConfig is required')
    config.validate()
    if (config.num_hidden_layers != 6 or len(config.target_layer_ids) != 8
            or config.block_size != 16 or tuple(config.layer_types) != _LAYERS):
        raise ValueError('incremental35B draft requires six layers, eight target taps, block16 and the original five-sliding/one-full topology')
    if type(context_rows) is not int or context_rows not in {16, 64}:
        raise ValueError('context_rows must select one static C16 or C64 gear')
    if type(max_draft_tokens) is not int or max_draft_tokens not in MAX_DRAFT_TOKEN_CHOICES:
        raise ValueError('max_draft_tokens must select one static 15, 7, 3 or 1 candidate gear')
    if type(cache_capacity) is not int or cache_capacity < context_rows:
        raise ValueError('cache_capacity must be an explicit integer covering one complete context gear')
    if cache_capacity + max(context_rows, max_draft_tokens + 1) > min(_INT32_MAX, config.max_position_embeddings):
        raise ValueError('cache/context/block positions exceed the declared RoPE or INT32 range')
    if config.head_dim % 2:
        raise ValueError('standard DFlash RoPE requires an even head dimension')
    if type(include_hidden) is not bool:
        raise ValueError('include_hidden must be an explicit boolean')


def validate_incremental_draft_controls(config, *, start_position, valid_rows, anchor, proposal_count,
                                      context_rows=16, cache_capacity=192, max_draft_tokens=15):
    """Validate plain host integers before constructing the INT32 wire inputs.

    The caller additionally owns cache lineage/finite initialization and must
    supply all committed feature rows in order, excluding the latest anchor.
    """
    _validate(config, context_rows, cache_capacity, True, max_draft_tokens)
    values = (start_position, valid_rows, anchor, proposal_count)
    if any(type(value) is not int or not 0 <= value <= _INT32_MAX for value in values):
        raise ValueError('incremental controls must be nonnegative explicit INT32-range integers')
    if (valid_rows > context_rows or start_position + valid_rows > cache_capacity
            or start_position > cache_capacity or anchor >= config.vocab_size
            or proposal_count > max_draft_tokens):
        raise ValueError('incremental controls exceed context/cache/token/proposal bounds')
    if (start_position + context_rows > config.max_position_embeddings
            or start_position + valid_rows + max_draft_tokens + 1 > config.max_position_embeddings
            or start_position + max(context_rows, valid_rows + max_draft_tokens + 1) > _INT32_MAX):
        raise ValueError('absolute computed context/block RoPE positions overflow their declared range')
    return {'start_position': start_position, 'valid_rows': valid_rows, 'anchor': anchor,
        'proposal_count': proposal_count, 'context_end': start_position + valid_rows}


class _IncrementalBuilder(_DraftBuilder):
    def __init__(self, config, context_rows, cache_capacity, max_draft_tokens=15):
        super().__init__(config, context_rows)
        self.capacity = cache_capacity
        self.b = max_draft_tokens + 1
        self.k = cache_capacity + self.b

    def _rope(self, positions, rows, name):
        c, g = self.c, self.g
        theta = _fp32(c.rope_theta)
        inverse = [_fp32(1.0 / _fp32(theta ** _fp32(index / c.head_dim)))
                   for index in range(0, c.head_dim, 2)]
        inverse = g.const([[inverse]], dtype='float32')
        promoted = g.op('cast', positions, shape=(1, rows), dtype='float32', attrs={'dtype': 'float32'})
        promoted = self.reshape(promoted, (1, rows, 1))
        frequency = g.op('mul', promoted, inverse, shape=(1, rows, c.head_dim // 2),
            dtype='float32', name=name + '_frequency')
        frequency = g.op('concat', frequency, frequency, shape=(1, rows, c.head_dim), dtype='float32', attrs={'axis': -1})
        cosine = g.op('cos', frequency, shape=frequency.shape, dtype='float32')
        sine = g.op('sin', frequency, shape=frequency.shape, dtype='float32')
        cosine = g.op('cast', cosine, shape=frequency.shape, dtype='float16', attrs={'dtype': 'float16'}, name=name + '_cosine')
        sine = g.op('cast', sine, shape=frequency.shape, dtype='float16', attrs={'dtype': 'float16'}, name=name + '_sine')
        return self.reshape(cosine, (1, 1, rows, c.head_dim)), self.reshape(sine, (1, 1, rows, c.head_dim))

    def _cache_update(self, old, new, gather_indices, write_mask, name):
        shape = (1, self.c.num_key_value_heads, self.capacity, self.c.head_dim)
        selected = self.g.op('gather', new, gather_indices, shape=shape, attrs={'axis': 2})
        return self.g.op('where', write_mask, selected, old, shape=shape, name=name)

    def _attention(self, x, context, context_rope, block_rope, positions, key_valid, inactive_witness,
                   gather_indices, write_mask, past_k, past_v, layer):
        c, g, b, total = self.c, self.g, self.b, self.k
        prefix = f'draft.layers.{layer}.self_attn'
        query = self.linear(x, prefix + '.q_proj.weight', c.query_width)
        query = self.reshape(query, (1, b, c.num_attention_heads, c.head_dim))
        query = self.norm(query, prefix + '.q_norm.weight')
        query = self.transpose(query, (0, 2, 1, 3))
        query = self.rotary(query, *block_rope)

        new_key = self.linear(context, prefix + '.k_proj.weight', c.key_value_width)
        new_key = self.reshape(new_key, (1, self.s, c.num_key_value_heads, c.head_dim))
        new_key = self.norm(new_key, prefix + '.k_norm.weight')
        new_key = self.transpose(new_key, (0, 2, 1, 3))
        new_key = self.rotary(new_key, *context_rope)
        new_value = self.linear(context, prefix + '.v_proj.weight', c.key_value_width)
        new_value = self.reshape(new_value, (1, self.s, c.num_key_value_heads, c.head_dim))
        new_value = self.transpose(new_value, (0, 2, 1, 3))
        present_k = self._cache_update(past_k, new_key, gather_indices, write_mask, f'present_k_{layer}')
        present_v = self._cache_update(past_v, new_value, gather_indices, write_mask, f'present_v_{layer}')

        block_key = self.linear(x, prefix + '.k_proj.weight', c.key_value_width)
        block_key = self.reshape(block_key, (1, b, c.num_key_value_heads, c.head_dim))
        block_key = self.norm(block_key, prefix + '.k_norm.weight')
        block_key = self.transpose(block_key, (0, 2, 1, 3))
        block_key = self.rotary(block_key, *block_rope)
        block_value = self.linear(x, prefix + '.v_proj.weight', c.key_value_width)
        block_value = self.reshape(block_value, (1, b, c.num_key_value_heads, c.head_dim))
        block_value = self.transpose(block_value, (0, 2, 1, 3))
        cache_shape = (1, c.num_key_value_heads, total, c.head_dim)
        key = g.op('concat', present_k, block_key, shape=cache_shape, attrs={'axis': 2})
        value = g.op('concat', present_v, block_value, shape=cache_shape, attrs={'axis': 2})
        key, value = self.repeat_kv(key), self.repeat_kv(value)
        visible = g.op('broadcast_to', key_valid, shape=(1, 1, b, total), dtype='bool', attrs={'shape': [1, 1, b, total]})
        if c.layer_types[layer] == 'sliding_attention':
            query_positions = self.slice(positions, (0, self.capacity), (1, b))
            query_positions = self.reshape(query_positions, (1, 1, b, 1))
            key_positions = self.reshape(positions, (1, 1, 1, total))
            causal = g.op('le', key_positions, query_positions, shape=(1, 1, b, total), dtype='bool')
            visible = g.op('logical_and', visible, causal, shape=(1, 1, b, total), dtype='bool')
            if c.use_sliding_window:
                distance = g.op('sub', query_positions, key_positions, shape=(1, 1, b, total), dtype='int32')
                local = g.op('lt', distance, g.const(c.sliding_window, dtype='int32'), shape=(1, 1, b, total), dtype='bool')
                visible = g.op('logical_and', visible, local, shape=(1, 1, b, total), dtype='bool')
        # Ignored query rows can otherwise be entirely masked with short K and
        # a small sliding window. Their NaN softmax would contaminate the next
        # layer's masked V through 0*NaN. Keep an anchor witness only for those
        # inactive queries; every active query mask and inactive key stays as-is.
        visible = g.op('logical_or', visible, inactive_witness, shape=(1, 1, b, total), dtype='bool')
        visible = self.reshape(visible, (1, 1, b, total), name=f'layer_{layer}_visible')
        scores = g.op('matmul', query, key, shape=(1, c.num_attention_heads, b, total),
            attrs={'transpose_b': True}, name=f'layer_{layer}_qk_fp16')
        scores = g.op('cast', scores, shape=scores.shape, dtype='float32', attrs={'dtype': 'float32'})
        scores = g.op('scalar_mul', scores, shape=scores.shape, dtype='float32', attrs={'value': c.head_dim ** -0.5})
        scores = g.op('where', visible, scores, g.const(float('-inf'), dtype='float32'), shape=scores.shape, dtype='float32')
        probability = g.op('softmax', scores, shape=scores.shape, dtype='float32', attrs={'axis': -1}, name=f'layer_{layer}_probability_fp32')
        probability = g.op('cast', probability, shape=probability.shape, dtype='float16', attrs={'dtype': 'float16'})
        mixed = g.op('matmul', probability, value, shape=(1, c.num_attention_heads, b, c.head_dim), name=f'layer_{layer}_pv_fp16')
        mixed = g.op('cast', mixed, shape=mixed.shape, dtype='float32', attrs={'dtype': 'float32'})
        mixed = g.op('cast', mixed, shape=mixed.shape, dtype='float16', attrs={'dtype': 'float16'})
        mixed = self.transpose(mixed, (0, 2, 1, 3))
        mixed = self.reshape(mixed, (1, b, c.query_width))
        return self.linear(mixed, prefix + '.o_proj.weight', c.hidden_size), present_k, present_v

    def build(self, include_hidden):
        c, g, rows, capacity, block = self.c, self.g, self.s, self.capacity, self.b
        features = g.input('target_features', (1, rows, c.feature_size), 'float16')
        start = g.input('start_position', (1,), 'int32')
        valid = g.input('valid_rows', (1,), 'int32')
        anchor = g.input('anchor', (1,), 'int32')
        proposal_count = g.input('proposal_count', (1,), 'int32')
        cache_shape = (1, c.num_key_value_heads, capacity, c.head_dim)
        past = [(g.input(f'past_k_{layer}', cache_shape, 'float16'), g.input(f'past_v_{layer}', cache_shape, 'float16'))
                for layer in range(c.num_hidden_layers)]
        end = g.op('add', start, valid, shape=(1,), dtype='int32', name='context_end')
        context_positions = g.op('add', start, g.const([list(range(rows))], dtype='int32'),
            shape=(1, rows), dtype='int32', name='context_positions')
        offsets = g.const([list(range(block))], dtype='int32')
        block_positions = g.op('add', end, offsets, shape=(1, block), dtype='int32', name='block_positions')
        cache_positions = g.const([list(range(capacity))], dtype='int32', name='cache_positions')
        positions = g.op('concat', cache_positions, block_positions, shape=(1, capacity + block),
            dtype='int32', attrs={'axis': 1}, name='positions')
        cache_valid = g.op('lt', cache_positions, end, shape=(1, capacity), dtype='bool')
        block_valid = g.op('le', offsets, proposal_count, shape=(1, block), dtype='bool', name='block_valid')
        inactive_query = g.op('lt', proposal_count, offsets, shape=(1, block), dtype='bool')
        inactive_query = self.reshape(inactive_query, (1, 1, block, 1))
        anchor_key = g.const([[[[False] * capacity + [True] + [False] * (block - 1)]]], dtype='bool')
        inactive_witness = g.op('logical_and', inactive_query, anchor_key,
            shape=(1, 1, block, capacity + block), dtype='bool', name='inactive_query_anchor_witness')
        key_valid = g.op('concat', cache_valid, block_valid, shape=(1, capacity + block), dtype='bool', attrs={'axis': 1})
        key_valid = self.reshape(key_valid, (1, 1, 1, capacity + block))
        absolute_rows = g.const(list(range(capacity)), dtype='int32')
        local = g.op('sub', absolute_rows, start, shape=(capacity,), dtype='int32')
        zero = g.const(0, dtype='int32')
        negative = g.op('lt', local, zero, shape=(capacity,), dtype='bool')
        safe = g.op('where', negative, zero, local, shape=(capacity,), dtype='int32')
        below_upper = g.op('lt', safe, g.const(rows, dtype='int32'), shape=(capacity,), dtype='bool')
        safe = g.op('where', below_upper, safe, g.const(rows - 1, dtype='int32'),
            shape=(capacity,), dtype='int32', name='cache_gather_indices')
        after_start = g.op('le', start, absolute_rows, shape=(capacity,), dtype='bool')
        before_end = g.op('lt', absolute_rows, end, shape=(capacity,), dtype='bool')
        write = g.op('logical_and', after_start, before_end, shape=(capacity,), dtype='bool')
        write = self.reshape(write, (1, 1, capacity, 1), name='cache_write_mask')
        block_ids = g.op('concat', self.reshape(anchor, (1, 1)),
            g.const([[c.mask_token_id] * (block - 1)], dtype='int32'), shape=(1, block),
            dtype='int32', attrs={'axis': 1}, name='block_ids')
        embedding = g.weight('target.embed_tokens.weight', (c.vocab_size, c.hidden_size))
        hidden = g.op('gather', embedding, block_ids, shape=(1, block, c.hidden_size), attrs={'axis': 0})
        hidden = g.op('scalar_mul', hidden, shape=hidden.shape, attrs={'value': c.input_embedding_scale})
        context = self.linear(features, 'draft.fc.weight', c.hidden_size)
        context = self.norm(context, 'draft.hidden_norm.weight', name='projected_context')
        context_rope, block_rope = self._rope(context_positions, rows, 'context_rope'), self._rope(block_positions, block, 'block_rope')
        updated = []
        for layer, (past_k, past_v) in enumerate(past):
            prefix = f'draft.layers.{layer}'
            normalized = self.norm(hidden, prefix + '.input_layernorm.weight')
            attention, present_k, present_v = self._attention(normalized, context, context_rope, block_rope,
                positions, key_valid, inactive_witness, safe, write, past_k, past_v, layer)
            updated.extend((present_k, present_v))
            hidden = g.op('add', hidden, attention, shape=hidden.shape)
            normalized = self.norm(hidden, prefix + '.post_attention_layernorm.weight')
            gate = self.linear(normalized, prefix + '.mlp.gate_proj.weight', c.intermediate_size)
            up = self.linear(normalized, prefix + '.mlp.up_proj.weight', c.intermediate_size)
            silu = g.op('silu', gate, shape=gate.shape)
            product = g.op('mul', silu, up, shape=up.shape)
            mlp = self.linear(product, prefix + '.mlp.down_proj.weight', c.hidden_size)
            hidden = g.op('add', hidden, mlp, shape=hidden.shape)
        hidden = self.norm(hidden, 'draft.norm.weight')
        hidden = self.slice(hidden, (0, 1, 0), (1, block - 1, c.hidden_size), name='draft_hidden')
        logits = self.linear(hidden, 'target.lm_head.weight', c.vocab_size, name='draft_logits')
        top1 = g.op('argmax', logits, shape=(1, block - 1), dtype='int32', attrs={'axis': -1, 'keepdims': False}, name='draft_top1')
        g.outputs = [top1, *([hidden] if include_hidden else []), *updated]
        return g.validate()


def build_incremental_draft(config, context_rows=16, cache_capacity=192, include_hidden=True, *, max_draft_tokens=15):
    """Construct one static C16/C64 cached context-delta graph, with 12 KV IOs."""
    _validate(config, context_rows, cache_capacity, include_hidden, max_draft_tokens)
    return _IncrementalBuilder(config, context_rows, cache_capacity, max_draft_tokens).build(include_hidden)


def incremental_draft_abi(config, context_rows=16, cache_capacity=192, include_hidden=True, *, max_draft_tokens=15):
    """Describe the host graph contract; actual compiled/native IO is pending."""
    graph = build_incremental_draft(config, context_rows, cache_capacity, include_hidden, max_draft_tokens=max_draft_tokens)
    metadata = graph.manifest()
    return {'schema_version': 2, 'artifact_kind': 'qwen35-incremental-draft-host-graph-abi',
        'variant': 'SIX_LAYER_CONTEXT_KV_CACHED', 'context_rows': context_rows, 'cache_capacity': cache_capacity,
        'block_rows': max_draft_tokens + 1, 'max_draft_tokens': max_draft_tokens,
        'trained_block_rows': config.block_size, 'proposal_count_range': [0, max_draft_tokens], 'graph': metadata,
        'candidate_gear_policy': 'STATIC_GRAPH_COMPUTE_WIDTH_NOT_OUTPUT_TRUNCATION',
        'verify_limit': 'SEPARATE_HOST_POLICY_NO_LARGER_THAN_MAX_DRAFT_TOKENS',
        'control_dtype': 'INT32', 'control_shape': [1], 'context_features': 'NEW_COMMITTED_ROWS_BEFORE_LATEST_ANCHOR_ONLY',
        'cache_layout': [1, config.num_key_value_heads, cache_capacity, config.head_dim],
        'cache_initialization': 'ZERO_INITIALIZE_ALL12_BUFFERS; VALID_CACHE_VALUES_MUST_BE_FINITE',
        'cache_write_interval': '[start_position,start_position+valid_rows)',
        'cache_preservation': 'WHERE_SELECTS_OLD_VALUE_EVERYWHERE_OUTSIDE_WRITE_INTERVAL',
        'positions': {'context': 'start_position+row_offset', 'block': 'start_position+valid_rows+row_offset',
            'historical_keys': 'ALREADY_ROTATED_NOT_REROTATED'},
        'temporary_block_kv': 'NEVER_PERSISTED', 'active_outputs': 'FIRST_PROPOSAL_COUNT; REMAINING_TOP1/HIDDEN_ROWS_NOT_COMMITTED',
        'full_layer_mask': 'PROPOSAL_COUNT_TAIL_MASKED_TOO; ACTIVE_BLOCK_KEYS_BIDIRECTIONAL',
        'inactive_queries': 'ANCHOR_WITNESS_KEEPS_IGNORED_ROWS_FINITE; ACTIVE_QUERY_MASKS_UNCHANGED',
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN', 'numerical_validation': 'NOT_RUN',
        'selected_ddk_primitive_support': 'UNVERIFIED_GATHER_WHERE_AND_ATTENTION_PRIMITIVES_ONLY_NO_NEW_CUSTOM_OPERATOR'}

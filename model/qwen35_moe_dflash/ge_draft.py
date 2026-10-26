"""Direct graph construction for the FP16 DFlash draft.

This module has no PyTorch, TorchAir or torch_npu dependency.  Its static shapes
and input/output order match ``graphs.DraftGraph`` after the existing FP16
lowering.  The graph is a cache-free recomputation baseline; the right-padded
prefix length remains a runtime input, not a tracing example baked into AIR.

Attention is expressed as primitive matrix multiplications and masked softmax.
Its floating point rounding can differ from fused PyTorch/GE kernels, so
reference and device accuracy remain separate validation gates.
"""

from __future__ import annotations

import struct

import numpy as np

from .config import Qwen35DFlashConfig
from .ge_ir import Graph, Value


def _fp32(value: float) -> float:
    """Round host-generated RoPE constants like the FP32 source buffer."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


class _DraftBuilder:
    def __init__(self, config: Qwen35DFlashConfig, sequence_length: int) -> None:
        self.c = config
        self.s = sequence_length
        self.b = config.block_size
        self.k = sequence_length + config.block_size
        self.g = Graph("draft")

    def reshape(self, x: Value, shape: tuple[int, ...], name: str | None = None) -> Value:
        return self.g.op("reshape", x, shape=shape, dtype=x.dtype,
                         attrs={"shape": list(shape)}, name=name)

    def transpose(self, x: Value, perm: tuple[int, ...]) -> Value:
        return self.g.op("transpose", x, shape=tuple(x.shape[i] for i in perm),
                         dtype=x.dtype, attrs={"perm": list(perm)})

    def slice(self, x: Value, starts: tuple[int, ...], sizes: tuple[int, ...],
              name: str | None = None) -> Value:
        return self.g.op("slice", x, shape=sizes, dtype=x.dtype,
                         attrs={"starts": list(starts), "sizes": list(sizes)}, name=name)

    def linear(self, x: Value, key: str, width: int, name: str | None = None) -> Value:
        weight = self.g.weight(key, (width, x.shape[-1]))
        return self.g.op("matmul", x, weight, shape=(*x.shape[:-1], width),
                         attrs={"transpose_b": True}, name=name)

    def norm(self, x: Value, key: str, name: str | None = None) -> Value:
        g = self.g
        square = g.op("mul", x, x, shape=x.shape)
        mean_shape = (*x.shape[:-1], 1)
        mean = g.op("reduce_mean", square, shape=mean_shape,
                    attrs={"axes": [-1], "keepdims": True})
        mean = g.op("scalar_add", mean, shape=mean_shape, attrs={"value": self.c.rms_norm_eps})
        reciprocal = g.op("rsqrt", mean, shape=mean_shape)
        normalized = g.op("mul", x, reciprocal, shape=x.shape)
        # DFlash stores the multiplier itself; Qwen3.5's (1 + weight) does not
        # apply to the draft checkpoint.
        return g.op("mul", normalized, g.weight(key, (x.shape[-1],)),
                    shape=x.shape, name=name)

    def rotary(self, x: Value, cosine: Value, sine: Value) -> Value:
        half = self.c.head_dim // 2
        shape = (*x.shape[:-1], half)
        first = self.slice(x, (0, 0, 0, 0), shape)
        second = self.slice(x, (0, 0, 0, half), shape)
        negative = self.g.op("neg", second, shape=shape)
        rotated = self.g.op("concat", negative, first, shape=x.shape, attrs={"axis": -1})
        left = self.g.op("mul", x, cosine, shape=x.shape)
        right = self.g.op("mul", rotated, sine, shape=x.shape)
        return self.g.op("add", left, right, shape=x.shape)

    def repeat_kv(self, x: Value) -> Value:
        c, g = self.c, self.g
        if c.num_key_value_groups == 1:
            return x
        # Match target repeat_heads: copy each KV head consecutively using
        # exact INT32 indices, without a rank-five expansion or arithmetic.
        indices = g.const([head for head in range(c.num_key_value_heads)
                           for _ in range(c.num_key_value_groups)], dtype="int32")
        return g.op("gather", x, indices,
                    shape=(1, c.num_attention_heads, self.k, c.head_dim),
                    dtype=x.dtype, attrs={"axis": 1})

    def attention(self, x: Value, context: Value, cosine: Value, sine: Value,
                  positions: Value, key_valid: Value, layer: int) -> Value:
        c, g, b, k = self.c, self.g, self.b, self.k
        prefix = f"draft.layers.{layer}.self_attn"
        query = self.linear(x, f"{prefix}.q_proj.weight", c.query_width)
        query = self.reshape(query, (1, b, c.num_attention_heads, c.head_dim))
        query = self.norm(query, f"{prefix}.q_norm.weight")
        query = self.transpose(query, (0, 2, 1, 3))

        key_context = self.linear(context, f"{prefix}.k_proj.weight", c.key_value_width)
        key_block = self.linear(x, f"{prefix}.k_proj.weight", c.key_value_width)
        value_context = self.linear(context, f"{prefix}.v_proj.weight", c.key_value_width)
        value_block = self.linear(x, f"{prefix}.v_proj.weight", c.key_value_width)
        key = g.op("concat", key_context, key_block, shape=(1, k, c.key_value_width),
                   attrs={"axis": 1})
        value = g.op("concat", value_context, value_block, shape=(1, k, c.key_value_width),
                     attrs={"axis": 1})
        key = self.reshape(key, (1, k, c.num_key_value_heads, c.head_dim))
        key = self.norm(key, f"{prefix}.k_norm.weight")
        key = self.transpose(key, (0, 2, 1, 3))
        value = self.reshape(value, (1, k, c.num_key_value_heads, c.head_dim))
        value = self.transpose(value, (0, 2, 1, 3))

        query_cosine = self.slice(cosine, (0, 0, self.s, 0), (1, 1, b, c.head_dim))
        query_sine = self.slice(sine, (0, 0, self.s, 0), (1, 1, b, c.head_dim))
        query = self.rotary(query, query_cosine, query_sine)
        key = self.rotary(key, cosine, sine)
        key, value = self.repeat_kv(key), self.repeat_kv(value)

        visible = g.op("broadcast_to", key_valid, shape=(1, 1, b, k), dtype="bool",
                       attrs={"shape": [1, 1, b, k]})
        if c.layer_types[layer] == "sliding_attention":
            query_positions = self.slice(positions, (0, self.s), (1, b))
            query_positions = self.reshape(query_positions, (1, 1, b, 1))
            key_positions = self.reshape(positions, (1, 1, 1, k))
            causal = g.op("le", key_positions, query_positions,
                          shape=(1, 1, b, k), dtype="bool")
            visible = g.op("logical_and", visible, causal, shape=(1, 1, b, k), dtype="bool")
            if c.use_sliding_window:
                distance = g.op("sub", query_positions, key_positions,
                                shape=(1, 1, b, k), dtype="int32")
                local = g.op("lt", distance, g.const(c.sliding_window, dtype="int32"),
                             shape=(1, 1, b, k), dtype="bool")
                visible = g.op("logical_and", visible, local,
                               shape=(1, 1, b, k), dtype="bool")
        # The full-attention layer is bidirectional across the complete draft
        # block; both kinds still exclude padded and anchor context rows.
        visible = self.reshape(visible, (1, 1, b, k), name=f"layer_{layer}_visible")
        scores = g.op("matmul", query, key, shape=(1, c.num_attention_heads, b, k),
                      attrs={"transpose_b": True})
        scores = g.op("scalar_mul", scores, shape=scores.shape, attrs={"value": c.head_dim ** -0.5})
        scores = g.op("where", visible, scores, g.const(float("-inf")), shape=scores.shape)
        probability = g.op("softmax", scores, shape=scores.shape, attrs={"axis": -1})
        mixed = g.op("matmul", probability, value,
                     shape=(1, c.num_attention_heads, b, c.head_dim))
        mixed = self.transpose(mixed, (0, 2, 1, 3))
        mixed = self.reshape(mixed, (1, b, c.query_width))
        return self.linear(mixed, f"{prefix}.o_proj.weight", c.hidden_size)

    def build(self) -> Graph:
        c, g, s, b, k = self.c, self.g, self.s, self.b, self.k
        features = g.input("target_features", (1, s, c.feature_size), "float16")
        ids = g.input("input_ids", (1, s), "int32")
        mask = g.input("attention_mask", (1, s), "int32")
        valid = g.op("cast", mask, shape=(1, s), dtype="bool", attrs={"dtype": "bool"})
        valid_int = g.op("cast", valid, shape=(1, s), dtype="int32", attrs={"dtype": "int32"})
        length = g.op("reduce_sum", valid_int, shape=(1, 1), dtype="int32",
                      attrs={"axes": [1], "keepdims": True})
        one = g.const(1, dtype="int32")
        anchor = g.op("sub", length, one, shape=(1, 1), dtype="int32", name="anchor_position")
        # Internal OMG GatherElements accepts floating data only. The HIAI
        # contract is batch1 with a nonempty valid prefix, so anchor is in
        # [0,S-1]. Gather from a 1D INT32 token vector without any float casts;
        # the [1,1] index shape is also the Gather output shape.
        token_vector = self.reshape(ids, (s,), name="input_ids_vector")
        anchor_ids = g.op("gather", token_vector, anchor, shape=(1, 1), dtype="int32", attrs={"axis": 0})
        block_ids = g.op("concat", anchor_ids,
                         g.const([[c.mask_token_id] * (b - 1)], dtype="int32"),
                         shape=(1, b), dtype="int32", attrs={"axis": 1}, name="block_ids")
        rows = g.const([list(range(s))], dtype="int32")
        preceding = g.op("lt", rows, anchor, shape=(1, s), dtype="bool")
        context_valid = g.op("logical_and", valid, preceding, shape=(1, s), dtype="bool", name="context_valid")
        context_int = g.op("cast", context_valid, shape=(1, s), dtype="int32", attrs={"dtype": "int32"})
        context_positions = g.op("cumsum", context_int, shape=(1, s), dtype="int32", attrs={"axis": 1})
        context_positions = g.op("sub", context_positions, one, shape=(1, s), dtype="int32")
        # Internal OMG rejects INT32 Maximum. Here the count is in [0,S-1],
        # so x=count-1 is in [-1,S-2]. Even a CPUCL Less implementation that
        # compares via the signed difference 0-x cannot overflow in this
        # bounded domain. Do not generalize this to arbitrary INT32 extrema.
        zero = g.const(0, dtype="int32")
        negative = g.op("lt", context_positions, zero, shape=(1, s), dtype="bool",
                        name="context_position_negative")
        context_positions = g.op("where", negative, zero, context_positions,
                                 shape=(1, s), dtype="int32")
        block_positions = g.op("add", anchor, g.const([list(range(b))], dtype="int32"), shape=(1, b), dtype="int32")
        positions = g.op("concat", context_positions, block_positions, shape=(1, k), dtype="int32",
                         attrs={"axis": 1}, name="positions")
        key_valid = g.op("concat", context_valid, g.const([[True] * b], dtype="bool"),
                         shape=(1, k), dtype="bool", attrs={"axis": 1})
        key_valid = self.reshape(key_valid, (1, 1, 1, k))

        embedding = g.weight("target.embed_tokens.weight", (c.vocab_size, c.hidden_size))
        hidden = g.op("gather", embedding, block_ids, shape=(1, b, c.hidden_size), attrs={"axis": 0})
        hidden = g.op("scalar_mul", hidden, shape=hidden.shape, attrs={"value": c.input_embedding_scale})
        context = self.linear(features, "draft.fc.weight", c.hidden_size)
        context = self.norm(context, "draft.hidden_norm.weight", name="projected_context")

        # Source rotary.inv_freq is a nonpersistent FP32 buffer, then converted
        # to FP16 by the deployment lowering.  Duplicate the frequency half;
        # this is full-head standard RoPE, not Qwen3.5 target partial/MRoPE.
        theta = _fp32(c.rope_theta)
        inverse = [_fp32(1.0 / _fp32(theta ** _fp32(i / c.head_dim)))
                   for i in range(0, c.head_dim, 2)]
        inverse = g.const([[inverse]])
        float_positions = g.op("cast", positions, shape=(1, k), attrs={"dtype": "float16"})
        float_positions = self.reshape(float_positions, (1, k, 1))
        # The nonempty prefix contract bounds positions to [0,S+B-2]. For
        # finite nonnegative FP16 operands, this outer product is also a K=1
        # BMM, with the same FP16 result and no negative-zero ambiguity. Keep
        # the position cast BEFORE it, including rounding above 2048. This
        # avoids the CPUCL Mul input-type mismatch seen in the compiled draft;
        # it is deliberately not a generic Mul rewrite or a static table.
        inverse_data = g.constants[inverse.name]
        use_bmm = (k - 2 <= 65504 and np.isfinite(inverse_data).all()
                   and not np.signbit(inverse_data).any())
        frequency = g.op("matmul" if use_bmm else "mul", float_positions, inverse,
                         shape=(1, k, c.head_dim // 2))
        frequency = g.op("concat", frequency, frequency, shape=(1, k, c.head_dim), attrs={"axis": -1})
        cosine = g.op("cos", frequency, shape=frequency.shape)
        sine = g.op("sin", frequency, shape=frequency.shape)
        cosine = self.reshape(cosine, (1, 1, k, c.head_dim))
        sine = self.reshape(sine, (1, 1, k, c.head_dim))

        for layer in range(c.num_hidden_layers):
            prefix = f"draft.layers.{layer}"
            normalized = self.norm(hidden, f"{prefix}.input_layernorm.weight")
            attention = self.attention(normalized, context, cosine, sine, positions, key_valid, layer)
            hidden = g.op("add", hidden, attention, shape=hidden.shape)
            normalized = self.norm(hidden, f"{prefix}.post_attention_layernorm.weight")
            gate = self.linear(normalized, f"{prefix}.mlp.gate_proj.weight", c.intermediate_size)
            up = self.linear(normalized, f"{prefix}.mlp.up_proj.weight", c.intermediate_size)
            silu = g.op("silu", gate, shape=gate.shape)
            product = g.op("mul", silu, up, shape=up.shape)
            mlp = self.linear(product, f"{prefix}.mlp.down_proj.weight", c.hidden_size)
            hidden = g.op("add", hidden, mlp, shape=hidden.shape)
        hidden = self.norm(hidden, "draft.norm.weight")
        hidden = self.slice(hidden, (0, 1, 0), (1, b - 1, c.hidden_size), name="draft_hidden")
        logits = self.linear(hidden, "target.lm_head.weight", c.vocab_size, name="draft_logits")
        top1 = g.op("argmax", logits, shape=(1, b - 1), dtype="int32",
                    attrs={"axis": -1, "keepdims": False}, name="draft_top1")
        g.outputs = [top1]
        return g


def build_draft(config: Qwen35DFlashConfig, sequence_length: int) -> Graph:
    """Construct the full drafter graph with weights supplied separately.

    Inputs, in order: FP16 target features ``[1,S,F]``, int32 token IDs
    ``[1,S]``, int32 right-padded 0/1 mask ``[1,S]`` with at least one valid
    token. Output: int32 Top1 ``[1,block_size-1]``. The HIAI runtime validates
    input values before execution, as for the retained PyTorch export path.
    """
    config.validate()
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if sequence_length > (1 << 31) - 1:
        raise ValueError("sequence_length must fit the INT32 position counter")
    if config.block_size < 2:
        raise ValueError("DFlash requires an anchor and at least one draft token")
    if config.head_dim % 2:
        raise ValueError("standard DFlash RoPE requires an even head_dim")
    return _DraftBuilder(config, sequence_length).build()

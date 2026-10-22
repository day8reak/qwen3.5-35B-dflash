"""Direct graph definition of the FP16 Qwen3.5 MoE text target.

This module does not import PyTorch or TorchAir. It deliberately retains the
existing recompute/static-all-expert deployment baseline, including chunk-64
Gated DeltaNet evaluation order. It is not a sparse execution optimization.
Canonical checkpoint names are ``target.layers.*`` etc.; the bundle writer
maps these to the checkpoint's ``model.language_model.*`` namespace.
"""
from __future__ import annotations

import math
import numpy as np

from .ge_ir import Graph


class _Builder:
    def __init__(self, graph):
        self.g = graph

    def constant(self, value, dtype="float16"):
        return self.g.const(np.asarray(value, dtype=dtype), dtype=dtype)

    def op(self, kind, *values, shape=None, dtype="float16", **attrs):
        return self.g.op(kind, *values, shape=tuple(shape or values[0].shape),
                         dtype=dtype, attrs=attrs)

    def view(self, x, shape):
        return self.g.op("reshape", x, shape=tuple(shape), dtype=x.dtype,
                         attrs={"shape": list(shape)})

    def permute(self, x, perm):
        return self.op("transpose", x, shape=[x.shape[p] for p in perm],
                       dtype=x.dtype, perm=list(perm))

    def slice(self, x, axis, start, size):
        axis %= len(x.shape)
        starts = [0] * len(x.shape)
        sizes = list(x.shape)
        starts[axis], sizes[axis] = start, size
        return self.op("slice", x, shape=sizes, dtype=x.dtype, starts=starts, sizes=sizes)

    def cat(self, values, axis):
        values = list(values)
        if len(values) == 1:
            return values[0]
        shape = list(values[0].shape)
        axis %= len(shape)
        shape[axis] = sum(v.shape[axis] for v in values)
        return self.op("concat", *values, shape=shape, dtype=values[0].dtype, axis=axis)

    def reduce(self, kind, x, axis, keep=True):
        axis %= len(x.shape)
        shape = list(x.shape)
        if keep:
            shape[axis] = 1
        else:
            shape.pop(axis)
        return self.op(kind, x, shape=shape, dtype=x.dtype, axes=[axis], keepdims=keep)

    def linear(self, x, key, width, weight=None):
        weight = self.g.weight(key, (width, x.shape[-1])) if weight is None else weight
        return self.op("matmul", x, weight, shape=(*x.shape[:-1], width),
                       transpose_a=False, transpose_b=True)

    def rms(self, x, key, eps, offset=True):
        square = self.op("mul", x, x)
        mean = self.reduce("reduce_mean", square, -1)
        scale = self.op("rsqrt", self.op("scalar_add", mean, value=eps))
        result = self.op("mul", x, scale)
        weight = self.g.weight(key, (x.shape[-1],))
        if offset:
            weight = self.op("add", self.constant(1), weight, shape=weight.shape)
        return self.op("mul", result, weight)

    def l2(self, x):
        norm = self.reduce("reduce_sum", self.op("mul", x, x), -1)
        return self.op("mul", x, self.op("rsqrt", self.op("scalar_add", norm, value=1e-6)))

    def repeat_heads(self, x, axis, factor):
        if factor == 1:
            return x
        ids = self.constant(np.repeat(np.arange(x.shape[axis]), factor), "int32")
        shape = list(x.shape)
        shape[axis] *= factor
        return self.op("gather", x, ids, shape=shape, axis=axis)

    def rotary(self, x, cos, sin):
        width = cos.shape[-1]
        rotated = self.slice(x, -1, 0, width)
        first = self.slice(rotated, -1, 0, width // 2)
        second = self.slice(rotated, -1, width // 2, width // 2)
        turned = self.cat([self.op("neg", second), first], -1)
        result = self.op("add", self.op("mul", rotated, cos), self.op("mul", turned, sin))
        if width < x.shape[-1]:
            result = self.cat([result, self.slice(x, -1, width, x.shape[-1] - width)], -1)
        return result


def _full_attention(b, x, prefix, c, mask, rotary):
    s, h, heads, kv, d = x.shape[1], c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    qgate = b.view(b.linear(x, prefix + ".q_proj.weight", heads*d*2), (1, s, heads, d*2))
    q = b.slice(qgate, -1, 0, d)
    gate = b.view(b.slice(qgate, -1, d, d), (1, s, heads*d))
    q = b.permute(b.rms(q, prefix + ".q_norm.weight", c["rms_norm_eps"]), (0, 2, 1, 3))
    k = b.view(b.linear(x, prefix + ".k_proj.weight", kv*d), (1, s, kv, d))
    k = b.permute(b.rms(k, prefix + ".k_norm.weight", c["rms_norm_eps"]), (0, 2, 1, 3))
    v = b.permute(b.view(b.linear(x, prefix + ".v_proj.weight", kv*d), (1, s, kv, d)), (0, 2, 1, 3))
    q, k = [b.rotary(t, *rotary) for t in (q, k)]
    k = b.repeat_heads(k, 1, heads // kv)
    v = b.repeat_heads(v, 1, heads // kv)
    scores = b.op("matmul", q, k, shape=(1, heads, s, s), transpose_a=False, transpose_b=True)
    scores = b.op("add", b.op("scalar_mul", scores, value=d**-0.5), mask)
    probs = b.op("softmax", scores, axis=-1)
    mixed = b.op("matmul", probs, v, shape=(1, heads, s, d), transpose_a=False, transpose_b=False)
    mixed = b.view(b.permute(mixed, (0, 2, 1, 3)), (1, s, heads*d))
    return b.linear(b.op("mul", mixed, b.op("sigmoid", gate)), prefix + ".o_proj.weight", h)


def _chunk_gdn(b, query, key, value, decay, beta):
    """The installed Transformers torch_chunk_gated_delta_rule, FP16-lowered.

    Keep padding and reduction widths of 64 even for tiny sequences: replacing
    this with a token recurrence changes FP16 accumulation and requires a new
    accuracy audit. No nonzero/gather routing or data-dependent Python loops.
    """
    sequence, heads, kdim, vdim = query.shape[1], query.shape[2], query.shape[3], value.shape[3]
    chunk = 64
    count = (sequence + chunk - 1) // chunk
    padded = count * chunk
    query, key = b.l2(query), b.l2(key)
    tensors = []
    for x in (query, key, value, beta, decay):
        x = b.permute(x, (0, 2, 1, 3) if len(x.shape) == 4 else (0, 2, 1))
        if sequence < padded:
            zeros_shape = list(x.shape)
            zeros_shape[2] = padded - sequence
            x = b.cat([x, b.constant(np.zeros(zeros_shape, dtype=np.float16))], 2)
        tensors.append(x)
    query, key, value, beta, decay = tensors
    query = b.op("scalar_mul", query, value=kdim**-0.5)
    v_beta = b.op("mul", value, b.view(beta, (1, heads, padded, 1)))
    k_beta = b.op("mul", key, b.view(beta, (1, heads, padded, 1)))
    query, key, value, k_beta, v_beta = [b.view(x, (1, heads, count, chunk, x.shape[-1])) for x in (query, key, value, k_beta, v_beta)]
    decay = b.op("cumsum", b.view(decay, (1, heads, count, chunk)), axis=-1)
    lower = b.constant(np.tril(np.ones((chunk, chunk), dtype=bool)), "bool")
    strict_lower = b.constant(np.tril(np.ones((chunk, chunk), dtype=bool), -1), "bool")
    diff = b.op("sub", b.view(decay, (1, heads, count, chunk, 1)),
                b.view(decay, (1, heads, count, 1, chunk)), shape=(1, heads, count, chunk, chunk))
    zero = b.constant(0)
    decay_mask = b.op("where", lower, diff, zero, shape=diff.shape)
    decay_mask = b.op("where", lower, b.op("exp", decay_mask), zero, shape=diff.shape)
    attn = b.op("matmul", k_beta, key, shape=diff.shape, transpose_a=False, transpose_b=True)
    attn = b.op("neg", b.op("where", strict_lower, b.op("mul", attn, decay_mask), zero, shape=diff.shape))
    for i in range(1, chunk):
        row = b.slice(b.slice(attn, -2, i, 1), -1, 0, i)
        sub = b.slice(b.slice(attn, -2, 0, i), -1, 0, i)
        row_flat = b.view(row, (1, heads, count, i))
        product = b.op("mul", b.view(row_flat, (1, heads, count, i, 1)), sub, shape=sub.shape)
        fixed = b.op("add", row_flat, b.reduce("reduce_sum", product, -2, False))
        fixed = b.view(fixed, (1, heads, count, 1, i))
        full_row = b.cat([fixed, b.slice(b.slice(attn, -2, i, 1), -1, i, chunk-i)], -1)
        rows = [b.slice(attn, -2, 0, i), full_row]
        if i + 1 < chunk:
            rows.append(b.slice(attn, -2, i+1, chunk-i-1))
        attn = b.cat(rows, -2)
    attn = b.op("add", attn, b.constant(np.eye(chunk, dtype=np.float16)))
    value = b.op("matmul", attn, v_beta, shape=(1, heads, count, chunk, vdim), transpose_a=False, transpose_b=False)
    exp_decay = b.view(b.op("exp", decay), (1, heads, count, chunk, 1))
    k_cum = b.op("matmul", attn, b.op("mul", k_beta, exp_decay),
                 shape=(1, heads, count, chunk, kdim), transpose_a=False, transpose_b=False)
    state = b.constant(np.zeros((1, heads, kdim, vdim), dtype=np.float16))
    out = []
    for i in range(count):
        take = lambda x: b.view(b.slice(x, 2, i, 1), (1, heads, *x.shape[3:]))
        q_i, k_i, v_i = [take(x) for x in (query, key, value)]
        dec_i = take(decay_mask)
        scores = b.op("matmul", q_i, k_i, shape=(1, heads, chunk, chunk), transpose_a=False, transpose_b=True)
        scores = b.op("mul", scores, dec_i)
        old_value = b.op("matmul", take(k_cum), state, shape=(1, heads, chunk, vdim), transpose_a=False, transpose_b=False)
        new_value = b.op("sub", v_i, old_value)
        g_i = take(decay)
        exp_i = b.view(b.op("exp", g_i), (1, heads, chunk, 1))
        inter = b.op("matmul", b.op("mul", q_i, exp_i), state,
                     shape=(1, heads, chunk, vdim), transpose_a=False, transpose_b=False)
        out.append(b.op("add", inter, b.op("matmul", scores, new_value,
                        shape=(1, heads, chunk, vdim), transpose_a=False, transpose_b=False)))
        if i + 1 < count:
            final = b.slice(g_i, -1, chunk-1, 1)
            factor = b.view(b.op("exp", b.op("sub", final, g_i, shape=g_i.shape)), (1, heads, chunk, 1))
            carry = b.op("mul", state, b.view(b.op("exp", final), (1, heads, 1, 1)))
            update = b.op("matmul", b.op("mul", k_i, factor), new_value,
                          shape=state.shape, transpose_a=True, transpose_b=False)
            state = b.op("add", carry, update)
    result = b.cat(out, 2)
    if padded > sequence:
        result = b.slice(result, 2, 0, sequence)
    return b.permute(result, (0, 2, 1, 3))


def _linear_attention(b, x, prefix, c):
    # The pinned upstream helper masks only batch > 1. This ABI has batch=1;
    # padded GDN rows remain present, matching prepare_fp16_pair exactly.
    s, h = x.shape[1], c["hidden_size"]
    kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd, kernel = c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
    kw, vw = kh*kd, vh*vd
    width = kw*2 + vw
    mixed = b.permute(b.linear(x, prefix + ".in_proj_qkv.weight", width), (0, 2, 1))
    conv = b.g.weight(prefix + ".conv1d.weight", (width, 1, kernel))
    # Conv1d cross-correlation: weight[tap] consumes t-(K-1)+tap.
    padded = b.cat([b.constant(np.zeros((1, width, kernel-1), dtype=np.float16)), mixed], -1)
    windows = b.cat([b.view(b.slice(padded, -1, tap, s), (width, s, 1)) for tap in range(kernel)], -1)
    # One matmul reduction avoids rounding each individual tap product to FP16.
    convolved = b.op("matmul", windows, b.view(conv, (width, kernel, 1)),
                     shape=(width, s, 1), transpose_a=False, transpose_b=False)
    mixed = b.permute(b.op("silu", b.view(convolved, (1, width, s))), (0, 2, 1))
    query = b.view(b.slice(mixed, -1, 0, kw), (1, s, kh, kd))
    key = b.view(b.slice(mixed, -1, kw, kw), (1, s, kh, kd))
    value = b.view(b.slice(mixed, -1, kw*2, vw), (1, s, vh, vd))
    query, key = [b.repeat_heads(t, 2, vh // kh) for t in (query, key)]
    beta = b.op("sigmoid", b.linear(x, prefix + ".in_proj_b.weight", vh))
    a = b.linear(x, prefix + ".in_proj_a.weight", vh)
    dt = b.g.weight(prefix + ".dt_bias", (vh,))
    a = b.op("softplus", b.op("add", a, dt), beta=1.0, threshold=20.0)
    decay = b.op("mul", b.op("neg", b.op("exp", b.g.weight(prefix + ".A_log", (vh,)))), a, shape=a.shape)
    core = _chunk_gdn(b, query, key, value, decay, beta)
    z = b.view(b.linear(x, prefix + ".in_proj_z.weight", vw), (1, s, vh, vd))
    core = b.rms(core, prefix + ".norm.weight", c["rms_norm_eps"], offset=False)
    core = b.op("mul", core, b.op("silu", z))
    return b.linear(b.view(core, (1, s, vw)), prefix + ".out_proj.weight", h)


def _moe(b, x, prefix, c):
    s, h, experts, topk = x.shape[1], c["hidden_size"], c["num_experts"], c["num_experts_per_tok"]
    middle, shared = c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
    flat = b.view(x, (s, h))
    logits = b.linear(flat, prefix + ".gate.weight", experts)
    probs = b.op("softmax", logits, axis=-1)
    values = b.op("topk_values", probs, shape=(s, topk), axis=-1, k=topk)
    indices = b.op("topk_indices", probs, shape=(s, topk), dtype="int32", axis=-1, k=topk)
    weights = b.op("div", values, b.reduce("reduce_sum", values, -1))
    gate_up = b.g.weight(prefix + ".experts.gate_up_proj", (experts, middle*2, h))
    down = b.g.weight(prefix + ".experts.down_proj", (experts, h, middle))
    result = b.constant(np.zeros((s, h), dtype=np.float16))
    for e in range(experts):
        selected = b.op("eq", indices, b.constant(e, "int32"), dtype="bool")
        route = b.op("where", selected, weights, b.constant(0), shape=weights.shape)
        scale = b.reduce("reduce_sum", route, -1)
        gate_weight = b.view(b.slice(gate_up, 0, e, 1), (middle*2, h))
        expert_down = b.view(b.slice(down, 0, e, 1), (h, middle))
        mixed = b.linear(flat, None, middle*2, weight=gate_weight)
        gate, up = b.slice(mixed, -1, 0, middle), b.slice(mixed, -1, middle, middle)
        expert = b.linear(b.op("mul", b.op("silu", gate), up), None, h, weight=expert_down)
        contribution = b.op("mul", expert, scale)
        active = b.op("gt", b.reduce("reduce_sum", b.op("cast", selected, dtype="int32"), -1), b.constant(0, "int32"), dtype="bool")
        contribution = b.op("where", active, contribution, b.constant(0), shape=contribution.shape)
        result = b.op("add", result, contribution)
    gate = b.linear(flat, prefix + ".shared_expert.gate_proj.weight", shared)
    up = b.linear(flat, prefix + ".shared_expert.up_proj.weight", shared)
    shared_out = b.linear(b.op("mul", b.op("silu", gate), up), prefix + ".shared_expert.down_proj.weight", h)
    shared_gate = b.op("sigmoid", b.linear(flat, prefix + ".shared_expert_gate.weight", 1))
    return b.view(b.op("add", result, b.op("mul", shared_out, shared_gate)), (1, s, h))


def _config(config, draft, sequence_length):
    c = dict(config.get("text_config", config))
    integer_keys = ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads", "num_key_value_heads",
                    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                    "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                    "shared_expert_intermediate_size")
    for key in integer_keys:
        if key not in c or isinstance(c[key], bool) or int(c[key]) != c[key] or int(c[key]) <= 0:
            raise ValueError(f"target {key} must be a positive integer")
        c[key] = int(c[key])
    c["head_dim"] = int(c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]))
    c["rms_norm_eps"] = float(c.get("rms_norm_eps", 1e-6))
    c["layer_types"] = list(c["layer_types"])
    if len(c["layer_types"]) != c["num_hidden_layers"] or set(c["layer_types"]) - {"linear_attention", "full_attention"}:
        raise ValueError("target layer_types must declare every full/linear attention layer")
    for key, expected in (("hidden_size", draft.hidden_size), ("vocab_size", draft.vocab_size), ("num_hidden_layers", draft.num_target_layers)):
        if c[key] != expected:
            raise ValueError(f"target/draft {key} mismatch")
    if c.get("attention_bias", False) or float(c.get("attention_dropout", 0)) != 0 or c.get("hidden_act", "silu") != "silu":
        raise ValueError("direct target requires bias-free, zero-dropout SiLU inference")
    if c["num_attention_heads"] % c["num_key_value_heads"] or c["linear_num_value_heads"] % c["linear_num_key_heads"]:
        raise ValueError("target query/value head counts must divide their KV/key head counts")
    if c["num_experts_per_tok"] > c["num_experts"]:
        raise ValueError("target top-k exceeds expert count")
    if not math.isfinite(c["rms_norm_eps"]) or c["rms_norm_eps"] <= 0:
        raise ValueError("target rms_norm_eps must be finite and positive")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError("sequence_length must be an integer >= 2")
    rope = dict(c.get("rope_parameters", {}))
    if rope.get("rope_type", "default") != "default":
        raise ValueError("direct target supports default text RoPE only")
    c["rotary_dim"] = int(c["head_dim"] * float(rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 0.25))))
    c["rope_theta"] = float(rope.get("rope_theta", c.get("rope_theta", 10000.0)))
    if not 0 < c["rotary_dim"] <= c["head_dim"] or c["rotary_dim"] % 2 or not math.isfinite(c["rope_theta"]) or c["rope_theta"] <= 0:
        raise ValueError("target default RoPE requires a positive even rotary width and finite positive theta")
    if sequence_length > int(c.get("max_position_embeddings", sequence_length)):
        raise ValueError("sequence_length exceeds target position limit")
    draft.validate()
    return c


def build_target(config, draft_config, sequence_length, *, head_chunk_rows=16):
    """Construct both target ABI outputs without importing a tensor framework."""
    c = _config(config, draft_config, sequence_length)
    if isinstance(head_chunk_rows, bool) or not isinstance(head_chunk_rows, int) or head_chunk_rows < 1:
        raise ValueError("head_chunk_rows must be a positive integer")
    s, h = sequence_length, c["hidden_size"]
    graph = Graph("target")
    b = _Builder(graph)
    ids = graph.input("input_ids", (1, s), "int32")
    attention = graph.input("attention_mask", (1, s), "int32")
    table = graph.weight("target.embed_tokens.weight", (c["vocab_size"], h))
    hidden = b.op("gather", table, ids, shape=(1, s, h), axis=0)
    valid = b.op("cast", attention, dtype="bool")
    causal = b.constant(np.tril(np.ones((1, 1, s, s), dtype=bool)), "bool")
    visible = b.op("logical_and", causal, b.view(valid, (1, 1, 1, s)), shape=(1, 1, s, s), dtype="bool")
    full_mask = b.op("where", visible, b.constant(0), b.constant(np.finfo(np.float16).min), shape=(1, 1, s, s))
    # The upstream RoPE buffer is computed in FP32 then explicitly cast by the
    # deployment precision pass. Keep that stored-buffer conversion here.
    inv = (1.0 / (np.float32(c["rope_theta"]) ** (np.arange(0, c["rotary_dim"], 2, dtype=np.float32) / c["rotary_dim"]))).astype(np.float16)
    frequency = b.op("matmul", b.constant(inv.reshape(1, -1, 1)), b.constant(np.arange(s, dtype=np.float16).reshape(1, 1, s)),
                     shape=(1, len(inv), s), transpose_a=False, transpose_b=False)
    frequency = b.permute(frequency, (0, 2, 1))
    angles = b.cat([frequency, frequency], -1)
    rotary = tuple(b.view(b.op(kind, angles), (1, 1, s, c["rotary_dim"])) for kind in ("cos", "sin"))
    captures = {}
    for index, kind in enumerate(c["layer_types"]):
        prefix = f"target.layers.{index}"
        normed = b.rms(hidden, prefix + ".input_layernorm.weight", c["rms_norm_eps"])
        mixed = (_full_attention(b, normed, prefix + ".self_attn", c, full_mask, rotary)
                 if kind == "full_attention" else _linear_attention(b, normed, prefix + ".linear_attn", c))
        hidden = b.op("add", hidden, mixed)
        normalized = b.rms(hidden, prefix + ".post_attention_layernorm.weight", c["rms_norm_eps"])
        hidden = b.op("add", hidden, _moe(b, normalized, prefix + ".mlp", c))
        if index in draft_config.target_layer_ids:
            captures[index] = hidden
    hidden = b.rms(hidden, "target.norm.weight", c["rms_norm_eps"])
    if c["num_hidden_layers"] - 1 in captures:
        captures[c["num_hidden_layers"] - 1] = hidden
    features = b.cat([captures[i] for i in draft_config.target_layer_ids], -1)
    top1 = []
    for start in range(0, s, head_chunk_rows):
        rows = b.slice(hidden, 1, start, min(head_chunk_rows, s-start))
        logits = b.linear(rows, "target.lm_head.weight", c["vocab_size"])
        top1.append(b.op("argmax", logits, shape=logits.shape[:-1], dtype="int32", axis=-1, keepdims=False))
    graph.outputs = [b.cat(top1, 1), features]
    return graph

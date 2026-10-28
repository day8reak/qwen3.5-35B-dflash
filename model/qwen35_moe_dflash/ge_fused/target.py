"""Stateful Qwen3.5 target constructed directly as a GE graph, without Torch.

All tokens in a call are valid and contiguous. Past length is a static graph
specialization; cache tensors are explicit values, never hidden mutations.
The optional fused roles describe a logical ND adapter contract. The installed
vendor interface must be supplied by a checked binding, not inferred here.
"""
from __future__ import annotations

import numpy as np

from ..ge_target import _Builder, _config, _moe
from .ir import Graph


FUSION_ROLES = frozenset({"gdn", "conv_state", "rms"})


class _StatefulBuilder(_Builder):
    def __init__(self, graph, fused_ops, bindings, *, gdn_indexed_copy="native", gdn_copy_report=None,
                 gdn_state_casts="reject"):
        super().__init__(graph, gdn_indexed_copy=gdn_indexed_copy, gdn_copy_report=gdn_copy_report)
        self.fused_ops = frozenset(fused_ops)
        self.bindings = bindings
        self.gdn_state_casts = gdn_state_casts
        self.gdn_state_contracts = []

    def embedding(self, ids, config):
        return self.op("gather", self.g.weight("target.embed_tokens.weight",
                       (config["vocab_size"], config["hidden_size"])), ids,
                       shape=(1, ids.shape[1], config["hidden_size"]), axis=0)

    def input_embedding(self, sequence_length, config):
        return self.embedding(self.g.input("input_ids", (1, sequence_length), "int32"), config)

    def moe(self, x, prefix, config, expert_dispatch):
        return _moe(self, x, prefix, config, expert_dispatch)

    def gdn_decay(self, a, prefix, heads):
        a = self.op("softplus", self.op("add", a, self.g.weight(prefix + ".dt_bias", (heads,))),
                    beta=1.0, threshold=20.0)
        return self.op("mul", self.op("neg", self.op("exp", self.g.weight(prefix + ".A_log", (heads,)))), a,
                       shape=a.shape)

    def gated_norm(self, core, gate, key, eps):
        return self.op("mul", self.rms(core, key, eps, offset=False), self.op("silu", gate))

    def rotary_tables(self, config, sequence_length, past_length):
        s, total = sequence_length, past_length + sequence_length
        if total - 1 > float(np.finfo(np.float16).max):
            raise ValueError("position exceeds this graph's finite FP16 RoPE range")
        inv = (1.0 / (np.float32(config["rope_theta"]) **
                      (np.arange(0, config["rotary_dim"], 2, dtype=np.float32) / config["rotary_dim"]))).astype(np.float16)
        # Form integer positions before casting; FP16 arange can round its step.
        positions = np.arange(past_length, total, dtype=np.int64).astype(np.float16)
        frequency = self.op("matmul", self.constant(inv.reshape(1, -1, 1)),
                            self.constant(positions.reshape(1, 1, s)), shape=(1, len(inv), s),
                            transpose_a=False, transpose_b=False)
        frequency = self.permute(frequency, (0, 2, 1))
        angles = self.cat([frequency, frequency], -1)
        return tuple(self.view(self.op(kind, angles), (1, 1, s, config["rotary_dim"]))
                     for kind in ("cos", "sin"))

    def bound_dtype(self, role, direction, logical, fallback):
        declared = getattr(self.bindings, "tensor_dtype", None)
        return declared(role, direction, logical) if callable(declared) else fallback

    def rms(self, x, key, eps, offset=True):
        if "rms" not in self.fused_ops:
            return super().rms(x, key, eps, offset)
        gamma = self.g.weight(key, (x.shape[-1],))
        # Main/q/k norms store HF zero-centered weights. Gated GDN norm stores
        # its effective gamma and must not receive this +1 conversion.
        if offset:
            gamma = self.op("scalar_add", gamma, value=1.0)
        rstd_dtype = self.bound_dtype("rms", "outputs", "rstd", x.dtype)
        if rstd_dtype not in {"float16", "float32"}:
            raise ValueError("fused RMS rstd dtype must be float16 or float32")
        result = self.bindings.call(
            self.g, "rms", inputs={"x": x, "gamma": gamma},
            outputs={"y": (x.shape, x.dtype),
                     "rstd": ((*x.shape[:-1], 1), rstd_dtype)},
            attrs={"epsilon": float(eps)}, name=key.removesuffix(".weight") + "/fused")
        return result["y"]


def _expanded_gdn(b, query, key, value, decay, beta, initial_state):
    """FP16 chunk delta rule with an explicit incoming and outgoing state.

This retains the existing chunk-64 arithmetic for multi-token calls. A single
token uses chunk 1. These FP16 evaluation orders need not be bitwise identical
across different call segmentations; this is not a device accuracy claim.
"""
    sequence, heads, kdim = query.shape[1:]
    vdim = value.shape[-1]
    if any(x.dtype != "float16" for x in (query, key, value, decay, beta, initial_state)):
        raise ValueError("expanded GDN requires float16 state and inputs")
    if initial_state.shape != (1, heads, kdim, vdim):
        raise ValueError("expanded GDN recurrent state shape mismatch")
    chunk = 1 if sequence == 1 else 64
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
    query = b.op("scalar_mul", query, value=kdim ** -0.5)
    v_beta = b.op("mul", value, b.view(beta, (1, heads, padded, 1)))
    k_beta = b.op("mul", key, b.view(beta, (1, heads, padded, 1)))
    query, key, value, k_beta, v_beta = [
        b.view(x, (heads, count, chunk, x.shape[-1]))
        for x in (query, key, value, k_beta, v_beta)]
    decay = b.op("cumsum", b.view(decay, (heads, count, chunk)), axis=-1,
                 scope="gdn_nonpositive")
    zero_row = b.constant(np.zeros((heads, count, 1, chunk), dtype=np.float16))

    def triangular_mask(x, *, strict=False, site):
        strategy, reason = b.gdn_copy_strategy("triangular_mask", x)
        if strategy == "gather":
            from ..ge_indexed_copy import triangular_mask_gather
            name = f"__gdn_copy/mask_{b.gdn_copy_report['counts']['total']}"
            result = triangular_mask_gather(b.g, x, strict=strict, name=name)
            return b.record_gdn_copy("triangular_mask", (x,), result, strategy, reason,
                                     strict=strict, site=site)
        rows = []
        for row in range(chunk):
            kept = row if strict else row + 1
            if kept == 0:
                rows.append(zero_row)
                continue
            parts = [b.slice(b.slice(x, -2, row, 1), -1, 0, kept)]
            if kept < chunk:
                parts.append(b.slice(zero_row, -1, 0, chunk - kept))
            rows.append(b.cat(parts, -1))
        return b.record_gdn_copy("triangular_mask", (x,), b.cat(rows, -2), strategy, reason,
                                 strict=strict, site=site)

    left = b.view(decay, (heads, count, chunk, 1))
    right = b.view(decay, (heads, count, 1, chunk))
    strategy, reason = b.gdn_copy_strategy("outer_sub", left, right)
    if strategy == "gather":
        from ..ge_indexed_copy import outer_sub_gather
        diff = outer_sub_gather(b.g, left, right,
                               name=f"__gdn_copy/outer_sub_{b.gdn_copy_report['counts']['total']}")
    else:
        diff = b.op("sub", left, right, shape=(heads, count, chunk, chunk))
    b.record_gdn_copy("outer_sub", (left, right), diff, strategy, reason)
    decay_mask = triangular_mask(diff, site="before_exp")
    decay_mask = triangular_mask(b.op("exp", decay_mask), site="after_exp")
    attn = b.op("matmul", k_beta, key, shape=diff.shape,
                transpose_a=False, transpose_b=True)
    attn = b.op("neg", triangular_mask(b.op("mul", attn, decay_mask), strict=True, site="before_neg"))
    for i in range(1, chunk):
        row = b.slice(b.slice(attn, -2, i, 1), -1, 0, i)
        sub = b.slice(b.slice(attn, -2, 0, i), -1, 0, i)
        row_flat = b.view(row, (heads, count, i))
        product = b.op("mul", b.view(row_flat, (heads, count, i, 1)), sub, shape=sub.shape)
        fixed = b.op("add", row_flat, b.reduce("reduce_sum", product, -2, False))
        fixed = b.view(fixed, (heads, count, 1, i))
        full_row = b.cat([fixed, b.slice(b.slice(attn, -2, i, 1), -1, i, chunk - i)], -1)
        rows = [b.slice(attn, -2, 0, i), full_row]
        if i + 1 < chunk:
            rows.append(b.slice(attn, -2, i + 1, chunk - i - 1))
        attn = b.cat(rows, -2)
    attn = b.op("add", attn, b.constant(np.eye(chunk, dtype=np.float16)))
    value = b.op("matmul", attn, v_beta, shape=(heads, count, chunk, vdim),
                 transpose_a=False, transpose_b=False)
    exp_decay = b.view(b.op("exp", decay), (heads, count, chunk, 1))
    k_cum = b.op("matmul", attn, b.op("mul", k_beta, exp_decay),
                 shape=(heads, count, chunk, kdim), transpose_a=False, transpose_b=False)
    state, out = initial_state, []
    for i in range(count):
        def take(x):
            return b.view(b.slice(x, 1, i, 1), (1, heads, *x.shape[2:]))

        q_i, k_i, v_i = [take(x) for x in (query, key, value)]
        scores = b.op("matmul", q_i, k_i, shape=(1, heads, chunk, chunk),
                      transpose_a=False, transpose_b=True)
        scores = b.op("mul", scores, take(decay_mask))
        old_value = b.op("matmul", take(k_cum), state, shape=(1, heads, chunk, vdim),
                         transpose_a=False, transpose_b=False)
        new_value = b.op("sub", v_i, old_value)
        g_i = take(decay)
        exp_i = b.view(b.op("exp", g_i), (1, heads, chunk, 1))
        inter = b.op("matmul", b.op("mul", q_i, exp_i), state,
                     shape=(1, heads, chunk, vdim), transpose_a=False, transpose_b=False)
        intra = b.op("matmul", scores, new_value,
                     shape=(1, heads, chunk, vdim), transpose_a=False, transpose_b=False)
        out.append(b.op("add", inter, intra))
        # Zero-padded q/k/v/beta and zero decay preserve the last valid state.
        final = b.slice(g_i, -1, chunk - 1, 1)
        factor = b.view(b.op("exp", b.op("sub", final, g_i, shape=g_i.shape)),
                        (1, heads, chunk, 1))
        carry = b.op("mul", state, b.view(b.op("exp", final), (1, heads, 1, 1)))
        update = b.op("matmul", b.op("mul", k_i, factor), new_value,
                      shape=state.shape, transpose_a=True, transpose_b=False)
        state = b.op("add", carry, update)
    result = b.cat(out, 2)
    if padded > sequence:
        result = b.slice(result, 2, 0, sequence)
    return b.permute(result, (0, 2, 1, 3)), state


def _linear_attention(b, x, prefix, c, history, recurrent):
    s, h = x.shape[1], c["hidden_size"]
    kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd, kernel = c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
    kw, vw = kh * kd, vh * vd
    width = 2 * kw + vw
    projected = b.linear(x, prefix + ".in_proj_qkv.weight", width)
    if "conv_state" in b.fused_ops:
        raw_history_size = b.bindings.conv_history_size(kernel)
        if raw_history_size not in {kernel - 1, kernel}:
            raise ValueError("conv_state binding must declare K or K-1 history width")
        history_policy = b.bindings.conv_history_slice
        if history_policy not in {"last_kernel_minus_one", "padded_tail"}:
            raise ValueError("conv_state binding must explicitly select the last K-1 history entries")
        result = b.bindings.call(
            b.g, "conv_state",
            inputs={"projected": projected, "history": history,
                    "update_control": b.constant(np.zeros(128, dtype=np.int32), "int32")},
            outputs={"history": ((1, width, raw_history_size), "float16"),
                     "padded": ((1, width, s + kernel - 1), "float16")},
            attrs={}, name=prefix + "/fused_conv_state")
        padded, next_history = result["padded"], result["history"]
        if history_policy == "padded_tail":
            # The documented y2 is history || projected. Its final K-1 entries
            # define the next logical state without assuming y1's chronology.
            next_history = b.slice(padded, -1, s, kernel - 1)
        elif raw_history_size == kernel:
            next_history = b.slice(next_history, -1, 1, kernel - 1)
    else:
        padded = b.cat([history, b.permute(projected, (0, 2, 1))], -1)
        next_history = b.slice(padded, -1, s, kernel - 1)
    conv = b.g.weight(prefix + ".conv1d.weight", (width, 1, kernel))
    windows = b.cat([b.view(b.slice(padded, -1, tap, s), (width, s, 1))
                     for tap in range(kernel)], -1)
    convolved = b.op("matmul", windows, b.view(conv, (width, kernel, 1)),
                     shape=(width, s, 1), transpose_a=False, transpose_b=False)
    mixed = b.permute(b.op("silu", b.view(convolved, (1, width, s))), (0, 2, 1))
    query = b.view(b.slice(mixed, -1, 0, kw), (1, s, kh, kd))
    key = b.view(b.slice(mixed, -1, kw, kw), (1, s, kh, kd))
    value = b.view(b.slice(mixed, -1, 2 * kw, vw), (1, s, vh, vd))
    query, key = [b.repeat_heads(t, 2, vh // kh) for t in (query, key)]
    beta = b.op("sigmoid", b.linear(x, prefix + ".in_proj_b.weight", vh))
    a = b.linear(x, prefix + ".in_proj_a.weight", vh)
    decay = b.gdn_decay(a, prefix, vh)
    if "gdn" in b.fused_ops:
        native_input_dtype = b.bound_dtype("gdn", "inputs", "initial_state", recurrent.dtype)
        native_output_dtype = b.bound_dtype("gdn", "outputs", "final_state", recurrent.dtype)
        if any(dtype not in {"float16", "float32"} for dtype in (native_input_dtype, native_output_dtype)):
            raise ValueError("fused GDN native state dtypes must be float16 or float32")
        needs_conversion = any(dtype != recurrent.dtype for dtype in (native_input_dtype, native_output_dtype))
        if needs_conversion and b.gdn_state_casts != "allow":
            raise ValueError("fused GDN state dtype differs from cache; set gdn_state_casts='allow' "
                             "to emit explicit state Cast operations")
        conversion_records = []

        def convert_state(source, dtype, direction):
            if source.dtype == dtype:
                return source
            result = b.op("cast", source, dtype=dtype)
            conversion_records.append({"direction": direction, "node": result.name,
                                       "source": source.name, "from_dtype": source.dtype,
                                       "to_dtype": dtype})
            return result

        initial_state = convert_state(recurrent, native_input_dtype, "input")
        result = b.bindings.call(
            b.g, "gdn", inputs={"query": query, "key": key, "value": value,
                                 "g": decay, "beta": beta, "initial_state": initial_state,
                                 "effective_length": b.constant([s], "int16")},
            outputs={"core_attn": ((1, s, vh, vd), "float16"),
                     "final_state": (recurrent.shape, native_output_dtype)},
            attrs={"chunk_size": 1 if s == 1 else 64, "output_final_state": True,
                   "use_qk_l2norm_in_kernel": True}, name=prefix + "/fused_gdn")
        core, next_recurrent = result["core_attn"], result["final_state"]
        next_recurrent = convert_state(next_recurrent, recurrent.dtype, "output")
        b.gdn_state_contracts.append({"scope": prefix, "cache_dtype": recurrent.dtype,
                                      "native_input_dtype": native_input_dtype,
                                      "native_output_dtype": native_output_dtype,
                                      "casts": conversion_records})
    else:
        core, next_recurrent = _expanded_gdn(b, query, key, value, decay, beta, recurrent)
    z = b.view(b.linear(x, prefix + ".in_proj_z.weight", vw), (1, s, vh, vd))
    core = b.gated_norm(core, z, prefix + ".norm.weight", c["rms_norm_eps"])
    return b.linear(b.view(core, (1, s, vw)), prefix + ".out_proj.weight", h), next_history, next_recurrent


def _full_attention(b, x, prefix, c, past_key, past_value, mask, rotary):
    s, h = x.shape[1], c["hidden_size"]
    heads, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    qgate = b.view(b.linear(x, prefix + ".q_proj.weight", heads * d * 2), (1, s, heads, d * 2))
    q = b.slice(qgate, -1, 0, d)
    gate = b.view(b.slice(qgate, -1, d, d), (1, s, heads * d))
    q = b.permute(b.rms(q, prefix + ".q_norm.weight", c["rms_norm_eps"]), (0, 2, 1, 3))
    k = b.view(b.linear(x, prefix + ".k_proj.weight", kv * d), (1, s, kv, d))
    k = b.permute(b.rms(k, prefix + ".k_norm.weight", c["rms_norm_eps"]), (0, 2, 1, 3))
    v = b.permute(b.view(b.linear(x, prefix + ".v_proj.weight", kv * d), (1, s, kv, d)), (0, 2, 1, 3))
    q, k = [b.rotary(t, *rotary) for t in (q, k)]
    if past_key is not None:
        k, v = b.cat([past_key, k], 2), b.cat([past_value, v], 2)
    next_key, next_value = k, v
    k, v = [b.repeat_heads(t, 1, heads // kv) for t in (k, v)]
    scores = b.op("matmul", q, k, shape=(1, heads, s, k.shape[2]),
                  transpose_a=False, transpose_b=True)
    scores = b.op("add", b.op("scalar_mul", scores, value=d ** -0.5), mask)
    probs = b.op("softmax", scores, axis=-1)
    mixed = b.op("matmul", probs, v, shape=(1, heads, s, d),
                 transpose_a=False, transpose_b=False)
    mixed = b.view(b.permute(mixed, (0, 2, 1, 3)), (1, s, heads * d))
    result = b.linear(b.op("mul", mixed, b.op("sigmoid", gate)), prefix + ".o_proj.weight", h)
    return result, next_key, next_value


def build_target(config, draft_config, sequence_length, *, past_length=0, fused_ops=(),
                 bindings=None, recurrent_dtype="float16", expert_dispatch="static",
                 head_chunk_rows=16, gdn_indexed_copy="native", gdn_copy_report=None,
                 gdn_state_casts="reject", builder_factory=None,
                 output_positions=None, return_logits=False):
    """Build a stateful target; selected fusion roles require explicit bindings.

    Outputs are target_top1, target_features, then conv/recurrent or key/value
    for every layer in layer order. Optional FP32 target_logits is appended last.
    output_positions selects zero-based current-call rows for the head only;
    negative indices are normalized, and all rows still update features/state.
    ``graph.state_abi`` records state routing. A graph for a different past
    length must be built to append more tokens.
    """
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if type(return_logits) is not bool:
        raise ValueError("return_logits must be bool")
    if output_positions is not None:
        if (not isinstance(output_positions, tuple) or not output_positions
                or any(type(value) is not int or not -sequence_length <= value < sequence_length
                       for value in output_positions)):
            raise ValueError("output_positions must be a nonempty tuple of valid zero-based token indices")
        output_positions = tuple(value % sequence_length for value in output_positions)
    if isinstance(past_length, bool) or not isinstance(past_length, int) or past_length < 0:
        raise ValueError("past_length must be a nonnegative integer")
    if isinstance(fused_ops, str):
        raise ValueError("fused_ops must be a collection of role names, not a string")
    selected = frozenset(fused_ops)
    if selected - FUSION_ROLES:
        raise ValueError(f"unsupported fused roles: {sorted(selected - FUSION_ROLES)}")
    if selected and (bindings is None or not callable(getattr(bindings, "call", None))):
        raise ValueError("selected fused operators require explicit custom bindings")
    if recurrent_dtype not in {"float16", "float32"}:
        raise ValueError("recurrent_dtype must be float16 or float32")
    if gdn_state_casts not in {"reject", "allow"}:
        raise ValueError("gdn_state_casts must be reject or allow")
    if recurrent_dtype != "float16" and "gdn" not in selected:
        raise ValueError("float32 recurrent state requires fused gdn")
    if "gdn" in selected and sequence_length > np.iinfo(np.int16).max:
        raise ValueError("fused GDN effective_length exceeds int16 capacity")
    if expert_dispatch not in {"static", "selected"}:
        raise ValueError("expert_dispatch must be static or selected")
    if isinstance(head_chunk_rows, bool) or not isinstance(head_chunk_rows, int) or head_chunk_rows < 1:
        raise ValueError("head_chunk_rows must be a positive integer")
    # Reuse structural validation; the old recompute interface requires >=2,
    # while this separate interface explicitly admits one-token decode.
    c = _config(config, draft_config, max(2, past_length + sequence_length))
    if c["linear_conv_kernel_dim"] < 2:
        raise ValueError("stateful conv history requires kernel width >= 2")
    if expert_dispatch == "selected" and c["num_experts"] > 2048:
        raise ValueError("selected expert dispatch requires at most 2048 experts")
    s, h, total = sequence_length, c["hidden_size"], past_length + sequence_length
    graph = Graph("stateful_target")
    factory = _StatefulBuilder if builder_factory is None else builder_factory
    if not callable(factory):
        raise ValueError("builder_factory must be callable or None")
    b = factory(graph, selected, bindings, gdn_indexed_copy=gdn_indexed_copy,
                gdn_copy_report=gdn_copy_report, gdn_state_casts=gdn_state_casts)
    graph.gdn_copy_report = b.gdn_copy_report
    hidden = b.input_embedding(s, c)
    visible = np.arange(total)[None, :] <= np.arange(past_length, total)[:, None]
    mask = b.constant(np.where(visible, 0, np.finfo(np.float16).min).reshape(1, 1, s, total))
    rotary = b.rotary_tables(c, s, past_length)
    captures, state_outputs, state_records = {}, [], []

    def incoming(index, role, shape, dtype="float16"):
        name = f"layer_{index}_{role}"
        if past_length:
            return graph.input(name, shape, dtype)
        return graph.const(np.zeros(shape, dtype=dtype), dtype=dtype, name=name + "_initial")

    def record(index, layer_type, role, old, new):
        name = f"layer_{index}_{role}_out"
        new = graph.op("reshape", new, shape=new.shape, dtype=new.dtype,
                       attrs={"shape": list(new.shape)}, name=name)
        state_outputs.append(new)
        state_records.append({
            "layer": index, "layer_type": layer_type, "role": role,
            "input": old.name if past_length else None, "output": new.name,
            "input_shape": list(old.shape) if old is not None else None,
            "output_shape": list(new.shape), "dtype": new.dtype,
            "initialization": "input" if past_length else "zeros"})

    for index, kind in enumerate(c["layer_types"]):
        prefix = f"target.layers.{index}"
        normed = b.rms(hidden, prefix + ".input_layernorm.weight", c["rms_norm_eps"])
        if kind == "linear_attention":
            width = (2 * c["linear_num_key_heads"] * c["linear_key_head_dim"]
                     + c["linear_num_value_heads"] * c["linear_value_head_dim"])
            history = incoming(index, "conv_history", (1, width, c["linear_conv_kernel_dim"] - 1))
            recurrent = incoming(index, "recurrent", (1, c["linear_num_value_heads"],
                                  c["linear_key_head_dim"], c["linear_value_head_dim"]), recurrent_dtype)
            mixed, next_history, next_recurrent = _linear_attention(
                b, normed, prefix + ".linear_attn", c, history, recurrent)
            record(index, kind, "conv_history", history, next_history)
            record(index, kind, "recurrent", recurrent, next_recurrent)
        else:
            shape = (1, c["num_key_value_heads"], past_length, c["head_dim"])
            old_key = incoming(index, "key", shape) if past_length else None
            old_value = incoming(index, "value", shape) if past_length else None
            mixed, new_key, new_value = _full_attention(
                b, normed, prefix + ".self_attn", c, old_key, old_value, mask, rotary)
            record(index, kind, "key", old_key, new_key)
            record(index, kind, "value", old_value, new_value)
        hidden = b.op("add", hidden, mixed)
        normalized = b.rms(hidden, prefix + ".post_attention_layernorm.weight", c["rms_norm_eps"])
        hidden = b.op("add", hidden, b.moe(normalized, prefix + ".mlp", c, expert_dispatch))
        if index in draft_config.target_layer_ids:
            captures[index] = hidden
    hidden = b.rms(hidden, "target.norm.weight", c["rms_norm_eps"])
    if c["num_hidden_layers"] - 1 in captures:
        captures[c["num_hidden_layers"] - 1] = hidden
    features = b.cat([captures[i] for i in draft_config.target_layer_ids], -1)
    if features.dtype != "float16":
        features = b.op("cast", features, dtype="float16")
    head_input = hidden
    if output_positions is not None:
        head_input = b.op("gather", hidden, b.constant(output_positions, "int32"),
                          shape=(1, len(output_positions), h), dtype=hidden.dtype, axis=1)
    top1, full_logits = [], []
    for start in range(0, head_input.shape[1], head_chunk_rows):
        rows = b.slice(head_input, 1, start, min(head_chunk_rows, head_input.shape[1] - start))
        logits = b.linear(rows, "target.lm_head.weight", c["vocab_size"])
        top1.append(b.op("argmax", logits, shape=logits.shape[:-1], dtype="int32", axis=-1, keepdims=False))
        if return_logits:
            full_logits.append(b.op("cast", logits, dtype="float32"))
    def named(value, name):
        return graph.op("reshape", value, shape=value.shape, dtype=value.dtype,
                        attrs={"shape": list(value.shape)}, name=name)
    graph.outputs = [named(b.cat(top1, 1), "target_top1"), named(features, "target_features"), *state_outputs]
    if return_logits:
        graph.outputs.append(named(b.cat(full_logits, 1), "target_logits"))
    graph.output_contract = {"head_positions": list(range(s)) if output_positions is None else list(output_positions),
                             "position_scope": "zero_based_current_call", "features": "all_sequence_positions",
                             "state_update": "all_sequence_positions", "full_logits": return_logits}
    graph.state_abi = {
        "schema_version": 1, "batch_size": 1, "sequence_length": s,
        "past_length": past_length, "next_past_length": total,
        "valid_tokens": "all_sequence_positions", "recurrent_dtype": recurrent_dtype,
        "gdn_state_casts": gdn_state_casts, "gdn_state_contracts": b.gdn_state_contracts,
        "conv_state_history_policy": bindings.conv_history_slice if "conv_state" in selected else "padded_tail",
        "fused_ops": sorted(selected), "states": state_records,
        "input_order": [v.name for v in graph.inputs], "output_order": [v.name for v in graph.outputs],
        "cache_layout": "ND", "cache_update": "functional",
        "gdn_indexed_copy": gdn_indexed_copy,
        "requires_matching_past_length_graph": True,
    }
    return graph.validate()

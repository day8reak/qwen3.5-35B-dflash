"""Host semantic tests for the rank-four GDN layout, not device validation."""
from pathlib import Path

import numpy as np
import pytest
import torch

from reference.ge_torch import evaluate_torch
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import DTYPES, Graph
from qwen35_moe_dflash.ge_target import _Builder, _chunk_gdn
from test_ge_static_broadcast import _serialize_both


# Frozen pre-layout graph definition: preserve the previous rank-five carriers
# and BOOL Where masks as an independent before/after construction reference.
def _legacy_chunk_gdn(b, query, key, value, decay, beta):
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
    # This scan sums -exp(A_log) * softplus(a + dt_bias), hence nonpositive
    # values. Mark the domain for the bounded FP16 prefix-reduction lowering.
    decay = b.op("cumsum", b.view(decay, (1, heads, count, chunk)), axis=-1,
                 scope="gdn_nonpositive")
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


def _graph(function, sequence, heads=2, kdim=4, vdim=3, batch=1):
    graph = Graph("gdn_layout")
    shapes = [(batch, sequence, heads, width) for width in (kdim, kdim, vdim)]
    shapes += [(batch, sequence, heads)] * 2
    inputs = [graph.input(name, shape) for name, shape in zip(("q", "k", "v", "g", "beta"), shapes)]
    graph.outputs = [function(_Builder(graph), *inputs)]
    return graph


def _feeds(graph, seed):
    rng = np.random.default_rng(seed)
    values = {value.name: rng.normal(0, .25, value.shape).astype(np.float16) for value in graph.inputs}
    values["g"] = -rng.uniform(.01, .5, values["g"].shape).astype(np.float16)
    values["beta"] = rng.uniform(.05, .95, values["beta"].shape).astype(np.float16)
    return values


@pytest.mark.parametrize("sequence", [1, 8, 64, 65, 128])
@pytest.mark.parametrize("heads,kdim,vdim", [(1, 3, 5), (3, 4, 2)])
def test_chunk_layout_and_masks_are_bitwise_equal_to_original_graph(sequence, heads, kdim, vdim):
    torch.set_num_threads(1)
    old = _graph(_legacy_chunk_gdn, sequence, heads, kdim, vdim)
    new = _graph(_chunk_gdn, sequence, heads, kdim, vdim)
    inputs = _feeds(old, 123 + sequence)
    for backend in (evaluate, evaluate_torch):
        expected, = backend(old, inputs)
        actual, = backend(new, inputs)
        if isinstance(actual, torch.Tensor):
            actual, expected = actual.numpy(), expected.numpy()
        assert actual.shape == (1, sequence, heads, vdim) and actual.dtype == np.float16
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert max(len(value.shape) for node in new.nodes for value in (*node.inputs, node.output)) <= 4
    assert not any(node.kind == "where" for node in new.nodes)
    assert all(value.dtype == "float16" for value in new._values.values())
    # Removing the singleton and replacing masks must not replace any arithmetic
    # stage, change contracted dimensions or reassociate the row corrections.
    arithmetic = {"add", "sub", "mul", "neg", "exp", "rsqrt", "reduce_sum", "reduce_mean", "scalar_mul", "scalar_add", "cumsum", "matmul"}
    def signature(graph):
        result = []
        for node in graph.nodes:
            if node.kind in arithmetic:
                matrices = [value.shape[-2:] for value in node.inputs] if node.kind == "matmul" else None
                attrs = dict(node.attrs)
                if "axes" in attrs:
                    rank = len(node.inputs[0].shape)
                    attrs["axes"] = [(axis % rank) - rank for axis in attrs["axes"]]
                result.append((node.kind, attrs, matrices))
        return result
    assert signature(old) == signature(new)


def _mask_boundaries(graph):
    exp = next(node for node in graph.nodes if node.kind == "exp" and node.output.shape[-2:] == (64, 64))
    diff = next(node for node in graph.nodes if node.kind == "sub" and node.output.shape == exp.output.shape)
    matmul = next(node for node in graph.nodes if node.kind == "matmul")
    weighted = next(node for node in graph.nodes if node.kind == "mul" and node.inputs[0] == matmul.output)
    neg = next(node for node in graph.nodes if node.kind == "neg")
    return [(diff.output, exp.inputs[0], False), (exp.output, weighted.inputs[1], False),
            (weighted.output, neg.inputs[0], True)]


def _between(graph, source, output):
    """Cut the actual mask subgraph at its arithmetic boundaries for raw-bit tests."""
    result = Graph("gdn_mask")
    mapping = {source.name: result.input("source", source.shape, source.dtype)}
    producers = {node.output.name: node for node in graph.nodes}
    def clone(value):
        if value.name in mapping:
            return mapping[value.name]
        if value.name in graph.constants:
            copied = result.const(graph.constants[value.name], value.dtype, name=value.name)
        else:
            node = producers[value.name]
            assert node.kind in {"slice", "concat"}
            copied = result.op(node.kind, *(clone(item) for item in node.inputs), shape=value.shape,
                               dtype=value.dtype, attrs=node.attrs, name=value.name)
        mapping[value.name] = copied
        return copied
    result.outputs = [clone(output)]
    return result


def _execute_mask_recipe(records, outputs, array):
    """Interpret serialized slice offsets and concat axes; never call GE/NPU."""
    values = {}
    for index, record in enumerate(records):
        output, = record["outputs"]
        args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
        if record["type"] == "Data":
            value = array
        elif record["type"] == "Const":
            filename, offset, size = record["constant"]
            with Path(filename).open("rb") as stream:
                stream.seek(offset)
                value = np.frombuffer(stream.read(size), dtype=DTYPES[output["dtype"]]).reshape(output["shape"])
        elif record["type"] == "Slice":
            value = args[0][tuple(slice(int(start), int(start + size)) for start, size in zip(args[1], args[2]))]
        elif record["type"] == "ConcatD":
            value = np.concatenate(args, axis=record["attrs"]["concat_dim"])
        else:
            raise AssertionError(f"unexpected mask operation: {record['type']}")
        assert value.shape == output["shape"] and value.dtype == DTYPES[output["dtype"]]
        values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


@pytest.mark.parametrize("mask_index", [0, 1, 2], ids=["before-exp", "after-exp", "strict-before-neg"])
def test_static_triangles_copy_retained_nonfinite_bits_and_zero_masked_cells(tmp_path, mask_index):
    graph = _graph(_chunk_gdn, 65)
    source, output, strict = _mask_boundaries(graph)[mask_index]
    mask_graph = _between(graph, source, output)
    # Include +/-0, +/-Inf, quiet and signalling NaN payloads, subnormals, and
    # negative/positive finite values on both sides of each triangular boundary.
    markers = np.array([0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01, 0x0001, 0x8001, 0x3c00, 0xbc00], dtype=np.uint16)
    array = np.resize(markers, source.shape).view(np.float16)
    keep = np.tril(np.ones((64, 64), dtype=bool), -1 if strict else 0)
    expected = np.where(keep, array, np.float16(0))
    for backend in (evaluate, evaluate_torch):
        actual, = backend(mask_graph, {"source": array})
        if isinstance(actual, torch.Tensor):
            actual = actual.numpy()
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    records, _, outputs = _serialize_both(mask_graph, tmp_path)
    actual, = _execute_mask_recipe(records, outputs, array)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert set(record["type"] for record in records) <= {"Data", "Const", "Slice", "ConcatD"}
    assert not actual.view(np.uint16)[~np.broadcast_to(keep, array.shape)].any()


@pytest.mark.parametrize("sequence", [8, 65])
def test_serialized_gdn_has_rank_four_or_less_and_no_boolean_mask_operations(tmp_path, sequence):
    graph = _graph(_chunk_gdn, sequence)
    records, _, _ = _serialize_both(graph, tmp_path)
    assert max(len(port["shape"]) for record in records for port in record["inputs"] + record["outputs"]) <= 4
    assert not any(record["type"] in {"Select", "SelectV2", "Cast", "CastT"} for record in records)
    assert not any(port["dtype"] == "bool" for record in records for port in record["inputs"] + record["outputs"])
    bmms = [record for record in records if record["type"] == "BatchMatMulV2"]
    assert bmms and all(len(port["shape"]) == 3 for record in bmms for port in record["inputs"] + record["outputs"])


def test_gdn_singleton_layout_rejects_a_non_singleton_batch():
    with pytest.raises(ValueError, match="batch=1"):
        _graph(_chunk_gdn, 8, batch=2)


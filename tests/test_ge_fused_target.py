"""Host graph semantics and explicit state/fusion wiring, without device claims."""
from itertools import combinations
import os
import subprocess
import sys

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_target import _Builder, _chunk_gdn, build_target as build_legacy
from qwen35_moe_dflash.ge_weights import tiny_weights
from qwen35_moe_dflash.ge_fused.ir import Graph
from qwen35_moe_dflash.ge_fused.target import build_target, _expanded_gdn


@pytest.fixture(scope="module")
def configs():
    target, draft = configurations(tiny=True)
    return target, Qwen35DFlashConfig.from_dict(draft)


def test_expanded_prefill_preserves_legacy_target_and_weight_contract(configs):
    c, d = configs
    graph, legacy = build_target(c, d, 8), build_legacy(c, d, 8)
    assert {k: v.shape for k, v in graph.weights.items()} == {k: v.shape for k, v in legacy.weights.items()}
    weights = tiny_weights({k: v.shape for k, v in graph.weights.items()})
    ids = np.array([[1, 2, 3, 9, 4, 8, 5, 6]], dtype=np.int32)
    actual = evaluate(graph, {"input_ids": ids}, weights)
    expected = evaluate(legacy, {"input_ids": ids, "attention_mask": np.ones_like(ids)}, weights)
    for a, e in zip(actual[:2], expected):
        np.testing.assert_array_equal(a, e)
    assert graph.state_abi["input_order"] == ["input_ids"]
    assert [v.name for v in graph.outputs[:2]] == ["target_top1", "target_features"]
    assert len(graph.outputs) == 2 + 2 * c["num_hidden_layers"]
    assert all(x.size for x in graph.constants.values())
    for state, array in zip(graph.state_abi["states"], actual[2:]):
        assert state["input"] is None and state["initialization"] == "zeros"
        assert list(array.shape) == state["output_shape"] and np.isfinite(array).all()
    assert graph.manifest()["state_abi"] == graph.state_abi


def test_multitoken_core_zero_state_matches_existing_chunk_arithmetic():
    shape = (1, 3, 2, 4)
    rng = np.random.default_rng(143)
    arrays = [rng.normal(0, .2, shape).astype(np.float16) for _ in range(2)]
    arrays += [rng.normal(0, .2, (1, 3, 2, 3)).astype(np.float16),
               -rng.uniform(.01, .1, (1, 3, 2)).astype(np.float16),
               rng.uniform(.2, .8, (1, 3, 2)).astype(np.float16)]
    old = Graph("old")
    inputs = [old.input(n, a.shape) for n, a in zip(("q", "k", "v", "g", "beta"), arrays)]
    states = []
    old.outputs = [_chunk_gdn(_Builder(old), *inputs, state_outputs=states)]
    old.outputs += states[-1:]
    graph = Graph("new")
    new_inputs = [graph.input(v.name, v.shape) for v in inputs]
    initial = graph.input("state", (1, 2, 4, 3))
    graph.outputs = list(_expanded_gdn(_Builder(graph), *new_inputs, initial))
    payload = {v.name: a for v, a in zip(inputs, arrays)}
    expected = evaluate(old, payload)
    actual = evaluate(graph, {**payload, "state": np.zeros(initial.shape, dtype=np.float16)})
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


def test_guarded_indexed_copy_option_preserves_stateful_outputs(configs):
    c, d = configs
    report = {}
    native = build_target(c, d, 3)
    gathered = build_target(c, d, 3, gdn_indexed_copy="gather", gdn_copy_report=report)
    assert gathered.gdn_copy_report is report
    assert report["counts"]["replaced"] == 12
    assert report["counts"]["retained"] == 0
    weights = tiny_weights({k: v.shape for k, v in native.weights.items()})
    inputs = {"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}
    for a, e in zip(evaluate(gathered, inputs, weights), evaluate(native, inputs, weights)):
        np.testing.assert_array_equal(a, e)
    decode = build_target(c, d, 1, past_length=3, gdn_indexed_copy="gather")
    assert decode.gdn_copy_report["counts"]["replaced"] == 0
    assert decode.gdn_copy_report["counts"]["retained"] == 12


def test_single_token_core_consumes_incoming_state_and_exports_decay_update():
    graph = Graph("recurrent_decode")
    q = graph.input("q", (1, 1, 1, 2))
    k = graph.input("k", q.shape)
    v = graph.input("v", (1, 1, 1, 3))
    g = graph.input("g", (1, 1, 1))
    beta = graph.input("beta", g.shape)
    state = graph.input("state", (1, 1, 2, 3))
    graph.outputs = list(_expanded_gdn(_Builder(graph), q, k, v, g, beta, state))
    initial = np.array([[[[.1, .2, .3], [.4, .5, .6]]]], dtype=np.float16)
    payload = {"q": np.array([[[[1., 0.]]]], dtype=np.float16),
               "k": np.zeros(k.shape, dtype=np.float16), "v": np.ones(v.shape, dtype=np.float16),
               "g": np.full(g.shape, -.5, dtype=np.float16),
               "beta": np.full(beta.shape, .5, dtype=np.float16), "state": initial}
    output, final = evaluate(graph, payload)
    decay = np.exp(np.float32(-.5)).astype(np.float16)
    expected_state = (initial.astype(np.float32) * np.float32(decay)).astype(np.float16)
    np.testing.assert_array_equal(final, expected_state)
    scaled_query = np.float16(np.float16(2 ** -.5) * np.float32(decay))
    expected_output = (initial[:, :, :1].astype(np.float32) * np.float32(scaled_query)).astype(np.float16)
    np.testing.assert_array_equal(output, expected_output)
    zero_output, zero_state = evaluate(graph, {**payload, "state": np.zeros_like(initial)})
    assert not zero_output.any() and not zero_state.any()
    assert np.any(output) and np.any(final)


def test_chunk_boundary_state_handoff_matches_unsplit_core():
    rng = np.random.default_rng(99)
    payload = {
        "q": rng.normal(0, .2, (1, 128, 2, 4)).astype(np.float16),
        "k": rng.normal(0, .2, (1, 128, 2, 4)).astype(np.float16),
        "v": rng.normal(0, .2, (1, 128, 2, 3)).astype(np.float16),
        "g": -rng.uniform(.01, .1, (1, 128, 2)).astype(np.float16),
        "beta": rng.uniform(.2, .8, (1, 128, 2)).astype(np.float16),
        "state": rng.normal(0, .1, (1, 2, 4, 3)).astype(np.float16),
    }

    def make(length):
        graph = Graph("core")
        inputs = [graph.input(name, (1, length, *payload[name].shape[2:]))
                  for name in ("q", "k", "v", "g", "beta")]
        initial = graph.input("state", payload["state"].shape)
        graph.outputs = list(_expanded_gdn(_Builder(graph), *inputs, initial))
        return graph

    expected, expected_state = evaluate(make(128), payload)
    graph = make(64)
    first = {k: v if k == "state" else v[:, :64] for k, v in payload.items()}
    out1, state1 = evaluate(graph, first)
    second = {k: state1 if k == "state" else v[:, 64:] for k, v in payload.items()}
    out2, state2 = evaluate(graph, second)
    np.testing.assert_array_equal(np.concatenate([out1, out2], axis=1), expected)
    np.testing.assert_array_equal(state2, expected_state)


def _feed_state(graph, prior_graph, prior_outputs, ids):
    states = {x["output"]: a for x, a in zip(prior_graph.state_abi["states"], prior_outputs[2:])}
    inputs = {"input_ids": ids}
    for state in graph.state_abi["states"]:
        inputs[state["input"]] = states[state["output"]]
    return inputs


def test_decode_abi_and_cache_append_are_explicit(configs):
    c, d = configs
    first, next_graph = build_target(c, d, 3), build_target(c, d, 1, past_length=3)
    weights = tiny_weights({k: v.shape for k, v in first.weights.items()})
    prior = evaluate(first, {"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}, weights)
    inputs = _feed_state(next_graph, first, prior, np.array([[4]], dtype=np.int32))
    following = evaluate(next_graph, inputs, weights)
    assert len(next_graph.inputs) == 1 + 2 * c["num_hidden_layers"]
    assert next_graph.state_abi["next_past_length"] == 4
    for record, array in zip(next_graph.state_abi["states"], following[2:]):
        old = inputs[record["input"]]
        if record["role"] in {"key", "value"}:
            np.testing.assert_array_equal(array[:, :, :3], old)
            assert array.shape[2] == 4
        elif record["role"] == "conv_history":
            np.testing.assert_array_equal(array[..., :-1], old[..., 1:])
        assert np.isfinite(array).all()
    assert np.any(following[1] != evaluate(next_graph, {**inputs, **{
        record["input"]: np.zeros_like(inputs[record["input"]])
        for record in next_graph.state_abi["states"]}}, weights)[1])


def test_full_attention_cached_decode_matches_full_prefix(configs):
    c, d = configs
    c = {**c, "layer_types": ["full_attention"] * c["num_hidden_layers"]}
    full, prefill, decode = build_target(c, d, 4), build_target(c, d, 3), build_target(c, d, 1, past_length=3)
    weights = tiny_weights({k: v.shape for k, v in full.weights.items()})
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int32)
    expected = evaluate(full, {"input_ids": tokens}, weights)
    before = evaluate(prefill, {"input_ids": tokens[:, :3]}, weights)
    actual = evaluate(decode, _feed_state(decode, prefill, before, tokens[:, 3:]), weights)
    np.testing.assert_array_equal(actual[0], expected[0][:, 3:])
    np.testing.assert_array_equal(actual[1], expected[1][:, 3:])
    for a, e in zip(actual[2:], expected[2:]):
        np.testing.assert_array_equal(a, e)


class RecordingBindings:
    """Synthetic schema to test graph wiring, never an installed operator claim."""
    conv_history_slice = "last_kernel_minus_one"

    def __init__(self, raw_history="kernel"):
        self.calls = []
        self.raw_history = raw_history

    def conv_history_size(self, kernel):
        return kernel if self.raw_history == "kernel" else kernel - 1

    def call(self, graph, role, *, inputs, outputs, attrs, name=None):
        self.calls.append((role, inputs, outputs, attrs, name))
        typed = {k: {"type": "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else "float", "value": v}
                 for k, v in attrs.items()}
        return graph.custom("Test" + role.replace("_", "").title(), inputs, outputs, typed, name)


class MixedStateBindings(RecordingBindings):
    def __init__(self, input_dtype="float16", output_dtype="float32", rstd_dtype="float32"):
        super().__init__()
        self.input_dtype, self.output_dtype, self.rstd_dtype = input_dtype, output_dtype, rstd_dtype

    def tensor_dtype(self, role, direction, logical):
        return {("gdn", "inputs", "initial_state"): self.input_dtype,
                ("gdn", "outputs", "final_state"): self.output_dtype,
                ("rms", "outputs", "rstd"): self.rstd_dtype}[(role, direction, logical)]


@pytest.mark.parametrize("cache_dtype,cast_direction", [("float16", "output"), ("float32", "input")])
def test_explicit_mixed_gdn_state_casts_keep_public_cache_contract(configs, cache_dtype, cast_direction):
    c, d = configs
    bindings = MixedStateBindings()
    graph = build_target(c, d, 1, past_length=3, fused_ops=("gdn",), bindings=bindings,
                         recurrent_dtype=cache_dtype, gdn_state_casts="allow")
    assert graph.state_abi["gdn_state_casts"] == "allow"
    contracts = graph.state_abi["gdn_state_contracts"]
    assert len(contracts) == 3
    producers = {node.output.name: node for node in graph.nodes}
    states = [record for record in graph.state_abi["states"] if record["role"] == "recurrent"]
    for contract, state in zip(contracts, states):
        assert contract["native_input_dtype"] == "float16"
        assert contract["native_output_dtype"] == "float32"
        assert contract["cache_dtype"] == state["dtype"] == cache_dtype
        assert len(contract["casts"]) == 1
        conversion = contract["casts"][0]
        assert conversion["direction"] == cast_direction
        assert conversion["from_dtype"] == "float32" and conversion["to_dtype"] == "float16"
        cast = producers[conversion["node"]]
        assert cast.kind == "cast" and cast.inputs[0].dtype == "float32" and cast.output.dtype == "float16"
        assert cast.inputs[0].name == conversion["source"]
    for role, inputs, outputs, _, _ in bindings.calls:
        assert role == "gdn"
        assert inputs["initial_state"].dtype == "float16"
        assert outputs["final_state"][1] == "float32"
    for value in graph.inputs:
        if value.name.endswith("_recurrent"):
            assert value.dtype == cache_dtype
    for value in graph.outputs:
        if value.name.endswith("_recurrent_out"):
            assert value.dtype == cache_dtype


@pytest.mark.parametrize("cache_dtype", ["float16", "float32"])
def test_mixed_gdn_state_requires_explicit_conversion_policy(configs, cache_dtype):
    c, d = configs
    with pytest.raises(ValueError, match="gdn_state_casts='allow'"):
        build_target(c, d, 1, past_length=3, fused_ops=("gdn",), bindings=MixedStateBindings(),
                     recurrent_dtype=cache_dtype)


@pytest.mark.parametrize("native_types", [("int32", "float32"), ("float16", "int32")])
def test_gdn_state_cast_option_does_not_permit_integer_state(configs, native_types):
    c, d = configs
    with pytest.raises(ValueError, match="native state dtypes"):
        build_target(c, d, 1, fused_ops=("gdn",), bindings=MixedStateBindings(*native_types),
                     gdn_state_casts="allow")


def test_fused_rms_preserves_explicit_fp32_auxiliary_output(configs):
    c, d = configs
    bindings = MixedStateBindings()
    graph = build_target(c, d, 1, fused_ops=("rms",), bindings=bindings)
    for role, _, outputs, _, _ in bindings.calls:
        assert role == "rms" and outputs["y"][1] == "float16" and outputs["rstd"][1] == "float32"
    for node in graph.nodes:
        if node.kind == "custom":
            assert {port: value.dtype for port, value in zip(node.output_ports, node.outputs)} == {
                "y": "float16", "rstd": "float32"}


@pytest.mark.parametrize("roles", [tuple(c) for n in range(4) for c in combinations(("gdn", "conv_state", "rms"), n)])
def test_each_fusion_combination_changes_only_selected_roles(configs, roles):
    c, d = configs
    bindings = RecordingBindings()
    graph = build_target(c, d, 3, fused_ops=roles, bindings=bindings)
    calls = bindings.calls
    assert {role for role, *_ in calls} == set(roles)
    assert graph.state_abi["fused_ops"] == sorted(roles)
    assert len(graph.outputs) == 10
    for role, inputs, outputs, attrs, name in calls:
        if role == "gdn":
            assert inputs["initial_state"].shape == (1, 2, 8, 8)
            effective = inputs["effective_length"]
            assert effective.dtype == "int16" and graph.constants[effective.name].tolist() == [3]
            assert attrs == {"chunk_size": 64, "output_final_state": True,
                             "use_qk_l2norm_in_kernel": True}
            assert outputs == {"core_attn": ((1, 3, 2, 8), "float16"),
                               "final_state": ((1, 2, 8, 8), "float16")}
        elif role == "conv_state":
            assert inputs["history"].shape == (1, 48, 3)
            assert outputs["history"] == ((1, 48, 4), "float16")
            assert outputs["padded"] == ((1, 48, 6), "float16")
    assert len([node for node in graph.nodes if node.kind == "custom"]) == len(calls)


def test_fused_fp32_state_decode_and_gamma_conventions(configs):
    c, d = configs
    bindings = RecordingBindings(raw_history="kernel_minus_one")
    graph = build_target(c, d, 1, past_length=7, fused_ops=("gdn", "conv_state", "rms"),
                         bindings=bindings, recurrent_dtype="float32")
    producers = {node.output.name: node for node in graph.nodes}
    for role, inputs, outputs, attrs, _ in bindings.calls:
        if role == "gdn":
            assert inputs["initial_state"].dtype == outputs["final_state"][1] == "float32"
            assert attrs["chunk_size"] == 1
        elif role == "rms":
            gamma = inputs["gamma"]
            if ".linear_attn.norm.weight" in gamma.name:
                assert gamma.name in graph.weights
            else:
                node = producers[gamma.name]
                assert node.kind == "scalar_add" and node.attrs["value"] == 1.0
        elif role == "conv_state":
            assert outputs["history"] == ((1, 48, 3), "float16")
    assert all(record["dtype"] == "float32" for record in graph.state_abi["states"] if record["role"] == "recurrent")


@pytest.mark.parametrize("kwargs,match", [
    ({"sequence_length": 0}, "positive integer"),
    ({"sequence_length": True}, "positive integer"),
    ({"past_length": -1}, "nonnegative"),
    ({"past_length": True}, "nonnegative"),
    ({"past_length": 128}, "position limit"),
    ({"fused_ops": "gdn"}, "collection"),
    ({"fused_ops": ("attention",)}, "unsupported fused"),
    ({"fused_ops": ("gdn",)}, "explicit custom bindings"),
    ({"recurrent_dtype": "float32"}, "requires fused gdn"),
    ({"recurrent_dtype": "int32"}, "float16 or float32"),
    ({"expert_dispatch": "unknown"}, "expert_dispatch"),
    ({"gdn_indexed_copy": "unknown"}, "gdn_indexed_copy"),
    ({"gdn_state_casts": "unknown"}, "gdn_state_casts"),
])
def test_invalid_contracts_fail_without_fallback(configs, kwargs, match):
    c, d = configs
    args = {"sequence_length": 3, **kwargs}
    with pytest.raises(ValueError, match=match):
        build_target(c, d, **args)


def test_conv_history_adapter_must_be_explicit(configs):
    c, d = configs
    bindings = RecordingBindings()
    bindings.conv_history_slice = "unknown"
    with pytest.raises(ValueError, match="explicitly select"):
        build_target(c, d, 3, fused_ops=("conv_state",), bindings=bindings)


@pytest.mark.parametrize("sequence,past", [(1, 3), (2, 3), (5, 0)])
def test_documented_padded_tail_keeps_state_independent_of_raw_history_order(configs, sequence, past):
    class DocumentConvSemantics(RecordingBindings):
        conv_history_slice = "padded_tail"

        def call(self, graph, role, *, inputs, outputs, attrs, name=None):
            assert role == "conv_state"
            builder = _Builder(graph)
            padded = builder.cat([inputs["history"], builder.permute(inputs["projected"], (0, 2, 1))], -1)
            # Deliberately unrelated y1: its ordering is unspecified in the MD.
            # This host contract double tests the adapter, not the vendor kernel.
            raw = graph.const(np.full(outputs["history"][0], 19, dtype=np.float16))
            return {"history": raw, "padded": padded}

    c, d = configs
    expanded = build_target(c, d, sequence, past_length=past)
    selected = build_target(c, d, sequence, past_length=past, fused_ops=("conv_state",),
                            bindings=DocumentConvSemantics())
    weights = tiny_weights({k: v.shape for k, v in expanded.weights.items()})
    rng = np.random.default_rng(73)
    payload = {v.name: rng.normal(0, .05, v.shape).astype(v.dtype) for v in expanded.inputs}
    payload["input_ids"] = np.arange(1, sequence + 1, dtype=np.int32).reshape(1, sequence)
    expected, actual = evaluate(expanded, payload, weights), evaluate(selected, payload, weights)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    assert selected.state_abi["conv_state_history_policy"] == "padded_tail"


def test_rope_positions_cannot_silently_overflow_fp16(configs):
    c, d = configs
    c = {**c, "max_position_embeddings": 131072}
    with pytest.raises(ValueError, match="finite FP16 RoPE range"):
        build_target(c, d, 1, past_length=70000)


def test_rope_casts_integer_positions_after_generating_full_sequence(configs):
    c, d = configs
    c = {**c, "max_position_embeddings": 4096}
    graph = build_target(c, d, 4, past_length=2049)
    frequency = next(node for node in graph.nodes if node.kind == "matmul")
    positions = graph.constants[frequency.inputs[1].name]
    np.testing.assert_array_equal(positions, np.array([[[2048, 2050, 2052, 2052]]], dtype=np.float16))
    np.testing.assert_array_equal(positions, np.arange(2049, 2053, dtype=np.int64).astype(np.float16).reshape(1, 1, 4))


def test_direct_stateful_builder_never_imports_torch(configs):
    code = """
import sys
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchair', 'torch_npu'}:
            raise RuntimeError('forbidden framework import: ' + fullname)
sys.meta_path.insert(0, RejectTorch())
from qwen35_moe_dflash.ge_fused.target import build_target
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.config import Qwen35DFlashConfig
c,d = configurations(tiny=True)
g = build_target(c,Qwen35DFlashConfig.from_dict(d),1)
assert g.outputs[0].shape == (1,1)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

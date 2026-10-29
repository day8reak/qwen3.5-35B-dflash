"""Diagnostic packing retains original graph identity and independent probes."""
from copy import deepcopy

import numpy as np
import pytest

from qwen35_moe_dflash.draft_ge_trace import instrument_draft_trace
from qwen35_moe_dflash.draft_layer_trace import trace_draft_graph, trace_names
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from test_ge_draft import _FP16PrimitiveOps, _fixture, _inputs


def _case():
    config, reference, weights = _fixture()
    reference.draft.set_ops(_FP16PrimitiveOps())
    reference.reference_rms_mode = "air4b-precision"
    return config, reference, weights, build_draft(config, 8)


@pytest.mark.parametrize("length", [1, 4, 8])
def test_all_trace_values_match_independent_reference_with_original_top1(length):
    config, reference, weights, graph = _case()
    before = deepcopy(graph)
    graph.outputs.append(next(node.output for node in graph.nodes if node.output.name == "draft_hidden"))
    instrumented, spec = instrument_draft_trace(graph)
    assert [row["name"] for row in spec] == list(trace_names(config))
    assert instrumented.nodes[:len(before.nodes)] == before.nodes
    assert instrumented.weights == graph.weights == before.weights
    assert graph.nodes == before.nodes
    assert [value.name for value in graph.outputs] == ["draft_top1", "draft_hidden"]
    assert [value.name for value in instrumented.outputs] == ["draft_top1", "draft_trace_values"]
    assert all(len(node.inputs) <= 16 for node in instrumented.nodes[len(before.nodes):]
               if node.kind == "concat")
    inputs = _inputs(config, length)
    top1, packed = evaluate(instrumented, inputs, weights)
    independent, report = trace_draft_graph(reference, inputs)
    np.testing.assert_array_equal(top1, independent["draft_top1"])
    assert report["status"] == "TRACED"
    offset = 0
    for row in spec:
        assert row["offset"] == offset
        actual = packed.reshape(-1)[offset:offset + row["elements"]].reshape(row["shape"])
        expected = independent[row["name"]]
        assert expected.shape == tuple(row["shape"])
        assert expected.dtype.name == row["dtype"]
        if row["dtype"] in {"bool", "int32"}:
            np.testing.assert_array_equal(actual, expected, err_msg=row["name"])
        else:
            np.testing.assert_allclose(actual, expected, atol=0.003, rtol=0.025, err_msg=row["name"])
        offset += row["elements"]
    assert packed.dtype == np.float32 and packed.shape == (1, 1, offset)


def test_subset_order_and_top1_are_preserved():
    config, _, weights, graph = _case()
    names = ["draft_top1", "positions", "context_fc", "layer_0_visible"]
    traced, spec = instrument_draft_trace(graph, names=names)
    inputs = _inputs(config, 4)
    original = evaluate(graph, inputs, weights)[0]
    actual, values = evaluate(traced, inputs, weights)
    assert [row["name"] for row in spec] == names
    np.testing.assert_array_equal(original, actual)
    np.testing.assert_array_equal(values.reshape(-1)[:3], actual.reshape(-1))


@pytest.mark.parametrize("names", [[], ["no_such_value"], ["draft_hidden", "draft_hidden"], "draft_hidden"])
def test_rejects_invalid_selection_without_mutating_graph(names):
    _, _, _, graph = _case()
    before = deepcopy(graph)
    with pytest.raises(ValueError):
        instrument_draft_trace(graph, names=names)
    assert graph.nodes == before.nodes and graph.outputs == before.outputs


def test_rejects_ambiguous_weight_producer_and_double_instrumentation():
    _, _, _, graph = _case()
    traced, _ = instrument_draft_trace(graph, names=["context_fc"])
    with pytest.raises(ValueError, match="already instrumented"):
        instrument_draft_trace(traced)
    source = next(node for node in graph.nodes if node.kind == "matmul" and
                  any(value.name == "draft.fc.weight" for value in node.inputs))
    graph.op("matmul", *source.inputs, shape=source.output.shape, attrs=source.attrs, name="duplicate_fc")
    with pytest.raises(ValueError, match="expected one producer"):
        instrument_draft_trace(graph)


def test_rejects_large_trace_before_appending_nodes(monkeypatch):
    _, _, _, graph = _case()
    before = deepcopy(graph)
    monkeypatch.setattr("qwen35_moe_dflash.draft_ge_trace.MAX_TRACE_BYTES", 4)
    with pytest.raises(ValueError, match="64 MiB"):
        instrument_draft_trace(graph)
    assert graph.nodes == before.nodes

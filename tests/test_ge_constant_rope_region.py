"""The static RoPE region folds completely or retains its original graph."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from qwen35_moe_dflash.ge_constants import fold_constants
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph


def _region(heads=2, sequence=8, *, left_kind="constant", concat_axis=2):
    graph = Graph("unnamed_static_rope_region")
    final_shape = (1, 1, sequence, 2 * heads)
    graph.input("context", final_shape)
    inverse = np.resize(np.array([.7, .01], dtype="<f2"), heads).reshape(1, heads, 1)
    if left_kind == "constant":
        left = graph.const(inverse, name="inverse_buffer")
    elif left_kind == "weight":
        left = graph.weight("external_inverse", inverse.shape)
    else:
        left = graph.input("dynamic_inverse", inverse.shape)
    right = graph.const(np.arange(sequence, dtype="<f2").reshape(1, 1, sequence),
                        name="position_buffer")
    nodes = {}

    def op(key, kind, *inputs, shape, attrs=None):
        value = graph.op(kind, *inputs, shape=shape, attrs=attrs, name=key)
        nodes[key] = graph.nodes[-1]
        return value

    product = op("frequency_product", "matmul", left, right, shape=(1, heads, sequence),
                 attrs={"transpose_a": False, "transpose_b": False})
    transposed = op("frequency_order", "transpose", product, shape=(1, sequence, heads),
                    attrs={"perm": [0, 2, 1]})
    angles = op("duplicated_angles", "concat", transposed, transposed,
                shape=(1, sequence, 2 * heads), attrs={"axis": concat_axis})
    cosine = op("cosine_values", "cos", angles, shape=angles.shape)
    cosine = op("cosine_ready", "reshape", cosine, shape=final_shape,
                attrs={"shape": list(final_shape)})
    sine = op("sine_values", "sin", angles, shape=angles.shape)
    sine = op("sine_ready", "reshape", sine, shape=final_shape,
              attrs={"shape": list(final_shape)})
    graph.outputs = [cosine, sine]
    return graph, nodes


def _live(graph):
    live = {value.name for value in graph.outputs}
    for node in reversed(graph.nodes):
        if node.output.name in live:
            live.update(value.name for value in node.inputs)
    return live


def _fold(graph, **kwargs):
    return fold_constants(graph, {}, _live(graph), **kwargs)


def _replace_output(graph, node, **changes):
    previous = node.output
    node.output = replace(previous, **changes)
    graph._values[previous.name] = node.output
    for consumer in graph.nodes:
        consumer.inputs = tuple(node.output if value is previous else value
                                for value in consumer.inputs)
    graph.outputs = [node.output if value is previous else value for value in graph.outputs]


def _assert_region_unfolded(graph, nodes):
    try:
        result = _fold(graph)
    except ValueError:
        # Invalid metadata may fail validation; it must never produce a
        # partially accepted replacement region.
        return
    assert not set(nodes).intersection(result)


@pytest.mark.parametrize("heads,sequence", [(2, 8), (32, 8), (2, 13)])
@pytest.mark.parametrize("concat_axis", [2, -1])
def test_complete_region_matches_each_original_fp16_boundary(heads, sequence, concat_axis):
    graph, nodes = _region(heads, sequence, concat_axis=concat_axis)
    original = deepcopy(graph)
    raw_constants = {key: value.tobytes() for key, value in graph.constants.items()}
    folded = _fold(graph)
    assert set(folded) == set(nodes)
    original.outputs = [node.output for node in original.nodes]
    expected = evaluate(original, {"context": np.zeros(graph.inputs[0].shape, dtype="<f2")})
    for (name, node), reference in zip(nodes.items(), expected):
        actual = folded[name]
        assert actual.shape == node.output.shape and actual.dtype == np.dtype("<f2")
        assert actual.nbytes == 2 * np.prod(node.output.shape)
        assert actual.flags.c_contiguous
        assert actual.tobytes() == reference.tobytes()
        assert not any(np.shares_memory(actual, source) for source in graph.constants.values())
    assert {key: value.tobytes() for key, value in graph.constants.items()} == raw_constants
    assert graph.nodes == _region(heads, sequence, concat_axis=concat_axis)[0].nodes


def test_trigonometry_uses_rounded_fp16_angles_not_the_unrounded_product():
    graph, _ = _region()
    folded = _fold(graph)
    a = graph.constants["inverse_buffer"].astype(np.float32)
    b = graph.constants["position_buffer"].astype(np.float32)
    unrounded = np.matmul(a, b).transpose(0, 2, 1)
    unrounded = np.concatenate([unrounded, unrounded], axis=-1)
    # FP16(.7) * FP16(7) is 4.9013671875 in FP32 and 4.90234375
    # after the source MatMul's FP16 boundary. Both trigs distinguish them.
    for name, operation in (("cosine_ready", np.cos), ("sine_ready", np.sin)):
        skipped_boundary = operation(unrounded).astype("<f2").reshape(1, 1, 8, 4)
        assert folded[name].tobytes() != skipped_boundary.tobytes()


class _PayloadForbidden(dict):
    def __getitem__(self, key):
        raise AssertionError("region budget must be checked before reading any constant payload")


@pytest.mark.parametrize("budget", [{"max_tensor_bytes": 63}, {"max_total_bytes": 383}])
def test_incomplete_region_budget_retains_all_seven_nodes_before_payload_access(budget):
    graph, _ = _region()
    graph.constants = _PayloadForbidden(graph.constants)
    assert _fold(graph, **budget) == {}


def test_exact_region_budget_counts_all_seven_outputs_once():
    graph, nodes = _region()
    folded = _fold(graph, max_tensor_bytes=64, max_total_bytes=384)
    assert set(folded) == set(nodes)
    assert sum(array.nbytes for array in folded.values()) == 384
    assert folded["cosine_ready"].nbytes == folded["sine_ready"].nbytes == 64


@pytest.mark.parametrize("fault", [
    "duplicate-perm", "bool-perm", "short-perm", "transpose-attribute", "transpose-shape",
    "concat-axis", "concat-bool-axis", "concat-shape", "concat-input",
    "trig-attribute", "trig-shape", "reshape-attribute", "reshape-shape", "reshape-dtype",
])
def test_unsupported_region_metadata_never_commits_a_prefix(fault):
    graph, nodes = _region()
    if fault == "duplicate-perm":
        nodes["frequency_order"].attrs["perm"] = [0, 0, 1]
    elif fault == "bool-perm":
        nodes["frequency_order"].attrs["perm"] = [False, 2, 1]
    elif fault == "short-perm":
        nodes["frequency_order"].attrs["perm"] = [0, 2]
    elif fault == "transpose-attribute":
        nodes["frequency_order"].attrs["unknown"] = True
    elif fault == "transpose-shape":
        _replace_output(graph, nodes["frequency_order"], shape=(1, 2, 8))
    elif fault == "concat-axis":
        nodes["duplicated_angles"].attrs["axis"] = 1
    elif fault == "concat-bool-axis":
        nodes["duplicated_angles"].attrs["axis"] = True
    elif fault == "concat-shape":
        _replace_output(graph, nodes["duplicated_angles"], shape=(1, 8, 5))
    elif fault == "concat-input":
        duplicate = nodes["frequency_order"].output
        nodes["duplicated_angles"].inputs = (duplicate, graph.inputs[0])
    elif fault == "trig-attribute":
        nodes["cosine_values"].attrs["unknown"] = 1
    elif fault == "trig-shape":
        _replace_output(graph, nodes["sine_values"], shape=(1, 4, 8))
    elif fault == "reshape-attribute":
        nodes["sine_ready"].attrs["unknown"] = 1
    elif fault == "reshape-shape":
        nodes["sine_ready"].attrs["shape"] = [1, 8, 4]
    else:
        _replace_output(graph, nodes["cosine_ready"], dtype="int32")
    _assert_region_unfolded(graph, nodes)


@pytest.mark.parametrize("intermediate", [
    "frequency_product", "frequency_order", "duplicated_angles", "cosine_values", "sine_values",
])
@pytest.mark.parametrize("fanout", ["graph-output", "dynamic", "static"])
def test_extra_live_intermediate_uses_retain_the_whole_region(intermediate, fanout):
    graph, nodes = _region()
    value = nodes[intermediate].output
    if fanout == "graph-output":
        graph.outputs.append(value)
    elif fanout == "static":
        graph.outputs.append(graph.op("neg", value, shape=value.shape, name="extra_static_use"))
    else:
        dynamic = graph.input("extra_dynamic_input", value.shape)
        graph.outputs.append(graph.op("add", value, dynamic, shape=value.shape, name="extra_dynamic_use"))
    assert _fold(graph) == {}


def test_dynamic_endpoint_fanout_does_not_leave_static_operations_in_the_region():
    graph, nodes = _region()
    cosine, sine = graph.outputs
    graph.outputs = [graph.op("mul", cosine, graph.inputs[0], shape=cosine.shape, name="rotary_cosine"),
                     graph.op("mul", sine, graph.inputs[0], shape=sine.shape, name="rotary_sine")]
    assert set(_fold(graph)) == set(nodes)


def test_static_endpoint_consumer_retains_the_whole_region():
    graph, _ = _region()
    cosine, sine = graph.outputs
    graph.outputs = [graph.op("neg", cosine, shape=cosine.shape, name="static_cosine_tail"), sine]
    assert _fold(graph) == {}


class _UnreadableWeight:
    def __fspath__(self):
        raise AssertionError("RoPE region must not access external weight payloads")


@pytest.mark.parametrize("left_kind", ["weight", "dynamic"])
def test_nonconstant_root_inputs_do_not_trigger_region_or_external_reads(left_kind):
    graph, _ = _region(left_kind=left_kind)
    assert fold_constants(graph, {"external_inverse": _UnreadableWeight()}, _live(graph)) == {}


@pytest.mark.parametrize("source", ["inverse_buffer", "position_buffer"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_operand_does_not_commit_any_region_node(source, value):
    graph, _ = _region()
    graph.constants[source].flat[0] = value
    assert _fold(graph) == {}


def test_nonfinite_fp16_product_does_not_commit_any_region_node():
    graph, _ = _region()
    graph.constants["inverse_buffer"].fill(np.finfo(np.float16).max)
    graph.constants["position_buffer"].fill(2)
    assert _fold(graph) == {}


def test_payload_descriptor_disagreement_is_rejected_before_any_region_is_returned():
    graph, nodes = _region()
    graph.constants["inverse_buffer"] = np.ones((1, 3, 1), dtype="<f2")
    _assert_region_unfolded(graph, nodes)

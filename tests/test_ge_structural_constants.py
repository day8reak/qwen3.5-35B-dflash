"""Small static structural folding copies bytes; it is not a device kernel test."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.ge_constants import fold_constants
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_target import _Builder


def _slice(graph, source, starts, sizes, *, shape=None):
    return graph.op("slice", source, shape=sizes if shape is None else shape,
                    attrs={"starts": starts, "sizes": sizes})


def _reshape(graph, source, shape, *, output_shape=None):
    return graph.op("reshape", source, shape=shape if output_shape is None else output_shape,
                    attrs={"shape": shape})


def _bits():
    # Include signed zeros, infinities, quiet/signalling NaN payloads, subnormals.
    return np.array([0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x7d01, 0xfe31, 1,
                     0x8001, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0xfbff, 0x03ff,
                     0x0400, 0x8400, 0x7e44, 0x7d32, 0x8000], dtype="<u2").reshape(4, 5)


@pytest.mark.parametrize("axis", [0, -2, 1, -1])
def test_builder_slice_normalizes_axes_and_preserves_all_fp16_bits(axis):
    graph = Graph("structural_bits")
    original = _bits()
    source = graph.const(original.view("<f2"))
    sliced = _Builder(graph).slice(source, axis, 1, 2)
    reshaped = _reshape(graph, sliced, (int(np.prod(sliced.shape)),))
    folded = fold_constants(graph, {}, set(graph._values))
    slices = [slice(None)] * 2
    slices[axis % 2] = slice(1, 3)
    expected = original[tuple(slices)]
    assert folded[sliced.name].dtype == np.dtype("<f2")
    assert folded[sliced.name].tobytes() == expected.tobytes()
    assert folded[reshaped.name].tobytes() == expected.reshape(-1).tobytes()
    assert graph.constants[source.name].tobytes() == original.tobytes()


def test_noncontiguous_multi_axis_slice_then_reshape_is_bit_exact():
    graph = Graph("multi_axis")
    source = graph.const(_bits().view("<f2"))
    sliced = _slice(graph, source, [1, 1], [2, 3])
    reshaped = _reshape(graph, sliced, (3, 2))
    folded = fold_constants(graph, {}, set(graph._values))
    expected = _bits()[1:3, 1:4]
    assert folded[sliced.name].shape == (2, 3)
    assert folded[reshaped.name].shape == (3, 2)
    assert folded[reshaped.name].tobytes() == expected.reshape(3, 2).tobytes()


def test_slice_minus_one_size_uses_remaining_extent():
    graph = Graph("remaining_extent")
    source = graph.const(_bits().view("<f2"))
    output = _slice(graph, source, [1, 2], [-1, -1], shape=(3, 3))
    folded = fold_constants(graph, {}, {output.name})
    assert folded[output.name].tobytes() == _bits()[1:, 2:].tobytes()


@pytest.mark.parametrize("starts,sizes,shape", [
    ([0], [2, 3], (2, 3)),
    ([0, 0], [2], (2, 3)),
    ([-1, 0], [2, 3], (2, 3)),
    ([0, 0], [-2, 3], (2, 3)),
    ([3, 0], [2, 3], (2, 3)),
    ([0, 4], [2, 2], (2, 2)),
    ([0, 0], [2, 3], (3, 2)),
    ([False, 0], [2, 3], (2, 3)),
    ([0.0, 0], [2, 3], (2, 3)),
])
def test_static_slice_rejects_invalid_descriptors(starts, sizes, shape):
    graph = Graph("invalid_slice")
    source = graph.const(np.zeros((4, 5), dtype="<f2"))
    output = _slice(graph, source, starts, sizes, shape=shape)
    with pytest.raises(ValueError):
        fold_constants(graph, {}, {output.name})


@pytest.mark.parametrize("shape,output_shape", [
    ([3, 3], (3, 3)),
    ([2, 10], (4, 5)),
    ([-1, 5], (4, 5)),
    ([True, 20], (1, 20)),
    ([4.0, 5], (4, 5)),
])
def test_static_reshape_rejects_invalid_descriptors(shape, output_shape):
    graph = Graph("invalid_reshape")
    source = graph.const(np.zeros((4, 5), dtype="<f2"))
    output = _reshape(graph, source, shape, output_shape=output_shape)
    with pytest.raises(ValueError):
        fold_constants(graph, {}, {output.name})


def test_static_input_bytes_must_match_declared_descriptor():
    graph = Graph("bad_storage")
    source = graph.const(np.zeros((4, 5), dtype="<f2"))
    output = _slice(graph, source, [0, 0], [1, 5])
    graph.constants[source.name] = np.zeros((4, 5), dtype="<f4")
    with pytest.raises(ValueError):
        fold_constants(graph, {}, {output.name})


def test_small_slice_of_large_original_weight_does_not_access_path():
    class UnreadablePath:
        def __fspath__(self):
            raise AssertionError("large source weight path must not be accessed")

    graph = Graph("large_weight")
    source = graph.weight("huge_experts", (32769,))
    sliced = _slice(graph, source, [0], [2])
    reshaped = _reshape(graph, sliced, (1, 2))
    folded = fold_constants(graph, {source.name: UnreadablePath()}, set(graph._values))
    assert sliced.name not in folded and reshaped.name not in folded


def test_structural_budget_counts_source_read_and_each_retained_output(tmp_path, monkeypatch):
    graph = Graph("structural_budget")
    source = graph.weight("weight", (2, 4))
    sliced = _slice(graph, source, [1, 0], [1, 4])
    reshaped = _reshape(graph, sliced, (4,))
    path = tmp_path / "weight.bin"
    original = _bits().reshape(-1)[:8].tobytes()
    path.write_bytes(original)
    calls = []
    original_open = Path.open

    def tracking_open(self, *args, **kwargs):
        calls.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    assert fold_constants(graph, {source.name: path}, set(graph._values), max_total_bytes=23) == {}
    assert calls == []
    folded = fold_constants(graph, {source.name: path}, set(graph._values), max_total_bytes=24)
    assert set(folded) == {sliced.name} and calls == [path]
    calls.clear()
    folded = fold_constants(graph, {source.name: path}, set(graph._values), max_total_bytes=32)
    assert set(folded) == {sliced.name, reshaped.name} and calls == [path]
    assert folded[reshaped.name].tobytes() == original[8:]
    with original_open(path, "rb") as stream:
        assert stream.read() == original


def test_two_slices_reuse_small_weight_read(tmp_path, monkeypatch):
    graph = Graph("shared_source")
    source = graph.weight("weight", (2, 4))
    first = _slice(graph, source, [0, 0], [1, 4])
    second = _slice(graph, source, [1, 0], [1, 4])
    path = tmp_path / "weight.bin"
    path.write_bytes(_bits().reshape(-1)[:8].tobytes())
    calls = []
    original_open = Path.open

    def tracking_open(self, *args, **kwargs):
        calls.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    folded = fold_constants(graph, {source.name: path}, set(graph._values), max_total_bytes=32)
    assert set(folded) == {first.name, second.name} and calls == [path]


@pytest.fixture(scope="module")
def tiny_structural_case(tmp_path_factory):
    from qwen35_moe_dflash.ge_export import build_pair, configurations, weight_shapes
    from qwen35_moe_dflash.ge_weights import tiny_weights

    target, draft = build_pair(*configurations(tiny=True), 8)
    weights = tiny_weights(weight_shapes([target, draft]))
    folder = tmp_path_factory.mktemp("tiny_structural_weights")
    paths = {}
    for index, (key, value) in enumerate(weights.items()):
        paths[key] = folder / f"{index}.bin"
        paths[key].write_bytes(value.tobytes())
    folded = fold_constants(target, paths, set(target._values))
    lowered = deepcopy(target)
    lowered.constants.update(folded)
    lowered.nodes = [node for node in lowered.nodes if node.output.name not in folded]
    return target, lowered, weights, paths, folded


def test_all_tiny_static_expert_slices_and_conv_views_are_folded(tiny_structural_case):
    target, lowered, weights, paths, folded = tiny_structural_case
    expert_slices = [node for node in target.nodes
                     if node.kind == "slice" and node.inputs[0].name in target.weights]
    conv_views = [node for node in target.nodes if node.kind == "reshape"
                  and node.inputs[0].name.endswith(".conv1d.weight")]
    assert len(expert_slices) == 32 and len(conv_views) == 3
    for node in expert_slices + conv_views:
        assert node.output.name in folded
        source = weights[node.inputs[0].name]
        if node.kind == "slice":
            expected = source[tuple(slice(start, start + size) for start, size
                                    in zip(node.attrs["starts"], node.attrs["sizes"]))]
        else:
            expected = source.reshape(node.output.shape)
        assert folded[node.output.name].tobytes() == expected.tobytes()
    assert not any(node.kind == "slice" and node.inputs[0].name in target.weights
                   for node in lowered.nodes)
    for name, path in paths.items():
        assert path.read_bytes() == weights[name].tobytes()


@pytest.mark.parametrize("tokens", [[1], [1, 2, 3], [7, 3, 26, 8, 16, 42, 29, 31]])
def test_tiny_whole_target_outputs_are_unchanged(tiny_structural_case, tokens):
    from qwen35_moe_dflash.ge_eval import evaluate

    target, lowered, weights, _, _ = tiny_structural_case
    ids = np.zeros((1, 8), dtype="<i4")
    mask = ids.copy()
    ids[0, :len(tokens)] = tokens
    mask[0, :len(tokens)] = 1
    inputs = {"input_ids": ids, "attention_mask": mask}
    for expected, actual in zip(evaluate(target, inputs, weights), evaluate(lowered, inputs, weights)):
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("external", [False, True])
def test_recipe_prunes_raw_weight_and_static_slice_reshape_chain(tmp_path, monkeypatch, external):
    from qwen35_moe_dflash.ge_cpp import emit_cpp
    from qwen35_moe_dflash import ge_recipe

    graph = Graph("structural_recipe")
    inp = graph.input("input", (3, 2))
    source = graph.weight("weight", (4, 5))
    sliced = _slice(graph, source, [1, 1], [2, 3])
    reshaped = _reshape(graph, sliced, (3, 2))
    graph.outputs = [graph.op("add", inp, reshaped, shape=inp.shape)]
    raw = _bits().tobytes()
    path = tmp_path / "weight.bin"
    path.write_bytes(raw)
    captured = []
    original_emit = ge_recipe.emit_recipe_cpp

    def capture(records, *args, **kwargs):
        captured.extend(records)
        return original_emit(records, *args, **kwargs)

    monkeypatch.setattr(ge_recipe, "emit_recipe_cpp", capture)
    emit_cpp(graph, {source.name: path}, external_weights=external, recipe_dir=tmp_path / "recipe")
    assert [record["type"] for record in captured] == ["Data", "Const", "Add"]
    assert captured[1]["name"] == reshaped.name
    assert captured[1]["constant"]["data"] == _bits()[1:3, 1:4].reshape(3, 2).tobytes()
    assert path.read_bytes() == raw

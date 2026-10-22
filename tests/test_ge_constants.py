"""Small constant folding preserves FP16 boundaries and bounded weight access."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.ge_constants import fold_constants
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph


def unary(g, kind, source, **attrs):
    return g.op(kind, source, shape=source.shape, attrs=attrs)


def test_small_raw_weight_folding_matches_oracle_and_preserves_file(tmp_path):
    g = Graph("constant_math")
    inp = g.input("input", (4,))
    weight = g.weight("weight", (4,))
    shifted = unary(g, "scalar_add", weight, value=1.0)
    scaled = unary(g, "scalar_mul", shifted, value=.5)
    exponent = unary(g, "exp", scaled)
    negative = unary(g, "neg", exponent)
    g.outputs = [shifted, scaled, exponent, negative]
    array = np.array([-.5, .0006, .75, 2.25], dtype="<f2")
    path = tmp_path / "weight.bin"
    path.write_bytes(array.tobytes())
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    folded = fold_constants(g, {"weight": path}, set(g._values))
    expected = evaluate(g, {inp.name: np.zeros(4, dtype="<f2")}, {"weight": array})
    for output, actual in zip(g.outputs, expected):
        np.testing.assert_array_equal(folded[output.name], actual)
        assert folded[output.name].dtype == np.dtype("<f2")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert len(g.nodes) == 4 and not g.constants


def test_rounds_at_every_fp16_node_boundary():
    g = Graph("rounding")
    original = g.const(np.array([.0006], dtype="<f2"))
    shifted = unary(g, "scalar_add", original, value=1.0)
    restored = unary(g, "scalar_add", shifted, value=-1.0)
    folded = fold_constants(g, {}, set(g._values))
    assert folded[restored.name].item() == np.float16(2**-10)
    assert folded[restored.name].item() != g.constants[original.name].item()


def test_skips_dynamic_nonlive_and_unsupported_nodes_without_weight_access():
    g = Graph("skip")
    dynamic = g.input("input", (4,))
    unary(g, "exp", dynamic)
    weight = g.weight("weight", (4,))
    unary(g, "neg", weight)  # deliberately not live
    unsupported = g.op("reshape", weight, shape=(2, 2), attrs={"shape": [2, 2]})
    after = unary(g, "neg", unsupported)
    assert fold_constants(g, {}, {dynamic.name, after.name, unsupported.name}) == {}


def test_large_tensors_are_skipped_before_path_access():
    class UnreadablePath:
        def __fspath__(self):
            raise AssertionError("large raw weight path was accessed")

    g = Graph("large")
    weight = g.weight("huge", (32769,))
    output = unary(g, "scalar_add", weight, value=1.0)
    assert fold_constants(g, {"huge": UnreadablePath()}, {output.name}) == {}


@pytest.mark.parametrize("payload", [b"\0" * 6, b"\0" * 10])
def test_eligible_weight_requires_exact_raw_byte_count(tmp_path, payload):
    g = Graph("bytes")
    weight = g.weight("weight", (4,))
    output = unary(g, "neg", weight)
    path = tmp_path / "weight.bin"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="byte count must be 8"):
        fold_constants(g, {"weight": path}, {output.name})


def test_budget_counts_cached_weight_once_and_each_retained_output(tmp_path, monkeypatch):
    g = Graph("budget")
    weight = g.weight("weight", (4,))
    first = unary(g, "neg", weight)
    second = unary(g, "scalar_add", weight, value=1.0)
    third = unary(g, "exp", second)
    path = tmp_path / "weight.bin"
    path.write_bytes(np.ones(4, dtype="<f2").tobytes())
    calls = []
    original_open = Path.open
    def tracking_open(self, *args, **kwargs):
        calls.append(self)
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tracking_open)
    assert fold_constants(g, {"weight": path}, set(g._values), max_total_bytes=15) == {}
    assert calls == []
    folded = fold_constants(g, {"weight": path}, set(g._values), max_total_bytes=24)
    assert set(folded) == {first.name, second.name}
    assert third.name not in folded and calls == [path]


def test_existing_graph_constants_only_charge_folded_output_budget():
    g = Graph("constant_budget")
    source = g.const(np.ones(4, dtype="<f2"))
    first = unary(g, "neg", source)
    second = unary(g, "neg", first)
    folded = fold_constants(g, {}, set(g._values), max_total_bytes=8)
    assert set(folded) == {first.name}
    assert second.name not in folded


@pytest.mark.parametrize("scalar", [np.nan, np.inf, -np.inf, 1e100, "invalid"])
def test_scalar_attributes_must_be_representable_finite_fp32(scalar):
    g = Graph("scalar")
    source = g.const(np.ones(4, dtype="<f2"))
    out = unary(g, "scalar_mul", source, value=scalar)
    with pytest.raises(ValueError, match="finite FP32"):
        fold_constants(g, {}, {out.name})


@pytest.mark.parametrize("argument,value", [("max_tensor_bytes", -1), ("max_total_bytes", True),
                                             ("max_total_bytes", 1.5)])
def test_rejects_invalid_budgets(argument, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        fold_constants(Graph("budget"), {}, set(), **{argument: value})


@pytest.fixture(scope="module")
def tiny_folded_case(tmp_path_factory):
    from copy import deepcopy
    from qwen35_moe_dflash.ge_export import build_pair, configurations, weight_shapes
    from qwen35_moe_dflash.ge_weights import tiny_weights

    target, draft = build_pair(*configurations(tiny=True), 8)
    weights = tiny_weights(weight_shapes([target, draft]))
    folder = tmp_path_factory.mktemp("fold-tiny-weights")
    paths = {}
    for index, (key, array) in enumerate(weights.items()):
        paths[key] = folder / f"{index}.bin"
        paths[key].write_bytes(array.tobytes())
    folded = fold_constants(target, paths, set(target._values))
    assert len(folded) == 17
    assert sum(value.nbytes for value in folded.values()) == 344
    assert fold_constants(draft, paths, set(draft._values)) == {}
    lowered = deepcopy(target)
    lowered.constants.update(folded)
    lowered.nodes = [node for node in lowered.nodes if node.output.name not in folded]
    return target, lowered, weights


@pytest.mark.parametrize("tokens", [[1], [1, 2, 3], [7, 3, 26, 8, 16, 42, 29, 31]])
def test_tiny_target_outputs_unchanged_after_constant_folding(tiny_folded_case, tokens):
    target, lowered, weights = tiny_folded_case
    ids = np.zeros((1, 8), dtype=np.int32)
    mask = ids.copy()
    ids[0, :len(tokens)] = tokens
    mask[0, :len(tokens)] = 1
    inputs = {"input_ids": ids, "attention_mask": mask}
    expected = evaluate(target, inputs, weights)
    actual = evaluate(lowered, inputs, weights)
    for left, right in zip(expected, actual):
        assert left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize("external", [False, True])
def test_recipe_contains_folded_const_and_prunes_weight_arithmetic(tmp_path, monkeypatch, external):
    from qwen35_moe_dflash.ge_cpp import emit_cpp
    from qwen35_moe_dflash import ge_recipe

    graph = Graph("folded_recipe")
    x = graph.input("input", (4,))
    weight = graph.weight("weight", (4,))
    shifted = unary(graph, "scalar_add", weight, value=1.0)
    graph.outputs = [graph.op("mul", x, shifted, shape=x.shape)]
    raw = np.array([-.5, .0006, .75, 2.25], dtype="<f2").tobytes()
    path = tmp_path / "weight.bin"
    path.write_bytes(raw)
    captured = []
    original_emit = ge_recipe.emit_recipe_cpp
    def capture(records, *args, **kwargs):
        captured.extend(records)
        return original_emit(records, *args, **kwargs)
    monkeypatch.setattr(ge_recipe, "emit_recipe_cpp", capture)
    folder = tmp_path / "recipe"
    emit_cpp(graph, {"weight": path}, external_weights=external, recipe_dir=folder)
    assert [record["type"] for record in captured] == ["Data", "Const", "Mul"]
    assert captured[1]["name"] == shifted.name
    expected = (np.frombuffer(raw, dtype="<f2").astype(np.float32) + np.float32(1)).astype("<f2")
    assert captured[1]["constant"]["data"] == expected.tobytes()
    assert (folder / "graph.recipe").read_bytes().startswith(b"Q35GE001")
    assert (folder / "constants.bin").read_bytes() == expected.tobytes()
    assert path.read_bytes() == raw

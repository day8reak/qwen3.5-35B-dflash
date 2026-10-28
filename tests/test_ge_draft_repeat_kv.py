"""Draft KV head copies remain byte-exact without rank-five graph carriers."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import _DraftBuilder, build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations
from reference.ge_torch import evaluate_torch
from test_ge_draft import _fixture, _inputs, _named
from test_ge_select_lowering import RecipeReader


class _LegacyDraftBuilder(_DraftBuilder):
    """Frozen pre-change broadcast definition for the complete graph oracle."""

    def repeat_kv(self, x):
        c, g = self.c, self.g
        if c.num_key_value_groups == 1:
            return x
        x = self.reshape(x, (1, c.num_key_value_heads, 1, self.k, c.head_dim))
        expanded = (1, c.num_key_value_heads, c.num_key_value_groups, self.k, c.head_dim)
        x = g.op("broadcast_to", x, shape=expanded, attrs={"shape": list(expanded)})
        return self.reshape(x, (1, c.num_attention_heads, self.k, c.head_dim))


def _head_graph(heads, groups):
    config = Qwen35DFlashConfig.from_dict(configurations(tiny=True)[1])
    config = replace(config, num_key_value_heads=heads, num_attention_heads=heads * groups, head_dim=4)
    builder = _DraftBuilder(config, 8)
    source = builder.g.input("kv", (1, heads, builder.k, config.head_dim), "float16")
    builder.g.outputs = [builder.repeat_kv(source)]
    return builder.g.validate()


@pytest.mark.parametrize("heads,groups", [(1, 1), (1, 4), (2, 2), (3, 3), (8, 4)])
def test_head_order_and_fp16_bits_are_exact_copies(heads, groups):
    graph = _head_graph(heads, groups)
    # Include both zeros, infinities, distinct quiet/signalling NaN payloads,
    # subnormals and ordinary finite values; each source head differs.
    patterns = np.array([0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0xfe12, 0x7c01, 0xfc03,
                         0x0001, 0x8001, 0x03ff, 0x83ff, 0x3c00, 0xbc00, 0x7bff, 0xfbff], np.uint16)
    bits = np.stack([np.resize(np.roll(patterns, head), (12, 4)) for head in range(heads)])[None]
    values = bits.view(np.float16)
    expected = np.repeat(bits, groups, axis=1)
    for evaluator in (evaluate, evaluate_torch):
        actual, = evaluator(graph, {"kv": values})
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint16), expected)
    if groups == 1:
        assert graph.outputs[0] is graph.inputs[0]
        assert not graph.nodes and not graph.constants
    else:
        node, = graph.nodes
        assert node.kind == "gather" and node.attrs == {"axis": 1}
        assert node.inputs[1].dtype == "int32"
        assert graph.constants[node.inputs[1].name].tolist() == [h for h in range(heads) for _ in range(groups)]
    assert max(len(value.shape) for value in graph._values.values()) <= 4


def _constant(record):
    path, offset, size = record["constant"]
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        return np.frombuffer(stream.read(size), dtype=np.int32)


def test_serialized_gather_uses_int32_head_indices_without_cast_or_rank5(tmp_path):
    graph = _head_graph(3, 2)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    gather, = [record for record in records if record["type"] == "GatherV2"]
    assert {record["type"] for record in records} == {"Data", "Const", "GatherV2"}
    source, indices, axis = gather["inputs"]
    assert source["shape"] == (1, 3, 12, 4) and source["dtype"] == "float16"
    assert indices["shape"] == (6,) and indices["dtype"] == axis["dtype"] == "int32"
    assert _constant(records[indices["src"]]).tolist() == [0, 0, 1, 1, 2, 2]
    assert _constant(records[axis["src"]]).tolist() == [1]
    assert gather["outputs"][0]["shape"] == (1, 6, 12, 4)
    assert max(len(port["shape"]) for record in records for port in record["inputs"] + record["outputs"]) <= 4


@pytest.fixture(scope="module")
def draft_case():
    config, _, weights = _fixture()
    current, legacy = build_draft(config, 8), _LegacyDraftBuilder(config, 8).build().validate()
    for graph in (current, legacy):
        graph.outputs += _named(graph, "draft_hidden", "draft_logits")
    return config, weights, current, legacy


@pytest.mark.parametrize("length", [1, 4, 8])
def test_full_draft_outputs_match_previous_graph_bitwise(draft_case, length):
    config, weights, current, legacy = draft_case
    assert current.weights == legacy.weights
    for evaluator in (evaluate, evaluate_torch):
        actual = evaluator(current, _inputs(config, length), weights)
        expected = evaluator(legacy, _inputs(config, length), weights)
        for got, want in zip(actual, expected):
            got, want = np.asarray(got), np.asarray(want)
            if got.dtype == np.float16:
                assert np.isfinite(got).all()
                np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))
            else:
                np.testing.assert_array_equal(got, want)


def test_full_draft_recipe_has_no_rank5_carriers(draft_case, tmp_path):
    _, weights, graph, legacy = draft_case
    paths = {}
    for index, (name, array) in enumerate(weights.items()):
        paths[name] = tmp_path / f"weight-{index}.bin"
        paths[name].write_bytes(array.tobytes())
    emit_cpp(graph, paths, recipe_dir=tmp_path / "recipe")
    records, _, _ = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert any(len(value.shape) == 5 for value in legacy._values.values())
    assert max(len(value.shape) for value in graph._values.values()) <= 4
    assert max(len(port["shape"]) for record in records for port in record["inputs"] + record["outputs"]) <= 4
    head_copies = [record for record in records if record["type"] == "GatherV2"
                   and len(record["inputs"][0]["shape"]) == 4
                   and record["inputs"][0]["dtype"] == "float16"]
    assert len(head_copies) == 4  # K and V for each of the two attention layers.
    assert all(record["inputs"][1]["dtype"] == "int32" for record in head_copies)

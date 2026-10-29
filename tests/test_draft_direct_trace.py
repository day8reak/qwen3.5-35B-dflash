"""Direct observations expose original values without added snapshot arithmetic."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.draft_direct_trace import (
    DEFAULT_DIRECT_NAMES, instrument_draft_direct_outputs,
)
from qwen35_moe_dflash.draft_layer_trace import trace_draft_graph
from qwen35_moe_dflash.draft_standalone import _graph, export_draft, prepare_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from test_draft_ge_trace import _case
from test_draft_standalone import fake_ge
from test_ge_draft import _inputs


def test_complete_default_direct_outputs_preserve_original_nodes_and_native_types():
    _, _, original = _graph(64)
    before = deepcopy(original)
    direct, spec = instrument_draft_direct_outputs(original)
    assert original.nodes == direct.nodes == before.nodes
    assert original.outputs == before.outputs
    assert direct.weights == before.weights and set(direct.constants) == set(before.constants)
    assert [row["name"] for row in spec] == ["draft_top1", "draft_hidden", *DEFAULT_DIRECT_NAMES]
    assert len(direct.outputs) == 16
    assert [value.name for value in direct.outputs[:2]] == ["draft_top1", "draft_hidden"]
    indexed = {row["name"]: row for row in spec}
    assert indexed["layer_0_output"]["source"] == indexed["layer_1_input"]["source"]
    assert indexed["layer_0_output"]["output_index"] == indexed["layer_1_input"]["output_index"]
    assert indexed["context_fc"]["dtype"] == "float16"
    assert indexed["layer_0_scores_fp32"]["dtype"] == "float32"
    for row in spec:
        value = direct.outputs[row["output_index"]]
        assert (row["source"], row["shape"], row["dtype"]) == (value.name, list(value.shape), value.dtype)


@pytest.mark.parametrize("length", [1, 4, 8])
def test_direct_outputs_match_original_values_and_independent_reference(length):
    config, reference, weights, graph = _case()
    direct, spec = instrument_draft_direct_outputs(graph)
    inputs = _inputs(config, length)
    values = evaluate(direct, inputs, weights)
    independent, _ = trace_draft_graph(reference, inputs, names=[row["name"] for row in spec])
    assert len(values) == 16
    for row in spec:
        observed = values[row["output_index"]]
        expected = independent[row["name"]]
        assert observed.dtype == expected.dtype
        if row["dtype"] == "int32":
            np.testing.assert_array_equal(observed, expected, err_msg=row["name"])
        else:
            np.testing.assert_allclose(observed, expected, atol=.003, rtol=.025, err_msg=row["name"])
    original = evaluate(graph, inputs, weights)[0]
    np.testing.assert_array_equal(values[0], original)


def test_ordered_subset_deduplicates_sources_and_keeps_terminal_outputs_first():
    _, _, _, graph = _case()
    direct, spec = instrument_draft_direct_outputs(graph, names=[
        "layer_1_input", "draft_hidden", "layer_0_output", "draft_top1", "context_fc"])
    assert [row["name"] for row in spec] == [
        "draft_top1", "draft_hidden", "layer_1_input", "layer_0_output", "context_fc"]
    assert len(direct.outputs) == 4
    assert spec[2]["output_index"] == spec[3]["output_index"] == 2


@pytest.mark.parametrize("names", [[], "context_fc", ["unknown"], ["context_fc", "context_fc"],
                                  ["layer_0_visible"], [["context_fc"]]])
def test_unsupported_or_ambiguous_native_probes_fail_before_graph_mutation(names):
    _, _, _, graph = _case()
    before = deepcopy(graph)
    with pytest.raises(ValueError):
        instrument_draft_direct_outputs(graph, names=names)
    assert graph.nodes == before.nodes and graph.outputs == before.outputs


def test_direct_output_budgets_and_already_instrumented_graph_are_rejected(monkeypatch):
    from qwen35_moe_dflash.draft_ge_trace import instrument_draft_trace
    _, _, _, graph = _case()
    with pytest.raises(ValueError, match="16-output"):
        instrument_draft_direct_outputs(graph, max_outputs=2)
    packed, _ = instrument_draft_trace(graph, names=["context_fc"])
    with pytest.raises(ValueError, match="uninstrumented"):
        instrument_draft_direct_outputs(packed)
    monkeypatch.setattr("qwen35_moe_dflash.draft_direct_trace.MAX_DIRECT_BYTES", 4)
    with pytest.raises(ValueError, match="64 MiB"):
        instrument_draft_direct_outputs(graph)


def test_export_direct_contract_and_recipe_expose_exact_original_output_ports(tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_standalone
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared, output = tmp_path / "prepared", tmp_path / "direct-air"
    prepare_draft(prepared, sequence_length=8, tiny=True)
    _, _, base = _graph(8, tiny=True)
    direct, spec = instrument_draft_direct_outputs(base)

    def logged(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock serializer; GE construction tested separately")
        else:
            Path(command[1]).write_bytes(b"mock direct AIR")

    monkeypatch.setattr(draft_standalone, "_logged", logged)
    manifest = export_draft(prepared, output, fake_ge(tmp_path), direct_names=DEFAULT_DIRECT_NAMES)
    assert manifest["direct_spec"] == spec and "trace_spec" not in manifest
    assert manifest["export_contract"].endswith("-direct-outputs-v1")
    assert manifest["direct_outputs"] == "ORIGINAL_VALUES_NATIVE_DTYPE_NO_OBSERVER_NODES"
    assert manifest["graphs"][0]["outputs"] == [value.name for value in direct.outputs]
    assert manifest["graphs"][0]["output_dtypes"] == ["torch." + value.dtype for value in direct.outputs]
    recipe = read_recipe(output / "draft/recipe/graph.recipe")
    actual = [(recipe["records"][index]["name"], port) for index, port in recipe["outputs"]]
    assert actual == [(value.name, 0) for value in direct.outputs]
    command = json.loads((output / "draft/export-command.json").read_text())
    assert command["export_contract"] == manifest["export_contract"]


def test_export_rejects_packed_and_direct_selection_before_reading_prepared(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        export_draft(tmp_path / "absent", tmp_path / "unused", tmp_path / "ge",
                     trace_names=["context_fc"], direct_names=["context_fc"])
    assert not (tmp_path / "unused").exists()

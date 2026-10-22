"""Host checks for native Softplus serialization and FP16 threshold equivalence.

These checks do not execute the DDK's Softplus kernel or establish target-device
accuracy. The mathematical comparison isolates removal of the threshold branch.
"""
from copy import deepcopy

import numpy as np
import pytest

from qwen35_moe_dflash import ge_recipe
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_export import build_pair, configurations
from qwen35_moe_dflash.ge_ir import Graph


def _capture_recipe(monkeypatch):
    """Capture records handed to the real writer without bypassing serialization."""
    records = []
    original = ge_recipe.emit_recipe_cpp

    def capture(actual_records, *args, **kwargs):
        records.extend(deepcopy(actual_records))
        return original(actual_records, *args, **kwargs)

    monkeypatch.setattr(ge_recipe, "emit_recipe_cpp", capture)
    return records


@pytest.mark.parametrize("attrs", [{}, {"beta": 1.0, "threshold": 20.0}])
def test_softplus_serializes_native_fp16_contract(tmp_path, monkeypatch, attrs):
    graph = Graph("native_softplus")
    x = graph.input("input", (1, 8, 2))
    graph.outputs = [graph.op("softplus", x, shape=x.shape, attrs=attrs, name="activation")]
    records = _capture_recipe(monkeypatch)
    emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert (tmp_path / "recipe/graph.recipe").is_file()
    activation, = [record for record in records if record["name"] == "activation"]
    assert activation["type"] == "Softplus"
    assert activation["attrs"] == {}
    assert activation["inputs"] == [{"port": "x", "src": 0, "src_port": "y",
                                      "shape": [1, 8, 2], "dtype": "float16"}]
    assert activation["outputs"] == [{"port": "y", "shape": [1, 8, 2], "dtype": "float16"}]
    assert not any(record["type"] == "SoftplusV2" for record in records)
    unrolled = emit_cpp(graph)
    assert ', "Softplus", {"x"}, {"y"}' in unrolled
    assert "SoftplusV2" not in unrolled


def test_tiny_target_uses_native_softplus_for_every_linear_attention_layer(tmp_path, monkeypatch):
    graph = build_pair(*configurations(tiny=True), 8)[0]
    expected_names = {node.output.name for node in graph.nodes if node.kind == "softplus"}
    assert len(expected_names) > 1
    paths = {key: tmp_path / f"weight-{index}.bin" for index, key in enumerate(graph.weights)}
    records = _capture_recipe(monkeypatch)
    # This audits the full graph's descriptors. Real payload preparation and
    # offline folding have separate tests and are intentionally disabled here.
    emit_cpp(graph, paths, external_weights=True, recipe_dir=tmp_path / "recipe",
             fold_constant_arithmetic=False)
    assert (tmp_path / "recipe/graph.recipe").is_file()
    native = [record for record in records if record["type"] == "Softplus"]
    assert {record["name"] for record in native} == expected_names
    assert not any(record["type"] == "SoftplusV2" for record in records)
    for record in native:
        source, = record["inputs"]
        result, = record["outputs"]
        assert source["port"] == "x" and result["port"] == "y"
        assert source["dtype"] == result["dtype"] == "float16"
        assert source["shape"] == result["shape"]
        assert record["attrs"] == {}


@pytest.mark.parametrize("attrs", [
    {"beta": 2.0}, {"beta": 0.0}, {"beta": -1.0}, {"beta": float("nan")},
    {"beta": float("inf")}, {"threshold": 10.0}, {"threshold": 19.0},
    {"threshold": float("nan")}, {"threshold": float("inf")},
])
def test_nondefault_softplus_parameters_are_rejected_before_writing(tmp_path, attrs):
    graph = Graph("unsupported_softplus")
    x = graph.input("input", (2,))
    graph.outputs = [graph.op("softplus", x, shape=x.shape, attrs=attrs)]
    with pytest.raises(NotImplementedError):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("input_dtype,output_dtype", [("int32", "int32"),
                                                       ("float16", "int32"),
                                                       ("int32", "float16")])
def test_native_softplus_rejects_non_fp16_contract(tmp_path, input_dtype, output_dtype):
    graph = Graph("unsupported_softplus_dtype")
    x = graph.input("input", (2,), input_dtype)
    graph.outputs = [graph.op("softplus", x, shape=x.shape, dtype=output_dtype)]
    with pytest.raises(NotImplementedError):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


def test_threshold20_removal_preserves_all_finite_fp16_rounded_values():
    bits = np.arange(1 << 16, dtype=np.uint16)
    inputs = bits.view(np.float16)
    inputs = inputs[np.isfinite(inputs)]
    assert inputs.size == 63488
    assert {0x0000, 0x8000}.issubset(set(inputs.view(np.uint16)))
    precise = inputs.astype(np.float64)
    native_formula = np.logaddexp(precise, 0)
    threshold_formula = np.where(precise > 20, precise, native_formula)
    # Stable FP64 formulas keep all intermediate exp/log arithmetic outside
    # FP16; only the operator's output boundary rounds to FP16. This comparison
    # is not a claim of bitwise equality with PyTorch or the target kernel.
    native_rounded = native_formula.astype(np.float16)
    threshold_rounded = threshold_formula.astype(np.float16)
    np.testing.assert_array_equal(native_rounded.view(np.uint16),
                                  threshold_rounded.view(np.uint16))
    assert np.count_nonzero(precise > 20) == 12031


def test_threshold20_removal_preserves_nonfinite_value_semantics():
    inputs = np.array([-np.inf, np.inf, np.nan], dtype=np.float16).astype(np.float64)
    with np.errstate(invalid="ignore"):
        native_formula = np.logaddexp(inputs, 0)
        threshold_formula = np.where(inputs > 20, inputs, native_formula)
    native = native_formula.astype(np.float16)
    threshold = threshold_formula.astype(np.float16)
    np.testing.assert_array_equal(native[:2].view(np.uint16), threshold[:2].view(np.uint16))
    assert native[0].view(np.uint16) == 0
    assert np.isposinf(native[1])
    assert np.isnan(native[2]) and np.isnan(threshold[2])

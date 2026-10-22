"""Execute serialized scan recipes on CPU; target reduction accuracy is pending."""
from pathlib import Path
import math

import numpy as np
import pytest
import torch

from qwen35_moe_dflash.ge_cpp import _Emitter, _opindex, emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations, prepare_pair, build_pair
from qwen35_moe_dflash.ge_ir import DTYPES, Graph, Value
from qwen35_moe_dflash.ge_recipe import emit_recipe_cpp
from qwen35_moe_dflash.ge_scan import emit_cumsum
from qwen35_moe_dflash.ge_target import _Builder, _linear_attention
from test_ge_select_lowering import RecipeReader


def _recipe(tmp_path, array, axis, *, scope=None):
    tmp_path = tmp_path / "recipe"
    graph = Graph("scan_test")
    source = graph.input("x", array.shape, str(array.dtype))
    output = graph.op("cumsum", source, shape=source.shape, dtype=source.dtype,
                      attrs={"axis": axis})
    graph.outputs = [output]
    emitter = _Emitter(graph, recipe=True)
    ref = emitter.op("x", "Data", [], [("y", source)], {"index": "int64_t{0}"})
    emitter.refs["x"] = (ref, "y")
    emit_cumsum(emitter, output, source, axis, scope=scope)
    outputs = [(_opindex(emitter.refs[output.name][0]), 0)]
    emit_recipe_cpp(emitter.records, graph.name, [0], outputs, tmp_path)
    return RecipeReader(tmp_path / "graph.recipe").read()


def _execute(records, outputs, array):
    values = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for index, record in enumerate(records):
            args = [values[(item["src"], item["src_port"])] for item in record["inputs"]]
            output, = record["outputs"]
            dtype = DTYPES[output["dtype"]]
            kind = record["type"]
            if kind == "Data":
                value = array
            elif kind == "Const":
                path, offset, size = record["constant"]
                with Path(path).open("rb") as stream:
                    stream.seek(offset)
                    payload = stream.read(size)
                assert len(payload) == size
                value = np.frombuffer(payload, dtype=dtype).reshape(output["shape"])
            elif kind == "Slice":
                value = args[0][tuple(slice(int(start), int(start + size))
                                      for start, size in zip(args[1], args[2]))]
            elif kind == "ReduceSum":
                # FP32 accumulation models the existing CPU oracle, not a
                # claim about the unavailable internal OMG/device kernel.
                assert dtype == np.float16
                assert record["attrs"]["keep_dims"]
                value = args[0].astype(np.float32).sum(
                    axis=tuple(int(a) for a in args[1]), keepdims=True)
            elif kind == "ConcatD":
                assert record["attrs"]["N"] == len(args)
                value = np.concatenate(args, axis=record["attrs"]["concat_dim"])
            elif kind == "Add":
                assert dtype in (np.int32, np.int64)
                value = np.add(args[0], args[1], dtype=dtype)
            else:
                raise AssertionError(f"unexpected scan operator {kind}")
            value = np.asarray(value, dtype=dtype)
            assert value.shape == output["shape"]
            values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


def _scan(tmp_path, array, axis, scope=None):
    records, _, outputs = _recipe(tmp_path, array, axis, scope=scope)
    actual, = _execute(records, outputs, array)
    return actual, records


@pytest.mark.parametrize("sequence", [8, 65])
def test_fp16_scan_of_actual_gdn_decay_graph(tmp_path, sequence):
    # Build the real model producer, including softplus, -exp(A_log),
    # transposition, zero padding, and chunk-64 reshape. Random tiny weights
    # are host evidence only; this is not a real-checkpoint quality claim.
    config, _ = configurations(tiny=True)
    graph = Graph("gdn_decay")
    x = graph.input("hidden", (1, sequence, config["hidden_size"]))
    _linear_attention(_Builder(graph), x, "target.layers.0.linear_attn", config)
    index, scan = next((i, node) for i, node in enumerate(graph.nodes) if node.kind == "cumsum")
    graph.nodes = graph.nodes[:index]
    graph.outputs = [scan.inputs[0]]
    rng = np.random.default_rng(481)
    feeds = {"hidden": rng.normal(0, .2, x.shape).astype(np.float16)}
    weights = {key: rng.normal(0, .2, value.shape).astype(np.float16)
               for key, value in graph.weights.items()}
    decay, = evaluate(graph, feeds, weights)
    assert decay.shape[-1] == 64 and np.all(decay <= 0)
    actual, records = _scan(tmp_path, decay, -1, "gdn_nonpositive")
    expected = torch.cumsum(torch.from_numpy(decay), -1)
    torch.testing.assert_close(torch.from_numpy(actual), expected)
    assert {r["type"] for r in records} == {"Data", "Const", "Slice", "ReduceSum", "ConcatD"}
    assert sum(r["type"] == "ReduceSum" for r in records) == 64
    assert all(p["dtype"] != "float32" for r in records for p in r["outputs"])


@pytest.mark.parametrize("axis", [1, -2])
def test_fp16_nonpositive_wide_dynamic_range(tmp_path, axis):
    rng = np.random.default_rng(611)
    array = -np.exp(rng.uniform(-16, 10, (3, 64, 2))).astype(np.float16)
    actual, _ = _scan(tmp_path, array, axis, "gdn_nonpositive")
    torch.testing.assert_close(torch.from_numpy(actual), torch.cumsum(torch.from_numpy(array), axis))


@pytest.mark.parametrize("marker", [-np.inf, np.nan])
def test_future_nonfinite_values_do_not_contaminate_prior_prefixes(tmp_path, marker):
    array = np.array([[0, -1, -2, marker, -3, -0., -4, -5]], dtype=np.float16)
    actual, records = _scan(tmp_path, array, -1, "gdn_nonpositive")
    assert np.isfinite(actual[0, :3]).all()
    np.testing.assert_array_equal(actual[0, :3], [0, -1, -3])
    torch.testing.assert_close(torch.from_numpy(actual), torch.cumsum(torch.from_numpy(array), -1), equal_nan=True)
    assert not {"Mul", "MatMul", "BatchMatMulV2", "Add"}.intersection(r["type"] for r in records)


def test_generic_signed_fp16_scan_is_rejected_due_to_cancellation(tmp_path):
    # This counterexample is intentionally retained: prefix reductions may
    # reorder FP32 additions. A generic signed-scan promise would be false.
    array = np.array([65504, .001, -65504, .001, 0, 0, 0, 0], dtype=np.float16)
    with pytest.raises(NotImplementedError, match="GDN nonpositive"):
        _recipe(tmp_path, array, 0)
    reductions = np.array([array[:n].astype(np.float32).sum() for n in range(1, 9)], np.float16)
    reference = torch.cumsum(torch.from_numpy(array), 0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.from_numpy(reductions), reference)


def test_gdn_scope_stays_bounded(tmp_path):
    array = np.full((1, 256), -0.125, np.float16)
    actual, _ = _scan(tmp_path / "allowed", array, -1, "gdn_nonpositive")
    torch.testing.assert_close(torch.from_numpy(actual), torch.cumsum(torch.from_numpy(array), -1))
    with pytest.raises(NotImplementedError, match="exceeds 256"):
        _recipe(tmp_path / "large", np.zeros((1, 257), np.float16), -1, scope="gdn_nonpositive")


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("length", [8, 128])
@pytest.mark.parametrize("axis", [1, -2])
def test_integer_scan_matches_modular_cumsum_with_overflow(tmp_path, dtype, length, axis):
    rng = np.random.default_rng(79)
    info = np.iinfo(dtype)
    array = rng.integers(info.min, info.max, (2, length, 3), dtype=dtype, endpoint=True)
    array[0, :3, 0] = [info.max, 1, 1]
    actual, records = _scan(tmp_path, array, axis)
    expected = np.cumsum(array, axis=axis, dtype=dtype)
    np.testing.assert_array_equal(actual, expected)
    torch.testing.assert_close(torch.from_numpy(actual), torch.cumsum(torch.from_numpy(array), axis,
                                                                    dtype=torch.from_numpy(array).dtype),
                               atol=0, rtol=0)
    assert sum(r["type"] == "Add" for r in records) == math.ceil(math.log2(length))
    assert {r["type"] for r in records} == {"Data", "Const", "Slice", "ConcatD", "Add"}


@pytest.mark.parametrize("dtype", [np.float16, np.int32, np.int64])
def test_single_element_axis(tmp_path, dtype):
    array = np.array([[-0.], [-3.]], dtype=dtype)
    actual, _ = _scan(tmp_path, array, 1, "gdn_nonpositive" if dtype == np.float16 else None)
    torch.testing.assert_close(torch.from_numpy(actual), torch.cumsum(torch.from_numpy(array), 1,
                                                                    dtype=torch.from_numpy(array).dtype))


@pytest.mark.parametrize("shape,axis", [((), 0), ((1, 0), 1), ((2, 3), 2), ((2, 3), -3), ((2, 3), True)])
def test_invalid_scan_geometry_fails_before_emission(shape, axis):
    source = Value("x", shape, "int32")
    with pytest.raises(ValueError):
        emit_cumsum(None, Value("y", shape, "int32"), source, axis)


def test_output_descriptor_must_match_source():
    source = Value("x", (2, 3), "int32")
    for output in (Value("y", (6,), "int32"), Value("y", (2, 3), "int64")):
        with pytest.raises(ValueError, match="shape and dtype"):
            emit_cumsum(None, output, source, 1)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_single_element_constant_scan_materializes_lazy_source(tmp_path, dtype):
    graph = Graph("constant_scan")
    graph.input("unused", (1,), dtype)
    array = np.array([[7], [-3]], dtype=DTYPES[dtype])
    source = graph.const(array, dtype=dtype, name="constant_input")
    output = graph.op("cumsum", source, shape=source.shape, dtype=dtype, attrs={"axis": 1})
    graph.outputs = [output]
    code = emit_cpp(graph, fold_constant_arithmetic=False)
    assert 'Op("constant_input", "Const"' in code
    assert '"Cumsum"' not in code
    assert code.count('Op("constant_input", "Const"') == 1
    emit_cpp(graph, recipe_dir=tmp_path / "recipe", fold_constant_arithmetic=False)
    records, _, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    actual, = _execute(records, outputs, np.array([0], dtype=DTYPES[dtype]))
    np.testing.assert_array_equal(actual, array)
    assert sum(record["name"] == "constant_input" for record in records) == 1


def test_serialized_scans_preserve_tiny_pair_outputs(tmp_path, monkeypatch):
    # Integrate the serialized scan executor into complete target/draft
    # evaluation. Everything outside cumsum retains the original oracle.
    # This covers the lowering's effect on logits/top1 and draft decisions,
    # without pretending to execute the unavailable OMG/device reduction.
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, sequence_length=8, tiny=True)
    target, draft = build_pair(plan["target_config"], plan["draft_config"], 8)
    weights = {key: np.fromfile(prepared / item["path"], dtype="<f2").reshape(item["shape"])
               for key, item in plan["weights"].items()}
    calls = {"float16": 0, "integer": 0}

    def serialized_cumsum(array, axis=None, dtype=None, out=None):
        assert dtype is None and out is None and axis is not None
        if array.dtype == np.float32:
            # evaluate() promotes the original FP16 tensor for its reference
            # accumulator. Recover its exact input boundary for the recipe.
            source = array.astype(np.float16)
            np.testing.assert_array_equal(source.astype(np.float32), array)
            assert np.all((source <= 0) | np.isnan(source))
            scope, key = "gdn_nonpositive", "float16"
        else:
            assert array.dtype in (np.int32, np.int64)
            source, scope, key = array, None, "integer"
        calls[key] += 1
        folder = tmp_path / "serialized_scans" / str(sum(calls.values()))
        result, _ = _scan(folder, source, axis, scope)
        return result

    for length in (1, 3, 8):
        ids = np.zeros((1, 8), dtype=np.int32)
        mask = ids.copy()
        ids[0, :length] = [7, 3, 26, 8, 16, 42, 29, 31][:length]
        mask[0, :length] = 1
        target_inputs = {"input_ids": ids, "attention_mask": mask}
        expected_top1, expected_features = evaluate(target, target_inputs, weights)
        expected_draft, = evaluate(draft, {**target_inputs, "target_features": expected_features}, weights)
        with monkeypatch.context() as patched:
            # NumPy is shared across modules. No baseline cumsum is called
            # within this context, and context exit restores the real API.
            patched.setattr(np, "cumsum", serialized_cumsum)
            actual_top1, actual_features = evaluate(target, target_inputs, weights)
            actual_draft, = evaluate(draft, {**target_inputs, "target_features": actual_features}, weights)
        np.testing.assert_array_equal(actual_top1, expected_top1)
        torch.testing.assert_close(torch.from_numpy(actual_features), torch.from_numpy(expected_features))
        np.testing.assert_array_equal(actual_draft, expected_draft)
    assert calls == {"float16": 9, "integer": 3}

"""Execute emitted contraction recipes on CPU; no OMG/device claim is made."""
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import build_pair, configurations
from qwen35_moe_dflash.ge_ir import DTYPES, Graph
from test_ge_select_lowering import RecipeReader


def _graph(kind="reduce_sum", shape=(2, 4), axes=(-1,), keepdims=True, dtype="float16"):
    graph = Graph("reduction_lowering")
    source = graph.input("x", shape, dtype)
    reduced = {axis % len(shape) for axis in axes}
    output_shape = tuple(1 if i in reduced else d for i, d in enumerate(shape)) if keepdims else tuple(
        d for i, d in enumerate(shape) if i not in reduced)
    graph.outputs = [graph.op(kind, source, shape=output_shape, dtype=dtype, name="result",
                              attrs={"axes": list(axes), "keepdims": keepdims})]
    return graph


def _recipe(graph, folder, mode="matmul", **kwargs):
    report = {}
    emit_cpp(graph, recipe_dir=folder, fp16_reductions=mode, reduction_report=report, **kwargs)
    records, _, outputs = RecipeReader(folder / "graph.recipe").read()
    return records, outputs, report


def _constant(record):
    path, offset, size = record["constant"]
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(size)
    assert len(payload) == size
    output, = record["outputs"]
    return np.frombuffer(payload, dtype=DTYPES[output["dtype"]]).reshape(output["shape"])


def _execute(records, outputs, feeds):
    """Independent recipe evaluator, with a rounding boundary at every record."""
    values = {}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for index, record in enumerate(records):
            args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
            output, = record["outputs"]
            dtype = DTYPES[output["dtype"]]
            kind, attrs = record["type"], record["attrs"]
            if kind == "Data":
                result = feeds[record["name"]]
            elif kind == "Const":
                result = _constant(record)
            elif kind == "Transpose":
                result = np.transpose(args[0], tuple(args[1]))
            elif kind == "Reshape":
                assert args[1].dtype == np.int32
                result = args[0].reshape(tuple(args[1]))
            elif kind == "MatMul":
                assert all(arg.dtype == np.float16 for arg in args)
                left, right = [arg.astype(np.float32) for arg in args]
                if attrs["transpose_x1"]:
                    left = np.swapaxes(left, -1, -2)
                if attrs["transpose_x2"]:
                    right = np.swapaxes(right, -1, -2)
                result = np.matmul(left, right)
            elif kind in {"ReduceSum", "ReduceMean"}:
                assert args[1].dtype == np.int32
                source = args[0].astype(np.float32) if args[0].dtype == np.float16 else args[0]
                fn = np.sum if kind == "ReduceSum" else np.mean
                options = {"dtype": dtype} if kind == "ReduceSum" and dtype.kind == "i" else {}
                result = fn(source, axis=tuple(args[1]), keepdims=bool(attrs["keep_dims"]), **options)
            elif kind in {"Mul", "Add"}:
                operands = [arg.astype(np.float32) if arg.dtype == np.float16 else arg for arg in args]
                result = np.multiply(*operands) if kind == "Mul" else np.add(*operands)
            elif kind == "Slice":
                result = args[0][tuple(slice(int(start), int(start + size))
                                        for start, size in zip(args[1], args[2]))]
            elif kind == "ConcatD":
                assert attrs["N"] == len(args)
                result = np.concatenate(args, axis=attrs["concat_dim"])
            else:
                raise AssertionError(f"unexpected recipe operator {kind}")
            result = np.asarray(result, dtype=dtype)
            assert result.shape == output["shape"]
            values[(index, output["port"])] = result
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


@pytest.mark.parametrize("kind", ["reduce_sum", "reduce_mean"])
@pytest.mark.parametrize("shape,axis", [((8,), 0), ((3, 1), -1), ((2, 8), 1),
                                        ((4, 3), 0), ((2, 4, 3), 1), ((2, 3, 4, 2), -2)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_emitted_contractions_match_original_reduction(tmp_path, kind, shape, axis, keepdims):
    graph = _graph(kind, shape, [axis], keepdims)
    original = pickle.dumps(graph)
    records, outputs, report = _recipe(graph, tmp_path / "recipe")
    # Dyadic signed values make permutation/row errors visible without relying
    # on an implementation-specific FP32 summation order.
    values = ((np.arange(math.prod(shape)).reshape(shape) % 17 - 8) / 8).astype(np.float16)
    actual, = _execute(records, outputs, {"x": values})
    expected, = evaluate(graph, {"x": values})
    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert pickle.dumps(graph) == original
    assert report["counts"]["total"] == report["counts"]["replaced"] == 1
    assert report["counts"]["retained"] == 0
    contract, = report["replaced"]
    assert contract["axis"] == axis % len(shape) and contract["axes"] == [axis]
    assert contract["K"] == shape[axis] and contract["M"] == math.prod(shape) // shape[axis]
    mm, = [record for record in records if record["type"] == "MatMul"]
    assert mm["attrs"] == {"transpose_x1": False, "transpose_x2": True}
    assert mm["inputs"][0]["shape"] == (contract["M"], contract["K"])
    assert mm["inputs"][1]["shape"] == (1, contract["K"])
    coefficient = 1.0 if kind == "reduce_sum" else 1.0 / shape[axis]
    np.testing.assert_array_equal(_constant(records[mm["inputs"][1]["src"]]),
                                  np.full((1, shape[axis]), coefficient, np.float16))
    assert not {"Cast", "ReduceSum", "ReduceMean"}.intersection(r["type"] for r in records)
    assert records[outputs[0][0]]["name"] == "result"


def test_fp16_product_boundary_and_prepared_weight_are_preserved(tmp_path):
    graph = Graph("product_reduction")
    source = graph.input("x", (2, 4))
    weight = graph.weight("weight", (2, 4))
    product = graph.op("mul", source, weight, shape=(2, 4), name="rounded_product")
    graph.outputs = [graph.op("reduce_sum", product, shape=(2, 1), name="result",
                              attrs={"axes": [1], "keepdims": True})]
    values = np.array([[0.7, 0.7, 0.7, 0.7], [-0.3, 0.3, 0.3, -0.3]], np.float16)
    weight_values = np.array([[0.7, 0.7, 0.7, 0.7], [0.2, 0.2, -0.2, 0.2]], np.float16)
    path = tmp_path / "weight.bin"
    path.write_bytes(weight_values.tobytes())
    original, weight_bytes = pickle.dumps(graph), path.read_bytes()
    records, outputs, report = _recipe(graph, tmp_path / "recipe", weight_paths={"weight": path})
    actual, = _execute(records, outputs, {"x": values})
    expected, = evaluate(graph, {"x": values}, {"weight": weight_values})
    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    producer, = [r for r in records if r["type"] == "Mul"]
    assert producer["outputs"][0]["dtype"] == "float16"
    assert report["replaced"][0]["source"]["name"] == "rounded_product"
    assert pickle.dumps(graph) == original and path.read_bytes() == weight_bytes


def test_mean_scales_before_accumulation_without_fp16_sum_overflow(tmp_path):
    graph = _graph("reduce_mean", (2, 4096), [1])
    values = np.full((2, 4096), 16, np.float16)
    records, outputs, _ = _recipe(graph, tmp_path / "recipe")
    actual, = _execute(records, outputs, {"x": values})
    torch.testing.assert_close(torch.from_numpy(actual), torch.full((2, 1), 16, dtype=torch.float16))
    assert np.isfinite(actual).all()
    assert not any(r["type"] in {"ReduceSum", "Mul", "Power"} for r in records)


@pytest.mark.parametrize("marker", [np.nan, -np.inf])
@pytest.mark.parametrize("axis", [0, -1])
def test_scan_uses_independent_prefix_slices_and_keeps_future_nonfinite_isolated(tmp_path, marker, axis):
    values = np.array([[0, -1, -2, marker, -3, -0., -4, -5]], np.float16)
    if axis == 0:
        values = values.T
    graph = Graph("scan_lowering")
    source = graph.input("x", values.shape)
    graph.outputs = [graph.op("cumsum", source, shape=values.shape, name="scan",
                              attrs={"axis": axis, "scope": "gdn_nonpositive"})]
    native, native_outputs, _ = _recipe(graph, tmp_path / "native", "native")
    records, outputs, report = _recipe(graph, tmp_path / "matmul")
    actual, = _execute(records, outputs, {"x": values})
    expected, = _execute(native, native_outputs, {"x": values})
    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected), equal_nan=True)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    assert np.isfinite(np.take(actual, [0, 1, 2], axis=axis)).all()
    assert report["counts"]["scan_prefix_replaced"] == 8
    assert [item["K"] for item in report["replaced"]] == list(range(1, 9))
    slices = [r for r in records if r["type"] == "Slice"]
    assert len(slices) == 8 and all(records[r["inputs"][0]["src"]]["name"] == "x" for r in slices)
    assert not {"Mul", "SelectV2", "Cast", "ReduceSum", "BatchMatMulV2"}.intersection(r["type"] for r in records)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_reduction_and_scan_are_byte_identical_between_modes(tmp_path, dtype):
    graph = _graph(dtype=dtype)
    scan = graph.op("cumsum", graph.inputs[0], shape=(2, 4), dtype=dtype, attrs={"axis": 1}, name="scan")
    graph.outputs.append(scan)
    assert emit_cpp(graph) == emit_cpp(graph, fp16_reductions="matmul")
    records, outputs, report = _recipe(graph, tmp_path / "recipe")
    info = np.iinfo(DTYPES[dtype])
    values = np.array([[info.max, 1, 1, -3], [0, 2, -1, 5]], dtype=DTYPES[dtype])
    actual_sum, actual_scan = _execute(records, outputs, {"x": values})
    np.testing.assert_array_equal(actual_sum, values.sum(axis=1, keepdims=True, dtype=values.dtype))
    np.testing.assert_array_equal(actual_scan, values.cumsum(axis=1, dtype=values.dtype))
    assert report["counts"]["integer_retained"] == 1
    assert report["counts"]["replaced"] == 0
    assert not {"Cast", "MatMul"}.intersection(r["type"] for r in records)


@pytest.mark.parametrize("kind,shape,axes,reason", [
    ("reduce_sum", (2, 4), [0, 1], "requires_single_axis"),
    ("reduce_mean", (2, 4), [], "requires_single_axis"),
    ("reduce_sum", (0, 4), [1], "requires_positive_static_shape"),
    ("reduce_mean", (2, 3), [1], "mean_requires_exact_fp16_reciprocal"),
    ("reduce_mean", (1, 1 << 25), [1], "mean_requires_exact_fp16_reciprocal"),
    ("reduce_sum", (1 << 31, 1), [1], "reshape_dimensions_exceed_int32"),
])
def test_valid_ineligible_reductions_remain_native_with_reason(kind, shape, axes, reason):
    graph = _graph(kind, shape, axes)
    report = {}
    assert emit_cpp(graph, fp16_reductions="matmul", reduction_report=report) == emit_cpp(graph)
    assert report["counts"]["replaced"] == 0 and report["counts"]["retained"] == 1
    assert report["retained"][0]["reason"] == reason


@pytest.mark.parametrize("axes", [[True], [1.0], [2], [-3], [1, -1], [1 << 31], None, "1", [[1]]])
@pytest.mark.parametrize("mode", ["native", "matmul"])
def test_malformed_axes_still_fail_before_recipe_serialization(tmp_path, axes, mode):
    graph = _graph()
    graph.nodes[-1].attrs["axes"] = axes
    with pytest.raises(ValueError):
        _recipe(graph, tmp_path / "recipe", mode)
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("shape,dtype", [((2,), "float16"), ((2, 1), "int32")])
def test_malformed_output_contract_rejected_in_matmul_mode(tmp_path, shape, dtype):
    graph = Graph("invalid_contract")
    source = graph.input("x", (2, 4))
    graph.outputs = [graph.op("reduce_sum", source, shape=shape, dtype=dtype,
                              attrs={"axes": [1], "keepdims": True})]
    with pytest.raises(ValueError, match="output (shape|dtype)"):
        _recipe(graph, tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("kind,dtype,expected_sha", [
    ("reduce_sum", "float16", "ed0a7f91ac68881fe67a1afdb697b0c55c5aaef4b3ead2ba45697cd06fb76237"),
    ("reduce_mean", "float16", "b138055b0d06656dbdc770e24cf06c9a8d7867d7d6545db6e788e38cf681088d"),
    ("reduce_sum", "int32", "3accf52a74e9c1082e677182f5bffc29ac4abd2e8fe87d2542bf3d7b3ea5611a"),
    ("cumsum", "float16", "0d61c0e5884d64044c2cc63c7e6ac67491bd5ccfe89caf2bfd15bdba9467d3ba"),
])
def test_default_cpp_matches_prechange_bytes(kind, dtype, expected_sha):
    graph = Graph("native_reduction_identity")
    source = graph.input("x", (2, 4), dtype)
    attrs = {"axis": -1, "scope": "gdn_nonpositive"} if kind == "cumsum" else {"axes": [-1], "keepdims": True}
    graph.outputs = [graph.op(kind, source, shape=(2, 4) if kind == "cumsum" else (2, 1),
                              dtype=dtype, attrs=attrs, name="result")]
    code = emit_cpp(graph)
    report = {"stale": "must be cleared"}
    assert emit_cpp(graph, fp16_reductions="native", reduction_report=report) == code
    assert hashlib.sha256(code.encode()).hexdigest() == expected_sha
    assert "stale" not in report and report["mode"] == "native"
    assert report["counts"]["replaced"] == 0


def test_policy_and_report_argument_validation():
    graph = _graph()
    for mode in (None, [], True, "automatic"):
        with pytest.raises(ValueError, match="fp16_reductions"):
            emit_cpp(graph, fp16_reductions=mode)
    with pytest.raises(TypeError, match="dictionary"):
        emit_cpp(graph, reduction_report=[])


def test_reports_deterministic_and_cover_actual_tiny_pair_including_nested_scans(tmp_path):
    for graph, expected_fp16, expected_integer, expected_scan in zip(
            build_pair(*configurations(tiny=True), 8), (421, 10), (16, 1), (192, 0)):
        before = pickle.dumps(graph)
        # Descriptor-only audit deliberately avoids inventing checkpoint data.
        # The separate small recipes above execute the emitted operations.
        weight_paths = {key: tmp_path / "weights" / key for key in graph.weights}
        kwargs = {"weight_paths": weight_paths, "external_weights": True, "fold_constant_arithmetic": False}
        native, _, native_report = _recipe(graph, tmp_path / (graph.name + "_native"), "native", **kwargs)
        records, _, report = _recipe(graph, tmp_path / (graph.name + "_matmul"), **kwargs)
        second = {}
        emit_cpp(graph, recipe_dir=tmp_path / (graph.name + "_repeat"), fp16_reductions="matmul",
                 reduction_report=second, **kwargs)
        assert json.dumps(report, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
        assert pickle.dumps(graph) == before
        assert report["counts"] == {"total": expected_fp16 + expected_integer,
            "replaced": expected_fp16, "retained": expected_integer, "fp16_replaced": expected_fp16,
            "integer_retained": expected_integer, "scan_prefix_replaced": expected_scan}
        assert native_report["counts"]["total"] == expected_fp16 + expected_integer
        reductions = [r for r in records if r["type"] in {"ReduceSum", "ReduceMean"}]
        assert len(reductions) == expected_integer
        assert all(r["inputs"][0]["dtype"] == r["outputs"][0]["dtype"] == "int32" for r in reductions)
        originals = {r["name"]: r for r in native if r["type"] in {"ReduceSum", "ReduceMean"}}
        for retained in reductions:
            original = originals[retained["name"]]
            assert retained["type"] == original["type"] and retained["attrs"] == original["attrs"]
            assert retained["outputs"] == original["outputs"]
            for left, right in zip(retained["inputs"], original["inputs"]):
                assert {k: left[k] for k in ("port", "shape", "dtype")} == {k: right[k] for k in ("port", "shape", "dtype")}
            np.testing.assert_array_equal(_constant(records[retained["inputs"][1]["src"]]),
                                          _constant(native[original["inputs"][1]["src"]]))
        names = {r["name"]: r for r in records}
        for contract in report["replaced"]:
            assert contract["source"]["dtype"] == contract["output"]["dtype"] == "float16"
            mm = names[f'__ge/{contract["name"]}/reduction/matmul']
            assert mm["type"] == "MatMul"
            assert mm["inputs"][0]["shape"] == (contract["M"], contract["K"])
            assert mm["inputs"][1]["shape"] == (1, contract["K"])
            assert names[contract["name"]]["outputs"][0]["shape"] == tuple(contract["output"]["shape"])
        assert all(r["reason"] == "non_fp16_source_or_output" for r in report["retained"])

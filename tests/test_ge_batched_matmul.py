"""Numerical/layout checks of serialized BMM normalization, not device evidence."""
import math

import numpy as np
import pytest
import torch

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import build_pair, configurations, prepare_pair
from qwen35_moe_dflash.ge_ir import DTYPES, Graph
from test_ge_select_lowering import RecipeReader
from test_ge_static_broadcast import _payload, _serialize_both


_NUMPY_MATMUL = np.matmul


def _graph(left_shape, right_shape, *, transpose_a=False, transpose_b=False,
           output_shape=None, left_dtype="float16", right_dtype="float16", dtype="float16",
           name="product"):
    graph = Graph("bmm_layout")
    left = graph.input("left", left_shape, left_dtype)
    right = graph.input("right", right_shape, right_dtype)
    if output_shape is None:
        output_shape = left_shape[:-2] + (left_shape[-1] if transpose_a else left_shape[-2],
                                         right_shape[-2] if transpose_b else right_shape[-1])
    graph.outputs = [graph.op("matmul", left, right, shape=output_shape, dtype=dtype, name=name,
                             attrs={"transpose_a": transpose_a, "transpose_b": transpose_b})]
    return graph


def _execute(records, outputs, feeds):
    """Interpret decoded control tensors and GE adj flags independently."""
    values = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for index, record in enumerate(records):
            args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
            output, = record["outputs"]
            dtype = DTYPES[output["dtype"]]
            kind = record["type"]
            if kind == "Data":
                result = feeds[record["name"]]
            elif kind == "Const":
                result = np.frombuffer(_payload(record), dtype=dtype).reshape(output["shape"])
            elif kind == "Reshape":
                assert args[1].dtype == np.int32
                assert record["attrs"] == {"axis": 0, "num_axes": -1}
                result = args[0].reshape(tuple(int(x) for x in args[1]))
                # A view must not create a new floating-point boundary.
                assert result.dtype == args[0].dtype
                assert result.tobytes() == args[0].tobytes()
            elif kind == "Transpose":
                assert args[1].dtype == np.int64
                result = np.transpose(args[0], tuple(int(x) for x in args[1]))
            elif kind in {"BatchMatMulV2", "MatMul"}:
                assert dtype == np.float16 and all(arg.dtype == np.float16 for arg in args)
                if kind == "BatchMatMulV2":
                    assert all(arg.ndim == 3 for arg in args)
                    assert args[0].shape[0] == args[1].shape[0]
                    assert record["attrs"]["offset_x"] == 0
                    keys = ("adj_x1", "adj_x2")
                else:
                    assert all(arg.ndim == 2 for arg in args)
                    keys = ("transpose_x1", "transpose_x2")
                a, b = (np.swapaxes(arg, -1, -2) if record["attrs"][key] else arg
                        for arg, key in zip(args, keys))
                # Match the existing CPU oracle's accumulation and original
                # FP16 output boundary; no actual internal kernel claim.
                result = _NUMPY_MATMUL(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
            else:
                raise AssertionError(f"unexpected normalized matmul operator: {kind}")
            assert result.dtype == dtype and result.shape == output["shape"]
            values[(index, output["port"])] = result
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


def _reference(left, right, transpose_a=False, transpose_b=False):
    a = np.swapaxes(left, -1, -2) if transpose_a else left
    b = np.swapaxes(right, -1, -2) if transpose_b else right
    with np.errstate(over="ignore", invalid="ignore"):
        return _NUMPY_MATMUL(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)


@pytest.mark.parametrize("batch", [(1, 2), (2, 3), (1, 2, 3)])
@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (False, True), (True, False), (True, True)])
def test_highrank_bmm_preserves_batch_order_and_transposes(tmp_path, batch, transpose_a, transpose_b):
    left_shape = batch + ((5, 3) if transpose_a else (3, 5))
    right_shape = batch + ((4, 5) if transpose_b else (5, 4))
    rng = np.random.default_rng(326)
    left = rng.normal(0, .2, left_shape).astype(np.float16)
    right = rng.normal(0, .2, right_shape).astype(np.float16)
    graph = _graph(left_shape, right_shape, transpose_a=transpose_a, transpose_b=transpose_b)
    records, _, outputs = _serialize_both(graph, tmp_path)
    bmm, = [record for record in records if record["type"] == "BatchMatMulV2"]
    assert bmm["attrs"]["adj_x1"] == False and bmm["attrs"]["adj_x2"] == transpose_b
    assert [port["shape"] for port in bmm["inputs"]] == [
        (math.prod(batch), 3, 5), (math.prod(batch),) + right_shape[-2:]]
    assert bmm["outputs"][0]["shape"] == (math.prod(batch), 3, 4)
    assert sum(record["type"] == "Reshape" for record in records) == 3
    kinds = {"Data", "Const", "Reshape", "BatchMatMulV2"}
    if transpose_a:
        kinds.add("Transpose")
    assert {record["type"] for record in records} == kinds
    assert all(port["dtype"] == "float16" for record in records if record["type"] != "Const"
               for port in record["outputs"])
    actual, = _execute(records, outputs, {"left": left, "right": right})
    np.testing.assert_array_equal(actual, _reference(left, right, transpose_a, transpose_b))
    a, b = torch.from_numpy(left), torch.from_numpy(right)
    if transpose_a:
        a = a.transpose(-1, -2)
    if transpose_b:
        b = b.transpose(-1, -2)
    torch.testing.assert_close(torch.from_numpy(actual), torch.matmul(a, b))


def test_batch_with_nan_cannot_contaminate_another_batch(tmp_path):
    batch = (2, 3, 2)
    left = np.zeros(batch + (3, 3), dtype=np.float16)
    right = np.zeros_like(left)
    for index in np.ndindex(batch):
        left[index] = np.eye(3, dtype=np.float16)
        right[index] = np.full((3, 3), np.ravel_multi_index(index, batch) + 1, np.float16)
    left[1, 0, 1] = np.nan
    records, _, outputs = _serialize_both(_graph(left.shape, right.shape), tmp_path)
    actual, = _execute(records, outputs, {"left": left, "right": right})
    np.testing.assert_array_equal(actual, _reference(left, right))
    for index in np.ndindex(batch):
        if index == (1, 0, 1):
            assert np.isnan(actual[index]).all()
        else:
            assert np.isfinite(actual[index]).all()
            np.testing.assert_array_equal(actual[index], right[index])


def test_observed_v1115_uses_flat_batches_and_original_output_shape(tmp_path):
    # Preserve the historical board-reported shape regression independently of
    # generated node IDs: the current GDN graph removes this redundant rank5.
    graph = _graph((1, 2, 1, 64, 64), (1, 2, 1, 64, 8), name="v1115")
    records, _, outputs = _serialize_both(graph, tmp_path)
    bmm, = [record for record in records if record["type"] == "BatchMatMulV2"]
    assert [port["shape"] for port in bmm["inputs"]] == [(2, 64, 64), (2, 64, 8)]
    assert records[outputs[0][0]]["outputs"][0]["shape"] == graph.outputs[0].shape
    rng = np.random.default_rng(1115)
    feeds = {name: rng.normal(0, .1, value.shape).astype(np.float16)
             for name, value in zip(("left", "right"), graph.inputs)}
    actual, = _execute(records, outputs, feeds)
    np.testing.assert_array_equal(actual, _reference(feeds["left"], feeds["right"]))


@pytest.mark.parametrize("left,right,output", [
    ((2, 3, 4, 5), (3, 2, 5, 6), (2, 3, 4, 6)),  # Same product does not mean paired batches.
    ((2, 1, 4, 5), (2, 3, 5, 6), (2, 3, 4, 6)),  # Broadcast requires a different lowering.
    ((1, 2, 4, 5), (2, 5, 6), (1, 2, 4, 6)),
    ((2, 3, 4, 5), (2, 3, 7, 6), (2, 3, 4, 6)),  # K mismatch.
    ((2, 3, 4, 5), (2, 3, 5, 6), (3, 2, 4, 6)),  # Output batch mismatch.
    ((2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 6, 4)),  # Output M/N mismatch.
])
def test_incompatible_shapes_are_rejected_before_serialization(tmp_path, left, right, output):
    with pytest.raises((ValueError, NotImplementedError)):
        emit_cpp(_graph(left, right, output_shape=output), recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("dtypes", [("int32", "float16", "float16"),
                                    ("float16", "float16", "int32")])
def test_mixed_dtypes_are_rejected(tmp_path, dtypes):
    graph = _graph((2, 3, 4, 5), (2, 3, 5, 6), left_dtype=dtypes[0],
                   right_dtype=dtypes[1], dtype=dtypes[2])
    with pytest.raises((ValueError, NotImplementedError)):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


def test_batch_product_checked_before_int32_narrowing(tmp_path):
    # Each individual dimension is legal; the flattened batch is not.
    batch = (46341, 46341)
    assert math.prod(batch) > (1 << 31) - 1
    graph = _graph(batch + (1, 1), batch + (1, 1))
    with pytest.raises(ValueError, match="INT32"):
        emit_cpp(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


def test_maximum_int32_batch_product_is_not_truncated(tmp_path):
    batch = (1, (1 << 31) - 1)
    records, _, _ = _serialize_both(_graph(batch + (1, 1), batch + (1, 1)), tmp_path)
    bmm, = [record for record in records if record["type"] == "BatchMatMulV2"]
    assert bmm["inputs"][0]["shape"][0] == (1 << 31) - 1
    # No data allocation occurs: this test inspects only small shape constants.


def test_existing_rank3_and_dense_projection_paths_remain_unchanged(tmp_path):
    rank3 = _graph((2, 3, 5), (2, 5, 4))
    records, _, _ = _serialize_both(rank3, tmp_path / "rank3")
    assert [record["type"] for record in records] == ["Data", "Data", "BatchMatMulV2"]
    dense = _graph((1, 2, 3, 5), (5, 4))
    records, _, _ = _serialize_both(dense, tmp_path / "dense")
    assert sum(record["type"] == "MatMul" for record in records) == 1
    assert sum(record["type"] == "Reshape" for record in records) == 2
    assert not any(record["type"] == "BatchMatMulV2" for record in records)


@pytest.mark.parametrize("transpose_b", [False, True])
def test_rank3_left_transpose_is_explicit_without_identity_reshapes(tmp_path, transpose_b):
    left_shape = (2, 5, 3)
    right_shape = (2, 4, 5) if transpose_b else (2, 5, 4)
    graph = _graph(left_shape, right_shape, transpose_a=True, transpose_b=transpose_b)
    records, _, outputs = _serialize_both(graph, tmp_path)
    assert not any(record["type"] == "Reshape" for record in records)
    assert sum(record["type"] == "Transpose" for record in records) == 1
    bmm, = [record for record in records if record["type"] == "BatchMatMulV2"]
    assert bmm["attrs"] == {"adj_x1": False, "adj_x2": transpose_b, "offset_x": 0}
    rng = np.random.default_rng(913)
    left = rng.normal(0, .2, left_shape).astype(np.float16)
    right = rng.normal(0, .2, right_shape).astype(np.float16)
    actual, = _execute(records, outputs, {"left": left, "right": right})
    np.testing.assert_array_equal(actual, _reference(left, right, True, transpose_b))


@pytest.mark.parametrize("sequence", [8, 65])
def test_all_tiny_pair_native_bmm_inputs_have_rank3(tmp_path, sequence):
    for graph in build_pair(*configurations(tiny=True), sequence):
        weights = {name: tmp_path / f"unopened-{index}.bin" for index, name in enumerate(graph.weights)}
        folder = tmp_path / graph.name
        emit_cpp(graph, weights, external_weights=True, fold_constant_arithmetic=False, recipe_dir=folder)
        records, _, _ = RecipeReader(folder / "graph.recipe").read()
        bmm = [record for record in records if record["type"] == "BatchMatMulV2"]
        assert bmm
        assert all(len(port["shape"]) == 3 for record in bmm for port in record["inputs"])
        assert all(record["inputs"][0]["shape"][0] == record["inputs"][1]["shape"][0]
                   for record in bmm)
        assert all(not record["attrs"]["adj_x1"] for record in bmm)


def test_serialized_bmm_subgraphs_preserve_tiny_pair_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, sequence_length=8, tiny=True)
    target, draft = build_pair(plan["target_config"], plan["draft_config"], 8)
    weights = {key: np.fromfile(prepared / item["path"], dtype="<f2").reshape(item["shape"])
               for key, item in plan["weights"].items()}
    calls = []

    def serialized_matmul(left, right, *args, **kwargs):
        if max(left.ndim, right.ndim) <= 3 or min(left.ndim, right.ndim) < 3:
            return _NUMPY_MATMUL(left, right, *args, **kwargs)
        assert not args and not kwargs
        assert left.dtype == right.dtype == np.float32
        # The oracle has already applied any transpose flags and promoted the
        # previous FP16 boundary. Recover those exact input values, then execute
        # the actual serialized replacement; all other graph ops stay unchanged.
        raw_left, raw_right = left.astype(np.float16), right.astype(np.float16)
        np.testing.assert_array_equal(raw_left.astype(np.float32), left)
        np.testing.assert_array_equal(raw_right.astype(np.float32), right)
        calls.append((left.shape, right.shape))
        graph = _graph(left.shape, right.shape)
        records, _, outputs = _serialize_both(graph, tmp_path / "bmm-subgraphs" / str(len(calls)))
        actual, = _execute(records, outputs, {"left": raw_left, "right": raw_right})
        return actual

    for prompt_length in (1, 3, 8):
        ids = np.zeros((1, 8), dtype=np.int32)
        mask = ids.copy()
        ids[0, :prompt_length] = [7, 3, 26, 8, 16, 42, 29, 31][:prompt_length]
        mask[0, :prompt_length] = 1
        inputs = {"input_ids": ids, "attention_mask": mask}
        expected_top1, expected_features = evaluate(target, inputs, weights)
        expected_draft, = evaluate(draft, {**inputs, "target_features": expected_features}, weights)
        with monkeypatch.context() as context:
            context.setattr(np, "matmul", serialized_matmul)
            actual_top1, actual_features = evaluate(target, inputs, weights)
            actual_draft, = evaluate(draft, {**inputs, "target_features": actual_features}, weights)
        np.testing.assert_array_equal(actual_top1, expected_top1)
        np.testing.assert_array_equal(actual_features, expected_features)
        np.testing.assert_array_equal(actual_draft, expected_draft)
    assert len(calls) == 3 * (23 + 4)

"""IR contracts and host-only serializer checks, without device claims."""
import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph


def test_tensor_ownership_reuse_and_dtype():
    g = Graph("ownership")
    x = g.input("x", (2, 3))
    w = g.weight("layer.weight", (3, 3))
    assert g.weight("layer.weight", (3, 3)) is w
    with pytest.raises(ValueError, match="inconsistent"):
        g.weight("layer.weight", (4, 3))
    with pytest.raises(ValueError, match="duplicate"):
        g.input("x", (2, 3))
    with pytest.raises(ValueError, match="FP16"):
        g.input("bad", (1,), "float32")
    other = Graph("other").input("x", (2, 3))
    with pytest.raises(ValueError, match="owned"):
        g.op("add", x, other, shape=(2, 3))


def test_masked_attention_fp16_rounding():
    g = Graph("attention")
    x = g.input("x", (1, 2, 3))
    m = g.input("mask", (1, 2, 3), "bool")
    masked = g.op("where", m, x, g.const(-np.inf), shape=x.shape)
    g.outputs = [g.op("softmax", masked, shape=x.shape, attrs={"axis": -1})]
    data = np.array([[[1, 2, 3], [-1, 0, 1]]], dtype=np.float16)
    mask = np.array([[[True, False, True], [False, True, True]]])
    y, = evaluate(g, {"x": data, "mask": mask})
    assert np.array_equal(y[~mask], [0, 0])
    np.testing.assert_allclose(y.sum(-1), 1, atol=0.001)
    assert y.dtype == np.float16
    assert "0,252" in emit_cpp(g)  # little-endian -inf, not finite sentinel.


def test_scalar_attributes_do_not_preround_to_half():
    g = Graph("scalar")
    x = g.input("x", (3,))
    g.outputs = [g.op("scalar_mul", x, shape=x.shape, attrs={"value": 8 ** -0.5})]
    a = np.array([0.125, 13, -2], dtype=np.float16)
    y, = evaluate(g, {"x": a})
    np.testing.assert_array_equal(y, (a.astype(np.float32) * np.float32(8 ** -0.5)).astype(np.float16))
    assert '"Power"' in emit_cpp(g)


def test_mixed_rank_matmul_and_topk_indices():
    g = Graph("projection")
    x = g.input("x", (1, 2, 3))
    w = g.weight("weight", (4, 3))
    y = g.op("matmul", x, w, shape=(1, 2, 4), attrs={"transpose_b": True})
    ids = g.op("topk_indices", y, shape=(1, 2, 2), dtype="int32", attrs={"k": 2, "axis": -1})
    g.outputs = [y, ids]
    a = np.array([[[1, 2, 3], [2, 3, 4]]], dtype=np.float16)
    weight = np.arange(12, dtype=np.float16).reshape(4, 3)
    actual, top = evaluate(g, {"x": a}, {"weight": weight})
    np.testing.assert_array_equal(actual, (a.astype(np.float32) @ weight.astype(np.float32).T).astype(np.float16))
    np.testing.assert_array_equal(top, [[[3, 2], [3, 2]]])
    code = emit_cpp(g, {"weight": "weight.bin"})
    assert '"MatMul"' in code and '"Reshape"' in code
    assert '"TopKV2"' in code and "{1}}" in code


def test_integral_gather_prefix_mask_and_argmax():
    g = Graph("indices")
    x = g.input("x", (2, 3), "int32")
    ids = g.input("indices", (2, 1), "int32")
    selected = g.op("gather_elements", x, ids, shape=(2, 1), dtype="int32", attrs={"axis": 1})
    summed = g.op("cumsum", x, shape=x.shape, dtype="int32", attrs={"axis": 1})
    g.outputs = [selected, summed]
    y, s = evaluate(g, {"x": np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32),
                        "indices": np.array([[0], [2]], dtype=np.int32)})
    np.testing.assert_array_equal(y, [[1], [1]])
    np.testing.assert_array_equal(s, [[1, 1, 2], [0, 1, 2]])


def test_external_weight_shape_type_attrs_and_big_embed_guard():
    g = Graph("external")
    x = g.input("x", (1, 3))
    w = g.weight("weight", (3, 3))
    g.outputs = [g.op("matmul", x, w, shape=(1, 3))]
    code = emit_cpp(g, {"weight": "weight.bin"}, external_weights=True)
    assert '"FileConstant"' in code
    assert '.SetAttr("dtype", ge::DT_FLOAT16)' in code
    assert '.SetAttr("shape", std::vector<int64_t>{3,3})' in code
    assert '.SetAttr("file_path", ge::AscendString(' in code
    assert "FindOpByName" not in code
    assert code.count("Verify(") == 3
    g2 = Graph("oversize")
    x = g2.input("x", (1,))
    g2.weight("large", (1_000_000_000,))
    g2.outputs = [x]
    with pytest.raises(ValueError, match="size limit"):
        emit_cpp(g2, {"large": "weight.bin"})


def test_unsupported_op_and_missing_weights_fail_before_cpp():
    g = Graph("invalid")
    x = g.input("x", (1,))
    g.outputs = [g.op("unknown", x, shape=(1,))]
    with pytest.raises(NotImplementedError):
        emit_cpp(g)
    with pytest.raises(NotImplementedError):
        evaluate(g, {"x": np.array([1], dtype=np.float16)})
    g.weight("needed", (1,))
    with pytest.raises(ValueError, match="missing weight"):
        emit_cpp(g)


def test_evaluator_rejects_input_dtype_and_shape():
    g = Graph("strict")
    x = g.input("x", (2,))
    g.outputs = [x]
    with pytest.raises(ValueError, match="dtype"):
        evaluate(g, {"x": np.ones(2, dtype=np.float32)})
    with pytest.raises(ValueError, match="expected"):
        evaluate(g, {"x": np.ones(3, dtype=np.float16)})


def test_topk_shared_node_dead_pruning_and_argmax_keepdims():
    g = Graph("shared")
    x = g.input("x", (1, 4))
    unused = g.const(0, name="unused_constant")
    g.op("add", x, unused, shape=x.shape, name="unused_node")
    values = g.op("topk_values", x, shape=(1, 2), attrs={"k": 2})
    indices = g.op("topk_indices", x, shape=(1, 2), dtype="int32", attrs={"k": 2})
    maximum = g.op("argmax", x, shape=(1, 1), dtype="int32", attrs={"axis": -1, "keepdims": True})
    g.outputs = [values, indices, maximum]
    code = emit_cpp(g)
    assert code.count('Op("' + values.name + '", "TopKV2"') == 1
    assert 'Op("' + indices.name + '", "TopKV2"' not in code
    assert "unused_constant" not in code and "unused_node" not in code
    assert '"ArgMaxV2"' in code and '"Reshape"' in code
    out = evaluate(g, {"x": np.array([[1, 4, 2, 3]], dtype=np.float16)})
    np.testing.assert_array_equal(out[0], [[4, 3]])
    np.testing.assert_array_equal(out[1], [[1, 3]])
    np.testing.assert_array_equal(out[2], [[1]])


@pytest.mark.parametrize("kind", ["concat", "slice", "transpose", "where"])
def test_bool_basic_type_operations_use_exact_integer_carriers(kind, tmp_path):
    g = Graph("boolean")
    x = g.input("x", (2, 2), "bool")
    if kind == "concat":
        y = g.op(kind, x, x, shape=(2, 4), dtype="bool", attrs={"axis": 1})
    elif kind == "slice":
        y = g.op(kind, x, shape=(1, 2), dtype="bool", attrs={"starts": [0, 0], "sizes": [1, 2]})
    elif kind == "transpose":
        y = g.op(kind, x, shape=(2, 2), dtype="bool", attrs={"perm": [1, 0]})
    else:
        y = g.op(kind, x, x, g.const(False, "bool"), shape=(2, 2), dtype="bool")
    g.outputs = [y]
    code = emit_cpp(g)
    assert f'"__ge/{y.name}/integer_output"' in code
    assert '.SetAttr("dst_type", int64_t{ge::DT_INT32})' in code
    assert '.SetAttr("dst_type", int64_t{ge::DT_BOOL})' in code


def test_large_graph_requires_recipe_before_source_generation():
    g = Graph("large")
    value = g.input("x", (1,))
    for _ in range(10_001):
        value = g.op("neg", value, shape=(1,))
    g.outputs = [value]
    with pytest.raises(ValueError, match="recipe_dir"):
        emit_cpp(g)

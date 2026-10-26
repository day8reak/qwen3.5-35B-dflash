"""Check emitted row-Mul layouts; these are host contracts, not board results."""
from collections import Counter
import hashlib
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.hashing import sha256_file
from test_ge_reduction_lowering import _execute
from test_ge_select_lowering import RecipeReader


SHAPES = [((1, 8, 2, 8), (1, 8, 2, 1), 16),
          ((1, 2, 64, 8), (1, 2, 64, 1), 128)]


def _graph(left_shape=SHAPES[0][0], right_shape=SHAPES[0][1], dtype="float16",
           output_shape=None, output_dtype=None):
    graph = Graph("row_mul_lowering")
    left = graph.input("left", left_shape, dtype)
    right = graph.input("right", right_shape, dtype)
    graph.outputs = [graph.op("mul", left, right,
                             shape=left_shape if output_shape is None else output_shape,
                             dtype=dtype if output_dtype is None else output_dtype, name="result")]
    return graph


def _recipe(graph, folder, **kwargs):
    report = {"stale": True}
    emit_cpp(graph, recipe_dir=folder, fp16_row_mul="compact", row_mul_report=report, **kwargs)
    records, inputs, outputs = RecipeReader(folder / "graph.recipe").read()
    assert "stale" not in report
    return records, inputs, outputs, report


def _assert_bits_and_classes(actual, expected):
    assert actual.dtype == expected.dtype == np.float16 and actual.shape == expected.shape
    for classification in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(classification(actual), classification(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(actual[finite].view(np.uint16), expected[finite].view(np.uint16))


def _reference(left, right):
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return (left.astype(np.float32) * right.astype(np.float32)).astype(np.float16)


@pytest.mark.parametrize("left_shape,right_shape,rows", SHAPES)
def test_two_exact_contracts_keep_recipe_ports_output_and_graph(tmp_path, left_shape, right_shape, rows):
    graph = _graph(left_shape, right_shape)
    original = pickle.dumps(graph)
    records, inputs, outputs, report = _recipe(graph, tmp_path / "recipe")
    assert pickle.dumps(graph) == original
    assert report["mode"] == "compact" and report["graph"] == graph.name
    assert report["counts"] == {"total": 1, "replaced": 1, "retained": 0}
    assert len(report["replaced"]) == 1 and not report["retained"]
    assert Counter(r["type"] for r in records) == {"Data": 2, "Const": 3, "Reshape": 3, "Mul": 1}
    assert [records[i]["name"] for i in inputs] == ["left", "right"]
    operation, = [r for r in records if r["type"] == "Mul"]
    assert [p["port"] for p in operation["inputs"]] == ["x1", "x2"]
    assert [p["shape"] for p in operation["inputs"]] == [(rows, 8), (rows, 1)]
    assert all(p["dtype"] == "float16" for p in operation["inputs"] + operation["outputs"])
    for port, expected_input in zip(operation["inputs"], inputs):
        view = records[port["src"]]
        assert view["type"] == "Reshape" and view["inputs"][0]["src"] == expected_input
    output, = outputs
    assert records[output[0]]["name"] == "result"
    assert records[output[0]]["outputs"][output[1]]["shape"] == left_shape
    for view in (r for r in records if r["type"] == "Reshape"):
        assert view["inputs"][1]["dtype"] == "int32"
    left = (((np.arange(rows * 8) * 7) % 127 - 63) / 32).astype(np.float16).reshape(left_shape)
    right = (((np.arange(rows) * 11) % 61 - 30) / 16).astype(np.float16).reshape(right_shape)
    actual, = _execute(records, outputs, {"left": left, "right": right})
    _assert_bits_and_classes(actual, _reference(left, right))
    unrolled_report = {}
    unrolled = emit_cpp(graph, fp16_row_mul="compact", row_mul_report=unrolled_report)
    assert unrolled_report == report
    assert unrolled.count(', "Mul",') == 1 and unrolled.count(', "Reshape",') == 3
    assert pickle.dumps(graph) == original


@pytest.mark.parametrize("left_shape,right_shape,rows", SHAPES)
def test_signed_zero_subnormal_overflow_and_nonfinite_positions(tmp_path, left_shape, right_shape, rows):
    graph = _graph(left_shape, right_shape)
    records, _, outputs, _ = _recipe(graph, tmp_path / "recipe")
    bits = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0x7bff, 0x7c00, 0x7e01], np.uint16)
    left = np.broadcast_to(bits.view(np.float16), (rows, 8)).copy().reshape(left_shape)
    scales = np.array([0., -0., 1., -1., 2., -2., .5, -.5, np.inf, -np.inf, np.nan, 1., -1., 2., .5, 1.], np.float16)
    right = np.tile(scales, rows // 16).reshape(right_shape)
    before = (left.tobytes(), right.tobytes())
    actual, = _execute(records, outputs, {"left": left, "right": right})
    expected = _reference(left, right)
    _assert_bits_and_classes(actual, expected)
    assert np.isnan(actual).any() and np.isposinf(actual).any() and np.isneginf(actual).any()
    assert np.signbit(actual[actual == 0]).any() and (~np.signbit(actual[actual == 0])).any()
    assert np.any((np.abs(actual) < np.finfo(np.float16).tiny) & (actual != 0))
    assert before == (left.tobytes(), right.tobytes())


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_non_fp16_stays_native_without_casts(dtype):
    graph = _graph(dtype=dtype)
    original, report = pickle.dumps(graph), {}
    native = emit_cpp(graph)
    assert emit_cpp(graph, fp16_row_mul="compact", row_mul_report=report) == native
    assert report["counts"] == {"total": 1, "replaced": 0, "retained": 1}
    assert report["retained"][0]["reason"] == "non_fp16"
    assert pickle.dumps(graph) == original


@pytest.mark.parametrize("left_shape,right_shape,output_shape", [
    ((1, 8, 2, 7), (1, 8, 2, 1), (1, 8, 2, 7)),
    ((1, 7, 2, 8), (1, 7, 2, 1), (1, 7, 2, 8)),
    ((1, 2, 63, 8), (1, 2, 63, 1), (1, 2, 63, 8)),
    ((2, 1, 64, 8), (2, 1, 64, 1), (2, 1, 64, 8)),
    ((16, 8), (16, 1), (16, 8)),
    ((1, 8, 2, 1), (1, 8, 2, 8), (1, 8, 2, 8)),
    ((1, 8, 2, 8), (1, 8, 2, 8), (1, 8, 2, 8)),
    ((1, 8, 2, 8), (8,), (1, 8, 2, 8)),
    ((0, 8, 2, 8), (0, 8, 2, 1), (0, 8, 2, 8)),
])
def test_unprobed_shapes_remain_byte_identical_to_native(left_shape, right_shape, output_shape):
    graph = _graph(left_shape, right_shape, output_shape=output_shape)
    report = {}
    assert emit_cpp(graph, fp16_row_mul="compact", row_mul_report=report) == emit_cpp(graph)
    assert report["counts"] == {"total": 1, "replaced": 0, "retained": 1}
    assert report["retained"][0]["reason"] == "outside_verified_shapes"


@pytest.mark.parametrize("output_shape,output_dtype", [((16, 8), "float16"), (SHAPES[0][0], "int32")])
def test_malformed_verified_output_fails_before_recipe_write(tmp_path, output_shape, output_dtype):
    graph = _graph(output_shape=output_shape, output_dtype=output_dtype)
    original = pickle.dumps(graph)
    with pytest.raises(ValueError, match="output shape/dtype"):
        _recipe(graph, tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists() and pickle.dumps(graph) == original


@pytest.mark.parametrize("source_kind", ["pending_constant", "folded_reshape", "embedded_weight", "external_weight"])
def test_rhs_constant_and_weight_references_materialize_in_order(tmp_path, source_kind):
    graph = Graph("row_mul_source")
    left = graph.input("left", SHAPES[0][0])
    scale = np.arange(16, dtype=np.float16).reshape(SHAPES[0][1]) / np.float16(8)
    paths, kwargs = {}, {}
    if source_kind == "pending_constant":
        right = graph.const(scale, name="right")
    elif source_kind == "folded_reshape":
        raw = graph.const(scale.ravel(), name="raw")
        right = graph.op("reshape", raw, shape=scale.shape, attrs={"shape": list(scale.shape)}, name="right")
    else:
        right = graph.weight("right", scale.shape)
        path = tmp_path / "weight.bin"
        path.write_bytes(scale.tobytes())
        paths["right"] = path
        kwargs["external_weights"] = source_kind == "external_weight"
    graph.outputs = [graph.op("mul", left, right, shape=left.shape, name="result")]
    original = pickle.dumps(graph)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    records, _, outputs, report = _recipe(graph, tmp_path / "recipe", weight_paths=paths, **kwargs)
    assert report["counts"]["replaced"] == 1 and pickle.dumps(graph) == original
    for index, record in enumerate(records):
        assert all(port["src"] < index for port in record["inputs"])
    operation, = [r for r in records if r["type"] == "Mul"]
    rhs = records[operation["inputs"][1]["src"]]
    source = records[rhs["inputs"][0]["src"]]
    assert source["type"] == ("FileConstant" if source_kind == "external_weight" else "Const")
    assert source["outputs"][0]["shape"] == scale.shape
    assert all(sha256_file(paths[name]) == digest for name, digest in hashes.items())
    if source_kind != "external_weight":
        values = np.arange(128, dtype=np.float16).reshape(left.shape) / np.float16(16)
        actual, = _execute(records, outputs, {"left": values})
        _assert_bits_and_classes(actual, _reference(values, scale))


def test_default_and_explicit_native_cpp_are_identical_and_clear_report():
    graph = _graph()
    original, report = pickle.dumps(graph), {"stale": True}
    implicit = emit_cpp(graph)
    assert emit_cpp(graph, fp16_row_mul="native", row_mul_report=report) == implicit
    assert ', "Reshape",' not in implicit
    assert report["mode"] == "native" and "stale" not in report
    assert report["counts"] == {"total": 1, "replaced": 0, "retained": 1}
    assert pickle.dumps(graph) == original


@pytest.mark.parametrize("left_shape,right_shape,dtype,digest", [
    (*SHAPES[0][:2], "float16", "c595bade2b313a29528af4407ea48668fe514e369e3724e114a3da4a7e14341b"),
    (*SHAPES[1][:2], "float16", "a113d9291e553aafa0104a3ec9fc1d39816e96d5a601fc82cf2f343979912c0c"),
    (*SHAPES[0][:2], "int32", "d670719e3819b1c8264e875a9e8fd818afdb00e0d20fa2aec422a65ead3d1a38"),
    ((2, 1, 64, 8), (2, 1, 64, 1), "float16", "1d63ebc87938c0f0ffac201bfc8624f12f7a3c2cc1611590513f9fa3251cca97"),
])
def test_default_cpp_matches_recorded_prechange_bytes(left_shape, right_shape, dtype, digest):
    # Recorded with the parent revision before row-Mul emission was added.
    graph = _graph(left_shape, right_shape, dtype)
    graph.name = "row_mul_identity"
    code = emit_cpp(graph)
    assert hashlib.sha256(code.encode()).hexdigest() == digest
    assert emit_cpp(graph, fp16_row_mul="native") == code


@pytest.mark.parametrize("name", [
    "__ge/result/row_mul/left", "__ge/result/row_mul/right", "__ge/result/row_mul/product",
    "__ge/__ge/result/row_mul/left/shape", "__ge/__ge/result/row_mul/right/shape",
    "__ge/result/shape",
])
def test_generated_name_collision_rejected_without_overwriting_graph_or_recipe(tmp_path, name):
    graph = _graph()
    # Even an unused original value owns its name; it must not be overwritten.
    graph.const([1], dtype="int32", name=name)
    original = pickle.dumps(graph)
    with pytest.raises(ValueError, match="generated name collision"):
        _recipe(graph, tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()
    assert pickle.dumps(graph) == original


@pytest.mark.parametrize("mode", ["fp32", "COMPACT", None, 0, []])
def test_invalid_mode_fails_before_recipe_directory(tmp_path, mode):
    with pytest.raises(ValueError, match="fp16_row_mul"):
        emit_cpp(_graph(), recipe_dir=tmp_path / "recipe", fp16_row_mul=mode)
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("report", [[], "report", 0])
def test_invalid_report_type_fails_before_recipe_directory(tmp_path, report):
    with pytest.raises(TypeError, match="row_mul_report"):
        emit_cpp(_graph(), recipe_dir=tmp_path / "recipe", row_mul_report=report)
    assert not (tmp_path / "recipe").exists()

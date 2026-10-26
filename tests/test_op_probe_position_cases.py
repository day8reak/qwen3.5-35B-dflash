"""Independent finite-value and serialized-ABI checks; no OMG or board run."""
from pathlib import Path
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import DTYPES
from qwen35_moe_dflash.op_probe_position_cases import position_cases
from test_ge_select_lowering import RecipeReader, _assert_select_contract


NAMES = (
    "rope_frequency_bmm", "rope_frequency_matmul2d", "rope_frequency_outer_mul",
    "position_cast_tiny_native", "position_cast_tiny_compact",
    "position_cast_rounding_native", "position_cast_rounding_compact",
    "attention_mask_and_select", "attention_mask_reshape_before_cast",
    "attention_mask_sequential_select", "attention_mask_lut_int32", "attention_mask_lut_fp16",
)
MASK_NAMES = NAMES[7:]


def _case(name):
    return next(case for case in position_cases() if case["name"] == name)


def _constant(record):
    path, offset, size = record["constant"]
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        data = stream.read(size)
    assert len(data) == size
    output, = record["outputs"]
    return np.frombuffer(data, DTYPES[output["dtype"]]).reshape(output["shape"])


def _execute(records, outputs, feeds):
    """Execute serialized records separately from the neutral-IR evaluator."""
    values = {}
    for index, record in enumerate(records):
        args = [values[(p["src"], p["src_port"])] for p in record["inputs"]]
        output, = record["outputs"]
        kind, attrs = record["type"], record["attrs"]
        if kind == "Data":
            result = feeds[record["name"]]
        elif kind == "Const":
            result = _constant(record)
        elif kind == "Reshape":
            assert args[1].dtype == np.int32
            result = args[0].reshape(tuple(args[1]))
        elif kind == "Cast":
            dtype = {1: "float16", 3: "int32", 9: "int64", 12: "bool"}[attrs["dst_type"]]
            assert dtype == output["dtype"]
            result = args[0].astype(DTYPES[dtype])
        elif kind in {"MatMul", "BatchMatMulV2"}:
            left, right = (arg.astype(np.float64) for arg in args)
            assert left.shape[-1] == right.shape[-2] == 1
            assert not attrs["transpose_x1" if kind == "MatMul" else "adj_x1"]
            assert not attrs["transpose_x2" if kind == "MatMul" else "adj_x2"]
            result = np.matmul(left, right)
        elif kind == "Mul":
            result = args[0].astype(np.float64) * args[1].astype(np.float64)
        elif kind == "LogicalAnd":
            result = np.logical_and(*args)
        elif kind == "GatherV2":
            assert args[1].dtype == np.int32 and args[2].dtype == np.int32
            assert np.all((args[1] == 0) | (args[1] == 1))
            result = np.take(args[0], args[1], axis=int(args[2]))
        elif kind == "BroadcastTo":
            result = np.broadcast_to(args[0], tuple(args[1]))
        elif kind == "SelectV2":
            assert args[0].dtype == np.bool_
            assert args[0].shape == args[1].shape == args[2].shape
            result = np.where(*args)
        else:
            raise AssertionError(f"unexpected position recipe operation {kind}")
        result = np.asarray(result, DTYPES[output["dtype"]])
        assert result.shape == output["shape"]
        values[(index, output["port"])] = result
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


def _recipe(case, folder):
    before = pickle.dumps(case["graph"])
    emit_cpp(case["graph"], recipe_dir=folder)
    assert pickle.dumps(case["graph"]) == before
    records, inputs, outputs = RecipeReader(folder / "graph.recipe").read()
    assert [records[index]["name"] for index in inputs] == list(case["inputs"])
    for index in inputs:
        assert records[index]["type"] == "Data"
    return records, outputs


def _mask_expected(mask):
    # Independent scalar predicate makes key/query axis reversal observable.
    result = np.full((1, 1, 8, 8), -65504, np.float16)
    for query in range(8):
        for key in range(8):
            if key <= query and mask[0, key] != 0:
                result[0, 0, query, key] = np.float16(0)
    return result


def test_catalogue_is_fresh_deterministic_bounded_and_has_no_external_bool():
    first, second = list(position_cases()), list(position_cases())
    assert tuple(case["name"] for case in first) == NAMES
    assert len({case["name"] for case in first}) == 12
    for left, right in zip(first, second):
        graph = left["graph"]
        assert graph is not right["graph"] and not graph.weights
        assert all(v.dtype in {"float16", "int32"} for v in graph.inputs + graph.outputs)
        assert set(left["inputs"]) == {v.name for v in graph.inputs}
        assert all(np.isfinite(a).all() for a in left["expected"])
        assert sum(a.nbytes for a in left["inputs"].values()) <= 128
        assert all(a.nbytes <= 128 for a in left["expected"])
        for name, data in left["inputs"].items():
            assert data.tobytes() == right["inputs"][name].tobytes()
            assert not np.shares_memory(data, right["inputs"][name])
        assert "SaveToFile" in emit_cpp(graph)


@pytest.mark.parametrize("name,kind,shape", [
    (NAMES[0], "BatchMatMulV2", (1, 2, 8)),
    (NAMES[1], "MatMul", (2, 8)),
    (NAMES[2], "Mul", (1, 2, 8)),
])
def test_rope_single_product_uses_original_io_and_independent_fp16_oracle(name, kind, shape, tmp_path):
    case = _case(name)
    assert [(v.shape, v.dtype) for v in case["graph"].inputs] == [
        ((1, 2, 1), "float16"), ((1, 1, 8), "float16")]
    feeds = case["inputs"]
    expected = (feeds["inverse"].astype(np.float64) * feeds["positions"].astype(np.float64)).astype(np.float16)
    assert case["equivalence_group"] == "rope_frequency_k1"
    assert case["expected"][0].tobytes() == expected.tobytes()
    records, outputs = _recipe(case, tmp_path / "recipe")
    arithmetic, = [r for r in records if r["type"] in {"MatMul", "BatchMatMulV2", "Mul"}]
    assert arithmetic["type"] == kind and arithmetic["outputs"][0]["shape"] == shape
    assert all(p["dtype"] == "float16" for p in arithmetic["inputs"] + arithmetic["outputs"])
    assert not any(r["type"] in {"Cast", "Add", "Power", "ReduceSum"} for r in records)
    actual, = _execute(records, outputs, feeds)
    assert actual.tobytes() == expected.tobytes()
    assert records[outputs[0][0]]["outputs"][0]["shape"] == (1, 2, 8)
    # Change both dynamic payloads; the serialized arithmetic must stay live.
    changed = {"inverse": np.array([0.375, -0.8125], np.float16).reshape(1, 2, 1),
               "positions": np.arange(1, 9, dtype=np.float16).reshape(1, 1, 8)}
    oracle = (changed["inverse"].astype(np.float64) * changed["positions"].astype(np.float64)).astype(np.float16)
    assert _execute(records, outputs, changed)[0].tobytes() == oracle.tobytes()


@pytest.mark.parametrize("name", NAMES[3:7])
def test_position_cast_layout_keeps_int32_input_and_expected_rounding(name, tmp_path):
    case = _case(name)
    expected = (np.arange(8, dtype=np.float16) if "tiny" in name else
                np.array([2047, 2048, 2048, 2052, 4096, 4096, 4096, 65504], np.float16)).reshape(1, 8)
    assert case["expected"][0].tobytes() == expected.tobytes()
    assert case["tolerances"] == [{"rtol": 0.0, "atol": 0.0}]
    records, outputs = _recipe(case, tmp_path / "recipe")
    cast, = [r for r in records if r["type"] == "Cast"]
    assert cast["inputs"][0]["dtype"] == "int32" and cast["outputs"][0]["dtype"] == "float16"
    assert cast["inputs"][0]["shape"] == ((8,) if "compact" in name else (1, 8))
    assert [r["type"] for r in records].count("Reshape") == (2 if "compact" in name else 0)
    assert _execute(records, outputs, case["inputs"])[0].tobytes() == expected.tobytes()
    if "rounding" in name:
        assert np.any(expected.astype(np.int32) != case["inputs"]["positions"])
        assert "lose integer precision" in case["description"]


@pytest.mark.parametrize("name", MASK_NAMES)
@pytest.mark.parametrize("fixture", ["valid", "invalid", "mixed"])
def test_mask_recipes_match_original_selection_for_all_binary_patterns(name, fixture, tmp_path):
    case = _case(name)
    mask = {"valid": np.ones((1, 8), np.int32), "invalid": np.zeros((1, 8), np.int32),
            "mixed": np.array([[0, 1, 0, 1, 1, 0, 1, 0]], np.int32)}[fixture]
    expected = _mask_expected(mask)
    assert case["input_domain"] == {"attention_mask": {"allowed_values": [0, 1]}}
    assert case["expected"][0].tobytes() == _mask_expected(case["inputs"]["attention_mask"]).tobytes()
    assert case["tolerances"] == [{"rtol": 0.0, "atol": 0.0}]
    records, outputs = _recipe(case, tmp_path / "recipe")
    _assert_select_contract(records)
    feeds = {"attention_mask": mask}
    assert _execute(records, outputs, feeds)[0].tobytes() == expected.tobytes()
    assert evaluate(case["graph"], feeds)[0].tobytes() == expected.tobytes()
    assert not any(r["type"] in {"Mul", "Add", "MatMul", "SoftmaxV2"} for r in records)
    assert all(p["dtype"] in DTYPES for r in records for p in r["inputs"] + r["outputs"])


@pytest.mark.parametrize("name", MASK_NAMES)
def test_mask_variants_expose_actual_cast_view_and_lut_contracts(name, tmp_path):
    case = _case(name)
    records, _ = _recipe(case, tmp_path / "recipe")
    types = [r["type"] for r in records]
    assert types.count("Data") == 1
    if name.endswith("and_select"):
        assert types.count("LogicalAnd") == 1 and types.count("SelectV2") == 1
        view, = [r for r in records if r["name"] == "valid_view"]
        assert view["type"] == "Reshape" and view["inputs"][0]["dtype"] == "bool"
        assert view["inputs"][0]["shape"] == (1, 8) and view["outputs"][0]["shape"] == (1, 1, 1, 8)
    elif name.endswith("reshape_before_cast"):
        view, = [r for r in records if r["name"] == "integer_mask_view"]
        assert view["inputs"][0]["dtype"] == "int32"
        cast, = [r for r in records if r["name"] == "valid"]
        assert cast["inputs"][0]["shape"] == (1, 1, 1, 8) and cast["outputs"][0]["dtype"] == "bool"
    elif name.endswith("sequential_select"):
        assert types.count("SelectV2") == 2 and "LogicalAnd" not in types
    else:
        assert "LogicalAnd" not in types and types.count("GatherV2") == 1 and types.count("SelectV2") == 1
        assert "ONLY" in case["description"] and "arbitrary nonzero" in case["description"]
        gather, = [r for r in records if r["type"] == "GatherV2"]
        table = records[gather["inputs"][0]["src"]]
        assert gather["inputs"][1]["dtype"] == "int32"
        dtype = np.int32 if name.endswith("int32") else np.float16
        np.testing.assert_array_equal(_constant(table), np.array([-65504, 0], dtype))
        assert types.count("Cast") == (1 if dtype == np.int32 else 0)


@pytest.mark.parametrize("name", MASK_NAMES[:3])
def test_non_lookup_mask_variants_preserve_nonzero_cast_semantics(name, tmp_path):
    case = _case(name)
    mask = np.array([[-1, 0, 2, -17, 0, 2147483647, 0, -2147483648]], np.int32)
    records, outputs = _recipe(case, tmp_path / "recipe")
    expected = _mask_expected(mask)
    assert _execute(records, outputs, {"attention_mask": mask})[0].tobytes() == expected.tobytes()

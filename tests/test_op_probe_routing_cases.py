"""Independent tiny-routing gold and emitted recipes, without board claims."""
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import DTYPES
from qwen35_moe_dflash.op_probe_routing_cases import ROUTING_CASE_NAMES, routing_cases
from test_ge_select_lowering import RecipeReader


def _suite():
    return {case["name"]: case for case in routing_cases()}


def _route_gold(inputs):
    result = np.zeros((8, 16), np.float16)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for expert in range(4):
            selected = inputs["ids"] == expert
            routed = np.where(selected, inputs["weights"], np.float16(0))
            scale = routed.astype(np.float32).sum(axis=1, keepdims=True).astype(np.float16)
            contribution = (inputs[f"expert_{expert}"].astype(np.float32)
                            * scale.astype(np.float32)).astype(np.float16)
            contribution = np.where(selected.any(axis=1, keepdims=True), contribution, np.float16(0))
            result = (result.astype(np.float32) + contribution.astype(np.float32)).astype(np.float16)
    return result


def _gold(name, inputs):
    if name == "moe_equal_cast_int32":
        return [(inputs["ids"] == 2).astype(np.int32)]
    if name == "moe_cast_bool_roundtrip":
        return [inputs["membership"].copy()]
    if name == "moe_count_reduce_int32":
        return [inputs["membership"].sum(axis=1, keepdims=True, dtype=np.int32)]
    if name == "moe_active_greater_int32":
        return [(inputs["counts"] > 0).astype(np.int32)]
    if name.startswith("moe_gate_"):
        return [np.where(inputs["active"] != 0, inputs["values"], np.float16(0))]
    if name.startswith("moe_route_"):
        return [_route_gold(inputs)]
    result = []
    for expert in range(4):
        selected = inputs["ids"] == expert
        if name.startswith("moe_membership_"):
            value = selected.astype(np.int32)
        elif name.startswith("moe_count_"):
            value = selected.sum(axis=1, keepdims=True, dtype=np.int32)
        elif name.startswith("moe_active_"):
            value = selected.any(axis=1, keepdims=True).astype(np.int32)
        else:
            assert name.startswith("moe_weights_")
            value = np.where(selected, inputs["weights"], np.float16(0))
        result.append(value)
    return result


def _assert_equal_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def test_batch_is_bounded_deterministic_and_records_its_integer_domains():
    first, second = list(routing_cases()), list(routing_cases())
    assert tuple(case["name"] for case in first) == ROUTING_CASE_NAMES
    assert len(first) == len(set(ROUTING_CASE_NAMES)) == 16
    for case, repeated in zip(first, second):
        graph = case["graph"]
        assert not graph.weights and len(graph.nodes) < 150
        assert set(case["inputs"]) == {v.name for v in graph.inputs}
        assert all(v.dtype in {"float16", "int32"} for v in graph.inputs + graph.outputs)
        assert all(np.isfinite(value).all() for value in case["expected"])
        assert json.loads(json.dumps(case["input_domain"])) == case["input_domain"]
        for name, value in case["inputs"].items():
            _assert_equal_bits(value, repeated["inputs"][name])
            if value.dtype == np.int32:
                assert name in case["input_domain"]
                assert set(value.ravel()) <= set(case["input_domain"][name]["allowed_values"])
        expected_tolerance = ({"rtol": .01, "atol": .002}
                              if case["name"].startswith("moe_route_") else {"rtol": 0., "atol": 0.})
        assert case["tolerances"] == [expected_tolerance] * len(case["expected"])
    ids, weights = first[-1]["inputs"]["ids"], first[-1]["inputs"]["weights"]
    assert ids.dtype == np.int32 and ids.shape == weights.shape == (8, 2)
    assert np.all(ids[:, 0] != ids[:, 1])
    np.testing.assert_array_equal(weights.astype(np.float32).sum(axis=1), np.ones(8))
    for expert in range(4):
        selected = ids == expert
        assert np.any(selected & (weights == 0)) and np.any(selected & (weights > 0))


@pytest.mark.parametrize("name", ROUTING_CASE_NAMES)
def test_references_and_candidate_graphs_match_independent_routing_gold(name):
    case = _suite()[name]
    gold = _gold(name, case["inputs"])
    for actual, reference, expected in zip(evaluate(case["graph"], case["inputs"]), case["expected"], gold):
        _assert_equal_bits(reference, expected)
        _assert_equal_bits(actual, expected)


def _execute_recipe(records, outputs, feeds):
    values = {}
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for index, record in enumerate(records):
            output, = record["outputs"]
            args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
            math_args = [a.astype(np.float32) if a.dtype == np.float16 else a for a in args]
            kind, attrs = record["type"], record["attrs"]
            if kind == "Data":
                value = feeds[record["name"]]
            elif kind == "Const":
                path, offset, size = record["constant"]
                with Path(path).open("rb") as stream:
                    stream.seek(offset)
                    raw = stream.read(size)
                assert len(raw) == size
                value = np.frombuffer(raw, DTYPES[output["dtype"]]).reshape(output["shape"])
            elif kind == "Cast":
                assert output["dtype"] in {"int32", "bool"}  # Never encode IDs as floats.
                value = args[0].astype(DTYPES[output["dtype"]])
            elif kind in {"Equal", "Greater"}:
                assert args[0].dtype == args[1].dtype == np.int32
                value = (np.equal if kind == "Equal" else np.greater)(*args)
            elif kind in {"Add", "Mul"}:
                value = (np.add if kind == "Add" else np.multiply)(*math_args)
            elif kind == "ReduceSum":
                assert args[0].dtype == args[1].dtype == np.int32
                value = np.sum(args[0], axis=tuple(args[1]), keepdims=bool(attrs["keep_dims"]))
            elif kind == "Slice":
                value = args[0][tuple(slice(int(a), int(a + b)) for a, b in zip(args[1], args[2]))]
            elif kind == "Reshape":
                assert args[1].dtype == np.int32
                value = args[0].reshape(tuple(args[1]))
            elif kind == "ConcatD":
                assert attrs["N"] == len(args) == 2 and attrs["concat_dim"] == 0
                value = np.concatenate(args, axis=0)
            elif kind == "GatherV2":
                assert args[1].dtype == args[2].dtype == np.int32 and args[2].shape == ()
                assert int(args[2]) == 0 and attrs["batch_dims"] == 0
                assert np.all(args[1] >= 0) and np.all(args[1] < args[0].shape[0])
                value = np.take(args[0], args[1], axis=0)
            elif kind == "BroadcastTo":
                value = np.broadcast_to(args[0], tuple(args[1]))
            elif kind == "SelectV2":
                assert args[0].dtype == np.bool_
                assert args[0].shape == args[1].shape == args[2].shape
                value = np.where(*args)
            elif kind == "MatMul":
                assert args[0].shape == (8, 2) and args[1].shape == (1, 2)
                assert attrs["transpose_x1"] == 0 and attrs["transpose_x2"] == 1
                np.testing.assert_array_equal(args[1], np.ones((1, 2), np.float16))
                value = math_args[0] @ math_args[1].T
            else:
                raise AssertionError(f"unhandled routing recipe op {kind}")
            value = np.asarray(value, dtype=DTYPES[output["dtype"]])
            assert value.shape == output["shape"]
            values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


@pytest.mark.parametrize("name", ROUTING_CASE_NAMES)
def test_actual_recipes_execute_integer_indices_and_original_fp16_boundaries(name, tmp_path):
    case = _suite()[name]
    emit_cpp(case["graph"], recipe_dir=tmp_path / "recipe")
    records, _, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    for actual, expected in zip(_execute_recipe(records, outputs, case["inputs"]), _gold(name, case["inputs"])):
        _assert_equal_bits(actual, expected)
    if name == "moe_route_native":
        assert sum(r["type"] == "Equal" for r in records) == 8
        # The active Greater BOOL goes directly to its BroadcastTo/Select;
        # no artificial BOOL -> INT32 -> BOOL roundtrip in the full context.
        for r in records:
            if r["type"] == "Cast" and r["inputs"][0]["dtype"] == "bool":
                assert records[r["inputs"][0]["src"]]["type"] == "Equal"
    if name == "moe_gate_select":
        assert any(r["type"] == "BroadcastTo" and r["inputs"][0]["dtype"] == "bool" for r in records)
    if name in {"moe_weights_gather", "moe_gate_gather", "moe_route_gather"}:
        assert not any(r["type"] in {"Cast", "CastT", "Equal", "Greater", "SelectV2", "BroadcastTo"} for r in records)
    if name.startswith("moe_route_"):
        assert sum(r["type"] == "MatMul" for r in records) == 4
        assert not any(r["type"] == "ReduceSum" and r["inputs"][0]["dtype"] == "float16" for r in records)


@pytest.mark.parametrize("stage", ["membership", "count", "active", "weights", "route"])
def test_all_sixteen_ordered_id_pairs_including_duplicate_experts(stage):
    suite = _suite()
    all_pairs = np.array([(a, b) for a in range(4) for b in range(4)], np.int32)
    for suffix in ("native", "gather"):
        case = suite[f"moe_{stage}_{suffix}"]
        for ids in (all_pairs[:8], all_pairs[8:]):
            inputs = {**case["inputs"], "ids": ids.copy()}
            for actual, expected in zip(evaluate(case["graph"], inputs), _gold(case["name"], inputs)):
                _assert_equal_bits(actual, expected)


@pytest.mark.parametrize("stage", ["weights", "gate"])
def test_host_copy_paths_preserve_nan_payloads_signed_zero_and_exclude_nonfinite(stage, tmp_path):
    bits = np.array([0x7e31, 0x8000, 0x7c00, 0xfc00, 0x7d03, 0xfe11, 1, 0x8001], np.uint16)
    for suffix in (("native", "gather") if stage == "weights" else ("select", "gather")):
        case = _suite()[f"moe_{stage}_{suffix}"]
        inputs = dict(case["inputs"])
        key, shape = ("weights", (8, 2)) if stage == "weights" else ("values", (8, 16))
        inputs[key] = np.resize(bits, shape).view(np.float16)
        before = inputs[key].tobytes()
        emit_cpp(case["graph"], recipe_dir=tmp_path / suffix)
        records, _, outputs = RecipeReader(tmp_path / suffix / "graph.recipe").read()
        for actuals in (evaluate(case["graph"], inputs), _execute_recipe(records, outputs, inputs)):
            for actual, expected in zip(actuals, _gold(case["name"], inputs)):
                _assert_equal_bits(actual, expected)
        assert inputs[key].tobytes() == before


@pytest.mark.parametrize("suffix", ["native", "gather"])
def test_host_full_route_keeps_selected_zero_active_and_masks_unselected_nan(suffix):
    case = _suite()[f"moe_route_{suffix}"]
    # Expert 2 is selected at zero weight on row 4 but absent on row 0.
    inputs = {name: value.copy() for name, value in case["inputs"].items()}
    inputs["expert_2"][[0, 4], :] = np.nan
    actual, = evaluate(case["graph"], inputs)
    expected = _route_gold(inputs)
    assert np.all(np.isfinite(actual[0])) and np.all(np.isnan(actual[4]))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    _assert_equal_bits(actual[np.isfinite(expected)], expected[np.isfinite(expected)])
    # The finite on-board fixture is unchanged; this is a separate host test.
    assert all(np.all(np.isfinite(value)) for value in case["inputs"].values())

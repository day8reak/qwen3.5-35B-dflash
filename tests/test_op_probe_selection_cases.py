"""Selection golds and serialized ports; host checks, not OMG/device execution."""
import math

import numpy as np
import pytest

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.op_probe_selection_cases import selection_cases
from test_ge_select_lowering import RecipeReader


NAMES = (
    *(f"selection_topk_rank{rank}_unique" for rank in (2, 3, 4)),
    *(f"selection_argmax_{layout}_{fixture}" for fixture in ("unique", "ties")
      for layout in ("rank3", "rank2", "rank2_keepdims", "rank4")),
    "selection_argmax_topk1_unique",
    *(f"selection_router_rank{rank}_unique" for rank in (2, 3, 4)),
)


def _gold(case):
    x = case["inputs"]["x"]
    if "argmax" in case["name"]:
        # Python's first occurrence handles both a second-written lower index
        # and signed-zero ties without borrowing the implementation's argmax.
        return [np.array([list(row).index(max(row)) for row in x.reshape(8, 64)], np.int32).reshape(1, 8)]
    router = "router" in case["name"]
    scores = x
    if router:
        rows = []
        for row in x:
            exponentials = [math.exp(float(v) - float(max(row))) for v in row]
            rows.append([v / sum(exponentials) for v in exponentials])
        scores = np.array(rows, np.float16)  # Original FP16 Softmax boundary.
    assert all(len(set(map(float, row))) == 4 for row in scores)
    ids = np.array([sorted(range(4), key=lambda i: -float(row[i]))[:2] for row in scores], np.int32)
    values = np.array([[row[i] for i in indices] for row, indices in zip(scores, ids)], np.float16)
    if router:
        # Sum rounds to FP16 before the division; never normalize in FP32 all
        # the way through to the graph output.
        totals = np.array([[sum(map(float, row))] for row in values], np.float16)
        values = (values.astype(np.float32) / totals.astype(np.float32)).astype(np.float16)
    return [values, ids]


@pytest.mark.parametrize("name", NAMES)
def test_selection_golds_and_serialized_contract(name, tmp_path):
    case = next(c for c in selection_cases() if c["name"] == name)
    gold = _gold(case)
    actual = evaluate(case["graph"], case["inputs"])
    router = "router" in name
    for index, (computed, expected, reference) in enumerate(zip(actual, gold, case["expected"])):
        if router and index == 0:
            assert case["tolerances"][index] == {"rtol": .01, "atol": .002}
            np.testing.assert_allclose(computed, expected, rtol=.01, atol=.002)
            np.testing.assert_allclose(reference, expected, rtol=.01, atol=.002)
        else:
            assert case["tolerances"][index] == {"rtol": 0., "atol": 0.}
            np.testing.assert_array_equal(computed, expected)
            np.testing.assert_array_equal(reference, expected)
    emit_cpp(case["graph"], recipe_dir=tmp_path / "recipe")
    records, inputs, outputs = RecipeReader(tmp_path / "recipe/graph.recipe").read()
    assert [records[i]["name"] for i in inputs] == ["x"]
    assert [records[i]["outputs"][port]["shape"] for i, port in outputs] == [a.shape for a in gold]
    assert [records[i]["outputs"][port]["dtype"] for i, port in outputs] == [str(a.dtype) for a in gold]
    topks = [(i, row) for i, row in enumerate(records) if row["type"] == "TopKV2"]
    if "argmax" in name and "topk1" not in name:
        assert not topks
        argmax, = [r for r in records if r["type"] == "ArgMaxV2"]
        assert argmax["attrs"]["dtype"] == 2  # Recipe dtype tag2 maps to GE DT_INT32.
        assert argmax["inputs"][1]["dtype"] == "int32"
        assert len(argmax["outputs"][0]["shape"]) == len(argmax["inputs"][0]["shape"]) - 1
    else:
        topk_index, topk = topks[0]
        assert len(topks) == 1
        assert [(v["port"], v["dtype"]) for v in topk["outputs"]] == [("values", "float16"), ("indices", "int32")]
        assert topk["attrs"] == {"sorted": 1, "dim": -1, "largest": 1}
        assert topk["inputs"][1]["dtype"] == "int32"
        if "topk1" not in name:
            # Both outputs must remain connected to the same actual TopKV2,
            # not merely two equivalent source-IR operations.
            connected = {port["src_port"] for row in records for port in row["inputs"]
                         if port["src"] == topk_index}
            connected |= {records[i]["outputs"][port]["port"] for i, port in outputs if i == topk_index}
            assert connected == {"values", "indices"}
    if router:
        kinds = [r["type"] for r in records]
        assert kinds.count("SoftmaxV2") == kinds.count("MatMul") == kinds.count("RealDiv") == 1
        assert "ReduceSum" not in kinds
        for row in records:
            if row["type"] in {"SoftmaxV2", "MatMul", "RealDiv"}:
                assert all(out["dtype"] == "float16" for out in row["outputs"])


def test_selection_catalogue_pairs_and_tie_fixture_are_meaningful():
    rows = list(selection_cases())
    assert tuple(c["name"] for c in rows) == NAMES
    ties = [c for c in rows if c["name"].endswith("_ties")]
    assert len(ties) == 4 and not any("topk" in c["name"] for c in ties)
    expected = ties[0]["expected"][0]
    assert expected[0, 0] == 0
    x = ties[0]["inputs"]["x"]
    assert np.signbit(x[0, 0]).any() and (x[0, 0] == 0).all()
    assert expected[0, 7] == 4  # Lower index was assigned second:49 then4.
    for group in {c["equivalence_group"] for c in rows}:
        pair = [c for c in rows if c["equivalence_group"] == group]
        assert all(c["inputs"]["x"].tobytes() == pair[0]["inputs"]["x"].tobytes() for c in pair)
        assert all(c["tolerances"] == pair[0]["tolerances"] for c in pair)

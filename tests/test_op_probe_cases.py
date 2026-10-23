"""Host contracts for device probes, without pretending to execute a device."""
import numpy as np

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.op_probe_cases import cases


def test_case_inputs_references_and_serialization_are_complete():
    suite = list(cases())
    assert len(suite) == 47
    assert len({c["name"] for c in suite}) == len(suite)
    for case in suite:
        graph = case["graph"]
        assert not graph.weights
        assert set(case["inputs"]) == {v.name for v in graph.inputs}
        assert len(case["expected"]) == len(graph.outputs) == len(case["tolerances"])
        assert all(v.dtype in {"float16", "int32"} for v in graph.inputs + graph.outputs)
        assert all(np.isfinite(a).all() for a in case["expected"])
        assert all(np.any(a != 0) for a in case["inputs"].values())
        assert "SaveToFile" in emit_cpp(graph)
        for actual, expected in zip(evaluate(graph, case["inputs"]), case["expected"]):
            np.testing.assert_array_equal(actual, expected)


def test_rank_pairs_have_identical_flattened_inputs_and_results():
    groups = {}
    for case in cases():
        if case["equivalence_group"]:
            groups.setdefault(case["equivalence_group"], []).append(case)
    assert len(groups) == 5
    for pair in groups.values():
        assert len(pair) == 2
        high, flat = pair
        assert len(high["graph"].inputs[0].shape) == 5
        assert len(flat["graph"].inputs[0].shape) == 4
        for name in high["inputs"]:
            np.testing.assert_array_equal(high["inputs"][name].ravel(), flat["inputs"][name].ravel())
        np.testing.assert_array_equal(high["expected"][0].ravel(), flat["expected"][0].ravel())
        assert high["expected"][0].size > 1
        assert np.unique(high["expected"][0]).size > 2


def test_dense_const_and_dynamic_preserve_values_and_topk_port_order():
    suite = {case["name"]: case for case in cases()}
    for dims in [f"m{m}" for m in (1, 2, 8, 16)] + [f"m8_n{n}" for n in (1, 2, 8)]:
        const = suite[f"matmul_{dims}_b_const"]
        dynamic = suite[f"matmul_{dims}_b_dynamic"]
        np.testing.assert_array_equal(const["expected"][0], dynamic["expected"][0])
        assert '"MatMul"' in emit_cpp(const["graph"])
    topk = suite["topk"]
    assert topk["expected"][0].dtype == np.float16
    assert topk["expected"][1].dtype == np.int32
    np.testing.assert_array_equal(topk["expected"][1], [[2, 7, 4], [5, 0, 3]])
    np.testing.assert_array_equal(topk["expected"][0], [[7, 5, 4], [9, 6, 4]])
    gather = suite["gather_int32"]
    assert np.all(gather["expected"][0] > 2 ** 24)
    assert all(t == {"rtol": 0.0, "atol": 0.0} for t in gather["tolerances"])


def test_probe_values_are_deterministic_and_bmm_uses_normalized_native_kind():
    left, right = list(cases()), list(cases())
    for a, b in zip(left, right):
        assert a["name"] == b["name"]
        for name in a["inputs"]:
            np.testing.assert_array_equal(a["inputs"][name], b["inputs"][name])
    bmm = next(case for case in left if case["name"] == "bmm_rank5_normalized")
    code = emit_cpp(bmm["graph"])
    assert '"BatchMatMulV2"' in code
    assert '"__ge/result/batch_left"' in code
    assert '"__ge/result/batch_right"' in code
    conv = next(case for case in left if case["name"] == "bmm_conv_batch48_m8_k4_n1")
    assert conv["inputs"]["x"].shape == (48, 8, 4)
    assert conv["inputs"]["b"].shape == (48, 4, 1)
    assert conv["expected"][0].shape == (48, 8, 1)
    np.testing.assert_array_equal(conv["expected"][0], np.matmul(conv["inputs"]["x"], conv["inputs"]["b"]))

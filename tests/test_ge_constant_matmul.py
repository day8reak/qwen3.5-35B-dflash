"""Bounded constant K=1 folding preserves the neutral IR matmul boundary."""
from dataclasses import replace

import numpy as np
import pytest

from qwen35_moe_dflash.ge_constants import fold_constants
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph


def product(left, right, *, transpose_a=False, transpose_b=False, dtype="float16"):
    graph = Graph("constant_matmul")
    graph.input("unused", (1,))
    a = graph.const(left, dtype=dtype, name="left")
    b = graph.const(right, dtype=dtype, name="right")
    a_shape = a.shape[:-2] + (a.shape[-2:][::-1] if transpose_a else a.shape[-2:])
    b_shape = b.shape[:-2] + (b.shape[-2:][::-1] if transpose_b else b.shape[-2:])
    shape = np.broadcast_shapes(a_shape[:-2], b_shape[:-2]) + (a_shape[-2], b_shape[-1])
    output = graph.op("matmul", a, b, shape=shape, dtype=dtype, name="result",
                      attrs={"transpose_a": transpose_a, "transpose_b": transpose_b})
    graph.outputs = [output]
    return graph


def folded(graph, **kwargs):
    return fold_constants(graph, {}, set(graph._values), **kwargs)


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (False, True),
                                                       (True, False), (True, True)])
@pytest.mark.parametrize("batched", [False, True])
def test_finite_k1_transpose_and_batch_products_match_oracle_bytes(transpose_a, transpose_b, batched):
    left = np.array([[1], [-2], [0.01000213623046875]], dtype=np.float16)
    right = np.array([[0, 1, 3, -7]], dtype=np.float16)
    if batched:
        left = np.broadcast_to(left, (2, 1, 3, 1)).copy()
        right = np.broadcast_to(right, (1, 5, 1, 4)).copy()
    if transpose_a:
        left = np.swapaxes(left, -1, -2)
    if transpose_b:
        right = np.swapaxes(right, -1, -2)
    graph = product(left, right, transpose_a=transpose_a, transpose_b=transpose_b)
    before = {name: array.tobytes() for name, array in graph.constants.items()}
    expected, = evaluate(graph, {"unused": np.zeros(1, dtype=np.float16)})
    actual = folded(graph)
    assert list(actual) == ["result"]
    assert actual["result"].dtype == np.dtype("<f2")
    assert actual["result"].shape == expected.shape
    assert actual["result"].tobytes() == expected.tobytes()
    assert before == {name: array.tobytes() for name, array in graph.constants.items()}


def test_signed_zero_follows_matmul_instead_of_elementwise_multiply():
    left = np.array([0x0000, 0x8000, 0x3C00, 0xBC00], dtype=np.uint16).view(np.float16)
    right = left.copy()
    graph = product(left[:, None], right[None, :])
    oracle, = evaluate(graph, {"unused": np.zeros(1, dtype=np.float16)})
    actual = folded(graph)["result"]
    assert actual.tobytes() == oracle.tobytes()
    # NumPy's matrix accumulator can canonicalize exact negative-zero products;
    # a naive outer multiply is not an interchangeable semantic oracle.
    multiply = (left.astype(np.float32)[:, None] * right.astype(np.float32)[None, :]).astype(np.float16)
    assert (actual.view(np.uint16) != multiply.view(np.uint16)).any()


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_nonfinite_operand_stays_unfolded_without_calling_matmul(operand, bad, monkeypatch):
    arrays = [np.ones((2, 1), dtype=np.float16), np.ones((1, 3), dtype=np.float16)]
    arrays[operand].flat[0] = bad
    graph = product(*arrays)
    monkeypatch.setattr(np, "matmul", lambda *args: pytest.fail("nonfinite operands must not be evaluated"))
    assert folded(graph) == {}


def test_finite_operands_with_nonfinite_fp16_result_stay_unfolded():
    graph = product([[65504]], [[2]])
    assert folded(graph) == {}


class PayloadForbidden(dict):
    def __getitem__(self, key):
        raise AssertionError("budget or eligibility must be checked before payload access")


@pytest.mark.parametrize("budget", [{"max_tensor_bytes": 11}, {"max_total_bytes": 11},
                                   {"max_tensor_bytes": 0}, {"max_total_bytes": 0}])
def test_byte_budgets_are_checked_before_any_constant_payload_lookup(budget):
    graph = product(np.ones((3, 1)), np.ones((1, 2)))
    graph.constants = PayloadForbidden(graph.constants)
    assert folded(graph, **budget) == {}


def test_exact_output_budget_does_not_charge_existing_constants_twice():
    graph = product(np.ones((3, 1)), np.ones((1, 2)))
    assert folded(graph, max_tensor_bytes=12, max_total_bytes=12)["result"].nbytes == 12


def test_total_budget_is_shared_with_prior_folds_and_other_matmuls():
    graph = Graph("shared_budget")
    left = graph.const(np.ones((2, 1)), name="left")
    right = graph.const(np.ones((1, 2)), name="right")
    first = graph.op("neg", left, shape=left.shape, name="unary")
    middle = graph.op("matmul", left, right, shape=(2, 2), name="first_product")
    last = graph.op("matmul", left, right, shape=(2, 2), name="second_product")
    result = fold_constants(graph, {}, {first.name, middle.name, last.name}, max_total_bytes=12)
    assert set(result) == {"unary", "first_product"}
    assert sum(array.nbytes for array in result.values()) == 12


class UnreadableWeight:
    def __fspath__(self):
        raise AssertionError("constant matmul must never resolve a weight path")


@pytest.mark.parametrize("external_on_left", [False, True])
def test_external_weight_operand_is_ineligible_without_resolving_path(external_on_left):
    graph = Graph("external")
    if external_on_left:
        left = graph.weight("external", (2, 1))
        right = graph.const(np.ones((1, 2)))
    else:
        left = graph.const(np.ones((2, 1)))
        right = graph.weight("external", (1, 2))
    output = graph.op("matmul", left, right, shape=(2, 2))
    graph.constants = PayloadForbidden(graph.constants)
    assert fold_constants(graph, {"external": UnreadableWeight()}, {output.name}) == {}


def test_previously_folded_constant_expression_is_not_a_direct_constant():
    graph = Graph("derived")
    left = graph.const(np.ones((2, 1)))
    right = graph.const(np.ones((1, 2)))
    derived = graph.op("neg", left, shape=left.shape)
    result = graph.op("matmul", derived, right, shape=(2, 2))
    assert set(folded(graph)) == {derived.name}
    assert result.name not in folded(graph)


def test_dynamic_operand_is_ineligible_before_payload_lookup():
    graph = Graph("dynamic")
    left = graph.input("x", (2, 1))
    right = graph.const(np.ones((1, 2)))
    graph.op("matmul", left, right, shape=(2, 2))
    graph.constants = PayloadForbidden(graph.constants)
    assert folded(graph) == {}


def test_k_greater_than_one_and_non_fp16_products_are_not_evaluated():
    graph = product(np.ones((2, 3)), np.ones((3, 4)))
    graph.constants = PayloadForbidden(graph.constants)
    assert folded(graph) == {}
    graph = product([[1]], [[2]], dtype="int32")
    graph.constants = PayloadForbidden(graph.constants)
    assert folded(graph) == {}


def test_dead_product_and_unrecognized_attributes_remain_unfolded():
    graph = product([[1]], [[2]])
    assert fold_constants(graph, {}, set()) == {}
    graph.nodes[0].attrs["unsupported_semantics"] = True
    graph.constants = PayloadForbidden(graph.constants)
    assert folded(graph) == {}


@pytest.mark.parametrize("key", ["transpose_a", "transpose_b"])
@pytest.mark.parametrize("value", [0, 1, "false", None, np.bool_(True)])
def test_transpose_attributes_require_actual_python_booleans(key, value):
    graph = product([[1]], [[2]])
    graph.nodes[0].attrs[key] = value
    with pytest.raises(ValueError, match="transpose attributes must be booleans"):
        folded(graph)


@pytest.mark.parametrize("fault,match", [
    ("arity", "requires two inputs"), ("contracted", "contracted dimensions differ"),
    ("batch", "batch shapes are incompatible"), ("output", "output shape mismatch"),
    ("negative", "positive static dimensions"), ("zero", "positive static dimensions"),
    ("float_dim", "positive static dimensions"), ("bool_dim", "positive static dimensions"),
    ("payload_shape", "tensor descriptor mismatch"), ("payload_dtype", "tensor descriptor mismatch"),
])
def test_malformed_eligible_metadata_or_payload_is_rejected(fault, match):
    graph = product([[1], [2]], [[3, 4]])
    node = graph.nodes[0]
    if fault == "arity":
        node.inputs = node.inputs[:1]
    elif fault == "contracted":
        node.inputs = (node.inputs[0], replace(node.inputs[1], shape=(2, 2)))
    elif fault == "batch":
        node.inputs = (replace(node.inputs[0], shape=(2, 2, 1)),
                       replace(node.inputs[1], shape=(3, 1, 2)))
    elif fault == "output":
        node.output = replace(node.output, shape=(1, 4))
    elif fault in {"negative", "zero", "float_dim", "bool_dim"}:
        dim = {"negative": -1, "zero": 0, "float_dim": 2.0, "bool_dim": True}[fault]
        node.inputs = (replace(node.inputs[0], shape=(dim, 1)), node.inputs[1])
    elif fault == "payload_shape":
        graph.constants["left"] = np.ones((3, 1), dtype=np.float16)
    else:
        graph.constants["left"] = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        folded(graph)

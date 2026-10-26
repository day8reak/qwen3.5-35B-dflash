"""Bounded TopK/ArgMax layout probes; no production selection rewrite.

TopK comparisons use unique scores, including after the FP16 Softmax boundary.
ArgMax ties require the original lowest-index policy. TopK1 is only a candidate
for unique maxima: its unspecified tie ordering is not an ArgMax substitute.
"""
from __future__ import annotations

import numpy as np

from .ge_ir import Graph
from .ge_reduction_matmul import reduce_via_matmul


def _view(graph, value, shape, name):
    return graph.op("reshape", value, shape=shape, dtype=value.dtype,
                    attrs={"shape": list(shape)}, name=name)


def _topk(graph, source, *, k=2, values_name="values", indices_name="indices"):
    shape = source.shape[:-1] + (k,)
    values = graph.op("topk_values", source, shape=shape,
                      attrs={"axis": -1, "k": k}, name=values_name)
    indices = graph.op("topk_indices", source, shape=shape, dtype="int32",
                       attrs={"axis": -1, "k": k}, name=indices_name)
    return values, indices


def _topk_graph(name, rank):
    graph = Graph(name)
    source = graph.input("x", (8, 4))
    if rank != 2:
        source = _view(graph, source, (1,) * (rank - 2) + source.shape, "layout")
    values, indices = _topk(graph, source,
                           values_name="values" if rank == 2 else "selected_values",
                           indices_name="indices" if rank == 2 else "selected_indices")
    if rank != 2:
        values = _view(graph, values, (8, 2), "values")
        indices = _view(graph, indices, (8, 2), "indices")
    graph.outputs = [values, indices]
    return graph


def _argmax_scores(ties):
    columns = np.arange(64, dtype=np.int32)
    permutation = (columns[None, :] * 17 + np.arange(8)[:, None] * 7) % 64
    scores = ((permutation - 32) / 8).astype(np.float16)
    if ties:
        scores = (-1 - permutation / 64).astype(np.float16)
        for row in range(8):
            scores[row, [(row * 7) % 64, (row * 7 + 19) % 64]] = np.float16(-0.5)
        # All equal, with both zero signs: the required index remains zero.
        scores[0] = np.float16(0)
        scores[0, [1, 32]] = np.float16(-0.0)
    return scores.reshape(1, 8, 64)


def _argmax_graph(name, layout):
    graph = Graph(name)
    source = graph.input("x", (1, 8, 64))
    if layout in {"rank2", "rank2_keepdims", "topk1"}:
        source = _view(graph, source, (8, 64), "layout")
    elif layout == "rank4":
        source = _view(graph, source, (1, 1, 8, 64), "layout")
    if layout == "topk1":
        result = graph.op("topk_indices", source, shape=(8, 1), dtype="int32",
                          attrs={"axis": -1, "k": 1}, name="selected_indices")
    else:
        keepdims = layout == "rank2_keepdims"
        shape = source.shape[:-1] + ((1,) if keepdims else ())
        result = graph.op("argmax", source, shape=shape, dtype="int32",
                          attrs={"axis": -1, "keepdims": keepdims},
                          name="result" if layout == "rank3" else "selected_indices")
    if layout != "rank3":
        result = _view(graph, result, (1, 8), "result")
    graph.outputs = [result]
    return graph


def _router_graph(name, rank, *, native_sum=False):
    graph = Graph(name)
    source = graph.input("x", (8, 4))
    if rank != 2:
        source = _view(graph, source, (1,) * (rank - 2) + source.shape, "layout")
    probs = graph.op("softmax", source, shape=source.shape, attrs={"axis": -1}, name="probs")
    values, indices = _topk(graph, probs, values_name="selected_values",
                           indices_name="indices" if rank == 2 else "selected_indices")
    if rank != 2:
        values = _view(graph, values, (8, 2), "flat_values")
        indices = _view(graph, indices, (8, 2), "indices")
    if native_sum:
        total = graph.op("reduce_sum", values, shape=(8, 1),
                         attrs={"axes": [-1], "keepdims": True}, name="total")
    else:
        total = reduce_via_matmul(graph, values, kind="reduce_sum", axis=-1,
                                  keepdims=True, name="total")
    weights = graph.op("div", values, total, shape=(8, 2), name="weights")
    graph.outputs = [weights, indices]
    return graph


def selection_cases():
    """Yield 15 contracts with unchanged FP16/INT32 external IO and tolerances."""
    # Lazy imports let the main case registry append this family without a
    # circular import or changes to existing case construction.
    from .op_probe_cases import _finish, _fp

    scores = _fp((8, 4), 3)
    reference = _topk_graph("selection_topk_reference", 2)
    for rank in (2, 3, 4):
        graph = _topk_graph(f"selection_topk_rank{rank}_unique", rank)
        yield _finish(graph, {"x": scores.copy()},
                      f"Sorted FP16 TopK k2 at internal rank{rank}; unique scores, "
                      "external [8,4] to FP16 values and INT32 IDs [8,2].",
                      exact=True, pair="selection_topk_unique", reference_graph=reference)

    for ties in (False, True):
        label = "ties" if ties else "unique"
        data = _argmax_scores(ties)
        reference = _argmax_graph(f"selection_argmax_reference_{label}", "rank3")
        for layout in ("rank3", "rank2", "rank2_keepdims", "rank4"):
            graph = _argmax_graph(f"selection_argmax_{layout}_{label}", layout)
            yield _finish(graph, {"x": data.copy()},
                          f"ArgMax {layout}, {label} scores; preserve [1,8,64] to "
                          "INT32 [1,8], lowest original index for tied maxima.",
                          exact=True, pair=f"selection_argmax_{label}", reference_graph=reference)

    graph = _argmax_graph("selection_argmax_topk1_unique", "topk1")
    reference = _argmax_graph("selection_argmax_topk1_reference", "rank3")
    yield _finish(graph, {"x": _argmax_scores(False)},
                  "TopK1 INT32 IDs as an ArgMax candidate for unique maxima only; "
                  "no assertion of lowest-index tie behavior.", exact=True,
                  pair="selection_argmax_unique", reference_graph=reference)

    reference = _router_graph("selection_router_reference", 2, native_sum=True)
    for rank in (2, 3, 4):
        graph = _router_graph(f"selection_router_rank{rank}_unique", rank)
        yield _finish(graph, {"x": scores.copy()},
                      f"FP16 Softmax then sorted TopK k2 at rank{rank}, two-slot MatMul "
                      "sum then FP16 division; reference retains native ReduceSum. "
                      "Unique probabilities; unchanged normalized weights and INT32 IDs ABI.",
                      pair="selection_router_unique", reference_graph=reference)

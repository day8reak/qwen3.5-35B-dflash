"""Embed emitted FP16 reduction recipes in a complete tiny host graph pair.

Only reduction/scan recipes are projected back into the neutral IR; all other
nodes retain the original oracle. This is not full serialized-model execution,
OMG validation, a device result, or a claim of model-quality accuracy.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import build_pair, prepare_pair
from qwen35_moe_dflash.ge_ir import DTYPES, Graph
from qwen35_moe_dflash.hashing import sha256_file
from test_ge_select_lowering import RecipeReader


def _project_recipe(clone, source, node, folder, counts):
    """Interpret actual emitted edges, constants and attrs, not lowering plans."""
    fixture = Graph("reduction_fixture")
    x = fixture.input("x", source.shape, source.dtype)
    fixture.outputs = [fixture.op(node.kind, x, shape=node.output.shape,
                                  dtype=node.output.dtype, attrs=dict(node.attrs), name="result")]
    report = {}
    emit_cpp(fixture, recipe_dir=folder, fp16_reductions="matmul", reduction_report=report)
    records, inputs, outputs = RecipeReader(folder / "graph.recipe").read()
    assert len(inputs) == len(outputs) == 1
    assert report["counts"]["retained"] == 0
    assert report["counts"]["replaced"] == (source.shape[node.attrs["axis"]]
                                               if node.kind == "cumsum" else 1)
    counts["scan_prefixes"] += report["counts"]["scan_prefix_replaced"]
    counts["matmuls"] += sum(record["type"] == "MatMul" for record in records)
    counts[node.kind] += 1
    values, literals = {}, {}

    def constant_input(record, index):
        port = record["inputs"][index]
        return literals[(port["src"], port["src_port"])].tolist()

    for index, record in enumerate(records):
        output, = record["outputs"]
        kind = record["type"]
        args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
        name = f"__projected/{node.output.name}/{index}"
        shape, dtype = output["shape"], output["dtype"]
        if kind == "Data":
            assert index == inputs[0] and shape == source.shape and dtype == source.dtype
            value = source
        elif kind == "Const":
            path, offset, size = record["constant"]
            with Path(path).open("rb") as stream:
                stream.seek(offset)
                payload = stream.read(size)
            assert len(payload) == size
            array = np.frombuffer(payload, dtype=DTYPES[dtype]).reshape(shape)
            value = clone.const(array, dtype=dtype, name=name)
            literals[(index, output["port"])] = array
        else:
            assert dtype == "float16"
            if kind == "MatMul":
                ir_kind = "matmul"
                attrs = {"transpose_a": bool(record["attrs"]["transpose_x1"]),
                         "transpose_b": bool(record["attrs"]["transpose_x2"])}
            elif kind == "Reshape":
                ir_kind, attrs = "reshape", {"shape": constant_input(record, 1)}
                assert tuple(attrs["shape"]) == shape
                args = args[:1]
            elif kind == "Transpose":
                ir_kind, attrs = "transpose", {"perm": constant_input(record, 1)}
                args = args[:1]
            elif kind == "Slice":
                ir_kind, attrs = "slice", {"starts": constant_input(record, 1),
                                           "sizes": constant_input(record, 2)}
                args = args[:1]
            elif kind == "ConcatD":
                assert record["attrs"]["N"] == len(args)
                ir_kind, attrs = "concat", {"axis": record["attrs"]["concat_dim"]}
            else:
                raise AssertionError(f"unexpected operation in emitted reduction recipe: {kind}")
            value = clone.op(ir_kind, *args, shape=shape, dtype=dtype, attrs=attrs, name=name)
        assert (value.shape, value.dtype) == (shape, dtype)
        values[(index, output["port"])] = value
    index, port = outputs[0]
    result = values[(index, records[index]["outputs"][port]["port"])]
    assert (result.shape, result.dtype) == (node.output.shape, node.output.dtype)
    return result


def _project_reductions(original, folder):
    clone, values, counts = Graph(original.name), {}, Counter()
    for value in original.inputs:
        values[value.name] = clone.input(value.name, value.shape, value.dtype)
    for name, array in original.constants.items():
        values[name] = clone.const(array, dtype=original._values[name].dtype, name=name)
    for name, value in original.weights.items():
        values[name] = clone.weight(name, value.shape, value.dtype)
    for node in original.nodes:
        args = [values[value.name] for value in node.inputs]
        if node.kind in {"reduce_mean", "reduce_sum", "cumsum"} and node.inputs[0].dtype == "float16":
            assert len(args) == 1 and node.output.dtype == "float16"
            value = _project_recipe(clone, args[0], node, folder / node.output.name, counts)
        else:
            value = clone.op(node.kind, *args, shape=node.output.shape, dtype=node.output.dtype,
                             attrs=dict(node.attrs), name=node.output.name)
        values[node.output.name] = value
    clone.outputs = [values[value.name] for value in original.outputs]
    clone.validate()
    assert clone.weights == original.weights
    assert [(value.shape, value.dtype) for value in clone.inputs + clone.outputs] == [
        (value.shape, value.dtype) for value in original.inputs + original.outputs]
    # Integer routes remain native and keep their original types/attributes.
    integer_reductions = lambda graph: [(node.kind, node.output.dtype, node.output.shape, node.attrs)
                                        for node in graph.nodes
                                        if node.kind in {"reduce_mean", "reduce_sum", "cumsum"}
                                        and node.inputs[0].dtype != "float16"]
    assert integer_reductions(clone) == integer_reductions(original)
    assert not any(node.kind in {"reduce_sum", "reduce_mean", "cumsum"}
                   and node.inputs[0].dtype == "float16" for node in clone.nodes)
    return clone, counts


def test_emitted_fp16_reductions_preserve_tiny_pair_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, sequence_length=8, tiny=True)
    original = build_pair(plan["target_config"], plan["draft_config"], 8)
    projected, counts = [], {}
    for graph in original:
        candidate, count = _project_reductions(graph, tmp_path / "recipes" / graph.name)
        projected.append(candidate)
        counts[graph.name] = count
    assert counts["target"] == {"reduce_mean": 14, "reduce_sum": 215, "cumsum": 3,
                                 "scan_prefixes": 192, "matmuls": 421}
    assert counts["draft"] == {"reduce_mean": 10, "scan_prefixes": 0, "matmuls": 10}
    weights = {}
    for key, item in plan["weights"].items():
        path = prepared / item["path"]
        assert sha256_file(path) == item["sha256"]
        value = np.fromfile(path, dtype="<f2").reshape(item["shape"])
        value.setflags(write=False)
        weights[key] = value
    for length in (1, 3, 8):
        ids = np.zeros((1, 8), dtype=np.int32)
        mask = np.zeros_like(ids)
        ids[0, :length] = [7, 3, 26, 8, 16, 42, 29, 31][:length]
        mask[0, :length] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        expected_top1, expected_features = evaluate(original[0], feeds, weights)
        expected_draft, = evaluate(original[1], {**feeds, "target_features": expected_features}, weights)
        actual_top1, actual_features = evaluate(projected[0], feeds, weights)
        actual_draft, = evaluate(projected[1], {**feeds, "target_features": actual_features}, weights)
        np.testing.assert_array_equal(actual_top1, expected_top1)
        assert actual_features.dtype == expected_features.dtype == np.float16
        assert np.isfinite(actual_features).all() and np.isfinite(expected_features).all()
        # Preserve the existing FP16 comparison gate; do not tune tolerances.
        torch.testing.assert_close(torch.from_numpy(actual_features), torch.from_numpy(expected_features))
        np.testing.assert_array_equal(actual_draft, expected_draft)
    for item in plan["weights"].values():
        assert sha256_file(prepared / item["path"]) == item["sha256"]

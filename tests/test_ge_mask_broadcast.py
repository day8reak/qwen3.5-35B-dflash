"""Exact integer Gather copies for the draft's dynamic BOOL attention masks.

These execute the emitted recipe with an independent NumPy interpreter. They
check host semantics and serialization, not CPUCL placement or board support.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np
import pytest

from qwen35_moe_dflash.draft_standalone import _graph
from qwen35_moe_dflash.ge_cpp import _Emitter, emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import DTYPES, Graph, Value
from qwen35_moe_dflash.ge_recipe_read import read_recipe


def _constant(record):
    storage = record["constant"]
    with Path(storage["path"]).open("rb") as stream:
        stream.seek(storage["offset"])
        payload = stream.read(storage["size"])
    assert len(payload) == storage["size"]
    output, = record["outputs"]
    return np.frombuffer(payload, dtype=DTYPES[output["dtype"]]).reshape(output["shape"])


def _run_recipe(recipe, feeds):
    """Interpret only the mask/Select primitives; never call the lowering helper."""
    values = {}
    binary = {"Add": np.add, "Sub": np.subtract, "Less": np.less,
              "LessEqual": np.less_equal, "LogicalAnd": np.logical_and}
    for index, record in enumerate(recipe["records"]):
        args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
        output, = record["outputs"]
        kind = record["type"]
        attrs = {key: value["value"] for key, value in record["attrs"].items()}
        if kind == "Data":
            value = feeds[record["name"]]
        elif kind == "Const":
            value = _constant(record)
        elif kind == "Cast":
            assert attrs["dst_type"] == {"bool": 12, "int32": 3}[output["dtype"]]
            value = args[0].astype(DTYPES[output["dtype"]])
        elif kind == "GatherV2":
            assert args[1].dtype == args[2].dtype == np.dtype("int32")
            value = np.take(args[0], args[1], axis=int(args[2].item()))
        elif kind == "Reshape":
            value = args[0].reshape(tuple(int(x) for x in args[1]))
        elif kind == "Slice":
            value = args[0][tuple(slice(int(start), int(start + size))
                                 for start, size in zip(args[1], args[2]))]
        elif kind == "ConcatD":
            value = np.concatenate(args, axis=attrs["concat_dim"])
        elif kind == "ReduceSum":
            value = args[0].sum(axis=tuple(int(x) for x in args[1]), keepdims=attrs["keep_dims"])
        elif kind in binary:
            value = binary[kind](*args)
        elif kind == "SelectV2":
            assert args[0].dtype == np.dtype("bool")
            assert args[0].shape == args[1].shape == args[2].shape
            value = np.where(*args)
        else:
            raise AssertionError(f"unexpected mask operation {kind}")
        # FP16 selection/copy must already have the exact dtype, so casting in
        # the interpreter cannot hide introduced arithmetic or quiet a NaN.
        if output["dtype"] == "float16":
            assert value.dtype == DTYPES["float16"]
        value = np.asarray(value, dtype=DTYPES[output["dtype"]])
        assert value.shape == tuple(output["shape"]), (record["name"], value.shape, output["shape"])
        values[(index, output["port"])] = value
    outputs = [values[(index, recipe["records"][index]["outputs"][port]["port"])]
               for index, port in recipe["outputs"]]
    return outputs, values


def _serialize(graph, folder, *, backend="legacy", **options):
    if backend == "legacy":
        emit, read = emit_cpp, read_recipe
    else:
        from qwen35_moe_dflash.ge_fused.emitter import emit_cpp as emit
        from qwen35_moe_dflash.ge_fused.recipe import read_recipe as read
    emit(graph, recipe_dir=folder / "recipe", **options)
    recipe = read(folder / "recipe/graph.recipe")
    cpp = emit(graph, **options)
    declared = [(json.loads(name), json.loads(kind)) for name, kind in re.findall(
        r'ops\.push_back\(Op\(("[^"\n]+"), ("[^"\n]+"),', cpp)]
    assert declared == [(r["name"], r["type"]) for r in recipe["records"]]
    literals = {int(index): bytes(int(byte) for byte in payload.split(",") if byte)
                for index, payload in re.findall(
                    r'SetConst\(ops\[(\d+)\], [^\n]*std::vector<uint8_t>\{([\d,]*)\}\);', cpp)}
    assert literals == {index: _constant(record).tobytes()
                        for index, record in enumerate(recipe["records"]) if record["type"] == "Const"}
    return recipe


def _mask_gathers(recipe):
    return [r for r in recipe["records"] if r["type"] == "GatherV2"
            and r["inputs"][0]["dtype"] == "int32" and len(r["inputs"][0]["shape"]) == 4]


def _check_copies(recipe, count):
    records = recipe["records"]
    gathers = _mask_gathers(recipe)
    assert len(gathers) == count
    for gather in gathers:
        source, indices, axis = gather["inputs"]
        output, = gather["outputs"]
        assert [p["port"] for p in gather["inputs"]] == ["x", "indices", "axis"]
        assert source["dtype"] == output["dtype"] == indices["dtype"] == axis["dtype"] == "int32"
        assert gather["attrs"] == {"batch_dims": {"type": "int", "value": 0},
                                    "is_preprocessed": {"type": "bool", "value": False},
                                    "negative_index_support": {"type": "bool", "value": False}}
        zeros, axis_data = _constant(records[indices["src"]]), _constant(records[axis["src"]])
        dimension = int(axis_data.item())
        assert dimension in (1, 2) and axis_data.shape == ()
        assert source["shape"][dimension] == 1
        assert zeros.shape == (output["shape"][dimension],) and not zeros.any()
        expected = list(source["shape"])
        expected[dimension] = len(zeros)
        assert output["shape"] == expected
        assert all(p.get("format", "ND") == "ND" for p in (*gather["inputs"], output))
    assert not any(r["type"] == "BroadcastTo" and r["outputs"][0]["dtype"] == "bool" for r in records)
    return gathers


def _mask_subgraph(*, tiny=False):
    _, config, graph = _graph(8 if tiny else 64, tiny=tiny)
    visible = [next(n.output for n in graph.nodes if n.output.name == f"layer_{layer}_visible")
               for layer in range(config.num_hidden_layers)]
    live = {v.name for v in visible}
    for node in reversed(graph.nodes):
        if node.output.name in live:
            live.update(v.name for v in node.inputs)
    graph.nodes = [n for n in graph.nodes if n.output.name in live]
    graph.inputs = [v for v in graph.inputs if v.name in live]
    graph.constants = {k: v for k, v in graph.constants.items() if k in live}
    graph.weights = {k: v for k, v in graph.weights.items() if k in live}
    assert not graph.weights and [v.name for v in graph.inputs] == ["attention_mask"]
    score_shape = (1, config.num_attention_heads, config.block_size,
                   graph.inputs[0].shape[1] + config.block_size)
    scores = graph.input("test_scores", score_shape)
    masked = [graph.op("where", mask, scores, graph.const(np.float16(-np.inf)),
                       shape=score_shape, name=f"test_masked_{layer}")
              for layer, mask in enumerate(visible)]
    graph.outputs = visible + masked
    return config, graph.validate()


@pytest.mark.parametrize("tiny", [False, True], ids=["full-six-layer-s64", "tiny-two-layer-s8"])
def test_every_mask_layer_matches_source_ir_for_short_partial_and_full_prefix(tmp_path, tiny):
    config, graph = _mask_subgraph(tiny=tiny)
    before = copy.deepcopy(graph.manifest())
    recipe = _serialize(graph, tmp_path)
    gathers = _check_copies(recipe, 2 * config.num_hidden_layers)
    sequence = graph.inputs[0].shape[1]
    score_shape = graph.inputs[1].shape
    patterns = np.array([0, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01, 1, 0x8001], dtype="<u2")
    scores = np.resize(patterns, score_shape).view("<f2")
    for length in (1, min(6, sequence), sequence):
        feeds = {"attention_mask": (np.arange(sequence)[None] < length).astype(np.int32),
                 "test_scores": scores}
        expected = evaluate(graph, feeds)
        actual, values = _run_recipe(recipe, feeds)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))
        for gather in gathers:
            index = recipe["records"].index(gather)
            assert set(np.unique(values[(index, "y")])).issubset({0, 1})
        # These layer-level semantics prevent an accidental all-true mask or
        # causal full-attention replacement from passing a mere copy test.
        masks = actual[:config.num_hidden_layers]
        assert masks[-1][0, 0, 0, -1]
        assert not masks[0][0, 0, 0, -1]
        assert not masks[-1][0, 0, :, length - 1:sequence].any()
    assert graph.manifest() == before


def test_full_draft_keeps_all_twelve_mask_output_names_and_no_bool_broadcast(tmp_path):
    _, config, graph = _graph(64)
    before = copy.deepcopy(graph.manifest())
    explicit = [n for n in graph.nodes if n.kind == "broadcast_to"]
    assert len(explicit) == 6 and explicit[-1].output.name == "v501"
    paths = {key: tmp_path / f"unopened-{index}.bin" for index, key in enumerate(graph.weights)}
    recipe = _serialize(graph, tmp_path, weight_paths=paths, external_weights=True,
                        fold_constant_arithmetic=False, draft_fc_policy="conv2d",
                        fp16_reductions="matmul", fp16_row_mul="compact")
    _check_copies(recipe, 12)
    by_name = {r["name"]: r for r in recipe["records"]}
    for node in explicit:
        record = by_name[node.output.name]
        assert record["type"] == "Cast" and record["outputs"][0]["dtype"] == "bool"
        assert record["outputs"][0]["shape"] == list(node.output.shape)
    selected = [r for r in recipe["records"] if r["type"] == "SelectV2"
                and r["outputs"][0]["dtype"] == "float16"]
    assert len(selected) == config.num_hidden_layers
    for select in selected:
        condition, then, other = select["inputs"]
        copied = recipe["records"][condition["src"]]
        assert copied["name"] == f"__ge/{select['name']}/broadcast_0" and copied["type"] == "Cast"
        assert condition["dtype"] == "bool" and then["dtype"] == other["dtype"] == "float16"
        assert condition["shape"] == then["shape"] == other["shape"] == [1, 32, 16, 80]
    assert [recipe["records"][i]["name"] for i, _ in recipe["outputs"]] == ["draft_top1", "draft_hidden"]
    assert graph.manifest() == before and not any(path.exists() for path in paths.values())


@pytest.mark.parametrize("backend", ["legacy", "stateful"])
@pytest.mark.parametrize("both_axes", [False, True])
def test_shared_emitter_copies_query_and_head_axes_without_touching_score_bits(tmp_path, backend, both_axes):
    if backend == "stateful":
        from qwen35_moe_dflash.ge_fused.ir import Graph as Builder
    else:
        Builder = Graph
    graph = Builder("mask_copy")
    condition = graph.input("condition", (1, 1, 1, 80), "bool")
    expanded_shape = (1, 32 if both_axes else 1, 16, 80)
    expanded = graph.op("broadcast_to", condition, shape=expanded_shape, dtype="bool",
                        attrs={"shape": list(expanded_shape)}, name="v501")
    scores = graph.input("scores", (1, 32, 16, 80))
    graph.outputs = [graph.op("where", expanded, scores, graph.const(np.float16(-np.inf)),
                              shape=scores.shape, name="selected")]
    recipe = _serialize(graph, tmp_path, backend=backend)
    _check_copies(recipe, 2)
    mask = (np.arange(80) % 3 != 0).reshape(condition.shape)
    patterns = np.array([0, 0x8000, 0x7c00, 0xfc00, 0x7e31, 0x7d01, 1, 0x8001], dtype="<u2")
    data = np.resize(patterns, scores.shape).view("<f2")
    (actual,), trace = _run_recipe(recipe, {"condition": mask, "scores": data})
    expected = np.where(mask, data, np.float16(-np.inf))
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"))
    assert {r["type"] for r in recipe["records"]} == {"Data", "Const", "Cast", "GatherV2", "SelectV2"}
    for index, record in enumerate(recipe["records"]):
        if record["outputs"][0]["dtype"] == "int32" and record["type"] in {"Cast", "GatherV2"}:
            assert set(np.unique(trace[(index, "y")])).issubset({0, 1})


@pytest.mark.parametrize("change", ["fp16", "int32", "output_dtype", "source_layout", "output_layout",
    "rank3", "batch2", "last_axis", "nonsingleton", "no_expansion", "zero", "negative",
    "float_dimension", "bool_dimension", "overflow", "index_limit"])
def test_mask_copy_rejects_outside_its_rank4_nd_singleton_contract_without_emission(change):
    source = Value("source", (1, 1, 1, 80), "bool")
    output = Value("output", (1, 32, 16, 80), "bool")
    if change in {"fp16", "int32"}:
        dtype = "float16" if change == "fp16" else "int32"
        source, output = replace(source, dtype=dtype), replace(output, dtype=dtype)
    elif change == "output_dtype": output = replace(output, dtype="int32")
    elif change == "source_layout": source = replace(source, format="NCHW")
    elif change == "output_layout": output = replace(output, format="NCHW")
    elif change == "rank3": source = replace(source, shape=(1, 1, 80))
    elif change == "batch2": source, output = replace(source, shape=(2, 1, 1, 80)), replace(output, shape=(2, 32, 16, 80))
    elif change == "last_axis": output = replace(output, shape=(1, 32, 16, 81))
    elif change == "nonsingleton": source = replace(source, shape=(1, 2, 1, 80))
    elif change == "no_expansion": output = replace(output, shape=source.shape)
    elif change == "zero": output = replace(output, shape=(1, 32, 0, 80))
    elif change == "negative": output = replace(output, shape=(1, 32, -1, 80))
    elif change == "float_dimension": output = replace(output, shape=(1, 32, 16.0, 80))
    elif change == "bool_dimension": source = replace(source, shape=(True, 1, 1, 80))
    elif change == "overflow": source, output = replace(source, shape=(1, 1, 1, 2**31)), replace(output, shape=(1, 32, 16, 2**31))
    elif change == "index_limit": output = replace(output, shape=(1, 4097, 16, 80))
    emitter = _Emitter(Graph("scope"))
    assert emitter.emit_mask_broadcast(source, output) is False
    assert emitter.records == [] and emitter.refs == {}


@pytest.mark.parametrize("backend", ["legacy", "stateful"])
@pytest.mark.parametrize("constant", [False, True], ids=["dynamic-mask", "constant-mask"])
@pytest.mark.parametrize("shape_control", [[1, 32, 16.0, 80], [True, 32, 16, 80]],
                         ids=["float-dimension", "bool-dimension"])
def test_rank4_shape_attributes_are_checked_before_folding_or_integer_copy(
        tmp_path, monkeypatch, backend, constant, shape_control):
    if backend == "legacy":
        Builder, emit = Graph, emit_cpp
    else:
        from qwen35_moe_dflash.ge_fused.ir import Graph as Builder
        from qwen35_moe_dflash.ge_fused.emitter import emit_cpp as emit
    graph = Builder("invalid_rank4_mask_control")
    condition = graph.input("condition", (1, 1, 1, 80), "bool")
    source = (graph.const(np.ones((1, 1, 1, 80), dtype=np.bool_), dtype="bool") if constant
              else condition)
    graph.outputs = [graph.op("broadcast_to", source, shape=(1, 32, 16, 80), dtype="bool",
                              attrs={"shape": shape_control}, name="v501"), condition]

    def no_lowering(*args, **kwargs):
        pytest.fail("malformed shape controls must be checked before folding or copying")
    monkeypatch.setattr(_Emitter, "static_broadcast", no_lowering)
    monkeypatch.setattr(_Emitter, "emit_mask_broadcast", no_lowering)
    with pytest.raises(ValueError, match="BroadcastTo dimensions must fit nonnegative INT32"):
        emit(graph, recipe_dir=tmp_path / "recipe")
    assert not (tmp_path / "recipe").exists()


def test_nonmask_broadcasts_keep_their_native_operator(tmp_path):
    graph = Graph("unrelated_broadcasts")
    source = graph.input("fp16", (1, 1, 1, 3))
    other = graph.input("rank2_bool", (1, 3), "bool")
    graph.outputs = [graph.op("broadcast_to", source, shape=(1, 2, 4, 3),
                              attrs={"shape": [1, 2, 4, 3]}),
                     graph.op("broadcast_to", other, shape=(2, 3), dtype="bool",
                              attrs={"shape": [2, 3]})]
    recipe = _serialize(graph, tmp_path)
    assert sum(r["type"] == "BroadcastTo" for r in recipe["records"]) == 2
    assert not any(r["type"] in {"Cast", "GatherV2"} for r in recipe["records"])

"""Integer token selection from decoded GE recipes, without floating carriers."""
import copy

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations, prepare_pair, weight_identity
from qwen35_moe_dflash.ge_ir import DTYPES, Graph, Node
from qwen35_moe_dflash.hashing import sha256_file
from test_ge_static_broadcast import _payload, _serialize_both


def _execute(records, outputs, feeds):
    values = {}
    for index, record in enumerate(records):
        args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
        output, = record["outputs"]
        kind, attrs = record["type"], record["attrs"]
        dtype = DTYPES[output["dtype"]]
        if kind == "Data":
            value = feeds[record["name"]]
        elif kind == "Const":
            value = np.frombuffer(_payload(record), dtype=dtype).reshape(output["shape"])
        elif kind == "Reshape":
            assert args[1].dtype == np.int32
            value = args[0].reshape(tuple(int(x) for x in args[1]))
            assert value.dtype == args[0].dtype and value.tobytes() == args[0].tobytes()
        elif kind == "Cast":
            # The real anchor producer casts mask INT32 -> BOOL -> INT32.
            # Token IDs must not travel through a floating carrier.
            assert dtype in (np.int32, np.bool_)
            assert attrs["dst_type"] == {"bool": 12, "int32": 3}[output["dtype"]]
            value = args[0].astype(dtype)
        elif kind == "ReduceSum":
            assert args[0].dtype == dtype == np.int32
            value = args[0].sum(axis=tuple(int(x) for x in args[1]), dtype=np.int32,
                                keepdims=bool(attrs["keep_dims"]))
        elif kind == "Sub":
            assert all(arg.dtype == np.int32 for arg in args)
            value = np.subtract(*args, dtype=np.int32)
        elif kind == "GatherV2":
            assert args[1].dtype == args[2].dtype == np.int32
            assert attrs == {"batch_dims": 0, "negative_index_support": 0, "is_preprocessed": 0}
            value = np.take(args[0], args[1], axis=int(args[2]))
        elif kind == "GatherElements":
            assert args[0].dtype == np.float16
            value = np.take_along_axis(args[0], args[1], axis=attrs["dim"])
        elif kind == "ConcatD":
            assert attrs["N"] == len(args)
            value = np.concatenate(args, axis=attrs["concat_dim"])
        elif kind == "Less":
            value = np.less(*args)
        elif kind == "LogicalAnd":
            value = np.logical_and(*args)
        else:
            raise AssertionError(f"unexpected integer-anchor operator {kind}")
        assert value.shape == output["shape"] and value.dtype == dtype
        values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


def _anchor_graph(sequence, *, tiny=False):
    _, raw = configurations(tiny=tiny)
    config = Qwen35DFlashConfig.from_dict(raw)
    graph = build_draft(config, sequence)
    context = next(i for i, node in enumerate(graph.nodes) if node.output.name == "context_valid")
    graph.nodes = graph.nodes[:context + 1]
    graph.weights = {}
    wanted = ("v4", "block_ids", "anchor_position", "context_valid")
    graph.outputs = [next(node.output for node in graph.nodes if node.output.name == name) for name in wanted]
    return graph, config


@pytest.mark.parametrize("sequence", [8, 65])
def test_actual_draft_anchor_recipe_handles_every_nonempty_prefix(tmp_path, sequence):
    graph, config = _anchor_graph(sequence)
    assert config.vocab_size == 248320
    anchor = next(node for node in graph.nodes if node.output.name == "v4")
    assert anchor.kind == "gather" and anchor.attrs["axis"] == 0
    assert anchor.inputs[0].name == "input_ids_vector"
    assert anchor.inputs[0].shape == (sequence,) and anchor.inputs[0].dtype == "int32"
    records, inputs, outputs = _serialize_both(graph, tmp_path)
    assert [records[index]["name"] for index in inputs] == ["target_features", "input_ids", "attention_mask"]
    assert [records[index]["outputs"][0]["dtype"] for index in inputs] == ["float16", "int32", "int32"]
    gather, = [record for record in records if record["type"] == "GatherV2"]
    assert gather["name"] == "v4"
    assert [port["port"] for port in gather["inputs"]] == ["x", "indices", "axis"]
    assert [port["dtype"] for port in gather["inputs"]] == ["int32"] * 3
    assert [port["shape"] for port in gather["inputs"]] == [(sequence,), (1, 1), ()]
    assert gather["outputs"][0]["shape"] == (1, 1) and gather["outputs"][0]["dtype"] == "int32"
    assert np.frombuffer(_payload(records[gather["inputs"][2]["src"]]), dtype="<i4").item() == 0
    assert not any(record["type"] == "GatherElements" for record in records)
    assert all(port["dtype"] in {"int32", "int64", "bool"}
               for record in records if record["type"] != "Data" for port in record["outputs"])

    # These are all valid real vocabulary IDs. Larger generic INT32 values
    # belong to the isolated copy test below, never to an embedding lookup.
    cases = np.array([2049, 65535, config.mask_token_id, config.vocab_size - 1,
                      1, 0, 32769, 131073], dtype=np.int32)
    assert np.all((cases >= 0) & (cases < config.vocab_size))
    ids = np.resize(cases, sequence).reshape(1, sequence)
    features = np.zeros((1, sequence, config.feature_size), dtype=np.float16)
    before_ids = ids.tobytes()
    for length in range(1, sequence + 1):
        mask = np.zeros((1, sequence), dtype=np.int32)
        mask[:, :length] = 1
        before_mask = mask.tobytes()
        actual_id, block, position, context = _execute(records, outputs,
            {"input_ids": ids, "attention_mask": mask, "target_features": features})
        assert actual_id.item() == int(ids[0, length - 1])
        assert actual_id.tobytes() == ids[:, length - 1:length].tobytes()
        assert position.item() == length - 1
        np.testing.assert_array_equal(block, np.array(
            [[ids[0, length - 1]] + [config.mask_token_id] * (config.block_size - 1)], dtype=np.int32))
        np.testing.assert_array_equal(context, np.arange(sequence)[None] < length - 1)
        assert ids.tobytes() == before_ids and mask.tobytes() == before_mask


def test_generic_int32_gather_preserves_values_beyond_float_exact_ranges(tmp_path):
    # Generic data-copy values, not a claim that out-of-vocabulary token IDs
    # are accepted by the model or its runtime validation.
    values = np.array([2049, 65535, (1 << 24) + 1, np.iinfo(np.int32).max,
                       np.iinfo(np.int32).min], dtype=np.int32)
    graph = Graph("int32_copy")
    source = graph.input("source", values.shape, "int32")
    index = graph.input("index", (1, 1), "int32")
    graph.outputs = [graph.op("gather", source, index, shape=(1, 1), dtype="int32", attrs={"axis": 0})]
    records, _, outputs = _serialize_both(graph, tmp_path)
    assert {record["type"] for record in records} == {"Data", "Const", "GatherV2"}
    assert all(port["dtype"] == "int32" for record in records for port in record["outputs"])
    for selected in range(values.size):
        actual, = _execute(records, outputs, {"source": values, "index": np.array([[selected]], np.int32)})
        assert actual.item() == int(values[selected])
        assert actual.tobytes() == values[selected:selected + 1].tobytes()
    assert np.float16(values[0]) != values[0]
    assert int(np.float32(values[2])) != int(values[2])


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
@pytest.mark.parametrize("recipe", [False, True])
def test_non_fp16_gather_elements_rejected_before_serialization(tmp_path, dtype, recipe):
    graph = Graph("unsupported_integer_gather_elements")
    source = graph.input("source", (1, 4), dtype)
    index = graph.input("index", (1, 1), "int32")
    graph.outputs = [graph.op("gather_elements", source, index, shape=(1, 1), dtype=dtype, attrs={"axis": 1})]
    options = {"recipe_dir": tmp_path / "recipe"} if recipe else {}
    with pytest.raises(NotImplementedError, match="GatherElements requires FP16 data.*integer Gather"):
        emit_cpp(graph, **options)
    assert not (tmp_path / "recipe").exists()


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_fp16_gather_elements_stays_native_and_preserves_payload_bits(tmp_path, index_dtype):
    bits = np.array([[0x0000, 0x8000, 0x7e35, 0x3c00], [0xfc00, 0x7c00, 0x7d01, 0x4000]], dtype="<u2")
    data = bits.view("<f2")
    indices = np.array([[2, 1], [0, 2]], dtype=DTYPES[index_dtype])
    graph = Graph("fp16_native_gather_elements")
    source = graph.input("source", data.shape)
    index = graph.input("index", indices.shape, index_dtype)
    graph.outputs = [graph.op("gather_elements", source, index, shape=indices.shape, attrs={"axis": 1})]
    records, _, outputs = _serialize_both(graph, tmp_path)
    assert [record["type"] for record in records] == ["Data", "Data", "GatherElements"]
    actual, = _execute(records, outputs, {"source": data, "index": indices})
    expected = np.take_along_axis(bits, indices, axis=1)
    np.testing.assert_array_equal(actual.view("<u2"), expected)


def _previous_anchor_semantics(graph):
    previous = copy.deepcopy(graph)
    ids = next(value for value in previous.inputs if value.name == "input_ids")
    retained = []
    for node in previous.nodes:
        if node.output.name == "input_ids_vector":
            continue
        if node.output.name == "v4":
            assert node.kind == "gather"
            node = Node("gather_elements", (ids, node.inputs[1]), node.output, {"axis": 1})
        retained.append(node)
    previous.nodes = retained
    return previous


def test_complete_tiny_draft_semantics_abi_and_prepared_hashes_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, 8, tiny=True)
    config = Qwen35DFlashConfig.from_dict(plan["draft_config"])
    current = build_draft(config, 8)
    previous = _previous_anchor_semantics(current)
    before_manifest = sha256_file(prepared / "prepared.json")
    before_weights = {key: sha256_file(prepared / record["path"]) for key, record in plan["weights"].items()}
    assert weight_identity(plan["weights"]) == plan["source"]["fp16_payload_sha256"]
    weights = {key: np.fromfile(prepared / record["path"], dtype="<f2").reshape(record["shape"])
               for key, record in plan["weights"].items()}
    current_manifest, previous_manifest = current.manifest(), previous.manifest()
    for field in ("inputs", "outputs", "weights"):
        assert current_manifest[field] == previous_manifest[field]
    assert [(value.name, value.dtype, value.shape) for value in current.inputs] == [
        ("target_features", "float16", (1, 8, config.feature_size)),
        ("input_ids", "int32", (1, 8)), ("attention_mask", "int32", (1, 8))]
    assert [(value.name, value.dtype, value.shape) for value in current.outputs] == [
        ("draft_top1", "int32", (1, config.block_size - 1))]
    rng = np.random.default_rng(817)
    ids = rng.integers(0, config.vocab_size, (1, 8), dtype=np.int32)
    features = rng.normal(0, .2, (1, 8, config.feature_size)).astype(np.float16)
    for length in range(1, 9):
        mask = np.zeros((1, 8), dtype=np.int32)
        mask[:, :length] = 1
        feeds = {"input_ids": ids, "target_features": features, "attention_mask": mask}
        expected, = evaluate(previous, feeds, weights)
        actual, = evaluate(current, feeds, weights)
        np.testing.assert_array_equal(actual, expected)
    assert sha256_file(prepared / "prepared.json") == before_manifest
    assert {key: sha256_file(prepared / record["path"]) for key, record in plan["weights"].items()} == before_weights

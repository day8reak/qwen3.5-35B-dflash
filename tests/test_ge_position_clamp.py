"""Bounded draft position clamp only; no generic INT32 extrema claim."""
import copy

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations, prepare_pair
from qwen35_moe_dflash.ge_ir import DTYPES, Graph, Node
from qwen35_moe_dflash.hashing import sha256_file
from test_ge_static_broadcast import _payload, _serialize_both


_WHERE = np.where


def _bounded_less(left, right):
    """Check the difference-based blue comparator is safe in this domain.

    This models the inspected arithmetic, not a device kernel invocation.
    Arbitrary signed INT32 pairs can overflow; those are deliberately excluded.
    """
    difference = right.astype(np.int64) - left.astype(np.int64)
    limits = np.iinfo(np.int32)
    assert np.all(difference >= limits.min) and np.all(difference <= limits.max)
    assert np.all(-difference >= limits.min) and np.all(-difference <= limits.max)
    model = np.subtract(right, left, dtype=np.int32).astype(np.float32) > 0
    np.testing.assert_array_equal(model, np.less(left, right))
    return model


def _execute(records, outputs, feeds):
    values = {}
    for index, record in enumerate(records):
        args = [values[(port["src"], port["src_port"])] for port in record["inputs"]]
        output, = record["outputs"]
        dtype, kind, attrs = DTYPES[output["dtype"]], record["type"], record["attrs"]
        if kind == "Data":
            value = feeds[record["name"]]
        elif kind == "Const":
            value = np.frombuffer(_payload(record), dtype=dtype).reshape(output["shape"])
        elif kind == "BroadcastTo":
            value = np.broadcast_to(args[0], tuple(int(x) for x in args[1]))
        elif kind == "SelectV2":
            assert args[0].dtype == np.bool_ and args[1].dtype == args[2].dtype == np.int32
            assert args[0].shape == args[1].shape == args[2].shape
            value = _WHERE(*args)
        elif kind == "Less":
            assert args[0].dtype == args[1].dtype == np.int32
            value = _bounded_less(*args)
        elif kind == "LogicalAnd":
            value = np.logical_and(*args)
        elif kind == "Cast":
            assert dtype in (np.int32, np.bool_)
            assert attrs["dst_type"] == {"int32": 3, "bool": 12}[output["dtype"]]
            value = args[0].astype(dtype)
        elif kind == "ReduceSum":
            assert args[0].dtype == dtype == np.int32
            value = args[0].sum(axis=tuple(int(x) for x in args[1]), dtype=np.int32,
                                keepdims=bool(attrs["keep_dims"]))
        elif kind in {"Sub", "Add"}:
            assert all(arg.dtype == np.int32 for arg in args)
            value = (np.subtract if kind == "Sub" else np.add)(*args, dtype=np.int32)
        elif kind == "Slice":
            value = args[0][tuple(slice(int(start), int(start + size))
                                  for start, size in zip(args[1], args[2]))]
        elif kind == "ConcatD":
            assert attrs["N"] == len(args)
            value = np.concatenate(args, axis=attrs["concat_dim"])
        else:
            raise AssertionError(f"unexpected position-producer operator {kind}")
        assert value.shape == output["shape"] and value.dtype == dtype
        values[(index, output["port"])] = value
    return [values[(index, records[index]["outputs"][port]["port"])] for index, port in outputs]


def _clamp_subgraph(draft):
    clamp = next(node for node in draft.nodes if node.output.name == "v12")
    assert clamp.kind == "where" and clamp.output.dtype == "int32"
    condition = next(node for node in draft.nodes if node.output is clamp.inputs[0])
    assert condition.kind == "lt" and condition.output.name == "context_position_negative"
    assert condition.inputs == (clamp.inputs[2], clamp.inputs[1])
    assert draft.constants[clamp.inputs[1].name].item() == 0
    graph = Graph("bounded_position_clamp")
    source = graph.input(clamp.inputs[2].name, clamp.inputs[2].shape, "int32")
    zero = graph.const(draft.constants[clamp.inputs[1].name], "int32", clamp.inputs[1].name)
    negative = graph.op("lt", source, zero, shape=source.shape, dtype="bool", name=condition.output.name)
    graph.outputs = [graph.op("where", negative, zero, source, shape=source.shape,
                              dtype="int32", name=clamp.output.name)]
    return graph


@pytest.mark.parametrize("sequence", [1, 8, 65])
def test_real_draft_v12_positions_for_every_nonempty_prefix(tmp_path, sequence):
    _, raw = configurations(tiny=False)
    config = Qwen35DFlashConfig.from_dict(raw)
    graph = build_draft(config, sequence)
    _clamp_subgraph(graph)  # Check the precise model-owned LT/Where contract.
    index, node = next((i, node) for i, node in enumerate(graph.nodes) if node.output.name == "v12")
    graph.nodes = graph.nodes[:index + 1]
    graph.weights = {}
    graph.outputs = [node.output] + [next(node.output for node in graph.nodes if node.output.name == name)
                                     for name in ("anchor_position", "context_valid")]
    records, inputs, outputs = _serialize_both(graph, tmp_path)
    assert [records[index]["name"] for index in inputs] == ["target_features", "input_ids", "attention_mask"]
    selected, = [record for record in records if record["type"] == "SelectV2"]
    assert selected["name"] == "v12"
    assert not any(record["type"] in {"Maximum", "Minimum"} for record in records)
    assert selected["outputs"][0]["shape"] == (1, sequence)
    assert all(port["dtype"] in {"int32", "int64", "bool"}
               for record in records if record["type"] != "Data" for port in record["outputs"])
    ids = np.full((1, sequence), config.vocab_size - 1, np.int32)
    features = np.zeros((1, sequence, config.feature_size), np.float16)
    for length in range(1, sequence + 1):
        mask = np.zeros((1, sequence), np.int32)
        mask[:, :length] = 1
        before = mask.tobytes()
        positions, anchor, context = _execute(records, outputs,
            {"target_features": features, "input_ids": ids, "attention_mask": mask})
        expected = np.array([[max(0, min(column, length - 2)) for column in range(sequence)]], np.int32)
        assert positions.tobytes() == expected.tobytes()
        if length == 1:
            assert not positions.any() and not context.any()
        assert anchor.item() == length - 1
        np.testing.assert_array_equal(context, np.arange(sequence)[None] < length - 1)
        assert mask.tobytes() == before


@pytest.mark.parametrize("sequence", [1, 65, 262128, (1 << 31) - 1])
def test_proven_position_domain_cannot_overflow_difference_comparison(sequence):
    # Scalar endpoint proof requires no enormous tensor/model allocation.
    # For every legal sequence, context count-1 lies in [-1,S-2].
    values = np.array([-1, sequence - 2], dtype=np.int32)
    zero = np.array(0, np.int32)
    result = _bounded_less(values, zero)
    np.testing.assert_array_equal(result, values == -1)


@pytest.mark.parametrize("sequence", [1 << 31, 1 << 40])
def test_position_counter_overflow_rejected_before_graph_allocation(sequence, monkeypatch):
    from qwen35_moe_dflash import ge_draft
    _, raw = configurations(tiny=True)
    config = Qwen35DFlashConfig.from_dict(raw)

    def unexpected_build(*args):
        raise AssertionError("oversized sequence must fail before graph or constant allocation")

    monkeypatch.setattr(ge_draft._DraftBuilder, "build", unexpected_build)
    with pytest.raises(ValueError, match="INT32 position counter"):
        build_draft(config, sequence)


def _previous_maximum_graph(draft):
    previous = copy.deepcopy(draft)
    retained = []
    for node in previous.nodes:
        if node.output.name == "context_position_negative":
            continue
        if node.output.name == "v12":
            assert node.kind == "where"
            node = Node("maximum", (node.inputs[2], node.inputs[1]), node.output, {})
        retained.append(node)
    previous.nodes = retained
    return previous


def test_serialized_clamp_preserves_complete_tiny_draft_outputs_weights_and_abi(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, 8, tiny=True)
    config = Qwen35DFlashConfig.from_dict(plan["draft_config"])
    draft = build_draft(config, 8)
    previous = _previous_maximum_graph(draft)
    abi_before = (draft.manifest()["inputs"], draft.manifest()["outputs"])
    assert abi_before == (previous.manifest()["inputs"], previous.manifest()["outputs"])
    hashes = {record["path"]: sha256_file(prepared / record["path"]) for record in plan["weights"].values()}
    manifest_hash = sha256_file(prepared / "prepared.json")
    weights = {key: np.fromfile(prepared / record["path"], dtype="<f2").reshape(record["shape"])
               for key, record in plan["weights"].items()}
    clamp = _clamp_subgraph(draft)
    records, _, outputs = _serialize_both(clamp, tmp_path / "serialized-clamp")
    assert {record["type"] for record in records} == {"Data", "Const", "Less", "SelectV2"}
    calls = []

    def serialized_where(*args):
        if len(args) != 3 or any(np.asarray(value).dtype != np.int32 for value in args[1:]):
            return _WHERE(*args)
        condition, zero, source = args
        assert zero.item() == 0 and source.shape == (1, 8)
        assert np.all(source >= -1) and np.all(source <= 6)
        np.testing.assert_array_equal(condition, source < 0)
        calls.append(source.copy())
        actual, = _execute(records, outputs, {clamp.inputs[0].name: source})
        return actual

    rng = np.random.default_rng(142)
    ids = rng.integers(0, config.vocab_size, (1, 8), dtype=np.int32)
    features = rng.normal(0, .2, (1, 8, config.feature_size)).astype(np.float16)
    for length in range(1, 9):
        mask = np.zeros((1, 8), np.int32)
        mask[:, :length] = 1
        feeds = {"input_ids": ids, "attention_mask": mask, "target_features": features}
        expected, = evaluate(previous, feeds, weights)
        with monkeypatch.context() as context:
            context.setattr(np, "where", serialized_where)
            actual, = evaluate(draft, feeds, weights)
        np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 8
    np.testing.assert_array_equal(calls[0], np.full((1, 8), -1, np.int32))
    assert (draft.manifest()["inputs"], draft.manifest()["outputs"]) == abi_before
    assert sha256_file(prepared / "prepared.json") == manifest_hash
    assert {path: sha256_file(prepared / path) for path in hashes} == hashes

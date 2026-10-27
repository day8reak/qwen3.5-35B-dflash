"""Host contracts for same-payload GDN core isolation, not device evidence."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash import op_probe
from qwen35_moe_dflash.artifacts import checked_file
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_target import _Builder, _chunk_gdn, _config, _linear_attention
from qwen35_moe_dflash.op_probe_gdn_accuracy import cases_from_prepared
from test_ge_select_lowering import RecipeReader
from test_op_probe_gdn_accuracy import case, prepared_source, _feeds, _outputs, _same_half, _weights


def _suite(case, **kwargs):
    return cases_from_prepared(case["prepared"], case["reference_path"], case["omc"], **kwargs)


def _linear_graph(observed):
    target, draft = configurations(tiny=True)
    graph = Graph("linear_observer_contract")
    x = graph.input("x", (1, 8, 16))
    config = _config(target, Qwen35DFlashConfig.from_dict(draft), 8)
    graph.outputs = [_linear_attention(_Builder(graph, gdn_indexed_copy="gather"), x,
                                      "target.layers.0.linear_attn", config, diagnostics=observed)]
    return graph


def _chunk_graph(observed):
    graph = Graph("chunk_observer_contract")
    inputs = [graph.input(name, (1, 8, 2, 8) if name in ("query", "key", "value") else (1, 8, 2))
              for name in ("query", "key", "value", "decay", "beta")]
    graph.outputs = [_chunk_gdn(_Builder(graph, gdn_indexed_copy="gather"), *inputs, diagnostics=observed)]
    return graph


@pytest.mark.parametrize("build", [_linear_graph, _chunk_graph])
def test_observer_references_existing_values_without_adding_nodes_constants_or_outputs(build):
    plain = build(None)
    observations = {}
    observed = build(observations)
    assert pickle.dumps(observed) == pickle.dumps(plain)
    assert observations
    assert all(observed._values[value.name] is value for value in observations.values())
    assert len(observed.outputs) == 1
    assert observations["padded_core"] is observations["chunk_0_padded_output"]
    assert observations["cropped_core"].shape == (1, 2, 8, 8)
    assert observations["chunk_core"].shape == (1, 8, 2, 8)


def test_new_suite_preserves_original_first_layer_and_default_three_cases(case, monkeypatch):
    def no_regeneration(*args, **kwargs):
        raise AssertionError("core diagnostics must consume the supplied prepared weights")
    monkeypatch.setattr("qwen35_moe_dflash.ge_weights.tiny_weights", no_regeneration)
    old, old_provenance = _suite(case)
    original = pickle.dumps(old)
    cases, provenance = _suite(case, core_stages=True)
    assert [item["name"] for item in cases] == ["gdn_layer0_core_stages", "gdn_core_replay"]
    full, replay = cases
    full_values, replay_values = map(_outputs, cases)
    original_top1, original_features = evaluate(case["graphs"][0], _feeds(), _weights(case))
    assert original_top1[0, :3].tolist() == [41, 41, 45]
    _same_half(full_values["layer0_final"], original_features[:, :, :16])
    _same_half(full_values["chunk_core"], _outputs(old[1])["chunk_core"])
    _same_half(replay_values["chunk_core"], full_values["chunk_core"])
    assert provenance["source"] == old_provenance["source"] == case["plan"]["source"]
    assert provenance["bindings"] == old_provenance["bindings"]
    for item in cases:
        assert len({value.name for value in item["graph"].outputs}) == len(item["graph"].outputs)
        assert len(item["output_labels"]) == len(item["expected"]) == len(item["graph"].outputs)
        assert item["emit_options"] == {"fp16_reductions": "matmul", "fp16_row_mul": "compact"}
        actual = evaluate(item["graph"], item["inputs"], item["weights"])
        for got, expected in zip(actual, item["expected"]):
            _same_half(got, expected)
    again, again_provenance = _suite(case)
    assert pickle.dumps(again) == original
    assert again_provenance == old_provenance


def test_replay_uses_exact_five_actual_cut_inputs_from_projected_and_convolved_weights(case):
    cases, provenance = _suite(case, core_stages=True)
    full, replay = cases
    values = _outputs(full)
    weights = _weights(case)
    projected = (values["input_norm"].astype(np.float32)
                 @ weights["target.layers.0.linear_attn.in_proj_qkv.weight"].astype(np.float32).T).astype(np.float16)
    _same_half(values["projected_qkv"], projected)
    padded = np.pad(projected.transpose(0, 2, 1), ((0, 0), (0, 0), (3, 0)))
    windows = np.stack([padded[0, :, tap:tap+8] for tap in range(4)], axis=-1)
    kernel = weights["target.layers.0.linear_attn.conv1d.weight"].reshape(48, 4, 1)
    convolved = (windows.astype(np.float32) @ kernel.astype(np.float32)).astype(np.float16)
    _same_half(values["convolved_qkv"], convolved)
    preactivation = convolved.reshape(1, 48, 8).astype(np.float32)
    activation = (preactivation / (1 + np.exp(-preactivation))).astype(np.float16).transpose(0, 2, 1)
    _same_half(values["activated_qkv"], activation)
    expected = {name: activation[:, :, offset:offset+16].reshape(1, 8, 2, 8)
                for name, offset in (("query", 0), ("key", 16), ("value", 32))}
    expected.update(decay=values["decay_input"], beta=values["beta_input"])
    assert list(replay["inputs"]) == ["query", "key", "value", "decay", "beta"]
    assert [value.name for value in replay["graph"].inputs] == list(replay["inputs"])
    assert not replay["weights"] and not replay["graph"].weights
    for name, array in replay["inputs"].items():
        _same_half(array, expected[name])
        assert array.flags.c_contiguous
        record = provenance["cut_inputs"][name]
        assert record["shape"] == list(array.shape) and record["dtype"] == "float16"
        assert record["bytes"] == array.nbytes
        assert record["sha256"] == hashlib.sha256(array.astype("<f2").tobytes(order="C")).hexdigest()
        assert record["source_stage"] == {"query": "query_input", "key": "key_input",
                                           "value": "value_input", "decay": "decay_input", "beta": "beta_input"}[name]
        _same_half(array, values[record["source_stage"]])
        assert full["graph"]._values[record["source_value_name"]].shape == array.shape
    assert provenance["policies"] == {"fp16_reductions": "matmul", "fp16_row_mul": "compact",
                                       "gdn_indexed_copy": "gather"}


def test_semantic_core_stages_preserve_padding_crop_transpose_and_first_token_diagonal(case):
    cases, provenance = _suite(case, core_stages=True)
    full, replay = cases
    values = _outputs(replay)
    full_values = _outputs(full)
    for label in values:
        _same_half(values[label], full_values[label])
    # S=8 retains the production chunk width64; each cropped row must come from
    # the first eight positions in both heads, then exchange head/sequence axes.
    padded = values["chunk_0_padded_output"]
    cropped, core = values["cropped_core"], values["chunk_core"]
    assert padded.shape == (1, 2, 64, 8)
    assert np.count_nonzero(padded[:, :, 8:, :]) == 0
    assert np.count_nonzero(padded[:, :, :8, :]) == 128
    _same_half(cropped, padded[:, :, :8, :])
    _same_half(core, cropped.transpose(0, 2, 1, 3))
    _same_half(values["chunk_0_intra"], padded)
    assert "padded_core" not in values  # Same existing Value is exposed once.
    for item in cases:
        stage_map = provenance["stage_maps"][item["name"]]
        assert stage_map["padded_core"] == stage_map["chunk_0_padded_output"]
        for label, record in stage_map.items():
            value = item["graph"].outputs[record["output_index"]]
            assert (record["name"], record["shape"], record["dtype"]) == (value.name, list(value.shape), value.dtype)
    np.testing.assert_array_equal(np.diagonal(values["decay_mask"], axis1=-2, axis2=-1), 1)
    np.testing.assert_array_equal(np.diagonal(values["triangular_transform"], axis1=-2, axis2=-1), 1)
    beta, raw_value = replay["inputs"]["beta"], replay["inputs"]["value"]
    for head in range(2):
        scaled_query = (values["normalized_query"][0, 0, head].astype(np.float32)
                        * np.float32(8 ** -.5)).astype(np.float16)
        key = values["normalized_key"][0, 0, head]
        dot = np.sum(scaled_query.astype(np.float32)*key.astype(np.float32), dtype=np.float32).astype(np.float16)
        weighted_value = (raw_value[0, 0, head].astype(np.float32)*np.float32(beta[0, 0, head])).astype(np.float16)
        first_token = (np.float32(dot)*weighted_value.astype(np.float32)).astype(np.float16)
        _same_half(core[0, 0, head], first_token)
        assert np.any(np.abs(first_token) >= np.finfo(np.float16).tiny)


def test_core_recipe_keeps_real_matmul_compact_and_gather_lowerings(case):
    cases, _ = _suite(case, core_stages=True)
    replay = cases[1]
    graph_before = pickle.dumps(replay["graph"])
    reductions, rows = {}, {}
    folder = case["root"] / "core-recipe"
    emit_cpp(replay["graph"], recipe_dir=folder, reduction_report=reductions,
             row_mul_report=rows, **replay["emit_options"])
    records, inputs, outputs = RecipeReader(folder / "graph.recipe").read()
    assert pickle.dumps(replay["graph"]) == graph_before
    assert [records[index]["name"] for index in inputs] == list(replay["inputs"])
    assert len(outputs) == len(replay["output_labels"])
    assert reductions["mode"] == "matmul" and reductions["counts"]["replaced"] > 0
    assert reductions["counts"]["scan_prefix_replaced"] == 64
    assert rows["mode"] == "compact" and rows["counts"]["replaced"] > 0
    assert any(item["type"] == "GatherV2" and item["name"].startswith("__indexed_copy/__gdn_copy/")
               for item in records)
    assert any(item["type"] == "MatMul" for item in records)


def test_core_cli_prepare_freezes_two_cases_and_exact_cut_inputs_without_vendor_tools(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only diagnostics must not call compiler, native builder or board tools")
    monkeypatch.setattr(op_probe, "compiler_resources", forbidden)
    monkeypatch.setattr(op_probe, "native_build", forbidden)
    monkeypatch.setattr(op_probe, "_logged", forbidden)
    output = case["root"] / "core-probe"
    args = op_probe.parser().parse_args([
        "--suite", "gdn-core-accuracy", "--prepared", str(case["prepared"]),
        "--reference", str(case["reference_path"]), "--omc", str(case["omc"]),
        "--prepare-only", "--output", str(output)])
    assert op_probe.run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "PREPARED" and manifest["board_requested"] is False
    assert [item["name"] for item in manifest["cases"]] == ["gdn_layer0_core_stages", "gdn_core_replay"]
    full, replay = manifest["cases"]
    assert full["output_labels"][:10] == ["layer0_final", "input_norm", "projected_qkv", "convolved_qkv",
        "activated_qkv", "query_input", "key_input", "value_input", "beta_input", "decay_input"]
    expected_replay_labels = ["normalized_query", "normalized_key", "scaled_query", "weighted_value",
        "decay_prefix", "decay_mask", "triangular_transform", "transformed_value", "chunk_0_scores",
        "chunk_0_new_value", "chunk_0_intra", "chunk_0_padded_output", "cropped_core", "chunk_core"]
    assert replay["output_labels"] == expected_replay_labels
    assert full["output_labels"][10:] == expected_replay_labels
    assert manifest["provenance"]["source"] == case["plan"]["source"]
    assert [item["name"] for item in replay["input_tensors"]] == ["query", "key", "value", "decay", "beta"]
    full_outputs = {item["label"]: item for item in full["outputs"]}
    for tensor in replay["input_tensors"]:
        cut = manifest["provenance"]["cut_inputs"][tensor["name"]]
        assert all(tensor[key] == cut[key] for key in ("sha256", "bytes", "shape", "dtype"))
        input_path = checked_file(Path(replay["directory"]), tensor)
        expected_path = checked_file(Path(full["directory"]), full_outputs[cut["source_stage"]]["reference"])
        assert input_path.read_bytes() == expected_path.read_bytes()
    for item in (full, replay):
        assert item["probe_suite"] == "gdn-core-accuracy"
        assert item["emit_options"] == {"fp16_reductions": "matmul", "fp16_row_mul": "compact"}
        for payload in item["construction_files"]:
            checked_file(Path(item["directory"]), payload)

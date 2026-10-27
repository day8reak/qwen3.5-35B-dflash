"""Independent host contracts for bounded GDN arithmetic/layout diagnostics.

Synthetic OMC fixtures bind provenance only. No device support or numerical
success is inferred from these host tests.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from qwen35_moe_dflash import op_probe
from qwen35_moe_dflash.artifacts import checked_file
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.op_probe_gdn_accuracy import cases_from_prepared as core_cases
from test_ge_select_lowering import RecipeReader
from test_op_probe_gdn_accuracy import case, prepared_source, _same_half, _weights


EXPECTED_NAMES = ["gc_io_actual", "gc_io_tagged", "gc_conv_layout_actual", "gc_conv_layout_tagged",
    "gc_conv_bmm_actual_const", "gc_conv_bmm_actual_dynamic", "gc_conv_bmm_tagged_const", "gc_conv_bmm_tagged_dynamic",
    "gc_l2_query_actual", "gc_l2_query_normal", "gc_l2_key_actual", "gc_l2_key_normal",
    "gc_pad_layout_actual", "gc_pad_layout_tagged", "gc_weighted_mul_actual_chain", "gc_weighted_mul_actual_padded",
    "gc_weighted_mul_tagged_chain", "gc_weighted_mul_tagged_padded", "gc_scan_actual_chain", "gc_scan_actual_padded",
    "gc_scan_moderate_chain", "gc_scan_moderate_padded", "gc_final_bmm_actual", "gc_final_bmm_tagged",
    "gc_core_normalized_cut", "gc_core_final_cut"]


def _suite(case):
    from qwen35_moe_dflash.op_probe_gdn_comprehensive import cases_from_prepared
    items, provenance = cases_from_prepared(case["prepared"], case["reference_path"], case["omc"])
    return {item["name"]: item for item in items}, provenance


def _outputs(item):
    return dict(zip(item["output_labels"], item["expected"]))


def _same_negated_half(actual, source):
    np.testing.assert_array_equal(actual.view(np.uint16), source.view(np.uint16) ^ np.uint16(0x8000))


def _original_stages(case):
    pair, provenance = core_cases(case["prepared"], case["reference_path"], case["omc"], core_stages=True)
    full, replay = pair
    return _outputs(full), replay["inputs"], provenance


def _matmul(left, right, *, transpose=False):
    if transpose:
        right = right.swapaxes(-1, -2)
    return (left.astype(np.float32) @ right.astype(np.float32)).astype(np.float16)


def _l2_stages(x):
    square = np.square(x.astype(np.float32)).astype(np.float16)
    total = np.sum(square.astype(np.float32), axis=-1, keepdims=True).astype(np.float16)
    epsilon = (total.astype(np.float32) + np.float32(1e-6)).astype(np.float16)
    gain = (1 / np.sqrt(epsilon.astype(np.float32))).astype(np.float16)
    result = (x.astype(np.float32) * gain.astype(np.float32)).astype(np.float16)
    return {"square": square, "sum": total, "epsilon": epsilon, "gain": gain, "result": result}


def test_batch_is_bounded_source_bound_and_does_not_regenerate_or_modify_payloads(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("comprehensive diagnostics must read the supplied weights")
    monkeypatch.setattr("qwen35_moe_dflash.ge_weights.tiny_weights", forbidden)
    before = {str(path.relative_to(case["root"])): sha256_file(path)
              for path in case["root"].rglob("*") if path.is_file()}
    original_graphs = pickle.dumps(case["graphs"])
    batch, provenance = _suite(case)
    assert len(batch) == 26 and set(batch) == set(EXPECTED_NAMES)
    assert provenance["source"] == case["plan"]["source"]
    assert provenance["bindings"] == {
        "prepared_manifest_sha256": sha256_file(case["prepared"] / "prepared.json"),
        "reference_sha256": sha256_file(case["reference_path"]),
        "omc_manifest_sha256": sha256_file(case["omc"] / "manifest.json")}
    assert provenance["case_names"] == list(batch)
    assert provenance["policies"] == {"fp16_reductions": "matmul", "fp16_row_mul": "compact",
                                       "gdn_indexed_copy": "gather"}
    for name, item in batch.items():
        graph = item["graph"]
        assert item["probe_suite"] == "gdn-comprehensive" and item["probe_family"] == "gdn"
        assert item["emit_options"] == {"fp16_reductions": "matmul", "fp16_row_mul": "compact"}
        assert len(graph.inputs) + len(graph.outputs) <= 32
        assert len(graph.outputs) == len(item["expected"]) == len(item["output_labels"])
        assert all(len(v.shape) <= 4 and np.prod(v.shape)*2 <= 32768 for v in graph.inputs + graph.outputs)
        for label, record in provenance["stage_maps"][name].items():
            value = graph.outputs[record["output_index"]]
            assert (value.name, list(value.shape), value.dtype) == (record["name"], record["shape"], record["dtype"])
        for label, array in item["inputs"].items():
            assert array.dtype == np.float16 and array.flags.c_contiguous and np.isfinite(array).all()
            assert provenance["fixture_inputs"][name][label] == {
                "shape": list(array.shape), "dtype": "float16", "bytes": array.nbytes,
                "sha256": hashlib.sha256(array.astype("<f2").tobytes()).hexdigest()}
    assert pickle.dumps(case["graphs"]) == original_graphs
    after = {str(path.relative_to(case["root"])): sha256_file(path)
             for path in case["root"].rglob("*") if path.is_file()}
    assert before == after


def test_io_controls_preserve_each_actual_shape_and_reverse_signs_exactly(case):
    batch, _ = _suite(case)
    original, cut, _ = _original_stages(case)
    for fixture in ("actual", "tagged"):
        item = batch["gc_io_" + fixture]
        assert list(item["inputs"]) == ["query", "key", "value", "decay", "beta", "conv_shape"]
        for name, actual in _outputs(item).items():
            source = item["inputs"][name]
            np.testing.assert_array_equal(actual.view(np.uint16), source.view(np.uint16) ^ np.uint16(0x8000))
            if fixture == "actual":
                _same_half(source, original["convolved_qkv"] if name == "conv_shape" else cut[name])
            elif source.size > 16:
                assert np.unique(source).size > 16
        assert all(tolerance == {"rtol": 0.0, "atol": 0.0} for tolerance in item["tolerances"])


@pytest.mark.parametrize("fixture", ["actual", "tagged"])
def test_convolution_views_and_static_dynamic_rhs_use_identical_exact_weights(case, fixture):
    batch, _ = _suite(case)
    layout = batch["gc_conv_layout_" + fixture]
    projected = layout["inputs"]["projected"]
    transposed = projected.transpose(0, 2, 1)
    padded = np.pad(transposed, ((0, 0), (0, 0), (3, 0)))
    windows = np.stack([padded[0, :, start:start+8] for start in range(4)], axis=-1)
    for label, expected in (("transposed", transposed), ("padded", padded), ("windows", windows)):
        _same_half(_outputs(layout)[label], expected)
    assert all(t["rtol"] == t["atol"] == 0 for t in layout["tolerances"])
    key = "target.layers.0.linear_attn.conv1d.weight"
    kernel = _weights(case)[key]
    expected = _matmul(windows, kernel.reshape(48, 4, 1))
    for mode in ("const", "dynamic"):
        item = batch[f"gc_conv_bmm_{fixture}_{mode}"]
        _same_half(item["inputs"]["windows"], windows)
        _same_half(item["inputs"]["projected"], projected)
        _same_half(item["weights"][key] if mode == "const" else item["inputs"]["kernel"], kernel)
        assert set(item["graph"].weights) == ({key} if mode == "const" else set())
        actual = _outputs(item)
        if mode == "const":
            assert "kernel_view" not in actual
        else:
            _same_half(actual["kernel_view"], kernel.reshape(48, 4, 1))
        _same_half(actual["produced_windows"], windows)
        _same_half(actual["convolved"], expected)
        _same_half(actual["produced_convolved"], expected)
        _same_half(actual["activation_view"], expected.reshape(1, 48, 8))
        x = expected.reshape(1, 48, 8).astype(np.float32)
        _same_half(actual["activated"], (x / (1 + np.exp(-x))).astype(np.float16).transpose(0, 2, 1))
    if fixture == "actual":
        original, _, _ = _original_stages(case)
        _same_half(projected, original["projected_qkv"])
        _same_half(expected, original["convolved_qkv"])


@pytest.mark.parametrize("kind", ["query", "key"])
@pytest.mark.parametrize("fixture", ["actual", "normal"])
def test_l2_exposes_half_boundaries_and_independent_dense_normal_domain_control(case, kind, fixture):
    batch, _ = _suite(case)
    item = batch[f"gc_l2_{kind}_{fixture}"]
    x, dense = item["inputs"]["x"], item["inputs"]["dense_x"]
    _same_half(dense, x.reshape(16, 8))
    expected = _l2_stages(x)
    values = _outputs(item)
    for label, reference in expected.items():
        _same_half(values[label], reference)
        _same_half(values["dense_" + label], reference.reshape(16, -1))
    assert np.max(np.abs(values["result"])) <= 1
    if fixture == "normal":
        assert np.all(expected["square"] >= np.finfo(np.float16).tiny)
        assert np.isfinite(expected["sum"]).all()
        assert np.unique(expected["sum"]).size >= 4  # Gain mispairing must be observable.
    else:
        _, cut, _ = _original_stages(case)
        _same_half(x, cut[kind])
        assert np.count_nonzero((expected["square"] > 0) & (expected["square"] < np.finfo(np.float16).tiny)) > 64
    # The dense branch must not consume a view produced by the rank4 branch.
    producers = {node.output.name: node for node in item["graph"].nodes}
    def leaves(value):
        return {value.name} if value.name not in producers else set().union(*(leaves(v) for v in producers[value.name].inputs))
    dense_output = item["graph"].outputs[item["output_labels"].index("dense_result")]
    assert leaves(dense_output) == {"dense_x"}


@pytest.mark.parametrize("fixture", ["actual", "tagged"])
def test_pad_and_weighted_mul_controls_distinguish_both_heads_and_dense_runtime_operands(case, fixture):
    batch, _ = _suite(case)
    layout = batch["gc_pad_layout_" + fixture]
    outputs = _outputs(layout)
    for kind, source in layout["inputs"].items():
        perm = (0, 2, 1, 3) if source.ndim == 4 else (0, 2, 1)
        transposed = source.transpose(perm)
        pad = ((0, 0), (0, 0), (0, 56)) + (((0, 0),) if source.ndim == 4 else ())
        padded = np.pad(transposed, pad)
        _same_half(outputs[kind + "_transposed"], transposed)
        _same_half(outputs[kind + "_padded"], padded)
        _same_half(outputs[kind + "_view"], padded.reshape(outputs[kind + "_view"].shape))
        assert np.count_nonzero(padded[:, :, :8]) == source.size
        assert np.count_nonzero(padded[:, :, 8:]) == 0
    left, right = outputs["value_padded"], outputs["beta_view"]
    expected = (left.astype(np.float32) * right.astype(np.float32)).astype(np.float16)
    for mode in ("chain", "padded"):
        item = batch[f"gc_weighted_mul_{fixture}_{mode}"]
        actual = _outputs(item)
        if mode == "padded":
            _same_negated_half(actual["value_carrier_neg"], left)
            _same_negated_half(actual["beta_carrier_neg"], right)
        else:
            _same_half(actual["value_carrier"], left)
            _same_half(actual["beta_carrier"], right)
        _same_half(actual["weighted"], expected)
        _same_half(actual["dense_weighted"], expected.reshape(128, 8))
        _same_half(item["inputs"]["dense_value"], left.reshape(128, 8))
        _same_half(item["inputs"]["dense_beta"], right.reshape(128, 1))
    if fixture == "actual":
        assert np.count_nonzero(expected[:, :, :8]) == 128


@pytest.mark.parametrize("fixture", ["actual", "moderate"])
def test_scan_retains_nonzero_normal_prefixes_through_zero_padding(case, fixture):
    batch, _ = _suite(case)
    chain, padded = (batch[f"gc_scan_{fixture}_{mode}"] for mode in ("chain", "padded"))
    source = chain["inputs"]["decay"]
    carrier = np.pad(source.transpose(0, 2, 1), ((0, 0), (0, 0), (0, 56))).reshape(2, 1, 64)
    expected = np.cumsum(carrier.astype(np.float32), axis=-1).astype(np.float16)
    _same_half(padded["inputs"]["decay_padded"], carrier)
    _same_half(_outputs(chain)["carrier"], carrier)
    _same_negated_half(_outputs(padded)["carrier_neg"], carrier)
    for item in (chain, padded):
        _same_half(_outputs(item)["prefix"], expected)
    assert np.all(expected < 0) and np.all(np.abs(expected) >= np.finfo(np.float16).tiny)
    assert np.count_nonzero(expected) == 128
    _same_half(expected[:, :, 8:], np.broadcast_to(expected[:, :, 7:8], (2, 1, 56)))


def test_observers_avoid_direct_data_and_constant_only_outputs_without_changing_consumers(case):
    batch, _ = _suite(case)
    for item in batch.values():
        graph = item["graph"]
        producers = {node.output.name: node for node in graph.nodes}
        runtime_values = {value.name for value in graph.inputs}
        for node in graph.nodes:
            if any(value.name in runtime_values for value in node.inputs):
                runtime_values.add(node.output.name)
        assert all(value.name in producers and value.name in runtime_values for value in graph.outputs)
        assert len({value.name for value in graph.outputs}) == len(graph.outputs)

    for fixture in ("actual", "tagged"):
        item = batch[f"gc_weighted_mul_{fixture}_padded"]
        nodes = {node.output.name: node for node in item["graph"].nodes}
        outputs = dict(zip(item["output_labels"], item["graph"].outputs))
        mul = nodes[outputs["weighted"].name]
        assert mul.kind == "mul" and [v.name for v in mul.inputs] == ["value_padded", "beta_padded"]
        for label, operand in zip(("value_carrier_neg", "beta_carrier_neg"), mul.inputs):
            observer = nodes[outputs[label].name]
            assert observer.kind == "neg" and observer.inputs == (operand,)
            assert item["tolerances"][item["output_labels"].index(label)] == {"rtol": 0, "atol": 0}
    for fixture in ("actual", "moderate"):
        item = batch[f"gc_scan_{fixture}_padded"]
        nodes = {node.output.name: node for node in item["graph"].nodes}
        outputs = dict(zip(item["output_labels"], item["graph"].outputs))
        scan, observer = nodes[outputs["prefix"].name], nodes[outputs["carrier_neg"].name]
        assert scan.kind == "cumsum" and scan.inputs[0].name == "decay_padded"
        assert observer.kind == "neg" and observer.inputs == scan.inputs


@pytest.mark.parametrize("constant_only", [False, True])
def test_factory_rejects_direct_input_and_constant_foldable_observers(constant_only):
    from qwen35_moe_dflash.ge_ir import Graph
    from qwen35_moe_dflash.op_probe_gdn_comprehensive import _make
    graph = Graph("invalid_observer")
    source = graph.input("x", (1, 8))
    if constant_only:
        source = graph.op("neg", graph.const(np.ones((1, 8), np.float16)), shape=(1, 8))
    with pytest.raises(ValueError, match="diagnostic output requires a runtime operation"):
        _make(graph, {"x": np.ones((1, 8), np.float16)}, {"carrier": source}, "invalid observer",
              {"fp16_reductions": "matmul", "fp16_row_mul": "compact"})


def test_final_bmm_and_core_cuts_retain_original_reachable_arithmetic(case):
    batch, _ = _suite(case)
    original_pair, _ = core_cases(case["prepared"], case["reference_path"], case["omc"], core_stages=True)
    original = original_pair[1]
    original_values = _outputs(original)
    source_nodes = {node.output.name: node for node in original["graph"].nodes}
    for fixture in ("actual", "tagged"):
        item = batch["gc_final_bmm_" + fixture]
        expected = _matmul(item["inputs"]["scores"], item["inputs"]["new_value"])
        _same_half(_outputs(item)["intra"], expected)
        if fixture == "actual":
            _same_half(expected, original_values["chunk_0_intra"])
    for name in ("gc_core_normalized_cut", "gc_core_final_cut"):
        item = batch[name]
        for label, value in _outputs(item).items():
            _same_half(value, original_values[label])
        for node in item["graph"].nodes:
            source = source_nodes[node.output.name]
            assert (node.kind, node.attrs, node.output.shape, node.output.dtype) == (
                source.kind, source.attrs, source.output.shape, source.output.dtype)
        for label, constant in item["graph"].constants.items():
            np.testing.assert_array_equal(constant, original["graph"].constants[label])
    final = batch["gc_core_final_cut"]
    assert set(final["inputs"]) == {"scores", "new_value", "inter"}
    assert np.count_nonzero(final["inputs"]["inter"]) == 0
    assert any(node.kind == "add" and any(v.name == "inter" for v in node.inputs) for node in final["graph"].nodes)
    assert set(batch["gc_core_normalized_cut"]["inputs"]) == {"normalized_query", "normalized_key", "value", "decay", "beta"}


def test_real_cli_freezes_all_cases_inputs_outputs_and_lowered_recipes_without_vendor_calls(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only comprehensive diagnostics must not invoke vendor/board tools")
    for name in ("compiler_resources", "native_build", "_logged"):
        monkeypatch.setattr(op_probe, name, forbidden)
    output = case["root"] / "comprehensive"
    args = op_probe.parser().parse_args(["--suite", "gdn-comprehensive", "--prepared", str(case["prepared"]),
        "--reference", str(case["reference_path"]), "--omc", str(case["omc"]), "--prepare-only", "--output", str(output)])
    assert op_probe.run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "PREPARED" and manifest["board_requested"] is False
    assert len(manifest["cases"]) == 26 and {item["name"] for item in manifest["cases"]} == set(EXPECTED_NAMES)
    all_records = {}
    for item in manifest["cases"]:
        folder = Path(item["directory"])
        for payload in item["construction_files"]:
            checked_file(folder, payload)
        for tensor in item["input_tensors"]:
            frozen = manifest["provenance"]["fixture_inputs"][item["name"]][tensor["name"]]
            assert all(frozen[key] == tensor[key] for key in ("shape", "dtype", "bytes", "sha256"))
        records, inputs, outputs = RecipeReader(folder / "recipe/graph.recipe").read()
        assert [records[index]["name"] for index in inputs] == [tensor["name"] for tensor in item["input_tensors"]]
        assert len(outputs) == len(item["outputs"]) == len(item["output_labels"])
        assert len(set(inputs)) == len(inputs) and len(set(outputs)) == len(outputs)
        assert not set(inputs) & {index for index, _ in outputs}
        assert all(records[index]["type"] not in {"Data", "Const"} for index, _ in outputs)
        for (index, _), output_record in zip(outputs, item["outputs"]):
            if output_record["label"].endswith("_neg"):
                assert records[index]["type"] == "Neg"
                assert output_record["rtol"] == output_record["atol"] == 0
                assert output_record["reference"]["sha256"] == sha256_file(folder / output_record["expected"])
        all_records[item["name"]] = records
    l2 = all_records["gc_l2_query_actual"]
    assert any(record["type"] == "MatMul" and "/reduction/matmul" in record["name"] for record in l2)
    assert any(record["type"] == "Mul" and "/row_mul/product" in record["name"] for record in l2)
    scan = all_records["gc_scan_actual_padded"]
    assert sum(record["type"] == "MatMul" and "/reduction/matmul" in record["name"] for record in scan) == 64
    core = all_records["gc_core_normalized_cut"]
    assert any(record["type"] == "GatherV2" and "__indexed_copy/__gdn_copy/" in record["name"] for record in core)
    const = all_records["gc_conv_bmm_actual_const"]
    dynamic = all_records["gc_conv_bmm_actual_dynamic"]
    assert any(record["type"] == "BatchMatMulV2" for record in const)
    assert any(record["type"] == "BatchMatMulV2" for record in dynamic)

"""Freeze weighted diagnostic probes without using vendor or board tools."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from qwen35_moe_dflash import op_probe
from qwen35_moe_dflash.artifacts import checked_file
from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.op_probe_cases import FP16_TOLERANCE, cases
from test_ge_select_lowering import RecipeReader
from test_op_probe import _compiler, _no_tool, run_root
from test_op_probe_gdn_accuracy import case as gdn_source_case, prepared_source


def _weighted_case():
    graph = Graph("weighted_diagnostic")
    x = graph.input("x", (1, 8, 2, 8))
    weight = graph.weight("target.projection.weight", (8, 8))
    projected = graph.op("matmul", x, weight, shape=x.shape, attrs={"transpose_b": True})
    mean = graph.op("reduce_mean", projected, shape=(1, 8, 2, 1),
                    attrs={"axes": [-1], "keepdims": True})
    result = graph.op("mul", projected, mean, shape=x.shape)
    graph.outputs = [result, mean]
    weights = {weight.name: (np.arange(64, dtype=np.float16).reshape(8, 8) / 64).T}
    weights[weight.name][0, 0] = np.float16(-0.0)
    inputs = {"x": (np.arange(128, dtype=np.float16).reshape(x.shape) - 64) / 64}
    return {"name": graph.name, "graph": graph, "description": "Test-only weighted diagnostic",
            "inputs": inputs, "weights": weights, "expected": evaluate(graph, inputs, weights),
            "tolerances": [dict(FP16_TOLERANCE) for _ in graph.outputs],
            "output_labels": ["scaled_projection", "row_mean"],
            "probe_suite": "gdn-accuracy", "probe_family": "gdn"}


def test_weight_payloads_are_frozen_and_hash_bound_before_recipe_emission(run_root):
    case = _weighted_case()
    record = op_probe.prepare_case(case, run_root / "probe")
    folder = Path(record["directory"])
    key, payload = next(iter(record["weights"].items()))
    path = checked_file(folder, payload)
    assert payload["path"] == "weights/w00000.bin" and payload["dtype"] == "float16"
    assert payload["shape"] == [8, 8] and payload["bytes"] == 128
    assert path.read_bytes() == case["weights"][key].astype("<f2").tobytes(order="C")
    assert path.read_bytes()[:2] == b"\x00\x80"  # Preserve signed zero.
    assert any(item["path"] == payload["path"] and item["sha256"] == payload["sha256"]
               for item in record["construction_files"])
    records, _, output_ports = RecipeReader(folder / "recipe/graph.recipe").read()
    frozen, = [item for item in records if item["name"] == key]
    assert frozen["type"] == "Const" and frozen["constant"] == (str(path), 0, 128)
    assert len(output_ports) == len(record["outputs"]) == 2
    assert record["output_labels"] == [output["label"] for output in record["outputs"]]
    assert all((output["rtol"], output["atol"]) == (.01, .002) for output in record["outputs"])
    # Later caller mutation cannot alter the frozen recipe input.
    case["weights"][key][:] = 0
    checked_file(folder, payload)
    assert path.read_bytes()[2:] != bytes(126)


@pytest.mark.parametrize("fault", ["missing", "extra", "dtype", "shape", "list", "nonfinite", "graph_dtype"])
def test_invalid_weights_are_rejected_without_implicit_conversion_or_files(run_root, fault):
    case = _weighted_case()
    key = next(iter(case["weights"]))
    if fault == "missing":
        case.pop("weights")
    elif fault == "extra":
        case["weights"]["extra"] = np.ones((8,), np.float16)
    elif fault == "dtype":
        case["weights"][key] = case["weights"][key].astype(np.float32)
    elif fault == "shape":
        case["weights"][key] = case["weights"][key].reshape(64)
    elif fault == "list":
        case["weights"][key] = case["weights"][key].tolist()
    elif fault == "nonfinite":
        case["weights"][key][0, 0] = np.nan
    else:
        graph = Graph("integer_weight")
        x = graph.input("x", (8,), "int32")
        weight = graph.weight(key, (8,), "int32")
        graph.outputs = [graph.op("add", x, weight, shape=x.shape, dtype="int32")]
        case.update(graph=graph, weights={key: np.ones(8, np.int32)}, expected=[np.ones(8, np.int32)],
                    tolerances=[dict(FP16_TOLERANCE)], output_labels=["integer_output"])
    with pytest.raises(ValueError, match="weight"):
        op_probe.prepare_case(case, run_root / "invalid")
    assert not (run_root / "invalid").exists()


def test_weight_tampering_is_rejected_before_any_compiler_call(run_root, monkeypatch):
    resources, commands = _compiler(monkeypatch, run_root)
    record = op_probe.prepare_case(_weighted_case(), run_root / "probe")
    path = Path(record["directory"]) / next(iter(record["weights"].values()))["path"]
    data = bytearray(path.read_bytes()); data[3] ^= 1; path.write_bytes(data)
    result = op_probe.compile_case(record, resources, 60)
    assert result["status"] == "FAILED" and "SHA256/size mismatch" in result["error"]
    assert not commands


@pytest.mark.parametrize("field", ["expected", "tolerances", "output_labels"])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_output_count_mismatch_never_silently_truncates_zip(run_root, field, change):
    case = _weighted_case()
    if change == "missing":
        case[field].pop()
    else:
        case[field].append(deepcopy(case[field][0]))
    with pytest.raises(ValueError, match="output"):
        op_probe.prepare_case(case, run_root / "invalid")
    assert not (run_root / "invalid").exists()


@pytest.mark.parametrize("options", [None, [], {"external_weights": True}, {"fold_constant_arithmetic": False},
                                      {"fp16_reductions": "other"}, {"fp16_row_mul": "other"},
                                      {"fp16_reductions": True}, {"fp16_row_mul": []}])
def test_emitter_options_accept_only_named_supported_policies(run_root, options):
    case = _weighted_case(); case["emit_options"] = options
    with pytest.raises(ValueError, match="emit_options"):
        op_probe.prepare_case(case, run_root / "invalid")
    assert not (run_root / "invalid").exists()


@pytest.mark.parametrize("options", [{}, {"fp16_reductions": "native", "fp16_row_mul": "native"},
                                      {"fp16_reductions": "matmul", "fp16_row_mul": "compact"}])
def test_explicit_policies_reach_real_recipe_and_default_stays_native(run_root, options):
    case = _weighted_case(); case["emit_options"] = options
    record = op_probe.prepare_case(case, run_root / "probe")
    records, _, _ = RecipeReader(Path(record["directory"]) / "recipe/graph.recipe").read()
    transformed = options.get("fp16_reductions") == "matmul"
    assert any(item["type"] == "ReduceMean" for item in records) is not transformed
    assert any("/reduction/matmul" in item["name"] for item in records) is transformed
    assert any("/row_mul/product" in item["name"] for item in records) is transformed
    assert record.get("emit_options", {}) == options


def test_historical_case_adds_no_weight_or_option_fields(run_root):
    record = op_probe.prepare_case(next(cases()), run_root / "probe")
    assert not {"weights", "emit_options", "output_labels"}.intersection(record)
    assert not (Path(record["directory"]) / "weights").exists()


def _factory(monkeypatch, result=None):
    calls = []
    provenance = {"source": {"payload": "test-hash"}, "scope": "mock fixture, no device evidence"}
    module = ModuleType("qwen35_moe_dflash.op_probe_gdn_accuracy")

    def factory(prepared, reference, omc):
        calls.append((prepared, reference, omc))
        return ([_weighted_case()] if result is None else result), provenance

    module.cases_from_prepared = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(op_probe, "compiler_resources", _no_tool)
    monkeypatch.setattr(op_probe, "native_build", _no_tool)
    return calls, provenance


def test_accuracy_suite_calls_factory_and_retains_provenance(run_root, monkeypatch):
    calls, provenance = _factory(monkeypatch)
    output = run_root / "accuracy"
    args = op_probe.parser().parse_args([
        "--suite", "gdn-accuracy", "--prepared", "prepared", "--reference", "reference", "--omc", "omc",
        "--prepare-only", "--output", str(output)])
    assert op_probe.run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert calls == [("prepared", "reference", "omc")]
    assert manifest["provenance"] == provenance
    assert manifest["selection"]["selected_cases"] == ["weighted_diagnostic"]
    assert manifest["cases"][0]["probe_suite"] == "gdn-accuracy"


@pytest.mark.parametrize("missing", ["prepared", "reference", "omc"])
def test_accuracy_suite_requires_all_source_arguments_before_factory(run_root, monkeypatch, missing):
    calls, _ = _factory(monkeypatch)
    argv = ["--suite", "gdn-accuracy", "--list-cases"]
    for name in ("prepared", "reference", "omc"):
        if name != missing:
            argv += ["--" + name, name]
    with pytest.raises(ValueError, match="requires --prepared, --reference and --omc"):
        op_probe.run(op_probe.parser().parse_args(argv))
    assert not calls


@pytest.mark.parametrize("selection", [[], ["--suite", "remaining-cpu"], ["--case", "silu"]])
@pytest.mark.parametrize("source", ["prepared", "reference", "omc"])
def test_source_arguments_rejected_for_historical_selection(monkeypatch, selection, source):
    calls, _ = _factory(monkeypatch)
    with pytest.raises(ValueError, match="require --suite gdn-accuracy"):
        op_probe.run(op_probe.parser().parse_args([*selection, "--" + source, "source", "--list-cases"]))
    assert not calls


def test_accuracy_suite_rejects_exclusions_before_factory(monkeypatch):
    calls, _ = _factory(monkeypatch)
    with pytest.raises(ValueError, match="does not accept --exclude-family"):
        op_probe.run(op_probe.parser().parse_args([
            "--suite", "gdn-accuracy", "--prepared", "p", "--reference", "r", "--omc", "o",
            "--exclude-family", "gdn", "--list-cases"]))
    assert not calls


def test_accuracy_case_listing_uses_factory_without_preparation(monkeypatch, capsys):
    calls, _ = _factory(monkeypatch)
    assert op_probe.run(op_probe.parser().parse_args([
        "--suite", "gdn-accuracy", "--prepared", "p", "--reference", "r", "--omc", "o",
        "--list-cases"])) == 0
    assert calls == [("p", "r", "o")] and "weighted_diagnostic:" in capsys.readouterr().out


def test_empty_accuracy_suite_is_rejected_before_tools(monkeypatch):
    _factory(monkeypatch, result=[])
    with pytest.raises(ValueError, match="no cases"):
        op_probe.run(op_probe.parser().parse_args([
            "--suite", "gdn-accuracy", "--prepared", "p", "--reference", "r", "--omc", "o",
            "--list-cases"]))


def test_real_accuracy_factory_prepares_all_weighted_cases_without_tools(gdn_source_case, monkeypatch):
    source = gdn_source_case
    monkeypatch.setattr(op_probe, "compiler_resources", _no_tool)
    monkeypatch.setattr(op_probe, "native_build", _no_tool)
    output = source["root"] / "probe"
    assert op_probe.run(op_probe.parser().parse_args([
        "--suite", "gdn-accuracy", "--prepared", str(source["prepared"]),
        "--reference", str(source["reference_path"]), "--omc", str(source["omc"]),
        "--prepare-only", "--output", str(output)])) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "PREPARED" and manifest["board_requested"] is False
    assert manifest["provenance"]["source"] == source["plan"]["source"]
    assert [len(case.get("weights", {})) for case in manifest["cases"]] == [19, 19, 0]
    assert [case["name"] for case in manifest["cases"]] == [
        "gdn_layer0_final", "gdn_layer0_stages", "gdn_layer0_rsqrt"]
    for case in manifest["cases"]:
        folder = Path(case["directory"])
        assert case["emit_options"] == {"fp16_reductions": "matmul", "fp16_row_mul": "compact"}
        assert len(case["outputs"]) == len(case["output_labels"])
        for payload in case["construction_files"]:
            checked_file(folder, payload)
        records, _, ports = RecipeReader(folder / "recipe/graph.recipe").read()
        assert len(ports) == len(case["outputs"])
        assert all((item["rtol"], item["atol"]) == (.01, .002) for item in case["outputs"])
        if case.get("weights"):
            assert any("/reduction/matmul" in record["name"] for record in records)


@pytest.mark.parametrize("fault", [None, "missing_ddk", "wrong_ddk", "wrong_omg"])
def test_accuracy_compilation_requires_original_ddk_and_omg_before_artifacts(run_root, monkeypatch, fault):
    _, provenance = _factory(monkeypatch)
    resources, commands = _compiler(monkeypatch, run_root)
    provenance.update(ddk_id="original-ddk", omg_sha256=sha256_file(resources[3]))
    monkeypatch.setenv("CPRO_DDK_ID", "original-ddk")
    if fault == "missing_ddk":
        monkeypatch.delenv("CPRO_DDK_ID")
    elif fault == "wrong_ddk":
        monkeypatch.setenv("CPRO_DDK_ID", "another-ddk")
    elif fault == "wrong_omg":
        resources[3].write_bytes(b"different compiler")
    output = run_root / "compiled"
    args = op_probe.parser().parse_args([
        "--suite", "gdn-accuracy", "--prepared", "p", "--reference", "r", "--omc", "o",
        "--compile-only", "--output", str(output)])
    if fault:
        with pytest.raises(ValueError, match="differs from original OMC"):
            op_probe.run(args)
        assert not commands and not output.exists()
    else:
        assert op_probe.run(args) == 0
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["status"] == "COMPILE_COMPLETE" and len(commands) == 3
        assert manifest["tools"]["omg_sha256"] == provenance["omg_sha256"]

"""Host diagnostic contracts; these are not model/device accuracy gates."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from reference import gdn_precision as diagnostic
from reference.ge_torch import evaluate_torch
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_target import _Builder, _chunk_gdn
from test_reference_backend import broken_npu, child, run_env


@pytest.fixture(autouse=True, scope="module")
def torch_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _inputs(sequence=65):
    rng = np.random.default_rng(765)
    return {"q": rng.normal(0, .25, (1, sequence, 2, 3)).astype(np.float32),
            "k": rng.normal(0, .25, (1, sequence, 2, 3)).astype(np.float32),
            "v": rng.normal(0, .25, (1, sequence, 2, 4)).astype(np.float32),
            "g": -rng.uniform(.01, .5, (1, sequence, 2)).astype(np.float32),
            "beta": rng.uniform(.05, .95, (1, sequence, 2)).astype(np.float32)}


def _observed(values, observe):
    graph = Graph("observed")
    inputs = [graph.input(name, values[name].shape) for name in diagnostic.NAMES]
    states = [] if observe else None
    output = _chunk_gdn(_Builder(graph), *inputs, state_outputs=states)
    graph.outputs = [output, *(states or [])]
    return evaluate_torch(graph, {name: value.astype(np.float16) for name, value in values.items()})


def test_observer_preserves_output_and_exposes_final_and_cross_chunk_state():
    values = _inputs()
    default, = _observed(values, False)
    observed, first_state, final_state = _observed(values, True)
    np.testing.assert_array_equal(default.numpy().view(np.uint16), observed.numpy().view(np.uint16))
    assert first_state.shape == final_state.shape == (1, 2, 3, 4)
    _, prefix_state = _observed({name: value[:, :64] for name, value in values.items()}, True)
    np.testing.assert_array_equal(prefix_state.numpy().view(np.uint16), first_state.numpy().view(np.uint16))
    assert not torch.equal(first_state, final_state)


def test_error_metrics_and_zero_reference_are_explicit():
    result = diagnostic.metrics(np.array([3., 4.]), np.array([0., 0.]))
    assert result["relative_l2"] == 1 and result["max_abs"] == 4
    assert result["status"] == "FINITE" and result["compared_finite_elements"] == 2
    assert diagnostic.metrics([0], [0])["relative_l2"] == 0
    zero = diagnostic.metrics([0], [1])
    assert zero["relative_l2"] is None and zero["relative_l2_status"] == "ZERO_REFERENCE_NORM"
    nonfinite = diagnostic.metrics([0, np.inf, 1], [np.nan, 0, 2])
    assert nonfinite["status"] == "NONFINITE" and nonfinite["max_abs"] == 1
    assert nonfinite["reference_nonfinite"] == nonfinite["actual_nonfinite"] == 1
    assert nonfinite["compared_finite_elements"] == 1
    assert diagnostic.metrics([np.nan], [np.inf])["max_abs"] is None
    json.dumps([zero, nonfinite], allow_nan=False)


def test_reference_gets_the_same_half_rounded_values_as_candidate(monkeypatch):
    values = _inputs()
    calls = []
    reference = diagnostic._reference_chunk
    def checked(*inputs):
        calls.append(inputs[0].shape[1])
        for name, actual in zip(diagnostic.NAMES, inputs):
            assert actual.dtype == torch.float32
            expected = values[name][:, :actual.shape[1]].astype(np.float16).astype(np.float32)
            np.testing.assert_array_equal(actual.numpy(), expected)
        return reference(*inputs)
    monkeypatch.setattr(diagnostic, "_reference_chunk", checked)
    result = diagnostic.diagnose(values, source={"kind": "NPZ_INPUTS"})
    assert result["status"] == "DIAGNOSTIC_COMPLETE" and result["accuracy_gate"] == "NOT_EVALUATED"
    assert calls == [64, 65] and result["chunk_size"] == 64
    assert result["device_execution"] == "NOT_RUN"
    assert [(row["token_start"], row["token_end"]) for row in result["chunks"]] == [(0, 64), (64, 65)]
    assert all(row[kind]["status"] == "FINITE" for row in result["chunks"] for kind in ("output", "state"))
    assert result["input_rounding"]["q"]["rounding_error"]["max_abs"] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["keys", "rank", "batch", "sequence", "keydim", "dtype", "positive-g", "beta"])
def test_npz_rejects_invalid_contract(tmp_path, fault):
    values = _inputs(8)
    if fault == "keys":
        values["query"] = values.pop("q")
    elif fault == "rank":
        values["g"] = values["g"][..., None]
    elif fault == "batch":
        values = {name: np.repeat(value, 2, axis=0) for name, value in values.items()}
    elif fault == "sequence":
        values["beta"] = values["beta"][:, :3]
    elif fault == "keydim":
        values["k"] = values["k"][..., :2]
    elif fault == "dtype":
        values["q"] = values["q"].astype(np.int32)
    elif fault == "positive-g":
        values["g"][0, 0, 0] = .1
    else:
        values["beta"][0, 0, 0] = 1.1
    path = tmp_path / "inputs.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        diagnostic.load_inputs(path)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e30])
def test_nonfinite_or_half_overflow_input_is_saved_without_json_nan(tmp_path, monkeypatch, value):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    values = _inputs(8)
    values["q"][0, 0, 0, 0] = value
    path = tmp_path / "inputs.npz"
    np.savez(path, **values)
    output = tmp_path / "diagnostic.json"
    assert diagnostic.main(["--inputs", str(path), "--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["status"] == "DIAGNOSTIC_NONFINITE_INPUT" and result["chunks"] == []
    assert result["input_rounding"]["q"]["rounding_error"]["actual_nonfinite"] == 1
    assert "NaN" not in output.read_text() and "Infinity" not in output.read_text()


def test_finite_inputs_can_report_nonfinite_fp16_arithmetic():
    values = _inputs(64)
    values["g"].fill(-65504)
    result = diagnostic.diagnose(values, source={"kind": "SYNTHETIC"})
    assert result["status"] == "DIAGNOSTIC_NONFINITE_RESULT"
    assert all(item["rounding_error"]["status"] == "FINITE" for item in result["input_rounding"].values())
    assert result["chunks"][0]["state"]["actual_nonfinite"] > 0
    assert result["chunks"][0]["state"]["reference_nonfinite"] == 0
    json.dumps(result, allow_nan=False)


def test_default_synthetic_command_records_scope_and_preserves_existing_output(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    output = tmp_path / "synthetic.json"
    assert diagnostic.main(["--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["source"]["kind"] == "SYNTHETIC" and result["source"]["shape"] == {"S": 128, "H": 2, "K": 8, "V": 8}
    assert result["status"] == "DIAGNOSTIC_COMPLETE" and len(result["chunks"]) == 2
    old = output.read_bytes()
    assert diagnostic.main(["--output", str(output)]) == 1
    assert output.read_bytes() == old


@pytest.mark.parametrize("sequence,heads,key_dim,value_dim", [(513, 1, 1, 1), (1, 9, 1, 1),
                        (1, 1, 129, 1), (1, 1, 1, 129), (512, 3, 1, 1)])
def test_npz_size_limits_precede_array_loading(tmp_path, monkeypatch, sequence, heads, key_dim, value_dim):
    values = {name: np.zeros((1, sequence, heads, width), np.float16)
              for name, width in (("q", key_dim), ("k", key_dim), ("v", value_dim))}
    values.update(g=np.zeros((1, sequence, heads), np.float16), beta=np.zeros((1, sequence, heads), np.float16))
    path = tmp_path / "large.npz"
    np.savez(path, **values)
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized arrays must be rejected before np.load")
    monkeypatch.setattr(diagnostic.np, "load", forbidden)
    with pytest.raises(ValueError, match="size limit.*slice"):
        diagnostic.load_inputs(path)
    with pytest.raises(ValueError, match="size limit.*slice"):
        diagnostic.validate_inputs(values)


def test_expanded_npz_size_is_bounded_before_reading_headers(tmp_path):
    path = tmp_path / "compressed.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in diagnostic.NAMES:
            archive.writestr(name + ".npy", b"\x00" * (diagnostic.MAX_NPZ_BYTES + 1 if name == "q" else 1))
    assert path.stat().st_size < diagnostic.MAX_NPZ_BYTES
    with pytest.raises(ValueError, match="expanded NPZ exceeds"):
        diagnostic.load_inputs(path)


@pytest.mark.parametrize("entry", ["script", "module"])
def test_cpu_entrypoints_disable_broken_npu_before_torch_import(tmp_path, broken_npu, entry):
    output = tmp_path / (entry + ".json")
    repo = Path(diagnostic.__file__).resolve().parents[1]
    args = [str(repo / "targets/cpro/scripts/diagnose_gdn_precision.py")] if entry == "script" else ["-m", "reference.gdn_precision"]
    result = child(tmp_path, [*args, "--output", str(output)], run_env(tmp_path, broken_npu))
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "DIAGNOSTIC_COMPLETE" and report["reference_backend"] == "cpu"
    assert report["source"]["kind"] == "SYNTHETIC"

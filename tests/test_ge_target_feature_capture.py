"""Offline feature diagnostics with real tiny host outputs and synthetic board metadata.

The OMC bytes and HIAI metadata below are test doubles, not device evidence.
No compiler, transport or board execution is performed.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from qwen35_moe_dflash.ge_eval import evaluate
from qwen35_moe_dflash.ge_export import build_pair, prepare_pair
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.ge_target_features import compare_target_features


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def host_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("target-feature-host")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("AI_RUN_DIR", str(root))
        plan = prepare_pair(root / "prepared", 8, tiny=True)
    graphs = build_pair(plan["target_config"], plan["draft_config"], 8, gdn_indexed_copy="gather")
    weights = {key: np.fromfile(root / "prepared" / item["path"], dtype="<f2").reshape(item["shape"])
               for key, item in plan["weights"].items()}
    ids, mask = np.zeros((1, 8), np.int32), np.zeros((1, 8), np.int32)
    ids[0, :3], mask[0, :3] = [1, 2, 3], 1
    top1, features = evaluate(graphs[0], {"input_ids": ids, "attention_mask": mask}, weights)
    return {"prepared": root / "prepared", "plan": plan, "graphs": graphs,
            "top1": top1[0].tolist(), "features": features}


@pytest.fixture
def case(tmp_path, monkeypatch, host_fixture):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared, omc, board = (tmp_path / name for name in ("prepared", "omc", "board"))
    prepared.mkdir()
    # The comparator needs exact prepared payloads, not generated C++ recipes.
    shutil.copyfile(host_fixture["prepared"] / "prepared.json", prepared / "prepared.json")
    shutil.copytree(host_fixture["prepared"] / "weights", prepared / "weights")
    omc.mkdir(); board.mkdir()
    plan = copy.deepcopy(host_fixture["plan"])
    manifest = {"artifact_kind": "qwen35-cpro-omc-pair", "status": "COMPILED", "abi_version": 2,
                "source": copy.deepcopy(plan["source"]), "precision": "torch.float16", "quantization": None,
                "feature_dtype": "torch.float16", "sequence_length": 8, "feature_width": 32,
                "vocab_size": 64, "draft_width": 3, "pad_token_id": 0, "eos_token_ids": [],
                "expert_dispatch": "static", "gdn_indexed_copy": "gather",
                "fp16_reductions": "matmul", "fp16_row_mul": "compact",
                "ddk_id": "test-ddk", "graphs": []}
    for graph in host_fixture["graphs"]:
        payload = omc / (graph.name + ".omc")
        payload.write_bytes(b"TEST DOUBLE, NOT OMC: " + graph.name.encode())
        manifest["graphs"].append({"name": graph.name, "omc": payload.name,
            "bytes": payload.stat().st_size, "sha256": sha256_file(payload),
            "inputs": [v.name for v in graph.inputs],
            "outputs": ["target_top1", "target_features"] if graph.name == "target" else ["draft_top1"],
            "input_shapes": [list(v.shape) for v in graph.inputs],
            "input_dtypes": ["torch." + v.dtype for v in graph.inputs],
            "output_shapes": [list(v.shape) for v in graph.outputs],
            "output_dtypes": ["torch." + v.dtype for v in graph.outputs]})
    _json(omc / "manifest.json", manifest)
    reference = {"status": "PASS", "source": copy.deepcopy(plan["source"]),
                 "sequence_length": 8, "precision": "torch.float16", "quantization": None,
                 "prepared_manifest_sha256": sha256_file(prepared / "prepared.json"),
                 "prompt_token_ids": [1, 2, 3], "max_new_tokens": 4,
                 "ordinary": {"token_ids": [45, 42, 60, 42], "stop_reason": "length"}}
    report = {"schema_version": 2, "abi_version": 2, "feature_dtype": "torch.float16",
              "status": "EXECUTION_COMPLETE", "mode": "smoke", "scope": "target-draft-execution",
              "evidence_kind": "device-hiai", "device_execution": "PASS", "accuracy": "NOT_EVALUATED",
              "performance": "NOT_EVALUATED", "runtime_version": "test-runtime", "firmware_id": "test-firmware",
              "ddk_id": "test-ddk", "target_sha256": manifest["graphs"][0]["sha256"],
              "draft_sha256": manifest["graphs"][1]["sha256"], "sequence_length": 8,
              "pad_token_id": 0, "eos_token_ids": [], "prompt_token_ids": [1, 2, 3],
              "execution": {"target_calls": 1, "draft_calls": 1,
                            "target_top1": list(host_fixture["top1"]), "draft_top1": [0, 1, 2]}}
    run = {"status": "EXECUTION_COMPLETE", "device_execution": "PASS", "capture_target_features": True,
           "source": copy.deepcopy(plan["source"]), "precision": "torch.float16", "quantization": None,
           "ddk_id": "test-ddk", "runtime_version": "test-runtime", "firmware_id": "test-firmware",
           "runner_sha256": "a" * 64}
    value = {"prepared": prepared, "omc": omc, "board": board, "reference_path": tmp_path / "reference.json",
             "output": tmp_path / "diagnostics", "plan": plan, "manifest": manifest,
             "reference": reference, "report": report, "run": run,
             "features": host_fixture["features"].copy(), "top1": list(host_fixture["top1"])}
    _capture(value, value["features"])
    _save(value)
    return value


def _capture(case, features):
    raw = np.asarray(features, dtype="<f2").tobytes()
    case["report"]["execution"]["target_features"] = {
        "dtype": "float16", "encoding": "hex-le", "shape": [1, 8, 32],
        "bytes": len(raw), "data": raw.hex()}


def _save(case):
    _json(case["omc"] / "manifest.json", case["manifest"])
    _json(case["reference_path"], case["reference"])
    _json(case["board"] / "benchmark.json", case["report"])
    case["run"].update(manifest_sha256=sha256_file(case["omc"] / "manifest.json"),
                       reference_sha256=sha256_file(case["reference_path"]),
                       benchmark_sha256=sha256_file(case["board"] / "benchmark.json"))
    _json(case["board"] / "result.json", case["run"])


def _compare(case):
    return compare_target_features(case["prepared"], case["omc"], case["board"],
                                   case["reference_path"], case["output"])


def test_same_weight_capture_preserves_original_feature_abi_and_is_only_diagnostic(case):
    result = _compare(case)
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert result["source_valid_top1"] == result["board_valid_top1"] == [41, 41, 45]
    assert result["first_valid_top1_mismatch"] is None
    assert result["first_generated_token"] == {"prompt_index": 2, "reference": 45,
        "source": 45, "board": 45, "source_matches_reference": True}
    assert [row["layer"] for row in result["layers"]] == [0, 2]
    assert [row["feature_columns"] for row in result["layers"]] == [[0, 16], [16, 32]]
    for layer in result["layers"]:
        assert layer["valid_tokens"]["numerically_equal"] is True
        assert layer["valid_tokens"]["elements"] == 3 * 16
        assert layer["valid_tokens"]["max_abs"] == 0
        assert [row["prompt_index"] for row in layer["per_token"]] == [0, 1, 2]
    assert "no acceptance threshold" in result["scope"]
    assert json.loads((case["output"] / "report.json").read_text()) == result
    assert (case["output"] / "report.txt").is_file()
    # Diagnostic output never rewrites an existing golden/reference or board report.
    assert sha256_file(case["reference_path"]) == case["run"]["reference_sha256"]
    assert sha256_file(case["board"] / "benchmark.json") == case["run"]["benchmark_sha256"]


def test_localizes_row2_layer2_difference_and_reports_host_logit_of_board_token(case):
    features = case["features"].copy()
    features[0, 2, 20] += np.float16(0.5)
    _capture(case, features)
    case["report"]["execution"]["target_top1"][2] = 55
    _save(case)
    result = _compare(case)
    assert result["first_valid_top1_mismatch"] == 2
    assert result["first_generated_token"]["board"] == 55
    assert result["layers"][0]["valid_tokens"]["numerically_equal"] is True
    affected = result["layers"][1]
    assert affected["valid_tokens"]["max_abs_index"] == [2, 4]
    assert [item["numerically_equal"] for item in affected["per_token"]] == [True, True, False]
    expected_delta = abs(float(features[0, 2, 20]) - float(case["features"][0, 2, 20]))
    assert affected["valid_tokens"]["max_abs"] == expected_delta
    head = result["source_head"]
    assert [item["token"] for item in head["source_top5"]] == [45, 9, 20, 39, 54]
    assert head["source_logit_of_board_token"] == 0.17724609375
    assert head["source_top1_margin"] == 0.004638671875


def test_padded_feature_and_top1_differences_are_excluded(case):
    features = case["features"].copy()
    features[:, 3:, :] = np.nan
    _capture(case, features)
    case["report"]["execution"]["target_top1"][3:] = [1] * 5
    _save(case)
    result = _compare(case)
    assert result["first_valid_top1_mismatch"] is None
    assert all(layer["valid_tokens"]["numerically_equal"] for layer in result["layers"])
    assert all(layer["valid_tokens"]["board_nonfinite"] == 0 for layer in result["layers"])


def test_valid_nan_capture_is_reported_without_nonstandard_json_numbers(case):
    features = case["features"].copy()
    features[0, 0, 0] = np.nan
    _capture(case, features)
    _save(case)
    result = _compare(case)
    affected = result["layers"][0]["valid_tokens"]
    assert affected["board_nonfinite"] == 1 and affected["finite_pairs"] == 47
    assert affected["numerically_equal"] is False
    assert "max_abs" not in affected
    assert result["layers"][1]["valid_tokens"]["numerically_equal"] is True
    def invalid_constant(value):
        raise AssertionError("non-standard JSON number: " + value)
    saved = json.loads((case["output"] / "report.json").read_text(), parse_constant=invalid_constant)
    assert saved == result


@pytest.mark.parametrize("fault", [
    "weight_payload", "omc_payload", "benchmark_hash", "manifest_hash", "reference_hash",
    "prepared_source", "omc_source", "reference_source", "board_source", "prepared_reference_hash",
    "prompt", "capture_missing", "capture_shape", "capture_bytes", "capture_encoding", "capture_hex",
    "capture_not_requested", "board_model_hash", "runtime", "firmware", "omc_output_dtype", "run_precision", "run_quantization", "run_ddk",
])
def test_corrupt_or_mismatched_provenance_rejected_before_evaluation_and_output(case, monkeypatch, fault):
    if fault == "weight_payload":
        item = next(iter(case["plan"]["weights"].values()))
        path = case["prepared"] / item["path"]
        path.write_bytes(path.read_bytes() + b"\x00")
    elif fault == "omc_payload":
        path = case["omc"] / "target.omc"
        path.write_bytes(path.read_bytes() + b"\x00")
    elif fault in {"benchmark_hash", "manifest_hash", "reference_hash"}:
        name = {"benchmark_hash": "benchmark_sha256", "manifest_hash": "manifest_sha256",
                "reference_hash": "reference_sha256"}[fault]
        case["run"][name] = "0" * 64
        _json(case["board"] / "result.json", case["run"])
    elif fault == "prepared_source":
        case["plan"]["source"]["fp16_payload_sha256"] = "0" * 64
        _json(case["prepared"] / "prepared.json", case["plan"])
    else:
        if fault.endswith("_source"):
            owner = {"omc_source": "manifest", "reference_source": "reference", "board_source": "run"}[fault]
            case[owner]["source"]["seed"] = 18
        elif fault == "prepared_reference_hash":
            case["reference"]["prepared_manifest_sha256"] = "0" * 64
        elif fault == "prompt":
            case["report"]["prompt_token_ids"] = [1, 2, 4]
        elif fault == "capture_missing":
            del case["report"]["execution"]["target_features"]
        elif fault == "capture_not_requested":
            case["run"]["capture_target_features"] = False
        elif fault == "board_model_hash":
            case["report"]["target_sha256"] = "0" * 64
        elif fault in {"runtime", "firmware"}:
            case["run"][fault + ("_version" if fault == "runtime" else "_id")] = "different"
        elif fault in {"run_precision", "run_quantization", "run_ddk"}:
            key, value = {"run_precision": ("precision", "torch.float32"),
                          "run_quantization": ("quantization", {"bits": 4}),
                          "run_ddk": ("ddk_id", "different")}[fault]
            case["run"][key] = value
        elif fault == "omc_output_dtype":
            case["manifest"]["graphs"][0]["output_dtypes"][1] = "torch.float32"
        else:
            capture = case["report"]["execution"]["target_features"]
            field, value = {"capture_shape": ("shape", [1, 4, 64]), "capture_bytes": ("bytes", 510),
                            "capture_encoding": ("encoding", "hex-be"), "capture_hex": ("data", "z" * 1024)}[fault]
            capture[field] = value
        _save(case)
    def unexpected_evaluate(*args, **kwargs):
        raise AssertionError("invalid provenance reached tensor evaluation")
    monkeypatch.setattr("qwen35_moe_dflash.ge_target_features.evaluate", unexpected_evaluate)
    with pytest.raises(ValueError):
        _compare(case)
    assert not case["output"].exists()


@pytest.mark.parametrize("tokens", [None, [], [True], [64]])
def test_reference_needs_real_generated_tokens(case, tokens):
    case["reference"]["ordinary"]["token_ids"] = tokens
    _save(case)
    with pytest.raises(ValueError, match="nonempty valid ordinary token"):
        _compare(case)
    assert not case["output"].exists()


def test_cli_routes_to_offline_comparator(case):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_model_air.py"
    env = dict(os.environ, AI_RUN_DIR=str(case["output"].parent), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(script), "compare-target-features",
        "--prepared", str(case["prepared"]), "--omc", str(case["omc"]),
        "--board-run", str(case["board"]), "--reference", str(case["reference_path"]),
        "--output", str(case["output"])], env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert '"status": "DIAGNOSTIC_COMPLETE"' in completed.stdout
    result = json.loads((case["output"] / "report.json").read_text())
    assert result["source_valid_top1"] == [41, 41, 45]
    assert result["accuracy"] == "NOT_EVALUATED"
    source = script.parents[3] / "model/qwen35_moe_dflash"
    assert result["host_code_sha256"]["ge_eval.py"] == sha256_file(source / "ge_eval.py")

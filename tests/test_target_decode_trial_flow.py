"""Join real trial preparation/validation to simulated target-tool transport."""
import hashlib
import json
from pathlib import Path
import runpy
import struct

import pytest

from qwen35_moe_dflash.target_decode_trial_inputs import (
    prepare_decode_trial_inputs,
    validate_decode_trial_inputs,
)
from test_target_decode_smoke_execution import case as decode_case, DecodeFakeHdc


def source_snapshot(root):
    result = {}
    for path in root.iterdir():
        info = path.stat()
        result[path.name] = {
            "identity": (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                         info.st_ctime_ns, info.st_nlink),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


@pytest.fixture
def trial_case(decode_case, tmp_path):
    case = decode_case
    (case["source"] / "conv_state_pos.bin").unlink()
    # This extra file is deliberately outside the required-input inventory.
    (case["source"] / "golden_logits.bin").write_bytes(b"unbound golden, never copied")
    before = source_snapshot(case["source"])
    prepared = tmp_path / "prepared-trial"
    manifest = prepare_decode_trial_inputs(case["source"], prepared,
        experimental_conv_current_query_count=True)
    assert validate_decode_trial_inputs(prepared) == manifest
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_target_decode_smoke.py"
    main = runpy.run_path(str(script))["main"]
    return dict(case, prepared=prepared, manifest=manifest, main=main,
        before=before, execution=tmp_path / "trial-execution")


def arguments(case, *, input_dir=None):
    return ["--job", str(case["job"]), "--runtime-probe", str(case["probe"]),
        "--input-dir", str(input_dir or case["manifest"]["input_dir"]),
        "--trial-input-receipt", str(case["prepared"] / "manifest.json"),
        "--output", str(case["execution"]), "--lock-wait-seconds", "0", "--apply"]


def test_real_trial_receipt_reaches_one_tool_call_with_assumption_and_sources_preserved(trial_case):
    case = trial_case
    assert case["main"](arguments(case)) == 0
    result = json.loads((case["execution"] / "result.json").read_text())
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["remote_exitcode"] == 0
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["state_continuity"] == "NOT_VALIDATED_SINGLE_TOOL_INVOCATION"
    assert result["prefill_state_feedback"] == "NOT_RUN"
    assert result["experimental_input_assumption"] == case["manifest"]["assumption"]
    assert result["experimental_input_assumption"]["contract_status"] == "NOT_A_VALIDATED_KERNEL_CONTRACT"
    assert result["trial_input_preservation"] == "UNCHANGED_ORIGINALS_DERIVED_INPUTS_AND_EXPLICIT_ASSUMPTION"
    assert result["trial_input_receipt"]["sha256"] == hashlib.sha256(
        (case["prepared"] / "manifest.json").read_bytes()).hexdigest()
    calls = DecodeFakeHdc.instances[0].calls
    assert sum(label == "execute-original-target" for _, label, _ in calls) == 1
    control = Path(case["manifest"]["input_dir"]) / "conv_state_pos.bin"
    assert len(control.read_bytes()) == 512
    assert hashlib.sha256(control.read_bytes()).hexdigest() == (
        "d839a3521723b8a55d09d8eed9848940b284828e4d09218202c3ee11046bc16d")
    assert not (case["source"] / "conv_state_pos.bin").exists()
    assert not (Path(case["manifest"]["input_dir"]) / "golden_logits.bin").exists()
    assert source_snapshot(case["source"]) == case["before"]
    assert validate_decode_trial_inputs(case["prepared"]) == case["manifest"]


@pytest.mark.parametrize("fault", ["assumption", "source"])
def test_changed_trial_assumption_or_original_input_blocks_before_hdc(trial_case, fault):
    case = trial_case
    if fault == "assumption":
        path = case["prepared"] / "manifest.json"
        changed = json.loads(path.read_text())
        changed["assumption"]["contract_status"] = "VALIDATED"
        path.write_text(json.dumps(changed))
    else:
        path = case["source"] / "effective_length.bin"
        path.write_bytes(b"\x02\x00")
    assert case["main"](arguments(case)) == 1
    result = json.loads((case["execution"] / "result.json").read_text())
    assert result["status"] == "BLOCKED", result
    assert result["device_execution"] == "NOT_RUN"
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert not DecodeFakeHdc.instances and not DecodeFakeHdc.sends


def test_receipt_cannot_authorize_a_different_input_directory(trial_case):
    case = trial_case
    assert case["main"](arguments(case, input_dir=case["source"])) == 1
    result = json.loads((case["execution"] / "result.json").read_text())
    assert result["status"] == "BLOCKED", result
    assert "input directory differs" in result["error"]
    assert result["device_execution"] == "NOT_RUN"
    assert result["input_preflight"]["missing_files"] == ["conv_state_pos.bin"]
    assert not DecodeFakeHdc.instances and not DecodeFakeHdc.sends
    assert source_snapshot(case["source"]) == case["before"]


def test_explicit_single_token_recipe_records_all_four_controls_and_keeps_supplied_history(decode_case, tmp_path):
    case = decode_case
    source = case["source"]
    (source / "conv_state_pos.bin").unlink()
    original_controls = {
        "output_pos.bin": ("i", 255),
        "allQLen.bin": ("q", 1224),
        "effective_length.bin": ("h", 256),
        "position_ids.bin": ("i", 256),
        "new_kv_cache_pos.bin": ("i", 256),
    }
    for name, (encoding, value) in original_controls.items():
        (source / name).write_bytes(struct.pack("<" + encoding, value))
    mask = struct.pack("<2048f", *([0.] * 257 + [float("-inf")] * (2048 - 257)))
    (source / "attention_mask.bin").write_bytes(mask)
    before = source_snapshot(source)
    prepared = tmp_path / "single-token-trial"
    manifest = prepare_decode_trial_inputs(source, prepared, experimental_single_token_controls=True)
    assert validate_decode_trial_inputs(prepared) == manifest
    assert len(manifest["copied_inputs"]) == 9
    generated = {row["file"]: row for row in manifest["generated_controls"]}
    assert set(generated) == {"conv_state_pos.bin", "output_pos.bin", "effective_length.bin", "allQLen.bin"}
    expected = {"output_pos.bin": ("i", 0), "allQLen.bin": ("q", 1), "effective_length.bin": ("h", 1)}
    inputs = Path(manifest["input_dir"])
    for name, (encoding, value) in expected.items():
        assert (inputs / name).read_bytes() == struct.pack("<" + encoding, value)
    conv = (inputs / "conv_state_pos.bin").read_bytes()
    assert struct.unpack("<128i", conv) == (1,) + (0,) * 127
    assert hashlib.sha256(conv).hexdigest() == "d839a3521723b8a55d09d8eed9848940b284828e4d09218202c3ee11046bc16d"
    for name in ("attention_mask.bin", "position_ids.bin", "new_kv_cache_pos.bin",
                 "past_conv_state.bin", "past_recurrent_state.bin", "past_k.bin", "past_v.bin"):
        assert (inputs / name).read_bytes() == (source / name).read_bytes()
    replacements = {row["file"]: row for row in manifest["assumption"]["control_replacements"]}
    assert set(replacements) == set(generated)
    assert replacements["output_pos.bin"]["old_values"] == [255]
    assert replacements["allQLen.bin"]["old_values"] == [1224]
    assert replacements["effective_length.bin"]["old_values"] == [256]
    assert replacements["conv_state_pos.bin"]["old_values"] is None
    execution = tmp_path / "single-token-execution"
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_target_decode_smoke.py"
    case = dict(case, manifest=manifest, prepared=prepared, execution=execution)
    assert runpy.run_path(str(script))["main"](arguments(case)) == 0
    result = json.loads((execution / "result.json").read_text())
    assert result["status"] == "EXECUTED_UNVALIDATED", result
    assert result["experimental_input_assumption"] == manifest["assumption"]
    assert result["experimental_input_assumption"]["contract_status"] == "NOT_A_VALIDATED_KERNEL_CONTRACT"
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["state_continuity"] == "NOT_VALIDATED_SINGLE_TOOL_INVOCATION"
    assert result["prefill_state_feedback"] == "NOT_RUN"
    assert sum(label == "execute-original-target" for _, label, _ in DecodeFakeHdc.instances[0].calls) == 1
    proof = json.loads((execution / "board/runtime-decode-proof.json").read_text())
    controls = {Path(row["path"]).name: row for row in proof["control_observations"]}
    assert controls["output_pos.bin"]["values"] == [0]
    assert controls["allQLen.bin"]["values"] == controls["effective_length.bin"]["values"] == [1]
    assert controls["conv_state_pos.bin"]["values"] == [1] + [0] * 127
    assert source_snapshot(source) == before
    assert not (source / "conv_state_pos.bin").exists()

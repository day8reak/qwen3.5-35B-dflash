"""Opt-in bounded target feature transport; protocol fixtures are not board evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash.cli import execute as cli_execute, parser as cli_parser
from qwen35_moe_dflash.hdc_run import _smoke_checks
from test_cli import bundle
from test_hdc_run import case, _calls, _run


def _board_args(root, tmp_path, mode="smoke", capture=True):
    argv = ["board-command", "--mode", mode, "--omc", str(root),
            "--prompt", str(tmp_path / "prompt.json"), "--board-model-dir", "/data/models",
            "--board-output", "/data/report.json", "--runtime-version", "test-runtime",
            "--firmware-id", "test-firmware", "--max-new-tokens", "2",
            "--output", str(tmp_path / "command.json")]
    if capture:
        argv.append("--capture-target-features")
    return cli_parser().parse_args(argv)


@pytest.mark.parametrize("capture", [False, True])
def test_board_command_capture_flag_is_opt_in_and_smoke_only(bundle, tmp_path, capture):
    root, _ = bundle
    cli_execute(_board_args(root, tmp_path, capture=capture))
    command = json.loads((tmp_path / "command.json").read_text())["argv"]
    if capture:
        index = command.index("--capture-target-features")
        assert command[index + 1] == "1"
    else:
        assert "--capture-target-features" not in command


@pytest.mark.parametrize("mode", ["generate", "benchmark"])
def test_capture_rejects_other_board_modes_before_creating_output(bundle, tmp_path, mode):
    root, _ = bundle
    with pytest.raises(ValueError, match="requires --mode smoke"):
        cli_execute(_board_args(root, tmp_path, mode))
    assert not (tmp_path / "command.json").exists()


@pytest.mark.parametrize("width,valid", [(2048, True), (2049, False)])
def test_board_command_enforces_logical_capture_byte_limit(bundle, tmp_path, width, valid):
    root, manifest = bundle
    manifest["feature_width"] = width  # S=16, FP16: limit is width2048.
    (root / "manifest.json").write_text(json.dumps(manifest))
    if valid:
        cli_execute(_board_args(root, tmp_path))
    else:
        with pytest.raises(ValueError, match="65536-byte"):
            cli_execute(_board_args(root, tmp_path))
        assert not (tmp_path / "command.json").exists()


def _report(case):
    manifest = case["manifest"]
    graphs = {g["name"]: g for g in manifest["graphs"]}
    reference = json.loads(case["reference"].read_text())
    size = 2 * manifest["sequence_length"] * manifest["feature_width"]
    # +1, -0, +Inf and NaN retain their raw little-endian FP16 bits. This gate
    # validates transport, so nonfinite numerical failures must remain visible.
    data = ("003c0080007c007e" * ((size + 7) // 8))[:2 * size]
    payload = {"dtype": "float16", "encoding": "hex-le",
               "shape": [1, manifest["sequence_length"], manifest["feature_width"]],
               "bytes": size, "data": data}
    report = {"schema_version": 2, "abi_version": 2, "feature_dtype": "torch.float16",
              "status": "EXECUTION_COMPLETE", "mode": "smoke", "scope": "target-draft-execution",
              "evidence_kind": "device-hiai", "device_execution": "PASS",
              "accuracy": "NOT_EVALUATED", "performance": "NOT_EVALUATED",
              "runtime_version": "runtime", "ddk_id": manifest["ddk_id"], "firmware_id": "firmware",
              "target_sha256": graphs["target"]["sha256"], "draft_sha256": graphs["draft"]["sha256"],
              "sequence_length": manifest["sequence_length"], "pad_token_id": manifest["pad_token_id"],
              "prompt_token_ids": reference["prompt_token_ids"], "eos_token_ids": manifest["eos_token_ids"],
              "execution": {"target_calls": 1, "draft_calls": 1,
                            "target_top1": [12] * manifest["sequence_length"],
                            "draft_top1": [13] * manifest["draft_width"], "target_features": payload}}
    return report, manifest, reference


def test_feature_capture_checks_accept_exact_bytes_without_numerical_interpretation(case):
    report, manifest, reference = _report(case)
    before = deepcopy(report)
    checks = _smoke_checks(report, manifest, reference, "runtime", "firmware", capture_target_features=True)
    assert all(checks.values()) and all(type(value) is bool for value in checks.values())
    assert report == before
    payload = report["execution"]["target_features"]
    assert bytes.fromhex(payload["data"])[:8] == b"\x00\x3c\x00\x80\x00\x7c\x00\x7e"


@pytest.mark.parametrize("field,value", [
    ("missing", None), ("payload", []), ("dtype", "float32"), ("encoding", "hex-be"),
    ("shape", [1, 8, 31]), ("shape", [True, 8, 32]), ("shape", "1,8,32"),
    ("bytes", 511), ("bytes", True), ("bytes", "512"), ("data", ""),
    ("data", "0" * 1023), ("data", "0" * 1026), ("data", "G" * 1024),
    ("data", "A" * 1024), ("data", " " * 1024), ("data", [0] * 1024),
])
def test_requested_capture_rejects_missing_malformed_or_mismatched_payload(case, field, value):
    report, manifest, reference = _report(case)
    if field == "missing":
        del report["execution"]["target_features"]
    elif field == "payload":
        report["execution"]["target_features"] = value
    else:
        report["execution"]["target_features"][field] = value
    checks = _smoke_checks(report, manifest, reference, "runtime", "firmware", capture_target_features=True)
    assert all(type(check) is bool for check in checks.values())
    assert not all(checks.values())


@pytest.mark.parametrize("width,valid", [(4096, True), (4097, False)])
def test_feature_report_capture_limit_uses_manifest_shape(case, width, valid):
    case["manifest"]["feature_width"] = width
    report, manifest, reference = _report(case)
    assert all(_smoke_checks(report, manifest, reference, "runtime", "firmware",
                             capture_target_features=True).values()) is valid


def test_default_smoke_does_not_require_new_field(case):
    report, manifest, reference = _report(case)
    del report["execution"]["target_features"]
    checks = _smoke_checks(report, manifest, reference, "runtime", "firmware")
    assert all(checks.values())
    assert not any(name.startswith("target_features") for name in checks)


def test_hdc_rejects_capture_without_execution_only_before_transport(case):
    case["argv"].append("--capture-target-features")
    result = _run(case)
    assert result.returncode != 0 and "requires --execution-only" in result.stderr
    assert not _calls(case) and not _calls(case, runner=True)


def test_hdc_rejects_oversized_capture_before_transport(case):
    manifest = case["manifest"]
    manifest["feature_width"] = 4097
    (case["bundle"] / "manifest.json").write_text(json.dumps(manifest))
    case["argv"] += ["--capture-target-features", "--execution-only"]
    result = _run(case)
    assert result.returncode != 0 and "65536-byte" in result.stderr
    assert not _calls(case) and not _calls(case, runner=True)


@pytest.mark.parametrize("include_payload", [False, True])
def test_hdc_capture_roundtrip_requires_payload_and_keeps_hash_binding(case, include_payload):
    case["argv"] += ["--execution-only", "--capture-target-features"]
    if include_payload:
        fake_runner = Path(case["env"]["FAKE_RUNNER"])
        source = fake_runner.read_text()
        anchor = '    if failure == "smoke-skipped-draft":'
        injected = '''    if flags.get("--capture-target-features") == "1":
        source["execution"]["target_features"] = {
            "dtype": "float16", "encoding": "hex-le", "shape": [1, 8, 32],
            "bytes": 512, "data": "003c" * 256}
'''
        assert anchor in source
        fake_runner.write_text(source.replace(anchor, injected + anchor))
    result = _run(case)
    record = json.loads((case["output"] / "result.json").read_text())
    command = json.loads((case["output"] / "board-command.json").read_text())["argv"]
    assert command[command.index("--capture-target-features") + 1] == "1"
    assert record["capture_target_features"] is True
    assert record["execution_checks"]["target_features_present"] is include_payload
    if include_payload:
        assert result.returncode == 0, result.stdout + result.stderr
        assert record["status"] == "EXECUTION_COMPLETE"
        path = case["output"] / "benchmark.json"
        assert record["benchmark_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        payload = json.loads(path.read_text())["execution"]["target_features"]
        assert payload["data"] == "003c" * 256
        assert not case["board"].exists()
    else:
        assert result.returncode != 0 and "target_features_present" in result.stderr
        assert record["status"] == "FAILED" and record["stage"] == "execution-report"
        assert case["board"].is_dir()

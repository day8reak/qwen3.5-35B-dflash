"""Host diagnostics/protocol tests; synthetic output never certifies a device kernel."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.ge_fused.gdn_diagnostics import (
    FILLS, collect_sources, compare_fills, validate_io,
)
from qwen35_moe_dflash.ge_fused.probes import _board_script
from qwen35_moe_dflash.hashing import sha256_file
from test_fused_board_run import case, FAKE_RUNNER, _executable, _calls


def test_prefill_patterns_zeros_and_variable_outputs_are_separate():
    outputs = [{"name": "core_attn", "dtype": "float16", "shape": [8]}]
    untouched = {fill: [bytes([fill]) * 16] for fill in FILLS}
    row = compare_fills(untouched, outputs)[0]
    assert row["signal"] == "PREFILL_PATTERN_OBSERVED" and row["bytes_following_prefill"] == 16
    zeros = {fill: [bytes(16)] for fill in FILLS}
    row = compare_fills(zeros, outputs)[0]
    assert row["signal"] == "ALL_ZERO_IN_ALL_RUNS" and row["bytes_following_prefill"] == 0
    variable = {fill: [np.full((8,), i + 1, dtype="<f2").tobytes()] for i, fill in enumerate(FILLS)}
    assert compare_fills(variable, outputs)[0]["signal"] == "OUTPUT_VARIES_BETWEEN_RUNS"
    partial = {fill: [bytes(8) + bytes([fill]) * 8] for fill in FILLS}
    assert compare_fills(partial, outputs)[0]["bytes_following_prefill"] == 8


def test_fp32_state_statistics_follow_declared_dtype():
    outputs = [{"name": "last_recurrent_state", "dtype": "float32", "shape": [2]}]
    data = {fill: [np.array([0, 1], dtype="<f4").tobytes()] for fill in FILLS}
    row = compare_fills(data, outputs)[0]
    assert row["nonzero_elements"] == [1, 1, 1] and row["nonfinite_elements"] == [0, 0, 0]
    assert row["signal"] == "IDENTICAL_OUTPUTS"
    with pytest.raises(ValueError, match="bytes"):
        compare_fills(data, [dict(outputs[0], shape=[3])])


def test_unwritten_padding_does_not_flag_valid_prefix_as_unwritten():
    outputs = [{"name": "core_attn", "dtype": "float16", "shape": [1, 8, 2, 2], "valid_prefix_length": 6}]
    payloads = {fill: [np.ones((1, 6, 2, 2), dtype="<f2").tobytes() + bytes([fill]) * 16] for fill in FILLS}
    row = compare_fills(payloads, outputs)[0]
    assert row["signal"] == "IDENTICAL_OUTPUTS" and row["bytes_following_prefill"] == 0
    assert row["padding_bytes_unchecked"] == 16 and row["raw_identical_across_runs"] is False
    assert row["nonzero_elements"] == [24, 24, 24]


@pytest.mark.parametrize("fill,cases", [(True, ["gdn_decode"]), (256, ["gdn_decode"]),
                                       (-1, ["gdn_decode"]), (165, ["cache_prefill"]), (90, [])])
def test_output_fill_is_explicit_bounded_gdn_only(fill, cases):
    with pytest.raises(ValueError, match="output fill"):
        _board_script(cases, output_fill_byte=fill)


def test_source_capture_is_bounded_read_only_and_does_not_import(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ddk = tmp_path / "ddk"
    source = ddk / "ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise AssertionError('must never import DDK code')\n# initial_state\n")
    digest = sha256_file(source)
    unrelated = source.with_name("unrelated.py")
    unrelated.write_text("unrelated")
    out = tmp_path / "capture"
    report = collect_sources(ddk, out)
    assert report["status"] == "COLLECTED" and len(report["files"]) == 1
    assert report["files"][0]["sha256"] == digest == sha256_file(source)
    assert (out / report["files"][0]["path"]).read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        collect_sources(ddk, ddk / "forbidden")


def _native(tmp_path):
    output = tmp_path / "output.bin"
    output.write_bytes(np.arange(4, dtype="<f2").tobytes())
    desc = {"index": 0, "name": "core_attn", "dtype": "float16", "shape": [4],
            "path": output.name, "bytes": 8, "sha256": sha256_file(output)}
    step = {"inputs": [{"bytes": 8, "sha256": "1" * 64}], "outputs": [desc]}
    native = {"outputs": [desc], "io_diagnostic": {"enabled": True, "output_fill_byte": 165,
        "process_called": True, "input_readback_point": "AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS",
        "inputs": [{"index": 0, "logical_bytes": 8, "allocated_bytes": 32, "readback_sha256": "1" * 64, "readback_matches_source": True}],
        "outputs": [{"index": 0, "logical_bytes": 8, "allocated_bytes_before": 32, "allocated_bytes_after": 32,
                     "checked_after_process": True, "fill_count_available": True,
                     "object_same": True, "buffer_same": True, "fill_byte_count": 0}]}}
    return native, step


def test_io_validation_checks_real_buffer_hashes_and_retained_fill_counts(tmp_path):
    native, step = _native(tmp_path)
    assert len(validate_io(native, step, tmp_path, 165)[0]) == 8
    changed = deepcopy(native)
    changed["io_diagnostic"]["inputs"][0]["readback_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="readback"):
        validate_io(changed, step, tmp_path, 165)
    changed = deepcopy(native)
    changed["io_diagnostic"]["outputs"][0]["object_same"] = False
    with pytest.raises(ValueError, match="identity"):
        validate_io(changed, step, tmp_path, 165)
    changed = deepcopy(native)
    changed["io_diagnostic"]["outputs"][0]["fill_byte_count"] = 8
    with pytest.raises(ValueError, match="retained-fill"):
        validate_io(changed, step, tmp_path, 165)
    with pytest.raises(ValueError, match="requested"):
        validate_io(native, step, tmp_path, 90)
    (tmp_path / "output.bin").write_bytes(bytes(8))
    with pytest.raises(ValueError, match="SHA256"):
        validate_io(native, step, tmp_path, 165)


DIAGNOSTIC_FAKE = r'''
if '--output-fill-byte' in flags:
    fill = int(flags['--output-fill-byte'])
    diagnostic = {'enabled': True, 'output_fill_byte': fill, 'inputs': [], 'outputs': [], 'process_called': True,
                  'input_readback_point': 'AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS'}
    for item in step['inputs']:
        data = (plan.parent / item['path']).read_bytes()
        diagnostic['inputs'].append({'index': item['index'], 'logical_bytes': len(data),
            'allocated_bytes': len(data), 'readback_sha256': hashlib.sha256(data).hexdigest(), 'readback_matches_source': True})
    for item in base['outputs']:
        data = bytes([fill]) * item['bytes']
        (output / item['path']).write_bytes(data)
        item['sha256'] = hashlib.sha256(data).hexdigest()
        diagnostic['outputs'].append({'index': item['index'], 'logical_bytes': len(data),
            'allocated_bytes_before': len(data), 'allocated_bytes_after': len(data),
            'object_same': True, 'buffer_same': True, 'fill_byte_count': len(data),
            'checked_after_process': True, 'fill_count_available': True})
    if os.environ.get('FUSED_TEST_FAILURE') == 'bad-readback':
        diagnostic['inputs'][0]['readback_sha256'] = '0' * 64
    base['io_diagnostic'] = diagnostic
'''


@pytest.mark.parametrize("failure,expected_runs", [("", 3), ("bad-readback", 1), ("execute", 1)])
def test_three_fill_orchestration_reuses_models_runner_and_checks_evidence(case, monkeypatch, failure, expected_runs):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_gdn_diagnostics.py"
    spec = importlib.util.spec_from_file_location("test_gdn_entry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "board", case["module"])
    for key in ("CPRO_DDK_ROOT", "DDK_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FUSED_TEST_FAILURE", failure)
    _executable(case["root"] / "fake-runner", FAKE_RUNNER.replace("base['status'] = 'EXECUTED'", DIAGNOSTIC_FAKE + "\nbase['status'] = 'EXECUTED'"))
    code = module.main(["--probe-run", str(case["probe"]), "--output", str(case["output"]), "--timeout-seconds", "20"])
    report = json.loads((case["output"] / "report.json").read_text())
    assert code == (1 if failure else 0), report
    assert len(report["runs"]) == expected_runs and len(case["built"]) == 1
    calls = [dict(zip(row[::2], row[1::2])) for row in _calls(case, runner=True) if row != ["--probe"]]
    assert len(calls) == expected_runs * 2
    assert [int(row["--output-fill-byte"]) for row in calls] == [fill for fill in FILLS[:expected_runs] for _ in range(2)]
    assert {Path(row["--fused-op-step"]).parent.name for row in calls} == {"gdn_prefill", "gdn_decode"}
    assert case["original"] == {str(path.relative_to(case["probe"])): sha256_file(path)
                                for path in case["probe"].rglob("*") if path.is_file()}
    if not failure:
        assert report["status"] == "DIAGNOSIS_COMPLETE" and len(report["cases"]) == 2
        assert all(item["signal"] == "PREFILL_PATTERN_OBSERVED" for row in report["cases"] for item in row["outputs"])
        assert report["numerical_acceptance"] == "NOT_DEFINED" and report["placement"] == "UNVERIFIED"
    else:
        assert report["status"] == "FAILED" and report["cases"] == []

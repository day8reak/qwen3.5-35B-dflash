import json

import numpy as np
import pytest

from qwen35_moe_dflash.op_probe_report import compare_outputs, write_report


def _case(tmp_path, reference, actual):
    source = tmp_path / "case"
    board = tmp_path / "board" / "gdn_layer0_stages"
    source.mkdir()
    board.mkdir(parents=True)
    reference = np.asarray(reference, dtype="<f2")
    np.asarray(actual, dtype="<f2").tofile(board / "output_0.bin")
    reference.tofile(source / "expected_0.bin")
    return {"name": "gdn_layer0_stages", "probe_suite": "gdn-accuracy", "probe_family": "gdn",
            "directory": str(source), "outputs": [{"name": "v100", "label": "chunk_core",
                "path": "output_0.bin", "expected": "expected_0.bin", "dtype": "float16",
                "shape": list(reference.shape), "rtol": .01, "atol": .002}]}, board


def test_tiny_zero_output_remains_visible_even_inside_existing_tolerance(tmp_path):
    case, board = _case(tmp_path, [.001, -.001, 0], [0, 0, 0])
    result = compare_outputs(case, board)
    assert result["status"] == "PASSED"  # Preserve the old acceptance formula.
    output, = result["outputs"]
    assert output["label"] == "chunk_core"
    signal = output["signal"]
    assert signal["actual"]["nonzero"] == 0
    assert signal["reference"]["nonzero"] == 2
    assert signal["relative_l2_to_reference"] == pytest.approx(1)
    assert signal["error_rms"] > 0
    assert signal["exact_elements"] == 1
    (board / "stdout").write_text(json.dumps({"status": "EXECUTED"}))
    (board / "exitcode").write_text("0\n")
    write_report(tmp_path, {"cases": [case]})
    report = (tmp_path / "report.txt").read_text()
    assert "output 0 (chunk_core)" in report
    assert '"nonzero": 0' in report and '"relative_l2_to_reference": 1.0' in report


def test_zero_reference_has_explicitly_undefined_relative_norm(tmp_path):
    case, board = _case(tmp_path, [0, 0], [0, .25])
    output, = compare_outputs(case, board)["outputs"]
    assert output["status"] == "FAILED"
    assert output["signal"]["relative_l2_to_reference"] is None
    assert output["signal"]["relative_l2_defined"] is False
    assert output["signal"]["actual"]["l2"] == .25
    json.dumps(output, allow_nan=False)


def test_nonfinite_output_still_fails_without_misleading_finite_metrics(tmp_path):
    case, board = _case(tmp_path, [1, 2], [float("nan"), 2])
    output, = compare_outputs(case, board)["outputs"]
    assert output["status"] == "FAILED" and output["reason"] == "NONFINITE"
    assert "signal" not in output
    json.dumps(output, allow_nan=False)

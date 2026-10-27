import json
import tarfile

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


@pytest.mark.parametrize("suite", ["gdn-accuracy", "gdn-core-accuracy", "gdn-comprehensive"])
def test_tiny_zero_output_remains_visible_even_inside_existing_tolerance(tmp_path, suite):
    case, board = _case(tmp_path, [.001, -.001, 0], [0, 0, 0])
    case["probe_suite"] = suite
    result = compare_outputs(case, board)
    assert result["status"] == "PASSED"  # Preserve the old acceptance formula.
    output, = result["outputs"]
    assert output["label"] == "chunk_core"
    signal = output["signal"]
    assert signal["actual"]["nonzero"] == 0
    assert signal["reference"]["nonzero"] == 2
    assert signal["all_zero_with_nonzero_reference"] is True
    assert signal["relative_l2_to_reference"] == pytest.approx(1)
    assert signal["error_rms"] > 0
    assert signal["exact_elements"] == 1
    (board / "stdout").write_text(json.dumps({"status": "EXECUTED"}))
    (board / "exitcode").write_text("0\n")
    write_report(tmp_path, {"cases": [case]})
    report = (tmp_path / "report.txt").read_text()
    assert "output 0 (chunk_core)" in report
    assert '"nonzero": 0' in report and '"relative_l2_to_reference": 1.0' in report
    assert '"all_zero_with_nonzero_reference": true' in report


def test_zero_reference_has_explicitly_undefined_relative_norm(tmp_path):
    case, board = _case(tmp_path, [0, 0], [0, .25])
    output, = compare_outputs(case, board)["outputs"]
    assert output["status"] == "FAILED"
    assert output["signal"]["relative_l2_to_reference"] is None
    assert output["signal"]["relative_l2_defined"] is False
    assert output["signal"]["all_zero_with_nonzero_reference"] is False
    assert output["signal"]["actual"]["l2"] == .25
    json.dumps(output, allow_nan=False)


def test_nonfinite_output_still_fails_without_misleading_finite_metrics(tmp_path):
    case, board = _case(tmp_path, [1, 2], [float("nan"), 2])
    output, = compare_outputs(case, board)["outputs"]
    assert output["status"] == "FAILED" and output["reason"] == "NONFINITE"
    assert "signal" not in output
    json.dumps(output, allow_nan=False)


def test_comprehensive_report_locates_head_loss_and_exposes_allocation_without_repacking(tmp_path):
    reference = np.ones((1, 2, 64, 8), dtype=np.float16)
    actual = reference.copy()
    actual[:, 1] = 0
    case, board = _case(tmp_path, reference, actual)
    case["probe_suite"] = "gdn-comprehensive"
    io = [{"direction": "output", "index": 0, "logical_shape": [1, 2, 64, 8],
           "logical_dtype": "float16", "logical_bytes": 2048,
           "runtime_shape": [1, 2, 64, 8], "runtime_format": 2,
           "runtime_dtype": "unavailable", "allocation_bytes": 4096}]
    (board / "stdout").write_text(json.dumps({"status": "EXECUTED", "io_tensors": io}))
    (board / "exitcode").write_text("0\n")
    write_report(tmp_path, {"cases": [case]})
    report = json.loads((tmp_path / "report.json").read_text())
    row, = report["cases"]
    assert row["runtime"]["native"]["io_tensors"] == io
    output, = row["numerical"]["outputs"]
    assert output["status"] == "FAILED"
    counts = output["signal"]["nonzero_by_axis"]
    assert counts["reference"][1] == {"axis": 1, "counts": [512, 512]}
    assert counts["actual"][1] == {"axis": 1, "counts": [512, 0]}
    assert counts["actual"][2] == {"axis": 2, "counts": [8] * 64}
    text = (tmp_path / "report.txt").read_text()
    assert '"allocation_bytes": 4096' in text
    assert '"counts": [512, 0]' in text
    assert "does not by itself establish row padding" in text
    summary = (tmp_path / "gdn-summary.txt").read_text()
    assert "axis1 nonzero actual[0=512,1=0] ref[0..1=512]" in summary
    assert "bytes=2048/4096" in summary
    assert "nonzero(actual/ref)=512/1024" in summary
    with tarfile.open(tmp_path / "results.tar.gz") as archive:
        assert "gdn-summary.txt" in archive.getnames()


def test_comprehensive_axis_counts_are_bounded_and_only_added_after_finite_checks(tmp_path):
    case, board = _case(tmp_path, np.ones((65, 2)), np.ones((65, 2)))
    case["probe_suite"] = "gdn-comprehensive"
    output, = compare_outputs(case, board)["outputs"]
    assert output["signal"]["nonzero_by_axis"]["actual"] == [{"axis": 1, "counts": [65, 65]}]
    bad = np.ones((65, 2), dtype="<f2")
    bad[0, 0] = np.nan
    bad.tofile(board / "output_0.bin")
    output, = compare_outputs(case, board)["outputs"]
    assert output["reason"] == "NONFINITE" and "signal" not in output


def test_existing_comprehensive_summary_is_not_overwritten_or_partially_rebuilt(tmp_path):
    case, _ = _case(tmp_path, [1], [1])
    case["probe_suite"] = "gdn-comprehensive"
    summary = tmp_path / "gdn-summary.txt"
    summary.write_text("existing evidence")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report(tmp_path, {"cases": [case]})
    assert summary.read_text() == "existing evidence"
    assert not (tmp_path / "report.json").exists()

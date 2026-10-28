"""Console diagnosis uses retained evidence without repeating the draft run."""
import json

import pytest

from qwen35_moe_dflash.artifacts import save_json
from test_full_draft_cli import job


@pytest.mark.parametrize("zero_reference", [False, True])
def test_executed_numerical_failure_shows_retained_comparison_without_retry(
        job, tmp_path, monkeypatch, capsys, zero_reference):
    from qwen35_moe_dflash import draft_board

    main, fresh, _, calls, _ = job
    summary = {"elements": 30, "finite_count": 30, "nonfinite_count": 0,
               "nonzero_count": 30, "finite_min": -1.0, "finite_max": 2.0,
               "l2_norm": 5.0, "l2_norm_status": "FINITE"}
    comparison = {
        "status": "FAILED", "device_execution": "PASSED", "numerical_validation": "FAILED",
        "host_input_readback": "MATCHED", "token_mismatches": 2,
        "actual_top1": [[8, 3, 4]], "expected_top1": [[8, 5, 6]],
        "hidden_diagnostic": {
            "status": "MISMATCH_ZERO_REFERENCE" if zero_reference else "FINITE_DIAGNOSTIC",
            "max_abs_error": 2.0, "error_l2": 5.0, "relative_l2": None if zero_reference else 0.5,
            "relative_l2_status": "RELATIVE_L2_UNDEFINED_ZERO_REFERENCE" if zero_reference else "DEFINED",
            "exact_match": False, "acceptance_threshold": "NOT_DEFINED",
            "expected": {**summary, **({"nonzero_count": 0, "l2_norm": 0.0,
                                         "l2_norm_status": "ZERO"} if zero_reference else {})},
            "actual": summary,
        },
        "reference_quality": "DEGENERATE_ZERO_HIDDEN" if zero_reference else "FINITE_NONZERO_HIDDEN",
        "accuracy_scope": "Top1 and hidden diagnostics on this fixture only",
        "error": "draft Top1 differs from PyTorch reference on identical inputs",
    }
    if zero_reference:
        comparison["reference_warning"] = "All reference hidden values are zero"
    retained = {}

    def mismatch_board(bundle, folder, **kwargs):
        calls.append("board")
        folder.mkdir()
        save_json(folder / "comparison.json", comparison)
        retained["bytes"] = (folder / "comparison.json").read_bytes()
        return {"status": "FAILED", "stage": "complete", "device_execution": "PASSED",
                "numerical_validation": "FAILED", "native_status": "EXECUTED", "remote_exitcode": 0,
                "error": comparison["error"], "comparison": str(folder / "comparison.json")}

    monkeypatch.setattr(draft_board, "execute_draft_bundle", mismatch_board)
    root = tmp_path / "failed-comparison"
    assert main([*fresh, "--output", str(root)]) == 1
    output = capsys.readouterr().out
    assert calls == ["prepared", "fixture", "air", "compilation", "bundle", "board"]
    assert "Retained numerical comparison:" in output
    assert 'device_execution="PASSED"' in output and 'host_input_readback="MATCHED"' in output
    assert "token_mismatches=2" in output
    assert "actual_top1=[[8, 3, 4]]" in output and "expected_top1=[[8, 5, 6]]" in output
    assert "Hidden expected:" in output and "Hidden actual:" in output
    assert "max_abs_error=2.0" in output and "error_l2=5.0" in output
    assert 'acceptance_threshold="NOT_DEFINED"' in output
    assert comparison["reference_quality"] in output and comparison["accuracy_scope"] in output
    if zero_reference:
        assert "relative_l2=null" in output and "RELATIVE_L2_UNDEFINED_ZERO_REFERENCE" in output
        assert comparison["reference_warning"] in output
    else:
        assert "relative_l2=0.5" in output
    assert (root / "board/comparison.json").read_bytes() == retained["bytes"]
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "FAILED" and result["device_execution"] == "PASSED"
    assert result["numerical_validation"] == "FAILED" and result["remote_exitcode"] == 0


def test_weight_inventory_console_is_compact_but_retained_result_is_complete(job, tmp_path):
    main, _, _, calls, _ = job
    result = {"output": str(tmp_path), "status": "FAILED", "weight_storage_validation": {
        "requested": "embedded-stream", "observed_weight_candidates": [
            {"path": f"generated/model/SubGraph_{i}.weight", "bytes": 1024, "sha256": "a" * 64}
            for i in range(53)]}}
    save_json(tmp_path / "result.json", result)
    original = (tmp_path / "result.json").read_bytes()
    printed = main.__globals__["_console_result"](result)
    storage = printed["weight_storage_validation"]
    assert storage["observed_weight_candidate_count"] == 53
    assert storage["observed_weight_candidate_bytes"] == 53 * 1024
    assert storage["details"] == str(tmp_path / "result.json")
    assert "SubGraph_" not in json.dumps(printed) and len(json.dumps(printed)) < 1024
    assert len(result["weight_storage_validation"]["observed_weight_candidates"]) == 53
    assert (tmp_path / "result.json").read_bytes() == original and calls == []

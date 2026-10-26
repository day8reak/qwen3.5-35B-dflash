"""Offline comparison explains drift without weakening existing accuracy gates."""
from copy import deepcopy
import hashlib
import json

import pytest

from qwen35_moe_dflash.cli import execute, parser
from test_cli import bundle


def _reports(manifest):
    sample = {"generated_token_ids": [3, 4], "stop_reason": "length"}
    mode = {"stable_generated_token_ids": [3, 4], "stable_stop_reason": "length",
            "measurements": [deepcopy(sample) for _ in range(10)]}
    board = {"status": "PASS", "evidence_kind": "device-hiai",
             "prompt_token_ids": [1, 2], "max_new_tokens": 2, "sequence_length": 16,
             "ordinary": mode, "dflash": deepcopy(mode),
             "target_sha256": manifest["graphs"][0]["sha256"],
             "draft_sha256": manifest["graphs"][1]["sha256"],
             "abi_version": 2, "feature_dtype": "torch.float16"}
    reference = {"status": "PASS", "source": manifest["source"],
                 "prompt_token_ids": [1, 2], "max_new_tokens": 2, "sequence_length": 16,
                 "ordinary": {"token_ids": [3, 4], "stop_reason": "length"},
                 "precision": "torch.float16", "quantization": None}
    return board, reference


def _compare(root, tmp_path, board, reference, expected_failed=(), expected_error=None):
    board_path, reference_path = tmp_path / "board.json", tmp_path / "reference.json"
    board_path.write_text(json.dumps(board))
    reference_path.write_text(json.dumps(reference))
    before = [path.read_bytes() for path in (board_path, reference_path)]
    output = tmp_path / "comparison.json"
    args = parser().parse_args(["compare", "--omc", str(root), "--report", str(board_path),
                               "--reference", str(reference_path), "--output", str(output)])
    if expected_failed:
        with pytest.raises(ValueError, match="comparison failed") as failure:
            execute(args)
        assert "diagnostic details: " + str(output) in str(failure.value)
        assert "board_modes(tokens/stop)=" in str(failure.value)
        if expected_error is not None:
            assert expected_error in str(failure.value)
    else:
        execute(args)
    result = json.loads(output.read_text())
    assert result["status"] == ("FAIL" if expected_failed else "PASS")
    assert {name for name, passed in result["checks"].items() if not passed} == set(expected_failed)
    assert result["board_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert result["reference_sha256"] == hashlib.sha256(before[1]).hexdigest()
    assert [path.read_bytes() for path in (board_path, reference_path)] == before
    return result["diagnostics"]


def _set_mode(mode, tokens, stop="length"):
    mode["stable_generated_token_ids"] = list(tokens)
    mode["stable_stop_reason"] = stop
    for row in mode["measurements"]:
        row["generated_token_ids"] = list(tokens)
        row["stop_reason"] = stop


def test_matching_results_preserve_all_acceptance_checks(bundle, tmp_path):
    root, manifest = bundle
    board, reference = _reports(manifest)
    details = _compare(root, tmp_path, board, reference)
    assert details["ordinary_vs_dflash"]["tokens_equal"]
    assert details["ordinary_vs_dflash"]["stop_equal"]
    for mode in details["modes"].values():
        assert mode["has_ten_measurements"]
        assert mode["repeatable_tokens"] and mode["repeatable_stop"]
        assert mode["measurements_match_reported_stable"]
        assert mode["stable_vs_reference"]["first_token_mismatch"] is None
        assert mode["first_measurement_reference_mismatch"] is None
        assert mode["first_measurement_repeatability_mismatch"] is None


def test_stable_board_modes_can_agree_with_each_other_and_disagree_with_reference(bundle, tmp_path):
    root, manifest = bundle
    board, reference = _reports(manifest)
    for mode in ("ordinary", "dflash"):
        _set_mode(board[mode], [3, 5])
    details = _compare(root, tmp_path, board, reference,
                       {"tokens", "dflash_tokens", "ten_stable_runs"},
                       'ordinary: count=10, repeatable(tokens/stop)=True/True, '
                       'stable/reference_first_token_mismatch={"generated_index":1,"expected_token":4,"actual_token":5')
    assert details["ordinary_vs_dflash"]["tokens_equal"]
    for mode in details["modes"].values():
        assert mode["repeatable_tokens"] and mode["repeatable_stop"]
        assert mode["measurements_match_reported_stable"]
        mismatch = mode["stable_vs_reference"]["first_token_mismatch"]
        assert mismatch == {"generated_index": 1, "expected_token": 4, "actual_token": 5,
                            "expected_present": True, "actual_present": True}
        assert mode["first_measurement_reference_mismatch"]["measurement_index"] == 0
        assert mode["first_measurement_repeatability_mismatch"] is None


def test_actual_measurement_drift_is_distinct_from_stable_header_reference_match(bundle, tmp_path):
    root, manifest = bundle
    board, reference = _reports(manifest)
    board["ordinary"]["measurements"][7]["generated_token_ids"] = [3, 5]
    details = _compare(root, tmp_path, board, reference, {"ten_stable_runs"})
    mode = details["modes"]["ordinary"]
    assert mode["stable_vs_reference"]["tokens_equal"]
    assert not mode["repeatable_tokens"] and mode["repeatable_stop"]
    assert not mode["measurements_match_reported_stable"]
    assert mode["first_measurement_reference_mismatch"]["measurement_index"] == 7
    assert mode["first_measurement_repeatability_mismatch"]["measurement_index"] == 7


def test_cross_mode_difference_is_reported_without_changing_reference_gate(bundle, tmp_path):
    root, manifest = bundle
    board, reference = _reports(manifest)
    _set_mode(board["dflash"], [3, 5])
    details = _compare(root, tmp_path, board, reference, {"dflash_tokens", "ten_stable_runs"})
    assert not details["ordinary_vs_dflash"]["tokens_equal"]
    assert details["ordinary_vs_dflash"]["first_token_mismatch"]["generated_index"] == 1
    assert all(mode["repeatable_tokens"] for mode in details["modes"].values())


@pytest.mark.parametrize("tokens,index,expected_present,actual_present", [
    ([3], 1, True, False), ([3, 4, 5], 2, False, True), ([], 0, True, False),
])
def test_length_mismatch_includes_presence_flags(bundle, tmp_path, tokens, index,
                                                 expected_present, actual_present):
    root, manifest = bundle
    board, reference = _reports(manifest)
    _set_mode(board["ordinary"], tokens)
    details = _compare(root, tmp_path, board, reference, {"tokens", "ten_stable_runs"})
    diff = details["modes"]["ordinary"]["stable_vs_reference"]
    assert diff["expected_length"] == 2 and diff["actual_length"] == len(tokens)
    mismatch = diff["first_token_mismatch"]
    assert mismatch["generated_index"] == index
    assert mismatch["expected_present"] is expected_present
    assert mismatch["actual_present"] is actual_present


def test_stop_mismatch_does_not_invent_a_token_mismatch(bundle, tmp_path):
    root, manifest = bundle
    board, reference = _reports(manifest)
    _set_mode(board["dflash"], [3, 4], "eos")
    details = _compare(root, tmp_path, board, reference, {"dflash_stop", "ten_stable_runs"})
    diff = details["modes"]["dflash"]["stable_vs_reference"]
    assert diff["tokens_equal"] and diff["first_token_mismatch"] is None
    assert not diff["stop_equal"]
    assert (diff["expected_stop"], diff["actual_stop"]) == ("length", "eos")


@pytest.mark.parametrize("count", [0, 9])
def test_incomplete_measurements_do_not_pass_or_claim_empty_repeatability(bundle, tmp_path, count):
    root, manifest = bundle
    board, reference = _reports(manifest)
    board["ordinary"]["measurements"] = board["ordinary"]["measurements"][:count]
    details = _compare(root, tmp_path, board, reference, {"ten_stable_runs"})
    mode = details["modes"]["ordinary"]
    assert mode["measurement_count"] == count and not mode["has_ten_measurements"]
    assert mode["repeatable_tokens"] is bool(count)
    assert mode["repeatable_stop"] is bool(count)


def test_incomplete_malformed_measurements_remain_failed_and_have_missing_detail(bundle, tmp_path):
    root, manifest = bundle
    board, reference = _reports(manifest)
    board["ordinary"]["measurements"] = [{}] * 9
    details = _compare(root, tmp_path, board, reference, {"ten_stable_runs"})
    mode = details["modes"]["ordinary"]
    assert not mode["repeatable_tokens"] and not mode["repeatable_stop"]
    assert not mode["measurements_match_reported_stable"]
    mismatch = mode["first_measurement_reference_mismatch"]
    assert mismatch["measurement_index"] == 0
    assert mismatch["actual_length"] is None and mismatch["actual_stop"] is None


def test_reported_stable_header_mismatch_does_not_replace_existing_reference_rejection(bundle, tmp_path):
    root, manifest = bundle
    board, reference = _reports(manifest)
    board["ordinary"]["stable_generated_token_ids"] = [3, 5]
    details = _compare(root, tmp_path, board, reference, {"tokens"})
    mode = details["modes"]["ordinary"]
    assert mode["repeatable_tokens"] and not mode["measurements_match_reported_stable"]
    assert mode["first_measurement_reference_mismatch"] is None

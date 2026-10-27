"""Selection boundaries when reusing an existing multi-operator probe batch."""
import json
import subprocess

import pytest

from qwen35_moe_dflash.ge_fused.probes import _board_script, build_fused_probes, compare_fused_probes
from qwen35_moe_dflash.hashing import sha256_file
from test_ge_fused_probes import setup, _fake_results, _mock_tools


MODEL_CASES = ("gdn_prefill", "gdn_decode", "cache_prefill", "cache_decode",
               "attention_prefill", "attention_decode")


def test_selected_comparison_keeps_source_identity_and_excludes_unrun_cases(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    _mock_tools(monkeypatch, fail="rms")
    root = tmp_path / "original"
    manifest = build_fused_probes(root, binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk", cases="all")
    assert manifest["status"] == "PARTIAL"
    before = {path: path.read_bytes() for path in (root / "manifest.json", root / "bundle/run-board.sh")}
    board = tmp_path / "board"
    selected = [case for case in manifest["cases"] if case["name"] in MODEL_CASES]
    _fake_results(root, board, {**manifest, "cases": selected})

    comparison = compare_fused_probes(root, board, tmp_path / "selected-comparison", allow_host_fake=True,
                                     case_names=list(reversed(MODEL_CASES)))
    assert [case["name"] for case in comparison["cases"]] == list(MODEL_CASES)
    assert all(case["process"] == "PASSED" for case in comparison["cases"])
    assert comparison["probe_manifest_sha256"] == sha256_file(root / "manifest.json")
    assert comparison["numerical_acceptance"] == "NOT_DEFINED"
    assert comparison["placement"] == "UNVERIFIED"

    original_scope = compare_fused_probes(root, board, tmp_path / "original-comparison", allow_host_fake=True)
    rows = {case["name"]: case for case in original_scope["cases"]}
    assert len(rows) == 9 and rows["rms"]["compile"] == "FAILED"
    assert rows["conv_prefill"]["result_validation"] == rows["conv_decode"]["result_validation"] == "FAILED"
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("selection", [[], (), ["gdn_decode", "gdn_decode"], ["unknown"],
                                        ["gdn_prefill"], [None], [{}], "gdn_decode", True])
def test_comparison_rejects_invalid_or_absent_selections_before_output(setup, tmp_path, selection):
    binding, _, _ = setup
    root = tmp_path / "source"
    build_fused_probes(root, binding, cases="gdn_decode")
    with pytest.raises(ValueError, match="nonempty unique subset of probe manifest names"):
        compare_fused_probes(root, tmp_path / "board", tmp_path / "invalid", case_names=selection)
    assert not (tmp_path / "invalid").exists()


def test_selection_does_not_bypass_validation_of_the_source_manifest(setup, tmp_path):
    binding, _, _ = setup
    root = tmp_path / "source"
    manifest = build_fused_probes(root, binding, cases="gdn_decode,rms")
    manifest["cases"][-1]["operator_type"] = "UnexpectedOperator"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="case contract disagrees"):
        compare_fused_probes(root, tmp_path / "board", tmp_path / "invalid", case_names=["gdn_decode"])
    assert not (tmp_path / "invalid").exists()


def test_selected_board_script_runs_only_requested_cases_and_continues_after_failure(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    script = bundle / "run-board.sh"
    script.write_text(_board_script(MODEL_CASES))
    runner = tmp_path / "fake-runner.sh"
    runner.write_text('''#!/bin/sh
printf '%s\\n' "$@"
case "$2" in */gdn_prefill/step.tsv) exit 7;; esac
exit 0
''')
    runner.chmod(0o700)
    results = tmp_path / "results"
    invoked = subprocess.run(["sh", str(script), str(runner), str(results), "test runtime", "test ddk", "test firmware"],
                             capture_output=True, text=True, check=False)
    assert invoked.returncode == 1
    assert {path.name for path in results.glob("*.exitcode")} == {name + ".exitcode" for name in MODEL_CASES}
    for name in MODEL_CASES:
        assert (results / (name + ".stdout")).read_text().splitlines() == [
            "--fused-op-step", str(bundle / name / "step.tsv"), "--output", str(results / name),
            "--expected-runtime-version", "test runtime", "--ddk-id", "test ddk", "--firmware-id", "test firmware"]
        assert (results / (name + ".exitcode")).read_text().strip() == ("7" if name == "gdn_prefill" else "0")


@pytest.mark.parametrize("selection", [["gdn_decode", "gdn_decode"], ["rms; exit 0"], "gdn_decode"])
def test_board_script_rejects_invalid_case_names(selection):
    with pytest.raises(ValueError, match="unique known probe names"):
        _board_script(selection)

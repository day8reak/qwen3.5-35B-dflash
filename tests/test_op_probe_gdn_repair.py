"""Contracts for the combined correction/precision/integrated GDN batch."""
import copy
import json

import numpy as np
import pytest

from qwen35_moe_dflash import op_probe
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.op_probe_gdn_repair import cases_from_prepared, _native_norm_names
from test_op_probe_gdn_accuracy import case, prepared_source
from test_ge_select_lowering import RecipeReader


def _batch(case):
    return cases_from_prepared(case["prepared"], case["reference_path"], case["omc"])


def test_repair_batch_binds_all_cases_and_preserves_reference_and_weights(case):
    before = {str(p): sha256_file(p) for p in case["root"].rglob("*") if p.is_file()}
    batch, provenance = _batch(case)
    assert len(batch) == 41
    assert len(set(provenance["case_names"])) == 41
    assert provenance["case_names"] == [c["name"] for c in batch]
    assert provenance["suite"] == "gdn-repair"
    assert all(c["probe_suite"] == "gdn-repair" for c in batch)
    assert len(provenance["native_norm_reduction_names"]) == 3
    assert provenance["bindings"]["reference_sha256"] == sha256_file(case["reference_path"])
    by_name = {c["name"]: c for c in batch}
    for policy in ("matmul", "native_norms"):
        final, stages = [by_name[f"gr_layer0_{policy}_{capture}"] for capture in ("final", "stages")]
        assert final["output_labels"] == ["layer0_final"]
        assert len(stages["output_labels"]) == 24
        np.testing.assert_array_equal(final["expected"][0].view(np.uint16), stages["expected"][0].view(np.uint16))
        for name in final["weights"]:
            np.testing.assert_array_equal(final["weights"][name], stages["weights"][name])
        overrides = final["emit_options"].get("fp16_native_reductions", [])
        assert overrides == (provenance["native_norm_reduction_names"] if policy == "native_norms" else [])
    for left, right in zip(by_name["gr_layer0_matmul_stages"]["expected"], by_name["gr_layer0_native_norms_stages"]["expected"]):
        np.testing.assert_array_equal(left.view(np.uint16), right.view(np.uint16))
    for c in batch:
        assert provenance["candidate_emit_options"][c["name"]] == c["emit_options"]
        assert len(c["graph"].inputs) + len(c["graph"].outputs) <= 32
        for label, value in zip(c["output_labels"], c["graph"].outputs):
            record = provenance["stage_maps"][c["name"]][label]
            assert record["name"] == value.name
            assert record["shape"] == list(value.shape)
    assert before == {str(p): sha256_file(p) for p in case["root"].rglob("*") if p.is_file()}


def test_integrated_native_override_affects_only_three_norm_reductions(case, tmp_path):
    batch, provenance = _batch(case)
    item = next(c for c in batch if c["name"] == "gr_layer0_native_norms_stages")
    report = {}
    weights = {}
    for i, (name, value) in enumerate(item["weights"].items()):
        path = tmp_path / f"w{i}.bin"
        value.tofile(path)
        weights[name] = path
    emit_cpp(item["graph"], weights, recipe_dir=tmp_path / "recipe", reduction_report=report, **item["emit_options"])
    selected = set(provenance["native_norm_reduction_names"])
    assert {v["name"] for v in report["retained"] if v["reason"] == "explicit_native_override"} == selected
    assert report["counts"]["scan_prefix_replaced"] > 0
    assert report["counts"]["fp16_replaced"] > 3
    ops, _, _ = RecipeReader(tmp_path / "recipe" / "graph.recipe").read()
    for name in selected:
        assert next(op for op in ops if op["name"] == name)["type"] in {"ReduceSum", "ReduceMean"}
    malformed = copy.deepcopy(item)
    malformed["output_labels"] = ["not_normalized_query" if x == "normalized_query" else x for x in malformed["output_labels"]]
    with pytest.raises((KeyError, ValueError)):
        _native_norm_names(malformed)


def test_repair_cli_prepares_whole_batch_and_summary_without_vendor_calls(case, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only must not touch vendor compiler or board")
    monkeypatch.setattr(op_probe, "compiler_resources", forbidden)
    output = tmp_path / "repair"
    args = op_probe.parser().parse_args(["--suite", "gdn-repair", "--prepared", str(case["prepared"]),
        "--reference", str(case["reference_path"]), "--omc", str(case["omc"]),
        "--prepare-only", "--output", str(output)])
    assert op_probe.run(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert len(manifest["cases"]) == len(report["cases"]) == 41
    summary = (output / "gdn-summary.txt").read_text()
    assert "gr_layer0_native_norms_stages" in summary
    assert "gc_conv_layout_actual" in summary
    assert "numerical=NOT_RUN" in summary


def test_repair_rejects_different_source_policy_before_mislabeling_baseline(monkeypatch):
    from qwen35_moe_dflash import op_probe_gdn_repair
    monkeypatch.setattr(op_probe_gdn_repair, "comprehensive_cases", lambda *args: ([], {
        "policies": {"fp16_reductions": "native", "fp16_row_mul": "compact", "gdn_indexed_copy": "gather"}}))
    with pytest.raises(ValueError, match="source policies"):
        cases_from_prepared(None, None, None)

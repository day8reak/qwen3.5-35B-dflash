"""CLI policy binding/order with real recipes; mocked compilation is not device evidence."""
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash.hashing import sha256_file
from test_full_draft_cli import job


def _receipt(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def policy_job(job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_standalone, existing_air, draft_compile_precision
    from qwen35_moe_dflash.ge_cpp import emit_cpp
    from qwen35_moe_dflash.ge_ir import Graph
    main, fresh, common, calls, failures = job
    ddk = tmp_path / "ddk"
    schema = ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
    schema.parent.mkdir(parents=True)
    schema.write_bytes(b"mock selected schema; verifier is mocked")
    omg = ddk / "tools/tools_omg/master/omg"
    omg.write_bytes(b"mock selected compiler")
    state = {"verification": "PASSED", "received": [], "schema": schema}
    original_export = draft_standalone.export_draft
    original_compile = existing_air.compile_existing_air

    def export(prepared, folder, ge, **kwargs):
        original_export(prepared, folder, ge, **kwargs)
        graph = Graph("draft")
        x = graph.input("input", (1, 16, 8), "float16")
        p = graph.op("cast", x, shape=x.shape, dtype="float32", name="promoted")
        square = graph.op("mul", p, p, shape=x.shape, dtype="float32", name="square")
        mean = graph.op("reduce_mean", square, shape=(1, 16, 1), dtype="float32",
                        attrs={"axes": [-1], "keepdims": True}, name="mean")
        epsilon = graph.op("scalar_add", mean, shape=mean.shape, dtype="float32",
                           attrs={"value": 1e-6}, name="epsilon")
        inv = graph.op("rsqrt", epsilon, shape=mean.shape, dtype="float32", name="inverse")
        norm = graph.op("mul", p, inv, shape=x.shape, dtype="float32", name="normalized")
        graph.outputs = [graph.op("cast", norm, shape=x.shape, dtype="float16", name="hidden")]
        emit_cpp(graph, {}, recipe_dir=folder / "draft/recipe")
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest["graphs"][0]["construction_files"] = [_receipt(folder / "draft/recipe/graph.recipe", folder)]
        (folder / "manifest.json").write_text(json.dumps(manifest))

    def compile_air(source, folder, selected_ddk, **kwargs):
        result = original_compile(source, folder, selected_ddk, **kwargs)
        if result["status"] != "COMPILED":
            return result
        result.update(ddk_root=str(ddk), omg_sha256=sha256_file(omg))
        policy = kwargs.get("cl_customization")
        state["received"].append(policy.read_bytes() if policy is not None else None)
        if policy is not None:
            destination = folder / "compile-work/options/cl-customization.json"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(policy.read_bytes())
            result["cl_customization"] = _receipt(destination, folder)
            result["requested_options"]["cl_customization"] = "../options/cl-customization.json"
        (folder / "report.json").write_text(json.dumps(result))
        return result

    def verify(omc, selected_schema, plan):
        calls.append("precision-check")
        assert selected_schema == schema
        assert len(plan["required_nodes"]) == 5
        return {"status": state["verification"], "required_nodes": 5,
                "matched_nodes": 5 if state["verification"] == "PASSED" else 0,
                "omc": _receipt(omc, omc.parent), "recipe": plan["recipe"],
                "nodes": [] if state["verification"] == "PASSED" else [
                    {"name": "square", "status": "MISMATCH", "errors": ["output is FLOAT16"]}]}

    monkeypatch.setattr(draft_standalone, "export_draft", export)
    monkeypatch.setattr(existing_air, "compile_existing_air", compile_air)
    monkeypatch.setattr(draft_compile_precision, "verify_compiled_precision", verify)
    # Exercise the new default rather than the legacy sequencing fixture's override.
    fresh = fresh[:-2]
    common = common[:-2]
    return main, fresh, common, calls, failures, state


def test_default_locks_source_fp32_and_verifies_before_any_bundle(policy_job, tmp_path):
    main, fresh, _, calls, _, state = policy_job
    assert main(fresh) == 0
    assert calls == ["prepared", "fixture", "air", "compilation", "precision-check", "bundle", "board"]
    root = tmp_path / "full-draft"
    result = json.loads((root / "result.json").read_text())
    assert result["precision_policy"] == "fp32-cpu"
    assert result["compiled_precision"]["status"] == "PASSED"
    config = json.loads(state["received"][0])
    strategy = config["npucl_strategy"]["npucl_graphs"]["draft"]
    assert strategy["groups_info"] == [{"npucl_implementation": 7}]
    assert all(row == {"group_id": 0} for row in strategy["nodes_info"].values())
    assert set(strategy["nodes_info"]) == {"square", "mean", "epsilon", "inverse", "normalized"}
    assert not list(root.glob(".draft-precision-*"))
    assert (root / "compilation/precision-check.json").is_file()


def test_retry_inherits_and_rechecks_policy_without_recompile_or_air_export(policy_job, tmp_path):
    main, fresh, _, calls, _, state = policy_job
    assert main(fresh) == 0
    root = tmp_path / "full-draft"
    compiler_receipt = (root / "compilation/report.json").read_bytes()
    calls.clear()
    assert main([]) == 0
    assert calls == ["precision-check", "board"]
    assert (root / "compilation/report.json").read_bytes() == compiler_receipt
    assert len(state["received"]) == 1
    assert [p.name for p in tmp_path.glob("full-draft*")] == ["full-draft"]


def test_switch_from_legacy_default_only_recompiles_and_rebundles(policy_job, tmp_path):
    main, fresh, common, calls, _, _ = policy_job
    assert main([*fresh, "--precision-policy", "compiler-default"]) == 0
    root = tmp_path / "full-draft"
    before = {name: _snapshot(root / name) for name in ("prepared", "fixture", "air")}
    # Missing policy emulates the prior version; omitted option must lock FP32.
    result = json.loads((root / "result.json").read_text())
    result.pop("precision_policy")
    (root / "result.json").write_text(json.dumps(result))
    calls.clear()
    assert main(common) == 0
    assert calls == ["compilation", "precision-check", "bundle", "board"]
    result = json.loads((root / "result.json").read_text())
    assert result["resume_invalidations"][-1]["reason"] == "DRAFT_PRECISION_POLICY_CHANGED"
    assert all(_snapshot(root / name) == values for name, values in before.items())


def test_explicit_reproduction_override_removes_old_customization(policy_job, tmp_path):
    main, fresh, common, calls, _, state = policy_job
    assert main(fresh) == 0
    calls.clear()
    assert main([*common, "--precision-policy", "compiler-default"]) == 0
    assert calls == ["compilation", "bundle", "board"]
    assert state["received"][-1] is None
    compiled = json.loads((tmp_path / "full-draft/compilation/report.json").read_text())
    assert "cl_customization" not in compiled


def test_changed_policy_bytes_invalidate_compilation_only(policy_job, tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_precision_policy
    main, fresh, common, calls, _, state = policy_job
    assert main(fresh) == 0
    old = draft_precision_policy.customization_bytes
    monkeypatch.setattr(draft_precision_policy, "customization_bytes", lambda plan: old(plan) + b"\n")
    calls.clear()
    assert main(common) == 0
    assert calls == ["compilation", "precision-check", "bundle", "board"]
    assert state["received"][0] != state["received"][1]


@pytest.mark.parametrize("payload", ["compilation/compile-work/options/cl-customization.json", "air/draft/recipe/graph.recipe"])
def test_tampered_policy_or_recipe_fails_before_old_board_is_deleted(policy_job, tmp_path, payload):
    main, fresh, _, calls, _, _ = policy_job
    assert main(fresh) == 0
    root = tmp_path / "full-draft"
    (root / "board/sentinel").write_text("previous board evidence")
    with (root / payload).open("ab") as out:
        out.write(b"changed")
    before = _snapshot(root)
    calls.clear()
    assert main([]) == 1
    assert calls == []
    assert _snapshot(root) == before


@pytest.mark.parametrize("status", ["FAILED", "NOT_VERIFIED"])
def test_verification_failure_blocks_bundle_and_next_retry_rechecks_same_omc(policy_job, tmp_path, capsys, status):
    main, fresh, _, calls, _, state = policy_job
    state["verification"] = status
    assert main(fresh) == 1
    root = tmp_path / "full-draft"
    assert calls == ["prepared", "fixture", "air", "compilation", "precision-check"]
    assert not (root / "bundle").exists() and not (root / "board").exists()
    assert "square: output is FLOAT16" in capsys.readouterr().out
    compiler_receipt = (root / "compilation/report.json").read_bytes()
    state["verification"] = "PASSED"
    calls.clear()
    assert main([]) == 0
    assert calls == ["precision-check", "bundle", "board"]
    assert (root / "compilation/report.json").read_bytes() == compiler_receipt
    assert json.loads((root / "compilation/precision-check.json").read_text())["status"] == "PASSED"


def test_embedded_is_rejected_before_reset(policy_job, tmp_path):
    main, fresh, _, calls, _, _ = policy_job
    assert main([*fresh, "--precision-policy", "compiler-default", "--weight-storage", "embedded"]) == 0
    root = tmp_path / "full-draft"
    (root / "board/sentinel").write_text("keep")
    before = _snapshot(root)
    calls.clear()
    assert main(["--precision-policy", "fp32-cpu"]) == 1
    assert calls == [] and _snapshot(root) == before


@pytest.mark.parametrize("mode", ["--compare-only", "--replay-reference", "--diagnose-numerics", "--inspect-only"])
def test_readonly_modes_reject_precision_selection_without_mutation(policy_job, tmp_path, mode):
    main, fresh, _, calls, _, _ = policy_job
    assert main(fresh) == 0
    root = tmp_path / "full-draft"
    before = _snapshot(root)
    calls.clear()
    assert main([mode, "--resume", str(root), "--precision-policy", "fp32-cpu"]) == 1
    assert calls == [] and _snapshot(root) == before


def test_prepare_only_records_default_without_ddk_or_recipe(policy_job, tmp_path):
    main, fresh, _, calls, _, _ = policy_job
    assert main([*fresh[:4], "--prepare-only"]) == 0
    assert calls == ["prepared", "fixture"]
    assert json.loads((tmp_path / "full-draft/result.json").read_text())["precision_policy"] == "fp32-cpu"


def test_explicit_schema_must_belong_to_selected_ddk_before_reset(policy_job, tmp_path):
    main, fresh, _, calls, _, state = policy_job
    assert main(fresh) == 0
    root = tmp_path / "full-draft"
    before = _snapshot(root)
    other = tmp_path / "another-schema.so"
    other.write_bytes(state["schema"].read_bytes())
    calls.clear()
    assert main(["--schema-library", str(other)]) == 1
    assert calls == [] and _snapshot(root) == before


@pytest.mark.parametrize("binding,owner", [
    ("compiler_report_sha256", "compilation"),
    ("export_manifest_sha256", "air"),
    ("fixture_report_sha256", "fixture"),
])
def test_reused_bundle_bound_to_different_source_refuses_board_after_current_precision_passes(
        policy_job, tmp_path, binding, owner):
    main, fresh, _, calls, _, _ = policy_job
    assert main(fresh) == 0
    root = tmp_path / "full-draft"
    result = json.loads((root / "result.json").read_text())
    source = result["components"][owner]
    prior_payload = json.loads((Path(source["directory"]) / source["manifest"]["path"]).read_text())
    prior_payload["historical_build"] = True
    prior_source = tmp_path / ("prior-" + owner + ".json")
    prior_source.write_text(json.dumps(prior_payload))
    manifest_path = root / "bundle/manifest.json"
    bundle = json.loads(manifest_path.read_text())
    bundle[binding] = sha256_file(prior_source)
    manifest_path.write_text(json.dumps(bundle))
    # This is a valid manifest receipt for a different source generation,
    # rather than a corrupt file that the ordinary receipt check would catch.
    receipt = {"directory": str(root / "bundle"), "manifest": _receipt(manifest_path, root / "bundle")}
    (root / "completed-bundle.json").write_text(json.dumps(receipt))
    result["components"]["bundle"] = receipt
    (root / "result.json").write_text(json.dumps(result))
    calls.clear()
    assert main([]) == 1
    assert calls == ["precision-check"]
    failed = json.loads((root / "result.json").read_text())
    assert failed["compiled_precision"]["status"] == "PASSED"
    assert failed["stage"] == "bundle"
    assert binding in failed["error"] and "refusing board execution" in failed["error"]
    assert not (root / "board").exists()

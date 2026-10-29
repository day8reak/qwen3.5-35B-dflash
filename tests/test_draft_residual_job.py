"""A residual candidate reuses fixed CPU goldens without adopting old stages."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from qwen35_moe_dflash.artifacts import save_json
from qwen35_moe_dflash.draft_board import _record
from qwen35_moe_dflash.draft_trace_job import run_trace
from qwen35_moe_dflash.hashing import sha256_file
from test_draft_standalone import fake_ge
from test_draft_trace_job import source


def _snapshot(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _cache_reference(root, monkeypatch):
    from qwen35_moe_dflash import draft_layer_trace
    assert run_trace(root, ge_root=None, prepare_only=True)["status"] == "PREPARED"
    monkeypatch.setattr(draft_layer_trace, "trace_prepared_draft", lambda *a, **k:
                        pytest.fail("candidate must reuse the retained CPU observations"))


def test_candidate_reuses_two_cached_goldens_and_preserves_every_old_file(source, monkeypatch):
    root, _ = source
    _cache_reference(root, monkeypatch)
    legacy = run_trace(root, ge_root=None, prepare_only=True, direct=True)
    assert legacy["status"] == "PREPARED"
    before = _snapshot(root)
    result = run_trace(root, ge_root=None, prepare_only=True, direct=True, residual_policy="cpu-fp32-add")
    assert result["status"] == "PREPARED", result
    lane = root / "residual-fix"
    assert result["output"] == str(lane)
    assert result["residual_policy"] == "cpu-fp32-add"
    fixture = json.loads((lane / "fixture/report.json").read_text())
    assert [r["name"] for r in fixture["expected_direct"]] == ["draft_top1", "draft_hidden"]
    original = json.loads((root / "fixture/report.json").read_text())
    assert [r["sha256"] for r in fixture["expected_direct"]] == [original["expected_top1"]["sha256"], original["expected_hidden"]["sha256"]]
    owner = json.loads((lane / "owner.json").read_text())
    assert owner["residual_policy"] == "cpu-fp32-add" and "ge_residual.py" in owner["code"]
    old_owner = json.loads((root / "direct-trace/owner.json").read_text())
    assert "residual_policy" not in old_owner and "ge_residual.py" not in old_owner["code"]
    owner_bytes = (lane / "owner.json").read_bytes()
    second = run_trace(root, ge_root=None, prepare_only=True, direct=True, residual_policy="cpu-fp32-add")
    assert second["status"] == "PREPARED"
    assert (lane / "owner.json").read_bytes() == owner_bytes
    assert {path: path.read_bytes() for path in before} == before
    assert (root / "prepared").is_dir() and (root / "fixture").is_dir()
    assert not (lane / "prepared").exists()


@pytest.mark.parametrize("passed", [False, True])
def test_candidate_precision_gate_and_compile_cache(source, monkeypatch, tmp_path, passed):
    from qwen35_moe_dflash import draft_standalone, existing_air, draft_compile_precision, draft_direct_abi, draft_board
    root, ddk = source
    _cache_reference(root, monkeypatch)
    ge = fake_ge(tmp_path)
    calls = []

    def logged(command, **kwargs):
        calls.append("export-tool")
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock compiler result")
        else:
            Path(command[1]).write_bytes(b"mock AIR; real source recipe tested")

    def compile_air(air, output, selected, **kwargs):
        calls.append("compile")
        output.mkdir()
        (output / "cl.json").write_bytes(kwargs["cl_customization"].read_bytes())
        (output / "model.omc").write_bytes(b"mock compiled OMC")
        report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED",
                  "ddk_root": str(selected), "ddk_id": kwargs["ddk_id"],
                  "requested_options": {"use_origin_format": "true", "weight_data_type": "FP16",
                                        "limit_partition_size": "false", "save_weights_as_external_data": "true"},
                  "omg_sha256": sha256_file(ddk / "tools/tools_omg/master/omg"),
                  "cl_customization": _record(output / "cl.json", output), "omc": _record(output / "model.omc", output)}
        save_json(output / "report.json", report)
        return report

    def verify(omc, schema, plan):
        calls.append("verify")
        sums = [n for n in plan["required_nodes"] if n["name"].startswith("__draft_residual/")]
        config = json.loads((root / "prepared/prepared.json").read_text())["draft_config"]
        assert len(sums) == config["num_hidden_layers"] * 2
        assert all(n["type"] == "Add" for n in sums)
        return {"status": "PASSED" if passed else "FAILED", "required_nodes": len(plan["required_nodes"]),
                "matched_nodes": len(plan["required_nodes"]) if passed else len(plan["required_nodes"]) - 1}

    def inspect_outputs(omc, schema, spec):
        assert passed, "failed precision must stop before output ABI inspection"
        calls.append("output-abi")
        assert [row["name"] for row in spec] == ["draft_top1", "draft_hidden"]
        return {"artifact_kind": "qwen35-draft-direct-output-abi-inspection", "status": "PASSED",
                "declared_spec": spec, "observed_spec": spec, "changes": [],
                "omc": _record(omc, omc.parent),
                "schema": {"library": str(schema), "library_sha256": sha256_file(schema)}}

    def forbidden(*args, **kwargs):
        pytest.fail("compile-only candidate must stop before bundle and board")

    monkeypatch.setattr(draft_standalone, "_logged", logged)
    monkeypatch.setattr(existing_air, "compile_existing_air", compile_air)
    monkeypatch.setattr(draft_compile_precision, "verify_compiled_precision", verify)
    monkeypatch.setattr(draft_direct_abi, "inspect_direct_output_abi", inspect_outputs)
    monkeypatch.setattr(draft_board, "prepare_draft_bundle", forbidden)
    monkeypatch.setattr(draft_board, "execute_draft_bundle", forbidden)
    before = _snapshot(root)
    result = run_trace(root, ge_root=ge, direct=True, residual_policy="cpu-fp32-add", compile_only=True)
    assert result["status"] == ("COMPILED" if passed else "FAILED"), result
    if passed:
        assert result["compiled_precision"]["status"] == "PASSED"
        assert result["compiled_output_abi"]["status"] == "PASSED"
        assert result["source_output_precision"] == "SOURCE_DTYPES_PRESERVED"
    else:
        assert "compiled precision did not pass" in result["error"]
    lane = root / "residual-fix"
    exported = json.loads((lane / "air/manifest.json").read_text())
    assert exported["residual_policy"] == "cpu-fp32-add"
    assert len(exported["direct_spec"]) == 2
    calls.clear()
    repeated = run_trace(root, ge_root=ge, direct=True, residual_policy="cpu-fp32-add", compile_only=True)
    assert repeated["status"] == ("COMPILED" if passed else "FAILED")
    assert calls == (["verify", "output-abi"] if passed else ["verify"])
    assert not (lane / "bundle").exists() and not (lane / "board").exists()
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("direct,policy", [(False, "cpu-fp32-add"), (True, "other")])
def test_invalid_candidate_selection_does_not_create_workspace(tmp_path, monkeypatch, direct, policy):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="residual"):
        run_trace(tmp_path / "missing", ge_root=None, direct=direct, residual_policy=policy)
    assert not (tmp_path / "missing").exists()


def test_candidate_cli_skips_packed_observation_review(tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_trace_job, draft_trace_integrity
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    old = tmp_path / "full-draft/layer-trace/board/result.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}")
    monkeypatch.setattr(draft_trace_integrity, "review_trace_integrity", lambda *a, **k: pytest.fail("old packed review must be skipped"))
    calls = []
    monkeypatch.setattr(draft_trace_job, "run_trace", lambda *a, **k: calls.append((a, k)) or {"status": "PREPARED"})
    path = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_draft_trace.py"
    spec = importlib.util.spec_from_file_location("candidate_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--direct", "--residual-policy", "cpu-fp32-add", "--prepare-only"]) == 0
    assert calls[0][1]["residual_policy"] == "cpu-fp32-add" and calls[0][1]["direct"] is True


def test_residual_wrapper_fixes_candidate_policy(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "run_draft_trace", SimpleNamespace(main=lambda args: calls.append(args) or 0))
    monkeypatch.setattr(sys, "argv", ["run_draft_residual.py", "--compile-only"])
    path = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_draft_residual.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(path), run_name="__main__")
    assert error.value.code == 0
    assert calls == [["--compile-only", "--direct", "--residual-policy", "cpu-fp32-add"]]

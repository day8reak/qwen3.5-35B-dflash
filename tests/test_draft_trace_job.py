"""Real tiny reference tracing; mock compiler receipts test orchestration gates only."""
import json
from pathlib import Path

import numpy as np
import pytest

from qwen35_moe_dflash.artifacts import save_json
from qwen35_moe_dflash.draft_board import _record
from qwen35_moe_dflash.draft_reference import prepare_draft_reference
from qwen35_moe_dflash.draft_standalone import prepare_draft
from qwen35_moe_dflash.draft_trace_job import run_trace


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    root = tmp_path / "full-draft"
    root.mkdir()
    prepare_draft(root / "prepared", sequence_length=8, tiny=True)
    prepare_draft_reference(root / "prepared", root / "fixture", valid_length=6, seed=17)
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"mock compiler identity")
    schema = omg.parent / "lib64/libai_fmk_graph_proto_host.so"
    schema.parent.mkdir()
    schema.write_bytes(b"mock schema identity")
    (root / "compilation").mkdir()
    save_json(root / "compilation/report.json", {"ddk_root": str(ddk), "ddk_id": "test-ddk"})
    names = {"prepared": "prepared.json", "fixture": "report.json", "compilation": "report.json"}
    job = {"artifact_kind": "qwen35-full-draft-job", "compiled_precision": {"status": "PASSED"},
           "components": {name: {"directory": str(root / name), "manifest": _record(root / name / file, root / name)}
                          for name, file in names.items()}}
    save_json(root / "result.json", job)
    from qwen35_moe_dflash import draft_review
    monkeypatch.setattr(draft_review, "review_draft_result", lambda _: {"device_execution": "PASSED"})
    return root, ddk


def test_prepare_trace_reuses_original_inputs_goldens_and_exact_reference(source):
    root, _ = source
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = run_trace(root, ge_root=None, prepare_only=True)
    assert result["status"] == "PREPARED", result
    reference = json.loads((root / "layer-trace/reference/report.json").read_text())
    assert all(row["status"] == "EXACT_MATCH" for row in reference["golden_reproduction"])
    fixture = json.loads((root / "layer-trace/fixture/report.json").read_text())
    original = json.loads((root / "fixture/report.json").read_text())
    assert fixture["inputs"] == original["inputs"]
    assert fixture["expected_hidden"] == original["expected_hidden"]
    assert fixture["expected_top1"] == original["expected_top1"]
    assert fixture["expected_trace"]["dtype"] == "float32"
    packed = np.fromfile(root / "layer-trace/fixture/draft_trace_values.bin", dtype="<f4")
    assert packed.size == sum(row["elements"] for row in fixture["trace_spec"])
    # The normal workspace lock is the only source-side metadata change.
    for path, raw in before.items():
        assert (root / path).read_bytes() == raw
    second = run_trace(root, ge_root=None, prepare_only=True)
    assert second["status"] == "PREPARED"
    assert (root / "layer-trace/reference/report.json").read_text() == json.dumps(reference, indent=2, ensure_ascii=False) + "\n"


def test_changed_cached_probe_is_rejected_before_compile(source):
    root, _ = source
    assert run_trace(root, ge_root=None, prepare_only=True)["status"] == "PREPARED"
    report = json.loads((root / "layer-trace/reference/report.json").read_text())
    (root / "layer-trace/reference" / report["tensors"][0]["path"]).write_bytes(b"changed")
    result = run_trace(root, ge_root=None, prepare_only=True)
    assert result["status"] == "FAILED" and "mismatch" in result["error"]
    assert not (root / "layer-trace/compilation").exists()


def test_changed_production_identity_preserves_trace_and_refuses_reuse(source):
    root, _ = source
    assert run_trace(root, ge_root=None, prepare_only=True)["status"] == "PREPARED"
    before = (root / "layer-trace/owner.json").read_bytes()
    with (root / "result.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="different source/code identity"):
        run_trace(root, ge_root=None, prepare_only=True)
    assert (root / "layer-trace/owner.json").read_bytes() == before


def test_trace_export_keeps_separate_contract_and_bound_packed_descriptor(source, monkeypatch, tmp_path):
    from qwen35_moe_dflash import draft_standalone
    from test_draft_standalone import fake_ge
    root, _ = source

    def logged(command, **kwargs):
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock serializer")
        else:
            Path(command[1]).write_bytes(b"mock AIR")

    monkeypatch.setattr(draft_standalone, "_logged", logged)
    output = tmp_path / "trace-export"
    names = ["context_fc", "projected_context", "draft_hidden"]
    exported = draft_standalone.export_draft(root / "prepared", output, fake_ge(tmp_path), trace_names=names)
    assert exported["export_contract"].endswith("-layer-trace-v1")
    assert exported["diagnostic_only"] is True
    assert [row["name"] for row in exported["trace_spec"]] == names
    graph = exported["graphs"][0]
    assert graph["outputs"] == ["draft_top1", "draft_trace_values"]
    assert graph["output_dtypes"] == ["torch.int32", "torch.float32"]
    assert graph["output_shapes"][1] == [1, 1, sum(row["elements"] for row in exported["trace_spec"])]
    command = json.loads((output / "draft/export-command.json").read_text())
    assert command["export_contract"] == exported["export_contract"]


@pytest.mark.parametrize("changed", [False, True])
def test_terminal_equivalence_distinguishes_trace_partition_changes(tmp_path, monkeypatch, changed):
    from qwen35_moe_dflash.draft_trace_job import production_equivalence
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    job, trace = tmp_path / "source", tmp_path / "trace"
    original = np.arange(24, dtype="<f2").reshape(1, 3, 8) / 4
    observed = original.astype("<f4")
    if changed:
        observed[0, 1, 3] += .5
    spec = [{"name": "draft_hidden", "shape": [1, 3, 8], "dtype": "float16", "offset": 5, "elements": 24}]
    for root, hidden in ((job, original), (trace, np.concatenate((np.zeros(5, dtype="<f4"), observed.reshape(-1))))):
        result = root / "board/results"
        result.mkdir(parents=True)
        np.asarray([[2, 3, 4]], dtype="<i4").tofile(result / "output_0000.bin")
        hidden.tofile(result / "output_0001.bin")
        save_json(result / "result.json", {"outputs": [_record(result / f"output_{i:04d}.bin", result) for i in range(2)]})
    report = production_equivalence(job, trace, spec)
    assert report["status"] == ("DIFFERENT" if changed else "EXACT")
    assert report["hidden"]["exact_match"] is (not changed)
    assert report["token_mismatches"] == 0


@pytest.mark.parametrize("passed", [False, True])
def test_precision_gate_precedes_bundle_and_compile_cache_is_checked(source, monkeypatch, passed):
    from qwen35_moe_dflash import draft_standalone, draft_compile_precision, existing_air, draft_board
    from qwen35_moe_dflash.ge_cpp import emit_cpp
    from qwen35_moe_dflash.hashing import sha256_file
    root, ddk = source
    calls = []

    def export(prepared, output, ge, **kwargs):
        output.mkdir()
        from qwen35_moe_dflash.draft_ge_trace import instrument_draft_trace
        graph = draft_standalone._graph(8, tiny=True)[2]
        graph, spec = instrument_draft_trace(graph, names=kwargs["trace_names"])
        plan = json.loads((prepared / "prepared.json").read_text())
        paths = {name: str(prepared / row["path"]) for name, row in plan["weights"].items()}
        emit_cpp(graph, paths, recipe_dir=output / "draft/recipe")
        (output / "draft.air").write_bytes(b"mock AIR; no vendor serializer executed")
        result = {"status": "EXPORTED", "trace_spec": spec, "graphs": [{"air": "draft.air",
                  "files": [_record(output / "draft.air", output)],
                  "construction_files": [_record(output / "draft/recipe/graph.recipe", output)]}]}
        save_json(output / "manifest.json", result)
        calls.append("export")
        return result

    def compile_air(air, output, selected, **kwargs):
        output.mkdir()
        policy = kwargs["cl_customization"].read_bytes()
        (output / "cl.json").write_bytes(policy)
        (output / "model.omc").write_bytes(b"mock OMC; precision decoder mocked")
        result = {"status": "COMPILED", "omg_sha256": sha256_file(ddk / "tools/tools_omg/master/omg"),
                  "cl_customization": _record(output / "cl.json", output), "omc": _record(output / "model.omc", output)}
        save_json(output / "report.json", result)
        calls.append("compile")
        return result

    def verify(omc, schema, plan):
        calls.append("verify")
        return {"status": "PASSED" if passed else "FAILED", "required_nodes": len(plan["required_nodes"]),
                "matched_nodes": len(plan["required_nodes"]) if passed else 0}

    def bundle(*args, **kwargs):
        calls.append("bundle")
        raise AssertionError("bundle must not be invoked in compile-only or failed precision check")

    monkeypatch.setattr(draft_standalone, "export_draft", export)
    monkeypatch.setattr(existing_air, "compile_existing_air", compile_air)
    monkeypatch.setattr(draft_compile_precision, "verify_compiled_precision", verify)
    monkeypatch.setattr(draft_board, "prepare_draft_bundle", bundle)
    result = run_trace(root, ge_root="mock-ge", compile_only=True)
    assert result["status"] == ("COMPILED" if passed else "FAILED"), result
    assert calls == ["export", "compile", "verify"]
    calls.clear()
    result = run_trace(root, ge_root="mock-ge", compile_only=True)
    assert calls == ["verify"]
    assert result["status"] == ("COMPILED" if passed else "FAILED")
    assert not (root / "layer-trace/bundle").exists()


@pytest.fixture
def retained_trace_board(tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_board
    from qwen35_moe_dflash.hashing import sha256_file
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    bundle, board = tmp_path / "bundle", tmp_path / "board"
    bundle.mkdir()
    (board / "results").mkdir(parents=True)
    save_json(bundle / "manifest.json", {"ddk_id": "test-ddk"})
    save_json(board / "results/result.json", {"status": "EXECUTED", "runtime_version": "runtime",
                                             "firmware_id": "firmware"})
    save_json(board / "runtime.json", {"runtime_version": "runtime", "npu_supported": True})
    for name in ("draft.stdout", "draft.stderr", "runtime.stderr", "results/output_0000.bin", "results/output_0001.bin"):
        (board / name).write_bytes(b"host mock evidence only")
    for name in ("draft.exitcode", "runtime.exitcode"):
        (board / name).write_bytes(b"0\n")
    execution = {"artifact_kind": "qwen35-standalone-draft-execution", "remote_exitcode": 0,
                 "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
                 "model_load_api": "legacy-buffer", "runtime_version": "runtime", "firmware_id": "firmware",
                 "ddk_id": "test-ddk", "retrieved_sha256": {
                     str(path.relative_to(board)): sha256_file(path) for path in board.rglob("*") if path.is_file()}}
    expected = {"status": "FRESH_COMPARISON", "device_execution": "HOST_MOCK_ONLY"}
    calls = []
    def compare(*args, **kwargs):
        calls.append((args, kwargs))
        return expected
    monkeypatch.setattr(draft_board, "compare_draft_result", compare)
    return bundle, board, execution, expected, calls


def test_trace_board_revalidates_all_transport_before_fresh_comparison(retained_trace_board):
    from qwen35_moe_dflash.draft_trace_job import _review_trace_board
    bundle, board, execution, expected, calls = retained_trace_board
    assert _review_trace_board(bundle, board, execution) == expected
    assert calls[0][1] == {"expected_model_load_api": "legacy-buffer"}


@pytest.mark.parametrize("mutation", ["empty_hashes", "missing_output", "changed_output", "false_exit",
                                      "exit_file", "runtime_probe", "native_runtime", "ddk", "loader"])
def test_trace_board_rejects_incomplete_or_conflicting_cached_evidence(retained_trace_board, mutation):
    from qwen35_moe_dflash.draft_trace_job import _review_trace_board
    from qwen35_moe_dflash.hashing import sha256_file
    bundle, board, execution, _, calls = retained_trace_board
    if mutation == "empty_hashes":
        execution["retrieved_sha256"] = {}
    elif mutation == "missing_output":
        execution["retrieved_sha256"].pop("results/output_0001.bin")
    elif mutation == "changed_output":
        (board / "results/output_0001.bin").write_bytes(b"altered")
    elif mutation == "false_exit":
        execution["remote_exitcode"] = False
    elif mutation == "exit_file":
        (board / "draft.exitcode").write_bytes(b"1\n")
        execution["retrieved_sha256"]["draft.exitcode"] = sha256_file(board / "draft.exitcode")
    elif mutation == "runtime_probe":
        (board / "runtime.json").write_text(json.dumps({"runtime_version": "other", "npu_supported": True}))
        execution["retrieved_sha256"]["runtime.json"] = sha256_file(board / "runtime.json")
    elif mutation == "native_runtime":
        execution["firmware_id"] = "other"
    elif mutation == "ddk":
        execution["ddk_id"] = "other"
    elif mutation == "loader":
        execution["model_load_api"] = "file-v2"
    with pytest.raises(ValueError):
        _review_trace_board(bundle, board, execution)
    assert not calls

"""Bounded contracts for the five documented single-op GE cases."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from qwen35_moe_dflash.ge_fused.bindings import document_bindings
from qwen35_moe_dflash.ge_fused.probes import build_fused_probes, compare_fused_probes, inspect_fused_probes
from qwen35_moe_dflash.ge_fused.recipe import read_recipe
from qwen35_moe_dflash.hashing import sha256_file


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(document_bindings(target="kirinx90", ddk_id="test-ddk", gdn_initial_state_dtype="float32")))
    ge = tmp_path / "ge"
    (ge / "include/graph").mkdir(parents=True)
    (ge / "lib64").mkdir()
    for name in ("graph.h", "operator.h"):
        (ge / "include/graph" / name).write_text("MOCK HEADER")
    (ge / "lib64/libgraph.so").write_bytes(b"MOCK LIBGRAPH")
    ddk = tmp_path / "ddk"
    binary = ddk / "tools/tools_omg/master/omg"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"MOCK OMG")
    binary.chmod(0o700)
    return binding, ge, ddk


def _mock_tools(monkeypatch, fail=None):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        folder = Path(kwargs["cwd"])
        if "--target=omc" in command:
            (folder / "dump_graph").mkdir()
            (folder / "kernel_meta").mkdir()
            (folder / "kernel_meta/worker.log").write_text("MOCK child compilation traceback\n")
            if folder.parent.name == fail:
                (folder / "dump_graph/02_after_infershape_1_fail.irpb").write_bytes(b"MOCK FAILED IR")
                raise subprocess.CalledProcessError(1, command)
            (folder / "dump_graph/after_infershape.irpb").write_bytes(b"MOCK SUCCESS IR")
            (folder / "probe.omc").write_bytes(b"MOCK OMC " + folder.parent.name.encode())
        else:
            path = command[command.index("-o") + 1] if "-o" in command else command[1]
            Path(path).write_bytes(b"MOCK HOST OUTPUT")
        kwargs["stdout"].write("Mock tools; no GE/OMG/device execution.\n")
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_five_source_graphs_are_exact_single_custom_ops(setup, tmp_path, monkeypatch):
    binding, _, _ = setup
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("source construction must not run a tool"))
    result = build_fused_probes(tmp_path / "source", binding)
    assert result["status"] == "SOURCE_READY" and len(result["cases"]) == 5
    root = tmp_path / "source"
    for case in result["cases"]:
        assert case["construction"] == "PASSED" and case["export"] == case["compile"] == "NOT_RUN"
        graph = json.loads((root / "cases" / case["name"] / "graph.json").read_text())
        assert len(graph["nodes"]) == len(graph["custom_ops"]) == 1
        assert graph["weights"] == {} and graph["constants"] == {}
        recipe = read_recipe(root / "cases" / case["name"] / "recipe/graph.recipe")
        assert [record["type"] for record in recipe["records"]] == ["Data"] * len(case["inputs"]) + [case["operator_type"]]
        abi = json.loads((root / case["operator_abi"]["path"]).read_text())
        assert [item["name"] for item in case["inputs"]] == [item["name"] for item in abi["inputs"]]
        assert len(case["outputs"]) == 2 and case["bundle_manifest"] is None
        if case["role"] == "gdn":
            assert case["inputs"][0]["shape"] == [1, case["sequence_length"], 32, 128]
            assert case["inputs"][5]["dtype"] == "float32" and case["inputs"][6]["dtype"] == "int16"
            assert all(value["type"] == "int" for value in abi["attrs"].values())
            assert case["input_policy"]["ranges"]["initial_state"]["nonzero"] == 32 * 128 * 128
        elif case["role"] == "conv_state":
            assert case["outputs"][0]["reference"] is None
            assert case["outputs"][0]["shape"] == [1, 8192, 4]
        else:
            assert case["inputs"][0]["name"] == "self" and case["inputs"][0]["shape"] == [1, 8, 2048]
            assert case["input_policy"]["ranges"]["gamma"]["min"] == .75
    assert not list(root.rglob("*.air")) and not list(root.rglob("*.omc"))


def test_export_compile_boundaries_and_continue_after_failure(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    calls = _mock_tools(monkeypatch, fail="gdn_prefill")
    result = build_fused_probes(tmp_path / "compiled", binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    assert result["status"] == "PARTIAL" and len(calls) == 15
    assert result["cases"][0]["export"] == "PASSED" and result["cases"][0]["compile"] == "FAILED"
    assert all(case["status"] == "COMPILED" for case in result["cases"][1:])
    failed = tmp_path / "compiled/cases/gdn_prefill/compile-work"
    assert (failed / "dump_graph/02_after_infershape_1_fail.irpb").read_bytes() == b"MOCK FAILED IR"
    assert (failed / "kernel_meta/worker.log").read_text() == "MOCK child compilation traceback\n"
    assert (failed / "input.air").is_file() and not (failed / "input.air").is_symlink()
    assert "Retained compiler work:" in result["cases"][0]["error"]
    inventory = {item["path"]: item for item in result["cases"][0]["files"]}
    worker_record = inventory["cases/gdn_prefill/compile-work/kernel_meta/worker.log"]
    assert worker_record["sha256"] == sha256_file(failed / "kernel_meta/worker.log")
    assert not list((tmp_path / "compiled").rglob(".ipc-*"))
    script = (tmp_path / "compiled/bundle/run-board.sh").read_text()
    assert "gdn_prefill/step.tsv" not in script and "rms/step.tsv" in script
    for case in result["cases"][1:]:
        work = tmp_path / "compiled" / case["compile_work"]
        assert (work / "dump_graph/after_infershape.irpb").read_bytes() == b"MOCK SUCCESS IR"
        assert (work / "kernel_meta/worker.log").is_file()
        root = tmp_path / "compiled/bundle" / case["name"]
        assert not list(root.rglob("worker.log"))  # compiler diagnostics are not uploaded as model payloads
        lines = (root / "step.tsv").read_text().splitlines()
        assert lines[0] == "QWEN35_FUSED_OP_STEP\t1" and lines[3] == "operator\t" + case["operator_type"]
        assert lines[2].startswith("operator_abi\toperator-abi.json\t")
        assert not any("target_top1" in line for line in lines)


def _fake_results(probe, result_dir, manifest):
    result_dir.mkdir()
    for case in manifest["cases"]:
        step_root = probe / "bundle" / case["name"]
        step = json.loads((step_root / "manifest.json").read_text())
        output = result_dir / case["name"]
        output.mkdir()
        records = []
        for desc in case["outputs"]:
            dtype = "<f2" if desc["dtype"] == "float16" else "<f4"
            data = (np.frombuffer((probe / desc["reference"]["path"]).read_bytes(), dtype=dtype).reshape(desc["shape"]).copy()
                    if desc["reference"] else np.zeros(desc["shape"], dtype=dtype))
            if case["name"] == "gdn_prefill" and desc["logical"] == "core_attn":
                data[:, desc["valid_prefix_length"]:] = np.nan  # padding is explicitly outside comparison
            path = output / f"output_{desc['index']:04d}.bin"
            path.write_bytes(data.tobytes())
            records.append({**step["outputs"][desc["index"]], "path": path.name,
                            "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        result = {"schema_version": 1, "status": "EXECUTED", "scope": "fused-operator-probe",
                  "operator_type": step["operator_type"], "operator_calls": 1, "target_calls": 0, "draft_calls": 0,
                  "operator_abi_sha256": step["operator_abi"]["sha256"], "step_plan_sha256": step["step_plan"]["sha256"],
                  "model_sha256": step["model"]["sha256"], "ddk_id": "test-ddk", "runtime_version": "FAKE TEST ONLY",
                  "evidence_kind": "host-fake-hiai", "inputs": step["inputs"], "outputs": records}
        (output / "result.json").write_text(json.dumps(result))


def test_comparison_requires_bound_results_and_excludes_undefined_padding(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    _mock_tools(monkeypatch)
    root = tmp_path / "compiled"
    manifest = build_fused_probes(root, binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    board = tmp_path / "board"
    _fake_results(root, board, manifest)
    rejected = compare_fused_probes(root, board, tmp_path / "rejected")
    assert all(case["result_validation"] == "FAILED" for case in rejected["cases"])
    result = compare_fused_probes(root, board, tmp_path / "compared", allow_host_fake=True)
    assert all(case["process"] == "PASSED" and case["numerical"] == "DIAGNOSTIC_COMPLETE" for case in result["cases"])
    diagnostic = result["cases"][0]["diagnostics"][0]
    assert diagnostic["compared_shape"] == [1, 6, 32, 128] and diagnostic["max_abs_error"] == 0
    assert diagnostic["raw_nonfinite"] == 2 * 32 * 128 and diagnostic["actual_nonfinite"] == 0
    assert result["cases"][2]["diagnostics"][0]["reference"] == "SHAPE_ONLY_UNSPECIFIED_STATE_ORDER"
    path = board / "rms/result.json"
    bad = json.loads(path.read_text())
    bad["model_sha256"] = "0" * 64
    path.write_text(json.dumps(bad))
    changed = compare_fused_probes(root, board, tmp_path / "changed", allow_host_fake=True)
    assert changed["cases"][-1]["result_validation"] == "FAILED"
    assert all(case["process"] == "PASSED" for case in changed["cases"][:-1])


def test_compile_requires_matching_declared_platform_and_ddk(setup, tmp_path):
    binding, ge, ddk = setup
    with pytest.raises(ValueError, match="binding DDK identity"):
        build_fused_probes(tmp_path / "bad", binding, ge_root=ge, ddk_root=ddk, ddk_id="different")
    data = json.loads(binding.read_text())
    data["target"] = "Ascend310P3"
    binding.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="platform kirinx90"):
        build_fused_probes(tmp_path / "bad", binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    assert not (tmp_path / "bad").exists()


def test_no_compiled_cases_script_fails_without_invocation(setup, tmp_path):
    binding, _, _ = setup
    root = tmp_path / "source"
    build_fused_probes(root, binding)
    result = subprocess.run([str(root / "bundle/run-board.sh"), "/runner-that-does-not-exist",
                             str(tmp_path / "board"), "test-runtime", "test-ddk", "test-firmware"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "No COMPILED" in result.stderr
    assert "no operator invocation attempted" in (tmp_path / "board/run-error.txt").read_text()
    assert (root / "report.txt").is_file()
    assert not list((tmp_path / "board").glob("*.exitcode"))


def test_failure_inspection_reads_existing_middle_errors_and_never_runs_ddk(setup, tmp_path, monkeypatch):
    binding, _, ddk = setup
    root = tmp_path / "source"
    build_fused_probes(root, binding)
    log = root / "cases/gdn_prefill/compile.log"
    log.write_text("No such file or directory: optional.so\n" * 20 +
                   "Traceback (most recent call last):\n  wrapper.py:24\nModuleNotFoundError: missing_dependency\n" +
                   "compiler process died\n" * 20)
    source = ddk / "ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('DO NOT IMPORT THIS INSTALLED MODULE')\n"
                      "def chunk_gated_delta_rule(initial_state, output_final_state):\n    pass\n")
    before = {p: p.read_bytes() for folder in (root, ddk) for p in folder.rglob('*') if p.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("inspection must not execute any tool"))
    output = tmp_path / "inspection"
    result = inspect_fused_probes(root, ddk, output)
    assert result['new_compilation'] == result['device_execution'] == 'NOT_RUN'
    assert result['cases'][0]['compile_work_retained'] is False
    assert 'ModuleNotFoundError: missing_dependency' in (output / 'report.txt').read_text()
    assert len(result['installed_sources']) == 1
    assert (output / result['installed_sources'][0]['path']).read_bytes() == source.read_bytes()
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert result['cases'][1]['logs'][0]['status'] == 'UNREADABLE'
    for forbidden in (root / 'inspection', ddk / 'inspection'):
        with pytest.raises(ValueError, match='separate'):
            inspect_fused_probes(root, ddk, forbidden)
        assert not forbidden.exists()


def test_hash_bound_runtime_failures_preserve_stage_without_numerical_claim(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    _mock_tools(monkeypatch)
    root = tmp_path / "compiled"
    manifest = build_fused_probes(root, binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    board = tmp_path / "board"
    _fake_results(root, board, manifest)
    stages = ["LOAD_OR_SETUP", "INPUT_BINDING", "PROCESS", "PROCESS"]
    for case, stage in zip(manifest["cases"], stages):
        path = board / case["name"] / "result.json"
        result = json.loads(path.read_text())
        for key in ("runtime_version", "operator_calls", "inputs"):
            result.pop(key)
        result.update(status="RUNTIME_FAILED", failure_stage=stage, reason="injected runtime failure",
                      expected_runtime_version="FAKE TEST ONLY", outputs=[])
        if case["name"] == "conv_decode":
            result["model_sha256"] = "0" * 64
        path.write_text(json.dumps(result))
    rejected = compare_fused_probes(root, board, tmp_path / "rejected")
    assert all(case["result_validation"] == "FAILED" for case in rejected["cases"])
    output = tmp_path / "compared"
    result = compare_fused_probes(root, board, output, allow_host_fake=True)
    for case, stage, load, process in zip(result["cases"], stages,
                                        ("UNVERIFIED", "PASSED", "PASSED"),
                                        ("NOT_RUN", "NOT_RUN", "FAILED")):
        assert case["failure_stage"] == stage and case["runtime"] == "FAILED"
        assert case["load"] == load and case["process"] == process
        assert case["result_validation"] == "PASSED" and case["numerical"] == "NOT_EVALUATED"
        assert "diagnostics" not in case and case["result_sha256"]
    assert result["cases"][3]["result_validation"] == "FAILED"
    assert result["cases"][3]["load"] == result["cases"][3]["process"] == "UNVERIFIED"
    assert result["cases"][4]["process"] == "PASSED"
    report = (output / "report.txt").read_text()
    assert all("Native " + stage + " failed" in report for stage in stages[:3])
    assert (output / "report.json").is_file()

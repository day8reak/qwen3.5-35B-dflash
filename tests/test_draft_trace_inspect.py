"""Retained trace inspection is read only and never invokes target tools."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from qwen35_moe_dflash.draft_trace_inspect import inspect_retained_trace_precision


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def receipt(path, root):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def retained(tmp_path):
    job = tmp_path / "full-draft"
    root = job / "layer-trace"
    write_json(job / "result.json", {"artifact_kind": "qwen35-full-draft-job"})
    spec = [{"name": "draft_hidden", "shape": [1, 15, 2048], "dtype": "float16", "offset": 0, "elements": 30720}]
    write_json(root / "owner.json", {"artifact_kind": "qwen35-draft-layer-trace-job-owner", "job": str(job),
               "source_job_sha256": hashlib.sha256((job / "result.json").read_bytes()).hexdigest(),
               "trace_spec": spec, "code": {"draft_trace_job.py": "old code is deliberately accepted"}})
    air = root / "air"
    recipe = air / "draft/recipe/graph.recipe"
    recipe.parent.mkdir(parents=True)
    recipe.write_bytes(b"mock retained precision recipe; no serializer or target library loaded")
    (air / "draft.air").write_bytes(b"mock retained diagnostic AIR")
    exported = {"artifact_kind": "qwen35-cpro-draft-air", "status": "EXPORTED", "diagnostic_only": True,
                "trace_spec": spec, "graphs": [{"name": "draft", "air": "draft.air",
                "outputs": ["draft_top1", "draft_trace_values"],
                "files": [receipt(air / "draft.air", air)], "construction_files": [receipt(recipe, air)]}]}
    write_json(air / "manifest.json", exported)
    write_json(root / "completed-air.json", receipt(air / "manifest.json", air))
    compilation = root / "compilation"
    snapshot = compilation / "compile-work/inputs/draft.air"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes((air / "draft.air").read_bytes())
    omc = compilation / "compile-work/generated/model/model.omc"
    omc.parent.mkdir(parents=True)
    omc.write_bytes(b"mock small external OMC; weight sidecars are intentionally absent")
    compiled = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "model_compilation": "PASSED",
                "input_snapshots": [{"role": "air", **receipt(snapshot, compilation)}], "omc": receipt(omc, compilation)}
    write_json(compilation / "report.json", compiled)
    write_json(root / "completed-compilation.json", receipt(compilation / "report.json", compilation))
    precision = {"artifact_kind": "qwen35-draft-compiled-precision-verification", "contract": "cpro-fp32-cpu-v1",
                 "status": "FAILED", "required_nodes": 3, "matched_nodes": 1, "error_count": 2,
                 "omc": receipt(omc, omc.parent), "recipe": receipt(recipe, recipe.parent),
                 "nodes": [{"name": "matched", "source_type": "Mul", "status": "MATCHED", "errors": []},
                           {"name": "missing", "source_type": "Reshape", "status": "MISSING", "errors": [{"kind": "CPU_NODE_MISSING"}]},
                           {"name": "boundary", "source_type": "Mul", "status": "MISMATCH", "errors": [{"kind": "DESCRIPTOR_DIFFERS"}]}]}
    write_json(root / "precision-check.json", precision)
    return job


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_inspection_accepts_original_code_and_preserves_failed_result_without_writes(retained, monkeypatch):
    monkeypatch.delenv("AI_RUN_DIR", raising=False)
    before = snapshot(retained)
    report = inspect_retained_trace_precision(retained)
    assert report["status"] == "INSPECTED"
    assert report["compiled_precision"]["status"] == "FAILED"
    assert report["compiled_precision"]["required_nodes"] == 3
    assert report["failed_node_count"] == 2
    assert report["failed_nodes"][0] == {"name": "missing", "source_type": "Reshape", "status": "MISSING",
                                        "errors": [{"kind": "CPU_NODE_MISSING"}]}
    assert snapshot(retained) == before


@pytest.mark.parametrize("mutation", ["job", "compilation_receipt", "omc", "recipe", "precision_recipe", "precision_omc", "counts", "symlink"])
def test_inspection_rejects_changed_or_unbound_evidence(retained, mutation):
    root = retained / "layer-trace"
    if mutation == "job":
        (retained / "result.json").write_text('{"artifact_kind":"qwen35-full-draft-job", "changed":true}')
    elif mutation == "compilation_receipt":
        path = root / "compilation/report.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "omc":
        (root / "compilation/compile-work/generated/model/model.omc").write_bytes(b"changed")
    elif mutation == "recipe":
        (root / "air/draft/recipe/graph.recipe").write_bytes(b"changed")
    elif mutation == "symlink":
        path = root / "precision-check.json"
        moved = path.with_name("moved-precision.json")
        path.rename(moved)
        path.symlink_to(moved)
    else:
        path = root / "precision-check.json"
        report = json.loads(path.read_text())
        if mutation == "precision_recipe":
            report["recipe"]["sha256"] = "0" * 64
        elif mutation == "precision_omc":
            report["omc"]["sha256"] = "0" * 64
        else:
            report["matched_nodes"] = 3
        write_json(path, report)
    with pytest.raises(ValueError):
        inspect_retained_trace_precision(retained)


def cli():
    path = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/run_draft_trace.py"
    spec = importlib.util.spec_from_file_location("draft_trace_inspect_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_inspect_cli_prints_all_failures_without_importing_run_trace(retained, monkeypatch, capsys):
    import sys
    monkeypatch.delenv("AI_RUN_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "qwen35_moe_dflash.draft_trace_job", None)
    assert cli()(["--job", str(retained), "--inspect-only"]) == 0
    captured = capsys.readouterr().out
    assert "Precision mismatch:" in captured and '"source_type": "Reshape"' in captured
    assert '"failed_node_count": 2' in captured


def test_normal_cli_keeps_existing_run_trace_signature(tmp_path, monkeypatch):
    from qwen35_moe_dflash import draft_trace_job
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    calls = []
    def run(job, **kwargs):
        calls.append((job, kwargs))
        return {"status": "PREPARED"}
    monkeypatch.setattr(draft_trace_job, "run_trace", run)
    assert cli()(["--prepare-only"]) == 0
    assert calls[0][0] == tmp_path / "full-draft"
    assert "inspect_only" not in calls[0][1]
    assert "include_boundaries" not in calls[0][1]
    assert calls[0][1]["prepare_only"] is True


def test_inspect_mode_is_exclusive_with_compile(monkeypatch):
    with pytest.raises(SystemExit) as error:
        cli()(["--inspect-only", "--compile-only"])
    assert error.value.code == 2


def boundary_document():
    raw_desc = {"dtype": 1, "shape": {"dim": [1, 1, 64]}, "layout": "ND"}
    graphop = {"name": "SubGraph_2", "type": "GraphOp", "input": ["SubGraph_1:3"],
               "input_desc": [raw_desc], "output_desc": [raw_desc], "attr": [
                   {"key": "cl_name", "value": {"s": "CPUCL"}},
                   {"key": "graphop_task_offset", "value": {"i": 0}},
                   {"key": "graphop_task_size", "value": {"i": 200}}]}
    children = [{"name": "SubGraph_2_data_0", "type": "Data", "input": [], "output_desc": [raw_desc],
                 "attr": [{"key": "index", "value": {"i": 0}}]},
                {"name": "__draft_trace/concat_0/0", "type": "ConcatD", "input": ["SubGraph_2_data_0:0"],
                 "input_desc": [raw_desc], "output_desc": [raw_desc], "attr": []},
                {"name": "SubGraph_2_netoutput", "type": "NetOutput", "input": ["__draft_trace/concat_0/0:0"],
                 "input_desc": [raw_desc], "attr": [{"key": "_parent_node_index", "value": {"i": 4}}]}]
    return {"graph": [{"name": "draft", "op": [graphop]}]}, [{"owner": "SubGraph_2", "graph": {"name": "SubGraph_2", "op": children}}]


def test_boundary_summary_keeps_raw_graphop_and_cpu_io_without_inferring_mapping():
    from qwen35_moe_dflash.draft_trace_inspect import summarize_boundary_document
    model, cpu = boundary_document()
    result = summarize_boundary_document(model, cpu, ["__draft_trace/concat_0/0"])
    main = result["main_boundary_nodes"][0]
    assert main["input"] == ["SubGraph_1:3"]
    assert main["input_desc"] == model["graph"][0]["op"][0]["input_desc"]
    boundaries = result["cpu_subgraphs"][0]["boundary_nodes"]
    assert boundaries[0]["attr"] == [{"key": "index", "value": {"i": 0}}]
    assert boundaries[1]["input"] == ["__draft_trace/concat_0/0:0"]
    assert boundaries[1]["attr"] == [{"key": "_parent_node_index", "value": {"i": 4}}]
    failed = result["cpu_subgraphs"][0]["failed_nodes"]
    assert failed[0]["input"] == ["SubGraph_2_data_0:0"]
    assert "mapping" in result["scope"]


@pytest.mark.parametrize("mutation", ["wrong_owner", "duplicate_node", "oversize_ports"])
def test_boundary_summary_rejects_ambiguous_or_truncated_metadata(mutation):
    from qwen35_moe_dflash.draft_trace_inspect import summarize_boundary_document
    model, cpu = boundary_document()
    if mutation == "wrong_owner":
        cpu[0]["owner"] = "unknown"
    elif mutation == "duplicate_node":
        cpu[0]["graph"]["op"].append(cpu[0]["graph"]["op"][0])
    else:
        model["graph"][0]["op"][0]["input"] = ["x:0"] * 2049
    with pytest.raises(ValueError):
        summarize_boundary_document(model, cpu)


def test_optional_boundary_decode_uses_bound_artifacts_without_writing(retained, monkeypatch):
    from qwen35_moe_dflash import draft_trace_inspect
    calls = []
    def decode(read, root, compiled, precision, library):
        calls.append((root, compiled, precision, library))
        return {"mock_static_decode": True}
    monkeypatch.setattr(draft_trace_inspect, "_decode_boundaries", decode)
    before = snapshot(retained)
    assert "boundaries" not in inspect_retained_trace_precision(retained)
    assert not calls
    report = inspect_retained_trace_precision(retained, include_boundaries=True, schema_library="chosen-schema")
    assert report["boundaries"] == {"mock_static_decode": True}
    assert calls[0][0] == retained / "layer-trace" and calls[0][-1] == "chosen-schema"
    assert snapshot(retained) == before


def test_boundary_cli_option_requires_read_only_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    assert cli()(["--include-boundaries"]) == 1
    assert "requires --inspect-only" in capsys.readouterr().err

"""One read-only coordinator preserves failures and uses schemas explicitly."""
import copy
import json
from pathlib import Path
import runpy

import pytest

from qwen35_moe_dflash import target_compile_failure as job_module
from qwen35_moe_dflash import target_reshape_metadata as shape_module
from test_original_air_inspect import selected
from test_target_compile_failure import case as failure_case
from test_target_reshape_metadata import graph, schemas as shape_schemas


@pytest.fixture
def cli(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_compile_failure.py"
    module = runpy.run_path(str(script))
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("CPRO_DDK_ROOT", raising=False)
    monkeypatch.delenv("DDK_PATH", raising=False)
    report = {"status": "INSPECTED", "failure": {"status": "FAILED", "returncode": 1},
        "job": str(tmp_path / "job"), "ddk": {"root": "/declared/ddk"},
        "snapshot_air_path": str(tmp_path / "snapshot/export.air"),
        "compiler_graph_paths": [str(tmp_path / f"dump{i}.irpb") for i in range(4)],
        "snapshots": {"hash_validation": "RECORDED_HASH_WITH_CURRENT_STAT", "count": 1154},
        "options": {"dynamic_dims": "64,64,64,64,64;1,1,1,1,1", "input_shape": "omitted large contract"}}
    calls = []
    monkeypatch.setattr(job_module, "inspect_target_compile_failure", lambda path: copy.deepcopy(report))
    def shape(path, **kwargs):
        calls.append((path, kwargs))
        return {"status": "INSPECTED", "input": {"path": path}, "matching_node_count": 1,
            "matches": [{"graph_name": "branches1", "node_name": "Reshape", "node_type": "Reshape",
                "shape_constant": {"status": "DECODED_TINY_INTEGER_TENSOR", "decoded_values": [64, 2048]},
                "neighborhood": {"nodes": [{"name": "fc", "type": "DynamicQuant", "private_metadata": "full report only"}]}}]}
    monkeypatch.setattr(shape_module, "inspect_target_reshape_metadata", shape)
    return module, report, calls


def test_default_inspects_only_recorded_compiler_dumps_not_a_candidate_origin_schema(cli, tmp_path, capsys):
    module, report, calls = cli
    assert module["main"](["--job", report["job"], "--output", str(tmp_path / "report.json")]) == 0
    assert len(calls) == 2 and all(call[1] == {"ddk_root": "/declared/ddk", "node_name": "Reshape"} for call in calls)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["failure"]["status"] == "FAILED"
    assert saved["source_air_schema"] == "NOT_PARSED_WITHOUT_EXPLICIT_ORIGIN_SCHEMA"
    assert saved["snapshots"]["hash_validation"] == "RECORDED_HASH_WITH_CURRENT_STAT"
    shown = json.loads(capsys.readouterr().out)
    assert "input_shape" not in shown["options"]
    assert shown["reshape_metadata"][0]["matches"][0]["shape_constant"]["decoded_values"] == [64, 2048]
    assert "private_metadata" not in shown["reshape_metadata"][0]["matches"][0]["nearby_nodes"][0]
    assert "private_metadata" in saved["reshape_metadata"][0]["matches"][0]["neighborhood"]["nodes"][0]


def test_explicit_origin_schema_is_only_used_for_snapshot_and_explicit_ddk_for_dumps(cli, capsys):
    module, report, calls = cli
    assert module["main"](["--job", report["job"], "--torchair-pb2", "/origin/schema.py",
                            "--ddk-root", "/selected/ddk", "--node", "Reshape_7"]) == 0
    assert calls[0] == (report["snapshot_air_path"], {"torchair_pb2": "/origin/schema.py", "node_name": "Reshape_7"})
    assert len(calls) == 3 and calls[1][1] == {"ddk_root": "/selected/ddk", "node_name": "Reshape_7"}
    assert json.loads(capsys.readouterr().out)["failure"]["returncode"] == 1


def test_missing_ddk_reports_unknown_without_guessing_or_calling_graph_parser(cli, capsys):
    module, report, calls = cli
    report["ddk"] = {}
    assert module["main"](["--job", report["job"]]) == 0
    assert not calls
    assert json.loads(capsys.readouterr().out)["compiler_graph_schema"] == "UNAVAILABLE_WITHOUT_SELECTED_DDK_ROOT"


def test_existing_diagnostic_output_is_not_overwritten(cli, tmp_path, capsys):
    module, report, _ = cli
    path = tmp_path / "retained.json"
    path.write_bytes(b"retained diagnostic")
    assert module["main"](["--job", report["job"], "--output", str(path)]) == 1
    assert path.read_bytes() == b"retained diagnostic"
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_unavailable_failed_job_remains_blocked_not_a_compile_success(cli, capsys):
    module, report, calls = cli
    report.update(status="BLOCKED", compiler_graph_paths=[], snapshot_air_path=None)
    assert module["main"](["--job", report["job"]]) == 1
    assert not calls and json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_real_receipts_and_encoded_compiler_graph_interoperate_without_weight_reads(
        failure_case, shape_schemas, tmp_path, monkeypatch, capsys):
    import os
    import subprocess

    ddk, _ = shape_schemas
    for key in ("graph", "retained_graph"):
        failure_case[key].write_bytes(graph())
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("inspection must not execute tools"))
    original = os.open
    def guarded(path, *args, **kwargs):
        assert Path(path) not in {failure_case["weight"], failure_case["partial"], failure_case["omg"]}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", guarded)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_compile_failure.py"
    module = runpy.run_path(str(script))
    assert module["main"](["--job", str(failure_case["job"]), "--ddk-root", str(ddk),
                            "--output", str(tmp_path / "actual-diagnostic.json")]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["failure"]["returncode"] == 1 and shown["snapshots"]["count"] == 2
    assert shown["source_air_schema"] == "NOT_PARSED_WITHOUT_EXPLICIT_ORIGIN_SCHEMA"
    assert len(shown["reshape_metadata"]) == 2
    for result in shown["reshape_metadata"]:
        assert result["status"] == "INSPECTED"
        assert result["matches"][0]["shape_constant"]["decoded_values"] == [64, 2048]
        assert any(row["file_path"] == "./weights/layer0/in_proj_qkv"
                   for row in result["matches"][0]["file_constant_operands"])

"""Existing OMC inspection using a mock ELF/OMG; no real compiler or board."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from qwen35_moe_dflash import existing_omc_inspect as module


HELP = """--om Input model\n--json Output json\n--mode 0: generate model; 1: model to json\n"""


def _record(root, path, role):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stat": module._stat(path), "role": role}


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ddk = tmp_path / "ddk"
    omg = ddk / "tools/tools_omg/master/omg"
    omg.parent.mkdir(parents=True)
    omg.write_bytes(b"\x7fELF MOCK EXECUTABLE; subprocess is intercepted")
    omg.chmod(0o755)
    root = tmp_path / "compilation"
    generated = root / "compile-work/generated"
    (generated / "weights").mkdir(parents=True)
    primary = generated / "model.omc"
    primary.write_bytes(b"MOCK OMC")
    sidecar = generated / "weights/model.bin"
    sidecar.write_bytes(b"MOCK EXTERNAL WEIGHTS")
    records = [_record(root, primary, "primary_omc"), _record(root, sidecar, "external_data_candidate")]
    report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED",
              "ddk_root": str(ddk), "ddk_id": "MOCK", "omg_path": str(omg),
              "omg_sha256": hashlib.sha256(omg.read_bytes()).hexdigest(),
              "omc": records[0], "generated_artifacts": records}
    (root / "report.json").write_text(json.dumps(report))
    document = {"graph": [{"name": "original", "op": [
        {"name": "b", "type": "Data", "attr": {"index": {"i": 9}},
         "output_desc": [{"dtype": 333, "shape": {"dim": [1, 2048]}}]},
        {"name": "a", "type": "RefData", "output_desc": [{"dtype": 777}], "attr": {"index": {"i": 2}}},
        {"name": "weights", "type": "Const", "attr": {"value": {"t": {"data": "DO_NOT_SUMMARIZE_CONST"}}}},
        {"name": "projection", "type": "QuantBatchMatmul", "input": ["b:0", "weights:0"],
         "input_desc": [{"dtype": 13}], "attr": {"offset": {"i": -1}, "tensor": {"data": "SECRET_TENSOR_BLOB"}}},
        {"name": "state", "type": "ChunkGatedDeltaRule", "attr": {"chunk_size": {"i": 64}}},
        {"name": "output", "type": "NetOutput", "input": ["state:1", "projection:0"],
         "input_desc": [{"dtype": 1}, {"dtype": 2}]},
    ]}]}
    calls = []
    state = {"root": root, "omg": omg, "ddk": ddk, "report": report,
             "output": tmp_path / "inspection", "calls": calls, "document": document,
             "behavior": "ok", "help": HELP}

    def tool(argv, **kwargs):
        calls.append({"argv": argv, "cwd": kwargs["cwd"], "timeout": kwargs["timeout"]})
        assert kwargs["cwd"] != generated
        assert root not in kwargs["cwd"].parents
        if argv[1:] == ["--help"]:
            kwargs["stdout"].write(state["help"].encode())
            return subprocess.CompletedProcess(argv, 0)
        assert argv[1:] == ["--mode=1", "--om=input.omc", "--json=model.json"]
        work = kwargs["cwd"]
        assert (work / "input.omc").read_bytes() == b"MOCK OMC"
        assert (work / "model.omc").is_symlink()
        assert (work / "weights/model.bin").read_bytes() == b"MOCK EXTERNAL WEIGHTS"
        behavior = state["behavior"]
        if behavior == "fail":
            return subprocess.CompletedProcess(argv, 7)
        if behavior == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if behavior == "absent":
            return subprocess.CompletedProcess(argv, 0)
        if behavior == "oversized":
            with (work / "model.json").open("wb") as stream:
                stream.seek(module.MAX_JSON_BYTES)
                stream.write(b"x")
        elif behavior == "symlink":
            (work / "model.json").symlink_to(generated / "model.omc")
        elif behavior == "malformed":
            (work / "model.json").write_text("{broken")
        else:
            (work / "model.json").write_text(json.dumps(state["document"]))
        kwargs["stdout"].write(b"Mock conversion only\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", tool)
    return state


def _inspect(case, **kwargs):
    return module.inspect_existing_omc(case["root"], case["output"], **kwargs)


def test_selected_omg_help_only_conversion_and_raw_metadata_without_abi_assumptions(case, monkeypatch):
    before = {str(p.relative_to(case["root"])): p.read_bytes() for p in case["root"].rglob("*") if p.is_file()}
    real_hash = module.sha256_file
    hashed = []
    def track_hash(path):
        hashed.append(Path(path))
        assert Path(path) == case["omg"]  # Never rehash huge OMC/weight payloads.
        return real_hash(path)
    monkeypatch.setattr(module, "sha256_file", track_hash)
    report = _inspect(case)
    assert report["status"] == "INSPECTED"
    assert report["compilation_status"] == "COMPILED"
    assert report["runtime_abi"] == "UNVERIFIED" and report["device_execution"] == "NOT_RUN"
    assert hashed == [case["omg"]]
    assert [row["argv"][1:] for row in case["calls"]] == [
        ["--help"], ["--mode=1", "--om=input.omc", "--json=model.json"]]
    assert [row["timeout"] for row in case["calls"]] == [30, 120]
    data = report["inspection"]
    assert data["op_type_counts"] == {"Data": 1, "RefData": 1, "Const": 1,
                                      "QuantBatchMatmul": 1, "ChunkGatedDeltaRule": 1, "NetOutput": 1}
    by_name = {row["raw_node"]["name"]: row for row in data["io_nodes"]}
    assert by_name["b"]["raw_node"]["output_desc"][0]["dtype"] == 333
    assert by_name["a"]["raw_node"]["attr"]["index"]["i"] == 2
    assert by_name["output"]["raw_node"]["input"] == ["state:1", "projection:0"]
    assert data["input_order"] == "NOT_INFERRED" and data["dtype_enums"] == "NOT_DECODED"
    serialized = json.dumps(report)
    assert "DO_NOT_SUMMARIZE_CONST" not in serialized and "SECRET_TENSOR_BLOB" not in serialized
    assert "tensor_payload_omitted" in serialized
    assert report["source_preservation"]["status"] == "UNCHANGED_BY_STAT"
    assert before == {str(p.relative_to(case["root"])): p.read_bytes() for p in case["root"].rglob("*") if p.is_file()}
    assert not (case["output"] / "work/input.omc").exists()
    assert not (case["output"] / "work/weights/model.bin").exists()
    assert json.loads((case["output"] / "report.json").read_text()) == report


@pytest.mark.parametrize("behavior", ["fail", "timeout", "absent", "oversized", "symlink", "malformed"])
def test_unavailable_inspection_never_changes_compilation_success(case, behavior):
    case["behavior"] = behavior
    result = _inspect(case, timeout_seconds=17)
    assert result["status"] == "UNAVAILABLE"
    assert result["compilation_status"] == "COMPILED" and result["error"]
    assert json.loads((case["root"] / "report.json").read_text())["status"] == "COMPILED"
    assert case["calls"][-1]["timeout"] == 17
    if behavior == "oversized":
        assert result["json"]["bytes"] > module.MAX_JSON_BYTES
        assert "sha256" not in result["json"] and "inspection" not in result


def test_help_without_exact_mode_one_support_skips_conversion(case):
    case["help"] = "--mode 0: generate; --om input; --json output"
    result = _inspect(case)
    assert result["status"] == "UNAVAILABLE"
    assert result["stage"] == "conversion-help"
    assert len(case["calls"]) == 1


@pytest.mark.parametrize("change", ["hash", "artifact", "non_elf", "outside_ddk"])
def test_identity_changes_are_retained_as_unavailable_before_invocation(case, change):
    if change == "hash":
        case["report"]["omg_sha256"] = "0" * 64
    elif change == "artifact":
        path = case["root"] / case["report"]["omc"]["path"]
        path.write_bytes(b"DIFFERENT SIZE")
    elif change == "non_elf":
        case["omg"].write_text("#!/bin/sh\nexit 0\n")
        case["report"]["omg_sha256"] = hashlib.sha256(case["omg"].read_bytes()).hexdigest()
    else:
        case["report"]["omg_path"] = "/usr/bin/env"
    (case["root"] / "report.json").write_text(json.dumps(case["report"]))
    result = _inspect(case)
    assert result["status"] == "UNAVAILABLE" and result["error"]
    assert case["calls"] == []


def test_explicit_recorded_elf_inside_same_ddk_is_allowed(case):
    selected = case["ddk"] / "selected/omg"
    selected.parent.mkdir()
    selected.write_bytes(case["omg"].read_bytes())
    selected.chmod(0o755)
    case["report"]["omg_path"] = str(selected)
    (case["root"] / "report.json").write_text(json.dumps(case["report"]))
    result = _inspect(case)
    assert result["status"] == "INSPECTED"
    assert all(row["argv"][0] == str(selected) for row in case["calls"])


def test_bounded_node_metadata_does_not_limit_all_type_counts(monkeypatch):
    monkeypatch.setattr(module, "MAX_IO_NODES", 1)
    monkeypatch.setattr(module, "MAX_SELECTED_NODES", 1)
    document = {"graph": [{"op": [
        {"name": "first", "type": "Data"}, {"name": "second", "type": "Data"},
        {"name": "q1", "type": "DynamicQuant"}, {"name": "q2", "type": "DynamicQuant"},
        {"name": "body", "type": "GraphOp", "attr": {"g": {"op": [
            {"name": "inner", "type": "NetOutput", "input_desc": [{"dtype": 99}]}]}}}
    ]}]}
    result = module.summarize_json(document)
    assert result["op_count"] == 6 and result["scan_complete"] is True
    assert result["io_nodes_omitted"] == 2 and result["selected_nodes_omitted"] == 1
    assert result["op_type_counts"]["DynamicQuant"] == 2


def test_existing_air_report_precondition_and_output_isolation(case):
    with pytest.raises(ValueError, match="outside"):
        module.inspect_existing_omc(case["root"], case["root"] / "inspection")
    case["report"]["status"] = "FAILED"
    (case["root"] / "report.json").write_text(json.dumps(case["report"]))
    with pytest.raises(ValueError, match="COMPILED"):
        _inspect(case)
    assert not case["output"].exists() and not case["calls"]


@pytest.mark.parametrize("document", [{"name": "unscoped", "type": "Data"},
                                       {"graph": [{"op": [{"type": "Data", "attr": {"bad": float("nan")}}]}]}])
def test_unrecognized_schema_or_nonfinite_json_is_unavailable(case, document):
    case["document"] = document
    result = _inspect(case)
    assert result["status"] == "UNAVAILABLE" and result["stage"] == "json-summary"
    assert (case["output"] / "report.json").is_file()

"""Explicit CPU boundary paths and receipt identities without vendor execution."""
from copy import deepcopy
import hashlib
import json
import subprocess

import pytest

from qwen35_moe_dflash.draft_output_provenance import inspect_draft_output_provenance, inspect_output_document
from test_draft_ir_inspect import _attr, _blob, _desc, _int, _library
from test_draft_precision_omc import _container


def _spec():
    return [{"name": name, "source": name, "shape": [1, 2], "dtype": dtype, "output_index": index}
            for index, (name, dtype) in enumerate((("draft_top1", "int32"), ("draft_hidden", "float16")))]


def _node(name, kind, outputs, inputs=(), refs=()):
    desc = lambda dtype: {"dtype": dtype, "shape": {"dim": [1, 2]}, "layout": "ND"}
    row = {"name": name, "type": kind, "output_desc": [desc(d) for d in outputs],
           "input_desc": [desc(d) for d in inputs]}
    if refs:
        row["input"] = list(refs)
    return row


@pytest.fixture
def document():
    source = _node("draft_hidden", "Add", [7])
    source["output_desc"][0]["attr"] = [{"key": "raw_reuse", "value": {"i": 17}}]
    cpu = {"owner": "cpu0", "cl_name": "CPUCL", "graph": {"name": "cpu0", "op": [source,
        _node("cast", "Cast", [7], [7], ["draft_hidden:0"]),
        _node("net", "NetOutput", [7], [7], ["cast:0"])]}}
    main = {"name": "main", "op": [_node("opaque", "GraphOp", [3]), _node("cpu0", "GraphOp", [7]),
        _node("net", "NetOutput", [], [3, 7], ["opaque:0", "cpu0:0"])]}
    return main, [cpu], _spec(), {"dtype": {"values": {"INT32": 3, "HALF": 7}}}


def test_cpu_cast_path_and_consumers_keep_raw_descriptor_and_do_not_decode_npu(document):
    result = inspect_output_document(*document)
    assert result["outputs"][0]["resolution"] == "OPAQUE_NON_CPU_TASK"
    hidden = result["outputs"][1]
    assert hidden["resolution"] == "EXACT_CPU_NAME_ON_PATH"
    assert [step["node"]["name"] for step in hidden["path"]] == ["net", "cast", "draft_hidden"]
    found = hidden["source_occurrences"][0]
    assert found["node"]["selected_output_descriptor"]["attr"][0]["key"] == "raw_reuse"
    assert found["consumers"]["edges"][0]["name"] == "cast"
    assert "not value-equivalence" in result["view_path_scope"]


def test_cpu_data_follows_explicit_parent_graphop_port(document):
    main, cpu, spec, enums = document
    cpu[0]["graph"]["op"][0] = _node("data0", "Data", [7], [7])
    cpu[0]["graph"]["op"][1]["input"] = ["data0:0"]
    main["op"][1].update(input=["upstream:0"], input_desc=[{"dtype": 7, "shape": {"dim": [1, 2]}, "layout": "ND"}])
    main["op"].append(_node("upstream", "GraphOp", [7]))
    upstream = {"owner": "upstream", "cl_name": "CPUCL", "graph": {"name": "upstream", "op": [
        _node("draft_hidden", "Add", [7]), _node("out", "NetOutput", [7], [7], ["draft_hidden:0"])]}}
    hidden = inspect_output_document(main, cpu + [upstream], spec, enums)["outputs"][1]
    assert hidden["resolution"] == "EXACT_CPU_NAME_ON_PATH"
    assert [step["owner"] for step in hidden["path"]] == ["cpu0", "cpu0", "cpu0", "upstream", "upstream"]


def test_absent_source_and_bad_cpu_interface_are_not_inferred_as_elimination(document):
    main, cpu, spec, enums = document
    cpu[0]["graph"]["op"][0]["name"] = "other"
    cpu[0]["graph"]["op"][1]["input"] = ["other:0"]
    hidden = inspect_output_document(main, cpu, spec, enums)["outputs"][1]
    assert hidden["resolution"] == "DIFFERENT_EXPLICIT_PRODUCER" and hidden["source_occurrences"] == []
    main["op"][1]["output_desc"].append(deepcopy(main["op"][1]["output_desc"][0]))
    assert inspect_output_document(main, cpu, spec, enums)["outputs"][1]["resolution"] == "UNRESOLVED_CPU_INTERFACE_COUNT"


def test_ambiguous_nodes_and_output_cardinality_are_rejected(document):
    main, cpu, spec, enums = document
    main["op"].append(deepcopy(main["op"][0]))
    with pytest.raises(ValueError, match="ambiguous"):
        inspect_output_document(main, cpu, spec, enums)
    main["op"].pop()
    main["op"][2]["input_desc"].pop()
    with pytest.raises(ValueError, match="count"):
        inspect_output_document(main, cpu, spec, enums)


@pytest.fixture
def retained(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("inspection must not execute tools"))
    import test_draft_ir_inspect as fixture
    original = fixture._schema
    def extended():
        raw = original()
        enum = _blob(1, "DataType") + b"".join(_blob(2, _blob(1, n) + _int(2, v)) for n, v in [("UNDEFINED", 0), ("FLOAT", 1), ("HALF", 7)])
        return raw.replace(_blob(5, enum), _blob(5, enum + _blob(2, _blob(1, "INT32") + _int(2, 3))))
    monkeypatch.setattr(fixture, "_schema", extended)
    library = tmp_path / "schema.so"
    _library(library)
    root = tmp_path / "direct-trace"
    root.mkdir()
    write = lambda path, doc: path.write_text(json.dumps(doc))
    record = lambda path: {"path": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    spec = _spec()
    write(root / "owner.json", {"artifact_kind": "qwen35-draft-direct-trace-job-owner", "trace_spec": spec})
    descs = [_desc(3, [1, 2]), _desc(7, [1, 2])]
    node = lambda name, kind, extra: _blob(1, name) + _blob(2, kind) + extra
    cpu = _blob(1, "cpu0") + b"".join(_blob(6, node(s["source"], "Add", _blob(34, d))) for s, d in zip(spec, descs))
    cpu += _blob(6, node("cpuout", "NetOutput", b"".join(_blob(5, s["source"] + ":0") + _blob(33, d) + _blob(34, d) for s, d in zip(spec, descs))))
    owner = node("cpu0", "GraphOp", _attr("cl_name", _blob(2, "CPUCL")) + _attr("graphop_task_offset", _int(3, 0)) + _attr("graphop_task_size", _int(3, len(cpu))) + b"".join(_blob(34, d) for d in descs))
    net = node("mainout", "NetOutput", b"".join(_blob(5, "cpu0:" + str(i)) + _blob(33, d) for i, d in enumerate(descs)))
    omc = _container(_blob(7, _blob(1, "main") + _blob(6, owner) + _blob(6, net)), cpu)
    for stage in ("fixture", "air", "compilation"):
        (root / stage).mkdir()
    omc_path = root / "compilation/model.omc"
    omc_path.write_bytes(omc)
    air_id = {"path": "draft.air", "bytes": 100, "sha256": "a" * 64}
    documents = {"fixture": {"direct_spec": spec}, "air": {"direct_spec": spec, "graphs": [{"name": "draft", "air": "draft.air", "files": [air_id]}]},
                 "compilation": {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "omc": record(omc_path), "input_snapshots": [{**air_id, "role": "air"}]}}
    for stage, doc in documents.items():
        path = root / stage / ("manifest.json" if stage == "air" else "report.json")
        write(path, doc)
        write(root / ("completed-" + stage + ".json"), record(path))
    binding = {"status": "PASSED", "declared_spec": spec, "observed_spec": spec, "omc": record(omc_path),
               "schema": {"library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}}
    write(root / "compiled-output-abi.json", binding)
    return root


def test_retained_readonly_inspection_binds_schema_omc_and_source_snapshot(retained):
    before = {p: p.read_bytes() for p in retained.rglob("*") if p.is_file()}
    result = inspect_draft_output_provenance(retained)
    assert result["status"] == "INSPECTED", result
    assert result["resolution_counts"] == {"EXACT_CPU_NAME_ON_PATH": 2}
    assert before == {p: p.read_bytes() for p in retained.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["receipt", "snapshot", "omc", "schema", "spec"])
def test_changed_retained_evidence_never_returns_producer_findings(retained, damage):
    if damage == "receipt":
        path = retained / "fixture/report.json"
        path.write_text(path.read_text() + " ")
    elif damage == "snapshot":
        path = retained / "compilation/report.json"
        doc = json.loads(path.read_text())
        doc["input_snapshots"][0]["sha256"] = "b" * 64
        path.write_text(json.dumps(doc))
        receipt = {"path": "report.json", "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (retained / "completed-compilation.json").write_text(json.dumps(receipt))
    else:
        path = retained / "compiled-output-abi.json"
        doc = json.loads(path.read_text())
        if damage == "omc":
            doc["omc"]["sha256"] = "b" * 64
        elif damage == "schema":
            doc["schema"]["library_sha256"] = "b" * 64
        else:
            doc["declared_spec"][1]["source"] = "other"
        path.write_text(json.dumps(doc))
    result = inspect_draft_output_provenance(retained)
    assert result["status"] == "UNAVAILABLE" and "outputs" not in result

def test_compact_cli_keeps_opaque_descriptors_consumers_and_math_result(retained, capsys):
    import runpy
    from pathlib import Path
    script = Path(__file__).parents[1] / "targets/cpro/scripts/inspect_draft_outputs.py"
    cli = runpy.run_path(str(script))
    assert cli["main"](["--job", str(retained), "--math"]) == 0
    result = json.loads(capsys.readouterr().out)
    row = result["outputs"][1]
    assert row["main_output_descriptor"]["dtype"] == 7
    assert row["netoutput_input_descriptor"]["shape"] == {"dim": [1, 2]}
    assert row["main_consumers"]["edges"][0]["name"] == "mainout"
    assert row["path"][0]["descriptor"]["dtype"] == 7
    assert result["output_math"]["status"] == "UNAVAILABLE"


def test_task_bounds_are_checked_before_cpu_graph_decode(retained):
    from qwen35_moe_dflash.omc_container import read_container
    from qwen35_moe_dflash.omc_wire import load_schema
    path = retained / "compilation/model.omc"
    raw = path.read_bytes()
    binding_path = retained / "compiled-output-abi.json"
    binding = json.loads(binding_path.read_text())
    schema = load_schema(binding["schema"]["library"])
    parts = read_container(path)["partitions"]
    model = schema.decode(memoryview(raw)[parts[0]["offset"]:parts[0]["offset"] + parts[0]["bytes"]], "ModelDef")
    attrs = model["graph"][0]["op"][0]["attr"]
    size = next(a["value"]["i"] for a in attrs if a["key"] == "graphop_task_size")
    updated = raw.replace(_attr("graphop_task_size", _int(3, size)), _attr("graphop_task_size", _int(3, size + 1)))
    assert len(updated) == len(raw) and updated != raw
    path.write_bytes(updated)
    omc = {"path": path.name, "bytes": len(updated), "sha256": hashlib.sha256(updated).hexdigest()}
    binding["omc"] = omc
    binding_path.write_text(json.dumps(binding))
    report_path = retained / "compilation/report.json"
    report = json.loads(report_path.read_text())
    report["omc"] = omc
    report_path.write_text(json.dumps(report))
    receipt = {"path": report_path.name, "bytes": report_path.stat().st_size, "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}
    (retained / "completed-compilation.json").write_text(json.dumps(receipt))
    result = inspect_draft_output_provenance(retained)
    assert result["status"] == "UNAVAILABLE" and "ownership outside bounds" in result["reason"]

"""CPU-payload inspection contracts; synthetic OMC and schema, no vendor runs."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import pytest

from qwen35_moe_dflash import omc_cpu_inspect as module
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.omc_inspect import inspect_json


def _bytes(document):
    return json.dumps(document).encode()


def _container(model, tasks):
    main = _bytes(model)
    table = struct.pack("<I", 2) + struct.pack("<III", 0, 999, len(main)) + struct.pack("<III", 2, 111, len(tasks))
    header = bytearray(256)
    header[:4] = b"IMOD"
    struct.pack_into("<I", header, 0x4C, len(table) + len(main) + len(tasks))
    return header + table + main + tasks


def _descriptor(dtype=2):
    return {"dtype": dtype, "shape": {"dim": [1, 8]}, "layout": "ND"}


def _cpu_graph(name="SubGraph_0"):
    return {"name": name, "op": [
        {"name": "producer", "type": "Data", "output_desc": [_descriptor()]},
        {"name": "insert_cast_4", "type": "CastT", "input": ["producer:0"],
         "input_desc": [_descriptor()], "output_desc": [_descriptor(1)],
         "attr": [{"key": "src_dtype", "value": {"i": 0}}, {"key": "dst_dtype", "value": {"i": 1}}]},
    ]}


def _owner(name, body, offset=0, cl="CPUCL"):
    return {"name": name, "type": "GraphOp", "input": [], "attr": [
        {"key": "cl_name", "value": {"s": cl}},
        {"key": "graphop_task_offset", "value": {"i": offset}},
        {"key": "graphop_task_size", "value": {"i": len(body)}},
    ]}


def _set_task(node, field, value):
    next(entry for entry in node["attr"] if entry["key"] == field)["value"]["i"] = value


def _write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(document))


def _write_pair(case):
    manifest = {"artifact_kind": "qwen35-cpro-omc-pair", "status": "COMPILED",
                "ddk_id": "fixture-ddk", "omg_sha256": sha256_file(case["binary"]), "graphs": []}
    prior = {"artifact_kind": "qwen35-omc-cast-inspection", "ddk_id": "fixture-ddk",
             "omg_sha256": sha256_file(case["binary"]), "graphs": []}
    for name in ("target", "draft"):
        source = case["bundle"] / f"{name}.omc"
        source.write_bytes(_container(case["models"][name], case["tasks"][name]))
        converted = case["prior"] / name / "model.json"
        _write_json(converted, case["json"][name])
        manifest["graphs"].append({"name": name, "omc": source.name,
                                  "bytes": source.stat().st_size, "sha256": sha256_file(source)})
        prior["graphs"].append({"name": name, "source_sha256": sha256_file(source),
                               "conversion": {"status": "CONVERTED", "json_bytes": converted.stat().st_size,
                                              "json_sha256": sha256_file(converted)}})
    _write_json(case["bundle"] / "manifest.json", manifest)
    prior["source_manifest_sha256"] = sha256_file(case["bundle"] / "manifest.json")
    _write_json(case["prior"] / "report.json", prior)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    binary = tmp_path / "ddk/tools/tools_omg/master/omg"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"Test fixture, never executable")
    library = binary.parent / "lib64/libai_fmk_graph_proto_host.so"
    library.parent.mkdir()
    library.write_bytes(b"Test schema source, not an ELF library")
    data = {"bundle": tmp_path / "bundle", "prior": tmp_path / "prior", "output": tmp_path / "inspection",
            "ddk": tmp_path / "ddk", "binary": binary, "library": library,
            "models": {}, "tasks": {}, "json": {}, "calls": []}
    data["bundle"].mkdir()
    for name in ("target", "draft"):
        body = _bytes(_cpu_graph())
        data["tasks"][name] = body
        data["models"][name] = {"graph": [{"name": name, "op": [_owner("SubGraph_0", body)]}]}
        data["json"][name] = copy.deepcopy(data["models"][name])
    _write_pair(data)

    class FakeSchema:
        evidence = {"library": str(library), "library_sha256": sha256_file(library)}
        enums = {}

        def decode(self, raw, kind):
            data["calls"].append((kind, raw))
            return json.loads(raw)

    data["schema"] = FakeSchema()
    monkeypatch.setattr(module, "load_schema", lambda path: data["schema"])

    def no_subprocess(*args, **kwargs):
        pytest.fail("offline CPU inspection must not invoke an executable")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    return data


def _inspect(case, **kwargs):
    return module.inspect_omc_cpu(case["bundle"], case["prior"], case["output"], case["ddk"],
                                  ddk_id="fixture-ddk", **kwargs)


def test_decodes_both_existing_cpu_slices_with_bound_evidence_and_no_execution(case):
    before = {str(path): sha256_file(path) for folder in (case["bundle"], case["prior"])
              for path in folder.rglob("*") if path.is_file()}
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert result["device_execution"] == "NOT_RUN"
    assert result["runtime_placement"] == "UNVERIFIED"
    assert result["source_manifest_sha256"] == before[str(case["bundle"] / "manifest.json")]
    assert result["prior_report_sha256"] == before[str(case["prior"] / "report.json")]
    for graph in result["graphs"]:
        assert graph["main_json_agreement"] == "MATCHED"
        cpu = graph["cpu_subgraphs"][0]
        assert cpu["status"] == "DECODED"
        assert cpu["serialized_graph_name"] == cpu["graphop_name"] == "SubGraph_0"
        assert cpu["slice_sha256"] == hashlib.sha256(case["tasks"][graph["name"]]).hexdigest()
        assert sha256_file(Path(cpu["decoded_json"]["path"])) == cpu["decoded_json"]["sha256"]
        cast = cpu["inspection"]["casts"][0]
        assert cast["raw_node"]["input_desc"][0]["dtype"] == 2
        assert cast["raw_node"]["attr"][0]["value"]["i"] == 0
        assert cast["same_graph_port_candidates"][0]["raw_reference"] == "producer:0"
    assert all(sha256_file(Path(path)) == digest for path, digest in before.items())
    assert json.loads((case["output"] / "report.json").read_text()) == result


@pytest.mark.parametrize("fault", ["prior-binding", "omc-bytes", "json-bytes", "ddk-id", "omg-bytes",
                                   "prior-ddk-id", "prior-omg-hash", "prior-source-hash", "conversion-status"])
def test_identity_failures_reject_before_output_or_schema_decode(case, fault):
    prior_path = case["prior"] / "report.json"
    prior = json.loads(prior_path.read_text())
    if fault == "prior-binding":
        prior["source_manifest_sha256"] = "0" * 64
    elif fault == "omc-bytes":
        with (case["bundle"] / "target.omc").open("ab") as stream:
            stream.write(b"changed")
    elif fault == "json-bytes":
        with (case["prior"] / "target/model.json").open("ab") as stream:
            stream.write(b" ")
    elif fault == "ddk-id":
        manifest_path = case["bundle"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["ddk_id"] = "different-ddk"
        _write_json(manifest_path, manifest)
        prior["source_manifest_sha256"] = sha256_file(manifest_path)
    elif fault == "omg-bytes":
        case["binary"].write_bytes(b"different binary")
    elif fault == "prior-ddk-id":
        prior["ddk_id"] = "different-ddk"
    elif fault == "prior-omg-hash":
        prior["omg_sha256"] = "0" * 64
    elif fault == "prior-source-hash":
        prior["graphs"][0]["source_sha256"] = "0" * 64
    elif fault == "conversion-status":
        prior["graphs"][0]["conversion"]["status"] = "FAILED"
    _write_json(prior_path, prior)
    with pytest.raises(ValueError):
        _inspect(case)
    assert not case["output"].exists()
    assert not case["calls"]


def test_main_structure_mismatch_prevents_cpu_decode_but_draft_continues(case):
    case["json"]["target"]["graph"][0]["op"][0]["name"] = "wrong-owner"
    _write_pair(case)
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    assert result["graphs"][0]["status"] == "UNSUPPORTED_LAYOUT_OR_SCHEMA"
    assert "CPU slices not interpreted" in result["graphs"][0]["error"]
    assert result["graphs"][1]["status"] == "CPU_GRAPHS_DECODED"
    assert [kind for kind, _ in case["calls"]] == ["ModelDef", "ModelDef", "GraphDef"]
    assert not list((case["output"] / "target").glob("cpu-*.json"))


@pytest.mark.parametrize("dimension", [1.9, True])
def test_nonintegral_dimensions_are_rejected_even_when_both_main_views_agree(case, dimension):
    node = case["models"]["target"]["graph"][0]["op"][0]
    node["input_desc"] = [_descriptor()]
    node["input_desc"][0]["shape"]["dim"][0] = dimension
    case["json"]["target"] = copy.deepcopy(case["models"]["target"])
    _write_pair(case)
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    assert "shape dimensions must be exact int64" in result["graphs"][0]["error"]
    assert result["graphs"][1]["status"] == "CPU_GRAPHS_DECODED"
    assert [kind for kind, _ in case["calls"]] == ["ModelDef", "ModelDef", "GraphDef"]


@pytest.mark.parametrize("fault", ["graph-null", "graph-entry-null", "op-null", "op-entry-null",
                                   "descriptors-null", "descriptor-null", "shape-null", "dimensions-null",
                                   "inputs-null", "input-nonstring"])
def test_malformed_main_structure_is_partial_and_draft_continues(case, fault):
    model = case["models"]["target"]
    graph = model["graph"][0]
    node = graph["op"][0]
    if fault == "graph-null":
        model["graph"] = None
    elif fault == "graph-entry-null":
        model["graph"] = [None]
    elif fault == "op-null":
        graph["op"] = None
    elif fault == "op-entry-null":
        graph["op"] = [None]
    elif fault == "descriptors-null":
        node["input_desc"] = None
    elif fault == "descriptor-null":
        node["input_desc"] = [None]
    elif fault == "shape-null":
        node["input_desc"] = [{"dtype": 2, "shape": None}]
    elif fault == "dimensions-null":
        node["input_desc"] = [{"dtype": 2, "shape": {"dim": None}}]
    elif fault == "inputs-null":
        node["input"] = None
    elif fault == "input-nonstring":
        node["input"] = [False]
    case["json"]["target"] = copy.deepcopy(model)
    _write_pair(case)
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    assert result["graphs"][0]["status"] == "UNSUPPORTED_LAYOUT_OR_SCHEMA"
    assert result["graphs"][0]["error"].startswith("ValueError:")
    assert result["graphs"][1]["status"] == "CPU_GRAPHS_DECODED"
    assert [kind for kind, _ in case["calls"]] == ["ModelDef", "ModelDef", "GraphDef"]


@pytest.mark.parametrize("fault", ["outside", "zero-size", "overlap"])
def test_all_owner_bounds_are_checked_before_any_cpu_slice_is_decoded(case, fault):
    owners = case["models"]["target"]["graph"][0]["op"]
    second = _owner("SubGraph_1", b"x", offset=len(case["tasks"]["target"]) - 1)
    if fault == "outside":
        _set_task(second, "graphop_task_offset", len(case["tasks"]["target"]) + 1)
    elif fault == "zero-size":
        _set_task(second, "graphop_task_size", 0)
    owners.append(second)
    case["json"]["target"] = copy.deepcopy(case["models"]["target"])
    _write_pair(case)
    result = _inspect(case)
    assert result["graphs"][0]["status"] == "UNSUPPORTED_LAYOUT_OR_SCHEMA"
    assert result["graphs"][1]["status"] == "CPU_GRAPHS_DECODED"
    assert [kind for kind, _ in case["calls"]] == ["ModelDef", "ModelDef", "GraphDef"]


def test_npu_payload_is_retained_as_opaque_owner_without_graph_decode(case):
    cpu = case["tasks"]["target"]
    opaque = b"Not a GraphDef; NPU payload stays opaque"
    case["tasks"]["target"] = cpu + opaque
    case["models"]["target"]["graph"][0]["op"].append(_owner("SubGraph_1", opaque, len(cpu), "NPUCL"))
    case["json"]["target"] = copy.deepcopy(case["models"]["target"])
    _write_pair(case)
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    target = result["graphs"][0]
    assert target["other_subgraphs"][0]["task"]["cl_name"] == "NPUCL"
    assert len(target["cpu_subgraphs"]) == 1
    assert all(raw != opaque for _, raw in case["calls"])


@pytest.mark.parametrize("fault", ["name", "duplicate-node", "malformed"])
def test_bad_cpu_slice_is_explicit_and_later_slices_and_draft_continue(case, fault):
    graph = _cpu_graph("wrong-name" if fault == "name" else "SubGraph_0")
    if fault == "duplicate-node":
        graph["op"].append(copy.deepcopy(graph["op"][0]))
    bad = b"not json" if fault == "malformed" else _bytes(graph)
    good = _bytes(_cpu_graph("SubGraph_1"))
    case["tasks"]["target"] = bad + good
    case["models"]["target"]["graph"][0]["op"] = [
        _owner("SubGraph_0", bad), _owner("SubGraph_1", good, len(bad))]
    case["json"]["target"] = copy.deepcopy(case["models"]["target"])
    _write_pair(case)
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    assert [row["status"] for row in result["graphs"][0]["cpu_subgraphs"]] == ["UNSUPPORTED_CPU_SLICE", "DECODED"]
    assert result["graphs"][1]["status"] == "CPU_GRAPHS_DECODED"
    assert not (case["output"] / "target/cpu-000.json").exists()
    assert (case["output"] / "target/cpu-001.json").is_file()


def test_schema_unavailable_retains_report_for_both_graphs_without_decode(case, monkeypatch):
    def unavailable(path):
        raise ValueError("no supported descriptor symbol")
    monkeypatch.setattr(module, "load_schema", unavailable)
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    assert "no supported descriptor symbol" in result["schema_error"]
    assert [row["status"] for row in result["graphs"]] == ["SCHEMA_UNAVAILABLE", "SCHEMA_UNAVAILABLE"]
    assert not case["calls"]
    assert (case["output"] / "report.txt").is_file()


def test_schema_library_change_is_rejected_before_result_report(case, monkeypatch):
    original = case["schema"].decode
    def changed(raw, kind):
        case["library"].write_bytes(b"changed schema")
        return original(raw, kind)
    monkeypatch.setattr(case["schema"], "decode", changed)
    with pytest.raises(ValueError, match="schema library changed"):
        _inspect(case)
    assert not (case["output"] / "report.json").exists()


def test_prior_json_budget_rejects_before_creating_output(case):
    with pytest.raises(ValueError, match="JSON exceeds"):
        _inspect(case, max_partition_bytes=1)
    assert not case["output"].exists()


def test_port_candidates_require_exact_unique_node_and_existing_port():
    graph = {"op": [
        {"name": "p", "type": "Data", "output_desc": [_descriptor()]},
        {"name": "p:0", "type": "Data", "output_desc": [_descriptor()]},
        {"name": "dup", "type": "Data"},
        {"name": "dup", "type": "Data", "output_desc": [_descriptor()]},
        {"name": "insert_cast_4", "type": "CastT", "input": ["p:0", "p:0:0", "p:1", "dup:0", "missing:0", 7]},
    ]}
    inspection = inspect_json(graph)
    module._port_candidates(graph, inspection)
    assert [row["raw_reference"] for row in inspection["casts"][0]["same_graph_port_candidates"]] == ["p:0", "p:0:0"]


def test_cast_and_port_candidate_counts_stay_bounded_and_keep_failure_node():
    producers = [{"name": f"p{i}", "type": "Data", "output_desc": [_descriptor()]} for i in range(12)]
    casts = [{"name": f"cast{i}", "type": "CastT"} for i in range(220)]
    graph = {"op": producers + casts + [{"name": "insert_cast_4", "type": "CastT",
                                         "input": [f"p{i}:0" for i in range(12)]}]}
    inspection = inspect_json(graph)
    module._port_candidates(graph, inspection)
    assert inspection["cast_object_count"] == 221
    assert inspection["reported_cast_count"] == 200
    failing = next(row for row in inspection["casts"] if row["name"] == "insert_cast_4")
    assert len(failing["same_graph_port_candidates"]) == 8

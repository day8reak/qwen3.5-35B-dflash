"""Real protobuf bytes and a synthetic selected-schema ELF; no model execution."""
import ctypes
import hashlib
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys

import pytest

from qwen35_moe_dflash import original_air_inspect as inspect
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.omc_wire import load_schema
from test_draft_ir_inspect import _attr, _blob, _desc, _field, _int, _message
import test_draft_ir_inspect as fixture


@pytest.fixture
def selected(tmp_path, monkeypatch):
    original = fixture._schema()
    def enum(names):
        return _blob(5, _blob(1, "DataType") + b"".join(_blob(2, _blob(1, name) + _int(2, number))
            for name, number in names))
    original = original.replace(enum([("UNDEFINED", 0), ("FLOAT", 1), ("HALF", 7)]),
                                enum([("UNDEFINED", 0), ("DT_FLOAT", 1), ("DT_FLOAT16", 7)]))
    old_graph = _message("GraphDef", _field("name", 1, 9), _field("op", 6, 11, True, "OpDef"))
    new_graph = _message("GraphDef", _field("name", 1, 9), _field("op", 6, 11, True, "OpDef"),
                         _field("output", 5, 9, True))
    original = original.replace(old_graph, new_graph)
    monkeypatch.setattr(fixture, "_schema", lambda: original)
    ddk = tmp_path / "selected-ddk"
    library = ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
    library.parent.mkdir(parents=True)
    fixture._library(library)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("inspection must not run OMG/HDC/any process"))
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: pytest.fail("inspection must not load the vendor library"))
    return ddk, library


def _node(name, kind, shape, dtype=7, index=None):
    raw = _blob(1, name) + _blob(2, kind) + _blob(34, _desc(dtype=dtype, shape=shape))
    return raw if index is None else raw + _attr("index", _int(3, index))


def _graph(*, sequence=55, netoutput=True, constants=7, mutation=None):
    profile = get_existing_air_profile("cpro-40block-64-1")
    nodes, state_descs = [], []
    for index, entry in enumerate(profile["input_shape"].split(";")):
        name, shape = entry.split(":")
        shape = [sequence if int(d) == -1 else int(d) for d in shape.split(",")]
        dtype = 77 if mutation == "unknown_dtype" and index == 0 else 7
        if mutation == "state_shape" and name == "RefData_0":
            shape[-1] = 3
        kind = "RefData" if name.startswith("RefData_") else "Data"
        if mutation != "missing_input" or index != 0:
            nodes.append(_node(name, kind, shape, dtype, index))
        if kind == "RefData":
            state_descs.append(_desc(shape=shape))
    logits_shape = [1, 1, 248319 if mutation == "logits_shape" else 248320]
    output_descs = [_desc(dtype=1, shape=logits_shape), *state_descs]
    if mutation == "output_dtype":
        output_descs[1] = _desc(dtype=1, shape=(1, 8192, 4))
    body = _blob(1, "opaque_body") + _blob(2, "UnverifiedBody") + b"".join(_blob(34, desc) for desc in output_descs)
    nodes.append(body)
    for index in range(constants):
        nodes.append(_node(f"weight_{index}", "FileConstant", [2048]) +
            _attr("file_path", _blob(2, f"./original_export/model_model_layers_{index}_input_layernorm_weight")) +
            _attr("dtype", _int(3, 7)) + _attr("offset", _int(3, index * 4096)) +
            _attr("value", _blob(12, _blob(2, b"PRIVATE_TENSOR_BYTES_MUST_NEVER_BE_PRINTED"))))
    edges = [f"opaque_body:{index}" for index in range(81 if mutation != "missing_output" else 80)]
    if mutation == "bad_producer":
        edges[1] = "does_not_exist:0"
    if netoutput:
        descs = output_descs[:len(edges)]
        if mutation == "output_producer_desc":
            descs = [*descs]
            descs[0] = _desc(shape=logits_shape)
        nodes.append(_blob(1, "Node_Output") + _blob(2, "NetOutput") +
                     b"".join(_blob(5, edge) for edge in edges) + b"".join(_blob(33, desc) for desc in descs))
    if mutation == "duplicate_op":
        nodes.append(nodes[0])
    graph = _blob(1, "original_target") + b"".join(_blob(6, node) for node in nodes)
    if not netoutput:
        graph += b"".join(_blob(5, edge) for edge in edges)
    return graph


def _container(main):
    header = bytearray(256)
    header[:4] = b"IMOD"
    table = struct.pack("<I", 2) + struct.pack("<III", 0, 0, len(main)) + struct.pack("<III", 2, len(main), 0)
    struct.pack_into("<I", header, 0x4c, len(table) + len(main))
    return bytes(header) + table + main


@pytest.mark.parametrize("kind", ["raw_graph", "raw_model", "container", "graph_outputs"])
def test_real_wire_candidates_are_distinguished_and_weights_and_taps_remain_unverified(tmp_path, selected, kind):
    ddk, library = selected
    graph = _graph(netoutput=kind != "graph_outputs")
    raw = graph if kind in {"raw_graph", "graph_outputs"} else _blob(7, graph)
    if kind == "container":
        raw = _container(raw)
    air = tmp_path / "export.air"
    air.write_bytes(raw)
    before = {path: path.stat() for path in (air, library)}
    result = inspect.inspect_original_air(air, ddk)
    assert result["status"] == "STRUCTURE_OBSERVED"
    expected = {"raw_graph": "RAW_GRAPHDEF", "graph_outputs": "RAW_GRAPHDEF", "raw_model": "RAW_MODELDEF", "container": "IMOD_MODEL_PARTITION"}
    assert result["observed_format"] == expected[kind]
    assert result["air"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["air"]["header"]["sha256"] == hashlib.sha256(raw[:64]).hexdigest()
    assert result["tap_resolution"]["status"] == "UNRESOLVED"
    assert result["external_weights"] == "NOT_OPENED_NOT_HASHED_NOT_CHECKED"
    candidate = next(row for row in result["candidates"] if row["status"] == "STRUCTURE_OBSERVED")["graph"]
    assert candidate["ordinary_inputs"][0]["shape"] == [1, 55, 2048]
    assert candidate["input_count"] == 89 and candidate["output_count"] == 81
    assert candidate["model_completeness"] == "NOT_ESTABLISHED_BY_BOUNDARY_CHECK"
    assert candidate["file_constants"]["count"] == 7 and candidate["file_constants"]["shown"] == 6
    assert candidate["file_constants"]["examples"][0]["declared_extent"] == {"offset": 0}
    assert len(json.dumps(result)) < 20000
    assert "PRIVATE_TENSOR_BYTES" not in json.dumps(result)
    # Reads may update filesystem atime; stable content/identity fields must not change.
    def identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    assert all(identity(path.stat()) == identity(stamp) for path, stamp in before.items())


@pytest.mark.parametrize("mutation", ["missing_input", "unknown_dtype", "state_shape", "missing_output",
                                      "logits_shape", "output_dtype", "bad_producer", "duplicate_op", "output_producer_desc"])
def test_parse_success_or_similar_names_cannot_promote_a_different_boundary(tmp_path, selected, mutation):
    ddk, _ = selected
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph(mutation=mutation)))
    result = inspect.inspect_original_air(air, ddk)
    assert result["status"] == "OPAQUE"
    assert all(row["status"] != "STRUCTURE_OBSERVED" for row in result["candidates"])
    assert result["tap_resolution"]["status"] == "UNRESOLVED"


def test_full_metadata_contains_explicit_edges_and_descriptors_without_any_tensor_blob(tmp_path, selected):
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph()))
    result = inspect.inspect_original_air(air, selected[0], full=True)
    graph = next(row["graph"] for row in result["candidates"] if row["status"] == "STRUCTURE_OBSERVED")
    assert len(graph["inputs"]) == 89 and len(graph["outputs"]) == 81
    assert graph["operations"][-1]["input"][0] == "opaque_body:0"
    assert len(graph["file_constants"]["examples"]) == 7
    assert "PRIVATE_TENSOR_BYTES" not in json.dumps(result)
    assert "value" in graph["operations"][-2]["attribute_names"]


def test_metadata_budget_discards_full_proof_instead_of_printing_unbounded_graph(tmp_path, selected, monkeypatch):
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph()))
    monkeypatch.setattr(inspect, "MAX_OUTPUT_BYTES", 1024)
    result = inspect.inspect_original_air(air, selected[0], full=True)
    assert result["status"] == "OPAQUE" and "observed_format" not in result
    assert all("graph" not in row for row in result["candidates"])
    assert result["output_limit"]


def test_schema_failure_preserves_only_identity_and_opaque_observations(tmp_path, selected):
    air = tmp_path / "export.air"
    air.write_bytes(b"OTHER_AIR_CONTAINER\x00\xff\x80")
    result = inspect.inspect_original_air(air, selected[0])
    assert result["status"] == "OPAQUE" and len(result["candidates"]) == 2
    assert result["source_preservation"] == "UNCHANGED_BY_STAT"
    absent = inspect.inspect_original_air(air, tmp_path / "missing-ddk")
    assert absent["schema"]["status"] == "UNAVAILABLE" and absent["status"] == "OPAQUE"
    assert absent["air"]["sha256"] == hashlib.sha256(air.read_bytes()).hexdigest()


def test_oversized_air_is_rejected_before_any_payload_read(tmp_path, monkeypatch):
    air = tmp_path / "oversized.air"
    with air.open("wb") as stream:
        stream.truncate(inspect.MAX_AIR_BYTES + 1)
    monkeypatch.setattr(inspect.os, "open", lambda *a, **k: pytest.fail("oversized AIR payload must not be opened"))
    result = inspect.inspect_original_air(air)
    assert result["status"] == "ERROR" and "64 MiB" in result["error"]
    assert "sha256" not in result["air"]


def test_symlink_is_not_read_and_changed_air_discards_decoded_proof(tmp_path, selected, monkeypatch):
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph()))
    link = tmp_path / "linked.air"
    link.symlink_to(air)
    assert inspect.inspect_original_air(link, selected[0])["status"] == "ERROR"
    schema = load_schema(selected[1])
    original_decode = schema.decode
    def changed(*args):
        result = original_decode(*args)
        air.write_bytes(b"CHANGED SOURCE")
        return result
    monkeypatch.setattr(schema, "decode", changed)
    monkeypatch.setattr(inspect, "load_schema", lambda library: schema)
    result = inspect.inspect_original_air(air, selected[0])
    assert result["status"] == "ERROR" and result["candidates"] == []
    assert result["source_preservation"] == "CHANGED_DURING_INSPECTION"
    assert "observed_format" not in result


def test_cli_defaults_to_small_stdout_json_without_run_directory_or_generated_files(tmp_path, selected, monkeypatch, capsys):
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph()))
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_original_target_air.py"
    before = {path for path in tmp_path.rglob("*")}
    monkeypatch.delenv("AI_RUN_DIR", raising=False)
    monkeypatch.setenv("CPRO_DDK_ROOT", str(selected[0]))
    monkeypatch.setattr(sys, "argv", [str(script), "--air", str(air)])
    assert runpy.run_path(str(script))["main"]() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "STRUCTURE_OBSERVED"
    assert before == {path for path in tmp_path.rglob("*")}


def test_cli_keeps_full_graph_in_report_and_prints_only_a_summary(tmp_path, selected, monkeypatch, capsys):
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph()))
    report = tmp_path / "run" / "source-air.json"
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CPRO_DDK_ROOT", str(selected[0]))
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_original_target_air.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--air", str(air), "--full", "--output", str(report)])
    assert runpy.run_path(str(script))["main"]() == 0
    summary = json.loads(capsys.readouterr().out)
    full = json.loads(report.read_text())
    assert summary["status"] == "STRUCTURE_OBSERVED"
    assert summary["report"] == str(report)
    assert all("graph" not in row for row in summary["candidates"])
    graph = next(row["graph"] for row in full["candidates"] if row["status"] == "STRUCTURE_OBSERVED")
    assert graph["operations"] and len(graph["outputs"]) == 81
    with pytest.raises(FileExistsError):
        runpy.run_path(str(script))["main"]()
    assert json.loads(report.read_text()) == full

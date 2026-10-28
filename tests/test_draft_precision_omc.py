"""Inspect bound OMC graph slices without executing vendor code or reading weights."""
import hashlib
import json
import struct
import copy

import pytest

from qwen35_moe_dflash import draft_precision_inspect as di
from qwen35_moe_dflash.omc_wire import Schema
from test_draft_ir_inspect import _attr, _blob, _desc, _int
from test_draft_precision_inspect import (
    _encoded_graph, _inspect, _json, _record, _refresh_compilation_receipt,
    retained_precision,
)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _owner(name, cl, offset, size, *, edge=None, dtype=7):
    raw = (_blob(1, name) + _blob(2, "GraphOp") + _attr("cl_name", _blob(2, cl))
           + _attr("graphop_task_offset", _int(3, offset))
           + _attr("graphop_task_size", _int(3, size))
           + _blob(34, _desc(dtype, (1, 15, 32))))
    if edge:
        raw += _blob(5, edge) + _blob(33, _desc(7, (1, 15, 32)))
    return raw


def _container(main, task):
    header = bytearray(256)
    header[:4] = b"IMOD"
    table = (struct.pack("<I", 3) + struct.pack("<III", 0, 0, len(main))
             + struct.pack("<III", 1, len(main), 0)
             + struct.pack("<III", 2, len(main), len(task)))
    struct.pack_into("<I", header, 0x4c, len(table) + len(main) + len(task))
    return bytes(header) + table + main + task


def _write_case(case, *, overlap=False, outside=False, cpu_name="SubGraph_0", cpu_raw=None,
                dtype=7, all_opaque=False, packing_damage=None):
    root, _, library, records, dump = case
    if dump.exists():
        dump.unlink()
    cpu = _encoded_graph(records, graph_name=cpu_name) if cpu_raw is None else cpu_raw
    # A valid protobuf would be misleading here: a non-CPU task is opaque even
    # when its bytes happen to resemble a graph or contain source node names.
    opaque = b"opaque-npu-task\xff\x00v579\x00draft_hidden"
    task = cpu + opaque
    offset = 0 if overlap else len(task) + 1 if outside else len(cpu)
    first = _owner("SubGraph_0", "NPUCL" if all_opaque else "CPUCL", 0, len(cpu))
    second = _owner("SubGraph_1", "NPUCL", offset, len(opaque), edge="SubGraph_0:0", dtype=dtype)
    output = (_blob(1, "NetOutput") + _blob(2, "NetOutput")
              + _blob(5, "SubGraph_1:0") + _blob(33, _desc(dtype, (1, 15, 32))))
    main = _blob(7, _blob(1, "compiled_draft") + _blob(6, first) + _blob(6, second) + _blob(6, output))
    raw = _container(main, task)
    omc = root / "compilation/compile-work/generated/model/model.omc"
    omc.parent.mkdir(parents=True, exist_ok=True)
    omc.write_bytes(raw)
    cp = root / "compilation/report.json"
    compiled = json.loads(cp.read_text())
    compiled.update(generated_artifacts=[], omc=_record(omc, root / "compilation"))
    _json(cp, compiled)
    _refresh_compilation_receipt(root)
    job = json.loads((root / "result.json").read_text())
    bundle = root / "bundle"
    # Deliberately absent. The outer numerical review binds the executed
    # payload; this inspector must use its receipt instead of rereading GBs.
    executed = {"path": "model/model.omc", "bytes": 2_800_000_000, "sha256": "a" * 64}
    packing = {
        "artifact_kind": "qwen35-omc-embedded-stream", "status": "EMBEDDED",
        "source": compiled["omc"], "output": executed,
        "schema": {"library_sha256": _hash(library.read_bytes())},
        "partitions": [{"type": 0, "bytes": len(main), "sha256": _hash(main), "preserved": "BYTE_IDENTICAL"},
                       {"type": 2, "bytes": len(task), "sha256": _hash(task), "preserved": "BYTE_IDENTICAL"}],
    }
    if packing_damage:
        if packing_damage in {"source", "output"}:
            packing[packing_damage] = {**packing[packing_damage], "sha256": "f" * 64}
        elif packing_damage == "schema":
            packing["schema"]["library_sha256"] = "f" * 64
        else:
            packing["partitions"][1]["sha256"] = "f" * 64
    _json(bundle / "weight-packing.json", packing)
    manifest = {"artifact_kind": "qwen35-standalone-draft-bundle", "weight_storage": "embedded-stream",
                "compiler_report_sha256": job["components"]["compilation"]["manifest"]["sha256"],
                "export_manifest_sha256": job["components"]["air"]["manifest"]["sha256"],
                "model": executed, "weight_packing": _record(bundle / "weight-packing.json", bundle)}
    _json(bundle / "manifest.json", manifest)
    job["components"]["bundle"] = {"directory": str(bundle), "manifest": _record(bundle / "manifest.json", bundle)}
    _json(root / "result.json", job)
    return omc


def test_packing_bound_cpu_slice_decodes_once_and_npu_payload_is_opaque(retained_precision, monkeypatch):
    case = retained_precision
    omc = _write_case(case)
    before = {p: p.read_bytes() for folder in case[:2] for p in folder.rglob("*") if p.is_file()}
    original, calls = Schema.decode, []

    def decode(self, raw, name):
        calls.append(name)
        return original(self, raw, name)

    monkeypatch.setattr(Schema, "decode", decode)
    report = _inspect(case)
    assert report["status"] == "INSPECTED", report.get("reason")
    assert calls == ["ModelDef", "GraphDef"]
    assert report["omc_inspection"]["binding"] == "PACKING_PRESERVED_MAIN_AND_TASK_BYTES"
    assert [r["evidence_kind"] for r in report["files"]] == ["OMC_MODELDEF", "OMC_CPU_GRAPHDEF"]
    assert report["files"][1]["matches"]
    assert report["summary"]["graphop_counts"] == {"CPUCL": 1, "NPUCL": 1}
    assert report["summary"]["output_connections"][0]["status"] == "RESOLVED"
    assert report["summary"]["output_connections"][0]["producer_output_descriptor"]["dtype"] == "float16"
    assert report["summary"]["output_connections"][0]["consumer_input_descriptor"]["shape"] == [1, 15, 32]
    assert report["summary"]["key_nodes"]
    assert not (case[0] / "bundle/model/model.omc").exists()
    assert not list(omc.parent.glob("*.weight"))
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("damage", ["source", "output", "schema", "task"])
def test_mismatched_packing_receipt_never_decodes_cpu_or_main(retained_precision, monkeypatch, damage):
    _write_case(retained_precision, packing_damage=damage)
    monkeypatch.setattr(Schema, "decode", lambda *a: pytest.fail("unbound source must not be decoded"))
    report = _inspect(retained_precision)
    assert report["status"] == "NOT_AVAILABLE"
    assert not report["files"]
    assert report["omc_inspection"]["binding"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("damage", ["overlap", "outside"])
def test_invalid_all_task_ownership_rejects_before_any_cpu_slice(retained_precision, monkeypatch, damage):
    _write_case(retained_precision, **{damage: True})
    original, calls = Schema.decode, []

    def decode(self, raw, name):
        calls.append(name)
        return original(self, raw, name)

    monkeypatch.setattr(Schema, "decode", decode)
    report = _inspect(retained_precision)
    assert report["status"] == "NOT_AVAILABLE"
    assert calls == ["ModelDef"]
    assert not report["omc_inspection"]["graphop_boundaries"]
    assert not report["files"]


@pytest.mark.parametrize("cpu", ["wrong-name", "invalid-protobuf"])
def test_bad_cpu_slice_preserves_only_bound_main_graph_evidence(retained_precision, cpu):
    _write_case(retained_precision, **({"cpu_name": "WrongGraph"} if cpu == "wrong-name" else {"cpu_raw": b"not a GraphDef"}))
    report = _inspect(retained_precision)
    assert report["status"] == "PARTIAL"
    assert report["files"][1]["status"] == "UNSUPPORTED_CPU_SLICE"
    assert not report["files"][1].get("matches")
    assert report["summary"]["output_connections"][0]["status"] == "RESOLVED"


def test_opaque_final_norm_still_reports_output_type_without_precision_claim(retained_precision, monkeypatch):
    _write_case(retained_precision, all_opaque=True, dtype=1)
    original, calls = Schema.decode, []

    def decode(self, raw, name):
        calls.append(name)
        return original(self, raw, name)

    monkeypatch.setattr(Schema, "decode", decode)
    report = _inspect(retained_precision)
    assert report["status"] == "INSPECTED"
    assert calls == ["ModelDef"]
    summary = report["summary"]
    assert not summary["descriptor_differences"]
    assert summary["output_connections"][0]["producer_output_descriptor"]["dtype"] == "float32"
    assert summary["output_connections"][0]["consumer_input_descriptor"]["dtype"] == "float32"
    assert summary["not_observed"] > 10
    assert len(summary["key_nodes"]) < 20
    assert all(row["status"] == "NOT_OBSERVED" for row in summary["key_nodes"])
    assert "intermediate-value" in summary["opaque_task_note"]


def test_changed_source_never_decodes_or_reports_bound_descriptors(retained_precision, monkeypatch):
    omc = _write_case(retained_precision)
    omc.write_bytes(omc.read_bytes() + b"corrupt")
    monkeypatch.setattr(Schema, "decode", lambda *a: pytest.fail("unbound source must not decode"))
    report = _inspect(retained_precision)
    assert report["status"] == "NOT_AVAILABLE"
    assert not report["files"]
    assert not report["summary"]["output_connections"]


def test_late_metadata_change_clears_decoded_omc_boundaries(retained_precision, monkeypatch):
    case = retained_precision
    _write_case(case)
    monkeypatch.setattr(di, "inspect_precision_log", lambda *args: {
        "status": "INSPECTED", "source": None, "lines": [{"line": 1, "text": "v581 support context"}]})
    original = Schema.decode

    def decode(self, raw, name):
        result = original(self, raw, name)
        if name == "GraphDef":
            path = case[0] / "bundle/weight-packing.json"
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(Schema, "decode", decode)
    report = _inspect(case)
    assert report["status"] == "NOT_AVAILABLE"
    assert not any(row.get("matches") for row in report["files"])
    assert report["summary"]["omc_binding"] == "NOT_ESTABLISHED"
    assert not report["summary"]["output_connections"]
    assert not report["summary"]["graphop_boundaries"]
    assert report["summary"]["compile_log_evidence"]["status"] == "NOT_AVAILABLE"
    assert not report["summary"]["compile_log_evidence"]["lines"]


def test_external_omc_byte_limit_is_enforced_before_decode(retained_precision, monkeypatch):
    _write_case(retained_precision)
    monkeypatch.setattr(di, "_MAX_OMC_BYTES", 20)
    monkeypatch.setattr(Schema, "decode", lambda *a: pytest.fail("oversized source must not decode"))
    report = _inspect(retained_precision)
    assert report["status"] == "NOT_AVAILABLE"
    assert "byte limit" in report["reason"]


def test_final_norm_inserted_cast_and_power_parameters_reach_summary(retained_precision, monkeypatch):
    import test_draft_ir_inspect as ir
    case = retained_precision
    # Add a float attribute to this fixture's selected schema; no decoder or
    # DDK enum substitution is used by the production path.
    schema = ir._schema()
    old = ir._message("AttrDef", ir._field("i", 3, 3), ir._field("s", 2, 12),
                      ir._field("t", 12, 11, False, "TensorDef"), ir._field("g", 13, 11, False, "GraphDef"))
    new = ir._message("AttrDef", ir._field("i", 3, 3), ir._field("s", 2, 12),
                      ir._field("t", 12, 11, False, "TensorDef"), ir._field("g", 13, 11, False, "GraphDef"),
                      ir._field("f", 4, 2))
    assert schema.count(old) == 1
    monkeypatch.setattr(ir, "_schema", lambda: schema.replace(old, new))
    ir._library(case[2])
    records = copy.deepcopy(case[3])
    wi = next(i for i, r in enumerate(records) if r["name"] == "draft.norm.weight")
    gamma = next(r for r in records if r["type"] == "Mul" and any(p["src"] == wi for p in r["inputs"]))
    cast_index = next(p["src"] for p in gamma["inputs"] if p["src"] != wi)
    cast = records[cast_index]
    assert cast["type"] == "Cast"
    product = records[cast["inputs"][0]["src"]]
    rsqrt = next(records[p["src"]] for p in product["inputs"] if records[p["src"]]["type"] == "Rsqrt")
    rsqrt["type"] = "Power"
    extra = copy.deepcopy(cast)
    extra.update(name="compiler_insert_cast_final", type="CastT")
    cast["inputs"][0]["src"] = len(records)
    cast["inputs"][0]["dtype"] = "float16"
    records.append(extra)
    # Encode real attributes, including the float32 power value, using the
    # selected fixture schema. The summary must retain integer dtype attrs
    # without translating them through the tensor descriptor enum domain.
    cpu = _encoded_graph(records, graph_name="SubGraph_0")
    cast_prefix = _blob(1, extra["name"]) + _blob(2, extra["type"])
    cast_attrs = _attr("src_dtype", _int(3, 1)) + _attr("dst_dtype", _int(3, 7))
    power_prefix = _blob(1, rsqrt["name"]) + _blob(2, "Power")
    power_attrs = _attr("power", bytes([(4 << 3) | 5]) + struct.pack("<f", -0.5))
    # Append attrs inside the length-delimited operation, never inject into
    # an enclosing protobuf length. Build only the changed operation again.
    def with_attrs(raw, prefix, attributes):
        from qwen35_moe_dflash.omc_wire import _fields
        chunks = []
        for number, wire, value in _fields(raw, [10000]):
            assert wire == 2
            chunks.append(_blob(number, bytes(value) + attributes if number == 6 and bytes(value).startswith(prefix) else bytes(value)))
        return b"".join(chunks)
    cpu = with_attrs(with_attrs(cpu, cast_prefix, cast_attrs), power_prefix, power_attrs)
    _write_case(case, cpu_raw=cpu)
    report = _inspect(case)
    assert report["status"] == "INSPECTED", report.get("reason")
    conversions = report["summary"]["final_norm_conversion_ancestors"]
    row, = [r for r in conversions if r["name"] == extra["name"]]
    assert row["attributes"] == [{"key": "src_dtype", "value": {"i": 1}}, {"key": "dst_dtype", "value": {"i": 7}}]
    assert row["input_descriptors"][0]["dtype"] == "float32"
    assert row["output_descriptors"][0]["dtype"] == "float16"
    power, = [r for r in report["summary"]["key_nodes"] if r["name"] == rsqrt["name"]]
    assert power["observed"][0]["type"] == "Power"
    assert power["observed"][0]["attributes"] == [{"key": "power", "value": {"f": -0.5}}]

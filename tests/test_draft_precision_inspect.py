"""Read retained precision descriptors using a selected schema, never GE/DDK execution."""
import hashlib
import json
import subprocess

import pytest

from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from test_draft_ir_inspect import _blob, _desc, _int, _library


def _json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def _record(path, root):
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _encoded_graph(records, *, graph_name="draft", dtype_changes=None, renamed=None):
    dtype_changes, renamed = dtype_changes or {}, renamed or {}
    encoded = _blob(1, graph_name)
    # These are the test library's schema values, deliberately unlike GE IDs.
    codes = {"float32": 1, "float16": 7, "int32": 3, "int64": 4, "bool": 5}
    for node in records:
        name = node["name"]
        op = _blob(1, renamed.get(name, name)) + _blob(2, node["type"])
        for edge in node["inputs"]:
            parent = records[edge["src"]]["name"]
            op += _blob(5, renamed.get(parent, parent) + ":0")
        for key, field in (("inputs", 33), ("outputs", 34)):
            for port in node[key]:
                dtype = dtype_changes.get(name, codes[port["dtype"]])
                descriptor = (_desc(dtype, port["shape"]) if dtype is not None else
                    _blob(3, b"".join(_int(1, dimension) for dimension in port["shape"])) + _blob(4, "ND"))
                op += _blob(field, descriptor)
        encoded += _blob(6, op)
    return encoded


def _dump(case, *graphs):
    root, air, library, records, path = case
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_blob(7, graph) for graph in graphs))
    report_path = root / "compilation/report.json"
    report = json.loads(report_path.read_text())
    report["generated_artifacts"] = [_record(path, root / "compilation")]
    _json(report_path, report)
    _refresh_compilation_receipt(root)
    return path


def _refresh_compilation_receipt(root):
    path = root / "result.json"
    job = json.loads(path.read_text())
    job["components"]["compilation"] = {"directory": str(root / "compilation"),
        "manifest": _record(root / "compilation/report.json", root / "compilation")}
    _json(path, job)


def _differences(report):
    return [{"graph_name": graph["graph_name"], "node": row["node"]["name"], **difference}
            for file in report["files"] for graph in file.get("matches", [])
            for row in graph["nodes"] for difference in row.get("descriptor_differences", [])]


@pytest.fixture
def retained_precision(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("inspection must not run tools"))
    root, air = tmp_path / "job", tmp_path / "air"
    library = tmp_path / "selected-schema.so"
    _library(library)
    graph = build_draft(Qwen35DFlashConfig.from_dict(configurations(tiny=True)[1]), 8)
    emit_cpp(graph, {name: tmp_path / "absent-weights" / f"{i}.bin" for i, name in enumerate(graph.weights)},
             external_weights=True, recipe_dir=air / "draft/recipe", fp16_row_mul="compact")
    recipe = air / "draft/recipe/graph.recipe"
    records = read_recipe(recipe)["records"]
    air_record = {"path": "draft/draft.air", "bytes": 8, "sha256": hashlib.sha256(b"fake AIR").hexdigest()}
    manifest = _json(air / "manifest.json", {
        "artifact_kind": "qwen35-cpro-draft-air", "status": "EXPORTED", "graphs": [{
            "name": "draft", "air": air_record["path"], "files": [air_record],
            "construction_files": [_record(recipe, air)]}]})
    _json(root / "result.json", {"artifact_kind": "qwen35-full-draft-job", "status": "FAILED",
        "components": {"air": {"directory": str(air), "manifest": _record(manifest, air)}}})
    _json(root / "compilation/report.json", {
        "artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED",
        "requested_options": {"weight_data_type": "FP16", "use_origin_format": "true"},
        "air": air_record, "input_snapshots": [{**air_record, "role": "air",
            "path": "compile-work/generated/draft.air"}], "generated_artifacts": []})
    path = root / "compilation/compile-work/generated/dump_graph/01_precision.irpb"
    case = root, air, library, records, path
    _dump(case, _encoded_graph(records))
    return case


def _inspect(case, **kwargs):
    from qwen35_moe_dflash.draft_precision_inspect import inspect_draft_precision
    return inspect_draft_precision(case[0], schema_library=case[2], **kwargs)


def test_matching_fp32_descriptors_are_decoded_once_and_source_files_unchanged(retained_precision, monkeypatch):
    from qwen35_moe_dflash.omc_wire import Schema
    case = retained_precision
    before = {p: p.read_bytes() for folder in case[:2] for p in folder.rglob("*") if p.is_file()}
    original, calls = Schema.decode, []
    def decode(self, data, type_name):
        calls.append(type_name)
        return original(self, data, type_name)
    monkeypatch.setattr(Schema, "decode", decode)
    report = _inspect(case)
    assert report["status"] == "INSPECTED"
    assert report["source_binding"] == "RECORDED_AIR_HASH_MATCHED"
    assert calls == ["ModelDef"]
    assert len(report["source_targets"]) > 1
    assert report["files"][0]["status"] == "DECODED"
    assert report["files"][0]["matches"]
    assert not _differences(report)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert not (case[1] / "draft/draft.air").exists()
    assert not (case[0].parent / "absent-weights").exists()


@pytest.mark.parametrize("damage", ["recipe", "manifest", "snapshot"])
def test_corrupted_source_binding_is_not_inspected(retained_precision, damage):
    case = retained_precision
    if damage == "recipe":
        path = case[1] / "draft/recipe/graph.recipe"
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif damage == "manifest":
        path = case[1] / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = case[0] / "compilation/report.json"
        report = json.loads(path.read_text())
        report["input_snapshots"][0]["sha256"] = "f" * 64
        _json(path, report)
        _refresh_compilation_receipt(case[0])
    report = _inspect(case)
    assert report["status"] == "NOT_AVAILABLE"
    assert "mismatch" in report["reason"] or "differs" in report["reason"]
    assert not any(file.get("matches") for file in report["files"])


def test_changed_dump_is_not_decoded_as_current_compilation(retained_precision):
    case = retained_precision
    case[4].write_bytes(case[4].read_bytes() + b"corrupt")
    report = _inspect(case)
    assert report["status"] != "INSPECTED"
    assert report["files"][0]["status"] == "INTEGRITY_MISMATCH"
    assert not report["files"][0].get("matches")


def test_fp32_mean_replaced_by_half_is_reported_using_selected_schema(retained_precision):
    case = retained_precision
    mean = next(record for record in case[3] if record["type"] == "ReduceMean")
    assert mean["outputs"][0]["dtype"] == "float32"
    _dump(case, _encoded_graph(case[3], dtype_changes={mean["name"]: 7}))
    report = _inspect(case)
    assert report["files"][0]["status"] == "DECODED"
    differences = _differences(report)
    assert differences
    detail = json.dumps(differences)
    assert mean["name"] in detail and "float32" in detail and "float16" in detail
    assert "HALF" in json.dumps(report["files"][0]["matches"])


def test_renamed_nodes_are_not_reported_as_precision_regressions(retained_precision):
    case = retained_precision
    renamed = {record["name"]: "fused_" + record["name"] for record in case[3]}
    _dump(case, _encoded_graph(case[3], renamed=renamed))
    report = _inspect(case)
    assert report["files"][0]["status"] == "DECODED"
    assert not report["files"][0]["matches"]
    assert not _differences(report)
    assert report["summary"]["not_observed"]


def test_same_names_in_distinct_graphs_keep_ancestry_and_differences_separate(retained_precision):
    case = retained_precision
    mean = next(record for record in case[3] if record["type"] == "ReduceMean")
    _dump(case, _encoded_graph(case[3], graph_name="correct_subgraph"),
          _encoded_graph(case[3], graph_name="changed_subgraph", dtype_changes={mean["name"]: 7}))
    report = _inspect(case)
    matches = report["files"][0]["matches"]
    assert {match["graph_name"] for match in matches} == {"correct_subgraph", "changed_subgraph"}
    differences = _differences(report)
    assert differences
    assert all(row["graph_name"] == "changed_subgraph" for row in differences)


def test_duplicate_names_within_one_graph_do_not_establish_descriptor_differences(retained_precision):
    case = retained_precision
    mean = next(record for record in case[3] if record["type"] == "ReduceMean")
    _dump(case, _encoded_graph([*case[3], mean], dtype_changes={mean["name"]: 7}))
    report = _inspect(case)
    rows = [row for graph in report["files"][0]["matches"] for row in graph["nodes"]
            if row["node"]["name"] == mean["name"]]
    assert len(rows) == 2 and all(row["name_status"] == "AMBIGUOUS" for row in rows)
    assert all(not row.get("descriptor_differences") for row in rows)
    assert not any(row["node"] == mean["name"] for row in _differences(report))


def test_omitted_dtype_is_not_guessed_from_operator_or_public_ge_ids(retained_precision):
    case = retained_precision
    mean = next(record for record in case[3] if record["type"] == "ReduceMean")
    _dump(case, _encoded_graph(case[3], dtype_changes={mean["name"]: None}))
    report = _inspect(case)
    row, = [row for graph in report["files"][0]["matches"] for row in graph["nodes"]
            if row["node"]["name"] == mean["name"]]
    assert all(desc["dtype_presence"] == "OMITTED_NOT_INTERPRETED"
               for key in ("input_desc", "output_desc") for desc in row["node"][key])
    assert not row["descriptor_differences"]
    assert report["summary"]["captured_output_dtypes"]["NOT_INTERPRETED"] > 0


def test_unrecorded_dump_is_not_decoded_as_compiler_evidence(retained_precision, monkeypatch):
    from qwen35_moe_dflash.omc_wire import Schema
    case = retained_precision
    path = case[0] / "compilation/report.json"
    compiled = json.loads(path.read_text())
    compiled["generated_artifacts"] = []
    _json(path, compiled)
    _refresh_compilation_receipt(case[0])
    monkeypatch.setattr(Schema, "decode", lambda *a, **kw: pytest.fail("unbound dump must not be decoded"))
    report = _inspect(case)
    assert report["status"] == "PARTIAL"
    assert report["files"][0]["status"] == "UNBOUND_DUMP"
    assert not report["files"][0].get("matches")


@pytest.mark.parametrize("missing", ["schema", "dump"])
def test_missing_schema_or_dumps_leave_unavailable_diagnostics(retained_precision, missing):
    case = retained_precision
    (case[2] if missing == "schema" else case[4]).unlink()
    report = _inspect(case)
    assert report["status"] in {"NOT_AVAILABLE", "PARTIAL"}
    assert not any(row.get("matches") for row in report["files"])


def test_opaque_recorded_dump_is_explicitly_unsupported(retained_precision):
    case = retained_precision
    case[4].write_bytes(b"not a ModelDef")
    report_path = case[0] / "compilation/report.json"
    compiled = json.loads(report_path.read_text())
    compiled["generated_artifacts"] = [_record(case[4], case[0] / "compilation")]
    _json(report_path, compiled)
    _refresh_compilation_receipt(case[0])
    report = _inspect(case)
    assert report["status"] != "INSPECTED"
    assert report["files"][0]["status"] == "UNSUPPORTED_DUMP"
    assert not report["files"][0].get("matches")


@pytest.mark.parametrize("mismatch", [None, "compiler_report_sha256", "export_manifest_sha256"])
def test_executed_bundle_must_bind_the_inspected_air_and_compiler_receipts(retained_precision, mismatch):
    case = retained_precision
    job_path = case[0] / "result.json"
    job = json.loads(job_path.read_text())
    manifest = {"artifact_kind": "qwen35-standalone-draft-bundle", "status": "PREPARED",
                "compiler_report_sha256": job["components"]["compilation"]["manifest"]["sha256"],
                "export_manifest_sha256": job["components"]["air"]["manifest"]["sha256"]}
    if mismatch:
        manifest[mismatch] = "f" * 64
    bundle = case[0] / "bundle"
    _json(bundle / "manifest.json", manifest)
    job["components"]["bundle"] = {"directory": str(bundle),
        "manifest": _record(bundle / "manifest.json", bundle)}
    _json(job_path, job)
    report = _inspect(case)
    if mismatch is None:
        assert report["executed_bundle_binding"] == "MATCHED"
        assert report["status"] == "INSPECTED" and report["files"][0]["matches"]
    else:
        assert report["executed_bundle_binding"] == "MISMATCH"
        assert report["status"] == "NOT_AVAILABLE"
        assert not any(file.get("matches") for file in report["files"])


def test_after_partition_dump_is_selected_before_many_earlier_filename_hints(retained_precision):
    case = retained_precision
    folder = case[4].parent
    case[4].unlink()
    raw = _blob(7, _encoded_graph(case[3]))
    dumps = []
    for index in range(18):
        path = folder / f"00_before_parse_{index:02d}.irpb"
        path.write_bytes(raw)
        dumps.append(path)
    mean = next(record for record in case[3] if record["type"] == "ReduceMean")
    later = folder / "99_after_partition.irpb"
    later.write_bytes(_blob(7, _encoded_graph(case[3], dtype_changes={mean["name"]: 7})))
    dumps.append(later)
    compiled_path = case[0] / "compilation/report.json"
    compiled = json.loads(compiled_path.read_text())
    compiled["generated_artifacts"] = [_record(path, case[0] / "compilation") for path in dumps]
    _json(compiled_path, compiled)
    _refresh_compilation_receipt(case[0])
    report = _inspect(case)
    assert report["files_omitted"] == 3
    assert report["files"][0]["path"].endswith("99_after_partition.irpb")
    assert report["files"][0]["status"] == "DECODED"
    assert any(row["node"] == mean["name"] and row["observed"] == "float16"
               for row in _differences(report))
    assert "filenames do not prove compilation order" in report["selection_policy"]

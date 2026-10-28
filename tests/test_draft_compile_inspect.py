"""Retained draft failure inspection; synthetic metadata, no vendor execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from qwen35_moe_dflash.draft_compile_inspect import inspect_draft_compile
from qwen35_moe_dflash.ge_recipe import emit_recipe_cpp


NODE = "__ge/v20/flat_output"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _record(path, root):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def retained(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("inspection must not run a tool"))
    air, job, output = tmp_path / "export", tmp_path / "failed-job", tmp_path / "inspection"
    def port(shape, name="y", **extra):
        return {"port": name, "shape": shape, "dtype": "float16", **extra}
    records = [
        {"name": "features", "type": "Data", "inputs": [], "outputs": [port([64, 16384])],
         "attrs": {"index": {"type": "int", "value": 0}}},
        {"name": "draft.fc.weight", "type": "FileConstant", "inputs": [], "outputs": [port([2048, 16384])],
         "attrs": {"file_path": {"type": "string", "value": "weights/not-present.bin"},
                   "shape": {"type": "ints", "value": [2048, 16384]}, "dtype": {"type": "int", "value": 1}}},
        {"name": NODE, "type": "MatMul", "inputs": [port([64, 16384], "x1", src=0, src_port="y"),
                                                        port([2048, 16384], "x2", src=1, src_port="y")],
         "outputs": [port([64, 2048])], "attrs": {"transpose_x1": {"type": "bool", "value": False},
                                                  "transpose_x2": {"type": "bool", "value": True}}},
    ]
    emit_recipe_cpp(records, "draft", [0], [(2, 0)], air / "draft/recipe")
    recipe = air / "draft/recipe/graph.recipe"
    air_identity = {"path": "draft.air", "bytes": 7, "sha256": hashlib.sha256(b"NOT AIR").hexdigest()}
    manifest = _write(air / "manifest.json", {"artifact_kind": "qwen35-cpro-draft-air", "status": "EXPORTED",
        "graphs": [{"name": "draft", "air": "draft.air", "files": [air_identity],
                    "construction_files": [_record(recipe, air)]}]})
    _write(job / "result.json", {"artifact_kind": "qwen35-full-draft-job", "status": "FAILED",
        "components": {"air": {"directory": str(air), "manifest": _record(manifest, air)}}})
    log = job / "compilation/compile.log"
    log.parent.mkdir(parents=True)
    log.write_text(f"I/AI_FMK InferShape for Graph [draft]\n"
                   f"I/AI_FMK [op:{NODE} type:MatMul] infershape success\n"
                   "D/AI_FMK Checking NPU store convolution\n"
                   f"E/AI_FMK CheckSupported: the op name [{NODE}] type [MatMul] don't support\n"
                   f"I/AI_FMK unrelated [{NODE}_other]\n")
    compiled = {"artifact_kind": "qwen35-existing-air-compilation", "status": "FAILED",
        "ddk_root": str(tmp_path / "selected-ddk"), "air": air_identity,
        "input_snapshots": [{**air_identity, "path": "compile-work/generated/draft.air", "role": "air"}],
        "requested_options": {"use_origin_format": "true", "weight_data_type": "FP16"},
        "compile_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "generated_artifacts": [{"path": "compile-work/generated/dump_graph/fail.irpb", "bytes": 19, "sha256": "a" * 64}]}
    _write(job / "compilation/report.json", compiled)
    # The IR decoder has a separate schema-derived suite. This test double
    # validates only how the metadata inspector delegates and labels absence.
    calls = []
    ir = ModuleType("qwen35_moe_dflash.draft_ir_inspect")
    def inspect(compilation_dir, schema_library, *, node_name):
        calls.append((compilation_dir, schema_library, node_name))
        return {"status": "SCHEMA_UNAVAILABLE", "evidence_kind": "TEST_DOUBLE", "files": []}
    ir.inspect_draft_ir = inspect
    monkeypatch.setitem(sys.modules, "qwen35_moe_dflash.draft_ir_inspect", ir)
    return job, air, output, calls


def test_readonly_failure_inspection_preserves_job_and_reports_declared_weight_and_log(retained):
    job, air, output, calls = retained
    before = {path: path.read_bytes() for root in (job, air) for path in root.rglob("*") if path.is_file()}
    result = inspect_draft_compile(job, output)
    assert result["status"] == "INSPECTED"
    assert result["association"]["status"] == "RECORDED_AIR_HASH_MATCHED"
    declared = result["declared_graph"]
    assert declared["node"]["type"] == "MatMul"
    assert declared["node"]["attrs"]["transpose_x2"]["value"] is True
    weight = declared["input_producers"][1]["node"]
    assert weight["name"] == "draft.fc.weight" and weight["type"] == "FileConstant"
    assert weight["outputs"][0]["shape"] == [2048, 16384]
    assert result["log"]["node_context"]["exact_node_occurrences"] == 2
    assert any("Checking NPU store" in row["text"] for row in result["log"]["node_context"]["lines"])
    assert result["log"]["hash_status"] == "MATCHED_COMPILER_RECORD"
    assert result["requested_options"]["weight_data_type"] == "FP16"
    assert calls[0] == (job / "compilation", job.parent / "selected-ddk/tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so", NODE)
    assert "draft.fc.weight" in (output / "report.txt").read_text()
    assert result["comparison"] == str(output / "report.txt")
    assert {path: path.read_bytes() for path in before} == before
    assert not (air / "draft.air").exists() and not (air / "weights/not-present.bin").exists()


@pytest.mark.parametrize("damage", ["recipe", "air_manifest", "log", "snapshot_identity"])
def test_hash_or_air_association_corruption_fails_before_output(retained, damage):
    job, air, output, _ = retained
    if damage == "recipe":
        path = air / "draft/recipe/graph.recipe"
        path.write_bytes(path.read_bytes() + b"CORRUPT")
    elif damage == "air_manifest":
        path = air / "manifest.json"
        path.write_text(path.read_text() + " ")
    elif damage == "log":
        (job / "compilation/compile.log").write_text("different log")
    else:
        path = job / "compilation/report.json"
        compiled = json.loads(path.read_text())
        compiled["input_snapshots"][0]["sha256"] = "f" * 64
        _write(path, compiled)
    with pytest.raises(ValueError, match="mismatch|identity differs"):
        inspect_draft_compile(job, output)
    assert not output.exists()


@pytest.mark.parametrize("missing", ["log", "report"])
def test_missing_compiler_evidence_is_partial_and_never_claims_association(retained, missing):
    job, _, output, _ = retained
    (job / "compilation" / ("compile.log" if missing == "log" else "report.json")).unlink()
    result = inspect_draft_compile(job, output)
    assert result["status"] == "PARTIAL"
    if missing == "log":
        assert result["log"]["status"] == "UNAVAILABLE"
    else:
        assert result["association"]["status"] == "UNAVAILABLE"
        assert result["log"]["exact_node_support"] == []


def test_explicit_schema_library_and_unknown_node_are_reported_without_fuzzy_match(retained):
    job, _, output, calls = retained
    library = job.parent / "explicit-library.so"
    result = inspect_draft_compile(job, output, node=NODE + "_missing", schema_library=library)
    assert result["declared_graph"]["status"] == "NOT_FOUND"
    assert result["log"]["node_context"]["exact_node_occurrences"] == 0
    assert calls[0][1] == library


def test_text_report_includes_actual_ir_descriptor_enum_shape_and_weight_producer(retained, monkeypatch):
    job, _, output, _ = retained
    def descriptor(shape):
        return {"raw": {"dtype": 19, "shape": {"dim": shape}, "layout": "ND"},
                "schema_enum_names": {"dtype": {"value": 19, "enum_type": "fixture.Type",
                                                 "names": ["FIXTURE_FP16"]}}}
    node = {"name": NODE, "type": "MatMul", "input_desc": [descriptor([64, 16384])],
            "output_desc": [descriptor([64, 2048])]}
    weight = {"name": "draft.fc.weight", "type": "Const", "input_desc": [],
              "output_desc": [descriptor([2048, 16384])]}
    ir = sys.modules["qwen35_moe_dflash.draft_ir_inspect"]
    monkeypatch.setattr(ir, "inspect_draft_ir", lambda *a, **kw: {
        "status": "INSPECTED", "files": [{"path": "dump/fail.irpb", "status": "TARGET_FOUND",
        "matches": [{"graph_name": "draft", "node": node,
                     "input_producers": [{"field": "input", "input_index": 1, "producer": weight}]}]}]})
    inspect_draft_compile(job, output)
    rendered = (output / "report.txt").read_text()
    assert "IR exact node" in rendered and "IR direct producer" in rendered
    assert '"dtype": 19' in rendered and "FIXTURE_FP16" in rendered
    assert '"dim": [2048, 16384]' in rendered and '"layout": "ND"' in rendered


def test_overlarge_log_rejected_before_reading_or_creating_output(retained, monkeypatch):
    from qwen35_moe_dflash import draft_compile_inspect as module
    job, _, output, _ = retained
    monkeypatch.setattr(module, "_MAX_LOG", 8)
    with pytest.raises(ValueError, match="byte limit"):
        inspect_draft_compile(job, output)
    assert not output.exists()


def test_existing_output_and_original_job_cannot_be_overwritten(retained):
    job, _, output, _ = retained
    with pytest.raises(ValueError, match="outside the original job"):
        inspect_draft_compile(job, job / "inspection")
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        inspect_draft_compile(job, output)
    assert marker.read_text() == "existing evidence"


def test_late_exact_node_k_limit_is_preserved_after_many_older_mentions():
    from qwen35_moe_dflash.draft_compile_inspect import _contexts
    lines = []
    for index in range(40):
        lines.extend([f"I/AI_FMK [{NODE}] inferred pass {index}", *["unrelated detail"] * 12])
    lines.extend(["E/AI_NPUCL matmul_nd_check_support.cc CheckBatchMatMulCubeDim Matrix k greater than 8000 is not supported",
                  f"E/AI_FMK CheckSupported: the op name [{NODE}] type [MatMul] don't support"])
    result = _contexts("\n".join(lines), NODE, [])
    assert len(result["lines"]) <= 96
    assert any("greater than 8000" in row["text"] for row in result["lines"])
    evidence, = result["adjacent_check_evidence"]
    assert evidence["kind"] == "K_LIMIT_CHECK_REJECTION" and evidence["reported_limit"] == 8000
    assert evidence["limit_line"] == len(lines) - 1
    assert evidence["node_rejection_line"] == len(lines)


@pytest.mark.parametrize("intervening", ["unrelated diagnostic", "CheckSupported: the op name [other_node] don't support",
                                        "other.cc IsSupport CheckBatchMatMulCubeSupport failed"])
def test_k_limit_not_associated_across_intervening_messages(intervening):
    from qwen35_moe_dflash.draft_compile_inspect import _contexts
    text = ("Matrix k greater than 8000 is not supported\n" + intervening + "\n" +
            f"CheckSupported: the op name [{NODE}] type [MatMul] don't support")
    assert _contexts(text, NODE, [])["adjacent_check_evidence"] == []


def test_k_limit_association_accepts_exact_known_support_wrapper():
    from qwen35_moe_dflash.draft_compile_inspect import _contexts
    text = ("E/AI_NPUCL matmul_nd_check_support.cc CheckBatchMatMulCubeDim Matrix k greater than 8000 is not supported\n"
            "E/AI_NPUCL batchmatmul_cube_check_support.cc IsSupport(157)::CheckBatchMatMulCubeSupport failed\n"
            f"E/AI_FMK CheckSupported: the op name [{NODE}] type [MatMul] don't support")
    evidence, = _contexts(text, NODE, [])["adjacent_check_evidence"]
    assert evidence["reported_limit"] == 8000
    assert evidence["limit_line"] == 1 and evidence["node_rejection_line"] == 3
    assert evidence["support_wrapper_lines"] == [2]


def test_k_limit_check_evidence_is_rendered_in_readonly_report(retained):
    job, _, output, _ = retained
    log = job / "compilation/compile.log"
    with log.open("a") as stream:
        stream.write("E/AI_NPUCL matmul_nd_check_support.cc CheckBatchMatMulCubeDim Matrix k greater than 8000 is not supported\n")
        stream.write(f"E/AI_FMK CheckSupported: the op name [{NODE}] type [MatMul] don't support\n")
    report_path = job / "compilation/report.json"
    compiled = json.loads(report_path.read_text())
    compiled["compile_log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    _write(report_path, compiled)
    result = inspect_draft_compile(job, output)
    assert result["log"]["node_context"]["adjacent_check_evidence"][0]["reported_limit"] == 8000
    assert "K_LIMIT_CHECK_REJECTION: reported_limit=8000" in (output / "report.txt").read_text()

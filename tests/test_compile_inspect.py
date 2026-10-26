"""Associate real recipe metadata with synthetic OMG logs; never execute a model."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_moe_dflash.compile_inspect import inspect_compile
from qwen35_moe_dflash.ge_recipe import emit_recipe_cpp
from qwen35_moe_dflash.hashing import sha256_file


def _port(name, shape, dtype="float16", **extra):
    return {"port": name, "shape": shape, "dtype": dtype, **extra}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    air, omc = tmp_path / "air", tmp_path / "omc"
    recipe = air / "target/recipe"
    records = [
        {"name": "x", "type": "Data", "inputs": [], "outputs": [_port("y", [2, 3])],
         "attrs": {"index": {"type": "int", "value": 0}}},
        {"name": "axes", "type": "Const", "inputs": [], "outputs": [_port("y", [1], "int32")],
         "constant": {"data": b"\x01\0\0\0"}},
        {"name": "projection", "type": "Relu", "inputs": [_port("x", [2, 3], src=0, src_port="y")],
         "outputs": [_port("y", [2, 3])]},
        {"name": "sum", "type": "ReduceSum", "inputs": [_port("x", [2, 3], src=2, src_port="y"),
         _port("axes", [1], "int32", src=1, src_port="y")], "outputs": [_port("y", [2, 1])],
         "attrs": {"keep_dims": {"type": "bool", "value": True}}},
        {"name": "out", "type": "Sigmoid", "inputs": [_port("x", [2, 1], src=3, src_port="y")],
         "outputs": [_port("y", [2, 1])]},
    ]
    emit_recipe_cpp(records, "target", [0], [(4, 0)], recipe)
    # A diagnostic must not need constant/weight payloads to inspect edges.
    (recipe / "constants.bin").unlink()
    path = recipe / "graph.recipe"
    air_doc = {"artifact_kind": "qwen35-cpro-air-pair", "status": "EXPORTED", "graphs": [
        {"name": "target", "construction_files": [{"path": "target/recipe/graph.recipe",
         "sha256": sha256_file(path), "bytes": path.stat().st_size}]}]}
    (air / "manifest.json").write_text(json.dumps(air_doc))
    log = omc / "target/omg.log"
    log.parent.mkdir(parents=True)
    text = ["start InferShape for graph[target] ....", "TCH iCompiler->Compile begin ..."]
    for name, typ in (("x", "Data"), ("axes", "Const"), ("projection", "Relu"),
                      ("sum", "ReduceSum"), ("out", "Activation"), ("clone_sum", "ReduceSum")):
        text.append(f"CheckSupported: the op name [{name}] type [{typ}] is not supported in npucl store [custom_ascendc_lib]")
    text.append("opDeviceSupport size: 6")
    for name, devices in (("x", "1,"), ("axes", "1,"), ("projection", "0,"),
                          ("sum", "1,"), ("out", "1,0,"), ("clone_sum", "1,")):
        text.append(f"opDeviceSupport: op={name}, devices=[{devices}]")
    text.append('Partition of type: NPUCL is more than 9," " will rollback to CPU')
    log.write_text("\n".join(text))
    compiled = {"artifact_kind": "qwen35-cpro-omc-pair", "status": "FAILED", "air_manifest_sha256": sha256_file(air / "manifest.json"),
                "graphs": [{"name": "target", "compiler_evidence": {"log_sha256": sha256_file(log)}}]}
    (omc / "failure.json").write_text(json.dumps(compiled))
    return air, omc


def test_bound_recipe_support_join_excludes_constants_and_never_guesses_clone(bundle, tmp_path):
    air, omc = bundle
    report = inspect_compile(air, omc, tmp_path / "report")
    assert report["support_snapshot"]["status"] == "complete"
    joined = report["association"]
    assert joined["exact_name_matches"] == 5 and joined["unmatched_names"] == ["clone_sum"]
    assert joined["excluded_boundary_candidates"] == ["x", "axes"]
    rows = {r["name"]: r for r in joined["cpu_only_compute_candidates"]}
    assert rows["sum"]["has_npu_candidate_input"] and rows["sum"]["has_npu_candidate_output"]
    assert rows["sum"]["inputs"][1]["dtype"] == "int32"
    assert rows["clone_sum"]["recipe_match"] == "unmatched" and "inputs" not in rows["clone_sum"]
    assert "not final partitions" in (tmp_path / "report/report.txt").read_text()
    assert report["device_execution"] == "NOT_RUN" and report["runtime_placement"] == "UNVERIFIED"


@pytest.mark.parametrize("kind", ["recipe", "log", "air_manifest"])
def test_mismatched_artifacts_are_rejected_before_output(bundle, tmp_path, kind):
    air, omc = bundle
    path = {"recipe": air / "target/recipe/graph.recipe", "log": omc / "target/omg.log",
            "air_manifest": air / "manifest.json"}[kind]
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        inspect_compile(air, omc, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_ambiguous_compilation_record_refused(bundle, tmp_path):
    air, omc = bundle
    (omc / "manifest.json").write_bytes((omc / "failure.json").read_bytes())
    with pytest.raises(ValueError, match="exactly one OMC"):
        inspect_compile(air, omc, tmp_path / "report")


def test_incomplete_support_list_is_not_promoted_to_complete(bundle, tmp_path):
    air, omc = bundle
    log = omc / "target/omg.log"
    log.write_text(log.read_text().replace("opDeviceSupport size: 6", "opDeviceSupport size: 7"))
    path = omc / "failure.json"
    doc = json.loads(path.read_text())
    doc["graphs"][0]["compiler_evidence"]["log_sha256"] = sha256_file(log)
    path.write_text(json.dumps(doc))
    result = inspect_compile(air, omc, tmp_path / "report")
    assert result["support_snapshot"]["status"] == "partial"
    assert "Support snapshot: partial" in (tmp_path / "report/report.txt").read_text()


def _replace_log(omc, text):
    log = omc / "target/omg.log"
    log.write_text(text)
    path = omc / "failure.json"
    doc = json.loads(path.read_text())
    doc["graphs"][0]["compiler_evidence"]["log_sha256"] = sha256_file(log)
    path.write_text(json.dumps(doc))


def test_known_different_graph_name_cannot_join_same_node_names(bundle, tmp_path):
    air, omc = bundle
    _replace_log(omc, (omc / "target/omg.log").read_text().replace("graph[target]", "graph[draft]"))
    with pytest.raises(ValueError, match="log graph name"):
        inspect_compile(air, omc, tmp_path / "report")


def test_unmatched_unknown_type_is_not_assumed_to_be_compute(bundle, tmp_path):
    air, omc = bundle
    lines = (omc / "target/omg.log").read_text().splitlines()
    _replace_log(omc, "\n".join(line for line in lines if "op name [clone_sum]" not in line))
    result = inspect_compile(air, omc, tmp_path / "report")
    joined = result["association"]
    assert [r["name"] for r in joined["unresolved_cpu_candidates"]] == ["clone_sum"]
    assert [r["name"] for r in joined["cpu_only_compute_candidates"]] == ["sum"]


@pytest.mark.parametrize("source", ["air_manifest", "compilation"])
def test_manifest_mutation_during_analysis_is_detected(bundle, tmp_path, monkeypatch, source):
    from qwen35_moe_dflash import compile_inspect
    air, omc = bundle
    original = compile_inspect.support_snapshot
    path = air / "manifest.json" if source == "air_manifest" else omc / "failure.json"

    def mutate(text):
        result = original(text)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(compile_inspect, "support_snapshot", mutate)
    with pytest.raises(ValueError, match="input changed"):
        inspect_compile(air, omc, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_cli_reads_existing_files_without_numpy_torch_or_compiler(bundle, tmp_path):
    air, omc = bundle
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_model_air.py"
    result = subprocess.run([sys.executable, "-S", str(script), "inspect-compile", "--air", str(air),
                             "--omc", str(omc), "--output", str(tmp_path / "report")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "DIAGNOSTIC_COMPLETE" in result.stdout and "ReduceSum" in result.stdout
    assert json.loads((tmp_path / "report/report.json").read_text())["status"] == "DIAGNOSTIC_COMPLETE"


_EARLY_LOG = """I/AI_FMK (12): start InferShape for graph[target] ....
I/AI_FMK (12): ReshapeFusionPass begin
I/AI_FMK (12): input=__ge/v10/row_mul/left output=result
E/AI_FMK (12): graph.cpp ByPassNode PreCheck failed
E/AI_FMK (12): pass_manager.cpp Pass Run failed: ReshapeFusion
E/AI_FMK (12): AfterInferShape failed
E/AI_FMK (12): GenerateOMCModel failed
E/OMG_TOOL (12): OMG generate offline model failed
"""


def _early_bundle(bundle, *, graph="target", text=_EARLY_LOG):
    air, omc = bundle
    record = json.loads((omc / "failure.json").read_text())
    old_target = record["graphs"][0]
    if graph == "draft":
        recipe = air / "draft/recipe"
        emit_recipe_cpp([{"name": "x", "type": "Data", "inputs": [],
                          "outputs": [_port("y", [2, 3])],
                          "attrs": {"index": {"type": "int", "value": 0}}}],
                        "draft", [0], [(0, 0)], recipe)
        recipe_path = recipe / "graph.recipe"
        air_doc = json.loads((air / "manifest.json").read_text())
        air_doc["graphs"].append({"name": "draft", "construction_files": [
            {"path": "draft/recipe/graph.recipe", "sha256": sha256_file(recipe_path),
             "bytes": recipe_path.stat().st_size}]})
        (air / "manifest.json").write_text(json.dumps(air_doc))
    log = omc / graph / "omg.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text(text.replace("graph[target]", f"graph[{graph}]"))
    record.update(graphs=[] if graph == "target" else [old_target], failed_graph=graph,
                  stage=f"{graph}-omg", log_path=str(log), returncode=1,
                  air_manifest_sha256=sha256_file(air / "manifest.json"))
    (omc / "failure.json").write_text(json.dumps(record))
    return air, omc


def test_old_early_failure_captures_log_hash_and_failure_chain_without_compiled_graph(bundle, tmp_path):
    cleanup = "E/TE_FUSION (12): ParallelCompilationDeinit failed\n" * 180
    air, omc = _early_bundle(bundle, text=_EARLY_LOG + cleanup)
    before = {p: sha256_file(p) for root in (air, omc) for p in root.rglob("*") if p.is_file()}
    report = inspect_compile(air, omc, tmp_path / "report")
    assert report["compilation_record"] == {
        "kind": "early_failure_record", "status": "FAILED", "failed_graph": "target",
        "stage": "target-omg", "requested_graph_has_compiled_record": False}
    assert report["sources"]["log"]["hash_origin"] == "diagnostic_capture"
    assert report["sources"]["log"]["historical_hash_status"] == "NOT_RECORDED"
    assert report["sources"]["log"]["sha256"] == sha256_file(omc / "target/omg.log")
    assert report["support_snapshot"]["status"] == "missing"
    assert report["support_snapshot"]["nodes"] == []
    assert report["association"]["candidate_counts"] == {}
    assert report["association"]["cpu_only_compute_candidates"] == []
    assert report["device_execution"] == "NOT_RUN" and report["runtime_placement"] == "UNVERIFIED"
    rows = report["failure_context"]["lines"]
    assert len(rows) <= 60 and all(len(row["text"]) <= 1000 for row in rows)
    assert {row["line"] for row in rows} >= {4, 5, 6, 8}
    rendered = (tmp_path / "report/report.txt").read_text()
    assert "Support snapshot: missing" in rendered and "ByPassNode PreCheck failed" in rendered
    assert "no historical compiler log hash" in rendered and "not a proven root cause" in rendered
    assert before == {p: sha256_file(p) for p in before}


def test_draft_failure_does_not_replace_prior_target_compiler_hash_binding(bundle, tmp_path):
    air, omc = _early_bundle(bundle, graph="draft")
    draft = inspect_compile(air, omc, tmp_path / "draft-report", graph="draft")
    assert draft["compilation_record"]["kind"] == "early_failure_record"
    assert draft["support_snapshot"]["status"] == "missing"
    target = inspect_compile(air, omc, tmp_path / "target-report")
    assert target["sources"]["log"]["hash_origin"] == "compiler_record"
    assert target["support_snapshot"]["status"] == "complete"
    assert "failure_context" not in target
    (omc / "target/omg.log").write_text("changed")
    with pytest.raises(ValueError, match="recorded compiler evidence"):
        inspect_compile(air, omc, tmp_path / "bad-target")


@pytest.mark.parametrize("stage", ["target-omg", "target-output-check", "target-collect"])
def test_only_attempted_early_compile_stages_are_supported(bundle, tmp_path, stage):
    air, omc = _early_bundle(bundle)
    path = omc / "failure.json"
    record = json.loads(path.read_text())
    record["stage"] = stage
    path.write_text(json.dumps(record))
    assert inspect_compile(air, omc, tmp_path / "report")["compilation_record"]["stage"] == stage


@pytest.mark.parametrize("key,value,match", [
    ("artifact_kind", "different-model", "artifact kind"),
    ("status", "COMPILED", "compilation status"),
    ("failed_graph", "draft", "early failed graph/stage"),
    ("stage", "target-setup", "early failed graph/stage"),
    ("stage", "target-placement-check", "early failed graph/stage"),
    ("stage", "draft-omg", "early failed graph/stage"),
    ("log_path", "target/omg.log", "log_path"),
    ("log_path", "/somewhere/target/omg.log", "log_path"),
    ("graphs", [{"name": "draft"}], "completed graph prefix"),
    ("graphs", None, "graphs list"),
])
def test_early_failure_identity_and_recorded_path_required(bundle, tmp_path, key, value, match):
    air, omc = _early_bundle(bundle)
    path = omc / "failure.json"
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        inspect_compile(air, omc, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_unattempted_draft_cannot_use_target_early_failure_log(bundle, tmp_path):
    air, omc = _early_bundle(bundle, graph="draft")
    path = omc / "failure.json"
    record = json.loads(path.read_text())
    record.update(graphs=[], failed_graph="target", stage="target-omg",
                  log_path=str(omc / "target/omg.log"))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="early failed graph/stage"):
        inspect_compile(air, omc, tmp_path / "report", graph="draft")


@pytest.mark.parametrize("data", [{}, {"log_sha256": "0" * 64}])
def test_existing_graph_record_never_falls_back_to_diagnostic_capture(bundle, tmp_path, data):
    air, omc = bundle
    path = omc / "failure.json"
    record = json.loads(path.read_text())
    record.update(failed_graph="target", stage="target-omg", log_path=str(omc / "target/omg.log"))
    record["graphs"][0]["compiler_evidence"] = data
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="recorded compiler evidence"):
        inspect_compile(air, omc, tmp_path / "report")


def test_unbounded_support_text_never_associates_main_nodes(bundle, tmp_path):
    air, omc = _early_bundle(bundle, text=(
        "start InferShape for graph[target] ....\n"
        "start InferShape for graph[ConstNodeGraph] ....\n"
        "opDeviceSupport size: 1\nopDeviceSupport: op=sum, devices=[1,]\n" + _EARLY_LOG))
    report = inspect_compile(air, omc, tmp_path / "report")
    assert report["support_snapshot"]["status"] == "partial"
    assert report["support_snapshot"]["unbound_support_text_observed"] is True
    assert report["support_snapshot"]["nodes"] == []
    assert report["association"]["candidate_counts"] == {}


def test_known_other_graph_log_rejected_even_without_support_table(bundle, tmp_path):
    air, omc = _early_bundle(bundle, text=_EARLY_LOG.replace("graph[target]", "graph[draft]"))
    with pytest.raises(ValueError, match="log graph name"):
        inspect_compile(air, omc, tmp_path / "report")


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_early_failure_log_cannot_follow_symlinks(bundle, tmp_path, kind):
    air, omc = _early_bundle(bundle)
    if kind == "file":
        log = omc / "target/omg.log"
        other = tmp_path / "other.log"
        log.rename(other)
        log.symlink_to(other)
    else:
        folder = omc / "target"
        other = tmp_path / "other-graph"
        folder.rename(other)
        folder.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="regular OMG log"):
        inspect_compile(air, omc, tmp_path / "report")


def test_early_failure_log_mutation_during_analysis_rejected(bundle, tmp_path, monkeypatch):
    from qwen35_moe_dflash import compile_inspect
    air, omc = _early_bundle(bundle)
    original = compile_inspect.support_snapshot
    def mutate(text):
        result = original(text)
        (omc / "target/omg.log").write_text("changed")
        return result
    monkeypatch.setattr(compile_inspect, "support_snapshot", mutate)
    with pytest.raises(ValueError, match="input changed"):
        inspect_compile(air, omc, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_successful_graph_keeps_historical_log_binding(bundle, tmp_path):
    air, omc = bundle
    record = json.loads((omc / "failure.json").read_text())
    record["status"] = "COMPILED"
    (omc / "failure.json").unlink()
    (omc / "manifest.json").write_text(json.dumps(record))
    report = inspect_compile(air, omc, tmp_path / "report")
    assert report["sources"]["log"]["historical_hash_status"] == "VERIFIED"
    assert report["compilation_record"]["requested_graph_has_compiled_record"] is True


@pytest.mark.parametrize("kind", ["recipe", "air_manifest"])
def test_early_failure_keeps_original_air_and_recipe_hash_guards(bundle, tmp_path, kind):
    air, omc = _early_bundle(bundle)
    path = air / ("target/recipe/graph.recipe" if kind == "recipe" else "manifest.json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        inspect_compile(air, omc, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_generic_failure_context_survives_cleanup_and_bounds_long_lines(bundle, tmp_path):
    text = ("start InferShape for graph[target] ....\nI/AI_FMK (12): a precursor\n"
            + "E/AI_FMK (12): graph compiler failed: " + "x" * 3000 + "\n"
            + "E/OMG_TOOL (12): OMG generate offline model failed\n"
            + "E/TE_FUSION (12): ParallelCompilationDeinit failed\n" * 150)
    air, omc = _early_bundle(bundle, text=text)
    report = inspect_compile(air, omc, tmp_path / "report")
    rows = report["failure_context"]["lines"]
    assert {row["line"] for row in rows} >= {2, 3, 4}
    assert max(row["line"] for row in rows) <= 6
    assert len(rows) <= 60 and max(len(row["text"]) for row in rows) == 1000


def test_early_failure_cli_remains_standard_library_only(bundle, tmp_path):
    air, omc = _early_bundle(bundle)
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_model_air.py"
    result = subprocess.run([sys.executable, "-S", str(script), "inspect-compile", "--air", str(air),
                             "--omc", str(omc), "--output", str(tmp_path / "report")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Support snapshot: missing" in result.stdout and "ByPassNode" in result.stdout

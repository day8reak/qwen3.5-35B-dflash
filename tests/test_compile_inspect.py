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
    compiled = {"status": "FAILED", "air_manifest_sha256": sha256_file(air / "manifest.json"),
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

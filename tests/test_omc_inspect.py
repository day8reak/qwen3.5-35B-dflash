"""Offline OMC inspection contracts using a fake OMG, without Torch or a board."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.omc_inspect import _scan_names, inspect_json, inspect_omc


# Exact relevant lines from the recorded blue OMG help; the active tool's help
# still gates conversion independently on every invocation.
HELP = """--om The model file to be converted to json.
--json The output json file path&name which is converted from a model.
--mode Run mode. 0(default): generate model; 1: model to json; 3: only pre-check;
"""


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    binary = tmp_path / "ddk/tools/tools_omg/master/omg"
    binary.parent.mkdir(parents=True)
    binary.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
config_path = Path(__file__).with_name('fake-config.json')
config = json.loads(config_path.read_text())
with config_path.with_name('fake-calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')
if sys.argv[1:] == ['--help']:
    print(config['help'])
    sys.exit(config.get('help_returncode', 0))
if sys.argv[1:] != ['--mode=1', '--om=input.omc', '--json=model.json']:
    print('unexpected arguments: only existing-OMC conversion is allowed')
    sys.exit(91)
assert Path('input.omc').is_file()
name = Path.cwd().name
behavior = config.get('behaviors', {}).get(name, 'ok')
if behavior == 'fail':
    print('intentional conversion failure')
    sys.exit(7)
if behavior == 'absent':
    sys.exit(0)
if behavior == 'wrong_filename':
    Path('other.json').write_text('{}')
    sys.exit(0)
if behavior == 'symlink':
    Path('model.json').symlink_to(Path('input.omc').resolve())
    sys.exit(0)
if behavior == 'directory':
    Path('model.json').mkdir()
    sys.exit(0)
if behavior == 'malformed':
    Path('model.json').write_text('{not-json')
    sys.exit(0)
document = config.get('documents', {}).get(name, {
    'graph': [{'name': name, 'op': [{'name': 'input', 'type': 'Data'}]}]})
Path('model.json').write_text(json.dumps(document))
''')
    binary.chmod(0o700)
    config = {"help": HELP}
    binary.with_name("fake-config.json").write_text(json.dumps(config))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {"artifact_kind": "qwen35-cpro-omc-pair", "status": "COMPILED",
                "ddk_id": "fake-omc-ddk", "omg_sha256": sha256_file(binary), "graphs": []}
    for name in ("target", "draft"):
        source = bundle / f"{name}.omc"
        source.write_bytes(b"TEST DOUBLE, NOT OMC: insert_cast_4\x00" + name.encode())
        manifest["graphs"].append({"name": name, "omc": source.name,
            "bytes": source.stat().st_size, "sha256": sha256_file(source),
            "compiler_evidence": {"main_partition_events": [{"NPU": 5, "CPU": 6}],
                                  "rollback_events": []}})
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return {"bundle": bundle, "ddk": tmp_path / "ddk", "binary": binary,
            "config": config, "manifest": manifest, "output": tmp_path / "inspection"}


def _configure(case, **kwargs):
    case["config"].update(kwargs)
    case["binary"].with_name("fake-config.json").write_text(json.dumps(case["config"]))


def _inspect(case, **kwargs):
    return inspect_omc(case["bundle"], case["output"], case["ddk"],
                       ddk_id="fake-omc-ddk", **kwargs)


def _calls(case):
    path = case["binary"].with_name("fake-calls.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_existing_pair_uses_active_help_and_only_om_conversion_flags(case):
    before = {p.name: sha256_file(p) for p in case["bundle"].iterdir()}
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert result["device_execution"] == "NOT_RUN"
    assert result["runtime_placement"] == "UNVERIFIED"
    assert [row["argv"] for row in _calls(case)] == [
        ["--help"], ["--mode=1", "--om=input.omc", "--json=model.json"],
        ["--mode=1", "--om=input.omc", "--json=model.json"]]
    assert [Path(row["cwd"]).name for row in _calls(case)[1:]] == ["target", "draft"]
    assert result["source_manifest_sha256"] == before["manifest.json"]
    assert result["omg_sha256"] == sha256_file(case["binary"])
    for graph in result["graphs"]:
        folder = case["output"] / graph["name"]
        assert graph["source_sha256"] == before[f"{graph['name']}.omc"]
        assert graph["main_partition_events"] == [{"NPU": 5, "CPU": 6}]
        assert graph["byte_scan"]["counts"] == {"insert_cast_4": 1}
        assert graph["conversion"]["json_sha256"] == sha256_file(folder / "model.json")
        assert graph["conversion"]["log_sha256"] == sha256_file(folder / "convert.log")
        assert not (folder / "input.omc").exists()
    assert before == {p.name: sha256_file(p) for p in case["bundle"].iterdir()}
    assert json.loads((case["output"] / "report.json").read_text()) == result


@pytest.mark.parametrize("help_text,returncode", [
    ("--mode Run mode. 0: generate model; --om input; --json output", 0),
    ("--mode 1: model to json; --om input", 0),
    ("--mode 0: generate model\n--framework 1: model to json; --om input; --json output", 0),
    ("--mode 1: json to model; --om input; --json output", 0),
    (HELP, 2),
])
def test_unconfirmed_active_help_skips_conversion_but_retains_byte_evidence(case, help_text, returncode):
    _configure(case, help=help_text, help_returncode=returncode)
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_UNSUPPORTED"
    assert [row["argv"] for row in _calls(case)] == [["--help"]]
    assert len(result["graphs"]) == 2
    assert all(row["conversion"]["status"] == "SKIPPED_HELP_UNCONFIRMED" for row in result["graphs"])
    assert all(row["byte_scan"]["counts"] == {"insert_cast_4": 1} for row in result["graphs"])


@pytest.mark.parametrize("change", ["ddk_identity", "compiler_hash", "source_hash", "source_missing"])
def test_identity_or_source_mismatch_refuses_before_output_or_help(case, change):
    if change == "ddk_identity":
        case["manifest"]["ddk_id"] = "other-ddk"
    elif change == "compiler_hash":
        case["manifest"]["omg_sha256"] = "0" * 64
    elif change == "source_hash":
        source = case["bundle"] / "draft.omc"
        source.write_bytes(b"x" * source.stat().st_size)
    else:
        (case["bundle"] / "draft.omc").unlink()
    (case["bundle"] / "manifest.json").write_text(json.dumps(case["manifest"]))
    with pytest.raises(ValueError):
        _inspect(case)
    assert not case["output"].exists()
    assert not _calls(case)


def test_target_conversion_failure_does_not_skip_draft(case):
    _configure(case, behaviors={"target": "fail"})
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    assert [(row["name"], row["conversion"]["status"]) for row in result["graphs"]] == [
        ("target", "FAILED"), ("draft", "CONVERTED")]
    assert result["graphs"][0]["conversion"]["returncode"] == 7
    assert "intentional conversion failure" in (case["output"] / "target/convert.log").read_text()
    assert len(_calls(case)) == 3


@pytest.mark.parametrize("behavior", ["absent", "wrong_filename", "symlink", "directory", "malformed"])
def test_zero_exit_without_usable_json_is_not_success_and_draft_continues(case, behavior):
    _configure(case, behaviors={"target": behavior})
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    assert result["graphs"][0]["conversion"]["status"] == "FAILED"
    assert result["graphs"][0]["conversion"]["error"]
    assert result["graphs"][1]["conversion"]["status"] == "CONVERTED"


def test_json_budget_retains_hash_bound_file_without_parsing(case):
    result = _inspect(case, max_json_bytes=8)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    for graph in result["graphs"]:
        assert graph["conversion"]["status"] == "JSON_RETAINED_NOT_PARSED"
        assert "inspection" not in graph
        converted = case["output"] / graph["name"] / "model.json"
        assert converted.stat().st_size == graph["conversion"]["json_bytes"] > 8
        assert sha256_file(converted) == graph["conversion"]["json_sha256"]


@pytest.mark.parametrize("document", [None, [], {"name": "insert_cast_4", "type": "CastT"},
                                       {"graph": [{"op": [{"name": "missing_type"}]}]}])
def test_unrecognized_json_does_not_claim_complete_inspection(case, document):
    _configure(case, documents={"target": document})
    result = _inspect(case)
    assert result["status"] == "DIAGNOSTIC_PARTIAL"
    target = result["graphs"][0]
    assert target["conversion"]["status"] == "JSON_SCHEMA_UNRECOGNIZED"
    assert target["inspection"]["schema_status"] == "NO_RECOGNIZED_OP_OBJECTS"


def test_casts_keep_raw_attributes_and_resolve_only_explicit_same_scope_names():
    cast = {"name": "insert_cast_4", "type": "CastT", "input": ["producer:0", "duplicate:0"],
            "src_name": ["producer", "duplicate", "outside"], "src_index": [0, 0, 0],
            "attr": {"src_dtype": {"i": 0}, "dst_dtype": {"i": 1},
                     "metadata": {"name": "not_an_op", "type": "CastT"}},
            "input_desc": [{"dtype": 1}], "output_desc": [{"dtype": 0}]}
    inner = {"name": "cpu_graph", "op": [
        {"name": "producer", "type": "MatMul", "output_desc": [{"dtype": 1}]},
        {"name": "duplicate", "type": "Data"}, {"name": "duplicate", "type": "Const"},
        cast, {"name": "consumer", "type": "Add", "srcName": ["insert_cast_4"]}]}
    document = {"graph": [{"op": [
        {"name": "outside", "type": "Data"},
        {"name": "subgraph", "type": "GraphOp", "attr": {"body": {"g": inner}}}]}]}
    result = inspect_json(document)
    assert result["named_typed_object_count"] == 7
    assert result["cast_object_count"] == 1
    row = result["casts"][0]
    assert row["raw_node"] == cast
    assert row["json_path"] == "$/graph/0/op/1/attr/body/g/op/3"
    assert row["op_array_path"] == "$/graph/0/op/1/attr/body/g/op"
    assert [parent["source_name"] for parent in row["producers"]] == ["producer"]
    assert [child["name"] for child in row["consumers"]] == ["consumer"]
    assert row["unresolved_inputs"] == ["duplicate", "outside"]
    assert row["raw_input_references"] == cast["input"]


def test_input_strings_are_retained_without_guessing_edge_encoding():
    result = inspect_json({"op": [{"name": "producer", "type": "Data"},
        {"name": "cast", "type": "Cast", "input": ["producer:0"]},
        {"name": "consumer", "type": "Add", "input": ["cast:0"]}]})
    row = result["casts"][0]
    assert row["raw_input_references"] == ["producer:0"]
    assert row["producers"] == row["consumers"] == []


def test_cast_name_is_preserved_across_nested_graph_lists_and_report_limit():
    operations = [{"name": f"cast_{i}", "type": "CastT"} for i in range(205)]
    operations.append({"name": "container", "type": "GraphOp", "attr": {"branches": {
        "list": {"g": [{"op": [{"name": "insert_cast_4", "type": "CastT"}]}]}}}})
    result = inspect_json({"graph": [{"op": operations}]})
    assert result["cast_object_count"] == 206
    assert result["reported_cast_count"] == len(result["casts"]) == 200
    failing = [row for row in result["casts"] if row["name"] == "insert_cast_4"]
    assert len(failing) == 1
    assert "/attr/branches/list/g/0/op/0" in failing[0]["json_path"]


def test_absent_casts_are_not_evidence_of_load_time_creation(case):
    result = _inspect(case)
    assert all(row["inspection"]["cast_object_count"] == 0 for row in result["graphs"])
    assert all(row["byte_scan"]["counts"] for row in result["graphs"])
    text = (case["output"] / "report.txt").read_text()
    assert "missing names do not prove Load-time insertion" in text
    assert "Byte matches are strings, not decoded nodes" in text
    assert "opaque payload coverage is unknown" in result["graphs"][0]["inspection"]["coverage"]


def test_report_previews_large_raw_attributes_but_keeps_complete_hashed_json(case):
    cast = {"name": "insert_cast_4", "type": "CastT", "attr": {"opaque": {"s": "x" * 100_000}}}
    _configure(case, documents={"target": {"graph": [{"op": [cast]}]}})
    result = _inspect(case)
    graph = result["graphs"][0]
    converted = case["output"] / "target/model.json"
    assert json.loads(converted.read_text())["graph"][0]["op"][0] == cast
    assert graph["conversion"]["json_sha256"] == sha256_file(converted)
    assert len(json.dumps(graph["inspection"])) < 5_000
    assert len((case["output"] / "report.txt").read_text()) < 10_000
    preview = graph["inspection"]["casts"][0]["raw_node"]["attr"]["opaque"]["s"]
    assert preview["length"] == 100_000


def test_neighbor_report_is_limited_without_inferring_missing_edges():
    parents = [{"name": f"producer_{i}", "type": "Data"} for i in range(20)]
    missing = [f"missing_{i}" for i in range(20)]
    cast = {"name": "insert_cast_4", "type": "CastT",
            "srcName": [row["name"] for row in parents] + missing}
    consumers = [{"name": f"consumer_{i}", "type": "Add", "src_name": ["insert_cast_4"]}
                 for i in range(20)]
    result = inspect_json({"op": parents + [cast] + consumers})
    row = result["casts"][0]
    assert len(row["producers"]) == len(row["consumers"]) == row["neighbor_limit"] == 8
    assert row["unresolved_inputs"] == missing[:8]
    assert row["producer_count"] == row["consumer_count"] == row["unresolved_input_count"] == 20
    assert row["raw_input_encoding"] == "NOT_INTERPRETED"


def test_streaming_scan_handles_chunk_and_overlap_boundaries_once(tmp_path):
    boundary = 1 << 20
    data = bytearray(b"x" * (2 * boundary + 200))
    inserts = [(0, b"insert_cast_4"), (boundary - 128 - 5, b"insert_cast_24"),
               (boundary - 5, b"insert_cast_123456789"),
               (2 * boundary - 128, b"insert_cast_4"),
               (len(data) - len(b"insert_cast_4"), b"insert_cast_4")]
    for offset, value in inserts:
        data[offset:offset + len(value)] = value
    path = tmp_path / "scan.omc"
    path.write_bytes(data)
    result = _scan_names(path)
    assert result["counts"] == {"insert_cast_4": 3, "insert_cast_24": 1,
                                 "insert_cast_123456789": 1}
    assert result["example_byte_offsets"]["insert_cast_4"] == [0, 2 * boundary - 128, inserts[-1][0]]
    assert result["example_byte_offsets"]["insert_cast_123456789"] == [boundary - 5]


def test_cli_runs_under_python_without_site_packages_or_torch(case, tmp_path):
    guard = tmp_path / "import-guard"
    guard.mkdir()
    (guard / "torch.py").write_text("raise AssertionError('inspect-omc must not import Torch')\n")
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/direct_model_air.py"
    env = dict(os.environ, PYTHONPATH=str(guard), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-S", str(script), "inspect-omc", "--omc", str(case["bundle"]),
               "--ddk-root", str(case["ddk"]), "--ddk-id", "fake-omc-ddk",
               "--output", str(case["output"])]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["status"] == "DIAGNOSTIC_COMPLETE"
    assert len(_calls(case)) == 3

"""Enforce source FP32 regions against actual encoded CPUCL task graphs."""
import copy
import hashlib
import json
import shutil
import struct
import subprocess

import pytest

from qwen35_moe_dflash import draft_compile_precision as dp
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.draft_ir_inspect import _enum_fields
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_draft import build_draft
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.ge_recipe_read import read_recipe
from qwen35_moe_dflash.omc_wire import Schema, load_schema
from qwen35_moe_dflash.omc_container import read_container
from test_draft_ir_inspect import _blob, _int
from test_draft_precision_inspect import _encoded_graph
from test_draft_precision_omc import _container, _owner
import test_draft_ir_inspect as schema_fixture


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("precision planning/verifying must not run tools"))
    # Extend the existing synthetic schema; its FLOAT=1/HALF=7 remain unlike GE.
    original = schema_fixture._schema()
    values = [("UNDEFINED", 0), ("FLOAT", 1), ("HALF", 7)]
    def enum(items):
        return _blob(5, _blob(1, "DataType") + b"".join(_blob(2, _blob(1, n) + _int(2, v)) for n, v in items))
    expanded = original.replace(enum(values), enum(values + [("INT32", 3), ("INT64", 4), ("BOOL", 5)]))
    monkeypatch.setattr(schema_fixture, "_schema", lambda: expanded)
    library = tmp_path / "selected.so"
    schema_fixture._library(library)
    graph = build_draft(Qwen35DFlashConfig.from_dict(configurations(tiny=False)[1]), 8)
    emit_cpp(graph, {n: tmp_path / "absent-weights" / n for n in graph.weights},
             external_weights=True, recipe_dir=tmp_path / "recipe", fold_constant_arithmetic=False)
    recipe = tmp_path / "recipe/graph.recipe"
    records = read_recipe(recipe)["records"]
    plan = dp.collect_precision_plan(recipe)
    schema = load_schema(library)
    compiled = schema.decode(_encoded_graph(records, graph_name="CPU_0"), "GraphDef")
    cpu = [{"owner": "CPU_0", "cl_name": "CPUCL", "graph": compiled, "descriptor_enum_fields": _enum_fields(schema)}]
    return {"root": tmp_path, "library": library, "recipe": recipe, "records": records, "plan": plan, "cpu": cpu}


def _node(case, name):
    return next(n for n in case["cpu"][0]["graph"]["op"] if n["name"] == name)


def _required(case, kind):
    return next(n for n in case["plan"]["required_nodes"] if n["type"] == kind)


def _errors(report):
    return [error["kind"] for row in report["nodes"] for error in row["errors"]]


def _omc(case, *, cpu_name="CPU_0", cl="CPUCL", overlap=False, outside=False):
    cpu = _encoded_graph(case["records"], graph_name=cpu_name)
    opaque = b"not protobuf\x00\xffNPU task"
    offset = 0 if overlap else len(cpu) + len(opaque) + 1 if outside else len(cpu)
    main = _blob(7, _blob(1, "compiled") + _blob(6, _owner("CPU_0", cl, 0, len(cpu)))
                 + _blob(6, _owner("NPU_0", "NPUCL", offset, len(opaque))))
    path = case["root"] / "model.omc"
    path.write_bytes(_container(main, cpu + opaque))
    return path


def test_plan_covers_all_26_norms_attention_and_rope_without_cast_names(case):
    plan = case["plan"]
    counts = plan["counts"]["types"]
    assert counts["ReduceMean"] == counts["Rsqrt"] == 26
    assert counts["SoftmaxV2"] == counts["SelectV2"] == 6
    assert counts["Cos"] == counts["Sin"] == 1
    assert counts["Mul"] == 53
    assert counts["Power"] == 32
    assert plan["counts"]["required_nodes"] == 153
    assert plan["counts"]["boundary_casts"] == 79
    assert not {"Cast", "CastT", "Const", "FileConstant", "Data"} & counts.keys()
    assert not str(case["root"]) in json.dumps(plan)
    assert plan["recipe"]["sha256"] == hashlib.sha256(case["recipe"].read_bytes()).hexdigest()
    assert not (case["root"] / "absent-weights").exists()


def test_plan_is_identical_after_relocation(case):
    moved = case["root"] / "moved/graph.recipe"
    moved.parent.mkdir()
    shutil.copyfile(case["recipe"], moved)
    assert dp.collect_precision_plan(moved) == case["plan"]


def test_cpu_fp32_descriptors_and_actual_producers_pass(case):
    report = dp.verify_document(case["plan"], case["cpu"])
    assert report["status"] == "PASSED", [(r["name"], r["errors"]) for r in report["nodes"] if r["errors"]]
    assert report["matched_nodes"] == 153
    assert report["error_count"] == 0
    assert all(e["status"] == "MATCHED" for n in report["nodes"] for e in n["fp32_input_edges"])


@pytest.mark.parametrize("kind,direction", [("Mul", "input_desc"), ("Rsqrt", "output_desc"),
    ("SoftmaxV2", "output_desc"), ("Cos", "output_desc"), ("Power", "input_desc")])
def test_fp16_compiled_port_blocks_the_policy(case, kind, direction):
    source = _required(case, kind)
    _node(case, source["name"])[direction][0]["dtype"] = 7
    report = dp.verify_document(case["plan"], case["cpu"])
    assert report["status"] == "FAILED"
    assert "DESCRIPTOR_DIFFERS" in _errors(report)


def test_matching_cpu_mean_alone_cannot_hide_surrounding_npu_half(case):
    source = _required(case, "ReduceMean")
    producer = source["inputs"][0]["producer"]
    case["cpu"][0]["graph"]["op"] = [n for n in case["cpu"][0]["graph"]["op"] if n["name"] != producer]
    report = dp.verify_document(case["plan"], case["cpu"])
    assert report["status"] == "FAILED"
    assert "CPU_NODE_MISSING" in _errors(report)
    assert "FP32_PRODUCER_UNRESOLVED" in _errors(report)


def test_cpu_upcast_cannot_conceal_reduced_mean_crossing_half_boundary(case):
    source = _required(case, "Rsqrt")
    target = _node(case, source["name"])
    desc = copy.deepcopy(target["input_desc"][0])
    case["cpu"][0]["graph"]["op"].append({"name": "reimported_after_half", "type": "Data", "output_desc": [desc]})
    target["input"][0] = "reimported_after_half:0"
    report = dp.verify_document(case["plan"], case["cpu"])
    assert report["status"] == "FAILED"
    assert "FP32_REQUIRED_EDGE_CHANGED" in _errors(report)


def test_changed_boundary_cast_name_is_not_a_missing_requirement(case):
    source = _required(case, "Mul")
    cast = source["inputs"][0]["producer"]
    _node(case, cast)["name"] = "compiler_inserted_cast"
    for n in case["cpu"][0]["graph"]["op"]:
        n["input"] = ["compiler_inserted_cast:0" if v == cast + ":0" else v for v in n.get("input", [])]
    assert dp.verify_document(case["plan"], case["cpu"])["status"] == "PASSED"


def test_explicit_inputs_take_precedence_over_stale_source_metadata(case):
    for node in case["cpu"][0]["graph"]["op"]:
        if node.get("input"):
            node["src_name"] = ["old_preoptimization_name"] * len(node["input"])
            node["src_index"] = [99] * len(node["input"])
    assert dp.verify_document(case["plan"], case["cpu"])["status"] == "PASSED"


def test_source_metadata_is_used_only_when_explicit_inputs_are_absent(case):
    for node in case["cpu"][0]["graph"]["op"]:
        if node.get("input"):
            edges = [value.rsplit(":", 1) for value in node.pop("input")]
            node["src_name"] = [name for name, _ in edges]
            node["src_index"] = [int(port) for _, port in edges]
    assert dp.verify_document(case["plan"], case["cpu"])["status"] == "PASSED"


@pytest.mark.parametrize("invalid", [[], ["malformed"], ["source:0", "extra:0"], "not-a-list"])
def test_malformed_explicit_inputs_cannot_fall_back_to_stale_metadata(case, invalid):
    source = _required(case, "Rsqrt")
    node = _node(case, source["name"])
    name, port = node["input"][0].rsplit(":", 1)
    node.update(src_name=[name], src_index=[int(port)], input=invalid)
    report = dp.verify_document(case["plan"], case["cpu"])
    assert report["status"] == "FAILED"
    assert "FP32_INPUT_EDGE_UNRESOLVED" in _errors(report)


def test_control_dependency_does_not_shift_explicit_data_port_indices(case):
    node = _node(case, _required(case, "Mul")["name"])
    node["input"].insert(0, "control_node:-1")
    assert dp.verify_document(case["plan"], case["cpu"])["status"] == "PASSED"


@pytest.mark.parametrize("damage", ["omitted_dtype", "wrong_shape", "missing_port", "renamed", "duplicate"])
def test_missing_or_ambiguous_compiled_precision_is_never_passed(case, damage):
    source = _required(case, "ReduceMean")
    node = _node(case, source["name"])
    if damage == "omitted_dtype":
        node["output_desc"][0].pop("dtype")
    elif damage == "wrong_shape":
        node["output_desc"][0]["shape"]["dim"][-1] = 17
    elif damage == "missing_port":
        node["input_desc"].pop()
    elif damage == "renamed":
        node["name"] = "unproven_fused_name"
    else:
        case["cpu"][0]["graph"]["op"].append(copy.deepcopy(node))
    assert dp.verify_document(case["plan"], case["cpu"])["status"] == "FAILED"


def test_same_source_name_in_other_cpu_graph_is_ambiguous(case):
    duplicate = copy.deepcopy(case["cpu"][0])
    duplicate["owner"] = duplicate["graph"]["name"] = "CPU_1"
    report = dp.verify_document(case["plan"], case["cpu"] + [duplicate])
    assert report["status"] == "FAILED"
    assert set(_errors(report)) == {"CPU_NODE_AMBIGUOUS"}


def test_actual_omc_cpu_task_decodes_once_and_never_decodes_npu_payload(case, monkeypatch):
    path = _omc(case)
    original, calls = Schema.decode, []
    def decode(self, raw, name):
        calls.append(name)
        return original(self, raw, name)
    monkeypatch.setattr(Schema, "decode", decode)
    before = path.read_bytes()
    report = dp.verify_compiled_precision(path, case["library"], case["plan"])
    assert report["status"] == "PASSED", report
    assert calls == ["ModelDef", "GraphDef"]
    assert report["graphop_counts"] == {"CPUCL": 1, "NPUCL": 1}
    assert report["omc"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def _append_trailer(path, raw, kind=8):
    original = path.read_bytes()
    parts = read_container(path)["partitions"]
    table = [(p["type"], p["table_word"], p["bytes"]) for p in parts]
    table.append((kind, sum(p["bytes"] for p in parts), len(raw)))
    payload = struct.pack("<I", len(table)) + b"".join(struct.pack("<III", *row) for row in table)
    payload += original[parts[0]["offset"]:] + raw
    header = bytearray(original[:256])
    struct.pack_into("<I", header, 0x4c, len(payload))
    path.write_bytes(bytes(header) + payload)


def _observed_weight_info():
    # Same keys/field types as the actual Next6.1.1 complete-tiny draft OMC.
    return {"WeightInfoLists": [{"subGraphName": "CPU_0", "variableSkip": False,
        "WeightBlockInfo": [{"hash": 12612883901365648434, "offset": 0,
                             "size": 16, "type": 0, "weightIdx": 0}]}]}


def test_observed_type8_weight_info_trailer_is_separate_from_cpu_precision(case, monkeypatch):
    path = _omc(case)
    trailer = json.dumps(_observed_weight_info(), separators=(",", ":")).encode()
    _append_trailer(path, trailer)
    original, calls = Schema.decode, []
    def decode(self, raw, name):
        calls.append(name)
        return original(self, raw, name)
    monkeypatch.setattr(Schema, "decode", decode)
    report = dp.verify_compiled_precision(path, case["library"], case["plan"])
    assert report["status"] == "PASSED", report
    assert report["matched_nodes"] == 153
    assert calls == ["ModelDef", "GraphDef"]
    assert report["auxiliary_partitions"] == [{"type": 8, "bytes": len(trailer),
        "sha256": hashlib.sha256(trailer).hexdigest(), "format": "WeightInfoLists JSON",
        "status": "RECOGNIZED_NOT_USED_FOR_PRECISION"}]


@pytest.mark.parametrize("damage", ["unknown_kind", "wrong_keys", "duplicate_key", "nonfinite", "float_offset"])
def test_unrecognized_auxiliary_partition_is_rejected_before_cpu_decode(case, monkeypatch, damage):
    path = _omc(case)
    data = _observed_weight_info()
    kind = 9 if damage == "unknown_kind" else 8
    if damage == "wrong_keys":
        data = {"OtherInfo": []}
    if damage == "float_offset":
        data["WeightInfoLists"][0]["WeightBlockInfo"][0]["offset"] = 0.0
    trailer = json.dumps(data).encode()
    if damage == "duplicate_key":
        trailer = b'{"WeightInfoLists": [], "WeightInfoLists": []}'
    if damage == "nonfinite":
        trailer = b'{"WeightInfoLists": NaN}'
    _append_trailer(path, trailer, kind)
    monkeypatch.setattr(Schema, "decode", lambda *args: pytest.fail("unknown metadata must stop before graph decode"))
    report = dp.verify_compiled_precision(path, case["library"], case["plan"])
    assert report["status"] == "NOT_VERIFIED"
    assert not report["nodes"]


@pytest.mark.parametrize("damage", ["overlap", "outside"])
def test_invalid_task_bounds_reject_before_any_cpu_graph_decode(case, monkeypatch, damage):
    path = _omc(case, **{damage: True})
    original, calls = Schema.decode, []
    def decode(self, raw, name):
        calls.append(name)
        return original(self, raw, name)
    monkeypatch.setattr(Schema, "decode", decode)
    report = dp.verify_compiled_precision(path, case["library"], case["plan"])
    assert report["status"] == "NOT_VERIFIED"
    assert calls == ["ModelDef"]
    assert not report["nodes"]


def test_wrong_cpu_graph_name_is_unverified(case):
    path = _omc(case, cpu_name="wrong")
    report = dp.verify_compiled_precision(path, case["library"], case["plan"])
    assert report["status"] == "NOT_VERIFIED"
    assert not report["nodes"]


def test_unknown_task_library_is_opaque_and_cannot_satisfy_cpu_policy(case, monkeypatch):
    path = _omc(case, cl="UNRECOGNIZED")
    original, calls = Schema.decode, []
    def decode(self, raw, name):
        calls.append(name)
        return original(self, raw, name)
    monkeypatch.setattr(Schema, "decode", decode)
    report = dp.verify_compiled_precision(path, case["library"], case["plan"])
    assert report["status"] == "FAILED"
    assert calls == ["ModelDef"]
    assert report["matched_nodes"] == 0


@pytest.mark.parametrize("which", ["omc", "schema"])
def test_late_identity_failure_discards_all_decoded_conclusions(case, monkeypatch, which):
    path = _omc(case)
    original = dp.verify_document
    def changed(*args):
        report = original(*args)
        target = path if which == "omc" else case["library"]
        with target.open("ab") as stream:
            stream.write(b"changed")
        return report
    monkeypatch.setattr(dp, "verify_document", changed)
    report = dp.verify_compiled_precision(path, case["library"], case["plan"])
    assert report["status"] == "NOT_VERIFIED"
    assert not report["nodes"]
    assert "changed" in report["reason"]

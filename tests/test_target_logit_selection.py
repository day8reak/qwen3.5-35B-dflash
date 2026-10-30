"""Bounded encoded AIR/control diagnostics never execute the original model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

from google.protobuf import descriptor_pb2
import pytest

from qwen35_moe_dflash import target_logit_selection as audit
from qwen35_moe_dflash.omc_wire import load_schema
from qwen35_moe_dflash.target_prefill_smoke_plan import _source
from test_draft_ir_inspect import _attr, _blob, _desc, _int
from test_original_air_inspect import _node, selected
from test_target_reshape_metadata import op, schemas
import test_draft_ir_inspect as fixture_schema


@pytest.fixture(autouse=True)
def _named_list_schema(schemas, monkeypatch):
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(fixture_schema._schema())
    value = next(row for row in descriptor.message_type if row.name == "ListValue").field.add()
    value.name, value.number, value.type, value.label = "s", 2, 9, 3
    raw = descriptor.SerializeToString()
    monkeypatch.setattr(fixture_schema, "_schema", lambda: raw)
    ddk, _ = schemas
    fixture_schema._library(ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so")


def named_ports(names=("x", "indices", "axis"), values=(0, 1, 2)):
    return (_attr("_input_name_key", _blob(1, b"".join(_blob(2, name) for name in names)))
            + _attr("_input_name_value", _blob(1, b"".join(_int(3, value) for value in values))))


def graph(*, axis=1, axis_dtype=42, axis_shape=(), source_shape=(1, 64, 2048),
          consumer_shape="same", passthrough=None, external_axis=False, axis_data=None,
          named=True, duplicate=False, output_control="arg1238_1", optional_ancestor=False):
    desc = _desc(dtype=1, shape=source_shape)
    control_desc = _desc(dtype=42, shape=(1,))
    axis_desc = _desc(dtype=axis_dtype, shape=axis_shape)
    selected_desc = _desc(dtype=1, shape=(1, 1, 2048))
    logits = _desc(dtype=1, shape=(1, 1, 248320))
    nodes = [_node("arg1238_1", "Data", [1], dtype=42),
             _node("other_control", "Data", [1], dtype=42),
             _node("hidden", "Data", source_shape, dtype=1)]
    if optional_ancestor:
        nodes[-1] = op("hidden", "ExampleOptionalOp", ["other_control:0", ""],
            [control_desc, control_desc], [desc])
    index = output_control + ":0"
    if passthrough:
        dtype = 91 if passthrough == "Cast" else 42
        target = _desc(dtype=dtype, shape=(1,))
        nodes.append(op("index_carrier", passthrough, [index], [control_desc], [target]))
        index = "index_carrier:0"
        control_desc = target
    if external_axis:
        nodes.append(_node("axis", "FileConstant", axis_shape, dtype=axis_dtype)
                     + _attr("file_path", _blob(2, "./private/never-open-this")))
    else:
        raw = axis_data if axis_data is not None else struct.pack("<i" if axis_dtype == 42 else "<q", axis)
        nodes.append(_node("axis", "Const", axis_shape, dtype=axis_dtype)
                     + _attr("value", _blob(12, _blob(1, axis_desc) + _blob(2, raw))))
    consumer = desc if consumer_shape == "same" else _desc(dtype=1, shape=consumer_shape)
    gather = op("select_row", "GatherV2", ["hidden:0", index, "axis:0"],
                [consumer, control_desc, axis_desc], [selected_desc])
    nodes.append(gather + (named_ports() if named else b""))
    if duplicate:
        nodes.append(op("select_row_again", "GatherV2", ["hidden:0", index, "axis:0"],
                        [consumer, control_desc, axis_desc], [selected_desc]) + named_ports())
        nodes.append(op("combine", "Add", ["select_row:0", "select_row_again:0"],
                        [selected_desc, selected_desc], [selected_desc]))
    head_input = "combine:0" if duplicate else "select_row:0"
    nodes.append(_node("head_weight", "FileConstant", [2048, 248320], dtype=1)
                 + _attr("file_path", _blob(2, "./weights/do-not-open")))
    nodes.append(op("head", "MatMul", [head_input, "head_weight:0"],
                    [selected_desc, _desc(dtype=1, shape=(2048, 248320))], [logits]))
    nodes.append(op("NetOutput", "NetOutput", ["head:0"], [logits], []))
    return _blob(1, "source") + b"".join(_blob(6, node) for node in nodes)


def analyze(schemas, raw, index=64):
    ddk, _ = schemas
    schema = load_schema(ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so")
    document = schema.decode(raw, "GraphDef")
    return audit.analyze_logit_selection(document, raw, schema, index)


def test_optional_input_holes_keep_port_positions_without_becoming_dependencies(schemas):
    report = analyze(schemas, graph(optional_ancestor=True))
    assert report["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED", report
    assert audit._inputs({"input": ["other_control:0", ""]}) == ["other_control:0", ""]


@pytest.mark.parametrize("axis_dtype,axis_shape", [(42, ()), (42, (1,)), (91, ())])
def test_exact_axis_data_and_actual_extent_prove_positive_index64_exceeds_axis64(schemas, axis_dtype, axis_shape):
    report = analyze(schemas, graph(axis_dtype=axis_dtype, axis_shape=axis_shape))
    assert report["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED", report
    row, = report["candidates"]
    assert row["serialized_axis_extent"] == 64
    assert row["nonnegative_valid_index_range"] == [0, 63]
    assert row["selected_index"] == 64 and row["axis"]["value"] == 1
    assert row["index_path"][-1]["node"] == "arg1238_1"
    assert "runtime out-of-range behavior" in report["scope"]


@pytest.mark.parametrize("kind", ["Identity", "Squeeze", "Unsqueeze", "Cast"])
def test_only_value_preserving_integer_carriers_keep_source_control(kind, schemas):
    report = analyze(schemas, graph(passthrough=kind))
    assert report["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED", report
    assert report["candidates"][0]["index_path"][0]["type"] == kind


@pytest.mark.parametrize("index,expected", [(0, "LOGIT_INDEX_WITHIN_SERIALIZED_EXTENT"),
    (63, "LOGIT_INDEX_WITHIN_SERIALIZED_EXTENT"), (-1, "LOGIT_SELECTION_OBSERVED_RANGE_UNVERIFIED")])
def test_other_indices_are_observations_not_accuracy_acceptance(schemas, index, expected):
    assert analyze(schemas, graph(), index)["status"] == expected


@pytest.mark.parametrize("options,reason", [
    ({"source_shape": ()}, "absent/dynamic"),
    ({"consumer_shape": (1, 63, 2048)}, "dimensions disagree"),
    ({"axis": 3}, "outside the serialized params rank"),
    ({"external_axis": True}, "embedded Const"),
    ({"axis_data": b"longer-than-one-integer"}, "exactly4 or8"),
    ({"passthrough": "Sub"}, "unproved value transform"),
    ({"named": False}, "port-name map"),
    ({"output_control": "other_control"}, "unproved value transform"),
])
def test_missing_extent_arithmetic_external_axis_or_missing_semantics_never_infers64(schemas, options, reason):
    report = analyze(schemas, graph(**options))
    assert report["status"] in {"LOGIT_SELECTION_UNVERIFIED", "LOGIT_SELECTION_OBSERVED_RANGE_UNVERIFIED"}, report
    assert any(reason in row.get("reason", "") for row in [*report["candidates"], *report["other_gathers"]])
    assert not any(row.get("index_range") == "OUTSIDE_SERIALIZED_AXIS_EXTENT" for row in report["candidates"])


def test_duplicate_control_gathers_are_not_resolved_by_first_match(schemas):
    report = analyze(schemas, graph(duplicate=True))
    assert report["control_gather_candidate_count"] == 2
    assert report["status"] == "LOGIT_SELECTION_UNVERIFIED"


def test_negative_axis_is_normalized_only_with_observed_rank(schemas):
    report = analyze(schemas, graph(axis=-2))
    assert report["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED"
    assert report["candidates"][0]["normalized_axis"] == 1


def test_source_control_requires_exact_successful_file_sha_identity_and_unique_name(tmp_path, monkeypatch):
    execution = tmp_path / "execution"
    execution.mkdir()
    control = tmp_path / "output_pos.bin"
    control.write_bytes(struct.pack("<i", 64))
    job = {"compilation_report_sha256": "a" * 64, "model_sha256": "b" * 64}
    row = {"compiled_candidates": ["arg1238_1"], "dtype": "INT32", "bytes": 4,
        "source_filename": "output_pos.bin", "semantic_slot_identity": "UNIQUE_TYPED_ND_VIEW", "source": str(control)}
    proof = {"inputs": [row], "source_inputs": [_source(control, 4)]}
    (execution / "plan.json").write_text(json.dumps(proof))
    monkeypatch.setattr(audit, "_bound_execution", lambda execution, bindings: ({}, {}, job))
    observed = audit._observed_control(execution, job, [])
    assert observed["value"] == 64 and observed["source_input_name"] == "arg1238_1"
    assert observed["sha256"] == hashlib.sha256(struct.pack("<i", 64)).hexdigest()
    control.write_bytes(struct.pack("<i", 63))
    with pytest.raises(ValueError, match="bytes/identity"):
        audit._observed_control(execution, job, [])


@pytest.mark.parametrize("fault", ["ambiguous", "different_model", "different_source", "wrong_type"])
def test_source_control_binding_rejects_ambiguous_or_foreign_saved_proof(tmp_path, monkeypatch, fault):
    execution = tmp_path / "execution"
    execution.mkdir()
    control = tmp_path / "output_pos.bin"
    control.write_bytes(struct.pack("<i", 64))
    job = {"compilation_report_sha256": "a" * 64, "model_sha256": "b" * 64}
    linked = deepcopy(job)
    row = {"compiled_candidates": ["arg1238_1"], "dtype": "INT32", "bytes": 4,
        "source_filename": "output_pos.bin", "semantic_slot_identity": "UNIQUE_TYPED_ND_VIEW", "source": str(control)}
    source = _source(control, 4)
    if fault == "ambiguous": row["compiled_candidates"].append("another")
    elif fault == "different_model": linked["model_sha256"] = "c" * 64
    elif fault == "different_source": source["sha256"] = "d" * 64
    else: row["dtype"] = "INT64"
    (execution / "plan.json").write_text(json.dumps({"inputs": [row], "source_inputs": [source]}))
    monkeypatch.setattr(audit, "_bound_execution", lambda execution, bindings: ({}, {}, linked))
    with pytest.raises(ValueError):
        audit._observed_control(execution, job, [])


def test_complete_audit_uses_shared_bound_air_loader_without_model_or_input_changes(tmp_path, schemas, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ddk, _ = schemas
    schema = load_schema(ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so")
    raw = graph()
    document = schema.decode(raw, "GraphDef")
    job = {"compilation_report_sha256": "a" * 64, "model_sha256": "b" * 64, "weight_payload_access": "NONE"}
    monkeypatch.setattr(audit, "load_bound_source_air", lambda jobdir, profile, bindings: (job, schema, raw, document, "GraphDef"))
    source = tmp_path / "output_pos.bin"
    source.write_bytes(struct.pack("<i", 64))
    control_record = _source(source, 4)
    execution = tmp_path / "execution"
    execution.mkdir()
    (execution / "plan.json").write_text(json.dumps({"inputs": [{"compiled_candidates": ["arg1238_1"],
        "dtype": "INT32", "bytes": 4, "source_filename": "output_pos.bin",
        "semantic_slot_identity": "UNIQUE_TYPED_ND_VIEW", "source": str(source)}], "source_inputs": [control_record]}))
    monkeypatch.setattr(audit, "_bound_execution", lambda execution, bindings: ({}, {}, job))
    before = source.stat()
    report = audit.inspect_target_logit_selection(tmp_path / "job", execution, tmp_path / "audit")
    assert report["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED", report
    assert report["input_mutation"] == report["weight_payload_access"] == "NONE"
    assert report["device_execution"] == report["model_compilation"] == "NOT_RUN"
    assert source.read_bytes() == struct.pack("<i", 64)
    assert source.stat().st_ctime_ns == before.st_ctime_ns
    assert (tmp_path / "audit/report.json").is_file()

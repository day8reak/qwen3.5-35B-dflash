"""Bounded encoded AIR/control diagnostics never execute the original model."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

from google.protobuf import descriptor_pb2
import pytest

from qwen35_moe_dflash import target_logit_selection as audit
from qwen35_moe_dflash.omc_wire import load_schema
from qwen35_moe_dflash.target_prefill_smoke_plan import _source
from qwen35_moe_dflash.target_compile_failure import _stamp
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.air_schema import load_torchair_schema
from test_air_schema import ge_descriptor, pb2
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
          named=True, duplicate=False, output_control="arg1238_1", optional_ancestor=False,
          src_metadata=None, src_only=False, control_edge=False, const_src_metadata=False,
          source_layout="ND", source_meta=None):
    desc = _int(2, 1) + _blob(3, b"".join(_int(1, dim) for dim in source_shape)) + _blob(4, source_layout)
    if source_meta is not None:
        desc += _blob(5, _blob(1, "_meta") + _blob(2, _blob(2, source_meta)))
        desc += _blob(5, _blob(1, "_fx_tensor_name") + _blob(2, _blob(2, "literal-source-hidden")))
    control_desc = _desc(dtype=42, shape=(1,))
    axis_desc = _desc(dtype=axis_dtype, shape=axis_shape)
    selected_desc = _desc(dtype=1, shape=(1, 1, 2048))
    logits = _desc(dtype=1, shape=(1, 1, 248320))
    nodes = [_node("arg1238_1", "Data", [1], dtype=42),
             _node("other_control", "Data", [1], dtype=42),
             op("hidden", "Data", [], [], [desc])]
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
        if const_src_metadata:
            nodes[-1] += _blob(24, "stale_const_metadata") + _int(25, 0)
    consumer = desc if consumer_shape == "same" else _desc(dtype=1, shape=consumer_shape)
    references = ["hidden:0", index, "axis:0"]
    gather = op("select_row", "GatherV2", [] if src_only else references + (["other_control:-1"] if control_edge else []),
                [consumer, control_desc, axis_desc], [selected_desc])
    if src_metadata is not None or src_only:
        for reference in (src_metadata if src_metadata is not None else references):
            name, port = reference.rsplit(":", 1)
            gather += _blob(24, name) + _int(25, int(port))
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


def test_stale_src_metadata_cannot_redirect_the_actual_logit_control_dependency(schemas):
    report = analyze(schemas, graph(src_metadata=["hidden:0", "other_control:0", "axis:0"], control_edge=True))
    assert report["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED", report
    candidate, = report["candidates"]
    assert candidate["index_path"][-1]["node"] == "arg1238_1"
    assert candidate["serialized_input_ports"]["indices"] == 1
    assert report["graph_edge_policy"]["policy"] == "OPDEF_INPUT_AUTHORITATIVE_GRAPH_CONNECTIVITY"


def test_src_metadata_without_explicit_graph_edges_cannot_prove_logit_selection(schemas):
    result = analyze(schemas, graph(src_only=True))
    assert result["status"] == "LOGIT_SELECTION_UNVERIFIED"
    assert result["control_gather_candidate_count"] == 0
    assert "exactly params,indices,axis operands" in result["other_gathers"][0]["reason"]


def test_serialized_zero_input_axis_const_ignores_stored_src_metadata(schemas):
    result = analyze(schemas, graph(const_src_metadata=True))
    assert result["status"] == "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED", result
    assert result["candidates"][0]["axis"]["value"] == 1
    assert result["control_gather_candidate_count"] == 1


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


def test_logit_to_gather_path_reports_actual_ports_and_descriptors(schemas):
    report = analyze(schemas, graph())
    candidate, = report["candidates"]
    path = candidate["logits_dependency_path"]
    assert not path["truncated"]
    assert [(row["node"]["name"], row["output_port"]) for row in path["nodes"]] == [
        ("head", 0), ("select_row", 0)]
    assert path["nodes"][0]["node"]["output_desc"][0]["shape"] == {"dim": [1, 1, 248320]}
    assert candidate["params_output_port"] == 0
    assert candidate["params_consumer_descriptor_index"] == 0
    assert candidate["params_shape_evidence"] == "BOTH_SERIALIZED_CONSUMER_AND_PRODUCER"


def test_descriptor_meta_and_layout_are_diagnostics_without_shape_substitution(schemas):
    ddk, _ = schemas
    schema = load_schema(ddk / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so")
    raw = graph(source_shape=(), source_layout="FRACTAL_NZ", source_meta="Tensor(fake_size=(1,64,2048))")
    document = schema.decode(raw, "GraphDef")
    report = audit.analyze_logit_selection(document, raw, schema, 64)
    candidate, = report["candidates"]
    assert report["status"] == "LOGIT_SELECTION_OBSERVED_RANGE_UNVERIFIED"
    desc = candidate["params_producer"]["output_desc"][0]
    assert desc["layout"] == "FRACTAL_NZ" and desc["fx_tensor_name"] == "literal-source-hidden"
    assert "64" in desc["declared_meta"] and "params_shape" not in candidate
    assert "profile dimensions are not substituted" in candidate["reason"]


@pytest.fixture
def producer_case(tmp_path, schemas, monkeypatch):
    """Real bounded source/OMC bytes; only the saved runtime-link gate is fake."""
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    ddk, _ = schemas
    producer = pb2(tmp_path / "original/ge_ir_pb2.py", ge_descriptor())
    selected_schema = load_torchair_schema(producer)
    copied = tmp_path / "copied-source-pb2.py"
    copied.write_bytes(producer.read_bytes())
    job, execution = tmp_path / "job", tmp_path / "execution"
    compilation = job / "compilation"
    air = compilation / "compile-work/generated/export.air"
    model = compilation / "compile-work/generated/model/model.omc"
    model.parent.mkdir(parents=True)
    execution.mkdir()
    receipt_path = tmp_path / "source-export-result.json"
    control = tmp_path / "output_pos.bin"
    control.write_bytes(struct.pack("<i", 64))
    def save(path, value):
        path.write_text(json.dumps(value))
    def build(**graph_options):
        body = graph(**graph_options)
        body += b"".join(_blob(6, _node(f"RefData_{index}", "RefData", [1, 8192, 4], dtype=7)) for index in range(80))
        air.write_bytes(_blob(7, body))
        compiled = _blob(7, graph())
        task = b"opaque fixture task; never executed"
        content = struct.pack("<I", 2) + struct.pack("<III", 0, 0, len(compiled)) + struct.pack("<III", 2, 0, len(task)) + compiled + task
        header = bytearray(256);header[:4] = b"IMOD";struct.pack_into("<I", header, 0x4C, len(content))
        model.write_bytes(header + content)
        def row(path, stat_key):
            return {"path": path.relative_to(compilation).as_posix(), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), stat_key: _stamp(path.stat())}
        snapshot, primary = dict(row(air, "snapshot_stat"), role="air"), row(model, "stat")
        report = {"artifact_kind": "qwen35-existing-air-compilation", "status": "COMPILED", "model_compilation": "PASSED",
            "returncode": 0, "platform": "kirinx90", "ddk_id": "fixture-ddk", "ddk_root": str(ddk),
            "source_preservation": {"status": "UNCHANGED_BY_STAT"}, "input_snapshots": [snapshot],
            "omc": primary, "generated_artifacts": [primary]}
        save(compilation / "report.json", report)
        save(compilation / "input-inventory.json", {"files": [snapshot]})
        report_sha = hashlib.sha256((compilation / "report.json").read_bytes()).hexdigest()
        save(job / "result.json", {"artifact_kind": "qwen35-original-target-job", "status": "COMPILED", "output": str(job),
            "compilation_report_sha256": report_sha})
        save(job / "compile-profile.json", get_existing_air_profile("cpro-40block-prefill-64"))
        air_identity = {"path": "/unavailable-original-machine/export.air", "bytes": snapshot["bytes"], "sha256": snapshot["sha256"]}
        receipt = {"artifact_kind": "qwen35-canonical-target-export", "status": "EXPORT_READY", "air": air_identity,
            "schema": selected_schema.evidence, "normalization": {"status": "OUTPUT_ORDER_CANONICALIZED", "sequence_length": 64,
                "patched_sha256": snapshot["sha256"], "schema": selected_schema.evidence}}
        save(receipt_path, receipt)
        linkage = {"model_sha256": primary["sha256"], "compilation_report_sha256": report_sha}
        monkeypatch.setattr(audit, "_bound_execution", lambda execution, bindings: ({}, {}, linkage))
        save(execution / "plan.json", {"inputs": [{"compiled_candidates": ["arg1238_1"], "dtype": "INT32", "bytes": 4,
            "source_filename": "output_pos.bin", "semantic_slot_identity": "UNIQUE_TYPED_ND_VIEW", "source": str(control)}],
            "source_inputs": [_source(control, 4)]})
        return {"job": job, "execution": execution, "receipt": receipt_path, "copied_pb2": copied,
            "producer": producer, "air": air, "model": model, "control": control}
    return build


@pytest.mark.parametrize("source_shape,index,expected", [
    ((1, 64, 2048), 64, "LOGIT_INDEX_OUT_OF_RANGE_OBSERVED"),
    ((1, 64, 2048), 63, "LOGIT_INDEX_WITHIN_SERIALIZED_EXTENT"),
    ((), 64, "LOGIT_SELECTION_OBSERVED_RANGE_UNVERIFIED")])
def test_static64_canonical_producer_and_pb2_copy_bind_real_source_graph(producer_case, tmp_path, monkeypatch,
                                                                       source_shape, index, expected):
    case = producer_case(source_shape=source_shape)
    if index != 64:
        case["control"].write_bytes(struct.pack("<i", index))
        saved = json.loads((case["execution"] / "plan.json").read_text())
        saved["source_inputs"] = [_source(case["control"], 4)]
        (case["execution"] / "plan.json").write_text(json.dumps(saved))
    # Only AST descriptor extraction is allowed; original producer path may be unavailable.
    case["producer"].unlink()
    source_files = [p for p in tmp_path.rglob("*") if p.is_file()]
    before = {p: p.read_bytes() for p in source_files}
    original_open = Path.open
    def no_weights(path, *args, **kwargs):
        assert path.suffix != ".weight" and "do-not-open" not in str(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_weights)
    report = audit.inspect_target_logit_selection(case["job"], case["execution"], tmp_path / "audit",
        source_torchair_pb2=case["copied_pb2"], source_export_receipt=case["receipt"])
    assert report["status"] == expected, report
    assert report["job"]["source_schema"]["schema_package"] == "ge.proto"
    assert report["job"]["source_schema"]["pb2_execution"] == "NOT_EXECUTED"
    assert report["stage"] == "complete" and report["numerical_validation"] == "NOT_RUN"
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert report["selection"]["candidates"][0]["logits_dependency_path"]["nodes"][-1]["node"]["name"] == "select_row"


@pytest.mark.parametrize("fault", ["air_sha", "schema_sha", "pb2_sha", "sequence", "unbound_pb2"])
def test_bad_explicit_producer_selection_never_falls_back_to_ddk(producer_case, tmp_path, fault):
    case = producer_case()
    receipt = json.loads(case["receipt"].read_text())
    if fault == "air_sha": receipt["air"]["sha256"] = "0" * 64
    elif fault == "schema_sha": receipt["schema"]["descriptor_sha256"] = "0" * 64
    elif fault == "pb2_sha": receipt["schema"]["pb2_sha256"] = "0" * 64
    elif fault == "sequence": receipt["normalization"]["sequence_length"] = 1
    case["receipt"].write_text(json.dumps(receipt))
    report = audit.inspect_target_logit_selection(case["job"], case["execution"], tmp_path / "audit",
        source_torchair_pb2=case["copied_pb2"],
        source_export_receipt=None if fault == "unbound_pb2" else case["receipt"])
    assert report["status"] == "FAILED" and report["stage"] == "read-bound-source-air", report
    assert "selection" not in report and report["device_execution"] == "NOT_RUN"


def test_cli_forwards_exact_source_schema_and_receipt_options(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_target_logit_selection.py"
    spec = importlib.util.spec_from_file_location("inspect_logit_selection_cli_under_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    def inspect(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "LOGIT_SELECTION_UNVERIFIED", "device_execution": "NOT_RUN"}
    monkeypatch.setattr(audit, "inspect_target_logit_selection", inspect)
    assert cli.main(["--job", "job", "--execution", "execution", "--output", str(tmp_path / "out"),
        "--source-torchair-pb2", "copied.py", "--source-export-receipt", "producer.json"]) == 0
    assert calls == [(("job", "execution", str(tmp_path / "out")),
        {"source_torchair_pb2": "copied.py", "source_export_receipt": "producer.json"})]
    assert json.loads(capsys.readouterr().out)["device_execution"] == "NOT_RUN"

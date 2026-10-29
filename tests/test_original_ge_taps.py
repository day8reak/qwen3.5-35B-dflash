"""Forty source-policy norm anchors on strictly decoded synthetic GE wire.

No vendor library executes and no weight file exists. Sparse original shapes
are retained verbatim; dimensions in the auxiliary plan remain constraints.
"""
import json
import struct

from google.protobuf import descriptor_pb2
import pytest

from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.original_air_inspect import inspect_original_air
from qwen35_moe_dflash.original_ge_target import audit_source, SOURCE_SHA256
from qwen35_moe_dflash.original_ge_taps import plan_original_ge_taps
from test_air_schema import ge_descriptor, pb2
from test_draft_ir_inspect import _attr, _blob, _desc, _int
from test_original_air_inspect import selected, _node, _scalar_node
from test_original_ge_target import documentation_case


TEMPLATE = "./glm_export_air_cpro_40block_55_quant/model_model_layers_{layer}_input_layernorm_weight"


def source_audit():
    return audit_source({"files": [{"path": name, "bytes": 100, "sha256": value}
                                   for name, value in SOURCE_SHA256.items()]})


def op(name, kind, inputs=(), input_descs=(), *, dtype=1, shape=(), output_descs=None):
    outputs = [_desc(dtype=dtype, shape=shape)] if output_descs is None else output_descs
    return (_blob(1, name) + _blob(2, kind)
            + b"".join(_blob(5, edge) for edge in inputs)
            + b"".join(_blob(33, desc) for desc in input_descs)
            + b"".join(_blob(34, desc) for desc in outputs))


def wire(sequence=55, *, half_stream=False, scalar_data=None):
    nodes, states, state_refs = [], [], []
    entries = get_existing_air_profile("cpro-40block-64-1")["input_shape"].split(";")
    for index, entry in enumerate(entries):
        name, dims = entry.split(":")
        dims = [sequence if int(dim) == -1 else int(dim) for dim in dims.split(",")]
        state = name.startswith("RefData_")
        dtype = 7 if state else 3 if name == "arg1_1" else 1
        nodes.append(_node(name, "RefData" if state else "Data", dims, dtype=dtype, index=index))
        if state:
            states.append(_desc(dtype=7, shape=dims))
            state_refs.append(name + ":0")
    nodes.append(_scalar_node("one", struct.pack("<f", 1.0) if scalar_data is None else scalar_data, dtype=1))
    nodes.append(op("embedding_cast", "Cast", ["arg1_1:0"], [_desc(dtype=3, shape=[1, sequence, 2048])]))
    nodes.append(op("scaled_embedding", "Mul", ["embedding_cast:0", "arg0_1:0"],
                    [_desc(dtype=1, shape=()), _desc(dtype=1, shape=[1, sequence, 1])]))
    previous, previous_dtype = "scaled_embedding:0", 1
    for layer in range(40):
        gamma, add, cast, norm = (f"gamma_{layer}", f"gamma_add_{layer}", f"data_cast_{layer}", f"norm_{layer}")
        nodes.append(_node(gamma, "FileConstant", [], dtype=1)
                     + _attr("file_path", _blob(2, TEMPLATE.format(layer=layer)))
                     + _attr("shape", _blob(1, _int(3, 2048)))
                     + _attr("dtype", _int(3, 0)))  # GE attr enum is not protobuf DT_FLOAT=1.
        nodes.append(op(add, "Add", [gamma + ":0", "one:0"],
                        [_desc(dtype=1, shape=[2048]), _desc(dtype=1, shape=())]))
        nodes.append(op(cast, "Cast", [previous], [_desc(dtype=previous_dtype, shape=())]))
        nodes.append(op(norm, "RmsNorm", [cast + ":0", add + ":0"],
                        [_desc(dtype=1, shape=()), _desc(dtype=1, shape=())],
                        output_descs=[_desc(dtype=1, shape=()), _desc(dtype=1, shape=())]))
        # Norm output0 fanout is normal and must not invalidate a gamma anchor.
        nodes.append(op(f"norm_side_{layer}", "SideConsumer", [norm + ":0"], [_desc(dtype=1, shape=())]))
        nodes.append(op(f"attn_{layer}", "SyntheticAttention", [norm + ":0"], [_desc(dtype=1, shape=())]))
        nodes.append(op(f"postattn_{layer}", "Add", [previous, f"attn_{layer}:0"],
                        [_desc(dtype=previous_dtype, shape=()), _desc(dtype=1, shape=())]))
        nodes.append(op(f"mlp_{layer}", "SyntheticMLP", [f"postattn_{layer}:0"], [_desc(dtype=1, shape=())]))
        previous_dtype = 7 if half_stream else 1
        nodes.append(op(f"layer_output_{layer}", "Add", [f"postattn_{layer}:0", f"mlp_{layer}:0"],
                        [_desc(dtype=1, shape=()), _desc(dtype=1, shape=())], dtype=previous_dtype))
        previous = f"layer_output_{layer}:0"
    nodes.append(op("logits", "SyntheticHead", [previous], [_desc(dtype=previous_dtype, shape=())]))
    outputs = ["logits:0"]
    for index, ref in enumerate(state_refs):
        nodes.append(op(f"state_output_{index}", "SyntheticStateUpdate", [ref], [states[index]], dtype=7))
        outputs.append(f"state_output_{index}:0")
    nodes.append(op("Node_Output", "NetOutput", outputs,
                    [_desc(dtype=1, shape=[1, 1, 248320]), *states], output_descs=[]))
    return _blob(7, _blob(1, "source_policy_fixture") + b"".join(_blob(6, node) for node in nodes))


@pytest.fixture
def report_factory(tmp_path, selected):
    # Reuse strict synthetic descriptor and extend its enum before AST loading.
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(ge_descriptor())
    enum = next(value for value in descriptor.enum_type if value.name == "DataType")
    item = enum.value.add()
    item.name, item.number = "DT_INT8", 3
    schema = pb2(tmp_path / "ge_ir_pb2.py", descriptor.SerializeToString())
    def make(**options):
        air = tmp_path / "export.air"
        air.write_bytes(wire(**options))
        report = inspect_original_air(air, None, full=True, torchair_pb2=schema)
        assert report["protobuf_decode"] == "PASSED"
        assert any(row.get("metadata_inspection") == "COMPLETE" for row in report["candidates"]), report
        return report
    return make


def graph(report):
    return next(row["graph"] for row in report["candidates"] if row.get("metadata_inspection") == "COMPLETE")


def node(report, name):
    return next(row for row in graph(report)["operations"] if row["name"] == name)


def plan(report, **options):
    return plan_original_ge_taps(report, source_audit(), TEMPLATE, sequence_length=options.get("sequence_length", 55))


def codes(result):
    return {row["code"] for row in result["pending"]}


@pytest.mark.parametrize("half_stream", [False, True])
def test_forty_anchors_ordered_dag_and_eight_sparse_taps_are_only_a_plan(report_factory, half_stream):
    report = report_factory(half_stream=half_stream)
    before = json.dumps(report, sort_keys=True)
    result = plan(report)
    assert result["status"] == "FEATURE_BRANCH_PLAN_REVIEW_REQUIRED", result
    assert result["bound_input_norm_anchor_count"] == 40
    assert all(row["norm_output0_consumers"] == 2 for row in result["input_norm_anchors"])
    assert result["ordered_dependencies"][-1]["predecessor_input_norm_layers"] == list(range(39))
    assert [tap["post_decoder_layer_index"] for tap in result["taps"]] == [1, 6, 11, 16, 22, 27, 32, 37]
    assert [tap["producer_reference"] for tap in result["taps"]] == [f"layer_output_{layer}:0" for layer in [1, 6, 11, 16, 22, 27, 32, 37]]
    assert all(tap["descriptor"]["shape"] == [] and tap["expected_shape"] == [1, 55, 2048]
               and tap["feature_dtype"] == "float16" for tap in result["taps"])
    assert result["input_norm_anchors"][0]["gamma_dtype_attribute"] == {"i": 0}
    assert result["input_norm_anchors"][0]["gamma_proto_descriptor"]["raw_dtype"] == 1
    assert result["producer_shape_observation"]["shape_inference"].startswith("NOT_PROVEN")
    assert result["feature_branch"]["concat"]["output_descriptor"]["shape"] == [1, 55, 16384]
    assert result["feature_branch"]["feature_output_index"] == 81
    assert len(result["feature_branch"]["preserved_output_references"]) == 81
    assert result["source_to_air_revision"] == "UNVERIFIED"
    assert result["model_compilation"] == result["device_execution"] == result["numerical_validation"] == "NOT_RUN"
    assert json.dumps(report, sort_keys=True) == before


@pytest.mark.parametrize("mutation,expected", [
    ("path", "EXACT_GAMMA_PATH_MISSING_OR_AMBIGUOUS"),
    ("duplicate_path", "EXACT_GAMMA_PATH_MISSING_OR_AMBIGUOUS"),
    ("shape_attribute", "GAMMA_SHAPE_ATTRIBUTE_REQUIRED"),
    ("vector_input", "GAMMA_CONSUMER_VECTOR_DECLARATION_REQUIRED"),
    ("scalar_hash", "EXACT_FP32_SCALAR_ONE_NOT_PROVEN"),
    ("scalar_dtype", "EXACT_FP32_SCALAR_ONE_NOT_PROVEN"),
    ("gamma_fanout", "GAMMA_ADD_CONSUMER_NOT_UNIQUE_INPUT_ZERO"),
    ("add_fanout", "GAMMA_ADD_OUTPUT_NOT_UNIQUE_NORM_GAMMA"),
    ("norm_port", "GAMMA_ADD_OUTPUT_NOT_UNIQUE_NORM_GAMMA"),
    ("cast_fanout", "PENDING_CAST_LINEAGE"),
    ("cast_type", "PENDING_CAST_LINEAGE"),
    ("nested_cast", "PREVIOUS_LAYER_OUTPUT_ADD_UNRESOLVED"),
    ("wrong_add", "PREVIOUS_LAYER_OUTPUT_ADD_UNRESOLVED"),
    ("skipped_layer", "ORDERED_INPUT_NORM_DEPENDENCIES_UNRESOLVED"),
    ("logits_skip", "ALL_FORTY_INPUT_NORMS_MUST_FEED_ORIGINAL_LOGITS"),
])
def test_missing_or_ambiguous_path_port_dtype_scalar_or_order_stays_pending(report_factory, mutation, expected):
    report = report_factory()
    constants = graph(report)["file_constants"]["examples"]
    if mutation == "path":
        constants[0]["file_path"] = "./other-export/weight"
    elif mutation == "duplicate_path":
        constants[1]["file_path"] = constants[0]["file_path"]
    elif mutation == "shape_attribute":
        constants[0]["shape_attribute"]["list_i"] = [1024]
    elif mutation == "vector_input":
        node(report, "gamma_add_0")["input_desc"][0]["shape"] = []
    elif mutation == "scalar_hash":
        node(report, "one")["scalar_constant"]["data"]["sha256"] = "0" * 64
        node(report, "one")["scalar_constant"]["is_standard_le_one"] = True
    elif mutation == "scalar_dtype":
        node(report, "one")["scalar_constant"]["descriptor"]["raw_dtype"] = 7
    elif mutation in {"gamma_fanout", "add_fanout", "cast_fanout"}:
        ref = {"gamma_fanout": "gamma_0:0", "add_fanout": "gamma_add_0:0", "cast_fanout": "data_cast_0:0"}[mutation]
        node(report, "norm_side_0")["input"] = [ref]
    elif mutation == "norm_port":
        node(report, "norm_0")["input"].reverse()
    elif mutation == "cast_type":
        node(report, "data_cast_2")["input_desc"][0]["schema_dtype_names"] = ["DT_INT8"]
    elif mutation == "nested_cast":
        node(report, "data_cast_2")["input"] = ["data_cast_1:0"]
        node(report, "norm_1")["input"][0] = "layer_output_0:0"
    elif mutation == "wrong_add":
        node(report, "data_cast_2")["input"] = ["mlp_1:0"]
    elif mutation == "skipped_layer":
        node(report, "data_cast_2")["input"] = ["layer_output_0:0"]
    elif mutation == "logits_skip":
        node(report, "logits")["input"] = ["layer_output_38:0"]
    result = plan(report)
    assert not result.get("feature_plan_available") and result["taps"] == []
    assert expected in codes(result), result


@pytest.mark.parametrize("data", [struct.pack("<f", 2.0), b"arbitrary-data", struct.pack("<e", 1.0)])
def test_only_proven_fp32_scalar_one_is_accepted(report_factory, data):
    result = plan(report_factory(scalar_data=data))
    assert "EXACT_FP32_SCALAR_ONE_NOT_PROVEN" in codes(result)


def test_sequence_from_real_input_is_not_defaulted_or_rewritten(report_factory):
    report = report_factory(sequence=55)
    result = plan(report, sequence_length=64)
    assert "ORIGINAL_EMBEDDING_SEQUENCE_ABI_UNRESOLVED" in codes(result)
    assert graph(report)["ordinary_inputs"][0]["shape"] == [1, 55, 2048]


def test_cast_shared_with_exact_residual_add_is_a_legal_source_stream(report_factory):
    report = report_factory()
    for layer in [0, 2, 7, 38]:
        node(report, f"postattn_{layer}")["input"][0] = f"data_cast_{layer}:0"
    result = plan(report)
    assert result["status"] == "FEATURE_BRANCH_PLAN_REVIEW_REQUIRED", result
    assert result["ordered_dependencies"][0]["producer_reference"] == "data_cast_0:0"
    assert result["taps"][0]["producer_reference"] == "layer_output_1:0"
    assert result["taps"][0]["cast_lineage"]["residual_consumer_edges"] == [("postattn_2", 0)]


def test_same_cast_residual_consumer_requires_this_norm_on_other_add_operand(report_factory):
    report = report_factory()
    node(report, "postattn_0")["input"] = ["data_cast_0:0", "scaled_embedding:0"]
    assert "PENDING_CAST_LINEAGE" in codes(plan(report))


def test_direct_previous_add_needs_no_cast_and_retains_sparse_descriptor(report_factory):
    report = report_factory()
    node(report, "norm_2")["input"][0] = "layer_output_1:0"
    result = plan(report)
    assert result["status"] == "FEATURE_BRANCH_PLAN_REVIEW_REQUIRED", result
    assert result["taps"][0]["cast_lineage"] is None
    assert result["taps"][0]["descriptor"]["shape"] == []


def test_changed_source_revision_and_incomplete_metadata_cannot_create_plan(report_factory):
    report = report_factory()
    audit = source_audit()
    next(row for row in audit["files"] if row["path"].endswith("utils.py"))["sha256"] = "a" * 64
    result = plan_original_ge_taps(report, audit, TEMPLATE, sequence_length=55)
    assert "SOURCE_AUDIT_REQUIRED" in codes(result)
    graph(report)["file_constants"]["shown"] -= 1
    assert "FULL_FILECONSTANT_METADATA_REQUIRED" in codes(plan(report))


def test_known_documentation_source_is_rechecked_from_actual_root_before_tap_binding(report_factory, documentation_case):
    from qwen35_moe_dflash import original_ge_target as ge
    root, _, _ = documentation_case
    report = report_factory()
    audit = ge.audit_source(ge.source_receipt(root), source_root=root)
    result = plan_original_ge_taps(report, audit, TEMPLATE, sequence_length=55)
    assert result["status"] == "FEATURE_BRANCH_PLAN_REVIEW_REQUIRED", result
    assert result["source_to_air_revision"] == "UNVERIFIED"
    (root / "utils.py").write_bytes((root / "utils.py").read_bytes() + b"# changed after first source audit\n")
    result = plan_original_ge_taps(report, audit, TEMPLATE, sequence_length=55)
    assert "SOURCE_AUDIT_REQUIRED" in codes(result) and result["taps"] == []

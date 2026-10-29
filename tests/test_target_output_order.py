"""Output ABI normalization preserves every original computation wire span."""
import builtins
import hashlib
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import pytest

from qwen35_moe_dflash.air_schema import load_torchair_schema
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash import target_output_order as ordering


def _schema(tmp_path):
    fdp = descriptor_pb2.FileDescriptorProto(name="ge_ir.proto", package="ge.proto", syntax="proto3")
    enum = fdp.enum_type.add(name="DataType")
    for name, number in (("DT_UNDEFINED", 0), ("DT_FLOAT", 1), ("DT_FLOAT16", 7), ("DT_INT32", 4)):
        enum.value.add(name=name, number=number)
    def message(name, fields, parent=None):
        result = (fdp.message_type if parent is None else parent.nested_type).add(name=name)
        for name, number, kind, repeated, target in fields:
            field = result.field.add(name=name, number=number, type=kind, label=3 if repeated else 1)
            if target:
                field.type_name = ".ge.proto." + target
        return result
    message("ModelDef", [("name", 1, 9, False, ""), ("graph", 7, 11, True, "GraphDef")])
    message("GraphDef", [("name", 1, 9, False, ""), ("input", 4, 9, True, ""),
                         ("output", 5, 9, True, ""), ("op", 6, 11, True, "OpDef")])
    message("OpDef", [("name", 1, 9, False, ""), ("type", 2, 9, False, ""), ("input", 5, 9, True, ""),
                      ("attr", 10, 11, True, "AttrEntry"), ("id", 20, 3, False, ""),
                      ("src_index", 22, 3, True, ""), ("is_input_const", 32, 8, True, ""),
                      ("input_desc", 33, 11, True, "TensorDescriptor"),
                      ("output_desc", 34, 11, True, "TensorDescriptor")])
    message("AttrEntry", [("key", 1, 9, False, ""), ("value", 2, 11, False, "AttrDef")])
    attribute = message("AttrDef", [("s", 2, 12, False, ""), ("i", 3, 3, False, ""),
                        ("b", 5, 8, False, ""), ("list", 1, 11, False, "AttrDef.ListValue")])
    message("ListValue", [("s", 2, 12, True, ""), ("i", 3, 3, True, ""),
                          ("b", 5, 8, True, ""), ("val_type", 20, 5, False, "")], parent=attribute)
    message("TensorDescriptor", [("dtype", 2, 14, False, "DataType"),
        ("shape", 3, 11, False, "ShapeDef"), ("layout", 4, 9, False, ""),
        ("attr", 5, 11, True, "AttrEntry"), ("real_dim_cnt", 16, 3, False, "")])
    message("ShapeDef", [("dim", 1, 3, True, "")])
    raw = fdp.SerializeToString()
    pb2 = tmp_path / "ge_ir_by_protoc_3_19_pb2.py"
    pb2.write_text("raise RuntimeError('never execute schema')\nDESCRIPTOR = p.AddSerializedFile(" + repr(raw) + ")\n")
    schema = load_torchair_schema(pb2)
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(raw)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))
    return schema, cls


def _desc(target, shape, dtype=7):
    target.dtype, target.layout = dtype, "ND"
    target.shape.dim.extend(shape)
    target.shape.SetInParent()
    target.real_dim_cnt = len(shape)


def _attr(node, name, member, value):
    row = node.attr.add(key=name)
    setattr(row.value, member, value)
    return row


def _list_attr(node, name, member, values):
    row = node.attr.add(key=name)
    getattr(row.value.list, member).extend(values)
    row.value.list.val_type = {"s": 1, "i": 2, "b": 4}[member]


@pytest.fixture
def source(tmp_path):
    schema, cls = _schema(tmp_path)
    model = cls(name="original quantized external target")
    graph = model.graph.add(name="original_target")
    profile = get_existing_air_profile("cpro-40block-64-1")
    state_shapes = []
    for index, entry in enumerate(profile["input_shape"].split(";")):
        name, shape = entry.split(":")
        shape = [64 if int(value) == -1 else int(value) for value in shape.split(",")]
        node = graph.op.add(name=name, type="RefData" if name.startswith("RefData_") else "Data", id=index)
        _desc(node.output_desc.add(), shape)
        _attr(node, "index", "i", index)
        if node.type == "RefData":
            state_shapes.append(shape)
    body = graph.op.add(name="full_40_layer_decoder", type="OriginalDecoder", id=89)
    # Original TorchAir producers may expose empty shape metadata, independently
    # of the fully materialized NetOutput input descriptors.
    _desc(body.output_desc.add(), [], 1)
    for _ in state_shapes:
        _desc(body.output_desc.add(), [])
    constant = graph.op.add(name="original_quantized_weight", type="FileConstant", id=90)
    _desc(constant.output_desc.add(), [2048])
    _attr(constant, "file_path", "s", b"./original_large_sidecars/weight.bin")
    _attr(constant, "dtype", "i", 0)  # Independent GE attribute dtype domain.
    feature = graph.op.add(name="ConcatV2_100", type="ConcatV2", id=91)
    _desc(feature.output_desc.add(), [], 7)
    net = graph.op.add(name="NetOutput", type="NetOutput", id=92)
    net.input.extend([body.name + ":0", feature.name + ":0", *[body.name + ":" + str(i + 1) for i in range(80)]])
    _desc(net.input_desc.add(), [1, 1, 248320], 1)
    _desc(net.input_desc.add(), [1, 64, 16384])
    for shape in state_shapes:
        _desc(net.input_desc.add(), shape)
    flags = [index % 5 == 1 for index in range(82)]
    net.is_input_const.extend(flags)
    _list_attr(net, "is_input_const", "b", flags)
    _list_attr(net, "_input_name_key", "s", [("__input" + str(i)).encode() for i in range(82)])
    _list_attr(net, "_input_name_value", "i", range(82))
    _attr(net, "true_branch_stream", "i", 0)
    _attr(net, "_inner_net_output", "b", True)
    _list_attr(net, "_datadump_original_op_names", "s", [b"original_target"])
    graph.output.extend(net.input)
    return schema, cls, model


def _nodes(model):
    return model.graph[0].op


def _net(model):
    return next(node for node in _nodes(model) if node.type == "NetOutput")


def _attrs(node):
    return {row.key: row.value for row in node.attr}


def _normalize(source):
    schema, cls, model = source
    raw = model.SerializeToString()
    patched, receipt = ordering.reorder_target_output_bytes(raw, schema)
    return cls.FromString(patched), receipt, raw, patched


def test_fixed_permutation_preserves_all_computation_input_and_weight_bytes(source):
    schema, cls, model = source
    new_model, receipt, raw, patched = _normalize(source)
    old, new = _net(model), _net(new_model)
    permutation = [0, *range(2, 82), 1]
    assert list(new.input) == [old.input[index] for index in permutation]
    assert [desc.SerializeToString() for desc in new.input_desc] == [old.input_desc[index].SerializeToString() for index in permutation]
    assert list(new_model.graph[0].output) == list(new.input)
    assert new_model.name == model.name
    original_nodes = {node.name: node.SerializeToString() for node in _nodes(model) if node.type != "NetOutput"}
    assert original_nodes == {node.name: node.SerializeToString() for node in _nodes(new_model) if node.type != "NetOutput"}
    assert receipt["status"] == "OUTPUT_ORDER_CANONICALIZED"
    assert receipt["computation_ops_byte_identical"] == len(original_nodes)
    assert receipt["file_constant_ops_byte_identical"] == 1 and receipt["new_op_count"] == len(_nodes(model))
    assert receipt["original_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["patched_sha256"] == hashlib.sha256(patched).hexdigest()
    assert receipt["input_validation"]["runtime_input_order"] == "UNVERIFIED"
    assert receipt["numerical_validation"] == receipt["device_execution"] == "NOT_RUN"
    assert "equal-shaped K/V lineage" in receipt["state_identity_scope"]
    assert model.SerializeToString() == raw
    # The synthetic schema uses FP16=7, proving the patcher does not hardcode
    # either the CANN GE DataType=1 or another protobuf schema's FP16=2.
    assert new.input_desc[81].dtype == 7


def test_input_name_mapping_uses_inverse_port_permutation_and_constants_use_forward_order(source):
    schema, cls, model = source
    original_values = list(reversed(range(82)))
    attrs = _attrs(_net(model))
    attrs["_input_name_value"].list.i[:] = original_values
    new_model, _, _, _ = _normalize(source)
    old, new = _net(model), _net(new_model)
    old_attrs, new_attrs = _attrs(old), _attrs(new)
    inverse = {old: new for new, old in enumerate([0, *range(2, 82), 1])}
    assert new_attrs["_input_name_key"].SerializeToString() == old_attrs["_input_name_key"].SerializeToString()
    assert list(new_attrs["_input_name_value"].list.i) == [inverse[old] for old in original_values]
    expected_flags = [old.is_input_const[index] for index in [0, *range(2, 82), 1]]
    assert list(new.is_input_const) == list(new_attrs["is_input_const"].list.b) == expected_flags
    for key in ("true_branch_stream", "_inner_net_output", "_datadump_original_op_names"):
        assert new_attrs[key].SerializeToString() == old_attrs[key].SerializeToString()


def test_missing_optional_mappings_graph_output_and_input_indices_are_not_invented(source):
    schema, cls, model = source
    net = _net(model)
    del net.attr[:]
    del net.is_input_const[:]
    del model.graph[0].output[:]
    for node in _nodes(model):
        if node.type in {"Data", "RefData"}:
            del node.attr[:]
    new_model, receipt, _, _ = _normalize(source)
    assert not _net(new_model).attr and not _net(new_model).is_input_const
    assert not new_model.graph[0].output
    assert receipt["graph_output_policy"] == "ABSENT_OR_EMPTY_UNCHANGED"
    assert all(row["serialized_index"] is None for row in receipt["input_validation"]["inputs"])


def test_indices_are_observed_not_assumed_global_input_order(source):
    schema, cls, model = source
    for node in _nodes(model):
        if node.type == "RefData":
            _attrs(node)["index"].i = int(node.name.split("_")[1])
    _, receipt, _, _ = _normalize(source)
    assert next(row for row in receipt["input_validation"]["inputs"] if row["name"] == "RefData_1")["serialized_index"] == 1
    assert receipt["input_validation"]["runtime_input_order"] == "UNVERIFIED"


@pytest.mark.parametrize("fault,match", [
    ("state_shape", "state geometry"), ("state_dtype", "state geometry"),
    ("state_name", "RefData_0..79"), ("missing_input", "9 Data and 80 RefData"),
    ("ordinary_shape", "ordinary input geometry"), ("ordinary_dtype", "boundary dtype"),
    ("missing_output", "82 explicit"), ("state_output_shape", "differs from RefData_0"),
    ("logits_dtype", "FP32 ND logits"), ("feature_shape", "exactly one FP16 ND feature"),
    ("feature_slot", "exactly one FP16 ND feature"), ("missing_producer", "producer or selected"),
    ("control", "control edges"), ("producer_dtype", "producer dtype/layout"),
    ("producer_shape", "producer shape"), ("graph_output", "GraphDef.output"),
    ("unknown_attr", "unreviewed NetOutput attributes"), ("unknown_top", "unreviewed NetOutput port fields"),
    ("incomplete_mapping", "mapping is incomplete"), ("mapping_duplicate", "unique names"),
    ("mapping_negative", "unique names"), ("const_count", "82 entries"),
    ("top_const_count", "82 boolean entries"), ("duplicate_name", "names must be present"),
])
def test_unsafe_or_unproven_boundaries_fail_closed(source, fault, match):
    schema, _, model = source
    nodes, net = _nodes(model), _net(model)
    state0 = next(node for node in nodes if node.name == "RefData_0")
    attrs = _attrs(net)
    if fault == "state_shape": state0.output_desc[0].shape.dim[-1] = 3
    elif fault == "state_dtype": state0.output_desc[0].dtype = 1
    elif fault == "state_name": state0.name = "wrong_state"
    elif fault == "missing_input": del nodes[0]
    elif fault == "ordinary_shape": nodes[0].output_desc[0].shape.dim[-1] = 1024
    elif fault == "ordinary_dtype": nodes[0].output_desc[0].dtype = 77
    elif fault == "missing_output": del net.input[-1]; del net.input_desc[-1]
    elif fault == "state_output_shape": net.input_desc[2].shape.dim[-1] = 5
    elif fault == "logits_dtype": net.input_desc[0].dtype = 7; nodes[89].output_desc[0].dtype = 7
    elif fault == "feature_shape": net.input_desc[1].shape.dim[-1] = 8192
    elif fault == "feature_slot":
        first = net.input_desc[1].SerializeToString(); second = net.input_desc[2].SerializeToString()
        net.input_desc[1].ParseFromString(second); net.input_desc[2].ParseFromString(first)
    elif fault == "missing_producer": net.input[1] = "absent:0"
    elif fault == "control": net.input[1] = "ConcatV2_100:-1"
    elif fault == "producer_dtype": next(node for node in nodes if node.type == "ConcatV2").output_desc[0].dtype = 1
    elif fault == "producer_shape": next(node for node in nodes if node.type == "ConcatV2").output_desc[0].shape.dim.extend([1, 1, 2048])
    elif fault == "graph_output": model.graph[0].output[1] = "different:0"
    elif fault == "unknown_attr": _attr(net, "unreviewed_input_port_order", "i", 1)
    elif fault == "unknown_top": net.src_index.extend(range(82))
    elif fault == "incomplete_mapping":
        for index, row in enumerate(net.attr):
            if row.key == "_input_name_key": del net.attr[index]; break
    elif fault == "mapping_duplicate": attrs["_input_name_key"].list.s[1] = attrs["_input_name_key"].list.s[0]
    elif fault == "mapping_negative": attrs["_input_name_value"].list.i[1] = -1
    elif fault == "const_count": del attrs["is_input_const"].list.b[-1]
    elif fault == "top_const_count": del net.is_input_const[-1]
    elif fault == "duplicate_name": nodes[-2].name = nodes[-3].name
    # For source output mutations, keep GraphDef.output consistent so the
    # targeted independent geometry/reference check is exercised first.
    if fault != "graph_output": model.graph[0].output[:] = net.input
    with pytest.raises(ValueError, match=match):
        ordering.reorder_target_output_bytes(model.SerializeToString(), schema)


def test_normalization_has_no_filesystem_or_external_weight_access(source, monkeypatch):
    schema, _, model = source
    raw = model.SerializeToString()
    monkeypatch.setattr(builtins, "open", lambda *a, **k: pytest.fail("normalization must not open files"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("normalization must not open paths"))
    _, receipt = ordering.reorder_target_output_bytes(raw, schema)
    assert receipt["weights"] == "NOT_OPENED_OR_COPIED_REFERENCES_BYTE_IDENTICAL"


def test_only_confirmed_source_physical_order_and_explicit_sequence_are_supported(source):
    schema, _, model = source
    raw = model.SerializeToString()
    patched, _ = ordering.reorder_target_output_bytes(raw, schema)
    with pytest.raises(ValueError, match="physical port 1"):
        ordering.reorder_target_output_bytes(patched, schema)
    with pytest.raises(ValueError, match="ordinary input geometry"):
        ordering.reorder_target_output_bytes(raw, schema, sequence_length=55)
    with pytest.raises(ValueError, match="sequence_length"):
        ordering.reorder_target_output_bytes(raw, schema, sequence_length=True)


def test_noncanonical_original_computation_wire_encoding_is_preserved_exactly(source):
    schema, cls, model = source
    raw = model.SerializeToString()
    name = b"full_40_layer_decoder"
    ordinary = b"\x0a" + bytes([len(name)]) + name
    overlong_tag = b"\x8a\x00" + bytes([len(name)]) + name
    assert raw.count(ordinary) == 1
    # Construct a valid test carrier with an overlong two-byte name tag. The
    # normalizer must retain that exact computation encoding, not reserialize
    # it through a message object and merely compare semantic values.
    original_body = next(node for node in _nodes(model) if node.type == "OriginalDecoder")
    body_raw = original_body.SerializeToString()
    overlong_body = body_raw.replace(ordinary, overlong_tag, 1)
    graph_raw = model.graph[0].SerializeToString()
    old_op = ordering._blob(6, body_raw)
    new_op = ordering._blob(6, overlong_body)
    graph_raw = graph_raw.replace(old_op, new_op, 1)
    raw = ordering._blob(1, model.name.encode()) + ordering._blob(7, graph_raw)
    patched, receipt = ordering.reorder_target_output_bytes(raw, schema)
    assert overlong_body in patched
    assert cls.FromString(patched).graph[0].name == model.graph[0].name
    assert receipt["computation_ops_byte_identical"] == len(_nodes(model)) - 1


def test_equal_shaped_state_lineage_is_never_inferred_or_resorted(source):
    _, _, model = source
    net = _net(model)
    # Layer3 full-attention K and V have equal geometry. This operation cannot
    # infer their lineage; it must retain whichever relative order the source
    # actually declared and state that device continuity remains unproven.
    left, right = 2 + 6, 2 + 7
    net.input[left], net.input[right] = net.input[right], net.input[left]
    model.graph[0].output[:] = net.input
    new_model, receipt, _, _ = _normalize(source)
    assert list(_net(new_model).input[1:81]) == list(net.input[2:82])
    assert "source-relative order" in receipt["state_identity_scope"]
    assert "not proven" in receipt["state_identity_scope"]

"""Independent protobuf fixtures for output-order metadata; no model execution."""
from copy import deepcopy

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import pytest

from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.omc_wire import Schema
from qwen35_moe_dflash.target_output_order_validation import validate_target_output_order
from test_draft_ir_inspect import _schema


@pytest.fixture
def case():
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(_schema())
    descriptor.name, descriptor.package = "ge_ir.proto", "ge.proto"

    def visit(message):
        for field in message.field:
            if field.type_name.startswith(".hiai.proto."):
                field.type_name = field.type_name.replace(".hiai.proto.", ".ge.proto.")
        for child in message.nested_type:
            visit(child)

    for enum in descriptor.enum_type:
        for value in enum.value:
            value.name = {"FLOAT": "DT_FLOAT", "HALF": "DT_FLOAT16", "UNDEFINED": "DT_UNDEFINED"}[value.name]
    for message in descriptor.message_type:
        visit(message)
        if message.name == "GraphDef":
            message.field.add(name="input", number=4, type=9, label=3)
            message.field.add(name="output", number=5, type=9, label=3)
        elif message.name == "ModelDef":
            message.field.add(name="name", number=1, type=9, label=1)
            message.field.add(name="version", number=2, type=3, label=1)
        elif message.name == "OpDef":
            for name, number, kind, repeated in (("is_input_const", 50, 8, True), ("has_out_attr", 51, 8, False),
                ("id", 52, 3, False), ("stream_id", 53, 3, False)):
                message.field.add(name=name, number=number, type=kind, label=3 if repeated else 1)
        elif message.name == "AttrDef":
            message.field.add(name="b", number=5, type=8, label=1)
            value = message.nested_type.add(name="ListValue")
            for name, number, kind, repeated in (("s", 2, 12, True), ("i", 3, 3, True),
                ("b", 5, 8, True), ("val_type", 20, 5, False)):
                value.field.add(name=name, number=number, type=kind, label=3 if repeated else 1)
            message.field.add(name="list", number=1, type=11, label=1, type_name=".ge.proto.AttrDef.ListValue")
    encoded = descriptor.SerializeToString()
    schema = Schema(encoded, expected_file="ge_ir.proto", expected_package="ge.proto")
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(encoded)
    Model = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))
    model = Model(name="source-A-host-fixture", version=17)
    graph = model.graph.add(name="complete-interface-host-fixture")
    state_descs = []

    def typed(value, dims, dtype=7):
        value.dtype, value.layout = dtype, "ND"
        value.shape.dim.extend(dims)

    for entry in get_existing_air_profile("cpro-40block-64-1")["input_shape"].split(";"):
        name, dims = entry.split(":")
        dims = [64 if int(value) == -1 else int(value) for value in dims.split(",")]
        node = graph.op.add(name=name, type="RefData" if name.startswith("RefData_") else "Data")
        typed(node.output_desc.add(), dims)
        if node.type == "RefData":
            state_descs.append(node.output_desc[0])
    weight = graph.op.add(name="original_weight", type="FileConstant")
    weight.attr.add(key="file_path").value.s = b"./original64/weight.bin"
    weight.attr.add(key="offset").value.i = 32
    typed(weight.output_desc.add(), [2048])
    body = graph.op.add(name="original_body", type="HostFixtureOnlyNotARealModel")
    body.input.extend(["arg1_1:0", "original_weight:0"])
    typed(body.output_desc.add(), [1, 1, 248320], 1)
    typed(body.output_desc.add(), [1, 64, 16384])
    for value in state_descs:
        body.output_desc.add().CopyFrom(value)
    net = graph.op.add(name="NetOutput", type="NetOutput", has_out_attr=True, id=901, stream_id=0)
    net.input.extend(f"original_body:{index}" for index in range(82))
    for value in body.output_desc:
        net.input_desc.add().CopyFrom(value)

    def add_metadata(net):
        net.is_input_const.extend(index % 3 == 1 for index in range(82))
        names = net.attr.add(key="_input_name_key").value.list
        names.s.extend(f"logical_name_{index}".encode() for index in range(82))
        net.attr.add(key="_input_name_value").value.list.i.extend(reversed(range(82)))
        flags = net.attr.add(key="is_input_const").value.list
        flags.b.extend(index % 5 == 2 for index in range(82))
        flags.val_type = 4
        net.attr.add(key="true_branch_stream").value.i = 0
        net.attr.add(key="_inner_net_output").value.b = True
        net.attr.add(key="_datadump_original_op_names").value.list.val_type = 1

    def normalize(original):
        result = deepcopy(original)
        net = next(node for node in result.graph[0].op if node.type == "NetOutput")
        order = [0] + list(range(2, 82)) + [1]
        edges, descs = list(net.input), [value.SerializeToString() for value in net.input_desc]
        del net.input[:]
        net.input.extend(edges[index] for index in order)
        del net.input_desc[:]
        for index in order:
            net.input_desc.add().ParseFromString(descs[index])
        if net.is_input_const:
            flags = list(net.is_input_const)
            del net.is_input_const[:]
            net.is_input_const.extend(flags[index] for index in order)
        for attr in net.attr:
            if attr.key == "_input_name_value":
                ports = list(attr.value.list.i)
                del attr.value.list.i[:]
                attr.value.list.i.extend(0 if index == 0 else 81 if index == 1 else index - 1 for index in ports)
            elif attr.key == "is_input_const":
                flags = list(attr.value.list.b)
                del attr.value.list.b[:]
                attr.value.list.b.extend(flags[index] for index in order)
        if len(result.graph[0].output) == 82:
            values = list(result.graph[0].output)
            del result.graph[0].output[:]
            result.graph[0].output.extend(values[index] for index in order)
        return result

    return {"schema": schema, "original": model, "normalize": normalize, "metadata": add_metadata}


def validate(case, old=None, new=None, *, kind="ModelDef"):
    old = old or case["original"]
    new = new or case["normalize"](old)
    raw = lambda value: value.SerializeToString() if kind == "ModelDef" else value.graph[0].SerializeToString()
    return validate_target_output_order(raw(old), raw(new), case["schema"])


@pytest.mark.parametrize("kind", ["ModelDef", "GraphDef"])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("graph_outputs", [False, True])
def test_only_fixed_order_is_validated_no_shape_based_state_or_device_pass(case, kind, metadata, graph_outputs):
    old = case["original"]
    if metadata:
        case["metadata"](old.graph[0].op[-1])
    if graph_outputs:
        old.graph[0].output.extend(old.graph[0].op[-1].input)
    result = validate(case, kind=kind)
    assert result["status"] == "PASS_METADATA_ORDER_ONLY"
    assert result["permutation"] == [0, *range(2, 82), 1]
    assert result["original_feature_index"] == 1 and result["normalized_feature_index"] == 81
    assert result["input_count"] == 89 and result["refdata_count"] == 80
    assert result["file_constants"]["count"] == 1
    assert result["original_computation_nodes_byte_identical"] == len(old.graph[0].op) - 1
    assert result["state_continuity"] == "UNVERIFIED" and result["device_execution"] == "NOT_RUN"
    assert result["numerical_validation"] == "NOT_EVALUATED"
    assert result["state_identity"].endswith("NOT_PROVEN_BY_SAME_SHAPED_DESCRIPTORS")


@pytest.mark.parametrize("mutation", ["wrong_edges", "wrong_descriptors", "feature_not_last", "computation", "fileconstant", "model", "graph", "operation_order", "extra_node", "graph_output", "const_top", "const_attr", "input_name_wrong_remap", "input_name_keys_reordered", "metadata_dropped"])
def test_unapproved_normalized_changes_cannot_pass(case, mutation):
    old = case["original"]
    case["metadata"](old.graph[0].op[-1])
    old.graph[0].output.extend(old.graph[0].op[-1].input)
    new = case["normalize"](old)
    graph, net = new.graph[0], new.graph[0].op[-1]
    if mutation == "wrong_edges":
        net.input[1], net.input[2] = net.input[2], net.input[1]
    elif mutation == "wrong_descriptors":
        net.input_desc[1].dtype = 1
    elif mutation == "feature_not_last":
        net.input[-1] = "original_body:2"
    elif mutation == "computation":
        graph.op[-2].type = "different-operation"
    elif mutation == "fileconstant":
        next(node for node in graph.op if node.type == "FileConstant").attr[1].value.i = 64
    elif mutation == "model":
        new.version = 18
    elif mutation == "graph":
        graph.input.append("unexpected:0")
    elif mutation == "operation_order":
        first, second = deepcopy(graph.op[0]), deepcopy(graph.op[1])
        graph.op[0].CopyFrom(second)
        graph.op[1].CopyFrom(first)
    elif mutation == "extra_node":
        graph.op.add(name="unapproved", type="Identity")
    elif mutation == "graph_output":
        graph.output[1] = "original_body:1"
    elif mutation == "const_top":
        net.is_input_const[-1] = not net.is_input_const[-1]
    elif mutation == "const_attr":
        flags = next(attr for attr in net.attr if attr.key == "is_input_const").value.list.b
        flags[-1] = not flags[-1]
    elif mutation == "input_name_wrong_remap":
        item = next(attr for attr in net.attr if attr.key == "_input_name_value").value.list.i
        del item[:]
        item.extend(reversed(range(82)))
    elif mutation == "input_name_keys_reordered":
        item = next(attr for attr in net.attr if attr.key == "_input_name_key").value.list.s
        values = list(item)
        del item[:]
        item.extend(values[0:1] + values[2:] + values[1:2])
    else:
        del net.attr[-1]
    assert validate(case, old, new)["status"] == "FAILED"


@pytest.mark.parametrize("mutation", ["unknown_attr", "src_metadata", "control", "unknown_port", "duplicate_ref", "feature_dtype", "feature_shape", "feature_layout", "producer_dtype", "producer_layout", "producer_shape", "RefData_type", "RefData_geometry", "RefData_dtype", "Data_geometry", "name_map_duplicate", "name_map_incomplete", "graph_output_representation"])
def test_unsupported_original_contract_is_not_made_valid_by_reordering(case, mutation):
    old = case["original"]
    graph, net = old.graph[0], old.graph[0].op[-1]
    if mutation == "unknown_attr":
        net.attr.add(key="unknown_port_metadata").value.i = 82
    elif mutation == "src_metadata":
        net.src_name.append("original_body")
    elif mutation == "control":
        net.input[3] = "original_body:-1"
    elif mutation == "unknown_port":
        net.input[3] = "original_body:999"
    elif mutation == "duplicate_ref":
        net.input[3] = net.input[2]
    elif mutation == "feature_dtype":
        net.input_desc[1].dtype = 1
    elif mutation == "feature_shape":
        net.input_desc[1].shape.dim[-1] = 8192
    elif mutation == "feature_layout":
        net.input_desc[1].layout = "NCHW"
    elif mutation == "producer_dtype":
        graph.op[-2].output_desc[1].dtype = 1
    elif mutation == "producer_layout":
        graph.op[-2].output_desc[1].layout = "NCHW"
    elif mutation == "producer_shape":
        graph.op[-2].output_desc[1].shape.dim[-1] = 8192
    elif mutation == "RefData_type":
        next(node for node in graph.op if node.name == "RefData_0").type = "Data"
    elif mutation == "RefData_geometry":
        next(node for node in graph.op if node.name == "RefData_0").output_desc[0].shape.dim[-1] = 5
    elif mutation == "RefData_dtype":
        next(node for node in graph.op if node.name == "RefData_0").output_desc[0].dtype = 1
    elif mutation == "Data_geometry":
        next(node for node in graph.op if node.name == "arg1_1").output_desc[0].shape.dim[-1] = 1024
    elif mutation in {"name_map_duplicate", "name_map_incomplete"}:
        case["metadata"](net)
        values = next(attr for attr in net.attr if attr.key == "_input_name_value").value.list.i
        if mutation == "name_map_duplicate":
            values[0] = values[1]
        else:
            del values[-1]
    else:
        graph.output.append("NetOutput:0")
    assert validate(case)["status"] == "FAILED"


def test_same_shaped_state_identity_is_deliberately_not_inferred(case):
    # L0/L1 convolution states share a shape. Relative order can be preserved
    # while external semantic labels are wrong; metadata cannot prove otherwise.
    old = case["original"]
    net = old.graph[0].op[-1]
    net.input[2], net.input[4] = net.input[4], net.input[2]
    result = validate(case)
    assert result["status"] == "PASS_METADATA_ORDER_ONLY"
    assert result["state_identity"].startswith("RELATIVE_STATE_OUTPUT_ORDER_ONLY")


def test_nonempty_unrecognized_schema_or_changed_root_format_is_rejected(case):
    old, new = case["original"], case["normalize"](case["original"])
    raw = old.SerializeToString()
    assert validate_target_output_order(raw + b"\xf8\x3f\x01", new.SerializeToString(), case["schema"])["status"] == "FAILED"
    assert validate_target_output_order(raw, new.graph[0].SerializeToString(), case["schema"])["status"] == "FAILED"


def test_independent_validator_accepts_actual_wire_writer_with_nonidentity_port_name_map(case):
    from qwen35_moe_dflash.target_output_order import reorder_target_output_bytes

    old = case["original"]
    case["metadata"](old.graph[0].op[-1])
    old.graph[0].output.extend(old.graph[0].op[-1].input)
    original = old.SerializeToString()
    normalized, _ = reorder_target_output_bytes(original, case["schema"])
    result = validate_target_output_order(original, normalized, case["schema"])
    assert result["status"] == "PASS_METADATA_ORDER_ONLY"

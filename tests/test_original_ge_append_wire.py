"""Real protobuf-encoded host fixtures; no AIR/weight files or vendor execution."""
from copy import deepcopy
import hashlib

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import pytest

from qwen35_moe_dflash.omc_wire import Schema, _fields
from qwen35_moe_dflash.original_ge_append_wire import append_feature_branch_wire
from test_draft_ir_inspect import _schema


PREFIX = "__dflash_features/"


@pytest.fixture
def case():
    # Reuse only the selected-schema field definitions. Fixtures below are
    # serialized by protobuf itself, independently of the appender's encoder.
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(_schema())
    descriptor.name, descriptor.package = "ge_ir.proto", "ge.proto"

    def visit(message):
        for field in message.field:
            if field.type_name.startswith(".hiai.proto."):
                field.type_name = field.type_name.replace(".hiai.proto.", ".ge.proto.")
        for child in message.nested_type:
            visit(child)

    for message in descriptor.message_type:
        visit(message)
        if message.name == "GraphDef":
            message.field.add(name="output", number=5, type=9, label=3)
        elif message.name == "ModelDef":
            message.field.add(name="name", number=1, type=9, label=1)
            message.field.add(name="version", number=2, type=3, label=1)
    raw_schema = descriptor.SerializeToString()
    schema = Schema(raw_schema, expected_file="ge_ir.proto", expected_package="ge.proto")
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(raw_schema)
    Graph = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.GraphDef"))
    Model = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))

    def typed(desc, shape=(1, 55, 2048), dtype=7):
        desc.dtype, desc.layout = dtype, "ND"
        desc.shape.dim.extend(shape)

    def node(graph, name, kind, inputs=(), shape=(1, 55, 2048), outputs=1):
        result = graph.op.add(name=name, type=kind)
        result.input.extend(inputs)
        for _ in inputs:
            typed(result.input_desc.add(), shape)
        for _ in range(outputs):
            typed(result.output_desc.add(), shape)
        return result

    original = Graph(name="original")
    node(original, "embeddings", "Data")
    constant = node(original, "weight", "FileConstant", shape=(2048,))
    item = constant.attr.add(key="file_path")
    item.value.s = b"./unchanged-original-external-weight"
    constant.attr.add(key="offset").value.i = 8192
    body = node(original, "opaque_body", "OriginalComputation", ("embeddings:0", "weight:0"), outputs=81)
    body.attr.add(key="computation-must-stay-byte-identical").value.i = 29
    node(original, "Node_Output", "NetOutput", tuple(f"opaque_body:{index}" for index in range(81)), outputs=0)
    original.op[-1].attr.add(key="original-output-metadata").value.i = 81
    branch = Graph(name="ge-built-host-fixture-branch")
    for index in range(8):
        node(branch, PREFIX + f"input_{index}", "Data")
        cast = node(branch, PREFIX + f"cast_{index}", "Cast", (PREFIX + f"input_{index}:0",))
        cast.attr.add(key="dst_type").value.i = 1  # Native attr enum; not the synthetic protobuf HALF=7.
    concat = node(branch, PREFIX + "concat", "ConcatD", tuple(PREFIX + f"cast_{index}:0" for index in range(8)),
                  shape=(1, 55, 16384))
    concat.attr.add(key="concat_dim").value.i = 2
    concat.attr.add(key="N").value.i = 8
    node(branch, "branch-output", "NetOutput", (PREFIX + "concat:0",), shape=(1, 55, 16384), outputs=0)
    taps = [{"producer_reference": f"opaque_body:{index}"} for index in range(8)]

    def encode(graph, kind="ModelDef"):
        if kind == "GraphDef":
            return graph.SerializeToString()
        model = Model(name="original-model-fields-retained", version=103)
        model.graph.add().CopyFrom(graph)
        return model.SerializeToString()

    return {"schema": schema, "original": original, "branch": branch, "taps": taps,
            "encode": encode, "Graph": Graph, "Model": Model, "schema_descriptor": descriptor}


def decoded_graph(raw, schema, kind):
    value = schema.decode(raw, kind)
    return value["graph"][0] if kind == "ModelDef" else value


def op_payloads(raw, schema, kind):
    # Extract payloads with the strict wire reader, without reserializing them.
    graph_raw = next(bytes(value) for number, wire, value in _fields(raw, [2_000_000])
                     if number == 7 and wire == 2) if kind == "ModelDef" else raw
    result = []
    for number, wire, payload in _fields(graph_raw, [2_000_000]):
        if number == 6:
            result.append((schema.decode(payload, "OpDef"), bytes(payload)))
    return result


def call(case, original_kind="ModelDef", branch_kind="ModelDef"):
    old = case["encode"](case["original"], original_kind)
    branch = case["encode"](case["branch"], branch_kind)
    return old, branch, append_feature_branch_wire(old, branch, case["schema"], case["taps"])


@pytest.mark.parametrize("original_kind", ["ModelDef", "GraphDef"])
@pytest.mark.parametrize("branch_kind", ["ModelDef", "GraphDef"])
@pytest.mark.parametrize("graph_outputs", [False, True])
def test_exact_nine_node_append_preserves_all_original_computation_bytes(case, original_kind, branch_kind, graph_outputs):
    if graph_outputs:
        case["original"].output.extend(f"opaque_body:{index}" for index in range(81))
    old, branch, (patched, receipt) = call(case, original_kind, branch_kind)
    graph = decoded_graph(patched, case["schema"], original_kind)
    original_nodes = op_payloads(old, case["schema"], original_kind)
    patched_nodes = op_payloads(patched, case["schema"], original_kind)
    original_names = [node["name"] for node, _ in original_nodes]
    new_names = [node["name"] for node, _ in patched_nodes]
    added_names = [PREFIX + f"cast_{index}" for index in range(8)] + [PREFIX + "concat"]
    assert new_names == original_names[:-1] + added_names + original_names[-1:]
    assert not any(node["type"] == "Data" and node["name"].startswith(PREFIX) for node in graph["op"])
    before = dict((node["name"], raw) for node, raw in original_nodes)
    after = dict((node["name"], raw) for node, raw in patched_nodes)
    for name, payload in before.items():
        if name != "Node_Output":
            assert after[name] == payload
    assert after["Node_Output"].startswith(before["Node_Output"])
    net = next(node for node in graph["op"] if node["name"] == "Node_Output")
    assert net["input"] == [f"opaque_body:{index}" for index in range(81)] + [PREFIX + "concat:0"]
    assert len(net["input_desc"]) == 82
    assert net["input_desc"][-1]["shape"]["dim"] == [1, 55, 16384]
    for index in range(8):
        cast = next(node for node in graph["op"] if node["name"] == PREFIX + f"cast_{index}")
        assert cast["input"] == [f"opaque_body:{index}"]
        template = next(node for node in decoded_graph(branch, case["schema"], branch_kind)["op"]
                        if node["name"] == cast["name"])
        assert {k: v for k, v in cast.items() if k != "input"} == {k: v for k, v in template.items() if k != "input"}
    if graph_outputs:
        assert graph["output"] == [f"opaque_body:{index}" for index in range(81)] + [PREFIX + "concat:0"]
    else:
        assert "output" not in graph
    if original_kind == "ModelDef":
        model = case["schema"].decode(patched, "ModelDef")
        assert model["name"] == "original-model-fields-retained" and model["version"] == 103
    assert receipt["status"] == "ORIGINAL_COMPUTATION_BYTES_PRESERVED"
    assert receipt["new_nodes"] == 9 and receipt["new_outputs"] == 82
    assert receipt["original_computation_nodes_byte_identical"] == len(original_nodes) - 1
    assert receipt["original_sha256"] == hashlib.sha256(old).hexdigest()
    assert receipt["patched_sha256"] == hashlib.sha256(patched).hexdigest()
    assert "accuracy" not in receipt and "device_execution" not in receipt


@pytest.mark.parametrize("reference", ["missing:0", "opaque_body:81", "opaque_body:-1", "opaque_body:01", "opaque_body:+1", "opaque_body:١", "opaque_body", ":0", None])
def test_unknown_or_noncanonical_tap_ports_are_rejected(case, reference):
    case["taps"][0]["producer_reference"] = reference
    with pytest.raises(ValueError):
        call(case)


@pytest.mark.parametrize("mutation", ["reserved_collision", "missing_netout", "two_netouts", "80_ports", "82_ports", "missing_desc", "control_minus_one", "control_caret", "unknown_original_producer", "unknown_original_port"])
def test_original_boundary_ambiguity_controls_or_collision_is_rejected(case, mutation):
    graph = case["original"]
    net = graph.op[-1]
    if mutation == "reserved_collision":
        graph.op[0].name = PREFIX + "existing"
    elif mutation == "missing_netout":
        del graph.op[-1]
    elif mutation == "two_netouts":
        graph.op.add().CopyFrom(net)
        graph.op[-1].name = "another-output"
    elif mutation == "80_ports":
        del net.input[-1]
        del net.input_desc[-1]
    elif mutation == "82_ports":
        net.input.append("opaque_body:0")
        net.input_desc.add().CopyFrom(net.input_desc[0])
    elif mutation == "missing_desc":
        del net.input_desc[-1]
    elif mutation == "control_minus_one":
        net.input[-1] = "opaque_body:-1"
    elif mutation == "control_caret":
        net.input[-1] = "^opaque_body"
    elif mutation == "unknown_original_producer":
        net.input[-1] = "missing:0"
    else:
        net.input[-1] = "opaque_body:81"
    with pytest.raises(ValueError):
        call(case)


@pytest.mark.parametrize("mutation", ["17_nodes", "19_nodes", "duplicate_name", "wrong_data", "wrong_cast", "cast_input_rewire", "two_cast_inputs", "concat_order", "concat_type", "two_concat_outputs", "missing_concat_output", "branch_netout_rewire", "branch_netout_two_inputs"])
def test_branch_count_identity_and_explicit_connections_are_required(case, mutation):
    graph = case["branch"]
    nodes = {node.name: node for node in graph.op}
    cast, concat = nodes[PREFIX + "cast_0"], nodes[PREFIX + "concat"]
    if mutation == "17_nodes":
        del graph.op[0]
    elif mutation == "19_nodes":
        graph.op.add(name="extra", type="Identity")
    elif mutation == "duplicate_name":
        cast.name = PREFIX + "cast_1"
    elif mutation == "wrong_data":
        nodes[PREFIX + "input_0"].type = "FileConstant"
    elif mutation == "wrong_cast":
        cast.type = "Identity"
    elif mutation == "cast_input_rewire":
        cast.input[0] = PREFIX + "input_1:0"
    elif mutation == "two_cast_inputs":
        cast.input.append(PREFIX + "input_1:0")
    elif mutation == "concat_order":
        concat.input[0], concat.input[1] = concat.input[1], concat.input[0]
    elif mutation == "concat_type":
        concat.type = "Concat"
    elif mutation == "two_concat_outputs":
        concat.output_desc.add().CopyFrom(concat.output_desc[0])
    elif mutation == "missing_concat_output":
        del concat.output_desc[:]
    elif mutation == "branch_netout_rewire":
        graph.op[-1].input[0] = PREFIX + "cast_0:0"
    else:
        graph.op[-1].input.append(PREFIX + "concat:0")
        graph.op[-1].input_desc.add().CopyFrom(graph.op[-1].input_desc[0])
    with pytest.raises(ValueError):
        call(case)


@pytest.mark.parametrize("which", ["original", "branch"])
@pytest.mark.parametrize("bad", [b"", b"\x00", b"\x3a\xff", b"\xf8\x3f\x01", b"\x0b", "not bytes"])
def test_malformed_or_unknown_schema_wire_is_rejected(case, which, bad):
    old, branch = case["encode"](case["original"]), case["encode"](case["branch"])
    if which == "original":
        old = bad
    else:
        branch = bad
    with pytest.raises(ValueError):
        append_feature_branch_wire(old, branch, case["schema"], case["taps"])


def test_multiple_root_graphs_and_duplicate_singular_fields_are_rejected(case):
    model = case["Model"](name="two-roots", version=1)
    model.graph.add().CopyFrom(case["original"])
    model.graph.add().CopyFrom(case["original"])
    old = model.SerializeToString()
    branch = case["encode"](case["branch"])
    with pytest.raises(ValueError):
        append_feature_branch_wire(old, branch, case["schema"], case["taps"])
    # Two name fields form ambiguous strict wire input even if a permissive
    # protobuf implementation would select/merge a value.
    old = case["encode"](case["original"]) + b"\x0a\x01x"
    with pytest.raises(ValueError):
        append_feature_branch_wire(old, branch, case["schema"], case["taps"])


def test_tap_count_and_selected_schema_field_contract_are_required(case):
    old, branch = case["encode"](case["original"]), case["encode"](case["branch"])
    with pytest.raises(ValueError, match="eight"):
        append_feature_branch_wire(old, branch, case["schema"], case["taps"][:-1])
    changed = deepcopy(case["schema"])
    message = changed._messages["ge.proto.OpDef"]
    field = next(key for key, value in message.items() if value.name == "input_desc")
    del message[field]
    with pytest.raises(ValueError):
        append_feature_branch_wire(old, branch, changed, case["taps"])


def test_new_ids_are_rebased_above_original_collision_range_without_changing_old_payloads(case):
    descriptor = deepcopy(case["schema_descriptor"])
    op = next(message for message in descriptor.message_type if message.name == "OpDef")
    # Deliberately not the public field number: the selected schema must bind it.
    op.field.add(name="id", number=137, type=3, label=1)
    schema = Schema(descriptor.SerializeToString(), expected_file="ge_ir.proto", expected_package="ge.proto")
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(descriptor.SerializeToString())
    Model = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))
    old, branch = Model(), Model()
    old.ParseFromString(case["encode"](case["original"]))
    branch.ParseFromString(case["encode"](case["branch"]))
    for node, identifier in zip(old.graph[0].op, (8, 9, 10, 16)):
        node.id = identifier
    for index, node in enumerate(branch.graph[0].op):
        node.id = index
    original_raw, branch_raw = old.SerializeToString(), branch.SerializeToString()
    patched, receipt = append_feature_branch_wire(original_raw, branch_raw, schema, case["taps"])
    graph = decoded_graph(patched, schema, "ModelDef")
    original_payloads = {node["name"]: payload for node, payload in op_payloads(original_raw, schema, "ModelDef")}
    patched_payloads = {node["name"]: payload for node, payload in op_payloads(patched, schema, "ModelDef")}
    ids = [node["id"] for node in graph["op"] if node["name"].startswith(PREFIX)]
    assert ids == list(range(17, 26))
    assert receipt["new_operation_ids"] == ids
    for name, payload in original_payloads.items():
        if name != "Node_Output":
            assert patched_payloads[name] == payload
    assert {node["name"]: node["id"] for node in graph["op"] if not node["name"].startswith(PREFIX)} == {
        "embeddings": 8, "weight": 9, "opaque_body": 10, "Node_Output": 16}

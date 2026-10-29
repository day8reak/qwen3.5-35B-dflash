"""Mechanical isolation/fidelity gates; mocked host API is never device evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import pytest

from qwen35_moe_dflash import original_air_roundtrip as probe
from test_draft_ir_inspect import _blob
from test_original_air_inspect import _graph
import test_draft_ir_inspect as fixture


def _pb2(path):
    descriptor = descriptor_pb2.FileDescriptorProto.FromString(fixture._schema())
    descriptor.name, descriptor.package = "ge_ir.proto", "ge.proto"
    def visit(message):
        for field in message.field:
            if field.type_name.startswith(".hiai.proto."):
                field.type_name = field.type_name.replace(".hiai.proto.", ".ge.proto.")
        for nested in message.nested_type:
            visit(nested)
    for message in descriptor.message_type:
        visit(message)
    path.write_text("raise RuntimeError('must not execute pb2')\nDESCRIPTOR = p.AddSerializedFile(" +
                    repr(descriptor.SerializeToString()) + ")\n")


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "run"))
    air = tmp_path / "original.air"
    air.write_bytes(_blob(7, _graph(constants=1)))
    schema = tmp_path / "schema.py"
    _pb2(schema)
    # A real sibling payload exists outside isolation and must never be read.
    weight = tmp_path / "original_export/model_model_layers_0_input_layernorm_weight"
    weight.parent.mkdir()
    weight.write_bytes(b"external payload must not be opened or copied")
    ge = tmp_path / "declared-ge"
    include, lib = ge / "include", ge / "lib64"
    for path in (include / "graph/graph.h", include / "graph/operator.h", include / "graph/types.h",
                 lib / "libgraph.so", lib / "libgraph_base.so", ge / "compiler"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"host-test toolchain identity only")
    monkeypatch.setattr(probe, "_toolchain", lambda supplied, *args: (ge, include, lib, str(ge / "compiler")))
    options = SimpleNamespace(mutate=None, native_failure=False, calls=[])
    def logged(command, *, folder, env, log, timeout):
        options.calls.append(command)
        log.write_text("host test fake GE call; no actual model or vendor execution\n")
        if command[0] == str(ge / "compiler"):
            output = Path(command[-1])
            output.write_bytes(b"fake binary")
            output.chmod(0o755)
            return
        assert folder == tmp_path / "run/probe/work"
        assert Path(command[1]).parent == folder
        assert not (folder / "original_export").exists()
        assert env["LD_LIBRARY_PATH"].split(":")[0] == str(lib)
        status = {"load_status": 1 if options.native_failure else 0, "save_status": None if options.native_failure else 0,
                  "model_execution": "NOT_RUN"}
        Path(command[3]).write_text(json.dumps(status))
        if options.native_failure:
            raise RuntimeError("retained native load failure")
        raw = Path(command[1]).read_bytes()
        if options.mutate == "node":
            raw = raw.replace(b"original_target", b"Original_target")
        elif options.mutate == "weight_path":
            raw = raw.replace(b"original_export", b"modified_export")
        elif options.mutate == "source":
            air.write_bytes(b"source was changed")
        Path(command[2]).write_bytes(raw)
    monkeypatch.setattr(probe, "_logged", logged)
    return air, schema, ge, tmp_path / "run/probe", weight, options


def test_only_air_is_staged_native_status_and_all_metadata_are_retained(case, monkeypatch):
    air, schema, ge, out, weight, options = case
    original_open = Path.open
    def no_payload(path, *args, **kwargs):
        if path == weight:
            pytest.fail("external weight must not be opened")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_payload)
    result = probe.probe_original_air_roundtrip(air, out, ge_root=ge, torchair_pb2=schema)
    assert result["status"] == "ROUNDTRIP_METADATA_PASSED"
    assert result["fidelity"]["byte_identical"]
    assert result["source_preservation"] == "UNCHANGED_STAT_AND_SHA256"
    assert result["native"]["load_status"] == result["native"]["save_status"] == 0
    assert result["feature_extension"] == "NOT_ATTEMPTED"
    assert result["feature_extension_gate"] == "METADATA_ONLY_PASSED"
    assert result["model_execution"] == result["model_compilation"] == result["device_execution"] == "NOT_RUN"
    assert result["external_weights"]["payloads"] == "NOT_SUPPLIED_IN_ISOLATED_WORKSPACE"
    assert not (out / "work/original_export").exists()
    assert len(options.calls) == 2
    assert result == json.loads((out / "result.json").read_text())
    assert result["toolchain"]["identities"] and result["commands"]["build"]
    source = (out / "work/probe.cpp").read_text()
    assert "LoadFromFile" in source and "SaveToFile" in source
    assert "SetOutputs" not in source and "SetInputs" not in source
    with pytest.raises(FileExistsError):
        probe.probe_original_air_roundtrip(air, out, ge_root=ge, torchair_pb2=schema)


@pytest.mark.parametrize("reference", ["/original/weights/value", "../value", "folder/../../value", "input.air", "probe.cpp"])
def test_unisolatable_or_colliding_weights_block_before_build(case, reference):
    air, schema, ge, out, _, options = case
    from qwen35_moe_dflash.air_schema import extract_torchair_descriptor
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(extract_torchair_descriptor(schema)[0])
    model = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))()
    model.ParseFromString(air.read_bytes())
    constant = next(node for node in model.graph[0].op if node.type == "FileConstant")
    next(row for row in constant.attr if row.key == "file_path").value.s = reference.encode()
    air.write_bytes(model.SerializeToString())
    result = probe.probe_original_air_roundtrip(air, out, ge_root=ge, torchair_pb2=schema)
    assert result["status"] == "BLOCKED" and result["feature_extension_gate"] == "BLOCKED"
    assert not options.calls
    assert not (out / "work/input.air").exists()


@pytest.mark.parametrize("mutation", ["node", "weight_path"])
def test_native_success_cannot_promote_changed_graph_or_fileconstant_metadata(case, mutation):
    air, schema, ge, out, _, options = case
    options.mutate = mutation
    result = probe.probe_original_air_roundtrip(air, out, ge_root=ge, torchair_pb2=schema)
    assert result["native"]["load_status"] == result["native"]["save_status"] == 0
    assert result["status"] == "ROUNDTRIP_METADATA_DIFFERENT"
    assert result["feature_extension_gate"] == "BLOCKED"
    assert result["fidelity"]["differences"]


def test_nested_fileconstant_cannot_bypass_workspace_isolation(case):
    air, schema, ge, out, _, options = case
    from qwen35_moe_dflash.air_schema import extract_torchair_descriptor
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(extract_torchair_descriptor(schema)[0])
    model = message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))()
    model.ParseFromString(air.read_bytes())
    source_constant = next(node for node in model.graph[0].op if node.type == "FileConstant")
    nested = model.graph[0].op[0].attr.add()
    nested.key = "nested_graph"
    nested.value.g.name = "nested"
    copied = nested.value.g.op.add()
    copied.CopyFrom(source_constant)
    next(row for row in copied.attr if row.key == "file_path").value.s = b"/outside/weights"
    air.write_bytes(model.SerializeToString())
    result = probe.probe_original_air_roundtrip(air, out, ge_root=ge, torchair_pb2=schema)
    assert result["status"] == "BLOCKED" and not options.calls
    assert "cannot be isolated" in result["error"]


def test_native_failure_retains_status_and_source_mutation_cannot_pass(case):
    air, schema, ge, out, _, options = case
    options.native_failure = True
    result = probe.probe_original_air_roundtrip(air, out, ge_root=ge, torchair_pb2=schema)
    assert result["status"] == "FAILED"
    assert result["native"]["load_status"] == 1 and result["native"]["save_status"] is None
    assert result["feature_extension_gate"] == "BLOCKED"


def test_missing_environment_is_concrete_blocker_without_guessing_another_ge(case):
    air, schema, _, out, _, options = case
    result = probe.probe_original_air_roundtrip(air, out, torchair_pb2=schema)
    assert result["status"] == "BLOCKED" and "declare --ge-root or CANN_ROOT" in result["error"]
    assert not options.calls
    assert (out / "work/probe.cpp").is_file()


def test_original_air_change_invalidates_an_otherwise_matching_roundtrip(case):
    air, schema, ge, out, _, options = case
    options.mutate = "source"
    result = probe.probe_original_air_roundtrip(air, out, ge_root=ge, torchair_pb2=schema)
    assert result["status"] == "FAILED" and result["source_preservation"] == "CHANGED"
    assert result["feature_extension_gate"] == "BLOCKED"


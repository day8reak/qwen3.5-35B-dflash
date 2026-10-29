"""Static TorchAir schema selection must never execute pb2 or import NPU code."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

from google.protobuf import descriptor_pb2
import pytest

from qwen35_moe_dflash import air_schema
from qwen35_moe_dflash import original_air_inspect as inspect
from qwen35_moe_dflash.omc_wire import Schema
from test_original_air_inspect import selected, _graph
from test_draft_ir_inspect import _blob
import test_draft_ir_inspect as fixture


def ge_descriptor():
    value = descriptor_pb2.FileDescriptorProto.FromString(fixture._schema())
    value.name, value.package = "ge_ir.proto", "ge.proto"
    def visit(message):
        for field in message.field:
            if field.type_name.startswith(".hiai.proto."):
                field.type_name = ".ge.proto." + field.type_name[len(".hiai.proto."):]
        for child in message.nested_type:
            visit(child)
    for message in value.message_type:
        visit(message)
    return value.SerializeToString()


def pb2(path, descriptor=None, style="new"):
    descriptor = ge_descriptor() if descriptor is None else descriptor
    expression = ("_descriptor_pool.Default().AddSerializedFile(" + repr(descriptor) + ")" if style == "new" else
                  "_descriptor.FileDescriptor(name='ge_ir.proto', serialized_pb=" + repr(descriptor) + ")")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("raise RuntimeError('generated module must never execute')\nDESCRIPTOR = " + expression + "\n")
    return path


@pytest.mark.parametrize("style", ["new", "old"])
def test_real_descriptor_literals_are_data_and_default_hiai_remains_strict(tmp_path, style):
    descriptor = ge_descriptor()
    path = pb2(tmp_path / "ge_ir_pb2.py", descriptor, style)
    raw, evidence = air_schema.extract_torchair_descriptor(path)
    assert raw == descriptor
    assert evidence["pb2_execution"] == "NOT_EXECUTED"
    assert evidence["descriptor_sha256"] == hashlib.sha256(descriptor).hexdigest()
    assert evidence["schema_package"] == "ge.proto"
    assert evidence["pb2_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="package hiai.proto"):
        Schema(descriptor)
    schema = air_schema.load_torchair_schema(path)
    assert schema.decode(_blob(2, "relative-weight.bin"), ".ge.proto.AttrDef")["s"] == "relative-weight.bin"
    with pytest.raises(ValueError, match="unknown protobuf field"):
        schema.decode(_blob(199, b""), "GraphDef")
    with pytest.raises(ValueError, match="unknown protobuf message"):
        schema.decode(b"", ".hiai.proto.GraphDef")


@pytest.mark.parametrize("fault", ["shim", "duplicate", "expression", "package", "filename", "dependency", "invalid_descriptor"])
def test_shims_expressions_wrong_schema_duplicates_and_dependencies_are_rejected(tmp_path, fault):
    path = tmp_path / "ge_ir_pb2.py"
    if fault == "shim":
        path.write_text("from torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2 import *\n")
    elif fault == "duplicate":
        pb2(path)
        path.write_text(path.read_text() + path.read_text())
    elif fault == "expression":
        path.write_text("DESCRIPTOR = pool.AddSerializedFile(__import__('os').urandom(8))\n")
    elif fault == "invalid_descriptor":
        pb2(path, b"not a descriptor")
    else:
        descriptor = descriptor_pb2.FileDescriptorProto.FromString(ge_descriptor())
        if fault == "package":
            descriptor.package = "hiai.proto"
        elif fault == "filename":
            descriptor.name = "other.proto"
        else:
            descriptor.dependency.append("unavailable.proto")
        pb2(path, descriptor.SerializeToString())
    with pytest.raises(ValueError):
        air_schema.load_torchair_schema(path)


def test_source_descriptor_and_ast_limits_fail_closed(tmp_path, monkeypatch):
    path = pb2(tmp_path / "ge_ir_pb2.py")
    monkeypatch.setattr(air_schema, "MAX_DESCRIPTOR_BYTES", 8)
    with pytest.raises(ValueError, match="descriptor exceeds"):
        air_schema.load_torchair_schema(path)
    monkeypatch.setattr(air_schema, "MAX_DESCRIPTOR_BYTES", 8 << 20)
    monkeypatch.setattr(air_schema, "_MAX_AST_NODES", 1)
    with pytest.raises(ValueError, match="AST node"):
        air_schema.load_torchair_schema(path)
    monkeypatch.setattr(air_schema, "MAX_PB2_BYTES", 8)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("oversized source must not be opened"))
    with pytest.raises(ValueError, match="8 MiB"):
        air_schema.load_torchair_schema(path)


def test_explicit_ge_schema_takes_precedence_and_bad_selection_never_falls_back(tmp_path, selected, monkeypatch):
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph()))
    schema = pb2(tmp_path / "selected-ge-ir-pb2.py")
    monkeypatch.setattr(inspect, "load_schema", lambda *a: pytest.fail("explicit TorchAir schema must take precedence"))
    result = inspect.inspect_original_air(air, selected[0], torchair_pb2=schema)
    assert result["status"] == "STRUCTURE_OBSERVED"
    assert result["observed_format"] == "RAW_MODELDEF"
    assert result["schema"]["schema_package"] == "ge.proto"
    assert result["schema"]["selection"] == "EXPLICIT_TORCHAIR_PB2"
    assert "PRIVATE_TENSOR_BYTES" not in json.dumps(result)
    schema.write_text("from torchair import dangerous_shim\n")
    wrong = inspect.inspect_original_air(air, selected[0], torchair_pb2=schema)
    assert wrong["status"] == "OPAQUE" and wrong["schema"]["status"] == "UNAVAILABLE"
    assert not wrong["candidates"]


def test_distribution_and_fixed_syspath_discovery_does_not_import_or_recurse(tmp_path, monkeypatch):
    root = tmp_path / "site"
    selected = pb2(root / "torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py")
    shim = selected.with_name("ge_ir_pb2.py")
    shim.write_text("from torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2 import *\n")
    marker = tmp_path / "package-imported"
    (root / "torchair/__init__.py").write_text("open(" + repr(str(marker)) + ", 'w').write('bad')\n")
    relative = selected.relative_to(root)
    dist = SimpleNamespace(files=[relative], locate_file=lambda path: root / path)
    monkeypatch.setattr(air_schema.metadata, "distributions", lambda: [dist])
    monkeypatch.setattr(sys, "path", [str(root)])
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: pytest.fail("no recursive sys.path scan"))
    result = air_schema.discover_torchair_schemas()
    assert result["status"] == "FOUND" and not marker.exists()
    rows = {row["path"]: row for row in result["candidates"]}
    assert rows[str(selected)]["status"] == "LITERAL_DESCRIPTOR"
    assert rows[str(selected)]["discovered_by"] == ["DISTRIBUTION_FILES", "SYS_PATH_FIXED_CANDIDATE"]
    assert rows[str(shim)]["status"] == "UNSUPPORTED" and "shim" in rows[str(shim)]["error"]


def test_find_cli_needs_no_air_and_writes_only_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(air_schema, "discover_torchair_schemas", lambda: {"status": "NOT_FOUND", "candidates": []})
    script = Path(__file__).resolve().parents[1] / "targets/cpro/scripts/inspect_original_target_air.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--find-torchair-schema"])
    assert runpy.run_path(str(script))["main"]() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "NOT_FOUND"
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(sys, "argv", [str(script)])
    with pytest.raises(SystemExit, match="2"):
        runpy.run_path(str(script))["main"]()


def test_schema_source_mutation_discards_decoded_proof(tmp_path, selected, monkeypatch):
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph()))
    path = pb2(tmp_path / "selected.py")
    schema = air_schema.load_torchair_schema(path)
    decode = schema.decode
    def changed(*args):
        value = decode(*args)
        path.write_text(path.read_text() + "# mutation\n")
        return value
    monkeypatch.setattr(schema, "decode", changed)
    monkeypatch.setattr(inspect, "load_torchair_schema", lambda _: schema)
    result = inspect.inspect_original_air(air, torchair_pb2=path)
    assert result["status"] == "ERROR" and not result["candidates"]
    assert "observed_format" not in result


@pytest.mark.parametrize("present_shape", [True, False])
def test_full_internal_scalar_shape_and_unavailable_descriptors_do_not_relax_boundaries(tmp_path, selected, present_shape):
    # A present, empty ShapeDef is rank zero; an absent ShapeDef is unknown.
    descriptor = _blob(3, b"") if present_shape else b""
    from test_draft_ir_inspect import _int
    scalar = (_blob(1, "internal_scalar") + _blob(2, "Const") +
              _blob(34, _int(2, 7) + descriptor))
    air = tmp_path / "export.air"
    air.write_bytes(_blob(7, _graph() + _blob(6, scalar)))
    result = inspect.inspect_original_air(air, selected[0], full=True)
    assert result["status"] == "STRUCTURE_OBSERVED"
    graph = next(row["graph"] for row in result["candidates"] if row["status"] == "STRUCTURE_OBSERVED")
    observed = graph["operations"][-1]["output_desc"][0]
    if present_shape:
        assert observed["shape"] == []
    else:
        assert observed["status"] == "UNAVAILABLE"
    with pytest.raises(ValueError):
        inspect._desc({"dtype": 7}, {"DT_FLOAT16": 7})


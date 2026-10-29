"""Export inventories read payloads only on explicit hash and never stage files."""
import hashlib
import json
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import pytest

from qwen35_moe_dflash import target_export_inventory as inventory
from test_draft_ir_inspect import _blob, _desc, _attr, _int
from test_original_air_inspect import _graph
import test_draft_ir_inspect as fixture


def _schema(path):
    value = descriptor_pb2.FileDescriptorProto.FromString(fixture._schema())
    value.name, value.package = "ge_ir.proto", "ge.proto"
    for enum in value.enum_type:
        for item in enum.value:
            item.name = {"HALF": "DT_FLOAT16", "FLOAT": "DT_FLOAT", "UNDEFINED": "DT_UNDEFINED"}.get(item.name, item.name)
    def visit(message):
        for field in message.field:
            if field.type_name.startswith(".hiai.proto."):
                field.type_name = field.type_name.replace(".hiai.proto.", ".ge.proto.")
        for nested in message.nested_type:
            visit(nested)
    for message in value.message_type:
        visit(message)
    descriptor = value.SerializeToString()
    path.write_text("raise RuntimeError('must not import this file')\nDESCRIPTOR = p.AddSerializedFile(" + repr(descriptor) + ")\n")
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("ge.proto.ModelDef"))


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "run"))
    folder = tmp_path / "source/glm_export_air_cpro_40block_64_quant"
    folder.mkdir(parents=True)
    air = folder / "export.air"
    graph = _graph(sequence=64, constants=0)
    extra = (_blob(1, "feature") + _blob(2, "ConcatD") +
             _blob(34, _desc(shape=())))  # FP16 producer shape deliberately unmaterialized.
    graph += _blob(6, extra)
    cls = _schema(tmp_path / "ge_ir_by_protoc_3_19_pb2.py")
    model = cls.FromString(_blob(7, graph))
    net = next(node for node in model.graph[0].op if node.type == "NetOutput")
    net.input.append("feature:0")
    net.input_desc.add().ParseFromString(_desc(shape=(1, 64, 16384)))
    for index, basename in enumerate(("weight0", "weight1", "weight0")):
        node = model.graph[0].op.add()
        node.name, node.type = "weight_" + str(index), "FileConstant"
        node.output_desc.add().ParseFromString(_desc(dtype=1, shape=()))
        attr = node.attr.add(); attr.key = "file_path"
        attr.value.s = ("./" + folder.name + "/" + basename).encode()
        attr = node.attr.add(); attr.key = "dtype"
        attr.value.i = 0  # GE attribute domain is separate from schema descriptor FLOAT=1.
    air.write_bytes(model.SerializeToString())
    weights = {name: folder / name for name in ("weight0", "weight1")}
    weights["weight0"].write_bytes(b"first weight")
    weights["weight1"].write_bytes(b"second weight")
    return {"air": air, "folder": folder, "schema": tmp_path / "ge_ir_by_protoc_3_19_pb2.py",
            "out": tmp_path / "run/inventory", "weights": weights, "model_class": cls}


def test_default_is_stat_only_unique_refs_and_retains_separate_raw_dtype_metadata(source, monkeypatch):
    actual_hash = inventory._hash_file
    def no_weight_read(path, expected):
        assert path == source["air"], "default inventory must not hash weight payloads"
        return actual_hash(path, expected)
    monkeypatch.setattr(inventory, "_hash_file", no_weight_read)
    before = {path for path in source["folder"].iterdir()}
    result = inventory.inventory_target_export(source["air"], source["out"], torchair_pb2=source["schema"])
    assert result["status"] == "INVENTORIED" and result["weight_identity"] == "STAT_ONLY"
    assert result["declared_abi"]["status"] == "MATCHED_INTERFACE_METADATA"
    assert result["declared_abi"]["producer_shape_status"] == "NOT_MATERIALIZED_OR_SCALAR_AMBIGUOUS"
    assert result["unique_reference_count"] == 2 and result["file_constant_count"] == 3
    assert result["total_weight_bytes"] == len(b"first weight") + len(b"second weight")
    assert all(row["sha256"] is None for row in result["files"])
    assert len(result["files"][0]["nodes"]) == 2
    assert result["file_constants"][0]["raw_output_desc"][0]["dtype"] == 1
    assert result["air_preservation"] == "UNCHANGED_STAT_AND_SHA256"
    assert result == json.loads((source["out"] / "inventory.json").read_text())
    assert before == {path for path in source["folder"].iterdir()}
    assert list(source["out"].iterdir()) == [source["out"] / "inventory.json"]


def test_explicit_stream_hash_reads_each_unique_payload_once_and_copies_nothing(source, monkeypatch):
    original = inventory._hash_file
    calls = []
    def traced(path, expected):
        if path != source["air"]:
            calls.append(path)
        return original(path, expected)
    monkeypatch.setattr(inventory, "_hash_file", traced)
    result = inventory.inventory_target_export(source["air"], source["out"],
                                               torchair_pb2=source["schema"], hash_weights=True)
    assert result["status"] == "INVENTORIED" and result["weight_identity"] == "SHA256_COMPLETE"
    assert len(calls) == 2
    for row in result["files"]:
        assert row["sha256"] == hashlib.sha256(source["weights"][row["basename"]].read_bytes()).hexdigest()
        assert row["reference"].startswith("./" + source["folder"].name + "/")
        assert row["relative_path"] == source["folder"].name + "/" + row["basename"]
    assert inventory.HASH_CHUNK_BYTES == 1 << 20
    assert len(list(source["out"].iterdir())) == 1


@pytest.mark.parametrize("fault", ["missing", "symlink_leaf", "symlink_parent", "absolute", "traversal", "nested", "feature_shape", "outputs", "dtype"])
def test_missing_unsafe_paths_and_wrong_interface_are_blocked(source, fault):
    if fault == "missing":
        source["weights"]["weight0"].unlink()
    elif fault == "symlink_leaf":
        source["weights"]["weight0"].unlink()
        source["weights"]["weight0"].symlink_to(source["weights"]["weight1"])
    elif fault == "symlink_parent":
        # The AIR itself must reject a symlink in any parent component.
        original = source["folder"]
        moved = original.with_name("moved")
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
    else:
        model = source["model_class"].FromString(source["air"].read_bytes())
        net = next(node for node in model.graph[0].op if node.type == "NetOutput")
        constant = next(node for node in model.graph[0].op if node.type == "FileConstant")
        if fault in {"absolute", "traversal", "nested"}:
            value = {"absolute": "/outside/weight0", "traversal": "../outside", "nested": "prefix/nested/weight0"}[fault]
            next(row for row in constant.attr if row.key == "file_path").value.s = value.encode()
        elif fault == "feature_shape":
            net.input_desc[-1].shape.dim[-1] = 2048
        elif fault == "outputs":
            del net.input[-1]; del net.input_desc[-1]
        else:
            net.input_desc[-1].dtype = 1
        source["air"].write_bytes(model.SerializeToString())
    result = inventory.inventory_target_export(source["air"], source["out"],
                                               torchair_pb2=source["schema"], hash_weights=True)
    assert result["status"] == "BLOCKED" and result["error"]
    assert result["weight_identity"] != "SHA256_COMPLETE"


def test_discovery_requires_one_descriptor_hash_and_chooses_smallest_source(source, monkeypatch):
    selected = inventory.load_torchair_schema(source["schema"])
    evidence = selected.evidence
    rows = [{"path": str(source["schema"]), "status": "LITERAL_DESCRIPTOR",
             "descriptor_sha256": evidence["descriptor_sha256"], "pb2_bytes": 500},
            {"path": "unselected-large.py", "status": "LITERAL_DESCRIPTOR",
             "descriptor_sha256": evidence["descriptor_sha256"], "pb2_bytes": 900}]
    monkeypatch.setattr(inventory, "discover_torchair_schemas", lambda: {"candidates": rows})
    result = inventory.inventory_target_export(source["air"], source["out"], discover_schema=True)
    assert result["status"] == "INVENTORIED"
    assert result["schema"]["pb2"] == str(source["schema"])
    rows[1]["descriptor_sha256"] = "0" * 64
    result = inventory.inventory_target_export(source["air"], source["out"].with_name("ambiguous"), discover_schema=True)
    assert result["status"] == "BLOCKED" and "multiple TorchAir descriptor hashes" in result["error"]


def test_payload_mutation_during_hash_cannot_become_a_reuse_manifest(source, monkeypatch):
    original = inventory._hash_file
    def changed(path, expected):
        result = original(path, expected)
        if path == source["weights"]["weight0"]:
            path.write_bytes(b"changed payload")
        return result
    monkeypatch.setattr(inventory, "_hash_file", changed)
    result = inventory.inventory_target_export(source["air"], source["out"],
                                               torchair_pb2=source["schema"], hash_weights=True)
    assert result["status"] == "BLOCKED" and result["source_preservation"] == "CHANGED"
    assert result["weight_identity"] == "NOT_VERIFIED"


def test_report_is_exclusive_and_no_implicit_schema_selection(source):
    first = inventory.inventory_target_export(source["air"], source["out"])
    assert first["status"] == "BLOCKED" and "declare --torchair-pb2" in first["error"]
    before = (source["out"] / "inventory.json").read_bytes()
    with pytest.raises(FileExistsError):
        inventory.inventory_target_export(source["air"], source["out"], torchair_pb2=source["schema"])
    assert (source["out"] / "inventory.json").read_bytes() == before


def test_explicit_55_shape_is_preserved_for_b_and_never_silently_changed_to64(source):
    model = source["model_class"].FromString(source["air"].read_bytes())
    net = next(node for node in model.graph[0].op if node.type == "NetOutput")
    net.input_desc[-1].shape.dim[1] = 55
    source["air"].write_bytes(model.SerializeToString())
    rejected = inventory.inventory_target_export(source["air"], source["out"], torchair_pb2=source["schema"])
    assert rejected["status"] == "BLOCKED"
    result = inventory.inventory_target_export(source["air"], source["out"].with_name("b55"),
                                               torchair_pb2=source["schema"], sequence_length=55)
    assert result["status"] == "INVENTORIED"
    assert result["declared_abi"]["feature_shape"] == [1, 55, 16384]


def test_explicit_reference_root_and_progress_are_metadata_only(source):
    rows = []
    result = inventory.inventory_target_export(source["air"], source["out"], torchair_pb2=source["schema"],
        reference_root=source["folder"].parent, progress=rows.append)
    assert result["status"] == "INVENTORIED"
    assert [row["completed"] for row in rows] == [1, 2]
    assert all(not row["payload_hashed"] for row in rows)


def test_large_payload_hash_requests_are_bounded_to_one_mib(source, monkeypatch):
    payload = b"x" * (2 * inventory.HASH_CHUNK_BYTES + 17)
    source["weights"]["weight0"].write_bytes(payload)
    original = inventory.os.fdopen
    reads = []
    class Bounded:
        def __init__(self, file): self.file = file
        def __enter__(self): return self
        def __exit__(self, *args): return self.file.__exit__(*args)
        def __getattr__(self, name): return getattr(self.file, name)
        def read(self, size):
            assert 0 < size <= 1 << 20
            reads.append(size)
            return self.file.read(size)
    monkeypatch.setattr(inventory.os, "fdopen", lambda *a, **k: Bounded(original(*a, **k)))
    result = inventory.inventory_target_export(source["air"], source["out"],
        torchair_pb2=source["schema"], hash_weights=True)
    assert result["status"] == "INVENTORIED" and reads.count(1 << 20) == 2
    weight = next(row for row in result["files"] if row["basename"] == "weight0")
    assert weight["sha256"] == hashlib.sha256(payload).hexdigest()


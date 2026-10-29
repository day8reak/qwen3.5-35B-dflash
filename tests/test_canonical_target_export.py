"""Small real-encoded AIR -> canonical boundary -> streamed identity inventory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from google.protobuf import descriptor_pb2
import pytest

from qwen35_moe_dflash import canonical_target_export as canonical
from qwen35_moe_dflash import target_export_inventory as inventory
from qwen35_moe_dflash.air_schema import extract_torchair_descriptor
from test_target_output_order import source as output_order_fixture


@pytest.fixture
def source(tmp_path, monkeypatch):
    schema, cls, model = output_order_fixture.__wrapped__(tmp_path)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "run"))
    root = tmp_path / "origin"
    air = root / "export-folder/export.air"
    air.parent.mkdir(parents=True)
    air.write_bytes(model.SerializeToString())
    weight = root / "original_large_sidecars/weight.bin"
    weight.parent.mkdir()
    weight.write_bytes(b"unchanged existing quantized sidecar")
    return {"air": air, "weight": weight, "root": root, "schema": Path(schema.evidence["pb2"]),
            "out": tmp_path / "run/canonical-export", "model": model, "model_class": cls}


def _run(source, **kwargs):
    return canonical.canonicalize_target_export(source["air"], source["out"],
        torchair_pb2=source["schema"], **kwargs)


def _snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def _inventory(source):
    return json.loads((source["out"] / "inventory/inventory.json").read_text())


def test_real_inventory_hashes_existing_source_weights_and_writes_only_small_air_metadata(source):
    before = _snapshot(source["root"])
    source_air = source["air"].read_bytes()
    rows = []
    result = _run(source, hash_weights=True, progress=rows.append)
    assert result["status"] == "EXPORT_READY", result
    assert result["weight_identity"] == "SHA256_COMPLETE"
    assert result["validation"]["status"] == "PASS_METADATA_ORDER_ONLY"
    assert result["feature_output_index"] == 81 and result["unique_reference_count"] == 1
    assert result["original_air"]["sha256"] == hashlib.sha256(source_air).hexdigest()
    assert result["original_preservation"] == "UNCHANGED_STAT_AND_SHA256"
    assert result["model_compilation"] == result["device_execution"] == result["numerical_validation"] == "NOT_RUN"
    assert result["weight_copies"] == "NONE"
    assert _snapshot(source["root"]) == before
    actual = source["model_class"].FromString((source["out"] / "export.air").read_bytes())
    net = next(node for node in actual.graph[0].op if node.type == "NetOutput")
    assert net.input[81] == "ConcatV2_100:0"
    assert list(net.input_desc[81].shape.dim) == [1, 64, 16384]
    manifest = _inventory(source)
    assert manifest["reference_root"] == str(source["root"])
    assert manifest["declared_abi"]["status"] == "MATCHED_INTERFACE_METADATA"
    assert manifest["files"][0]["sha256"] == hashlib.sha256(source["weight"].read_bytes()).hexdigest()
    assert result["inventory"]["sha256"] == hashlib.sha256((source["out"] / "inventory/inventory.json").read_bytes()).hexdigest()
    assert rows == [{"completed": 1, "total": 1, "basename": "weight.bin", "bytes": source["weight"].stat().st_size,
                     "payload_hashed": True}]
    assert set(_snapshot(source["out"])) == {"export.air", "normalization.json", "validation.json", "result.json", "inventory/inventory.json"}


def test_payload_stream_reads_are_bounded_and_no_payload_copy_occurs(source, monkeypatch):
    source["weight"].write_bytes(b"x" * ((2 << 20) + 19))
    original = inventory.os.fdopen
    sizes = []
    class Bounded:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def __getattr__(self, name): return getattr(self.stream, name)
        def read(self, size):
            assert 0 < size <= 1 << 20
            sizes.append(size)
            return self.stream.read(size)
    monkeypatch.setattr(inventory.os, "fdopen", lambda *a, **k: Bounded(original(*a, **k)))
    result = _run(source, hash_weights=True)
    assert result["status"] == "EXPORT_READY", result
    assert sizes.count(1 << 20) == 2
    assert result["total_weight_bytes"] == (2 << 20) + 19
    assert sum(path.stat().st_size for path in source["out"].rglob("*") if path.is_file()) < 1 << 20


def test_completed_resume_rechecks_stat_without_hashing_payload_again(source, monkeypatch):
    first = _run(source, hash_weights=True)
    assert first["status"] == "EXPORT_READY"
    old_manifest = (source["out"] / "inventory/inventory.json").read_bytes()
    old_air = (source["out"] / "export.air").read_bytes()
    old_result = (source["out"] / "result.json").read_bytes()
    original = inventory._hash_file
    def no_payload(path, expected):
        assert path != source["weight"], "completed resume must not reread weight payloads"
        return original(path, expected)
    monkeypatch.setattr(inventory, "_hash_file", no_payload)
    rows = []
    result = _run(source, hash_weights=True, resume=True, progress=rows.append)
    assert result["status"] == "EXPORT_READY", result
    assert result["inventory"]["reused"] is True and not rows
    assert result["weight_identity"] == "SHA256_COMPLETE"
    assert (source["out"] / "inventory/inventory.json").read_bytes() == old_manifest
    assert (source["out"] / "export.air").read_bytes() == old_air
    archived = [path.read_bytes() for path in (source["out"] / "attempts").rglob("*") if path.is_file()]
    assert old_result in archived
    # A second complete resume must retain the same manifest, never degrade
    # the cached hash identity or create another AIR/weight payload.
    again = _run(source, hash_weights=True, resume=True)
    assert again["status"] == "EXPORT_READY" and again["inventory"]["reused"] is True
    assert (source["out"] / "inventory/inventory.json").read_bytes() == old_manifest


def test_stat_only_inventory_can_be_explicitly_upgraded_to_stream_hash_with_old_evidence_retained(source):
    first = _run(source)
    assert first["status"] == "EXPORT_READY" and first["weight_identity"] == "STAT_ONLY"
    old_inventory = (source["out"] / "inventory/inventory.json").read_bytes()
    result = _run(source, hash_weights=True, resume=True)
    assert result["status"] == "EXPORT_READY", result
    assert result["weight_identity"] == "SHA256_COMPLETE" and result["inventory"]["reused"] is False
    archived = [path.read_bytes() for path in (source["out"] / "attempts").rglob("*") if path.is_file()]
    assert old_inventory in archived


def test_failed_owned_inventory_retry_retains_old_evidence_and_reuses_same_air(source):
    content = source["weight"].read_bytes()
    source["weight"].unlink()
    first = _run(source, hash_weights=True)
    assert first["status"] == "BLOCKED" and first["inventory"]["status"] == "BLOCKED"
    failed_inventory = (source["out"] / "inventory/inventory.json").read_bytes()
    old_air = (source["out"] / "export.air").read_bytes()
    source["weight"].write_bytes(content)
    result = _run(source, hash_weights=True, resume=True)
    assert result["status"] == "EXPORT_READY", result
    assert result["inventory"]["reused"] is False
    assert (source["out"] / "export.air").read_bytes() == old_air
    archived = [path.read_bytes() for path in (source["out"] / "attempts").rglob("*") if path.is_file()]
    assert failed_inventory in archived


@pytest.mark.parametrize("fault", ["unknown_out", "source_air", "normalized_air", "schema"])
def test_unknown_or_changed_identity_is_rejected_without_touching_retained_evidence(source, fault):
    assert _run(source, hash_weights=True)["status"] == "EXPORT_READY"
    if fault == "unknown_out": (source["out"] / "unowned.user-file").write_bytes(b"keep this")
    elif fault == "source_air":
        source["model"].name = "different original source receipt"
        source["air"].write_bytes(source["model"].SerializeToString())
    elif fault == "normalized_air":
        model = source["model_class"].FromString((source["out"] / "export.air").read_bytes())
        model.name = "different normalized file"
        (source["out"] / "export.air").write_bytes(model.SerializeToString())
    else:
        descriptor, _ = extract_torchair_descriptor(source["schema"])
        value = descriptor_pb2.FileDescriptorProto.FromString(descriptor)
        value.message_type.add(name="UnusedButDifferentSchema")
        source["schema"].write_text("DESCRIPTOR = p.AddSerializedFile(" + repr(value.SerializeToString()) + ")\n")
    before = _snapshot(source["out"])
    result = _run(source, hash_weights=True, resume=True)
    assert result["status"] == "BLOCKED", result
    assert _snapshot(source["out"]) == before


def test_changed_weight_stat_blocks_resume_before_payload_read(source, monkeypatch):
    assert _run(source, hash_weights=True)["status"] == "EXPORT_READY"
    source["weight"].write_bytes(b"changed source weight")
    actual_hash = inventory._hash_file
    def no_payload(path, expected):
        assert path != source["weight"], "changed payload metadata must block cached reuse before a hash read"
        return actual_hash(path, expected)
    monkeypatch.setattr(inventory, "_hash_file", no_payload)
    result = _run(source, hash_weights=True, resume=True)
    assert result["status"] == "BLOCKED" and "payload metadata changed" in result["error"]


def test_existing_unknown_task_is_not_adopted_or_overwritten(source):
    source["out"].mkdir(parents=True)
    (source["out"] / "result.json").write_text('{"artifact_kind":"another-task"}\n')
    before = _snapshot(source["out"])
    result = _run(source, resume=True)
    assert result["status"] == "BLOCKED"
    assert _snapshot(source["out"]) == before


def test_normalized_air_is_published_as_complete_bytes_with_no_partial_left(source, monkeypatch):
    original_link = canonical.os.link
    observations = []
    def checked_link(partial, destination, *args, **kwargs):
        assert not Path(destination).exists()
        raw = Path(partial).read_bytes()
        model = source["model_class"].FromString(raw)
        net = next(node for node in model.graph[0].op if node.type == "NetOutput")
        assert len(net.input) == 82 and net.input[81] == "ConcatV2_100:0"
        observations.append(hashlib.sha256(raw).hexdigest())
        return original_link(partial, destination, *args, **kwargs)
    monkeypatch.setattr(canonical.os, "link", checked_link)
    result = _run(source, hash_weights=True)
    assert result["status"] == "EXPORT_READY", result
    assert observations == [result["air"]["sha256"]]
    assert not (source["out"] / ".export.air.partial").exists()


def test_cli_executes_metadata_only_path_and_publishes_explicit_inventory(source):
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / "targets/cpro/scripts/canonicalize_target_export.py"),
        "--air", str(source["air"]), "--torchair-pb2", str(source["schema"]),
        "--hash-weights", "--sequence-length", "64", "--output", str(source["out"])]
    completed = subprocess.run(command, env=dict(os.environ), capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "EXPORT_READY" and result["weight_identity"] == "SHA256_COMPLETE"
    assert "hashed 1/1" in completed.stderr


def test_output_parent_symlink_cannot_be_hidden_by_path_resolution(source):
    run = source["out"].parent
    run.mkdir()
    target = run / "target"
    target.mkdir()
    link = run / "parent-link"
    link.symlink_to(target, target_is_directory=True)
    attempted = link / "new-task"
    before = _snapshot(target)
    try:
        result = canonical.canonicalize_target_export(source["air"], attempted,
            torchair_pb2=source["schema"], hash_weights=True)
    except (ValueError, OSError):
        result = {"status": "BLOCKED"}
    assert result["status"] == "BLOCKED", result
    assert _snapshot(target) == before and not (target / "new-task").exists()


def test_schema_source_mutation_during_inventory_cannot_publish_ready_status(source, monkeypatch):
    original = inventory._hash_file
    mutated = []
    def changed(path, expected):
        digest = original(path, expected)
        if path == source["weight"]:
            source["schema"].write_bytes(source["schema"].read_bytes() + b"\n")
            mutated.append(True)
        return digest
    monkeypatch.setattr(inventory, "_hash_file", changed)
    result = _run(source, hash_weights=True)
    assert mutated and result["status"] == "BLOCKED", result
    assert "schema" in result["error"].lower()


def test_modified_failed_inventory_is_not_adopted_as_owned_retry_metadata(source):
    source["weight"].unlink()
    result = _run(source, hash_weights=True)
    assert result["status"] == "BLOCKED" and result["inventory"]["status"] == "BLOCKED"
    manifest_path = source["out"] / "inventory/inventory.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["error"] = "unowned replacement metadata"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    before = _snapshot(source["out"])
    result = _run(source, hash_weights=True, resume=True)
    assert result["status"] == "BLOCKED", result
    assert _snapshot(source["out"]) == before

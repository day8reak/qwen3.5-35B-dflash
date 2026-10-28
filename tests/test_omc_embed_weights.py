"""Streaming repack contracts using synthetic protobuf/ELF, never vendor code."""
import hashlib
from pathlib import Path
import shutil
import struct

import pytest

from qwen35_moe_dflash import omc_embed_weights as module
from qwen35_moe_dflash.hashing import sha256_file
from qwen35_moe_dflash.omc_container import read_container, read_partition
from test_omc_wire import blob, elf, integer


def _node(name, offset, size):
    result = blob(1, name) + blob(2, "GraphOp")
    for key, value in (("graphop_weight_offset", offset), ("graphop_weight_size", size)):
        result += blob(10, blob(1, key) + blob(2, integer(3, value)))
    return result


def _image(nodes, *, kinds=(0, 1, 2), words=None, weights=b""):
    main = blob(7, blob(1, "draft") + b"".join(blob(6, node) for node in nodes))
    payloads = [main, weights, b"opaque-compiled-program\x00\xff"]
    offsets = [0, len(main), len(main) + len(weights)] if words is None else words
    table = struct.pack("<I", 3) + b"".join(struct.pack("<III", kind, off, len(data))
        for kind, off, data in zip(kinds, offsets, payloads))
    # Nonzero opaque header bytes must survive byte for byte.
    header = bytearray((index * 11) % 256 for index in range(256))
    header[:4] = b"IMOD"
    struct.pack_into("<I", header, 0x4C, len(table) + sum(map(len, payloads)))
    return bytes(header) + table + b"".join(payloads)


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "model.omc"
    # Graph order deliberately differs from offset order. Never sort filenames
    # lexically: SubGraph_10's offset follows SubGraph_2 in this fixture.
    source.write_bytes(_image([_node("SubGraph_10", 5, 7), _node("SubGraph_2", 0, 5)]))
    sidecars = []
    for name, data in (("SubGraph_10", b"abcdefg"), ("SubGraph_2", b"12345")):
        path = tmp_path / (name + ".weight")
        path.write_bytes(data)
        sidecars.append({"path": str(path), "bytes": len(data), "sha256": sha256_file(path)})
    library = tmp_path / "libai_fmk_graph_proto_host.so"
    library.write_bytes(elf()[0])
    return {"source": source, "output": tmp_path / "embedded.omc", "library": library,
            "sidecars": sidecars}


def _convert(case, **kwargs):
    return module.embed_external_weights(case["source"], case["output"], case["library"],
        expected_source_sha256=kwargs.pop("expected_source_sha256", sha256_file(case["source"])),
        sidecars=case["sidecars"], **kwargs)


def test_streams_correct_order_and_preserves_graph_tasks_header_and_sources(case, monkeypatch):
    before = {path: sha256_file(path) for path in case["source"].parent.iterdir()}
    original = case["source"].read_bytes()
    monkeypatch.setattr(module, "_CHUNK_BYTES", 3)
    updates = []
    report = _convert(case, progress=updates.append)
    assert report["status"] == "EMBEDDED"
    assert report["device_execution"] == "NOT_RUN"
    assert report["numerical_validation"] == "NOT_EVALUATED"
    assert report["source"]["sha256"] == before[case["source"]]
    assert report["schema"]["library_sha256"] == before[case["library"]]
    assert [row["graphop_name"] for row in report["weights"]] == ["SubGraph_2", "SubGraph_10"]
    assert len(updates) == 2
    assert report["output"]["bytes"] == len(original) + 12
    assert report["output"]["sha256"] == sha256_file(case["output"])
    current = case["output"].read_bytes()
    assert current == _image([_node("SubGraph_10", 5, 7), _node("SubGraph_2", 0, 5)],
                             weights=b"12345abcdefg")
    assert current[:0x4C] == original[:0x4C]
    assert current[0x50:256] == original[0x50:256]
    source_parts = read_container(case["source"])["partitions"]
    output_parts = read_container(case["output"])["partitions"]
    assert read_partition(case["output"], output_parts[1]) == b"12345abcdefg"
    for index in (0, 2):
        assert read_partition(case["output"], output_parts[index]) == read_partition(case["source"], source_parts[index])
        assert report["partitions"][index]["preserved"] == "BYTE_IDENTICAL"
    assert output_parts[2]["table_word"] == output_parts[0]["bytes"] + 12
    assert all(sha256_file(path) == digest for path, digest in before.items())
    assert not list(case["output"].parent.glob("*.partial"))


@pytest.mark.parametrize("nodes, match", [
    ([_node("SubGraph_2", 1, 5), _node("SubGraph_10", 6, 7)], "gapless"),
    ([_node("SubGraph_2", 0, 5), _node("SubGraph_10", 4, 7)], "gapless"),
    ([_node("SubGraph_2", 0, 5), _node("SubGraph_2", 5, 7)], "unique"),
    ([_node("../SubGraph_2", 0, 5)], "safe filename"),
    ([_node("SubGraph_2", 0, 6), _node("SubGraph_10", 6, 7)], "size differs"),
    ([_node("SubGraph_2", 0, 5)], "cover every"),
    ([_node("unknown", 0, 5)], "missing hash-bound"),
    ([_node("SubGraph_2", -1, 5)], "uint32"),
])
def test_rejects_inconsistent_graph_weight_ownership_without_output(case, nodes, match):
    case["source"].write_bytes(_image(nodes))
    with pytest.raises(ValueError, match=match):
        _convert(case)
    assert not case["output"].exists()
    assert not list(case["output"].parent.glob("*.partial"))


@pytest.mark.parametrize("fault, match", [
    ("unknown-partition", "partitions"), ("nonempty", "empty weights"),
    ("table-offset", "cumulative"), ("source-hash", "source OMC hash"),
    ("schema-hash", "schema library hash"), ("sidecar-size", "size differs"),
    ("sidecar-hash", "sidecar hash differs"), ("sidecar-symlink", "non-symlink"),
    ("source-symlink", "non-symlink"), ("sidecar-outside", "beside"),
])
def test_rejects_unbound_or_unsupported_inputs_and_removes_partial(case, fault, match):
    kwargs = {}
    nodes = [_node("SubGraph_2", 0, 5), _node("SubGraph_10", 5, 7)]
    if fault == "unknown-partition":
        case["source"].write_bytes(_image(nodes, kinds=(0, 8, 2)))
    elif fault == "nonempty":
        case["source"].write_bytes(_image(nodes, weights=b"x"))
    elif fault == "table-offset":
        case["source"].write_bytes(_image(nodes, words=[1, 2, 3]))
    elif fault == "source-hash":
        kwargs["expected_source_sha256"] = "0" * 64
    elif fault == "schema-hash":
        kwargs["expected_schema_sha256"] = "0" * 64
    elif fault == "sidecar-size":
        case["sidecars"][0]["bytes"] += 1
    elif fault == "sidecar-hash":
        case["sidecars"][0]["sha256"] = "0" * 64
    elif fault in ("sidecar-symlink", "source-symlink"):
        path = Path(case["sidecars"][0]["path"]) if fault == "sidecar-symlink" else case["source"]
        actual = path.with_suffix(".actual")
        path.rename(actual)
        path.symlink_to(actual)
    elif fault == "sidecar-outside":
        case["sidecars"][0]["path"] = str(case["source"].parent.parent / "SubGraph_10.weight")
    with pytest.raises(ValueError, match=match):
        _convert(case, **kwargs)
    assert not case["output"].exists()
    assert not list(case["output"].parent.glob("*.partial"))


def test_prechecks_disk_space_before_creating_output(case, monkeypatch):
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(OSError, match="needs") as caught:
        _convert(case)
    assert caught.value.errno == 28
    assert not case["output"].exists()
    assert not list(case["output"].parent.glob("*.partial"))


def test_existing_output_is_never_overwritten(case):
    case["output"].write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        _convert(case)
    assert case["output"].read_bytes() == b"existing"


def test_concurrent_destination_is_never_overwritten(case):
    def concurrent_writer(_):
        case["output"].write_bytes(b"other writer")
    with pytest.raises(FileExistsError):
        _convert(case, progress=concurrent_writer)
    assert case["output"].read_bytes() == b"other writer"
    assert not list(case["output"].parent.glob("*.partial"))


def test_sidecar_mutation_before_stream_rejects_and_removes_partial(case):
    def mutation(_):
        Path(case["sidecars"][1]["path"]).write_bytes(b"12346")
    with pytest.raises(ValueError, match="source changed|hash differs"):
        _convert(case, progress=mutation)
    assert not case["output"].exists()
    assert not list(case["output"].parent.glob("*.partial"))


def test_uint32_output_limit_rejects_before_large_copy(case, monkeypatch):
    monkeypatch.setattr(module, "_UINT32_MAX", 1024)
    case["sidecars"][0]["bytes"] = 900
    path = Path(case["sidecars"][0]["path"])
    path.write_bytes(b"x" * 900)
    case["sidecars"][0]["sha256"] = sha256_file(path)
    case["source"].write_bytes(_image([_node("SubGraph_2", 0, 5), _node("SubGraph_10", 5, 900)]))
    with pytest.raises(ValueError, match="native uint32"):
        _convert(case)
    assert not case["output"].exists()

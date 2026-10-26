import hashlib
import struct

import pytest

from qwen35_moe_dflash.omc_container import read_container, read_partition


def container_bytes(parts=((0, 0, b"model"), (2, 0, b"cpu graph"))):
    table = struct.pack("<I", len(parts)) + b"".join(
        struct.pack("<III", kind, word, len(payload)) for kind, word, payload in parts)
    payload = table + b"".join(payload for _, _, payload in parts)
    header = bytearray(256)
    header[:4] = b"IMOD"
    struct.pack_into("<I", header, 0x4C, len(payload))
    return header + payload


def test_reads_directory_and_exact_payloads_without_mutating_source(tmp_path):
    image = container_bytes(((2, 0xFFFFFFFF, b"cpu graph"), (17, 913, b"other"), (0, 234, b"model")))
    path = tmp_path / "target.omc"
    path.write_bytes(image)
    before = hashlib.sha256(image).hexdigest()
    result = read_container(path)
    assert result == {
        "layout": "IMOD_256_TABLE12_REFERENCE_LAYOUT",
        "file_bytes": len(image),
        "header_sha256": hashlib.sha256(image[:256]).hexdigest(),
        "partitions": [
            {"type": 2, "table_word": 0xFFFFFFFF, "offset": 296, "bytes": 9},
            {"type": 17, "table_word": 913, "offset": 305, "bytes": 5},
            {"type": 0, "table_word": 234, "offset": 310, "bytes": 5},
        ],
    }
    assert [read_partition(path, row) for row in result["partitions"]] == [b"cpu graph", b"other", b"model"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("fault, message", [
    ("short-header", "truncated 256-byte header"),
    ("magic", "magic is not IMOD"),
    ("header-length", "header payload length"),
    ("short-count", "truncated partition count"),
    ("zero-count", "partition count"),
    ("large-count", "partition count"),
    ("short-table", "partition table exceeds"),
    ("overflow", "overflow"),
    ("outside", "partition extends beyond"),
    ("unclaimed", "do not cover"),
])
def test_rejects_malformed_layout_without_payload_decode(tmp_path, fault, message):
    image = container_bytes()
    if fault == "short-header":
        image = image[:255]
    elif fault == "magic":
        image[:4] = b"OMC0"
    elif fault == "header-length":
        struct.pack_into("<I", image, 0x4C, len(image))
    elif fault == "short-count":
        image = image[:258]
        struct.pack_into("<I", image, 0x4C, 2)
    elif fault == "zero-count":
        image = container_bytes(())
    elif fault == "large-count":
        struct.pack_into("<I", image, 256, 36)
    elif fault == "short-table":
        struct.pack_into("<I", image, 256, 4)
    elif fault == "overflow":
        struct.pack_into("<I", image, 268, 0xFFFFFFFF)
    elif fault == "outside":
        struct.pack_into("<I", image, 268, len(image))
    elif fault == "unclaimed":
        image.extend(b"unclaimed")
        struct.pack_into("<I", image, 0x4C, len(image) - 256)
    path = tmp_path / "target.omc"
    path.write_bytes(image)
    with pytest.raises(ValueError, match=message):
        read_container(path)


@pytest.mark.parametrize("types, required", [((1, 2), 0), ((0, 1), 2), ((0, 0, 2), 0), ((0, 2, 2), 2)])
def test_requires_unambiguous_model_and_compiled_payload_partitions(tmp_path, types, required):
    path = tmp_path / "target.omc"
    path.write_bytes(container_bytes(tuple((kind, 0, b"x") for kind in types)))
    with pytest.raises(ValueError, match=f"exactly one type {required}"):
        read_container(path)


def test_accepts_maximum_table_and_zero_length_unrelated_partition(tmp_path):
    parts = ((0, 100, b"m"), (2, 200, b"c")) + tuple((17, i, b"") for i in range(33))
    path = tmp_path / "target.omc"
    path.write_bytes(container_bytes(parts))
    result = read_container(path)
    assert len(result["partitions"]) == 35
    assert read_partition(path, result["partitions"][-1]) == b""


@pytest.mark.parametrize("record", [None, {}, {"offset": -1, "bytes": 1},
    {"offset": 0, "bytes": -1}, {"offset": True, "bytes": 1},
    {"offset": 0, "bytes": False}, {"offset": 0.0, "bytes": 1}])
def test_rejects_invalid_range_records_before_reading(tmp_path, record):
    with pytest.raises(ValueError):
        read_partition(tmp_path / "nonexistent.omc", record)


@pytest.mark.parametrize("budget", [0, -1, True, 2.5])
def test_rejects_invalid_budgets_before_reading(tmp_path, budget):
    with pytest.raises(ValueError, match="positive integer"):
        read_partition(tmp_path / "nonexistent.omc", {"offset": 0, "bytes": 0}, budget)


def test_partition_budget_prevents_large_allocation_or_file_access(tmp_path):
    with pytest.raises(ValueError, match="byte budget"):
        read_partition(tmp_path / "nonexistent.omc", {"offset": 0, "bytes": 65 << 20})


def test_detects_source_truncation_after_directory_read(tmp_path):
    path = tmp_path / "target.omc"
    path.write_bytes(container_bytes())
    record = read_container(path)["partitions"][1]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="extends beyond"):
        read_partition(path, record)


def test_rejects_out_of_file_empty_range(tmp_path):
    path = tmp_path / "target.omc"
    path.write_bytes(container_bytes())
    with pytest.raises(ValueError, match="extends beyond"):
        read_partition(path, {"offset": path.stat().st_size + 1, "bytes": 0})

"""Bounded reader for the observed IMOD container, without native ABI calls.

The layout follows the inspected blue DDK's UnifiedModel/ModelPartitions
implementation. It is a reference layout, not a claim about every OMC version.
Only the magic, payload length and partition table have known meanings here.
Callers must bind the source hash and validate the decoded graph against their
matching OMG output before interpreting compiled subgraph payloads.
"""
import hashlib
import os
from pathlib import Path
import stat
import struct


_HEADER_BYTES = 256
_UINT32_MAX = (1 << 32) - 1
_LAYOUT = "IMOD_256_TABLE12_REFERENCE_LAYOUT"


def _invalid(reason):
    return ValueError(f"unsupported or malformed OMC container: {reason}")


def _file_bytes(stream):
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise _invalid("source must be a regular file")
    return info.st_size


def read_container(path: Path):
    """Read only the fixed header and bounded table; require model/task parts.

    Table records are three little-endian uint32 words. The inspected reader
    ignores the middle word and derives payload offsets by accumulating sizes.
    Preserve that word without calling it an offset or assigning it semantics.
    """
    with Path(path).open("rb") as stream:
        file_bytes = _file_bytes(stream)
        header = stream.read(_HEADER_BYTES)
        if len(header) != _HEADER_BYTES:
            raise _invalid("truncated 256-byte header")
        if header[:4] != b"IMOD":
            raise _invalid("magic is not IMOD")
        payload_bytes = struct.unpack_from("<I", header, 0x4C)[0]
        if payload_bytes + _HEADER_BYTES != file_bytes:
            raise _invalid("header payload length does not match file size")
        raw_count = stream.read(4)
        if len(raw_count) != 4:
            raise _invalid("truncated partition count")
        count = struct.unpack("<I", raw_count)[0]
        if not 1 <= count <= 35:
            raise _invalid("partition count must be between 1 and 35")
        table_bytes = 4 + 12 * count
        if table_bytes > payload_bytes:
            raise _invalid("partition table exceeds the payload")
        table = stream.read(12 * count)
        if len(table) != 12 * count:
            raise _invalid("truncated partition table")

        consumed, partitions = table_bytes, []
        for kind, table_word, size in struct.iter_unpack("<III", table):
            if size > _UINT32_MAX - consumed:
                raise _invalid("partition sizes overflow the 32-bit payload length")
            end = consumed + size
            if end > payload_bytes:
                raise _invalid("partition extends beyond the payload")
            partitions.append({"type": kind, "table_word": table_word,
                               "offset": _HEADER_BYTES + consumed, "bytes": size})
            consumed = end
        if consumed != payload_bytes:
            raise _invalid("partition sizes do not cover the payload exactly")
        for required in (0, 2):
            if sum(record["type"] == required for record in partitions) != 1:
                raise _invalid(f"CPU payload inspection requires exactly one type {required} partition")
        if _file_bytes(stream) != file_bytes:
            raise _invalid("source changed size during inspection")
    return {"layout": _LAYOUT, "file_bytes": file_bytes,
            "header_sha256": hashlib.sha256(header).hexdigest(), "partitions": partitions}


def read_partition(path: Path, record, max_bytes=64 << 20):
    """Read one already identified range with a strict allocation budget."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("partition byte budget must be a positive integer")
    if not isinstance(record, dict):
        raise ValueError("partition record must be an object")
    offset, size = record.get("offset"), record.get("bytes")
    if type(offset) is not int or offset < 0 or type(size) is not int or size < 0:
        raise ValueError("partition offset and bytes must be nonnegative integers")
    if size > max_bytes:
        raise ValueError(f"partition exceeds the byte budget {max_bytes}")
    with Path(path).open("rb") as stream:
        file_bytes = _file_bytes(stream)
        if offset > file_bytes or size > file_bytes - offset:
            raise ValueError("partition range extends beyond the current file")
        stream.seek(offset)
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("partition read was truncated")
        if _file_bytes(stream) != file_bytes:
            raise ValueError("source changed size during partition read")
    return data

"""Stream the observed external-weight IMOD variant into an embedded container.

The reference DDK writer emits [main=0, weights=1, targets=2], with cumulative
payload offsets in each table entry. GeneralCompiledModel::SetWeights slices
type 1 using each GraphOp's graphop_weight_offset/graphop_weight_size. The
external writer retains those attributes while emitting an empty type 1.

This conversion accepts only that bounded variant. It copies the graph, target
programs and weight bytes unchanged, and updates only the header payload length
and partition table. It invokes no vendor code and proves no device support.
"""
import errno
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile

from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
from qwen35_moe_dflash.toolchain.omc_wire import load_schema


_CHUNK_BYTES = 8 << 20
_MAX_MAIN_BYTES = 64 << 20
_UINT32_MAX = (1 << 32) - 1
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_REFERENCE = {
    "layout": "IMOD_256_TABLE12_REFERENCE_LAYOUT",
    "libomg_sha256": "83957988db0bb9837595b7c095771172951e7f7c7e58eab414e643111d6a465f",
    "libfmk_common_sha256": "5cf71293558640324f890a2af78636cec08764ae6421c494cda6797bec70ba01",
    "writer": "SaveCompiledModelToBuffer types 0/1/2; GetPartitionTable cumulative offsets",
    "loader": "GeneralCompiledModel::SetWeights graphop_weight_offset/graphop_weight_size slices",
    "symbol_ranges": {
        "libomg_SaveCompiledModelToBuffer": "0x1d6502-0x1d6728",
        "libomg_SaveModelFileHeader": "0x1d4b96-0x1d4d45",
        "libomg_CreateCompiledModelBuffer": "0x1d56ff-0x1d5be0",
        "libomg_SetWeights_GraphOp_lambda": "0x1bbe7b-0x1bc05c",
        "libomg_SaveToBuffer_external_lambda": "0x1b8941-0x1b8b4a",
        "libfmk_common_GetPartitionTable": "0x11314-0x1144e",
    },
    "scope": "Static reference write/read paths; selected DDK graph schema and input invariants checked separately.",
}


def _digest(value, label):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"source must be a regular non-symlink file: {path}")
    return info


def _open_source(path, identity):
    stream = os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb")
    if _identity(os.fstat(stream.fileno())) != identity:
        stream.close()
        raise ValueError(f"source changed before read: {path}")
    return stream


def _check_unchanged(path, identity, stream=None):
    if (_identity(_regular(path)) != identity or
            (stream is not None and _identity(os.fstat(stream.fileno())) != identity)):
        raise ValueError(f"source changed during conversion: {path}")


def _copy(stream, size, output=None, output_digest=None):
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, _CHUNK_BYTES))
        if not chunk:
            raise ValueError("source truncated during conversion")
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
            output_digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _attribute(node, key):
    attrs = node.get("attr")
    if not isinstance(attrs, list):
        raise ValueError("GraphOp attributes must be decoded key/value entries")
    values = [entry.get("value") for entry in attrs
              if isinstance(entry, dict) and entry.get("key") == key]
    if len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError(f"missing or ambiguous {node.get('name')}.{key}")
    value = values[0].get("i")
    if type(value) is not int or not 0 <= value <= _UINT32_MAX:
        raise ValueError(f"{node.get('name')}.{key} must be uint32")
    return value


def _weight_plan(model, records, parent):
    graphs = model.get("graph") if isinstance(model, dict) else None
    if not isinstance(graphs, list) or len(graphs) != 1 or not isinstance(graphs[0], dict):
        raise ValueError("embedded streaming requires exactly one decoded main graph")
    nodes = graphs[0].get("op")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("decoded main graph has no operations")
    sidecars = {}
    if not isinstance(records, list) or not records:
        raise ValueError("hash-bound sidecar records are required")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), (str, Path)):
            raise ValueError("invalid sidecar record")
        path = Path(record["path"]).absolute()
        if path.parent.resolve() != parent or path.name in sidecars:
            raise ValueError("sidecars must have unique names beside the source OMC")
        size = record.get("bytes")
        if type(size) is not int or not 0 < size <= _UINT32_MAX:
            raise ValueError("sidecar bytes must be a positive uint32")
        info = _regular(path)
        if info.st_size != size:
            raise ValueError(f"sidecar size differs from its manifest: {path.name}")
        sidecars[path.name] = {"path": path, "bytes": size,
                               "sha256": _digest(record.get("sha256"), "sidecar sha256"),
                               "identity": _identity(info)}
    plan, names = [], set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("main graph operations must be objects")
        if node.get("type") != "GraphOp":
            continue
        name = node.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise ValueError("GraphOp names must be unique safe filename stems")
        names.add(name)
        filename = name + ".weight"
        if filename not in sidecars:
            raise ValueError(f"missing hash-bound sidecar: {filename}")
        row = dict(sidecars[filename])
        row.update(graphop_name=name, offset=_attribute(node, "graphop_weight_offset"))
        if _attribute(node, "graphop_weight_size") != row["bytes"]:
            raise ValueError(f"GraphOp weight size differs from sidecar: {filename}")
        plan.append(row)
    if not plan or len(plan) != len(sidecars):
        raise ValueError("GraphOp ownership must cover every supplied sidecar exactly once")
    plan.sort(key=lambda row: row["offset"])
    cursor = 0
    for row in plan:
        if row["offset"] != cursor:
            raise ValueError("GraphOp weight offsets must form one gapless nonoverlapping partition")
        cursor += row["bytes"]
        if cursor > _UINT32_MAX:
            raise ValueError("combined weights exceed the uint32 partition limit")
    return plan, cursor


def embed_external_weights(source_omc, output_omc, schema_library, *,
                           expected_source_sha256, sidecars, expected_schema_sha256=None,
                           reserve_bytes=64 << 20, progress=None):
    """Create a new, atomically published OMC; never replace or modify inputs.

    ``sidecars`` are manifest-bound {path, bytes, sha256} records. The schema must
    be the selected compilation DDK's libai_fmk_graph_proto_host.so. The caller
    owns binding that selection to its compiler manifest; this function records
    and optionally verifies the library digest. No whole weight is allocated.
    """
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    expected_source_sha256 = _digest(expected_source_sha256, "source sha256")
    source, output = Path(source_omc).absolute(), Path(output_omc).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    source_identity = _identity(_regular(source))
    with _open_source(source, source_identity) as stream:
        source_hash = _copy(stream, source_identity[2])
        _check_unchanged(source, source_identity, stream)
    if source_hash != expected_source_sha256:
        raise ValueError("source OMC hash differs from its manifest")
    container = read_container(source)
    parts = container["partitions"]
    if [row["type"] for row in parts] != [0, 1, 2] or parts[1]["bytes"] != 0:
        raise ValueError("requires external OMC partitions [0, 1, 2] with empty weights partition")
    cursor = 0
    for part in parts:
        if part["table_word"] != cursor:
            raise ValueError("partition table words must be cumulative payload offsets")
        cursor += part["bytes"]
    schema = load_schema(schema_library)
    schema_hash = schema.evidence.get("library_sha256")
    _digest(schema_hash, "schema library sha256")
    if expected_schema_sha256 is not None and schema_hash != _digest(expected_schema_sha256, "expected schema sha256"):
        raise ValueError("schema library hash differs from selected DDK")
    main = read_partition(source, parts[0], max_bytes=_MAX_MAIN_BYTES)
    plan, weight_bytes = _weight_plan(schema.decode(main, "ModelDef"), sidecars, source.parent.resolve())
    payload_bytes = container["file_bytes"] - 256 + weight_bytes
    if payload_bytes > _UINT32_MAX or payload_bytes + 256 > _UINT32_MAX:
        raise ValueError("embedded OMC exceeds the native uint32 file-size limit")
    output_bytes = payload_bytes + 256
    free = shutil.disk_usage(output.parent).free
    if free < output_bytes + reserve_bytes:
        raise OSError(errno.ENOSPC, f"embedded OMC needs {output_bytes + reserve_bytes} bytes including reserve; {free} available")
    with _open_source(source, source_identity) as stream:
        header = bytearray(stream.read(256))
    struct.pack_into("<I", header, 0x4C, payload_bytes)
    sizes = [parts[0]["bytes"], weight_bytes, parts[2]["bytes"]]
    table = bytearray(struct.pack("<I", 3))
    cursor = 0
    for kind, size in enumerate(sizes):
        table.extend(struct.pack("<III", kind, cursor, size))
        cursor += size
    temporary = None
    output_hash, partitions = hashlib.sha256(), []
    try:
        fd, name = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".partial", dir=output.parent)
        temporary = Path(name)
        with os.fdopen(fd, "wb") as target, _open_source(source, source_identity) as original:
            for block in (header, table):
                target.write(block)
                output_hash.update(block)
            for part in parts:
                if part["type"] == 1:
                    combined = hashlib.sha256()
                    for index, row in enumerate(plan):
                        if progress:
                            progress(f"Embedding weights {index + 1}/{len(plan)}: {row['path'].name} ({row['bytes']} bytes)")
                        with _open_source(row["path"], row["identity"]) as weight:
                            digest = hashlib.sha256()
                            remaining = row["bytes"]
                            while remaining:
                                block = weight.read(min(remaining, _CHUNK_BYTES))
                                if not block:
                                    raise ValueError("weight sidecar truncated during conversion")
                                digest.update(block)
                                combined.update(block)
                                target.write(block)
                                output_hash.update(block)
                                remaining -= len(block)
                            _check_unchanged(row["path"], row["identity"], weight)
                        if digest.hexdigest() != row["sha256"]:
                            raise ValueError(f"sidecar hash differs from its manifest: {row['path'].name}")
                    partitions.append({"type": 1, "bytes": weight_bytes, "sha256": combined.hexdigest()})
                else:
                    original.seek(part["offset"])
                    digest = _copy(original, part["bytes"], target, output_hash)
                    partitions.append({"type": part["type"], "bytes": part["bytes"],
                                       "sha256": digest, "preserved": "BYTE_IDENTICAL"})
            _check_unchanged(source, source_identity, original)
            for row in plan:
                _check_unchanged(row["path"], row["identity"])
            if target.tell() != output_bytes:
                raise ValueError("embedded output size differs from declared partitions")
            target.flush()
            os.fsync(target.fileno())
        verified = read_container(temporary)
        if [p["bytes"] for p in verified["partitions"]] != sizes:
            raise ValueError("written container failed partition validation")
        # Atomic publication without overwriting an existing output (including a
        # symlink created while streaming). Source and partial share no inodes.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"schema_version": 1, "artifact_kind": "qwen35-omc-embedded-stream",
            "status": "EMBEDDED", "source": {"path": str(source), "bytes": container["file_bytes"],
                                                  "sha256": source_hash},
            "schema": schema.evidence, "reference_layout": dict(_REFERENCE),
            "weights": [{key: str(value) if key == "path" else value
                         for key, value in row.items() if key != "identity"} for row in plan],
            "partitions": partitions,
            "output": {"path": str(output), "bytes": output_bytes, "sha256": output_hash.hexdigest()},
            "changes": ["header payload length at 0x4c", "partition table weight size and subsequent offsets",
                        "type 1 payload populated from unchanged sidecars"],
            "device_execution": "NOT_RUN", "numerical_validation": "NOT_EVALUATED",
            "scope": "Container conversion only; graph, compiled targets and FP16 weight bytes unchanged. Device support unverified."}

"""Isolated GE LoadFromFile/SaveToFile probe; no graph edits or model run.

Only a bounded AIR is copied. Explicit FileConstant paths must stay relative
inside a new workspace, where their payloads are never supplied. Absolute,
parent-traversing and unknown references are rejected before invoking GE.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat

from .air_schema import load_torchair_schema
from .artifacts import output_path, save_json
from .ge_export import _logged
from .ge_fused.export import _toolchain
from .hashing import sha256_file


MAX_AIR_BYTES = 64 << 20
_MAX_METADATA_BYTES = 64 << 20
_RESERVED = {"input.air", "roundtrip.air", "probe.cpp", "probe", "native.json", "build.log", "roundtrip.log", "files.strace"}
_CPP = r"""#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>
#include "graph/graph.h"
#include "graph/types.h"

int main(int argc, char **argv) {
  if (argc != 4) return 10;
  const rlimit limit{64ULL << 20, 64ULL << 20};
  if (setrlimit(RLIMIT_FSIZE, &limit) != 0) return 11;
  const int fd = open(argv[2], O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  if (fd < 0) return 12;
  close(fd);
  ge::Graph graph("original_air_roundtrip_probe");
  const auto loaded = graph.LoadFromFile(argv[1]);
  std::cout << "LOAD_STATUS=" << static_cast<uint64_t>(loaded) << std::endl;
  if (loaded != ge::GRAPH_SUCCESS) {
    std::ofstream(argv[3]) << "{\"load_status\":" << static_cast<uint64_t>(loaded)
                          << ",\"save_status\":null,\"model_execution\":\"NOT_RUN\"}\n";
    return 2;
  }
  const auto saved = graph.SaveToFile(argv[2]);
  std::cout << "SAVE_STATUS=" << static_cast<uint64_t>(saved) << std::endl;
  std::ofstream(argv[3]) << "{\"load_status\":" << static_cast<uint64_t>(loaded)
                        << ",\"save_status\":" << static_cast<uint64_t>(saved)
                        << ",\"model_execution\":\"NOT_RUN\"}\n";
  return saved == ge::GRAPH_SUCCESS ? 0 : 3;
}
"""


def _stamp(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_air(path):
    path = Path(path).expanduser()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_AIR_BYTES:
        raise ValueError("AIR must be a nonempty regular non-symlink file at most 64 MiB")
    before = _stamp(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_AIR_BYTES + 1)
    if len(raw) != before[2] or _stamp(path) != before:
        raise ValueError("AIR changed during bounded read")
    return raw, {"path": str(path.resolve()), "bytes": len(raw), "stat": list(before),
                 "sha256": hashlib.sha256(raw).hexdigest()}


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        result = [_canonical(item) for item in value]
        if result and all(isinstance(item, dict) and isinstance(item.get("key"), str)
                          and not set(item) - {"key", "value"} for item in result):
            if len({item["key"] for item in result}) != len(result):
                raise ValueError("duplicate protobuf map keys in graph metadata")
            result.sort(key=lambda item: item["key"])
        return result
    return value


def _digest(value):
    raw = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("decoded graph metadata exceeds comparison budget")
    return hashlib.sha256(raw).hexdigest()


def _decode(schema, raw):
    candidates = []
    errors = {}
    for kind in ("ModelDef", "GraphDef"):
        try:
            decoded = schema.decode(raw, kind)
            graphs = decoded.get("graph") if kind == "ModelDef" else [decoded]
            if not isinstance(graphs, list) or len(graphs) != 1 or not isinstance(graphs[0], dict):
                raise ValueError("one root graph is required")
            graph = graphs[0]
            nodes = graph.get("op")
            if not isinstance(nodes, list) or not 1 <= len(nodes) <= 32768:
                raise ValueError("one nonempty bounded root operation list is required")
            names = [node.get("name") for node in nodes]
            if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                raise ValueError("root operation names must be present and unique")
            candidates.append((kind, decoded, graph))
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            errors[kind] = str(error)[:256]
    if len(candidates) != 1:
        raise ValueError("AIR has no unique strict raw ModelDef/GraphDef: " + json.dumps(errors))
    return candidates[0]


def _attributes(node):
    rows = node.get("attr", [])
    if not isinstance(rows, list):
        raise ValueError("invalid root attribute map")
    result = {}
    for row in rows:
        key = row.get("key")
        if not isinstance(key, str) or key in result:
            raise ValueError("duplicate or missing root attribute key")
        result[key] = row.get("value", {})
    return result


def _proof(kind, decoded, graph):
    nodes = graph["op"]
    constants, bindings = [], []
    def file_constants(value):
        if isinstance(value, dict):
            if value.get("type") == "FileConstant":
                yield value
            for item in value.values():
                yield from file_constants(item)
        elif isinstance(value, list):
            for item in value:
                yield from file_constants(item)
    # Nested subgraphs cannot bypass the relative-path isolation gate.
    for node in file_constants(decoded):
        path = _attributes(node).get("file_path", {}).get("s")
        if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
            raise ValueError("FileConstant requires a bounded explicit POSIX file_path")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {".", ""}:
            raise ValueError("FileConstant path cannot be isolated without payload access: " + path)
        if relative.parts[0] in _RESERVED:
            raise ValueError("FileConstant path collides with probe-owned files: " + path)
        bindings.append(path)
        constants.append({"name": node["name"], "file_path": path, "metadata_sha256": _digest(node)})
    outputs = {"graph_outputs": graph.get("output", []),
               "netoutputs": [node for node in nodes if node.get("type") == "NetOutput"]}
    return {"format": kind, "root_name": graph.get("name"), "node_count": len(nodes),
            "node_type_counts": dict(sorted(Counter(node.get("type", "") for node in nodes).items())),
            "node_order_sha256": _digest([node["name"] for node in nodes]),
            "root_nodes_sha256": _digest({node["name"]: node for node in nodes}),
            "outputs_sha256": _digest(outputs), "file_constants_sha256": _digest(constants),
            "file_constants": constants, "decoded_model_sha256": _digest(decoded)}, bindings


def _unavailable_payloads(work, bindings):
    for value in bindings:
        path = work / value
        # Work is new and contains no symlinked directories; never resolve/stat
        # a reference against the original AIR's directory.
        if path.exists() or path.is_symlink():
            raise ValueError("FileConstant payload unexpectedly exists in isolated workspace: " + value)


def probe_original_air_roundtrip(air, output, *, ge_root=None, torchair_pb2=None,
                                cxx="c++", cxx11_abi=0, timeout_seconds=120, trace_files=False):
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    work = out / "work"
    work.mkdir()
    (work / "probe.cpp").write_text(_CPP)
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-air-roundtrip-probe",
              "status": "BLOCKED", "output": str(out), "model_execution": "NOT_RUN",
              "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN",
              "graph_changes": "NONE", "feature_extension": "NOT_ATTEMPTED",
              "feature_extension_gate": "BLOCKED",
              "external_weights": {"probe_reads": "NONE", "probe_copies": "NONE",
                                   "opaque_library_file_access": "NOT_SYSCALL_AUDITED"},
              "scope": "Host GE serialization and schema-bound metadata fidelity only; no complete-target, numerical or device pass"}
    source, source_before, bindings, identity_files = None, None, [], {}
    stage = "source"
    try:
        raw, source = _read_air(air)
        source_before = tuple(source["stat"])
        result["source_air"] = source
        if not torchair_pb2:
            raise ValueError("explicit --torchair-pb2 is required for strict metadata comparison")
        schema = load_torchair_schema(torchair_pb2)
        result["schema"] = schema.evidence
        kind, decoded, graph = _decode(schema, raw)
        proof, bindings = _proof(kind, decoded, graph)
        result["source_metadata"] = proof
        _unavailable_payloads(work, bindings)
        result["external_weights"].update(file_constant_paths=bindings, payloads="NOT_SUPPLIED_IN_ISOLATED_WORKSPACE")
        with (work / "input.air").open("xb") as stream:
            stream.write(raw)
        (work / "input.air").chmod(0o444)
        if not ge_root:
            raise ValueError("declare --ge-root or CANN_ROOT; no other installation is selected automatically")
        stage = "toolchain"
        ge, include, lib, compiler = _toolchain(ge_root, cxx, cxx11_abi, timeout_seconds)
        paths = [include / "graph/graph.h", include / "graph/operator.h", include / "graph/types.h",
                 lib / "libgraph.so", Path(compiler).resolve(), Path(torchair_pb2).resolve(),
                 Path(__file__), Path(__file__).with_name("ge_export.py"),
                 Path(__file__).parent / "ge_fused/export.py", work / "probe.cpp"]
        if (lib / "libgraph_base.so").is_file():
            paths.append(lib / "libgraph_base.so")
        identity_files = {str(path.absolute()): {"bytes": path.stat().st_size, "sha256": sha256_file(path),
                                               "resolved_path": str(path.resolve())} for path in paths}
        result["toolchain"] = {"ge_root": str(ge), "cxx11_abi": cxx11_abi, "identities": identity_files,
                               "scope": "Direct headers/libraries/compiler/probe/helper identities; transitive shared-library closure is not claimed"}
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(lib), env.get("LD_LIBRARY_PATH", ""))))
        env["ASCEND_PROCESS_LOG_PATH"] = str(out / "runtime-log")
        build = [compiler, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
                 "-I", str(include), str(work / "probe.cpp"), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
        if (lib / "libgraph_base.so").is_file():
            build.append("-lgraph_base")
        build.extend(["-o", str(work / "probe")])
        command = [str(work / "probe"), str(work / "input.air"), str(work / "roundtrip.air"), str(work / "native.json")]
        if trace_files:
            tracer = shutil.which("strace")
            if tracer is None:
                raise ValueError("--trace-files requires an available strace executable")
            result["external_weights"]["opaque_library_file_access"] = "TRACE_RETAINED_REQUIRES_REVIEW"
            result["strace"] = {"path": str(Path(tracer).resolve()), "sha256": sha256_file(Path(tracer).resolve())}
            command = [tracer, "-f", "-e", "trace=%file", "-o", str(work / "files.strace"), *command]
        result["commands"] = {"build": build, "roundtrip": command, "cwd": str(work),
                              "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]}
        stage = "build"
        _logged(build, folder=work, env=env, log=work / "build.log", timeout=timeout_seconds)
        if not (work / "probe").is_file() or (work / "probe").is_symlink():
            raise ValueError("compiler did not produce a regular probe")
        result["probe_binary"] = {"bytes": (work / "probe").stat().st_size, "sha256": sha256_file(work / "probe")}
        _unavailable_payloads(work, bindings)
        stage = "load-save"
        _logged(command, folder=work, env=env, log=work / "roundtrip.log", timeout=timeout_seconds)
        stage = "fidelity"
        native = json.loads((work / "native.json").read_text())
        result["native"] = native
        if native.get("load_status") != 0 or native.get("save_status") != 0:
            raise ValueError("GE did not report successful LoadFromFile and SaveToFile")
        roundtrip, record = _read_air(work / "roundtrip.air")
        result["roundtrip_air"] = record
        rt_kind, rt_decoded, rt_graph = _decode(schema, roundtrip)
        rt_proof, rt_bindings = _proof(rt_kind, rt_decoded, rt_graph)
        _unavailable_payloads(work, bindings + rt_bindings)
        result["roundtrip_metadata"] = rt_proof
        compared = ("format", "root_name", "node_count", "node_type_counts", "node_order_sha256",
                    "root_nodes_sha256", "outputs_sha256", "file_constants_sha256", "decoded_model_sha256")
        differences = [key for key in compared if proof[key] != rt_proof[key]]
        result.update(status="ROUNDTRIP_METADATA_PASSED" if not differences else "ROUNDTRIP_METADATA_DIFFERENT",
                      fidelity={"status": "PASSED" if not differences else "FAILED", "differences": differences,
                                "byte_identical": raw == roundtrip,
                                "scope": "All decoded protobuf fields compared; protobuf map-entry ordering alone is normalized"},
                      feature_extension_gate="METADATA_ONLY_PASSED" if not differences else "BLOCKED")
    except Exception as error:
        result.update(stage=stage, error=f"{type(error).__name__}: {error}")
        if stage in {"build", "load-save", "fidelity"}:
            result["status"] = "FAILED"
        native_path = work / "native.json"
        if native_path.is_file() and native_path.stat().st_size <= 4096:
            try:
                result["native"] = json.loads(native_path.read_text())
            except (ValueError, OSError):
                pass
    finally:
        if source is not None:
            try:
                unchanged = (_stamp(Path(source["path"])) == source_before
                             and sha256_file(Path(source["path"])) == source["sha256"])
            except OSError:
                unchanged = False
            result["source_preservation"] = "UNCHANGED_STAT_AND_SHA256" if unchanged else "CHANGED"
            if not unchanged:
                result.update(status="FAILED", error="source AIR changed during probe", feature_extension_gate="BLOCKED")
        changed = []
        for path, record in identity_files.items():
            try:
                same = (Path(path).is_file() and str(Path(path).resolve()) == record["resolved_path"]
                        and sha256_file(Path(path)) == record["sha256"])
            except OSError:
                same = False
            if not same:
                changed.append(path)
        if changed:
            result.update(status="FAILED", error="declared probe/toolchain identity changed", changed_identity_files=changed,
                          feature_extension_gate="BLOCKED")
        if (work / "files.strace").is_file():
            result["file_trace"] = {"path": str(work / "files.strace"), "bytes": (work / "files.strace").stat().st_size,
                                    "sha256": sha256_file(work / "files.strace")}
        result["logs"] = {name: {"bytes": (work / name).stat().st_size, "sha256": sha256_file(work / name)}
                          for name in ("build.log", "roundtrip.log") if (work / name).is_file()}
        save_json(out / "result.json", result)
    return result


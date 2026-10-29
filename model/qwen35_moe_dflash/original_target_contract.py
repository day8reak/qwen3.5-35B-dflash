"""Schema-bound original target boundaries and selected runtime source evidence.

This produces observations for the original RefData adapter. It never creates
an explicit-state manifest, assigns meanings to argN inputs, or runs a model.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
import re

from .artifacts import output_path, save_json
from .draft_ir_inspect import _enum_fields
from .existing_air import _stat
from .full_target import _recorded_file, read_metadata
from .hashing import sha256_file
from .omc_container import read_container, read_partition
from .omc_wire import load_schema


_SOURCE_PATTERN = re.compile(
    r"RefData|ref_data|past_key_values|new_kv_cache_pos|kv_len|output_pos|"
    r"input_args|example_inputs|dynamic_dims|dynamic_shapes|export\.air|"
    r"InitWeights|SetDynamic|SetInput.*Shape|SetInput.*Dim|Create.*Tensor|"
    r"RestoreFromFile|GetInputTensor|GetOutputTensor|\bProcess\(|\bRun\(")
_HEADER_PATTERN = re.compile(
    r"RefData|refData|ref_data|Dynamic|dynamic|InitWeights|Weight.*(?:Path|Buffer|File)|"
    r"RestoreFromFile|GetInputTensor|GetOutputTensor|SetInput|Create.*Tensor|"
    r"\bRun\(|\bProcess\(|class .*Model|struct .*Options|NDTensorDesc")
_HEADERS = ("HiAiModelManagerService.h", "model/built_model.h", "model_manager/model_manager.h",
            "model_manager/model_manager_types.h", "tensor/nd_tensor_desc.h", "tensor/nd_tensor_buffer.h",
            "tensor/buffer.h", "util/base_types.h")


def boundary_report(result):
    lines = ["Original target compiled boundaries: " + result["status"],
             "Runtime ABI, RefData persistence and numerical execution remain unverified."]
    if result.get("error"):
        lines.append(result["error"])
    boundary = result.get("boundaries")
    if boundary:
        lines.extend(["Visible root types: " + str(boundary["input_type_counts"]),
                      "User profile: " + boundary["declared_profile_check"]["status"],
                      "Rows below follow serialized operation order, not verified HIAI runtime slots."])
        groups = {}
        for row in boundary["inputs_in_serialized_op_order"]:
            desc = (str(row["shape"]), str(row["schema_dtype_names"]), str(row["raw_dtype"]), str(row["layout"]))
            if row["type"] == "RefData":
                groups.setdefault(desc, []).append(row["name"])
            else:
                lines.append(f"Data {row['name']} serialized_index={row['serialized_data_index']} shape={desc[0]} dtype={desc[1]} raw_dtype={desc[2]} layout={desc[3]}")
        for desc, names in groups.items():
            lines.append("RefData " + ",".join(names) + f" shape={desc[0]} dtype={desc[1]} raw_dtype={desc[2]} layout={desc[3]}")
        for row in boundary["outputs"]:
            lines.append(f"Output {row['netoutput']}[{row['output_index']}] <- {row['producer_reference']} shape={row['shape']} dtype={row['schema_dtype_names']} raw_dtype={row['raw_dtype']} layout={row['layout']}")
        for key in ("missing_names", "extra_names", "differences"):
            if boundary["declared_profile_check"][key]:
                lines.append(key + ": " + str(boundary["declared_profile_check"][key]))
    return "\n".join(lines) + "\n"


def _attributes(node):
    entries = node.get("attr", [])
    if not isinstance(entries, list):
        raise ValueError("selected schema attributes must be key/value entries")
    result = {}
    for row in entries:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or row["key"] in result:
            raise ValueError("missing or duplicate compiled attribute key")
        result[row["key"]] = row.get("value", {})
    return result


def _descriptor(raw, enums):
    if not isinstance(raw, dict):
        raise ValueError("compiled tensor descriptor must be an object")
    shape = raw.get("shape", {}).get("dim")
    if (not isinstance(shape, list) or len(shape) > 16
            or any(type(d) is not int or d < -1 for d in shape)):
        raise ValueError("compiled input/output shape is missing or unsupported")
    names = [name for name, value in enums.get("dtype", {}).get("values", {}).items()
             if "dtype" in raw and type(raw["dtype"]) is int and value == raw["dtype"]]
    return {"shape": shape, "raw_dtype": raw.get("dtype"), "schema_dtype_names": names,
            "layout": raw.get("layout"), "dtype_status": "DECODED" if names else "UNRESOLVED"}


def summarize_boundaries(model, enums, profile):
    """Describe one root graph only; nested task Data nodes are not host I/O.

    Data indices are serialized graph indices, not proven HIAI runtime slots.
    The user shape list is never used as the actual input order or dtype map.
    """
    graphs = model.get("graph")
    if not isinstance(graphs, list) or len(graphs) != 1:
        raise ValueError("original target requires one unambiguous root graph")
    graph = graphs[0]
    ops = graph.get("op")
    if not isinstance(ops, list) or not 1 <= len(ops) <= 32768:
        raise ValueError("unsupported original target root operation count")
    by_name = {}
    for node in ops:
        name = node.get("name") if isinstance(node, dict) else None
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("missing or duplicate original target root operation name")
        by_name[name] = node
    inputs, outputs = [], []
    for position, node in enumerate(ops):
        kind = node.get("type")
        if kind in {"Data", "RefData"}:
            descs = node.get("output_desc")
            if not isinstance(descs, list) or len(descs) != 1:
                raise ValueError("original root input must have one output descriptor")
            attrs = _attributes(node)
            index = attrs.get("index", {}).get("i")
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("invalid serialized original target input index")
            inputs.append({"name": node["name"], "type": kind, "serialized_op_index": position,
                           "serialized_data_index": index, **_descriptor(descs[0], enums)})
        elif kind == "NetOutput":
            descs, edges = node.get("input_desc"), node.get("input")
            if not isinstance(descs, list) or not isinstance(edges, list):
                raise ValueError("original NetOutput must have ordered descriptors and references")
            values = []
            for edge in edges:
                if not isinstance(edge, str) or ":" not in edge:
                    raise ValueError("unresolved original NetOutput reference")
                name, port = edge.rsplit(":", 1)
                if not re.fullmatch(r"-?(0|[1-9][0-9]*)", port):
                    raise ValueError("malformed original NetOutput reference")
                port = int(port)
                if port == -1:
                    continue
                producer = by_name.get(name)
                if port < 0 or producer is None or port >= len(producer.get("output_desc", [])):
                    raise ValueError("missing original NetOutput producer/descriptor")
                values.append((edge, producer["output_desc"][port]))
            if len(values) != len(descs) or not descs or len(descs) > 512:
                raise ValueError("original NetOutput port count is unsupported or ambiguous")
            for index, ((edge, producer), raw) in enumerate(zip(values, descs)):
                desc = _descriptor(raw, enums)
                if desc != _descriptor(producer, enums):
                    raise ValueError("original NetOutput differs from its actual producer descriptor")
                outputs.append({"netoutput": node["name"], "output_index": index,
                                "producer_reference": edge, **desc})
    if not inputs or not outputs:
        raise ValueError("original target root has no visible input/output boundary")
    expected = {entry.split(":", 1)[0]: [int(d) for d in entry.split(":", 1)[1].split(",")]
                for entry in profile["input_shape"].split(";")}
    actual = {row["name"]: row for row in inputs}
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    differences = []
    for name in expected.keys() & actual.keys():
        shape, observed = expected[name], actual[name]["shape"]
        # A dynamic descriptor or either frozen gear is an observation; actual
        # gear selection support still requires the runtime/API evidence.
        if len(shape) != len(observed) or any(
                (a != b if a != -1 else b not in {-1, 1, 64}) for a, b in zip(shape, observed)):
            differences.append({"name": name, "declared": shape, "observed": observed})
        expected_kind = "RefData" if name.startswith("RefData_") else "Data"
        if actual[name]["type"] != expected_kind:
            differences.append({"name": name, "declared_type": expected_kind, "observed_type": actual[name]["type"]})
    indices = [row["serialized_data_index"] for row in inputs]
    indices_complete = all(type(i) is int for i in indices) and set(indices) == set(range(len(inputs)))
    return {"graph_name": graph.get("name"), "root_op_counts": dict(sorted(Counter(n.get("type") for n in ops).items())),
            "inputs_in_serialized_op_order": inputs, "outputs": outputs,
            "input_type_counts": dict(Counter(row["type"] for row in inputs)),
            "serialized_input_indices": "CONTIGUOUS" if indices_complete else "INCOMPLETE_OR_NONCONTIGUOUS",
            "declared_profile_check": {"status": "MATCHED_VISIBLE_BOUNDARIES" if not (missing or extra or differences) else "DIFFERENT_VISIBLE_BOUNDARIES",
                "missing_names": missing, "extra_names": extra, "differences": sorted(differences, key=lambda r: r["name"])},
            "runtime_input_order": "UNVERIFIED", "dynamic_gear_execution": "NOT_RUN",
            "refdata_persistence": "UNVERIFIED", "input_semantic_mapping": "REQUIRES_ORIGINAL_EXPORT_AND_RUNTIME_SOURCE"}


def inspect_target_contract(compilation_dir, output_dir, profile):
    root, out = Path(compilation_dir).resolve(), output_path(output_dir)
    report = read_metadata(root / "report.json")
    if (report.get("artifact_kind") != "qwen35-existing-air-compilation" or report.get("status") != "COMPILED"
            or report.get("model_compilation") != "PASSED"):
        raise ValueError("original target contract inspection requires a completed compilation")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-target-contract",
              "status": "UNAVAILABLE", "compilation_report_sha256": sha256_file(root / "report.json"),
              "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN", "runtime_abi": "UNVERIFIED",
              "device_ready": False, "scope": "Selected-schema root boundaries only; no task arithmetic, runtime binding, state update or weight-loading validation"}
    try:
        omc = _recorded_file(root, report["omc"], "stat")
        before = _stat(omc)
        result["omc"] = dict(report["omc"], identity_check="SIZE_STAT_AND_RECOMPUTED_SHA256")
        library = Path(report["ddk_root"]) / "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
        schema = load_schema(library)
        result["schema"] = schema.evidence
        container = read_container(omc)
        main = next(p for p in container["partitions"] if p["type"] == 0)
        raw = read_partition(omc, main, max_bytes=64 << 20)
        model = schema.decode(raw, "ModelDef")
        result.update(main_partition_sha256=hashlib.sha256(raw).hexdigest(), container=container,
                      boundaries=summarize_boundaries(model, _enum_fields(schema), profile))
        if (_stat(omc) != before or sha256_file(library) != schema.evidence["library_sha256"]
                or sha256_file(root / "report.json") != result["compilation_report_sha256"]):
            raise ValueError("target OMC, schema or compilation report changed during inspection")
        result["status"] = "OBSERVED"
    except (OSError, ValueError, TypeError, KeyError, StopIteration, RecursionError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    save_json(out / "report.json", result)
    (out / "report.txt").write_text(boundary_report(result))
    return result


def capture_runtime_sources(output_dir, *, include=None, source_root=None):
    """Retain bounded code excerpts locally; never import or transmit sources."""
    out = output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "artifact_kind": "qwen35-original-target-runtime-sources",
              "status": "CAPTURED", "headers": [], "sources": [], "errors": [],
              "scope": "Local current-source excerpts only; source-to-AIR revision agreement and dynamic gear, RefData or external-weight API behavior remain unverified"}
    total = 0
    def capture(path, root, pattern):
        nonlocal total
        if path.is_symlink() or root not in path.resolve().parents or not path.is_file():
            return
        before = _stat(path)
        if before[2] > 1 << 20 or total + before[2] > 16 << 20:
            report["errors"].append("source read budget exceeded: " + str(path))
            return
        raw = path.read_bytes()
        if _stat(path) != before:
            raise ValueError("runtime source changed while reading: " + str(path))
        total += len(raw)
        lines = raw.decode("utf-8", errors="replace").splitlines()
        hits = [i for i, line in enumerate(lines) if pattern.search(line)]
        if not hits:
            return
        selected = sorted({j for i in hits for j in range(max(0, i - 5), min(len(lines), i + 9))})
        return {"path": str(path.relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "matching_lines": len(hits), "truncated": len(selected) > 240,
                "lines": [{"line": i + 1, "text": lines[i][:1600]} for i in selected[:240]]}
    if include:
        include = Path(include).resolve()
        report["hiai_include"] = str(include)
        for name in _HEADERS:
            path = include / name
            if not path.is_file():
                report["errors"].append("selected header missing: " + name)
            else:
                row = capture(path, include, _HEADER_PATTERN)
                if row:
                    report["headers"].append(row)
    else:
        report["errors"].append("HIAI_INCLUDE_DIR/--hiai-include not supplied")
    if source_root:
        source_root = Path(source_root).resolve()
        report["source_root"] = str(source_root)
        entries = 0
        if not source_root.is_dir():
            report["errors"].append("original source root is missing")
        for folder, dirs, files in os.walk(source_root, followlinks=False):
            base = Path(folder)
            entries += len(dirs) + len(files)
            if entries > 4096:
                report["errors"].append("original source entry budget exceeded")
                break
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {
                "__pycache__", "weights", "checkpoints", "build", "output", "outputs", "logs"}
                and not (base / d).is_symlink() and len((base / d).relative_to(source_root).parts) <= 3)
            for name in sorted(files):
                if Path(name).suffix in {".py", ".cpp", ".h", ".cc"} and not name.startswith("."):
                    row = capture(base / name, source_root, _SOURCE_PATTERN)
                    if row:
                        report["sources"].append(row)
            if len(report["sources"]) > 64:
                report["errors"].append("original source match budget exceeded")
                break
    else:
        report["errors"].append("--source-root not supplied; argN semantics remain unavailable")
    if report["errors"] or not report["headers"] or not report["sources"]:
        report["status"] = "INCOMPLETE"
    save_json(out / "report.json", report)
    text = ["Original target runtime sources: " + report["status"], report["scope"]]
    for group in ("headers", "sources"):
        for row in report[group]:
            text.append("\n" + group.upper() + " " + row["path"])
            text.extend(f"L{line['line']}: {line['text']}" for line in row["lines"])
            if row["truncated"]:
                text.append("[excerpt truncated; use the recorded source file for the full function]")
    text.extend(report["errors"])
    (out / "report.txt").write_text("\n".join(text) + "\n")
    return report

"""Read selected-DDK frontend declarations and retained Top1 compiler IR.

No vendor code is imported or executed. Backend opinfo is deliberately not
used as an operator prototype. A compiled CPU node is a comparison example,
not a complete frontend schema or evidence of NPU placement.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import re

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.draft.ir_inspect import (
    _enum_fields, _graphs, _read, _select, inspect_draft_ir,
)
from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
from qwen35_moe_dflash.toolchain.omc_cpu_inspect import _task
from qwen35_moe_dflash.toolchain.omc_wire import load_schema


_TYPES = ("ArgMaxExt2", "ReduceMax", "GatherV2D")
_NAMES = ["argmax_ext2", "local_argmax", "local_value", "winning_tile", "local_index"]
_CASES = ("top1_ext2_fp16", "top1_ext2_flat_fp16", "top1_ext2_tiled_fp16")
_LIBRARY = "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
_LIMITS = dict(entries=20000, source_file_bytes=2 << 20, source_total_bytes=32 << 20,
               excerpts=24, excerpt_lines=90, excerpt_characters=8000, output_bytes=2 << 20)


def _frontend_sources(root):
    """Bounded text search including generic math/array prototype headers."""
    result = dict(status="INSPECTED", excerpts=[], notes=[],
                  coverage=dict(entries=0, files_read=0, bytes=0, symlinks_skipped=0))
    used = result["coverage"]
    # Prioritize headers; do not spend the file budget on TBE Python or binaries.
    seeds = [root / p for p in ("ddk/ai_ddk_lib/include/graph/op", "include", "ddk/include", "tools/tools_omg/master/include",
                                "ddk", "tools/tools_omg/master")]
    result["roots"] = [dict(path=str(p), exists=p.is_dir()) for p in seeds]
    queue, seen = deque(seeds), set()
    ignored = {"__pycache__", ".git", "tbe", "ccec_compiler", "bisheng", "node_modules", "tikcpp", "ascendc"}
    result["excluded_directory_names"] = sorted(ignored)
    token = re.compile(r"\b(?:ArgMaxExt2(?:Verify|Infer)?|ReduceMax(?:Verify|Infer)?|GatherV2D(?:Verify|Infer)?)\b")
    macro = re.compile(r"\b(?:HIAI_)?REG_OP\s*\(\s*(ArgMaxExt2|ReduceMax|GatherV2D)\s*\)")
    end_macro = re.compile(r"\b(?:HIAI_OP_END|OP_END_FACTORY_REG)\s*\(")

    def note(path, message):
        result["status"] = "PARTIAL"
        if len(result["notes"]) < 24:
            result["notes"].append(dict(path=str(path), error=str(message)[:512]))

    while queue and used["entries"] < _LIMITS["entries"]:
        path = queue.popleft()
        try:
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(root):
                used["symlinks_skipped"] += 1
                note(path, "symlink skipped; no recursive external SDK scan")
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if not path.exists():
                continue
            used["entries"] += 1
            if path.is_dir():
                children = []
                for child in path.iterdir():
                    if child.name in ignored:
                        continue
                    if used["entries"] + len(queue) + len(children) >= _LIMITS["entries"]:
                        note(path, "directory entry budget reached")
                        break
                    children.append(child)
                queue.extend(sorted(children, key=lambda p: (p.name != "include", p.name)))
                continue
            if path.suffix.lower() not in {".h", ".hpp", ".inc", ".cc", ".cpp"}:
                continue
            raw = _read(path, min(_LIMITS["source_file_bytes"], _LIMITS["source_total_bytes"] - used["bytes"]))
            used["bytes"] += len(raw)
            used["files_read"] += 1
            source = raw.decode("utf-8", errors="replace")
            if not token.search(source):
                continue
            lines, covered = source.splitlines(), set()
            # Declarations precede examples/comments so a nearby mention does
            # not consume a short context window and hide the actual prototype.
            declarations = [i for i, line in enumerate(lines) if macro.search(line)]
            related = [i for i, line in enumerate(lines) if token.search(line) and not macro.search(line)]
            for i in declarations + related:
                line = lines[i]
                if i in covered or not token.search(line):
                    continue
                if len(result["excerpts"]) >= _LIMITS["excerpts"]:
                    note(path, "excerpt budget reached")
                    return result
                declaration = macro.search(line)
                start, stop = max(0, i - 3), min(len(lines), i + 12)
                if declaration:
                    stop = min(len(lines), i + _LIMITS["excerpt_lines"])
                    for end in range(i, stop):
                        if end_macro.search(lines[end]):
                            stop = end + 1
                            break
                covered.update(range(start, stop))
                excerpt = "\n".join(f"L{n + 1}: {lines[n]}" for n in range(start, stop))
                result["excerpts"].append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                    kind=("HIAI_REG_OP_TEXT" if "HIAI_REG_OP" in line else "REG_OP_TEXT") if declaration else "RELATED_SOURCE_TEXT",
                    first_line=start + 1, last_line=stop,
                    excerpt=excerpt[:_LIMITS["excerpt_characters"]],
                    truncated=len(excerpt) > _LIMITS["excerpt_characters"] or
                        bool(declaration and not end_macro.search("\n".join(lines[i:stop]))),
                    scope="PRESENT_TEXT_NOT_PROOF_THIS_DECLARATION_WAS_LOADED"))
        except (OSError, ValueError) as error:
            note(path, error)
        if used["bytes"] >= _LIMITS["source_total_bytes"]:
            note(root, "source byte budget reached")
            break
    if queue:
        note(root, "scan incomplete at inspection budget")
    if not result["excerpts"]:
        note(root, "no matching frontend text within scanned roots/budgets; support remains unknown")
    return result


def _reference(root, schema):
    """Read checked CPU slices only; never materialize a model's weight part."""
    root = Path(root).resolve()
    manifest = json.loads(_read(root / "manifest.json", 4 << 20))
    packing = json.loads(_read(root / "weight-packing.json", 4 << 20))
    if (manifest.get("artifact_kind") != "qwen35-incremental-draft-pytorch-omc"
            or manifest.get("status") != "PYTORCH_DRAFT_OMC_COMPILED"
            or manifest.get("omc") != "model.omc"):
        raise ValueError("reference requires the retained compiled PyTorch Draft")
    if packing.get("status") != "EMBEDDED" or packing.get("schema") != schema.evidence:
        raise ValueError("reference packing schema differs from selected DDK")
    model = root / "model.omc"
    if model.is_symlink():
        raise ValueError("reference OMC must not be a symlink")
    before = model.stat()
    identity = dict(path=str(model), bytes=before.st_size, sha256=sha256_file(model))
    container = read_container(model)
    main = next(p for p in container["partitions"] if p["type"] == 0)
    tasks = next(p for p in container["partitions"] if p["type"] == 2)
    raw = read_partition(model, main, max_bytes=16 << 20)
    document = schema.decode(raw, "ModelDef")
    enums, spans, cpu = _enum_fields(schema), [], []
    for graph, _ in _graphs(document):
        for node in graph["op"]:
            if node.get("type") != "GraphOp":
                continue
            task = _task(node)
            start, size = task["offset"], task["bytes"]
            if not size or start > tasks["bytes"] or size > tasks["bytes"] - start:
                raise ValueError("task exceeds compiled task partition")
            if any(start < end and left < start + size for left, end in spans):
                raise ValueError("task slices overlap")
            spans.append((start, start + size))
            if task["cl_name"] == "CPUCL":
                cpu.append((node["name"], task))
    if len(cpu) > 16 or sum(t["bytes"] for _, t in cpu) > 32 << 20:
        raise ValueError("reference CPU slices exceed inspection budget")
    result = dict(status="INSPECTED", model=identity, main_partition_sha256=hashlib.sha256(raw).hexdigest(),
                  cpu_subgraphs=[], scope="COMPILED_CPU_EXAMPLE_NOT_FRONTEND_SCHEMA_OR_NPU_SUPPORT")
    for name, task in cpu:
        payload = read_partition(model, dict(offset=tasks["offset"] + task["offset"], bytes=task["bytes"]),
                                 max_bytes=16 << 20)
        graph = schema.decode(payload, "GraphDef")
        names = [n.get("name") for n in graph.get("op", [])]
        if (graph.get("name") != name or not names or len(names) != len(set(names))
                or any(not isinstance(n, str) or not n for n in names)):
            raise ValueError("CPU graph owner/name mismatch or ambiguous operation names")
        chosen = [n for n in graph["op"] if n.get("type") in _TYPES]
        if len(chosen) > 16:
            raise ValueError("too many reference reduction nodes")
        matches = [match for n in chosen for match in _select({"graph": [graph]}, n["name"], enums)]
        result["cpu_subgraphs"].append(dict(owner=name, task=task,
            sha256=hashlib.sha256(payload).hexdigest(), matches=matches))
    after = model.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("reference OMC changed during inspection")
    return result


def inspect_top1_ir(compiled, ddk_root, reference_compiled=None):
    if not ddk_root or not Path(ddk_root).is_dir():
        raise ValueError("provide the selected --ddk-root")
    root, compiled = Path(ddk_root).resolve(), Path(compiled).resolve()
    if not compiled.is_dir():
        raise FileNotFoundError(compiled)
    result = dict(artifact_kind="qwen35-draft-top1-ir-inspection", status="PARTIAL",
        scope="READ_ONLY_FRONTEND_TEXT_FAILED_IR_AND_COMPILED_CPU_EXAMPLES",
        model_compilation="NOT_RUN", device_execution="NOT_RUN", numerical_validation="NOT_RUN",
        ddk_root=str(root), limits=_LIMITS, cases=[],
        limitations=["Backend opinfo does not define the frontend's input ports or attribute defaults.",
            "Source snippets do not prove registration in the running OMG; missing text does not prove lack of support.",
            "Failed dumps describe a rejected graph; inferred shapes may be incomplete or wrong.",
            "CPU nodes are compiled examples, not a complete frontend contract or NPU placement proof.",
            "Only the selected DDK protobuf schema is decoded. Tensor bytes remain hashes, not decoded axis values.",
            "OMC CPU slices use checked reference-layout ranges and owner names; no OMG JSON cross-check."])
    library = root / _LIBRARY
    try:
        schema = load_schema(library)
        result["schema"] = schema.evidence
    except (OSError, ValueError) as error:
        schema = None
        result["schema_error"] = str(error)
    for case in _CASES:
        folder = compiled / case / "omc"
        row = dict(case=case, status="UNINSPECTED")
        result["cases"].append(row)
        try:
            path = folder / "omg-environment.json"
            raw = _read(path, 2 << 20)
            environment = json.loads(raw)
            recorded = environment.get("ddk_root")
            if not isinstance(recorded, str) or not Path(recorded).is_absolute() or Path(recorded).resolve() != root:
                raise ValueError("saved compiler DDK root differs from selected DDK; dump not decoded")
            row["environment"] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), ddk_root=recorded)
            if schema is None:
                row["status"] = "SCHEMA_UNAVAILABLE"
                continue
            row["ir"] = inspect_draft_ir(folder, library, node_names=_NAMES)
            row["status"] = row["ir"]["status"]
        except (OSError, ValueError, TypeError, RecursionError) as error:
            row.update(status="UNSUPPORTED", error=str(error)[:512])
    result["frontend_sources"] = _frontend_sources(root)
    if reference_compiled is not None:
        try:
            if schema is None:
                raise ValueError("selected DDK schema unavailable")
            result["reference"] = _reference(reference_compiled, schema)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            result["reference"] = dict(status="UNSUPPORTED", error=str(error)[:512])
    if (all(row["status"] == "INSPECTED" for row in result["cases"])
            and result["frontend_sources"]["status"] == "INSPECTED"
            and result.get("reference", {}).get("status", "INSPECTED") == "INSPECTED"):
        result["status"] = "INSPECTED"
    result["next"] = "RECONCILE_FRONTEND_PORTS_AXIS_AND_REDUCTION_SHAPES_BEFORE_CHANGING_OR_RECOMPILING_PROBES"
    if len(json.dumps(result, ensure_ascii=True).encode()) > _LIMITS["output_bytes"]:
        raise ValueError("Top1 IR inspection exceeds output budget")
    return result

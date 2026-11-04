"""Read selected-DDK frontend declarations and retained Top1 compiler IR.

No vendor code is imported or executed. Backend opinfo is deliberately not
used as an operator prototype. A compiled CPU node is a comparison example,
not a complete frontend schema or evidence of NPU placement.
"""
from collections import deque
from collections import Counter
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
_CASES = ("top1_ext2_fp16", "top1_ext2_flat_fp16", "top1_ext2_tiled_fp16",
          "top1_ext2_tiled_flat_fp16", "top1_ext2_tiled_transposed_fp16", "top1_ext2_tiled_485_fp16")
_LIBRARY = "tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so"
_LIMITS = dict(entries=20000, source_file_bytes=2 << 20, source_total_bytes=32 << 20,
               excerpts=24, excerpt_lines=90, excerpt_characters=8000, output_bytes=2 << 20)


def _frontend_sources(root, *, types=_TYPES, declarations_only=False):
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
    names = '|'.join(re.escape(name) for name in types)
    token = re.compile(r"\b(?:" + names + r")(?:Verify|Infer)?\b")
    macro = re.compile(r"\b(?:HIAI_)?REG_OP\s*\(\s*(" + names + r")\s*\)")
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
            related = ([] if declarations_only else
                       [i for i, line in enumerate(lines) if token.search(line) and not macro.search(line)])
            for i in declarations + related:
                line = lines[i]
                if (i in covered and not macro.search(line)) or not token.search(line):
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
    return _compiled_model(root / "model.omc", schema)


def _node_summary(node, enums):
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    keys = {"axis", "axes", "dimension", "output_type", "keep_dims", "outmaxval", "topk", "dst_type",
            "k", "dim", "sorted", "largest", "type", "keep_dim"}
    return dict(name=node.get("name"), type=node.get("type"), input=node.get("input", []),
        inputs=[_observed_desc(d, enums) for d in node.get("input_desc", [])],
        outputs=[_observed_desc(d, enums) for d in node.get("output_desc", [])],
        attributes=[a for a in node.get("attr", []) if a.get("key") in keys])


def _compiled_model(model, schema, *, compact=False):
    """Decode bounded CPU task slices; NPU task payloads remain opaque."""
    if model.is_symlink():
        raise ValueError("OMC must not be a symlink")
    before = model.stat()
    identity = dict(path=str(model), bytes=before.st_size, sha256=sha256_file(model))
    container = read_container(model)
    main = next(p for p in container["partitions"] if p["type"] == 0)
    tasks = next(p for p in container["partitions"] if p["type"] == 2)
    raw = read_partition(model, main, max_bytes=16 << 20)
    document = schema.decode(raw, "ModelDef")
    enums, spans, cpu = _enum_fields(schema), [], []
    main_nodes, counts = [], Counter()
    for graph, _ in _graphs(document):
        if compact:
            names = [n.get("name") for n in graph["op"]]
            if (len(names) > 1024 or len(names) != len(set(names))
                    or any(not isinstance(n, str) or not n for n in names)):
                raise ValueError("main graph operation names/count unsupported")
        for node in graph["op"]:
            if compact:
                main_nodes.append(dict(graph=graph.get("name"), **_node_summary(node, enums)))
            if node.get("type") != "GraphOp":
                continue
            task = _task(node)
            counts[task["cl_name"]] += 1
            if compact:
                main_nodes[-1]["owner"] = task["cl_name"]
            start, size = task["offset"], task["bytes"]
            if not size or start > tasks["bytes"] or size > tasks["bytes"] - start:
                raise ValueError("task exceeds compiled task partition")
            if any(start < end and left < start + size for left, end in spans):
                raise ValueError("task slices overlap")
            spans.append((start, start + size))
            if task["cl_name"] == "CPUCL":
                cpu.append((node["name"], task))
    if len(cpu) > 64 or sum(t["bytes"] for _, t in cpu) > 32 << 20:
        raise ValueError("reference CPU slices exceed inspection budget")
    result = dict(status="INSPECTED", model=identity, main_partition_sha256=hashlib.sha256(raw).hexdigest(),
                  cpu_subgraphs=[], scope="COMPILED_CPU_EXAMPLE_NOT_FRONTEND_SCHEMA_OR_NPU_SUPPORT")
    if compact:
        if not counts:
            raise ValueError("no GraphOp owners found")
        result.update(main_nodes=main_nodes, graphop_counts=dict(counts),
            scope="COMPILED_MAIN_BOUNDARIES_AND_CPU_OPERATIONS_ONLY; NPU_TASK_INTERNALS_NOT_DECODED")
    for name, task in cpu:
        payload = read_partition(model, dict(offset=tasks["offset"] + task["offset"], bytes=task["bytes"]),
                                 max_bytes=16 << 20)
        graph = schema.decode(payload, "GraphDef")
        names = [n.get("name") for n in graph.get("op", [])]
        if (graph.get("name") != name or not names or len(names) != len(set(names))
                or len(names) > 1024 or any(not isinstance(n, str) or not n for n in names)):
            raise ValueError("CPU graph owner/name mismatch or ambiguous operation names")
        if compact:
            result["cpu_subgraphs"].append(dict(owner=name, task=task,
                sha256=hashlib.sha256(payload).hexdigest(),
                operations=[_node_summary(n, enums) for n in graph["op"]]))
            continue
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


def inspect_top1_placement(compiled, ddk_root):
    """Inspect already compiled Top1 cases without compiling or writing files."""
    from .top1_probe import CASES
    if not ddk_root or not Path(ddk_root).is_dir():
        raise ValueError("provide the selected --ddk-root")
    root, sdk = Path(compiled).resolve(), Path(ddk_root).resolve()
    batch_raw = _read(root / "result.json", 4 << 20)
    batch = json.loads(batch_raw)
    if (batch.get("artifact_kind") != "qwen35-draft-vector-probe"
            or batch.get("status") not in {"COMPILED", "PARTIAL"}):
        raise ValueError("expected completed probe compilation")
    rows = batch.get("cases", [])
    names = [r["case"] for r in rows]
    if not names or len(names) != len(set(names)) or any(n not in CASES for n in names):
        raise ValueError("expected distinct Top1 cases")
    schema = load_schema(sdk / _LIBRARY)
    result = dict(artifact_kind="qwen35-draft-top1-placement-inspection", status="INSPECTED",
        model_compilation="NOT_RUN", device_execution="NOT_RUN", numerical_validation="NOT_RUN",
        source_batch_status=batch["status"], source_batch_sha256=hashlib.sha256(batch_raw).hexdigest(),
        schema=schema.evidence, cases=[],
        scope="READ_ONLY_COMPILED_PROBE_MAIN_BOUNDARIES_AND_CPU_TASKS",
        limitations=["NPU_GRAPH_PRESENT does not place any particular reduction on NPU.",
            "Only CPU task GraphDefs are decoded; absent CPU names are not NPU operator mapping proof.",
            "Tensor descriptors and explicit input references are reported, not measured transfers or internal arithmetic.",
            "Empty/missing shape dimensions remain as decoded; no expected shape is filled in.",
            "Reference OMC layout with selected schema, owner and range checks; no fresh OMG JSON conversion."])
    for row in rows:
        entry = dict(case=row["case"], status="UNINSPECTED")
        result["cases"].append(entry)
        try:
            if row.get("status") != "COMPILED" or row.get("model_compilation") != "PASSED":
                raise ValueError("case did not compile; no placement decoded")
            folder = root / row["case"] / "omc"
            report_raw = _read(folder / "report.json", 16 << 20)
            report = json.loads(report_raw)
            environment = json.loads(_read(folder / "omg-environment.json", 2 << 20))
            if (report.get("status") != "COMPILED" or report.get("platform") != "kirinx90"
                    or any(Path(d.get("ddk_root") or "").resolve() != sdk for d in (report, environment))):
                raise ValueError("saved compiler status/platform/DDK differs from selected inspection")
            identity = report["omc"]
            relative = Path(identity["path"])
            model = folder / relative
            if (relative.is_absolute() or ".." in relative.parts or model.is_symlink()
                    or not model.resolve().is_relative_to(folder.resolve())
                    or model.resolve() != (root / row["omc"]).resolve()):
                raise ValueError("probe model path differs from saved compilation")
            if model.stat().st_size > 64 << 20:
                raise ValueError("Top1 probe exceeds 64 MiB")
            if (model.stat().st_size != identity["bytes"] or sha256_file(model) != identity["sha256"]):
                raise ValueError("probe OMC differs from compiler inventory")
            inspection = _compiled_model(model, schema, compact=True)
            if (inspection["model"]["sha256"] != identity["sha256"]
                    or inspection["graphop_counts"] != row.get("graphop_counts")):
                raise ValueError("decoded probe identity/owners differ from saved result")
            entry.update(status="INSPECTED", compiler_report_sha256=hashlib.sha256(report_raw).hexdigest(),
                         **{k: v for k, v in inspection.items() if k != "status"})
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
            entry.update(status="UNSUPPORTED", error=str(error)[:512])
            result["status"] = "PARTIAL"
    result["next"] = "LOCATE_CPU_REDUCTIONS_AND_BOUNDARY_SHAPES; DEVICE_NUMERICS_AND_LATENCY_STILL_UNVERIFIED"
    if len(json.dumps(result, ensure_ascii=True).encode()) > _LIMITS["output_bytes"]:
        raise ValueError("Top1 placement inspection exceeds output budget")
    return result


def placement_summary(report):
    """Compact per-case observations, without inferring opaque NPU operators."""
    cases = []
    for case in report["cases"]:
        row = {key: case[key] for key in ("case", "status", "error", "graphop_counts") if key in case}
        if case["status"] == "INSPECTED":
            row["boundaries"] = [dict(name=n["name"], owner=n["owner"],
                input=n["input"], inputs=n["inputs"], outputs=n["outputs"])
                for n in case["main_nodes"] if n["type"] == "GraphOp"]
            row["cpu_operations"] = [dict(owner=g["owner"], **n)
                for g in case["cpu_subgraphs"] for n in g["operations"]
                if n["type"] not in {"Const", "Data", "NetOutput"}]
            row["main_operations_outside_tasks"] = [n for n in case["main_nodes"]
                if n["type"] not in {"Const", "Data", "GraphOp", "NetOutput"}]
            owners = case["graphop_counts"]
            row["npu_only_candidate"] = bool(owners.get("NPUCL") and set(owners) == {"NPUCL"}
                and not row["cpu_operations"] and not row["main_operations_outside_tasks"])
            row["placement_gate"] = ("NPU_TASKS_ONLY_AT_DECODED_GRAPH_LEVEL"
                if row["npu_only_candidate"] else "CPU_OTHER_OWNER_OR_UNASSIGNED_MAIN_COMPUTE")
        cases.append(row)
    from .top1_alternatives import FULL_CASES
    from .top1_bitcarrier import FULL_CASES as BITCARRIER_FULL_CASES
    return dict(status=report["status"], cases=cases,
                npu_only_cases=[r["case"] for r in cases if r.get("npu_only_candidate")],
                npu_only_full_top1_candidates=[r["case"] for r in cases
                    if r.get("npu_only_candidate") and r["case"] in (*FULL_CASES, *BITCARRIER_FULL_CASES)],
                scope="COMPILED_CPU_OPERATIONS_AND_BOUNDARIES_NOT_NPU_KERNEL_INTERNALS_OR_PERFORMANCE")


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

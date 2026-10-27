"""Explicit single-custom-op GE cases for the documented fused bindings.

Construction, AIR serialization, OMG compilation and device execution are
separate evidence. NumPy references describe FP32 mathematics, not kernel
rounding. No kernel is installed and no primitive fallback graph is generated.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from ..artifacts import files_under, output_path, save_json
from ..compile import (_environment_record, _omg_ipc_environment, _stateful_json,
                       omg_environment)
from ..compiler_evidence import compile_evidence
from ..ge_export import _logged
from ..hashing import sha256_file
from .bindings import OperatorBindings, document_probe_spec
from .emitter import emit_cpp
from .export import _step_file, _structure_graph, _toolchain
from .ir import DTYPES, CustomNode, Graph
from .recipe import read_recipe


_LEGACY_CASES = (("gdn_prefill", "gdn", 8), ("gdn_decode", "gdn", 1),
          ("conv_prefill", "conv_state", 8), ("conv_decode", "conv_state", 1),
          ("rms", "rms", 8))
_CASES = _LEGACY_CASES + (("cache_prefill", "cache", 8), ("cache_decode", "cache", 1),
                         ("attention_prefill", "attention", 8), ("attention_decode", "attention", 1))
_TYPES = {"gdn": "ChunkGatedDeltaRule", "conv_state": "LinearAttentionConvstateUpdate", "rms": "AdnRmsNorm",
          "cache": "CacheUpdate", "attention": "AdnFusedInferAttention"}


def _select_cases(cases):
    if cases is None:
        return list(_LEGACY_CASES)
    if cases == "all":
        return list(_CASES)
    if isinstance(cases, str):
        cases = cases.split(",")
    if (not isinstance(cases, (list, tuple)) or not cases
            or any(not isinstance(name, str) for name in cases)):
        raise ValueError("cases must be a nonempty unique subset of known probe names")
    known = {name for name, _, _ in _CASES}
    if len(set(cases)) != len(cases) or not set(cases) <= known:
        raise ValueError("cases must be a nonempty unique subset of known probe names")
    return [case for case in _CASES if case[0] in cases]


def _validate_probe_manifest(manifest):
    if (manifest.get("artifact_kind") != "qwen35-fused-op-probes"
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1):
        raise ValueError("expected a fused operator probe manifest")
    selected = manifest.get("selected_cases", [item[0] for item in _LEGACY_CASES])
    if not isinstance(selected, list) or selected != [item[0] for item in _select_cases(selected)]:
        raise ValueError("probe selected_cases must be a canonical nonempty subset")
    rows = manifest.get("cases")
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or [row.get("name") for row in rows] != selected):
        raise ValueError("probe cases must exactly match selected_cases (legacy manifests require all five)")
    for row, (name, role, sequence) in zip(rows, _select_cases(selected)):
        if (row.get("role") != role or type(row.get("sequence_length")) is not int or row["sequence_length"] != sequence
                or row.get("operator_type") != _TYPES[role]):
            raise ValueError(f"probe case contract disagrees with known case {name}")


def inspect_fused_probes(probe_run, ddk_root, output_dir):
    """Read existing failures and installed interface text; never import or run it.

    This reports evidence, not an automatically repaired binding. Compiler
    intermediates are available only when retained by the generating revision.
    """
    root, ddk = Path(probe_run).resolve(), Path(ddk_root).resolve()
    manifest = _stateful_json(root / "manifest.json")
    _validate_probe_manifest(manifest)
    if not ddk.is_dir():
        raise ValueError("ddk_root must be the existing selected DDK directory")
    output = output_path(output_dir)
    if output == root or root in output.parents or output == ddk or ddk in output.parents:
        raise ValueError("inspection output must be separate from the existing probe run and DDK")
    output.mkdir(parents=True, exist_ok=False)
    # Ignore noisy framework E/ logs and ImportWarning; retain exception
    # context and the exact interface/compiler failure, even in a long log.
    fault = re.compile(r"Traceback|\b(?:ModuleNotFoundError|ImportError|RuntimeError|TypeError|ValueError|"
                       r"AttributeError|OSError|FileNotFoundError|KeyError|SyntaxError)\b|"
                       r"fatal error:|LLVM ERROR|Segmentation fault|core dumped|undefined symbol|"
                       r"compiler process died|Input .*must be|infershape func failed|"
                       r"GE recipe failed|port order mismatch|Get format configs failed|"
                       r"find op format config failed|Not supported OP|"
                       r"No such file or directory|\bKilled\b|\bAborted\b", re.I)
    crash = re.compile(r"Traceback|\b[A-Za-z]+Error:|fatal error:|LLVM ERROR|Segmentation fault|"
                       r"core dumped|undefined symbol|\bKilled\b|\bAborted\b", re.I)
    context = re.compile(r"outputs1:|soc_version:|File\[.*\.py\]|GetKernelbinAddr.*not valid")
    source_hint = re.compile(r"^\s*(?:def |class |@)|register|initial_state|last_recurrent_state|"
                             r"effective_length|llmConfig|llm_config|conv_state|rstd|"
                             r"get_soc|dtype|shape|chunk_size|output_final_state", re.I)
    reports, platform_paths = [], set()

    def read_text(path, base, relative, pattern, limit):
        record = {"source": str(path), "status": "UNREADABLE"}
        try:
            resolved = path.resolve()
            if base not in resolved.parents or not resolved.is_file():
                raise ValueError("file missing or resolves outside the selected source root")
            size = resolved.stat().st_size
            if size > 32 * 1024 * 1024:
                record.update(status="SIZE_LIMIT", bytes=size)
                return record
            payload = resolved.read_bytes()
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            lines = payload.decode("utf-8", errors="replace").splitlines()
            hits = [i for i, line in enumerate(lines) if pattern.search(line) and "ImportWarning" not in line]
            selected = (hits[:limit // 2] + hits[-limit // 2:]) if len(hits) > limit else hits
            if pattern is fault:
                primary = [i for i in hits if crash.search(lines[i])]
                selected = sorted(set(primary[:limit] + selected[:2] + selected[-2:]))
            indices = sorted({j for i in selected for j in range(max(0, i - 2), min(len(lines), i + 4))})
            if pattern is fault:
                # Individual regex hits omit caller frames: retain whole bounded tracebacks.
                for start in [i for i, line in enumerate(lines) if "Traceback (most recent call last):" in line][:4]:
                    end = min(len(lines), start + 60)
                    for j in range(start + 1, end):
                        if re.match(r"^[\w.]+(?:Error|Exception):", lines[j]):
                            end = j + 1
                            break
                    indices = sorted(set(indices).union(range(start, end)))
                for match in re.finditer(r'File "(/[^"\n]+/tbe/common/platform/platform_info\.py)", line \d+', "\n".join(lines)):
                    platform_paths.add(Path(match.group(1)))
            record.update(status="READ", **_record(destination, output), total_lines=len(lines),
                          matched_lines=len(hits), omitted_matches=max(0, len(hits) - len(selected)),
                          excerpts=[{"line": i + 1, "text": lines[i]} for i in indices])
        except (OSError, ValueError) as error:
            record["error"] = str(error)
        return record

    for case in manifest.get("cases", []):
        if case.get("name") not in {entry[0] for entry in _CASES}:
            raise ValueError("unexpected probe case name")
        name = case["name"]
        folder = root / "cases" / name
        logs = [folder / "compile.log"]
        if (folder / "export.log").is_file():
            logs.append(folder / "export.log")
        work = folder / "compile-work"
        if work.is_dir() and not work.is_symlink():
            logs += sorted(p for p in work.rglob("*") if p.suffix in {".log", ".err", ".txt"})[:12]
        item = {"name": name, "construction": case.get("construction"), "export": case.get("export"),
                "compile": case.get("compile"), "compile_work_retained": work.is_dir(), "logs": []}
        env_path = folder / "omg-environment.json"
        item["compiler_environment"] = {"source": str(env_path), "status": "UNREADABLE"}
        try:
            if root not in env_path.resolve().parents or env_path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("environment record outside run or exceeds size limit")
            saved_env = _stateful_json(env_path)
            keys = ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "DDK_PATH", "TOOLCHAIN_HOME", "HIAI_VERSION",
                    "CCE_LIB_DIR", "SOC_VERSION", "KIRIN_CONF_JSON_PATH", "ASCEND_OPP_PATH", "PYTHONHOME")
            safe_env = {key: value for key, value in saved_env.get("environment", {}).items()
                        if key in keys and isinstance(value, str)}
            dest = output / "logs" / name / "omg-environment.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_json(dest, {"environment": safe_env})
            item["compiler_environment"].update(status="READ", source_sha256=sha256_file(env_path),
                                                environment=safe_env, **_record(dest, output))
        except (OSError, ValueError, TypeError, AttributeError) as error:
            item["compiler_environment"]["error"] = str(error)
        for i, path in enumerate(logs):
            rec = read_text(path, root, f"logs/{name}/{i:02d}-{path.name}", fault, 8)
            if rec["status"] == "READ" and i == 0:
                # Descriptor/SoC evidence is separate from the error excerpts.
                saved = (output / rec["path"]).read_text(errors="replace").splitlines()
                rec["descriptor_context"] = [{"line": j + 1, "text": line} for j, line in enumerate(saved)
                                             if context.search(line)][:12]
            item["logs"].append(rec)
        reports.append(item)
    sources = []
    ops = ddk / "ddk/ascendc/ops"
    names = ("chunkgateddeltarule", "linearattentionconvstateupdate", "adnrmsnorm",
             "cacheupdate", "adnfusedinferattention")
    if ops.is_dir():
        for path in sorted(ops.rglob("*")):
            name = re.sub(r"[^a-z0-9]", "", path.stem.lower())
            if (path.suffix.lower() in {".py", ".h", ".hpp", ".cpp", ".json", ".ini"}
                    and any(key in name for key in names)):
                relative = path.relative_to(ddk)
                sources.append(read_text(path, ddk, "installed/" + relative.as_posix(), source_hint, 8))
                if path.suffix.lower() in {".cpp", ".h", ".hpp"} and sources[-1]["status"] == "READ":
                    lines = (output / sources[-1]["path"]).read_text(errors="replace").splitlines()
                    state = re.compile(r"initial_state|last_recurrent_state|\byz\b")
                    calls = [i for i, line in enumerate(lines) if "SetGlobalBuffer" in line
                             and state.search("\n".join(lines[i:i + 3]))][:8]
                    receivers = {match.group(1) for i in calls
                                 for match in re.finditer(r"(\w+)\s*\.SetGlobalBuffer", lines[i])}
                    declarations = [i for i, line in enumerate(lines) if "GlobalTensor" in line]
                    declarations.sort(key=lambda i: not any(re.search(r"\b" + name + r"\b", lines[i]) for name in receivers))
                    hits = calls + declarations[:12 - len(calls)]
                    indices = sorted({j for i in hits for j in range(max(0, i - 1), min(len(lines), i + 4))})
                    sources[-1]["state_storage"] = [{"line": i + 1, "text": lines[i]} for i in indices]
                if len(sources) == 24:
                    break
    # Format/implementation registrations often live in shared configuration
    # files, not in a filename containing the op name. Read bounded text only.
    registration = {"roots": [], "files_read": 0, "bytes_read": 0, "limited": False,
                    "records": [], "scope": "Text matches only; absence does not prove missing registration or kernel."}
    registration_pattern = re.compile(r"LinearAttentionConvstateUpdate|linear_attention_convstate_update|"
                                      r"AdnRmsNorm|adn_rms_norm|\bRmsNorm\b|\brms_norm\b")
    for relative_root in ("ddk/ascendc/ops", "ddk/tbe/ops", "ddk/ascendc/conf", "ddk/ascendcpp/conf",
                          "tools/tools_omg/master/config", "tools/tools_omg/master/conf"):
        search_root = ddk / relative_root
        registration["roots"].append({"path": str(search_root), "exists": search_root.is_dir()})
        if not search_root.is_dir() or search_root.is_symlink():
            continue
        for path in search_root.rglob("*"):
            if path.suffix.lower() not in {".json", ".ini", ".cfg", ".conf", ".xml"}:
                continue
            if registration["files_read"] >= 2048 or len(registration["records"]) >= 12:
                registration["limited"] = True
                break
            try:
                resolved = path.resolve()
                if ddk not in resolved.parents or not resolved.is_file():
                    continue
                size = resolved.stat().st_size
                if size > 8 << 20 or registration["bytes_read"] + size > 64 << 20:
                    registration["limited"] = True
                    continue
                registration["files_read"] += 1
                payload = resolved.read_bytes()
                registration["bytes_read"] += len(payload)
                if not registration_pattern.search(payload.decode("utf-8", errors="replace")):
                    continue
                relative = path.relative_to(ddk)
                registration["records"].append(read_text(
                    path, ddk, "registration/" + relative.as_posix(), registration_pattern, 6))
            except OSError:
                registration["limited"] = True
    platform_sources = []
    for index, path in enumerate(sorted(platform_paths)[:4]):
        # Only this exact traceback-named module is eligible outside the run/DDK.
        if not path.is_absolute() or path.parts[-4:] != ("tbe", "common", "platform", "platform_info.py") or path.is_symlink():
            continue
        record = read_text(path, path.parent.resolve(), f"installed/platform-{index}/platform_info.py", re.compile(r"a^"), 1)
        if record["status"] == "READ":
            text = (output / record["path"]).read_text(errors="replace")
            try:
                tree = ast.parse(text)
                nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                         or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name in {"te_update_version", "te_set_version"}]
                indices = sorted({i for node in nodes for i in range(node.lineno - 1, node.end_lineno)})
                lines = text.splitlines()
                record["excerpts"] = [{"line": i + 1, "text": lines[i]} for i in indices]
            except SyntaxError as error:
                record["parse_error"] = str(error)
        platform_sources.append(record)
    directories = []
    for relative in ("tools/platform/kirinx90", "ddk/ascendcpp/conf", "ddk/ascendc/conf",
                     "ddk/tbe/lib64", "tools/tools_ascendc/package/python"):
        path = ddk / relative
        item = {"path": str(path), "exists": path.is_dir()}
        try:
            item["entry_sample"] = sorted(p.name for p in path.iterdir())[:12] if path.is_dir() else []
        except OSError as error:
            item["error"] = str(error)
        directories.append(item)
    libraries = []
    for name in ("libai_npucore_ascendc_kernel.so", "libai_npucore_ascendcpp.so"):
        path = ddk / "tools/tools_omg/master/lib64" / name
        libraries.append({"path": str(path), "exists": path.is_file()})
    result = {"schema_version": 1, "artifact_kind": "qwen35-fused-probe-inspection", "status": "INSPECTED",
              "probe_manifest_sha256": sha256_file(root / "manifest.json"), "ddk_root": str(ddk),
              "declared_ddk_id": manifest.get("ddk_id"), "cases": reports, "installed_sources": sources,
              "installed_registration": registration,
              "library_paths": libraries, "platform_sources": platform_sources, "ddk_directories": directories,
              "new_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
              "scope": "Read-only excerpts and copies; no DDK code imported, no bindings changed, no installed-kernel claim.",
              "limits": "At most 24 interface files, 4 traceback-named platform modules, 14 logs/case, 4 tracebacks/log (60 lines each); 32 MiB/file. Registration scan: 2048 files, 8 MiB/file, 64 MiB payload; at most 12 matches re-read for copying. Absence is not proof of no registration. Old compiler work may be removed."}
    save_json(output / "report.json", result)
    lines = ["Existing fused probe failure inspection (no compilation or device execution)",
             f"DDK identity (declared): {manifest.get('ddk_id')}", ""]
    for case in reports:
        lines.append(f"{case['name']}: export={case['export']} compile={case['compile']} retained_work={case['compile_work_retained']}")
        env = case["compiler_environment"]
        lines.append(f"  Compiler environment [{env['status']}]: " + json.dumps(env.get("environment", {})))
        for record in case["logs"]:
            lines.append(f"  {record['source']} [{record['status']}]")
            for entry in record.get("excerpts", []) + record.get("descriptor_context", []):
                lines.append(f"  L{entry['line']}: {entry['text']}")
            if record.get("omitted_matches"):
                lines.append(f"  Further matches: {record['omitted_matches']}; full copied text: {record['path']}")
    lines.append("\nInstalled interface text (read, never imported):")
    for record in sources:
        lines.append(f"{record['source']} [{record['status']}]")
        for entry in record.get("excerpts", []) + record.get("state_storage", []):
            lines.append(f"  L{entry['line']}: {entry['text']}")
    if not sources:
        lines.append("No matching interface source filename found under selected DDK/ddk/ascendc/ops; registration is unresolved.")
    lines.append("\nSelected DDK registration text (read only):")
    lines.append(json.dumps({key: value for key, value in registration.items() if key != "records"}))
    for record in registration["records"]:
        lines.append(f"{record['source']} [{record['status']}]")
        lines.extend(f"  L{entry['line']}: {entry['text']}" for entry in record.get("excerpts", []))
    lines.append("\nTraceback-named TBE platform source (read/AST only; never imported):")
    for record in platform_sources:
        lines.append(f"{record['source']} [{record['status']}]")
        for entry in record.get("excerpts", []):
            lines.append(f"  L{entry['line']}: {entry['text']}")
    lines.append("\nSelected DDK directory samples (non-recursive):")
    lines.extend(json.dumps(item) for item in directories)
    lines += ["", "Library path existence is diagnostic, not proof of the compile failure cause."]
    lines.extend(json.dumps(item) for item in libraries)
    lines.extend(["", result["limits"], "Full evidence: report.json, logs/, installed/."])
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    return result


def _record(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _pattern(shape, scale, phase=0, dtype="float16"):
    values = ((np.arange(math.prod(shape), dtype=np.int32) + phase) % 127 - 63).astype(np.float32)
    values[values == 0] = .5
    return (values * np.float32(scale)).reshape(shape).astype(DTYPES[dtype])


def _case_values(role, sequence, bindings, conv_history_width=3):
    """Use nonzero signed inputs and effective gamma; no checkpoint needed."""
    if role == "gdn":
        state_dtype = bindings.tensor_dtype(role, "inputs", "initial_state")
        if state_dtype not in {"float16", "float32"}:
            raise ValueError("GDN probe initial_state requires declared float16 or float32")
        shape = (1, sequence, 32, 128)
        inputs = {"query": _pattern(shape, .01), "key": _pattern(shape, .01, 37),
                  "value": _pattern(shape, .005, 71),
                  "g": -(.005 + np.arange(sequence * 32, dtype=np.float32) % 19 * .003).reshape(1, sequence, 32).astype("<f2"),
                  "beta": (.2 + np.arange(sequence * 32, dtype=np.float32) % 31 / 64).reshape(1, sequence, 32).astype("<f2"),
                  "initial_state": _pattern((1, 32, 128, 128), .0001, 19, state_dtype),
                  "effective_length": np.asarray([6 if sequence == 8 else 1], dtype="<i2")}
        q, k, v, decay, beta = [inputs[key].astype(np.float32) for key in ("query", "key", "value", "g", "beta")]
        q *= 1 / np.sqrt(np.sum(q * q, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
        k *= 1 / np.sqrt(np.sum(k * k, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
        q *= np.float32(128 ** -.5)
        state = inputs["initial_state"].astype(np.float32)
        output = np.zeros(shape, dtype=np.float32)
        for position in range(int(inputs["effective_length"][0])):
            decayed = state * np.exp(decay[:, position])[..., None, None]
            prediction = np.sum(decayed * k[:, position, :, :, None], axis=-2, dtype=np.float32)
            delta = (v[:, position] - prediction) * beta[:, position, :, None]
            state = decayed + k[:, position, :, :, None] * delta[..., None, :]
            output[:, position] = np.sum(state * q[:, position, :, :, None], axis=-2, dtype=np.float32)
        refs = {"core_attn": output, "final_state": state}
        attrs = {"chunk_size": 64 if sequence > 1 else 1, "output_final_state": True,
                 "use_qk_l2norm_in_kernel": True}
        shapes = {"core_attn": shape, "final_state": state.shape}
    elif role == "conv_state":
        inputs = {"projected": _pattern((1, sequence, 8192), .003, 7),
                  "history": _pattern((1, 8192, conv_history_width), .004, 31),
                  "update_control": np.zeros((128,), dtype="<i4")}
        padded = np.concatenate((inputs["history"][:, :, -3:], inputs["projected"].transpose(0, 2, 1)), axis=-1)
        shapes = {"history": (1, 8192, bindings.conv_history_size(4)), "padded": padded.shape}
        refs, attrs = {"history": None, "padded": padded if conv_history_width == 3 else None}, {}
    elif role == "cache":
        inputs = {"x": _pattern((64, 32, 32, 16), .001),
                  "updates": _pattern((sequence, 32, 16), .002, 17),
                  "targetBlock": np.asarray([3], dtype="<i4"),
                  "offsetInBlock": np.asarray(0 if sequence == 8 else 7, dtype="<i4")}
        shapes, refs, attrs = {"x": inputs["x"].shape}, {"x": None}, {}
    elif role == "attention":
        mask = np.full((1, 1, sequence, 2048), -10000, dtype="<f2")
        for row in range(sequence):
            mask[0, 0, row, :row + 1 if sequence > 1 else 8] = 0
        inputs = {"query": _pattern((1, 256, sequence, 16), .002),
                  "key0": _pattern((64, 32, 32, 16), .001, 17),
                  "value0": _pattern((64, 32, 32, 16), .003, 37),
                  "atten_mask": mask,
                  "actual_seq_lengths_q": np.asarray([sequence], dtype="<i8"),
                  "actual_seq_lengths_kv": np.asarray([2048], dtype="<i8"),
                  "block_table": np.arange(64, dtype="<i4").reshape(1, 64),
                  "all_seq_lengths_q": np.asarray([sequence], dtype="<i8")}
        shapes, refs, attrs = {"attention_out": inputs["query"].shape}, {"attention_out": None}, {}
    else:
        inputs = {"x": _pattern((1, sequence, 2048), .03125, 11),
                  "gamma": (.75 + np.arange(2048, dtype=np.float32) % 17 / 32).astype("<f2")}
        x = inputs["x"].astype(np.float32)
        gain = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
        refs = {"y": x * gain * inputs["gamma"].astype(np.float32), "rstd": gain}
        shapes, attrs = {"y": x.shape, "rstd": gain.shape}, {"epsilon": 1e-6}
    for key in refs:
        if refs[key] is not None:
            refs[key] = np.asarray(refs[key], dtype=DTYPES[bindings.tensor_dtype(role, "outputs", key)])
    return inputs, shapes, refs, attrs


def _construct_case(root, name, role, sequence, bindings, conv_history_width=3):
    folder = root / "cases" / name
    folder.mkdir(parents=True)
    values, shapes, references, attrs = _case_values(role, sequence, bindings, conv_history_width)
    probe_only = role in {"cache", "attention"}
    spec = document_probe_spec(role) if probe_only else bindings.manifest()["operators"][role]
    conv_candidate = role == "conv_state" and conv_history_width == 4
    if conv_candidate:
        spec = deepcopy(spec)
        spec["inputs"]["history"]["shape"] = [1, 8192, 4]
    graph, inputs = Graph(name), {}
    for logical, port in spec["inputs"].items():
        value = values[logical]
        inputs[logical] = graph.input(port["port"], value.shape, str(value.dtype))
    if probe_only or conv_candidate:
        for logical, value in inputs.items():
            OperatorBindings._tensor_contract(spec["inputs"][logical], value.shape, value.dtype, logical)
        for logical, desc in spec["outputs"].items():
            OperatorBindings._tensor_contract(desc, shapes[logical], desc["dtype"], logical)
        raw = graph.custom(spec["op_type"], {spec["inputs"][key]["port"]: value for key, value in inputs.items()},
                           {desc["port"]: (shapes[key], desc["dtype"]) for key, desc in spec["outputs"].items()},
                           spec["attrs"] if probe_only else spec.get("static_attrs", {}), name=name + "_fused")
        outputs = {key: raw[desc["port"]] for key, desc in spec["outputs"].items()}
    else:
        outputs = bindings.call(graph, role, inputs=inputs,
                                outputs={key: (shape, bindings.tensor_dtype(role, "outputs", key))
                                         for key, shape in shapes.items()}, attrs=attrs, name=name + "_fused")
    graph.outputs = list(outputs.values())
    graph.validate()
    assert len(graph.nodes) == 1 and isinstance(graph.nodes[0], CustomNode) and not graph.weights and not graph.constants
    node = graph.nodes[0]
    abi = {"schema_version": 1, "artifact_kind": "qwen35-fused-operator-abi",
           "role": role, "operator_type": node.op_type, "binding_sha256": bindings.sha256,
           "attrs": node.attrs, "inputs": [], "outputs": []}
    if probe_only:
        abi["probe_contract"] = spec
    if conv_candidate:
        abi["probe_override"] = {"conv_history_width": 4, "status": "UNVERIFIED_CANDIDATE",
                                 "scope": "Installed infer-shape requires x2 width4; y1/y2 shapes and control semantics still document-derived; no numerical reference."}
    for direction, ports, tensors in (("inputs", node.input_ports, node.inputs),
                                      ("outputs", node.output_ports, node.outputs)):
        abi[direction] = [{"index": index, "name": port, "dtype": tensor.dtype, "shape": list(tensor.shape)}
                          for index, (port, tensor) in enumerate(zip(ports, tensors))]
    save_json(folder / "operator-abi.json", abi)
    save_json(folder / "graph.json", _structure_graph(graph))
    (folder / "probe.cpp").write_text(emit_cpp(graph, {}, recipe_dir=folder / "recipe", fold_constant_arithmetic=False))
    recipe = read_recipe(folder / "recipe/graph.recipe")
    if [record["type"] for record in recipe["records"]] != ["Data"] * len(inputs) + [node.op_type]:
        raise ValueError("single-op probe recipe must contain only its Data inputs and one declared custom operator")
    input_records, output_records = [], []
    for index, (logical, desc) in enumerate(zip(spec["inputs"], abi["inputs"])):
        path = folder / f"input_{index:04d}.bin"
        path.write_bytes(values[logical].tobytes())
        input_records.append({**desc, **_record(path, root)})
    for index, (logical, desc) in enumerate(zip(spec["outputs"], abi["outputs"])):
        item = {**desc, "logical": logical, "reference": None,
                "comparison": ("SHAPE_ONLY_UNVERIFIED_CONTRACT" if probe_only or conv_candidate else
                               "SHAPE_ONLY_UNSPECIFIED_STATE_ORDER") if references[logical] is None else "FP32_LOGIC_DIAGNOSTIC"}
        if role == "gdn" and logical == "core_attn":
            item["valid_prefix_length"] = int(values["effective_length"][0])
        if references[logical] is not None:
            path = folder / f"reference_{index:04d}.bin"
            path.write_bytes(references[logical].tobytes())
            item["reference"] = _record(path, root)
        output_records.append(item)
    return {"name": name, "role": role, "sequence_length": sequence, "operator_type": node.op_type,
            "status": "SOURCE_READY", "construction": "PASSED", "export": "NOT_RUN", "compile": "NOT_RUN",
            "load": "NOT_RUN", "process": "NOT_RUN", "numerical": "NOT_RUN", "placement": "UNVERIFIED",
            "operator_abi": _record(folder / "operator-abi.json", root),
            "inputs": input_records, "outputs": output_records, "bundle_manifest": None,
            "input_policy": {"generator": "deterministic modular tags; no RNG or checkpoint",
                             "ranges": {key: {"min": float(value.min()), "max": float(value.max()),
                                               "nonzero": int(np.count_nonzero(value))}
                                        for key, value in values.items()}}}


def _export_case(folder, ge_root, cxx, cxx11_abi, timeout_seconds):
    ge, include, lib, compiler = _toolchain(ge_root, cxx, cxx11_abi, timeout_seconds)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    command = [compiler, "-std=c++17", "-O2", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
               "-I", str(include), str(folder / "probe.cpp"), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
    if (lib / "libgraph_base.so").is_file():
        command.append("-lgraph_base")
    command.extend(["-o", str(folder / "probe_exporter")])
    save_json(folder / "build-command.json", {"argv": command})
    _logged(command, folder=folder, env=env, log=folder / "build.log", timeout=timeout_seconds)
    binary = folder / "probe_exporter"
    if binary.is_symlink() or not binary.is_file() or not binary.stat().st_size:
        raise RuntimeError("host compiler did not produce an exporter")
    command = [str(binary), str(folder / "probe.air")]
    save_json(folder / "export-command.json", {"argv": command})
    _logged(command, folder=folder, env=env, log=folder / "export.log", timeout=timeout_seconds)
    air = folder / "probe.air"
    if air.is_symlink() or not air.is_file() or not air.stat().st_size:
        raise RuntimeError("GE exporter did not produce a nonempty regular AIR")
    return {"ge_root": str(ge), "libgraph_sha256": sha256_file(lib / "libgraph.so"), "cxx11_abi": cxx11_abi}


def _compile_case(folder, ddk_root, timeout_seconds, tbe_diagnosis=None):
    omg, env = omg_environment(ddk_root)
    if not os.access(omg, os.X_OK):
        raise PermissionError("selected OMG executable is not executable")
    destination = folder / "omc"
    destination.mkdir()
    command = [str(omg), "--framework=1", "--model=input.air", "--output=probe", "--target=omc",
               "--platform=kirinx90", "--use_origin_format=true", "--weight_data_type=FP16"]
    scratch = folder / "compile-work"
    scratch.mkdir()
    # Retain compiler workers' tracebacks, kernel metadata and failed IR dumps.
    # These single-op AIR files have no external weights; a regular input copy
    # also keeps the retained case inventory free of our own temporary links.
    shutil.copyfile(folder / "probe.air", scratch / "input.air")
    save_json(folder / "compile-command.json", {
        "argv": command, "cwd": "compile-work",
        "diagnostics_retention": "Compiler work retained on success and failure; IPC temporary files cleaned."})
    compatibility = None
    invocation, outcome = "NOT_ATTEMPTED", "NOT_RUN"
    try:
        if tbe_diagnosis is not None:
            from ..compile import prepare_tbe_compatibility
            env, compatibility = prepare_tbe_compatibility(ddk_root, tbe_diagnosis,
                                                            folder / "tbe-compat", inherited=env)
        with _omg_ipc_environment(scratch, env) as (local_env, _):
            save_json(folder / "omg-environment.json", _environment_record(ddk_root, local_env))
            invocation = "ATTEMPTED"
            _logged(command, folder=scratch, env=local_env, log=folder / "compile.log", timeout=timeout_seconds)
        model = scratch / "probe.omc"
        if model.is_symlink() or not model.is_file() or not model.stat().st_size:
            raise RuntimeError("OMG succeeded without a nonempty regular probe.omc")
        for path in scratch.iterdir():
            if path.name.startswith("probe") and not path.is_symlink():
                shutil.move(str(path), str(destination / path.name))
        outcome = "PASSED"
    except Exception as error:
        if invocation == "ATTEMPTED":
            outcome = "FAILED"
        raise RuntimeError(f"{error}\nRetained compiler work: {scratch}") from error
    finally:
        if tbe_diagnosis is not None:
            audit = {"status": "NOT_PREPARED" if compatibility is None else "NOT_OBSERVED"}
            if compatibility is not None and invocation == "ATTEMPTED":
                from ..compile import verify_tbe_compatibility
                try:
                    audit = verify_tbe_compatibility(compatibility)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    audit = {"status": "AUDIT_FAILED", "error": str(error)}
            save_json(folder / "tbe-compatibility-result.json", {
                "compiler_invocation": invocation, "compile": outcome,
                "audit": audit, "metadata": compatibility,
                "scope": "Prepared files alone do not establish worker use; compiler outcome and adaptation audit are separate."})
    return {"omg_sha256": sha256_file(omg),
            "compile_work": str(scratch.relative_to(folder.parents[1])),
            "compiler_evidence": compile_evidence((folder / "compile.log").read_text(errors="replace"))}


def _wire(model, abi, operator_type, inputs, outputs):
    rows = ["QWEN35_FUSED_OP_STEP\t1", f"model\t{model['path']}\t{model['sha256']}",
            f"operator_abi\t{abi['path']}\t{abi['sha256']}", f"operator\t{operator_type}"]
    for item in inputs:
        rows.append("\t".join(("input", str(item["index"]), item["name"], item["dtype"],
                               ",".join(map(str, item["shape"])), item["path"], item["sha256"])))
    for item in outputs:
        rows.append("\t".join(("output", str(item["index"]), item["name"], item["dtype"], ",".join(map(str, item["shape"])))))
    return ("\n".join(rows) + "\n").encode("ascii")


def _bundle_case(root, case):
    folder = root / "bundle" / case["name"]
    folder.mkdir()
    abi_source = _step_file(root, case["operator_abi"])
    shutil.copyfile(abi_source, folder / "operator-abi.json")
    abi_record = _record(folder / "operator-abi.json", folder)
    abi = _stateful_json(abi_source)
    inputs = []
    for item in case["inputs"]:
        source = _step_file(root, item)
        path = folder / f"input_{item['index']:04d}.bin"
        shutil.copyfile(source, path)
        inputs.append({**item, **_record(path, folder)})
    source_models = root / "cases" / case["name"] / "omc"
    shutil.copytree(source_models, folder / "model")
    payloads = files_under(folder / "model", folder)
    model = _record(folder / "model/probe.omc", folder)
    wire = _wire(model, abi_record, case["operator_type"], inputs, abi["outputs"])
    (folder / "step.tsv").write_bytes(wire)
    result = {"schema_version": 1, "artifact_kind": "qwen35-fused-op-step", "status": "PREPARED",
              "operator_type": case["operator_type"], "operator_abi": abi_record,
              "inputs": inputs, "outputs": abi["outputs"], "model": model, "model_payloads": payloads,
              "step_plan": _record(folder / "step.tsv", folder)}
    save_json(folder / "manifest.json", result)
    return _record(folder / "manifest.json", root)


def _board_script(cases, *, output_fill_byte=None):
    """Return the board batch script for ordered, already compiled case names."""
    known = {name for name, _, _ in _CASES}
    if (not isinstance(cases, (list, tuple)) or any(not isinstance(name, str) for name in cases)
            or len(set(cases)) != len(cases) or not set(cases) <= known):
        raise ValueError("board script cases must be unique known probe names")
    if output_fill_byte is not None and (type(output_fill_byte) is not int
            or not 0 <= output_fill_byte <= 255 or not cases
            or not set(cases) <= {"gdn_prefill", "gdn_decode"}):
        raise ValueError("output fill diagnostic requires GDN cases and a byte in 0..255")
    diagnostic = "" if output_fill_byte is None else f" --output-fill-byte {output_fill_byte}"
    script = '''#!/bin/sh
set -u
[ "$#" -eq 5 ] || { echo "usage: run-board.sh RUNNER NEW_RESULTS_DIR RUNTIME_VERSION DDK_ID FIRMWARE_ID" >&2; exit 2; }
runner=$1
results=$2
runtime=$3
ddk=$4
firmware=$5
bundle=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
[ ! -e "$results" ] || { echo "results directory already exists" >&2; exit 2; }
mkdir -p "$results" || exit 2
overall=0
'''
    if not cases:
        script += '''echo "No COMPILED fused operator cases; export and compile the source graphs before running." >&2
printf '%s\\n' "No COMPILED cases; no operator invocation attempted." >"$results/run-error.txt"
exit 2
'''
    for name in cases:
        script += f'''"$runner" --fused-op-step "$bundle/{name}/step.tsv" --output "$results/{name}" --expected-runtime-version "$runtime" --ddk-id "$ddk" --firmware-id "$firmware"{diagnostic} >"$results/{name}.stdout" 2>"$results/{name}.stderr"
code=$?
printf '%s\\n' "$code" >"$results/{name}.exitcode"
[ "$code" -eq 0 ] || overall=1
'''
    return script + 'exit "$overall"\n'


def build_fused_probes(output_dir, bindings_path, *, ge_root=None, ddk_root=None, ddk_id=None,
                       cxx="c++", cxx11_abi=0, timeout_seconds=3600, cases=None, tbe_diagnosis=None,
                       conv_history_width=3):
    selected = _select_cases(cases)
    selected_roles = {role for _, role, _ in selected}
    bindings = OperatorBindings.from_file(bindings_path)
    bindings.require(selected_roles - {"cache", "attention"})
    declared = bindings.manifest()
    if any(declared["operators"][role]["op_type"] != _TYPES[role] for role in selected_roles - {"cache", "attention"}):
        raise ValueError("selected probes require the corresponding documented operator types")
    if (ddk_root is None) != (ddk_id is None) or (ddk_root is not None and ge_root is None):
        raise ValueError("compilation requires ge_root, ddk_root and ddk_id together")
    if ddk_root is not None and (declared["ddk_id"] != ddk_id or declared["target"] != "kirinx90"):
        raise ValueError("binding DDK identity and target must match selected DDK and OMG platform kirinx90")
    if type(cxx11_abi) is not int or cxx11_abi not in (0, 1) or timeout_seconds <= 0:
        raise ValueError("invalid C++ ABI or timeout")
    if type(conv_history_width) is not int or conv_history_width not in (3, 4):
        raise ValueError("conv_history_width must be 3 (document) or 4 (unverified installed-shape candidate)")
    if tbe_diagnosis is not None and ddk_root is None:
        raise ValueError("tbe_diagnosis requires an actual DDK compilation")
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "bundle").mkdir()
    shutil.copyfile(bindings_path, root / "bindings.json")
    if sha256_file(root / "bindings.json") != bindings.sha256:
        raise ValueError("bindings changed while copying")
    cases = []
    for name, role, sequence in selected:
        case = {"name": name, "role": role, "sequence_length": sequence, "operator_type": _TYPES[role],
                "status": "FAILED", "construction": "NOT_RUN", "export": "NOT_RUN", "compile": "NOT_RUN",
                "load": "NOT_RUN", "process": "NOT_RUN", "numerical": "NOT_RUN", "placement": "UNVERIFIED"}
        stage = "construction"
        try:
            case = _construct_case(root, name, role, sequence, bindings, conv_history_width)
            folder = root / "cases" / name
            if ge_root is not None:
                stage = "export"
                case["ge_toolchain"] = _export_case(folder, ge_root, cxx, cxx11_abi, timeout_seconds)
                case.update(status="EXPORTED", export="PASSED", air=_record(folder / "probe.air", root))
            if ddk_root is not None:
                stage = "compile"
                case.update(_compile_case(folder, ddk_root, timeout_seconds, tbe_diagnosis))
                case.update(status="COMPILED", compile="PASSED", ddk_id=ddk_id, platform="kirinx90")
                case["model"] = _record(folder / "omc/probe.omc", root)
                stage = "bundle"
                case["bundle_manifest"] = _bundle_case(root, case)
        except Exception as error:
            case.update(status="FAILED", failed_stage=stage, error=f"{type(error).__name__}: {error}")
            case[stage] = "FAILED"
        compatibility_result = root / "cases" / name / "tbe-compatibility-result.json"
        if compatibility_result.is_file():
            case["tbe_compatibility"] = _stateful_json(compatibility_result)
        case["files"] = files_under(root / "cases" / name, root) if (root / "cases" / name).exists() else []
        cases.append(case)
        if ddk_root is not None:
            print(json.dumps({"case": name, "export": case["export"], "compile": case["compile"],
                              "completed": len(cases), "total": len(selected)}), flush=True)
    compiled = [case["name"] for case in cases if case["status"] == "COMPILED"]
    (root / "bundle/run-board.sh").write_text(_board_script(compiled))
    (root / "bundle/run-board.sh").chmod(0o700)
    statuses = {case["status"] for case in cases}
    result = {"schema_version": 1, "artifact_kind": "qwen35-fused-op-probes",
              "status": next(iter(statuses)) if len(statuses) == 1 else "PARTIAL",
              "bindings": _record(root / "bindings.json", root), "binding_contract": declared,
              "ddk_id": ddk_id, "platform": "kirinx90" if ddk_root is not None else None,
              "selected_cases": [name for name, _, _ in selected],
              "cases": cases, "run_script": _record(root / "bundle/run-board.sh", root),
              "reference_policy": "FP32 sequential GDN and RMS logic; outputs rounded to declared dtype; not kernel emulation. Conv y1 ordering is unspecified. Width4 conv, cache and attention outputs have no semantic reference or accuracy acceptance.",
              "device_execution": "NOT_RUN", "placement": "UNVERIFIED"}
    save_json(root / "manifest.json", result)
    _report(root / "report.txt", cases)
    return result


def _report(path, cases):
    lines = ["Documented fused operator probes", "", "case | construction | export | compile | load | process | numerical diagnostic | placement"]
    for case in cases:
        lines.append(" | ".join(str(case.get(key, "NOT_RUN")) for key in
                                ("name", "construction", "export", "compile", "load", "process", "numerical", "placement")))
        if case.get("error"):
            lines.append("  " + case["error"])
        if case.get("tbe_compatibility"):
            compatibility = case["tbe_compatibility"]
            audit = compatibility["audit"]
            events = [event for event in audit.get("events", []) if isinstance(event, dict)]
            summary = {"status": audit["status"],
                       "applied_count": sum(event.get("event") == "APPLIED" for event in events),
                       "rejected_count": sum(event.get("event") == "REJECTED" for event in events),
                       "pids": sorted({event["pid"] for event in events
                                       if type(event.get("pid")) is int and event["pid"] > 0}),
                       "audit_log": audit.get("audit_log") or (compatibility.get("metadata") or {}).get("audit_log")}
            error = audit.get("error") or next((event.get("error") or event.get("reason") for event in events
                                               if event.get("event") == "REJECTED" and
                                               (event.get("error") or event.get("reason"))), None)
            if error:
                summary["error"] = error
            lines.append("  TBE compatibility: " + json.dumps({
                "compiler_invocation": compatibility["compiler_invocation"],
                "compile": compatibility["compile"], "audit": summary}, allow_nan=False))
        for item in case.get("diagnostics", []):
            lines.append("  " + json.dumps(item, allow_nan=False))
    lines.extend(["", "Numerical values are diagnostics without an acceptance threshold. Runtime placement remains UNVERIFIED.",
                  "Source/recipe contains one declared custom node; this does not identify the runtime kernel or prove device support."])
    path.write_text("\n".join(lines) + "\n")


def _diagnostic(actual, reference, desc):
    result = {"index": desc["index"], "name": desc["name"], "shape": desc["shape"], "dtype": desc["dtype"],
              "raw_nonfinite": int(np.count_nonzero(~np.isfinite(actual))), "reference": desc["comparison"]}
    if reference is None:
        return result
    if "valid_prefix_length" in desc:
        length = desc["valid_prefix_length"]
        result.update(valid_prefix_length=length, padded_positions_unchecked=actual.shape[1] - length)
        actual, reference = actual[:, :length], reference[:, :length]
    result.update(compared_shape=list(actual.shape), actual_nonfinite=int(np.count_nonzero(~np.isfinite(actual))),
                  actual_nonzero=int(np.count_nonzero(actual)))
    expected = reference.astype(np.float64)
    value = actual.astype(np.float64)
    finite = np.isfinite(expected) & np.isfinite(value)
    delta = value[finite] - expected[finite]
    norm = float(np.linalg.norm(expected[finite]))
    result.update(reference_nonfinite=int(np.count_nonzero(~np.isfinite(reference))),
                  finite_pairs=int(np.count_nonzero(finite)),
                  max_abs_error=float(np.abs(delta).max()) if delta.size else None,
                  relative_l2=float(np.linalg.norm(delta)) / norm if norm else None,
                  all_zero_with_nonzero_reference=bool(not np.count_nonzero(actual) and np.count_nonzero(reference)))
    if result["all_zero_with_nonzero_reference"]:
        result["signal_loss"] = "ALL_ZERO_OUTPUT_WITH_NONZERO_REFERENCE"
    return result


def compare_fused_probes(probe_run, board_run, output_dir, *, allow_host_fake=False, case_names=None):
    """Check complete native result identity, then report threshold-free errors."""
    if type(allow_host_fake) is not bool:
        raise ValueError("allow_host_fake must be boolean")
    root, board = Path(probe_run).resolve(), Path(board_run).resolve()
    manifest = _stateful_json(root / "manifest.json")
    _validate_probe_manifest(manifest)
    names = {case["name"] for case in manifest["cases"]}
    if case_names is not None:
        if (not isinstance(case_names, (list, tuple)) or not case_names
                or any(not isinstance(name, str) for name in case_names)
                or len(set(case_names)) != len(case_names) or not set(case_names) <= names):
            raise ValueError("case_names must be a nonempty unique subset of probe manifest names")
        names = set(case_names)
    bindings = OperatorBindings.from_file(_step_file(root, manifest["bindings"]))
    if bindings.manifest() != manifest["binding_contract"]:
        raise ValueError("probe bindings differ from recorded contract")
    cases = []
    for original in manifest["cases"]:
        if original["name"] not in names:
            continue
        case = {key: original.get(key, "NOT_RUN") for key in
                ("name", "construction", "export", "compile", "load", "process", "numerical", "placement")}
        case["placement"] = "UNVERIFIED"
        if original["status"] != "COMPILED":
            case["error"] = original.get("error", "case has no compiled OMC; no device result consumed")
            cases.append(case)
            continue
        try:
            for record in original["files"]:
                _step_file(root, record)
            abi = _stateful_json(_step_file(root, original["operator_abi"]))
            step_path = _step_file(root, original["bundle_manifest"])
            step_root, step = step_path.parent, _stateful_json(step_path)
            if (step.get("artifact_kind") != "qwen35-fused-op-step" or step.get("status") != "PREPARED"
                    or step.get("operator_type") != original["operator_type"]
                    or abi.get("operator_type") != original["operator_type"]
                    or abi.get("binding_sha256") != bindings.sha256
                    or step.get("outputs") != abi["outputs"]):
                raise ValueError("bundle operator contract mismatch")
            abi_file = _step_file(step_root, step["operator_abi"])
            if _stateful_json(abi_file) != abi:
                raise ValueError("bundle ABI differs from constructed operator")
            _step_file(step_root, step["model"])
            source_model = _step_file(root, original["model"])
            if step["model"]["sha256"] != sha256_file(source_model) or step["model"]["bytes"] != source_model.stat().st_size:
                raise ValueError("bundle model differs from compiled operator OMC")
            if step["model"] not in step["model_payloads"]:
                raise ValueError("bundle model missing from payload inventory")
            for item in step["model_payloads"]:
                _step_file(step_root, item)
            if len(step["inputs"]) != len(original["inputs"]):
                raise ValueError("bundle input count mismatch")
            for actual, expected in zip(step["inputs"], original["inputs"]):
                if any(actual.get(key) != expected[key] for key in ("index", "name", "dtype", "shape", "bytes", "sha256")):
                    raise ValueError("bundle input differs from deterministic source input")
                _step_file(step_root, actual)
            plan = _step_file(step_root, step["step_plan"])
            if plan.read_bytes() != _wire(step["model"], step["operator_abi"], step["operator_type"], step["inputs"], step["outputs"]):
                raise ValueError("bundle plan differs from recorded operator boundary")
            result_root = board / original["name"]
            result_path = result_root / "result.json"
            if result_root.is_symlink() or result_path.is_symlink():
                raise ValueError("native result must not use symlinks")
            result = _stateful_json(result_path)
            expected = {"schema_version": 1, "scope": "fused-operator-probe",
                        "operator_type": step["operator_type"], "target_calls": 0, "draft_calls": 0,
                        "operator_abi_sha256": step["operator_abi"]["sha256"],
                        "step_plan_sha256": step["step_plan"]["sha256"], "model_sha256": step["model"]["sha256"],
                        "ddk_id": manifest["ddk_id"]}
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("native result does not bind this single-op invocation")
            if any(type(result.get(key)) is not int for key in ("schema_version", "target_calls", "draft_calls")):
                raise ValueError("native result counters must be integers")
            evidence = result.get("evidence_kind")
            if evidence not in {"device-hiai", "host-fake-hiai"} or (evidence == "host-fake-hiai" and not allow_host_fake):
                raise ValueError("host-fake result requires explicit allow_host_fake; device evidence otherwise required")
            if result.get("status") == "RUNTIME_FAILED":
                stages = {"LOAD_OR_SETUP": ("UNVERIFIED", "NOT_RUN"),
                          "INPUT_BINDING": ("PASSED", "NOT_RUN"), "PROCESS": ("PASSED", "FAILED")}
                stage = result.get("failure_stage")
                if (stage not in stages or result.get("outputs") != []
                        or not isinstance(result.get("reason"), str) or not result["reason"]
                        or not isinstance(result.get("expected_runtime_version"), str)
                        or not result["expected_runtime_version"]):
                    raise ValueError("native failure record lacks a supported stage or failure context")
                load, process = stages[stage]
                case.update(load=load, process=process, numerical="NOT_EVALUATED",
                            failure_stage=stage, runtime="FAILED", result_validation="PASSED",
                            error=f"Native {stage} failed: {result['reason']}", evidence_kind=evidence,
                            result_sha256=sha256_file(result_path),
                            expected_runtime_version=result["expected_runtime_version"])
                cases.append(case)
                continue
            if (result.get("status") != "EXECUTED" or type(result.get("operator_calls")) is not int
                    or result["operator_calls"] != 1):
                raise ValueError("native result does not record one executed operator invocation")
            if not isinstance(result.get("runtime_version"), str) or not result["runtime_version"]:
                raise ValueError("native result lacks runtime version")
            if len(result.get("inputs", [])) != len(step["inputs"]) or len(result.get("outputs", [])) != len(step["outputs"]):
                raise ValueError("native result boundary count mismatch")
            for actual, expected_input in zip(result["inputs"], step["inputs"]):
                if any(actual.get(key) != expected_input[key] for key in ("index", "name", "dtype", "shape", "bytes", "sha256")):
                    raise ValueError("native input inventory differs from the bound inputs")
            diagnostics, seen = [], set()
            for actual, desc, source_desc in zip(result["outputs"], step["outputs"], original["outputs"]):
                if (any(actual.get(key) != desc[key] for key in ("index", "name", "dtype", "shape"))
                        or actual.get("bytes") != math.prod(desc["shape"]) * DTYPES[desc["dtype"]].itemsize
                        or actual.get("path") in seen):
                    raise ValueError("native output inventory differs from the declared ports")
                seen.add(actual["path"])
                values = np.frombuffer(_step_file(result_root, actual).read_bytes(), dtype=DTYPES[desc["dtype"]]).reshape(desc["shape"])
                reference = source_desc["reference"]
                if reference is not None:
                    reference = np.frombuffer(_step_file(root, reference).read_bytes(), dtype=DTYPES[desc["dtype"]]).reshape(desc["shape"])
                diagnostics.append(_diagnostic(values, reference, source_desc))
            case.update(load="PASSED", process="PASSED", numerical="DIAGNOSTIC_COMPLETE", diagnostics=diagnostics,
                        result_validation="PASSED", evidence_kind=evidence,
                        result_sha256=sha256_file(result_path), runtime_version=result["runtime_version"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            case.update(load="UNVERIFIED", process="UNVERIFIED", numerical="NOT_EVALUATED",
                        error=f"{type(error).__name__}: {error}", result_validation="FAILED")
        cases.append(case)
    output = output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-fused-op-comparison", "status": "DIAGNOSTIC_COMPLETE",
              "probe_manifest_sha256": sha256_file(root / "manifest.json"), "cases": cases,
              "numerical_acceptance": "NOT_DEFINED", "placement": "UNVERIFIED", "allow_host_fake": allow_host_fake}
    save_json(output / "report.json", result)
    _report(output / "report.txt", cases)
    return result

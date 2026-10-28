"""Direct GE target/draft preparation and host AIR serialization.

Only NumPy and the standard library are needed here. PyTorch/TorchAir export
remains a separate, unchanged route. Full-size FP16 is a conversion diagnostic;
the packed integer-weight GE mapping is not implemented by this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .artifacts import checked_file, files_under, output_path, save_json
from .config import Qwen35DFlashConfig
from .hashing import sha256_file
from .resources import repository_root
from .tool_logs import log_excerpt as _log_excerpt


def configurations(*, tiny=False):
    specs = repository_root() / "specs"
    target = json.loads((specs / "target-config.json").read_text())["text_config"]
    draft = json.loads((specs / "draft-config.json").read_text())
    if tiny:
        target.update(vocab_size=64, hidden_size=16, num_hidden_layers=4,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                      linear_num_key_heads=2, linear_num_value_heads=2,
                      linear_key_head_dim=8, linear_value_head_dim=8,
                      linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
                      moe_intermediate_size=16, shared_expert_intermediate_size=16,
                      max_position_embeddings=128, pad_token_id=0,
                      layer_types=["linear_attention"] * 3 + ["full_attention"],
                      rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                       "partial_rotary_factor": .5, "mrope_section": [1, 1, 0]})
        draft.update(hidden_size=16, intermediate_size=32, vocab_size=64,
                     num_hidden_layers=2, num_target_layers=4, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
                     layer_types=["sliding_attention", "full_attention"], sliding_window=8)
        draft["dflash_config"] = {"block_size": 4, "mask_token_id": 63, "target_layer_ids": [0, 2]}
    return target, draft


def build_pair(target_config, draft_config, sequence_length, *, expert_dispatch="static",
               gdn_indexed_copy="native", gdn_copy_reports=None):
    from .ge_target import build_target
    from .ge_draft import build_draft
    config = Qwen35DFlashConfig.from_dict(draft_config)
    if (sequence_length < 2 or sequence_length + config.block_size > config.max_position_embeddings
            or sequence_length * config.feature_size * 2 > 2**32 - 1):
        raise ValueError("sequence length exceeds graph, RoPE or HIAI capacity")
    target_copy_report = {}
    graphs = [build_target(target_config, config, sequence_length, expert_dispatch=expert_dispatch,
                           gdn_indexed_copy=gdn_indexed_copy, gdn_copy_report=target_copy_report),
              build_draft(config, sequence_length)]
    if gdn_copy_reports is not None:
        gdn_copy_reports.clear()
        gdn_copy_reports.update(target=target_copy_report,
                                draft={"schema_version": 1, "graph": "draft", "mode": gdn_indexed_copy,
                                       "counts": {"total": 0, "replaced": 0, "retained": 0,
                                                  "mask_replaced": 0, "sub_replaced": 0}, "entries": []})
    expected = [
        ([(1, sequence_length)] * 2, ["int32", "int32"],
         [(1, sequence_length), (1, sequence_length, config.feature_size)], ["int32", "float16"]),
        ([(1, sequence_length, config.feature_size), (1, sequence_length), (1, sequence_length)],
         ["float16", "int32", "int32"], [(1, config.block_size - 1)], ["int32"]),
    ]
    for graph, (ins, ind, outs, outd) in zip(graphs, expected):
        graph.validate()
        if ([v.shape for v in graph.inputs] != ins or [v.dtype for v in graph.inputs] != ind
                or [v.shape for v in graph.outputs] != outs or [v.dtype for v in graph.outputs] != outd):
            raise ValueError(f"{graph.name}: direct GE graph differs from HIAI ABI v2")
        if any(v.dtype != "float16" for v in graph.weights.values()):
            raise ValueError("floating GE path requires FP16 weights")
    return graphs


def weight_shapes(graphs):
    result = {}
    for graph in graphs:
        for key, value in graph.weights.items():
            shape = tuple(value.shape)
            if key in result and result[key] != shape:
                raise ValueError(f"inconsistent shared weight shape: {key}")
            result[key] = shape
    return result


def graph_summary(graph):
    def tensor(v):
        return {"name": v.name, "shape": list(v.shape), "dtype": v.dtype}
    return {"name": graph.name, "nodes": len(graph.nodes),
            "operations": dict(sorted(Counter(n.kind for n in graph.nodes).items())),
            "inputs": [tensor(v) for v in graph.inputs], "outputs": [tensor(v) for v in graph.outputs],
            "weight_tensors": len(graph.weights),
            "weight_bytes": sum(math.prod(v.shape) * 2 for v in graph.weights.values())}


def weight_identity(records):
    payload = {key: {field: item[field] for field in ("shape", "dtype", "bytes", "sha256")}
               for key, item in sorted(records.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _expert_execution(target, dispatch):
    return {"experts_per_token": target["num_experts_per_tok"] if dispatch == "selected" else target["num_experts"],
            "implementation": "per-token Gather + BatchMatMul" if dispatch == "selected" else "all-expert MatMul",
            "merge_order": "ascending expert ID, sequential FP16 additions",
            "selected_weight_bytes_per_token": (target["num_experts_per_tok"] * 3 * target["hidden_size"]
                                                 * target["moe_intermediate_size"] * 2
                                                 if dispatch == "selected" else 0),
            "workspace_peak": "NOT_MEASURED"}


def plan_pair(output_dir, sequence_length=8, *, tiny=True, expert_dispatch="static"):
    target, draft = configurations(tiny=tiny)
    graphs = build_pair(target, draft, sequence_length, expert_dispatch=expert_dispatch)
    root = output_path(output_dir); root.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-ge-plan", "status": "PLANNED",
              "tiny": tiny, "sequence_length": sequence_length,
              "target_config": target, "draft_config": draft,
              "graphs": [graph_summary(g) for g in graphs],
              "unique_weight_bytes": sum(math.prod(s) * 2 for s in weight_shapes(graphs).values()),
              "precision": "float16", "quantization": None, "expert_dispatch": expert_dispatch,
              "expert_execution": _expert_execution(target, expert_dispatch),
              "device_validation": "PENDING", "board_memory_fit": "NOT_ESTABLISHED",
              "limitations": ["Full-size FP16 is not a 32GB deployment.",
                              ("Selected weights are gathered per token; this is not native grouped GEMM and may increase memory traffic."
                               if expert_dispatch == "selected" else "All experts execute; this is not a sparse performance implementation."),
                              "Packed quantized GE operators and full-weight quality are pending."]}
    save_json(root / "plan.json", result)
    return result


def _locked_checkpoints(target_path, draft_path):
    lock = json.loads((repository_root() / "specs/upstream.lock.json").read_text())
    for path, identity in zip((target_path, draft_path), lock):
        for name, record in identity["files"].items():
            if "sha256" in record and sha256_file(Path(path) / name) != record["sha256"]:
                raise ValueError(f"{identity['repo']}: {name} differs from locked revision")
    return lock


def prepare_pair(output_dir, sequence_length=8, *, tiny=True, target_path=None, draft_path=None,
                 float_baseline=False, external_weights=False, expert_dispatch="static"):
    from .ge_cpp import emit_cpp
    from .ge_weights import SafeTensorFiles, checkpoint_key, tiny_weights
    if not tiny and not float_baseline:
        raise ValueError("full-size direct GE currently requires --float-baseline; packed quantization is pending")
    if not tiny and (target_path is None or draft_path is None):
        raise ValueError("full-size preparation requires both original checkpoint directories")
    # Reject checkpoint/config problems before allocating the output or reading
    # many gigabytes. No remote code or automatic model downloads are enabled.
    lock = None if tiny else _locked_checkpoints(target_path, draft_path)
    if not tiny:
        external_weights = True  # >2GB protobuf limit; never embed 35B weights.
    root = output_path(output_dir)
    plan = plan_pair(root, sequence_length, tiny=tiny, expert_dispatch=expert_dispatch)
    stage = "prepare-weights"
    try:
        graphs = build_pair(plan["target_config"], plan["draft_config"], sequence_length,
                            expert_dispatch=plan["expert_dispatch"])
        shapes = weight_shapes(graphs)
        folder = root / "weights"; folder.mkdir()
        records = {}
        if tiny:
            values = tiny_weights(shapes)
            for number, (key, value) in enumerate(sorted(values.items())):
                path = folder / f"w{number:05d}.bin"
                with path.open("xb") as stream:
                    stream.write(value.tobytes())
                records[key] = {"path": str(path.relative_to(root)), "shape": list(value.shape),
                                "dtype": "float16", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            identity = {"kind": "ge-tiny-numpy-random", "seed": 17, "generator": "numpy.default_rng",
                        "numpy": np.__version__, "not_upstream_weights": True}
        else:
            readers = {"target": SafeTensorFiles(target_path), "draft": SafeTensorFiles(draft_path)}
            shards = {"target": set(), "draft": set()}
            for number, (key, shape) in enumerate(sorted(shapes.items())):
                owner, source_key = checkpoint_key(key)
                path = folder / f"w{number:05d}.bin"
                entry = readers[owner].write_fp16(source_key, shape, path)
                shards[owner].add(entry["source_shard"])
                records[key] = {**entry, "path": str(path.relative_to(root)),
                                "shape": list(shape), "dtype": "float16"}
            identity = {"kind": "original-checkpoint-fp16-conversion-diagnostic",
                        "checkpoints": [{"owner": owner, "repo": pin["repo"], "revision": pin["revision"],
                                         "shards": readers[owner].source_files(shards[owner])}
                                        for owner, pin in zip(("target", "draft"), lock)]}
        stage = "emit-cpp"
        identity["fp16_payload_sha256"] = weight_identity(records)
        paths = {key: str(root / item["path"]) for key, item in records.items()}
        construction = {}
        for graph in graphs:
            folder = root / graph.name; folder.mkdir()
            (folder / f"{graph.name}.cpp").write_text(emit_cpp(
                graph, paths, external_weights=external_weights, recipe_dir=folder / "recipe"))
            construction[graph.name] = files_under(folder, root)
        result = {**plan, "status": "PREPARED", "artifact_kind": "qwen35-ge-prepared",
                  "source": identity, "weights": records, "external_weights": external_weights,
                  "prepared_root": str(root), "files": files_under(root / "weights", root),
                  "construction_files": construction}
        save_json(root / "prepared.json", result)
        return result
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": stage,
                                          "error": f"{type(error).__name__}: {error}"})
        raise


def _logged(command, *, folder, env, log, timeout):
    try:
        with log.open("x") as stream:
            subprocess.run(command, cwd=folder, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        raise RuntimeError(f"{error}\nFull log: {log}\n{_log_excerpt(log)}") from error


def export_pair(prepared_dir, output_dir, ge_root, *, cxx="c++", cxx11_abi=0, timeout_seconds=3600,
                fp16_reductions="native", fp16_row_mul="native", gdn_indexed_copy="native"):
    from .ge_cpp import emit_cpp
    if fp16_reductions not in {"native", "matmul"}:
        raise ValueError("fp16_reductions must be native or matmul")
    if fp16_row_mul not in {"native", "compact"}:
        raise ValueError("fp16_row_mul must be native or compact")
    if gdn_indexed_copy not in {"native", "gather"}:
        raise ValueError("gdn_indexed_copy must be native or gather")
    prepared = Path(prepared_dir).resolve()
    manifest_path = prepared / "prepared.json"
    plan = json.loads(manifest_path.read_text())
    if plan.get("artifact_kind") != "qwen35-ge-prepared" or plan.get("status") != "PREPARED":
        raise ValueError("expected a successful direct GE preparation")
    dispatch = plan.get("expert_dispatch", "static")
    if dispatch not in {"static", "selected"}:
        raise ValueError("prepared expert_dispatch must be static or selected")
    execution = _expert_execution(plan["target_config"], dispatch)
    if "expert_execution" in plan and plan["expert_execution"] != execution:
        raise ValueError("prepared expert execution metadata differs from expert_dispatch/configuration")
    if cxx11_abi not in (0, 1) or timeout_seconds <= 0:
        raise ValueError("invalid C++ ABI or timeout")
    ge = Path(ge_root).expanduser().resolve()
    include, lib = ge / "include", ge / "lib64"
    for path in (include / "graph/graph.h", include / "graph/operator.h", lib / "libgraph.so"):
        if not path.is_file():
            raise FileNotFoundError(f"matching real GE headers/libraries required: {path}")
    compiler = shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError(f"C++ compiler not found: {cxx}")
    gdn_copy_reports = {}
    graphs = build_pair(plan["target_config"], plan["draft_config"], plan["sequence_length"],
                        expert_dispatch=dispatch, gdn_indexed_copy=gdn_indexed_copy,
                        gdn_copy_reports=gdn_copy_reports)
    shapes = weight_shapes(graphs)
    if set(shapes) != set(plan["weights"]):
        raise ValueError("prepared weight set does not match target/draft graphs")
    if plan["source"].get("fp16_payload_sha256") != weight_identity(plan["weights"]):
        raise ValueError("prepared source identity differs from its actual FP16 weight records")
    sources = {}
    for key, shape in shapes.items():
        item = plan["weights"][key]
        rel = Path(item["path"])
        if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "weights" or ".." in rel.parts:
            raise ValueError(f"invalid prepared weight path: {key}")
        if tuple(item["shape"]) != shape or item["dtype"] != "float16" or item["bytes"] != math.prod(shape) * 2:
            raise ValueError(f"invalid prepared weight descriptor: {key}")
        sources[key] = checked_file(prepared, item)
    if not plan["external_weights"] and sum(math.prod(s) * 2 for s in shapes.values()) > (1800 << 20):
        raise ValueError("embedded weights exceed AIR protobuf budget; use external weights")
    root = output_path(output_dir); root.mkdir(parents=True, exist_ok=False)
    records = []
    stage = "copy-weights"
    try:
        (root / "weights").mkdir()
        paths = {}
        for key, source in sources.items():
            destination = root / plan["weights"][key]["path"]
            # Hardlink immutable source payloads in the same run to avoid
            # another 70GB copy. Each payload is hash-checked before and after.
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            paths[key] = str(destination)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(lib), env.get("LD_LIBRARY_PATH", ""))))
        for graph in graphs:
            name = graph.name; folder = root / name; folder.mkdir()
            source = folder / f"{name}.cpp"
            stage = f"{name}-lowering"
            reduction_report = {}
            row_mul_report = {}
            source.write_text(emit_cpp(graph, paths, external_weights=plan["external_weights"],
                                       recipe_dir=folder / "recipe", fp16_reductions=fp16_reductions,
                                       reduction_report=reduction_report, fp16_row_mul=fp16_row_mul,
                                       row_mul_report=row_mul_report))
            save_json(folder / "reduction-lowering.json", reduction_report)
            save_json(folder / "row-mul-lowering.json", row_mul_report)
            save_json(folder / "gdn-copy-lowering.json", gdn_copy_reports[name])
            construction = files_under(folder, root)
            for item in construction:
                checked_file(root, item)
            binary, air = folder / f"{name}_exporter", folder / f"{name}.air"
            command = [compiler, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
                       "-I", str(include), str(source), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
            if (lib / "libgraph_base.so").is_file():
                command.append("-lgraph_base")
            command.extend(["-o", str(binary)])
            save_json(folder / "build-command.json", {"argv": command})
            stage = f"{name}-build"
            _logged(command, folder=folder, env=env, log=folder / "build.log", timeout=timeout_seconds)
            if not binary.is_file() or not binary.stat().st_size:
                raise RuntimeError("compiler did not produce a nonempty exporter")
            command = [str(binary), str(air)]
            save_json(folder / "export-command.json", {"argv": command})
            stage = f"{name}-serialize"
            _logged(command, folder=folder, env=env, log=folder / "export.log", timeout=timeout_seconds)
            if not air.is_file() or air.is_symlink() or not air.stat().st_size:
                raise RuntimeError(f"{name}: exporter did not produce a nonempty regular AIR")
            for item in construction:
                checked_file(root, item)
            files = [{"path": str(air.relative_to(root)), "bytes": air.stat().st_size, "sha256": sha256_file(air)}]
            if plan["external_weights"]:
                files += [{k: plan["weights"][key][k] for k in ("path", "bytes", "sha256")}
                          for key in graph.weights]
            records.append({"name": name, "air": str(air.relative_to(root)),
                            "inputs": (["input_ids", "attention_mask"] if name == "target" else
                                       ["target_features", "input_ids", "attention_mask"]),
                            "outputs": ["target_top1", "target_features"] if name == "target" else ["draft_top1"],
                            "input_shapes": [list(v.shape) for v in graph.inputs],
                            "input_dtypes": ["torch." + v.dtype for v in graph.inputs],
                            "output_shapes": [list(v.shape) for v in graph.outputs],
                            "output_dtypes": ["torch." + v.dtype for v in graph.outputs], "files": files,
                            "builder_sha256": sha256_file(source), "construction_files": construction,
                            "reduction_lowering": {"mode": reduction_report["mode"],
                                                   "counts": reduction_report["counts"]},
                            "row_mul_lowering": {"mode": row_mul_report["mode"],
                                                 "counts": row_mul_report["counts"]},
                            "gdn_copy_lowering": {"mode": gdn_copy_reports[name]["mode"],
                                                  "counts": gdn_copy_reports[name]["counts"]}})
        for item in plan["weights"].values():
            checked_file(root, item)
        config = Qwen35DFlashConfig.from_dict(plan["draft_config"])
        audit = {g.name: {"status": "PASS", "policy": "fp16-weights-air4b-precision" if g.name == "draft" else "fp16",
                          "scope": "declared GE graph tensor dtypes; not compiler or runtime precision",
                          "floating_intermediate_dtypes": sorted({n.output.dtype for n in g.nodes
                                                                   if n.output.dtype.startswith("float")}),
                          "violations": [], "quantized_matmuls": []} for g in graphs}
        manifest = {"schema_version": 2, "abi_version": 2, "artifact_kind": "qwen35-cpro-air-pair",
                    "pipeline": ["ge-cpp", "air", "omg", "omc", "hiai"], "export_backend": "direct-ge",
                    "status": "EXPORTED", "device_validation": "PENDING", "runtime": "hiai",
                    "sequence_length": plan["sequence_length"], "draft_width": config.block_size - 1,
                    "feature_width": config.feature_size, "vocab_size": config.vocab_size,
                    "precision": "torch.float16", "feature_dtype": "torch.float16", "precision_audit": audit,
                    "quantization": None, "state_policy": "recompute",
                    "expert_dispatch": dispatch, "expert_execution": execution,
                    "fp16_reductions": fp16_reductions,
                    "fp16_row_mul": fp16_row_mul,
                    "gdn_indexed_copy": gdn_indexed_copy,
                    "source": plan["source"], "pad_token_id": 0 if plan["tiny"] else 248044,
                    "eos_token_ids": [] if plan["tiny"] else [248044, 248046],
                    "export_root": str(root), "graphs": records, "external_weights": plan["external_weights"],
                    "prepared_manifest_sha256": sha256_file(manifest_path),
                    "ge_root": str(ge), "libgraph_sha256": sha256_file(lib / "libgraph.so"),
                    "cxx11_abi": cxx11_abi, "board_memory_fit": "NOT_ESTABLISHED"}
        save_json(root / "manifest.json", manifest)
        return manifest
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": stage,
                                          "fp16_reductions": fp16_reductions,
                                          "fp16_row_mul": fp16_row_mul,
                                          "gdn_indexed_copy": gdn_indexed_copy,
                                          "device_validation": "PENDING", "error": str(error)})
        raise

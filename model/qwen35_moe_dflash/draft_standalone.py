"""Prepare and serialize the complete DFlash draft without constructing target.

Only the target embedding and full-vocabulary head are read. The target decoder,
its quantized weights, and its AIR are not dependencies of this standalone lane.
Synthetic fixtures establish execution structure, not proposal acceptance.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil

from .artifacts import checked_file, files_under, output_path, save_json
from .config import Qwen35DFlashConfig
from .ge_draft import build_draft
from .ge_export import configurations, graph_summary, weight_identity, _logged
from .hashing import sha256_file
from .resources import repository_root


# Version the GE contract separately from weights and the CPU reference.
# Graph changes require a new AIR; reference arithmetic has its own contract.
# Real selected-DDK acceptance remains a separate gate.
CURRENT_DRAFT_EXPORT_CONTRACT = "cpro-fileconstant-relative-int-fc-conv2d-mask-gather-air4b-precision-v6"


def _graph(sequence_length, *, tiny=False, draft_config=None, include_hidden=True):
    raw = configurations(tiny=tiny)[1] if draft_config is None else draft_config
    config = Qwen35DFlashConfig.from_dict(raw)
    if (isinstance(sequence_length, bool) or not isinstance(sequence_length, int)
            or sequence_length < 2 or sequence_length + config.block_size > config.max_position_embeddings
            or sequence_length * config.feature_size * 2 > 2**32 - 1):
        raise ValueError("sequence length exceeds draft graph, RoPE or HIAI capacity")
    graph = build_draft(config, sequence_length)
    if not isinstance(include_hidden, bool):
        raise ValueError("include_hidden must be a boolean")
    if include_hidden:
        hidden = next(node.output for node in graph.nodes if node.output.name == "draft_hidden")
        graph.outputs.append(hidden)
    graph.validate()
    target_keys = {key for key in graph.weights if key.startswith("target.")}
    if target_keys != {"target.embed_tokens.weight", "target.lm_head.weight"}:
        raise ValueError("standalone draft must depend only on target embedding and head")
    return raw, config, graph


def plan_draft(output_dir, sequence_length=64, *, tiny=False, include_hidden=True):
    """Describe the full six-layer graph by default; tiny is a host fixture."""
    raw, config, graph = _graph(sequence_length, tiny=tiny, include_hidden=include_hidden)
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-draft-ge-plan", "status": "PLANNED",
              "tiny": tiny, "sequence_length": sequence_length, "draft_config": raw,
              "include_hidden": include_hidden,
              "draft_layers": config.num_hidden_layers, "draft_block_size": config.block_size,
              "draft_width": config.block_size - 1, "feature_width": config.feature_size,
              "vocab_size": config.vocab_size, "graphs": [graph_summary(graph)],
              "unique_weight_bytes": sum(math.prod(v.shape) * 2 for v in graph.weights.values()),
              "target_dependencies": ["embed_tokens.weight", "lm_head.weight"],
              "target_decoder": "NOT_CONSTRUCTED", "precision": "float16", "quantization": None,
              "state_policy": "prefix-recompute", "device_validation": "NOT_RUN",
              "numerical_validation": "NOT_RUN", "acceptance_validation": "NOT_RUN",
              "limitations": ["Target features are an external input; synthetic features do not validate acceptance.",
                              "FP16 conversion baseline; no whole-model accuracy or speed claim."]}
    save_json(root / "plan.json", result)
    return result


def _source_tensor_identity(reader, key, shape, digest):
    """Bind the hash collected during conversion to the selected source extent."""
    source, offset, count, dtype = reader.tensor(key, shape)
    return {"tensor": key, "shard": source.name, "offset": offset, "bytes": count * (4 if dtype == "F32" else 2),
            "dtype": dtype, "shape": list(shape), "sha256": digest}


def _standalone_checkpoint_metadata(target_path, draft_path):
    """Require the compute configurations; inventory optional packaging metadata.

    Standalone draft needs only two target tensors, so a target index covering
    all 40 layers is not required. Matching JSON metadata does not establish the
    provenance of weight bytes; selected tensor hashes are recorded separately.
    """
    lock = json.loads((repository_root() / "specs/upstream.lock.json").read_text())
    identities = []
    for owner, directory, pin in zip(("target", "draft"), (target_path, draft_path), lock):
        root = Path(directory).resolve()
        config = root / "config.json"
        expected = pin["files"]["config.json"]["sha256"]
        if not config.is_file() or sha256_file(config) != expected:
            raise ValueError(f"{pin['repo']}: config.json differs from locked configuration")
        metadata = {"config.json": {"status": "MATCHED", "sha256": expected,
                                     "required_for_draft": True}}
        # Neither generation settings nor an all-target-layer index changes
        # draft math. A present index is still validated by SafeTensorFiles.
        for name in ("model.safetensors.index.json", "generation_config.json"):
            path = root / name
            if not path.is_file():
                metadata[name] = {"status": "ABSENT_ALLOWED", "required_for_draft": False}
                continue
            digest = sha256_file(path)
            expected = pin["files"].get(name, {}).get("sha256")
            metadata[name] = {"status": ("MATCHED" if digest == expected else
                                          "DIFFERS_FROM_LOCK" if expected else "OBSERVED_UNPINNED"),
                              "sha256": digest, "required_for_draft": False}
        identities.append({"owner": owner, "repo": pin["repo"], "expected_revision": pin["revision"],
                           "revision_verification": "NOT_ESTABLISHED", "metadata": metadata})
    return identities


def prepare_draft(output_dir, sequence_length=64, *, target_path=None, draft_path=None,
                  tiny=False, include_hidden=True, progress=None):
    """Stream only draft tensors plus the target embedding/head into FP16 files."""
    from .ge_cpp import emit_cpp
    from .ge_weights import SafeTensorFiles, checkpoint_key, tiny_weights
    if not tiny and (target_path is None or draft_path is None):
        raise ValueError("full draft needs original draft and target checkpoint directories (embedding/head only)")
    _, _, graph = _graph(sequence_length, tiny=tiny, include_hidden=include_hidden)
    shapes = {key: value.shape for key, value in graph.weights.items()}
    identities = None if tiny else _standalone_checkpoint_metadata(target_path, draft_path)
    readers = {}
    if not tiny:
        required = {owner: {checkpoint_key(key)[1] for key in shapes if checkpoint_key(key)[0] == owner}
                    for owner in ("target", "draft")}
        readers = {owner: SafeTensorFiles(path, allow_indexless_shards=True, required_keys=required[owner])
                   for owner, path in (("target", target_path), ("draft", draft_path))}
        # Reject a missing head or any malformed draft tensor before writing
        # gigabytes of already valid weights into a failed preparation.
        for key, shape in shapes.items():
            owner, source_key = checkpoint_key(key)
            _, _, _, dtype = readers[owner].tensor(source_key, shape)
            if dtype != "BF16":
                raise ValueError(f"{source_key}: original draft/shared weights must be BF16")
    root = output_path(output_dir)
    plan = plan_draft(root, sequence_length, tiny=tiny, include_hidden=include_hidden)
    stage = "prepare-weights"
    try:
        folder = root / "weights"
        folder.mkdir()
        records = {}
        if tiny:
            for number, (key, value) in enumerate(sorted(tiny_weights(shapes).items())):
                path = folder / f"w{number:05d}.bin"
                with path.open("xb") as stream:
                    stream.write(value.tobytes())
                records[key] = {"path": str(path.relative_to(root)), "shape": list(value.shape),
                                "dtype": "float16", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            identity = {"kind": "ge-tiny-numpy-random", "seed": 17, "not_upstream_weights": True}
        else:
            tensors = {"target": [], "draft": []}
            for number, (key, shape) in enumerate(sorted(shapes.items())):
                if progress:
                    progress({"stage": stage, "message": f"Draft weight {number + 1}/{len(shapes)}: {key}"})
                owner, source_key = checkpoint_key(key)
                path = folder / f"w{number:05d}.bin"
                entry = readers[owner].write_fp16(source_key, shape, path)
                source = _source_tensor_identity(readers[owner], source_key, shape, entry["source_sha256"])
                tensors[owner].append(source)
                records[key] = {**entry, "path": str(path.relative_to(root)),
                                "shape": list(shape), "dtype": "float16"}
            identity = {"kind": "original-checkpoint-fp16-draft-conversion",
                        "checkpoints": [{**entry, "selected_tensors": tensors[entry["owner"]],
                                         "discovery": readers[entry["owner"]].discovery}
                                        for entry in identities]}
        stage = "emit-cpp"
        identity["fp16_payload_sha256"] = weight_identity(records)
        paths = {key: str(root / item["path"]) for key, item in records.items()}
        folder = root / "draft"
        folder.mkdir()
        (folder / "draft.cpp").write_text(emit_cpp(graph, paths, external_weights=True, recipe_dir=folder / "recipe"))
        result = {**plan, "artifact_kind": "qwen35-draft-ge-prepared", "status": "PREPARED",
                  "source": identity, "weights": records, "external_weights": True,
                  "prepared_root": str(root),
                  "files": [{key: item[key] for key in ("path", "bytes", "sha256")} for item in records.values()],
                  "construction_files": {"draft": files_under(folder, root)}}
        save_json(root / "prepared.json", result)
        return result
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": stage,
                                          "error": f"{type(error).__name__}: {error}"})
        raise


def export_draft(prepared_dir, output_dir, ge_root, *, cxx="c++", cxx11_abi=0,
                 timeout_seconds=3600, fp16_reductions="matmul", fp16_row_mul="compact", progress=None,
                 trace_names=None, direct_names=None, residual_policy="native"):
    """Build one real GE exporter and serialize one complete draft AIR."""
    from .ge_cpp import emit_cpp
    from .ge_residual import POLICIES, EXPORT_SUFFIX
    if residual_policy not in POLICIES:
        raise ValueError("draft residual policy must be native or cpu-fp32-add")
    if fp16_reductions not in {"native", "matmul"} or fp16_row_mul not in {"native", "compact"}:
        raise ValueError("invalid FP16 lowering policy")
    if cxx11_abi not in (0, 1) or timeout_seconds <= 0:
        raise ValueError("invalid C++ ABI or timeout")
    if trace_names is not None and direct_names is not None:
        raise ValueError("packed trace and direct outputs are mutually exclusive")
    prepared = Path(prepared_dir).resolve()
    manifest_path = prepared / "prepared.json"
    plan = json.loads(manifest_path.read_text())
    if plan.get("artifact_kind") != "qwen35-draft-ge-prepared" or plan.get("status") != "PREPARED":
        raise ValueError("expected successful standalone draft preparation")
    if plan.get("external_weights") is not True:
        raise ValueError("standalone draft requires external weights")
    expected_config = configurations(tiny=plan["tiny"])[1]
    if plan["draft_config"] != expected_config:
        raise ValueError("prepared draft configuration differs from locked topology")
    _, config, graph = _graph(plan["sequence_length"], tiny=plan["tiny"], draft_config=plan["draft_config"],
                             include_hidden=plan["include_hidden"])
    trace_spec = None
    direct_spec = None
    if trace_names is not None:
        from .draft_ge_trace import instrument_draft_trace
        graph, trace_spec = instrument_draft_trace(graph, names=trace_names)
    if direct_names is not None:
        from .draft_direct_trace import instrument_draft_direct_outputs
        graph, direct_spec = instrument_draft_direct_outputs(graph, names=direct_names)
    suffix = ("-layer-trace-v1" if trace_spec is not None else
              "-direct-outputs-v1" if direct_spec is not None else "")
    export_contract = CURRENT_DRAFT_EXPORT_CONTRACT + (EXPORT_SUFFIX if residual_policy != "native" else "") + suffix
    if set(plan["weights"]) != set(graph.weights):
        raise ValueError("prepared weight set differs from complete draft graph")
    if plan["source"].get("fp16_payload_sha256") != weight_identity(plan["weights"]):
        raise ValueError("prepared source identity differs from weight records")
    sources = {}
    for key, value in graph.weights.items():
        item = plan["weights"][key]
        rel = Path(item["path"])
        if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "weights" or ".." in rel.parts:
            raise ValueError(f"invalid prepared weight path: {key}")
        if tuple(item["shape"]) != value.shape or item["dtype"] != "float16" or item["bytes"] != math.prod(value.shape) * 2:
            raise ValueError(f"invalid prepared weight descriptor: {key}")
        sources[key] = checked_file(prepared, item)
    ge = Path(ge_root).expanduser().resolve()
    include, lib = ge / "include", ge / "lib64"
    for path in (include / "graph/graph.h", include / "graph/operator.h", lib / "libgraph.so"):
        if not path.is_file():
            raise FileNotFoundError(f"matching real GE headers/libraries required: {path}")
    compiler = shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError(f"C++ compiler not found: {cxx}")
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    stage = "copy-weights"
    try:
        (root / "weights").mkdir()
        paths = {}
        for key, source in sources.items():
            destination = root / plan["weights"][key]["path"]
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            paths[key] = str(destination)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(lib), env.get("LD_LIBRARY_PATH", ""))))
        folder = root / "draft"
        folder.mkdir()
        source = folder / "draft.cpp"
        reduction_report, row_mul_report, projection_report, residual_report = {}, {}, {}, {}
        stage = "draft-lowering"
        source.write_text(emit_cpp(graph, paths, external_weights=True, recipe_dir=folder / "recipe",
                                   fp16_reductions=fp16_reductions, reduction_report=reduction_report,
                                   fp16_row_mul=fp16_row_mul, row_mul_report=row_mul_report,
                                   file_constant_policy="cpro-relative-int", file_constant_root=root,
                                   draft_fc_policy="conv2d", draft_fc_report=projection_report,
                                   draft_residual_policy=residual_policy, draft_residual_report=residual_report))
        if len(residual_report["replaced"]) != (config.num_hidden_layers * 2 if residual_policy != "native" else 0):
            raise ValueError("draft residual lowering did not match the complete locked topology")
        expected_replacements = 0 if plan["tiny"] else 1
        if len(projection_report["replaced"]) != expected_replacements:
            raise ValueError("draft.fc Conv2D lowering did not match the locked topology")
        if expected_replacements and progress:
            progress({"stage": stage, "message": "draft.fc K=16384: one full-dot Conv2D with unchanged FP16 weights; target support unverified"})
        save_json(folder / "reduction-lowering.json", reduction_report)
        save_json(folder / "row-mul-lowering.json", row_mul_report)
        save_json(folder / "projection-lowering.json", projection_report)
        if residual_policy != "native":
            save_json(folder / "residual-lowering.json", residual_report)
            if progress:
                progress({"stage": stage, "message": "Draft residuals: explicit FP32 Adds with original FP16 result boundaries; compiled CPUCL ownership and numerical validation pending"})
        construction = files_under(folder, root)
        binary, air = folder / "draft_exporter", root / "draft.air"
        command = [compiler, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}", "-I", str(include),
                   str(source), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
        if (lib / "libgraph_base.so").is_file():
            command.append("-lgraph_base")
        command.extend(["-o", str(binary)])
        save_json(folder / "build-command.json", {"argv": command})
        stage = "draft-build"
        if progress:
            progress({"stage": stage, "message": f"Building draft GE exporter; log: {folder / 'build.log'}"})
        _logged(command, folder=folder, env=env, log=folder / "build.log", timeout=timeout_seconds)
        if not binary.is_file() or not binary.stat().st_size:
            raise RuntimeError("compiler did not produce nonempty draft exporter")
        command = [str(binary), str(air)]
        save_json(folder / "export-command.json", {"argv": command, "cwd": str(root),
                                                   "export_contract": export_contract})
        stage = "draft-serialize"
        if progress:
            progress({"stage": stage, "message": f"Serializing full draft AIR; log: {folder / 'export.log'}"})
        # Both host FileConstant validation and OMG resolve weights relative to
        # the AIR/compiler working directory. The C++ source stays in draft/.
        _logged(command, folder=root, env=env, log=folder / "export.log", timeout=timeout_seconds)
        if not air.is_file() or air.is_symlink() or not air.stat().st_size:
            raise RuntimeError("exporter did not produce a nonempty regular draft AIR")
        for item in construction:
            checked_file(root, item)
        for item in plan["weights"].values():
            checked_file(root, item)
        files = [{"path": str(air.relative_to(root)), "bytes": air.stat().st_size, "sha256": sha256_file(air)}]
        files += [{k: plan["weights"][key][k] for k in ("path", "bytes", "sha256")} for key in graph.weights]
        record = {"name": "draft", "air": str(air.relative_to(root)),
                  "inputs": [v.name for v in graph.inputs], "outputs": [v.name for v in graph.outputs],
                  "input_shapes": [list(v.shape) for v in graph.inputs],
                  "input_dtypes": ["torch." + v.dtype for v in graph.inputs],
                  "output_shapes": [list(v.shape) for v in graph.outputs],
                  "output_dtypes": ["torch." + v.dtype for v in graph.outputs], "files": files,
                  "builder_sha256": sha256_file(source), "construction_files": construction,
                  "reduction_lowering": {"mode": reduction_report["mode"], "counts": reduction_report["counts"]},
                  "row_mul_lowering": {"mode": row_mul_report["mode"], "counts": row_mul_report["counts"]},
                  "projection_lowering": projection_report, "residual_lowering": residual_report}
        manifest = {"schema_version": 1, "artifact_kind": "qwen35-cpro-draft-air", "status": "EXPORTED",
                    "export_contract": export_contract,
                    "rms_normalization": "FP32_NORMALIZE_CAST_FP16_THEN_DIRECT_SCALE",
                    "attention_precision": "FP16_MATMUL_FP32_SCORES_SOFTMAX_FP16_PROBABILITY",
                    "rope_precision": "FP32_ANGLES_COS_SIN_CAST_FP16",
                    "file_constant_policy": "cpro-relative-int", "file_constant_path_base": "air-parent",
                    "file_constant_dtype_attr": "int", "file_constant_target_validation": "NOT_RUN",
                    "pipeline": ["ge-cpp", "air", "omg", "omc", "hiai"], "export_backend": "direct-ge",
                    "device_validation": "NOT_RUN", "numerical_validation": "NOT_RUN",
                    "acceptance_validation": "NOT_RUN", "sequence_length": plan["sequence_length"],
                    "draft_layers": config.num_hidden_layers, "draft_block_size": config.block_size,
                    "draft_width": config.block_size - 1, "feature_width": config.feature_size,
                    "vocab_size": config.vocab_size, "precision": "torch.float16", "quantization": None,
                    "state_policy": "prefix-recompute", "target_decoder": "NOT_CONSTRUCTED",
                    "tiny": plan["tiny"], "include_hidden": plan["include_hidden"],
                    "source": plan["source"], "export_root": str(root), "graphs": [record],
                    "external_weights": True, "prepared_manifest_sha256": sha256_file(manifest_path),
                    "ge_root": str(ge), "libgraph_sha256": sha256_file(lib / "libgraph.so"),
                    "cxx11_abi": cxx11_abi, "fp16_reductions": fp16_reductions, "fp16_row_mul": fp16_row_mul,
                    "residual_policy": residual_policy}
        if trace_spec is not None:
            manifest.update(trace_spec=trace_spec, diagnostic_only=True)
        if direct_spec is not None:
            manifest.update(direct_spec=direct_spec, diagnostic_only=True,
                            direct_outputs="ORIGINAL_VALUES_NATIVE_DTYPE_NO_OBSERVER_NODES")
        save_json(root / "manifest.json", manifest)
        return manifest
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": stage,
                                          "export_contract": export_contract,
                                          "device_validation": "NOT_RUN", "error": str(error)})
        raise

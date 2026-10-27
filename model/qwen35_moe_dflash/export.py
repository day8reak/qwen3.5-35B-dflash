"""Production export: PyTorch -> TorchAir AIR (including external weights)."""
import importlib
import importlib.metadata
import json
import operator
from pathlib import Path
import torch
from .artifacts import output_path, files_under, save_json, working_directory
from .precision import prepare_fp16_pair
from .diagnostics import format_error
from .backends import require_unmodified_air_backend


def require_torchair():
    # Fail before opening/allocating the 35B checkpoint.
    require_unmodified_air_backend()
    try:
        # Also loads the bundled TorchAir when PyTorch backend autoload is off.
        importlib.import_module("torch_npu")
        module = importlib.import_module("torchair")
    except Exception as error:
        raise RuntimeError("AIR export requires a matching TorchAir/torch_npu/CANN environment; "
                           f"dependency import failed: {format_error(error)}") from error
    if not callable(getattr(module, "dynamo_export", None)):
        raise RuntimeError("installed TorchAir has no dynamo_export API")
    return module


def export_air(target, draft, output_dir, sequence_length, *, torchair_module=None,
               source_identity=None, dispatch="static", quantization=None):
    """Export two stateless batch-one graphs under the explicit FP16 policy.

    static dispatch avoids data-dependent NonZero lengths. It computes all
    experts and is a correctness baseline, not a sparse-performance claim.
    """
    require_unmodified_air_backend()
    torchair = torchair_module if torchair_module is not None else require_torchair()
    if sequence_length < 2:
        raise ValueError("sequence_length must exceed one")
    if sequence_length * target.config.feature_size * 2 > 2**32 - 1:
        raise ValueError("feature tensor exceeds HIAI uint32 buffer capacity")
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    try:
        target, draft, precision_audit = prepare_fp16_pair(
            target, draft, sequence_length, dispatch=dispatch, quantization=quantization is not None)
        if quantization is not None:
            ops = precision_audit["target"]["quantized_matmuls"]
            if len(ops) != quantization["matrix_count"]:
                raise ValueError("quantized AIR needs one native packed matmul per planned matrix; "
                                 "fake-quantized/dequantized dense graphs are not accepted")
            if any(o["input_dtypes"][0] not in ("torch.float16", "torch.int16") or
                   o["input_dtypes"][1] not in ("torch.int8", "torch.uint8", "torch.int32", "torch.int4")
                   for o in ops):
                raise ValueError("W4/A16 requires 16-bit activations and packed integer weights")
            packed_bytes = sum(o["weight_bytes"] for o in ops)
            if packed_bytes != (quantization["quantized_parameters"] + 1)//2:
                raise ValueError("native quantized graph weight bytes do not match packed INT4; "
                                 "review the internal DDK storage layout before permitting export")
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": "precision-and-packing-audit",
                                        "error": f"{type(error).__name__}: {error}"})
        raise
    device = next(target.parameters()).device
    ids = torch.zeros((1, sequence_length), dtype=torch.int32, device=device)
    mask = torch.zeros_like(ids)
    ids[0, :2] = torch.tensor([1, 2], device=device); mask[0, :2] = 1
    features = torch.zeros((1, sequence_length, target.config.feature_size), dtype=torch.float16, device=device)
    defs = [("target", target, (ids, mask), ["input_ids", "attention_mask"],
             ["target_top1", "target_features"]),
            ("draft", draft, (features, ids, mask), ["target_features", "input_ids", "attention_mask"],
             ["draft_top1"])]
    records = []
    manifest = {
        "schema_version": 2, "abi_version": 2, "artifact_kind": "qwen35-cpro-air-pair",
        "pipeline": ["pytorch", "air", "omg", "omc", "hiai"],
        "status": "EXPORTING", "device_validation": "PENDING", "runtime": "hiai",
        "sequence_length": sequence_length, "draft_width": target.config.block_size - 1,
        "feature_width": target.config.feature_size, "vocab_size": target.config.vocab_size,
        "precision": "torch.float16", "feature_dtype": "torch.float16",
        "precision_audit": precision_audit, "quantization": quantization, "state_policy": "recompute",
        "expert_dispatch": dispatch, "source": source_identity,
        "pad_token_id": 248044 if target.config.vocab_size == 248320 else 0,
        # Random tiny fixtures have no EOS, matching the CPU demo.
        "eos_token_ids": [248044, 248046] if target.config.vocab_size == 248320 else [],
        "export_root": str(root), "graphs": records,
        "environment": {"torch": torch.__version__, "torchair": getattr(torchair, "__version__", "unknown"),
                        "transformers": importlib.metadata.version("transformers")},
    }
    try:
        for name, module, args, inputs, outputs in defs:
            folder = root / name; folder.mkdir()
            with working_directory(folder), torch.inference_mode():
                torchair.dynamo_export(*args, model=module.eval(), export_path=str(folder),
                                      export_name=name, dynamic=False)
            air_files = list(folder.rglob("*.air"))
            if len(air_files) != 1 or not air_files[0].stat().st_size:
                raise RuntimeError(f"{name}: expected exactly one nonempty AIR file")
            records.append({"name": name, "air": str(air_files[0].relative_to(root)),
                            "inputs": inputs, "outputs": outputs,
                            "input_shapes": [list(a.shape) for a in args],
                            "input_dtypes": [str(a.dtype) for a in args],
                            "output_shapes": ([[1, sequence_length], [1, sequence_length, target.config.feature_size]]
                                              if name == "target" else [[1, target.config.block_size - 1]]),
                            "output_dtypes": (["torch.int32", "torch.float16"] if name == "target" else ["torch.int32"]),
                            "files": files_under(folder, root)})
        manifest["status"] = "EXPORTED"
        save_json(root / "manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_json(root / "failure.json", manifest)
        raise
    return manifest


def _hierarchical_pytorch_graphs(target, draft, sequence_length):
    """Build executable FX graphs of the existing export forward definitions.

    Decoder layers remain real PyTorch leaf modules, retaining their complete
    attention/GDN/MoE implementations and parameter banks. This is an explicit
    hierarchy, not an ATen lowering. The connections below match
    ExportTargetGraph.forward and DraftGraph.forward; changed-input equivalence
    is tested against those definitions using actual tiny weights.
    """
    from torch import nn, fx
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence_length must be an integer greater than one")
    device = next(target.parameters()).device
    dtype = next(target.parameters()).dtype
    c = target.config
    target_root = nn.Module()
    target_root.language_model = target.language_model
    target_root.lm_head = target.lm_head
    rows = torch.arange(sequence_length, device=device)
    positions = rows.view(1, -1)
    target_root.register_buffer("text_positions", positions)
    target_root.register_buffer("rope_positions", positions.unsqueeze(0).expand(3, -1, -1))
    target_root.register_buffer("causal_visible", (rows[:, None] >= rows[None, :]).view(1, 1, sequence_length, sequence_length))
    target_root.register_buffer("mask_zero", torch.zeros((), dtype=dtype, device=device))
    target_root.register_buffer("mask_sentinel", torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device))
    g = fx.Graph()
    ids, mask = g.placeholder("input_ids"), g.placeholder("attention_mask")
    hidden = g.call_module("language_model.embed_tokens", (g.call_method("long", (ids,)),))
    rotary = g.call_module("language_model.rotary_emb", (hidden, g.get_attr("rope_positions")))
    valid = g.call_method("bool", (mask,))
    key_valid = g.call_function(operator.getitem, (valid, (slice(None), None, None, slice(None))))
    visible = g.call_function(operator.and_, (g.get_attr("causal_visible"), key_valid))
    full_mask = g.call_function(torch.where, (visible, g.get_attr("mask_zero"), g.get_attr("mask_sentinel")))
    captures = {}
    for index, layer in enumerate(target.language_model.layers):
        hidden = g.call_module(f"language_model.layers.{index}", (hidden,), {
            "position_embeddings": rotary,
            "attention_mask": mask if layer.block_type == "linear_attention" else full_mask,
            "position_ids": g.get_attr("text_positions"), "past_key_values": None, "use_cache": False})
        if index in c.target_layer_ids:
            captures[index] = hidden
    hidden = g.call_module("language_model.norm", (hidden,))
    if len(target.language_model.layers) - 1 in captures:
        captures[len(target.language_model.layers) - 1] = hidden
    features = g.call_function(torch.cat, (tuple(captures[i] for i in c.target_layer_ids),), {"dim": -1})
    top1 = []
    for start in range(0, sequence_length, target.head_chunk_rows):
        piece = g.call_function(operator.getitem, (hidden, (slice(None), slice(start, start + target.head_chunk_rows))))
        logits = g.call_module("lm_head", (piece,))
        top1.append(g.call_method("to", (g.call_method("argmax", (logits, -1)), torch.int32)))
    g.output((g.call_function(torch.cat, (tuple(top1),), {"dim": 1}), g.call_method("float", (features,))))
    g.lint()
    target_fx = fx.GraphModule(target_root, g, "FullTargetHierarchy")

    draft_root = nn.Module()
    draft_root.draft, draft_root.embedding, draft_root.lm_head = draft.draft, draft.embedding, draft.lm_head
    draft_root.register_buffer("context_rows", rows.view(1, -1))
    draft_root.register_buffer("block_rows", torch.arange(c.block_size, device=device).view(1, -1))
    draft_root.register_buffer("mask_tokens", torch.full((1, c.block_size - 1), c.mask_token_id, dtype=torch.long, device=device))
    g = fx.Graph()
    features, ids, mask = (g.placeholder(name) for name in ("target_features", "input_ids", "attention_mask"))
    valid = g.call_method("bool", (mask,))
    length = g.call_method("sum", (g.call_method("long", (valid,)), -1), {"keepdim": True})
    anchor = g.call_function(operator.sub, (length, 1))
    anchor_ids = g.call_method("gather", (g.call_method("long", (ids,)), 1, anchor))
    block = g.call_function(torch.cat, ((anchor_ids, g.get_attr("mask_tokens")),), {"dim": -1})
    context_valid = g.call_function(operator.and_, (valid, g.call_function(operator.lt, (g.get_attr("context_rows"), anchor))))
    context_positions = g.call_method("clamp_min", (
        g.call_function(operator.sub, (g.call_method("cumsum", (g.call_method("long", (context_valid,)), -1)), 1)), 0))
    block_positions = g.call_function(operator.add, (anchor, g.get_attr("block_rows")))
    positions = g.call_function(torch.cat, ((context_positions, block_positions),), {"dim": -1})
    noise = g.call_function(operator.mul, (g.call_module("embedding", (block,)), c.input_embedding_scale))
    context = g.call_module("draft.hidden_norm", (g.call_module("draft.fc", (g.call_method("to", (features, dtype)),)),))
    rotary = g.call_module("draft.rotary", (positions, dtype))
    cosine, sine = (g.call_function(operator.getitem, (rotary, index)) for index in (0, 1))
    hidden = noise
    for index in range(len(draft.draft.layers)):
        hidden = g.call_module(f"draft.layers.{index}", (hidden, context, cosine, sine, None, positions, context_valid))
    hidden = g.call_module("draft.norm", (hidden,))
    hidden = g.call_function(operator.getitem, (hidden, (slice(None), slice(1, None), slice(None))))
    logits = g.call_module("lm_head", (hidden,))
    g.output(g.call_method("to", (g.call_method("argmax", (logits, -1)), torch.int32)))
    g.lint()
    return target_fx, fx.GraphModule(draft_root, g, "FullDraftHierarchy")


def build_full_pytorch(output_dir, sequence_length=128):
    """Construct the complete locked 35B target and six-layer draft on meta.

    Produces actual executable hierarchical FX GraphModules with meta tensors,
    readable forward code, graph edges, and every parameter descriptor. No
    checkpoint is opened and no tensor values or AIR artifact are fabricated.
    Real full-checkpoint AIR export remains the existing export-air command.
    """
    from .quantization import meta_graphs
    from .export_moe import prepare_export_target
    from .resources import repository_root
    from .hashing import sha256_file
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence_length must be an integer greater than one")
    target, draft = meta_graphs()
    if sequence_length > target.config.max_position_embeddings:
        raise ValueError("sequence length exceeds the locked model context capacity")
    target = prepare_export_target(target, mode="static")
    models = _hierarchical_pytorch_graphs(target, draft, sequence_length)
    parameters, shared, aliases = {}, {}, []
    for name, model in zip(("target", "draft"), models):
        parameters[name] = {}
        for key, tensor in model.named_parameters():
            if tensor.device.type != "meta":
                raise ValueError("full structure construction unexpectedly allocated a real parameter")
            alias = shared.setdefault(id(tensor), f"parameter_{len(shared):05d}")
            parameters[name][key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                     "device": "meta", "elements": tensor.numel(),
                                     "logical_bytes": tensor.numel() * tensor.element_size(), "storage_id": alias}
            aliases.append((alias, tensor.numel(), tensor.numel() * tensor.element_size()))
        if any(t.device.type != "meta" for t in model.buffers()):
            raise ValueError("full structure construction unexpectedly allocated a real model buffer")
    # Check the complete target parameter contract against the independent fork.
    from .fused.model import required_weight_shapes
    expected = required_weight_shapes(target.language_model.config.to_dict())
    actual = {"target." + key.removeprefix("language_model."): tuple(value["shape"])
              for key, value in parameters["target"].items()}
    if actual != expected:
        raise ValueError("full PyTorch FX graph does not retain the complete target parameter contract")
    if len(target.language_model.layers) != 40 or len(draft.draft.layers) != 6:
        raise ValueError("build-full-pytorch requires the locked target40/draft6 architecture")
    if models[0].lm_head is not models[1].lm_head or models[0].language_model.embed_tokens is not models[1].embedding:
        raise ValueError("draft must share the actual target embedding and LM head modules")
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    try:
        from torch.fx import Node
        def encode(value):
            if isinstance(value, Node):
                return {"node": value.name}
            if isinstance(value, slice):
                return {"slice": [value.start, value.stop, value.step]}
            if isinstance(value, (tuple, list)):
                return [encode(item) for item in value]
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (torch.dtype, torch.device)):
                return str(value)
            return value
        graphs = []
        for name, model in zip(("target", "draft"), models):
            folder = root / name
            folder.mkdir()
            nodes = [{"name": node.name, "op": node.op,
                      "target": (f"{node.target.__module__}.{node.target.__name__}" if callable(node.target) else node.target),
                      "args": encode(node.args), "kwargs": encode(node.kwargs)} for node in model.graph.nodes]
            save_json(folder / "graph.json", {"representation": "torch.fx.GraphModule", "level": "decoder-layer-hierarchy", "nodes": nodes})
            (folder / "forward.py").write_text(model.code)
            (folder / "graph.txt").write_text(str(model.graph) + "\n")
            (folder / "modules.txt").write_text(str(model) + "\n")
            layers = []
            for index in range(40 if name == "target" else 6):
                path = f"language_model.layers.{index}" if name == "target" else f"draft.layers.{index}"
                layer = model.get_submodule(path)
                layers.append({"index": index, "module": path, "class": type(layer).__name__,
                               "layer_type": target.language_model.config.layer_types[index] if name == "target" else draft.draft.config.layer_types[index],
                               "parameter_elements": sum(t.numel() for t in layer.parameters()),
                               "submodules": [{"name": key, "class": type(child).__name__} for key, child in layer.named_modules() if key]})
            save_json(folder / "layers.json", layers)
            graphs.append({"name": name, "node_count": len(nodes), "decoder_layers": len(layers),
                           "parameter_tensors": len(parameters[name]),
                           "parameter_elements": sum(item["elements"] for item in parameters[name].values()),
                           "files": files_under(folder, root)})
        save_json(root / "parameters.json", parameters)
        # A single save retains shared module identities. All tensors are meta;
        # this archive contains no learned values and is not an inference model.
        torch.save({"target": models[0], "draft": models[1]}, root / "models.meta.fx.pt")
        unique = {alias: (elements, size) for alias, elements, size in aliases}
        specs = repository_root() / "specs"
        manifest = {"schema_version": 1, "artifact_kind": "qwen35-full-pytorch-structure", "status": "META_STRUCTURE_READY",
                    "tiny": False, "sequence_length": sequence_length, "target_layers": 40, "draft_layers": 6,
                    "gdn_layers": 30, "full_attention_layers": 10, "experts_per_layer": 256, "experts_per_token": 8,
                    "feature_layers": list(target.config.target_layer_ids), "feature_width": target.config.feature_size,
                    "draft_block_size": target.config.block_size, "parameter_dtype": "torch.float16",
                    "unique_parameter_elements": sum(item[0] for item in unique.values()),
                    "unique_logical_parameter_bytes": sum(item[1] for item in unique.values()),
                    "checkpoint_loaded": False, "tensor_value_execution": "NOT_RUN", "air_export": "NOT_RUN",
                    "device_execution": "NOT_RUN", "aten_lowering": "NOT_RUN",
                    "graph_scope": "Executable hierarchical FX connections; each decoder retains its real PyTorch submodules. Decoder internals are not expanded into ATen nodes.",
                    "arithmetic_policy": "Existing upstream/export PyTorch implementations; only parameter storage is FP16. The production FP16 lowering pass has not run.",
                    "state_policy": "recompute", "stateful_torchair_export": "NOT_IMPLEMENTED",
                    "shared_modules": ["target.language_model.embed_tokens=draft.embedding", "target.lm_head=draft.lm_head"],
                    "config_sha256": {name: sha256_file(specs / name) for name in ("target-config.json", "draft-config.json")},
                    "graphs": graphs, "files": files_under(root, root)}
        save_json(root / "manifest.json", manifest)
        return manifest
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": "full-pytorch-structure",
                                          "error": f"{type(error).__name__}: {error}"})
        raise

"""Explicit FP16 graph policy, including upstream FP32 normalization/GDN.

This is a numerical change requested by the deployment contract. It is not
an exact BF16/FP32 optimization. Keep the unmodified upstream reference for
quality comparisons. Hardware accumulators inside an op remain compiler owned.
"""
import torch
from torch.utils._pytree import tree_map, tree_leaves


def _quant_state(name):
    return any(part in name.split(".") for part in ("quant_op", "npu_simulator"))


def _quant_node(node):
    if str(node.target).startswith("qwen35.dopt_reference_linear"):
        return True
    if str(node.target).startswith("npu.") and "quant" in str(node.target):
        return True
    for _, module_type in node.meta.get("nn_module_stack", {}).values():
        name = str(module_type)
        if "dopt" in name and any(c in name for c in ("QLinear", "QClass", "Quantizer", "Grid")):
            return True
    return False


def floating_dtype_audit(program, *, quantization=False):
    """Inspect inferred FX tensor dtypes, not only parameter storage."""
    violations = []
    quantization_exceptions = []
    counts = {}
    for node in program.graph.nodes:
        for value in tree_leaves(node.meta.get("val")):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                key = str(value.dtype)
                counts[key] = counts.get(key, 0) + 1
                if value.dtype != torch.float16:
                    item = {"node": node.name, "op": str(node.target), "dtype": key}
                    (quantization_exceptions if quantization and _quant_node(node) else violations).append(item)
    for name, tensor in program.state_dict.items():
        if tensor.is_floating_point() and tensor.dtype != torch.float16:
            item = {"state": name, "dtype": str(tensor.dtype)}
            (quantization_exceptions if quantization and _quant_state(name) else violations).append(item)
    quantized = []
    for node in program.graph.nodes:
        op = str(node.target)
        if "npu" in op and "quant" in op and "matmul" in op:
            tensors = [arg.meta.get("val") for arg in node.args[:2] if isinstance(arg, torch.fx.Node)]
            quantized.append({"op": op, "input_dtypes": [str(t.dtype) for t in tensors],
                              "weight_bytes": tensors[1].numel() * tensors[1].element_size()
                              if len(tensors) == 2 else None})
    return {"policy": "fp16-except-vendor-quantization" if quantization else "fp16",
            "status": "PASS" if not violations else "FAIL", "floating_node_values": counts,
            "quantization_exceptions": quantization_exceptions,
            "violations": violations, "quantized_matmuls": quantized}


def lower_fp16(module, inputs, *, quantization=False):
    """Capture, replace explicit floating dtype requests, then re-infer/check.

    No graph output may silently remain FP32/BF16. Integer masks/indices and
    packed quantized data are untouched. Inputs must already obey the ABI.
    Weight-only quantization must be inserted before this pass.
    """
    def cast_state(owner):
        for name, tensor in [*owner.named_parameters(), *owner.named_buffers()]:
            if tensor.is_floating_point() and not (quantization and _quant_state(name)):
                parent_name, _, attribute = name.rpartition(".")
                parent = owner.get_submodule(parent_name) if parent_name else owner
                converted = tensor.to(torch.float16)
                setattr(parent, attribute, torch.nn.Parameter(converted, requires_grad=tensor.requires_grad)
                        if isinstance(tensor, torch.nn.Parameter) else converted)
    cast_state(module)
    with torch.inference_mode():
        captured = torch.export.export(module, tuple(inputs), strict=True)
    graph = captured.module()
    cast_state(graph)
    rewritten = 0

    def half_dtype(value):
        nonlocal rewritten
        if isinstance(value, torch.dtype) and value in (torch.float32, torch.float64, torch.bfloat16):
            rewritten += 1
            return torch.float16
        return value

    for node in graph.graph.nodes:
        if quantization and _quant_node(node):
            continue
        node.args = tree_map(half_dtype, node.args)
        node.kwargs = tree_map(half_dtype, dict(node.kwargs))
        # A finite FP32 masking sentinel cannot be constructed as FP16.
        # Preserve the masking intent without clipping model activations.
        if node.target in (torch.ops.aten.full.default, torch.ops.aten.full_like.default):
            args = list(node.args)
            if (len(args) > 1 and isinstance(args[1], float)
                    and args[1] == torch.finfo(torch.float32).min):
                args[1] = torch.finfo(torch.float16).min
                node.args = tuple(args)
    graph.graph.lint()
    graph.recompile()
    # Re-export invalidates stale pre-rewrite tensor metadata and proves the
    # lowered graph can be captured again by the AIR frontend.
    with torch.inference_mode():
        verified = torch.export.export(graph, tuple(inputs), strict=True)
    audit = floating_dtype_audit(verified, quantization=quantization)
    audit["explicit_dtype_rewrites"] = rewritten
    if audit["status"] != "PASS":
        raise ValueError(f"FP16 lowering left non-FP16 computation: {audit['violations'][:8]}")
    result = verified.module()
    # Runtime metadata is deliberately outside the tensor graph.
    result.config = module.config
    return result, audit


def prepare_fp16_pair(target, draft, sequence_length, *, dispatch="static", quantization=False):
    from qwen35_moe_dflash.toolchain.export_moe import prepare_export_target
    device = next(target.parameters()).device
    ids = torch.zeros((1, sequence_length), dtype=torch.int32, device=device)
    mask = torch.zeros_like(ids)
    ids[0, :2] = torch.tensor([1, 2], device=device)
    mask[0, :2] = 1
    features = torch.zeros((1, sequence_length, target.config.feature_size),
                           dtype=torch.float16, device=device)
    target = prepare_export_target(target, mode=dispatch)
    # DraftGraph uses config through its inner model.
    draft.config = draft.draft.config
    target, target_audit = lower_fp16(target, (ids, mask), quantization=quantization)
    draft, draft_audit = lower_fp16(draft, (features, ids, mask))
    return target, draft, {"target": target_audit, "draft": draft_audit}

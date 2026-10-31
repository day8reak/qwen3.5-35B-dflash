"""DOPT W4/A16 candidate; no home-grown packed NPU format.

Calibration is offline and may use FP32 statistics. Deployment activations
outside quantized regions are FP16. Simulation checkpoints are not compressed AIR.
"""
import importlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

from qwen35_moe_dflash.common.artifacts import checked_file, files_under, output_path, save_json, working_directory
from qwen35_moe_dflash.reference.weights import sha256_file


class QuantizableExpert(nn.Module):
    def __init__(self, gate_up, down, activation):
        super().__init__()
        # Parameter views share the checkpoint storage. Do not allocate another
        # 35B copy while exposing fused expert banks to DOPT's Linear visitor.
        self.gate_up_proj = nn.Linear(gate_up.shape[1], gate_up.shape[0], bias=False, device="meta")
        self.down_proj = nn.Linear(down.shape[1], down.shape[0], bias=False, device="meta")
        self.gate_up_proj.weight = nn.Parameter(gate_up, requires_grad=False)
        self.down_proj.weight = nn.Parameter(down, requires_grad=False)
        self.activation = activation

    def forward(self, hidden):
        gate, up = self.gate_up_proj(hidden).chunk(2, dim=-1)
        return self.down_proj(self.activation(gate) * up)


class QuantizableExperts(nn.Module):
    """Expose every expert matrix; preserve dynamic top-k weights in the graph."""
    export_dispatch = "static"

    def __init__(self, original):
        super().__init__()
        self.layers = nn.ModuleList([
            QuantizableExpert(original.gate_up_proj[i], original.down_proj[i], original.act_fn)
            for i in range(original.num_experts)
        ])

    def forward(self, hidden, selected, weights):
        result = torch.zeros_like(hidden)
        for index, expert in enumerate(self.layers):
            route = selected == index
            weight = torch.where(route, weights, torch.zeros_like(weights)).sum(-1)
            contribution = (expert(hidden) * weight[:, None]).to(hidden.dtype)
            result = result + torch.where(route.any(-1, keepdim=True), contribution,
                                          torch.zeros_like(contribution))
        return result


def expose_experts(target):
    count = 0
    for layer in target.language_model.layers:
        if hasattr(layer.mlp, "experts"):
            original = layer.mlp.experts
            if not isinstance(original, QuantizableExperts):
                layer.mlp.experts = QuantizableExperts(original)
            count += len(layer.mlp.experts.layers)
    if not count:
        raise ValueError("no Qwen3.5 MoE experts found")
    return count


def meta_graphs():
    """Count the real model architecture without allocating/downloading weights."""
    from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
    from qwen35_moe_dflash.config import Qwen35DFlashConfig
    from qwen35_moe_dflash.reference.graphs import TargetGraph, DraftGraph
    from qwen35_moe_dflash.model import DFlashDraftModel
    from qwen35_moe_dflash.common.resources import repository_root
    specs = repository_root() / "specs"
    config = Qwen3_5MoeTextConfig(**json.loads((specs / "target-config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    draft_config = Qwen35DFlashConfig.from_dict(json.loads((specs / "draft-config.json").read_text()))
    with torch.device("meta"):
        owner = Qwen3_5MoeForCausalLM(config).half().eval()
        draft = DFlashDraftModel(draft_config, dtype=torch.float16).eval()
    return (TargetGraph(owner.model, owner.lm_head, draft_config).eval(),
            DraftGraph(draft, owner.model.embed_tokens, owner.lm_head).eval())


def make_plan(target, draft, *, group_size=128):
    if group_size not in (128, 256):
        raise ValueError("blue DOPT documented group sizes are 128 and 256")
    expert_count = expose_experts(target)
    matrices = []
    for name, layer in target.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        # Preserve vocabulary projections, routing and small GDN gate predictors.
        if name == "lm_head" or name.endswith((".shared_expert_gate", ".in_proj_a", ".in_proj_b")):
            continue
        rows, columns = layer.weight.shape
        if columns % group_size:
            raise ValueError(f"{name}: K={columns} must be divisible by group_size={group_size}")
        matrices.append({"name": name, "shape": [rows, columns], "parameters": rows * columns,
                         "groups": rows * (columns // group_size),
                         "expert": ".mlp.experts.layers." in name})
    expert_matrices = sum(m["expert"] for m in matrices)
    if expert_matrices != expert_count * 2:
        raise ValueError("DOPT plan omitted one or more fused gate/up or down expert matrices")
    target_count = sum(p.numel() for p in target.parameters())
    draft_count = sum(p.numel() for p in draft.parameters())
    quantized = sum(m["parameters"] for m in matrices)
    groups = sum(m["groups"] for m in matrices)
    # Scale and offset use FP16 in the observed blue GroupMinMaxGridWeight path.
    # Storage in OMC may include additional layout/alignment or LUT parameters.
    packed_bytes = (quantized + 1) // 2
    affine_bytes = groups * 4
    retained_bytes = (target_count - quantized) * 2
    draft_bytes = draft_count * 2  # Includes its own embedding + untied head.
    weights_total = packed_bytes + affine_bytes + retained_bytes + draft_bytes
    return {"schema_version": 1, "status": "CANDIDATE_NOT_DEVICE_VALIDATED",
            "method": "DOPT Quant_act_weight_eco / group_min_max PTQ",
            "weight_bits": 4, "activation_bits": 16, "activation_quantization": True,
            "input_quantizer_compute_dtype": "torch.float32",
            "non_quant_compute_dtype": "torch.float16",
            "group_size": group_size, "draft_precision": "torch.float16",
            "expert_count": expert_count, "expert_matrix_count": expert_matrices, "matrices": matrices,
            "parameter_count": {"target": target_count, "draft_with_embedding_head": draft_count,
                                "quantized_target": quantized, "retained_target": target_count-quantized},
            "weight_budget": {"packed_int4_bytes": packed_bytes, "fp16_scale_offset_bytes": affine_bytes,
                              "target_fp16_bytes": retained_bytes, "draft_fp16_bytes": draft_bytes,
                              "total_bytes": weights_total, "total_gib": weights_total / 2**30,
                              "excluded": ["compiler layout/LUT/alignment", "HIAI workspace",
                                           "load-time copies", "activations", "IO and cache"]},
            "quality_gate": "PENDING full-weight held-out evaluation; no quantization quality PASS"}


def memory_budget(plan, *, available_gib=18, sequence_length=128, feature_width=16384):
    if available_gib <= 0 or sequence_length < 2 or feature_width < 1:
        raise ValueError("positive memory/feature width and sequence_length > 1 required")
    # Persistent target/draft IO only; real intermediate/workspace peak unknown.
    io = sequence_length * feature_width * 2 * 2 + sequence_length * 4 * 5 + 15 * 4
    lower = plan["weight_budget"]["total_bytes"] + io
    available = int(available_gib * 2**30)
    return {"assumption": "available_gib is usable process budget, not physical RAM",
            "available_bytes": available, "sequence_length": sequence_length,
            "weight_plus_io_estimate_bytes": lower, "remaining_before_workspace_bytes": available-lower,
            "status": "INSUFFICIENT_FOR_ESTIMATE" if lower >= available else "UNVERIFIED_HEADROOM",
            "fit_proven": False, "hugetlb_added_to_budget": False}


def dopt_config(plan):
    return {"layer_strategy": {m["name"]: {
        "type": "<class 'torch.nn.modules.linear.Linear'>",
        "quant_strategy": "Quant_act_weight_eco",
        "weight": {"bit": 4, "group_size": plan["group_size"], "weight_algo": "group_min_max"},
        "input": {"bit": 16, "input_algo": "ema"}
    } for m in plan["matrices"]}}


def load_dopt(root):
    root = Path(root).resolve()
    if not (root / "dopt/dopt_lm").is_dir():
        raise FileNotFoundError("dopt_root must contain dopt/dopt_lm")
    sys.path.insert(0, str(root))
    module = importlib.import_module("dopt.dopt_lm.do_opt")
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(root):
        raise ValueError("another DOPT version is already imported; start a clean process")
    for method in ("optimize_model", "set_quant_state", "set_calibrate_state", "get_air_model"):
        if not callable(getattr(module, method, None)):
            raise RuntimeError(f"this DOPT is missing {method}")
    return module, {"api_path": str(origin.relative_to(root)), "api_sha256": sha256_file(origin),
                    "component_files": files_under(root / "dopt/dopt_lm", root)}


def require_dopt_air(root):
    """Check the actual public-package extension before loading 35B weights."""
    load_dopt(root)
    linear = importlib.import_module("dopt.dopt_lm.modules.qlinear")
    if not hasattr(linear, "NpuQLinear"):
        raise RuntimeError("DOPT AIR adapter has no NpuQLinear; install the matching torch_npu extension "
                           "for this DDK (stock CUDA/CPU PyTorch is insufficient)")
    torch_npu = importlib.import_module("torch_npu")
    if not callable(getattr(torch_npu, "npu_quant_matmul_v4", None)):
        raise RuntimeError("DOPT requires torch_npu.npu_quant_matmul_v4; the installed extension differs")


def _quantizer_float_input(module, args):
    return (args[0].float(), *args[1:])


def _quantizer_half_output(module, args, output):
    return output.to(torch.float16)


def stabilize_input_quantizers(model, matrix_names):
    """Keep division by tiny A16 scales out of FP16.

    The observed vendor FP16 simulator can overflow before INT16 clipping and
    produce NaN. User permits quantized regions to use other dtypes. This
    policy is recorded in the candidate identity and needs device comparison.
    """
    for name in matrix_names:
        layer = model.get_submodule(name)
        if not hasattr(layer, "quant_op"):
            raise RuntimeError(f"DOPT did not quantize {name}")
        quantizer = layer.quant_op.input_quantizer
        if not getattr(quantizer, "_qwen35_fp32_input", False):
            quantizer.register_forward_pre_hook(_quantizer_float_input)
            quantizer.register_forward_hook(_quantizer_half_output)
            quantizer._qwen35_fp32_input = True


def calibrate(target, draft, prompts, output_dir, *, sequence_length, dopt_root,
              source_identity, group_size=128, dopt_module=None):
    """Save a vendor simulation checkpoint; this is not a packed device model."""
    if not prompts or any(not p or len(p) > sequence_length for p in prompts):
        raise ValueError("provide nonempty calibration token sequences within the fixed gear")
    if any(type(t) is not int or not 0 <= t < target.config.vocab_size for p in prompts for t in p):
        raise ValueError("calibration token outside vocabulary")
    if dopt_module is None:
        dopt, vendor = load_dopt(dopt_root)
    else:
        dopt, vendor = dopt_module, {"kind": "TEST_DOUBLE"}
    target.half(); draft.half()
    plan = make_plan(target, draft, group_size=group_size)
    root = output_path(output_dir); root.mkdir(parents=True, exist_ok=False)
    config_file = root / "dopt_config.json"
    save_json(config_file, dopt_config(plan))
    save_json(root / "plan.json", plan)
    save_json(root / "calibration.json", {"token_ids": prompts})
    result = {"schema_version": 1, "artifact_kind": "qwen35-dopt-simulation",
              "status": "CALIBRATING", "source": source_identity, "plan": plan,
              "sequence_length": sequence_length, "vendor": vendor,
              "compressed_device_model": False, "device_validation": "PENDING",
              "calibration_precision": "FP16 weights/inputs; vendor/upstream FP32 statistics allowed offline",
              "deployment_precision": "INT4 weights / vendor 16-bit activation quantization; non-quant graph FP16"}
    try:
        with working_directory(root):
            quantized = dopt.optimize_model(target, str(config_file))
        # The user explicitly permits both weight and activation quantization.
        # A16 here denotes vendor quantization, not necessarily FP16 matmul input.
        dopt.set_quant_state(quantized, weight_state=True, input_state=True)
        dopt.set_calibrate_state(quantized, True)
        stabilize_input_quantizers(quantized, [m["name"] for m in plan["matrices"]])
        quantized.eval()
        selected = dict(quantized.named_modules())
        for matrix in plan["matrices"]:
            layer = selected[matrix["name"]]
            if not hasattr(layer, "quant_op"):
                raise RuntimeError(f"DOPT did not quantize {matrix['name']}")
        device = next(quantized.parameters()).device
        with torch.inference_mode():
            for prompt in prompts:
                ids = torch.zeros((1, sequence_length), dtype=torch.int32, device=device)
                mask = torch.zeros_like(ids)
                ids[0, :len(prompt)] = torch.tensor(prompt, device=device)
                mask[0, :len(prompt)] = 1
                top1, features = quantized(ids, mask)
                if not torch.isfinite(features).all():
                    raise FloatingPointError("nonfinite calibration features")
        dopt.set_calibrate_state(quantized, False)
        # Strict load into the same exposed architecture on export. Never
        # pickle module instances or assume a .pth is a compressed checkpoint.
        torch.save(quantized.state_dict(), root / "target-simulation.pth")
        result["files"] = files_under(root, root)
        result["status"] = "CALIBRATED"
        save_json(root / "manifest.json", result)
    except Exception as error:
        result["status"] = "FAILED"; result["error"] = f"{type(error).__name__}: {error}"
        save_json(root / "failure.json", result)
        raise
    return result


def _restore(target, draft, bundle, dopt_root, *, source_identity):
    root = Path(bundle).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("artifact_kind") != "qwen35-dopt-simulation" or manifest.get("status") != "CALIBRATED":
        raise ValueError("expected a complete DOPT simulation checkpoint")
    if source_identity != manifest.get("source"):
        raise ValueError("quantized checkpoint source weights differ")
    for item in manifest["files"]:
        checked_file(root, item)
    target.half(); draft.half()
    plan = make_plan(target, draft, group_size=manifest["plan"]["group_size"])
    if plan != manifest["plan"]:
        raise ValueError("quantization model/matrix plan changed")
    dopt, vendor = load_dopt(dopt_root)
    if vendor != manifest["vendor"]:
        raise ValueError("DOPT binary changed since calibration")
    quantized = dopt.optimize_model(target, str(root / "dopt_config.json"))
    state = torch.load(root / "target-simulation.pth", map_location="cpu", weights_only=True, mmap=True)
    quantized.load_state_dict(state, strict=True)
    dopt.set_quant_state(quantized, weight_state=True, input_state=True)
    dopt.set_calibrate_state(quantized, False)
    stabilize_input_quantizers(quantized, [m["name"] for m in plan["matrices"]])
    identity = {"method": plan["method"], "weight_bits": 4, "activation_bits": 16,
                "input_quantizer_compute_dtype": "torch.float32",
                "activation_quantization": True, "non_quant_compute_dtype": "torch.float16",
                "group_size": plan["group_size"],
                "matrix_count": len(plan["matrices"]), "expert_matrix_count": plan["expert_matrix_count"],
                "quantized_parameters": plan["parameter_count"]["quantized_target"],
                "calibration_manifest_sha256": sha256_file(root / "manifest.json"),
                "vendor": vendor, "quality_status": "PENDING", "device_validation": "PENDING"}
    return quantized.eval(), draft, identity, dopt, plan


def restore_for_air(target, draft, bundle, dopt_root, *, source_identity):
    quantized, draft, identity, dopt, _ = _restore(
        target, draft, bundle, dopt_root, source_identity=source_identity)
    try:
        air_target = dopt.get_air_model(quantized)
    except Exception as error:
        raise RuntimeError("DOPT get_air_model failed; install its matching torch_npu/AIR dependencies. "
                           "Do not export fake-quant weights as a compressed model.") from error
    if not isinstance(air_target, nn.Module):
        raise RuntimeError("DOPT get_air_model did not return a torch Module")
    return air_target.eval(), draft, identity


def restore_for_reference(target, draft, bundle, dopt_root, *, source_identity):
    from qwen35_moe_dflash.reference.reference_quant import bridge_reference
    quantized, draft, identity, _, plan = _restore(
        target, draft, bundle, dopt_root, source_identity=source_identity)
    return bridge_reference(quantized, [m["name"] for m in plan["matrices"]]), draft, identity

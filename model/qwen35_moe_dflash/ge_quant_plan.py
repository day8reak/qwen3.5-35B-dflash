"""Portable target-weight quantization inventory, not a GE or device artifact.

This describes host cache bytes only. INT4 is packed along the checkpoint K
axis, with the first signed two's-complement nibble in the low four bits.
I4ZN storage, QPARAM encoding and target quantized operator bindings are not
established by this plan. No checkpoint or target SDK is opened here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re

from .config import Qwen35DFlashConfig
from .ge_fused.target import build_target
from .ge_weights import checkpoint_key
from .resources import repository_root


GROUP_SIZE = 256
_DOCUMENT = "docs/qwen35-moe-operators-and-quantization-analysis.md"
_QUANT_SECTION_SHA256 = "2fd31c290bcc1c4606520a691acbef123d6ac91efae4bb60038b73fdfdf1f060"
_LINEAR_SUFFIXES = frozenset({
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
    "linear_attn.out_proj.weight", "self_attn.q_proj.weight",
    "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate.weight", "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight",
})
_FP16_SUFFIXES = frozenset({
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "linear_attn.conv1d.weight", "linear_attn.dt_bias", "linear_attn.A_log",
    "linear_attn.norm.weight", "mlp.shared_expert_gate.weight",
})


def target_tensor_quantization(key, shape):
    """Describe one recognized target tensor without reading its values.

    This helper accepts small test shapes, but quantized K axes must still
    contain complete groups of 256. Unknown names fail rather than silently
    falling into an unquantized or differently quantized policy.
    """
    shape = tuple(shape)
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError(f"{key}: shape must contain positive integers")
    layer = re.fullmatch(r"target\.layers\.([0-9]+)\.(.+)", key)
    suffix = layer.group(2) if layer else None
    if key == "target.embed_tokens.weight":
        kind = "embedding_int8"
    elif key == "target.lm_head.weight" or suffix in _LINEAR_SUFFIXES:
        kind = "linear_int4"
    elif suffix in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}:
        kind = "expert_int4"
    elif key == "target.norm.weight" or suffix in _FP16_SUFFIXES:
        kind = "float16"
    else:
        raise ValueError(f"unrecognized target weight: {key}")
    owner, source = checkpoint_key(key)
    if owner != "target":
        raise ValueError("target quantization plan cannot include draft weights")
    parameters = math.prod(shape)
    record = {
        "key": key, "source_tensor": source, "shape": list(shape),
        "source_expected_dtype": "BF16", "kind": kind, "parameters": parameters,
        "group_size": GROUP_SIZE if kind.endswith("int4") else None,
        "weight_shape": list(shape), "weight_dtype": "float16",
        "weight_bytes": parameters * 2, "scale_shape": None,
        "scale_dtype": None, "scale_bytes": 0,
        "weight_transform": "CAST_ONLY_NO_OFFSET",
    }
    if kind in {"linear_int4", "expert_int4"}:
        expected_rank = 3 if kind == "expert_int4" else 2
        if len(shape) != expected_rank or shape[-1] % GROUP_SIZE:
            raise ValueError(f"{key}: INT4 requires rank {expected_rank} and K divisible by {GROUP_SIZE}")
        scale_shape = [*shape[:-1], shape[-1] // GROUP_SIZE]
        record.update(
            weight_shape=[*shape[:-1], shape[-1] // 2], weight_dtype="uint8",
            weight_bytes=parameters // 2, scale_shape=scale_shape,
            scale_dtype="float32", scale_bytes=math.prod(scale_shape) * 4,
            weight_transform="PORTABLE_SIGNED_INT4_K_LOW_NIBBLE_FIRST",
            packed_axis=len(shape) - 1, scale_axis=len(shape) - 1,
        )
    elif kind == "embedding_int8":
        if len(shape) != 2:
            raise ValueError(f"{key}: embedding requires rank two")
        record.update(weight_dtype="int8", weight_bytes=parameters,
                      scale_shape=[shape[0]], scale_dtype="float32",
                      scale_bytes=shape[0] * 4, scale_axis=1,
                      weight_transform="SYMMETRIC_INT8_PER_ROW")
    if key == "target.norm.weight" or (suffix is not None and (
            suffix.endswith("layernorm.weight") or suffix.endswith("norm.weight"))):
        # Preserve the existing graph's treatment of raw HF norm parameters.
        # Main/q/k norms add one in the graph; gated GDN norm does not.
        record["checkpoint_weight_addend"] = 0
        record["graph_norm_weight_addend"] = 0 if suffix == "linear_attn.norm.weight" else 1
    return record


def _sources():
    root = repository_root()
    lock_path = root / "specs/upstream.lock.json"
    locks = json.loads(lock_path.read_text())
    by_repo = {item["repo"]: item for item in locks}
    configs, sources = {}, {}
    for name, repo in (("target", "Qwen/Qwen3.5-35B-A3B"),
                       ("draft", "z-lab/Qwen3.5-35B-A3B-DFlash")):
        path = f"specs/{name}-config.json"
        payload = (root / path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        locked = by_repo[repo]
        if digest != locked["files"]["config.json"]["sha256"]:
            raise ValueError(f"{name} configuration differs from upstream lock")
        if re.fullmatch(r"[0-9a-f]{40}", locked["revision"]) is None:
            raise ValueError(f"{name} upstream revision must be pinned")
        configs[name] = json.loads(payload)
        sources[name] = {"repo": repo, "revision": locked["revision"],
                         "config": {"path": path, "sha256": digest}}
    document = (root / _DOCUMENT).read_bytes()
    start = document.find("## 6. ".encode())
    end = document.find("## 附".encode(), start + 1)
    if (start < 0 or end < 0
            or hashlib.sha256(document[start:end]).hexdigest() != _QUANT_SECTION_SHA256):
        raise ValueError("quantization document section changed; review the portable plan")
    sources["document"] = {"path": _DOCUMENT, "sha256": hashlib.sha256(document).hexdigest(),
                           "section": "6. 量化方案分析", "section_sha256": _QUANT_SECTION_SHA256}
    sources["upstream_lock"] = {"path": "specs/upstream.lock.json",
                                "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()}
    return configs, sources


def target_quantization_plan():
    """Inventory the complete locked target graph, without opening weights.

    A fresh graph is constructed once per call, so source/configuration edits
    cannot reuse a stale process-wide inventory. Only its weight metadata is
    retained; this return value has no graph, AIR or OMC execution contract.
    """
    configs, sources = _sources()
    graph = build_target(configs["target"], Qwen35DFlashConfig.from_dict(configs["draft"]),
                         1, past_length=1, fused_ops=(), expert_dispatch="selected")
    tensors = [target_tensor_quantization(key, graph.weights[key].shape)
               for key in sorted(graph.weights)]
    counts = dict(sorted(Counter(item["kind"] for item in tensors).items()))
    by_kind = {}
    for kind in counts:
        members = [item for item in tensors if item["kind"] == kind]
        by_kind[kind] = {"tensor_count": len(members), **{
            field: sum(item[field] for item in members)
            for field in ("parameters", "weight_bytes", "scale_bytes")}}
        by_kind[kind]["payload_bytes"] = by_kind[kind]["weight_bytes"] + by_kind[kind]["scale_bytes"]
    weight_bytes = sum(item["weight_bytes"] for item in tensors)
    scale_bytes = sum(item["scale_bytes"] for item in tensors)
    return {
        "schema_version": 1, "status": "PORTABLE_QUANTIZATION_PLAN",
        "scope": "Full target text-model host quantization cache only; no draft or vision weights.",
        "deployment_ready": False, "hardware_format": "UNRESOLVED",
        "hardware_mapping": {"I4ZN": "NOT_IMPLEMENTED", "QPARAM": "NOT_IMPLEMENTED",
                             "GE_quantized_bindings": "UNVERIFIED"},
        "group_size": GROUP_SIZE,
        "cache_format": "ROW_MAJOR_K_PACKED_INT4_LOW_NIBBLE_FIRST_FP32_SCALES_V1",
        "int4_encoding": "Signed two's-complement nibbles; K[2*j] low, K[2*j+1] high; no transpose.",
        "weight_access": "NOT_RUN", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "graph_inventory": {"sequence_length": 1, "past_length": 1,
                            "expert_dispatch": "selected", "fused_ops": [],
                            "num_hidden_layers": configs["target"]["text_config"]["num_hidden_layers"]},
        "sources": sources, "tensors": tensors,
        "summary": {"tensor_count": len(tensors), "counts": counts,
                    "parameter_count": sum(item["parameters"] for item in tensors),
                    "weight_bytes": weight_bytes, "scale_bytes": scale_bytes,
                    "payload_bytes": weight_bytes + scale_bytes, "by_kind": by_kind},
    }

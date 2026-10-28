"""Local original-W4 target construction; no requantization or implicit GE ABI.

Only selected expert banks are gathered. The external embedding is supplied as
INT8 rows and FP32 row scales. The graph uses the existing functional cache ABI,
not the original AIR's paged RefData ABI, and makes no device success claim.
"""
from copy import deepcopy
import math

import numpy as np

from .quant_bindings import QuantOperatorBindings
from .target import _StatefulBuilder, build_target


class _QuantBuilder(_StatefulBuilder):
    def __init__(self, *args, quant_bindings, quant_descriptors, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = quant_bindings
        self.descriptors = quant_descriptors
        self.sources = {}

    def input_embedding(self, sequence_length, config):
        shape = (1, sequence_length, config["hidden_size"])
        rows = self.g.input("inputs_embeds", shape, "int8")
        scales = self.g.input("embedding_scale", (1, sequence_length, 1), "float32")
        expanded = self.op("cast", rows, dtype="float32")
        dequantized = self.op("mul", expanded, scales, dtype="float32")
        return self.op("cast", dequantized, dtype="float16")

    def quantized_weights(self, key, logical_shape, *, expert=False):
        if key not in self.descriptors:
            raise ValueError(f"original quantized tensor metadata is missing: {key}")
        desc = self.descriptors[key]
        logical_shape = tuple(logical_shape)
        n, k = logical_shape[-2:]
        prefix = logical_shape[:-2]
        packed_shape = (*prefix, k, n // 8)
        scale_shape = (*prefix, k // 256, n)
        if (tuple(desc.get("logical_shape", ())) != logical_shape or desc.get("group_size") != 256
                or n % 16 or k % 256 or tuple(desc.get("packed_shape", ())) != packed_shape
                or desc.get("packed_dtype") != "int32" or tuple(desc.get("scale_shape", ())) != scale_shape
                or desc.get("kind") != ("expert" if expert else "linear")):
            raise ValueError(f"original W4 carrier/scale shape contract mismatch: {key}")
        encoding = desc.get("scale_encoding")
        if expert:
            if encoding not in {"QPARAM_REQUIRED", "QPARAM_VENDOR_CONVERTED"}:
                raise ValueError(f"expert requires explicit vendor QPARAM conversion metadata: {key}")
            if encoding == "QPARAM_REQUIRED" and desc.get("scale_dtype") != "float32":
                raise ValueError(f"pending expert scales must retain their FP32 carrier: {key}")
            if encoding == "QPARAM_VENDOR_CONVERTED" and desc.get("scale_dtype") != "int64":
                raise ValueError(f"converted expert QPARAM must have int64 carrier: {key}")
        elif encoding != "FP32_GROUP_SCALE" or desc.get("scale_dtype") != "float32":
            raise ValueError(f"ordinary quantized linear scale must stay FP32: {key}")
        self.sources[key] = deepcopy(desc)
        weight = self.g.weight(key, packed_shape, "int32")
        scale = self.g.weight(key + ".quant_scale", scale_shape, "int64" if expert else "float32")
        return weight, scale

    def dynamic_quant(self, x):
        if x.dtype != "float16" or len(x.shape) != 2:
            raise ValueError("dynamic quant requires flattened FP16 [tokens, hidden]")
        result = self.quant.call(self.g, "dynamic_quant", inputs={"x": x},
            outputs={"quantized": (x.shape, "int8"), "scale": ((x.shape[0],), "float32")},
            name=f"quant_{len(self.g.nodes)}")
        return result["quantized"], result["scale"]

    def linear(self, x, key, width, weight=None):
        if key is not None and key.endswith(".shared_expert_gate.weight"):
            return super().linear(x, key, width, weight)
        if weight is not None or key is None:
            raise ValueError("quantized target requires named original linear tensors")
        w, scale = self.quantized_weights(key, (width, x.shape[-1]))
        rows = math.prod(x.shape[:-1])
        flat = self.view(x, (rows, x.shape[-1])) if len(x.shape) != 2 else x
        quantized, token_scale = self.dynamic_quant(flat)
        result = self.quant.call(self.g, "quant_matmul",
            inputs={"x": quantized, "weight": w, "scale": scale,
                    "token_scale": self.view(token_scale, (rows, 1))},
            outputs={"y": ((rows, width), "float16")}, name=f"qlinear_{len(self.g.nodes)}")["y"]
        return self.view(result, (*x.shape[:-1], width)) if len(x.shape) != 2 else result

    def grouped(self, x, weight, scale, width):
        groups = weight.shape[0]
        if x.shape[0] != groups:
            raise ValueError("selected GMM requires one token per selected expert")
        quantized, token_scale = self.dynamic_quant(x)
        return self.quant.call(self.g, "grouped_matmul",
            inputs={"x": quantized, "weight": weight, "scale": scale,
                    "token_scale": token_scale, "group_list": self.constant(np.ones(groups, dtype=np.int64), "int64")},
            outputs={"y": ((groups, width), "float16")},
            attrs={"group_list_type": 1, "split_item": 3, "group_type": 0, "act_type": 0},
            name=f"qgroup_{len(self.g.nodes)}")["y"]

    def moe(self, x, prefix, config, expert_dispatch):
        s, h = x.shape[1:]
        experts, topk = config["num_experts"], config["num_experts_per_tok"]
        middle, shared = config["moe_intermediate_size"], config["shared_expert_intermediate_size"]
        if expert_dispatch != "selected" or experts > 2048:
            raise ValueError("original quantized MoE requires selected dispatch with at most 2048 experts")
        flat = self.view(x, (s, h))
        probs = self.op("softmax", self.linear(flat, prefix + ".gate.weight", experts), axis=-1)
        values = self.op("topk_values", probs, shape=(s, topk), axis=-1, k=topk)
        ids = self.op("topk_indices", probs, shape=(s, topk), dtype="int32", axis=-1, k=topk)
        routes = self.op("div", values, self.reduce("reduce_sum", values, -1)) if config.get("norm_topk_prob", True) else values
        gate_up, gate_scale = self.quantized_weights(prefix + ".experts.gate_up_proj", (experts, middle * 2, h), expert=True)
        down, down_scale = self.quantized_weights(prefix + ".experts.down_proj", (experts, h, middle), expert=True)
        rows = []
        for token in range(s):
            selected = self.view(self.slice(ids, 0, token, 1), (topk,))
            negative = self.op("neg", self.op("cast", selected, dtype="float16"))
            order = self.op("topk_indices", negative, shape=(topk,), dtype="int32", axis=-1, k=topk)
            sorted_values = self.op("topk_values", negative, shape=(topk,), axis=-1, k=topk)
            selected = self.op("cast", self.op("neg", sorted_values), dtype="int32")
            route = self.view(self.slice(routes, 0, token, 1), (topk,))
            route = self.op("gather", route, order, shape=(topk,), axis=0)

            def gather(bank):
                return self.op("gather", bank, selected, shape=(topk, *bank.shape[1:]), dtype=bank.dtype, axis=0)

            row = self.slice(flat, 0, token, 1)
            projected = self.grouped(self.cat([row] * topk, 0), gather(gate_up), gather(gate_scale), middle * 2)
            gate, up = self.slice(projected, -1, 0, middle), self.slice(projected, -1, middle, middle)
            activated = self.op("mul", self.op("silu", gate), up)
            projected = self.grouped(activated, gather(down), gather(down_scale), h)
            weighted = self.op("mul", projected, self.view(route, (topk, 1)))
            summed = self.slice(weighted, 0, 0, 1)
            for slot in range(1, topk):
                summed = self.op("add", summed, self.slice(weighted, 0, slot, 1))
            rows.append(summed)
        result = self.cat(rows, 0)
        gate = self.linear(flat, prefix + ".shared_expert.gate_proj.weight", shared)
        up = self.linear(flat, prefix + ".shared_expert.up_proj.weight", shared)
        shared_out = self.linear(self.op("mul", self.op("silu", gate), up), prefix + ".shared_expert.down_proj.weight", h)
        shared_gate = self.op("sigmoid", self.linear(flat, prefix + ".shared_expert_gate.weight", 1))
        return self.view(self.op("add", result, self.op("mul", shared_out, shared_gate)), (1, s, h))


def build_quantized_target(config, draft_config, sequence_length, *, quant_bindings,
                           quant_descriptors, past_length=0, fused_ops=(), bindings=None,
                           recurrent_dtype="float16", head_chunk_rows=16, gdn_state_casts="reject"):
    if not isinstance(quant_bindings, QuantOperatorBindings):
        raise ValueError("quantized direct GE requires explicit QuantOperatorBindings; no installed schema is assumed")
    if sequence_length not in {1, 64}:
        raise ValueError("original quantized target supports static prefill64 and decode1 only")
    declared = config.get("text_config", config)
    if "kv_cache_max_len" in declared:
        capacity = declared["kv_cache_max_len"]
        maximum = declared.get("max_position_embeddings", capacity)
        if (type(capacity) is not int or capacity < 1 or type(maximum) is not int or maximum < 1
                or type(past_length) is not int or past_length < 0):
            raise ValueError("quantized target requires positive integer context capacities and nonnegative past_length")
        if past_length + sequence_length > min(capacity, maximum):
            raise ValueError("quantized target past_length + sequence_length exceeds kv_cache_max_len / position capacity")
    if not isinstance(quant_descriptors, dict) or not quant_descriptors:
        raise ValueError("original quantized tensor descriptors are required; no requantization fallback")
    builders = []

    def factory(*args, **kwargs):
        value = _QuantBuilder(*args, quant_bindings=quant_bindings, quant_descriptors=quant_descriptors, **kwargs)
        builders.append(value)
        return value

    graph = build_target(config, draft_config, sequence_length, past_length=past_length,
        fused_ops=fused_ops, bindings=bindings, recurrent_dtype=recurrent_dtype,
        expert_dispatch="selected", head_chunk_rows=head_chunk_rows, gdn_state_casts=gdn_state_casts,
        builder_factory=factory)
    graph.quant_bindings = quant_bindings
    graph.quant_weight_sources = builders[0].sources
    graph.quantization_contract = {
        "source": "ORIGINAL_W4_NO_REQUANTIZATION", "scale_group_size": 256,
        "expert_dispatch": "selected_per_token", "expert_count_evaluated_per_token": config.get("text_config", config)["num_experts_per_tok"],
        "embedding": "external_INT8_rows_times_FP32_scale_then_FP16",
        "pending_expert_qparam": [key for key, desc in graph.quant_weight_sources.items() if desc["scale_encoding"] == "QPARAM_REQUIRED"],
        "ge_bindings": quant_bindings.manifest(), "device_execution": "NOT_RUN",
        "runtime_abi": "functional_stateful_target_not_original_paged_RefData",
        "normalization": "raw_HF_norm_weights_not_loader_prefolded",
        "kv_cache_max_len": declared.get("kv_cache_max_len"),
    }
    return graph


def emit_quantized_cpp(graph, weight_paths, *, recipe_dir, external_weights=True,
                       fold_constant_arithmetic=False, allow_pending_qparam_structure=False):
    """Generate source/recipe, rejecting unconverted QPARAM in an export path.

    allow_pending_qparam_structure permits descriptor-only inspection. It does
    not create weights or authorize using FP32 scales as QPARAM carriers.
    """
    from .emitter import emit_cpp

    bindings = getattr(graph, "quant_bindings", None)
    if not isinstance(bindings, QuantOperatorBindings) or recipe_dir is None:
        raise ValueError("quantized graph export requires explicit bindings and recipe_dir")
    pending = graph.quantization_contract["pending_expert_qparam"]
    if pending and not allow_pending_qparam_structure:
        raise ValueError("expert QPARAM must be produced by the vendor converter before export: " + pending[0])
    source = emit_cpp(graph, weight_paths, recipe_dir=recipe_dir, external_weights=external_weights,
                      fold_constant_arithmetic=fold_constant_arithmetic)
    return bindings.extend_recipe_cpp(source)

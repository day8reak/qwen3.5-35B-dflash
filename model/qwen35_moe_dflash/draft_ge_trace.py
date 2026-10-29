"""Append bounded diagnostic outputs without rewriting the original draft IR.

The probes follow existing producer connections and weight keys. Original
arithmetic, node names and order are retained; the new graph is a diagnostic
artifact whose compiled placement still needs independent verification.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from types import SimpleNamespace

from .ge_ir import Graph


MAX_TRACE_BYTES = 64 << 20
TRACE_OUTPUT = "draft_trace_values"
_CONCAT_FAN_IN = 16


def _trace_points(graph):
    """Resolve semantic probes from connections, never from generated vN names."""
    nodes = graph.nodes
    by_name = {node.output.name: node for node in nodes}

    def one(kind=None, *, inputs=None, source=None, weight=None, label="probe"):
        matches = [node for node in nodes
                   if (kind is None or node.kind == kind)
                   and (inputs is None or node.inputs == tuple(inputs))
                   and (source is None or source in node.inputs)
                   and (weight is None or any(value.name == weight for value in node.inputs))]
        if len(matches) != 1:
            raise ValueError(f"draft trace {label}: expected one producer, found {len(matches)}")
        return matches[0].output

    def named(name, kind=None):
        node = by_name.get(name)
        if node is None or (kind is not None and node.kind != kind):
            raise ValueError("draft trace is missing expected node: " + name)
        return node.output

    def norm(key):
        return one("mul", weight=key, label=key)

    def projected(key, source=None):
        return one("matmul", weight=key, source=source, label=key)

    def rotary(source):
        # x*cos is the only direct Mul use of the pre-RoPE query/key.
        left = one("mul", source=source, label="rotary product")
        return one("add", source=left, label="rotary result")

    layer_ids = sorted({int(match.group(1)) for key in graph.weights
                        if (match := re.fullmatch(r"draft\.layers\.(\d+)\.input_layernorm\.weight", key))})
    if not layer_ids or layer_ids != list(range(len(layer_ids))):
        raise ValueError("draft trace requires contiguous complete draft layers")
    points = {name: named(name) for name in ("block_ids", "context_valid", "positions",
                                             "projected_context", "rope_cosine", "rope_sine")}
    embedding = one("gather", weight="target.embed_tokens.weight", label="embedding")
    points["noise_embedding"] = one("scalar_mul", source=embedding, label="noise embedding")
    points["context_fc"] = projected("draft.fc.weight")
    hidden = points["noise_embedding"]
    context = points["projected_context"]
    for layer in layer_ids:
        prefix = f"layer_{layer}_"
        key = f"draft.layers.{layer}."
        attn = key + "self_attn."
        values = {"input": hidden, "input_norm": norm(key + "input_layernorm.weight")}
        values["q_proj"] = projected(attn + "q_proj.weight")
        values["q_norm"] = norm(attn + "q_norm.weight")
        for kind in ("k", "v"):
            values[kind + "_context"] = projected(attn + kind + "_proj.weight", context)
            values[kind + "_block"] = projected(attn + kind + "_proj.weight", values["input_norm"])
        values["k_norm"] = norm(attn + "k_norm.weight")
        query = one("transpose", source=values["q_norm"], label=prefix + "query transpose")
        key_transposed = one("transpose", source=values["k_norm"], label=prefix + "key transpose")
        values["query"], values["key"] = rotary(query), rotary(key_transposed)
        joined = one("concat", inputs=(values["v_context"], values["v_block"]), label=prefix + "value concat")
        reshaped = one("reshape", source=joined, label=prefix + "value reshape")
        values["value"] = one("transpose", source=reshaped, label=prefix + "value transpose")
        values["visible"] = named(prefix + "visible", "reshape")
        values["qk_fp16"] = named(prefix + "qk_fp16", "matmul")
        values["probability_fp32"] = named(prefix + "probability_fp32", "softmax")
        probability_node = by_name[values["probability_fp32"].name]
        if len(probability_node.inputs) != 1 or by_name[probability_node.inputs[0].name].kind != "where":
            raise ValueError("draft trace softmax must consume masked scores")
        values["scores_fp32"] = probability_node.inputs[0]
        values["pv_fp16"] = named(prefix + "pv_fp16", "matmul")
        values["attention"] = projected(attn + "o_proj.weight")
        values["post_attention"] = one("add", inputs=(hidden, values["attention"]), label=prefix + "residual")
        values["post_attention_norm"] = norm(key + "post_attention_layernorm.weight")
        values["mlp_gate"] = projected(key + "mlp.gate_proj.weight")
        values["mlp_up"] = projected(key + "mlp.up_proj.weight")
        values["mlp_silu"] = one("silu", source=values["mlp_gate"], label=prefix + "silu")
        values["mlp_product"] = one("mul", inputs=(values["mlp_silu"], values["mlp_up"]), label=prefix + "swiglu")
        values["mlp_down"] = projected(key + "mlp.down_proj.weight")
        hidden = one("add", inputs=(values["post_attention"], values["mlp_down"]), label=prefix + "output")
        values["output"] = hidden
        points.update((prefix + name, value) for name, value in values.items())
    points["draft_hidden"] = named("draft_hidden", "slice")
    points["draft_top1"] = named("draft_top1", "argmax")
    return points, len(layer_ids)


def instrument_draft_trace(graph, *, names=None):
    """Return a copied diagnostic graph and ordered packed-output descriptors.

    The first output remains original Top1. A second FP32 ``[1,1,N]`` output
    carries selected original tensors, with their shapes/dtypes/offsets in
    ``spec``. INT32 token/position probes stay exact in the supported domain;
    BOOL becomes 0/1. Masked score -Inf values are retained intentionally.
    """
    if not isinstance(graph, Graph) or graph.name != "draft":
        raise ValueError("draft trace requires a draft construction graph")
    graph.validate()
    if not graph.outputs or graph.outputs[0].name != "draft_top1":
        raise ValueError("draft trace requires original Top1 as first output")
    if any(value.name.startswith("__draft_trace/") or value.name == TRACE_OUTPUT
           for value in graph._values.values()):
        raise ValueError("draft graph is already instrumented or contains reserved trace names")
    points, layers = _trace_points(graph)
    # Only name metadata is shared with the independent PyTorch observer.
    from .draft_layer_trace import trace_names
    supported = trace_names(SimpleNamespace(num_hidden_layers=layers))
    if names is None:
        selected = supported
    elif isinstance(names, (str, bytes)):
        raise ValueError("draft trace names must be an ordered sequence")
    else:
        selected = tuple(names)
    if (not selected or any(not isinstance(name, str) or name not in points for name in selected)
            or len(selected) != len(set(selected)) or len(selected) > 256):
        raise ValueError("draft trace selection is empty, duplicated, unsupported or exceeds count budget")
    spec, offset = [], 0
    for name in selected:
        value = points[name]
        elements = math.prod(value.shape)
        if (value.dtype not in {"float16", "float32", "int32", "bool"}
                or not value.shape or any(type(dim) is not int or dim < 1 for dim in value.shape)):
            raise ValueError("unsupported draft trace descriptor: " + name)
        if (offset + elements) * 4 > MAX_TRACE_BYTES:
            raise ValueError("packed draft trace exceeds 64 MiB")
        spec.append({"name": name, "shape": list(value.shape), "dtype": value.dtype,
                     "offset": offset, "elements": elements})
        offset += elements
    result = deepcopy(graph)
    packed = []
    for index, row in enumerate(spec):
        original = result._values[points[row["name"]].name]
        value = original
        if value.dtype != "float32":
            value = result.op("cast", value, shape=value.shape, dtype="float32", attrs={"dtype": "float32"},
                              name=f"__draft_trace/{index}/fp32")
        flat = result.op("reshape", value, shape=(1, 1, row["elements"]), dtype="float32",
                         attrs={"shape": [1, 1, row["elements"]]}, name=f"__draft_trace/{index}/flat")
        packed.append(flat)
    # Keep the public concat interface bounded even for all six layers. Each
    # level only copies consecutive chunks, so offsets and element order stay
    # identical to the flat spec. A selected single tensor needs no concat.
    level = 0
    while len(packed) > _CONCAT_FAN_IN:
        groups = []
        for index in range(0, len(packed), _CONCAT_FAN_IN):
            chunk = packed[index:index + _CONCAT_FAN_IN]
            if len(chunk) == 1:
                groups.append(chunk[0])
            else:
                size = sum(value.shape[-1] for value in chunk)
                groups.append(result.op("concat", *chunk, shape=(1, 1, size), dtype="float32",
                                        attrs={"axis": -1}, name=f"__draft_trace/concat_{level}/{index}"))
        packed = groups
        level += 1
    if len(packed) == 1:
        trace = result.op("reshape", packed[0], shape=(1, 1, offset), dtype="float32",
                          attrs={"shape": [1, 1, offset]}, name=TRACE_OUTPUT)
    else:
        trace = result.op("concat", *packed, shape=(1, 1, offset), dtype="float32",
                          attrs={"axis": -1}, name=TRACE_OUTPUT)
    result.outputs = [result.outputs[0], trace]
    result.validate()
    return result, spec

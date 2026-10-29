"""Bounded v3 CPU draft activations for a matching diagnostic AIR graph.

Hooks observe the existing reference and return its original operator results.
The score/probability probes are independent repetitions of the v3 primitive
formula; they never replace attention, weights, inputs, or retained goldens.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import checked_file, output_path, save_json
from .draft_reference import DRAFT_REFERENCE_CONTRACT, _load_graph, _validated_prepared, _write_tensor
from .draft_reference_trace import _summary
from .hashing import sha256_file


TRACE_CONTRACT = "pytorch-air4b-draft-layer-trace-v1"
_INPUT_NAMES = ("target_features", "input_ids", "attention_mask")
_MAX_TENSOR_BYTES = 8 << 20
_MAX_TOTAL_BYTES = 64 << 20


def trace_names(config):
    """Stable execution-order names shared with the diagnostic GE selector.

    q_norm/k_norm precede transpose and RoPE; query/key follow RoPE and
    precede KV repetition. value is transposed and precedes KV repetition.
    scores_fp32 includes scaling and masking; its masked entries are -Inf.
    The complete logits are deliberately omitted from the bounded trace.
    """
    names = ["block_ids", "context_valid", "positions", "noise_embedding",
             "context_fc", "projected_context", "rope_cosine", "rope_sine"]
    for layer in range(config.num_hidden_layers):
        names.extend(f"layer_{layer}_{suffix}" for suffix in (
            "input", "input_norm", "q_proj", "q_norm", "k_context", "k_block",
            "v_context", "v_block", "k_norm", "query", "key", "value", "visible",
            "qk_fp16", "scores_fp32", "probability_fp32", "pv_fp16", "attention",
            "post_attention", "post_attention_norm", "mlp_gate", "mlp_up",
            "mlp_silu", "mlp_product", "mlp_down", "output"))
    return tuple(names + ["draft_hidden", "draft_top1"])


class _AttentionObserver:
    def __init__(self, original, layer, emit, wanted):
        self.original, self.prefix, self.emit, self.wanted = original, f"layer_{layer}_", emit, wanted

    def __getattr__(self, name):
        return getattr(self.original, name)

    def rotary(self, query, key, cosine, sine):
        result = self.original.rotary(query, key, cosine, sine)
        self.emit(self.prefix + "query", result[0])
        self.emit(self.prefix + "key", result[1])
        return result

    def attention(self, query, key, value, attention_mask, scale, key_value_groups):
        import torch
        from .ops import _repeat_kv

        self.emit(self.prefix + "value", value)
        if attention_mask is None:
            visible = torch.ones((query.shape[0], 1, query.shape[-2], key.shape[-2]), dtype=torch.bool)
        else:
            visible = attention_mask
        self.emit(self.prefix + "visible", visible)
        # Preserve the original call, including its actual FP16 PV output.
        result = self.original.attention(query, key, value, attention_mask, scale, key_value_groups)
        self.emit(self.prefix + "pv_fp16", result)
        if any(self.prefix + suffix in self.wanted for suffix in
               ("qk_fp16", "scores_fp32", "probability_fp32")):
            repeated = _repeat_kv(key, key_value_groups)
            raw = torch.matmul(query.half(), repeated.half().transpose(-1, -2))
            scores = raw.float() * scale
            if attention_mask is not None:
                scores = torch.where(attention_mask, scores,
                                     torch.tensor(-float("inf"), dtype=torch.float32))
            self.emit(self.prefix + "qk_fp16", raw)
            self.emit(self.prefix + "scores_fp32", scores)
            self.emit(self.prefix + "probability_fp32", torch.softmax(scores, -1))
        return result


class _MLPObserver:
    def __init__(self, original, layer, emit):
        self.original, self.prefix, self.emit = original, f"layer_{layer}_", emit

    def __getattr__(self, name):
        return getattr(self.original, name)

    def swiglu(self, gate, up):
        import torch.nn.functional as functional
        result = self.original.swiglu(gate, up)
        self.emit(self.prefix + "mlp_silu", functional.silu(gate))
        self.emit(self.prefix + "mlp_product", result)
        return result


def _validated_inputs(graph, inputs):
    if isinstance(inputs, dict):
        if set(inputs) != set(_INPUT_NAMES):
            raise ValueError("trace input names differ from draft ABI")
        arrays = [inputs[name] for name in _INPUT_NAMES]
    else:
        arrays = list(inputs)
    if (len(arrays) != 3 or any(not isinstance(item, np.ndarray) for item in arrays)
            or [item.dtype for item in arrays] != [np.dtype("<f2"), np.dtype("<i4"), np.dtype("<i4")]
            or any(not item.flags.c_contiguous for item in arrays)):
        raise ValueError("trace requires contiguous retained FP16/INT32/INT32 NumPy arrays")
    config = graph.draft.config
    features, ids, mask = arrays
    if (features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 1
            or features.shape[-1] != config.feature_size
            or ids.shape != features.shape[:2] or mask.shape != ids.shape):
        raise ValueError("trace input shapes differ from draft ABI")
    if (not np.isfinite(features).all() or np.any(ids < 0) or np.any(ids >= config.vocab_size)
            or not np.isin(mask, [0, 1]).all() or not mask[0, 0]
            or np.any(np.diff(mask.astype(np.int64), axis=1) > 0)):
        raise ValueError("trace inputs violate finite features/token/right-padded mask contract")
    return arrays


def trace_draft_graph(graph, inputs, *, names=None, progress=None):
    """Return ``(ordered name->NumPy tensors, JSON-safe trace report)``.

    This runs the original loaded ``air4b-precision`` graph exactly once.
    Every attached hook and replaced dispatcher is restored on failure too.
    ``names`` selects a bounded subset in the supplied order; missing/duplicate
    names are errors. No threshold or golden file is changed by this helper.
    """
    import torch
    if (graph.training or getattr(graph, "reference_rms_mode", None) != "air4b-precision"
            or any(item.device.type != "cpu" or item.dtype != torch.float16 for item in graph.parameters())):
        raise ValueError("layer trace requires an eval-mode CPU FP16 air4b-precision reference")
    arrays = _validated_inputs(graph, inputs)
    supported = trace_names(graph.draft.config)
    selected = supported if names is None else tuple(names)
    if (not selected or len(selected) != len(set(selected)) or any(name not in supported for name in selected)):
        raise ValueError("trace selection is empty, duplicated or contains unknown names")
    if len(selected) > 256:
        raise ValueError("trace selection exceeds tensor count budget")
    wanted, captured, summaries, handles, original_dispatch = set(selected), {}, {}, [], []
    total_bytes = 0

    def emit(name, tensor):
        nonlocal total_bytes
        if name not in wanted:
            return
        if name in captured:
            raise ValueError("trace node executed more than once: " + name)
        count = tensor.numel() * tensor.element_size()
        if count > _MAX_TENSOR_BYTES or total_bytes + count > _MAX_TOTAL_BYTES:
            raise ValueError("draft trace tensor byte budget exceeded: " + name)
        total_bytes += count
        value = tensor.detach().contiguous().numpy().copy()
        captured[name] = value
        summaries[name] = {"name": name, **_summary(tensor), "bytes": count,
                           "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        if progress:
            progress("CPU trace: " + name)

    def output_hook(name):
        return lambda node, args, output: emit(name, output)

    def input_hook(name):
        return lambda node, args: emit(name, args[0])

    def draft_inputs(node, args):
        emit("noise_embedding", args[1])
        emit("positions", args[2].to(torch.int32))
        emit("context_valid", args[4])

    try:
        features, ids, mask = [torch.from_numpy(item) for item in arrays]
        anchor = mask.to(torch.int64).sum(-1, keepdim=True) - 1
        block = torch.cat((ids.gather(1, anchor), torch.full((1, graph.draft.config.block_size - 1),
                          graph.draft.config.mask_token_id, dtype=torch.int32)), dim=-1)
        emit("block_ids", block)
        handles.append(graph.draft.register_forward_pre_hook(draft_inputs))
        for module, name in ((graph.draft.fc, "context_fc"), (graph.draft.hidden_norm, "projected_context")):
            handles.append(module.register_forward_hook(output_hook(name)))
        def rotary_hook(node, args, output):
            emit("rope_cosine", output[0])
            emit("rope_sine", output[1])
        handles.append(graph.draft.rotary.register_forward_hook(rotary_hook))
        for index, layer in enumerate(graph.draft.layers):
            prefix, attn = f"layer_{index}_", layer.self_attn
            handles.append(layer.register_forward_pre_hook(input_hook(prefix + "input")))
            handles.append(layer.register_forward_hook(output_hook(prefix + "output")))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(input_hook(prefix + "post_attention")))
            for module, suffix in ((layer.input_layernorm, "input_norm"), (attn.q_proj, "q_proj"),
                    (attn.q_norm, "q_norm"), (attn.k_norm, "k_norm"), (attn, "attention"),
                    (layer.post_attention_layernorm, "post_attention_norm"), (layer.mlp.gate_proj, "mlp_gate"),
                    (layer.mlp.up_proj, "mlp_up"), (layer.mlp.down_proj, "mlp_down")):
                handles.append(module.register_forward_hook(output_hook(prefix + suffix)))
            for module, suffix in ((attn.k_proj, "k"), (attn.v_proj, "v")):
                calls = []
                def projected(node, args, output, *, suffix=suffix, calls=calls, prefix=prefix):
                    if len(calls) > 1:
                        raise ValueError("unexpected repeated draft K/V projection")
                    emit(prefix + suffix + ("_context" if not calls else "_block"), output)
                    calls.append(True)
                handles.append(module.register_forward_hook(projected))
            original_dispatch.append((attn, attn.ops))
            attn.ops = _AttentionObserver(attn.ops, index, emit, wanted)
            original_dispatch.append((layer.mlp, layer.mlp.ops))
            layer.mlp.ops = _MLPObserver(layer.mlp.ops, index, emit)
        handles.append(graph.draft.norm.register_forward_hook(
            lambda node, args, output: emit("draft_hidden", output[:, 1:, :])))
        with torch.inference_mode():
            top1 = graph(features, ids, mask)
        emit("draft_top1", top1)
        missing = wanted - set(captured)
        if missing:
            raise ValueError("missing requested draft trace nodes: " + ", ".join(sorted(missing)))
        ordered = {name: captured[name] for name in selected}
        report = {"schema_version": 1, "artifact_kind": "qwen35-draft-layer-reference-trace",
                  "status": "TRACED", "contract": TRACE_CONTRACT,
                  "reference_contract": DRAFT_REFERENCE_CONTRACT,
                  "torch_version": str(torch.__version__), "total_bytes": total_bytes,
                  "tensors": [{"index": index, **summaries[name]} for index, name in enumerate(selected)],
                  "target_execution": "NOT_RUN", "device_execution": "NOT_RUN",
                  "score_probes": "Independent repetition of v3 QK/scale/mask/softmax; original attention result retained",
                  "scope": "CPU intermediate probes; caller must verify prepared/input identities and retained golden reproduction."}
        return ordered, report
    finally:
        for handle in handles:
            handle.remove()
        for module, original in original_dispatch:
            module.ops = original


def _read_array(root, record):
    path = checked_file(root, record)
    dtype = np.dtype(record["dtype"])
    if dtype not in (np.dtype("float16"), np.dtype("int32")):
        raise ValueError("retained draft tensor has an unsupported dtype")
    if math.prod(record["shape"]) * dtype.itemsize != record["bytes"]:
        raise ValueError("retained draft tensor shape and byte count differ")
    array = np.fromfile(path, dtype=dtype).reshape(record["shape"])
    if hashlib.sha256(array.tobytes()).hexdigest() != record["sha256"]:
        raise ValueError("retained draft tensor changed after verification")
    return array


def trace_prepared_draft(prepared_dir, fixture_dir, output_dir, *, names=None, progress=None):
    """Trace hash-locked prepared weights and retained v3 inputs, without regeneration.

    Full hidden and Top1 must reproduce the retained golden byte-for-byte.
    The new output directory contains only selected small probes and metadata.
    """
    prepared, plan, config, paths = _validated_prepared(prepared_dir)
    fixture = Path(fixture_dir).resolve()
    prepared_digest = sha256_file(prepared / "prepared.json")
    fixture_digest = sha256_file(fixture / "report.json")
    reference = json.loads((fixture / "report.json").read_text())
    if (reference.get("status") != "REFERENCE_COMPLETE"
            or reference.get("reference_contract") != DRAFT_REFERENCE_CONTRACT
            or reference.get("prepared_manifest_sha256") != prepared_digest
            or reference.get("draft_config") != plan["draft_config"]
            or reference.get("sequence_length") != plan["sequence_length"]):
        raise ValueError("trace requires the matching complete retained v3 reference fixture")
    records = reference.get("inputs", [])
    expected_shapes = [(1, plan["sequence_length"], config.feature_size),
                       (1, plan["sequence_length"]), (1, plan["sequence_length"])]
    if len(records) != 3 or any(row.get("index") != index or row.get("name") != name
            or row.get("shape") != list(expected_shapes[index])
            or row.get("dtype") != ("float16" if index == 0 else "int32")
            for index, (name, row) in enumerate(zip(_INPUT_NAMES, records))):
        raise ValueError("retained trace fixture input ABI differs")
    inputs = [_read_array(fixture, row) for row in records]
    requested = trace_names(config) if names is None else tuple(names)
    required = tuple(dict.fromkeys((*requested, "draft_hidden", "draft_top1")))
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-draft-layer-reference-trace",
              "contract": TRACE_CONTRACT, "status": "FAILED",
              "prepared_manifest_sha256": prepared_digest, "fixture_report_sha256": fixture_digest,
              "source": plan["source"], "inputs": records, "reference_contract": DRAFT_REFERENCE_CONTRACT}
    identities = {path: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                  for path in paths.values()}
    try:
        graph = _load_graph(config, paths, rms_mode="air4b-precision")
        values, trace = trace_draft_graph(graph, inputs, names=required, progress=progress)
        reproduction = []
        for name, key in (("draft_hidden", "expected_hidden"), ("draft_top1", "expected_top1")):
            original = _read_array(fixture, reference[key])
            if original.dtype != values[name].dtype or original.shape != values[name].shape or original.tobytes() != values[name].tobytes():
                raise ValueError("current v3 trace does not reproduce retained " + name + " byte-for-byte")
            reproduction.append({"name": name, "status": "EXACT_MATCH", "sha256": reference[key]["sha256"]})
        if (sha256_file(prepared / "prepared.json") != prepared_digest
                or sha256_file(fixture / "report.json") != fixture_digest
                or any((path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns) != identity
                       for path, identity in identities.items())):
            raise ValueError("draft trace source files changed during execution")
        selected_rows = {row["name"]: row for row in trace["tensors"]}
        descriptors = []
        for index, name in enumerate(requested):
            descriptor = _write_tensor(root, name, values[name], index)
            descriptors.append({**selected_rows[name], **descriptor})
        result.update(trace)
        result.update(status="REFERENCE_TRACE_COMPLETE", tensors=descriptors,
                      golden_reproduction=reproduction, total_bytes=sum(row["bytes"] for row in descriptors))
        save_json(root / "report.json", result)
        return result
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save_json(root / "report.json", result)
        raise

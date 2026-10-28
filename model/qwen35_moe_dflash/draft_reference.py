"""Standalone PyTorch draft diagnostics using the exact prepared FP16 weights.

The complete draft and the target embedding/head are loaded; no target decoder
is constructed. Synthetic target features test computation, not proposal quality.
The primitive reference follows the existing GE FP16 boundaries, independently
of GE graph evaluation, and is not the original BF16/FP32 upstream oracle.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .artifacts import checked_file, output_path, save_json
from .config import Qwen35DFlashConfig
from .ge_export import configurations, weight_identity
from .hashing import sha256_file


def _validated_prepared(prepared_dir):
    root = Path(prepared_dir).expanduser().resolve()
    manifest = root / "prepared.json"
    plan = json.loads(manifest.read_text())
    if (plan.get("artifact_kind") != "qwen35-draft-ge-prepared"
            or plan.get("status") != "PREPARED" or type(plan.get("tiny")) is not bool
            or plan.get("precision") != "float16" or plan.get("quantization") is not None):
        raise ValueError("expected a successful standalone FP16 draft preparation")
    if plan.get("draft_config") != configurations(tiny=plan["tiny"])[1]:
        raise ValueError("prepared draft configuration differs from the locked topology")
    config = Qwen35DFlashConfig.from_dict(plan["draft_config"])
    sequence = plan.get("sequence_length")
    if (type(sequence) is not int or sequence < 2
            or sequence + config.block_size > config.max_position_embeddings):
        raise ValueError("prepared draft sequence length is invalid")
    shapes = {"draft." + name: shape for name, shape in config.required_tensor_shapes().items()}
    shapes.update({"target." + name: (config.vocab_size, config.hidden_size)
                   for name in ("embed_tokens.weight", "lm_head.weight")})
    records = plan.get("weights", {})
    if set(records) != set(shapes):
        raise ValueError("prepared weights must cover exactly the draft, target embedding and full head")
    if plan.get("source", {}).get("fp16_payload_sha256") != weight_identity(records):
        raise ValueError("prepared source identity differs from weight records")
    paths = {}
    for key, shape in shapes.items():
        record = records[key]
        if (record.get("shape") != list(shape) or record.get("dtype") != "float16"
                or record.get("bytes") != math.prod(shape) * 2):
            raise ValueError(f"invalid prepared FP16 weight descriptor: {key}")
        paths[key] = checked_file(root, record)
    return root, plan, config, paths


def _load_graph(config, paths):
    # This module does not import Transformers or TorchAir. Setting the public
    # PyTorch autoload control before importing torch also avoids initializing
    # a vendor NPU backend in a host-only reference process.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from torch import nn
    from .graphs import DraftGraph
    from .model import DFlashDraftModel
    from .ops import TorchDFlashOps, _repeat_kv

    class FP16Primitives(TorchDFlashOps):
        def rms_norm(self, x, weight, eps):
            return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * weight

        def attention(self, query, key, value, attention_mask, scale, key_value_groups):
            key = _repeat_kv(key, key_value_groups)
            value = _repeat_kv(value, key_value_groups)
            scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            if attention_mask is not None:
                scores = torch.where(attention_mask, scores,
                                     torch.tensor(-float("inf"), dtype=scores.dtype, device=scores.device))
            return torch.matmul(torch.softmax(scores, -1), value)

    class FP16Rotary(nn.Module):
        def __init__(self, inverse):
            super().__init__()
            self.register_buffer("inv_freq", inverse.half(), persistent=False)

        def forward(self, positions, dtype):
            frequencies = positions.half().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
            angles = torch.cat((frequencies, frequencies), dim=-1)
            return angles.cos(), angles.sin()

    # Meta construction avoids allocating and initializing another copy of the
    # 2+ GiB parameter set. Private mmap storage remains writable to PyTorch,
    # while any accidental write cannot alter the hash-locked source file.
    draft = DFlashDraftModel(config, ops=FP16Primitives(), device="meta", dtype=torch.float16)
    draft.rotary = FP16Rotary(draft.rotary.inv_freq)
    embedding = nn.Embedding(config.vocab_size, config.hidden_size, device="meta", dtype=torch.float16)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="meta", dtype=torch.float16)
    graph = DraftGraph(draft, embedding, head)
    parameters = {"draft." + name: (draft, name, value.shape)
                  for name, value in draft.named_parameters()}
    parameters["target.embed_tokens.weight"] = (embedding, "weight", embedding.weight.shape)
    parameters["target.lm_head.weight"] = (head, "weight", head.weight.shape)
    if set(paths) != set(parameters):
        raise ValueError("reference parameter mapping does not cover the prepared draft")
    for key, (owner, name, shape) in parameters.items():
        path = paths[key]
        values = np.memmap(path, mode="r", dtype="<f2")
        for start in range(0, values.size, 1 << 20):
            if not np.isfinite(values[start:start + (1 << 20)]).all():
                raise ValueError(f"nonfinite prepared draft weight: {key}")
        del values
        tensor = torch.from_file(str(path), shared=False, size=math.prod(shape), dtype=torch.float16).reshape(shape)
        parent_name, _, attribute = name.rpartition(".")
        parent = owner.get_submodule(parent_name) if parent_name else owner
        setattr(parent, attribute, nn.Parameter(tensor, requires_grad=False))
    graph.config = config
    return graph.eval()


def load_draft_reference(prepared_dir):
    """Load only prepared draft/embedding/head parameters on CPU using mmap."""
    _, _, config, paths = _validated_prepared(prepared_dir)
    return _load_graph(config, paths)


def _write_tensor(root, name, array, index):
    path = root / f"{name}.bin"
    with path.open("xb") as stream:
        stream.write(array.tobytes(order="C"))
    return {"index": index, "name": name, "shape": list(array.shape), "dtype": array.dtype.name,
            "path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def prepare_draft_reference(prepared_dir, output_dir, *, seed=17, valid_length=None, run_reference=True):
    """Write three deterministic diagnostic inputs and complete draft goldens.

    The features are synthetic and explicitly do not establish target feature
    compatibility, language quality, or DFlash acceptance. Exact Top1 comparison
    and hidden-error diagnostics use the complete configured vocabulary/layers.
    """
    prepared, plan, config, paths = _validated_prepared(prepared_dir)
    sequence = plan["sequence_length"]
    valid_length = min(8, sequence - 1) if valid_length is None else valid_length
    if type(valid_length) is not int or not 1 <= valid_length <= sequence:
        raise ValueError("valid_length must be within the nonempty right-padded prefix")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    features = rng.normal(0, .2, (1, sequence, config.feature_size)).astype("<f2")
    ids = rng.integers(0, config.vocab_size, (1, sequence), dtype=np.int32)
    ids[:, valid_length:] = 0
    mask = np.zeros((1, sequence), dtype="<i4")
    mask[:, :valid_length] = 1
    inputs = [("target_features", features), ("input_ids", ids), ("attention_mask", mask)]
    result = {"schema_version": 1, "artifact_kind": "qwen35-draft-reference", "status": "INPUTS_PREPARED",
              "prepared_manifest_sha256": sha256_file(prepared / "prepared.json"),
              "source": plan["source"], "draft_config": plan["draft_config"], "tiny": plan["tiny"],
              "sequence_length": sequence, "valid_length": valid_length, "seed": seed,
              "target_features_source": "SYNTHETIC_DIAGNOSTIC_NOT_TARGET_EXECUTION",
              "target_decoder": "NOT_CONSTRUCTED", "state_policy": "prefix-recompute",
              "reference": "PYTORCH_FP16_PRIMITIVES", "precision": "float16", "device": "cpu",
              "numerical_acceptance": {"top1_mismatches": 0, "hidden_threshold": "NOT_DEFINED"},
              "scope": "Complete draft computation diagnostic; not BF16 source quality, acceptance or speed evidence.",
              "inputs": [_write_tensor(root, name, array, i) for i, (name, array) in enumerate(inputs)],
              "expected_top1": None, "expected_hidden": None}
    try:
        if run_reference:
            graph = _load_graph(config, paths)
            import torch
            captured = []
            handle = graph.draft.norm.register_forward_hook(
                lambda module, args, output: captured.append(output[:, 1:, :].detach()))
            logits_finite = []
            head_handle = graph.lm_head.register_forward_hook(
                lambda module, args, output: logits_finite.append(bool(torch.isfinite(output).all())))
            try:
                with torch.inference_mode():
                    top1 = graph(*(torch.from_numpy(array) for _, array in inputs))
            finally:
                handle.remove()
                head_handle.remove()
            if len(captured) != 1 or logits_finite != [True] or not torch.isfinite(captured[0]).all():
                raise FloatingPointError("nonfinite or missing complete draft reference output")
            result["expected_top1"] = _write_tensor(root, "draft_top1", top1.numpy().astype("<i4"), 0)
            result["expected_hidden"] = _write_tensor(root, "draft_hidden", captured[0].numpy().astype("<f2"), 1)
            result.update(status="REFERENCE_COMPLETE", torch_version=torch.__version__,
                          draft_layers_executed=len(graph.draft.layers), full_vocabulary_rows=config.vocab_size)
        save_json(root / "report.json", result)
        return result
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save_json(root / "report.json", result)
        raise

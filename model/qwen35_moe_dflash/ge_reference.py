"""Optional PyTorch reference for the exact weights of a prepared GE tiny pair.

Run this in a separate CPU reference process. Direct GE preparation/export never
imports this module or requires PyTorch, torch_npu, or TorchAir.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .artifacts import checked_file, save_json
from .hashing import sha256_file


def reference_tiny(prepared_dir, output_path, prompt=(1, 2, 3), max_new_tokens=4):
    """Compare ordinary and DFlash CPU generation with prepared NumPy weights.

    The NumPy and PyTorch random generators produce different checkpoints even
    for the same seed. Reading every hash-locked prepared payload, rather than
    reseeding ``reference.tiny.create``, makes board comparison meaningful.
    """
    from .ge_export import configurations, weight_identity

    prepared = Path(prepared_dir).expanduser().resolve()
    manifest_path = prepared / "prepared.json"
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_target, expected_draft = configurations(tiny=True)
    if (plan.get("artifact_kind") != "qwen35-ge-prepared" or plan.get("status") != "PREPARED"
            or plan.get("tiny") is not True or plan.get("precision") != "float16"
            or plan.get("quantization") is not None):
        raise ValueError("reference-tiny requires an unquantized FP16 prepared GE tiny pair")
    if (plan.get("target_config") != expected_target or plan.get("draft_config") != expected_draft):
        raise ValueError("prepared configuration differs from the supported 64-vocabulary/16-hidden tiny fixture")
    if (not isinstance(plan.get("source"), dict)
            or plan["source"].get("kind") != "ge-tiny-numpy-random"
            or plan["source"].get("not_upstream_weights") is not True):
        raise ValueError("prepared tiny weight identity is missing or invalid")
    sequence = plan.get("sequence_length")
    if (type(sequence) is not int or sequence < 2
            or sequence + expected_draft["dflash_config"]["block_size"] > expected_draft["max_position_embeddings"]):
        raise ValueError("prepared sequence length is outside the tiny fixture capacity")
    records = plan.get("weights")
    if not isinstance(records, dict) or not records:
        raise ValueError("prepared weight manifest is empty")
    if plan["source"].get("fp16_payload_sha256") != weight_identity(records):
        raise ValueError("prepared source identity differs from its actual FP16 weight records")
    paths = {}
    for key, record in records.items():
        shape = record.get("shape")
        if (not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape)
                or record.get("dtype") != "float16" or record.get("bytes") != math.prod(shape) * 2):
            raise ValueError(f"invalid prepared FP16 weight descriptor: {key}")
        paths[key] = checked_file(prepared, record)

    # Must precede the first torch/Transformers import of this command. This
    # optional reference path does not initialize the AIR/NPU Python backend.
    from .backends import configure_reference_backend
    configure_reference_backend("cpu")
    import numpy as np
    import torch
    from reference.tiny import create
    from .precision import prepare_fp16_pair
    from .runtime import TorchExecutor, generate

    target, draft = create()
    parameters = {f"target.{name}": parameter for name, parameter in target.language_model.named_parameters()}
    parameters["target.lm_head.weight"] = target.lm_head.weight
    parameters.update({f"draft.{name}": parameter for name, parameter in draft.draft.named_parameters()})
    if set(records) != set(parameters):
        missing = sorted(set(parameters) - set(records))
        extra = sorted(set(records) - set(parameters))
        raise ValueError(f"prepared weights do not cover the tiny reference exactly: missing={missing}, extra={extra}")
    with torch.no_grad():
        for key, parameter in parameters.items():
            record = records[key]
            if tuple(record["shape"]) != tuple(parameter.shape):
                raise ValueError(f"prepared weight shape differs from tiny reference: {key}")
            array = np.fromfile(paths[key], dtype="<f2").reshape(parameter.shape)
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite prepared tiny weight: {key}")
            parameter.copy_(torch.from_numpy(array))
    target, draft, precision_audit = prepare_fp16_pair(target, draft, sequence)
    executor = TorchExecutor(target, draft, sequence, pad_token_id=0)
    prompt = list(prompt)
    ordinary = generate(executor, prompt, ordinary=True, max_new_tokens=max_new_tokens, eos_token_ids=())
    dflash = generate(executor, prompt, max_new_tokens=max_new_tokens, eos_token_ids=())
    match = (ordinary.token_ids, ordinary.stop_reason) == (dflash.token_ids, dflash.stop_reason)
    result = {"status": "PASS" if match else "FAIL", "evidence_kind": "framework-reference",
              "source": plan["source"], "prepared_manifest_sha256": sha256_file(manifest_path),
              "prompt_token_ids": prompt, "device": str(executor.device),
              "precision": "torch.float16", "quantization": None, "precision_audit": precision_audit,
              "sequence_length": sequence, "max_new_tokens": max_new_tokens,
              "ordinary": ordinary.to_dict(), "dflash": dflash.to_dict()}
    save_json(output_path, result)
    if not match:
        raise RuntimeError("ordinary and DFlash reference tokens differ for prepared GE weights")
    return result

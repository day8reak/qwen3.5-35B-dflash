"""Production export: PyTorch -> TorchAir AIR (including external weights)."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import torch
from .artifacts import output_path, files_under, save_json, working_directory
from .precision import prepare_fp16_pair


def require_torchair():
    # Fail before opening/allocating the 35B checkpoint.
    try:
        module = importlib.import_module("torchair")
    except ImportError as error:
        raise RuntimeError("AIR export requires a matching TorchAir/torch_npu/CANN environment; "
                           "CPU PyTorch and ONNX cannot produce the production AIR") from error
    if not callable(getattr(module, "dynamo_export", None)):
        raise RuntimeError("installed TorchAir has no dynamo_export API")
    return module


def export_air(target, draft, output_dir, sequence_length, *, torchair_module=None,
               source_identity=None, dispatch="static", quantization=None):
    """Export two stateless batch-one graphs under the explicit FP16 policy.

    static dispatch avoids data-dependent NonZero lengths. It computes all
    experts and is a correctness baseline, not a sparse-performance claim.
    """
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

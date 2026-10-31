"""Optional CPU ONNX mathematical cross-check. Never the production OMC input."""
from pathlib import Path
import json
import torch

from qwen35_moe_dflash.reference.weights import sha256_file
from qwen35_moe_dflash.toolchain.export_moe import enable_export_dispatch


def export_graphs(target, draft, output_dir, sequence_length):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    if sequence_length < 2:
        raise ValueError("sequence_length must exceed one")
    enable_export_dispatch(target.language_model)
    ids = torch.zeros((1, sequence_length), dtype=torch.int32)
    mask = torch.zeros_like(ids)
    ids[0, :2] = torch.tensor([1, 2])
    mask[0, :2] = 1
    features = torch.zeros((1, sequence_length, target.config.feature_size), dtype=torch.float32)
    records = []
    definitions = [
        ("target", target, (ids, mask), ["input_ids", "attention_mask"],
         ["target_top1", "target_features"]),
        ("draft", draft, (features, ids, mask), ["target_features", "input_ids", "attention_mask"],
         ["draft_top1"]),
    ]
    for name, module, args, inputs, outputs in definitions:
        folder = root / name
        folder.mkdir()
        path = folder / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                module, args, str(path), input_names=inputs, output_names=outputs,
                opset_version=17, dynamo=False, do_constant_folding=False,
                external_data=True,
            )
        import onnx
        onnx.checker.check_model(str(path))
        records.append({
            "name": name, "onnx": str(path.relative_to(root)),
            "inputs": inputs, "outputs": outputs,
            "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                       "sha256": sha256_file(p)}
                      for p in sorted(folder.iterdir()) if p.is_file()],
        })
    manifest = {
        "schema_version": 1, "artifact_kind": "qwen35-host-onnx-reference-pair",
        "status": "EXPORTED", "target_compilation": "PENDING",
        "device_validation": "PENDING", "sequence_length": sequence_length,
        "draft_width": target.config.block_size - 1,
        "feature_width": target.config.feature_size, "vocab_size": target.config.vocab_size,
        "precision": str(next(target.parameters()).dtype),
        "state_policy": "recompute", "graphs": records,
        "export_dispatch": "all experts enumerated; dynamic token routing; zero-weight sentinel",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

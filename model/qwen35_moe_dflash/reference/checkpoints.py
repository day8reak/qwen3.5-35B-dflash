"""Read-only checkpoint audit and complete payload identity; no downloads."""
import json
from pathlib import Path
from safetensors import safe_open
from qwen35_moe_dflash.common.resources import repository_root
from qwen35_moe_dflash.reference.weights import audit_dflash_checkpoint, sha256_file


def audit_checkpoints(target_dir, draft_dir, *, hash_weights=True):
    target, draft = Path(target_dir).resolve(), Path(draft_dir).resolve()
    lock = json.loads((repository_root() / "specs/upstream.lock.json").read_text())
    errors, records = [], []
    for root, pin in zip([target, draft], lock):
        for name, expected in pin["files"].items():
            if "sha256" not in expected: continue
            path = root / name
            if not path.is_file() or sha256_file(path) != expected["sha256"]:
                errors.append(f"{pin['repo']}: {name} differs from the locked upstream revision")
    if errors: return {"status": "FAIL", "errors": errors}
    draft_audit = audit_dflash_checkpoint(draft, verify_model_hash=hash_weights)
    errors.extend(draft_audit["errors"])
    index = json.loads((target / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    for filename in sorted(set(mapping.values())):
        path = (target / filename).resolve()
        if target not in path.parents or not path.is_file():
            errors.append(f"target shard missing or outside checkpoint: {filename}"); continue
        expected = {k for k, v in mapping.items() if v == filename}
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
            if keys != expected:
                errors.append(f"shard key set differs from index: {filename}")
            for key in keys:
                if (key.startswith("model.language_model.") or key == "lm_head.weight"):
                    if checkpoint.get_slice(key).get_dtype() != "BF16":
                        errors.append(f"target text parameter is not original BF16: {key}")
        records.append({"file": filename, "bytes": path.stat().st_size,
                        "sha256": sha256_file(path) if hash_weights else None})
    return {"schema_version": 1, "status": "FAIL" if errors else "PASS", "errors": errors,
            "target": {"repo": lock[0]["repo"], "revision": lock[0]["revision"], "shards": records},
            "draft": {"repo": lock[1]["repo"], "revision": lock[1]["revision"],
                      "sha256": draft_audit["model_sha256"], "parameters": draft_audit["parameter_count"]},
            "weights_hashed": hash_weights}

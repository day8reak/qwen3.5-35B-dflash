"""Pair-level direct graph definitions vs the preserved PyTorch generation path."""
import os

import numpy as np
import torch

from qwen35_moe_dflash.ge_export import prepare_pair, build_pair
from qwen35_moe_dflash.ge_reference import reference_tiny
from qwen35_moe_dflash import runtime
from reference.ge_torch import evaluate_torch


def test_direct_pair_matches_reference_through_dflash_scheduler(tmp_path, monkeypatch):
    from qwen35_moe_dflash import backends
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    # reference_tiny normally owns a fresh CLI process. Restore its persistent
    # backend policy when invoking it inside the shared pytest process.
    monkeypatch.setattr(backends, "_REFERENCE_DEVICE", backends._REFERENCE_DEVICE)
    monkeypatch.setattr(utils, "is_torch_npu_available", utils.is_torch_npu_available)
    monkeypatch.setattr(imports, "is_torch_npu_available", imports.is_torch_npu_available)
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD", ""))
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    prepared = tmp_path / "prepared"
    plan = prepare_pair(prepared, 8, tiny=True)
    target_ir, draft_ir = build_pair(plan["target_config"], plan["draft_config"], 8)
    weights = {key: torch.from_numpy(np.fromfile(prepared / item["path"], dtype="<f2").reshape(item["shape"]))
               for key, item in plan["weights"].items()}
    original = runtime.TorchExecutor
    counts = {"target": 0, "draft": 0}

    class ComparedExecutor(original):
        def target(self, prefix):
            expected = super().target(prefix)
            top1, features = evaluate_torch(target_ir, {"input_ids": self.ids, "attention_mask": self.mask}, weights)
            assert top1[0].tolist() == expected
            torch.testing.assert_close(features, self.features)
            self.direct_features = features
            counts["target"] += 1
            return top1[0].tolist()

        def draft(self):
            expected = super().draft()
            actual = evaluate_torch(draft_ir, {"target_features": self.direct_features,
                                              "input_ids": self.ids, "attention_mask": self.mask}, weights)[0]
            assert actual[0].tolist() == expected
            counts["draft"] += 1
            return actual[0].tolist()

    monkeypatch.setattr(runtime, "TorchExecutor", ComparedExecutor)
    result = reference_tiny(prepared, tmp_path / "reference.json", max_new_tokens=4)
    assert result["status"] == "PASS"
    assert counts["target"] >= 4 and counts["draft"] >= 1
    assert result["ordinary"]["token_ids"] == result["dflash"]["token_ids"]

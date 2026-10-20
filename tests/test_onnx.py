"""Cross-input checks catch frozen routing, lengths, masks and feature alignment."""
import numpy as np
import pytest
import torch
from reference.tiny import create
from qwen35_moe_dflash.onnx_reference import export_graphs


def test_tiny_onnx_new_routes_lengths_and_token_ids(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    target, draft = create()
    import copy
    unmodified_target = copy.deepcopy(target)
    export_graphs(target, draft, tmp_path / "onnx", 8)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(tmp_path / "onnx" / name / "model.onnx"),
                             sess_options=options, providers=["CPUExecutionProvider"])
        for name in ["target", "draft"]
    ]
    for tokens in [[4], [8, 7, 2], [5, 2, 31, 50, 11], list(range(1, 9))]:
        ids = torch.zeros((1, 8), dtype=torch.int32)
        mask = torch.zeros_like(ids)
        ids[0, :len(tokens)] = torch.tensor(tokens)
        mask[0, :len(tokens)] = 1
        with torch.inference_mode():
            expected_top1, features = unmodified_target(ids, mask)
            expected_draft = draft(features, ids, mask)
        top1, actual_features = sessions[0].run(
            None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
        np.testing.assert_array_equal(top1[:, :len(tokens)], expected_top1.numpy()[:, :len(tokens)])
        np.testing.assert_allclose(actual_features[:, :len(tokens)], features.numpy()[:, :len(tokens)],
                                   atol=1e-5, rtol=1e-4)
        actual_draft = sessions[1].run(
            None, {"target_features": actual_features, "input_ids": ids.numpy(),
                   "attention_mask": mask.numpy()})[0]
        np.testing.assert_array_equal(actual_draft, expected_draft.numpy())

"""Original Cpro-style target inputs over the functional PyTorch state ABI.

This wrapper follows the supplied model's embedding and selected-logit entry
points. It does not claim compatibility with the original AIR's paged/NZ
RefData tensors. All tokens are valid, contiguous, and causally masked by the
target's attention adapter.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import FusedTargetModel, FusedTargetOutput
from .session import TargetState


class OriginalCproTargetWrapper(nn.Module):
    """Return FP32 logits for selected positions plus features and new state.

    ``output_pos=-1`` selects the final token, ``None`` selects every token.
    Positions index this invocation's sequence, not the accumulated cache.
    ``past_key_values`` is a TargetState; the returned ``state`` is passed to
    the next invocation. Caller-owned state is never updated in place.
    """

    def __init__(self, target: FusedTargetModel):
        super().__init__()
        if not isinstance(target, FusedTargetModel) or target.quantization is None:
            raise TypeError("OriginalCproTargetWrapper requires an original-quantized FusedTargetModel")
        self.target = target
        self.eval()

    @staticmethod
    def _positions(value, expected, name, shapes):
        if value is None:
            return
        if (not isinstance(value, Tensor) or value.dtype not in (torch.int32, torch.int64)
                or tuple(value.shape) not in shapes or value.device != expected.device
                or not torch.equal(value.reshape(-1).long(), expected)):
            raise ValueError(f"{name} must match the contiguous positions of this TargetState")

    @staticmethod
    def _length(value, length, device, name, dtypes):
        if value is None:
            return
        if (not isinstance(value, Tensor) or value.dtype not in dtypes
                or tuple(value.shape) != (1,) or value.device != device
                or value.item() != length):
            raise ValueError(f"{name} must equal the unpadded input sequence length")

    @torch.inference_mode()
    def forward(self, input_ids=None, *, inputs_embeds=None, embed_scale=None,
                past_key_values=None, output_pos=-1, position_ids=None,
                new_kv_cache_pos=None, effective_length=None, allQLen=None) -> FusedTargetOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if embed_scale is not None and inputs_embeds is None:
            raise ValueError("embed_scale requires inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, TargetState):
            raise TypeError("past_key_values must be functional TargetState, not original AIR RefData/NZ tensors")
        value = input_ids if inputs_embeds is None else inputs_embeds
        rank = 2 if inputs_embeds is None else 3
        if (not isinstance(value, Tensor) or value.ndim != rank or value.shape[0] != 1
                or value.shape[1] < 1 or value.device != self.target.device):
            raise ValueError("input must be a nonempty batch-one sequence on the target device")
        length = value.shape[1]
        start = 0 if past_key_values is None else past_key_values.position
        expected = torch.arange(start, start + length, dtype=torch.int64, device=value.device)
        self._positions(position_ids, expected, "position_ids", {(1, length)})
        self._positions(new_kv_cache_pos, expected, "new_kv_cache_pos", {(length,), (1, length)})
        self._length(effective_length, length, value.device, "effective_length",
                     (torch.int16, torch.int32, torch.int64))
        self._length(allQLen, length, value.device, "allQLen", (torch.int32, torch.int64))
        if inputs_embeds is None:
            return self.target(input_ids, past_key_values, output_pos=output_pos, return_logits=True)
        return self.target.forward_embeddings(inputs_embeds, past_key_values,
            embed_scale=embed_scale, output_pos=output_pos, return_logits=True)

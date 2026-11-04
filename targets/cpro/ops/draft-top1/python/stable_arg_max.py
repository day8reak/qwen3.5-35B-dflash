"""Portable Torch schema/Meta and explicit TorchAir lowering for StableArgMax.

CPU dispatch is a reference only. There is intentionally no PrivateUse1/device
fallback. Import and register_ge() before exporting; a device execution adapter
can later implement this same schema without changing model call sites.
"""
import torch

TORCH_NAME = 'custom_ops::stable_arg_max'
GE_TYPE = 'StableArgMax'
KERNEL_ENTRY = 'stable_arg_max'
_LIBRARY = torch.library.Library('custom_ops', 'FRAGMENT')
_LIBRARY.define('stable_arg_max(Tensor x) -> Tensor')
_GE_MODULE = None


def _validate(x):
    if x.dtype != torch.float16 or x.layout != torch.strided or not x.is_contiguous():
        raise ValueError('StableArgMax requires contiguous strided FP16 input')
    if x.dim() < 1 or x.dim() > 8:
        raise ValueError('StableArgMax requires rank 1..8')
    for dim in x.shape:
        torch._check(dim > 0, lambda: 'StableArgMax rejects empty dimensions')
    torch._check(x.numel() <= 2147483647, lambda: 'StableArgMax element count exceeds INT32 range')


@torch.library.register_fake(TORCH_NAME)
def _meta(x):
    _validate(x)
    # Metadata only: never inspect input values or substitute a symbolic dim.
    return x.new_empty(x.shape[:-1], dtype=torch.int32)


def _cpu_reference(x):
    _validate(x)
    if not bool(torch.isfinite(x).all()):
        raise ValueError('StableArgMax contract requires finite FP16 values')
    return torch.argmax(x.float(), dim=-1).to(torch.int32)


_LIBRARY.impl('stable_arg_max', _cpu_reference, 'CPU')


def stable_arg_max(x):
    """Top1 over the last dimension, INT32 output of shape x.shape[:-1]."""
    return torch.ops.custom_ops.stable_arg_max.default(x)


def register_ge(torchair):
    """Register the converter once in the selected TorchAir export process."""
    global _GE_MODULE
    op = torch.ops.custom_ops.stable_arg_max.default
    if _GE_MODULE is torchair:
        return op
    if _GE_MODULE is not None:
        raise RuntimeError('StableArgMax is already bound to another TorchAir module')

    @torchair.register_fx_node_ge_converter(op)
    def convert(x, meta_outputs=None):
        return torchair.ge.custom_op(GE_TYPE, inputs={'x': x}, attrs={}, outputs=['indices'])

    _GE_MODULE = torchair
    return op


class StableArgMax(torch.nn.Module):
    def forward(self, x):
        return stable_arg_max(x)

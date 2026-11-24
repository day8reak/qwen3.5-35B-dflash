"""PyTorch schema, FakeTensor rule and TorchAir lowering for the CPro SiLU kernel."""
from itertools import count

import torch
import torch.nn.functional as F

from .silu_patch_air import GE_TYPE, port_names, vector_desc

torch.library.define('my_ops::silu_patch', '(Tensor x) -> Tensor')


def _check_vector(x):
    if x.dtype != torch.float16 or x.ndim != 1 or not x.is_contiguous():
        raise ValueError('silu_patch expects a contiguous FP16 vector')


@torch.library.register_fake('my_ops::silu_patch')
def _fake(x):
    _check_vector(x)
    return x.new_empty(x.shape, dtype=torch.float32)


@torch.library.impl('my_ops::silu_patch', 'CPU')
def _cpu_reference(x):
    _check_vector(x)
    return F.silu(x.float())


def silu_patch(x):
    """Call the one-dimensional kernel and restore the logical shape, keeping FP32."""
    flat = x.to(torch.float16).contiguous().view(-1)
    return torch.ops.my_ops.silu_patch.default(flat).reshape(x.shape)


def silu_same_dtype(x):
    return silu_patch(x).to(x.dtype)


def register_ge(torchair):
    sequence = count()

    @torchair.register_fx_node_ge_converter(torch.ops.my_ops.silu_patch.default)
    def convert(x, meta_outputs=None):
        _check_vector(x.meta)
        length = int(x.meta.shape[0])  # The native CPro exporter uses dynamic=False.
        z = torchair.ge.custom_op(GE_TYPE, inputs={'x': x}, outputs=['z'],
                                  node_name=f'silu_patch_n{length}_{next(sequence)}')
        z.set_meta(x.meta.new_empty((length,), dtype=torch.float32))
        vector_desc(z.node.input_desc[0], length)
        vector_desc(z.desc, length)
        port_names(z.node, 'x', 'z')
        return z

"""Host-only bridge to frozen vendor simulation, never an AIR/NPU operator.

DOPT's Cython QLinear cannot be inlined by Dynamo. An opaque reference call
keeps the actual vendor quantizer math while FP16 lowering handles the rest
of the model. Its registry is process-local; exported reference programs are
not portable artifacts. AIR export explicitly rejects this operator.
"""
import uuid
import torch
from torch import nn

_VENDOR_MODULES = {}


@torch.library.custom_op("qwen35::dopt_reference_linear", mutates_args=())
def dopt_reference_linear(x: torch.Tensor, weight: torch.Tensor, key: str) -> torch.Tensor:
    module = _VENDOR_MODULES[key]
    with torch.no_grad():
        result = torch.func.functional_call(module, {"weight": weight}, (x,), strict=False)
    if not torch.isfinite(result).all():
        raise FloatingPointError("DOPT reference quantized matmul produced nonfinite values")
    return result


@dopt_reference_linear.register_fake
def _fake(x, weight, key):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


class DoptReferenceLinear(nn.Module):
    def __init__(self, vendor_module):
        super().__init__()
        self.quant_op = vendor_module
        self.key = uuid.uuid4().hex
        _VENDOR_MODULES[self.key] = vendor_module

    def forward(self, x):
        return dopt_reference_linear(x, self.quant_op.weight, self.key)


def bridge_reference(model, matrix_names):
    for name in matrix_names:
        module = model.get_submodule(name)
        if not hasattr(module, "quant_op"):
            raise ValueError(f"{name} is not a DOPT calibrated layer")
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, DoptReferenceLinear(module))
    return model.eval()

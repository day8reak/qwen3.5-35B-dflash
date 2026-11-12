"""Process-local GE lowering for the integer CacheUpdate index experiment."""
from contextlib import contextmanager
import functools
import importlib
import inspect

import torch

from .source_regions import definition


@contextmanager
def cache_ge_integer_mul(enabled, block_size=32):
    """Lower only this candidate's INT32 block * 32 with complete metadata.

    Some mul.Tensor converters call dtype_promote on a newly created GE Mul
    before its output metadata is set. That inserts a Cast from an unresolved
    output, subsequently serialized as FP32. Build the typed integer operation
    at conversion time; never patch serialized descriptors or relax AIR checks.
    This context is used only for a single Cache candidate in its export worker.
    """
    if not enabled:
        yield None
        return
    if type(block_size) is not int or block_size != 32:
        raise ValueError('integer cache GE adapter requires validated block_size=32')
    module = importlib.import_module('torchair._ge_concrete_graph.fx2ge_converter')
    ge = importlib.import_module('torchair._ge_concrete_graph.ge_apis')
    from torchair.ge import DataType
    op = torch.ops.aten.mul.Tensor
    getter = getattr(module, '_get_converter', None)
    registry = getattr(module, '_CONVERTERS', None)
    if not callable(getter) or not isinstance(registry, dict):
        raise ValueError('installed TorchAir converter registry differs; inspect fx2ge_converter.py')
    original = getter(op)
    if not callable(original):
        raise ValueError('selected TorchAir has no aten.mul.Tensor converter')
    signature = inspect.signature(original)
    if not {'self', 'other', 'meta_outputs'} <= set(signature.parameters):
        raise ValueError('selected mul.Tensor converter signature differs: ' + str(signature))
    _, source = definition(inspect.unwrap(original))
    evidence = dict(calls=0, restored=False, original_converter=source,
        policy='CACHE_CANDIDATE_ONLY; INT32_BLOCK_TIMES_32; META_BEFORE_CONSUMERS',
        block_size=block_size)

    @functools.wraps(original)
    def adapted(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        block, factor, output_spec = (values.get(k) for k in ('self', 'other', 'meta_outputs'))
        meta = getattr(block, 'meta', None)
        # The registry wrapper may receive a raw FakeTensor before it turns it
        # into TensorSpec. Support both representations, with the same checks.
        raw_meta = isinstance(output_spec, torch.Tensor)
        output_dtype = getattr(output_spec, 'dtype', None)
        output_shape = list(output_spec.shape) if raw_meta else getattr(output_spec, 'size', None)
        if (getattr(block, 'dtype', None) != DataType.DT_INT32
                or type(factor) is not int or factor != block_size
                or output_dtype != (torch.int32 if raw_meta else DataType.DT_INT32)
                or not isinstance(meta, torch.Tensor) or meta.dtype != torch.int32
                or meta.ndim != 1 or output_shape != list(meta.shape)):
            raise ValueError('cache GE adapter expects INT32 [indices] * scalar 32 with matching INT32 FX output')
        if evidence['calls']:
            raise ValueError('integer cache candidate contains more than one mul.Tensor')
        factor_tensor = ge.Const(factor, dtype=DataType.DT_INT32)
        product = ge.Mul(block, factor_tensor)
        # This is metadata for a newly emitted integer operation, derived from
        # the validated FX operands. No existing float value/bytes are relabeled.
        product.set_meta(output_spec if raw_meta else meta * factor)
        if product.dtype != DataType.DT_INT32:
            raise ValueError('GE Mul did not retain INT32 output metadata')
        evidence['calls'] += 1
        evidence['multiply'] = dict(block=block.tensor, factor=factor_tensor.tensor,
                                    product=product.tensor, dtype='int32')
        return product

    existed, previous = op in registry, registry.get(op)
    had_attr, previous_attr = hasattr(op, '_ge_converter'), getattr(op, '_ge_converter', None)
    try:
        registry[op] = adapted
        op._ge_converter = adapted
        if getter(op) is not adapted:
            raise ValueError('selected TorchAir did not select the integer cache converter adapter')
        yield evidence
        if evidence['calls'] != 1:
            raise ValueError('candidate requires exactly one adapted integer cache multiplication')
    finally:
        if existed:
            registry[op] = previous
        else:
            registry.pop(op, None)
        if had_attr:
            op._ge_converter = previous_attr
        else:
            delattr(op, '_ge_converter')
        evidence['restored'] = getter(op) is original

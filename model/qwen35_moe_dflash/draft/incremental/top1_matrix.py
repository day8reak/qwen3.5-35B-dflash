"""Bounded Top1 placement matrix: isolate shape and composition effects."""
import math

import numpy as np

from qwen35_moe_dflash.graph.fused.ir import Graph
from qwen35_moe_dflash.graph.type_contracts import GE_DTYPE_IDS


GEOMETRIES = (
    ('r1_d512', (1, 512), 1),
    ('r15_d485', (15, 485), 1),
    ('r15_d512', (15, 512), 1),
    ('r485_d512', (485, 512), 1),
    ('r7275_d512', (7275, 512), 1),
    ('r7680_d485', (7680, 485), 1),
    ('b15_r485_d512', (15, 485, 512), 2),
    ('d512_r7275_axis0', (512, 7275), 0),
)
PRIMITIVES = {f'top1_{kind}_{label}_fp16': (kind, shape, axis)
              for label, shape, axis in GEOMETRIES for kind in ('argmax', 'max')}
# Full-width ArgMax is already observed on CPU. Test the unobserved value
# reduction at that width; its input stays full vocabulary.
PRIMITIVES['top1_max_r15_d248320_fp16'] = ('max', (15, 248320), 1)
COMPOSITIONS = ('top1_ext2_tiled_fp16', 'top1_ext2_tiled_flat_fp16',
                'top1_ext2_tiled_transposed_fp16', 'top1_ext2_tiled_485_fp16')
MATRIX_CASES = (*PRIMITIVES, *COMPOSITIONS)


def fixture(shape, axis):
    """Identical canonical rows across rank/transpose controls; all finite.

    Nonfinite Top1 semantics remain covered by the complete-vocabulary cases.
    Finite primitive Max outputs admit exact byte comparison without changing
    the validator's existing finite-output rule.
    """
    out_shape = shape[:axis] + shape[axis + 1:]
    width = shape[axis]
    rows = np.random.default_rng(482).integers(-64, 64,
        size=(math.prod(out_shape), width)).astype('<f2') / np.float16(8)
    for i, row in enumerate(rows):
        mode = i % 8
        if mode == 0:
            row[-1] = 8
        elif mode == 1:
            row[:] = 1
        elif mode == 2:
            row[0] = 8
        elif mode == 3:
            row[[0, -1]] = 9
        elif mode == 4:
            row[:] = -8
            row[-1] = -1
        elif mode == 5:
            row[:] = 0
        elif mode == 6:
            row[:] = 0
            row[width // 2] = np.float16(2 ** -24)
        else:
            row[width // 2] = 8
    return np.moveaxis(rows.reshape((*out_shape, width)), -1, axis).copy()


def graph_case(case, frontend_headers):
    kind, shape, axis = PRIMITIVES[case]
    output_shape = shape[:axis] + shape[axis + 1:]
    dtype = 'int32' if kind == 'argmax' else 'float32'
    g = Graph('draft_top1_primitive_probe')
    source = g.input('logits', shape, 'float16')
    x = g.op('cast', source, shape=shape, dtype='float32', name='frontend_logits')
    if kind == 'argmax':
        axes = g.const(np.array(axis, dtype='<i4'), dtype='int32', name='axis')
        output = g.custom('ArgMaxExt2', {'x': x, 'axis': axes},
            {'y': (output_shape, dtype)},
            {'output_type': {'type': 'int', 'value': GE_DTYPE_IDS[dtype]},
             'keep_dims': {'type': 'bool', 'value': False},
             'outmaxval': {'type': 'bool', 'value': False},
             'topk': {'type': 'int', 'value': 1}}, name='reduction')['y']
    else:
        axes = g.const(np.array([axis], dtype='<i4'), dtype='int32', name='axes')
        output = g.custom('ReduceMax', {'x': x, 'axes': axes},
            {'y': (output_shape, dtype)},
            {'keep_dims': {'type': 'bool', 'value': False}}, name='reduction')['y']
    g.outputs = [output]
    g.output_contract = dict(input_shape=list(shape), input_dtype='float16',
        output_shape=list(output_shape), output_dtype=dtype, operator=kind,
        reduction_axis=axis, frontend_data_dtype='float32',
        input_conversion='FP16_TO_FP32_EXACT_FINITE_VALUE_WIDENING',
        frontend_headers=dict(frontend_headers),
        reference='DIRECT_NUMPY_REDUCTION_HOST_CROSS_CHECKED_WITH_PYTORCH',
        fixture='FINITE_HALF_VALUES_TIES_NEGATIVE_VALUES_ZERO_AND_POSITIVE_SUBNORMAL',
        acceptance='EXACT_OUTPUT_BYTES_ALL_REPEATS',
        nonfinite='NOT_IN_PRIMITIVE_FIXTURE; COVERED_BY_FULL_TOP1_COMPOSITIONS',
        scope='ISOLATED_REDUCTION_SHAPE_PLACEMENT_NOT_FULL_TOP1_OR_DRAFT')
    data = fixture(shape, axis)
    golden = (np.argmax(data, axis=axis).astype('<i4') if kind == 'argmax'
              else np.max(data, axis=axis).astype('<f4'))
    return g.validate(), [data], [golden]

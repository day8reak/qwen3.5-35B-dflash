"""Full-vocabulary Top1 placement candidates; no learned weights or model edits.

Legacy GE schemas are from the pinned TorchAir raw-op definitions. Native CPro
candidates use the user's selected HIAI frontend headers, corroborated by the
retained Draft's CPU ArgMaxExt2. FP16 input values widen to FP32 for that
frontend's data ports. Actual compiler selection and placement remain unverified.
All candidates must return the original smallest-index argmax, including across
tiles. Nonfinite rows test behavior rather than assuming a vendor NaN policy.
"""
import numpy as np

from qwen35_moe_dflash.graph.fused.ir import Graph
from qwen35_moe_dflash.graph.type_contracts import GE_DTYPE_IDS

NATIVE_CASES = ('top1_ext2_fp16', 'top1_ext2_flat_fp16', 'top1_ext2_tiled_fp16')
NATIVE_SCHEMA = ('USER_OBSERVED_SELECTED_DDK_ddk/ai_ddk_lib/include/graph/op/'
                 'math_defs.h:ArgMaxExt2,ReduceMax; array_defs.h:GatherV2D; '
                 'COMPILED_DRAFT_CPU_ARGMAXEXT2_CORROBORATES_AXIS_AND_DEFAULTS')
FRONTEND_HEADERS = {
    'ddk/ai_ddk_lib/include/graph/op/math_defs.h':
        '50dbd5318f2bfbc584d83a97de4fc6aed2751b0bcf3f653fd767b68f58fd827e',
    'ddk/ai_ddk_lib/include/graph/op/array_defs.h':
        '978d175bfb5b74f0a3d47d4803974f0ced33354557cf0764ec7137b54e311b65',
}
CASES = ('top1_argmax_fp16', 'top1_with_value_fp16', 'top1_tiled_fp16', *NATIVE_CASES)
SHAPE = (1, 15, 248320)
TILE = 512
LABELS = ('all_equal', 'first', 'last', 'index_65535', 'index_65536',
          'tile_boundary_tie', 'uint16_boundary_tie', 'large_index_tie',
          'negative_only', 'signed_zero_tie', 'positive_infinity_tie',
          'all_negative_infinity', 'nan_across_tiles', 'large_unique', 'last_nan')


def fixture():
    """One adversarial row per proposed token; int32 output is always finite."""
    data = np.full(SHAPE, -8, dtype='<f2')
    rows = data[0]
    rows[0] = 1
    for row, index in ((1, 0), (2, 248319), (3, 65535), (4, 65536), (13, 200003)):
        rows[row, index] = 8
    for row, indices in ((5, (511, 512)), (6, (65535, 65536)),
                         (7, (131071, 131072))):
        rows[row, list(indices)] = 8
    rows[8, 180001] = -1
    rows[9, 0], rows[9, 1] = np.float16(-0.), np.float16(0.)
    rows[10, [65536, 248319]] = np.inf
    rows[11] = -np.inf
    rows[12, [512, 200003]] = np.nan
    rows[14, -1] = np.nan
    # numpy and original PyTorch argmax agreement is checked by host tests;
    # neither the two-level candidate nor its local indices define this golden.
    expected = np.argmax(data, axis=-1).astype('<i4')
    return data, expected


def contract():
    return dict(input_shape=list(SHAPE), input_dtype='float16',
        output_shape=[1, 15], output_dtype='int32', vocabulary=SHAPE[-1],
        reference='DIRECT_NUMPY_ARGMAX_HOST_CROSS_CHECKED_WITH_PYTORCH',
        row_labels=list(LABELS), tie_rule='SMALLEST_GLOBAL_INDEX',
        nonfinite='FIRST_NAN_OR_FIRST_MAXIMUM_AS_ORIGINAL_PYTORCH; DEVICE_UNVERIFIED',
        tiled_geometry=[485, TILE], padding=0, index_arithmetic='INT32',
        acceptance='EXACT_INT32_BYTES_ALL_ROWS_ALL_REPEATS',
        scope='SYNTHETIC_FULL_VOCABULARY_REDUCTION_ONLY_NOT_FULL_DRAFT')


def graph_case(case):
    if case not in CASES:
        raise ValueError('unknown full-vocabulary Top1 probe')
    g = Graph('draft_top1_probe')
    x = g.input('logits', SHAPE, 'float16')
    if case in NATIVE_CASES:
        # Selected frontend headers list FLOAT (FP32), not FLOAT16. Widening
        # preserves finite FP16 values; NaN/tie behavior still needs device tests.
        x = g.op('cast', x, shape=SHAPE, dtype='float32', name='frontend_logits')

    def reshape(value, shape, name):
        return g.op('reshape', value, shape=shape, dtype=value.dtype,
                    attrs={'shape': list(shape)}, name=name)

    def maximum(value, shape, name):
        return g.custom('ArgMaxWithValue', {'x': value},
            {'indice': (shape, 'int32'), 'values': (shape, 'float16')},
            {'dimension': {'type': 'int', 'value': -1},
             'keep_dims': {'type': 'bool', 'value': False}}, name=name)

    def ext2(value, shape, dimension, name):
        # HIAI frontend uses a tensor axis. The retained CPU node has a 4-byte
        # INT32 Const with empty dimensions; emit a scalar, as in ArgMaxV2.
        axis = g.const(np.array(dimension, dtype='<i4'), dtype='int32', name=name + '_axis')
        return g.custom('ArgMaxExt2', {'x': value, 'axis': axis}, {'y': (shape, 'int32')},
            {'output_type': {'type': 'int', 'value': GE_DTYPE_IDS['int32']},
             'keep_dims': {'type': 'bool', 'value': False},
             'outmaxval': {'type': 'bool', 'value': False},
             'topk': {'type': 'int', 'value': 1}}, name=name)['y']

    if case == 'top1_argmax_fp16':
        axis = g.const(np.array(-1, dtype='<i4'), dtype='int32', name='axis')
        output = g.custom('ArgMaxV2', {'x': x, 'dimension': axis},
            {'y': ((1, 15), 'int32')},
            {'dtype': {'type': 'dtype', 'value': 'int32'}}, name='argmax')['y']
    elif case == 'top1_with_value_fp16':
        output = maximum(x, (1, 15), 'argmax_value')['indice']
    elif case == 'top1_ext2_fp16':
        output = ext2(x, (1, 15), 2, 'argmax_ext2')
    elif case == 'top1_ext2_flat_fp16':
        output = reshape(ext2(reshape(x, (15, SHAPE[-1]), 'rows'), (15,),
                              1, 'argmax_ext2'), (1, 15), 'tokens')
    elif case == 'top1_ext2_tiled_fp16':
        tiles = SHAPE[-1] // TILE
        tiled = reshape(x, (15, tiles, TILE), 'tiles')
        local_index = ext2(tiled, (15, tiles), 2, 'local_argmax')
        axes = g.const(np.array([2], dtype='<i4'), dtype='int32', name='local_value_axes')
        local_value = g.custom('ReduceMax', {'x': tiled, 'axes': axes},
            {'y': ((15, tiles), 'float32')},
            {'keep_dims': {'type': 'bool', 'value': False}}, name='local_value')['y']
        winner = ext2(local_value, (15,), 1, 'winning_tile')
        row_offsets = g.const(np.arange(15, dtype='<i4') * tiles,
                              dtype='int32', name='row_offsets')
        selector = g.op('add', row_offsets, winner, shape=(15,), dtype='int32', name='selector')
        selected = g.custom('GatherV2D',
            {'x': reshape(local_index, (15 * tiles,), 'flat_indices'), 'indices': selector},
            {'y': ((15,), 'int32')}, {'axis': {'type': 'int', 'value': 0}},
            name='local_index')['y']
        stride = g.const(np.array(TILE, dtype='<i4'), dtype='int32', name='tile_stride')
        offset = g.op('mul', winner, stride, shape=(15,), dtype='int32', name='tile_offset')
        output = reshape(g.op('add', offset, selected, shape=(15,), dtype='int32',
                              name='global_index'), (1, 15), 'tokens')
    else:
        tiles = SHAPE[-1] // TILE
        local = maximum(reshape(x, (1, 15, tiles, TILE), 'tiles'),
                        (1, 15, tiles), 'local_max')
        winner = maximum(local['values'], (1, 15), 'winning_tile')['indice']
        gather_index = reshape(winner, (1, 15, 1), 'winning_tile_axis')
        selected = g.custom('GatherElements', {'x': local['indice'], 'index': gather_index},
            {'y': ((1, 15, 1), 'int32')},
            {'dim': {'type': 'int', 'value': 2}}, name='local_index')['y']
        stride = g.const(np.array(TILE, dtype='<i4'), dtype='int32', name='tile_stride')
        offset = g.op('mul', winner, stride, shape=(1, 15), dtype='int32', name='tile_offset')
        output = g.op('add', offset, reshape(selected, (1, 15), 'selected_index'),
                      shape=(1, 15), dtype='int32', name='global_index')
    g.outputs = [output]
    g.output_contract = contract()
    if case in NATIVE_CASES:
        g.output_contract.update(frontend_data_dtype='float32',
            input_conversion='FP16_TO_FP32_EXACT_FINITE_VALUE_WIDENING',
            axis_input='INT32_SCALAR_ARGMAX; INT32_VECTOR_REDUCEMAX',
            frontend_headers=dict(FRONTEND_HEADERS),
            schema_scope='USER_REPORTED_SELECTED_HIAI_HEADERS_NOT_KERNEL_PLACEMENT_PROOF')
    if case == 'top1_ext2_tiled_fp16':
        g.output_contract['local_value_reduction'] = 'SEPARATE_REDUCEMAX_NAN_AND_TIE_CONSISTENCY_DEVICE_UNVERIFIED'
    data, expected = fixture()
    return g.validate(), [data], [expected]

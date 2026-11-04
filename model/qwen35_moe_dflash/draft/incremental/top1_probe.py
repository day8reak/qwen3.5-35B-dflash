"""Full-vocabulary Top1 placement candidates; no learned weights or model edits.

GE schemas are from the pinned TorchAir raw-op definitions used by the vector
probe tool. Schema availability is not selected X90 DDK implementation proof.
All candidates must return the original smallest-index argmax, including across
tiles. Nonfinite rows test behavior rather than assuming a vendor NaN policy.
"""
import numpy as np

from qwen35_moe_dflash.graph.fused.ir import Graph

CASES = ('top1_argmax_fp16', 'top1_with_value_fp16', 'top1_tiled_fp16')
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

    def reshape(value, shape, name):
        return g.op('reshape', value, shape=shape, dtype=value.dtype,
                    attrs={'shape': list(shape)}, name=name)

    def maximum(value, shape, name):
        return g.custom('ArgMaxWithValue', {'x': value},
            {'indice': (shape, 'int32'), 'values': (shape, 'float16')},
            {'dimension': {'type': 'int', 'value': -1},
             'keep_dims': {'type': 'bool', 'value': False}}, name=name)

    if case == 'top1_argmax_fp16':
        axis = g.const(np.array(-1, dtype='<i4'), dtype='int32', name='axis')
        output = g.custom('ArgMaxV2', {'x': x, 'dimension': axis},
            {'y': ((1, 15), 'int32')},
            {'dtype': {'type': 'dtype', 'value': 'int32'}}, name='argmax')['y']
    elif case == 'top1_with_value_fp16':
        output = maximum(x, (1, 15), 'argmax_value')['indice']
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
    data, expected = fixture()
    return g.validate(), [data], [expected]

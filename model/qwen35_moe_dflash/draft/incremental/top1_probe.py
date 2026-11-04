"""Full-vocabulary Top1 placement candidates; no learned weights or model edits.

Legacy GE schemas are from the pinned TorchAir raw-op definitions. Native CPro
candidates retain the user's selected backend opinfo expression for diagnosis.
OMG rejected its one-input ArgMaxExt2; export is blocked by the observed arity
check pending the selected frontend IR contract. Backend registration is not
a frontend schema or selected X90 DDK implementation/placement proof.
All candidates must return the original smallest-index argmax, including across
tiles. Nonfinite rows test behavior rather than assuming a vendor NaN policy.
"""
import numpy as np

from qwen35_moe_dflash.graph.fused.ir import Graph
from qwen35_moe_dflash.graph.type_contracts import GE_DTYPE_IDS

NATIVE_CASES = ('top1_ext2_fp16', 'top1_ext2_flat_fp16', 'top1_ext2_tiled_fp16')
NATIVE_SCHEMA = ('USER_OBSERVED_SELECTED_DDK_tools/tools_omg/master/lib64/config/'
                 'npu_generated_opinfo.json:ArgMaxExt2,ReduceMax; '
                 'OMG_REJECTED_ONE_INPUT_FRONTEND_CONTRACT_PENDING')
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

    def reshape(value, shape, name):
        return g.op('reshape', value, shape=shape, dtype=value.dtype,
                    attrs={'shape': list(shape)}, name=name)

    def maximum(value, shape, name):
        return g.custom('ArgMaxWithValue', {'x': value},
            {'indice': (shape, 'int32'), 'values': (shape, 'float16')},
            {'dimension': {'type': 'int', 'value': -1},
             'keep_dims': {'type': 'bool', 'value': False}}, name=name)

    def ext2(value, shape, dimension, name):
        # NPU kernel metadata lists dimension only. OMG's IR verifier also
        # requires output_type; an INT32 output descriptor does not supply it.
        # This is a GE enum-valued integer, not the protobuf tensor dtype ID.
        return g.custom('ArgMaxExt2', {'x': value}, {'y': (shape, 'int32')},
            {'dimension': {'type': 'int', 'value': dimension},
             'output_type': {'type': 'int', 'value': GE_DTYPE_IDS['int32']}}, name=name)['y']

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
        # Retain the rejected experiment's backend opinfo form for inspection.
        # Its frontend axis contract and inferred output shape are unverified.
        local_value = g.custom('ReduceMax', {'x': tiled},
            {'y': ((15, tiles), 'float16')},
            {'axes': {'type': 'ints', 'value': [2]},
             'keep_dims': {'type': 'bool', 'value': False}}, name='local_value')['y']
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
    if case == 'top1_ext2_tiled_fp16':
        g.output_contract['local_value_reduction'] = 'SEPARATE_REDUCEMAX_NAN_AND_TIE_CONSISTENCY_DEVICE_UNVERIFIED'
    data, expected = fixture()
    return g.validate(), [data], [expected]

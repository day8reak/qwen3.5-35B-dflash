"""Builtin TopK and adjacent-pair Top1 candidates, for finite FP16 logits.

No custom kernel or forced placement. The pinned GE emitter supplies TopKV2 /
SelectV2 schemas; selected-DDK compilation, placement and device equivalence
are separate gates. Historical NaN-contract probes remain unchanged.
"""
import hashlib

import numpy as np

from qwen35_moe_dflash.graph.fused.ir import Graph
from .top1_matrix import fixture as primitive_fixture


TOPK_SHAPES = {
    'top1_topk_r15_d512_fp16': (15, 512),
    'top1_topk_r485_d512_fp16': (485, 512),
    'top1_topk_r7275_d512_fp16': (7275, 512),
    'top1_topk_r15_d485_fp16': (15, 485),
}
FULL_CASES = ('top1_topk_direct_fp16', 'top1_topk_tiled_fp16',
              'top1_topk_split15_fp16', 'top1_tree_full_fp16')
CASES = (*TOPK_SHAPES, 'top1_select_int32', 'top1_gather_int32',
         'top1_compose_int32', 'top1_tree_r15_d485_fp16', *FULL_CASES)
FRONTEND_TYPES = ('TopKV2', 'TopK', 'TopKD', 'SelectV2', 'Select', 'Greater',
                  'Cast', 'CastT', 'GatherV2D', 'Add', 'Mul', 'Slice', 'Reshape', 'ConcatD')
FINITE_LABELS = ('all_equal', 'first', 'last', 'index_65535', 'index_65536',
    'tile_boundary_tie', 'uint16_boundary_tie', 'large_index_tie', 'negative_only',
    'signed_zero_tie', 'max_finite_tie', 'all_min_finite', 'subnormal_tie_across_tiles',
    'large_unique', 'last_adjacent_finite')


def full_fixture():
    from .top1_probe import fixture
    data, _ = fixture()
    rows = data[0]
    rows[10, [65536, 248319]] = np.finfo(np.float16).max
    rows[11] = -np.finfo(np.float16).max
    rows[12] = 0
    rows[12, [512, 200003]] = np.float16(2 ** -24)
    rows[14, -1] = np.nextafter(np.float16(-8), np.float16(0))
    assert np.isfinite(data).all()
    return data, np.argmax(data, axis=-1).astype('<i4')


def reshape(g, x, shape, name):
    return g.op('reshape', x, shape=shape, dtype=x.dtype,
                attrs={'shape': list(shape)}, name=name)


def sliced(g, x, starts, sizes, name):
    return g.op('slice', x, shape=sizes, dtype=x.dtype,
                attrs={'starts': list(starts), 'sizes': list(sizes)}, name=name)


def topk(g, x, name):
    shape = (*x.shape[:-1], 1)
    # The emitter caches these two primitive outputs as one TopKV2 node.
    return tuple(g.op(kind, x, shape=shape, dtype=dtype,
                     attrs={'k': 1, 'axis': -1}, name=name + suffix)
                 for kind, dtype, suffix in (('topk_values', x.dtype, '_value'),
                                            ('topk_indices', 'int32', '_index')))


def select(g, condition, right, left, name):
    return g.op('where', condition, right, left, shape=left.shape,
                dtype=left.dtype, name=name)


def gather(g, x, indices, name):
    return g.custom('GatherV2D', {'x': x, 'indices': indices},
                    {'y': (indices.shape, 'int32')},
                    {'axis': {'type': 'int', 'value': 0}}, name=name)['y']


def compose(g, winner, local, name):
    stride = g.const(np.array(512, dtype='<i4'), dtype='int32', name=name + '_stride')
    offset = g.op('mul', winner, stride, shape=winner.shape, dtype='int32', name=name + '_offset')
    return g.op('add', offset, local, shape=local.shape, dtype='int32', name=name)


def tree(g, x):
    """Adjacent contiguous groups; carry odd tails without sentinel padding.

    Indices are relative to each group until the last level. This avoids a
    full-vocabulary index constant and never converts indices through floats.
    Strict greater chooses right; equality retains the earlier left group.
    """
    rows, width = x.shape
    index, stride, level = None, 1, 0
    while width > 1:
        pairs, odd = divmod(width, 2)
        prefix = f'tree_{level}'

        def halves(value, tag):
            even = sliced(g, value, (0, 0), (rows, pairs * 2), prefix + tag + '_even') if odd else value
            packed = reshape(g, even, (rows, pairs, 2), prefix + tag + '_pairs')
            return tuple(reshape(g, sliced(g, packed, (0, 0, side), (rows, pairs, 1),
                                 prefix + tag + f'_side_{side}'), (rows, pairs),
                                 prefix + tag + f'_half_{side}') for side in (0, 1))

        left, right = halves(x, '_value')
        take_right = g.op('gt', right, left, shape=left.shape, dtype='bool', name=prefix + '_right_wins')
        value = select(g, take_right, right, left, prefix + '_value')
        if index is None:
            chosen = g.op('cast', take_right, shape=left.shape, dtype='int32', name=prefix + '_index')
        else:
            li, ri = halves(index, '_index')
            step = g.const(np.array(stride, dtype='<i4'), dtype='int32', name=prefix + '_stride')
            ri = g.op('add', ri, step, shape=ri.shape, dtype='int32', name=prefix + '_right_offset')
            chosen = select(g, take_right, ri, li, prefix + '_index')
        if odd:
            tail = sliced(g, x, (0, width - 1), (rows, 1), prefix + '_tail_value')
            # At the first level each original element has relative index 0.
            tail_i = (g.const(np.zeros((rows, 1), dtype='<i4'), dtype='int32', name=prefix + '_tail_zero')
                      if index is None else sliced(g, index, (0, width - 1), (rows, 1), prefix + '_tail_index'))
            value = g.op('concat', value, tail, shape=(rows, pairs + 1), dtype=x.dtype,
                         attrs={'axis': 1}, name=prefix + '_values_with_tail')
            chosen = g.op('concat', chosen, tail_i, shape=(rows, pairs + 1), dtype='int32',
                          attrs={'axis': 1}, name=prefix + '_indices_with_tail')
        x, index, width = value, chosen, pairs + odd
        stride *= 2
        level += 1
    if index is None:
        raise ValueError('tree probe needs width >= 2')
    return index


def graph_case(case):
    if case not in CASES:
        raise ValueError('unknown Top1 alternative')
    g = Graph('draft_top1_alternative_probe')
    base_contract = dict(reference='DIRECT_NUMPY_HOST_CROSS_CHECKED_WITH_PYTORCH',
        acceptance='EXACT_OUTPUT_BYTES_ALL_REPEATS', index_arithmetic='INT32_NO_FLOAT_CARRIER',
        input_domain='FINITE_FP16_LOGITS_OR_DECLARED_INT32_HELPER_INPUTS',
        tie_rule='SMALLEST_GLOBAL_INDEX_REQUIRED', nonfinite='OUTSIDE_THIS_FINITE_INPUT_PROBE_CONTRACT',
        schema_scope='PINNED_GE_EMITTER_AND_OBSERVED_GATHERV2D; SELECTED_DDK_PLACEMENT_UNVERIFIED',
        scope='BUILTIN_TOP1_CANDIDATE_NOT_FULL_DRAFT', custom_kernel=False)
    if case in ('top1_select_int32', 'top1_gather_int32', 'top1_compose_int32'):
        shape = (15,)
        if case == 'top1_select_int32':
            arrays = [np.array([1, 2, -1, 0, -0., 8, 8, -8, 3, 4, 2**-24, 65504, -65504, 1, 9], dtype='<f2'),
                      np.array([2, 1, -1, -0., 0, 8, 7, -1, 4, 3, 0, 65472, -65504, 1, 8], dtype='<f2'),
                      np.array([0, 65535, 65536, 131071, 131072, 248318, 248319, 512, 511, 1, 2, 3, 4, 5, 6], dtype='<i4'),
                      np.array([248319, 65536, 65535, 131072, 131071, 0, 1, 511, 512, 2, 1, 4, 3, 6, 5], dtype='<i4')]
            lv, rv, li, ri = [g.input(n, shape, d) for n, d in
                             (('left_value', 'float16'), ('right_value', 'float16'),
                              ('left_index', 'int32'), ('right_index', 'int32'))]
            lv, rv = [g.op('cast', v, shape=shape, dtype='float32', name=n)
                      for v, n in ((lv, 'frontend_left'), (rv, 'frontend_right'))]
            condition = g.op('gt', rv, lv, shape=shape, dtype='bool', name='right_wins')
            selected = select(g, condition, rv, lv, 'selected_value')
            g.outputs = [g.op('cast', selected, shape=shape, dtype='float16', name='value'),
                         select(g, condition, ri, li, 'selected_index')]
            take = arrays[1] > arrays[0]
            golden = [np.where(take, arrays[1], arrays[0]), np.where(take, arrays[3], arrays[2])]
            base_contract['tie_rule'] = 'SELECT_LEFT_ON_EQUAL; HELPER_INDICES_ARE_OPAQUE_PAYLOADS'
        elif case == 'top1_gather_int32':
            source = (np.arange(7275, dtype='<i4') * 37) % 248320
            selector = np.array([0, 484, 485, 7274, 1, 512, 1024, 2048, 4096, 6000, 7000, 100, 200, 300, 400], dtype='<i4')
            source[selector[:4]] = [65535, 65536, 131072, 248319]
            x, indices = g.input('indices', (7275,), 'int32'), g.input('selector', shape, 'int32')
            g.outputs = [gather(g, x, indices, 'selected_index')]
            arrays, golden = [source, selector], [source[selector]]
        else:
            winner = np.array([0, 1, 127, 128, 255, 256, 484, 484, 1, 2, 3, 4, 5, 6, 7], dtype='<i4')
            local = np.array([0, 511, 511, 0, 511, 0, 511, 0, 0, 1, 2, 3, 4, 5, 6], dtype='<i4')
            a, b = g.input('winning_tile', shape, 'int32'), g.input('local_index', shape, 'int32')
            g.outputs = [compose(g, a, b, 'global_index')]
            arrays, golden = [winner, local], [winner * np.int32(512) + local]
    else:
        full = case in FULL_CASES
        data, expected = full_fixture() if full else (None, None)
        shape = (1, 15, 248320) if full else TOPK_SHAPES.get(case, (15, 485))
        if data is None:
            data = primitive_fixture(shape, len(shape) - 1)
            expected = np.argmax(data, axis=-1).astype('<i4')
        source = g.input('logits', shape, 'float16')
        x = g.op('cast', source, shape=shape, dtype='float32', name='frontend_logits')
        arrays, golden = [data], [expected]
        if case in TOPK_SHAPES:
            values, indices = topk(g, x, 'local')
            values = g.op('cast', values, shape=values.shape, dtype='float16', name='max_value')
            g.outputs = [values, indices]
            indices_np = expected[..., None]
            golden = [np.take_along_axis(data, indices_np, axis=-1), indices_np]
        elif case == 'top1_topk_direct_fp16':
            _, indices = topk(g, reshape(g, x, (15, 248320), 'rows'), 'global')
            g.outputs = [reshape(g, indices, (1, 15), 'tokens')]
        elif case in ('top1_topk_tiled_fp16', 'top1_topk_split15_fp16'):
            tiles = reshape(g, x, (7275, 512), 'tiles')
            if case == 'top1_topk_split15_fp16':
                outputs = [topk(g, sliced(g, tiles, (row * 485, 0), (485, 512), f'row_{row}'),
                                f'local_{row}') for row in range(15)]
                values, indices = [g.op('concat', *(p[port] for p in outputs), shape=(7275, 1),
                                       dtype=dtype, attrs={'axis': 0}, name=name)
                                   for port, dtype, name in ((0, 'float32', 'local_values'), (1, 'int32', 'local_indices'))]
            else:
                values, indices = topk(g, tiles, 'local')
            _, winner = topk(g, reshape(g, values, (15, 485), 'candidate_values'), 'winner')
            winner = reshape(g, winner, (15,), 'winning_tile')
            offsets = g.const(np.arange(15, dtype='<i4') * 485, dtype='int32', name='row_offsets')
            selector = g.op('add', offsets, winner, shape=(15,), dtype='int32', name='selector')
            local = gather(g, reshape(g, indices, (7275,), 'candidate_indices'), selector, 'selected_index')
            g.outputs = [reshape(g, compose(g, winner, local, 'global_index'), (1, 15), 'tokens')]
        else:
            index = tree(g, reshape(g, x, (15, shape[-1]), 'rows'))
            g.outputs = [reshape(g, index, (1, 15) if full else (15,), 'tokens')]
        base_contract.update(frontend_data_dtype='float32',
                             input_conversion='FP16_TO_FP32_EXACT_FINITE_VALUE_WIDENING')
        if full:
            base_contract.update(vocabulary=248320, row_labels=list(FINITE_LABELS), padding=0,
                                 scope='FINITE_FULL_VOCABULARY_TOP1_NOT_FULL_DRAFT')
    g.output_contract = base_contract
    g.validate()
    # Primitive graph manifests omit individual edges/attrs/constants. Bind the
    # exact emitted source as well so validation cannot accept a stale recipe.
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    g.output_contract['emitted_cpp_sha256'] = hashlib.sha256(
        emit_cpp(g, fold_constant_arithmetic=False).encode()).hexdigest()
    return g, arrays, [np.ascontiguousarray(a) for a in golden]

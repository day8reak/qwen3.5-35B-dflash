"""Last builtin Top1 attempt: select opaque FP16 words, then GE Bitcast.

The words are storage, never numerical indices. Some encode NaNs, infinities,
signed zero or subnormals. Selected-DDK compilation and exhaustive device
byte equality are mandatory: float conversion or arithmetic masking is unsafe.
"""
import hashlib

import numpy as np

from qwen35_moe_dflash.graph.fused.ir import Graph
from .top1_alternatives import full_fixture, reshape, sliced
from .top1_matrix import fixture


VOCAB = 248320
CASES = ('top1_bits_payload', 'top1_bits_tree_offset65500', 'top1_bits_tree_full')
FULL_CASES = (CASES[-1],)
DEPENDENCIES = {CASES[1]: (CASES[0],), CASES[2]: CASES[:2]}
FRONTEND_TYPES = ('Bitcast', 'SelectV2', 'Select', 'Greater', 'BroadcastTo',
                  'ConcatD', 'Slice', 'Reshape')


def words(ids):
    """Little-endian INT32 bytes split into two FP16 storage lanes, no cast."""
    packed = np.ascontiguousarray(ids, dtype='<i4').view('<f2').reshape(*ids.shape, 2)
    return tuple(np.ascontiguousarray(packed[..., lane]) for lane in (0, 1))


def choose(g, condition, right, left, name):
    return g.op('where', condition, right, left, shape=condition.shape,
                dtype=left.dtype, name=name)


def unpack(g, low, high):
    # Flatten first: rank-2 ConcatD avoids unrelated rank-3 layout lowering.
    count = int(np.prod(low.shape))
    lanes = [reshape(g, x, (count, 1), 'word_' + tag)
             for x, tag in ((low, 'low'), (high, 'high'))]
    packed = g.op('concat', *lanes, shape=(count, 2), dtype='float16',
                  attrs={'axis': 1}, name='packed_words')
    return g.custom('Bitcast', {'x': packed}, {'y': ((count,), 'int32')},
        {'type': {'type': 'dtype', 'value': 'int32'},
         'keep_dim': {'type': 'bool', 'value': False}}, name='tokens_bits')['y']


def tree(g, x, *, base=0):
    """Stable adjacent-pair reduction. Index words only undergo selection/copy."""
    rows, width = x.shape
    payload, level = None, 0
    while width > 1:
        pairs, odd = divmod(width, 2)
        prefix = f'bits_{level}'

        def halves(value, tag):
            even = (sliced(g, value, (0, 0), (rows, pairs * 2), prefix + tag + '_even')
                    if odd else value)
            packed = reshape(g, even, (rows, pairs, 2), prefix + tag + '_pairs')
            return tuple(reshape(g, sliced(g, packed, (0, 0, side), (rows, pairs, 1),
                prefix + tag + f'_side_{side}'), (rows, pairs), prefix + tag + f'_half_{side}')
                for side in (0, 1))

        left, right = halves(x, '_value')
        take = g.op('gt', right, left, shape=left.shape, dtype='bool', name=prefix + '_take_right')
        value = choose(g, take, right, left, prefix + '_value')
        selected = []
        for lane in (0, 1):
            tag = f'_word{lane}'
            if payload is None:
                ids = np.arange(base, base + pairs * 2, dtype='<i4').reshape(1, pairs, 2)
                li, ri = [g.const(words(ids[..., side])[lane], dtype='float16',
                                 name=prefix + tag + f'_side{side}') for side in (0, 1)]
            else:
                li, ri = halves(payload[lane], tag)
            chosen = choose(g, take, ri, li, prefix + tag)
            if odd:
                tail = (g.const(words(np.full((rows, 1), base + width - 1, dtype='<i4'))[lane],
                                dtype='float16', name=prefix + tag + '_tail')
                        if payload is None else sliced(g, payload[lane], (0, width - 1),
                                                       (rows, 1), prefix + tag + '_tail'))
                chosen = g.op('concat', chosen, tail, shape=(rows, pairs + 1), dtype='float16',
                              attrs={'axis': 1}, name=prefix + tag + '_with_tail')
            selected.append(chosen)
        if odd:
            tail = sliced(g, x, (0, width - 1), (rows, 1), prefix + '_value_tail')
            value = g.op('concat', value, tail, shape=(rows, pairs + 1), dtype=x.dtype,
                         attrs={'axis': 1}, name=prefix + '_value_with_tail')
        x, payload, width, level = value, selected, pairs + odd, level + 1
    if payload is None:
        raise ValueError('tree needs width >= 2')
    return unpack(g, *payload)


def graph_case(case):
    if case not in CASES:
        raise ValueError('unknown bit-carrier Top1 case')
    g = Graph('draft_top1_bitcarrier_probe')
    contract = dict(reference='DIRECT_NUMPY_INTEGER_BYTES_OR_ARGMAX_HOST_CROSS_CHECKED_WITH_PYTORCH',
        acceptance='EXACT_OUTPUT_BYTES_ALL_REPEATS', custom_kernel=False,
        index_encoding='LITTLE_ENDIAN_INT32_AS_TWO_OPAQUE_FP16_WORDS_THEN_BITCAST',
        payload_arithmetic='FORBIDDEN_IN_SOURCE; DEVICE_BIT_PRESERVATION_UNVERIFIED',
        payload_domain='ALL_TOKEN_IDS_0_TO_248319_INCLUDING_FP16_NAN_INF_ZERO_SUBNORMAL_BIT_PATTERNS',
        logit_domain='FINITE_FP16', tie_rule='SMALLEST_GLOBAL_INDEX',
        nonfinite_logits='OUTSIDE_THIS_FINITE_INPUT_CONTRACT',
        schema_scope='PINNED_GE_BITCAST_PROTOTYPE; SELECTED_DDK_SHAPE_SUPPORT_AND_PLACEMENT_UNVERIFIED',
        scope='LAST_BUILTIN_ATTEMPT_NOT_FULL_DRAFT', dependencies=list(DEPENDENCIES.get(case, ())))
    if case == CASES[0]:
        # Every ID is selected twice from each branch. Each unselected branch
        # carries a different ID, including nonfinite patterns next to finite.
        data = np.repeat(np.array([[-1], [1], [-1], [1]], dtype='<f2'), VOCAB, axis=1)
        source = g.input('selector', data.shape, 'float16')
        zero = g.const(np.array(0, dtype='<f2'), dtype='float16', name='zero')
        take = g.op('gt', source, zero, shape=data.shape, dtype='bool', name='take_right')
        ids = np.arange(VOCAB, dtype='<i4').reshape(1, VOCAB)
        left, right = words(ids), words(ids[:, ::-1])
        selected = []
        for lane in (0, 1):
            a, b = [g.const(v[lane], dtype='float16', name=f'{tag}_{lane}')
                    for tag, v in (('left', left), ('right', right))]
            selected.append(choose(g, take, b, a, f'selected_{lane}'))
        g.outputs = [unpack(g, *selected)]
        golden = np.concatenate([ids, ids[:, ::-1], ids, ids[:, ::-1]], axis=0).reshape(-1)
        contract.update(scope='EXHAUSTIVE_PAYLOAD_BROADCAST_SELECT_CONCAT_BITCAST_ONLY',
                        selected_ids_per_branch=VOCAB, selections_per_id_per_branch=2)
    else:
        full = case in FULL_CASES
        if full:
            data, golden = full_fixture()
            base = 0
        else:
            data, base = fixture((15, 485), 1), 65500
            golden = np.argmax(data, axis=-1).astype('<i4') + np.int32(base)
        source = g.input('logits', data.shape, 'float16')
        index = tree(g, reshape(g, source, (15, data.shape[-1]), 'rows'), base=base)
        g.outputs = [reshape(g, index, golden.shape, 'tokens')]
        contract.update(vocabulary=VOCAB if full else 485, index_base=base, padding=0)
    g.output_contract = contract
    g.validate()
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    contract['emitted_cpp_sha256'] = hashlib.sha256(
        emit_cpp(g, fold_constant_arithmetic=False).encode()).hexdigest()
    return g, [data], [np.ascontiguousarray(golden)]

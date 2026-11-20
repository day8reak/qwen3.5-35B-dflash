"""Explain the measured gated product using the split probes that reproduce hidden."""
import json

import numpy as np

from ...common.accuracy import floating_metrics
from ..layer_probe.common import read_array
from .gdr_stage_data import logical
from .segment_trace_compare import _effect, _magnitude


def _fmt(value):
    return 'n/a' if value is None else f'{value:.6g}'


def _product_scope(ref, board, start, end):
    # Use the formula for the CPro gate in substitutions so an invalid SiLU tap
    # cannot contaminate the attribution. Check a qualified direct read separately.
    def value(tensors, role):
        return tensors[role][0, start:end].astype(np.float64)
    ref_lhs, board_lhs = (value(t, 'mul_lhs') for t in (ref, board))
    ref_product, board_product = (value(t, 'mul_output') for t in (ref, board))
    ref_gate = value(ref, 'mul_rhs')
    z = value(board, 'silu_input')
    formula_gate = z*np.exp(-np.logaddexp(0., -z))
    lhs_delta, gate_delta, delta = board_lhs-ref_lhs, formula_gate-ref_gate, board_product-ref_product
    predicted = board_lhs*formula_gate

    comparisons = [
        ('310P lhs * measured rhs / 310P product', ref_lhs*ref_gate, ref_product),
        ('only CPro lhs / 310P product', board_lhs*ref_gate, ref_product),
        ('only CPro gate (formula) / 310P product', ref_lhs*formula_gate, ref_product),
        ('CPro lhs * SiLU(z) / measured CPro product', predicted, board_product),
        ('measured CPro / 310P product', board_product, ref_product),
    ]
    if 'mul_rhs' in board:
        measured_gate = value(board, 'mul_rhs')
        comparisons.extend([
            ('measured CPro rhs / SiLU(CPro z)', measured_gate, formula_gate),
            ('CPro lhs * measured rhs / CPro product', board_lhs*measured_gate, board_product),
        ])
    for name, actual, expected in comparisons:
        m = floating_metrics(actual, expected)
        print(f'  {name:46s} L2={_fmt(m["relative_l2"]):>11s}  cos={_fmt(m["cosine"])}')

    terms = [
        ('310P product roundoff/residual', ref_lhs*ref_gate-ref_product),
        ('lhs change weighted by 310P rhs', lhs_delta*ref_gate),
        ('gate change at 310P lhs (formula)', ref_lhs*gate_delta),
        ('lhs-gate interaction (formula)', lhs_delta*gate_delta),
        ('CPro measured minus reconstructed', board_product-predicted),
    ]
    print('  term                                              rel_to_310P_product    aligned_pct')
    for name, term in terms:
        magnitude, projection = _effect(term, ref_product, delta)
        print(f'  {name:48s} {_fmt(magnitude):>19s}  {_fmt(projection):>13s}')

    residual_ratio = _magnitude(board_product-predicted, delta)
    print(f'  ||CPro measured minus reconstructed|| / ||measured CPro minus 310P|| = {_fmt(residual_ratio)}')
    squared = delta**2
    row_error = squared.sum(axis=-1)
    total = row_error.sum()
    if total:
        i, head = np.unravel_index(row_error.argmax(), row_error.shape)
        print(f'  largest measured product-error row: token={i+start} head={head} '
              f'share={row_error[i, head]/total:.2%}')
    if ref['core'].dtype == np.float16:
        core = value(ref, 'core')
        mask = (core != 0) & (np.abs(core) < np.finfo(np.float16).tiny) & (value(board, 'core') == 0)
        share = squared[mask].sum()/total if total else None
        print(f'  310P FP16 core subnormal and CPro core zero: {mask.sum()}/{mask.size} elements; '
              f'share of measured product squared error={_fmt(100*share if share is not None else None)}%')


def audit_product(root, layer):
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest['physical_rows'], manifest['valid_rows'], manifest['last_valid_index'],
            manifest['chunk_size'], manifest['reference_backend']) != (64, 19, 18, 64, '310p'):
        raise ValueError('Expected a native 310P physical64/valid19 prefill dump')
    info = manifest['layers'][str(layer)]
    roles = ('core', 'mul_lhs', 'silu_input', 'mul_output')
    sources = set()
    for role in roles:
        matches = [g for g in info['graphs'].values() if role in g['aliases']]
        if len(matches) != 1 or not matches[0]['valid_hidden_equal_bits'] or not matches[0]['module_inputs_equal_baseline']:
            raise ValueError(f'L{layer}/{role}: need identical module inputs and reproduced valid19 hidden')
        sources.add(matches[0]['source_air_sha256'])
    if len(sources) != 1:
        raise ValueError('Product observations have different original AIRs')

    def read(backend, names):
        tensors = {name: logical(read_array(root, info['tensors'][name][backend]), name) for name in names}
        if not all(np.isfinite(a[:, :19]).all() for a in tensors.values()):
            raise ValueError(f'Nonfinite valid {backend} operand or product')
        return tensors
    ref = read('310p', (*roles, 'mul_rhs'))
    rhs_graph, = [g for g in info['graphs'].values() if 'mul_rhs' in g['aliases']]
    use_rhs = (rhs_graph['valid_hidden_equal_bits'] and rhs_graph['module_inputs_equal_baseline']
               and rhs_graph['source_air_sha256'] in sources)
    board = read('cpro', (*roles, 'mul_rhs') if use_rhs else roles)
    print(f'L{layer} CPro operands from hidden-reproducing probes. CPro gate factor from SiLU(silu_input).')
    print('  aligned_pct = signed projection onto the measured product difference.')
    for start, end, scope in ((0, 19, 'valid19'), (18, 19, 'token18')):
        print(scope)
        _product_scope(ref, board, start, end)

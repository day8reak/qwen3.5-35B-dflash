"""Check relations between the saved operands and outputs of the 64/19 prefill."""
import hashlib
import json

import numpy as np

from ...common.accuracy import floating_metrics
from ..layer_probe.common import read_array
from .decode_cache_trace import _record
from .gdr_stage_data import case_id
from .gdr_stage_report import _compiled_air
from .segment_trace_compare import board_case


def _print_gap(label, actual, expected):
    def numbers(x, y):
        m = floating_metrics(x, y)
        return '/'.join('n/a' if m[k] is None else f'{m[k]:.6g}' for k in ('relative_l2', 'cosine'))
    valid = actual[:, :19]
    print(f'{label:35s} {numbers(valid, expected[:, :19]):>22s} {numbers(valid[:, 18], expected[:, 18]):>22s}'
          f'  nonzero(actual/expected)={np.count_nonzero(valid)}/{np.count_nonzero(expected[:, :19])}'
          f' of {valid.size} nonfinite(actual)={np.count_nonzero(~np.isfinite(valid))}')


def _relations(t, epsilon):
    # Pointwise consistency on recorded operands. Every activation uses [B,T,H,D].
    x = t['rms_input'].astype(np.float64)
    z = t['silu_input'].astype(np.float64)
    with np.errstate(invalid='ignore', over='ignore', under='ignore', divide='ignore'):
        yield 'rms_input / cast(core)*', t['rms_input'], t['core'].astype(t['rms_input'].dtype)
        if epsilon is not None:
            rms = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + epsilon)
            yield 'rms_output / RMS(x)*gamma', t['rms_output'], rms*t['gamma']
        yield 'silu_input / cast(gate)', t['silu_input'], t['gate'].astype(t['silu_input'].dtype)
        yield 'silu_output / SiLU(silu_input)', t['silu_output'], z*np.exp(-np.logaddexp(0., -z))
        yield 'mul_output / (lhs*rhs)', t['mul_output'], t['mul_lhs'].astype(np.float64)*t['mul_rhs']
        yield 'gated_return / cast(mul_output)', t['gated_return'], t['mul_output'].astype(t['gated_return'].dtype)


def audit_dump(root, trace, layers):
    manifest = json.loads((root / 'manifest.json').read_text())
    trace_air = _compiled_air(trace)
    for layer in layers:
        info = manifest['layers'][str(layer)]
        base_id = f'l{layer:02}_c00_gdn_layer_hidden_o0'
        _, _, base, _ = board_case(trace, base_id)
        print(f'L{layer}   comparison                         valid19 L2/cos         token18 L2/cos')
        for group in ('gdr', 'norm'):
            case = case_id(layer, group)
            _, _, _, record = _record(trace_air, case)
            evidence = info['graphs'][group]
            if (record['air']['sha256'] != evidence['trace_air_sha256']
                    or record['capture_sha256'] != manifest['capture_sha256']):
                raise ValueError(f'{case}: dump and recovered graphs use different AIR or capture')
            _, _, outputs, _ = board_case(trace, case)
            hidden = outputs['golden_hidden']
            equal = hidden[:, :19].tobytes() == base['golden_hidden'][:, :19].tobytes()
            if equal != evidence['valid_hidden_equal_bits']:
                raise ValueError(f'{case}: dump hidden reproduction differs from recovery')
            _print_gap(f'{group} hidden / original (bits={equal})', hidden, base['golden_hidden'])
            for role, key in evidence['aliases'].items():
                spec = info['tensors'][role]['cpro']
                if hashlib.sha256(outputs[key].tobytes()).hexdigest() != spec['sha256']:
                    raise ValueError(f'{case}/{role}: dump differs from the recovered output')
        print('* CPro core and rms_input come from two different compiled graphs.')
        for backend in ('310p', 'cpro'):
            print(backend)
            tensors = {role: read_array(root, specs[backend]) for role, specs in info['tensors'].items()}
            eps = info['epsilon'] if backend == '310p' else info['graphs']['norm']['air_rms_epsilon']
            for name, actual, expected in _relations(tensors, eps):
                _print_gap(name, actual, expected)

"""Summarize already-recovered PyTorch prefix tensors without model execution."""
import math

import numpy as np

from .accuracy_diagnose import hidden_stats
from .embedding_probe import DTYPES


def _difference(actual, expected, atol, rtol):
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(e).all())
    bad = ~np.isfinite(a) | (np.abs(a-e) > atol+rtol*np.abs(e))
    norm = float(np.linalg.norm(e))
    return dict(elements=int(a.size), failing_elements=int(bad.sum()), finite=finite,
                max_abs_error=float(np.max(np.abs(a-e))) if finite else None,
                relative_l2_error=float(np.linalg.norm(a-e)/norm) if finite and norm else None)


def diagnose(fixture, native, meta, comparison, report):
    """The caller rechecks all ten calls and fixture hashes before this read."""
    rows = []
    source = meta.get('source_evidence', {})
    extraction = source.get('extraction', {})
    for case in meta['cases']:
        name = case['case']
        if name not in comparison['validated_cases']:
            continue
        total = sum(s['bytes'] for s in case['outputs'])
        if total > 4 << 20:
            raise ValueError('prefix diagnosis is limited to 4 MiB of output per call')
        raw = (native/name/'0.bin').read_bytes()
        if len(raw) != total:
            raise ValueError('prefix diagnostic snapshot extent changed')
        outputs, arrays, offset = [], {}, 0
        summary = next(c for c in comparison['cases'] if c['case'] == name)
        for spec in case['outputs']:
            shape = spec['shape']
            if not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise ValueError('prefix diagnosis requires static positive dimensions')
            dtype = np.dtype(DTYPES[spec['dtype']])
            if spec['bytes'] != math.prod(shape)*dtype.itemsize:
                raise ValueError('prefix diagnostic dtype/shape extent differs')
            actual = np.frombuffer(raw[offset:offset+spec['bytes']], dtype=dtype).reshape(shape)
            offset += spec['bytes']
            expected = np.fromfile(fixture/name/f'golden_{spec["index"]}.bin', dtype=dtype).reshape(shape)
            a_rows, e_rows = actual.reshape(-1, shape[-1]), expected.reshape(-1, shape[-1])
            if len(a_rows) > 64:
                raise ValueError('prefix diagnosis is limited to 64 rows')
            status = next(s for s in summary['outputs'] if s['name'] == spec['name'])
            detail = dict(name=spec['name'], dtype=spec['dtype'], shape=shape, index=spec['index'],
                          **{k: v for k, v in status.items() if k != 'name'},
                          actual=hidden_stats(actual), reference=hidden_stats(expected),
                          rows=[dict(row=i, **_difference(a, e, meta['atol'], meta['rtol']),
                                     bytes_equal=a.tobytes() == e.tobytes(),
                                     actual_zero=bool(np.all(a == 0)))
                                for i, (a, e) in enumerate(zip(a_rows, e_rows))])
            # The prefix has one anchor row followed by repeated mask rows.
            detail['samples'] = [dict(row=i,
                actual_first8=[float(v) if np.isfinite(v) else None for v in a_rows[i, :8]],
                reference_first8=[float(v) if np.isfinite(v) else None for v in e_rows[i, :8]])
                for i in range(min(2, len(a_rows)))]
            if spec['name'] == 'ids_readback':
                detail['ids'] = dict(actual=actual.reshape(-1).tolist(), expected=expected.reshape(-1).tolist())
            outputs.append(detail)
            arrays[spec['name']] = actual
        row = dict(case=name, passed=summary['passed'],
                   repeat_bitwise_stable=summary['repeat_bitwise_stable'],
                   graphop_counts=summary['graphop_counts'], model_sha256=case['model']['sha256'],
                   numerics=summary['numerics'], outputs=outputs)
        if {'rms_x', 'rms_gamma', 'input_rms'} <= arrays.keys():
            x, gamma, y = (arrays[n] for n in ('rms_x', 'rms_gamma', 'input_rms'))
            epsilon = extraction.get('epsilon')
            if (not isinstance(epsilon, (float, int)) or not math.isfinite(epsilon) or epsilon <= 0
                    or x.shape != (16, gamma.size) or y.shape != (1,16,gamma.size)):
                raise ValueError('retained RMS tap geometry/epsilon differs')
            derived = dict(scope='CAPTURED_LOGICAL_EDGE_VALUES_NOT_KERNEL_INPUT_READBACK_OR_ACCEPTANCE_REFERENCE')
            observation = extraction.get('gamma_observation')
            factor_ok = True
            if observation is not None:
                factor = arrays.get(observation.get('factor_output'))
                if (observation.get('method') != 'SOURCE_GAMMA_TIMES_RUNTIME_UNITY'
                        or observation.get('required_value') != 1.0 or factor is None
                        or factor.shape != (1, gamma.size)):
                    raise ValueError('retained gamma observation contract differs')
                factor_ok = bool(np.all(factor == 1.0))
                derived.update(gamma_observation=observation, factor_readback_is_unity=factor_ok)
            if not factor_ok:
                derived['status'] = 'NONUNITY_GAMMA_FACTOR; DERIVED_COMPARISON_SKIPPED'
            elif all(np.isfinite(a).all() for a in (x, gamma, y)):
                with np.errstate(over='ignore', invalid='ignore'):
                    expected_from_taps = x * (np.float32(1)/np.sqrt(
                        np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32)+np.float32(epsilon))) * gamma
                if np.isfinite(expected_from_taps).all():
                    derived.update(status='COMPUTED', **_difference(
                        y.reshape(x.shape), expected_from_taps, meta['atol'], meta['rtol']))
                else:
                    derived['status'] = 'NONFINITE_DERIVED_VALUES'
            else:
                derived['status'] = 'NONFINITE_CAPTURED_VALUES'
            row['rms_from_captured_inputs'] = derived
        rows.append(row)
    return dict(artifact_kind='qwen35-draft-prefix-diagnosis', status='ANALYZED',
                source_status=report['status'], numerical_validation=comparison['numerical_validation'],
                model_execution='NOT_RUN', model_compilation='NOT_RUN', board_attempt=report.get('board_attempt'),
                scope='LOCAL_RETAINED_OUTPUTS; TEN_CALLS_RECOMPARED; ROW_DETAILS_FROM_ITERATION_0',
                source_air_sha256=extraction.get('source_air_sha256'),
                original_model_sha256=source.get('original_model_sha256'),
                source_float_paths=extraction.get('float_paths'), epsilon=extraction.get('epsilon'),
                unit_scale_bypass=extraction.get('unit_scale_bypass'),
                gamma_observation=extraction.get('gamma_observation'),
                scale=extraction.get('scale'), atol=meta['atol'], rtol=meta['rtol'], cases=rows)

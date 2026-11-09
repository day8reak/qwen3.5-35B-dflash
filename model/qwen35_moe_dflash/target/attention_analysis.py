"""Explain retained Target probe output bits without changing numerical gates."""
from pathlib import Path

import numpy as np

from . import cache_probe


def cache_pattern(actual, original, expected, written):
    """Compare physical FP16 bit images, including both signs of zero."""
    actual, original, expected = (a.view('<u2') for a in (actual, original, expected))
    bad = actual != expected
    writes_bad = int(np.count_nonzero(bad & written))
    untouched_bad = int(np.count_nonzero(bad & ~written))
    untouched_nonzero = int(np.count_nonzero(actual[~written]))
    if not np.any(bad):
        pattern = 'WHOLE_CACHE_EXACT'
    elif not writes_bad and not untouched_nonzero:
        pattern = 'WRITES_EXACT_UNTOUCHED_POSITIVE_ZERO'
    elif not np.any(actual):
        pattern = 'ALL_POSITIVE_ZERO'
    elif np.array_equal(actual, original):
        pattern = 'ORIGINAL_CACHE_UNCHANGED'
    else:
        pattern = 'OTHER_MISMATCH'
    examples = []
    for coord in np.argwhere(bad)[:3]:
        index = tuple(coord)
        examples.append(dict(coordinate=coord.tolist(), region='written' if written[index] else 'untouched',
            actual_bits=f'0x{int(actual[index]):04x}', expected_bits=f'0x{int(expected[index]):04x}',
            original_bits=f'0x{int(original[index]):04x}'))
    return dict(pattern=pattern, written_elements=int(written.sum()),
        untouched_elements=int((~written).sum()), mismatched_written_elements=writes_bad,
        mismatched_untouched_elements=untouched_bad,
        untouched_nonzero_bits=untouched_nonzero,
        mismatched_original_elements=int(np.count_nonzero(actual != original)),
        first_mismatches=examples)


def analyze(fixture, native, meta, comparison, retained):
    """Called after receipt, fixture hash and snapshot extent validation.

    Also bind every snapshot and receipt to the previously recovered report.
    Patterns describe output contents; they cannot establish kernel ownership,
    runtime aliasing, input mutation, or permission to ignore untouched bytes.
    """
    fixture, native = Path(fixture), Path(native)
    previous = {row['case']: row for row in retained['cases']}
    cases = {row['case']: row for row in meta['cases']}
    rows = []
    for row in comparison['cases']:
        name = row['case']
        old = previous[name]
        if row['runner_receipt_sha256'] != old['runner_receipt_sha256']:
            raise ValueError('runner receipt changed after recovery: '+name)
        old_hashes = [(i['index'], i['sha256']) for i in old['invocations']]
        new_hashes = [(i['index'], i['sha256']) for i in row['invocations']]
        if old_hashes != new_hashes:
            raise ValueError('output snapshots changed after recovery: '+name)
        summary = dict(case=name, status=row['status'], calls=len(new_hashes),
            repeat_bitwise_stable=row['repeat_bitwise_stable'], finite=row['finite'])
        rows.append(summary)
        if row['family'] == 'gdn':
            summary['outputs'] = {}
            for output in row['invocations'][0]['outputs']:
                values = [i['outputs'][output] for i in row['invocations']]
                summary['outputs'][output] = {k: (max(v[k] for v in values)
                    if all(v[k] is not None for v in values) else None)
                    for k in ('max_abs', 'rmse', 'relative_l2')}
                cosines = [v['cosine'] for v in values]
                summary['outputs'][output]['min_cosine'] = (min(cosines)
                    if all(v is not None for v in cosines) else None)
            continue
        case = cases[name]
        by_name = {s['name']: s for s in case['inputs']}
        inputs = [np.fromfile(fixture/name/f'input_{by_name[n]["index"]}.bin',
                    dtype='<f2' if dtype == 'float16' else '<i4').reshape(by_name[n]['shape'])
                  for n, dtype in zip(cache_probe.INPUTS, cache_probe.DTYPES)]
        expected = cache_probe.physical_reference(*inputs)
        outputs = case['outputs']
        if (len(outputs) != 1 or tuple(outputs[0]['shape']) != cache_probe.SHAPE
                or outputs[0]['dtype'] != 'float16'):
            raise ValueError('cache analysis requires the frozen whole-cache output')
        golden = (fixture/name/f'golden_{outputs[0]["index"]}.bin').read_bytes()
        if expected.tobytes() != golden:
            raise ValueError('retained cache golden differs from independent physical reference')
        written = np.zeros(cache_probe.SHAPE, dtype=bool)
        written[inputs[2], :, inputs[3], :] = True
        variants = {}
        for invocation in row['invocations']:
            index, digest = invocation['index'], invocation['sha256']
            if digest not in variants:
                actual = np.fromfile(native/name/f'{index}.bin', dtype='<f2').reshape(cache_probe.SHAPE)
                variants[digest] = dict(sha256=digest, calls=[],
                    **cache_pattern(actual, inputs[0], expected, written))
            variants[digest]['calls'].append(index)
        summary['output_patterns'] = list(variants.values())
    return dict(artifact_kind='qwen35-target-attention-retained-analysis',
        status='ANALYZED', numerical_validation=comparison['numerical_validation'],
        model_execution='NOT_RUN', device_query='NOT_RUN',
        snapshot_binding='MATCHES_RETAINED_RECOVERY_SHA256',
        coverage_complete=comparison['coverage_complete'], cases=rows,
        scope='OUTPUT_CONTENT_PATTERNS_ONLY; ALIAS_AND_INPUT_MUTATION_UNOBSERVED',
        all_operators_passed=False)

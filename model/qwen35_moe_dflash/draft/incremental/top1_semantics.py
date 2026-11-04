"""Additional inputs for existing Top1 OMCs; no graph or compiler changes."""
from copy import deepcopy

import numpy as np

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.run_layout import _save
from .top1_matrix import COMPOSITIONS

SMALL_CASES = ('top1_argmax_r1_d512_fp16', 'top1_argmax_r15_d485_fp16',
               'top1_argmax_r15_d512_fp16', 'top1_argmax_r485_d512_fp16')
CONTROL_CASES = ('top1_argmax_r7275_d512_fp16', *COMPOSITIONS)
BASE_CASES = (*SMALL_CASES, *CONTROL_CASES)
SCENARIOS = ('all_equal', 'first', 'last', 'finite_tie', 'negative_only',
             'signed_zero_tie', 'positive_inf_tie', 'all_negative_inf',
             'first_nan', 'last_nan', 'multiple_nan', 'all_nan', 'subnormal')
EXTRA_CASES = tuple(f'{case}__{scenario}' for case in SMALL_CASES for scenario in SCENARIOS)


def fixture(shape, scenario):
    if scenario not in SCENARIOS or tuple(shape) not in {(1, 512), (15, 485), (15, 512), (485, 512)}:
        raise ValueError('unsupported Top1 semantic fixture')
    rows, width = shape
    rr, cc = np.ogrid[:rows, :width]
    x = (((rr * 29 + cc * 13) % 31 - 16) / 4).astype('<f2')
    if scenario == 'all_equal':
        x.fill(1)
    elif scenario in ('first', 'last'):
        x[:, 0 if scenario == 'first' else -1] = 8
    elif scenario == 'finite_tie':
        x[:, [width // 2, width - 1]] = 8
    elif scenario == 'negative_only':
        x.fill(-8)
        x[np.arange(rows), (np.arange(rows) * 37 + width - 1) % width] = -1
    elif scenario == 'signed_zero_tie':
        x.fill(-1)
        x[:, width // 2], x[:, -1] = np.float16(-0.), np.float16(0.)
    elif scenario == 'positive_inf_tie':
        x[:, [width // 2, width - 1]] = np.inf
    elif scenario == 'all_negative_inf':
        x.fill(-np.inf)
    elif scenario == 'first_nan':
        x[:, 0], x[:, -1] = np.nan, 8
    elif scenario == 'last_nan':
        x[:, 0], x[:, -1] = 8, np.nan
    elif scenario == 'multiple_nan':
        x[:, 0] = 8
        x[:, [width // 3, width - 1]] = np.nan
    elif scenario == 'all_nan':
        x.fill(np.nan)
    elif scenario == 'subnormal':
        x.fill(0)
        x[np.arange(rows), (np.arange(rows) * 37 + width - 1) % width] = np.float16(2 ** -24)
    return x, np.argmax(x, axis=1).astype('<i4')


def expand(out, meta):
    """Called after the original nine recipes, ABI and fixtures were verified."""
    if [row['case'] for row in meta['cases']] != list(BASE_CASES):
        raise ValueError('Top1 semantics requires the fixed nine compiled controls')
    # Collect every original control before additional inputs: an SDK failure
    # still stops the supervisor, so retain useful baseline coverage first.
    cases = [dict(deepcopy(row), compiled_case=row['case'], scenario='original_fixture')
             for row in meta['cases']]
    for original in meta['cases']:
        base = original['case']
        if base not in SMALL_CASES:
            continue
        if (len(original['inputs']) != 1 or len(original['outputs']) != 1
                or original['inputs'][0]['dtype'] != 'float16'
                or original['outputs'][0]['dtype'] != 'int32'):
            raise ValueError('Top1 semantic fixture ABI differs')
        for scenario in SCENARIOS:
            row = deepcopy(original)
            name = base + '__' + scenario
            row.update(case=name, compiled_case=base, scenario=scenario)
            row['model']['remote'] = name + '/model.omc'
            x, golden = fixture(original['inputs'][0]['shape'], scenario)
            folder = out / 'fixture' / name
            folder.mkdir()
            for key, prefix, value in (('inputs', 'input', x), ('outputs', 'golden', golden)):
                spec = row[key][0]
                if list(value.shape) != spec['shape'] or value.nbytes != spec['bytes']:
                    raise ValueError('Top1 semantic buffer differs from compiled ABI')
                path = folder / f'{prefix}_{spec["index"]}.bin'
                value.tofile(path)
                spec['fixture_sha256'] = sha256_file(path)
            (folder / 'plan.txt').write_bytes((out / 'fixture' / base / 'plan.txt').read_bytes())
            cases.append(row)
    result = dict(meta, cases=cases, exact_bytes=True,
        scope='EXISTING_TOP1_OMCS_NEW_INPUT_SEMANTICS_AND_API_TIMING_NOT_FULL_DRAFT',
        top1_semantics=dict(compiled_models=len(BASE_CASES), execution_cases=len(cases),
            small_cases=list(SMALL_CASES), control_cases=list(CONTROL_CASES),
            scenarios=['original_fixture', *SCENARIOS], model_compilation='NOT_RUN',
            reference='DIRECT_NUMPY_ARGMAX_HOST_CROSS_CHECKED_WITH_PYTORCH',
            contract='FIRST_NAN_ELSE_FIRST_MAXIMUM; EXACT_INT32_BYTES_ALL_TEN_REPEATS'))
    _save(out / 'fixture' / 'manifest.json', result)
    return result


def summary(meta, cases):
    groups = []
    for base in BASE_CASES:
        requested = [r for r in meta['cases'] if r['compiled_case'] == base]
        completed = [r for r in cases if r['compiled_case'] == base]
        groups.append(dict(compiled_case=base, requested=len(requested), completed=len(completed),
            model_sha256=requested[0]['model']['sha256'],
            graphop_counts=requested[0]['graphop_counts'],
            inputs=[{k: spec[k] for k in ('shape', 'dtype')} for spec in requested[0]['inputs']],
            passed=sum(r['passed'] for r in completed),
            failed_scenarios=[r['scenario'] for r in completed if not r['passed']],
            repeat_bitwise_stable=bool(completed) and all(r['repeat_bitwise_stable'] for r in completed),
            observations=[{k: r[k] for k in ('scenario', 'passed', 'mean_ms', 'min_ms', 'max_ms',
                'mismatched_elements_all_calls', 'mismatch_examples')} for r in completed]))
    return dict(meta['top1_semantics'], groups=groups,
        timing_scope='PER_EXISTING_MODEL_AND_FIXTURE; API_WALL_AFTER_3_WARMUPS_10_CALLS; NOT_ADDITIVE_OR_FULL_DRAFT_SPEEDUP')

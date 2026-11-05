"""Bounded same-input tail on the retained OMC, separate from full accuracy."""
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.execution.runtime_config import read_json
from .accuracy import metrics, read_snapshot

VERSION = 1
CALLS = 20


def request(meta, index):
    if type(index) is not int or not 0 <= index < len(meta['cases']):
        raise ValueError('KV reproduction case is outside the retained schedule')
    if meta['cases'][index]['index'] != index or not meta['cases'][index]['valid']:
        raise ValueError('KV reproduction requires an appending case')
    return dict(version=VERSION, case=index, original_calls=2*len(meta['cases']),
                planned_calls=CALLS, commit_outputs=False)


def validate_report(report, requested, meta):
    if requested != request(meta, requested['case']):
        raise ValueError('invalid retained KV reproduction request')
    if (any(report.get(k) != v for k, v in requested.items())
            or report.get('evidence_kind') != 'device-hiai'
            or report.get('status') not in ('EXECUTED', 'FAILED')):
        raise ValueError('KV reproduction report differs from the requested execution')
    rows = report.get('calls', [])
    if (not isinstance(rows, list) or len(rows) > CALLS
            or (report['status'] == 'EXECUTED' and len(rows) != CALLS)
            or (report['status'] == 'FAILED' and
                (len(rows) == CALLS or report.get('failure', {}).get('call') != len(rows)))):
        raise ValueError('incomplete or invalid KV reproduction call inventory')
    inputs = {r['index']: r for r in meta['input_files'] if r['case'] == requested['case']}
    for ordinal, row in enumerate(rows):
        if (row.get('call') != ordinal or row.get('strategy') != ('reuse' if ordinal % 2 else 'rewrite')
                or type(row.get('api_us')) is not int or row['api_us'] < 0):
            raise ValueError('KV reproduction call order or timing differs')
        for phase in ('input_before', 'input_after'):
            checks = row.get(phase, [])
            if len(checks) != len(meta['inputs']):
                raise ValueError('KV reproduction input readback inventory is incomplete')
            for i, check in enumerate(checks):
                if (check.get('index') != i or check.get('matches') is not True
                        or any(check.get(k) != inputs[i][k] for k in ('bytes', 'sha256'))):
                    raise ValueError('KV reproduction input readback differs from retained fixture')
    return [f'kv-repeat-{i}.bin' for i in range(len(rows))]


def analyze(fixture, native, requested):
    fixture, native = Path(fixture), Path(native)
    meta = read_json(fixture/'manifest.json')
    report = read_json(native/'kv-reproduction.json')
    names = validate_report(report, requested, meta)
    step = meta['cases'][requested['case']]
    index, start, valid, count = (step[k] for k in ('index', 'start', 'valid', 'proposals'))
    end = start+valid
    atol, rtol = (meta['comparison'][k] for k in ('atol', 'rtol'))
    original = {mode: read_snapshot(native/f'{mode}-{index}.bin', meta['outputs'])
                for mode in ('isolated', 'feedback')}
    def signatures(banks):
        return [b''.join(a[:, :, pos].tobytes() for a in banks) for pos in range(start, end)]
    def duplicates(rows):
        return [[start+i, start+j] for i in range(valid) for j in range(i+1, valid) if rows[i] == rows[j]]
    original_rows = {mode: signatures(values[2:]) for mode, values in original.items()}
    golden = [np.fromfile(fixture/'golden'/str(index)/f'kv_{j}.bin', dtype='<f2')
              .reshape(s['shape']) for j, s in enumerate(meta['outputs'][2:])]
    old = []
    for spec in meta['outputs'][2:]:
        slot = next(s['index'] for s in meta['inputs'] if s['name'] == spec['name'].replace('present_', 'past_'))
        old.append(np.fromfile(fixture/'inputs'/str(index)/f'{slot}.bin', dtype='<f2').reshape(spec['shape']))
    outside = np.ones(meta['cache_capacity'], dtype=bool); outside[start:end] = False
    results, summaries, previous = [], [], {}
    for call, name in zip(report['calls'], names):
        values = read_snapshot(native/name, meta['outputs'])
        banks = values[2:]
        row = dict(call=call['call'], strategy=call['strategy'], passed=True, tensors=[])
        if count:
            ids = np.fromfile(fixture/'golden'/str(index)/'top1.bin', dtype='<i4')
            row['top1_mismatches'] = int(np.count_nonzero(values[0].reshape(-1)[:count] != ids))
            hidden = np.fromfile(fixture/'golden'/str(index)/'hidden.bin', dtype='<f2').reshape(1, count, -1)
            metric = dict(name='draft_hidden', **metrics(values[1][:, :count], hidden, atol, rtol))
            row['tensors'].append(metric)
            row['passed'] &= metric['passed'] and row['top1_mismatches'] == 0
        for j, (got, ref) in enumerate(zip(banks, golden)):
            preserved = got[:, :, outside].tobytes() == old[j][:, :, outside].tobytes()
            metric = dict(name=meta['outputs'][j+2]['name'], **metrics(got, ref, atol, rtol),
                          preserved=preserved, appended=metrics(got[:, :, start:end], ref[:, :, start:end], atol, rtol))
            metric['passed'] &= preserved
            row['tensors'].append(metric); row['passed'] &= metric['passed']
        # Complete post-RoPE K and V signatures; equality is evidence of a row
        # duplicate, not proof of a particular internal index or faulty operator.
        current_rows = signatures(banks)
        appended = b''.join(a[:, :, start:end].tobytes() for a in banks)
        strategy = call['strategy']
        summary = dict(call=call['call'], strategy=strategy, passed=row['passed'],
            exact_kv_row_duplicates=duplicates(current_rows),
            appended_equal_previous_same_strategy=(appended == previous[strategy] if strategy in previous else None),
            appended_equal_original={mode: all(a[:, :, start:end].tobytes() == b[:, :, start:end].tobytes()
                for a, b in zip(banks, original[mode][2:])) for mode in original},
            changed_positions_vs_original={mode: [start+i for i in range(valid) if current_rows[i] != rows[i]]
                                           for mode, rows in original_rows.items()},
            preservation_passed=all(t['preserved'] for t in row['tensors'] if 'preserved' in t),
            max_appended_relative_l2=max((t['appended']['relative_l2_error'] for t in row['tensors']
                if t.get('appended', {}).get('relative_l2_error') is not None), default=None))
        previous[strategy] = appended
        row.update(summary); results.append(row); summaries.append(summary)
    summary = dict(status='ANALYZED' if report['status'] == 'EXECUTED' else 'INCOMPLETE',
        native_status=report['status'], source_case=step, completed_calls=len(names), planned_calls=CALLS,
        numerical_validation=('FAILED' if any(not r['passed'] for r in results) else
                              'PASSED' if len(names) == CALLS else 'NOT_EVALUATED'),
        input_readback='CPU_BUFFER_HASHES_BEFORE_AND_AFTER_PROCESS; NOT_NPU_READBACK',
        original_schedule='UNCHANGED; TAIL_AFTER_BOTH_SEQUENCES; SAME_MODEL_LOAD',
        original_case=dict(exact_kv_row_duplicates={mode: duplicates(rows) for mode, rows in original_rows.items()},
            changed_positions_between_modes=[start+i for i in range(valid)
                if original_rows['isolated'][i] != original_rows['feedback'][i]],
            reference_exact_kv_row_duplicates=duplicates(signatures(golden))),
        commit_outputs=False, calls=summaries,
        limitations=['Readback can perturb visibility; this tail cannot reproduce every runtime history.',
                    'No internal Gather index buffer is observed; stable repeats do not establish full-model accuracy.'])
    if 'failure' in report:
        summary['failure'] = report['failure']
    return dict(summary=summary, comparison=meta['comparison'], model=meta['model'], cases=results)

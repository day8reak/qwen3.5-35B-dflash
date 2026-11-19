"""Read slot and token comparisons from a recovered four-layer valid19 execution."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.accuracy import floating_metrics
from ..layer_probe.common import DTYPES, read_array, sha
from .segment_export import CASE, OUTPUTS


def _load(recovery):
    report = json.loads((recovery / 'result.json').read_text())
    if (report.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'
            or report.get('validated_cases') != [CASE]):
        raise ValueError('Expected a completed four-layer segmented prefill recovery')
    source = Path(report['recovery_source'])
    request = json.loads((source / 'request.json').read_text())
    fixture = source / 'fixture'
    if (request['board_attempt'] != report['board_attempt']
            or sha(fixture / 'manifest.json') != request['fixture_sha256']):
        raise ValueError('Recovery and launch fixture do not match')
    case, = json.loads((fixture / 'manifest.json').read_text())['cases']
    specs = case['outputs']
    if (case['case'] != CASE or case['kind'] != 'segment_prefill'
            or [s['semantic_name'] for s in specs] != list(OUTPUTS)):
        raise ValueError('Expected the segmented logits and eight state outputs')
    rows = {r['output_name']: r for r in report['cases']}
    snapshot = recovery / 'native' / CASE / '0.bin'
    if snapshot.stat().st_size != sum(s['bytes'] for s in specs):
        raise ValueError('Snapshot extent differs from the compiled outputs')
    actual, golden = {}, {}
    with snapshot.open('rb') as stream:
        for index, spec in enumerate(specs):
            name = spec['semantic_name']
            row = rows[name]
            raw = stream.read(spec['bytes'])
            if (spec['index'] != index or row['output_index'] != index or row['case'] != CASE
                    or hashlib.sha256(raw).hexdigest() != row['output_sha256'][0]):
                raise ValueError(f'Saved invocation 0 does not match its compared slot: {name}')
            actual[name] = np.frombuffer(raw, dtype=DTYPES[spec['dtype']]).reshape(spec['shape'])
            golden[name] = read_array(fixture, dict(path=f'{CASE}/golden_{index}.bin',
                bytes=spec['fixture_bytes'], sha256=spec['fixture_sha256'],
                dtype=spec['golden_dtype'], shape=spec['shape']))
    return actual, golden, rows


def _rel(actual, reference):
    value = floating_metrics(actual, reference)['relative_l2']
    return f'{value:.6g}' if value is not None and math.isfinite(value) else 'n/a'


def inspect(recovery):
    actual, golden, rows = _load(recovery)
    logits = rows['logits']['invocations'][0]
    lines = [f"Token 18 logits top1: CPro={logits['actual_top1']} 310P={logits['native_top1']} "
             f"propagated CPU={logits.get('cpu_top1', 'unavailable')}"]
    if 'vs_propagated_cpu' in logits:
        lines.append('CPro vs propagated CPU, relative L2 (CPU conv history uses a different tail):')
        for name in OUTPUTS:
            if not name.startswith('conv_state_'):
                cpu = rows[name]['invocations'][0].get('vs_propagated_cpu', {})
                value = cpu.get('relative_l2')
                lines.append(f'  {name:<19} ' + ('n/a' if value is None else f'{value:.6g}'))
    lines.append('Invocation 0. Rows: CPro slots. Columns: 310P references. Values: relative L2.')
    caches = ('key_cache_3', 'value_cache_3')
    for name in caches:
        if actual[name].shape != (64, 32, 32, 16) or rows[name]['invocations'][0]['valid_rows'] != 19:
            raise ValueError(f'Expected the first 19 valid K/V positions on physical page 0: {name}')
        actual[name] = actual[name][0, :, :19, :]
        golden[name] = golden[name][0, :, :19, :]

    for names in (('conv_state_0', 'conv_state_1', 'conv_state_2'),
                  ('recurrent_state_0', 'recurrent_state_1', 'recurrent_state_2'), caches):
        lines.append('')
        lines.append('CPro'.ljust(20) + ''.join(f'{name:>19}' for name in names))
        for name in names:
            lines.append(f'{name:<20}' + ''.join(f'{_rel(actual[name], golden[ref]):>19}' for ref in names))
    lines.append('\nValid token     key rel_L2   value rel_L2')
    for token in range(19):
        values = [_rel(actual[name][:, token], golden[name][:, token]) for name in caches]
        lines.append(f'{token:11d} {values[0]:>14} {values[1]:>14}')
    return '\n'.join(lines)

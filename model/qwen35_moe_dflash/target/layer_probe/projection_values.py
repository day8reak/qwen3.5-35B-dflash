"""Compare CPU projections with observable values in retained whole-module outputs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import DTYPES, KIND, checked, metrics, read_array, sha


def _json(path):
    return json.loads(path.read_text())


def _observable(cpu, board, original, case):
    if cpu.ndim != 3 or cpu.shape[0] != 1 or board.shape != original.shape:
        raise ValueError('expected matching state shapes and CPU projection [1,T,C]')
    _, tokens, channels = cpu.shape
    if case['kind'] == 'gdn_layer':
        if (case.get('convstate_mode') != 'reference' or board.ndim != 3
                or board.shape[:2] != (1, channels) or not 0 < board.shape[2] <= tokens):
            raise ValueError('requires explicit concat/tail with T >= state width')
        width = board.shape[-1]
        return cpu[0, -width:], board[0].T, original[0].T

    contract = case['full_attention_contract']
    block = contract['block_size']
    table = np.asarray(contract['block_table'])
    if (case.get('reference_basis') != 'ORIGINAL_FULL_ATTENTION_CAPTURE_WITH_AUDITED_CORE'
            or contract['sequence_length'] != tokens or board.ndim != 4 or block < 1
            or board.shape[2:] != (block, 16) or board.shape[1] * 16 != channels
            or table.shape[:1] != (1,) or table.ndim != 2 or table.dtype.kind not in 'iu'
            or table.shape[1] * block < tokens):
        raise ValueError('requires original fresh Prefill cache and captured block table')
    ids = np.arange(tokens)
    pages = table[0, :(tokens+block-1)//block]
    if pages.min() < 0 or pages.max() >= board.shape[0] or np.unique(pages).size != pages.size:
        raise ValueError('invalid active cache pages')
    def rows(cache):
        return cache.transpose(0, 2, 1, 3)[pages[ids//block], ids % block].reshape(tokens, channels)
    return cpu[0], rows(board), rows(original)


def compare_values(cpu_dir, group):
    cpu_dir, group = Path(cpu_dir), Path(group)
    cpu_report = _json(cpu_dir/'report.json')
    fixture = group/'execution/fixture'
    meta = _json(fixture/'manifest.json')
    source = group/'omc/manifest.json'
    if (meta.get('artifact_kind') != KIND or cpu_report.get('scope') != 'SELECTED_QLINEAR_GOLDEN_MATH_ONLY'
            or sha(source) != meta['source_evidence']['manifest_sha256']
            or cpu_report['capture_sha256'] != _json(source)['capture_sha256']):
        raise ValueError('CPU output and device fixture must derive from the same capture')
    compared = _json(group/'comparison/report.json')['cases']
    lines = []
    for case in meta['cases']:
        layer, kind, name = case['layer'], case['kind'], case['case']
        if (case['phase'] != 'prefill' or case['call'] != 0 or name != f'l{layer:02d}_c00_{kind}_all'
                or (layer, kind) not in ((0, 'gdn_layer'), (1, 'gdn_layer'), (2, 'gdn_layer'), (3, 'full_attention_layer'))):
            raise ValueError(f'unexpected Prefill case {name}')
        point, output = (('linear_attn_in_proj_qkv', 'golden_new_conv_state') if kind == 'gdn_layer'
                         else ('self_attn_v_proj', 'golden_value_cache'))
        case_id = f'l{layer:02d}_c00_{point}'
        cpu_case, = (r for r in cpu_report['cases'] if r['id'] == case_id and r['status'] == 'COMPUTED')
        specs = case['outputs']
        spec, = (s for s in specs if s['semantic_name'] == output)
        i = spec['index']
        if ([s['index'] for s in specs] != list(range(len(specs))) or spec['dtype'] != 'float16'
                or spec['golden_dtype'] != 'float16'
                or any(s['bytes'] != math.prod(s['shape'])*np.dtype(DTYPES[s['dtype']]).itemsize for s in specs)):
            raise ValueError('compiled output tuple or extent differs from fixture')
        original_path = checked(fixture, dict(path=f'{name}/golden_{i}.bin', bytes=spec['fixture_bytes'],
                                              sha256=spec['fixture_sha256']))
        original = np.fromfile(original_path, dtype='<f2').reshape(spec['shape'])
        native = group/'recovery/native'/name
        receipt = _json(native/'result.json')
        previous, = (r for r in compared if r['case'] == name and r.get('output_name') == output)
        hashes = previous['output_sha256']
        if (receipt.get('evidence_kind') != 'device-hiai' or receipt.get('status') != 'EXECUTED'
                or receipt.get('repetitions') != meta['repetitions'] or receipt.get('input_reset') != 'copy-each-call'
                or not previous['stable'] or not hashes or len(set(hashes)) != 1 or len(hashes) != meta['repetitions']):
            raise ValueError(f'{name}: need stable recovered device outputs')
        sample = native/'0.bin'
        if sample.stat().st_size != sum(s['bytes'] for s in specs):
            raise ValueError('device sample extent differs from compiled output tuple')
        board = np.fromfile(sample, dtype='<f2', count=math.prod(spec['shape']),
                            offset=sum(s['bytes'] for s in specs[:i])).reshape(spec['shape'])
        if hashlib.sha256(board.tobytes()).hexdigest() != hashes[0]:
            raise ValueError('recovered sample differs from recorded comparison')
        for label in ('cpu_reference', 'replayed_quant_reference'):
            if label not in cpu_case:
                continue
            cpu = read_array(cpu_dir, cpu_case[label]['tensor'])
            if cpu.dtype != np.float16:
                raise ValueError('expected FP16 projection reference')
            cpu_rows, board_rows, original_rows = _observable(cpu, board, original, case)
            for pair, actual, reference in (('board_vs_cpu', board_rows, cpu_rows),
                                            ('cpu_vs_310p', cpu_rows, original_rows),
                                            ('board_vs_310p', board_rows, original_rows)):
                values = metrics(actual, reference)
                lines.append(dict(case=case_id, reference=label, pair=pair, tokens=actual.shape[0],
                    elements=actual.size, **{k: values[k] for k in ('relative_l2', 'cosine', 'max_abs', 'passed')}))
    if not lines:
        raise ValueError('no Prefill projection values found')
    return lines

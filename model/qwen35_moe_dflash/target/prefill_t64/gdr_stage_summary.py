"""Build a portable index and a stage-ordered summary for the recovered 64/19 measurements."""
import json

import numpy as np

from ..layer_probe.common import identity, read_array
from .gdr_stage_data import DESCRIPTIONS, GDR, NORM


ENTRY = {
    'hidden': '整层入口 hidden，尚未做本层投影',
    'conv': '进入本层时的卷积历史',
    'state': '进入本层时的 recurrent state，保留各后端输入 ABI 的 dtype',
    'effective_length': '传给整层的有效 token 数',
    'linear_update_tensor': '分段推理交给本层卷积更新的输入',
}


def write_inputs(root, layer, reference, module, baseline, case, fixture_case, info, tensor_map, rows, score):
    fixture = baseline / 'execution/fixture' / case
    specs = {s['semantic_name']: s for s in fixture_case['inputs']}
    if not specs.keys() <= module['inputs'].keys():
        raise ValueError(f'{case}: board fixture contains an input absent from the capture')
    info['inputs'] = {}
    for role, native in module['inputs'].items():
        ref = read_array(reference, native)
        values = [('310p', ref, reference / native['path'])]
        slot = specs.get(role)
        if slot is not None:
            board = dict(path=f"input_{slot['index']}.bin", dtype=slot['dtype'], shape=slot['shape'],
                         bytes=slot['fixture_bytes'], sha256=slot['fixture_sha256'])
            act = read_array(fixture, board)
            if act.shape != ref.shape or act.tobytes() != ref.astype(act.dtype).tobytes():
                raise ValueError(f'{case}/{role}: compiled fixture differs from the 310P module input')
            values.append(('cpro', act, fixture / board['path']))
        entry = info['inputs'][role] = {}
        stage = 'entry_' + role
        for backend, value, src in values:
            path = root / f'l{layer}' / backend / 'inputs' / (role + '.bin')
            path.parent.mkdir(parents=True, exist_ok=True)
            value.tofile(path)
            entry[backend] = dict(identity(path, root), shape=list(value.shape), dtype=value.dtype.name)
            tensor_map.append(dict(layer=layer, backend=backend, tensor=stage, step='整层入口', meaning=ENTRY[role],
                baseline_hidden_reproduced='', path=path.relative_to(root).as_posix(), shape=json.dumps(list(value.shape)),
                dtype=value.dtype.name, bytes=value.nbytes, source=str(src), source_slot=slot['index'] if backend == 'cpro' else '',
                source_offset_bytes=0, source_length_bytes=value.nbytes, measurement='MODULE_INPUT' if slot else 'REFERENCE_INPUT_ONLY',
                source_case=case if backend == 'cpro' else ''))
        if slot is None:
            continue
        if role == 'hidden':
            for scope, index in [('valid19', slice(0, 19)), ('token18', 18)]:
                score(rows, layer, stage, 'cpro_vs_310p', scope, act[:, index], ref[:, index], reproduced='')
        else:
            score(rows, layer, stage, 'cpro_vs_310p', 'tensor', act, ref, reproduced='')


def check_measured_product(root, info, rows, score):
    for backend in ('310p', 'cpro'):
        lhs, rhs, product = (read_array(root, info['tensors'][r][backend]) for r in ('mul_lhs', 'mul_rhs', 'mul_output'))
        expected = lhs.astype(np.float64) * rhs
        for scope, idx in [('valid19', slice(0, 19)), ('token18', 18)]:
            score(rows, 0, 'mul_output', backend + '_product_vs_measured_lhs_times_rhs', scope,
                  product[:, idx], expected[:, idx], reproduced=True if backend == 'cpro' else '')


def write_overview(root, rows, layers):
    order = [*('entry_' + k for k in ENTRY), *GDR, *NORM, 'golden_hidden']
    rows.sort(key=lambda r: (r['layer'], order.index(r['stage']), r['comparison'] != 'cpro_vs_310p',
                             r['comparison'], r['scope'] == 'token18'))
    lines = ['# GDN prefill 逐层精度汇总', '', '物理 T64，有效 token 0–18，GDR chunk=64。各层使用各自的 310P 整层入口。',
             '逐 token 张量保留全部 64 行。valid19 和 token18 分别评分，状态、权重及长度按整个张量评分。',
             '浮点数显示相对 L2 / cosine。PASS 和 FAIL 按五项指标判定，整数按精确相等判定。',
             '仅诊断表示回读图未复现原始有效 hidden 或模块输入。其 CPro 文件在 diagnostic/ 下，原始读数保留在 CSV。',
             'L0 的 SiLU 输入和输出来自同次调用。其他合格读数来自相同整层输入、复现基线 hidden 的探针。310P 单端入口见文件索引中的 REFERENCE_INPUT_ONLY。',
             '', '[逐项指标](summary.csv) · [每个 bin 的步骤、dtype、shape 和来源](tensor-map.csv) · [SiLU 同次调用检查](silu-result.json)', '']
    for idx, info in layers.items():
        lines.extend([f'## L{idx}', '', '| 计算位置 | 张量或检查 | valid19 或全张量 L2 / cos | token18 L2 / cos | 结果 |',
                      '|---|---|---:|---:|---|'])
        groups = {}
        for row in (r for r in rows if r['layer'] == int(idx)):
            stage = row['stage']
            if stage.startswith('entry_'):
                measurement = 'MODULE_INPUT'
            elif row['comparison'].startswith('310p_'):
                measurement = 'REFERENCE_CHECK'
            else:
                graph, = [g for g in info['graphs'].values() if stage in g['aliases']]
                measurement = ('BASELINE' if stage == 'golden_hidden' else 'QUALIFIED') if (
                    graph['valid_hidden_equal_bits'] and graph['module_inputs_equal_baseline']) else 'DIAGNOSTIC_ONLY'
            row['measurement'] = measurement
            row['result'] = 'DIAGNOSTIC_ONLY' if measurement == 'DIAGNOSTIC_ONLY' else ('PASS' if row['passed'] else 'FAIL')
            groups.setdefault((stage, row['comparison']), []).append(row)
        for (stage, comparison), scores in groups.items():
            by_scope = {}
            for row in scores:
                if row['result'] == 'DIAGNOSTIC_ONLY':
                    value = '—'
                elif row['actual_dtype'].startswith('int'):
                    value = '相等' if row['equal_bits'] else '不同'
                else:
                    value = ' / '.join('n/a' if row[k] is None else f'{row[k]:.6g}' for k in ('relative_l2', 'cosine'))
                by_scope[row['scope']] = value
            position = '整层入口' if stage.startswith('entry_') else DESCRIPTIONS[stage][0]
            checks = {'310p_vs_silu_of_own_input': '310P 实测 / SiLU(310P 输入)',
                      'cpro_vs_silu_of_own_input': 'CPro 实测 / SiLU(CPro 输入)',
                      '310p_product_vs_measured_lhs_times_rhs': '310P 实测 / 两侧实测操作数之积',
                      'cpro_product_vs_measured_lhs_times_rhs': 'CPro 实测 / 两侧实测操作数之积'}
            label = checks.get(comparison, stage)
            status = '仅诊断' if scores[0]['result'] == 'DIAGNOSTIC_ONLY' else ('PASS' if all(r['passed'] for r in scores) else 'FAIL')
            lines.append(f"| {position} | {label} | {by_scope.get('valid19', by_scope.get('tensor', '—'))} | {by_scope.get('token18', '—')} | {status} |")
        lines.append('')
    (root / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

"""Package the existing L0 chunk64 operator measurements without model or board access."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from ...common.accuracy import ELEMENTWISE_TOLERANCE, ERROR_RATIO_FIELDS, ERROR_RATIO_LIMIT, THRESHOLDS
from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import identity, metrics, read_array, save_json
from .gdr_math import compute
from .gdr_stage_data import DESCRIPTIONS, GDR, STATIC, TOKEN_OPERANDS, logical

BUNDLE = 'gdr-l0-valid19'


def _comparison(name, scope, actual, reference, label='cpro'):
    m = metrics(actual, reference)
    fields = (*THRESHOLDS, 'mean_abs', 'max_abs', *ERROR_RATIO_FIELDS)
    row = {'tensor': name, 'scope': scope, f'{label}_dtype': actual.dtype.name, 'reference_dtype': reference.dtype.name}
    row.update(gate=m['gate'], passed=m['passed'], failed_metrics=','.join(m.get('failed_metrics', [])),
               **{k: m.get(k) for k in fields})
    row[f'{label}_zeros'] = int(np.count_nonzero(actual == 0))
    row['reference_zeros'] = int(np.count_nonzero(reference == 0))
    zero_column = f'reference_fp16_subnormals_with_{label}_zero'
    row[zero_column] = None
    row[f'{label}_nonfinite'] = m['nonfinite_actual']
    row['reference_nonfinite'] = m['nonfinite_golden']
    if reference.dtype == np.float16:
        small = (reference != 0) & (np.abs(reference) < np.finfo(np.float16).tiny)
        row[zero_column] = int(np.count_nonzero(small & (actual == 0)))
    return row


def _compare_scopes(name, actual, golden, *, label='cpro'):
    if name in STATIC:
        yield _comparison(name, 'tensor', actual, golden, label)
        return
    yield _comparison(name, 'valid19', actual[:, :19], golden[:, :19], label)
    for t in range(19):
        yield _comparison(name, f'token{t}', actual[:, t], golden[:, t], label)


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_cpu(root, attrs, measured, files):
    if attrs['chunk_size'] != 64 or not attrs['use_qk_l2norm_in_kernel'] or not attrs['output_final_state']:
        raise ValueError('CPU replay expects normalized Q/K, chunk64 and a returned state')
    length = int(measured['effective_length']['310p'].item())
    inputs = [measured[name]['310p'][:, :length] for name in TOKEN_OPERANDS]
    core, state = compute(*inputs, measured['initial_state']['310p'])
    rows = []
    for name, scope, golden in [('core', 'valid19', core), ('final_state', 'tensor', state)]:
        path = root / 'cpu' / scope / 'output' / (name + '.bin')
        path.parent.mkdir(parents=True, exist_ok=True)
        golden.tofile(path)
        files.append(dict(**identity(path, root), tensor=name, io='output', backend='cpu', scope=scope,
                          dtype=golden.dtype.str, shape=list(golden.shape), input_backend='310p'))
        for backend, actual in measured[name].items():
            for row in _compare_scopes(name, actual, golden, label='actual'):
                rows.append(dict(actual_backend=backend, reference_backend='cpu', **row))
    _write_csv(root / 'cpu-comparison.csv', rows)
    return dict(input_backend='310p', input_tensors=[*TOKEN_OPERANDS, 'initial_state', 'effective_length'],
                valid_rows=length, algorithm='gdr_math.compute', accumulation_dtype='float32',
                core_dtype=core.dtype.name, final_state_dtype=state.dtype.name)


def package(dump, output, *, cpu_golden=False):
    source = (dump / 'manifest.json').read_bytes()
    manifest = json.loads(source)
    if (manifest['physical_rows'], manifest['valid_rows'], manifest['last_valid_index'],
            manifest['chunk_size'], manifest['reference_backend']) != (64, 19, 18, 64, '310p'):
        raise ValueError('Expected a measured L0 chunk64 T64/valid19 stage dump with 310P reference')
    layer = manifest['layers']['0']
    graph = layer['graphs']['gdr']
    if not (graph['valid_hidden_equal_bits'] and graph['module_inputs_equal_baseline']):
        raise ValueError('The L0 GDR readout did not reproduce the baseline module inputs and valid hidden')
    target = output_path(output)
    if not target.name.endswith('.tar.gz'):
        raise ValueError('Choose a .tar.gz output filename')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as folder:
        root = Path(folder) / BUNDLE
        root.mkdir()
        rows, files = [], []
        description = [
            '# L0 prefill · ChunkGatedDeltaRule · T64 / valid19', '',
            '310p 和 cpro 分别保存各自算子入口、出口的实测数据。整层入口取自 310P，两端分别计算 Q/K/V/g/beta。',
            '复现原调用时使用同一后端 full64/ 下的逐 token 张量和 tensor/ 下的状态、长度。'
            'valid19/ 是分析视图，按时间轴取第 0–18 行。token18 是最后一个有效位置。原始 padding 行也可能有非零值。',
            '全部 bin 使用小端、C 顺序、无文件头。manifest.json 逐文件给出相对路径、dtype、shape 和 SHA256。', '',
            '| 文件名 | 方向 | 含义 | full64 或 tensor 形状 | 310P / CPro dtype |',
            '|---|---|---|---|---|',
        ]
        measured = {}
        for name in GDR:
            role = 'output' if name in ('core', 'final_state') else 'input'
            step, meaning = DESCRIPTIONS[name]
            pair = {}
            for backend in ('310p', 'cpro'):
                spec = layer['tensors'][name][backend]
                x = pair[backend] = logical(read_array(dump, spec), name)
                if name == 'effective_length' and (x.dtype != np.int16 or x.item() != 19):
                    raise ValueError(f'{backend}: expected effective_length INT16 [19]')
                if name != 'effective_length' and x.dtype.kind != 'f':
                    raise ValueError(f'{backend}/{name}: expected a floating tensor')
                views = [('tensor', x)] if name in STATIC else [('full64', x), ('valid19', x[:, :19])]
                for scope, value in views:
                    path = root / backend / scope / role / (name + '.bin')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value.tofile(path)
                    record = identity(path, root)
                    if scope != 'valid19' and record['sha256'] != spec['sha256']:
                        raise ValueError(f'{backend}/{name}: exported bytes differ from the source dump')
                    files.append(dict(**record, tensor=name, io=role, backend=backend, scope=scope,
                                      dtype=value.dtype.str, shape=list(value.shape),
                                      source_dump_path=spec['path'], source_sha256=spec['sha256']))
            a, b = pair['cpro'], pair['310p']
            description.append(f"| {name}.bin | {step} | {meaning} | {list(b.shape)} | {b.dtype} / {a.dtype} |")
            rows.extend(_compare_scopes(name, a, b))
            if cpu_golden:
                measured[name] = pair
        extra = {}
        if cpu_golden:
            extra['cpu_reference'] = _write_cpu(root, layer['gdr_attributes'], measured, files)
            description.extend([
                '', 'CPU golden 使用包内 310P 实测的 Q、K、V、g、beta、初态和有效长度。CPU 只对有效行进行 FP32 归一化和递推。',
                'cpu/valid19/output/core.bin 保存有效 19 行读出，返回 dtype 与 query 相同；'
                'cpu/tensor/output/final_state.bin 保存处理完 token 18 的 FP32 状态。',
                'cpu-comparison.csv 比较两端实测输出与这份 CPU golden。CPro 对 CPU 的差异也包含进入算子前的输入差异。',
            ])
        description.extend([
            '', 'comparison.csv 中每行比较 CPro 和 310P 的同名张量。时间张量分别统计 valid19 和 19 个单 token；状态和长度按整个张量比较。',
            'passed 使用原五项浮点指标，整数长度使用精确相等。error_ratio 是按 rtol=0.001、atol=0.00001、'
            'equal_nan=True 计算的超差元素比例，error_ratio_passed 表示该比例 ≤ 0.001。',
            'reference_fp16_subnormals_with_cpro_zero 统计 310P 为 FP16 次正规数、CPro 同位置为零的元素个数。',
            '', '例如，读取 CPro 的有效 core 输出：', '',
            '```python', 'import json, numpy as np',
            'from pathlib import Path', 'root = Path("gdr-l0-valid19")',
            'files = json.loads((root / "manifest.json").read_text())["files"]',
            's = next(f for f in files if f["path"] == "cpro/valid19/output/core.bin")',
            'core = np.fromfile(root / s["path"], dtype=s["dtype"]).reshape(s["shape"])', '```',
        ])
        (root / 'README.md').write_text('\n'.join(description) + '\n', encoding='utf-8')
        _write_csv(root / 'comparison.csv', rows)
        save_json(root / 'manifest.json', dict(
            operator='ChunkGatedDeltaRule', layer=0, physical_rows=64, valid_rows=19, last_valid_index=18,
            chunk_size=64, reference_backend='310p', input_basis=manifest['input_basis'],
            source_dump_manifest_sha256=hashlib.sha256(source).hexdigest(), capture_sha256=manifest['capture_sha256'],
            reference_gdr_attributes=layer['gdr_attributes'], cpro_gdr_graph=graph,
            elementwise_tolerance=dict(**ELEMENTWISE_TOLERANCE, max_error_ratio=ERROR_RATIO_LIMIT), files=files, **extra))
        temp = Path(folder) / 'bundle.tar.gz'
        with tarfile.open(temp, 'w:gz', compresslevel=3) as archive:
            archive.add(root, arcname=BUNDLE)
        share_for_transfer(temp)
        temp.replace(target)
    print(f'{len(files)} bin files · {target.stat().st_size} bytes · {target}')

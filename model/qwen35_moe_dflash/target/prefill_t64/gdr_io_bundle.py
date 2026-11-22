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
from .gdr_stage_data import DESCRIPTIONS, GDR, STATIC, logical

BUNDLE = 'gdr-l0-valid19'


def _comparison(name, scope, actual, reference):
    m = metrics(actual, reference)
    fields = (*THRESHOLDS, 'mean_abs', 'max_abs', *ERROR_RATIO_FIELDS)
    row = dict(tensor=name, scope=scope, cpro_dtype=actual.dtype.name, reference_dtype=reference.dtype.name,
               gate=m['gate'], passed=m['passed'], failed_metrics=','.join(m.get('failed_metrics', [])),
               **{k: m.get(k) for k in fields})
    row.update(cpro_zeros=int(np.count_nonzero(actual == 0)),
               reference_zeros=int(np.count_nonzero(reference == 0)),
               reference_fp16_subnormals_with_cpro_zero=None,
               cpro_nonfinite=m['nonfinite_actual'], reference_nonfinite=m['nonfinite_golden'])
    if reference.dtype == np.float16:
        small = (reference != 0) & (np.abs(reference) < np.finfo(np.float16).tiny)
        row['reference_fp16_subnormals_with_cpro_zero'] = int(np.count_nonzero(small & (actual == 0)))
    return row


def package(dump, output):
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
            if name in STATIC:
                rows.append(_comparison(name, 'tensor', a, b))
            else:
                rows.append(_comparison(name, 'valid19', a[:, :19], b[:, :19]))
                for t in range(19):
                    rows.append(_comparison(name, f'token{t}', a[:, t], b[:, t]))
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
        with (root / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_json(root / 'manifest.json', dict(
            operator='ChunkGatedDeltaRule', layer=0, physical_rows=64, valid_rows=19, last_valid_index=18,
            chunk_size=64, reference_backend='310p', input_basis=manifest['input_basis'],
            source_dump_manifest_sha256=hashlib.sha256(source).hexdigest(), capture_sha256=manifest['capture_sha256'],
            reference_gdr_attributes=layer['gdr_attributes'], cpro_gdr_graph=graph,
            elementwise_tolerance=dict(**ELEMENTWISE_TOLERANCE, max_error_ratio=ERROR_RATIO_LIMIT), files=files))
        temp = Path(folder) / 'bundle.tar.gz'
        with tarfile.open(temp, 'w:gz', compresslevel=3) as archive:
            archive.add(root, arcname=BUNDLE)
        share_for_transfer(temp)
        temp.replace(target)
    print(f'{len(files)} bin files · {target.stat().st_size} bytes · {target}')

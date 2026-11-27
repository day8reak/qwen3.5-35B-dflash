#!/usr/bin/env python3
"""Format a GDN board/matrices.csv as error_ratio.txt and cosine.txt beside it."""
import argparse
import csv
import json
import sys
from pathlib import Path

BACKENDS = ('cpu', '310p', 'cpro')
METRICS = ('error_ratio', 'cosine')


def read_groups(path, metrics=METRICS):
    groups = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = {'stage', 'tensor', 'source_variable', 'scope', 'metric', 'actual/reference', *BACKENDS}
        missing = columns.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        for row in reader:
            key = (row['stage'], row['tensor'], row['scope'])
            description = row['source_variable']
            original, matrices = groups.setdefault(key, (description, {}))
            if original != description:
                raise ValueError(f'{path}:{reader.line_num}: inconsistent description for {key}')
            metric = row['metric']
            if metric in metrics:
                matrix = matrices.setdefault(metric, {})
                actual = row['actual/reference']
                if actual not in BACKENDS or actual in matrix:
                    raise ValueError(f'{path}:{reader.line_num}: unexpected or duplicate row {actual!r} for {key} {metric}')
                matrix[actual] = [float(row[b]) if row[b] else None for b in BACKENDS]
    if not groups:
        raise ValueError(f'{path}: no tensor rows')
    return groups


def a8_cosine(board, stage, tensor, scope):
    reference = board.parent / 'reference'
    manifest = reference / 'manifest.json'
    checkpoint = board / stage / 'stage.json'
    if not (manifest.is_file() and checkpoint.is_file()):
        return {}, '缺少同次运行的参考或板端清单'

    from qwen35_moe_dflash.common.accuracy import floating_metrics
    from qwen35_moe_dflash.target.layer_probe.common import read_array, sha
    from qwen35_moe_dflash.target.prefill_t64.gdn_layer_stages import STAGES, compare_view

    if scope != 'valid19' or tensor not in STAGES[stage][1]:
        raise ValueError(f'{stage}/{tensor}: unexpected INT8 comparison scope or stage')
    saved = json.loads(checkpoint.read_bytes())
    if saved['source']['fixture'] != sha(manifest):
        raise ValueError(f'{checkpoint}: stage and reference belong to different runs')
    records = json.loads(manifest.read_bytes())['arrays']
    values = {}
    missing = []
    for backend in BACKENDS:
        folder, spec = ((board / stage, saved['outputs'][tensor]) if backend == 'cpro'
                        else (reference, records[backend][tensor]))
        if not (folder / spec['path']).is_file():
            missing.append(backend)
            continue
        if spec['dtype'] != 'int8':
            raise ValueError(f"{tensor}/{backend}: expected INT8 codes, got {spec['dtype']}")
        values[backend] = compare_view(tensor, read_array(folder, spec)).astype('float64')
    matrix = {b: [floating_metrics(actual, values[r])['cosine'] if r in values else None
                  for r in BACKENDS] for b, actual in values.items()}
    note = '按保留的 INT8 编码计算'
    if missing:
        note += '，缺少 ' + '、'.join(missing) + ' 的 bin'
    return matrix, note


def render(csv_path, metrics=METRICS):
    groups = read_groups(csv_path, metrics)
    reports = {}
    for metric in metrics:
        blocks = []
        for (stage, tensor, scope), (description, matrices) in groups.items():
            matrix = matrices.get(metric, {})
            note = ''
            if not matrix and metric == 'cosine' and tensor.endswith('_a8'):
                matrix, note = a8_cosine(csv_path.parent, stage, tensor, scope)
            lines = [f'{tensor}  {scope}  {metric}', f'{stage}  |  {description}']
            if note:
                lines.append(note)
            lines.append(r'actual\reference' + ''.join(f'{b:>15}' for b in ('CPU', '310P', 'CPro')))
            for actual in BACKENDS:
                values = matrix.get(actual, (None,) * len(BACKENDS))
                cells = ['n/a' if v is None else format(v, '.9g') for v in values]
                lines.append(f'{actual:<16}' + ''.join(f'{v:>15}' for v in cells))
            blocks.append('\n'.join(lines))
        reports[metric] = '\n\n'.join(blocks) + '\n'
    for metric, report in reports.items():
        dest = csv_path.with_name(metric + '.txt')
        dest.write_text(report, encoding='utf-8')
        print(f'{dest}  ({len(groups)} 个变量/范围)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
    try:
        render(args.csv)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')

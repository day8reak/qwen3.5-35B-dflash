"""Normalize retained HIAI operator CSVs; no SDK, model execution or network.

CSV columns follow the selected run skill's profiling-contract.md. An unknown
vendor schema is retained and rejected, never interpreted by guessed units.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import io
from pathlib import Path
import re

from .artifacts import output_path, save_json
from .hashing import sha256_file


OP_HEADER = ('op name', 'start time', 'total time(us)', 'device type')
MODEL_HEADER = ('item event', 'enter time', 'total time(us)')
EVENTS = ('load begin', 'load end', 'inference begin', 'inference end', 'unload begin', 'unload end')
SCOPE = 'CSV_DURATION_ATTRIBUTION_NOT_CRITICAL_PATH_NPU_ONLY_OR_STEADY_STATE'


def _duration(value):
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError('invalid microsecond duration: ' + repr(value)) from error
    if not number.is_finite() or not 0 <= number <= Decimal('1e15'):
        raise ValueError('duration must be finite, nonnegative and bounded')
    return float(number)


def _rows(path):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 64 << 20:
        raise ValueError('CSV must be a nonempty regular file of at most 64 MiB: ' + str(path))
    raw = path.read_bytes()
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')), strict=True)
    try:
        rows = [tuple(cell.strip() for cell in row) for row in reader if any(cell.strip() for cell in row)]
    except csv.Error as error:
        raise ValueError('malformed profiling CSV: ' + str(path)) from error
    if len(rows) < 2 or rows[0] not in (OP_HEADER, MODEL_HEADER):
        raise ValueError('unsupported or empty profiling CSV: ' + str(path) + '; header=' + repr(rows[:1]))
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError('profiling CSV row width differs from its header: ' + str(path))
    return rows


def aggregate(operators, key):
    grouped = {}
    total = sum(row['total_us'] for row in operators)
    for row in operators:
        item = grouped.setdefault(row[key], {key: row[key], 'count': 0, 'total_us': 0.0})
        item['count'] += 1
        item['total_us'] += row['total_us']
    result = sorted(grouped.values(), key=lambda row: (-row['total_us'], row[key]))
    for row in result:
        row.update(total_ms=row['total_us'] / 1000, mean_ms=row['total_us'] / row['count'] / 1000,
                   operator_sum_percent=100 * row['total_us'] / total if total else None)
    return result


def parse_profile(paths):
    operators, models, sources = [], [], []
    seen = set()
    for value in paths:
        path = Path(value).absolute()
        identity = path.stat()
        inode = (identity.st_dev, identity.st_ino)
        if inode in seen:
            raise ValueError('duplicate profiling file/inode: ' + str(path))
        seen.add(inode)
        before = (identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns)
        digest = sha256_file(path)
        rows = _rows(path)
        source = str(path)
        if rows[0] == OP_HEADER:
            for index, (name, start, duration, device) in enumerate(rows[1:], 2):
                if not name or not start or not device:
                    raise ValueError('blank operator name, start or device: ' + source)
                match = re.fullmatch(r'(.+)\(([^()]*)\)', name, re.DOTALL)
                us = _duration(duration)
                operators.append({'source_csv': source, 'source_row': index, 'name': name,
                    'node_name': match.group(1) if match else name,
                    'operator_type': match.group(2) if match and match.group(2) else 'UNKNOWN',
                    'device_type': device, 'start_time_raw': start, 'total_us': us, 'total_ms': us / 1000})
        else:
            events = {}
            for name, start, duration in rows[1:]:
                if name not in EVENTS or name in events or not start:
                    raise ValueError('unknown/duplicate/blank model event: ' + source)
                events[name] = {'enter_time_raw': start, 'total_us': _duration(duration)}
            if set(events) != set(EVENTS):
                raise ValueError('incomplete model lifecycle CSV: ' + source)
            models.append({'source_csv': source, 'events': events,
                **{phase + '_ms': events[phase + ' end']['total_us'] / 1000 for phase in ('load', 'inference', 'unload')}})
        after = path.stat()
        if before != (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or digest != sha256_file(path):
            raise ValueError('profiling source changed during parsing: ' + source)
        sources.append({'path': source, 'bytes': after.st_size, 'sha256': digest, 'header': list(rows[0])})
    if not operators:
        raise ValueError('no operator CSV rows; successful tool exit alone is not profiling evidence')
    operators.sort(key=lambda row: (-row['total_us'], row['source_csv'], row['source_row']))
    total = sum(row['total_us'] for row in operators)
    for rank, row in enumerate(operators, 1):
        row.update(rank=rank, operator_sum_percent=100 * row['total_us'] / total if total else None)
    return {'schema_version': 1, 'status': 'OPERATOR_PROFILE_CSV_OBSERVED', 'scope': SCOPE,
        'numerical_validation': 'NOT_EVALUATED', 'profiling_activation': 'NOT_ESTABLISHED_BY_CSV_PARSER_ALONE',
        'sources': sources, 'operator_count': len(operators), 'operator_duration_sum_ms': total / 1000,
        'operators': operators, 'by_operator_type': aggregate(operators, 'operator_type'),
        'by_device_type': aggregate(operators, 'device_type'), 'model_profiles': models,
        'model_timing_status': 'PER_CSV_EVENTS_NO_CROSS_MODEL_SUM' if models else 'UNAVAILABLE',
        'timestamp_policy': 'OPAQUE_NO_CLOCK_ORDER_OR_OVERLAP_INFERENCE',
        'sum_scope': 'ALL_RECORDED_ROWS_NO_HIERARCHY_DEDUPLICATION; may include overlapping or nested regions',
        'device_labels': 'LITERAL_VENDOR_LABELS_NO_NUMERIC_ENUM_GUESSES'}


def _write_csv(path, rows, keys):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _cell(value):
    return str(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def write_tables(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / 'operators.csv', report['operators'], ('rank', 'node_name', 'operator_type', 'device_type',
        'total_us', 'total_ms', 'operator_sum_percent', 'source_csv', 'source_row', 'start_time_raw'))
    for key, filename in (('operator_type', 'operator-types.csv'), ('device_type', 'devices.csv')):
        _write_csv(output / filename, report['by_' + key],
            (key, 'count', 'total_us', 'total_ms', 'mean_ms', 'operator_sum_percent'))
    lines = ['# Draft 算子耗时', '',
        '独立 profiling 采样；未验收精度。耗时总和可能包含重叠或嵌套执行，不能当作整次推理时延。', '',
        f"算子记录：{report['operator_count']}；记录耗时总和：{report['operator_duration_sum_ms']:.3f} ms。", '',
        '| 排名 | 算子 | 类型 | 设备 | 耗时 ms | 占算子记录总和 % |',
        '|---:|---|---|---|---:|---:|']
    for row in report['operators'][:30]:
        percent = row['operator_sum_percent']
        lines.append('| ' + ' | '.join(_cell(v) for v in (row['rank'], row['node_name'], row['operator_type'],
            row['device_type'], f"{row['total_ms']:.3f}", f'{percent:.2f}' if percent is not None else '—')) + ' |')
    for key, title in (('operator_type', '算子类型汇总'), ('device_type', '设备标签汇总')):
        lines.extend(['', '## ' + title, '', '| 类别 | 记录数 | 合计 ms |', '|---|---:|---:|'])
        for row in report['by_' + key]:
            lines.append(f"| {_cell(row[key])} | {row['count']} | {row['total_ms']:.3f} |")
    if report['model_profiles']:
        lines.extend(['', '## 厂商模型事件', '', '| 源CSV | 加载 ms | 推理 ms | 卸载 ms |', '|---|---:|---:|---:|'])
        for row in report['model_profiles']:
            lines.append('| ' + ' | '.join([_cell(Path(row['source_csv']).name)] +
                [f"{row[phase + '_ms']:.3f}" for phase in ('load', 'inference', 'unload')]) + ' |')
    if 'capture' in report:
        lines.extend(['', '厂商服务导出的历史记录及模型／PID归属待核对；此表保留全部源CSV，暂不认定每行均属于本次Draft。'])
    lines.extend(['', '完整算子表保存在 operators.csv；类型和设备汇总分别保存在 operator-types.csv、devices.csv。', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(output / 'report.json', report)


def summarize_profile(directory, output):
    root = Path(directory).absolute()
    out = output_path(output)
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('summary output and raw profile directories must be separate')
    paths = sorted(root.rglob('*.csv'))
    if len(paths) > 1024 or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('too many CSV files or symlinks in profile directory')
    report = parse_profile(paths)
    write_tables(report, out)
    return report

"""Read operator and kernel names as reported by msprof for the frozen row-count probe."""
import csv
import json
import re
from collections import Counter


def _key(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


_NAMES = {'opname', 'optype', 'operatorname', 'operatortype', 'kernelname', 'taskname'}
_FIELDS = _NAMES | {'tasktype', 'kerneltype', 'type', 'blockdim', 'mixblockdim',
                    'tilingkey', 'inputshapes', 'inputformats', 'inputdatatypes',
                    'outputshapes', 'deviceid'}
_MATMUL = re.compile(r'matmul|gemm|a8w4|w4a8|qbmm')


def _task_table(path, root):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        names = [field for field in headers if _key(field) in _NAMES]
        fields = [field for field in headers if _key(field) in _FIELDS]
        matches = Counter()
        examples = []
        for row in reader:
            labels = [row.get(field) or '' for field in names]
            if any(_MATMUL.search(_key(label)) for label in labels):
                matches[tuple((row.get(field) or '').strip() for field in fields)] += 1
            elif len(examples) < 8:
                label = next((value for value in labels if value.strip() not in ('', 'N/A')), None)
                if label and label not in examples:
                    examples.append(label)
    table = dict(file=str(path.relative_to(root)), named_matmul_rows=sum(matches.values()),
                 records=[dict(fields=dict(zip(fields, values)), csv_rows=count)
                          for values, count in matches.items()])
    if not matches:
        table.update(columns=headers, other_name_examples=examples)
    return table


def inspect_row_profiles(root):
    root = root.resolve(strict=True)
    identity = None
    cases = []
    for m in (8, 64, 256):
        report = json.loads((root/f'm{m}'/'report.json').read_text())
        if report.get('scope') != 'ORIGINAL_FUSED_MATMUL_REPEATED_T8_ROWS' or report.get('probe_rows') != m:
            raise ValueError(f'm{m} does not contain the expected row-count probe')
        binding = tuple(report[key] for key in ('case', 'capture_sha256', 'quant_replay_sha256', 'cpu_report_sha256'))
        if identity is not None and binding != identity:
            raise ValueError(f'm{m} used a different case, capture, quant replay or CPU reference')
        identity = binding
        profile = root/f'profile-{m}'
        tables = []
        for pattern in ('op_summary*.csv', 'task_time*.csv'):
            for path in sorted(profile.rglob(pattern)):
                if path.is_file():
                    tables.append(_task_table(path, root))
        status = 'NO_EXPORTED_CSV'
        if tables:
            status = 'FOUND_NAMED_ROWS' if any(t['named_matmul_rows'] for t in tables) else 'NO_NAMED_MATMUL'
        row = dict(M=m, accuracy=(report.get('row_comparison') or {}).get('original', 'INCOMPLETE'),
                   profile_status=status, tables=tables)
        if not tables:
            row['raw_profile_dirs'] = [str(p.relative_to(root)) for p in sorted(profile.rglob('PROF_*')) if p.is_dir()]
        cases.append(row)
    complete = all(c['profile_status'] == 'FOUND_NAMED_ROWS' for c in cases)
    return dict(status='INSPECTED' if complete else 'PARTIAL', case=identity[0], cases=cases,
                scope='EXPORTED_CSV_ROWS_BY_PROFILE_DIRECTORY; KERNEL_BINARY_AND_TAIL_HANDLING_UNVERIFIED')

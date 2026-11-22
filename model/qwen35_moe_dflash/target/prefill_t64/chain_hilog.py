"""Read the device log window for a completed, failed decoder probe."""
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile

from ...common.artifacts import output_path
from .chain_board import load


def _window(snapshot, dest, start, end):
    # HiLog records local month/day and time. Device marker mtimes use the same clock.
    # tar/FS mtimes and the text format can differ in fractional precision.
    lo = (start - timedelta(seconds=1)).strftime('%m-%d %H:%M:%S')
    hi = (end + timedelta(seconds=1)).strftime('%m-%d %H:%M:%S')
    stamp = re.compile(r'^(\d\d-\d\d \d\d:\d\d:\d\d)\.\d+\s')
    relevant = re.compile(r'hiai|npu|ai_fmk|ai_infra|aicore|aicpu|hcl|nnrt|davinci|kernel|tensor.probe|gated.?delta|grouped.?matmul|routing', re.I)
    failure = re.compile(r'\s[EF]\s|fail|error|timeout|exception', re.I)
    first, last = [], deque(maxlen=20)
    total = candidates = dated = 0
    first_stamp = last_stamp = None
    with snapshot.open(errors='replace') as src, dest.open('x') as out:
        for line in src:
            m = stamp.match(line)
            if not m:
                continue
            dated += 1
            t = m[1]
            if first_stamp is None:
                first_stamp = t
            last_stamp = t
            if not (lo <= t <= hi if lo <= hi else t >= lo or t <= hi):
                continue
            out.write(line)
            total += 1
            if failure.search(line) and relevant.search(line):
                candidates += 1
                if len(first) < 20:
                    first.append(line.rstrip())
                else:
                    last.append(line.rstrip())
    print(f'Device HiLog records parsed={dated}, in window={total}, runtime error candidates={candidates}')
    if dated:
        print(f'First and last timestamp in snapshot: {first_stamp} to {last_stamp}')
    print(f'All records in this time window (shared across processes): {dest}')
    if candidates > 40:
        print('Showing the first and last 20 candidates in order.')
    for line in [*first, *last]:
        print(line)
    if not total:
        print('No matching device log records for this attempt. Check snapshot coverage and format.')


def diagnose(case_dir, hilog=None):
    import validate_draft_vector as api

    case_dir = output_path(case_dir)
    case = case_dir.name
    plan = load(case_dir/'execution/request.json')
    recovered = load(case_dir/'recovery/result.json')
    failure = recovered.get('execution_diagnostics', {}).get('first_failure', {})
    token = plan['token']
    board = PurePosixPath(plan['board_attempt'])
    if (not re.fullmatch(r'l\d\d_c00_decoder_o0', case) or plan['cases'] != [case]
            or not re.fullmatch('[0-9a-f]{32}', token) or str(board) != api.PREFIX + token
            or recovered.get('board_attempt') != str(board) or failure.get('case') != case
            or not failure.get('exitcode')):
        raise ValueError('Expected the retained request and recovery of one failed decoder case')
    exe, serial, server = api._environment()
    if (serial, server) != (plan['serial'], plan['hdc_server']):
        raise ValueError('Select the HDC board and server recorded in execution/request.json')
    out = Path(tempfile.mkdtemp(dir=case_dir/'recovery', prefix='process-hilog-'))
    hdc = api.Hdc(exe, serial, server, out, 30)
    if hilog is None:
        snapshot = out/'hilog.txt'
        with snapshot.open('xb') as stream:
            subprocess.run([*hdc.prefix, 'shell', 'hilog -x'], stdout=stream, check=True, timeout=30)
    else:
        snapshot = hilog.resolve(strict=True)
    print(f'HiLog snapshot: {snapshot}', flush=True)
    # Read only the original supervisor's markers and the output file extents.
    command = ('cd ' + shlex.quote(str(board)) + ' && ' + api._guard(board, token)
               + f' && test "$(cat started)" = {token} && test "$(cat complete)" = {token}'
               + f' && test "$(cat {case}.exitcode)" = {int(failure["exitcode"])}'
               + f' && stat -c "%y" started && stat -c "%y" {case}.exitcode'
               + f' && for f in outputs/{case}/[0-9]*.bin; do'
               + ' if test -f "$f"; then stat -c "%n %s" "$f" || exit 1; fi; done')
    _, details = hdc.shell(command, 'failed-case-markers')
    lines = details.splitlines()
    start, end = [datetime.fromisoformat(t[:19]) for t in lines[:2]]
    if end < start:
        raise ValueError('The device exit marker predates the start of this request')
    print(f'Device run window: {lines[0]} to {lines[1]}')
    print('Retained numbered output files and byte counts: ' + (', '.join(lines[2:]) or 'none'))
    _window(snapshot, out/'window.log', start, end)

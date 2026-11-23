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


def _snapshot(hdc, out, hilog):
    if hilog is not None:
        return hilog.resolve(strict=True)
    path = out / 'hilog.txt'
    with path.open('xb') as stream:
        try:
            subprocess.run([*hdc.prefix, 'shell', 'hilog -x'], stdout=stream, check=True, timeout=180)
        except subprocess.TimeoutExpired:
            if not path.stat().st_size:
                raise RuntimeError(f'No HiLog bytes received. Supply a snapshot with --hilog. See {out}')
            print('HiLog capture timed out. Checking the time coverage of the partial snapshot.', flush=True)
    return path


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
    snapshot = _snapshot(hdc, out, hilog)
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


def diagnose_v2(replay_dir, hilog=None):
    import validate_draft_vector as api

    replay_dir = output_path(replay_dir)
    case_dir = replay_dir.parent.parent
    plan = load(case_dir / 'execution/request.json')
    report = load(replay_dir / 'result.json')
    token, board = plan['token'], PurePosixPath(plan['board_attempt'])
    work = board / replay_dir.name
    if (not re.fullmatch(r'v2-[0-9a-f]{32}', replay_dir.name) or replay_dir.parent.name != 'recovery'
            or not re.fullmatch('[0-9a-f]{32}', token) or str(board) != api.PREFIX + token
            or report['case'] != case_dir.name or plan['cases'] != [report['case']]
            or report['previous_attempt'] != str(board) or report['device_work'] != str(work)
            or report.get('status') != 'FAILED' or not report.get('tool_exitcode')):
        raise ValueError('Expected the recovery/v2-<id> directory of a completed, failed V2 tool run')
    exe, serial, server = api._environment()
    if (serial, server) != (plan['serial'], plan['hdc_server']):
        raise ValueError('Select the HDC board and server in execution/request.json')
    out = Path(tempfile.mkdtemp(dir=replay_dir, prefix='process-hilog-'))
    hdc = api.Hdc(exe, serial, server, out, 30)
    # Old replays created an empty stderr at launch. A nonempty stderr can
    # have a later mtime, so use the preceding V1 completion as a lower bound.
    command = (api._guard(board, token) + f' && cd {shlex.quote(str(work))}'
               + f' && test "$(cat tool.exitcode)" = {int(report["tool_exitcode"])}'
               + ' && if test -f tool.started; then marker=tool.started;'
               + ' elif test -f tool.stderr && test ! -s tool.stderr; then marker=tool.stderr;'
               + ' else marker=../complete; fi'
               + ' && echo "$marker" && stat -c "%y" "$marker" && stat -c "%y" tool.exitcode'
               + ' && sed -n "s/.*client_tid=\\([0-9][0-9]*\\).*/\\1/p" tool.stdout')
    _, details = hdc.shell(command, 'v2-log-markers')
    marker, start_text, end_text, *tids = details.splitlines()
    start, end = (datetime.fromisoformat(t[:19]) for t in (start_text, end_text))
    if end < start:
        raise ValueError('The V2 exit marker predates the start marker')
    print(f'V2 device window ({marker} to tool.exitcode): {start_text} to {end_text}', flush=True)
    snapshot = _snapshot(hdc, out, hilog)
    print(f'HiLog snapshot: {snapshot}', flush=True)
    window = out / 'window.log'
    _window(snapshot, window, start, end)
    header = re.compile(r'^\S+\s+\S+\s+(\d+)\s+(\d+)\s+[VDIWEF]\s+\S*/model_run_tool[^\s/:]*[/:]')
    with window.open() as stream:
        pids = {m[1] for line in stream if (m := header.match(line)) and m[2] in tids}
    if not pids:
        print(f'No matching model_run_tool thread in this window for client_tid={",".join(tids) or "unknown"}.')
        return
    client_log = out / 'client.log'
    with window.open() as stream, client_log.open('x') as client:
        for line in stream:
            if (m := header.match(line)) and m[1] in pids:
                client.write(line)
    print(f'V2 client_tid={",".join(tids)} matched PID {", ".join(sorted(pids))}. Client records: {client_log}')

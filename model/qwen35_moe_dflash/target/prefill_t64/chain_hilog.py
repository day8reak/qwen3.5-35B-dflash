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


def _window(snapshot, dest, start, end, *, signals=None):
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
            if (signals.search(line) if signals else failure.search(line) and relevant.search(line)):
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
    _client_records(window, out, tids)


def _client_records(window, out, tids):
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


def diagnose_decoder_pair(run, hilog=None, *, names=('L0', 'L3')):
    """Read retained standalone decoder logs and share one device snapshot across their windows."""
    from ...execution.hdc_run import Hdc
    from ...execution.serial_transport import _environment
    from ..layer_probe.common import sha

    env = _environment()
    out = Path(tempfile.mkdtemp(dir=run, prefix='decoder-diagnostics-'))
    print(f'Decoder diagnostics: {out}', flush=True)
    hdc = Hdc(*env, out, 30)
    windows = {}
    complete = True
    for name in names:
        board = run / name / 'board'
        folder = out / name
        folder.mkdir()
        try:
            attempt = load(board / 'attempt.json')
            remote = attempt['remote']
            if (not re.fullmatch(r'/data/local/tmp/qwen35_prefill40_[a-f0-9]{32}', remote)
                    or (attempt['device'], attempt['server']) != env[1:3]):
                raise ValueError('Select the board and server recorded in attempt.json')
            _, identity = hdc.shell(f'cd {remote} && sha256sum run.sh && cat tool.exitcode', name + '-identity')
            digest, code = identity.splitlines()
            if digest.split()[0] != sha(board / 'run.sh') or not code.strip().isdigit():
                raise ValueError('Remote script or completion record differs from this attempt')
            print(f'\n{name} HIAI V2 exit={code.strip()}  {remote}', flush=True)
            tids = []
            for filename in ('tool.stdout', 'tool.stderr'):
                _, body = hdc.shell(f'cd {remote} && cat {filename}', name + '-' + filename)
                (folder / filename).write_text(body + '\n' if body else '')
                if filename == 'tool.stdout':
                    tids = re.findall(r'\bclient_tid=(\d+)', body)
                print(f'{filename}\n{body}', flush=True)
            if int(code) == 0:
                continue
            # Older scripts created stderr at launch. A write can move its mtime later.
            _, bounds = hdc.shell(
                f'cd {remote} && if test -f tool.started; then marker=tool.started; '
                'else marker=tool.stderr; fi && echo "$marker" && wc -c < tool.stderr '
                '&& stat -c "%y" "$marker" tool.exitcode', name + '-times')
            marker, stderr_bytes, start, end = bounds.splitlines()
            if marker == 'tool.stderr' and int(stderr_bytes):
                print(f'{name}: nonempty stderr; its mtime covers only the end of execution.')
            begin, finish = (datetime.fromisoformat(t[:19]) for t in (start, end))
            if begin > finish:
                raise ValueError('Device start marker is later than the exit record')
            print(f'{name} device window ({marker} to tool.exitcode) {start} to {end}', flush=True)
            windows[name] = begin, finish, tids
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            complete = False
            print(f'{name} diagnostics incomplete: {exc}', flush=True)

    if windows:
        snapshot = _snapshot(hdc, out, hilog)
        print(f'Device HiLog: {snapshot}', flush=True)
        signals = re.compile(r'run failed stubName|rtModelExecute failed|hwts exception info|'
                             r'graph executor execute return error|tensor buffer.*not match|FillInputBuffer fail', re.I)
        for name, (begin, finish, tids) in windows.items():
            print(f'\n{name} device-window candidates (shared across processes)', flush=True)
            window = out / name / 'window.log'
            _window(snapshot, window, begin, finish, signals=signals)
            _client_records(window, out / name, tids)
    return complete

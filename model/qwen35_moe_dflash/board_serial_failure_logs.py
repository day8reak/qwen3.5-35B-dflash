"""Collect bounded retained logs from one already completed failed board request.

This is separate from execution/recovery: it never loads models, examines model
payloads, changes locks, launches a process, or attributes a reused PID to a run.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import uuid

from .artifacts import output_path, save_json
from .board_serial_transport import _environment, _regular_guard, _remote
from .hdc_run import Hdc
from .target_compile_failure import _read_small, _safe


_TAIL_BYTES = 64 << 10
_METADATA_BYTES = 8 << 20
_SHA = re.compile(r'[0-9a-f]{64}')
_KIND = 'qwen35-board-serial-failure-logs'


def _object(path):
    raw, identity = _read_small(path, _METADATA_BYTES)
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate diagnostic source metadata key: ' + key)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError('diagnostic source metadata must be an object')
    return value, identity


def _pid(value):
    return type(value) is int and 0 < value <= (1 << 31) - 1


def _source(recovery):
    root, _ = _safe(recovery, directory=True)
    state, checkpoint = _object(root / 'checkpoint.json')
    result, result_id = _object(root / 'result.json')
    keys = ('artifact_kind', 'status', 'stage', 'board_attempt', 'board_lock_token',
        'completion', 'config_sha256', 'serial', 'hdc_server', 'launcher', 'loop',
        'loop_report', 'recovery_source', 'recovery_source_identity', 'recovery_model_execution')
    if any(state.get(key) != result.get(key) for key in keys):
        raise ValueError('completed recovery checkpoint and result identities differ')
    completion = state.get('completion', {})
    token = state.get('board_lock_token')
    if (state.get('artifact_kind') != 'qwen35-board-serial-dflash-execution'
            or state.get('status') != 'FAILED' or state.get('stage') != 'complete'
            or state.get('recovery_model_execution') != 'FORBIDDEN'
            or not isinstance(state.get('recovery_source'), str)
            or not isinstance(state.get('recovery_source_identity'), dict)
            or re.fullmatch(r'[0-9a-f]{32}', str(token)) is None
            or state.get('board_attempt') != '/data/local/tmp/qwen35_board_serial_' + str(token)
            or not isinstance(completion, dict)
            or type(completion.get('remote_exitcode')) is not int
            or not 1 <= completion['remote_exitcode'] <= 255
            or not _pid(completion.get('launcher_process_id'))
            or _SHA.fullmatch(str(state.get('config_sha256'))) is None
            or not isinstance(state.get('launcher'), str)
            or not 0 < len(state['launcher'].encode()) <= 65536):
        raise ValueError('failure logs require a completed failed recovery and its original launch journal')
    loop, loop_id = _object(root / 'results' / 'loop.json')
    if (state.get('loop_report') != str(root / 'results' / 'loop.json')
            or state.get('loop') != loop or loop.get('status') != 'FAILED'
            or loop.get('config_sha256') != state['config_sha256']
            or not _pid(loop.get('process_id'))):
        raise ValueError('retained failure report differs from the completed recovery journal')
    return root, state, (checkpoint, result_id, loop_id)


def _binding_command(state, loop_id):
    board = _remote(state['board_attempt'])
    checks = ['test -d ' + shlex.quote(str(board)), 'test ! -L ' + shlex.quote(str(board)),
        'test -d ' + shlex.quote(str(board / 'results')), 'test ! -L ' + shlex.quote(str(board / 'results'))]
    for name, expected in (('tool.pid', str(state['completion']['launcher_process_id'])),
            ('tool.exitcode', str(state['completion']['remote_exitcode']))):
        path = board / name
        checks += [_regular_guard(path, size=len(expected) + 1),
            'test "$(cat ' + shlex.quote(str(path)) + ')" = ' + shlex.quote(expected)]
    for name, digest in (('config.tsv', state['config_sha256']),
            ('run.sh', hashlib.sha256(state['launcher'].encode()).hexdigest()),
            ('results/loop.json', loop_id['sha256'])):
        path = board / name
        checks += [_regular_guard(path), 'test "$(wc -c < ' + shlex.quote(str(path)) + ')" -le ' + str(_METADATA_BYTES),
            'serial_hash=$(sha256sum ' + shlex.quote(str(path)) + ')',
            'serial_hash=${serial_hash%% *}', 'test "$serial_hash" = ' + shlex.quote(digest)]
    return ' && '.join(checks)


def _tail_command(path, marker):
    quoted = shlex.quote(str(path))
    # The producer exit is encoded together with the tail, so a failing tail
    # cannot be hidden by a successful base64 process at the end of a pipe.
    return ('command -v base64 >/dev/null && command -v tail >/dev/null && '
        + _regular_guard(path) + ' && test -r ' + quoted
        + ' && serial_log_bytes=$(wc -c < ' + quoted + ')'
        + ' && printf "LOG_BYTES=%s\\n" "$serial_log_bytes"'
        + ' && { { tail -c ' + str(_TAIL_BYTES) + ' ' + quoted
        + '; serial_tail_rc=$?; printf "\\n' + marker + '=%s\\n" "$serial_tail_rc"; } | base64; }'
        + ' && ' + _regular_guard(path)
        + ' && test "$(wc -c < ' + quoted + ')" = "$serial_log_bytes"')


def _decode_tail(text, marker):
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError('failure log response exceeds the bounded tail frame')
    first, separator, encoded = text.replace('\r\n', '\n').partition('\n')
    match = re.fullmatch(r'LOG_BYTES=\s*([0-9]{1,18})', first)
    if not separator or match is None:
        raise ValueError('failure log response lacks its exact size/base64 frame')
    try:
        raw = base64.b64decode(''.join(encoded.split()), validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('failure log tail is not valid base64') from error
    suffix = ('\n' + marker + '=0\n').encode()
    if not raw.endswith(suffix):
        raise ValueError('remote tail did not report successful completion')
    raw = raw[:-len(suffix)]
    size = int(match[1])
    if len(raw) != min(size, _TAIL_BYTES):
        raise ValueError('failure log tail extent differs from its bounded source size')
    return raw, size


def _hilog_command(pid, marker):
    if not _pid(pid):
        raise ValueError('HiLog snapshot requires the recorded positive native PID')
    return ('command -v base64 >/dev/null || exit 127; command -v tail >/dev/null || exit 127; '
        + 'serial_hilog=$(hilog -x -P ' + str(pid) + ' 2>&1); serial_hilog_rc=$?; '
        + 'printf "HILOG_EXIT=%s\\n" "$serial_hilog_rc"; '
        + 'serial_log_bytes=$(printf "%s" "$serial_hilog" | wc -c); '
        + 'printf "LOG_BYTES=%s\\n" "$serial_log_bytes"; '
        + '{ printf "%s" "$serial_hilog" | tail -c ' + str(_TAIL_BYTES)
        + '; serial_tail_rc=$?; printf "\\n' + marker + '=%s\\n" "$serial_tail_rc"; } | base64')


def collect_board_serial_failure_logs(recovery_dir, output_dir, *, timeout_seconds=30, hilog=False):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 120:
        raise ValueError('failure-log timeout must be an integer from1 through120 seconds')
    if type(hilog) is not bool:
        raise ValueError('hilog must be an explicit boolean')
    out = output_path(output_dir)
    source = Path(recovery_dir).resolve()
    if out == source or source in out.parents:
        raise ValueError('failure logs require a separate new output directory')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, artifact_kind=_KIND, status='FAILED', output=str(out),
        recovery=str(source), model_execution='NOT_RUN', model_compilation='NOT_RUN',
        remote_mutation='NONE', lock_operations='NONE', process_lookup='NOT_PERFORMED',
        tail_byte_limit=_TAIL_BYTES, logs=[],
        scope='Retained completed-attempt log tails only; no inference, lock changes or SDK root-cause acceptance')
    hdc = None
    try:
        source, state, identities = _source(recovery_dir)
        executable, serial, server = _environment()
        if (serial, server) != (state.get('serial'), state.get('hdc_server')):
            raise ValueError('failure-log collection must use the original recorded device and HDC server')
        report.update(source_bindings=list(identities), board_attempt=state['board_attempt'],
            launcher_process_id=state['completion']['launcher_process_id'],
            native_process_id=state['loop']['process_id'], remote_exitcode=state['completion']['remote_exitcode'],
            config_sha256=state['config_sha256'], loop_sha256=identities[2]['sha256'])
        hdc = Hdc(executable, serial, server, out, timeout_seconds)
        binding = _binding_command(state, identities[2])
        hdc.shell(binding, 'bind-completed-failure-log-attempt', timeout=timeout_seconds)
        for name in ('supervisor.stdout', 'supervisor.stderr'):
            marker = '__QWEN35_TAIL_' + uuid.uuid4().hex + '__'
            remote = PurePosixPath(state['board_attempt']) / name
            _, encoded = hdc.shell(_tail_command(remote, marker), 'collect-failure-log-' + name,
                timeout=timeout_seconds)
            raw, size = _decode_tail(encoded, marker)
            local = out / (name + '.tail')
            local.write_bytes(raw)
            report['logs'].append(dict(name=name, remote=str(remote), path=str(local),
                source_bytes=size, bytes=len(raw), truncated=size > len(raw),
                sha256=hashlib.sha256(raw).hexdigest(), full_log_sha256='NOT_COMPUTED'))
        if hilog:
            marker = '__QWEN35_TAIL_' + uuid.uuid4().hex + '__'
            _, encoded = hdc.shell(_hilog_command(state['loop']['process_id'], marker),
                'collect-failure-hilog-snapshot', timeout=timeout_seconds)
            first, separator, tail = encoded.replace('\r\n', '\n').partition('\n')
            matched = re.fullmatch(r'HILOG_EXIT=([0-9]{1,3})', first)
            if not separator or matched is None or int(matched[1]) > 255:
                raise ValueError('HiLog response lacks its original query exit status')
            raw, size = _decode_tail(tail, marker)
            path = out / 'hilog.pid-snapshot.tail'
            path.write_bytes(raw)
            exitcode = int(matched[1])
            report['hilog'] = dict(status='SNAPSHOT_COLLECTED' if exitcode == 0 else 'QUERY_FAILED',
                native_pid_filter=state['loop']['process_id'], remote_exitcode=exitcode, path=str(path),
                bytes=len(raw), captured_text_bytes=size, truncated=size > len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                scope='Current HiLog ring snapshot filtered by recorded PID; PID reuse/rotation may include other runs or omit this run; not exact-attempt proof',
                text_capture='STDOUT_STDERR_COMBINED_SHELL_COMMAND_SUBSTITUTION_TRAILING_NEWLINES_REMOVED')
            if exitcode:
                raise RuntimeError('HiLog read-only snapshot query exited with status ' + str(exitcode))
        hdc.shell(binding, 'recheck-completed-failure-log-attempt', timeout=timeout_seconds)
        for record in identities:
            _, current = _read_small(record['path'], _METADATA_BYTES)
            if current != record:
                raise ValueError('source recovery changed during failure-log collection')
        report.update(status='BOARD_SERIAL_FAILURE_LOGS_COLLECTED',
            next='Inspect the two retained log tails with the original failure report; no model was restarted')
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',
            next='Retain this diagnostic result and review its exact read-only collection failure')
    finally:
        report['hdc_invocations'] = hdc.count if hdc else 0
        save_json(out / 'result.json', report)
    return report

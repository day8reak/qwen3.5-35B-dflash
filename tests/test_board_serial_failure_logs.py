"""Read-only failure diagnostics never launch HDC or a model in host tests."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from qwen35_moe_dflash import board_serial_failure_logs as logs


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('SN', 'recorded-device')
    monkeypatch.setenv('BIP', '127.0.0.1:8790')
    root = tmp_path / 'recovery'
    loop = dict(status='FAILED', process_id=os.getpid(), config_sha256='b' * 64,
        error='AiModelMngerClient::Load returned1', generated_tokens=[])
    _json(root / 'results/loop.json', loop)
    state = dict(artifact_kind='qwen35-board-serial-dflash-execution', status='FAILED', stage='complete',
        recovery_model_execution='FORBIDDEN', recovery_source=str(tmp_path / 'old-board'),
        recovery_source_identity=dict(sha256='d' * 64), board_lock_token='a' * 32,
        board_attempt='/data/local/tmp/qwen35_board_serial_' + 'a' * 32,
        completion=dict(remote_exitcode=1, launcher_process_id=12345),
        config_sha256='b' * 64, serial='recorded-device', hdc_server='127.0.0.1:8790',
        launcher='#!/system/bin/sh\nold-exact-launch\n', loop=loop,
        loop_report=str(root / 'results/loop.json'),
        lock_state='NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT')
    for name in ('checkpoint.json', 'result.json'): _json(root / name, state)
    class FakeHdc:
        instances = []
        failure = None
        hilog_exit = 0
        data = {'supervisor.stdout': b'hash startup\n' * 10000, 'supervisor.stderr': b'Load returned1\x00\xff\n',
            'hilog': b'10-31 SDK Load: missing sidecar\n'}
        def __init__(self, executable, serial, server, output, timeout):
            self.count = 0; self.calls = []
            self.__class__.instances.append(self)
        def shell(self, command, label, **kwargs):
            self.count += 1; self.calls.append((command, label))
            assert '/proc/' not in command
            assert not any(word in command for word in ('mkdir ', 'rm ', 'chmod ', 'nohup ', '.execution-lock', 'InitWeights'))
            if self.failure == label: raise RuntimeError('deliberate read-only transport failure')
            if label in {'bind-completed-failure-log-attempt', 'recheck-completed-failure-log-attempt'}:
                assert 'config.tsv' in command and 'loop.json' in command and 'tool.pid' in command and 'tool.exitcode' in command
                return 0, ''
            marker = re.search(r'__QWEN35_TAIL_[0-9a-f]{32}__', command)[0]
            name = 'hilog' if label == 'collect-failure-hilog-snapshot' else label.removeprefix('collect-failure-log-')
            raw = self.data[name]
            framed = raw[-logs._TAIL_BYTES:] + ('\n' + marker + '=0\n').encode()
            prefix = ('HILOG_EXIT=' + str(self.hilog_exit) + '\n') if name == 'hilog' else ''
            return 0, prefix + 'LOG_BYTES=' + str(len(raw)) + '\n' + base64.b64encode(framed).decode()
        def send(self, *args, **kwargs): raise AssertionError('read-only collection must never send')
        def receive(self, *args, **kwargs): raise AssertionError('bounded tails do not retrieve whole log files')
    monkeypatch.setattr(logs, 'Hdc', FakeHdc)
    return root, state, FakeHdc


@pytest.mark.parametrize('with_hilog', [False, True])
def test_collects_bounded_tails_preserves_bytes_and_accepts_reused_pid(completed, tmp_path, with_hilog):
    root, state, Hdc = completed
    before = {p: p.read_bytes() for p in root.rglob('*.json')}
    result = logs.collect_board_serial_failure_logs(root, tmp_path / 'diagnostic', hilog=with_hilog)
    assert result['status'] == 'BOARD_SERIAL_FAILURE_LOGS_COLLECTED', result.get('error')
    assert result['native_process_id'] == os.getpid()  # A live reused PID is irrelevant after known completion.
    assert result['model_execution'] == 'NOT_RUN' and result['lock_operations'] == 'NONE'
    assert result['remote_mutation'] == 'NONE' and result['process_lookup'] == 'NOT_PERFORMED'
    assert Hdc.instances[0].count == (5 if with_hilog else 4)
    for row in result['logs']:
        expected = Hdc.data[row['name']][-logs._TAIL_BYTES:]
        assert Path(row['path']).read_bytes() == expected
        assert row['sha256'] == hashlib.sha256(expected).hexdigest()
        assert row['truncated'] == (len(Hdc.data[row['name']]) > logs._TAIL_BYTES)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    if with_hilog:
        assert result['hilog']['remote_exitcode'] == 0
        assert 'not exact-attempt proof' in result['hilog']['scope']
    else:
        assert 'hilog' not in result


@pytest.mark.parametrize('damage', ['not-complete', 'success-exit', 'checkpoint-mismatch', 'loop-mismatch',
    'wrong-attempt', 'wrong-server', 'wrong-device', 'loop-symlink'])
def test_wrong_original_identity_rejected_before_transport(completed, tmp_path, monkeypatch, damage):
    root, state, Hdc = completed
    if damage == 'not-complete': state['stage'] = 'running-detached'
    elif damage == 'success-exit': state['completion']['remote_exitcode'] = 0
    elif damage == 'wrong-attempt': state['board_attempt'] += 'wrong'
    elif damage == 'wrong-server': monkeypatch.setenv('BIP', '127.0.0.2:8790')
    elif damage == 'wrong-device': monkeypatch.setenv('SN', 'other-device')
    for name in ('checkpoint.json', 'result.json'): _json(root / name, state)
    if damage == 'checkpoint-mismatch': _json(root / 'result.json', dict(state, config_sha256='e' * 64))
    elif damage == 'loop-mismatch': _json(root / 'results/loop.json', dict(state['loop'], config_sha256='e' * 64))
    elif damage == 'loop-symlink':
        path = root / 'results/loop.json'; path.rename(root / 'loop-other.json'); path.symlink_to(root / 'loop-other.json')
    result = logs.collect_board_serial_failure_logs(root, tmp_path / 'rejected')
    assert result['status'] == 'FAILED' and not Hdc.instances


@pytest.mark.parametrize('label', ['bind-completed-failure-log-attempt', 'collect-failure-log-supervisor.stderr',
    'recheck-completed-failure-log-attempt'])
def test_collection_failure_never_retries_a_model_or_changes_locks(completed, tmp_path, label):
    root, _, Hdc = completed; Hdc.failure = label
    result = logs.collect_board_serial_failure_logs(root, tmp_path / 'failed')
    assert result['status'] == 'FAILED' and 'read-only transport failure' in result['error']
    assert result['model_execution'] == 'NOT_RUN' and result['lock_operations'] == 'NONE'
    assert sum(item == label for _, item in Hdc.instances[0].calls) == 1


def test_hilog_query_failure_keeps_real_exit_and_error_tail(completed, tmp_path):
    root, _, Hdc = completed; Hdc.hilog_exit = 9; Hdc.data['hilog'] = b'Unsupported hilog option'
    result = logs.collect_board_serial_failure_logs(root, tmp_path / 'hilog-failed', hilog=True)
    assert result['status'] == 'FAILED'
    assert result['hilog']['remote_exitcode'] == 9 and result['hilog']['status'] == 'QUERY_FAILED'
    assert Path(result['hilog']['path']).read_bytes() == b'Unsupported hilog option'
    assert len(result['logs']) == 2


@pytest.mark.parametrize('raw', [b'', b'\x00\xff\n\r\x01', b'A\n' * 100000], ids=['empty', 'binary', 'truncated'])
def test_real_shell_tail_frame_is_bounded_and_binary_exact(tmp_path, raw):
    source = tmp_path / 'log'; source.write_bytes(raw)
    marker = '__QWEN35_TAIL_test__'
    result = subprocess.run(['/bin/sh', '-c', logs._tail_command(source, marker)], capture_output=True, text=True, check=True)
    observed, size = logs._decode_tail(result.stdout.rstrip('\r\n'), marker)
    assert observed == raw[-logs._TAIL_BYTES:] and size == len(raw)


def test_real_shell_failed_tail_is_not_hidden_by_successful_base64(tmp_path):
    source = tmp_path / 'log'; source.write_bytes(b'input')
    program = tmp_path / 'tail'; program.write_text('#!/bin/sh\nprintf partial\nexit 7\n'); program.chmod(0o755)
    marker = '__QWEN35_TAIL_test__'
    result = subprocess.run(['/bin/sh', '-c', logs._tail_command(source, marker)],
        env=dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH']), capture_output=True, text=True, check=True)
    with pytest.raises(ValueError, match='successful completion'):
        logs._decode_tail(result.stdout, marker)


@pytest.mark.parametrize('exitcode', [0, 9])
def test_real_shell_hilog_exit_survives_tail_base64_pipeline(tmp_path, exitcode):
    program = tmp_path / 'hilog'; program.write_text('#!/bin/sh\nprintf "SDK diagnostic\\n\\n"\nexit ' + str(exitcode) + '\n'); program.chmod(0o755)
    marker = '__QWEN35_TAIL_test__'
    result = subprocess.run(['/bin/sh', '-c', logs._hilog_command(63072, marker)],
        env=dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH']), capture_output=True, text=True, check=True)
    first, _, body = result.stdout.partition('\n')
    assert first == 'HILOG_EXIT=' + str(exitcode)
    assert logs._decode_tail(body, marker) == (b'SDK diagnostic', 14)


@pytest.mark.parametrize('damage', ['invalid-base64', 'wrong-size', 'oversized-frame', 'failed-tail'])
def test_malformed_or_unbounded_tail_frames_are_rejected(damage):
    marker = 'marker'
    raw = b'hello\nmarker=0\n'
    text = 'LOG_BYTES=5\n' + base64.b64encode(raw).decode()
    if damage == 'invalid-base64': text += '!'
    elif damage == 'wrong-size': text = text.replace('LOG_BYTES=5', 'LOG_BYTES=6')
    elif damage == 'oversized-frame': text += 'A' * 100001
    else: text = 'LOG_BYTES=5\n' + base64.b64encode(b'hello\nmarker=3\n').decode()
    with pytest.raises(ValueError): logs._decode_tail(text, marker)


def test_real_shell_attempt_binding_rejects_different_loop_and_keeps_files(tmp_path):
    board = tmp_path / 'board'; (board / 'results').mkdir(parents=True)
    contents = {'tool.pid': b'12345\n', 'tool.exitcode': b'1\n', 'run.sh': b'launcher\n',
        'config.tsv': b'config\n', 'results/loop.json': b'{"status":"FAILED"}'}
    for name, raw in contents.items(): (board / name).write_bytes(raw)
    state = dict(board_attempt=str(board), completion=dict(launcher_process_id=12345, remote_exitcode=1),
        config_sha256=hashlib.sha256(contents['config.tsv']).hexdigest(), launcher='launcher\n')
    identity = dict(sha256=hashlib.sha256(contents['results/loop.json']).hexdigest())
    command = logs._binding_command(state, identity)
    assert '/proc/' not in command and subprocess.run(['/bin/sh', '-c', command]).returncode == 0
    (board / 'results/loop.json').write_bytes(b'changed')
    assert subprocess.run(['/bin/sh', '-c', command]).returncode != 0
    assert all((board / name).read_bytes() == raw for name, raw in contents.items() if name != 'results/loop.json')


def test_cli_dispatches_only_read_only_diagnostics(tmp_path, monkeypatch, capsys):
    script = Path(__file__).parents[1] / 'targets/cpro/scripts/collect_board_serial_failure_logs.py'
    spec = importlib.util.spec_from_file_location('failure_log_cli_test', script)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    calls = []
    def collect(recovery, output, **kwargs):
        calls.append((recovery, output, kwargs)); return dict(status='BOARD_SERIAL_FAILURE_LOGS_COLLECTED')
    monkeypatch.setattr(logs, 'collect_board_serial_failure_logs', collect)
    assert cli.main(['--recovery', 'old-recovery', '--output', 'new', '--hilog']) == 0
    assert calls == [('old-recovery', 'new', dict(timeout_seconds=30, hilog=True))]
    with pytest.raises(SystemExit): cli.main(['--recovery', 'old-recovery', '--output', 'new', '--apply'])
    assert len(calls) == 1

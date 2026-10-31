"""Transport fixtures exercise uncertainty and recovery; no real board calls."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

import pytest

from qwen35_moe_dflash import draft_profile_board as board


HASH = 'a' * 64
TOOL = '/data/local/tmp/tools_sysdbg/model_run_tool_internal'


class FakeHdc:
    remote = {}
    calls = []
    fail = None
    parser_help = '--result_path       Profiling & Dump data output directory.'
    busy = False
    foreign = False
    timeout = 60

    def __init__(self, executable, serial, server, output, timeout):
        self.timeout = timeout

    def shell(self, command, label, **kwargs):
        self.calls.append((label, command))
        if self.fail == label:
            raise RuntimeError('fixture HDC connection lost without exit marker')
        if label == 'profile-firmware':
            return 0, 'fixture-firmware'
        if label in ('profile-model-tool-identity', 'profile-parser-tool-identity'):
            return 0, HASH + '  ' + (TOOL if 'model' in label else board.DEFAULT_PARSER)
        if label == 'acquire-global-model-io-lock':
            return 0, 'BUSY' if self.busy else 'ACQUIRED'
        if label == 'profile-parser-help':
            marker = re.search(r'__QWEN35_TOOL_INSPECT_[a-f0-9]+__', command).group()
            return 0, self.parser_help + '\n' + marker + '=1'
        if label == 'profile-completion':
            remote_root = re.search(board.PREFIX + r'[a-f0-9]{32}', command).group()
            return 0, self.remote.get(remote_root + '/complete', b'PENDING').decode()
        if label == 'hash-result':
            path = shlex.split(command)[3].rstrip(';')
            raw = self.remote.get(path)
            return 0, '__MISSING__' if raw is None else hashlib.sha256(raw).hexdigest() + '  ' + path
        if label == 'profile-file-inventory':
            return 0, '  91 ./prof_data/fixture_op.csv'
        if label == 'profile-recovery-lock-owner':
            return 0, 'OTHER' if self.foreign else 'OWNED'
        return 0, ''

    def send(self, local, remote, *args):
        self.remote[str(remote)] = Path(local).read_bytes()

    def receive(self, remote, local, **kwargs):
        Path(local).write_bytes(self.remote[str(remote)])


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('SN', 'fixture-serial')
    monkeypatch.setenv('BIP', 'fixture-server:1234')
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    inputs = []
    for index, size in enumerate((16, 4, 4)):
        path = prepared / f'input_{index}.bin'
        path.write_bytes(bytes(size))
        _, row = board._read_small(path, 1024)
        inputs.append(dict(row, index=index, dtype='fixture-only', shape=[1]))
    plan = {'artifact_kind': board.KIND, 'model': {'path': '/unused/fixture.omc', 'bytes': 32, 'sha256': HASH},
        'tool': {'path': TOOL, 'sha256': HASH}, 'inputs': inputs, 'firmware_id': 'fixture-firmware',
        'input_scope': 'SYNTHETIC_TEST_ONLY', 'timing_scope': 'FAKE_TRANSPORT_NO_DEVICE'}
    monkeypatch.setattr(board, 'bind_profile', lambda *args: dict(plan))
    monkeypatch.setattr(board, 'Hdc', FakeHdc)
    for key, value in dict(remote={}, calls=[], fail=None, busy=False, foreign=False,
            parser_help='--result_path       Profiling & Dump data output directory.').items():
        monkeypatch.setattr(FakeHdc, key, value)
    return {'root': tmp_path, 'prepared': prepared, 'execution': tmp_path / 'execution', 'plan': plan}


def start(case):
    return board.start_profile(case['prepared'], 'fixture-inspection', case['execution'], lock_wait_seconds=0)


def complete(case):
    plan = json.loads((case['execution'] / 'request.json').read_text())
    root = plan['board_attempt']
    FakeHdc.remote[root + '/complete'] = plan['token'].encode()
    for name in ('model.stdout', 'model.stderr', 'parser.stdout', 'parser.stderr', 'supervisor.stdout', 'supervisor.stderr'):
        FakeHdc.remote[root + '/' + name] = b'fixture-log'
    FakeHdc.remote[root + '/model.pid'] = b'12345\n'
    FakeHdc.remote[root + '/model.exitcode'] = b'0\n'
    FakeHdc.remote[root + '/parser.exitcode'] = b'0\n'
    raw = b'op name,start time,total time(us),device type\na(MatMul),opaque,10,NPU\nb(Add),opaque,20,CPU\n'
    # Inventory fixture size is checked against retrieved content, not trusted.
    assert len(raw) == 91
    FakeHdc.remote[root + '/prof_data/fixture_op.csv'] = raw
    return root


def test_one_detached_profile_uses_original_cache_and_explicit_three_input_dtypes(case):
    result = start(case)
    assert result['status'] == 'DRAFT_OPERATOR_PROFILE_STARTED', result
    assert result['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    request = json.loads((case['execution'] / 'request.json').read_text())
    assert '--times=1' in request['model_argv'] and '--enable_item=1' in request['model_argv']
    assert '--modelmanager_itf=V1' in request['model_argv']
    assert '--input_type=FP16,INT32,INT32' in request['model_argv']
    assert not any('direct_from_file' in arg for arg in request['model_argv'])
    assert sum(label == 'profile-launch' for label, _ in FakeHdc.calls) == 1
    assert not any(name.endswith('.omc') for name in FakeHdc.remote)
    assert not any(label == 'release-global-model-io-lock' for label, _ in FakeHdc.calls)
    subprocess.run(['sh', '-n', str(case['execution'] / 'profile.sh')], check=True)


def test_recover_pending_does_not_rerun_or_release(case):
    start(case)
    FakeHdc.calls.clear()
    result = board.recover_profile(case['execution'], case['root'] / 'pending')
    assert result['status'] == 'DRAFT_OPERATOR_PROFILE_PENDING'
    assert [label for label, _ in FakeHdc.calls] == ['profile-completion']


def test_completed_recovery_writes_tables_without_claiming_unbound_rows_are_draft(case):
    start(case)
    complete(case)
    FakeHdc.calls.clear()
    result = board.recover_profile(case['execution'], case['root'] / 'recovery')
    assert result['status'] == 'DRAFT_PROFILE_CAPTURED_REQUIRES_ATTRIBUTION_REVIEW', result
    assert result['operator_count'] == 2
    assert result['lock_state'] == 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
    report = json.loads((case['root'] / 'recovery/summary/report.json').read_text())
    assert 'REQUIRE_REVIEW' in report['capture']['attribution_status']
    assert all(label != 'profile-launch' for label, _ in FakeHdc.calls)


@pytest.mark.parametrize('stage', ['profile-launch', 'acquire-global-model-io-lock'])
def test_uncertain_launch_or_acquisition_never_releases_locks_or_retries(case, monkeypatch, stage):
    monkeypatch.setattr(FakeHdc, 'fail', stage)
    result = start(case)
    assert result['status'] == 'FAILED' and result['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    assert not any(label == 'release-global-model-io-lock' for label, _ in FakeHdc.calls)
    assert sum(label == stage for label, _ in FakeHdc.calls) == 1


def test_parser_help_failure_prevents_model_launch_and_releases_owned_locks(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, 'parser_help', 'Another instance is running or some error occurs')
    result = start(case)
    assert result['status'] == 'FAILED' and result['stage'] == 'parser-help'
    assert not any(label == 'profile-launch' for label, _ in FakeHdc.calls)
    assert sum(label == 'release-global-model-io-lock' for label, _ in FakeHdc.calls) == 2


def test_busy_existing_lock_does_not_get_removed(case, monkeypatch):
    monkeypatch.setattr(FakeHdc, 'busy', True)
    result = start(case)
    assert 'BOARD_MODEL_IO_BUSY' in result['error']
    assert not any(label == 'release-global-model-io-lock' for label, _ in FakeHdc.calls)


def test_recovery_preserves_foreign_new_owner(case, monkeypatch):
    start(case)
    complete(case)
    FakeHdc.calls.clear()
    monkeypatch.setattr(FakeHdc, 'foreign', True)
    result = board.recover_profile(case['execution'], case['root'] / 'recovery')
    assert result['operator_count'] == 2, result
    assert not any(label == 'release-global-model-io-lock' for label, _ in FakeHdc.calls)


def test_completed_model_failure_retains_logs_and_does_not_rerun(case):
    start(case)
    root = complete(case)
    FakeHdc.remote[root + '/model.exitcode'] = b'1\n'
    FakeHdc.calls.clear()
    result = board.recover_profile(case['execution'], case['root'] / 'failed-model')
    assert result['status'] == 'FAILED'
    assert 'converter failed' in result['error']
    assert (case['root'] / 'failed-model/model.stderr').read_bytes() == b'fixture-log'
    assert not any(label == 'profile-launch' for label, _ in FakeHdc.calls)
    assert result['lock_state'] == 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'


def test_retrieval_disconnect_preserves_remote_data_and_allows_separate_recovery(case, monkeypatch):
    start(case)
    complete(case)
    monkeypatch.setattr(FakeHdc, 'fail', 'hash-result')
    first = board.recover_profile(case['execution'], case['root'] / 'interrupted')
    assert first['status'] == 'FAILED' and 'connection lost' in first['error']
    monkeypatch.setattr(FakeHdc, 'fail', None)
    FakeHdc.calls.clear()
    second = board.recover_profile(case['execution'], case['root'] / 'second-recovery')
    assert second['status'] == 'DRAFT_PROFILE_CAPTURED_REQUIRES_ATTRIBUTION_REVIEW', second
    assert not any(label == 'profile-launch' for label, _ in FakeHdc.calls)


def test_inventory_rejects_escape_duplicates_and_large_exports():
    assert board._inventory('  1 ./prof_data/test_op.csv') == [{'path': 'prof_data/test_op.csv', 'bytes': 1}]
    for text in ('1 ./prof_data/../escape.csv', '1 /other.csv', '1 ./prof_data/a\n1 ./prof_data/a',
                 '999999999 ./prof_data/a'):
        with pytest.raises(ValueError):
            board._inventory(text)

"""Retained completed HDC ledger recovery never invokes or uploads a model."""
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shlex
import shutil

import pytest

from qwen35_moe_dflash import target_decode_continuation_recovery as recovery
from test_target_decode_continuation_run import case as run_case, emit, identity
from test_target_state_lineage import save


TOKEN = 'a' * 32


def call(board, label, command, content, serial, server, number):
    marker = '__QWEN35_HDC_' + str(number).zfill(32) + '__'
    payload = '( ' + command + '\n); rc=$?; printf \'\\n' + marker + '=%s\\n\' "$rc"; exit "$rc"'
    path = board / 'calls' / (str(number).zfill(4) + '-' + label + '.json')
    save(path, {'argv': ['fixture-hdc', '-t', serial, '-s', server, 'shell', 'sh -c ' + shlex.quote(payload)],
        'label': label, 'returncode': 0, 'stdout': content + '\n' + marker + '=0\n', 'stderr': ''})
    return path


class RecoveryHdc:
    instances = []
    case = None
    fault = None
    global_owner = TOKEN
    cache_present = True

    def __init__(self, executable, serial, server, output, timeout):
        self.count, self.calls, self.output = 0, [], output
        assert serial == os.environ['SN'] and server == os.environ['BIP']
        self.__class__.instances.append(self)

    def send(self, *args, **kwargs):
        pytest.fail('recovery may never upload inputs, weights, scripts or models')

    def shell(self, command, label, **kwargs):
        self.count += 1; self.calls.append((command, label))
        assert label not in {'execute-original-target', 'acquire-global-model-io-lock', 'acquire-execution-lock'}
        case = self.case
        if self.fault == 'disconnected': raise RuntimeError('missing remote marker: Device not founded or connected')
        if label == 'recovery-device-handshake':
            plan = case['plan']; return 0, plan['tool_binding']['sha256'] + '  ' + plan['tool_binding']['path'] + '\n' + plan['firmware_id'] + ' '
        if label in {'recovery-check-original-attempt', 'recovery-recheck-original-attempt', 'recovery-confirm-locks-absent'}:
            status = 'PROCESS_PRESENT' if self.fault == 'live_pid' else 'PROCESS_ABSENT'
            owner = self.global_owner if self.global_owner else 'ABSENT'
            if self.fault == 'foreign_owner': owner = 'b' * 32
            return 0, 'EXIT=0\nPID=5337\n' + status + '\nGLOBAL=' + owner + '\nCACHE=' + ('PRESENT' if self.cache_present else 'ABSENT')
        if label == 'recovery-output-symlinks': return 0, ''
        if label == 'recovery-output-files': return 0, '\n'.join(sorted(name for name in case['remote'] if '/outputs/' in name))
        if label == 'recovery-output-extents':
            return 0, '\n'.join(str(path.stat().st_size) + ' ' + name for name, path in sorted(case['remote'].items()))
        if label == 'recovery-output-sha256':
            return 0, '\n'.join(identity(path)['sha256'] + '  ' + name for name, path in sorted(case['remote'].items()))
        if label == 'recovery-release-original-locks':
            assert 'rmdir ' in command and 'rm -' not in command and 'kill' not in command
            assert TOKEN in command and case['plan']['compilation_report_sha256'] in command
            if self.fault == 'release_timeout': raise TimeoutError('uncertain lock release transport')
            self.__class__.global_owner, self.__class__.cache_present = None, False
            return 0, ''
        pytest.fail('unexpected recovery command: ' + label)

    def receive(self, remote, local):
        self.count += 1; self.calls.append((str(remote), 'receive'))
        source = self.case['remote'][str(remote)]
        if self.fault == 'missing_receive' and '/outputs/' in str(remote): return
        shutil.copyfile(source, local)
        if self.fault == 'bad_receive' and '/outputs/' in str(remote):
            with local.open('r+b') as stream: stream.write(b'x')


@pytest.fixture
def case(tmp_path, monkeypatch):
    value = run_case.__wrapped__(tmp_path, monkeypatch)
    plan = value['plan']
    remote_root, observed, paths = emit(value)
    cache = PurePosixPath('/data/local/tmp/qwen3_5_moe') / ('codex-target-' + plan['compilation_report_sha256'][:20])
    attempt = cache / ('run-' + 'c' * 32)
    stdout = remote_root / 'tool.stdout'
    stdout.write_text(stdout.read_text().replace(observed['board_attempt'], str(attempt)))
    observed['retrieved']['tool.stdout'] = identity(stdout)
    observed['board_attempt'] = str(attempt)
    remote = {str(attempt / name): remote_root / name for name in observed['retrieved']}
    inputs = [dict(row, host_source=row['source'], source=str(attempt / 'source-inputs' / row['source_filename'])) for row in plan['inputs']]
    failed_dir = tmp_path / 'failed-continuation'
    board = failed_dir / 'board'
    board.mkdir(parents=True)
    save(failed_dir / 'plan.json', plan)
    save(board / 'runtime-continuation-proof.json', plan)
    by_path = {row['path']: row for row in plan['source_inputs']}
    host_inputs = []
    for row in plan['inputs']:
        source = by_path[row['source']]
        host_inputs.append({'path': row['source_filename'], 'bytes': row['bytes'], 'sha256': row['sha256'],
            'stat': [source['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')]})
    board_plan = dict(plan, inputs=inputs, host_input_root=plan['input_dir'], host_inputs=host_inputs)
    save(board / 'plan.json', board_plan); save(board / 'input-evidence.json', {'inputs': inputs})
    primary_relative = Path(plan['primary_model_binding']['path']).relative_to(Path(plan['compilation']) / 'compile-work/generated')
    primary = cache / 'payload' / primary_relative.as_posix()
    argv = [plan['tool_binding']['path'], '--model=' + str(primary),
        '--input=' + ','.join(str(attempt / row['file']) for row in plan['inputs']), '--times=1', '--modelmanager_itf=V2',
        '--weight_dir=' + str(primary.parent), '--enable_item=1', '--direct_from_file=true',
        '--input_type=' + ','.join(row['tool_input_type'] for row in plan['inputs']),
        '--output_type=' + ','.join(row['tool_output_type'] for row in plan['outputs']), '--output_dir=' + str(attempt / 'outputs')]
    save(board / 'command.json', {'argv': argv, 'tool_sha256_observation': 'fixture-original-tool'})
    execute = ('cd ' + shlex.quote(str(attempt)) + ' && { LD_LIBRARY_PATH=' + shlex.quote(str(PurePosixPath(argv[0]).parent))
        + ' ' + shlex.join(argv) + ' > tool.stdout 2> tool.stderr & tool_pid=$!; printf \'%s\\n\' "$tool_pid" > tool.pid; '
        + 'wait "$tool_pid"; rc=$?; printf \'%s\\n\' "$rc" > tool.exitcode; exit "$rc"; }')
    serial, server = os.environ['SN'], os.environ['BIP']
    call(board, 'execute-original-target', execute, '', serial, server, 1)
    inventory = '\n'.join(identity(path)['sha256'] + '  ' + name for name, path in sorted(remote.items()) if '/outputs/' in name)
    call(board, 'output-inventory', 'find ' + str(attempt / 'outputs') + ' -type f -exec sha256sum {} \\;', inventory, serial, server, 2)
    call(board, 'acquire-execution-lock', 'mkdir ' + str(cache / '.execution-lock'), '', serial, server, 3)
    first = next(name for name in observed['retrieved'] if name.startswith('outputs/'))
    failed = dict(observed, status='FAILED', stage='retrieve-outputs', device_execution='FAILED',
        board_model_cache=str(cache), board_global_lock=str(recovery._GLOBAL), remote_execution_lock=str(cache / '.execution-lock'),
        board_lock_token=TOKEN, runtime_continuation_prepared=str(value['prepared']), serial=serial, hdc_server=server,
        inputs=inputs, lock_state='RETAINED_OR_UNCERTAIN', error='HDC success without local output_10',
        retrieval_errors=[{'path': 'tool.exitcode', 'error': 'disconnected'}],
        retrieved={first: observed['retrieved'][first]}, plan_sha256=identity(board / 'plan.json')['sha256'],
        model_payloads=[{'path': str(primary), 'bytes': plan['primary_model_binding']['bytes'], 'sha256': plan['model_sha256']}])
    save(board / 'result.json', failed)
    save(failed_dir / 'result.json', dict(failed, artifact_kind='qwen35-target-decode-continuation-job', mode='APPLY',
        output=str(failed_dir), prepared=str(value['prepared']), plan=str(failed_dir / 'plan.json'), board_result=str(board / 'result.json')))
    monkeypatch.setattr(recovery, 'Hdc', RecoveryHdc)
    RecoveryHdc.instances, RecoveryHdc.fault, RecoveryHdc.global_owner, RecoveryHdc.cache_present = [], None, TOKEN, True
    result = dict(value, failed_dir=failed_dir, failed_board=board, failed=failed, remote=remote,
        recovery_out=tmp_path / 'recovery-new', attempt=attempt, remote_outputs=remote_root, paths=paths)
    RecoveryHdc.case = result
    return result


def recover(case):
    return recovery.recover_decode_continuation(case['failed_dir'], case['recovery_out'])


def test_completed_same_attempt_recovers82_outputs4_logs_and_all80_byte_chains_without_rerun_or_upload(case):
    watched = [path for path in case['failed_dir'].rglob('*') if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_ctime_ns) for path in watched}
    result = recover(case)
    assert result['status'] == 'DECODE_CONTINUATION_OUTPUTS_RECOVERED_UNVALIDATED', result
    assert result['feedback_input_count'] == 80 and result['bound_child_output_count'] == 82
    assert result['state_feedback'] == 'ALL80_PARENT_OUTPUT_BYTES_USED_BY_CHILD_INPUTS'
    assert len(result['retrieved']) == 86 and result['tool_process_id'] == 5337
    assert result['model_inference'] == result['input_upload'] == result['model_upload'] == 'NOT_RUN'
    calls = RecoveryHdc.instances[0].calls
    assert sum(label == 'receive' for _, label in calls) == 86
    assert sum(label == 'recovery-release-original-locks' for _, label in calls) == 1
    assert result['lock_state'] == 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'
    assert before == {path: (path.read_bytes(), path.stat().st_ctime_ns) for path in watched}
    saved = json.loads((case['recovery_out'] / 'board/result.json').read_text())
    assert saved['recovery_only'] == 'NO_MODEL_RERUN_NO_INPUT_OR_WEIGHT_UPLOAD'
    assert saved['original_failed_execution'] == str(case['failed_dir'])


def test_both_original_locks_absent_never_acquires_or_cleans_any_lock(case):
    RecoveryHdc.global_owner, RecoveryHdc.cache_present = None, False
    result = recover(case)
    assert result['status'] == 'DECODE_CONTINUATION_OUTPUTS_RECOVERED_UNVALIDATED', result
    assert not any(label == 'recovery-release-original-locks' for _, label in RecoveryHdc.instances[0].calls)


@pytest.mark.parametrize('fault', ['foreign_owner', 'live_pid', 'disconnected', 'missing_receive', 'bad_receive', 'release_timeout'])
def test_unsafe_or_incomplete_transport_never_promotes_success_or_blindly_cleans_locks(case, fault):
    RecoveryHdc.fault = fault
    result = recover(case)
    assert result['status'] == 'FAILED' and result['error'], result
    assert result['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    assert not (case['recovery_out'] / 'state-chain.json').exists()
    if fault != 'release_timeout':
        assert not any(label == 'recovery-release-original-locks' for instance in RecoveryHdc.instances for _, label in instance.calls)


@pytest.mark.parametrize('fault', ['old_execute_marker', 'old_inventory_marker', 'old_inventory_hash', 'wrong_command', 'input_hash', 'wrong_attempt', 'wrong_server'])
def test_old_ledger_full_inventory_and_preexecution_input_proof_are_required_before_any_device_access(case, fault):
    board = case['failed_board']
    if fault in {'old_execute_marker', 'old_inventory_marker', 'old_inventory_hash'}:
        filename = '0001-execute-original-target.json' if fault == 'old_execute_marker' else '0002-output-inventory.json'
        path = board / 'calls' / filename
        record = json.loads(path.read_text())
        if fault.endswith('marker'): record['stdout'] = re.sub(r'(__QWEN35_HDC_[0-9a-f]+__)=0', r'\1=7', record['stdout'])
        else: record['stdout'] = 'f' * 64 + record['stdout'][64:]
        save(path, record)
    elif fault == 'wrong_command':
        path = board / 'command.json'; value = json.loads(path.read_text()); value['argv'].append('--times=2'); save(path, value)
    elif fault == 'input_hash':
        path = board / 'input-evidence.json'; value = json.loads(path.read_text()); value['inputs'][3]['sha256'] = 'f' * 64; save(path, value)
    else:
        path = board / 'result.json'; value = json.loads(path.read_text())
        value['board_attempt' if fault == 'wrong_attempt' else 'hdc_server'] = '/data/local/tmp/other' if fault == 'wrong_attempt' else 'other-server:8710'
        save(path, value)
    result = recover(case)
    assert result['status'] == 'FAILED' and result['error'], result
    if fault != 'old_inventory_hash': assert not RecoveryHdc.instances
    else: assert not any(label == 'receive' for _, label in RecoveryHdc.instances[0].calls)


def test_byte_chain_is_validated_before_releasing_original_locks(case):
    path = case['remote_outputs'] / 'tool.stdout'
    text = path.read_text(); path.write_text(text + text.splitlines()[0] + '\n')
    result = recover(case)
    assert result['status'] == 'FAILED' and 'slot' in result['error']
    assert not any(label == 'recovery-release-original-locks' for _, label in RecoveryHdc.instances[0].calls)


def test_global_absent_with_anonymous_cache_lock_never_claims_or_deletes_it(case):
    RecoveryHdc.global_owner = None
    result = recover(case)
    assert result['status'] == 'FAILED' and 'lock' in result['error']
    assert not any(label == 'recovery-release-original-locks' for _, label in RecoveryHdc.instances[0].calls)


def test_old_call_cannot_hide_an_additional_failure_marker_before_its_zero_marker(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    board = tmp_path / 'board'; command = 'find /data/local/tmp/attempt/outputs -type f'
    path = call(board, 'output-inventory', command, '', 'device', 'server:8710', 1)
    record = json.loads(path.read_text())
    marker = re.search(r'(__QWEN35_HDC_[0-9a-f]{32}__)=0', record['stdout'])[1]
    record['stdout'] = marker + '=1\n' + record['stdout']; save(path, record)
    with pytest.raises(ValueError, match='one final successful'):
        recovery._framed_call(board, 'output-inventory', command, 'device', 'server:8710', {}, [])


class LocalShell:
    def shell(self, command, label):
        import subprocess
        result = subprocess.run(['sh', '-c', command], capture_output=True, text=True, timeout=5)
        if result.returncode: raise RuntimeError(label + ': nonzero remote marker equivalent')
        return result.returncode, result.stdout.rstrip('\r\n')


def local_locks(tmp_path, monkeypatch):
    lock, cache = tmp_path / 'global-lock', tmp_path / 'cache'
    attempt = cache / 'attempt'; (attempt / 'outputs').mkdir(parents=True)
    lock.mkdir(); (lock / 'owner').write_text(TOKEN + '\n')
    (cache / '.execution-lock').mkdir(); (cache / '.compilation-sha256').write_text('d' * 64 + '\n')
    (attempt / 'tool.exitcode').write_text('0\n'); (attempt / 'tool.pid').write_text('2147483647\n')
    monkeypatch.setattr(recovery, '_GLOBAL', PurePosixPath(str(lock)))
    return lock, PurePosixPath(str(cache)), PurePosixPath(str(attempt))


def test_guarded_real_shell_releases_only_recorded_owner_empty_cache_and_completed_pid(tmp_path, monkeypatch):
    lock, cache, attempt = local_locks(tmp_path, monkeypatch)
    hdc = LocalShell()
    state = recovery._remote_checks(hdc, attempt, cache, 'd' * 64, TOKEN, 'initial')
    assert state == {'pid': 2147483647, 'global': TOKEN, 'cache': 'PRESENT'}
    recovery._release(hdc, attempt, cache, 'd' * 64, TOKEN, state)
    assert not lock.exists() and not Path(cache / '.execution-lock').exists()
    assert recovery._remote_checks(hdc, attempt, cache, 'd' * 64, TOKEN, 'final')['global'] == 'ABSENT'


@pytest.mark.parametrize('fault', ['global_symlink', 'cache_symlink', 'extra_fifo', 'owner_extra_bytes', 'marker_extra_bytes', 'exit_changed', 'pid_changed'])
def test_real_shell_lock_guards_preserve_owner_on_unsafe_paths_or_changed_completion(tmp_path, monkeypatch, fault):
    lock, cache, attempt = local_locks(tmp_path, monkeypatch)
    state = {'pid': 2147483647, 'global': TOKEN, 'cache': 'PRESENT'}
    if fault == 'global_symlink':
        shutil.rmtree(lock); lock.symlink_to(tmp_path / 'missing-lock', target_is_directory=True)
    elif fault == 'cache_symlink':
        Path(cache / '.execution-lock').rmdir(); Path(cache / '.execution-lock').symlink_to(tmp_path / 'missing-cache-lock', target_is_directory=True)
    elif fault == 'extra_fifo': os.mkfifo(lock / 'unexpected')
    elif fault == 'owner_extra_bytes': (lock / 'owner').write_text(TOKEN + '\nextra\n')
    elif fault == 'marker_extra_bytes': Path(cache / '.compilation-sha256').write_text('d' * 64 + '\nextra\n')
    elif fault == 'exit_changed': Path(attempt / 'tool.exitcode').write_text('1\n')
    else: Path(attempt / 'tool.pid').write_text('1\n')
    with pytest.raises((RuntimeError, ValueError)):
        if fault in {'global_symlink', 'cache_symlink'}:
            recovery._remote_checks(LocalShell(), attempt, cache, 'd' * 64, TOKEN, 'unsafe')
        else:
            recovery._release(LocalShell(), attempt, cache, 'd' * 64, TOKEN, state)
    if fault != 'global_symlink': assert (lock / 'owner').exists()


def test_recovery_cli_exposes_only_failed_execution_retrieval_and_new_output(case, capsys):
    module = runpy.run_path(str(Path(__file__).parents[1] / 'targets/cpro/scripts/recover_target_decode_continuation.py'))
    assert module['main'](['--execution', str(case['failed_dir']), '--output', str(case['recovery_out'])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'DECODE_CONTINUATION_OUTPUTS_RECOVERED_UNVALIDATED'
    assert result['model_inference'] == result['input_upload'] == result['model_upload'] == 'NOT_RUN'

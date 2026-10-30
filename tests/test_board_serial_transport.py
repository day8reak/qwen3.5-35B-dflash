"""Request-level transport tests; fake HDC never masquerades as a real NPU."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

import pytest

from qwen35_moe_dflash import board_serial_transport as transport
from qwen35_moe_dflash import board_serial_dflash as preparation


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        'identity': transport._full(path)}


@pytest.fixture
def prepared_request(tmp_path, monkeypatch):
    root = tmp_path / 'prepared'; root.mkdir()
    model = _write(tmp_path / 'target/model.omc', b'target-model')
    row = {'path': 'compile-work/generated/model/model.omc', 'bytes': model['bytes'], 'sha256': model['sha256']}
    weight = {'path': 'compile-work/generated/model/SubGraph_0.weight', 'bytes': 19000000000, 'sha256': '9' * 64}
    draft = _write(tmp_path / 'draft/model.omc', b'draft-model')
    abi = _write(tmp_path / 'draft/draft-abi.json', b'{"full6":true}')
    tables = {name: _write(tmp_path / (name + '.bin'), data) for name, data in
        (('weight', b'original-row'), ('scale', b'scale'))}
    base = {'compilation': str(tmp_path / 'target'), 'compilation_report_sha256': 'a' * 64,
        'runtime_version': 'runtime-v', 'firmware_id': 'firmware-v', 'ddk_id': 'ddk-v',
        'tool_binding': {'path': '/data/local/tmp/tools/tool', 'sha256': 'b' * 64},
        'outputs': [{'bytes': 993280}, {'bytes': 2097152}]}
    manifest = {'target_binding': {'base_plan': base}, 'target_compiled': {'omc': row},
        'target_model': model, 'target_payloads': [
            {'local': model['path'], 'remote': 'model/model.omc', **{k: row[k] for k in ('bytes', 'sha256')}, 'record': row},
            {'local': str(tmp_path / 'NOT-PRESENT-19GB.weight'), 'remote': 'model/SubGraph_0.weight',
             **{k: weight[k] for k in ('bytes', 'sha256')}, 'record': weight}],
        'draft': {'model_source': draft, 'abi': abi}, 'tables': tables,
        'prompt_token_ids': [1] * 8, 'max_new_tokens': 8, 'max_proposals_per_round': 2,
        'minimum_draft_rounds': 2, 'board_model_root': '/data/local/tmp/qwen3_5_moe'}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(preparation, 'validate_prepared', lambda path: deepcopy(manifest))
    def config(value, paths):
        assert paths['target_tool_record'] == {'bytes': 1048576, 'sha256': 'b' * 64}
        assert paths['draft_runner_record']['sha256']
        return b'QWEN35_BOARD_SERIAL\t1\n'
    monkeypatch.setattr(preparation, 'make_config', config)
    monkeypatch.setenv('SN', 'serial0'); monkeypatch.setenv('BIP', '127.0.0.1:8790')
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    def build(out, timeout):
        runner = _write(out / 'native/draft', b'runner')
        supervisor = _write(out / 'native/supervisor', b'supervisor')
        uploads = [{'local': row['path'], 'remote': remote, **{key: row[key] for key in ('sha256', 'bytes')}}
            for row, remote in ((runner, 'draft-runner'), (supervisor, 'board-serial'))]
        return uploads, ['/system/lib64'], {'mode': 'host-test-fixture'}
    monkeypatch.setattr(transport, '_build_payloads', build)
    class FakeHdc:
        instances = []
        disconnect_after_launch = False
        fail_stage = None
        foreign_owner = False
        def __init__(self, executable, serial, server, output, timeout):
            self.count = 0; self.timeout = timeout; self.output = output
            self.calls = []; self.sends = []; self.receives = []
            self.__class__.instances.append(self)
        def shell(self, command, label, **kwargs):
            self.count += 1; self.calls.append((label, command))
            if self.fail_stage == label:
                raise RuntimeError('deliberate HDC failure at ' + label)
            if label == 'whole-loop-device-preflight':
                return 0, 'firmware-v\nTOOL_BYTES=1048576'
            if label == 'acquire-global-model-io-lock':
                return 0, 'ACQUIRED'
            if label == 'whole-loop-board-disk':
                return 0, 'Filesystem 1K-blocks Used Available Use% Mounted\nx 9999999999 0 9999999999 0% /data'
            if label == 'launch-detached-board-serial':
                assert 'nohup sh ' in command and '</dev/null' in command
                assert '&& {' in command and 'LAUNCHED' in command
                if self.disconnect_after_launch:
                    raise RuntimeError('HDC disconnected after remote launch')
                return 0, 'LAUNCHED'
            if label == 'poll-whole-loop-completion':
                return 0, 'EXIT=0\nPID=12345\nPROCESS_ABSENT'
            if label == 'inspect-whole-loop-locks':
                if self.foreign_owner:
                    raise RuntimeError('foreign lock owner rejected by actual guard')
                return 0, 'OWNED'
            if label == 'release-whole-loop-locks':
                return 0, 'RELEASED'
            return 0, ''
        def send(self, local, remote, label='send'):
            self.count += 1; self.sends.append((str(local), str(remote)))
        def receive(self, remote, local, **kwargs):
            self.receives.append((remote, local))
            raise AssertionError('this test does not retrieve payloads')
    monkeypatch.setattr(transport, 'Hdc', FakeHdc)
    observed_inventory = []
    def cached(hdc, rows):
        observed_inventory.append(deepcopy(rows))
        return {row['remote']: 'OK' for row in rows}
    monkeypatch.setattr(transport, '_cache_inventory', cached)
    return root, manifest, FakeHdc, observed_inventory


def test_stage_once_detach_has_no_token_dependent_transport_or_weight_copy(prepared_request, tmp_path):
    root, manifest, FakeHdc, inventory = prepared_request
    result = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert result['status'] == 'BOARD_SERIAL_DFLASH_STARTED', result.get('error')
    assert result['device_execution'] == 'RUNNING_OR_COMPLETION_PENDING'
    assert result['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    hdc = FakeHdc.instances[0]
    assert len(inventory) == 1 and len(inventory[0]) == 5
    assert sum(label == 'launch-detached-board-serial' for label, _ in hdc.calls) == 1
    assert not any('poll' in label or 'release' in label for label, _ in hdc.calls)
    assert not any('model.omc' in local or 'NOT-PRESENT' in local or local.endswith('weight.bin') for local, _ in hdc.sends)
    assert result['intertoken_hdc_calls'] == 0
    assert not list((tmp_path / 'execution').rglob('*.omc'))
    for label, command in hdc.calls:
        if label == 'create-whole-loop-upload-directories':
            assert '.so' not in command and 'config.tsv' not in command
    checkpoint = json.loads((tmp_path / 'execution/checkpoint.json').read_text())
    assert checkpoint['launch_requested'] and checkpoint['board_lock_token']
    assert checkpoint['launcher'].startswith('#!/system/bin/sh\nset -Cu\n')


def test_launch_response_loss_retains_prelaunch_checkpoint_and_recover_never_relaunches(prepared_request, tmp_path, monkeypatch):
    root, manifest, FakeHdc, _ = prepared_request
    FakeHdc.disconnect_after_launch = True
    failed = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert failed['status'] == 'FAILED' and failed['launch_requested']
    assert failed['uncertain'] and failed['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    old = (tmp_path / 'execution/checkpoint.json').read_bytes()
    def retrieve(hdc, out, state, prepared):
        assert state['completion']['launcher_process_id'] == 12345
        state.update(device_execution='EXECUTED_OUTPUTS_RETRIEVED', completed_native_pids=[12346, 12347])
        return {'status': 'test-only-completed'}
    monkeypatch.setattr(transport, '_retrieve', retrieve)
    recovered = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert recovered['status'] == 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED', recovered.get('error')
    assert recovered['lock_state'] == 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'
    assert (tmp_path / 'execution/checkpoint.json').read_bytes() == old
    hdc = FakeHdc.instances[-1]
    assert not hdc.sends and not hdc.receives
    assert not any('launch' in label or 'acquire' in label for label, _ in hdc.calls)
    release = next(command for label, command in hdc.calls if label == 'release-whole-loop-locks')
    assert 'test ! -d /proc/12346' in release and 'test ! -d /proc/12347' in release


def test_recovery_foreign_owner_never_collects_or_releases(prepared_request, tmp_path, monkeypatch):
    root, _, FakeHdc, _ = prepared_request
    transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    FakeHdc.foreign_owner = True
    monkeypatch.setattr(transport, '_retrieve', lambda *args: pytest.fail('foreign locks must stop recovery'))
    result = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert result['status'] == 'FAILED' and result['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    assert not any('release' in label or 'launch' in label for label, _ in FakeHdc.instances[-1].calls)


@pytest.mark.parametrize('label', ['own-target-cache', 'acquire-target-cache-lock', 'verify-checksums', 'set-runtime-permissions'])
def test_prelaunch_failures_never_start_model_and_keep_durable_journal(prepared_request, tmp_path, label):
    root, _, FakeHdc, _ = prepared_request
    FakeHdc.fail_stage = label
    result = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert result['status'] == 'FAILED' and not result['launch_requested']
    assert result['global_held']
    assert not any('launch' in name for name, _ in FakeHdc.instances[-1].calls)
    saved = json.loads((tmp_path / 'execution/checkpoint.json').read_text())
    assert saved['phase'] == label + '-requested' or saved['phase'] == 'upload-runtime-requested'
    assert saved['uncertain']


def test_ctrl_c_during_remote_mutation_preserves_token_and_exact_pending_stage(prepared_request, tmp_path):
    root, _, FakeHdc, _ = prepared_request
    original = FakeHdc.shell
    def interrupted(self, command, label, **kwargs):
        if label == 'acquire-target-cache-lock':
            raise KeyboardInterrupt()
        return original(self, command, label, **kwargs)
    FakeHdc.shell = interrupted
    result = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert result['interruption'].startswith('LOCAL_CONTROLLER_INTERRUPTED')
    assert result['global_held'] and result['board_lock_token'] and result['uncertain']
    assert json.loads((tmp_path / 'execution/checkpoint.json').read_text())['phase'] == 'acquire-target-cache-lock-requested'


@pytest.mark.parametrize('text', ['OK /a\nOK /a', 'OK /a', 'OK /a\nSEND /b\nEXTRA', '[Fail]no device'])
def test_remote_cache_inventory_requires_complete_unique_framed_records(text):
    class Device:
        timeout = 60
        def shell(self, *args, **kwargs):
            return 0, text
    rows = [{'remote': '/a', 'bytes': 1, 'sha256': 'a' * 64}, {'remote': '/b', 'bytes': 2, 'sha256': 'b' * 64}]
    with pytest.raises(ValueError, match='inventory'):
        transport._cache_inventory(Device(), rows)


def test_zero_exit_hdc_connection_error_is_not_remote_completion(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, '[Fail]Device not founded', ''))
    device = transport.Hdc('hdc', 'SN', 'host:8790', tmp_path, 1)
    with pytest.raises(RuntimeError, match='missing unique remote exit marker'):
        device.shell('printf ok', 'completion')


def test_private_library_mode_still_creates_referenced_attempt_libdir(prepared_request, tmp_path):
    root, _, FakeHdc, _ = prepared_request
    result = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert result['status'] == 'BOARD_SERIAL_DFLASH_STARTED', result.get('error')
    command = next(command for label, command in FakeHdc.instances[0].calls
        if label == 'create-whole-loop-upload-directories')
    assert result['board_attempt'] + '/lib' in command
    assert not any('/lib/' in remote for _, remote in FakeHdc.instances[0].sends)


def test_inventory_commands_are_bounded_and_cache_hash_uses_declared_timeout():
    seen = []
    class Device:
        timeout = 7200
        def shell(self, command, label, **kwargs):
            seen.append((command, kwargs))
            paths = []
            for line in command.splitlines():
                # Exact script literal output path (each row is independently hashed).
                paths.append(line.rsplit("printf 'SEND %s\\n' ", 1)[1].split(';', 1)[0])
            return 0, '\n'.join('OK ' + path for path in paths)
    rows = [{'remote': '/data/cache/' + ('x' * 150) + str(index), 'bytes': 19000000000,
        'sha256': 'a' * 64} for index in range(80)]
    assert set(transport._cache_inventory(Device(), rows).values()) == {'OK'}
    assert len(seen) > 1
    assert all(len(command.encode()) <= 12000 and options['timeout'] == 7200 for command, options in seen)


@pytest.mark.parametrize('state', ['PROCESS_PRESENT', 'PROCESS_ABSENT\nEXTRA', 'PROCESS_ABSENT\nPID=8'])
def test_active_or_ambiguous_completion_cannot_release(state):
    class Device:
        timeout = 60
        def shell(self, *args, **kwargs):
            return 0, 'EXIT=0\nPID=100\n' + state
    value = {'board_attempt': '/data/local/tmp/owned'}
    if state == 'PROCESS_PRESENT':
        assert transport._completion(Device(), value) is None
    else:
        with pytest.raises(ValueError, match='uncertain'):
            transport._completion(Device(), value)


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'a//b', 'a/./b'])
def test_retrieval_manifest_rejects_escaping_paths(path):
    raw = ('QWEN35_BOARD_SERIAL_RETRIEVAL\t1\nfile\t' + path + '\t' + 'a' * 64 + '\t3\n').encode()
    with pytest.raises(ValueError):
        transport._inventory(raw)


def test_retrieval_manifest_rejects_duplicate_and_excessive_files():
    row = 'file\tloop.json\t' + 'a' * 64 + '\t3\n'
    with pytest.raises(ValueError, match='duplicate'):
        transport._inventory(('QWEN35_BOARD_SERIAL_RETRIEVAL\t1\n' + row + row).encode())
    with pytest.raises(ValueError, match='budget'):
        transport._inventory(('QWEN35_BOARD_SERIAL_RETRIEVAL\t1\nfile\tloop.json\t' + 'a' * 64 + '\t99999999999\n').encode())


def _failed_native(root, manifest, *, active=False, missing_exit=False):
    base = manifest['target_binding']['base_plan']
    state = {'config_sha256': 'c' * 64}
    loop = {'status': 'FAILED', 'evidence_kind': 'device-hiai', 'config_sha256': state['config_sha256'],
        'target_model_sha256': manifest['target_model']['sha256'],
        'draft_model_sha256': manifest['draft']['model_source']['sha256'],
        'draft_abi_sha256': manifest['draft']['abi']['sha256'], 'runtime_version': base['runtime_version'],
        'ddk_id': base['ddk_id'], 'firmware_id': base['firmware_id'], 'process_id': 100,
        'counts': {'target_native_calls': 1, 'draft_native_calls': 0}}
    progress = {'status': 'FAILED', 'supervisor_pid': 100, 'child_status': 'RUNNING' if active else 'EXITED',
        'last_child_pid': 101, 'last_child_exitcode': None if active else 1}
    records = [_write(root / 'loop.json', json.dumps(loop).encode()),
        _write(root / 'progress.json', json.dumps(progress).encode()), _write(root / 'target/step/tool.pid', b'101\n')]
    if not missing_exit:
        records.append(_write(root / 'target/step/tool.exitcode', b'1\n'))
    (root / 'retrieval-manifest.tsv').write_text('QWEN35_BOARD_SERIAL_RETRIEVAL\t1\n' + ''.join(
        'file\t' + str(Path(row['path']).relative_to(root)) + '\t' + row['sha256'] + '\t' + str(row['bytes']) + '\n'
        for row in records))
    return state


def test_known_completed_native_failure_retains_evidence_without_becoming_success(prepared_request, tmp_path):
    _, manifest, _, _ = prepared_request
    root = tmp_path / 'failed-native'
    state = _failed_native(root, manifest)
    loop, pids = transport._validate_completed_failure(manifest, root, state)
    assert loop['status'] == 'FAILED' and pids == [100, 101]


@pytest.mark.parametrize('options', [{'active': True}, {'missing_exit': True}])
def test_failed_supervisor_unknown_child_exit_cannot_release(prepared_request, tmp_path, options):
    _, manifest, _, _ = prepared_request
    root = tmp_path / 'failed-native'
    state = _failed_native(root, manifest, **options)
    with pytest.raises(ValueError, match='uncertain|no known exit'):
        transport._validate_completed_failure(manifest, root, state)


@pytest.mark.parametrize('fault', [None, 'foreign', 'extra', 'active-child', 'owner-symlink'])
def test_real_shell_release_guards_exact_owner_empty_target_and_all_child_exits(tmp_path, monkeypatch, fault):
    token = 'a' * 32; global_lock = PurePosixPath(str(tmp_path / 'global'))
    target = tmp_path / 'target'; draft = tmp_path / 'draft'; board = tmp_path / 'attempt'
    for path in (Path(global_lock), target / '.execution-lock', draft / '.execution-lock', board):
        path.mkdir(parents=True)
    (Path(global_lock) / 'owner').write_text(('f' * 32 if fault == 'foreign' else token) + '\n')
    (target / '.compilation-sha256').write_text('b' * 64 + '\n')
    (draft / '.model-sha256').write_text('c' * 64 + '\n')
    (draft / '.execution-lock/owner').write_text(token + '\n')
    (board / 'target-lock.owner').write_text(token + '\n')
    (board / 'tool.pid').write_text('2147483647\n'); (board / 'tool.exitcode').write_text('0\n')
    if fault == 'extra':
        (target / '.execution-lock/extra').write_text('foreign')
    if fault == 'owner-symlink':
        owner = draft / '.execution-lock/owner'; owner.unlink(); owner.symlink_to(board / 'target-lock.owner')
    monkeypatch.setattr(transport, '_GLOBAL', global_lock)
    class Shell:
        timeout = 60
        def shell(self, command, label, **kwargs):
            result = subprocess.run(['sh', '-c', command], capture_output=True, text=True, timeout=5)
            if result.returncode:
                raise RuntimeError('guarded release refused')
            return result.returncode, result.stdout.rstrip('\n')
    state = {'board_attempt': str(board), 'target_cache': str(target), 'draft_cache': str(draft),
        'board_lock_token': token, 'compilation_report_sha256': 'b' * 64, 'draft_model_sha256': 'c' * 64,
        'completed_native_pids': [os.getpid()] if fault == 'active-child' else [2147483646]}
    state['completion'] = transport._completion(Shell(), state)
    if fault is None:
        assert transport._lock_checks(Shell(), state, release=True) == 'RELEASED'
        assert not Path(global_lock).exists() and not (target / '.execution-lock').exists()
    else:
        with pytest.raises(RuntimeError, match='refused'):
            transport._lock_checks(Shell(), state, release=True)
        assert Path(global_lock).exists() and (target / '.execution-lock').exists()

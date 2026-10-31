"""Request-level transport tests; fake HDC never masquerades as a real NPU."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
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
        if value.get('execution_mode') == 'resident-v1':
            assert 'target_tool_record' not in paths
            assert 'draft_runner_record' not in paths
        else:
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
                return 0, 'firmware-v' + ('\nTOOL_BYTES=1048576' if 'TOOL_BYTES' in command else '')
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


@pytest.mark.parametrize('fault', [None, 'foreign', 'extra', 'active-child', 'owner-symlink',
    'prefill', 'prefill-foreign', 'prefill-extra'])
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
    if fault and fault.startswith('prefill'):
        prefill = tmp_path / 'prefill'; (prefill / '.execution-lock').mkdir(parents=True)
        (prefill / '.compilation-sha256').write_text('d' * 64 + '\n')
        (board / 'prefill-lock.owner').write_text(('f' * 32 if fault == 'prefill-foreign' else token) + '\n')
        if fault == 'prefill-extra':
            (prefill / '.execution-lock/foreign').write_text('foreign')
        state.update(prefill_cache=str(prefill), prefill_compilation_report_sha256='d' * 64)
    state['completion'] = transport._completion(Shell(), state)
    if fault in {None, 'prefill'}:
        assert transport._lock_checks(Shell(), state, release=True) == 'RELEASED'
        assert not Path(global_lock).exists() and not (target / '.execution-lock').exists()
        if fault == 'prefill':
            assert not (prefill / '.execution-lock').exists()
    else:
        with pytest.raises(RuntimeError, match='refused'):
            transport._lock_checks(Shell(), state, release=True)
        assert Path(global_lock).exists() and (target / '.execution-lock').exists()


def _retained_transport_fixture(FakeHdc, state, monkeypatch, *, fault=None):
    """Fake only transport; semantic native validation has independent C++ tests."""
    loop = {'counts': {'target_native_calls': 16, 'draft_native_calls': 2},
        'generated_tokens': [2] * 8, 'process_id': 12346, 'target_steps': [], 'draft_steps': []}
    result_data = {'loop.json': json.dumps(loop).encode(), 'progress.json': b'{"status":"complete"}',
        'target/final/payload.bin': b'actual-retained-state', 'target/final/empty.stderr': b''}
    inventory = 'QWEN35_BOARD_SERIAL_RETRIEVAL\t1\n' + ''.join('file\t' + name + '\t'
        + hashlib.sha256(raw).hexdigest() + '\t' + str(len(raw)) + '\n' for name, raw in result_data.items())
    attempt = state['board_attempt']
    remote = {attempt + '/' + name: raw for name, raw in {
        'supervisor.stdout': b'supervisor finished', 'supervisor.stderr': b'', 'tool.pid': b'12345\n',
        'tool.exitcode': b'0\n', 'run.sh': state['launcher'].encode()}.items()}
    remote.update({attempt + '/results/' + name: raw for name, raw in result_data.items()})
    remote[attempt + '/results/retrieval-manifest.tsv'] = inventory.encode()
    original_shell = FakeHdc.shell
    def shell(self, command, label, **kwargs):
        response = original_shell(self, command, label, **kwargs)
        if label == 'hash-result':
            path = re.search(r'if test -f ([^;]+); then', command)[1]
            if path not in remote:
                return 0, '__MISSING__'
            return 0, hashlib.sha256(remote[path]).hexdigest() + '  ' + path
        return response
    def receive(self, remote_path, local, **kwargs):
        self.count += 1; self.receives.append((str(remote_path), str(local)))
        raw = remote[str(remote_path)]
        if str(remote_path).endswith('/payload.bin'):
            if fault == 'missing':
                return ''  # A zero client exit is insufficient without an artifact.
            if fault == 'changed':
                raw = b'X' * len(raw)  # Changed after successful batch remote SHA.
            if fault == 'truncated':
                raw = raw[:-1]
        Path(local).parent.mkdir(parents=True, exist_ok=True); Path(local).write_bytes(raw)
        return ''
    monkeypatch.setattr(FakeHdc, 'shell', shell)
    monkeypatch.setattr(FakeHdc, 'receive', receive)
    monkeypatch.setattr(transport, 'validate_retrieved_loop', lambda *args, **kwargs: loop)
    return result_data


def test_completed_inventory_receives_remove_duplicate_per_file_hash_rpcs(prepared_request, tmp_path, monkeypatch):
    root, _, FakeHdc, _ = prepared_request
    launched = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    data = _retained_transport_fixture(FakeHdc, launched, monkeypatch)
    result = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert result['status'] == 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED', result.get('error')
    hdc = FakeHdc.instances[-1]
    # Only five launcher artifacts and the inventory itself use _receive's hash
    # RPC. Every inventory member is hashed remotely in the bounded batch once.
    assert sum(label == 'hash-result' for label, _ in hdc.calls) == len(transport._ATTEMPT_LOGS) + 1
    assert sum(label == 'verify-final-loop-inventory' for label, _ in hdc.calls) == 1
    assert len(hdc.receives) == len(transport._ATTEMPT_LOGS) + 1 + len(data)
    summary = result['retrieval_transport']
    assert summary['duplicate_per_file_remote_hash_queries_avoided'] == len(data)
    assert summary['file_receive_requests'] == len(data)
    assert summary['inventory_payload_bytes'] == sum(map(len, data.values()))
    assert (tmp_path / 'recovery/results/target/final/empty.stderr').read_bytes() == b''
    assert result['lock_release'] == 'RELEASED' and not hdc.sends


@pytest.mark.parametrize('fault,reason', [('changed', 'SHA differs'), ('truncated', 'extent differs'),
    ('missing', 'FileNotFoundError')])
def test_bad_receive_after_batch_hash_keeps_exact_attempt_locks(prepared_request, tmp_path, monkeypatch, fault, reason):
    root, _, FakeHdc, _ = prepared_request
    launched = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    _retained_transport_fixture(FakeHdc, launched, monkeypatch, fault=fault)
    result = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert result['status'] == 'FAILED' and reason in result['error']
    assert result['lock_state'] == 'RETAINED_OR_UNCERTAIN'
    hdc = FakeHdc.instances[-1]
    assert not any('release' in label or 'launch' in label for label, _ in hdc.calls)
    assert not hdc.sends


def test_inventory_bound_receive_rejects_mid_read_identity_change(tmp_path, monkeypatch):
    payload = b'actual state bytes'
    row = {'path': 'nested/state.bin', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    class Device:
        timeout = 60
        def receive(self, remote, local, **kwargs):
            Path(local).write_bytes(payload)
    original = transport._identity; calls = 0
    def changed(info):
        nonlocal calls
        value = original(info); calls += 1
        if calls >= 3:
            value['ctime_ns'] += 1
        return value
    monkeypatch.setattr(transport, '_identity', changed)
    with pytest.raises(ValueError, match='changed'):
        transport._receive_verified_inventory_file(Device(), PurePosixPath('/remote'), tmp_path, row)


def _compact_reports(manifest, *, config_sha='c' * 64, failed=False):
    base = manifest['target_binding']['base_plan']
    counts = {'target_native_calls': 16, 'draft_native_calls': 7, 'considered': 7,
        'accepted': 0, 'corrections': 7, 'bonuses': 0}
    children = [{'pid': 1001 + index, 'exitcode': 0} for index in range(23)]
    if failed:
        counts = {key: 0 for key in counts}; counts['target_native_calls'] = 1
        children = [{'pid': 1001, 'exitcode': 1}]
    lifecycle = {'launched_child_count': len(children), 'completed_child_count': len(children),
        'active_child_count': 0, 'last_child_pid': children[-1]['pid'], 'last_child_exitcode': children[-1]['exitcode'],
        'child_status': 'EXITED', 'children': children}
    generated = [] if failed else [2] * manifest['max_new_tokens']
    loop = {'schema_version': 2, 'artifact_kind': 'qwen35-board-local-serial-dflash',
        'dump': False, 'retrieval_mode': 'FINAL_TOKENS_ONLY', 'evidence_kind': 'host-fake-hiai',
        'model_compilation': 'NOT_RUN',
        'status': 'FAILED' if failed else 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED',
        'config_sha256': config_sha, 'target_model_sha256': manifest['target_model']['sha256'],
        'draft_model_sha256': manifest['draft']['model_source']['sha256'],
        'draft_abi_sha256': manifest['draft']['abi']['sha256'], 'runtime_version': base['runtime_version'],
        'ddk_id': base['ddk_id'], 'firmware_id': base['firmware_id'], 'process_id': 1000,
        'state_policy': 'ADVANCE_ONLY_COMMITTED_TOKENS_NO_SPECULATIVE_STATE',
        'control_semantics': 'EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION',
        'prompt': manifest['prompt_token_ids'], 'generated_tokens': generated,
        'full_prefix': [] if failed else manifest['prompt_token_ids'] + generated,
        'counts': counts, 'completed_rounds': counts['draft_native_calls'], 'child_lifecycle': lifecycle,
        'eos_policy': 'DISABLED_FIXED_BUDGET_SMOKE', 'acceptance_scope': 'CONSIDERED_K_ONLY_NOT_ALL15',
        'numerical_validation': 'NOT_RUN', 'performance': 'NOT_EVALUATED'}
    if failed:
        loop.update(error='target failed', stage='target-child', failed_child_pid=1001, failed_child_exitcode=1)
    prompt_calls = min(len(manifest['prompt_token_ids']), counts['target_native_calls'])
    phase_counts = {'prompt_ingest': prompt_calls, 'decode': counts['target_native_calls'] - prompt_calls,
        'draft': counts['draft_native_calls']}
    loop['phase_latency'] = {'timing_scope': 'CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD', **{
        key: {'calls': calls, 'total_ms': calls * 10, 'mean_ms': 10.0 if calls else None}
        for key, calls in phase_counts.items()}}
    loop['phase_latency']['prompt_ingest']['mechanism'] = 'DECODE1_SERIAL_PROMPT_INGESTION'
    loop['acceptance'] = {'accepted': counts['accepted'], 'considered': counts['considered'],
        'rate': counts['accepted'] / counts['considered'] if counts['considered'] else None}
    loop['supervisor_timing'] = {'scope': 'MONOTONIC_WALL_CLOCK_NOT_NPU_INFERENCE',
        'startup_source_hash_ms': 10, 'total_wall_ms': 1000,
        'target_child_process_wall_ms': counts['target_native_calls'] * 10,
        'draft_child_process_wall_ms': counts['draft_native_calls'] * 10,
        'target_child_processes_timed': counts['target_native_calls'], 'draft_child_processes_timed': counts['draft_native_calls']}
    progress = {'status': loop['status'], 'supervisor_pid': loop['process_id'], 'counts': counts,
        'supervisor_timing': loop['supervisor_timing'],
        'generated_tokens': generated, **{key: value for key, value in lifecycle.items() if key != 'children'}}
    return loop, progress


def _compact_files(root, loop, progress, *, extra=False):
    root.mkdir(parents=True, exist_ok=True)
    data = {'loop.json': json.dumps(loop).encode(), 'progress.json': json.dumps(progress).encode()}
    if extra:
        data['old/output_0.bin'] = b'unexpected-tensor'
    for name, raw in data.items():
        _write(root / name, raw)
    (root / 'retrieval-manifest.tsv').write_text('QWEN35_BOARD_SERIAL_RETRIEVAL\t1\n' + ''.join(
        'file\t' + name + '\t' + hashlib.sha256(raw).hexdigest() + '\t' + str(len(raw)) + '\n'
        for name, raw in data.items()))
    return data


def test_compact_token_validation_is_explicit_and_does_not_claim_full_state_chain(prepared_request, tmp_path):
    _, manifest, _, _ = prepared_request; manifest['dump'] = False
    loop, progress = _compact_reports(manifest); root = tmp_path / 'compact'
    _compact_files(root, loop, progress)
    assert transport.validate_retrieved_loop(manifest, root, _expected_evidence_kind='host-fake-hiai',
        _expected_config_sha256='c' * 64) == loop
    with pytest.raises(ValueError, match='exact config/model/runtime'):
        transport.validate_retrieved_tokens(manifest, root)  # Production rejects fake evidence.
    with pytest.raises(ValueError):
        transport.validate_retrieved_loop({**manifest, 'dump': True}, root, _expected_evidence_kind='host-fake-hiai')
    assert transport._dump_mode({}) is True


@pytest.mark.parametrize('damage', ['extra-file', 'schema', 'model', 'config', 'tokens', 'counts', 'state-claim',
    'integer-dump', 'float-schema', 'bool-prompt',
    'active', 'starting', 'unknown-exit', 'duplicate-pid', 'progress', 'missing-ledger'])
def test_compact_mode_rejects_wrong_scope_identity_and_uncertain_completion(prepared_request, tmp_path, damage):
    _, manifest, _, _ = prepared_request; manifest['dump'] = False
    loop, progress = _compact_reports(manifest)
    if damage == 'schema': loop['schema_version'] = 1
    if damage == 'integer-dump': loop['dump'] = 0
    if damage == 'float-schema': loop['schema_version'] = 2.0
    if damage == 'bool-prompt': loop['prompt'] = [True] * len(loop['prompt'])
    if damage == 'model': loop['target_model_sha256'] = 'd' * 64
    if damage == 'config': loop['config_sha256'] = 'd' * 64
    if damage == 'tokens': loop['full_prefix'][0] = 7
    if damage == 'counts': loop['counts']['considered'] = 6
    if damage == 'state-claim': loop['final_state_files'] = []
    if damage in {'active', 'starting'}:
        loop['child_lifecycle']['child_status'] = progress['child_status'] = 'RUNNING' if damage == 'active' else 'STARTING'
    if damage == 'unknown-exit': loop['child_lifecycle']['children'][-1]['exitcode'] = None
    if damage == 'duplicate-pid': loop['child_lifecycle']['children'][-1]['pid'] = 1001
    if damage == 'progress': progress['supervisor_pid'] = 999
    if damage == 'missing-ledger': del loop['child_lifecycle']['children']
    root = tmp_path / 'compact'; _compact_files(root, loop, progress, extra=damage == 'extra-file')
    with pytest.raises(ValueError):
        transport.validate_retrieved_tokens(manifest, root, _expected_evidence_kind='host-fake-hiai', _expected_config_sha256='c' * 64)


def test_compact_known_failure_retains_small_error_and_requires_all_waited_children(prepared_request, tmp_path):
    _, manifest, _, _ = prepared_request; manifest['dump'] = False
    loop, progress = _compact_reports(manifest, failed=True); root = tmp_path / 'compact'
    _compact_files(root, loop, progress)
    assert transport.validate_retrieved_tokens(manifest, root, _expected_evidence_kind='host-fake-hiai',
        _expected_config_sha256='c' * 64, _remote_exitcode=1)['status'] == 'FAILED'
    with pytest.raises(ValueError):
        transport.validate_retrieved_tokens(manifest, root, _expected_evidence_kind='host-fake-hiai', _remote_exitcode=0)


@pytest.mark.parametrize('failed,fault,resident', [(False, None, False), (True, None, False),
    (True, 'active-child', False), (True, None, True), (True, 'malformed-error', False),
    (True, None, 'v1'), (True, None, 'v1-historical'), (True, None, 'v1-prefill'),
    (True, 'malformed-prefill', 'v1-prefill')])
def test_compact_recovery_receives_only_tokens_progress_and_two_completion_files(prepared_request, tmp_path, monkeypatch, capsys, failed, fault, resident):
    root, manifest, FakeHdc, _ = prepared_request; manifest['dump'] = False
    if resident:
        _enable_prefill_request(root, manifest, tmp_path)
    v1 = resident in {'v1', 'v1-historical', 'v1-prefill'}
    if v1:
        manifest.update(execution_mode='resident-v1', target_lifecycle_policy=preparation.TARGET_LIFECYCLE_POLICY,
            model_load_api=preparation.V1_MODEL_LOAD_API, model_load_apis=dict(preparation.V1_MODEL_LOAD_APIS),
            resident_abi=_write(root / 'resident-abi.tsv', b'QWEN35_RESIDENT_ABI\t1\n'),
            prefill_abi=_write(root / 'prefill-abi.tsv', b'QWEN35_RESIDENT_ABI\t1\n'))
        old_build = transport._build_payloads
        monkeypatch.setattr(transport, '_build_payloads', lambda out, timeout, **kwargs: old_build(out, timeout))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    launched = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    if resident is True:
        # A retained historical V2 journal is constructed as test evidence;
        # new execution of this mode is now forbidden before transport setup.
        manifest.update(execution_mode='resident-v2', target_lifecycle_policy=preparation.V2_TARGET_LIFECYCLE_POLICY,
            resident_abi=_write(root / 'resident-abi.tsv', b'QWEN35_RESIDENT_ABI\t1\n'))
        (root / 'manifest.json').write_text(json.dumps(manifest))
        checkpoint = tmp_path / 'execution/checkpoint.json'
        old = json.loads(checkpoint.read_text())
        old.update(execution_mode='resident-v2', target_lifecycle_policy=preparation.V2_TARGET_LIFECYCLE_POLICY,
            prepared_manifest=transport._json(root / 'manifest.json')[1])
        checkpoint.write_text(json.dumps(old))
    if resident == 'v1-historical':
        # Retained 707d825-style request: no mixed-source fields, original policy.
        manifest['target_lifecycle_policy'] = preparation.HISTORICAL_V1_LIFECYCLE_POLICY
        for key in ('model_load_api', 'model_load_apis'): manifest.pop(key)
        (root / 'manifest.json').write_text(json.dumps(manifest))
        checkpoint = tmp_path / 'execution/checkpoint.json'
        old = json.loads(checkpoint.read_text())
        old.update(target_lifecycle_policy=preparation.HISTORICAL_V1_LIFECYCLE_POLICY,
            prepared_manifest=transport._json(root / 'manifest.json')[1])
        for key in ('model_load_api', 'model_load_apis'): old.pop(key)
        checkpoint.write_text(json.dumps(old))
    loop, progress = _compact_reports(manifest, config_sha=launched['config_sha256'], failed=failed)
    if v1:
        loop, progress = _v1_compact(manifest)
        loop.update(status='FAILED', config_sha256=launched['config_sha256'], stage='resident-close', error='V1 UnLoadModel returned1')
        life = loop['resident_lifecycle']
        life.update(session_id=hashlib.sha256((loop['config_sha256'] + ':1000').encode()).hexdigest(),
            stage='SESSION_CLEANUP', status='FAILED', poisoned=True, cleanup_error=loop['error'])
        life['target_v1_lifecycle'].update(unload_status=1, failure_stage='UNLOAD', cleanup_error=loop['error'])
        progress.update(status='FAILED', resident_call_status='FAILED_CLOSED')
        if resident == 'v1-prefill':
            loop.update(stage='prefill-v1-load', error='HIAI Load failed: 1 [model=qwen35_resident_prefill]',
                generated_tokens=[], full_prefix=[], completed_rounds=0)
            loop['counts'] = {key: 0 for key in loop['counts']}
            for key, value in list(life.items()):
                if type(value) is int: life[key] = 0 if key != 'process_id' else 1000
            life.update(stage='CLOSED', status='CLOSED', poisoned=False, cleanup_error='')
            for kind in ('target', 'draft'):
                life[kind + '_v1_lifecycle'] = _v1_life(kind, 0, attempted=False,
                    model_load_api=manifest['model_load_apis'][kind])
            prefill = _v1_life('prefill', 0, model_load_api='legacy-file')
            prefill.update(loaded=False, load_status=1, failure_stage='LEGACY_LOAD')
            loop['prefill_lifecycle'] = prefill
            loop['acceptance'] = dict(accepted=0, considered=0, rate=None)
            for kind in ('prompt_ingest', 'prefill', 'decode', 'draft'):
                loop['phase_latency'][kind].update(calls=0, total_ms=1 if kind in {'prompt_ingest', 'prefill'} else 0,
                    mean_ms=None)
            timing = loop['supervisor_timing']
            timing.update({key: 0 for key, value in timing.items() if type(value) is int})
            timing.update(prefill_model_load_ms=1, prefill_v1_total_ms=1, total_wall_ms=20)
            progress.update(generated_tokens=[], counts=loop['counts'], supervisor_timing=timing)
            # A previous recovery journal may retain the old misleading summary.
            checkpoint = tmp_path / 'execution/checkpoint.json'; stale = json.loads(checkpoint.read_text())
            stale.update(resident_failure_stage='CLOSED', prefill_failure_stage='STALE_STAGE')
            checkpoint.write_text(json.dumps(stale))
    elif resident:
        loop, progress = _resident_compact(manifest)
        loop.update(status='FAILED', config_sha256=launched['config_sha256'], stage='resident-load',
            error='INIT_WEIGHTS_AFTER_INIT: InitWeights failed: 23', generated_tokens=[], full_prefix=manifest['prompt_token_ids'],
            failed_child_pid=1001, failed_child_exitcode=0, completed_rounds=0)
        loop['counts'].update(target_native_calls=0, draft_native_calls=0, considered=0, accepted=0, corrections=0, bonuses=0)
        life = loop['resident_lifecycle']; life.update(session_id=hashlib.sha256((loop['config_sha256'] + ':1000').encode()).hexdigest(),
            stage='TARGET_LOAD/INIT_WEIGHTS_AFTER_INIT', status='FAILED', poisoned=True,
            target_load_attempts=1, target_loads=0, draft_load_attempts=0, draft_loads=0, target_close_calls=0, draft_close_calls=0,
            target_runs=0, target_run_attempts=0, draft_runs=0, draft_run_attempts=0, target_load_ms=139, draft_load_ms=0,
            target_run_ms=0, draft_run_ms=0, close_ms=0)
        for name in ('decode', 'draft'): loop['phase_latency'][name].update(calls=0, total_ms=0, mean_ms=None)
        loop['acceptance'].update(accepted=0, considered=0, rate=None)
        loop['supervisor_timing'].update(resident_target_load_ms=139, resident_draft_load_ms=0,
            resident_target_run_ms=0, resident_draft_run_ms=0, resident_close_ms=0)
        progress.update(status='FAILED', generated_tokens=[], resident_call_status='FAILED_CLOSED')
    if fault == 'active-child':
        loop['child_lifecycle']['child_status'] = progress['child_status'] = 'RUNNING'
        loop['child_lifecycle']['active_child_count'] = progress['active_child_count'] = 1
        loop['child_lifecycle']['children'][-1]['exitcode'] = None
    if fault == 'malformed-error':
        loop['error'] = {'unverified_error_text': 'must not become a summary error'}
    if fault == 'malformed-prefill':
        loop['prefill_lifecycle']['load_status'] = True
    data = _compact_files(tmp_path / 'native-results', loop, progress)
    remote = {launched['board_attempt'] + '/results/' + name: raw for name, raw in data.items()}
    remote[launched['board_attempt'] + '/results/retrieval-manifest.tsv'] = (tmp_path / 'native-results/retrieval-manifest.tsv').read_bytes()
    remote.update({launched['board_attempt'] + '/tool.pid': b'12345\n',
        launched['board_attempt'] + '/tool.exitcode': b'1\n' if failed else b'0\n'})
    original = FakeHdc.shell
    def shell(self, command, label, **kwargs):
        result = original(self, command, label, **kwargs)
        if label == 'poll-whole-loop-completion' and failed:
            return 0, 'EXIT=1\nPID=12345\nPROCESS_ABSENT'
        if label == 'hash-result':
            path = re.search(r'if test -f ([^;]+); then', command)[1]
            assert path in remote, 'compact retrieval must not request logs or tensors'
            return 0, hashlib.sha256(remote[path]).hexdigest() + '  ' + path
        return result
    def receive(self, source, destination, **kwargs):
        self.count += 1; self.receives.append((str(source), str(destination)))
        assert str(source) in remote
        Path(destination).parent.mkdir(parents=True, exist_ok=True); Path(destination).write_bytes(remote[str(source)])
    monkeypatch.setattr(FakeHdc, 'shell', shell); monkeypatch.setattr(FakeHdc, 'receive', receive)
    actual = transport.validate_retrieved_tokens
    monkeypatch.setattr(transport, 'validate_retrieved_tokens', lambda *args, **kwargs:
        actual(*args, **kwargs, _expected_evidence_kind='host-fake-hiai'))
    result = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert result['status'] == ('FAILED' if failed else 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED'), result.get('error')
    if not failed:
        assert result['generated_token_ids'] == loop['generated_tokens']
    assert result['evidence_scope'] == 'TOKENS_AND_SUPERVISOR_COMPLETION_ONLY_NO_HOST_STATE_CHAIN_REPLAY'
    assert result['state_continuity'] == 'NOT_REPLAYED_COMPACT_RESULT' and result['dump'] is False
    if not fault:
        assert result['acceptance']['rate'] == (None if failed and (not v1 or resident == 'v1-prefill') else 0)
        assert result['phase_latency']['prompt_ingest']['calls'] == (0 if resident == 'v1-prefill' else 1 if failed else 8)
        assert 'Recover this exact' not in result['next']
        assert result['counts'] == loop['counts'] and result['generated_token_ids'] == loop['generated_tokens']
        if failed:
            expected_stage = 'prefill-v1-load' if resident == 'v1-prefill' else 'resident-close' if v1 else 'resident-load' if resident else 'target-child'
            assert result['stage'] == 'complete' and result['execution_failure_stage'] == expected_stage
            assert result['error'] == result['execution_error'] == loop['error']
            if resident == 'v1-prefill':
                assert result['prefill_lifecycle'] == loop['prefill_lifecycle']
                assert result['prefill_failure_stage'] == 'LEGACY_LOAD'
                assert result['prefill_lifecycle']['load_status'] == 1
                assert 'resident_failure_stage' not in result
                assert result['resident_lifecycle']['stage'] == 'CLOSED'
                assert result['counts']['prefill_native_calls'] == result['counts']['target_native_calls'] == result['counts']['draft_native_calls'] == 0
                assert result['generated_token_ids'] == [] and result['completed_native_pids'] == [1000]
            elif v1:
                assert result['resident_failure_stage'] == 'SESSION_CLEANUP'
                assert result['counts']['target_native_calls'] == 8 and result['counts']['draft_native_calls'] == 7
                assert result['generated_token_ids'] == loop['generated_tokens']
                assert result['completed_native_pids'] == [1000]
                expected_api = 'legacy-buffer' if resident == 'v1-historical' else 'legacy-v1-mixed'
                assert result['model_load_api'] == result['resident_lifecycle']['model_load_api'] == expected_api
                assert not any('launch' in label or 'acquire' in label for label, _ in FakeHdc.instances[-1].calls)
            elif resident:
                assert result['resident_failure_stage'] == 'TARGET_LOAD/INIT_WEIGHTS_AFTER_INIT'
                assert result['counts'] == dict(target_native_calls=0, draft_native_calls=0, prefill_native_calls=1,
                    considered=0, accepted=0, corrections=0, bonuses=0)
            import importlib.util
            script = Path(__file__).parents[1] / 'targets/cpro/scripts/run_board_serial_dflash.py'
            spec = importlib.util.spec_from_file_location('failed_recovery_cli', script)
            cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
            monkeypatch.setattr(transport, 'recover_board_serial', lambda *a, **k: result)
            assert cli.main(['--recover', str(tmp_path / 'execution'), '--output', str(tmp_path / 'summary')]) == 1
            summary = json.loads(capsys.readouterr().out.splitlines()[-1])
            assert summary['error'] == loop['error'] and summary['execution_failure_stage'] == expected_stage
            assert summary['stage'] == 'complete' and summary['counts'] == loop['counts']
            if resident == 'v1-prefill':
                assert summary['prefill_lifecycle'] == result['prefill_lifecycle']
                assert summary['prefill_failure_stage'] == 'LEGACY_LOAD'
                assert 'resident_failure_stage' not in summary
            elif resident:
                assert summary['resident_failure_stage'] == result['resident_failure_stage']
    assert len(FakeHdc.instances[-1].receives) == 5
    assert not (tmp_path / 'recovery/supervisor.stdout').exists()
    if fault:
        assert result['lock_state'] == 'RETAINED_OR_UNCERTAIN' and 'lock_release' not in result
        assert result['error'] != loop['error'] and 'execution_error' not in result and 'execution_failure_stage' not in result
        if fault == 'malformed-prefill':
            assert 'prefill_lifecycle' not in result and 'prefill_failure_stage' not in result
    else:
        assert result['lock_release'] == 'RELEASED'


@pytest.mark.parametrize('damage', ['rate', 'nan', 'mean', 'phase-count', 'prefill-label', 'npu-label', 'phase-sum'])
def test_summary_rate_and_phase_latency_cannot_misstate_counts_or_timing_scope(prepared_request, tmp_path, damage):
    _, manifest, _, _ = prepared_request; manifest['dump'] = False
    loop, progress = _compact_reports(manifest)
    if damage == 'rate': loop['acceptance']['rate'] = .5
    if damage == 'nan': loop['acceptance']['rate'] = float('nan')
    if damage == 'mean': loop['phase_latency']['decode']['mean_ms'] = 9
    if damage == 'phase-count': loop['phase_latency']['prompt_ingest']['calls'] = 1
    if damage == 'prefill-label': loop['phase_latency']['prompt_ingest']['mechanism'] = 'PREFILL_MODEL'
    if damage == 'npu-label': loop['phase_latency']['timing_scope'] = 'NPU_INFERENCE'
    if damage == 'phase-sum': loop['supervisor_timing']['target_child_process_wall_ms'] += 1
    root = tmp_path / 'compact'; _compact_files(root, loop, progress)
    with pytest.raises(ValueError):
        transport.validate_retrieved_tokens(manifest, root, _expected_evidence_kind='host-fake-hiai')


def test_recovery_cannot_override_mode_pinned_by_original_preparation(prepared_request, tmp_path):
    root, manifest, FakeHdc, _ = prepared_request; manifest['dump'] = False
    (root / 'manifest.json').write_text(json.dumps(manifest))
    transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    path = tmp_path / 'execution/checkpoint.json'; checkpoint = json.loads(path.read_text())
    checkpoint['dump'] = True; path.write_text(json.dumps(checkpoint))
    result = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert result['status'] == 'FAILED' and 'cannot change the pinned dump/config mode' in result['error']
    assert len(FakeHdc.instances) == 1 and result['lock_state'] == 'RETAINED_OR_UNCERTAIN'


def _enable_prefill_request(root, manifest, tmp_path):
    model = _write(tmp_path / 'prefill/model.omc', b'original-prefill')
    row = {'path': 'compile-work/generated/model/model.omc', 'bytes': model['bytes'], 'sha256': model['sha256']}
    base = deepcopy(manifest['target_binding']['base_plan'])
    base.update(compilation=str(tmp_path / 'prefill'), compilation_report_sha256='d' * 64, inputs=[])
    manifest['prefill'] = {'model': model, 'base_plan': base, 'compiled': {'omc': row},
        'control_recipe': transport._PREFILL_RECIPE, 'state_mapping': [],
        'payloads': [{'local': model['path'], 'remote': 'model/model.omc', 'bytes': model['bytes'],
            'sha256': model['sha256'], 'record': row}]}
    (root / 'manifest.json').write_text(json.dumps(manifest))


def test_prefill_cache_is_independent_owned_locked_and_recovered_without_relaunch(prepared_request, tmp_path, monkeypatch):
    root, manifest, FakeHdc, inventories = prepared_request
    _enable_prefill_request(root, manifest, tmp_path)
    original = preparation.make_config
    def config(value, paths):
        assert paths['prefill_model'].startswith('/data/local/tmp/qwen3_5_moe/codex-target-' + 'd' * 20)
        assert paths['prefill_model'] != paths['target_model']
        return original(value, paths)
    monkeypatch.setattr(preparation, 'make_config', config)
    launched = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert launched['status'] == 'BOARD_SERIAL_DFLASH_STARTED', launched.get('error')
    assert launched['prefill_lock_held'] and launched['prefill_cache'] != launched['target_cache']
    assert [row['kind'] for row in inventories[0]].count('prefill') == 1
    assert any(label == 'acquire-prefill-cache-lock' and 'prefill-lock.owner' in command
        for label, command in FakeHdc.instances[0].calls)
    monkeypatch.setattr(transport, '_retrieve', lambda hdc, out, state, value: state.update(completed_native_pids=[12346]))
    recovered = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert recovered['status'] == 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED', recovered.get('error')
    assert not recovered['prefill_lock_held']
    assert recovered['prompt_ingestion'] == transport._PREFILL_RECIPE
    release = next(command for label, command in FakeHdc.instances[-1].calls if label == 'release-whole-loop-locks')
    assert 'prefill-lock.owner' in release and launched['prefill_cache'] + '/.execution-lock' in release
    assert not any('launch' in label or 'acquire' in label for label, _ in FakeHdc.instances[-1].calls)


@pytest.mark.parametrize('label', ['own-prefill-cache', 'acquire-prefill-cache-lock'])
def test_prefill_cache_mutation_failure_retains_exact_lock_journal(prepared_request, tmp_path, label):
    root, manifest, FakeHdc, _ = prepared_request
    _enable_prefill_request(root, manifest, tmp_path); FakeHdc.fail_stage = label
    result = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert result['status'] == 'FAILED' and result['phase'] == label + '-requested'
    assert result['lock_state'] == 'RETAINED_OR_UNCERTAIN' and not result['launch_requested']
    assert result['target_lock_held'] and result['global_held']
    assert not any('launch' in name or 'release' in name for name, _ in FakeHdc.instances[0].calls)


def test_prefill_cache_miss_uses_strict_prefill_compiler_record(prepared_request, tmp_path, monkeypatch):
    root, manifest, _, _ = prepared_request; _enable_prefill_request(root, manifest, tmp_path)
    from qwen35_moe_dflash import original_target_board
    captured = []
    monkeypatch.setattr(original_target_board, '_recorded_board_payload', lambda directory, row, compiled, **kwargs:
        captured.append((directory, row, compiled, kwargs)) or Path('strictly-validated'))
    rows = transport._static_rows(manifest, PurePosixPath('/target'), PurePosixPath('/draft'), PurePosixPath('/tables'), PurePosixPath('/prefill'))
    row = next(row for row in rows if row['kind'] == 'prefill')
    assert transport._local_upload(row, manifest) == Path('strictly-validated')
    assert captured == [(Path(manifest['prefill']['base_plan']['compilation']), row['record'],
        manifest['prefill']['compiled'], {'smoke_proof': manifest['prefill']['base_plan']})]


def test_resident_stages_pinned_descriptor_abi_selects_new_binary_and_never_relaunches(prepared_request, tmp_path, monkeypatch):
    root, manifest, FakeHdc, _ = prepared_request
    _enable_prefill_request(root, manifest, tmp_path)
    abi = _write(root / 'resident-abi.tsv', b'QWEN35_RESIDENT_ABI\t1\n')
    prefill_abi = _write(root / 'prefill-abi.tsv', b'QWEN35_RESIDENT_ABI\t1\nprefill_input\t0\targ1_1\tINT8\t1\t0\t1,64,2048\t131072\n')
    manifest.update(execution_mode='resident-v1', target_lifecycle_policy=preparation.TARGET_LIFECYCLE_POLICY,
        model_load_api=preparation.V1_MODEL_LOAD_API, model_load_apis=dict(preparation.V1_MODEL_LOAD_APIS),
        resident_abi=abi, prefill_abi=prefill_abi)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    old = transport._build_payloads; selected = []
    def build(out, timeout, *, resident=False):
        selected.append(resident)
        return old(out, timeout)
    monkeypatch.setattr(transport, '_build_payloads', build)
    old_config = preparation.make_config
    def config(value, paths):
        assert paths['resident_abi'].endswith('/resident-abi.tsv')
        assert paths['prefill_abi'].endswith('/prefill-abi.tsv')
        return old_config(value, paths)
    monkeypatch.setattr(preparation, 'make_config', config)
    started = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert started['status'] == 'BOARD_SERIAL_DFLASH_STARTED', started.get('error')
    assert selected == ['resident-v1'] and started['execution_mode'] == 'resident-v1'
    assert any(local == abi['path'] and remote.endswith('/resident-abi.tsv') for local, remote in FakeHdc.instances[0].sends)
    assert any(local == prefill_abi['path'] and remote.endswith('/prefill-abi.tsv') for local, remote in FakeHdc.instances[0].sends)
    assert not any('TOOL_BYTES' in command or 'sha256sum /data/local/tmp/tools/tool' in command
        for _, command in FakeHdc.instances[0].calls)
    monkeypatch.setattr(transport, '_retrieve', lambda hdc, out, state, value: state.update(completed_native_pids=[12346]))
    recovered = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovered')
    assert recovered['status'] == 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED', recovered.get('error')
    assert not any('launch' in label or 'acquire' in label for label, _ in FakeHdc.instances[-1].calls)


@pytest.mark.parametrize('damage', ['missing-map', 'wrong-map', 'wrong-session-api'])
def test_recovery_refuses_changed_new_v1_source_selection_before_transport(prepared_request, tmp_path, monkeypatch, damage):
    root, manifest, FakeHdc, _ = prepared_request
    _enable_prefill_request(root, manifest, tmp_path)
    manifest.update(execution_mode='resident-v1', target_lifecycle_policy=preparation.TARGET_LIFECYCLE_POLICY,
        model_load_api=preparation.V1_MODEL_LOAD_API, model_load_apis=dict(preparation.V1_MODEL_LOAD_APIS),
        resident_abi=_write(root / 'resident-abi.tsv', b'QWEN35_RESIDENT_ABI\t1\n'),
        prefill_abi=_write(root / 'prefill-abi.tsv', b'QWEN35_RESIDENT_ABI\t1\n'))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    build = transport._build_payloads
    monkeypatch.setattr(transport, '_build_payloads', lambda out, timeout, **kwargs: build(out, timeout))
    started = transport.execute_board_serial(root, tmp_path / 'execution', detach=True)
    assert started['status'] == 'BOARD_SERIAL_DFLASH_STARTED'
    checkpoint = tmp_path / 'execution/checkpoint.json'
    state = json.loads(checkpoint.read_text())
    if damage == 'missing-map': state.pop('model_load_apis')
    elif damage == 'wrong-map': state['model_load_apis']['target'] = 'legacy-buffer'
    else: state['model_load_api'] = 'legacy-buffer'
    checkpoint.write_text(json.dumps(state))
    recovered = transport.recover_board_serial(tmp_path / 'execution', tmp_path / 'recovery')
    assert recovered['status'] == 'FAILED' and 'source representations' in recovered['error']
    assert len(FakeHdc.instances) == 1


def _resident_compact(manifest):
    loop, progress = _compact_reports(manifest)
    loop.update(schema_version=3, execution_mode='resident-v2', resident_abi_sha256=manifest['resident_abi']['sha256'])
    counts = loop['counts']; counts['target_native_calls'] = 8; counts['prefill_native_calls'] = 1
    loop.update(prefill_model_sha256=manifest['prefill']['model']['sha256'], prompt_ingestion=transport._PREFILL_RECIPE,
        control_semantics='EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED')
    life = dict(session_id=hashlib.sha256((loop['config_sha256'] + ':1000').encode()).hexdigest(), process_id=1000, model_load_api='file-v2', stage='CLOSED', status='CLOSED',
        target_load_attempts=1, draft_load_attempts=1, target_loads=1, draft_loads=1, target_runs=8, target_run_attempts=8, draft_runs=7, draft_run_attempts=7,
        target_close_calls=1, draft_close_calls=1, target_load_ms=30, draft_load_ms=40,
        target_run_ms=80, draft_run_ms=70, close_ms=20, closed=True, poisoned=False)
    life.update(target_model_name='qwen35_resident_target', draft_model_name='qwen35_resident_draft',
        target_model_name_scope='SDK_SETNAME_GETNAME_VALIDATED_BEFORE_INIT', draft_model_name_scope='SDK_SETNAME_GETNAME_VALIDATED_BEFORE_INIT',
        target_lifecycle=dict(weight_init_policy='EXPERIMENTAL_INIT_WEIGHTS_BEFORE_INIT_NO_FALLBACK', ordering_validation='NOT_VALIDATED_ON_DEVICE',
            **{key: True for key in ('weights_init_attempted', 'weights_initialized', 'manager_init_attempted', 'manager_initialized',
                'manager_deinit_called', 'weights_deinit_called', 'closed')}, weights_deinit_status=0, run_attempts=8, successful_runs=8))
    if manifest.get('target_lifecycle_policy'):
        loop['target_lifecycle_policy'] = manifest['target_lifecycle_policy']
        life['target_lifecycle'].update(weight_init_policy='INIT_BEFORE_WEIGHTS_NO_FALLBACK',
            weight_cleanup_policy='DEINIT_WEIGHTS_BEFORE_DEINIT')
    loop['resident_lifecycle'] = life
    progress.update(resident_lifecycle=life, resident_call_status='RETURNED')
    child = dict(launched_child_count=1, completed_child_count=1, active_child_count=0,
        last_child_pid=1001, last_child_exitcode=0, child_status='EXITED', children=[dict(pid=1001, exitcode=0)])
    loop['child_lifecycle'] = child
    phases = loop['phase_latency']; phases['timing_scope'] = 'RESIDENT_RUN_WALL_EXCLUDES_MODEL_LOAD_AND_CLOSE'
    phases.update(prefill=dict(calls=1, total_ms=10, mean_ms=10),
        prompt_ingest=dict(calls=1, total_ms=10, mean_ms=10, latency_alias='prefill', mechanism=transport._PREFILL_RECIPE),
        decode=dict(calls=8, total_ms=80, mean_ms=10))
    for key in ('prefill', 'prompt_ingest'): phases[key]['timing_scope'] = 'CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD'
    for key in ('decode', 'draft'): phases[key]['timing_scope'] = 'RESIDENT_RUN_INCLUDES_TENSOR_COPIES_READBACK_HASH_AND_BUFFER_CHECKS_NOT_NPU_ONLY'
    timing = loop['supervisor_timing']; timing.update(target_child_process_wall_ms=0, draft_child_process_wall_ms=0,
        target_child_processes_timed=0, draft_child_processes_timed=0, prefill_child_processes_timed=1,
        prefill_child_process_wall_ms=10, resident_target_load_ms=30, resident_draft_load_ms=40,
        resident_target_run_ms=80, resident_draft_run_ms=70, resident_close_ms=20)
    progress.update({key: value for key, value in child.items() if key != 'children'})
    return loop, progress


@pytest.mark.parametrize('damage', [None, 'open', 'two-loads', 'poisoned', 'run-count', 'fake-child', 'wrong-abi', 'load-in-run',
    'missing-prefill-wall', 'memory-type'])
def test_resident_compact_distinguishes_calls_loads_and_real_prefill_child(prepared_request, tmp_path, damage):
    root, manifest, _, _ = prepared_request; _enable_prefill_request(root, manifest, tmp_path)
    manifest.update(dump=False, execution_mode='resident-v2', resident_abi=dict(sha256='e' * 64))
    loop, progress = _resident_compact(manifest)
    if damage == 'open': loop['resident_lifecycle']['closed'] = False
    elif damage == 'two-loads': loop['resident_lifecycle']['target_loads'] = 2
    elif damage == 'poisoned': loop['resident_lifecycle']['poisoned'] = True
    elif damage == 'run-count': loop['resident_lifecycle']['target_runs'] = 7
    elif damage == 'fake-child': loop['child_lifecycle']['launched_child_count'] = progress['launched_child_count'] = 16
    elif damage == 'wrong-abi': loop['resident_abi_sha256'] = 'f' * 64
    elif damage == 'load-in-run': loop['phase_latency']['timing_scope'] = 'CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD'
    elif damage == 'missing-prefill-wall': loop['supervisor_timing']['total_wall_ms'] = 240
    elif damage == 'memory-type': loop['resident_memory'] = dict(scope='PROCESS_RSS_NOT_NPU_ALLOCATED_MEMORY',
        snapshots=[dict(phase='after-close', status='OBSERVED', vm_rss_kib=True)])
    out = tmp_path / 'retained'; _compact_files(out, loop, progress)
    if damage:
        with pytest.raises(ValueError):
            transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai')
    else:
        checked = transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai')
        assert checked['resident_lifecycle']['closed'] and checked['child_lifecycle']['children'] == [dict(pid=1001, exitcode=0)]


@pytest.mark.parametrize('prepared_new,observed_new', [(False, False), (False, True), (True, False), (True, True)])
def test_resident_recovery_policies_cannot_be_silently_substituted(prepared_request, tmp_path, prepared_new, observed_new):
    root, manifest, _, _ = prepared_request; _enable_prefill_request(root, manifest, tmp_path)
    manifest.update(dump=False, execution_mode='resident-v2', resident_abi=dict(sha256='e' * 64))
    if prepared_new: manifest['target_lifecycle_policy'] = preparation.V2_TARGET_LIFECYCLE_POLICY
    observed = deepcopy(manifest)
    if observed_new: observed['target_lifecycle_policy'] = preparation.V2_TARGET_LIFECYCLE_POLICY
    else: observed.pop('target_lifecycle_policy', None)
    loop, progress = _resident_compact(observed); out = tmp_path / 'policy-evidence'; _compact_files(out, loop, progress)
    if prepared_new == observed_new:
        assert transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai') == loop
    else:
        with pytest.raises(ValueError, match='policy|lifecycle'):
            transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai')


def _v1_life(name, processes, *, attempted=True, model_load_api=None):
    return dict(model_load_api=model_load_api or preparation.V1_MODEL_LOAD_APIS[name], lifecycle_policy=preparation.HISTORICAL_V1_LIFECYCLE_POLICY,
        model_name='qwen35_resident_' + name,
        runtime_tensor_names='NOT_QUERIED_V1_SAVED_NAMED_ABI_ONLY',
        runtime_dtypes='DECLARED_BOUND_ABI_NOT_QUERIED',
        external_weight_resolution='MODEL_PATH_PARENT_SIDECARS_UNVALIDATED_ON_DEVICE',
        client_init_attempted=attempted, client_initialized=attempted, load_attempted=attempted,
        loaded=attempted, unload_called=attempted, closed=attempted,
        client_init_status=0 if attempted else None, load_status=0 if attempted else None,
        unload_status=0 if attempted else None, process_attempts=processes,
        successful_processes=processes, failure_stage='', cleanup_error='')


def _v1_compact(manifest):
    loop, progress = _resident_compact(manifest)
    loop.update(execution_mode='resident-v1', target_lifecycle_policy=manifest['target_lifecycle_policy'],
        prefill_abi_sha256=manifest['prefill_abi']['sha256'], prefill_lifecycle=_v1_life('prefill', 1))
    life = loop['resident_lifecycle']; life.pop('target_lifecycle')
    session_api, model_apis = preparation.v1_load_contract(manifest)
    life.update(model_load_api=session_api, target_v1_lifecycle=_v1_life('target', 8),
        draft_v1_lifecycle=_v1_life('draft', 7))
    if manifest['target_lifecycle_policy'] == preparation.TARGET_LIFECYCLE_POLICY:
        loop['model_load_apis'] = dict(model_apis)
    loop['prefill_lifecycle']['model_load_api'] = model_apis['prefill']
    for kind in ('target', 'draft'):
        life[kind + '_v1_lifecycle']['model_load_api'] = model_apis[kind]
    for kind in ('target', 'draft'):
        life[kind + '_model_name_scope'] = 'V1_MODEL_DESCRIPTION_LABEL_NOT_RUNTIME_TENSOR_NAME'
    child = dict(launched_child_count=0, completed_child_count=0, active_child_count=0,
        last_child_pid=None, last_child_exitcode=None, child_status='NONE', children=[])
    loop['child_lifecycle'] = child
    progress.update({key: value for key, value in child.items() if key != 'children'})
    phases = loop['phase_latency']
    for kind in ('prefill', 'prompt_ingest'):
        phases[kind]['timing_scope'] = 'PREFILL_V1_LOAD_PROCESS_UNLOAD_WALL'
    for kind in ('decode', 'draft'):
        phases[kind]['native_api'] = 'Process'
    loop['supervisor_timing'].update(prefill_child_processes_timed=0, prefill_child_process_wall_ms=0,
        prefill_model_load_ms=1, prefill_process_ms=8, prefill_model_unload_ms=1, prefill_v1_total_ms=10)
    return loop, progress


@pytest.mark.parametrize('damage', [None, 'v2-api', 'v2-policy', 'fake-child', 'wrong-prefill-abi',
    'missing-prefill', 'bool-status', 'process-count', 'unclosed-prefill', 'legacy-session-api',
    'target-api', 'prefill-api', 'draft-api', 'missing-api-map', 'mixed-api-map'])
def test_all_v1_compact_is_explicit_without_v2_weight_or_child_evidence(prepared_request, tmp_path, damage):
    root, manifest, _, _ = prepared_request; _enable_prefill_request(root, manifest, tmp_path)
    manifest.update(dump=False, execution_mode='resident-v1', target_lifecycle_policy=preparation.TARGET_LIFECYCLE_POLICY,
        model_load_api=preparation.V1_MODEL_LOAD_API, model_load_apis=dict(preparation.V1_MODEL_LOAD_APIS),
        resident_abi=dict(sha256='e' * 64), prefill_abi=dict(sha256='f' * 64))
    loop, progress = _v1_compact(manifest)
    life = loop['resident_lifecycle']
    if damage == 'v2-api': life['model_load_api'] = 'file-v2'
    elif damage == 'v2-policy': loop['target_lifecycle_policy'] = preparation.V2_TARGET_LIFECYCLE_POLICY
    elif damage == 'fake-child': loop['child_lifecycle']['children'] = [dict(pid=1001, exitcode=0)]
    elif damage == 'wrong-prefill-abi': loop['prefill_abi_sha256'] = '0' * 64
    elif damage == 'missing-prefill': loop.pop('prefill_lifecycle')
    elif damage == 'bool-status': life['target_v1_lifecycle']['load_status'] = False
    elif damage == 'process-count': life['target_v1_lifecycle']['successful_processes'] = 7
    elif damage == 'unclosed-prefill': loop['prefill_lifecycle']['closed'] = False
    elif damage == 'legacy-session-api': life['model_load_api'] = 'legacy-buffer'
    elif damage == 'target-api': life['target_v1_lifecycle']['model_load_api'] = 'legacy-buffer'
    elif damage == 'prefill-api': loop['prefill_lifecycle']['model_load_api'] = 'legacy-buffer'
    elif damage == 'draft-api': life['draft_v1_lifecycle']['model_load_api'] = 'legacy-file'
    elif damage == 'missing-api-map': loop.pop('model_load_apis')
    elif damage == 'mixed-api-map': loop['model_load_apis']['draft'] = 'legacy-file'
    out = tmp_path / 'v1-evidence'; _compact_files(out, loop, progress)
    if damage:
        with pytest.raises(ValueError):
            transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai')
    else:
        assert transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai') == loop
        assert not loop['child_lifecycle']['children']
        assert 'weights_init_attempted' not in life['target_v1_lifecycle']


@pytest.mark.parametrize('mode', ['resident-v2', 'resident-v1'])
def test_old_resident_preparation_cannot_start_new_execution(prepared_request, tmp_path, mode):
    root, manifest, FakeHdc, _ = prepared_request
    manifest.update(execution_mode=mode, target_lifecycle_policy=(preparation.V2_TARGET_LIFECYCLE_POLICY
        if mode == 'resident-v2' else preparation.HISTORICAL_V1_LIFECYCLE_POLICY))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    result = transport.execute_board_serial(root, tmp_path / 'must-not-launch', detach=True)
    assert result['status'] == 'FAILED' and 'recover-only' in result['error']
    assert not FakeHdc.instances


@pytest.mark.parametrize('prepared_new,observed_new', [(False, False), (True, True), (False, True), (True, False)])
def test_v1_recovery_requires_the_original_descriptor_policy(prepared_request, tmp_path, prepared_new, observed_new):
    root, manifest, _, _ = prepared_request; _enable_prefill_request(root, manifest, tmp_path)
    manifest.update(dump=False, execution_mode='resident-v1',
        target_lifecycle_policy=preparation.HISTORICAL_V1_LIFECYCLE_POLICY,
        resident_abi=dict(sha256='e' * 64), prefill_abi=dict(sha256='f' * 64))
    def choose(value, newer):
        if newer:
            value.update(target_lifecycle_policy=preparation.TARGET_LIFECYCLE_POLICY,
                model_load_api=preparation.V1_MODEL_LOAD_API, model_load_apis=dict(preparation.V1_MODEL_LOAD_APIS))
        return value
    observed = choose(deepcopy(manifest), observed_new)
    choose(manifest, prepared_new)
    loop, progress = _v1_compact(observed)
    out = tmp_path / 'original-v1-policy'; _compact_files(out, loop, progress)
    if prepared_new == observed_new:
        assert transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai') == loop
    else:
        with pytest.raises(ValueError, match='policy|source|lifecycle'):
            transport.validate_retrieved_loop(manifest, out, _expected_evidence_kind='host-fake-hiai')


@pytest.mark.parametrize('failure', ['init', 'descriptor', 'path-cleared'])
@pytest.mark.parametrize('historical', [False, True])
def test_v1_prefill_init_or_descriptor_failure_before_resident_load_is_recoverable(prepared_request, tmp_path, failure, historical):
    root, manifest, _, _ = prepared_request; _enable_prefill_request(root, manifest, tmp_path)
    manifest.update(dump=False, execution_mode='resident-v1', target_lifecycle_policy=preparation.TARGET_LIFECYCLE_POLICY,
        model_load_api=preparation.V1_MODEL_LOAD_API, model_load_apis=dict(preparation.V1_MODEL_LOAD_APIS),
        resident_abi=dict(sha256='e' * 64), prefill_abi=dict(sha256='f' * 64))
    if historical:
        manifest['target_lifecycle_policy'] = preparation.HISTORICAL_V1_LIFECYCLE_POLICY
        for key in ('model_load_api', 'model_load_apis'): manifest.pop(key)
    _, model_apis = preparation.v1_load_contract(manifest)
    loop, progress = _v1_compact(manifest)
    error = {'descriptor': 'V1 descriptor shape mismatch after Load0', 'init': 'V1 client Init returned7',
        'path-cleared': 'SetModelBuffer did not preserve exact model path'}[failure]
    loop.update(status='FAILED', stage='prefill-v1-load', error=error, generated_tokens=[], full_prefix=[], completed_rounds=0)
    loop['counts'] = {key: 0 for key in loop['counts']}
    life = loop['resident_lifecycle']
    for key, value in list(life.items()):
        if type(value) is int: life[key] = 0 if key != 'process_id' else 1000
    life.update(stage='NOT_LOADED', status='CLOSED',
        target_v1_lifecycle=_v1_life('target', 0, attempted=False, model_load_api=model_apis['target']),
        draft_v1_lifecycle=_v1_life('draft', 0, attempted=False, model_load_api=model_apis['draft']))
    prefill = _v1_life('prefill', 0, attempted=failure == 'descriptor', model_load_api=model_apis['prefill'])
    if failure == 'descriptor':
        prefill['failure_stage'] = 'LEGACY_TENSOR_DESCRIPTORS'
    elif failure == 'init':
        prefill.update(client_init_attempted=True, client_init_status=7, failure_stage='CLIENT_INIT', closed=True)
    else:
        prefill.update(client_init_attempted=True, client_initialized=True, client_init_status=0,
            failure_stage='MODEL_SOURCE_BINDING', closed=True)
    loop['prefill_lifecycle'] = prefill
    loop['acceptance'] = dict(accepted=0, considered=0, rate=None)
    for kind in ('prompt_ingest', 'prefill', 'decode', 'draft'):
        loop['phase_latency'][kind].update(calls=0, total_ms=10 if kind in {'prompt_ingest', 'prefill'} else 0, mean_ms=None)
    loop['supervisor_timing'].update({key: 0 for key in loop['supervisor_timing']})
    loop['supervisor_timing'].update(scope='MONOTONIC_WALL_CLOCK_NOT_NPU_INFERENCE',
        prefill_model_load_ms=10, prefill_v1_total_ms=10, total_wall_ms=20)
    progress.update(status='FAILED', generated_tokens=[], counts=loop['counts'], resident_call_status='FAILED_CLOSED',
        supervisor_timing=loop['supervisor_timing'])
    out = tmp_path / 'failed-before-session'; _compact_files(out, loop, progress)
    checked = transport.validate_retrieved_tokens(manifest, out, _expected_evidence_kind='host-fake-hiai', _remote_exitcode=1)
    assert checked['error'] == error
    assert checked['counts']['target_native_calls'] == 0
    assert checked['resident_lifecycle']['target_load_attempts'] == 0
    assert checked['child_lifecycle']['children'] == []
    assert checked['resident_lifecycle']['model_load_api'] == ('legacy-buffer' if historical else 'legacy-v1-mixed')
    if failure == 'path-cleared':
        assert checked['generated_tokens'] == [] and not checked['prefill_lifecycle']['load_attempted']


@pytest.mark.parametrize('kind', ['target', 'draft'])
def test_failed_v1_constructor_preserves_successful_sdk_load_before_shape_failure(prepared_request, tmp_path, kind):
    root, manifest, _, _ = prepared_request; _enable_prefill_request(root, manifest, tmp_path)
    manifest.update(dump=False, execution_mode='resident-v1', target_lifecycle_policy=preparation.TARGET_LIFECYCLE_POLICY,
        model_load_api=preparation.V1_MODEL_LOAD_API, model_load_apis=dict(preparation.V1_MODEL_LOAD_APIS),
        resident_abi=dict(sha256='e' * 64), prefill_abi=dict(sha256='f' * 64))
    loop, _ = _v1_compact(manifest); life = loop['resident_lifecycle']
    loop['counts'].update(target_native_calls=0, draft_native_calls=0, considered=0, accepted=0, corrections=0, bonuses=0)
    for model in ('target', 'draft'):
        constructed = kind == 'draft' and model == 'target'
        attempted = constructed or model == kind
        life.update({model + '_load_attempts': int(attempted), model + '_loads': int(constructed),
            model + '_close_calls': int(constructed), model + '_runs': 0, model + '_run_attempts': 0,
            model + '_v1_lifecycle': _v1_life(model, 0, attempted=attempted)})
    life.update(poisoned=True, stage=kind.upper() + '_LOAD/LEGACY_TENSOR_DESCRIPTORS', status='FAILED')
    observed = life[kind + '_v1_lifecycle']; observed['failure_stage'] = 'LEGACY_TENSOR_DESCRIPTORS'
    checked = transport._resident_lifecycle(loop, manifest, success=False)
    assert checked[kind + '_loads'] == 0 and observed['loaded'] is True
    assert checked[kind + '_close_calls'] == 0 and observed['unload_called'] is True
    with pytest.raises(ValueError):
        transport._resident_lifecycle(loop, manifest, success=True)


def test_build_selects_v1_supervisor_and_does_not_stage_unused_v2_runner(tmp_path, monkeypatch):
    runner = tmp_path / 'native/build/qwen35_dflash_hiai'
    _write(runner, b'old-runner-only-build-anchor')
    v1 = runner.with_name('qwen35_board_resident_v1'); _write(v1, b'new-all-v1-runtime')
    monkeypatch.setattr(transport, '_build_runner', lambda *args: runner)
    monkeypatch.setattr(transport, 'dynamic_info', lambda path: {})
    monkeypatch.setenv('BOARD_HIAI_LIB_DIR', '/system/lib64')
    uploads, _, _ = transport._build_payloads(tmp_path, 10, resident='resident-v1')
    assert [(row['local'], row['remote']) for row in uploads] == [(str(v1), 'board-serial')]
    assert uploads[0]['sha256'] == hashlib.sha256(v1.read_bytes()).hexdigest()


def test_failed_v1_io_validation_preserves_sdk_process_success_without_promoting_verified_call():
    observed = _v1_life('target', 1)
    observed.update(failure_stage='OUTPUT_BUFFER_CHECK', cleanup_error='')
    assert transport._legacy_v1_lifecycle(observed, name='qwen35_resident_target', processes=0, model_load_api='legacy-file', success=False)['successful_processes'] == 1
    with pytest.raises(ValueError):
        transport._legacy_v1_lifecycle(observed, name='qwen35_resident_target', processes=0, model_load_api='legacy-file', success=True)
    observed['successful_processes'] = 2
    with pytest.raises(ValueError):
        transport._legacy_v1_lifecycle(observed, name='qwen35_resident_target', processes=0, model_load_api='legacy-file', success=False)


@pytest.mark.parametrize('actual,accepted', [([1, 64, 2048], True), ([1, 1, 64, 2048], True),
    ([64, 2048, 1, 1], False), ([1, 2048, 64], False)])
def test_v1_actual_shape_is_checked_without_sdk_tensor_name_claims(actual, accepted):
    row = dict(index=0, shape=actual, raw_format=17, bytes=131072)
    receipt = dict(runtime_tensor_names='NOT_QUERIED_V1_SAVED_NAMED_ABI_ONLY',
        runtime_dtypes='DECLARED_BOUND_ABI_NOT_QUERIED', runtime_abi_observed_on_load=dict(inputs=[row], outputs=[row]))
    declaration = dict(inputs=[dict(shape=[1, 64, 2048], bytes=131072)], outputs=[dict(shape=[1, 64, 2048], bytes=131072)])
    if accepted:
        transport._v1_observed_abi(receipt, declaration)
    else:
        with pytest.raises(ValueError, match='shape/extent'):
            transport._v1_observed_abi(receipt, declaration)


def test_v1_target_and_prefill_format_must_match_bound_raw_enum():
    row = dict(index=0, shape=[1, 2, 3, 4], raw_format=17, bytes=48)
    receipt = dict(runtime_tensor_names='NOT_QUERIED_V1_SAVED_NAMED_ABI_ONLY',
        runtime_dtypes='DECLARED_BOUND_ABI_NOT_QUERIED', runtime_abi_observed_on_load=dict(inputs=[row], outputs=[row]))
    desc = dict(shape=[1, 2, 3, 4], bytes=48, raw_format=17)
    declaration = dict(inputs=[desc], outputs=[desc])
    transport._v1_observed_abi(receipt, declaration)
    declaration['inputs'] = [dict(desc, raw_format=9)]
    with pytest.raises(ValueError, match='shape/extent'):
        transport._v1_observed_abi(receipt, declaration)

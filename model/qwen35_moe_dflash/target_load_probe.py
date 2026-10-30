"""One cached target load-order experiment; recovery never launches a model."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shlex
import uuid

from . import board_serial_transport as transport
from .artifacts import output_path, save_json
from .draft_board import _receive
from .hdc_run import Hdc, _check_local, _file_record
from .target_compile_failure import _read_small


KIND = 'qwen35-target-load-probe-execution'
POLICY = 'init-before-weights'
_FILES = ('probe.json', 'tool.stdout', 'tool.stderr', 'tool.pid', 'tool.exitcode', 'run.sh')


def _prepared(directory):
    from .board_serial_dflash import validate_prepared
    root = Path(directory).resolve()
    if (root / 'prepared' / 'manifest.json').is_file():
        root /= 'prepared'
    manifest = validate_prepared(root)
    _, identity = transport._json(root / 'manifest.json')
    return root, manifest, identity


def _rows(manifest, cache):
    rows = []
    compiled = manifest['target_compiled']
    for value in manifest['target_payloads']:
        record = value['record']; relative = transport._relative(value['remote'])
        if (record not in compiled['generated_artifacts']
                or record['path'] != 'compile-work/generated/' + str(relative)
                or any(value[key] != record[key] for key in ('bytes', 'sha256'))
                or type(value['bytes']) is not int or value['bytes'] <= 0
                or transport._SHA.fullmatch(str(value['sha256'])) is None):
            raise ValueError('target cache inventory differs from the pinned completed compilation')
        rows.append(dict(remote=str(cache / 'payload' / relative), bytes=value['bytes'], sha256=value['sha256'], record=record))
    if not 1 <= len(rows) <= 8192 or len({row['remote'] for row in rows}) != len(rows):
        raise ValueError('target cache inventory is incomplete or ambiguous')
    primary = [row for row in rows if row['record'] == compiled['omc']]
    if (len(primary) != 1 or any(primary[0][key] != manifest['target_model'][key] for key in ('bytes', 'sha256'))
            or {row['record']['path'] for row in rows} != {row['path'] for row in compiled['generated_artifacts']}):
        raise ValueError('target cache inventory does not contain the exact complete original model')
    return rows, primary[0]['remote']


def _launcher(state):
    command = ' '.join(shlex.quote(value) for value in state['native_argv'])
    return ('#!/system/bin/sh\nset -Cu\numask 077\nmkdir launch.guard || exit 125\n'
        "printf '%s\\n' $$ > tool.pid\nLD_LIBRARY_PATH=" + shlex.quote(state['library_path'])
        + ' ' + command + ' > tool.stdout 2> tool.stderr\nprobe_status=$?\n'
        + "printf '%s\\n' \"$probe_status\" > tool.exitcode\nexit \"$probe_status\"\n")


def _argv(manifest, board, primary):
    base = manifest['target_binding']['base_plan']
    return [str(board / 'runner'), '--target-load-probe', primary, '--model-sha256', manifest['target_model']['sha256'],
        '--weight-directory', str(PurePosixPath(primary).parent), '--load-order', POLICY, '--expected-runtime-version', base['runtime_version'],
        '--ddk-id', base['ddk_id'], '--firmware-id', base['firmware_id'], '--output', str(board / 'probe.json')]


def _release_unlaunched(hdc, out, state):
    """Only known, completed prelaunch work can relinquish this exact token."""
    if not state.get('global_held') or state.get('launch_requested') or state.get('uncertain'):
        return
    token = state['board_lock_token']; global_lock = transport._GLOBAL
    guard = (transport._owner_guard(global_lock, 'owner', token)
        + ' && test "$(find ' + shlex.quote(str(global_lock)) + ' | wc -l)" -eq 2')
    action = ''
    if state.get('cache_held'):
        cache = PurePosixPath(state['target_cache']); lock = cache / '.execution-lock'; board = PurePosixPath(state['board_attempt'])
        guard += (' && ' + transport._owner_guard(cache, '.compilation-sha256', state['compilation_report_sha256'])
            + ' && ' + transport._regular_guard(board / 'target-lock.owner', size=33)
            + ' && read -r load_probe_owner < ' + shlex.quote(str(board / 'target-lock.owner'))
            + ' && test "$load_probe_owner" = ' + shlex.quote(token)
            + ' && test -d ' + shlex.quote(str(lock)) + ' && test ! -L ' + shlex.quote(str(lock))
            + ' && test "$(find ' + shlex.quote(str(lock)) + ' | wc -l)" -eq 1')
        action = 'rmdir ' + shlex.quote(str(lock)) + ' && '
    action += 'rm ' + shlex.quote(str(global_lock / 'owner')) + ' && rmdir ' + shlex.quote(str(global_lock))
    transport._checkpoint(out, state, 'release-unlaunched-locks-requested', uncertain=True)
    _, acknowledgement = hdc.shell(guard + ' && ' + action + " && printf 'RELEASED\\n'", 'release-unlaunched-load-probe-locks')
    if acknowledgement != 'RELEASED':
        raise ValueError('unlaunched target-load lock release is uncertain')
    transport._checkpoint(out, state, 'unlaunched-locks-released', global_held=False, cache_held=False, uncertain=False,
        lock_release='RELEASED_BEFORE_MODEL_LAUNCH')


def _locks(hdc, state, *, release=False):
    board = PurePosixPath(state['board_attempt']); cache = PurePosixPath(state['target_cache'])
    lock = cache / '.execution-lock'; token = state['board_lock_token']; global_lock = transport._GLOBAL
    absent = ' && '.join('test ! -e ' + shlex.quote(str(path)) + ' && test ! -L ' + shlex.quote(str(path))
        for path in (global_lock, lock))
    guard = (transport._owner_guard(global_lock, 'owner', token) + ' && '
        + transport._owner_guard(cache, '.compilation-sha256', state['compilation_report_sha256'])
        + ' && ' + transport._regular_guard(board / 'target-lock.owner', size=33)
        + ' && read -r load_probe_owner < ' + shlex.quote(str(board / 'target-lock.owner'))
        + ' && test "$load_probe_owner" = ' + shlex.quote(token)
        + ' && test -d ' + shlex.quote(str(lock)) + ' && test ! -L ' + shlex.quote(str(lock))
        + ' && test "$(find ' + shlex.quote(str(lock)) + ' | wc -l)" -eq 1'
        + ' && test "$(find ' + shlex.quote(str(global_lock)) + ' | wc -l)" -eq 2')
    action = "printf 'OWNED\\n'"
    if release:
        completion = transport._completion(hdc, state)
        if completion is None or completion != state.get('completion') or not state.get('native_process_id'):
            raise ValueError('target load probe completion is uncertain; owned locks retained')
        for name, value in (('tool.pid', completion['launcher_process_id']), ('tool.exitcode', completion['remote_exitcode'])):
            guard += (' && ' + transport._regular_guard(board / name, size=len(str(value)) + 1)
                + ' && test "$(cat ' + shlex.quote(str(board / name)) + ')" = ' + shlex.quote(str(value)))
        for pid in (completion['launcher_process_id'], state['native_process_id']):
            if type(pid) is not int or not 0 < pid <= (1 << 31) - 1:
                raise ValueError('target load probe PID is invalid')
            guard += ' && test ! -d /proc/' + str(pid)
        action = ('rmdir ' + shlex.quote(str(lock)) + ' && rm ' + shlex.quote(str(global_lock / 'owner'))
            + ' && rmdir ' + shlex.quote(str(global_lock)) + " && printf 'RELEASED\\n'")
    _, text = hdc.shell('if ' + absent + "; then printf 'ABSENT\\n'; else " + guard + ' && ' + action + '; fi',
        'release-target-load-locks' if release else 'inspect-target-load-locks', timeout=min(60, hdc.timeout))
    if text not in ({'ABSENT', 'RELEASED'} if release else {'ABSENT', 'OWNED'}):
        raise ValueError('target load probe locks are foreign, partial or uncertain')
    return text


def validate_load_receipt(value, manifest, state, *, _expected_evidence_kind='device-hiai'):
    """Validate the declared experiment, never infer Run readiness from load."""
    base = manifest['target_binding']['base_plan']; exitcode = state['completion']['remote_exitcode']
    expected = dict(schema_version=1, artifact_kind='qwen35-hiai-target-load-probe', evidence_kind=_expected_evidence_kind, model_load_api='file-v2',
        status='TARGET_LOAD_PROBE_PASSED' if exitcode == 0 else 'TARGET_LOAD_PROBE_FAILED', load_order=POLICY,
        model_name='qwen35_resident_target',
        numerical_validation='NOT_RUN',
        model_path=state['model_path'], model_sha256=manifest['target_model']['sha256'], model_bytes=manifest['target_model']['bytes'],
        weight_directory=str(PurePosixPath(state['model_path']).parent), expected_runtime_version=base['runtime_version'],
        ddk_id=base['ddk_id'], firmware_id=base['firmware_id'], run_called=False, process_called=False, tensor_allocations_requested=0)
    if (any(value.get(key) != item or type(value.get(key)) is not type(item) for key, item in expected.items())
            or type(value.get('process_id')) is not int or not 0 < value['process_id'] <= (1 << 31) - 1
            or type(value.get('elapsed_ms')) is not int or value['elapsed_ms'] < 0
            or type(value.get('model_content_unchanged')) is not bool
            or not isinstance(value.get('runtime_stage'), str) or not value['runtime_stage']):
        raise ValueError('load-only native receipt differs from this exact model/order/runtime invocation')
    life = value.get('lifecycle', {})
    if (not isinstance(life, dict) or life.get('weight_init_policy') != 'EXPERIMENTAL_INIT_BEFORE_WEIGHTS_LOAD_PROBE_ONLY_NO_FALLBACK'
            or life.get('ordering_validation') != 'NOT_VALIDATED_ON_DEVICE' or life.get('closed') is not True
            or any(type(life.get(key)) is not int or life[key] != 0 for key in ('run_attempts', 'successful_runs'))
            or type(life.get('weights_deinit_status')) is not int
            or not isinstance(life.get('cleanup_error'), str)):
        raise ValueError('load-only lifecycle is unclosed or claims model execution')
    for prefix in ('manager', 'weights'):
        attempted, initialized, status = life.get(prefix + '_init_attempted'), life.get(prefix + '_initialized'), life.get(prefix + '_init_status')
        if (type(attempted) is not bool or type(initialized) is not bool or initialized and (not attempted or type(status) is not int or status != 0)
                or status is not None and type(status) is not int or not attempted and status is not None):
            raise ValueError('load-only lifecycle invents an SDK call attempt or returned status')
    if life['weights_init_attempted'] and not life['manager_initialized']:
        raise ValueError('load-only receipt violated Init-before-InitWeights; no fallback is permitted')
    if (life.get('manager_deinit_called') is not life['manager_init_attempted']
            or life.get('weights_deinit_called') is not life['weights_initialized']):
        raise ValueError('load-only cleanup differs from the actually acquired model scope')
    if exitcode == 0:
        if (value.get('runtime_version') != base['runtime_version'] or value.get('runtime_stage') != 'CLOSED'
                or not life['manager_initialized'] or not life['weights_initialized']
                or type(life.get('weights_deinit_status')) is not int or life['weights_deinit_status'] != 0 or life['cleanup_error']
                or value.get('model_content_unchanged') is not True):
            raise ValueError('successful load-only receipt lacks exact API successes and clean closure')
    elif not isinstance(value.get('error'), str) or not value['error'].strip():
        raise ValueError('failed load-only receipt omits its actual failure')
    return value


def _retrieve(hdc, out, state, manifest):
    board = PurePosixPath(state['board_attempt']); pins = {}
    for name in _FILES:
        hdc.shell(transport._regular_guard(board / name) + ' && test "$(wc -c < ' + shlex.quote(str(board / name))
            + ')" -le 1048576', 'bound-load-probe-evidence', timeout=min(30, hdc.timeout))
        if _receive(hdc, board, out, name) is None:
            raise ValueError('completed load probe is missing ' + name)
        pins[name] = transport._full(out / name)
    if ((out / 'run.sh').read_text() != state['launcher']
            or (out / 'tool.pid').read_text() != str(state['completion']['launcher_process_id']) + '\n'
            or (out / 'tool.exitcode').read_text() != str(state['completion']['remote_exitcode']) + '\n'):
        raise ValueError('load probe launcher/completion evidence changed')
    value, identity = transport._json(out / 'probe.json', 1 << 20)
    validate_load_receipt(value, manifest, state)
    if any(transport._full(out / name) != stamp for name, stamp in pins.items()):
        raise ValueError('load probe evidence changed during validation')
    state.update(native_report=identity, native=value, native_process_id=value['process_id'],
        runtime_stage=value.get('runtime_stage'), lifecycle=value['lifecycle'], elapsed_ms=value['elapsed_ms'],
        device_execution='LOAD_ONLY_COMPLETED_NO_RUN', counts=dict(target_run_calls=0, draft_calls=0, prefill_calls=0))
    if state['completion']['remote_exitcode'] != 0:
        state.update(error=value['error'], execution_failure_stage=value.get('runtime_stage'))


def _finish(out, state, hdc):
    state['lock_state'] = ('RETAINED_OR_UNCERTAIN' if state.get('global_held') or state.get('cache_held') or state.get('uncertain')
        else 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT')
    state['hdc_invocations'] = hdc.count if hdc else 0
    state['next'] = ('Use --apply with a separate new output to run this one declared load-only experiment.' if state.get('status') == 'TARGET_LOAD_PROBE_PLANNED'
        else 'Retain this closed load-only observation; no inference or multi-model residency was validated.' if state.get('stage') == 'complete'
        else 'Recover this exact execution; never relaunch an uncertain load or delete its locks.' if state.get('launch_requested')
        else 'Inspect the retained deployment checkpoint before any retry or owner-safe lock cleanup.')
    transport._checkpoint(out, state, state.get('phase', 'failed'))
    save_json(out / 'result.json', state)
    return state


def run_target_load_probe(prepared, output, *, apply=False, detach=False, timeout_seconds=600, lock_wait_seconds=1800):
    if (type(apply) is not bool or type(detach) is not bool or detach and not apply or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200
            or type(lock_wait_seconds) is not int or not 0 <= lock_wait_seconds <= 7200):
        raise ValueError('bounded load-probe timeouts and explicit detach boolean required')
    out = output_path(output); supplied = Path(prepared).resolve()
    bound_root = supplied / 'prepared' if (supplied / 'prepared' / 'manifest.json').is_file() else supplied
    if out == bound_root or bound_root in out.parents:
        raise ValueError('load-probe output must be outside immutable preparation')
    out.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, artifact_kind=KIND, status='FAILED', output=str(out), model_compilation='NOT_RUN',
        device_execution='NOT_RUN', model_inference='NOT_RUN', numerical_validation='NOT_EVALUATED', load_order=POLICY,
        board_lock_token=uuid.uuid4().hex, global_lock=str(transport._GLOBAL), global_held=False, cache_held=False,
        uncertain=False, launch_requested=False, automatic_model_retry='FORBIDDEN')
    hdc = None
    try:
        root, manifest, identity = _prepared(prepared); base = manifest['target_binding']['base_plan']
        digest = base['compilation_report_sha256']; cache = transport._remote(manifest['board_model_root']) / ('codex-target-' + digest[:20])
        rows, primary = _rows(manifest, cache); board = PurePosixPath('/data/local/tmp/qwen35_target_load_' + state['board_lock_token'])
        state.update(prepared=str(root), prepared_manifest=identity, target_cache=str(cache), board_attempt=str(board),
            compilation_report_sha256=digest, model_path=primary)
        transport._checkpoint(out, state, 'host-bound')
        if not apply:
            state.update(status='TARGET_LOAD_PROBE_PLANNED', stage='complete', cache_policy='CACHE_HIT_ONLY_NO_MODEL_UPLOAD',
                declared_target_payload_bytes=sum(row['bytes'] for row in rows))
            return _finish(out, state, hdc)
        executable, serial, server = transport._environment()
        state.update(serial=serial, hdc_server=server, hdc_executable=executable)
        print('target-load-probe: building native runner; existing target OMC/cache is reused', flush=True)
        uploads, search, deployment = transport._build_payloads(out, timeout_seconds)
        uploads = [dict(row, remote='runner') if row['remote'] == 'draft-runner' else row
            for row in uploads if row['remote'] != 'board-serial']
        state['libraries'] = deployment; state['library_path'] = ':'.join([str(board / 'lib'), *search])
        hdc = Hdc(executable, serial, server, out, timeout_seconds)
        _, firmware = hdc.shell('command -v sha256sum >/dev/null && command -v nohup >/dev/null && command -v find >/dev/null && param get const.ohos.fullname', 'load-probe-device-preflight')
        if firmware.strip() != base['firmware_id'].strip():
            raise ValueError('load probe firmware differs from the pinned runtime proof')
        transport._mutate(hdc, out, state, 'mkdir ' + shlex.quote(str(board)), 'create-load-probe-attempt')
        from .target_model_io_probe import _acquire_global
        transport._checkpoint(out, state, 'acquire-global-requested', uncertain=True)
        _acquire_global(hdc, transport._GLOBAL, state['board_lock_token'], lock_wait_seconds)
        transport._checkpoint(out, state, 'global-locked', global_held=True, uncertain=False)
        hdc.shell(transport._owner_guard(cache, '.compilation-sha256', digest), 'require-existing-target-cache')
        transport._mutate(hdc, out, state, 'mkdir ' + shlex.quote(str(cache / '.execution-lock'))
            + " && printf '%s\\n' " + shlex.quote(state['board_lock_token']) + ' > ' + shlex.quote(str(board / 'target-lock.owner')), 'acquire-target-cache-lock')
        transport._checkpoint(out, state, 'cache-locked', cache_held=True)
        print('target-load-probe: checking the existing target cache once; missing files are never uploaded', flush=True)
        decisions = transport._cache_inventory(hdc, rows)
        state['cache_missing_bytes'] = sum(row['bytes'] for row in rows if decisions[row['remote']] != 'OK')
        if state['cache_missing_bytes']:
            raise ValueError('cache-hit-only load probe requires every original target payload; no model upload attempted')
        state['cache_inventory'] = [{key: row[key] for key in ('remote', 'bytes', 'sha256')} for row in rows]
        state['native_argv'] = _argv(manifest, board, primary)
        state['launcher'] = _launcher(state); (out / 'run.sh').write_text(state['launcher'])
        uploads.append(_file_record(out / 'run.sh', 'run.sh'))
        transport._directories(hdc, out, state, board, [board / 'lib'], 'create-load-probe-library-directory')
        transport._checkpoint(out, state, 'upload-runner-requested', uncertain=True, runtime_uploads=uploads)
        for row in uploads:
            _check_local(row); hdc.send(row['local'], board / row['remote'])
            hdc.shell(transport._regular_guard(board / row['remote'], size=row['bytes'])
                + ' && probe_sha=$(sha256sum ' + shlex.quote(str(board / row['remote']))
                + ') && probe_sha=${probe_sha%% *} && test "$probe_sha" = ' + shlex.quote(row['sha256']), 'verify-load-probe-upload')
        transport._checkpoint(out, state, 'runner-uploaded', uncertain=False)
        transport._mutate(hdc, out, state, 'chmod 755 ' + shlex.quote(str(board / 'runner')), 'set-load-probe-runner-permissions')
        _, current_manifest, current_identity = _prepared(root)
        if current_manifest != manifest or current_identity != identity:
            raise ValueError('load-probe preparation changed before the single launch')
        transport._checkpoint(out, state, 'launch-requested', launch_requested=True, uncertain=True, device_execution='ATTEMPTED')
        _, acknowledgement = hdc.shell('cd ' + shlex.quote(str(board)) + ' && { nohup sh run.sh </dev/null >launcher.stdout 2>launcher.stderr & printf \'LAUNCHED\\n\'; }', 'launch-target-load-probe')
        if acknowledgement != 'LAUNCHED':
            raise ValueError('load probe launch acknowledgement is uncertain')
        transport._checkpoint(out, state, 'launch-acknowledged', uncertain=False)
        completion = None if detach else transport._wait(hdc, out, state, timeout_seconds, 15)
        if completion is None:
            state.update(status='TARGET_LOAD_PROBE_STARTED', stage='running-detached', device_execution='RUNNING_OR_COMPLETION_PENDING')
        else:
            _locks(hdc, state); _retrieve(hdc, out, state, manifest)
            transport._checkpoint(out, state, 'release-locks-requested', uncertain=True)
            state['lock_release'] = _locks(hdc, state, release=True)
            state.update(status='TARGET_LOAD_PROBE_PASSED' if completion['remote_exitcode'] == 0 else 'FAILED',
                stage='complete', global_held=False, cache_held=False, uncertain=False)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stage=state.get('phase', 'host-preflight'))
        if hdc is not None:
            try:
                _release_unlaunched(hdc, out, state)
            except BaseException as cleanup:
                state['prelaunch_cleanup_error'] = f'{type(cleanup).__name__}: {cleanup}'
    return _finish(out, state, hdc)


def recover_target_load_probe(execution, output, *, timeout_seconds=60):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200:
        raise ValueError('bounded positive recovery timeout required')
    source = Path(execution).resolve(); out = output_path(output)
    if source == out or source in out.parents:
        raise ValueError('recovery output must be a separate new directory')
    out.mkdir(parents=True, exist_ok=False)
    state, identity = transport._json(source / 'checkpoint.json')
    state = dict(state, status='FAILED', output=str(out), recovery_source=str(source), recovery_source_identity=identity,
        device_execution='NOT_RUN_BY_RECOVERY', recovery_model_execution='FORBIDDEN')
    hdc = None
    try:
        token = state.get('board_lock_token')
        if (state.get('artifact_kind') != KIND or transport._TOKEN.fullmatch(str(token)) is None or not state.get('launch_requested')
                or state.get('global_lock') != str(transport._GLOBAL)
                or state.get('board_attempt') != '/data/local/tmp/qwen35_target_load_' + token):
            raise ValueError('recovery requires the exact original target load attempt and token')
        root, manifest, prepared_id = _prepared(state['prepared']); base = manifest['target_binding']['base_plan']
        cache = transport._remote(manifest['board_model_root']) / ('codex-target-' + base['compilation_report_sha256'][:20])
        _, primary = _rows(manifest, cache)
        if (state['target_cache'] != str(cache) or state['model_path'] != primary or state['load_order'] != POLICY
                or state['compilation_report_sha256'] != base['compilation_report_sha256']
                or state['prepared_manifest'] != prepared_id or state['native_argv'] != _argv(manifest, PurePosixPath(state['board_attempt']), primary)
                or state['launcher'] != _launcher(state)):
            raise ValueError('recovery model/cache/order differs from the original pinned experiment')
        executable, serial, server = transport._environment()
        if (serial, server) != (state['serial'], state['hdc_server']):
            raise ValueError('target load recovery must use the identical device/server')
        hdc = Hdc(executable, serial, server, out, timeout_seconds)
        _locks(hdc, state)
        completion = transport._wait(hdc, out, state, timeout_seconds, 15)
        if completion is None:
            state.update(status='TARGET_LOAD_PROBE_RUNNING_OR_UNCERTAIN', stage='running-detached')
        else:
            _retrieve(hdc, out, state, manifest)
            transport._checkpoint(out, state, 'release-locks-requested', uncertain=True)
            state['lock_release'] = _locks(hdc, state, release=True)
            state.update(status='TARGET_LOAD_PROBE_PASSED' if completion['remote_exitcode'] == 0 else 'FAILED',
                stage='complete', global_held=False, cache_held=False, uncertain=False)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stage=state.get('phase', 'recover-preflight'))
    return _finish(out, state, hdc)

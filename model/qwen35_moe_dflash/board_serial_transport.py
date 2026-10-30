"""Stage once and supervise one detached board-local serial DFlash request.

No HDC call advances an individual token.  A lost launch response is never
replayed: recovery only reads the recorded attempt and releases its exact locks
after the supervisor and every recorded child have completed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import time
import uuid

from .artifacts import output_path, save_json
from .draft_board import _build_runner, _receive
from .elf_dynamic import dynamic_info
from .hashing import sha256_file
from .hdc_run import Hdc, _check_local, _file_record, _relative, _text
from .hiai_deployment import BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers, discover_libraries
from .target_compile_failure import _read_small, _safe
from .target_prefill_smoke_plan import _identity


_KIND = 'qwen35-board-serial-dflash-execution'
_GLOBAL = PurePosixPath('/data/local/tmp/.qwen35-target-model-io-lock')
_SHA = re.compile(r'[0-9a-f]{64}')
_TOKEN = re.compile(r'[0-9a-f]{32}')
_ATTEMPT_LOGS = ('supervisor.stdout', 'supervisor.stderr', 'tool.pid', 'tool.exitcode', 'run.sh')
_MAX_FILE = 128 << 20
_MAX_TOTAL = 2 << 30


def _json(path, maximum=64 << 20):
    raw, identity = _read_small(path, maximum)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('board serial metadata must be an object: ' + str(path))
    return value, identity


def _full(path):
    path, stamp = _safe(path)
    full = _identity(path.stat())
    if [full[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
        raise ValueError('board serial file changed while snapshotting: ' + str(path))
    return full


def _checkpoint(out, state, phase, **updates):
    """Durable before-operation journal, including when Python is interrupted."""
    state.update(updates, phase=phase)
    temporary = out / ('checkpoint-' + uuid.uuid4().hex + '.pending')
    raw = json.dumps(state, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, out / 'checkpoint.json')
    descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remote(value):
    _text(str(value), 'remote path')
    path = PurePosixPath(value)
    if (not path.is_absolute() or str(path) != str(value) or str(path) == '/'
            or any(part in {'', '.', '..'} for part in str(value).split('/')[1:])
            or re.fullmatch(r'/[A-Za-z0-9_./-]+', str(path)) is None):
        raise ValueError('board serial remote path must be a normalized absolute safe path')
    return path


def _regular_guard(path, *, size=None):
    quoted = shlex.quote(str(path))
    guard = 'test -f ' + quoted + ' && test ! -L ' + quoted
    if size is not None:
        guard += ' && test "$(wc -c < ' + quoted + ')" -eq ' + str(size)
    return guard


def _owner_guard(directory, filename, owner):
    path = directory / filename
    return ('test -d ' + shlex.quote(str(directory)) + ' && test ! -L ' + shlex.quote(str(directory))
        + ' && ' + _regular_guard(path, size=len(owner) + 1)
        + ' && read -r serial_owner < ' + shlex.quote(str(path))
        + ' && test "$serial_owner" = ' + shlex.quote(owner))


def _mutate(hdc, out, state, command, label, *, timeout=None, **updates):
    _checkpoint(out, state, label + '-requested', uncertain=True, **updates)
    result = hdc.shell(command, label, timeout=min(60, hdc.timeout) if timeout is None else timeout)
    _checkpoint(out, state, label + '-completed', uncertain=False)
    return result


def _batches(commands, budget=12000):
    batch, size = [], 0
    for command in commands:
        if len(command.encode()) > budget:
            raise ValueError('one remote inventory operation exceeds the bounded shell command budget')
        if batch and size + len(command.encode()) + 4 > budget:
            yield batch; batch, size = [], 0
        batch.append(command); size += len(command.encode()) + 4
    if batch:
        yield batch


def _environment():
    serial, server = _text(os.environ.get('SN'), 'SN'), _text(os.environ.get('BIP'), 'BIP')
    if (serial.startswith('-') or server.startswith('-') or re.fullmatch(r'[^\s:]+:[0-9]+', server) is None
            or not 0 < int(server.rsplit(':', 1)[1]) < 65536):
        raise ValueError('SN and the selected HDC server BIP host:port are required')
    return _text(os.environ.get('HDC', 'hdc'), 'HDC'), serial, server


def _owned_directory(hdc, out, state, directory, filename, owner, label):
    parent = directory.parent
    command = ('test -d ' + shlex.quote(str(parent)) + ' && test ! -L ' + shlex.quote(str(parent))
        + ' && if mkdir ' + shlex.quote(str(directory)) + ' 2>/dev/null; then printf \'%s\\n\' '
        + shlex.quote(owner) + ' > ' + shlex.quote(str(directory / filename))
        + '; else ' + _owner_guard(directory, filename, owner) + '; fi')
    _mutate(hdc, out, state, command, label)


def _directories(hdc, out, state, root, paths, label):
    children = set()
    for path in paths:
        path = _remote(path)
        if root not in path.parents and path != root:
            raise ValueError('remote directory escapes its owned cache')
        while path != root:
            children.add(path); path = path.parent
    command = []
    for path in sorted(children, key=lambda item: (len(item.parts), str(item))):
        quoted = shlex.quote(str(path))
        command.append('test ! -L ' + quoted + ' && if test -e ' + quoted
            + '; then test -d ' + quoted + '; else mkdir ' + quoted + '; fi')
    if command:
        _mutate(hdc, out, state, ' && '.join(command), label)


def _cache_inventory(hdc, rows):
    """One actual remote SHA pass over the whole request's static payloads."""
    commands = []
    for row in rows:
        path = shlex.quote(row['remote'])
        commands.append('test ! -L ' + path + ' || exit 1; if test -e ' + path + '; then '
            + 'test -f ' + path + ' || exit 1; serial_hash=$(sha256sum ' + path + ') || exit 1; '
            + 'serial_hash=${serial_hash%% *}; test "${#serial_hash}" -eq 64 || exit 1; '
            + 'case "$serial_hash" in *[!0-9a-f]*) exit 1;; esac; '
            + 'test "$serial_hash" = ' + shlex.quote(row['sha256']) + ' || exit 1; '
            + 'test "$(wc -c < ' + path + ')" -eq ' + str(row['bytes']) + ' || exit 1; '
            + 'printf \'OK %s\\n\' ' + path + '; else printf \'SEND %s\\n\' ' + path + '; fi')
    responses = []
    for batch in _batches(commands):
        _, text = hdc.shell('\n'.join(batch), 'inspect-whole-loop-cache', timeout=hdc.timeout)
        responses.append(text)
    text = '\n'.join(responses)
    decisions = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(OK|SEND) (/[A-Za-z0-9_./-]+)', line)
        if not match or match[2] in decisions:
            raise ValueError('remote static-payload inventory is malformed or duplicated')
        decisions[match[2]] = match[1]
    if set(decisions) != {row['remote'] for row in rows}:
        raise ValueError('remote static-payload inventory is incomplete')
    return decisions


def _local_upload(row, manifest):
    if row['kind'] == 'target':
        from .original_target_board import _recorded_board_payload
        return _recorded_board_payload(Path(manifest['target_binding']['base_plan']['compilation']),
            row['record'], manifest['target_compiled'], smoke_proof=manifest['target_binding']['base_plan'])
    source = Path(row['local'])
    before = _full(source)
    if row.get('identity') is not None and before != row['identity']:
        raise ValueError('uncached source differs from the immutable preparation identity: ' + str(source))
    if before['bytes'] != row['bytes'] or sha256_file(source) != row['sha256'] or _full(source) != before:
        raise ValueError('uncached source changed or differs from its SHA256: ' + str(source))
    return source


def _upload_cached(hdc, out, state, rows, decisions, manifest):
    from .target_model_io_probe import _space
    _, disk = hdc.shell('df -Pk ' + shlex.quote(state['board_attempt']), 'whole-loop-board-disk')
    missing = sum(row['bytes'] for row in rows if decisions[row['remote']] == 'SEND')
    free = _space(disk)
    # Final states plus intermediate per-step files coexist until successful retrieval.
    steps = len(manifest['prompt_token_ids']) + manifest['max_new_tokens']
    output_bytes = sum(row['bytes'] for row in manifest['target_binding']['base_plan']['outputs'])
    input_bytes = sum(row['bytes'] for row in manifest['target_binding']['base_plan'].get('inputs', []))
    draft_bytes = manifest['max_new_tokens'] * ((2 << 20) + 61440 + (1 << 20))
    feature_bytes = steps * 32768
    reserve = (128 << 20) + output_bytes * (steps + 1) + input_bytes * steps + draft_bytes + feature_bytes
    state['board_disk_budget'] = {'free_bytes': free, 'missing_upload_bytes': missing,
        'retained_target_input_bytes': input_bytes * steps, 'retained_target_output_bytes': output_bytes * steps,
        'draft_input_output_metadata_reserve_bytes': draft_bytes, 'final_feature_bytes': feature_bytes,
        'retained_output_reserve_bytes': reserve, 'required_bytes': missing + reserve}
    print(f'board-serial: whole-request cache verified; missing upload={missing} bytes; retained-data reserve={reserve} bytes', flush=True)
    if free < missing + reserve:
        raise ValueError('board disk budget is insufficient for static payloads and retained loop outputs')
    for row in rows:
        if decisions[row['remote']] == 'OK':
            continue
        local = _local_upload(row, manifest)
        remote = PurePosixPath(row['remote']); partial = remote.with_name(remote.name + '.serial-partial')
        _mutate(hdc, out, state, 'test ! -e ' + shlex.quote(str(partial)) + ' && test ! -L '
            + shlex.quote(str(partial)), 'check-static-partial-absent')
        _checkpoint(out, state, 'send-static-payload-requested', uncertain=True,
            pending_payload={'remote': str(remote), 'bytes': row['bytes'], 'sha256': row['sha256']})
        hdc.send(local, partial)
        _mutate(hdc, out, state, _regular_guard(partial, size=row['bytes'])
            + ' && serial_hash=$(sha256sum ' + shlex.quote(str(partial)) + ') && serial_hash=${serial_hash%% *}'
            + ' && test "$serial_hash" = ' + shlex.quote(row['sha256'])
            + ' && test ! -e ' + shlex.quote(str(remote)) + ' && test ! -L ' + shlex.quote(str(remote))
            + ' && mv ' + shlex.quote(str(partial)) + ' ' + shlex.quote(str(remote)), 'publish-static-payload',
            timeout=hdc.timeout)
    state['cached_payloads'] = [{key: row[key] for key in ('remote', 'bytes', 'sha256', 'kind')}
        | {'decision': decisions[row['remote']]} for row in rows]


def _build_payloads(out, timeout):
    runner = _build_runner(out, timeout)
    supervisor = runner.parent / 'qwen35_board_serial'
    _safe(supervisor); dynamic_info(supervisor)
    uploads = [_file_record(runner, 'draft-runner'), _file_record(supervisor, 'board-serial')]
    paths = list(BOARD_LIBRARY_SEARCH_PATHS)
    private = os.environ.get('BOARD_HIAI_LIB_DIR')
    if private:
        _remote(private); paths.insert(0, private)
        return uploads, paths, {'mode': 'existing-board-directory', 'path': private}
    libraries = {}
    discoveries = []
    # discover_libraries binds result.binary_sha256 to the original HIAI
    # executable. The sibling supervisor uses the same selected SDK C++ runtime
    # and is separately ELF/SHA bound; it is not that recorded HIAI executable.
    for binary in (runner,):
        found = discover_libraries(binary, out / 'native')
        discoveries.append(found['metadata'])
        for path in found['libraries']:
            row = _file_record(path.resolve(), 'lib/' + str(_relative(path.name)))
            previous = libraries.get(path.name)
            if previous and (previous['sha256'], previous['bytes']) != (row['sha256'], row['bytes']):
                raise ValueError('runner dependency basenames have conflicting library identities')
            libraries[path.name] = row
    dependencies = [{'name': PurePosixPath(row['remote']).name,
        'kind': 'runner' if row in uploads else 'library', 'sha256': row['sha256'], **dynamic_info(row['local'])}
        for row in [*uploads, *libraries.values()]]
    blockers = candidate_blockers(dependencies, require_pair=False)
    if blockers:
        raise ValueError('cannot select board libraries: ' + '; '.join(blockers))
    uploads.extend(row for name, row in libraries.items() if name not in DDK_HCL_EXCLUDED)
    return uploads, paths, {'mode': 'candidate-without-ddk-hcl', 'discoveries': discoveries,
        'dependencies': dependencies, 'backend_selection': 'UNVERIFIED'}


def _static_rows(manifest, target_cache, draft_cache, table_cache):
    rows = []
    for row in manifest['target_payloads']:
        relative = _relative(row['remote'])
        rows.append({'local': row['local'], 'remote': str(target_cache / 'payload' / relative),
            'bytes': row['bytes'], 'sha256': row['sha256'], 'record': row['record'], 'kind': 'target'})
    model = manifest['draft']['model_source']
    rows.append({'local': model['path'], 'remote': str(draft_cache / 'model/model.omc'),
        'bytes': model['bytes'], 'sha256': model['sha256'], 'identity': model.get('identity'), 'kind': 'draft'})
    for name in ('weight', 'scale'):
        source = manifest['tables'][name]
        rows.append({'local': source['path'], 'remote': str(table_cache / (name + '.bin')),
            'bytes': source['bytes'], 'sha256': source['sha256'], 'identity': source['identity'], 'kind': 'embedding-' + name})
    if (not rows or len(rows) > 8192 or len({row['remote'] for row in rows}) != len(rows)
            or any(type(row['bytes']) is not int or row['bytes'] <= 0 or _SHA.fullmatch(row['sha256']) is None for row in rows)):
        raise ValueError('whole-loop static payload inventory is invalid or ambiguous')
    return rows


def _launcher(state):
    board = PurePosixPath(state['board_attempt'])
    argv = [str(board / 'board-serial'), '--config', str(board / 'config.tsv'), '--output', str(board / 'results')]
    return ('#!/system/bin/sh\nset -Cu\ncd ' + shlex.quote(str(board)) + ' || exit 125\n'
        + 'exec 3> launch.guard\n'  # Retained, attempt-local launch evidence.
        + 'printf \'%s\\n\' "$$" > tool.pid\n'
        + 'LD_LIBRARY_PATH=' + shlex.quote(state['supervisor_library_path']) + ' ' + shlex.join(argv)
        + ' > supervisor.stdout 2> supervisor.stderr\n'
        + 'serial_rc=$?\nprintf \'%s\\n\' "$serial_rc" > tool.exitcode\nexit "$serial_rc"\n')


def _completion(hdc, state):
    board = PurePosixPath(state['board_attempt'])
    # A PID can be reused: any live /proc entry is conservative uncertainty.
    command = ('test -d ' + shlex.quote(str(board)) + ' && test ! -L ' + shlex.quote(str(board))
        + ' && if test -e ' + shlex.quote(str(board / 'tool.exitcode')) + ' || test -L '
        + shlex.quote(str(board / 'tool.exitcode')) + '; then '
        + _regular_guard(board / 'tool.exitcode') + ' && test "$(wc -c < '
        + shlex.quote(str(board / 'tool.exitcode')) + ')" -le 4 && '
        + _regular_guard(board / 'tool.pid') + ' && test "$(wc -c < '
        + shlex.quote(str(board / 'tool.pid')) + ')" -le 11'
        + ' && printf \'EXIT=\' && cat ' + shlex.quote(str(board / 'tool.exitcode'))
        + ' && printf \'PID=\' && cat ' + shlex.quote(str(board / 'tool.pid'))
        + ' && read -r serial_pid < ' + shlex.quote(str(board / 'tool.pid'))
        + ' && case "$serial_pid" in \'\'|*[!0-9]*) exit 1;; esac'
        + ' && if test -d "/proc/$serial_pid"; then printf \'PROCESS_PRESENT\\n\'; else printf \'PROCESS_ABSENT\\n\'; fi'
        + '; else printf \'PENDING\\n\'; fi')
    _, text = hdc.shell(command, 'poll-whole-loop-completion', timeout=min(30, hdc.timeout))
    if text == 'PENDING':
        return None
    match = re.fullmatch(r'EXIT=(0|[1-9][0-9]{0,2})\nPID=([1-9][0-9]{0,9})\nPROCESS_(ABSENT|PRESENT)', text.replace('\r', ''))
    if not match or int(match[1]) > 255 or int(match[2]) > (1 << 31) - 1:
        raise ValueError('supervisor completion is active, malformed, or uncertain: ' + repr(text))
    if match[3] == 'PRESENT':
        return None
    return {'remote_exitcode': int(match[1]), 'launcher_process_id': int(match[2]),
        'completion_proof': 'RETAINED_EXITCODE_AND_PID_ABSENT_IN_ONE_FRAMED_SHELL'}


def _lock_checks(hdc, state, *, release=False):
    target, draft = PurePosixPath(state['target_cache']), PurePosixPath(state['draft_cache'])
    board, token = PurePosixPath(state['board_attempt']), state['board_lock_token']
    target_lock, draft_lock = target / '.execution-lock', draft / '.execution-lock'
    both_absent = ' && '.join('test ! -e ' + shlex.quote(str(path)) + ' && test ! -L ' + shlex.quote(str(path))
        for path in (_GLOBAL, target_lock, draft_lock))
    owned = (_owner_guard(_GLOBAL, 'owner', token) + ' && '
        + _owner_guard(target, '.compilation-sha256', state['compilation_report_sha256']) + ' && '
        + _owner_guard(draft, '.model-sha256', state['draft_model_sha256']) + ' && '
        + _owner_guard(draft_lock, 'owner', token)
        + ' && ' + _regular_guard(board / 'target-lock.owner', size=33)
        + ' && read -r serial_target_token < ' + shlex.quote(str(board / 'target-lock.owner'))
        + ' && test "$serial_target_token" = ' + shlex.quote(token)
        + ' && test -d ' + shlex.quote(str(target_lock)) + ' && test ! -L ' + shlex.quote(str(target_lock))
        + ' && test "$(find ' + shlex.quote(str(target_lock)) + ' | wc -l)" -eq 1'
        + ' && test "$(find ' + shlex.quote(str(_GLOBAL)) + ' | wc -l)" -eq 2'
        + ' && test "$(find ' + shlex.quote(str(draft_lock)) + ' | wc -l)" -eq 2')
    if release:
        completion = _completion(hdc, state)
        if completion is None or completion != state.get('completion'):
            raise ValueError('supervisor completion changed before exact-owner lock release')
        # Repeat completion inside the actual release shell, not only in a previous query.
        owned += (' && ' + _regular_guard(board / 'tool.exitcode', size=len(str(completion['remote_exitcode'])) + 1)
            + ' && test "$(cat ' + shlex.quote(str(board / 'tool.exitcode')) + ')" = '
            + shlex.quote(str(completion['remote_exitcode'])) + ' && '
            + _regular_guard(board / 'tool.pid', size=len(str(completion['launcher_process_id'])) + 1)
            + ' && test "$(cat ' + shlex.quote(str(board / 'tool.pid')) + ')" = '
            + shlex.quote(str(completion['launcher_process_id']))
            + ' && test ! -d /proc/' + str(completion['launcher_process_id']))
        for pid in state.get('completed_native_pids', []):
            if type(pid) is not int or not 0 < pid <= (1 << 31) - 1:
                raise ValueError('invalid completed native PID before lock release')
            owned += ' && test ! -d /proc/' + str(pid)
        action = ('rmdir ' + shlex.quote(str(target_lock)) + ' && rm ' + shlex.quote(str(draft_lock / 'owner'))
            + ' && rmdir ' + shlex.quote(str(draft_lock)) + ' && rm ' + shlex.quote(str(_GLOBAL / 'owner'))
            + ' && rmdir ' + shlex.quote(str(_GLOBAL)) + ' && printf \'RELEASED\\n\'')
    else:
        action = 'printf \'OWNED\\n\''
    _, text = hdc.shell('if ' + both_absent + '; then printf \'ABSENT\\n\'; else '
        + owned + ' && ' + action + '; fi', 'release-whole-loop-locks' if release else 'inspect-whole-loop-locks',
        timeout=min(60, hdc.timeout))
    if text not in ({'ABSENT', 'RELEASED'} if release else {'ABSENT', 'OWNED'}):
        raise ValueError('whole-loop lock ownership is foreign, partial, or uncertain')
    return text


def _inventory(raw):
    lines = raw.decode('utf-8').splitlines()
    if not lines or lines[0] != 'QWEN35_BOARD_SERIAL_RETRIEVAL\t1':
        raise ValueError('supervisor retrieval inventory has no supported header')
    rows, seen, total = [], set(), 0
    for line in lines[1:]:
        fields = line.split('\t')
        if len(fields) != 4 or fields[0] != 'file' or _SHA.fullmatch(fields[2]) is None or not fields[3].isdigit():
            raise ValueError('supervisor retrieval inventory row is malformed')
        relative = str(_relative(fields[1])); size = int(fields[3])
        if relative in seen or size > _MAX_FILE or len(rows) >= 16384:
            raise ValueError('supervisor retrieval inventory is duplicate or exceeds bounded file budgets')
        seen.add(relative); total += size
        if total > _MAX_TOTAL:
            raise ValueError('supervisor retrieval inventory exceeds its total byte budget')
        rows.append({'path': relative, 'sha256': fields[2], 'bytes': size})
    if not {'loop.json', 'progress.json'} <= seen:
        raise ValueError('supervisor retrieval inventory omits loop/progress reports')
    return rows


def validate_retrieved_loop(manifest, directory, *, _expected_evidence_kind='device-hiai',
                            _expected_config_sha256=None):
    """Verify actual retained supervisor bytes and the request's native evidence."""
    root = Path(directory)
    inventory_before = _full(root / 'retrieval-manifest.tsv')
    raw, _ = _read_small(root / 'retrieval-manifest.tsv', 8 << 20)
    if _full(root / 'retrieval-manifest.tsv') != inventory_before:
        raise ValueError('retrieval inventory changed before current audit baselines')
    rows = _inventory(raw)
    by_path = {row['path']: row for row in rows}
    snapshots = {}
    for row in rows:
        path = root / row['path']; before = _full(path)
        if before['bytes'] != row['bytes'] or sha256_file(path) != row['sha256'] or _full(path) != before:
            raise ValueError('retrieved supervisor file differs from its final inventory: ' + row['path'])
        snapshots[str(path)] = before
    loop, _ = _json(root / 'loop.json')
    if (loop.get('status') != 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED'
            or loop.get('evidence_kind') != _expected_evidence_kind
            or loop.get('prompt') != manifest['prompt_token_ids']
            or len(loop.get('generated_tokens', [])) != manifest['max_new_tokens']
            or loop.get('full_prefix') != [*manifest['prompt_token_ids'], *loop['generated_tokens']]):
        raise ValueError('supervisor loop is not this complete real request')
    counts = loop.get('counts', {})
    expected_target = len(manifest['prompt_token_ids']) + manifest['max_new_tokens']
    if (counts.get('target_native_calls') != expected_target
            or type(counts.get('draft_native_calls')) is not int
            or counts['draft_native_calls'] < manifest['minimum_draft_rounds']
            or len(loop.get('target_steps', [])) != expected_target
            or len(loop.get('draft_steps', [])) != counts['draft_native_calls']
            or len(loop.get('final_state_files', [])) != 80 or len(loop.get('final_target_outputs', [])) != 82):
        raise ValueError('supervisor did not retain all actual calls, states, and outputs')
    for token in loop['full_prefix']:
        if type(token) is not int or not 0 <= token < 248320:
            raise ValueError('supervisor token is outside the complete vocabulary')
    base = manifest['target_binding']['base_plan']
    expected = {'schema_version': 1, 'artifact_kind': 'qwen35-board-local-serial-dflash',
        'target_model_sha256': manifest['target_model']['sha256'],
        'draft_model_sha256': manifest['draft']['model_source']['sha256'],
        'draft_abi_sha256': manifest['draft']['abi']['sha256'],
        'runtime_version': base['runtime_version'], 'ddk_id': base['ddk_id'], 'firmware_id': base['firmware_id'],
        'state_policy': 'ADVANCE_ONLY_COMMITTED_TOKENS_NO_SPECULATIVE_STATE',
        'control_semantics': 'EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION'}
    if any(loop.get(key) != value for key, value in expected.items()):
        raise ValueError('supervisor model, environment, or scheduling identity differs from preparation')
    if (_SHA.fullmatch(str(loop.get('config_sha256'))) is None
            or _expected_config_sha256 is not None and loop['config_sha256'] != _expected_config_sha256):
        raise ValueError('supervisor did not execute this pinned native config')

    def record(value, *, actual=False):
        if (not isinstance(value, dict) or type(value.get('bytes')) is not int or value['bytes'] <= 0
                or _SHA.fullmatch(str(value.get('sha256'))) is None):
            raise ValueError('invalid supervisor file record')
        name = str(_relative(value.get('path')))
        core = {key: value[key] for key in ('path', 'bytes', 'sha256')}
        if actual and by_path.get(name) != core:
            raise ValueError('native evidence is absent from the verified retrieval inventory: ' + name)
        return core

    def load_record(value):
        row = record(value, actual=True)
        obj, identity = _json(root / row['path'])
        if any(identity[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('native receipt changed after retrieval')
        return obj

    def logs(receipt_record, pid):
        folder = PurePosixPath(receipt_record['path']).parent
        for name in ('tool.stdout', 'tool.stderr', 'tool.pid', 'tool.exitcode', 'command.json'):
            if str(folder / name) not in by_path:
                raise ValueError('native child logs or command are incomplete')
        if ((root / folder / 'tool.pid').read_text() != str(pid) + '\n'
                or (root / folder / 'tool.exitcode').read_text() != '0\n'):
            raise ValueError('native child PID/exit differs from its successful receipt')
        return _json(root / folder / 'command.json')[0]

    mapping = manifest['target_binding']['state_mapping']
    if (len(mapping) != 80 or len({row['source_refdata'] for row in mapping}) != 80
            or len({row['runtime_input_index'] for row in mapping}) != 80
            or len({row['runtime_output_index'] for row in mapping}) != 80):
        raise ValueError('prepared state mapping is not an80-owner bijection')
    previous, target_receipts, child_pids, zeros = None, [], set(), {}
    all_features, _ = _read_small(root / record(loop['final_features'], actual=True)['path'], 4 << 20)
    if len(all_features) != expected_target * 32768:
        raise ValueError('retained final features do not cover every committed target step')
    for position, event in enumerate(loop['target_steps']):
        receipt = load_record(event['receipt']); pid = event.get('pid')
        if (type(pid) is not int or not 0 < pid <= (1 << 31) - 1 or pid in child_pids
                or any(receipt.get(key) != value for key, value in {
                    'schema_version': 1, 'artifact_kind': 'qwen35-board-serial-target-step',
                    'evidence_kind': _expected_evidence_kind, 'status': 'EXECUTED_OUTPUTS_BOUND',
                    'exitcode': 0, 'pid': pid, 'absolute_position': position,
                    'model_sha256': manifest['target_model']['sha256'],
                    'token_id': loop['full_prefix'][position], 'next_token_id': event.get('next_token_id'),
                    'control_semantics': 'EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION'}.items())
                or event.get('token_id') != receipt['token_id']):
            raise ValueError('target native step identity/token order is inconsistent')
        child_pids.add(pid); logs(event['receipt'], pid)
        inputs, outputs = receipt.get('inputs', []), receipt.get('outputs', [])
        if len(inputs) != 89 or len(outputs) != 82:
            raise ValueError('target native receipt omits ordered89/82 IO')
        for index, (value, descriptor) in enumerate(zip(inputs, base['inputs'])):
            record(value)
            if (value.get('index') != index or value.get('name') != descriptor['name']
                    or value['bytes'] != descriptor['bytes'] or value.get('tool_type') != descriptor['tool_input_type']):
                raise ValueError('target native input differs from its named compiled ABI')
        for index, (value, descriptor) in enumerate(zip(outputs, base['outputs'])):
            record(value)
            if (value.get('index') != index or value.get('runtime_name') != descriptor['runtime_tensor_name']
                    or value['bytes'] != descriptor['bytes'] or value.get('tool_type') != descriptor['tool_output_type']):
                raise ValueError('target native output differs from its named compiled ABI')
        feedback = {row.get('source_refdata'): row for row in receipt.get('state_feedback', [])}
        if len(receipt.get('state_feedback', [])) != 80 or len(feedback) != 80:
            raise ValueError('target native state feedback is incomplete or duplicated')
        for owner in mapping:
            row = feedback.get(owner['source_refdata'], {})
            a, b = owner['runtime_input_index'], owner['runtime_output_index']
            if (row.get('runtime_input_index') != a or row.get('runtime_output_index') != b
                    or row.get('input') != inputs[a]):
                raise ValueError('target native feedback differs from the proved state-owner mapping')
            if previous is None:
                size = inputs[a]['bytes']
                if size not in zeros:
                    zeros[size] = hashlib.sha256(bytes(size)).hexdigest()
                if (row.get('initialization') != 'ZERO_FP16' or 'parent_output' in row
                        or inputs[a]['sha256'] != zeros[size]):
                    raise ValueError('target initial state is not the explicit80 zero buffers')
            elif (record(row.get('parent_output')) != record(previous[b])
                    or inputs[a]['sha256'] != previous[b]['sha256'] or inputs[a]['bytes'] != previous[b]['bytes']):
                raise ValueError('target input bytes do not match the exact preceding80 outputs')
        from .original_target_smoke_backend import _controls
        controls = _controls(position)
        for name, width, table in (('arg1_1', 2048, manifest['tables']['weight']),
                                   ('arg0_1', 4, manifest['tables']['scale'])):
            before = _full(table['path'])
            if before != table['identity']:
                raise ValueError('original embedding table changed since whole-request preparation')
            fd = os.open(table['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                if _identity(os.fstat(fd)) != before:
                    raise ValueError('original embedding table changed before row read')
                data = os.pread(fd, width, receipt['token_id'] * width)
                if len(data) != width or _identity(os.fstat(fd)) != before or _full(table['path']) != before:
                    raise ValueError('original embedding table changed during row read')
            finally:
                os.close(fd)
            controls[name] = data
        for value in inputs:
            if value['name'] in controls and hashlib.sha256(controls[value['name']]).hexdigest() != value['sha256']:
                raise ValueError('target input differs from original embedding row or declared query controls')
        feature = all_features[position * 32768:(position + 1) * 32768]
        if outputs[81]['sha256'] != hashlib.sha256(feature).hexdigest():
            raise ValueError('draft cumulative features differ from an actual target output chunk')
        if position and receipt['token_id'] != target_receipts[-1]['next_token_id']:
            if position >= len(manifest['prompt_token_ids']):
                raise ValueError('a committed generated token differs from current target greedy logits')
        previous = outputs; target_receipts.append(receipt)
    if [record(row, actual=True) for row in loop['final_target_outputs']] != [record(row) for row in previous]:
        raise ValueError('final82 retained outputs differ from the final native step')
    final_states = {row.get('source_refdata'): row for row in loop['final_state_files']}
    if len(final_states) != 80:
        raise ValueError('final state owners are duplicated')
    for owner in mapping:
        row = final_states.get(owner['source_refdata'], {})
        if (row.get('runtime_input_index') != owner['runtime_input_index']
                or row.get('runtime_output_index') != owner['runtime_output_index']
                or record(row, actual=True) != record(previous[owner['runtime_output_index']])):
            raise ValueError('final80 retained files differ from the proved state-owner outputs')
    for event in loop['draft_steps']:
        receipt = load_record(event['receipt']); pid = event.get('pid')
        if (type(pid) is not int or not 0 < pid <= (1 << 31) - 1 or pid in child_pids
                or any(receipt.get(key) != value for key, value in {'schema_version': 1,
                    'artifact_kind': 'qwen35-board-serial-draft-step', 'evidence_kind': _expected_evidence_kind,
                    'pid': pid, 'exitcode': 0, 'model_sha256': manifest['draft']['model_source']['sha256']}.items())):
            raise ValueError('draft native step identity is inconsistent')
        child_pids.add(pid); logs(event['receipt'], pid)
        prefix = receipt.get('prefix', [])
        if not 1 <= len(prefix) <= 64 or prefix != loop['full_prefix'][:len(prefix)]:
            raise ValueError('draft did not receive the complete ordered committed prefix')
        raw_feature = all_features[:len(prefix) * 32768]
        if receipt.get('feature_prefix_sha256') != hashlib.sha256(raw_feature).hexdigest():
            raise ValueError('draft logical feature prefix differs from actual target history')
        native = load_record(receipt['native_receipt'])
        if (native.get('status') != 'EXECUTED' or native.get('evidence_kind') != _expected_evidence_kind
                or native.get('process_id') != pid or native.get('target_calls') != 0 or native.get('draft_calls') != 1
                or native.get('model_sha256') != manifest['draft']['model_source']['sha256']
                or native.get('draft_abi_sha256') != manifest['draft']['abi']['sha256']
                or native.get('runtime_version') != base['runtime_version'] or native.get('sequence_length') != 64
                or native.get('ddk_id') != base['ddk_id'] or native.get('firmware_id') != base['firmware_id']
                or native.get('model_load_api') != 'legacy-buffer'):
            raise ValueError('draft native receipt does not bind this exact full model call')
        expected_inputs = [raw_feature + bytes((64 - len(prefix)) * 32768),
            struct.pack('<64i', *(prefix + [0] * (64 - len(prefix)))),
            struct.pack('<64i', *([1] * len(prefix) + [0] * (64 - len(prefix))))]
        if len(receipt.get('inputs', [])) != 3 or len(native.get('inputs', [])) != 3:
            raise ValueError('draft complete-prefix input evidence is incomplete')
        for index, data in enumerate(expected_inputs):
            row = receipt['inputs'][index]; actual = native['inputs'][index]
            if (row.get('index') != index or row.get('bytes') != len(data)
                    or row.get('sha256') != hashlib.sha256(data).hexdigest()
                    or any(actual.get(key) != row[key] for key in ('index', 'bytes', 'sha256'))):
                raise ValueError('draft native inputs differ from the full right-padded actual prefix')
        diagnostic = native.get('io_diagnostic', {})
        if (diagnostic.get('enabled') is not True or diagnostic.get('process_called') is not True
                or diagnostic.get('output_fill_byte') != 0
                or diagnostic.get('input_readback_point') != 'AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS'
                or len(diagnostic.get('inputs', [])) != 3 or len(diagnostic.get('outputs', [])) != 2):
            raise ValueError('draft native input buffer readback evidence is incomplete')
        for index, data in enumerate(expected_inputs):
            readback = diagnostic['inputs'][index]
            if (readback.get('index') != index or readback.get('readback_matches_source') is not True
                    or readback.get('readback_sha256') != hashlib.sha256(data).hexdigest()):
                raise ValueError('draft actual buffer readback differs from full-prefix input bytes')
        for index, value in enumerate(diagnostic['outputs']):
            if (value.get('index') != index or value.get('checked_after_process') is not True
                    or value.get('object_same') is not True or value.get('buffer_same') is not True):
                raise ValueError('draft output buffers changed through native Process')
        step_path = str(PurePosixPath(event['receipt']['path']).parent / 'bundle/step.tsv')
        if step_path not in by_path or native.get('step_plan_sha256') != by_path[step_path]['sha256']:
            raise ValueError('draft native step recipe differs from the actual retained invocation')
        native_folder = PurePosixPath(receipt['native_receipt']['path']).parent
        top = native_folder / 'output_0000.bin'; hidden = native_folder / 'output_0001.bin'
        if str(top) not in by_path or str(hidden) not in by_path:
            raise ValueError('draft actual two output files are missing')
        if len(native.get('outputs', [])) != 2:
            raise ValueError('draft native output evidence is incomplete')
        for index, name in enumerate((top, hidden)):
            row = native['outputs'][index]; retained = by_path[str(name)]
            if (row.get('index') != index or row.get('path') != name.name
                    or any(row.get(key) != retained[key] for key in ('sha256', 'bytes'))):
                raise ValueError('draft actual output file differs from native result bytes')
        proposals = list(struct.unpack('<15i', (root / top).read_bytes()))
        if (proposals != event.get('proposals') or proposals != receipt.get('proposals')
                or any(not 0 <= token < 248320 for token in proposals)):
            raise ValueError('draft proposals differ from actual native Top1 bytes')
    if len(loop.get('commits', [])) != manifest['max_new_tokens']:
        raise ValueError('generated-token commit evidence is incomplete')
    for index, row in enumerate(loop['commits']):
        position = len(manifest['prompt_token_ids']) + index
        if (row.get('position') != position or row.get('target_step_index') != position
                or row.get('token_id') != loop['generated_tokens'][index]
                or row.get('role') != target_receipts[position]['role']):
            raise ValueError('generated-token commits differ from actual target calls')
    rounds = loop.get('rounds', [])
    if len(rounds) != counts['draft_native_calls']:
        raise ValueError('draft rounds differ from actual native proposal calls')
    committed = len(manifest['prompt_token_ids']) + 1
    if loop['commits'][0]['role'] != 'initial_target_next_anchor':
        raise ValueError('first generated token is not the eager target-next anchor')
    totals = dict(considered=0, accepted=0, corrections=0, bonuses=0)
    for number, (round_row, draft_event) in enumerate(zip(rounds, loop['draft_steps'])):
        draft = load_record(draft_event['receipt'])
        prefix_before = loop['full_prefix'][:committed]
        proposals = draft_event['proposals']
        if (round_row.get('round') != number or round_row.get('prefix_before') != prefix_before
                or draft['prefix'] != prefix_before or round_row.get('proposals') != proposals
                or round_row.get('anchor_position') != committed - 1
                or round_row.get('anchor_token_id') != prefix_before[-1]):
            raise ValueError('draft round prefix/anchor differs from current committed state')
        limit = min(manifest['max_proposals_per_round'], expected_target - committed)
        used = matched = 0; mismatch = False; new = []
        for index in range(limit):
            token = target_receipts[committed - 1]['next_token_id']
            okay = proposals[index] == token
            role = 'accepted_proposal' if okay else 'target_correction'
            if target_receipts[committed]['role'] != role or loop['full_prefix'][committed] != token:
                raise ValueError('round acceptance/correction differs from actual target greedy output')
            new.append(token); committed += 1; used += 1
            totals['considered'] += 1
            if okay:
                matched += 1; totals['accepted'] += 1
            else:
                mismatch = True; totals['corrections'] += 1
                if round_row.get('mismatch_index') != index or round_row.get('correction_token_id') != token:
                    raise ValueError('round mismatch evidence is inconsistent')
                break
        if not mismatch and used == limit and committed < expected_target:
            token = target_receipts[committed - 1]['next_token_id']
            if (target_receipts[committed]['role'] != 'all_considered_accepted_bonus'
                    or loop['full_prefix'][committed] != token or round_row.get('bonus_token_id') != token):
                raise ValueError('round bonus differs from actual target greedy output')
            new.append(token); committed += 1; totals['bonuses'] += 1
        if (round_row.get('considered') != used or round_row.get('accepted') != matched
                or round_row.get('discarded_unverified') != 15 - used or round_row.get('committed') != new
                or round_row.get('prefix_after') != loop['full_prefix'][:committed]):
            raise ValueError('round accounting differs from actual committed token evidence')
    if committed != expected_target or any(counts.get(key) != value for key, value in totals.items()):
        raise ValueError('loop totals or considered-only acceptance accounting is inconsistent')
    supervisor = loop.get('process_id')
    if type(supervisor) is not int or not 0 < supervisor <= (1 << 31) - 1 or supervisor in child_pids:
        raise ValueError('native supervisor PID is missing or aliases a child')
    for path, before in snapshots.items():
        if _full(path) != before:
            raise ValueError('retrieved native evidence changed during the final audit')
    if _full(root / 'retrieval-manifest.tsv') != inventory_before:
        raise ValueError('retrieval inventory changed during the final audit')
    return loop


def _validate_completed_failure(manifest, directory, state):
    """Known failed execution is recoverable evidence, never a successful loop."""
    root = Path(directory); loop, _ = _json(root / 'loop.json'); progress, _ = _json(root / 'progress.json')
    base = manifest['target_binding']['base_plan']
    expected = {'status': 'FAILED', 'evidence_kind': 'device-hiai', 'config_sha256': state['config_sha256'],
        'target_model_sha256': manifest['target_model']['sha256'],
        'draft_model_sha256': manifest['draft']['model_source']['sha256'],
        'draft_abi_sha256': manifest['draft']['abi']['sha256'],
        'runtime_version': base['runtime_version'], 'ddk_id': base['ddk_id'], 'firmware_id': base['firmware_id']}
    if any(loop.get(key) != value for key, value in expected.items()):
        raise ValueError('failed supervisor lacks the complete bound model/config identity; locks retained')
    raw, _ = _read_small(root / 'retrieval-manifest.tsv', 8 << 20)
    rows = _inventory(raw); names = {row['path'] for row in rows}; pids = []; exits = {}
    for row in rows:
        path = root / row['path']
        if _full(path)['bytes'] != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError('failed supervisor evidence differs from its retained inventory')
        if PurePosixPath(row['path']).name != 'tool.pid':
            continue
        exit_path = str(PurePosixPath(row['path']).with_name('tool.exitcode'))
        if exit_path not in names:
            raise ValueError('a failed supervisor child has no known exit; locks retained')
        pid_text, exit_text = path.read_text(), (root / exit_path).read_text()
        if (re.fullmatch(r'[1-9][0-9]{0,9}\n', pid_text) is None or int(pid_text) > (1 << 31) - 1
                or re.fullmatch(r'(0|[1-9][0-9]{0,2})\n', exit_text) is None or int(exit_text) > 255):
            raise ValueError('failed supervisor child PID/exit evidence is malformed')
        pids.append(int(pid_text))
        if int(pid_text) in exits:
            raise ValueError('failed supervisor reused a child PID ambiguously')
        exits[int(pid_text)] = int(exit_text)
    child_present = any(key in progress for key in ('child_status', 'last_child_pid', 'last_child_exitcode'))
    counts = loop.get('counts', {})
    counted = counts.get('target_native_calls', 0) + counts.get('draft_native_calls', 0)
    if (progress.get('status') != 'FAILED' or progress.get('supervisor_pid') != loop.get('process_id')
            or counted != len(pids)
            or child_present and (progress.get('child_status') != 'EXITED'
                or progress.get('last_child_pid') not in exits
                or type(progress.get('last_child_exitcode')) is not int
                or exits[progress['last_child_pid']] != progress['last_child_exitcode'])
            or loop.get('failed_child_pid') is not None and (loop['failed_child_pid'] not in exits
                or loop.get('failed_child_exitcode') != exits[loop['failed_child_pid']])):
        raise ValueError('failed supervisor has an active or uncertain final child; locks retained')
    supervisor = loop.get('process_id')
    if type(supervisor) is not int or not 0 < supervisor <= (1 << 31) - 1 or supervisor in pids:
        raise ValueError('failed supervisor PID is unverified')
    return loop, [supervisor, *pids]


def _retrieve(hdc, out, state, manifest):
    board = PurePosixPath(state['board_attempt'])
    state['stage'] = 'retrieve-completed-loop'
    print('board-serial: supervisor completed; retrieving retained outputs and receipts once', flush=True)
    for name in _ATTEMPT_LOGS:
        if _receive(hdc, board, out, name) is None:
            raise ValueError('completed supervisor attempt is missing ' + name)
    if (out / 'run.sh').read_text() != state['launcher']:
        raise ValueError('retained launcher differs from this one recorded launch')
    if (out / 'tool.pid').read_text() != str(state['completion']['launcher_process_id']) + '\n' or (
            out / 'tool.exitcode').read_text() != str(state['completion']['remote_exitcode']) + '\n':
        raise ValueError('retrieved completion files differ from the read-only process proof')
    remote_results = board / 'results'; local_results = out / 'results'
    if _receive(hdc, remote_results, local_results, 'retrieval-manifest.tsv') is None:
        raise ValueError('completed supervisor has no retrieval inventory')
    raw, inventory_id = _read_small(local_results / 'retrieval-manifest.tsv', 8 << 20)
    rows = _inventory(raw)
    # Independently compare the entire declared final inventory on the board.
    checks = []
    for row in rows:
        path = remote_results / row['path']
        checks.append(_regular_guard(path, size=row['bytes']) + ' && serial_hash=$(sha256sum '
            + shlex.quote(str(path)) + ') && serial_hash=${serial_hash%% *} && test "$serial_hash" = '
            + shlex.quote(row['sha256']))
    for batch in _batches(checks):
        hdc.shell(' && '.join(batch), 'verify-final-loop-inventory', timeout=hdc.timeout)
    for row in rows:
        if _receive(hdc, remote_results, local_results, row['path']) != row['sha256']:
            raise ValueError('retrieved loop file SHA differs from the completed supervisor inventory')
    if state['completion']['remote_exitcode'] == 0:
        loop = validate_retrieved_loop(manifest, local_results, _expected_config_sha256=state['config_sha256'])
        completed = [loop['process_id'], *[row['pid'] for row in loop['target_steps']],
            *[row['pid'] for row in loop['draft_steps']]]
    else:
        loop, completed = _validate_completed_failure(manifest, local_results, state)
    if _full(local_results / 'retrieval-manifest.tsv')['bytes'] != inventory_id['bytes'] or sha256_file(
            local_results / 'retrieval-manifest.tsv') != inventory_id['sha256']:
        raise ValueError('retrieval inventory changed during verification')
    state.update(loop_report=str(local_results / 'loop.json'), loop=loop,
        retrieval_inventory=inventory_id,
        device_execution='EXECUTED_OUTPUTS_RETRIEVED' if state['completion']['remote_exitcode'] == 0 else 'FAILED_OUTPUTS_RETRIEVED',
        completed_native_pids=completed)
    if state['completion']['remote_exitcode'] == 0:
        state.update(counts=loop['counts'], generated_tokens=loop['generated_tokens'])
    return loop


def _wait(hdc, out, state, timeout_seconds, poll_seconds):
    started = time.monotonic()
    while True:
        completion = _completion(hdc, state)
        if completion is not None:
            _checkpoint(out, state, 'remote-completed', completion=completion, uncertain=False)
            return completion
        if time.monotonic() - started >= timeout_seconds:
            _checkpoint(out, state, 'running-detached', uncertain=True)
            return None
        # This is coarse request-level polling, never an inter-token trigger.
        time.sleep(min(poll_seconds, max(.001, timeout_seconds - (time.monotonic() - started))))


def execute_board_serial(prepared_dir, output_dir, *, timeout_seconds=7200, lock_wait_seconds=1800,
                         poll_seconds=15, detach=False):
    """Stage once, detach once, and retrieve once; disconnects retain the journal."""
    from .board_serial_dflash import validate_prepared, make_config
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or type(lock_wait_seconds) is not int or lock_wait_seconds < 0
            or type(poll_seconds) not in {int, float} or not 0 < poll_seconds <= 60 or type(detach) is not bool):
        raise ValueError('positive finite whole-request timeout/poll and nonnegative lock wait required')
    prepared, out = Path(prepared_dir).resolve(), output_path(output_dir)
    if prepared == out or prepared in out.parents:
        raise ValueError('execution must be outside immutable preparation')
    out.mkdir(parents=True, exist_ok=False)
    state = {'schema_version': 1, 'artifact_kind': _KIND, 'status': 'FAILED', 'prepared': str(prepared),
        'output': str(out), 'device_execution': 'NOT_RUN', 'model_compilation': 'NOT_RUN',
        'numerical_validation': 'NOT_EVALUATED', 'state_continuity': 'BYTE_CHAIN_ONLY_EXPERIMENTAL_CONTROLS',
        'board_lock_token': uuid.uuid4().hex, 'global_lock': str(_GLOBAL), 'global_held': False,
        'target_lock_held': False, 'draft_lock_held': False, 'launch_requested': False, 'uncertain': False,
        'automatic_model_retry': 'FORBIDDEN', 'intertoken_hdc_calls': 0}
    hdc = None
    try:
        manifest = validate_prepared(prepared)
        _, prep_id = _json(prepared / 'manifest.json')
        base = manifest['target_binding']['base_plan']
        executable, serial, server = _environment()
        token = state['board_lock_token']; digest = base['compilation_report_sha256']
        target = _remote(manifest['board_model_root']) / ('codex-target-' + digest[:20])
        draft = PurePosixPath('/data/local/tmp/qwen35_real_draft_' + manifest['draft']['model_source']['sha256'])
        table_digest = hashlib.sha256((manifest['tables']['weight']['sha256'] + manifest['tables']['scale']['sha256']).encode()).hexdigest()
        tables = PurePosixPath('/data/local/tmp/qwen35_embedding_' + table_digest)
        board = PurePosixPath('/data/local/tmp/qwen35_board_serial_' + token)
        state.update(prepared_manifest=prep_id, serial=serial, hdc_server=server, hdc_executable=executable,
            board_attempt=str(board), target_cache=str(target), draft_cache=str(draft), tables_cache=str(tables),
            compilation_report_sha256=digest, draft_model_sha256=manifest['draft']['model_source']['sha256'],
            tables_cache_sha256=table_digest, runtime_version=base['runtime_version'], firmware_id=base['firmware_id'])
        _checkpoint(out, state, 'host-preflight')
        print('board-serial: building current native supervisor and draft runner; existing OMCs are reused', flush=True)
        uploads, search, deployment = _build_payloads(out, timeout_seconds)
        state['libraries'] = deployment
        hdc = Hdc(executable, serial, server, out, timeout_seconds)
        tool = _remote(base['tool_binding']['path'])
        preflight = ('command -v sha256sum >/dev/null && command -v nohup >/dev/null && command -v find >/dev/null && '
            + _regular_guard(tool) + ' && serial_hash=$(sha256sum ' + shlex.quote(str(tool))
            + ') && serial_hash=${serial_hash%% *} && test "$serial_hash" = ' + shlex.quote(base['tool_binding']['sha256'])
            + ' && param get const.ohos.fullname')
        preflight += ' && printf \'TOOL_BYTES=\' && wc -c < ' + shlex.quote(str(tool))
        _, firmware = hdc.shell(preflight, 'whole-loop-device-preflight')
        observed = firmware.splitlines()
        if (len(observed) != 2 or observed[0].strip() != base['firmware_id'].strip()
                or re.fullmatch(r'TOOL_BYTES=[1-9][0-9]*', observed[1]) is None):
            raise ValueError('board tool or firmware differs from the immutable decode proof')
        _mutate(hdc, out, state, 'mkdir ' + shlex.quote(str(board)), 'create-whole-loop-attempt')
        from .target_model_io_probe import _acquire_global
        print('board-serial: waiting for the shared board model lock', flush=True)
        _checkpoint(out, state, 'acquire-global-requested', uncertain=True)
        _acquire_global(hdc, _GLOBAL, token, lock_wait_seconds)
        _checkpoint(out, state, 'acquire-global-completed', global_held=True, uncertain=False)
        _owned_directory(hdc, out, state, target, '.compilation-sha256', digest, 'own-target-cache')
        _mutate(hdc, out, state, 'mkdir ' + shlex.quote(str(target / '.execution-lock'))
            + ' && printf \'%s\\n\' ' + shlex.quote(token) + ' > ' + shlex.quote(str(board / 'target-lock.owner')),
            'acquire-target-cache-lock')
        _checkpoint(out, state, 'target-cache-locked', target_lock_held=True)
        _owned_directory(hdc, out, state, draft, '.model-sha256', state['draft_model_sha256'], 'own-draft-cache')
        _checkpoint(out, state, 'acquire-draft-cache-lock-requested', uncertain=True)
        _acquire_global(hdc, draft / '.execution-lock', token, 0)
        _checkpoint(out, state, 'draft-cache-locked', draft_lock_held=True, uncertain=False)
        _owned_directory(hdc, out, state, tables, '.tables-sha256', table_digest, 'own-embedding-cache')
        rows = _static_rows(manifest, target, draft, tables)
        for cache in (target, draft, tables):
            _directories(hdc, out, state, cache, [PurePosixPath(row['remote']).parent for row in rows
                if cache in PurePosixPath(row['remote']).parents], 'create-static-cache-directories')
        _checkpoint(out, state, 'inspect-static-cache-requested', uncertain=True)
        print('board-serial: checking target, draft and embedding cache SHA256 once for the whole request', flush=True)
        decisions = _cache_inventory(hdc, rows)
        _checkpoint(out, state, 'static-cache-inspected', uncertain=False)
        _upload_cached(hdc, out, state, rows, decisions, manifest)
        target_model = next(row['remote'] for row in rows if row['kind'] == 'target' and row['record'] == manifest['target_compiled']['omc'])
        draft_model = draft / 'model/model.omc'; attempt_model = board / 'draft-model.omc'
        _mutate(hdc, out, state, _regular_guard(draft_model, size=manifest['draft']['model_source']['bytes'])
            + ' && ln ' + shlex.quote(str(draft_model)) + ' ' + shlex.quote(str(attempt_model))
            + ' && test ' + shlex.quote(str(draft_model)) + ' -ef ' + shlex.quote(str(attempt_model)), 'link-cached-draft-model')
        abi = manifest['draft']['abi']; uploads.append({'local': abi['path'], 'remote': 'draft-abi.tsv',
            'bytes': abi['bytes'], 'sha256': abi['sha256']})
        library_paths = [str(board / 'lib'), *search]
        state['supervisor_library_path'] = ':'.join(library_paths)
        remote_paths = {'attempt': str(board), 'target_model': target_model,
            'target_weight_dir': str(PurePosixPath(target_model).parent), 'target_tool': str(tool),
            'draft_model': str(attempt_model), 'draft_runner': str(board / 'draft-runner'),
            'draft_runner_record': {key: uploads[0][key] for key in ('bytes', 'sha256')},
            'target_tool_record': {'sha256': base['tool_binding']['sha256'], 'bytes': int(observed[1].split('=')[1])},
            'board_runner': str(board / 'board-serial'), 'weight_table': str(tables / 'weight.bin'),
            'scale_table': str(tables / 'scale.bin'), 'draft_abi': str(board / 'draft-abi.tsv'),
            'config': str(board / 'config.tsv'), 'results': str(board / 'results'),
            'target_library_path': str(tool.parent), 'draft_library_path': ':'.join(library_paths)}
        config = make_config(manifest, remote_paths)
        if not isinstance(config, bytes):
            raise TypeError('board serial config serializer must return bytes')
        (out / 'config.tsv').write_bytes(config)
        uploads.append(_file_record(out / 'config.tsv', 'config.tsv'))
        state['config_sha256'] = uploads[-1]['sha256']
        state['remote_paths'] = remote_paths
        state['launcher'] = _launcher(state)
        (out / 'run.sh').write_text(state['launcher'])
        uploads.append(_file_record(out / 'run.sh', 'run.sh'))
        _directories(hdc, out, state, board, [board / 'lib', *[(board / row['remote']).parent for row in uploads
            if PurePosixPath(row['remote']).parent != PurePosixPath('.')]], 'create-whole-loop-upload-directories')
        _checkpoint(out, state, 'upload-runtime-requested', uncertain=True, runtime_uploads=uploads)
        for entry in uploads:
            _check_local(entry)
            _checkpoint(out, state, 'upload-runtime-requested', uncertain=True, pending_runtime_payload=entry)
            hdc.send(entry['local'], board / entry['remote'])
        checksum = out / 'payloads.sha256'
        checksum.write_text(''.join(entry['sha256'] + '  ' + entry['remote'] + '\n' for entry in uploads))
        _checkpoint(out, state, 'upload-runtime-requested', uncertain=True, pending_runtime_payload={'path': str(checksum)})
        hdc.send(checksum, board / checksum.name, 'send-checksums')
        hdc.shell('cd ' + shlex.quote(str(board)) + ' && sha256sum -c ' + shlex.quote(checksum.name), 'verify-checksums')
        _checkpoint(out, state, 'runtime-uploaded', uncertain=False)
        _mutate(hdc, out, state, 'chmod 755 ' + shlex.quote(str(board / 'draft-runner'))
            + ' ' + shlex.quote(str(board / 'board-serial')) + ' ' + shlex.quote(str(board / 'run.sh')), 'set-runtime-permissions')
        if validate_prepared(prepared) != manifest:
            raise ValueError('immutable preparation changed before the one launch')
        command = ('cd ' + shlex.quote(str(board)) + ' && test ! -e tool.pid && test ! -L tool.pid'
            + ' && test ! -e tool.exitcode && test ! -L tool.exitcode && test ! -e launch.guard && test ! -L launch.guard'
            + ' && { nohup sh ' + shlex.quote(str(board / 'run.sh'))
            + ' </dev/null > launcher.stdout 2> launcher.stderr & printf \'LAUNCHED\\n\'; }')
        _checkpoint(out, state, 'launch-requested', launch_requested=True, uncertain=True,
            device_execution='ATTEMPTED', launch_command=command)
        print('board-serial: starting detached autonomous board loop; host polling does not drive tokens', flush=True)
        _, launch_text = hdc.shell(command, 'launch-detached-board-serial', timeout=min(30, hdc.timeout))
        if launch_text != 'LAUNCHED':
            raise ValueError('detached launch lacks the exact acknowledgement; completion remains uncertain')
        _checkpoint(out, state, 'launch-acknowledged', uncertain=False)
        completion = None if detach else _wait(hdc, out, state, timeout_seconds, poll_seconds)
        if completion is None:
            state.update(status='BOARD_SERIAL_DFLASH_STARTED' if detach else 'BOARD_SERIAL_RUNNING_OR_UNCERTAIN',
                stage='running-detached', device_execution='RUNNING_OR_COMPLETION_PENDING')
        else:
            _lock_checks(hdc, state)
            _retrieve(hdc, out, state, manifest)
            _checkpoint(out, state, 'release-locks-requested', uncertain=True)
            state['lock_release'] = _lock_checks(hdc, state, release=True)
            state.update(status='BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED' if completion['remote_exitcode'] == 0 else 'FAILED', stage='complete',
                global_held=False, target_lock_held=False, draft_lock_held=False, uncertain=False)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stage=state.get('phase', 'preflight'))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            state['interruption'] = 'LOCAL_CONTROLLER_INTERRUPTED; NO_REMOTE_CANCELLATION_OR_RELAUNCH'
    finally:
        state['lock_state'] = ('RETAINED_OR_UNCERTAIN' if state['global_held'] or state['target_lock_held']
            or state['draft_lock_held'] or state['uncertain'] else 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT')
        state['hdc_invocations'] = hdc.count if hdc else 0
        state['next'] = ('Recover this exact execution directory; never relaunch or delete uncertain locks'
            if state['launch_requested'] and state['status'] != 'BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED'
            else 'Retain complete native evidence; numerical accuracy and performance remain unvalidated')
        _checkpoint(out, state, state.get('phase', 'failed'))
        save_json(out / 'result.json', state)
    return state


def recover_board_serial(execution_dir, output_dir, *, timeout_seconds=60, poll_seconds=15):
    """Read/poll/retrieve the identical detached request; never launch a model."""
    from .board_serial_dflash import validate_prepared
    if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or type(poll_seconds) not in {int, float} or not 0 < poll_seconds <= 60):
        raise ValueError('positive finite request recovery timeout and bounded poll interval required')
    execution, _ = _safe(execution_dir, directory=True)
    out = output_path(output_dir)
    if execution == out or execution in out.parents:
        raise ValueError('recovery output must be a separate new directory')
    out.mkdir(parents=True, exist_ok=False)
    state, identity = _json(execution / 'checkpoint.json')
    state = dict(state, recovery_source=str(execution), recovery_source_identity=identity, output=str(out),
        status='FAILED', device_execution='NOT_RUN_BY_RECOVERY', recovery_model_execution='FORBIDDEN')
    hdc = None
    try:
        if (state.get('artifact_kind') != _KIND or _TOKEN.fullmatch(str(state.get('board_lock_token'))) is None
                or not state.get('launch_requested') or state.get('global_lock') != str(_GLOBAL)
                or state.get('board_attempt') != '/data/local/tmp/qwen35_board_serial_' + state['board_lock_token']):
            raise ValueError('recovery requires this exact journaled detached launch and ownership token')
        for key in ('target_cache', 'draft_cache', 'tables_cache', 'board_attempt'):
            _remote(state[key])
        if (state['target_cache'].split('/')[-1] != 'codex-target-' + state['compilation_report_sha256'][:20]
                or state['draft_cache'] != '/data/local/tmp/qwen35_real_draft_' + state['draft_model_sha256']
                or state['launcher'] != _launcher(state)):
            raise ValueError('recovery journal cache/model/launcher identities are inconsistent')
        manifest = validate_prepared(state['prepared'])
        base = manifest['target_binding']['base_plan']
        if (state['compilation_report_sha256'] != base['compilation_report_sha256']
                or state['draft_model_sha256'] != manifest['draft']['model_source']['sha256']
                or state['runtime_version'] != base['runtime_version'] or state['firmware_id'] != base['firmware_id']):
            raise ValueError('recovery checkpoint differs from the exact prepared model and environment identities')
        _, current = _json(Path(state['prepared']) / 'manifest.json')
        if current != state['prepared_manifest']:
            raise ValueError('immutable preparation changed after the original launch')
        executable, serial, server = _environment()
        if (serial, server) != (state['serial'], state['hdc_server']):
            raise ValueError('recovery must use the same device and HDC server')
        hdc = Hdc(executable, serial, server, out, timeout_seconds)
        _checkpoint(out, state, 'recover-read-only')
        print('board-serial: recovering the same retained attempt; checking completion without relaunch', flush=True)
        _lock_checks(hdc, state)
        completion = _wait(hdc, out, state, timeout_seconds, poll_seconds)
        if completion is None:
            state.update(status='BOARD_SERIAL_RUNNING_OR_UNCERTAIN', stage='running-detached')
        else:
            _retrieve(hdc, out, state, manifest)
            _checkpoint(out, state, 'release-locks-requested', uncertain=True)
            state['lock_release'] = _lock_checks(hdc, state, release=True)
            state.update(status='BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED' if completion['remote_exitcode'] == 0 else 'FAILED', stage='complete',
                global_held=False, target_lock_held=False, draft_lock_held=False, uncertain=False)
    except BaseException as error:
        state.update(error=f'{type(error).__name__}: {error}', stage=state.get('phase', 'recovery-preflight'))
    finally:
        state['lock_state'] = ('RETAINED_OR_UNCERTAIN' if state.get('global_held') or state.get('target_lock_held')
            or state.get('draft_lock_held') or state.get('uncertain') else 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT')
        state['hdc_invocations'] = hdc.count if hdc else 0
        _checkpoint(out, state, state.get('phase', 'recovery-failed'))
        save_json(out / 'result.json', state)
    return state

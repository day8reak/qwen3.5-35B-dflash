"""Recover one completed continuation's retained files without model execution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex

from .artifacts import output_path, save_json
from .hdc_run import Hdc
from . import original_decode_row_transfer as json_io
from .original_target_board import _board_path, _hash_lines
from .target_compile_failure import _relative, _safe
from .target_decode_continuation_run import inspect_continuation_outputs
from .target_decode_output_inventory import _descriptors
from .target_prefill_output_inventory import _hash_retrieved, _logged_slots
from .target_prefill_smoke_plan import _identity


_KIND = 'qwen35-target-decode-continuation-recovery'
_GLOBAL = PurePosixPath('/data/local/tmp/.qwen35-target-model-io-lock')
_LOGS = ('tool.stdout', 'tool.stderr', 'tool.exitcode', 'tool.pid')
_MAX_LOG = 8 << 20


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _pin(path, snapshots, expected=None):
    path, stamp = _safe(path)
    full = _identity(path.stat())
    if [full[k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
        raise ValueError('recovery source changed before current snapshot')
    if expected is not None and (full != expected if isinstance(expected, dict) else stamp != expected):
        raise ValueError('recovery source differs from its fresh reader identity: ' + str(path))
    if str(path) in snapshots and snapshots[str(path)] != full:
        raise ValueError('recovery source changed within this invocation: ' + str(path))
    snapshots[str(path)] = full


def _json(path, snapshots, identities, maximum=32 << 20):
    _pin(path, snapshots)
    value, identity = json_io._json(path, maximum)
    _pin(path, snapshots, identity['stat'])
    identities.append(identity)
    return value


def _stable(snapshots):
    for path, identity in snapshots.items():
        if _identity(_safe(path)[0].stat()) != identity:
            raise ValueError('recovery source changed within this invocation: ' + path)


def _framed_call(board, label, expected_command, serial, server, snapshots, identities):
    paths = list((board / 'calls').glob('*-' + label + '.json'))
    if len(paths) != 1:
        raise ValueError('recovery requires exactly one original completed HDC call: ' + label)
    record = _json(paths[0], snapshots, identities)
    argv = record.get('argv')
    if (record.get('label') != label or type(record.get('returncode')) is not int or record['returncode'] != 0
            or record.get('error') or not isinstance(argv, list) or len(argv) != 7
            or argv[1:6] != ['-t', serial, '-s', server, 'shell']):
        raise ValueError('original HDC call did not complete on the bound device/server: ' + label)
    shell = shlex.split(argv[6])
    if len(shell) != 3 or shell[:2] != ['sh', '-c']:
        raise ValueError('original HDC shell wrapper is malformed')
    markers = re.findall(r'__QWEN35_HDC_[0-9a-f]{32}__', shell[2])
    if len(markers) != 1:
        raise ValueError('original HDC command lacks a unique completion marker')
    marker = markers[0]
    expected = '( ' + expected_command + '\n); rc=$?; printf \'\\n' + marker + '=%s\\n\' "$rc"; exit "$rc"'
    if shell[2] != expected:
        raise ValueError('original HDC command differs from this exact continuation attempt: ' + label)
    stdout = record.get('stdout')
    if not isinstance(stdout, str):
        raise ValueError('original HDC call has no raw stdout')
    matches = list(re.finditer(r'^' + re.escape(marker) + r'=([0-9]+)\r?$', stdout, re.MULTILINE))
    if len(matches) != 1 or matches[0][1] != '0' or stdout[matches[0].end():].strip():
        raise ValueError('original HDC call lacks one final successful remote marker: ' + label)
    return stdout[:matches[0].start()].rstrip('\r\n')


def _load_failed(execution, snapshots, identities):
    from .target_decode_continuation import validate_decode_continuation
    root, _ = _safe(execution, directory=True)
    outer = _json(root / 'result.json', snapshots, identities)
    if (outer.get('artifact_kind') != 'qwen35-target-decode-continuation-job' or outer.get('status') != 'FAILED'
            or outer.get('mode') != 'APPLY' or outer.get('output') != str(root)
            or outer.get('board_result') != str(root / 'board/result.json')
            or outer.get('plan') != str(root / 'plan.json')):
        raise ValueError('expected the retained failed continuation apply job')
    plan = validate_decode_continuation(outer['prepared'])
    for source in plan['source_inputs']:
        _pin(source['path'], snapshots, source['identity'])
    for state in plan['state_feedback_bindings']:
        _pin(state['output_file']['path'], snapshots, state['output_file']['stat'])
    _pin(plan['primary_model_binding']['path'], snapshots, plan['primary_model_binding']['identity'])
    board = root / 'board'
    observed = _json(board / 'result.json', snapshots, identities)
    saved_plan = _json(root / 'plan.json', snapshots, identities)
    proof = _json(board / 'runtime-continuation-proof.json', snapshots, identities)
    if not _same(plan, saved_plan) or not _same(plan, proof):
        raise ValueError('failed execution proof differs from the freshly validated continuation preparation')
    if (observed.get('artifact_kind') != 'qwen35-original-target-tool-execution'
            or observed.get('status') != 'FAILED' or observed.get('stage') != 'retrieve-outputs'
            or type(observed.get('remote_exitcode')) is not int or observed['remote_exitcode'] != 0
            or observed.get('runtime_continuation_prepared') != outer['prepared']
            or observed.get('source_preservation') != 'UNCHANGED_CONTINUATION_PROOF_AND_ALL89_PREPARED_INPUT_SHA256'):
        raise ValueError('recovery requires completed inference with a retrieval-only failure and preserved original inputs')
    serial, server = observed.get('serial'), observed.get('hdc_server')
    if (not isinstance(serial, str) or not serial or serial.startswith('-')
            or not isinstance(server, str) or re.fullmatch(r'[^\s:]+:[0-9]+', server) is None
            or not 0 < int(server.rsplit(':', 1)[1]) < 65536
            or any(os.environ.get(key) not in {None, expected} for key, expected in (('SN', serial), ('BIP', server)))):
        raise ValueError('recovery must use the same recorded device and HDC server')
    attempt, cache = _board_path(observed['board_attempt']), _board_path(observed['board_model_cache'])
    digest = plan['compilation_report_sha256']
    if (attempt.parent != cache or re.fullmatch(r'run-[0-9a-f]{32}', attempt.name) is None
            or cache.name != 'codex-target-' + digest[:20]
            or observed.get('board_global_lock') != str(_GLOBAL)
            or observed.get('remote_execution_lock') != str(cache / '.execution-lock')
            or re.fullmatch(r'[0-9a-f]{32}', str(observed.get('board_lock_token'))) is None):
        raise ValueError('failed attempt/cache/lock paths or ownership token are inconsistent')
    transport_plan = _json(board / 'plan.json', snapshots, identities)
    evidence = _json(board / 'input-evidence.json', snapshots, identities)
    remote_inputs = [dict(row, host_source=row['source'], source=str(attempt / 'source-inputs' / row['source_filename']))
        for row in plan['inputs']]
    expected_transport = dict(plan, inputs=remote_inputs, host_input_root=plan['input_dir'], host_inputs=[
        {'path': Path(row['path']).name, 'bytes': row['bytes'], 'sha256': row['sha256'],
         'stat': [row['identity'][key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')]}
        for row in plan['source_inputs']])
    # Original transport preserves first runtime-slot encounter order.
    by_name = {row['path']: row for row in expected_transport['host_inputs']}
    expected_transport['host_inputs'] = [by_name[row['source_filename']] for row in plan['inputs']]
    if (not _same(transport_plan, expected_transport) or not _same(evidence, {'inputs': remote_inputs})
            or not _same(observed.get('inputs'), remote_inputs)
            or observed.get('plan_sha256') != identities[-2]['sha256']):
        raise ValueError('failed board plan/input-evidence differs from the89 proved runtime input byte hashes')
    primary_relative = Path(plan['primary_model_binding']['path']).relative_to(Path(plan['compilation']) / 'compile-work/generated')
    primary = cache / 'payload' / primary_relative.as_posix()
    payloads = [row for row in observed.get('model_payloads', []) if row.get('path') == str(primary)]
    if (len(payloads) != 1 or payloads[0].get('sha256') != plan['model_sha256']
            or payloads[0].get('bytes') != plan['primary_model_binding']['bytes']):
        raise ValueError('failed execution does not bind the identical compiled primary model')
    command = _json(board / 'command.json', snapshots, identities)
    argv = [plan['tool_binding']['path'], '--model=' + str(primary),
        '--input=' + ','.join(str(attempt / row['file']) for row in plan['inputs']), '--times=1',
        '--modelmanager_itf=V2', '--weight_dir=' + str(primary.parent), '--enable_item=1', '--direct_from_file=true',
        '--input_type=' + ','.join(row['tool_input_type'] for row in plan['inputs']),
        '--output_type=' + ','.join(row['tool_output_type'] for row in plan['outputs']), '--output_dir=' + str(attempt / 'outputs')]
    if not _same(command.get('argv'), argv):
        raise ValueError('retained tool command differs from the exact single completed continuation')
    execute = ('cd ' + shlex.quote(str(attempt)) + ' && { LD_LIBRARY_PATH=' + shlex.quote(str(PurePosixPath(argv[0]).parent))
        + ' ' + shlex.join(argv) + ' > tool.stdout 2> tool.stderr & tool_pid=$!; printf \'%s\\n\' "$tool_pid" > tool.pid; '
        + 'wait "$tool_pid"; rc=$?; printf \'%s\\n\' "$rc" > tool.exitcode; exit "$rc"; }')
    _framed_call(board, 'execute-original-target', execute, serial, server, snapshots, identities)
    inventory_command = 'find ' + shlex.quote(str(attempt / 'outputs')) + ' -type f -exec sha256sum {} \\;'
    original_inventory = _framed_call(board, 'output-inventory', inventory_command, serial, server, snapshots, identities)
    hashes = _hash_lines(original_inventory, attempt / 'outputs')
    if len(hashes) != 82:
        raise ValueError('original successful output inventory must contain exactly82 distinct SHA256 entries')
    cache_acquired = _framed_call(board, 'acquire-execution-lock', 'mkdir ' + shlex.quote(str(cache / '.execution-lock')),
        serial, server, snapshots, identities)
    if cache_acquired:
        raise ValueError('original execution-lock acquisition had unexpected output')
    for name, row in observed.get('retrieved', {}).items():
        if name.startswith('outputs/') and hashes.get(name[len('outputs/'):]) != row.get('sha256'):
            raise ValueError('partial old retrieval disagrees with the original full output inventory')
    _stable(snapshots)
    return root, outer, plan, observed, attempt, cache, hashes, serial, server


def _remote_checks(hdc, attempt, cache, digest, token, label):
    # No new lock is acquired. Only the original owner may be released later.
    safe = [cache, attempt, attempt / 'outputs']
    command = ' && '.join('test -d ' + shlex.quote(str(path)) + ' && test ! -L ' + shlex.quote(str(path)) for path in safe)
    owner = shlex.quote(str(cache / '.compilation-sha256'))
    command += (' && test -f ' + owner + ' && test ! -L ' + owner
        + ' && test "$(wc -c < ' + owner + ')" -eq 65 && read -r recovery_digest < ' + owner
        + ' && test "$recovery_digest" = ' + shlex.quote(digest))
    for filename in ('tool.exitcode', 'tool.pid'):
        path = shlex.quote(str(attempt / filename))
        command += ' && test -f ' + path + ' && test ! -L ' + path + ' && test "$(wc -c < ' + path + ')" -le 16'
    command += (' && printf \'EXIT=\' && cat ' + shlex.quote(str(attempt / 'tool.exitcode'))
        + ' && printf \'PID=\' && cat ' + shlex.quote(str(attempt / 'tool.pid'))
        + ' && read -r recovery_pid < ' + shlex.quote(str(attempt / 'tool.pid'))
        + ' && if test -d "/proc/$recovery_pid"; then printf \'PROCESS_PRESENT\\n\'; else printf \'PROCESS_ABSENT\\n\'; fi')
    command += (' && if test -e ' + shlex.quote(str(_GLOBAL)) + ' || test -L ' + shlex.quote(str(_GLOBAL)) + '; then test -d ' + shlex.quote(str(_GLOBAL))
        + ' && test ! -L ' + shlex.quote(str(_GLOBAL)) + ' && test -f ' + shlex.quote(str(_GLOBAL / 'owner'))
        + ' && test ! -L ' + shlex.quote(str(_GLOBAL / 'owner')) + ' && test "$(wc -c < ' + shlex.quote(str(_GLOBAL / 'owner'))
        + ')" -eq 33 && printf \'GLOBAL=\' && cat ' + shlex.quote(str(_GLOBAL / 'owner'))
        + '; else printf \'GLOBAL=ABSENT\\n\'; fi'
        + ' && if test -e ' + shlex.quote(str(cache / '.execution-lock')) + ' || test -L ' + shlex.quote(str(cache / '.execution-lock')) + '; then test -d ' + shlex.quote(str(cache / '.execution-lock'))
        + ' && test ! -L ' + shlex.quote(str(cache / '.execution-lock')) + ' && printf \'CACHE=PRESENT\\n\'; else printf \'CACHE=ABSENT\\n\'; fi')
    _, text = hdc.shell(command, label)
    match = re.fullmatch(r'EXIT=0\r?\nPID=([1-9][0-9]{0,9})\r?\nPROCESS_ABSENT\r?\nGLOBAL=([0-9a-f]{32}|ABSENT)\r?\nCACHE=(PRESENT|ABSENT)', text)
    if not match or match[2] not in {token, 'ABSENT'} or match[2] == 'ABSENT' and match[3] != 'ABSENT':
        raise ValueError('remote child exit/PID or retained original lock ownership is unsafe or uncertain: ' + repr(text))
    return {'pid': int(match[1]), 'global': match[2], 'cache': match[3]}


def _inventory(hdc, attempt, original_hashes, plan):
    outputs = attempt / 'outputs'
    _, listed = hdc.shell('find ' + shlex.quote(str(outputs)) + ' -type l', 'recovery-output-symlinks')
    if listed:
        raise ValueError('retained output tree contains a symlink')
    _, listed = hdc.shell('find ' + shlex.quote(str(outputs)) + ' -type f', 'recovery-output-files')
    paths = [_board_path(line.rstrip('\r')) for line in listed.splitlines()]
    expected = {str(outputs / relative) for relative in original_hashes}
    if len(paths) != 82 or len(set(paths)) != 82 or {str(path) for path in paths} != expected:
        raise ValueError('retained output file set differs from the original completed82-file inventory')
    all_paths = sorted(paths) + [attempt / name for name in _LOGS]
    command = ('for recovery_file in ' + ' '.join(shlex.quote(str(path)) for path in all_paths)
        + '; do test -f "$recovery_file" && test ! -L "$recovery_file" || exit 1; '
        + 'printf \'%s %s\\n\' "$(wc -c < "$recovery_file")" "$recovery_file"; done')
    _, text = hdc.shell(command, 'recovery-output-extents')
    extents = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(0|[1-9][0-9]*) (/[A-Za-z0-9_./-]+)\r?', line)
        if not match or match[2] in extents:
            raise ValueError('retained output extent response is malformed or duplicated')
        extents[match[2]] = int(match[1])
    if set(extents) != {str(path) for path in all_paths}:
        raise ValueError('remote extents do not cover exactly82 outputs and4 logs')
    wanted = sorted(row['bytes'] for row in plan['outputs'])
    if sorted(extents[str(path)] for path in paths) != wanted or any(extents[str(attempt / name)] > _MAX_LOG for name in _LOGS):
        raise ValueError('retained output/log byte extents exceed the frozen descriptor or log budgets')
    _, text = hdc.shell('sha256sum ' + ' '.join(shlex.quote(str(path)) for path in all_paths), 'recovery-output-sha256')
    hashes = _hash_lines(text, attempt)
    if len(hashes) != 86 or any(hashes.get('outputs/' + name) != digest for name, digest in original_hashes.items()):
        raise ValueError('fresh remote SHA256 differs from the original completed output inventory')
    return {name: {'sha256': digest, 'bytes': extents[str(attempt / name)]} for name, digest in hashes.items()}


def _release(hdc, attempt, cache, digest, token, state):
    if state['global'] == 'ABSENT':
        return {'status': 'BOTH_LOCKS_ALREADY_ABSENT', 'board_mutation': 'NONE'}
    guard = ('test -d ' + shlex.quote(str(cache)) + ' && test ! -L ' + shlex.quote(str(cache))
        + ' && test -d ' + shlex.quote(str(attempt)) + ' && test ! -L ' + shlex.quote(str(attempt))
        + ' && test -d ' + shlex.quote(str(_GLOBAL)) + ' && test ! -L ' + shlex.quote(str(_GLOBAL))
        + ' && test -f ' + shlex.quote(str(_GLOBAL / 'owner')) + ' && test ! -L ' + shlex.quote(str(_GLOBAL / 'owner'))
        + ' && test "$(wc -c < ' + shlex.quote(str(_GLOBAL / 'owner')) + ')" -eq 33'
        + ' && read -r recovery_owner < ' + shlex.quote(str(_GLOBAL / 'owner'))
        + ' && test "$recovery_owner" = ' + shlex.quote(token)
        + ' && test "$(find ' + shlex.quote(str(_GLOBAL)) + ' | wc -l)" -eq 2'
        + ' && test "$(find ' + shlex.quote(str(_GLOBAL)) + ' -type f | wc -l)" -eq 1'
        + ' && test "$(find ' + shlex.quote(str(_GLOBAL)) + ' -type d | wc -l)" -eq 1'
        + ' && test -z "$(find ' + shlex.quote(str(_GLOBAL)) + ' -type l)"'
        + ' && test ! -L ' + shlex.quote(str(cache / '.compilation-sha256'))
        + ' && test "$(wc -c < ' + shlex.quote(str(cache / '.compilation-sha256')) + ')" -eq 65'
        + ' && read -r recovery_digest < ' + shlex.quote(str(cache / '.compilation-sha256'))
        + ' && test "$recovery_digest" = ' + shlex.quote(digest)
        + ' && test -f ' + shlex.quote(str(attempt / 'tool.exitcode'))
        + ' && test ! -L ' + shlex.quote(str(attempt / 'tool.exitcode'))
        + ' && test "$(wc -c < ' + shlex.quote(str(attempt / 'tool.exitcode')) + ')" -eq 2'
        + ' && read -r recovery_exit < ' + shlex.quote(str(attempt / 'tool.exitcode')) + ' && test "$recovery_exit" = 0'
        + ' && test -f ' + shlex.quote(str(attempt / 'tool.pid')) + ' && test ! -L ' + shlex.quote(str(attempt / 'tool.pid'))
        + ' && test "$(wc -c < ' + shlex.quote(str(attempt / 'tool.pid')) + ')" -le 16'
        + ' && read -r recovery_pid < ' + shlex.quote(str(attempt / 'tool.pid'))
        + ' && test "$recovery_pid" = ' + str(state['pid']) + ' && test ! -d /proc/' + str(state['pid']))
    if state['cache'] == 'PRESENT':
        guard += ' && test -d ' + shlex.quote(str(cache / '.execution-lock')) + ' && test ! -L ' + shlex.quote(str(cache / '.execution-lock'))
        guard += ' && rmdir ' + shlex.quote(str(cache / '.execution-lock'))
    else:
        guard += ' && test ! -e ' + shlex.quote(str(cache / '.execution-lock')) + ' && test ! -L ' + shlex.quote(str(cache / '.execution-lock'))
    # Both removals are guarded by the original owner; never recurse or signal.
    guard += ' && rm ' + shlex.quote(str(_GLOBAL / 'owner')) + ' && rmdir ' + shlex.quote(str(_GLOBAL))
    hdc.shell(guard, 'recovery-release-original-locks')
    return {'status': 'ORIGINAL_OWNER_LOCKS_RELEASED_AFTER_COMPLETE_RETRIEVAL', 'board_mutation': 'ORIGINAL_OWNED_EMPTY_LOCKS_ONLY'}


def recover_decode_continuation(execution, output, *, timeout_seconds=300):
    out = output_path(output); out.mkdir(parents=True, exist_ok=False)
    board_out = out / 'board'; board_out.mkdir()
    result = {'schema_version': 1, 'artifact_kind': _KIND, 'status': 'FAILED', 'stage': 'validate-failed-execution',
        'output': str(out), 'execution': str(Path(execution).absolute()), 'device_execution': 'NOT_RUN_RECOVERY_ONLY',
        'model_compilation': 'NOT_RUN', 'model_upload': 'NOT_RUN', 'input_upload': 'NOT_RUN', 'model_inference': 'NOT_RUN',
        'numerical_validation': 'NOT_RUN_NO_REFERENCE', 'state_continuity': 'NOT_VALIDATED',
        'lock_state': 'RETAINED_OR_UNCERTAIN', 'retrieval_errors': [], 'retrieved': {}}
    snapshots, identities, hdc = {}, [], None
    try:
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200:
            raise ValueError('recovery timeout_seconds must be an integer in1..7200')
        root, outer, plan, failed, attempt, cache, original_hashes, serial, server = _load_failed(execution, snapshots, identities)
        result.update(prepared=outer['prepared'], board_attempt=str(attempt), board_model_cache=str(cache),
            serial=serial, hdc_server=server, recovered_original_pid=failed.get('tool_process_id'))
        save_json(out / 'plan.json', plan)
        save_json(out / 'source-evidence.json', {'source_identities': identities,
            'original_output_sha256': original_hashes, 'source_current_identities': snapshots})
        hdc = Hdc(os.environ.get('HDC', 'hdc'), serial, server, board_out, timeout_seconds)
        result['stage'] = 'verify-same-board-tool-and-firmware'
        _, handshake = hdc.shell('sha256sum ' + shlex.quote(plan['tool_binding']['path'])
            + ' && param get const.ohos.fullname', 'recovery-device-handshake')
        lines = handshake.splitlines()
        if (len(lines) != 2 or lines[0] != plan['tool_binding']['sha256'] + '  ' + plan['tool_binding']['path']
                or lines[1].strip() != plan['firmware_id'].strip()):
            raise ValueError('recovery board tool or firmware differs from the original runtime proof')
        result['stage'] = 'verify-completed-original-attempt' 
        state = _remote_checks(hdc, attempt, cache, plan['compilation_report_sha256'], failed['board_lock_token'], 'recovery-check-original-attempt')
        if failed.get('tool_process_id') is not None and failed['tool_process_id'] != state['pid']:
            raise ValueError('fresh remote PID differs from the original execution receipt')
        result['tool_process_id'] = state['pid']
        inventory = _inventory(hdc, attempt, original_hashes, plan)
        result['stage'] = 'retrieve-original-outputs'
        # Logs first establish actual slot paths before any large output receive.
        for name in [*_LOGS, *sorted(key for key in inventory if key.startswith('outputs/'))]:
            if name.startswith('outputs/') and not any(key.startswith('outputs/') for key in result['retrieved']):
                raw = (board_out / 'tool.stdout').read_bytes()
                mapped = _logged_slots(raw, failed, _descriptors(plan), [dict(row, path=key) for key, row in inventory.items() if key.startswith('outputs/')])
                if mapped['status'] != 'BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG' or mapped.get('unlogged_retrieved_files'):
                    raise ValueError('original stdout does not explicitly bind all82 retained output slots')
            local = board_out / name
            local.parent.mkdir(parents=True, exist_ok=True)
            hdc.receive(attempt / name, local)
            identity, raw = _hash_retrieved(local, inventory[name], capture=name in _LOGS)
            if name == 'tool.exitcode' and raw != b'0\n':
                raise ValueError('recovered exit-code file is not the completed original zero exit')
            if name == 'tool.pid' and raw.strip() != str(state['pid']).encode():
                raise ValueError('recovered PID file differs from the same original attempt')
            result['retrieved'][name] = {key: identity[key] for key in ('sha256', 'bytes')}
            _pin(local, snapshots, identity['stat'])
        if not _same(inventory, _inventory(hdc, attempt, original_hashes, plan)):
            raise ValueError('remote outputs or logs changed during recovery')
        final = _remote_checks(hdc, attempt, cache, plan['compilation_report_sha256'], failed['board_lock_token'], 'recovery-recheck-original-attempt')
        if final != state:
            raise ValueError('remote completion or lock ownership changed during recovery')
        from .target_decode_continuation import validate_decode_continuation
        if not _same(validate_decode_continuation(outer['prepared']), plan):
            raise ValueError('continuation preparation changed during recovery')
        _stable(snapshots)
        observed = dict(failed, status='EXECUTED_UNVALIDATED', stage='recovered-original-outputs',
            device_execution='EXECUTED_OUTPUTS_RETRIEVED', remote_exitcode=0, tool_process_id=state['pid'],
            retrieval_errors=[], retrieved=result['retrieved'], lock_state=('OWNED_COMPLETED_LOCKS_RETAINED_FOR_RECOVERY' if state['global'] != 'ABSENT' else 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'),
            recovery_only='NO_MODEL_RERUN_NO_INPUT_OR_WEIGHT_UPLOAD', original_failed_execution=str(root),
            source_preservation='UNCHANGED_CONTINUATION_PROOF_AND_ALL89_PREPARED_INPUT_SHA256')
        observed.pop('error', None); observed.pop('evidence_error', None); observed.pop('lock_release_error', None)
        if state['global'] != 'ABSENT':
            observed['recovery_completion'] = {'exitcode': 0, 'pid': state['pid'], 'pid_absent': True,
                'original_owner_token': failed['board_lock_token'], 'global_owner_verified': True,
                'cache_owner_verified': True, 'compilation_report_sha256': plan['compilation_report_sha256'],
                'board_attempt': str(attempt)}
        chain = inspect_continuation_outputs(plan, board_out, observed,
            allow_retained_completed_lock=state['global'] != 'ABSENT')
        if not _same(validate_decode_continuation(outer['prepared']), plan):
            raise ValueError('continuation preparation changed during recovered byte-chain inspection')
        _stable(snapshots)
        result['stage'] = 'release-original-owned-locks'
        result['lock_recovery'] = _release(hdc, attempt, cache, plan['compilation_report_sha256'], failed['board_lock_token'], state)
        released = _remote_checks(hdc, attempt, cache, plan['compilation_report_sha256'], failed['board_lock_token'], 'recovery-confirm-locks-absent')
        if released != {**state, 'global': 'ABSENT', 'cache': 'ABSENT'}:
            raise ValueError('original locks were not both absent after recovery')
        observed['lock_state'] = 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'
        result['lock_state'] = observed['lock_state']
        _stable(snapshots)
        save_json(board_out / 'result.json', observed)
        save_json(out / 'state-chain.json', chain)
        result.update(status='DECODE_CONTINUATION_OUTPUTS_RECOVERED_UNVALIDATED', stage='complete',
            lock_state=observed['lock_state'], board_result=str(board_out / 'result.json'), state_chain=str(out / 'state-chain.json'),
            state_feedback=chain['status'], state_continuity=chain['state_continuity'],
            feedback_input_count=chain['feedback_input_count'], bound_child_output_count=chain['bound_child_output_count'],
            logits=chain['bound_tensors']['logits']['statistics'], feature=chain['bound_tensors']['feature']['statistics'])
    except Exception as error:
        result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    result['hdc_invocations'] = hdc.count if hdc else 0
    save_json(out / 'result.json', result)
    return result

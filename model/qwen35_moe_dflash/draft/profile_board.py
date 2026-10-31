"""One detached vendor-tool Draft profile, followed by retrieval without rerun.

Uses the user-observed internal CLI, not an invented HIAI maintenance ABI.
Profiling data exported by the vendor may include historical service records;
these remain separate by source CSV and are never silently attributed to Draft.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import uuid

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.execution.serial_transport import _environment
from qwen35_moe_dflash.draft.board import _receive
from qwen35_moe_dflash.draft.profile_plan import KIND, bind_profile, model_argv, bind_accuracy_profile, check_model_help
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.hdc_run import Hdc
from qwen35_moe_dflash.target.board_tool_inspect import _shell_result, _tool_hash, _tool_path
from qwen35_moe_dflash.target.source_inventory import _read_small
from qwen35_moe_dflash.execution.lock import _acquire_global, _release_global


GLOBAL_LOCK = PurePosixPath('/data/local/tmp/.qwen35-target-model-io-lock')
DEFAULT_PARSER = '/data/local/tmp/tools_sysdbg/data_proc_tool_internal'
PREFIX = '/data/local/tmp/qwen35_draft_profile_'
q = shlex.quote


def _guard(board, token):
    return ('test -d ' + q(str(board)) + ' && test ! -L ' + q(str(board))
        + ' && test ! -L ' + q(str(board / 'owner')) + ' && read -r owner < ' + q(str(board / 'owner'))
        + ' && test "$owner" = ' + q(token))


def _lock_guard(lock, token):
    return _guard(lock, token)


def supervisor(plan):
    board = PurePosixPath(plan['board_attempt'])
    cache = PurePosixPath(plan['board_model_cache'])
    env = 'LD_LIBRARY_PATH=' + q(plan['library_path'])
    command = 'echo $$ > model.pid\nexec env ' + env + ' ' + shlex.join(plan['model_argv'])
    parse = [plan['parser_tool'], '--result_path=' + str(board / 'prof_data')]
    return '\n'.join([
        '#!/bin/sh', 'umask 077', 'cd ' + q(str(board)) + ' || exit 91',
        _guard(board, plan['token']) + ' || exit 92',
        _lock_guard(GLOBAL_LOCK, plan['token']) + ' || exit 93',
        _lock_guard(cache / '.execution-lock', plan['token']) + ' || exit 94',
        'timeout -k 5 ' + str(plan['timeout_seconds']) + ' sh -c ' + q(command)
            + ' > model.stdout 2> model.stderr',
        'model_status=$?', 'printf "%s\\n" "$model_status" > model.exitcode',
        # Post-process only after the model command exited. No second inference.
        'timeout -k 5 180 env ' + env + ' ' + shlex.join(parse) + ' > parser.stdout 2> parser.stderr',
        'parser_status=$?', 'printf "%s\\n" "$parser_status" > parser.exitcode',
        'printf "%s\\n" ' + q(plan['token']) + ' > complete.tmp',
        'mv complete.tmp complete', 'exit 0', ''])


def _identity_guard(path, row):
    value = q(str(path))
    return ('test -f ' + value + ' && test ! -L ' + value
        + ' && test "$(wc -c < ' + value + ')" -eq ' + str(row['bytes'])
        + ' && test "$(sha256sum ' + value + ' | cut -d \' \' -f 1)" = ' + q(row['sha256']))


def start_profile(prepared, inspection, output, *, parser_tool=DEFAULT_PARSER,
                  timeout_seconds=600, lock_wait_seconds=1800, accuracy=None,
                  allow_numerical_failure=False,
                  model_tool="/data/local/tmp/tools_sysdbg/model_run_tool_internal"):
    if allow_numerical_failure and not accuracy:
        raise ValueError('--allow-numerical-failure requires --accuracy')
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200:
        raise ValueError('timeout must be 1..7200 seconds')
    if type(lock_wait_seconds) is not int or not 0 <= lock_wait_seconds <= 7200:
        raise ValueError('lock wait must be 0..7200 seconds')
    parser_tool = _tool_path(parser_tool)
    out = output_path(output)
    source = Path(accuracy or prepared).absolute()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('profile output must be separate from retained prepared inputs')
    from qwen35_moe_dflash.execution.run_layout import reuse_test_output
    reuse_test_output(out)
    out.mkdir(parents=True, exist_ok=False)
    result = {'artifact_kind': KIND, 'status': 'FAILED', 'stage': 'bind-inputs', 'output': str(out),
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN', 'numerical_validation': 'NOT_EVALUATED'}
    hdc = None
    held = []
    launching = False
    acquiring = False
    token = uuid.uuid4().hex
    try:
        plan = (bind_accuracy_profile(accuracy, model_tool,
                    allow_numerical_failure=allow_numerical_failure) if accuracy else
                bind_profile(prepared, inspection, refresh_output=out / 'input-revalidation'))
        result.update({k:plan[k] for k in ('source_numerical_validation','profiling_scope') if k in plan})
        if plan.get('input_revalidation'):
            result['input_revalidation'] = {key: plan['input_revalidation'][key]
                for key in ('status', 'identity_change_count', 'prepared')}
        executable, serial, server = _environment()
        hdc = Hdc(executable, serial, server, out, 60)
        board = PurePosixPath(PREFIX + token)
        cache = PurePosixPath('/data/local/tmp/qwen35_real_draft_' + plan['model']['sha256'])
        library_path = str(PurePosixPath(plan['tool']['path']).parent)
        if PurePosixPath(parser_tool).parent != PurePosixPath(library_path):
            raise ValueError('profile model tool and parser must use the same selected tool directory')
        plan.update(token=token, serial=serial, hdc_server=server, board_attempt=str(board),
            board_model_cache=str(cache), parser_tool=parser_tool, library_path=library_path,
            timeout_seconds=timeout_seconds)
        plan['model_argv'] = model_argv(plan, board, cache / 'model/model.omc')
        result.update(board_attempt=str(board), token=token, global_lock=str(GLOBAL_LOCK), cache_lock=str(cache / '.execution-lock'))
        result['stage'] = 'tool-preflight'
        _, firmware = hdc.shell('param get const.ohos.fullname', 'profile-firmware')
        if firmware.strip() != plan['firmware_id'].strip():
            raise ValueError('board firmware differs from the retained real-feature execution')
        for name, path in (('model', plan['tool']['path']), ('parser', parser_tool)):
            _, text = hdc.shell('test -x ' + q(path) + ' && test ! -L ' + q(path) + ' && sha256sum ' + q(path),
                               'profile-' + name + '-tool-identity')
            digest = _tool_hash(text, path)
            if name == 'model' and plan['tool']['sha256'] is not None and digest != plan['tool']['sha256']:
                raise ValueError('model profiling tool changed since its help inspection')
            plan[name + '_tool_sha256'] = digest
        hdc.shell('command -v timeout >/dev/null && command -v nohup >/dev/null && command -v sha256sum >/dev/null',
                  'profile-shell-tools')
        result['stage'] = 'wait-board-lock'
        print('draft-profile: waiting for exclusive board access; no new model compilation', flush=True)
        acquiring = True
        try:
            _acquire_global(hdc, GLOBAL_LOCK, token, lock_wait_seconds)
        except ValueError as error:
            if 'BOARD_MODEL_IO_BUSY:' in str(error):
                acquiring = False
            raise
        acquiring = False
        held.append(GLOBAL_LOCK)
        result['stage'] = 'cached-model'
        hdc.shell('test -d ' + q(str(cache)) + ' && test ! -L ' + q(str(cache)), 'profile-existing-cache')
        acquiring = True
        try:
            _acquire_global(hdc, cache / '.execution-lock', token, 0)
        except ValueError as error:
            if 'BOARD_MODEL_IO_BUSY:' in str(error):
                acquiring = False
            raise
        acquiring = False
        held.append(cache / '.execution-lock')
        # Require the already deployed identical OMC. Never silently upload 2.8GB.
        hdc.shell(_identity_guard(cache / 'model/model.omc', plan['model']), 'profile-cache-model-sha', timeout=180)
        plan['cache_validation'] = 'FULL_SHA256_MATCHED_CACHE_WARMED_BEFORE_DIAGNOSTIC_LOAD'
        hdc.shell('umask 077 && mkdir ' + q(str(board)) + ' && printf "%s\\n" ' + q(token)
            + ' > ' + q(str(board / 'owner')) + ' && mkdir ' + q(str(board / 'outputs'))
            + ' ' + q(str(board / 'prof_data')) + ' ' + q(str(board / 'help'))
            + ' ' + q(str(board / 'help/model')) + ' ' + q(str(board / 'help/parser')), 'profile-create-attempt')
        if accuracy:
            status, text = _shell_result(hdc, 'cd '+q(str(board/'help/model'))
                +' && timeout -k 1 15 env LD_LIBRARY_PATH='+q(library_path)
                +' '+q(plan['tool']['path'])+' --help', 'profile-model-help', 25)
            save_json(out/'model-help.json', {'returncode':status,'text':text})
            if status >= 128:
                raise ValueError('model tool help failed before inference')
            check_model_help(text)
            plan['tool']['sha256'] = plan['model_tool_sha256']
        result['stage'] = 'parser-help'
        status, text = _shell_result(hdc, 'cd ' + q(str(board / 'help/parser')) + ' && timeout -k 1 15 env LD_LIBRARY_PATH='
            + q(library_path) + ' ' + q(parser_tool) + ' --help', 'profile-parser-help', 25)
        save_json(out / 'parser-help.json', {'returncode': status, 'text': text})
        if status >= 128 or not re.search(r'--result_path\s+Profiling\s*&\s*Dump data output directory', text):
            raise ValueError('parser help did not establish --result_path; see parser-help.json (no model run)')
        plan['parser_help_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        result['stage'] = 'upload-inputs'
        for row in plan['inputs']:
            path = Path(row['path'])
            _, current = _read_small(path, 3 << 20)
            if any(current[k] != row[k] for k in ('bytes', 'sha256')):
                raise ValueError('input changed before transfer: ' + str(path))
            remote = board / f"input_{row['index']}.bin"
            hdc.send(path, remote)
            hdc.shell(_identity_guard(remote, row), 'profile-check-input-' + str(row['index']))
        script = supervisor(plan)
        (out / 'profile.sh').write_text(script, encoding='utf-8')
        plan['supervisor_sha256'] = hashlib.sha256(script.encode()).hexdigest()
        save_json(out / 'request.json', plan)
        result['request_sha256'] = sha256_file(out / 'request.json')
        hdc.send(out / 'profile.sh', board / 'profile.sh')
        hdc.shell(_identity_guard(board / 'profile.sh', {'bytes': len(script.encode()), 'sha256': plan['supervisor_sha256']})
            + ' && printf "%s\\n" ' + q(result['request_sha256']) + ' > ' + q(str(board / 'request.sha256')),
            'profile-check-supervisor')
        result['stage'] = 'launch'
        launching = True
        hdc.shell('cd ' + q(str(board)) + ' && ' + _guard(board, token)
            + ' && { nohup sh ./profile.sh < /dev/null > supervisor.stdout 2> supervisor.stderr & '
            + 'supervisor_pid=$!; printf "%s\\n" "$supervisor_pid" > supervisor.pid; }', 'profile-launch')
        result.update(status='DRAFT_OPERATOR_PROFILE_STARTED', stage='running-detached',
            device_execution='ONE_PROFILED_CALL_RUNNING_OR_COMPLETION_PENDING',
            next='Recover this exact output with --recover; never repeat the model call to retrieve data')
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        if launching:
            result['device_execution'] = 'LAUNCH_UNCERTAIN_RETAINED_NO_AUTOMATIC_RETRY'
    finally:
        if not launching and not acquiring and hdc:
            for lock in reversed(held[:]):
                try:
                    _release_global(hdc, lock, token)
                    held.remove(lock)
                except Exception as error:
                    result['lock_release_error'] = str(error)
                    break
        result['lock_state'] = 'RETAINED_OR_UNCERTAIN' if held or launching or acquiring else 'NO_KNOWN_OWNED_LOCK'
        save_json(out / 'result.json', result)
    return result


def _inventory(text):
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(r'\s*([0-9]+)\s+(\./prof_data/[A-Za-z0-9_./-]+)', line)
        if not match:
            raise ValueError('unrecognized profiling file inventory line: ' + repr(line))
        size, name = int(match[1]), match[2][2:]
        if '..' in PurePosixPath(name).parts or not 0 <= size <= 64 << 20:
            raise ValueError('profiling file path/extent exceeds retrieval contract')
        rows.append({'path': name, 'bytes': size})
    if len(rows) > 1024 or sum(row['bytes'] for row in rows) > 256 << 20 or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('profile inventory is duplicated or exceeds 256 MiB / 1024 files')
    return rows


def recover_profile(execution, output):
    root = Path(execution).absolute()
    out = output_path(output)
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('recovery needs a separate new output directory')
    from qwen35_moe_dflash.execution.run_layout import reuse_test_output
    reuse_test_output(out, recovery_source=root)
    out.mkdir(parents=True, exist_ok=False)
    result = {'artifact_kind': KIND, 'recovery_model_execution':'FORBIDDEN',
        'recovery_source':str(root), 'status': 'FAILED', 'stage': 'recover', 'output': str(out),
        'model_compilation': 'NOT_RUN', 'model_relaunch': 'NONE', 'numerical_validation': 'NOT_EVALUATED',
        'retrieved': {}, 'lock_state': 'RETAINED_OR_UNCERTAIN'}
    hdc = plan = None
    complete = False
    try:
        raw, identity = _read_small(root / 'request.json', 4 << 20)
        plan = json.loads(raw)
        result.update({k:plan[k] for k in ('source_numerical_validation','profiling_scope') if k in plan})
        token = plan['token']
        if (plan.get('artifact_kind') != KIND or re.fullmatch('[0-9a-f]{32}', token) is None
                or plan['board_attempt'] != PREFIX + token
                or re.fullmatch('[0-9a-f]{64}', plan['model']['sha256']) is None
                or plan['board_model_cache'] != '/data/local/tmp/qwen35_real_draft_' + plan['model']['sha256']):
            raise ValueError('invalid retained profiling request paths or identity')
        executable, serial, server = _environment()
        if (serial, server) != (plan['serial'], plan['hdc_server']):
            raise ValueError('recover requires the original selected serial and HDC server')
        hdc = Hdc(executable, serial, server, out, 60)
        board = PurePosixPath(plan['board_attempt'])
        result.update(board_attempt=str(board), request=identity)
        guard = _guard(board, token) + ' && test "$(cat ' + q(str(board / 'request.sha256')) + ')" = ' + q(identity['sha256'])
        _, text = hdc.shell(guard + ' && if test -f ' + q(str(board / 'complete')) + '; then cat '
            + q(str(board / 'complete')) + '; else printf "PENDING\\n"; fi', 'profile-completion')
        if text == 'PENDING':
            result.update(status='DRAFT_OPERATOR_PROFILE_PENDING', stage='running',
                next='Recover the same execution into the same result slot after completion; no model relaunch')
            return result
        if text != token:
            raise ValueError('completion token differs from this retained profile attempt')
        complete = True
        for name in ('model.stdout', 'model.stderr', 'model.pid', 'model.exitcode', 'parser.stdout',
                     'parser.stderr', 'parser.exitcode', 'supervisor.stdout', 'supervisor.stderr'):
            digest = _receive(hdc, board, out, name)
            if digest is None:
                raise ValueError('missing completion evidence: ' + name)
            result['retrieved'][name] = digest
        pid = (out / 'model.pid').read_text().strip()
        if not re.fullmatch('[1-9][0-9]{0,9}', pid):
            raise ValueError('invalid actual model-tool PID')
        result['model_process_id'] = int(pid)
        result['model_exitcode'] = int((out / 'model.exitcode').read_text().strip())
        result['parser_exitcode'] = int((out / 'parser.exitcode').read_text().strip())
        # Only new files below the attempt-owned result_path are retrieved. The
        # vendor may export historical server records: retain that claim gap.
        _, listing = hdc.shell(guard + ' && cd ' + q(str(board))
            + ' && test -z "$(find prof_data -type l -print)" && find ./prof_data -type f -exec wc -c {} \\;',
            'profile-file-inventory')
        inventory = _inventory(listing)
        save_json(out / 'artifact-inventory.json', inventory)
        for row in inventory:
            digest = _receive(hdc, board, out, row['path'])
            if digest is None or (out / row['path']).stat().st_size != row['bytes']:
                raise ValueError('profiling artifact missing or changed extent: ' + row['path'])
            result['retrieved'][row['path']] = digest
        if result['model_exitcode'] != 0 or result['parser_exitcode'] != 0:
            raise ValueError('model or profiling converter failed; raw logs/artifacts retained')
        csvs = [{'path': str(out / row['path']), 'bytes': row['bytes'],
                 'sha256': result['retrieved'][row['path']]}
                for row in inventory if row['path'].lower().endswith('.csv')]
        if not any(row['bytes'] > 0 for row in csvs):
            raise ValueError('no nonempty profiling CSV retrieved; raw logs/artifacts retained')
        # Deliver vendor CSVs unchanged. Host-side schema interpretation is not
        # part of capture success and must not trigger another model invocation.
        result['capture'] = {'request': identity, 'model': plan['model'], 'model_process_id': int(pid),
            'input_scope': plan['input_scope'], 'timing_scope': plan['timing_scope'],
            'attribution_status': 'VENDOR_SERVICE_EXPORT_HISTORY_AND_MODEL_MEMBERSHIP_REQUIRE_REVIEW'}
        result.update(status='DRAFT_PROFILE_CAPTURED_REQUIRES_ATTRIBUTION_REVIEW', stage='complete',
            device_execution='ONE_VENDOR_TOOL_CALL_COMPLETED', csv_directory=str(out / 'prof_data'),
            csv_count=len(csvs), csv_files=csvs, csv_analysis='NOT_REQUESTED_RAW_FILES_ONLY',
            next='Open the original CSVs in csv_directory; model membership requires review. No model rerun needed.')
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if complete and hdc and plan:
            errors = []
            for lock in (PurePosixPath(plan['board_model_cache']) / '.execution-lock', GLOBAL_LOCK):
                try:
                    # Re-recovery after completed retrieval may find no owned
                    # lock or a new owner's lock; never remove the latter.
                    _, value = hdc.shell('if test ! -e ' + q(str(lock)) + '; then printf "ABSENT\\n"; '
                        + 'elif ' + _lock_guard(lock, plan['token']) + '; then printf "OWNED\\n"; else printf "OTHER\\n"; fi',
                        'profile-recovery-lock-owner')
                    if value == 'OWNED':
                        _release_global(hdc, lock, plan['token'])
                    elif value not in ('ABSENT', 'OTHER'):
                        raise ValueError('unrecognized lock response')
                except Exception as error:
                    errors.append(str(error))
            result['lock_state'] = 'RETAINED_OR_UNCERTAIN' if errors else 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
            if errors:
                result['lock_release_errors'] = errors
        save_json(out / 'result.json', result)
    return result

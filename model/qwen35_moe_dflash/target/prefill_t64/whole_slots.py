"""Check the positional files of a completed whole-prefill invocation against HIAI V2."""
import json
import os
from pathlib import Path
import re
import shlex
import uuid

from ...execution.hdc_run import Hdc
from ...execution.lock import _acquire_global, _release_global
from ...execution.serial_transport import _environment
from ...toolchain.hiai_deployment import BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, discover_libraries
from ..decode_smoke_plan import _names
from ..layer_probe.common import save_json, sha
from ..original_target_board import _board_path
from ..prefill_smoke_plan import _TOOL_TYPES, _nd_view_key, _shape


def read(path):
    return json.loads(path.read_text())


def invocation_inputs(script, compiled, remote, model):
    lines = [line for line in script.splitlines() if line.startswith('LD_LIBRARY_PATH=')]
    line, = lines
    library_path, tool, *args = shlex.split(line)
    _board_path(tool)
    def option(key):
        value, = [item[len(key):] for item in args if item.startswith(key)]
        return value

    if option('--model=') != str(model) or option('--modelmanager_itf=') != 'V2':
        raise ValueError('Saved invocation does not use this model with HIAI V2')
    by_path = {str(remote / f"inputs/{s['index']:03d}.bin"): s for s in compiled}
    paths = option('--input=').split(',')
    if len(paths) != len(compiled) or set(paths) != set(by_path):
        raise ValueError('Invocation and compiled input file sets differ')
    rows = [by_path[path] for path in paths]
    if option('--input_type=').split(',') != [_TOOL_TYPES[s['dtype'].upper()] for s in rows]:
        raise ValueError('Saved --input_type differs from the supplied files')
    return rows, _board_path(library_path.split('=', 1)[1])


def compare_inputs(supplied, native):
    by_name = {s['name']: s for s in supplied}
    observed = native['inputs']
    if (len(by_name) != len(supplied) or len(observed) != len(supplied)
            or native['input_count'] != len(observed)):
        raise ValueError('Input counts or compiled Data names disagree')
    try:
        names = _names(native, 'inputs')
    except ValueError as exc:
        return dict(status='UNVERIFIED', reason=str(exc), slots=[])
    if set(names) != set(by_name):
        return dict(status='UNVERIFIED', reason='HIAI and compiled Data names differ',
                    runtime_only=sorted(set(names) - by_name.keys()),
                    compiled_only=sorted(by_name.keys() - set(names)), slots=[])
    rows = []
    for i, (name, desc, sent) in enumerate(zip(names, observed, supplied)):
        if desc['index'] != i:
            raise ValueError('HIAI descriptor indices are out of order')
        wanted = by_name[name]
        key = _nd_view_key((wanted['dtype'], tuple(wanted['shape'])))
        if (desc['raw_format'] != native['format_enum_values']['ND']
                or desc['raw_dtype'] != native['dtype_enum_values'][wanted['dtype'].upper()]
                or _nd_view_key((wanted['dtype'], _shape(desc))) != key):
            status = 'DESCRIPTOR_DIFFERS'
        elif sent['name'] == name:
            status = 'MATCH'
        elif (sent['sha256'] == wanted['sha256']
              and _nd_view_key((sent['dtype'], tuple(sent['shape']))) == key):
            status = 'SAME_INPUT_BYTES'
        else:
            status = 'WRONG_INPUT'
        rows.append(dict(slot=i, supplied=sent['semantic_name'], expected=wanted['semantic_name'],
                         supplied_name=sent['name'], runtime_name=name, status=status))
    kinds = {row['status'] for row in rows}
    if 'WRONG_INPUT' in kinds:
        verdict = 'INPUT_ORDER_MISMATCH'
    elif 'DESCRIPTOR_DIFFERS' in kinds:
        verdict = 'UNVERIFIED'
    elif 'SAME_INPUT_BYTES' in kinds:
        verdict = 'EQUIVALENT_INPUT_BYTES'
    else:
        verdict = 'INPUT_ORDER_MATCH'
    return dict(status=verdict, slots=rows)


def previous_four_replay(run, out, device, server, model_hash, model_name, firmware):
    """Find reusable model files and observed descriptors independently."""
    previous = observed = None
    attempts = run.glob('replay-*/L0-L3/board/attempt.json')
    for path in sorted(attempts, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if path.parent == out:
            continue
        try:
            attempt = read(path)
            remote = attempt['remote']
            if ((attempt['device'], attempt['server'], attempt['omc_sha256']) != (device, server, model_hash)
                    or not re.fullmatch(r'/data/local/tmp/qwen35_prefill40_[a-f0-9]{32}', remote)):
                continue
            if previous is None:
                previous = _board_path(remote)
            if not firmware:
                break
            desc = path.with_name('model-io.json')
            native = read(desc)
            if (native['status'] == 'IO_DESCRIPTORS_OBSERVED' and native['evidence_kind'] == 'device-hiai'
                    and native['run_called'] is False and native['process_called'] is False
                    and native['model_path'] == f'{remote}/payload/{model_name}'
                    and native['model_sha256'] == model_hash and native['firmware_id'] == firmware
                    and native['runtime_version'] == native['expected_runtime_version']):
                observed = desc, native
                break
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return previous, observed


def observe_model_io(hdc, config, env, work, out, model, model_sha, tool_lib):
    """Read V2 descriptors for the uploaded model while the caller holds the board lock."""
    cfg = read(config)
    native_dir = (config.parent / cfg.get('native_directory', 'native')).resolve()
    runner = native_dir / 'build/qwen35_dflash_hiai'
    files = [(runner, 'draft-runner')]
    search = list(BOARD_LIBRARY_SEARCH_PATHS)
    if os.environ.get('BOARD_HIAI_LIB_DIR'):
        search.insert(0, str(_board_path(os.environ['BOARD_HIAI_LIB_DIR'])))
    else:
        for lib in discover_libraries(runner, native_dir)['libraries']:
            if lib.name not in DDK_HCL_EXCLUDED:
                files.append((lib, 'lib/' + lib.name))
    hdc.shell(f'mkdir {work}/lib', 'create-probe-lib')
    for path, name in files:
        hdc.send(path, work/name)
    hdc.shell(f'chmod 700 {work}/draft-runner', 'prepare-probe')
    cmd = ['env', 'LD_LIBRARY_PATH=' + ':'.join([str(tool_lib), str(work / 'lib'), *search]),
           str(work / 'draft-runner'), '--model-io-only', str(model), '--descriptor-stage', 'restore-only',
           '--model-sha256', model_sha, '--expected-runtime-version', env['runtime_version'],
           '--ddk-id', env['ddk_id'], '--firmware-id', env['firmware_id'].strip(), '--output', str(work / 'model-io.json')]
    # Preserve the runner's exit code independently of whether it wrote a diagnostic JSON.
    restore = (f'cd {model.parent} && {{ {shlex.join(cmd)} 2>&1; probe_rc=$?; '
               f'if test -s {work}/model-io.json; then has_json=1; else has_json=0; fi; '
               'printf "\\n__PROBE_RESULT__=%d,%d\\n" "$probe_rc" "$has_json"; }')
    available = None
    deployed = {path.name for path, _ in files}
    while True:
        _, log = hdc.shell(restore, 'restore')
        message, marker, status = log.rpartition('__PROBE_RESULT__=')
        if not marker or not re.fullmatch(r'\d+,[01]', status):
            raise RuntimeError(f'{out}: descriptor probe returned no exit status\n{log}')
        rc, has_json = map(int, status.split(','))
        if has_json:
            break
        failure = f'{out}: descriptor probe exited {rc} without model-io.json\n{message.strip()}'
        # The resident model tool and our separate descriptor binary have different link dependencies.
        missing = set(re.findall(r'^Error loading shared library ([\w.+-]+\.so(?:\.\d+)*): '
                                 r'No such file or directory', message, re.MULTILINE))
        if not rc or not missing or missing & (DDK_HCL_EXCLUDED | deployed):
            raise RuntimeError(failure)
        if available is None:
            try:
                available = {lib.name: lib for lib in discover_libraries(runner, native_dir)['libraries']}
            except (ValueError, OSError) as exc:
                raise RuntimeError(f'{failure}\nCould not resolve dependencies from {native_dir}: {exc}') from exc
        if not missing <= available.keys():
            raise RuntimeError(f'{failure}\nMissing from the probe build: {sorted(missing - available.keys())}')
        for name in sorted(missing):
            path = available[name]
            print(f'Descriptor probe requires {name}; uploading to {work}/lib', flush=True)
            hdc.send(path, work/'lib'/name)
        deployed.update(missing)
    local = out / 'model-io.json'
    hdc.receive(work / local.name, local)
    native = read(local)
    if (rc or native.get('status') != 'IO_DESCRIPTORS_OBSERVED' or native.get('evidence_kind') != 'device-hiai'
            or native.get('model_sha256') != model_sha or native.get('model_path') != str(model)
            or native.get('run_called') is not False or native.get('process_called') is not False):
        raise ValueError(f"Could not observe this model's HIAI descriptors: {native.get('reason', local)}")
    return native


def check_slots(run, config):
    previous = run / 'board'
    attempt = read(previous / 'attempt.json')
    omc = read(run / 'omc/manifest.json')
    primary = omc['primary']
    remote = _board_path(attempt['remote'])
    if (not re.fullmatch(r'/data/local/tmp/qwen35_prefill40_[a-f0-9]{32}', str(remote))
            or primary['sha256'] != attempt['omc_sha256']):
        raise ValueError('Attempt and retained OMC identities disagree')
    model = _board_path(str(remote / 'payload' / Path(primary['path']).name))
    rows, tool_lib = invocation_inputs((previous / 'run.sh').read_text(), omc['io']['inputs'], remote, model)
    receipt = set((previous / 'files.sha256').read_text().splitlines())
    input_checks = [s['sha256'] + f"  inputs/{s['index']:03d}.bin" for s in rows]
    if not set(input_checks) <= receipt:
        raise ValueError('Input hashes differ from the deployment receipt')

    exe, sn, bip = _environment()
    if (sn, bip) != (attempt['device'], attempt['server']):
        raise ValueError('SN/BIP differ from the board that produced these logits')
    cfg = read(config)
    env = read(config.parent / cfg['assets'])['environment']
    token = uuid.uuid4().hex
    out = previous / ('slot-check-' + token)
    out.mkdir()
    print(f'Slot diagnostic: {out}', flush=True)
    work = remote / out.name
    hdc = Hdc(exe, sn, bip, out, 180)
    _, fw = hdc.shell('param get const.ohos.fullname', 'firmware')
    if fw.strip() != env['firmware_id'].strip():
        raise ValueError('Firmware differs from runtime configuration')
    lock = remote.parent / '.qwen35-target-model-io-lock'
    _acquire_global(hdc, lock, token, 0)
    try:
        hdc.shell(f'test "$(cat {remote}/tool.exitcode)" = 0 && mkdir {work}', 'retained-run')
        checks = out/'checks.sha256'
        checks.write_text('\n'.join([*input_checks, sha(previous/'run.sh') + '  run.sh']) + '\n')
        hdc.send(checks, work/checks.name)
        hdc.shell(f'cd {remote} && sha256sum -c {work}/{checks.name}', 'verify')
        native = observe_model_io(hdc, config, env, work, out, model, primary['sha256'], tool_lib)
    finally:
        _release_global(hdc, lock, token)
    report = compare_inputs(rows, native)
    save_json(out / 'slots.json', report)
    for s in report['slots']:
        if s['status'] != 'MATCH' or not re.fullmatch(r'l\d+_state[01]', s['supplied']):
            print(f"slot={s['slot']:2d} supplied={s['supplied']} expected={s['expected']} {s['status']}")
    print(report['status'], report.get('reason', ''), out / 'slots.json', flush=True)
    return report

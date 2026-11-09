#!/usr/bin/env python3
"""Prepare, compile, execute and compare isolated Target GDN/CacheUpdate cases.

This command DOES access the selected X90 board. Resume retrieves the same
pending request and never relaunches it. Runtime configuration is reused only
for the SDK, firmware identity and shared native runner, not its model paths.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
import validate_draft_vector as device
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from qwen35_moe_dflash.target.attention_validation import KIND, MODULES


def announce(family, stage):
    print(f'target-operator [{family}] {stage}', file=sys.stderr, flush=True)


def finish(root, state):
    rows = list(state['families'].values())
    state['all_operators_passed'] = False
    if state.get('status') != 'PENDING':
        state['status'] = ('VALIDATION_COMPLETE' if len(rows) == len(state['request']['families'])
            and all(r.get('status') in ('PASSED', 'METRICS_RECORDED') for r in rows) else 'PARTIAL')
    _save(root/'result.json', state)
    return state


def run(root, request, *, wait_seconds=1800, poll_seconds=10):
    state = read_json(root/'result.json') if (root/'result.json').is_file() else dict(
        artifact_kind=KIND, status='RUNNING', families={}, request=request)
    if state.get('artifact_kind') != KIND or state.get('request') != request:
        raise ValueError('resume request differs from the retained run')
    state['status'] = 'RUNNING'
    config = Path(request['config'])
    if sha256_file(config) != request['config_sha256']:
        raise ValueError('runtime config changed since this validation began')
    cfg = read_json(config)
    assets = (config.parent/cfg['assets']).resolve()
    if sha256_file(assets) != request['assets_sha256']:
        raise ValueError('runtime environment assets changed since this validation began')
    env = read_json(assets)['environment']
    for family in request['families']:
        if state['families'].get(family, {}).get('finished'):
            continue
        folder = root/family
        folder.mkdir(exist_ok=True)
        probe = Path(request['reuse'].get(family) or folder/'probe')
        execution = folder/'execution'
        row = state['families'][family] = dict(status='RUNNING', stage='prepare',
            model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN')
        _save(root/'result.json', state)
        try:
            # A retained request always takes precedence over compilation/start.
            if not (execution/'request.json').is_file():
                if execution.exists():
                    raise ValueError('previous preparation/launch failed; retain it and choose a new --output')
                module = MODULES[family]
                if not probe.exists():
                    announce(family, 'generating deterministic inputs and independent reference')
                    module.prepare(probe)
                else:
                    module.load_prepared(probe)
                row['stage'] = 'compile'
                if not (probe/'compile.json').is_file():
                    announce(family, 'compiling isolated t64/t1 models')
                    module.compile_prepared(probe, ge_root=request['ge_root'],
                        ddk_root=request['ddk_root'], ddk_id=env['ddk_id'],
                        cxx=request['cxx'], cxx11_abi=request['cxx11_abi'])
                compiled = read_json(probe/'compile.json')
                row['model_compilation'] = compiled.get('status', 'FAILED')
                if compiled.get('status') not in ('COMPILED', 'PARTIAL'):
                    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
                    row['compiler_diagnostics'] = {name: compilation_diagnostics(probe/'models'/name/'omc')
                        for name in ('t64', 't1')}
                    raise ValueError('no compiled case; see '+str(probe/'compile.json'))
                row['stage'] = 'execute'
                announce(family, 'binding compiled IO, deploying and executing on X90')
                launched = device.start(SimpleNamespace(config=str(config), compiled=str(probe),
                    ddk_root=request['ddk_root'], cases=None, apply=True,
                    output_initialization=request['output_initialization']), execution)
                row['launch'] = launched
                if launched['status'] != 'VECTOR_VALIDATION_STARTED':
                    raise RuntimeError(launched.get('error', launched['status']))
            row['stage'] = 'recover'
            row['device_execution'] = 'RUNNING_OR_COMPLETION_PENDING'
            _save(root/'result.json', state)
            deadline = time.monotonic()+wait_seconds
            while True:
                # Separate immutable receipt per poll; no evidence deletion.
                ordinal = 1
                while (folder/f'recovery-{ordinal:04d}').exists():
                    ordinal += 1
                recovery = folder/f'recovery-{ordinal:04d}'
                recovered = device.recover(execution, recovery)
                row['recovery'] = str(recovery/'result.json')
                if recovered['status'] != 'VECTOR_VALIDATION_PENDING':
                    row.update(recovered, stage='complete')
                    row['finished'] = recovered.get('lock_state') == 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
                    announce(family, 'comparison '+recovered['status']+'; '+str(recovery/'result.json'))
                    if not row['finished']:
                        state['status'] = 'PENDING'
                        row['stage'] = 'recovery-required'
                        return finish(root, state)
                    break
                row['board_progress'] = recovered.get('board_progress')
                if time.monotonic() >= deadline:
                    state['status'] = 'PENDING'
                    row['status'] = 'PENDING'
                    announce(family, 'still pending; use --resume with this same --output')
                    return finish(root, state)
                announce(family, 'waiting for snapshots: '+json.dumps(row['board_progress'], ensure_ascii=False))
                _save(root/'result.json', state)
                time.sleep(min(poll_seconds, max(0, deadline-time.monotonic())))
        except Exception as error:
            row.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            announce(family, row['error'])
            # Never proceed to another model after an uncertain device launch.
            if (execution/'request.json').exists():
                state['status'] = 'PENDING'
                row['stage'] = 'recovery-required'
                return finish(root, state)
            row['finished'] = True
        _save(root/'result.json', state)
    return finish(root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config', help='existing runtime/runtime.json')
    parser.add_argument('--families', nargs='+', choices=tuple(MODULES), default=['gdn', 'cache'])
    parser.add_argument('--reuse-gdn', help='existing prepared/compiled GDN probe directory')
    parser.add_argument('--reuse-cache', help='existing prepared/compiled CacheUpdate probe directory')
    parser.add_argument('--ge-root', default=os.environ.get('CANN_ROOT'))
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--cxx11-abi', type=int, choices=(0, 1), default=0)
    parser.add_argument('--output-initialization', choices=('zero', 'untouched'), default='zero')
    parser.add_argument('--wait-seconds', type=int, default=1800)
    parser.add_argument('--poll-seconds', type=int, default=10)
    args = parser.parse_args()
    if args.wait_seconds < 0 or not 1 <= args.poll_seconds <= 60:
        parser.error('wait must be nonnegative and poll must be 1..60 seconds')
    root = output_path(args.output)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent/('.'+root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            request = read_json(root/'request.json')
        else:
            if root.exists():
                parser.error('output exists; use --resume for this request or select a new output')
            if not args.config or not args.ge_root or not args.ddk_root:
                parser.error('--config, CANN_ROOT and CPRO_DDK_ROOT (or explicit roots) are required')
            if len(set(args.families)) != len(args.families):
                parser.error('families must be distinct')
            config = Path(args.config).resolve()
            cfg = read_json(config)
            assets = (config.parent/cfg['assets']).resolve()
            env = read_json(assets)['environment']
            if not all(env.get(k) for k in ('ddk_id', 'runtime_version', 'firmware_id')):
                parser.error('runtime assets need ddk_id, runtime_version and firmware_id')
            device._environment()
            native = (config.parent/cfg.get('native_directory', 'native')).resolve()
            binary = native/'build/qwen35_tensor_probe'
            if not binary.is_file() or not (native/'result.json').is_file():
                parser.error('build runtime/native first with targets/cpro/scripts/build_hiai.py')
            built = read_json(native/'result.json')
            if (built.get('status') != 'LINKED' or
                    built.get('diagnostic_binaries', {}).get(binary.name) !=
                    dict(bytes=binary.stat().st_size, sha256=sha256_file(binary))):
                parser.error('rebuild runtime/native with build_hiai.py --replace; runner identity is stale')
            device._check_probe_dtypes(dict(output_initialization=args.output_initialization,
                cases=[MODULES[family].build_graph(64).manifest() for family in args.families]), built)
            reuse = {family: str(output_path(value)) for family, value in (
                ('gdn', args.reuse_gdn), ('cache', args.reuse_cache)) if value}
            if any(Path(p) == root or root in Path(p).parents or Path(p) in root.parents for p in reuse.values()):
                parser.error('reused probes and validation output must be separate')
            request = dict(config=str(config), config_sha256=sha256_file(config),
                assets_sha256=sha256_file(assets), families=args.families, reuse=reuse,
                ge_root=str(Path(args.ge_root).resolve()), ddk_root=str(Path(args.ddk_root).resolve()),
                cxx=args.cxx, cxx11_abi=args.cxx11_abi, output_initialization=args.output_initialization)
            root.mkdir()
            _save(root/'request.json', request)
        state = run(root, request, wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds)
    print(json.dumps(dict(status=state['status'], report=str(root/'result.json'),
        families={k: {n: v.get(n) for n in ('status', 'stage', 'model_compilation',
            'device_execution', 'numerical_validation', 'error', 'recovery')}
            for k, v in state['families'].items()}, all_operators_passed=False), ensure_ascii=False))
    return 2 if state['status'] == 'PENDING' else 0 if state['status'] == 'VALIDATION_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())

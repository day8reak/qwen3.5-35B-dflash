#!/usr/bin/env python3
"""Replay original L0 CPro gated norm and W4 out_proj prefixes against measured 310P outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parents[2] / 'model'))
sys.dont_write_bytecode = True

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.common.command_trace import note


def board(run, args, *, stages=None, summarize=None, basis=None):
    from qwen35_moe_dflash.target.layer_probe.common import sha
    from qwen35_moe_dflash.target.prefill_t64.chain_compile import _pull
    from qwen35_moe_dflash.target.prefill_t64 import gdn_tail_probe as tail
    stages = stages or tail.STAGES
    summarize = summarize or tail.summarize
    basis = basis or tail.BASIS
    air = run/'air'
    if not air.exists():
        _pull(air, air, args.remote_air or air)
    manifest = json.loads((air/'manifest.json').read_text())
    if manifest['golden_source'] != basis:
        raise ValueError(f'Expected {basis} AIR batch at {air}')
    expected = {tail.case_id(s) for s in stages}
    actual = [case['id'] for case in manifest['cases']]
    if sorted(actual) != sorted(expected):
        raise ValueError(f'{air}: selected {sorted(expected)}, but the fixture contains {actual}')
    ddk = args.ddk_root.resolve(strict=True)
    compile_record = run/'omc/compile.json'
    cpu_silu = getattr(args, 'cpu_silu', False)
    if cpu_silu and not compile_record.exists():
        from qwen35_moe_dflash.target.layer_probe.batch import compile_batch
        compile_batch(air, run/'omc', ddk, args.ddk_id, use_origin_format=True,
                      tbe_diagnosis=args.tbe_diagnosis, cpu_silu=True)
    elif not compile_record.exists():
        subprocess.run([sys.executable, TOOLS/'target_layer_probe.py', 'compile', '--batch', air,
                        '--output', run/'omc', '--ddk-root', ddk, '--ddk-id', args.ddk_id,
                        '--compiler-profile', 'isolated-kirinx90', '--tbe-diagnosis', args.tbe_diagnosis,
                        '--use-origin-format'], check=True)
    else:
        note(f'[reuse OMC] {compile_record}')
    build = json.loads(compile_record.read_text())
    if bool(build.get('cpu_silu')) != cpu_silu:
        raise ValueError('Select a fresh run for a different SiLU placement')
    if (build['manifest_sha256'] != sha(air/'manifest.json') or build['ddk_id'] != args.ddk_id
            or set(build['cases']) != expected
            or any(x['status'] != 'COMPILED' for x in build['cases'].values())):
        raise ValueError(f'Incomplete or mismatched compilation. See {compile_record}')
    if getattr(args, 'runner', 'probe') == 'model-tool':
        from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import execute
        name, = expected
        execute(run, name, ddk, args.runtime_config.resolve())
        return summarize(run, v2=True)
    validate = [sys.executable, TOOLS/'validate_draft_vector.py']
    if not (run/'execution/request.json').is_file():
        subprocess.run([*validate, '--config', args.runtime_config, '--compiled', run/'omc', '--ddk-root', ddk,
                        '--output', run/'execution', '--apply'], check=True)
    else:
        note(f"[reuse board request] {run/'execution/request.json'}; board script: {run/'execution/launch.sh'}")
    result = run/'recovery/result.json'
    if not result.is_file() or json.loads(result.read_text()).get('stage') != 'complete':
        recovered = subprocess.run([*validate, '--recover', run/'execution', '--wait', '--output', run/'recovery'])
        if recovered.returncode and not result.is_file():
            recovered.check_returncode()
    else:
        note(f'[reuse board outputs] {result}')
    record = json.loads(result.read_text())
    if record.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED':
        raise ValueError(f"{record.get('error',record['status'])}; see {result}")
    return summarize(run)


def board_arguments(execute):
    execute.add_argument('--ddk-root', type=Path, required=True)
    execute.add_argument('--ddk-id', default=os.environ.get('CPRO_DDK_ID') or '100.520.020.010')
    execute.add_argument('--remote-air', type=Path, help='diagnostic AIR batch on 78 (default RUN/air)')
    execute.add_argument('--runtime-config', type=Path, required=True)
    execute.add_argument('--tbe-diagnosis', type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('prepare', help='on 78: measure both cores on 310P and cut existing CPro AIR')
    execute = commands.add_parser('board', help='on 35: pull, compile, execute and compare all six prefixes')
    report = commands.add_parser('report', help='recompute from saved board results')
    for action in (create, execute, report):
        action.add_argument('--run', type=Path, required=True)
    create.add_argument('--module-air', type=Path, required=True, help='original attention AIR batch containing models/l00_c00_gdn_layer_hidden_o0')
    create.add_argument('--source-root', type=Path, required=True, help='310P source containing inference_segment.py')
    create.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    for flag in ('core', 'gold-core', 'z'):
        create.add_argument('--' + flag, type=Path, required=True, help='FP16 bin; 19 or 64 rows in token/head/channel order')
    board_arguments(execute)
    args = parser.parse_args()
    from qwen35_moe_dflash.target.prefill_t64 import gdn_tail_probe as tail
    run = output_path(args.run)
    if args.action == 'prepare':
        tail.prepare(run, args.module_air.resolve(), args.core.resolve(), args.gold_core.resolve(),
                     args.z.resolve(), args.source_root.resolve(), args.config)
        return 0
    ok = board(run, args) if args.action == 'board' else tail.summarize(run)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Run a complete decoder accuracy case with the configured reference, exporter or board."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model'))
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import MODES, PHASES, case_name, groups


def required(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f'Set {name} in the environment for this host.')
    return Path(value).expanduser().resolve()


def source_config(action):
    nz = action == 'reference'
    src = required('TARGET_SOURCE' if nz else 'CPRO_SOURCE')
    cfg = os.environ.get('NZ_CONFIG' if nz else 'TARGET_CONFIG', 'config/qwen3_5_moe.yaml')
    return src, (src/cfg).resolve(strict=True)


def board_config():
    root = required('AI_RUN_DIR')
    return (required('CPRO_DDK_ROOT'),
            Path(os.getenv('RUNTIME_CONFIG', root/'runtime/runtime.json')).resolve(strict=True),
            Path(os.getenv('TBE_DIAGNOSIS', root/'common/tbe-diagnosis')).resolve())


def native40(args, run, remote):
    tool = Path(__file__).with_name('decoder4_original.py')
    argv = [sys.executable, '-u', str(tool), args.action, '--aligned40', '--run', str(run)]
    if args.action == 'board':
        from decoder4_original import FULL_GRAPH_RUN
        ddk, config, diag = board_config()
        argv.extend(['--ddk', ddk, '--runtime-config', config, '--tbe-diagnosis', diag, '--remote-air',
                     remote/FULL_GRAPH_RUN/'air'])
    else:
        src, cfg = source_config(args.action)
        argv.extend(['--source', src, '--config', cfg, '--reference', os.getenv('ALIGNED40_CAPTURE', run/'reference')])
    return subprocess.run(argv).returncode


def prepare_stage(args, run):
    from qwen35_moe_dflash.common.artifacts import files_under, share_for_transfer
    from qwen35_moe_dflash.target.layer_probe.common import checked, save_json, sha
    src, cfg = source_config(args.action)
    fixture = run/'fixture'
    package = run/'package'
    stamp = dict(mode=args.mode, phase=args.phase)
    if args.action == 'reference':
        from qwen35_moe_dflash.target.prefill_t64 import decoder_case_reference, cpu_reference, split_cpu_reference
        inputs = ('inference_segment.py', 'inference.py', 'utils.py', 'models/export_model_wrapper_qwen3_5_moe.py',
                  'models/modeling_qwen3_5_moe_hiai_nz.py')
        binding = dict(**stamp, config_sha256=sha(cfg), source_files={k: sha(src/k) for k in inputs},
                       math_sources={m.__name__: sha(m.__file__) for m in
                                     (decoder_case_reference, cpu_reference, split_cpu_reference)})
        if fixture.exists():
            ref = json.loads((fixture/'reference.json').read_bytes())
            if ref.get('binding') != binding:
                raise ValueError('Existing reference belongs to a different case or source')
            for key in ('cpu', 'outputs', 'inputs'):
                for row in ref[key].values():
                    checked(fixture, row)
            return
        with tempfile.TemporaryDirectory(dir=run, prefix='.capture-') as tmp:
            working = Path(tmp)/'fixture'
            working.mkdir()
            ref = decoder_case_reference.capture(working, args.mode, args.phase, src, cfg)
            ref['binding'] = binding
            decoder_case_reference.cpu_reference(working, ref, src)
            working.rename(fixture)
        return
    from qwen35_moe_dflash.target.prefill_t64 import decoder_case_air
    from qwen35_moe_dflash.target.prefill_t64 import source_model, source_exporter, decoder_four_silu
    from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import SOURCE_FILES
    ref = json.loads((fixture/'reference.json').read_bytes())
    if (ref['mode'], ref['phase']) != (args.mode, args.phase):
        raise ValueError('Reference differs from selected case')
    stamp.update(reference_sha256=sha(fixture/'reference.json'), config=sha(cfg),
                 cpro_sources={str(p): sha(src/p) for p in SOURCE_FILES},
                 adapters={m.__name__: sha(m.__file__) for m in (decoder_case_air, source_model, source_exporter, decoder_four_silu)})
    if package.exists():
        prior = json.loads((package/'manifest.json').read_bytes())
        if any(prior[k] != v for k, v in stamp.items()):
            raise ValueError('Existing export uses different sources or reference')
        for row in prior['files']:
            checked(package, row)
        print(f'Using verified package {package}', flush=True)
        return
    working = run/'.package-in-progress'
    if not working.exists():
        with tempfile.TemporaryDirectory(dir=run, prefix='.export-') as tmp:
            staged = Path(tmp)/'package'
            staged.mkdir()
            shutil.copytree(fixture, staged/'fixture', copy_function=os.link)
            save_json(staged/'binding.json', stamp)
            staged.rename(working)
    if json.loads((working/'binding.json').read_bytes()) != stamp:
        raise ValueError(f'{working}: interrupted export belongs to another source or fixture. Use a new run.')
    decoder_case_air.export(working, fixture, ref, src, cfg)
    files = [r for r in files_under(working, working) if r['path'] != 'manifest.json']
    save_json(working/'manifest.json', dict(**stamp, files=files))
    share_for_transfer(working)
    working.rename(package)
    size = sum(f['bytes'] for f in files)/2**30
    print(f'{len(groups(args.mode))} AIR, transfer {size:.2f} GiB: {package}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'reference', 'air', 'board'))
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--phase', choices=PHASES, required=True)
    parser.add_argument('--run', type=Path, help='local case directory below AI_RUN_DIR')
    parser.add_argument('--remote-run', type=Path, help='case directory on the export host')
    parser.add_argument('--recovery', action='store_true', help='reuse completed, matching per-OMC board outputs')
    a = parser.parse_args()
    schedule = groups(a.mode)
    if a.action == 'plan':
        print(json.dumps(dict(mode=a.mode, phase=a.phase, rows=64 if a.phase == 'prefill' else 1,
                              valid=19 if a.phase == 'prefill' else 1,
                              logits_position=18 if a.phase == 'prefill' else 19, omcs=len(schedule),
                              graphs=[case_name(group) for group in schedule]), indent=2))
        return 0
    if a.recovery and (a.action, a.mode) != ('board', 'layerwise'):
        parser.error('--recovery applies to board --mode layerwise')
    from qwen35_moe_dflash.common.artifacts import output_path
    native = (a.mode, a.phase) == ('40', 'prefill')
    relative = 'prefill40-original-aligned-silu-softplus-t64-valid19-01' if native else f'decoder-{a.mode}-{a.phase}'
    legacy_run = os.getenv('ALIGNED40_RUN') if native else None
    run = output_path(a.run or os.getenv('DECODER_RUN') or legacy_run or required('AI_RUN_DIR')/'target/tests'/relative)
    remote = a.remote_run or Path(os.getenv('DECODER_REMOTE_RUN', str(run)))
    if not remote.is_absolute():
        parser.error('--remote-run must be the absolute path on the export host')
    run.mkdir(parents=True, exist_ok=True)
    with (run/'.decoder-compare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if native:
            return native40(a, run, remote)
        if a.action == 'board':
            from decoder_cases_board import board
            ddk, config, diag = board_config()
            return 0 if board(run, a.mode, a.phase, remote, ddk, config, diag, a.recovery) else 1
        prepare_stage(a, run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

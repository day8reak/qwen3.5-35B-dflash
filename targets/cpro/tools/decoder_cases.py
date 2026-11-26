#!/usr/bin/env python3
"""Run a complete decoder accuracy case with the configured reference, exporter or board."""
import argparse
import errno
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
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import MODES, PHASES, case_name, directory_name, groups


def required(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f'Set {name} in the environment for this host.')
    return Path(value).expanduser().resolve()


def source_config(action):
    nz = action == 'reference'
    if nz:
        src = required('TARGET_SOURCE')
    elif os.getenv('CPRO_SOURCE'):
        src = required('CPRO_SOURCE')
    else:
        src = required('CPRO_PATCHED_SOURCE')
    cfg = os.environ.get('NZ_CONFIG' if nz else 'TARGET_CONFIG', 'config/qwen3_5_moe.yaml')
    return src, (src/cfg).resolve(strict=True)


def board_config(step='board', *, before_setup=False):
    root = required('AI_RUN_DIR')
    ddk = os.getenv('CPRO_DDK_ROOT') or os.getenv('DDK_PATH')
    if step == 'execute':
        ddk = None
    elif not ddk and not (before_setup and os.getenv('DECODER_DDK_SETUP')):
        raise SystemExit('Set CPRO_DDK_ROOT or load a DDK setup script that exports DDK_PATH.')
    return (Path(ddk).expanduser().resolve() if ddk else None,
            Path(os.getenv('RUNTIME_CONFIG', root/'runtime/runtime.json')).resolve(strict=True) if step != 'compile' else None,
            Path(os.getenv('TBE_DIAGNOSIS', root/'common/tbe-diagnosis')).resolve())


def prepared_source(run):
    from qwen35_moe_dflash.target.original_target_source import SOURCE_FILES, cpro_python_files
    from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import stage
    chosen = os.getenv('CPRO_PATCHED_SOURCE')
    dest = Path(chosen).expanduser().resolve() if chosen else run/'source'
    if not dest.is_dir() or not any(dest.iterdir()):
        original = required('CPRO_SOURCE')
        if original == dest:
            raise ValueError('Cannot generate patches from an empty CPRO_SOURCE')
        helpers = list(cpro_python_files(original))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.cpro-source-') as tmp:
            staged = Path(tmp)/'source'
            stage(original, staged, aligned=True, layers=40)
            # Keep Python helpers imported by local model customizations beside the patched files.
            for src in helpers:
                target = staged/src.relative_to(original)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, target)
            try:
                staged.rename(dest)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not dest.is_dir():
                    raise
    missing = [str(name) for name in SOURCE_FILES if not (dest/name).is_file()]
    if missing:
        raise ValueError(f'{dest}: incomplete CPro source, missing {missing}')
    print(f'Using CPro source: {dest}', flush=True)
    return dest


def board_upload_flags(args):
    flags = []
    if getattr(args, 'replace_inputs', False):
        flags.append('--replace-inputs')
    if getattr(args, 'replace_weights', False):
        flags.append('--replace-weights')
    return flags


def native40(args, run, remote):
    tool = Path(__file__).with_name('decoder4_original.py')
    argv = [sys.executable, '-u', str(tool), args.action, '--aligned40', '--gdn-conv', args.gdn_conv, '--run', str(run)]
    if args.action in ('board', 'execute'):
        argv.extend(board_upload_flags(args))
    if args.action in ('compile', 'execute', 'board'):
        ddk, config, diag = board_config(args.action)
        if ddk:
            argv.extend(['--ddk', ddk, '--tbe-diagnosis', diag, '--remote-air', remote/'air'])
        if config:
            argv.extend(['--runtime-config', config])
    else:
        src, cfg = source_config(args.action)
        argv.extend(['--source', src, '--config', cfg, '--reference', run/'reference'])
    if args.action == 'air':
        argv.extend(['--prepared-source', prepared_source(run)])
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
    from qwen35_moe_dflash.target.original_target_source import cpro_python_files
    from qwen35_moe_dflash.target.prefill_t64 import source_model, source_exporter, decoder_four_silu
    prepared = prepared_source(run)
    ref = json.loads((fixture/'reference.json').read_bytes())
    if (ref['mode'], ref['phase']) != (args.mode, args.phase):
        raise ValueError('Reference differs from selected case')
    if args.gdn_conv == 'taps':
        from qwen35_moe_dflash.target.prefill_t64.gdn_taps import taps_source
        if args.mode != 'full':
            prepared = taps_source(prepared, run/'source-taps')
        stamp['gdn_conv'] = 'taps'
    stamp.update(reference_sha256=sha(fixture/'reference.json'), config=sha(cfg),
                 cpro_source=str(prepared),
                 cpro_sources={str(p.relative_to(prepared)): sha(p) for p in cpro_python_files(prepared)},
                 adapters={m.__name__: sha(m.__file__) for m in (decoder_case_air, source_model, source_exporter, decoder_four_silu)})
    exporter = decoder_case_air.export
    if (args.mode, args.phase) == ('4', 'prefill'):
        import decoder_four_native_case
        exporter = decoder_four_native_case.export
        stamp['adapters']['decoder_four_native_case'] = sha(decoder_four_native_case.__file__)
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
        raise ValueError(f'{working}: interrupted export belongs to another source or fixture. Run prepare again.')
    exporter(working, fixture, ref, src, cfg, prepared, gdn_conv=args.gdn_conv)
    files = [r for r in files_under(working, working) if r['path'] != 'manifest.json']
    save_json(working/'manifest.json', dict(**stamp, files=files))
    share_for_transfer(working)
    working.rename(package)
    size = sum(f['bytes'] for f in files)/2**30
    print(f'{len(groups(args.mode))} AIR, transfer {size:.2f} GiB: {package}', flush=True)


def worker(args, action, root, remote):
    receiver = {'air': 'compile', 'golden': 'air', 'omc': 'execute'}.get(getattr(args, 'artifact', None))
    selection = {
        'reference': ('DECODER_NZ_ENV', 'qwen3_5_w4', 'DECODER_NZ_SETUP'),
        'air': ('DECODER_EXPORT_ENV', 'qwen3_5_cpro', 'DECODER_EXPORT_SETUP'),
        'compile': ('DECODER_COMPILE_ENV', os.getenv('DECODER_BOARD_ENV', 'custom_pypto'), 'DECODER_DDK_SETUP'),
        'execute': ('DECODER_EXECUTE_ENV', os.getenv('DECODER_BOARD_ENV', 'custom_pypto'), 'DECODER_EXECUTE_SETUP'),
        'board': ('DECODER_BOARD_ENV', 'custom_pypto', 'DECODER_DDK_SETUP'),
    }
    selection['diagnose'] = selection['execute']
    environment_key, default_env, setup_key = selection[receiver if action == 'transfer' else action]
    if action == 'transfer':
        default_env = os.getenv(environment_key, default_env)
        environment_key = 'DECODER_TRANSFER_ENV'
    name = os.getenv(environment_key, default_env)
    current = name in ('current', os.getenv('CONDA_DEFAULT_ENV'), os.getenv('CONDA_PREFIX'))
    cmd = [] if current else [os.getenv('CONDA_EXE') or 'conda', 'run', '--no-capture-output',
                              '-p' if os.path.isabs(name) else '-n', name]
    setup = os.getenv(setup_key)
    if (action in ('reference', 'air') or (action == 'transfer' and receiver == 'air')) and not setup:
        setup = os.getenv('DECODER_CANN_SETUP')
    if action == 'transfer' and os.getenv('DECODER_TRANSFER_ENV'):
        setup = None
    python = sys.executable if current else 'python'
    if setup:
        script = Path(setup).expanduser().resolve(strict=True)
        if not script.is_file():
            raise ValueError(f'Expected an environment setup script at {script}')
        # Activate the vendor toolchain inside this worker's conda environment.
        # Keep its changes local and pass argv as positional arguments, including paths with spaces.
        cmd.extend(['bash', '-c', 'decoder_setup_script=$1; shift; decoder_python=$(command -v "$1") || exit 1; shift; '
                    'load_setup() { source "$decoder_setup_script"; }; load_setup || exit $?; exec "$decoder_python" "$@"',
                    'decoder-env', str(script), python])
    else:
        cmd.append(python)
    cmd.extend(['-u', __file__, '_'+action, '--mode', args.mode, '--phase', args.phase,
                '--run', str(root), '--remote-run', str(remote)])
    if action == 'transfer':
        cmd.insert(cmd.index('_transfer')+1, args.artifact)
        if args.local_source:
            cmd.extend(['--local-source', str(args.local_source)])
    if getattr(args, 'gdn_conv', 'conv2d') == 'taps':
        cmd.extend(['--gdn-conv', 'taps'])
    if action in ('board', 'execute'):
        cmd.extend(board_upload_flags(args))
    if args.recovery:
        cmd.append('--recovery')
    label = {'transfer': '接收上游产物', 'reference': '运行 310P 参考并计算 CPU golden', 'air': '从选定的 CPro 源码导出 AIR',
             'compile': '编译 OMC 并保存 I/O 描述', 'execute': '上板执行已编译 OMC 并比较三端结果',
             'diagnose': '读取已失败的板端运行日志', 'board': '读取 AIR、编译 OMC、上板并比较三端结果'}[action]
    print(f'[{directory_name(args.mode, args.phase)}] {label}', flush=True)
    if setup:
        print(f'  加载环境脚本 {script}（{name}）', flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'source', 'prepare', 'board', 'golden', 'air', 'compile', 'execute',
                                           'diagnose', '_diagnose', 'transfer', '_transfer', '_reference', '_air', '_compile', '_execute', '_board'))
    parser.add_argument('artifact', nargs='?', choices=('golden', 'air', 'omc'), help='upstream output to receive')
    parser.add_argument('--local-source', type=Path, help='transfer from a mounted source case directory')
    parser.add_argument('--mode', choices=MODES, default=os.getenv('DECODER_MODE', '40'))
    parser.add_argument('--phase', choices=PHASES, default=os.getenv('DECODER_PHASE', 'prefill'))
    parser.add_argument('--gdn-conv', choices=('conv2d', 'taps'), default=os.getenv('DECODER_GDN_CONV', 'conv2d'),
                        help='CPro implementation of the GDN depthwise convolution')
    parser.add_argument('--run-root', type=Path, help='parent directory for the named cases below AI_RUN_DIR')
    parser.add_argument('--run', type=Path, help='explicit path ending in the canonical case name')
    parser.add_argument('--remote-run', type=Path, help='case directory on the export host')
    parser.add_argument('--patched-source', type=Path, help='editable CPro source directory on the export host')
    parser.add_argument('--replace-inputs', action='store_true', help='upload every board input again')
    parser.add_argument('--replace-weights', action='store_true', help='upload every external board weight again')
    parser.add_argument('--recovery', action='store_true', help='resume the current layerwise board run')
    a = parser.parse_args()
    if (a.action in ('transfer', '_transfer')) != bool(a.artifact) or (a.local_source and not a.artifact):
        parser.error('transfer requires one of golden, air or omc; --local-source applies to transfer only')
    if a.patched_source:
        os.environ['CPRO_PATCHED_SOURCE'] = str(a.patched_source.expanduser().resolve())
    if a.mode not in MODES or a.phase not in PHASES:
        parser.error('invalid DECODER_MODE or DECODER_PHASE')
    if a.gdn_conv not in ('conv2d', 'taps'):
        parser.error('invalid DECODER_GDN_CONV')
    slug = directory_name(a.mode, a.phase)
    if a.action == 'plan':
        schedule = groups(a.mode)
        print(json.dumps(dict(directory=slug, mode=a.mode, phase=a.phase, gdn_conv=a.gdn_conv, rows=64 if a.phase == 'prefill' else 1,
                              prompt='310P YAML prompts[0]', omcs=len(schedule),
                              graphs=[case_name(group) for group in schedule]), indent=2))
        return 0
    if a.action == 'source':
        if not os.getenv('CPRO_PATCHED_SOURCE'):
            parser.error('source needs --patched-source DIR or CPRO_PATCHED_SOURCE in decoder.env')
        prepared_source(None)
        return 0
    if a.recovery and (a.action not in ('board', '_board', 'execute', '_execute', 'compile', '_compile', 'transfer', '_transfer') or a.mode != 'layerwise'):
        parser.error('--recovery applies to board --mode layerwise')
    from qwen35_moe_dflash.common.artifacts import output_path
    parent = a.run_root or Path(os.getenv('DECODER_RUN_ROOT', required('AI_RUN_DIR')/'target/tests'))
    selected = a.run or parent/slug
    root = output_path(selected)
    if selected.name != slug or root.name != slug or selected.is_symlink():
        parser.error(f'case directory must be named {slug}. Set --run-root to change its parent.')
    remote_parent = os.getenv(f'DECODER_{a.artifact.upper()}_REMOTE_ROOT') if a.artifact else None
    remote = a.remote_run or Path(remote_parent or os.getenv('DECODER_REMOTE_RUN_ROOT', root.parent))/slug
    if not remote.is_absolute() or '..' in remote.parts:
        parser.error('remote case must be an absolute path on the export host')
    native = (a.mode, a.phase) == ('40', 'prefill')
    if a.action == '_diagnose':
        from split_stage_log import diagnose_case
        return 0 if diagnose_case(root, a.mode, a.phase) else 1
    if a.action.startswith('_'):
        a.action = a.action[1:]
        site = root/('board' if a.action in ('compile', 'execute', 'board') else 'export')
        from decoder_handoff import publish, receive
        if a.action == 'transfer':
            receive(root, remote, a.artifact, a)
            return 0
        if native:
            code = native40(a, site, remote/'export')
            if code:
                return code
        elif a.action in ('compile', 'execute', 'board'):
            from decoder_cases_board import board
            ddk, config, diag = board_config(a.action)
            if not board(site, a.mode, a.phase, remote/'export', ddk, config, diag, a.recovery,
                         gdn_conv=a.gdn_conv, step=a.action, replace_inputs=a.replace_inputs, replace_weights=a.replace_weights):
                return 1
        else:
            prepare_stage(a, site)
        if a.action in ('reference', 'air', 'compile'):
            publish(root, {'reference': 'golden', 'air': 'air', 'compile': 'omc'}[a.action], a)
        return 0
    # Lock outside the replaceable case and hold it across both conda processes.
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent/f'.{slug}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = root/'.decoder-case'
        if root.exists() and any(root.iterdir()) and (marker.is_symlink() or not marker.is_file() or marker.read_text() != slug):
            raise ValueError(f'{root} contains files from an unknown run; choose another --run-root')
        for key in ('CPRO_PATCHED_SOURCE', 'TARGET_SOURCE', 'CPRO_SOURCE', 'NZ_CONFIG', 'TARGET_CONFIG', 'DECODER_ENV',
                    'CPRO_TRANSFER_CONFIG', 'RUNTIME_CONFIG', 'TBE_DIAGNOSIS'):
            if os.getenv(key) and Path(os.environ[key]).expanduser().resolve().is_relative_to(root):
                raise ValueError(f'{key} must be outside the replaceable result directory {root}')
        if a.action == 'diagnose':
            worker(a, 'diagnose', root, remote)
            return 0
        if a.action in ('prepare', 'golden'):
            source_config('reference')
            if a.action == 'prepare':
                source_config('air')
                if os.getenv('CPRO_PATCHED_SOURCE'):
                    prepared_source(None)
            if root.exists():
                shutil.rmtree(root)
        root.mkdir(exist_ok=True)
        marker.write_text(slug)
        if a.action == 'transfer':
            worker(a, 'transfer', root, remote)
            return 0
        if a.action in ('golden', 'prepare'):
            (root/'export').mkdir()
            worker(a, 'reference', root, remote)
            if a.action == 'prepare':
                worker(a, 'air', root, remote)
        elif a.action == 'air':
            source_config('air')
            site = root/'export'
            reference = site/('reference/manifest.json' if native else 'fixture/reference.json')
            if not reference.is_file():
                raise FileNotFoundError(f'Provide the golden stage outputs here: {reference}')
            for role in ('air', 'omc'):
                (root/f'.handoff-{role}.json').unlink(missing_ok=True)
            for name in ('air', 'package', '.package-in-progress', 'source', 'source-taps'):
                path = site/name
                if path.is_symlink():
                    raise ValueError(f'Unexpected export directory symlink: {path}')
                if path.is_dir():
                    shutil.rmtree(path)
            worker(a, 'air', root, remote)
        elif a.action == 'execute':
            board_config('execute')
            site = root/'board'
            if site.is_symlink() or not site.is_dir():
                raise ValueError(f'Provide the compiled case at {site}')
            for filename in ('comparison.json', 'matrices.csv'):
                (site/filename).unlink(missing_ok=True)
            worker(a, 'execute', root, remote)
        else:
            board_config(a.action, before_setup=True)
            site = root/'board'
            if site.is_symlink():
                raise ValueError(f'Unexpected board directory symlink: {site}')
            bundle = 'air' if native else 'package'
            shared = root/'export'/bundle
            if a.action == 'compile' and not (shared/'manifest.json').is_file():
                raise FileNotFoundError('Receive the AIR package with decoder_transfer.sh air before compiling.')
            (root/'.handoff-omc.json').unlink(missing_ok=True)
            if site.exists() and not a.recovery:
                shutil.rmtree(site)
            site.mkdir(exist_ok=True)
            if a.action == 'compile' or (root == remote and (root/'export').exists()):
                if not (shared/'manifest.json').is_file():
                    raise ValueError(f'Complete prepare before board: {shared}')
                if not (site/bundle).exists():
                    (site/bundle).symlink_to(Path('..')/'export'/bundle, target_is_directory=True)
            separate = any(os.getenv(key) for key in ('DECODER_COMPILE_ENV', 'DECODER_EXECUTE_ENV', 'DECODER_EXECUTE_SETUP'))
            if a.action == 'board' and separate:
                worker(a, 'compile', root, remote)
                worker(a, 'execute', root, remote)
            else:
                worker(a, a.action, root, remote)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

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
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import (
    CLI_EXPORT_CONTROLS, EXPORT_CONTROLS, FOUR_LAYER_CONTROLS, MODES, PHASES, case_name, directory_name, groups,
)


def required(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f'Set {name} in the environment for this host.')
    return Path(value).expanduser().resolve()


def patched_source_variable(gdn_conv):
    return 'CPRO_ORIGINAL_PATCHED_SOURCE' if gdn_conv == 'original' else 'CPRO_PATCHED_SOURCE'


def source_convolution(args):
    # All four-layer controls use the source with native ConvState.
    if any(getattr(args, key, False) for key in FOUR_LAYER_CONTROLS):
        return 'original'
    return args.gdn_conv


def source_config(action, gdn_conv='conv2d', *, no_patch=False):
    nz = action == 'reference'
    if nz:
        src = required('TARGET_SOURCE')
    elif no_patch or os.getenv('CPRO_SOURCE'):
        src = required('CPRO_SOURCE')
    else:
        src = required(patched_source_variable(gdn_conv))
    if no_patch and not nz:
        from qwen35_moe_dflash.target.original_target_source import EXPORTER
        (src/EXPORTER).resolve(strict=True)
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


def prepared_source(run, gdn_conv='conv2d'):
    from qwen35_moe_dflash.target.original_target_source import SOURCE_FILES, cpro_python_files
    from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import stage
    native = gdn_conv == 'original'
    chosen = os.getenv(patched_source_variable(gdn_conv))
    dest = Path(chosen).expanduser().resolve() if chosen else run/'source'
    if not dest.is_dir() or not any(dest.iterdir()):
        original = required('CPRO_SOURCE')
        if original == dest:
            raise ValueError('Cannot generate patches from an empty CPRO_SOURCE')
        helpers = list(cpro_python_files(original))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.cpro-source-') as tmp:
            staged = Path(tmp)/'source'
            stage(original, staged, aligned=True, layers=40, native_convstate=native)
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
    if getattr(args, 'original_board_args', False):
        flags.append('--original-board-args')
    return flags


def native40(args, run, remote):
    tool = Path(__file__).with_name('decoder4_original.py')
    argv = [sys.executable, '-u', str(tool), args.action, '--aligned40', '--gdn-conv', args.gdn_conv, '--run', str(run)]
    if args.rmsnorm_patch and args.action != 'reference':
        argv.append('--rmsnorm-patch')
    if args.hidden_trace and args.action != 'reference':
        argv.append('--hidden-trace')
    if args.action in ('board', 'execute'):
        argv.extend(board_upload_flags(args))
    if args.action in ('compile', 'execute', 'board'):
        ddk, config, diag = board_config(args.action)
        if ddk:
            argv.extend(['--ddk', ddk, '--tbe-diagnosis', diag, '--remote-air', remote/'air'])
        if config:
            argv.extend(['--runtime-config', config])
    else:
        src, cfg = source_config(args.action, args.gdn_conv)
        argv.extend(['--source', src, '--config', cfg, '--reference', run/'reference'])
    if args.action == 'air':
        argv.extend(['--prepared-source', prepared_source(run, args.gdn_conv)])
        if args.gdn_conv == 'original' and os.getenv('DECODER_CONV_CONTROL'):
            argv.extend(['--conv-control', required('DECODER_CONV_CONTROL')])
    return subprocess.run(argv).returncode


def prepare_stage(args, run):
    from qwen35_moe_dflash.common.artifacts import files_under, share_for_transfer
    from qwen35_moe_dflash.target.layer_probe.common import checked, save_json, sha
    source_conv = source_convolution(args)
    src, cfg = source_config(args.action, source_conv, no_patch=args.no_patch)
    fixture = run/'fixture'
    package = run/'package'
    stamp = dict(mode=args.mode, phase=args.phase)
    if args.dump:
        stamp['dump'] = True
    if args.action == 'reference':
        from qwen35_moe_dflash.target.prefill_t64 import (
            chain_reference, cpu_reference, decoder_case_reference, decoder_embedding, split_cpu_reference,
        )
        inputs = ('inference_segment.py', 'inference.py', 'utils.py', 'models/export_model_wrapper_qwen3_5_moe.py',
                  'models/modeling_qwen3_5_moe_hiai_nz.py')
        binding = dict(**stamp, config_sha256=sha(cfg), source_files={k: sha(src/k) for k in inputs},
                       math_sources={m.__name__: sha(m.__file__) for m in
                                     (decoder_case_reference, cpu_reference, split_cpu_reference,
                                      chain_reference, decoder_embedding)})
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
            if args.dump:
                from l0_accuracy_reference import capture_reference
                capture_reference(working, working/'l0-steps-reference', src, cfg)
            working.rename(fixture)
        return
    from qwen35_moe_dflash.target.original_target_source import cpro_python_files
    prepared = src if args.no_patch else prepared_source(run, source_conv)
    ref = json.loads((fixture/'reference.json').read_bytes())
    if args.dump and not (fixture/'l0-steps-reference/values.npz').is_file():
        raise ValueError('The dump run requires the L0 intermediate golden from decoder_prepare.sh --dump')
    if (ref['mode'], ref['phase']) != (args.mode, args.phase):
        raise ValueError('Reference differs from selected case')
    if args.mode != '40' and not ref.get('embedding_inputs'):
        raise ValueError('Regenerate the decoder golden to capture the external embedding rows and scales')
    if args.gdn_conv == 'taps':
        from qwen35_moe_dflash.target.prefill_t64.gdn_taps import taps_source
        if args.mode != 'full':
            prepared = taps_source(prepared, run/'source-taps')
    if args.gdn_conv != 'conv2d' or source_conv == 'original':
        stamp['gdn_conv'] = args.gdn_conv
    if source_conv == 'original' and not getattr(args, 'physical_convstate', False) and os.getenv('DECODER_CONV_CONTROL'):
        stamp['conv_control'] = sha(required('DECODER_CONV_CONTROL'))
    controls = {name: getattr(args, name) for name in EXPORT_CONTROLS if getattr(args, name, False)}
    stamp.update(controls)
    stamp.update(reference_sha256=sha(fixture/'reference.json'), config=sha(cfg),
                 cpro_source=str(prepared),
                 cpro_sources={str(p.relative_to(prepared)): sha(p) for p in cpro_python_files(prepared)},
                 adapters={})
    from qwen35_moe_dflash.target.prefill_t64 import decoder_case_air
    if not args.no_patch:
        from qwen35_moe_dflash.target.prefill_t64 import source_model, source_exporter, decoder_four_silu, decoder_source
        stamp['adapters'] = {m.__name__: sha(m.__file__)
                             for m in (decoder_case_air, source_model, source_exporter, decoder_four_silu, decoder_source)}
        if args.rmsnorm_patch:
            from qwen35_moe_dflash.target.prefill_t64 import decoder_rmsnorm
            stamp['adapters'][decoder_rmsnorm.__name__] = sha(decoder_rmsnorm.__file__)
    options = dict(gdn_conv=args.gdn_conv, rmsnorm_patch=args.rmsnorm_patch)
    native_four = (args.mode, args.phase) == ('4', 'prefill')
    if native_four:
        import decoder_four_native_case
        exporter = decoder_four_native_case.export
        options['no_patch'] = args.no_patch
        options.update(controls)
        stamp['adapters']['decoder_four_native_case'] = sha(decoder_four_native_case.__file__)
    else:
        exporter = decoder_case_air.export
        options['shared_gate_fp16_sigmoid'] = args.shared_gate_fp16_sigmoid
        options['qlinear_scale_fp16'] = args.qlinear_scale_fp16
    if package.exists():
        prior = json.loads((package/'manifest.json').read_bytes())
        if (any(prior.get(k, False) != stamp.get(k, False) for k in EXPORT_CONTROLS)
                or any(prior.get(k) != v for k, v in stamp.items())):
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
    if args.no_patch:
        exporter(working, fixture, ref, src, cfg, prepared, **options)
    else:
        from qwen35_moe_dflash.target.prefill_t64.decoder_source import export_source
        with tempfile.TemporaryDirectory(dir=run, prefix='decoder-source-') as tmp:
            adapted = export_source(prepared, Path(tmp)/'source', rmsnorm_patch=args.rmsnorm_patch,
                                    shared_gate_fp16_sigmoid=args.shared_gate_fp16_sigmoid,
                                    qlinear_scale_fp16=args.qlinear_scale_fp16,
                                    fold_norm_gamma=not native_four, native_prefill=native_four)
            exporter(working, fixture, ref, src, cfg, adapted, **options)
    files = [r for r in files_under(working, working) if r['path'] != 'manifest.json']
    save_json(working/'manifest.json', dict(**stamp, files=files))
    share_for_transfer(working)
    working.rename(package)
    print(f'[CPro AIR] 本次导出的模型与 Python 源码：{package/"air-source"}', flush=True)
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
                    'load_setup() { source "$decoder_setup_script"; }; load_setup || exit $?; '
                    'export QWEN35_DECODER_SETUP_LOADED="$decoder_setup_script"; exec "$decoder_python" "$@"',
                    'decoder-env', str(script), python])
    else:
        cmd.append(python)
    cmd.extend(['-u', __file__, '_'+action, '--mode', args.mode, '--phase', args.phase,
                '--run', str(root), '--remote-run', str(remote)])
    if action == 'transfer':
        cmd.insert(cmd.index('_transfer')+1, args.artifact)
        if args.local_source:
            cmd.extend(['--local-source', str(args.local_source)])
    cmd.extend(['--gdn-conv', args.gdn_conv])
    for control in CLI_EXPORT_CONTROLS:
        value = getattr(args, control, False)
        if value:
            cmd.append('--' + control.replace('_', '-'))
            if isinstance(value, str):
                cmd.append(value)
    if action in ('board', 'execute'):
        cmd.extend(board_upload_flags(args))
    if args.dump:
        cmd.append('--dump')
    if getattr(args, 'cache_existing_golden', False):
        cmd.append('--cache-existing-golden')
    if getattr(args, 'gate_probe', False):
        cmd.extend(['--gate-probe', args.gate_probe])
        if args.rmsnorm_air:
            cmd.extend(['--rmsnorm-air', args.rmsnorm_air])
    if args.recovery:
        cmd.append('--recovery')
    label = {'transfer': '接收上游产物', 'reference': '运行 310P 参考并计算 CPU golden', 'air': '从选定的 CPro 源码导出 AIR',
             'compile': '编译 OMC 并保存 I/O 描述', 'execute': '上板执行已编译 OMC 并比较三端结果',
             'diagnose': '诊断已保留的 OMG 编译或板端失败记录', 'board': '读取 AIR、编译 OMC、上板并比较三端结果'}[action]
    if action == 'reference' and (args.mode, args.phase) == ('40', 'prefill'):
        label = '归档已有 40 层 golden' if args.cache_existing_golden else '准备 40 层 golden，优先复用 CPU 和 310P 参考'
    if getattr(args, 'gate_probe', False):
        label = '从四层 AIR 切出并编译门控对照' if action == 'compile' else '上板执行门控对照'
        if args.gate_probe == 'l0-embedding':
            label = '采集原 310P embedding 查表及乘法，与四层 CPro 输入逐 token 对照'
        elif args.gate_probe == 'l0-shared-gate':
            label = {'reference': '提取共享专家标量门控权重与已采集的参考值',
                     'compile': '用已回传的 L0 张量离线复算共享专家门控'}[action]
        elif args.gate_probe == 'l0-shared-gate-fp16':
            label = {'compile': '编译 L0 共享门控 FP16 MatMul → FP32 Cast 对照',
                     'execute': '用相同 L0 输入上板，对比标量门控和共享专家加权输出'}[action]
        elif args.gate_probe == 'l0-input-rmsnorm-omc':
            label = '把已有 RMSNorm 单算子 AIR 编译为同目录的 OMC'
        elif args.gate_probe in ('l0-input-rmsnorm', 'l0-input-rmsnorm-air'):
            label = '上板单测 L0 input_layernorm' if action == 'execute' else '提取 L0 input_layernorm 和真实 310P 输入输出'
        elif args.gate_probe == 'l0-steps-same-input':
            label = {'compile': '从 embedding 乘法之后切 L0 并编译诊断 OMC',
                     'execute': '用实测 310P 入口运行 L0 三端对照'}[action]
        elif args.gate_probe == 'l0-steps':
            label = {'reference': '采集 L0 真实 310P 中间量并计算 CPU 对照', 'compile': '从四层 AIR 导出 L0 中间量并编译',
                     'execute': '使用四层真实输入上板，按计算顺序生成 L0 三端矩阵'}[action]
        elif args.gate_probe == 'report':
            label = '对照四层和门控小图的已有 AIR、编译日志及上板记录'
        elif args.gate_probe == 'l0-decay-report':
            label = '查看现有 L0 AIR 和 OMC 的 Softplus 衰减支路'
        elif args.gate_probe == 'l0-dump':
            label = '复用已有 OMC，回收 L0 算子 dump'
        elif args.gate_probe == 'l0-original-omg':
            label = {'compile': '用原 CPro OMG 参数编译已有 L0 AIR', 'execute': '用原 V2 参数执行对照 OMC'}[action]
        elif args.gate_probe == 'gdn-bisect':
            label = {'compile': '从已有单层 GDN AIR 分段切图并编译 OMC',
                     'execute': '逐个上板并汇总复现同一加载错误的最小切片'}[action]
        elif args.gate_probe in ('gdn-export', 'gdn-whole'):
            label = {'compile': '从当前四层 AIR 导出完整 L0 linear attention',
                     'execute': '上板执行完整 L0 linear attention'}[action]
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
    parser.add_argument('--gdn-conv', choices=('conv2d', 'taps', 'original'),
                        help='CPro implementation of the GDN depthwise convolution')
    activation_mode = parser.add_mutually_exclusive_group()
    activation_mode.add_argument('--no-patch', action='store_true',
                                 help='run the exporter directly from CPRO_SOURCE for four-layer T64 prefill')
    activation_mode.add_argument('--native-activations', action='store_true',
                                 help='keep the aligned four-layer native GDN graph and restore stock SiLU and Softplus')
    activation_mode.add_argument('--silu-only', action='store_true',
                                 help='use FP16 SiluPatch and stock Softplus in the aligned four-layer T64 model')
    activation_mode.add_argument('--softplus-only', action='store_true',
                                 help='use stock SiLU and FP16 SoftplusPatch in the aligned four-layer T64 model')
    activation_mode.add_argument('--gdn-silu-only', action='store_true',
                                 help='use SiluPatch only in GDN, stock SiLU in MoE, and keep GDN SoftplusPatch')
    activation_mode.add_argument('--gdn-silu-site', choices=('conv', 'gate'), default=False,
                                 help='enable SiluPatch only after the GDN convolution or only at its RMS gate; keep SoftplusPatch')
    activation_mode.add_argument('--all-silu-no-gate-cast', action='store_true',
                                 help='use all GDN and MoE SiluPatch calls and omit the explicit FP32 cast after the GDN gate')
    activation_mode.add_argument('--rms-silu-token-view', action='store_true',
                                 help='select a previously exported four-layer [64,4096] AIR for replay')
    activation_mode.add_argument('--rms-silu-layers', default=False,
                                 choices=('none', '0', '1', '2', '0,1', '0,2', '1,2', '0,1,2'),
                                 help='use FP16 SiluPatch on the selected RMS gates and FP16 F.silu on the other gates')
    parser.add_argument('--physical-convstate', action='store_true',
                        help='replace the four-layer T64 ConvState update with concatenation and physical-tail slices')
    parser.add_argument('--run-root', type=Path, help='parent directory for the named cases below AI_RUN_DIR')
    parser.add_argument('--run', type=Path, help='explicit path ending in the canonical case name')
    parser.add_argument('--remote-run', type=Path, help='case directory on the export host')
    parser.add_argument('--patched-source', type=Path, help='editable CPro source directory on the export host')
    parser.add_argument('--replace-inputs', action='store_true', help='upload every board input again')
    parser.add_argument('--replace-weights', action='store_true', help='upload every external board weight again')
    parser.add_argument('--original-board-args', action='store_true',
                        help="use the owner's V2 flags with compiled I/O shapes on the current OMC")
    parser.add_argument('--rmsnorm-patch', action='store_true', help='use the custom FP16 RmsnormPatch in the CPro decoder')
    parser.add_argument('--hidden-trace', action='store_true', help='return post-decoder hidden states from the native 40-layer prefill')
    parser.add_argument('--dump', action='store_true',
                        help='capture L0 Python intermediates and dump one complete linear prefill OMC on the board')
    parser.add_argument('--cache-existing-golden', action='store_true',
                        help='archive the completed 40-layer prefill golden from this case')
    parser.add_argument('--recovery', action='store_true', help='resume the current layerwise board run')
    parser.add_argument('--gate-probe', nargs='?', const='slices',
                        choices=('slices', 'upstream', 'report', 'gdn-export', 'gdn-whole', 'gdn-bisect', 'l0-steps', 'l0-steps-same-input', 'l0-embedding', 'l0-shared-gate', 'l0-shared-gate-fp16',
                                 'l0-input-rmsnorm', 'l0-input-rmsnorm-air', 'l0-input-rmsnorm-omc',
                                 'l0-decay-report', 'l0-original-omg', 'l0-dump'), help=argparse.SUPPRESS)
    parser.add_argument('--rmsnorm-air', help='AIR filename in the L0 RMSNorm single-op directory (default: export.air)')
    a = parser.parse_args()
    # Keep these settings in artifact manifests so an older AIR cannot silently replace this recipe.
    aligned_prefill = a.mode in ('linear', '4', '40') and a.phase == 'prefill' and not a.no_patch
    a.shared_gate_fp16_sigmoid = aligned_prefill
    a.qlinear_scale_fp16 = aligned_prefill
    if a.hidden_trace and (a.mode, a.phase) != ('40', 'prefill'):
        parser.error('--hidden-trace requires --mode 40 --phase prefill')
    if a.cache_existing_golden and ((a.mode, a.phase) != ('40', 'prefill')
                                    or a.action not in ('golden', '_reference') or a.gate_probe):
        parser.error('--cache-existing-golden requires golden --mode 40 --phase prefill')
    if a.dump and ((a.mode, a.phase) != ('linear', 'prefill') or a.gate_probe):
        parser.error('--dump requires --mode linear --phase prefill with the regular decoder stages')
    if a.rms_silu_token_view and a.action in ('prepare', 'golden', 'air', '_air'):
        parser.error('--rms-silu-token-view selects historical AIRs. Omit it when exporting the native gate shape.')
    if a.no_patch and a.rmsnorm_patch:
        parser.error('--no-patch and --rmsnorm-patch select different RMSNorm implementations')
    if a.rmsnorm_air is not None and (a.gate_probe != 'l0-input-rmsnorm-omc'
                                      or Path(a.rmsnorm_air).name != a.rmsnorm_air
                                      or Path(a.rmsnorm_air).suffix != '.air'):
        parser.error('--rmsnorm-air takes a .air filename for l0-input-rmsnorm-omc')
    probe_actions = ('compile', 'execute', '_compile', '_execute')
    if a.gate_probe == 'l0-embedding':
        probe_actions = ('golden', '_reference')
    elif a.gate_probe == 'l0-shared-gate':
        probe_actions = ('golden', '_reference', 'compile', '_compile')
    elif a.gate_probe in ('l0-input-rmsnorm-air', 'l0-input-rmsnorm-omc', 'l0-decay-report'):
        probe_actions = ('compile', '_compile')
    elif a.gate_probe == 'l0-steps':
        probe_actions += ('golden', '_reference')
    elif a.gate_probe == 'l0-dump':
        probe_actions = ('execute', '_execute')
    probe_mode = 'linear' if a.gate_probe in ('l0-original-omg', 'l0-decay-report', 'l0-dump') else '4'
    if a.gate_probe == 'l0-dump' and a.mode == '4':
        probe_mode = '4'
    if a.gate_probe and ((a.mode, a.phase) != (probe_mode, 'prefill') or
                         a.action not in probe_actions):
        parser.error(f'{a.gate_probe} requires --mode {probe_mode} --phase prefill and its supported stage')
    if any(getattr(a, name) for name in FOUR_LAYER_CONTROLS):
        if (a.mode, a.phase) != ('4', 'prefill'):
            parser.error('use --mode 4 --phase prefill for this four-layer control')
        if a.no_patch and a.patched_source:
            parser.error('--no-patch uses CPRO_SOURCE directly with the original GDN convolution')
        if a.no_patch and a.physical_convstate:
            parser.error('--physical-convstate changes the CPro model and cannot be used with --no-patch')
        if a.action == 'source':
            parser.error('prepare the editable base with source --gdn-conv original')
        a.gdn_conv = a.gdn_conv or ('conv2d' if a.physical_convstate else 'original')
        if a.no_patch and a.gdn_conv != 'original':
            parser.error('--no-patch runs the original GDN convolution from CPRO_SOURCE')
        if a.gdn_conv == 'taps':
            parser.error('four-layer activation controls support --gdn-conv original or conv2d')
    else:
        a.gdn_conv = a.gdn_conv or os.getenv('DECODER_GDN_CONV', 'conv2d')
    if (a.action in ('transfer', '_transfer')) != bool(a.artifact) or (a.local_source and not a.artifact):
        parser.error('transfer requires one of golden, air or omc; --local-source applies to transfer only')
    source_conv = source_convolution(a)
    patch_variable = patched_source_variable(source_conv)
    if a.patched_source:
        os.environ[patch_variable] = str(a.patched_source.expanduser().resolve())
    if a.mode not in MODES or a.phase not in PHASES:
        parser.error('invalid DECODER_MODE or DECODER_PHASE')
    if a.gdn_conv == 'original' and (a.phase != 'prefill' or a.mode not in ('4', '40')):
        parser.error('--gdn-conv original applies to 4 or 40 layers with --phase prefill')
    if a.gdn_conv not in ('conv2d', 'taps', 'original'):
        parser.error('invalid DECODER_GDN_CONV')
    slug = directory_name(a.mode, a.phase)
    if a.action == 'plan':
        schedule = groups(a.mode)
        print(json.dumps(dict(directory=slug, mode=a.mode, phase=a.phase, gdn_conv=a.gdn_conv, rows=64 if a.phase == 'prefill' else 1,
                              prompt='310P YAML prompts[0]', dump=a.dump, hidden_trace=a.hidden_trace, rmsnorm_patch=a.rmsnorm_patch, no_patch=a.no_patch, native_activations=a.native_activations,
                              silu_only=a.silu_only, softplus_only=a.softplus_only, gdn_silu_only=a.gdn_silu_only,
                              gdn_silu_site=a.gdn_silu_site, all_silu_no_gate_cast=a.all_silu_no_gate_cast,
                              rms_silu_layers=a.rms_silu_layers, rms_silu_token_view=a.rms_silu_token_view,
                              physical_convstate=a.physical_convstate, shared_gate_fp16_sigmoid=a.shared_gate_fp16_sigmoid,
                              qlinear_scale_fp16=a.qlinear_scale_fp16,
                              omcs=len(schedule),
                              graphs=[case_name(group) for group in schedule]), indent=2))
        return 0
    if a.action == 'source':
        if not os.getenv(patch_variable):
            parser.error(f'source needs --patched-source DIR or {patch_variable} in decoder.env')
        prepared_source(None, a.gdn_conv)
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
    if native:
        cache_root = required('AI_RUN_DIR')/'target/golden'/slug
        if root.is_relative_to(cache_root) or cache_root.is_relative_to(root):
            parser.error('Select an experiment --run-root separate from the persistent 40-layer golden directory')
    if a.action == '_diagnose':
        from split_stage_log import diagnose_case
        return 0 if diagnose_case(root, a.mode, a.phase) else 1
    if a.action.startswith('_'):
        a.action = a.action[1:]
        if a.gate_probe:
            if a.gate_probe == 'l0-decay-report':
                from decoder_gate_report import report_linear_decay
                report_linear_decay(root, board_config('compile')[0])
                return 0
            if a.gate_probe in ('l0-original-omg', 'l0-dump'):
                from decoder_cases_board import board
                ddk, config, diag = board_config(a.action)
                original = a.gate_probe == 'l0-original-omg'
                return 0 if board(root/'board', a.mode, a.phase, remote/'export', ddk, config, diag, False,
                                  step=a.action, gdn_conv=a.gdn_conv, original_omg=original, rmsnorm_patch=a.rmsnorm_patch,
                                  shared_gate_fp16_sigmoid=a.shared_gate_fp16_sigmoid, qlinear_scale_fp16=a.qlinear_scale_fp16,
                                  original_board_args=original or a.original_board_args, operator_dump=not original,
                                  replace_inputs=a.replace_inputs, replace_weights=a.replace_weights) else 1
            if a.gate_probe == 'l0-embedding':
                from l0_embedding import diagnose
                return 0 if diagnose(root, *source_config('reference')) else 1
            if a.gate_probe == 'l0-shared-gate':
                from l0_shared_gate import prepare, report
                if a.action == 'reference':
                    prepare(root)
                else:
                    report(root, remote)
                return 0
            if a.gate_probe in ('l0-steps', 'l0-steps-same-input'):
                if a.action == 'reference':
                    from l0_accuracy_reference import prepare
                    prepare(root, *source_config('reference'))
                    return 0
                from l0_accuracy_report import run
                ddk, config, _ = board_config(a.action)
                return 0 if run(root, remote, a.action, ddk, config, a) else 1
            if a.gate_probe == 'l0-shared-gate-fp16':
                from l0_shared_gate_fp16 import run
                ddk, config, _ = board_config(a.action)
                return 0 if run(root, remote, a.action, ddk, config, a) else 1
            if a.gate_probe in ('l0-input-rmsnorm', 'l0-input-rmsnorm-air', 'l0-input-rmsnorm-omc'):
                from l0_input_rmsnorm import run
                ddk, config, _ = (None, None, None) if a.gate_probe.endswith('-air') else board_config(a.action)
                return 0 if run(root, remote, a.action, ddk, config, a) else 1
            if a.gate_probe == 'report':
                from decoder_gate_report import report
                report(root)
                return 0
            from decoder_gate_probe import probe
            ddk, config, _ = (None, None, None) if a.gate_probe == 'gdn-export' else board_config(a.action)
            return 0 if probe(root, a.action, ddk, config, mode=a.gate_probe,
                              replace_inputs=a.replace_inputs, replace_weights=a.replace_weights) else 1
        site = root/('board' if a.action in ('compile', 'execute', 'board') else 'export')
        from decoder_handoff import publish, receive
        if a.action == 'transfer':
            receive(root, remote, a.artifact, a)
            return 0
        if native and a.action == 'reference':
            from decoder_golden_cache import prepare
            prepare(root, cache_root, a, *source_config('reference'))
            if a.cache_existing_golden:
                return 0
        elif native:
            code = native40(a, site, remote/'export')
            if code:
                return code
        elif a.action in ('compile', 'execute', 'board'):
            from decoder_cases_board import board
            ddk, config, diag = board_config(a.action)
            if not board(site, a.mode, a.phase, remote/'export', ddk, config, diag, a.recovery,
                         gdn_conv=a.gdn_conv, step=a.action, replace_inputs=a.replace_inputs, replace_weights=a.replace_weights,
                         original_board_args=a.original_board_args, no_patch=a.no_patch, rmsnorm_patch=a.rmsnorm_patch, dump=a.dump,
                         shared_gate_fp16_sigmoid=a.shared_gate_fp16_sigmoid, qlinear_scale_fp16=a.qlinear_scale_fp16,
                         native_activations=a.native_activations, silu_only=a.silu_only, softplus_only=a.softplus_only,
                         gdn_silu_only=a.gdn_silu_only, gdn_silu_site=a.gdn_silu_site,
                         all_silu_no_gate_cast=a.all_silu_no_gate_cast, physical_convstate=a.physical_convstate,
                         rms_silu_layers=a.rms_silu_layers, rms_silu_token_view=a.rms_silu_token_view):
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
        if a.cache_existing_golden:
            worker(a, 'reference', root, remote)
            return 0
        for key in ('CPRO_PATCHED_SOURCE', 'CPRO_ORIGINAL_PATCHED_SOURCE', 'DECODER_CONV_CONTROL', 'TARGET_SOURCE', 'CPRO_SOURCE', 'NZ_CONFIG', 'TARGET_CONFIG', 'DECODER_ENV',
                    'CPRO_TRANSFER_CONFIG', 'RUNTIME_CONFIG', 'TBE_DIAGNOSIS'):
            if os.getenv(key) and Path(os.environ[key]).expanduser().resolve().is_relative_to(root):
                raise ValueError(f'{key} must be outside the replaceable result directory {root}')
        if a.action == 'diagnose' or a.gate_probe:
            worker(a, 'reference' if a.action == 'golden' else a.action, root, remote)
            return 0
        if (a.native_activations or a.silu_only or a.softplus_only) and a.action in ('board', 'compile', 'execute', 'transfer'):
            from decoder_four_inputs import remember_original_inputs
            remember_original_inputs(root, baseline='no_patch' if a.native_activations else 'native_activations')
        if a.action in ('prepare', 'golden'):
            source_config('reference')
            if a.action == 'prepare':
                source_config('air', source_conv, no_patch=a.no_patch)
                if not a.no_patch and os.getenv(patch_variable):
                    prepared_source(None, source_conv)
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
            source_config('air', source_conv, no_patch=a.no_patch)
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
            for filename in ('comparison.json', 'matrices.csv', 'hidden_cosine.png', 'hidden_error_ratio.png'):
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
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        # A decoder worker has already printed its result or original traceback to this terminal.
        if __file__ not in error.cmd:
            raise
        code = error.returncode
        raise SystemExit(code if code > 0 else 128 - code) from None

"""Package the owner's quantized four-layer T64 export for the decoder comparison."""
from difflib import unified_diff
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np
import yaml

from decoder4_original import (EXPORT_DIR, EXPORTER, four_precision, native_export_argv, original_filename,
                               prefill_inputs, select_native_depth)
from parse_model_json import GE_PROTO
from qwen35_moe_dflash.common.artifacts import files_under
from qwen35_moe_dflash.target.layer_probe.common import save_json, sized_file
from qwen35_moe_dflash.target.original_target_source import WRAPPER
from qwen35_moe_dflash.target.prefill_t64.chain_air import checkpoint_paths
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import case_name
from qwen35_moe_dflash.target.prefill_t64.decoder_source import save_air_source
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import inspect_air
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import (
    MODELING, configure_gate_export, inline_old_silu, inspect_activation_air, omit_gate_output_cast,
    restore_native_activations, select_gdn_silu, token_view_rms_gate,
)
from qwen35_moe_dflash.target.prefill_t64.source_model import adapt_four_layer_convolution


def four_layer_entry(prepared, work, *, native_activations=False, silu_only=False, softplus_only=False,
                     gdn_silu_only=False, gdn_silu_site=False, all_silu_no_gate_cast=False,
                     explicit_conv2d=False, physical_convstate=False, rms_layers=None, rms_silu_token_view=False):
    entry = prepared/EXPORTER
    text = entry.read_text()
    # Editable sources generated before multi-depth export counted 30 GDR calls even at L=4.
    old = 'with cpro_export_contracts(True, chunk_size=64, gdn_layers=30):'
    base_model = (prepared/MODELING).read_text()
    model = inline_old_silu(base_model)
    if gdn_silu_only or gdn_silu_site:
        model = select_gdn_silu(model, gdn_silu_site)
    if all_silu_no_gate_cast or rms_layers is not None:
        model = omit_gate_output_cast(model, select_layers=rms_layers is not None)
    if rms_silu_token_view:
        model = token_view_rms_gate(model)
    if rms_layers is not None:
        text = configure_gate_export(text, rms_layers)
    restore_activations = native_activations or silu_only or softplus_only
    edit_model = restore_activations or explicit_conv2d or physical_convstate or model != base_model
    if old not in text and not edit_model:
        return entry
    overlay = work/'source'
    overlay.mkdir()
    for path in prepared.iterdir():
        if edit_model and path.name == 'models':
            models = overlay/'models'
            models.mkdir()
            for module in path.iterdir():
                if module.name not in (Path(MODELING).name, '__pycache__'):
                    (models/module.name).symlink_to(module.resolve(), target_is_directory=module.is_dir())
            if restore_activations:
                text, model = restore_native_activations(text, model, keep_silu=silu_only, keep_softplus=softplus_only)
            if explicit_conv2d or physical_convstate:
                model = adapt_four_layer_convolution(model, explicit_conv2d=explicit_conv2d,
                                                     physical_convstate=physical_convstate)
            (overlay/MODELING).write_text(model)
        elif path.name not in (EXPORTER, '__pycache__'):
            (overlay/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (overlay/EXPORTER).write_text(text.replace(old, old.replace('gdn_layers=30', 'gdn_layers=3')))
    return overlay/EXPORTER


def export(root, fixture, ref, source, config, prepared, *, gdn_conv, no_patch=False, native_activations=False,
           silu_only=False, softplus_only=False, gdn_silu_only=False, gdn_silu_site=False,
           all_silu_no_gate_cast=False, physical_convstate=False, rms_silu_layers=False, rms_silu_token_view=False):
    dest = root/case_name(ref['layers'])
    if dest.is_dir():
        saved = json.loads((dest/'case.json').read_bytes())
        if (saved.get('air_directory') != EXPORT_DIR or saved.get('gdn_conv', 'conv2d') != gdn_conv
                or saved.get('no_patch', False) != no_patch
                or saved.get('native_activations', False) != native_activations
                or saved.get('silu_only', False) != silu_only or saved.get('softplus_only', False) != softplus_only
                or saved.get('gdn_silu_only', False) != gdn_silu_only
                or saved.get('gdn_silu_site', False) != gdn_silu_site
                or saved.get('all_silu_no_gate_cast', False) != all_silu_no_gate_cast
                or saved.get('rms_silu_token_view', False) != rms_silu_token_view
                or saved.get('rms_silu_layers', False) != rms_silu_layers
                or saved.get('physical_convstate', False) != physical_convstate):
            raise ValueError(f'{dest}: existing four-layer export uses a different source mode')
        for row in saved['files']:
            sized_file(dest, row)
        return
    cfg = yaml.safe_load(config.read_text())
    cfg.update({k: str(v) for k, v in checkpoint_paths(source, config, ref, 4).items()})
    for key in ('embedding_weight_path', 'embedding_scale_path'):
        cfg[key] = str((source/cfg[key]).resolve(strict=True))
    with tempfile.TemporaryDirectory(dir=root.parent, prefix='native-four-') as tmp:
        work = Path(tmp)
        case = work/'case'
        case.mkdir()
        restrict_silu_to_gdn = gdn_silu_only or bool(gdn_silu_site)
        rms_layers = None
        if rms_silu_layers:
            rms_layers = () if rms_silu_layers == 'none' else tuple(map(int, rms_silu_layers.split(',')))
        gate_control = all_silu_no_gate_cast or rms_layers is not None or rms_silu_token_view
        activation_control = native_activations or silu_only or softplus_only or restrict_silu_to_gdn or gate_control
        use_native_tools = gdn_conv == 'original' or activation_control or physical_convstate
        explicit_conv2d = (activation_control or physical_convstate) and gdn_conv == 'conv2d'
        physical_history = physical_convstate or not use_native_tools
        depth = select_native_depth(cfg, work, 4) if use_native_tools else None
        export_config = work/'export.yaml'
        export_config.write_text(yaml.safe_dump(cfg))
        entry = ((source/EXPORTER).resolve(strict=True) if no_patch else
                 four_layer_entry(prepared, work, native_activations=native_activations, silu_only=silu_only,
                                  softplus_only=softplus_only, gdn_silu_only=gdn_silu_only, gdn_silu_site=gdn_silu_site,
                                  all_silu_no_gate_cast=all_silu_no_gate_cast, explicit_conv2d=explicit_conv2d,
                                  physical_convstate=physical_convstate, rms_layers=rms_layers,
                                  rms_silu_token_view=rms_silu_token_view))
        save_air_source(entry.parent, root/'air-source')
        argv = native_export_argv(entry, 4, export_config)
        env = os.environ.copy()
        trace = work/'activations.txt'
        if no_patch:
            print(f'[L00-L03] --no-patch 原样使用 CPRO_SOURCE: {source}', flush=True)
        else:
            package = str(Path(__file__).resolve().parents[3]/'model')
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (package, env.get('PYTHONPATH'))))
            if physical_convstate:
                print('[L00-L03] ConvState：拼接旧状态与本次 QKV，末 4 行写回，末 67 行送卷积。\n'
                      '  默认对齐源码补丁：L3 RoPE 按 position_ids 取表，FP32 旋转 Q/K 后转回 hidden dtype；'
                      'W4 matmul 使用 FP16 权重 scale 和逐 token scale；GDR 使用 FP16 算子端口；'
                      'MoE 使用量化后的 INT8 激活并对齐门控 dtype。', flush=True)
            if physical_convstate or silu_only or restrict_silu_to_gdn or gate_control:
                with (case/'source.diff').open('w', encoding='utf-8') as changes:
                    for name in (EXPORTER, MODELING, WRAPPER, 'utils.py'):
                        base = source/name
                        before = base.read_text().splitlines(keepends=True) if base.is_file() else []
                        after = (entry.parent/name).read_text().splitlines(keepends=True)
                        changes.writelines(unified_diff(before, after, fromfile=f'base/{name}', tofile=f'export/{name}'))
                print(f'[L00-L03] 与 {source} 的实际源码差异将保存在 export/package/{dest.name}/source.diff', flush=True)
            if native_activations:
                print(f'[L00-L03] {prepared}：保留对齐修复，使用原生 SiLU 和 Softplus', flush=True)
            else:
                env['QWEN35_SILU_SHAPES'] = str(trace)
                if silu_only:
                    print(f'[L00-L03] {prepared}：直接调用 torch.ops.npu.silu_patch，FP16 输入和输出；使用原生 Softplus', flush=True)
                elif rms_silu_token_view:
                    print('[L00-L03] L0–L2 RMS 门控用 FP16 SiluPatch [64,4096]，恢复 head 视图后做 FP32 乘法', flush=True)
                elif rms_layers is not None:
                    print(f'[L00-L03] RMS 门控 SiluPatch 层={rms_silu_layers}，其余 RMS 门控使用 FP16 F.silu。'
                          '卷积、MoE 使用 SiluPatch，GDN 衰减使用 SoftplusPatch。', flush=True)
                elif all_silu_no_gate_cast:
                    print('[L00-L03] GDN 和 MoE 全部使用 SiluPatch；GDN 门控的 FP16 输出直接参与 RMSNorm 乘法。'
                          '衰减分支使用 SoftplusPatch。', flush=True)
                elif softplus_only:
                    print(f'[L00-L03] {prepared}：使用原生 SiLU 和 FP16 SoftplusPatch', flush=True)
                elif restrict_silu_to_gdn:
                    sites = {'conv': '卷积后', 'gate': 'RMS 门控'}.get(gdn_silu_site, '卷积后和 RMS 门控')
                    print(f'[L00-L03] GDN 仅在{sites}使用 FP16 SiluPatch，MoE 共享和路由专家使用原生 SiLU，'
                          'GDN 衰减使用 FP16 SoftplusPatch。', flush=True)
                else:
                    print(f'[L00-L03] {prepared}：使用 FP16 SiluPatch 和 FP16 SoftplusPatch', flush=True)
        print(f'[L00-L03] 使用原 CPro wrapper 导出四层 T64 AIR\n(cd {work} && {shlex.join(argv)})', flush=True)
        subprocess.run(argv, cwd=work, env=env, check=True)
        (work/EXPORT_DIR).rename(case/EXPORT_DIR)
        air = case/EXPORT_DIR/'export.air'
        boundary = inspect_air(air, layers=4, physical_history=physical_history, require_control=not physical_history,
                               conv2d_rows=64 if explicit_conv2d else None)
        if physical_convstate:
            print(f'[L00-L03] AIR 实测 ConvState：原更新算子 {boundary["convstate_update_nodes"]} 个，'
                  f'显式拼接的缓存 {boundary["physical_conv_caches"]} 份', flush=True)
        if no_patch:
            observations = dict(no_patch=True, operator_counts=boundary['operator_counts'])
            print(f'[L00-L03] 原样导出 AIR 实测节点数: {boundary["operator_counts"]}', flush=True)
        elif native_activations:
            counts = boundary['operator_counts']
            print(f'[L00-L03] 原生激活 AIR 节点数: {counts}', flush=True)
            if counts['SiluPatch'] or counts['SoftplusPatch']:
                raise ValueError(f'Custom activation remains in native-activation AIR: {counts}')
            observations = dict(native_activations=True, operator_counts=counts)
        else:
            observations = inspect_activation_air(air, trace, layers=4, silu=not softplus_only, softplus=not silu_only,
                                                  check_convs=not use_native_tools, gdn_conv=gdn_conv,
                                                  moe_silu=not restrict_silu_to_gdn, gdn_silu_site=gdn_silu_site,
                                                  all_silu_no_gate_cast=all_silu_no_gate_cast or rms_layers is not None,
                                                  rms_gate_count=None if rms_layers is None else len(rms_layers),
                                                  rms_silu_token_view=rms_silu_token_view)
            if activation_control or physical_convstate:
                counts = boundary['operator_counts']
                if gdn_silu_only:
                    observations['gdn_silu_only'] = True
                if gdn_silu_site:
                    observations['gdn_silu_site'] = gdn_silu_site
                if all_silu_no_gate_cast:
                    observations['all_silu_no_gate_cast'] = True
                if rms_silu_token_view:
                    observations['rms_silu_token_view'] = True
                if rms_layers is not None:
                    observations['rms_silu_layers'] = rms_silu_layers
                if silu_only or softplus_only:
                    observations['softplus_only' if softplus_only else 'silu_only'] = True
                observations['operator_counts'] = counts
                print(f'[L00-L03] AIR 实测 SiluPatch={counts["SiluPatch"]}, SoftplusPatch={counts["SoftplusPatch"]}',
                      flush=True)
        if not no_patch:
            if explicit_conv2d:
                observations['gdn_convs'] = boundary['gdn_convs']
            precision = four_precision(air, check_shared_gates=True)
            if precision['mismatches']:
                raise ValueError('\n'.join(precision['mismatches']))
            observations['precision'] = precision
        control = None if physical_history else os.getenv('DECODER_CONV_CONTROL')
        measured = prefill_inputs(case/'inputs', fixture, cfg, control, physical_history=physical_history,
                                  decoder_fixture=True)
        files = files_under(case, case)
        by_path = {r['path']: r for r in files}
        names, feeds = {}, {}
        for port in boundary['inputs']:
            raw = GE_PROTO[port['dtype']]
            dtype = {'FP16': 'float16', 'FP32': 'float32'}.get(raw, raw.lower())
            filename = original_filename(port['name'], port, dtype, fresh_zero_states=True)
            if physical_history and filename == 'conv_state_pos.bin':
                raise ValueError(f'{air}: physical ConvState AIR still exposes {port["name"]} as INT32[128]')
            row = by_path['inputs/' + filename]
            if row['bytes'] != math.prod(port['shape']) * np.dtype(dtype).itemsize:
                raise ValueError(f'{filename} does not match native AIR port {port}')
            role = Path(filename).stem
            feeds[role] = dict(row, shape=port['shape'], dtype=dtype)
            names[port['name']] = role
        outputs = [dict(name=row['role'], edge=row['edge']) for row in boundary['outputs']]
        save_json(case/'case.json', dict(air_directory=EXPORT_DIR, layers=ref['layers'], argv=argv,
                  gdn_conv=gdn_conv, native_convstate=not physical_history, native_tool_args=use_native_tools,
                  physical_convstate=physical_convstate, **({'export_configuration': depth} if depth else {}),
                  inputs=names, input_files=feeds, outputs=outputs, files=files, fixture=measured,
                  **observations))
        case.rename(dest)
    print(f'{dest.name} 原生 AIR 已导出，{len(names)} 个输入，输出 {[r["name"] for r in outputs]}', flush=True)

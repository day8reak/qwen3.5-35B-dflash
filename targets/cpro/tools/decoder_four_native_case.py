"""Package the owner's quantized four-layer T64 export for the decoder comparison."""
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
from qwen35_moe_dflash.target.prefill_t64.chain_air import checkpoint_paths
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import case_name
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import inspect_air
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import MODELING, inspect_activation_air, restore_native_activations
from qwen35_moe_dflash.target.prefill_t64.source_model import native_gdn_with_conv2d


def four_layer_entry(prepared, work, *, native_activations=False, silu_only=False, softplus_only=False, explicit_conv2d=False):
    entry = prepared/EXPORTER
    text = entry.read_text()
    # Editable sources generated before multi-depth export counted 30 GDR calls even at L=4.
    old = 'with cpro_export_contracts(True, chunk_size=64, gdn_layers=30):'
    restore_activations = native_activations or silu_only or softplus_only
    if old not in text and not restore_activations:
        return entry
    overlay = work/'source'
    overlay.mkdir()
    for path in prepared.iterdir():
        if restore_activations and path.name == 'models':
            models = overlay/'models'
            models.mkdir()
            for module in path.iterdir():
                if module.name not in (Path(MODELING).name, '__pycache__'):
                    (models/module.name).symlink_to(module.resolve(), target_is_directory=module.is_dir())
            text, model = restore_native_activations(text, (prepared/MODELING).read_text(),
                                                     keep_silu=silu_only, keep_softplus=softplus_only)
            if explicit_conv2d:
                model = native_gdn_with_conv2d(model)
            (overlay/MODELING).write_text(model)
        elif path.name not in (EXPORTER, '__pycache__'):
            (overlay/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (overlay/EXPORTER).write_text(text.replace(old, old.replace('gdn_layers=30', 'gdn_layers=3')))
    return overlay/EXPORTER


def export(root, fixture, ref, source, config, prepared, *, gdn_conv, no_patch=False, native_activations=False,
           silu_only=False, softplus_only=False):
    dest = root/case_name(ref['layers'])
    if dest.is_dir():
        saved = json.loads((dest/'case.json').read_bytes())
        if (saved.get('air_directory') != EXPORT_DIR or saved.get('gdn_conv', 'conv2d') != gdn_conv
                or saved.get('no_patch', False) != no_patch
                or saved.get('native_activations', False) != native_activations
                or saved.get('silu_only', False) != silu_only or saved.get('softplus_only', False) != softplus_only):
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
        activation_control = native_activations or silu_only or softplus_only
        explicit_conv2d = activation_control and gdn_conv == 'conv2d'
        physical_history = gdn_conv != 'original' and not activation_control
        depth = select_native_depth(cfg, work, 4) if not physical_history else None
        export_config = work/'export.yaml'
        export_config.write_text(yaml.safe_dump(cfg))
        entry = ((source/EXPORTER).resolve(strict=True) if no_patch else
                 four_layer_entry(prepared, work, native_activations=native_activations, silu_only=silu_only,
                                  softplus_only=softplus_only, explicit_conv2d=explicit_conv2d))
        argv = native_export_argv(entry, 4, export_config)
        env = os.environ.copy()
        trace = work/'activations.txt'
        if no_patch:
            print(f'[L00-L03] --no-patch 原样使用 CPRO_SOURCE: {source}', flush=True)
        else:
            package = str(Path(__file__).resolve().parents[3]/'model')
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (package, env.get('PYTHONPATH'))))
            if native_activations:
                print(f'[L00-L03] {prepared}：保留对齐修复，使用原生 SiLU 和 Softplus', flush=True)
            else:
                env['QWEN35_SILU_SHAPES'] = str(trace)
                if silu_only:
                    print(f'[L00-L03] {prepared}：使用 FP16 SiluPatch 和原生 Softplus', flush=True)
                elif softplus_only:
                    print(f'[L00-L03] {prepared}：使用原生 SiLU 和 FP16 SoftplusPatch', flush=True)
        print(f'[L00-L03] 使用原 CPro wrapper 导出四层 T64 AIR\n(cd {work} && {shlex.join(argv)})', flush=True)
        subprocess.run(argv, cwd=work, env=env, check=True)
        (work/EXPORT_DIR).rename(case/EXPORT_DIR)
        air = case/EXPORT_DIR/'export.air'
        boundary = inspect_air(air, layers=4, physical_history=physical_history, require_control=not physical_history,
                               conv2d_rows=64 if explicit_conv2d else None)
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
                                                  check_convs=physical_history, gdn_conv=gdn_conv)
            if silu_only or softplus_only:
                counts = boundary['operator_counts']
                observations['softplus_only' if softplus_only else 'silu_only'] = True
                observations['operator_counts'] = counts
                print(f'[L00-L03] AIR 实测 SiluPatch={counts["SiluPatch"]}, SoftplusPatch={counts["SoftplusPatch"]}',
                      flush=True)
        if not no_patch:
            if explicit_conv2d:
                observations['gdn_convs'] = boundary['gdn_convs']
            precision = four_precision(air)
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
            row = by_path['inputs/' + filename]
            if row['bytes'] != math.prod(port['shape']) * np.dtype(dtype).itemsize:
                raise ValueError(f'{filename} does not match native AIR port {port}')
            role = Path(filename).stem
            feeds[role] = dict(row, shape=port['shape'], dtype=dtype)
            names[port['name']] = role
        outputs = [dict(name=row['role'], edge=row['edge']) for row in boundary['outputs']]
        save_json(case/'case.json', dict(air_directory=EXPORT_DIR, layers=ref['layers'], argv=argv,
                  gdn_conv=gdn_conv, native_convstate=not physical_history, **({'export_configuration': depth} if depth else {}),
                  inputs=names, input_files=feeds, outputs=outputs, files=files, fixture=measured,
                  **observations))
        case.rename(dest)
    print(f'{dest.name} 原生 AIR 已导出，{len(names)} 个输入，输出 {[r["name"] for r in outputs]}', flush=True)

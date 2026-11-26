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
from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import inspect_activation_air


def four_layer_entry(prepared, work):
    entry = prepared/EXPORTER
    text = entry.read_text()
    # Editable sources generated before multi-depth export counted 30 GDR calls even at L=4.
    old = 'with cpro_export_contracts(True, chunk_size=64, gdn_layers=30):'
    if old not in text:
        return entry
    overlay = work/'source'
    overlay.mkdir()
    for path in prepared.iterdir():
        if path.name not in (EXPORTER, '__pycache__'):
            (overlay/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (overlay/EXPORTER).write_text(text.replace(old, old.replace('gdn_layers=30', 'gdn_layers=3')))
    return overlay/EXPORTER


def export(root, fixture, ref, source, config, prepared, *, gdn_conv):
    dest = root/case_name(ref['layers'])
    if dest.is_dir():
        saved = json.loads((dest/'case.json').read_bytes())
        if saved.get('air_directory') != EXPORT_DIR:
            raise ValueError(f'{dest}: expected the native four-layer export')
        for row in saved['files']:
            sized_file(dest, row)
        return
    cfg = yaml.safe_load(config.read_text())
    cfg.update({k: str(v) for k, v in checkpoint_paths(source, config, ref).items()})
    for key in ('embedding_weight_path', 'embedding_scale_path'):
        cfg[key] = str((source/cfg[key]).resolve(strict=True))
    with tempfile.TemporaryDirectory(dir=root.parent, prefix='native-four-') as tmp:
        work = Path(tmp)
        case = work/'case'
        case.mkdir()
        physical_history = gdn_conv != 'original'
        depth = select_native_depth(cfg, work, 4) if not physical_history else None
        export_config = work/'export.yaml'
        export_config.write_text(yaml.safe_dump(cfg))
        argv = native_export_argv(four_layer_entry(prepared, work), 4, export_config)
        trace = work/'activations.txt'
        package = str(Path(__file__).resolve().parents[3]/'model')
        env = dict(os.environ, QWEN35_SILU_SHAPES=str(trace),
                   PYTHONPATH=os.pathsep.join(filter(None, (package, os.getenv('PYTHONPATH')))))
        print(f'[L00-L03] 使用原 CPro wrapper 导出四层 T64 AIR\n(cd {work} && {shlex.join(argv)})', flush=True)
        subprocess.run(argv, cwd=work, env=env, check=True)
        (work/EXPORT_DIR).rename(case/EXPORT_DIR)
        air = case/EXPORT_DIR/'export.air'
        boundary = inspect_air(air, layers=4, physical_history=physical_history, require_control=not physical_history)
        activation = inspect_activation_air(air, trace, layers=4, softplus=True, check_convs=physical_history, gdn_conv=gdn_conv)
        precision = four_precision(air)
        if precision['mismatches']:
            raise ValueError('\n'.join(precision['mismatches']))
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
                  gdn_conv=gdn_conv, **({'export_configuration': depth} if depth else {}),
                  inputs=names, input_files=feeds, outputs=outputs, files=files, fixture=measured,
                  precision=precision, **activation))
        case.rename(dest)
    print(f'{dest.name} 原生 AIR 已导出，{len(names)} 个输入，输出 {[r["name"] for r in outputs]}', flush=True)

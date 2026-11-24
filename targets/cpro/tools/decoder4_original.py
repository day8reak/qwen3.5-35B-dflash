#!/usr/bin/env python3
"""Run the owner's unmodified four-layer Prefill64 exporter and replay its native I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))

from qwen35_moe_dflash.target.layer_probe.common import checked, identity, save_json, sha
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.toolchain.air_input_shape import inspect_input_shape

EXPORTER = 'export_glm_model_qwen3_5_moe.py'
EXPORT_DIR = 'glm_export_air_cpro_4block_64_quant'
KIND = 'original-cpro-four-layer-prefill64'


def valid19_inputs(folder, reference, cfg, conv_control):
    import numpy as np
    from qwen35_moe_dflash.target.layer_probe.common import read_array
    from qwen35_moe_dflash.target.prefill_t64.chain_reference import BASIS, VALID19_IDS_SHA256

    meta = json.loads((reference / 'manifest.json').read_bytes())
    if (meta.get('reference'), meta.get('status'), meta.get('physical_rows'), meta.get('valid_rows'),
            meta.get('output_pos')) != (BASIS, 'CAPTURED', 64, 19, 18):
        raise ValueError(f'Expected a measured T64/valid19 reference: {reference}')
    def tensor(name):
        return read_array(reference, meta['inputs'][name])

    ids = tensor('input_ids').astype('<i4')
    if ids.shape != (1, 64) or hashlib.sha256(ids.tobytes()).hexdigest() != VALID19_IDS_SHA256:
        raise ValueError('Prompt differs from the valid19 comparison')
    vocab = int(cfg.get('vocab_size', 248320))
    emb = np.memmap(cfg['embedding_weight_path'], mode='r', dtype='i1').reshape(vocab, 2048)
    scales = np.memmap(cfg['embedding_scale_path'], mode='r', dtype='<f4').reshape(vocab, 1)
    quant_embed, embed_scale = emb[ids], scales[ids]
    positions = np.arange(64, dtype='<i4')
    if (not np.array_equal(tensor('position_ids').reshape(-1), positions) or
            not np.array_equal(tensor('new_kv_cache_pos').reshape(-1), positions) or
            tensor('effective_length').tolist() != [19]):
        raise ValueError('Reference positions or effective length differ from fresh prefill')
    mask = tensor('attention_mask').astype('<f4')
    allowed = (positions[:, None] < 19) & (np.arange(2048)[None] <= positions[:, None])
    if mask.shape != (1, 1, 64, 2048) or not np.array_equal(mask[0, 0] == 0, allowed):
        raise ValueError('Expected the retained valid19 causal mask')
    if conv_control:
        ctrl = np.fromfile(conv_control, dtype='<i4')
        if ctrl.shape != (128,) or conv_control.stat().st_size != 512:
            raise ValueError('Expected 128 INT32 values for the CPro ConvState control')
        policy = f'provided {conv_control}'
    else:
        ctrl = np.zeros(128, dtype='<i4')
        ctrl[0] = 19
        policy = 'trial [19, 0 x 127] derived from observed valid64 [64, 0 x 127]; valid19 semantics unverified'
    arrays = dict(inputs_embeds=quant_embed, embed_scale=embed_scale, attention_mask=mask,
                  position_ids=positions[None], new_kv_cache_pos=positions,
                  output_pos=np.array([18], dtype='<i4'), allQLen=np.array([19], dtype='<i8'),
                  effective_length=np.array([19], dtype='<i2'), conv_state_pos=ctrl,
                  past_conv_state=np.zeros((1, 8192, 4), dtype='<f2'),
                  past_recurrent_state=np.zeros((1, 32, 128, 128), dtype='<f2'),
                  past_kv_zero=np.zeros((64, 32, 32, 16), dtype='<f2'))
    folder.mkdir()
    for name, array in arrays.items():
        array.tofile(folder / (name + '.bin'))
    print(f'Original runtime input fixture: valid19, output_pos=18, ConvState: {policy}', flush=True)
    measured = tensor('embedding')
    if measured.shape != (1, 64, 2048) or measured.dtype != np.float16:
        raise ValueError('Expected the measured FP16 decoder entry [1,64,2048]')
    # Compare at the diagnostic graph's FP16 boundary. The original graph still receives both raw files.
    entry = (quant_embed.astype(np.float32) * embed_scale).astype(np.float16)
    changes = int(np.count_nonzero(entry[:, :19].view(np.uint16) != measured[:, :19].view(np.uint16)))
    print(f'Original embedding * scale, rounded to FP16 vs 310P layer entry: {changes}/{19*2048} changed (valid19)')
    return dict(reference_sha256=sha(reference / 'manifest.json'), token_sha256=VALID19_IDS_SHA256,
                valid19_embedding_changed_fp16_elements=changes, conv_state_control=policy)


def export(run, source, config, reference, conv_control):
    import yaml
    source = source.resolve(strict=True)
    config = (source / config).resolve(strict=True)
    configured = yaml.safe_load(config.read_text())
    cfg = dict(configured)
    entry = source / EXPORTER
    sources = {n: sha(source / n) for n in (EXPORTER, 'models/modeling_qwen3_5_moe_hiai_cpro.py',
               'models/export_model_wrapper_qwen3_5_moe.py', 'utils.py')}
    argv = [sys.executable, '-u', str(entry), 'mode=export', 'quant_mode=enable',
            'num_hidden_layers=4', 'seq_len=64', 'topK=16', 'config=' + str(config)]
    run.mkdir(parents=True, exist_ok=True)
    dest = run / 'air'
    if dest.exists():
        raise FileExistsError(f'Original export already exists: {dest}. Select a new ORIGINAL4_RUN to re-export.')
    # The owner's exporter deletes its relative export directory on startup.
    with tempfile.TemporaryDirectory(dir=run, prefix='export-') as temp:
        work = Path(temp)
        # Resolve relative resource paths like the original project cwd while isolating new output.
        for key in ('model_file', 'quanted_pth', 'embedding_weight_path', 'embedding_scale_path'):
            original = Path(cfg[key])
            real = (source / original).resolve(strict=True)
            if not original.is_absolute():
                cfg[key] = str(real)
        if cfg != configured:
            resolved = work / 'export-config.yaml'
            resolved.write_text(yaml.safe_dump(cfg))
            argv[-1] = 'config=' + str(resolved)
        fixture = valid19_inputs(work / 'inputs', reference, cfg, conv_control)
        print(f'(cd {work} && {shlex.join(argv)})', flush=True)
        subprocess.run(argv, cwd=work, check=True)
        air = work / EXPORT_DIR / 'export.air'
        abi = inspect_input_shape(air)
        files = [identity(p, work) for p in sorted(air.parent.rglob('*')) if p.is_file()]
        files.extend(identity(p, work) for p in sorted((work / 'inputs').glob('*.bin')))
        save_json(work / 'manifest.json', dict(kind=KIND, argv=argv, sources=sources, fixture=fixture,
                  config_sha256=sha(config), inputs=abi['inputs'], files=files))
        bundle = work / 'bundle'
        bundle.mkdir()
        for name in (EXPORT_DIR, 'inputs', 'manifest.json'):
            (work / name).rename(bundle / name)
        bundle.rename(dest)
    share_for_transfer(dest)
    print(f'Original AIR: {dest / EXPORT_DIR / "export.air"}')


def ssh_bundle(client, remote, staged):
    from decoder4_prefill import ssh_file
    remote = Path(remote)
    staged.mkdir()
    manifest = staged / 'manifest.json'
    ssh_file(client, remote / manifest.name, manifest)
    meta = json.loads(manifest.read_bytes())
    if meta['kind'] != KIND:
        raise ValueError(f'Expected original four-layer AIR at {remote}')
    for row in meta['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts) < 2 or rel.parts[0] not in (EXPORT_DIR, 'inputs'):
            raise ValueError(f'Invalid original export path: {rel}')
        target = staged / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        ssh_file(client, remote / rel, target, row)


def build(run, ddk, diagnosis, remote):
    from decoder4_prefill import pull_air
    from qwen35_moe_dflash.toolchain.compile import omg_environment, prepare_tbe_compatibility, _omg_ipc_environment

    bundle = run / 'air'
    if not bundle.exists():
        pull_air(run, remote, ssh_download=ssh_bundle)
    meta = json.loads((bundle / 'manifest.json').read_bytes())
    if meta['kind'] != KIND:
        raise ValueError(f'Expected original four-layer export: {bundle}')
    for row in meta['files']:
        checked(bundle, row)
    dest = run / 'omc'
    out = Path(tempfile.mkdtemp(dir=run, prefix='omg-'))
    graph = bundle / EXPORT_DIR / 'export.air'
    shape = inspect_input_shape(graph)['input_shape']
    # Keep the directory component in the original FileConstant paths.
    (out / EXPORT_DIR).symlink_to(graph.parent, target_is_directory=True)
    omg, env = omg_environment(ddk)
    if diagnosis:
        env, _ = prepare_tbe_compatibility(ddk, diagnosis, out / 'tbe-compat', inherited=env)
    argv = [str(omg), '--framework=1', f'--model={EXPORT_DIR}/export.air', '--output=model',
            '--target=omc', '--platform=kirinx90', '--use_origin_format=true', '--weight_data_type=FP16',
            '--limit_partition_size=false', '--save_weights_as_external_data=true', '--input_shape=' + shape]
    print(f'(cd {out} && {shlex.join(argv)})\nOMG log: {out / "compile.log"}', flush=True)
    with _omg_ipc_environment(out, env) as (child, _), (out / 'compile.log').open('x') as log:
        subprocess.run(argv, cwd=out, env=child, stdout=log, stderr=subprocess.STDOUT, check=True)
    model, = out.rglob('*.omc')
    weights = [identity(p, out) for p in sorted(model.parent.rglob('*.weight'))]
    if not weights:
        raise ValueError(f'OMG did not produce the requested external weights: {out}')
    save_json(out / 'manifest.json', dict(air_manifest_sha256=sha(bundle / 'manifest.json'), argv=argv,
              primary=identity(model, out), weights=weights))
    out.rename(dest)
    print(f'Compiled original OMC: {dest / model.relative_to(out)}', flush=True)


def original_filename(name, src, dtype, *, fresh_zero_states):
    """Match the original exporter's descriptors, before singleton dimensions can fold."""
    dims = tuple(src['shape'])
    known = {
        ('int8', (1, 64, 2048)): 'inputs_embeds',
        ('float32', (1, 64, 1)): 'embed_scale',
        ('float32', (1, 1, 64, 2048)): 'attention_mask',
        ('int32', (1, 64)): 'position_ids', ('int32', (64,)): 'new_kv_cache_pos',
        ('int32', (1,)): 'output_pos', ('int64', (1,)): 'allQLen',
        ('int16', (1,)): 'effective_length', ('int32', (128,)): 'conv_state_pos',
    }
    if (dtype, dims) in known:
        return known[dtype, dims] + '.bin'
    if fresh_zero_states and dtype == 'float16':
        # All caches with the same view contain identical zero bytes at fresh prefill.
        empty = {(1, 8192, 4): 'past_conv_state', (1, 32, 128, 128): 'past_recurrent_state',
                 (64, 32, 32, 16): 'past_kv_zero'}
        if dims in empty:
            return empty[dims] + '.bin'
    raise ValueError(f'Cannot identify {name} {dtype}{dims}. Supply {name}.bin or an input-map JSON.')


def board(run, ddk, config, directory, input_map):
    import numpy as np
    from parse_model_json import GE_PROTO, dtype as ge_dtype, inspect, read_omc
    from qwen35_moe_dflash.target.prefill_smoke_plan import _nd_view_key
    from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import run_v2
    from qwen35_moe_dflash.execution.runtime_config import read_json
    from qwen35_moe_dflash.target.prefill_t64.chain_hilog import _window
    from qwen35_moe_dflash.execution.serial_transport import _environment
    from qwen35_moe_dflash.execution.hdc_run import Hdc
    from datetime import datetime

    compiled = read_json(run / 'omc/manifest.json')
    meta = read_json(run / 'air/manifest.json')
    if meta['kind'] != KIND or sha(run / 'air/manifest.json') != compiled['air_manifest_sha256']:
        raise ValueError('The compiled model belongs to a different original AIR bundle')
    model = checked(run / 'omc', compiled['primary'])
    weights = [checked(run / 'omc', row) for row in compiled['weights']]
    abi = inspect(read_omc(model, ddk_root=ddk)[0], enum=GE_PROTO)
    source = {s['name']: s for s in meta['inputs']}
    def specs(direction):
        rows = []
        for row in abi[direction]:
            if row['slot'] != len(rows) or not row['logical_bytes'] or row['issues']:
                raise ValueError(f'Incomplete compiled descriptor: {row}')
            dtype = {'FP16': 'float16', 'FP32': 'float32'}.get(row['dtype'], row['dtype'].lower())
            rows.append(dict(name=row['node'] if direction == 'inputs' else row['edge'],
                             shape=row['shape'], dtype=dtype, bytes=row['logical_bytes']))
        return rows

    inputs, outputs = specs('inputs'), specs('outputs')
    fresh = directory == (run / 'air/inputs').resolve()
    if fresh:
        for row in meta['files']:
            if Path(row['path']).parts[0] == 'inputs':
                checked(run / 'air', row)
    mapping = read_json(input_map) if input_map else {}
    if mapping and set(mapping) != {s['name'] for s in inputs}:
        raise ValueError('input-map must supply one filename for every compiled input node')
    for i, spec in enumerate(outputs):
        print(f"output {i}: {spec['name']} {spec['dtype']} {spec['shape']}", flush=True)
    paths = []
    for i, spec in enumerate(inputs):
        name = spec['name']
        if mapping:
            path = directory / mapping[name]
        elif (directory / (name + '.bin')).is_file():
            path = directory / (name + '.bin')
        else:
            if name not in source:
                raise ValueError(f'Missing AIR node {name}; supply the original input map')
            src = source[name]
            if _nd_view_key((ge_dtype(src['dtype_raw'], GE_PROTO), src['shape'])) != _nd_view_key(
                    (abi['inputs'][i]['dtype'], spec['shape'])):
                raise ValueError(f'AIR and compiled typed view differ for {name}: {src} vs {spec}')
            path = directory / original_filename(name, src, spec['dtype'], fresh_zero_states=fresh)
        if path.stat().st_size != spec['bytes']:
            raise ValueError(f'{name} {path}: expected {spec["bytes"]} bytes')
        if path.name == 'output_pos.bin':
            pos = int(np.fromfile(path, dtype='<i4').item())
            if not 0 <= pos < 64:
                raise ValueError(f'{path}: output_pos={pos}, expected a zero-based row in [0, 63]')
        paths.append(path)
        spec.update(sha256=sha(path), file=str(path))
        print(f"slot {i}: {name} {spec['dtype']} {spec['shape']} <- {path}", flush=True)
    runtime = read_json(config.parent / read_json(config)['assets'])['environment']
    out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    save_json(out / 'io.json', dict(inputs=inputs, outputs=outputs, order=abi['input_order_basis']))
    print(f'Original board attempt: {out}', flush=True)
    try:
        run_v2(out, 'original L0-L3', model, inputs, paths, outputs, runtime, external_weights=weights)
    except RuntimeError:
        receipt = read_json(out / 'result.json')
        if receipt.get('exitcode'):
            hdc = Hdc(*_environment(), out, 60)
            try:
                remote = receipt['remote']
                for filename in ('tool.stdout', 'tool.stderr'):
                    hdc.receive(remote + '/' + filename, out / filename)
                _, markers = hdc.shell(f'cd {remote} && stat -c "%y" tool.started tool.exitcode', 'times')
                start, end = [datetime.fromisoformat(t[:19]) for t in markers.splitlines()]
                from qwen35_moe_dflash.target.prefill_t64.chain_hilog import _snapshot
                _window(_snapshot(hdc, out, None), out / 'window.log', start, end)
                print(out / 'window.log')
            except Exception as diagnostic_error:
                print(f'Could not collect failure details: {diagnostic_error}', file=sys.stderr)
        raise
    print(f'Original four-layer prefill executed: {out / "result.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('air', 'compile', 'board'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--conv-control', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    parser.add_argument('--ddk', type=Path)
    parser.add_argument('--tbe-diagnosis', type=Path)
    parser.add_argument('--remote-air', type=Path)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--input-map', type=Path)
    args = parser.parse_args()
    run = output_path(args.run)
    if args.action == 'air':
        if not args.source or not args.reference:
            parser.error('air requires --source and --reference')
        export(run, args.source, args.config, args.reference.resolve(), args.conv_control)
        return
    if not args.ddk:
        parser.error('compile and board require --ddk')
    if not (run / 'omc/manifest.json').exists():
        build(run, args.ddk.resolve(), args.tbe_diagnosis, args.remote_air or run / 'air')
    if args.action == 'board':
        if not args.runtime_config:
            parser.error('board requires --runtime-config')
        directory = args.inputs or run / 'air/inputs'
        board(run, args.ddk.resolve(), args.runtime_config.resolve(), directory.resolve(), args.input_map)


if __name__ == '__main__':
    main()

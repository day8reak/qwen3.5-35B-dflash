#!/usr/bin/env python3
"""Export and replay the native four-layer Prefill64 graph, optionally using SiluPatch."""
import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))

from qwen35_moe_dflash.target.layer_probe.common import checked, identity, save_json, sha
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.common.tool_logs import log_excerpt
from qwen35_moe_dflash.toolchain.air_input_shape import inspect_input_shape

EXPORTER = 'export_glm_model_qwen3_5_moe.py'
EXPORT_DIR = 'glm_export_air_cpro_4block_64_quant'
KIND = 'original-cpro-four-layer-prefill64'
VECTOR_PATCH_KIND = KIND + '-silu-patch'
VECTOR_ALIGNED_KIND = KIND + '-aligned-silu-patch'
PATCH_KIND = VECTOR_PATCH_KIND + '-native-shape'
ALIGNED_KIND = VECTOR_ALIGNED_KIND + '-native-shape'
REFERENCE_INPUT_NAMES = ('embedding', 'position_ids', 'new_kv_cache_pos', 'effective_length', 'attention_mask')


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
                reference_inputs={k: meta['inputs'][k]['sha256'] for k in REFERENCE_INPUT_NAMES},
                valid19_embedding_changed_fp16_elements=changes, conv_state_control=policy)


def export(run, source, config, reference, conv_control, *, kind=KIND, baseline=None):
    import yaml
    default_config = Path('config/qwen3_5_moe.yaml')
    if kind != KIND:
        base_manifest = baseline / 'air/manifest.json'
        base = json.loads(base_manifest.read_bytes())
        if base['kind'] != KIND:
            raise ValueError(f'Expected the native CPro export in {baseline}')
        if source is None:
            source = Path(next(v for v in base['argv'] if Path(v).name == EXPORTER)).parent
        if config is None:
            saved = next(v.split('=', 1)[1] for v in base['argv'] if v.startswith('config='))
            choices = [Path(base.get('config_path', saved)), source / default_config]
            config = next((p.resolve() for p in choices if p.is_file() and sha(p) == base['config_sha256']), None)
            if config is None:
                raise ValueError('Set TARGET_CONFIG (or --config) to the YAML used by the successful original export')
    source = source.resolve(strict=True)
    config = (source / (config or default_config)).resolve(strict=True)
    configured = yaml.safe_load(config.read_text())
    cfg = dict(configured)
    entry = source / EXPORTER
    sources = {n: sha(source / n) for n in (EXPORTER, 'models/modeling_qwen3_5_moe_hiai_cpro.py',
               'models/modeling_qwen3_5_moe_hiai_nz.py',
               'models/export_model_wrapper_qwen3_5_moe.py', 'utils.py')}
    run.mkdir(parents=True, exist_ok=True)
    dest = run / 'air'
    if dest.exists():
        raise FileExistsError(f'Export already exists: {dest}. Use the existing AIR or choose a new run directory.')
    if kind != KIND:
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import inspect_silu_air, stage
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import inspect_air, same_boundary
        if base['sources'] != sources or base['config_sha256'] != sha(config):
            raise ValueError(f'Source/config differs from the successful original at {baseline}')
        graph_path = EXPORT_DIR + '/export.air'
        original = checked(base_manifest.parent, next(row for row in base['files'] if row['path'] == graph_path))
        base_boundary = inspect_air(original)
        staged = Path(tempfile.mkdtemp(dir=run, prefix='silu-source-'))
        entry = stage(source, staged, aligned=kind == ALIGNED_KIND)
        print(f'SiluPatch model and exporter for review: {staged}', flush=True)
    # The owner's exporter deletes its relative export directory on startup.
    with tempfile.TemporaryDirectory(dir=run, prefix='export-') as temp:
        work = Path(temp)
        # Resolve relative resource paths like the original project cwd while isolating new output.
        for key in ('model_file', 'quanted_pth', 'embedding_weight_path', 'embedding_scale_path'):
            original = Path(cfg[key])
            real = (source / original).resolve(strict=True)
            if not original.is_absolute():
                cfg[key] = str(real)
        export_config = config
        if cfg != configured:
            export_config = work / 'export-config.yaml'
            export_config.write_text(yaml.safe_dump(cfg))
        argv = [sys.executable, '-u', str(entry), 'mode=export', 'quant_mode=enable',
                'num_hidden_layers=4', 'seq_len=64', 'topK=16', f'config={export_config}', 'quant_bits=4']
        if kind == KIND:
            fixture = valid19_inputs(work / 'inputs', reference, cfg, conv_control)
        else:
            inputs = work / 'inputs'
            inputs.mkdir()
            for row in base['files']:
                rel = Path(row['path'])
                if rel.parent == Path('inputs'):
                    shutil.copyfile(checked(base_manifest.parent, row), inputs / rel.name)
            if conv_control and sha(conv_control) != sha(inputs / 'conv_state_pos.bin'):
                raise ValueError('ConvState control differs from the original fixture')
            fixture = dict(base['fixture'])
            if reference:
                prior = reference / 'manifest.json'
                if sha(prior) != fixture['reference_sha256']:
                    raise ValueError(f'{reference} differs from the measured reference of the original export')
                measured = json.loads(prior.read_bytes())['inputs']
                fixture['reference_inputs'] = {k: measured[k]['sha256'] for k in REFERENCE_INPUT_NAMES}
            print(f'Reusing original valid19 input files from {base_manifest.parent / "inputs"}', flush=True)
        print(f'(cd {work} && {shlex.join(argv)})', flush=True)
        env = os.environ.copy()
        shape_trace = work / 'silu-shapes.txt'
        if kind != KIND:
            env['QWEN35_SILU_SHAPES'] = str(shape_trace)
        subprocess.run(argv, cwd=work, env=env, check=True)
        air = work / EXPORT_DIR / 'export.air'
        variant = {}
        if kind != KIND:
            variant['silu_patch'] = inspect_silu_air(air, shape_trace)
            boundary = inspect_air(air)
            same_boundary(base_boundary, boundary)
            variant.update(boundary=boundary, original_manifest_sha256=sha(base_manifest))
            print(f'Native AIR boundary matched: {len(boundary["inputs"])} inputs, '
                  f'{[row["role"] for row in boundary["outputs"]]}', flush=True)
            variant['silu_patch']['source_sha256'] = {n: sha(staged / n) for n in (
                EXPORTER, 'models/modeling_qwen3_5_moe_hiai_cpro.py')}
        abi = inspect_input_shape(air)
        files = [identity(p, work) for p in sorted(air.parent.rglob('*')) if p.is_file()]
        files.extend(identity(p, work) for p in sorted((work / 'inputs').glob('*.bin')))
        if kind != KIND:
            recorded = {r['path']: r['sha256'] for r in base['files'] if r['path'].startswith('inputs/')}
            current = {r['path']: r['sha256'] for r in files if r['path'].startswith('inputs/')}
            if recorded != current:
                raise ValueError('The generated input files differ from the original four-layer fixture')
        save_json(work / 'manifest.json', dict(kind=kind, **variant,
                  argv=argv, sources=sources, fixture=fixture,
                  config_path=str(config), config_sha256=sha(config), inputs=abi['inputs'], files=files))
        bundle = work / 'bundle'
        bundle.mkdir()
        for name in (EXPORT_DIR, 'inputs', 'manifest.json'):
            (work / name).rename(bundle / name)
        bundle.rename(dest)
    share_for_transfer(dest)
    print(f'{kind} AIR: {dest / EXPORT_DIR / "export.air"}')


def ssh_bundle(client, remote, staged, *, kind=KIND):
    from decoder4_prefill import ssh_file
    remote = Path(remote)
    staged.mkdir()
    manifest = staged / 'manifest.json'
    ssh_file(client, remote / manifest.name, manifest)
    meta = json.loads(manifest.read_bytes())
    if meta['kind'] != kind:
        raise ValueError(f'Expected {kind} AIR at {remote}, got {meta["kind"]}')
    for row in meta['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts) < 2 or rel.parts[0] not in (EXPORT_DIR, 'inputs'):
            raise ValueError(f'Invalid original export path: {rel}')
        target = staged / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        ssh_file(client, remote / rel, target, row)


def compile_log_text(path):
    excerpt = log_excerpt(path)
    if excerpt == 'Log tail:\n':
        excerpt = 'OMG produced an empty compile.log.'
    return f'OMG log: {path}\n{excerpt}'


def compile_omg(out, air_path, ddk, diagnosis, platform):
    from qwen35_moe_dflash.toolchain.compile import omg_environment, prepare_tbe_compatibility, _omg_ipc_environment

    shape = inspect_input_shape(out / air_path)['input_shape']
    omg, env = omg_environment(ddk, soc_version=platform)
    if diagnosis:
        env, _ = prepare_tbe_compatibility(ddk, diagnosis, out / 'tbe-compat', inherited=env)
    platform_args = [f'--platform={platform}'] if platform is not None else []
    argv = [str(omg), '--framework=1', f'--model={air_path}', '--output=model',
            '--target=omc', *platform_args, '--use_origin_format=true', '--weight_data_type=FP16',
            '--limit_partition_size=false', '--save_weights_as_external_data=true', '--input_shape=' + shape]
    log_path = out / 'compile.log'
    command = f'(cd {shlex.quote(str(out))} && {shlex.join(argv)})'
    invocation = f'{command}\nOMG SOC_VERSION={env.get("SOC_VERSION", "<unset>")}'
    print(f'{invocation}\nOMG log: {log_path}', flush=True)
    with _omg_ipc_environment(out, env) as (child, _), log_path.open('x') as log:
        print(invocation, file=log, flush=True)
        result = subprocess.run(argv, cwd=out, env=child, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode, argv, env.get('SOC_VERSION')


def build(run, ddk, diagnosis, remote, *, bundle, kind, platform):
    from decoder4_prefill import pull_air

    run.mkdir(parents=True, exist_ok=True)
    if not bundle.exists():
        with tempfile.TemporaryDirectory(dir=bundle.parent, prefix='.fetch-') as tmp:
            pull_air(Path(tmp), remote, ssh_download=partial(ssh_bundle, kind=kind))
            incoming = Path(tmp) / 'air'
            if json.loads((incoming / 'manifest.json').read_bytes())['kind'] != kind:
                raise ValueError(f'Expected {kind} AIR at {remote}')
            incoming.rename(bundle)
    meta = json.loads((bundle / 'manifest.json').read_bytes())
    if meta['kind'] != kind:
        raise ValueError(f'Expected {kind} AIR at {bundle}, got {meta["kind"]}')
    for row in meta['files']:
        checked(bundle, row)
    local_air = run / 'air'
    if local_air != bundle:
        try:
            local_air.symlink_to(os.path.relpath(bundle, run), target_is_directory=True)
        except FileExistsError:
            if not local_air.samefile(bundle):
                raise ValueError(f'{local_air} must refer to the selected AIR bundle {bundle}')
    dest = run / 'omc'
    out = Path(tempfile.mkdtemp(dir=run, prefix='omg-'))
    # Keep the directory component in the original FileConstant paths.
    (out / EXPORT_DIR).symlink_to(bundle / EXPORT_DIR, target_is_directory=True)
    code, argv, soc = compile_omg(out, EXPORT_DIR + '/export.air', ddk, diagnosis, platform)
    if code:
        raise RuntimeError(f'{kind}: OMG exited {code}\n{compile_log_text(out / "compile.log")}')
    model, = out.rglob('*.omc')
    weights = [identity(p, out) for p in sorted(model.parent.rglob('*.weight'))]
    if not weights:
        raise ValueError(f'OMG did not produce the requested external weights: {out}')
    save_json(out / 'manifest.json', dict(air_manifest_sha256=sha(bundle / 'manifest.json'), argv=argv,
              soc_version=soc,
              primary=identity(model, out), weights=weights))
    out.rename(dest)
    print(f'Compiled {kind} OMC: {dest / model.relative_to(out)}', flush=True)


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


def board(run, ddk, config, directory, input_map, *, kind=KIND, baseline=None):
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
    if meta['kind'] != kind or sha(run / 'air/manifest.json') != compiled['air_manifest_sha256']:
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
    original_inputs = {}
    if baseline:
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import inspect_air, latest_replay, output_roles, view
        prior = latest_replay(baseline)
        original_io = read_json(prior / 'io.json')
        receipt = read_json(prior / 'result.json')
        original_omc = read_json(prior.parent / 'omc/manifest.json')
        if (sha(prior.parent / 'air/manifest.json') != meta['original_manifest_sha256']
                or original_omc['air_manifest_sha256'] != meta['original_manifest_sha256']
                or original_omc['primary']['sha256'] != receipt['omc_sha256']
                or [r['sha256'] for r in original_io['inputs']] != receipt['input_hashes']):
            raise ValueError(f'Original replay is not bound to the expected baseline: {prior}')
        original_inputs = {r['name']: r for r in original_io['inputs']}
        def descriptors(rows):
            return {r['name']: (r['dtype'], view(r['shape']), r['bytes']) for r in rows}
        if descriptors(inputs) != descriptors(original_io['inputs']):
            raise ValueError('Compiled four-layer inputs differ from the successful original replay')
        def output_types(rows, boundary):
            return {role: (r['dtype'], view(r['shape']), r['bytes']) for role, r in
                    zip(output_roles(rows, boundary['outputs']), rows)}
        original_boundary = inspect_air(prior.parent / 'air' / EXPORT_DIR / 'export.air')
        if output_types(outputs, meta['boundary']) != output_types(original_io['outputs'], original_boundary):
            raise ValueError('Compiled four-layer outputs differ from the successful original replay')
        print(f'Compared compiled ports with {prior}. Reusing its input hashes.', flush=True)
    for i, spec in enumerate(outputs):
        print(f"output {i}: {spec['name']} {spec['dtype']} {spec['shape']}", flush=True)
    paths = []
    for i, spec in enumerate(inputs):
        name = spec['name']
        if fresh and not mapping and name in original_inputs and Path(original_inputs[name]['file']).is_file():
            path = Path(original_inputs[name]['file'])
        elif mapping:
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
        if original_inputs and spec['sha256'] != original_inputs[name]['sha256']:
            raise ValueError(f'{name}: input bytes differ from the successful original replay')
        print(f"slot {i}: {name} {spec['dtype']} {spec['shape']} <- {path}", flush=True)
    runtime = read_json(config.parent / read_json(config)['assets'])['environment']
    out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    save_json(out / 'io.json', dict(inputs=inputs, outputs=outputs, order=abi['input_order_basis']))
    print(f'{kind} board attempt: {out}', flush=True)
    try:
        run_v2(out, 'original L0-L3' if kind == KIND else kind, model, inputs, paths, outputs, runtime,
               external_weights=weights)
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
    print(f'{kind} executed: {out / "result.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('air', 'compile', 'board', 'log'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--conv-control', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--ddk', type=Path)
    parser.add_argument('--tbe-diagnosis', type=Path)
    parser.add_argument('--remote-air', type=Path)
    parser.add_argument('--runtime-config', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--input-map', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--silu-patch', action='store_true', help='change only the SiLU sites in the native CPro graph')
    mode.add_argument('--aligned', action='store_true', help='native boundary, reviewed RoPE/expert fixes and SiluPatch')
    parser.add_argument('--baseline', type=Path, help='native baseline run (or its replay for board comparison)')
    args = parser.parse_args()
    base = output_path(args.run)
    bundle = base / 'air'
    run = output_path(base / 'no-platform') if args.aligned and args.action != 'air' else base
    if args.action == 'log':
        for folder in (run, base):
            logs = [p for p in (*folder.glob('omg-*/compile.log'), folder / 'omc/compile.log') if p.is_file()]
            if logs:
                newest = max(logs, key=lambda p: (p.stat().st_mtime_ns, p))
                print(compile_log_text(newest))
                return
        parser.error(f'No OMG compile.log under {base}')
    kind = ALIGNED_KIND if args.aligned else PATCH_KIND if args.silu_patch else KIND
    if kind != KIND and args.action in ('air', 'board') and args.baseline is None:
        parser.error(f'variant {args.action} requires --baseline')
    if args.action == 'air':
        if kind == KIND and (not args.source or not args.reference):
            parser.error('native export requires --source and --reference')
        export(run, args.source, args.config, args.reference, args.conv_control, kind=kind, baseline=args.baseline)
        return
    if not args.ddk:
        parser.error('compile and board require --ddk')
    compiled = run / 'omc/manifest.json'
    if not compiled.exists():
        build(run, args.ddk.resolve(), args.tbe_diagnosis, args.remote_air or bundle, kind=kind,
              bundle=bundle, platform=None if args.aligned else 'kirinx90')
    else:
        manifest = bundle / 'manifest.json'
        record = json.loads(compiled.read_bytes())
        if (json.loads(manifest.read_bytes())['kind'] != kind or record['air_manifest_sha256'] != sha(manifest)
                or not (run / 'air').samefile(bundle)):
            raise ValueError(f'Cached OMC does not match the requested {kind} AIR at {run}')
        if args.aligned and any(arg.split('=', 1)[0] == '--platform' for arg in record['argv']):
            raise ValueError(f'Cached OMC at {run} was built with an explicit --platform')
        print(f'Reusing {kind} OMC from {compiled.parent}', flush=True)
    if args.action == 'board':
        if not args.runtime_config:
            parser.error('board requires --runtime-config')
        directory = args.inputs or run / 'air/inputs'
        board(run, args.ddk.resolve(), args.runtime_config.resolve(), directory.resolve(), args.input_map, kind=kind,
              baseline=args.baseline)


if __name__ == '__main__':
    main()

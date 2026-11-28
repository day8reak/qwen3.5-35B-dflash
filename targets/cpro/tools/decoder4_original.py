#!/usr/bin/env python3
"""Export and replay the original CPro Prefill64 graph with validated CPro adaptations."""
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

from qwen35_moe_dflash.target.layer_probe.common import checked, identity, save_json, sha, sized_file
from qwen35_moe_dflash.common.artifacts import files_under, output_path, share_for_transfer
from qwen35_moe_dflash.common.tool_logs import log_excerpt
from qwen35_moe_dflash.toolchain.air_input_shape import inspect_input_shape

EXPORTER = 'export_glm_model_qwen3_5_moe.py'
EXPORT_DIR = 'glm_export_air_cpro_4block_64_quant'
KIND = 'original-cpro-four-layer-prefill64'
VECTOR_PATCH_KIND = KIND + '-silu-patch'
VECTOR_ALIGNED_KIND = KIND + '-aligned-silu-patch'
PATCH_KIND = VECTOR_PATCH_KIND + '-native-shape-fp16'
ALIGNED_KIND = VECTOR_ALIGNED_KIND + '-native-shape-fp16'
L0_KIND = 'original-cpro-l0-prefill64-aligned-silu-patch-split-v3-fp16'
SPLIT_KIND = 'original-cpro-decoder-prefill64-aligned-silu-patch-split-v3-fp16'
L0_DIR = 'glm_export_air_cpro_1block_64_quant'
FULL_KIND = 'original-cpro-40-layer-prefill64-aligned-silu-softplus-fp16-conv2d'
TAPS_FULL_KIND = FULL_KIND.removesuffix('conv2d') + 'taps'
NATIVE_CONV_FULL_KIND = FULL_KIND.removesuffix('conv2d') + 'original'
FULL_KINDS = (FULL_KIND, TAPS_FULL_KIND, NATIVE_CONV_FULL_KIND)
FULL_DIR = 'glm_export_air_cpro_40block_64_quant'
REFERENCE_INPUT_NAMES = ('embedding', 'position_ids', 'new_kv_cache_pos', 'effective_length', 'attention_mask')


def four_precision(air, layers=4, *, check_shared_gates=False, rmsnorm_patch=False,
                   shared_gate_fp16_sigmoid=False, qlinear_scale_fp16=False):
    from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
    from qwen35_moe_dflash.target.prefill_t64.l0_split import prefill_precision

    _, graph = graph_message(air.read_bytes(), 'NetOutput')
    result = prefill_precision(graph, range(layers), head=True, qlinear_scale_fp16=qlinear_scale_fp16)
    if rmsnorm_patch:
        from qwen35_moe_dflash.target.prefill_t64.decoder_rmsnorm import inspect_ports
        result['rmsnorm_nodes'] = inspect_ports(graph)
    if check_shared_gates:
        from qwen35_moe_dflash.target.prefill_t64.decoder_source import inspect_shared_gates
        result['shared_gates'] = inspect_shared_gates(graph, layers, fp16_sigmoid=shared_gate_fp16_sigmoid)
    status = 'MISMATCH' if result['mismatches'] else 'MATCH'
    print(f'{layers}-layer AIR {air} — {status}: {len(result["matmuls"])} W4 matmuls, {len(result["gdrs"])} GDRs')
    for issue in result['mismatches']:
        print(' ', issue)
    return result


def check_full_precision(meta, bundle, rmsnorm_patch, hidden_trace=False):
    if meta.get('hidden_trace', False) != hidden_trace:
        raise ValueError(f'{bundle}: --hidden-trace selection differs from the exported AIR')
    if not (meta.get('shared_gate_fp16_sigmoid') and meta.get('qlinear_scale_fp16')):
        raise ValueError(f'{bundle}: regenerate the 40-layer AIR with FP16 QLinear weight scales and shared gates')
    if bool(meta.get('precision', {}).get('rmsnorm_nodes')) != rmsnorm_patch:
        raise ValueError(f'{bundle}: --rmsnorm-patch selection differs from the exported AIR')


def prefill_inputs(folder, reference, cfg, conv_control, *, physical_history=False, decoder_fixture=False,
                   yaml_prompt=False, lookup=False, require_same_entry=False):
    import numpy as np
    from qwen35_moe_dflash.target.layer_probe.common import read_array
    from qwen35_moe_dflash.target.prefill_t64.chain_reference import BASIS, VALID19_IDS_SHA256, YAML_BASIS

    index = reference / ('reference.json' if decoder_fixture else 'manifest.json')
    meta = json.loads(index.read_bytes())
    valid = meta['valid_rows']
    yaml_prompt = yaml_prompt or physical_history or decoder_fixture
    basis = YAML_BASIS if yaml_prompt else BASIS
    if decoder_fixture:
        matched = (meta['mode'], meta['phase'], meta['layers'], meta['logits_position'], meta['prompt_length']) == (
            '4', 'prefill', [0, 1, 2, 3], valid-1, valid)
    else:
        matched = (meta.get('reference'), meta.get('status'), meta.get('output_pos')) == (basis, 'CAPTURED', valid-1)
    if not matched or meta['physical_rows'] != 64 or not 1 <= valid <= 64 or (not yaml_prompt and valid != 19):
        raise ValueError(f'Unexpected measured T64 reference: {reference}')
    def tensor(name):
        return read_array(reference, meta['inputs'][name])

    captured_ids = tensor('input_ids')
    vocab = int(cfg.get('vocab_size', 248320))
    if captured_ids.shape != (1, 64) or not np.issubdtype(captured_ids.dtype, np.integer):
        raise ValueError('Expected integer prompt token IDs with shape [1,64]')
    if np.any(captured_ids < 0) or np.any(captured_ids >= vocab):
        raise ValueError('Captured token ID outside the embedding vocabulary')
    ids = captured_ids.astype('<i4', copy=False)
    digest = hashlib.sha256(ids.tobytes()).hexdigest()
    if not yaml_prompt and digest != VALID19_IDS_SHA256:
        raise ValueError('Unexpected prompt token IDs')
    if lookup:
        embedding_files = {'input_ids': ids}
    elif decoder_fixture:
        from qwen35_moe_dflash.target.prefill_t64.decoder_embedding import retained_embedding
        rows = retained_embedding(reference, meta)
        if rows is None:
            raise ValueError('Recreate the four-layer golden with decoder_golden.sh to capture the external embedding operands')
        quant_embed, embed_scale = rows(ids)
        embedding_files = dict(inputs_embeds=quant_embed, embed_scale=embed_scale)
    else:
        emb = np.memmap(cfg['embedding_weight_path'], mode='r', dtype='i1').reshape(vocab, 2048)
        scales = np.memmap(cfg['embedding_scale_path'], mode='r', dtype='<f4').reshape(vocab, 1)
        quant_embed, embed_scale = emb[ids], scales[ids]
        embedding_files = dict(inputs_embeds=quant_embed, embed_scale=embed_scale)
    positions = np.arange(64, dtype='<i4')
    if (not np.array_equal(tensor('position_ids').reshape(-1), positions) or
            not np.array_equal(tensor('new_kv_cache_pos').reshape(-1), positions) or
            tensor('effective_length').tolist() != [valid]):
        raise ValueError('Reference positions or effective length differ from fresh prefill')
    mask = tensor('attention_mask').astype('<f4')
    allowed = (positions[:, None] < valid) & (np.arange(2048)[None] <= positions[:, None])
    if mask.shape != (1, 1, 64, 2048) or not np.array_equal(mask[0, 0] == 0, allowed):
        raise ValueError(f'Expected the measured valid{valid} causal mask')
    if conv_control:
        conv_control = Path(conv_control)
        ctrl = np.fromfile(conv_control, dtype='<i4')
        if ctrl.shape != (128,) or conv_control.stat().st_size != 512:
            raise ValueError('Expected 128 INT32 values for the CPro ConvState control')
        policy = f'provided {conv_control}'
    else:
        ctrl = np.zeros(128, dtype='<i4')
        ctrl[0] = valid
        policy = f'trial [{valid}, 0 x 127] derived from observed valid64 control; partial-length semantics unverified'
    arrays = dict(**embedding_files, attention_mask=mask,
                  position_ids=positions[None], new_kv_cache_pos=positions,
                  output_pos=np.array([valid-1], dtype='<i4'), allQLen=np.array([valid], dtype='<i8'),
                  effective_length=np.array([valid], dtype='<i2'), conv_state_pos=ctrl,
                  past_conv_state=np.zeros((1, 8192, 4), dtype='<f2'),
                  past_recurrent_state=np.zeros((1, 32, 128, 128), dtype='<f2'),
                  past_kv_zero=np.zeros((64, 32, 32, 16), dtype='<f2'))
    if physical_history:
        del arrays['conv_state_pos']
        policy = 'explicit last four physical QKV rows (60..63); no ConvState control input'
    folder.mkdir()
    for name, array in arrays.items():
        array.tofile(folder / (name + '.bin'))
    print(f'Original runtime input fixture: valid{valid}, output_pos={valid-1}, ConvState: {policy}', flush=True)
    entry_name = 'hidden' if decoder_fixture else 'embedding'
    fixture = dict(reference_sha256=sha(index), token_sha256=digest, valid_rows=valid, conv_state_control=policy,
                   reference_inputs={k: meta['inputs'][entry_name if k == 'embedding' else k]['sha256']
                                     for k in REFERENCE_INPUT_NAMES})
    if lookup:
        print('[40/prefill] wrapper else: input_ids INT32[1,64] → embed_tokens', flush=True)
        return dict(**fixture, prompt=meta['prompt'], embedding='lookup')
    measured = tensor(entry_name)
    if measured.shape != (1, 64, 2048) or measured.dtype != np.float16:
        raise ValueError('Expected the measured FP16 decoder entry [1,64,2048]')
    # Compare at the diagnostic graph's FP16 boundary. The original graph still receives both raw files.
    entry = (quant_embed.astype(np.float32) * embed_scale).astype(np.float16)
    different = entry.view(np.uint16) != measured.view(np.uint16)
    if require_same_entry:
        count = int(np.count_nonzero(different))
        fixture['embedding_changed_fp16_elements_all_rows'] = count
        print(f'[embedding] CPro 送板编码 × scale 转 FP16 vs 310P L0 入口，全部 64 行差异 {count}/{entry.size}', flush=True)
        if count:
            raise ValueError('Four-layer board embedding operands differ from the measured 310P L0 input')
    changes = int(np.count_nonzero(different[:, :valid]))
    print(f'Original embedding * scale, rounded to FP16 vs 310P layer entry: {changes}/{valid*2048} changed')
    tail = {}
    if physical_history:
        count = int(np.count_nonzero(different[:, -4:]))
        tail['physical_tail_embedding_changed_fp16_elements'] = count
        print(f'L0 入口（主机重建 embedding × scale → FP16）与 310P 实测 hidden：'
              f'物理行 60–63 有 {count}/8192 个元素位模式不同。', flush=True)
    return dict(**fixture, embedding='external', **tail,
                **({'prompt': meta['prompt'], 'embedding_changed_fp16_elements': changes} if yaml_prompt
                   else {'valid19_embedding_changed_fp16_elements': changes}))


def native_export_argv(entry, layers, config):
    return [sys.executable, '-u', str(entry), 'mode=export', 'quant_mode=enable',
            f'num_hidden_layers={layers}', 'seq_len=64', 'topK=16', f'config={config}', 'quant_bits=4']


def select_native_depth(cfg, work, layers):
    # The owner constructs the module from HF config before setting the requested cache depth.
    model = Path(cfg['model_file']).resolve(strict=True)
    original = model/'config.json'
    settings = json.loads(original.read_bytes())
    text = settings.get('text_config', settings)
    types = text['layer_types']
    if len(types) < layers:
        raise ValueError(f'{original}: requested {layers} layers, found {len(types)} layer_types')
    text['num_hidden_layers'] = layers
    text['layer_types'] = types[:layers]
    view = work/'hf-model'
    view.mkdir()
    for path in model.iterdir():
        if path.name != 'config.json':
            (view/path.name).symlink_to(path, target_is_directory=path.is_dir())
    save_json(view/'config.json', settings)
    cfg.update(model_file=str(view), quant_bits=4)
    print(f'[CPro AIR] HF config: {layers} layers, W4 checkpoint: {cfg["quanted_pth"]}', flush=True)
    return dict(model_config=str(original), num_hidden_layers=layers, layer_types=text['layer_types'],
                quant_mode='enable', quant_bits=4, quanted_pth=cfg['quanted_pth'])


def full_source_hashes(source):
    from qwen35_moe_dflash.target.original_target_source import cpro_python_files
    return {str(path.relative_to(source)): sha(path) for path in cpro_python_files(source)}


def export(run, source, config, reference, conv_control, *, kind=KIND, baseline=None, l0_capture=None, prepared=None,
           layer=None, head=True, standalone=True, rmsnorm_patch=False, hidden_trace=False):
    import yaml
    split = kind in (L0_KIND, SPLIT_KIND)
    full = kind in FULL_KINDS
    native_convstate = kind == NATIVE_CONV_FULL_KIND
    uses_baseline = kind not in (KIND, *FULL_KINDS)
    aligned = split or kind in (ALIGNED_KIND, *FULL_KINDS)
    layers = 40 if full else (1 if split else 4)
    export_dir = f'glm_export_air_cpro_{layers}block_64_quant'
    default_config = Path('config/qwen3_5_moe.yaml')
    if uses_baseline:
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
    sources = {n: sha((prepared or source) / n) for n in (EXPORTER, 'models/modeling_qwen3_5_moe_hiai_cpro.py',
               'models/modeling_qwen3_5_moe_hiai_nz.py',
               'models/export_model_wrapper_qwen3_5_moe.py', 'utils.py')}
    run.mkdir(parents=True, exist_ok=True)
    dest = run / 'air'
    if full and (dest/'manifest.json').is_file():
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import stage
        saved = json.loads((dest/'manifest.json').read_bytes())
        same_control = True
        if native_convstate and saved['kind'] == kind:
            if conv_control:
                same_control = conv_control.read_bytes() == (dest/'inputs/conv_state_pos.bin').read_bytes()
            else:
                same_control = saved['fixture']['conv_state_control'].startswith('trial [')
        if (same_control and saved.get('hidden_trace', False) == hidden_trace
                and saved['fixture'].get('embedding') == 'lookup' and saved['kind'] == kind
                and saved.get('boundary', {}).get('token_ids_input')
                and saved['sources'] == sources and saved['config_sha256'] == sha(config)
                and saved['fixture']['reference_sha256'] == sha(reference/'manifest.json')):
            with tempfile.TemporaryDirectory(dir=run, prefix='check-source-') as tmp:
                if prepared is None:
                    stage(source, Path(tmp)/'base', aligned=True, layers=layers, native_convstate=native_convstate)
                tree = prepared or Path(tmp)/'base'
                if kind == TAPS_FULL_KIND:
                    from qwen35_moe_dflash.target.prefill_t64.gdn_taps import taps_source
                    tree = taps_source(tree, Path(tmp)/'variant')
                from qwen35_moe_dflash.target.prefill_t64.decoder_source import export_source
                tree = export_source(tree, Path(tmp)/'decoder', lookup=True, rmsnorm_patch=rmsnorm_patch,
                                     native_prefill=True, hidden_trace=hidden_trace)
                if full_source_hashes(tree) == saved['silu_patch']['source_sha256']:
                    for row in saved['files']:
                        checked(dest, row)
                    print(f'Reusing verified 40-layer AIR: {dest}', flush=True)
                    return
    if dest.exists():
        raise FileExistsError(f'Export already exists: {dest}. Use the existing AIR or choose a new run directory.')
    if kind != KIND:
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import inspect_activation_air, stage
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import inspect_air, same_boundary
        if uses_baseline and (base['sources'] != sources or base['config_sha256'] != sha(config)):
            raise ValueError(f'Source/config differs from the successful original at {baseline}')
        graph_path = EXPORT_DIR + '/export.air'
        if uses_baseline and not split:
            original = checked(base_manifest.parent, next(row for row in base['files'] if row['path'] == graph_path))
            base_boundary = inspect_air(original)
        staged = prepared or Path(tempfile.mkdtemp(dir=run, prefix='silu-source-'))
        entry = (staged / EXPORTER if prepared else
                 stage(source, staged, aligned=aligned, l0=split, layer=layer, head=head, layers=layers,
                       native_convstate=native_convstate))
        if kind == TAPS_FULL_KIND:
            from qwen35_moe_dflash.target.prefill_t64.gdn_taps import taps_source
            staged = taps_source(staged, run/'source-taps')
            entry = staged / EXPORTER
        print(f'SiluPatch model and exporter for review: {staged}', flush=True)
    # The owner's exporter deletes its relative export directory on startup.
    with tempfile.TemporaryDirectory(dir=run, prefix='export-') as temp:
        work = Path(temp)
        if full:
            from qwen35_moe_dflash.target.prefill_t64.decoder_source import export_source, save_air_source
            from qwen35_moe_dflash.target.prefill_t64.chain_air import checkpoint_paths
            staged = export_source(staged, work/'decoder', lookup=True, rmsnorm_patch=rmsnorm_patch,
                                   native_prefill=True, hidden_trace=hidden_trace)
            entry = staged / EXPORTER
            save_air_source(staged, work/'air-source')
            measured = json.loads((reference / 'manifest.json').read_bytes())
            cfg.update({k: str(v) for k, v in checkpoint_paths(source, config, measured).items()})
        resource_keys = ('model_file', 'quanted_pth') if full else (
            'model_file', 'quanted_pth', 'embedding_weight_path', 'embedding_scale_path')
        for key in resource_keys:
            original = Path(cfg[key])
            real = (source / original).resolve(strict=True)
            if not original.is_absolute():
                cfg[key] = str(real)
        depth = select_native_depth(cfg, work, layers) if native_convstate else None
        export_config = config
        if cfg != configured:
            export_config = work / 'export-config.yaml'
            export_config.write_text(yaml.safe_dump(cfg))
        argv = native_export_argv(entry, layers, export_config)
        if not uses_baseline:
            fixture = prefill_inputs(work / 'inputs', reference, cfg, conv_control,
                                    physical_history=full and not native_convstate, yaml_prompt=full, lookup=full)
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
        if aligned:
            package = str(Path(__file__).resolve().parents[3] / 'model')
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (package, env.get('PYTHONPATH'))))
        subprocess.run(argv, cwd=work, env=env, check=True)
        air = work / export_dir / 'export.air'
        variant = {'export_configuration': depth} if depth else {}
        if kind != KIND:
            variant.update(inspect_activation_air(
                air, shape_trace, layers=layers, full_only=layer is not None and layer % 4 == 3,
                softplus=aligned, check_convs=full and not native_convstate,
                gdn_conv='taps' if kind == TAPS_FULL_KIND else 'conv2d'))
            if uses_baseline:
                variant['original_manifest_sha256'] = sha(base_manifest)
            if split:
                from qwen35_moe_dflash.target.prefill_t64.l0_split import pack_reference, plan
                index = layer or 0
                chain = plan(air.read_bytes(), variant['silu_patch']['nodes'], layer=index, head=head,
                             verify_precision=True)
                variant['precision'] = chain['precision']
                if index % 4 != 3:
                    fixture['conv_state_control'] = 'last 4 physical QKV rows (60..63 for T64)'
                variant.update(layer=index, head=head, standalone=standalone)
                variant['reference'] = pack_reference(l0_capture, work, fixture, layer=index, head=head,
                                                      standalone=standalone)
            else:
                boundary = inspect_air(air, layers=layers, physical_history=full and not native_convstate,
                                       require_control=native_convstate, hidden_trace=hidden_trace,
                                       lookup_vocab_size=int(cfg.get('vocab_size', 248320)) if full else None)
                if uses_baseline:
                    same_boundary(base_boundary, boundary)
                variant['boundary'] = boundary
                print(f'Native AIR boundary: {len(boundary["inputs"])} inputs, '
                      f'{[row["role"] for row in boundary["outputs"]]}', flush=True)
                if aligned:
                    variant['precision'] = four_precision(air, layers, check_shared_gates=full, rmsnorm_patch=rmsnorm_patch,
                                                          shared_gate_fp16_sigmoid=full, qlinear_scale_fp16=full)
                    if variant['precision']['mismatches']:
                        raise ValueError('Native prefill export differs from the reviewed W4/GDR port contracts')
            staged_files = [EXPORTER, 'models/modeling_qwen3_5_moe_hiai_cpro.py']
            if aligned:
                staged_files.append('utils.py')
            if split:
                staged_files.append('models/export_model_wrapper_qwen3_5_moe.py')
            variant['silu_patch']['source_sha256'] = (full_source_hashes(staged) if full else
                                                     {n: sha(staged / n) for n in staged_files})
        abi = inspect_input_shape(air)
        if full:
            from parse_model_json import GE_PROTO
            ports = [(GE_PROTO[p['dtype_raw']], p['shape']) for p in abi['inputs']]
            if ('INT8', [1, 64, 2048]) in ports or ('FP32', [1, 64, 1]) in ports:
                raise ValueError('Expected 40-layer lookup AIR without external embedding inputs')
        files = [identity(p, work) for p in sorted(air.parent.rglob('*')) if p.is_file()]
        files.extend(identity(p, work) for p in sorted((work / 'inputs').glob('*.bin')))
        if uses_baseline:
            recorded = {r['path']: r['sha256'] for r in base['files'] if r['path'].startswith('inputs/')}
            current = {r['path']: r['sha256'] for r in files if r['path'].startswith('inputs/')}
            if recorded != {k: v for k, v in current.items() if k != 'inputs/layer_entry.bin'}:
                raise ValueError('The generated input files differ from the original four-layer fixture')
        if full:
            from qwen35_moe_dflash.target.prefill_t64.native_40 import pack_goldens
            files.extend(files_under(work/'air-source', work))
            variant.update(shared_gate_fp16_sigmoid=True, qlinear_scale_fp16=True)
            if hidden_trace:
                variant['hidden_trace'] = True
            variant['goldens'], variant['tokens'] = pack_goldens(reference, run, work, hidden_trace=hidden_trace)
            files.extend(r for backend in variant['goldens'].values() for r in backend.values())
            files.append(variant['tokens'])
        if split:
            files.extend(variant['reference']['tensors'].values())
        save_json(work / 'manifest.json', dict(kind=kind, **variant,
                  argv=argv, sources=sources, fixture=fixture,
                  config_path=str(config), config_sha256=sha(config), inputs=abi['inputs'], files=files))
        bundle = work / 'bundle'
        bundle.mkdir()
        contents = [export_dir, 'inputs', 'manifest.json']
        if full:
            contents.append('air-source')
        if split or full:
            contents.append('golden')
        for name in contents:
            (work / name).rename(bundle / name)
        bundle.rename(dest)
    if full:
        print(f'[CPro AIR] 本次导出的模型与 Python 源码：{dest/"air-source"}', flush=True)
        print(f'40-layer AIR bundle for OMC compilation: {sum(row["bytes"] for row in files) / 2**30:.2f} GiB', flush=True)
    share_for_transfer(dest)
    print(f'{kind} AIR: {dest / export_dir / "export.air"}')


def report_full_disk(meta, path, *, incoming=True, linked=0):
    if meta['kind'] not in FULL_KINDS:
        raise ValueError(f'Expected a native 40-layer prefill AIR, got {meta["kind"]}')
    bundle_size = sum(r['bytes'] for r in meta['files'])
    graph_size = sum(r['bytes'] for r in meta['files'] if Path(r['path']).parts[0] == FULL_DIR)
    free = shutil.disk_usage(path).free
    transfer = bundle_size - linked if incoming else 0
    print(f'OMC compilation filesystem {path}: {free / 2**30:.2f} GiB free, '
          f'AIR bundle {bundle_size / 2**30:.2f} GiB ({transfer / 2**30:.2f} GiB to download). '
          f'If OMG outputs match the {graph_size / 2**30:.2f} GiB graph and weights, '
          f'{(free - transfer - graph_size) / 2**30:.2f} GiB would remain for scratch and other usage. '
          'Actual compiler peak is unknown.', flush=True)
    if transfer > free:
        raise OSError(f'Insufficient free space to download {transfer:,} bytes of the AIR bundle')


def ssh_bundle(client, remote, staged, *, kind=KIND, previous=None, fetch=None):
    from decoder4_prefill import ssh_file
    fetch = fetch or ssh_file
    remote = Path(remote)
    staged.mkdir()
    manifest = staged / 'manifest.json'
    fetch(client, remote / manifest.name, manifest)
    meta = json.loads(manifest.read_bytes())
    if meta['kind'] != kind:
        raise ValueError(f'Expected {kind} AIR at {remote}, got {meta["kind"]}')
    roots = (L0_DIR, 'inputs', 'golden') if kind in (L0_KIND, SPLIT_KIND) else (EXPORT_DIR, 'inputs')
    if kind in FULL_KINDS:
        roots = (FULL_DIR, 'inputs', 'golden', 'air-source')
    missing = []
    linked = 0
    prior_ids = {}
    if previous is not None and (previous/'manifest.json').is_file():
        prior_meta = json.loads((previous/'manifest.json').read_bytes())
        prior_ids = {r['path']: r['sha256'] for r in prior_meta['files']}
    for row in meta['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts) < 2 or rel.parts[0] not in roots:
            raise ValueError(f'Invalid original export path: {rel}')
        target = staged / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if prior_ids.get(row['path']) == row['sha256']:
            try:
                prior = sized_file(previous, row)
                os.link(prior, target)
            except (OSError, ValueError):
                pass  # Changed file, different filesystem or hard links unavailable.
            else:
                linked += row['bytes']
                continue
        missing.append((rel, target, row))
    if kind in FULL_KINDS:
        print(f'Reused {linked / 2**30:.2f} GiB from the previous AIR manifest', flush=True)
        report_full_disk(meta, staged, linked=linked)
    for rel, target, row in missing:
        fetch(client, remote / rel, target, row)


def sftp_file(client, remote, local, record=None):
    from pull_artifacts import download
    download(client, str(remote), local)
    if record is not None and local.stat().st_size != record['bytes']:
        raise ValueError(f'Incomplete AIR file: {remote}')


def compile_log_text(path):
    excerpt = log_excerpt(path)
    if excerpt == 'Log tail:\n':
        excerpt = 'OMG produced an empty compile.log.'
    return f'OMG log: {path}\n{excerpt}'


def compile_omg(out, air_path, ddk, diagnosis, platform, *, use_origin_format=True, launcher=(), external_weights=True):
    from qwen35_moe_dflash.toolchain.compile import omg_environment, prepare_tbe_compatibility, _omg_ipc_environment

    shape = inspect_input_shape(out / air_path)['input_shape']
    omg, env = omg_environment(ddk, soc_version=platform)
    if diagnosis:
        env, _ = prepare_tbe_compatibility(ddk, diagnosis, out / 'tbe-compat', inherited=env)
    platform_args = [f'--platform={platform}'] if platform is not None else []
    argv = [str(omg), '--framework=1', f'--model={air_path}', '--output=model',
            '--target=omc', *platform_args, f'--use_origin_format={str(use_origin_format).lower()}',
            '--weight_data_type=FP16',
            '--limit_partition_size=false', f'--save_weights_as_external_data={str(external_weights).lower()}',
            '--input_shape=' + shape]
    log_path = out / 'compile.log'
    cmd = [*launcher, *argv]
    command = f'(cd {shlex.quote(str(out))} && {shlex.join(cmd)})'
    invocation = f'{command}\nOMG SOC_VERSION={env.get("SOC_VERSION", "<unset>")}'
    print(f'[编译 OMC] {out / air_path}\n{invocation}\nOMG log: {log_path}', flush=True)
    with _omg_ipc_environment(out, env) as (child, _), log_path.open('x') as log:
        print(invocation, file=log, flush=True)
        result = subprocess.run(cmd, cwd=out, env=child, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode, argv, env.get('SOC_VERSION')


def compile_native_prefill(out, air_path, ddk, *, external_weights=True):
    """Use the owner's Linux OMG entry with the fixed gear of this prefill export."""
    shape = inspect_input_shape(out/air_path)['input_shape']
    wrapper = ddk/'tools/tools_omg/omg'
    argv = [str(wrapper), '--model', air_path, '--output', 'model', '--framework', '1',
            '--limit_partition_size=false', f'--save_weights_as_external_data={str(external_weights).lower()}',
            '--input_shape=' + shape]
    setup = ddk/'tools/tools_ascendc/set_ascendc_env.sh'
    # A configured compile worker has already sourced its own DDK setup.
    if os.getenv('DECODER_DDK_SETUP') and os.getenv('QWEN35_DECODER_SETUP_LOADED') == str(setup.resolve()):
        cmd = argv
    else:
        cmd = ['bash', '-c', 'setup=$1; shift; load_setup() { source "$setup"; }; '
               'load_setup || exit $?; exec "$@"', 'native-omg', str(setup), *argv]
    log = out/'compile.log'
    command = f'(cd {shlex.quote(str(out))} && {shlex.join(cmd)})'
    print(f'[编译原生 T64 OMC] {command}\nOMG log: {log}', flush=True)
    with log.open('x') as stream:
        print(command, file=stream, flush=True)
        result = subprocess.run(cmd, cwd=out, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode, argv, os.environ.get('SOC_VERSION')


def build(run, ddk, diagnosis, remote, *, bundle, kind, platform, external_weights=True, rmsnorm_patch=False,
          hidden_trace=False):
    from decoder4_prefill import pull_air

    run.mkdir(parents=True, exist_ok=True)
    if not bundle.exists():
        with tempfile.TemporaryDirectory(dir=bundle.parent, prefix='.fetch-') as tmp:
            previous = run.parent/'air' if kind in FULL_KINDS else None
            download = partial(ssh_bundle, kind=kind, previous=previous if previous and previous.is_dir() else None)
            pull_air(Path(tmp), remote, ssh_download=download,
                     sftp_download=partial(download, fetch=sftp_file) if kind in FULL_KINDS else None)
            incoming = Path(tmp) / 'air'
            if json.loads((incoming / 'manifest.json').read_bytes())['kind'] != kind:
                raise ValueError(f'Expected {kind} AIR at {remote}')
            incoming.rename(bundle)
    meta = json.loads((bundle / 'manifest.json').read_bytes())
    if kind in FULL_KINDS:
        check_full_precision(meta, bundle, rmsnorm_patch, hidden_trace)
        if not meta['boundary'].get('token_ids_input'):
            raise ValueError(f'{bundle}: rerun decoder_air.sh for 40-layer prefill with INT32 embedding indices before compiling')
    if meta['kind'] != kind:
        raise ValueError(f'Expected {kind} AIR at {bundle}, got {meta["kind"]}')
    for row in meta['files']:
        sized_file(bundle, row)
    local_air = run / 'air'
    if local_air != bundle:
        try:
            local_air.symlink_to(os.path.relpath(bundle, run), target_is_directory=True)
        except FileExistsError:
            if not local_air.samefile(bundle):
                raise ValueError(f'{local_air} must refer to the selected AIR bundle {bundle}')
    dest = run / 'omc'
    if kind in FULL_KINDS:
        report_full_disk(meta, run, incoming=False)
    if kind in FULL_KINDS:
        out = run/'compile'
        out.mkdir()
    else:
        out = Path(tempfile.mkdtemp(dir=run, prefix='omg-'))
    # Keep the directory component in the original FileConstant paths.
    directory = FULL_DIR if kind in FULL_KINDS else EXPORT_DIR
    (out / directory).symlink_to(bundle / directory, target_is_directory=True)
    if kind == NATIVE_CONV_FULL_KIND:
        code, argv, soc = compile_native_prefill(out, directory + '/export.air', ddk)
    else:
        code, argv, soc = compile_omg(out, directory + '/export.air', ddk, diagnosis, platform,
                                      external_weights=external_weights)
    if code:
        raise RuntimeError(f'{kind}: OMG exited {code}\n{compile_log_text(out / "compile.log")}')
    model, = out.rglob('*.omc')
    weights = [identity(p, out) for p in sorted(model.parent.rglob('*.weight'))]
    if bool(weights) != external_weights:
        raise ValueError(f'Unexpected OMG weight files for external_weights={external_weights}: {out}')
    descriptors = {}
    if kind in FULL_KINDS:
        from parse_model_json import GE_PROTO, inspect, read_omc
        descriptors['abi'] = inspect(read_omc(model, ddk_root=ddk)[0], enum=GE_PROTO)
    save_json(out / 'manifest.json', dict(**descriptors, air_manifest_sha256=sha(bundle / 'manifest.json'), argv=argv,
              soc_version=soc,
              primary=identity(model, out), weights=weights))
    out.rename(dest)
    print(f'Compiled {kind} OMC: {dest / model.relative_to(out)}', flush=True)


def original_filename(name, src, dtype, *, fresh_zero_states, token_ids_input=None):
    """Match the original exporter's descriptors, before singleton dimensions can fold."""
    dims = tuple(src['shape'])
    if name == token_ids_input:
        if (dtype, dims) != ('int32', (1, 64)):
            raise ValueError(f'{name}: expected INT32[1,64] for input_ids, got {dtype}{dims}')
        return 'input_ids.bin'
    known = {
        ('int64', (1, 64)): 'input_ids',
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


def board(run, ddk, config, directory, input_map, *, kind=KIND, baseline=None, load_control=None, replace_replay=False,
          replace_inputs=False, replace_weights=False, original_board_args=False, input_overrides=None, replay_dir=None,
          dump_edges=()):
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
    if kind in FULL_KINDS and meta['fixture'].get('embedding') != 'lookup':
        raise ValueError('This 40-layer OMC uses external embedding. Regenerate the 40-layer AIR and OMC for token-ID inference.')
    if meta['kind'] != kind or sha(run / 'air/manifest.json') != compiled['air_manifest_sha256']:
        raise ValueError('The compiled model belongs to a different original AIR bundle')
    model = sized_file(run / 'omc', compiled['primary'])
    weights = [sized_file(run / 'omc', row) for row in compiled['weights']]
    abi = compiled.get('abi')
    if abi is None:
        if ddk is None:
            raise ValueError('This OMC predates saved I/O descriptors. Run the compile stage for this export.')
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
    input_hashes = {(run/'air'/r['path']).resolve(): r['sha256']
                    for r in meta['files'] if fresh and Path(r['path']).parts[0] == 'inputs'}
    mapping = read_json(input_map) if input_map else {}
    if mapping and set(mapping) != {s['name'] for s in inputs}:
        raise ValueError('input-map must supply one filename for every compiled input node')
    original_inputs = {}
    if kind != KIND and not baseline:
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import output_roles
        output_roles(outputs, meta['boundary']['outputs'])
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
    failed_inputs = {r['name']: r for r in load_control[0]['inputs']} if load_control else {}
    paths = []
    for i, spec in enumerate(inputs):
        name = spec['name']
        recorded_input = failed_inputs.get(name) or original_inputs.get(name)
        if fresh and not mapping and recorded_input and Path(recorded_input['file']).is_file():
            path = Path(recorded_input['file'])
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
            path = directory / original_filename(name, src, spec['dtype'], fresh_zero_states=fresh,
                                                  token_ids_input=meta.get('boundary', {}).get('token_ids_input'))
        if input_overrides:
            path = input_overrides.get(path.name, path)
        if path.stat().st_size != spec['bytes']:
            raise ValueError(f'{name} {path}: expected {spec["bytes"]} bytes')
        if path.name == 'output_pos.bin':
            pos = int(np.fromfile(path, dtype='<i4').item())
            if not 0 <= pos < 64:
                raise ValueError(f'{path}: output_pos={pos}, expected a zero-based row in [0, 63]')
        paths.append(path)
        spec.update(sha256=input_hashes.get(path.resolve()) or sha(path), file=str(path))
        if original_inputs and spec['sha256'] != original_inputs[name]['sha256']:
            raise ValueError(f'{name}: input bytes differ from the successful original replay')
        print(f"slot {i}: {name} {spec['dtype']} {spec['shape']} <- {path}", flush=True)
    io = dict(inputs=inputs, outputs=outputs, order=abi['input_order_basis'])
    if load_control:
        previous_io, previous_receipt, _ = load_control
        def signature(rows):
            return {row['name']: (row['dtype'], tuple(row['shape']), row['sha256']) for row in rows}
        if (len(previous_io['inputs']) != len(inputs) or signature(previous_io['inputs']) != signature(inputs)
                or len(signature(inputs)) != len(inputs)):
            raise ValueError('Embedded-weight control must use the same named input slots and bytes as the failed run')
        if tuple(_environment()[1:3]) != (previous_receipt['device'], previous_receipt['server']):
            raise ValueError('Select the SN and BIP of the failed run for the embedded-weight control')
        io['load_control'] = dict(external_omc_sha256=previous_receipt['omc_sha256'],
                                  previous_board_attempt=previous_receipt['remote'],
                                  previous_exitcode=previous_receipt['exitcode'])
    runtime_config = read_json(config)
    runtime = read_json(config.parent / runtime_config['assets'])['environment']
    wait_seconds = runtime_config.get('defaults', {}).get('lock_wait_seconds', 1800) if dump_edges else 0
    if type(wait_seconds) is not int or not 0 <= wait_seconds <= 7200:
        raise ValueError(f'{config}: defaults.lock_wait_seconds must be an integer in [0, 7200]')
    if kind in FULL_KINDS:
        out = replay_dir or run/'replay'
        if replace_replay and out.is_dir():
            shutil.rmtree(out)
        out.mkdir()
    else:
        out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    save_json(out / 'io.json', io)
    print(f'{kind} board attempt: {out}', flush=True)
    try:
        run_v2(out, 'original L0-L3' if kind == KIND else kind, model, inputs, paths, outputs, runtime,
               decoder_shapes=(abi.get('input_shape'), abi.get('output_shape')), shape_config=config,
               original_board_args=original_board_args or kind == NATIVE_CONV_FULL_KIND,
               external_weights=weights, build_records=(compiled['primary'], compiled['weights']),
               replace_inputs=replace_inputs, replace_weights=replace_weights,
               board_case='40-prefill-t64' if kind in FULL_KINDS else '4-prefill-t64',
               hdc_timeout=7200 if kind in FULL_KINDS else 660, dump_edges=dump_edges,
               lock_wait_seconds=wait_seconds)
    except RuntimeError:
        if not (out / 'result.json').is_file():
            raise
        receipt = read_json(out / 'result.json')
        if receipt.get('exitcode'):
            hdc = Hdc(*_environment(), out, 60)
            try:
                remote = receipt['remote']
                attempt = receipt['attempt']
                for filename in ('tool.stdout', 'tool.stderr'):
                    _, body = hdc.shell(f'cd {remote} && test "$(cat .qwen35-hdc-owner)" = {attempt} && '
                                        f'cat {filename}', filename)
                    (out/filename).write_text(body + '\n' if body else '')
                _, markers = hdc.shell(f'cd {remote} && test "$(cat .qwen35-hdc-owner)" = {attempt} && '
                                       'stat -c "%y" tool.started tool.exitcode', 'times')
                start, end = [datetime.fromisoformat(t[:19]) for t in markers.splitlines()]
                from qwen35_moe_dflash.target.prefill_t64.chain_hilog import _snapshot
                _window(_snapshot(hdc, out, None), out / 'window.log', start, end)
                print(out / 'window.log')
            except Exception as diagnostic_error:
                print(f'Could not collect failure details: {diagnostic_error}', file=sys.stderr)
        raise
    print(f'{kind} executed: {out / "result.json"}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-inputs', action='store_true')
    parser.add_argument('--replace-weights', action='store_true')
    parser.add_argument('--original-board-args', action='store_true')
    parser.add_argument('action', choices=('air', 'compile', 'execute', 'board', 'log', 'audit', 'reference'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--prepared-source', type=Path, help='run the edited aligned40 exporter and model directly')
    parser.add_argument('--rmsnorm-patch', action='store_true', help='use RmsnormPatch in the aligned40 PyTorch export')
    parser.add_argument('--hidden-trace', action='store_true', help='append all 40 post-decoder hidden states')
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
    mode.add_argument('--aligned', action='store_true', help='native boundary, reviewed RoPE/expert fixes, SiluPatch and SoftplusPatch')
    mode.add_argument('--aligned40', action='store_true', help='one original L0–L39 native S64 prefill AIR and OMC')
    baselines = parser.add_mutually_exclusive_group()
    baselines.add_argument('--baseline-if-available', type=Path, help='use a successful local original replay when present')
    baselines.add_argument('--baseline', type=Path, help='native baseline run (or its replay for board comparison)')
    parser.add_argument('--retry-load', type=Path, metavar='FAILED_REPLAY',
                        help='rebuild aligned AIR with embedded weights and replay the same inputs')
    parser.add_argument('--replay', type=Path, help='include a retained four-layer load failure in the offline audit')
    parser.add_argument('--gdn-conv', choices=('conv2d', 'taps', 'original'), default='conv2d')
    args = parser.parse_args()
    if args.hidden_trace and not args.aligned40:
        parser.error('--hidden-trace requires --aligned40')
    if args.gdn_conv != 'conv2d' and not args.aligned40:
        parser.error('For this native exporter selecting a GDN convolution requires --aligned40')
    if args.retry_load and (args.action != 'board' or not args.aligned):
        parser.error('--retry-load requires board --aligned')
    if args.replay and args.action != 'audit':
        parser.error('--replay requires audit')
    base = output_path(args.run)
    bundle = base / 'air'
    if args.action == 'reference':
        if not args.aligned40 or not args.source or not args.config:
            parser.error('reference requires --aligned40, --source and --config')
        from qwen35_moe_dflash.target.prefill_t64.native_40 import capture_and_cpu
        base.mkdir(parents=True, exist_ok=True)
        capture_and_cpu(base, (args.reference or base/'reference').resolve(), args.source.resolve(), args.config)
        return
    if args.aligned40 and (args.inputs or args.input_map or (args.conv_control and args.gdn_conv != 'original')):
        parser.error('the 40-layer comparison uses inputs from its own 310P capture')
    if args.aligned40 and (args.baseline or args.baseline_if_available or args.action == 'audit'):
        parser.error('select the measured 40-layer capture with --reference for --aligned40')
    if args.action == 'audit':
        meta = json.loads((bundle/'manifest.json').read_bytes())
        if meta['kind'] != ALIGNED_KIND:
            parser.error('audit expects the aligned FP16 SiluPatch four-layer run')
        if args.replay:
            from split_stage_log import four_layer_load_report
            replay = args.replay.resolve(strict=True)
            if not (replay.parent/'air').samefile(bundle):
                parser.error('the selected replay and audit run must use the same AIR bundle')
            four_layer_load_report(replay)
        record, = [r for r in meta['files'] if r['path'] == EXPORT_DIR + '/export.air']
        print(f'Saved AIR sha256={record["sha256"]}')
        four_precision(checked(bundle, record))
        return
    run = output_path(base / 'no-platform') if args.aligned and args.action != 'air' else base
    if args.action == 'log':
        for folder in (run, base):
            logs = [p for p in (*folder.glob('omg-*/compile.log'), folder / 'omc/compile.log', folder / 'compile/compile.log') if p.is_file()]
            if logs:
                newest = max(logs, key=lambda p: (p.stat().st_mtime_ns, p))
                print(compile_log_text(newest))
                return
        parser.error(f'No OMG compile.log under {base}')
    full_kind = {'conv2d': FULL_KIND, 'taps': TAPS_FULL_KIND, 'original': NATIVE_CONV_FULL_KIND}[args.gdn_conv]
    kind = full_kind if args.aligned40 else (ALIGNED_KIND if args.aligned else PATCH_KIND if args.silu_patch else KIND)
    if kind not in (KIND, *FULL_KINDS) and args.action == 'air' and args.baseline is None:
        parser.error(f'variant {args.action} requires --baseline')
    if args.action == 'air':
        if args.aligned40:
            args.reference = (args.reference or base/'reference').resolve(strict=True)
        if kind in (KIND, *FULL_KINDS) and (not args.source or not args.reference):
            parser.error('native export requires --source and --reference')
        export(run, args.source, args.config, args.reference, args.conv_control, kind=kind, baseline=args.baseline,
               prepared=args.prepared_source.resolve(strict=True) if args.prepared_source else None,
               rmsnorm_patch=args.rmsnorm_patch, hidden_trace=args.hidden_trace)
        return
    if args.action == 'execute' and not args.aligned40:
        parser.error('execute requires --aligned40')
    if not args.ddk and args.action != 'execute':
        parser.error('compile and board require --ddk')
    control = None
    if args.retry_load:
        from split_stage_log import four_layer_load_report
        replay = args.retry_load.resolve(strict=True)
        if replay.parent != run:
            parser.error(f'--retry-load must identify a replay directly under {run}')
        control = four_layer_load_report(replay)
        command = control[2]['argv']
        if '--save_weights_as_external_data=true' not in command or not control[2]['weights']:
            parser.error('--retry-load requires an external-weight compiler control')
        if (Path(command[0]).resolve() != (args.ddk/'tools/tools_omg/master/omg').resolve()
                or control[2]['soc_version'] != os.environ.get('SOC_VERSION')):
            parser.error('use the same DDK and SOC_VERSION as the failed external-weight build')
        print(f'Available host space before control build: {shutil.disk_usage(run).free:,} bytes', flush=True)
        run = run/'embedded'
    compiled = run / 'omc/manifest.json'
    if not compiled.exists():
        if args.action == 'execute':
            raise FileNotFoundError(f'Run compile and make its outputs available here: {compiled}')
        build(run, args.ddk.resolve(), args.tbe_diagnosis, args.remote_air or bundle, kind=kind,
              bundle=bundle, platform=None if args.aligned or args.aligned40 else 'kirinx90',
              external_weights=not args.retry_load, rmsnorm_patch=args.rmsnorm_patch, hidden_trace=args.hidden_trace)
    else:
        manifest = bundle / 'manifest.json'
        air_meta = json.loads(manifest.read_bytes())
        if args.aligned40:
            check_full_precision(air_meta, bundle, args.rmsnorm_patch, args.hidden_trace)
        record = json.loads(compiled.read_bytes())
        if (air_meta['kind'] != kind or record['air_manifest_sha256'] != sha(manifest)
                or not (run / 'air').samefile(bundle)):
            raise ValueError(f'Cached OMC does not match the requested {kind} AIR at {run}')
        if (args.aligned or args.aligned40) and any(arg.split('=', 1)[0] == '--platform' for arg in record['argv']):
            raise ValueError(f'Cached OMC at {run} was built with an explicit --platform')
        print(f'Reusing {kind} OMC from {compiled.parent}', flush=True)
        if kind == ALIGNED_KIND and 'precision' not in air_meta:
            print('This cached AIR predates the four-layer W4/GDR port check. Inspect it with aligned-audit.')
    if control:
        reference = control[2]
        trial = json.loads(compiled.read_bytes())
        expected = ['--save_weights_as_external_data=false' if a == '--save_weights_as_external_data=true' else a
                    for a in reference['argv']]
        if (trial['argv'] != expected or trial['weights'] or trial['soc_version'] != reference['soc_version']
                or trial['air_manifest_sha256'] != reference['air_manifest_sha256']):
            raise ValueError('Embedded control differs in compiler command, SOC_VERSION, AIR bundle or weight mode')
        print(f'Embedded-weight comparison with {args.retry_load}: same AIR and compiler options except weight storage.')
    if args.action in ('execute', 'board'):
        if not args.runtime_config:
            parser.error('board requires --runtime-config')
        if args.baseline_if_available:
            from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import latest_replay
            args.baseline = latest_replay(args.baseline_if_available, required=False)
        if kind != KIND and args.baseline is None:
            print('Binding board inputs using compiled slots and the exported AIR fixture.', flush=True)
        directory = args.inputs or run / 'air/inputs'
        replay = board(run, args.ddk.resolve() if args.ddk else None, args.runtime_config.resolve(), directory.resolve(), args.input_map, kind=kind,
              baseline=args.baseline, load_control=control, replace_replay=args.action == 'execute',
              replace_inputs=args.replace_inputs, replace_weights=args.replace_weights, original_board_args=args.original_board_args)
        if args.aligned40:
            from qwen35_moe_dflash.target.prefill_t64.native_40 import compare
            from l0_silu_omcs import write_matrices
            passed = compare(bundle, replay, write_matrices)
            if args.hidden_trace:
                from plot_decoder_hidden import plot
                try:
                    plot(run/'matrices.csv', run)
                except ModuleNotFoundError as exc:
                    if exc.name != 'matplotlib':
                        raise
                    print(f'Matplotlib 未安装。可在绘图机运行 python3 targets/cpro/tools/plot_decoder_hidden.py {run}/matrices.csv')
            return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

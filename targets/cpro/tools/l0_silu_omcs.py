#!/usr/bin/env python3
"""Compile and replay the one-layer native CPro decoder with four singleton SiluPatch OMCs."""
import argparse
import csv
import errno
from functools import partial
import hashlib
import json
from math import prod
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from decoder4_original import (L0_DIR, L0_KIND, compile_log_text, compile_omg, export,
                               original_filename, ssh_bundle)
from decoder4_prefill import pull_air, ssh_file
from export_silu_patch_air import build_air
from parse_model_json import GE_PROTO, inspect, read_omc
from qwen35_moe_dflash.common.accuracy import ERROR_RATIO_FIELDS, THRESHOLDS
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, read_array, save_json, sha, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import run_v2
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.layer_template import parameterize
from qwen35_moe_dflash.target.prefill_t64.l0_split import CPU_BASIS, contracts, plan, score, slice_air


def load(path):
    return json.loads(path.read_bytes())


def select_board_defaults(args):
    """Use the completed L0 W4/residual plan unless a diagnostic variant was requested."""
    if not args.split_decoder_projections and not getattr(args, 'embed_decoder_hidden', False):
        args.split_decoder_residual = True


def typed_view(dtype, shape):
    return dtype, tuple(d for d in shape if d != 1)


def bind(step, wires, abi):
    inputs = step['inputs']
    names = {('x' if step['name'].endswith('_silu') else f'l0_feed_{i}'): e for i, e in enumerate(inputs)}
    io = {}
    for direction in ('inputs', 'outputs'):
        pending = list(step[direction])
        rows = []
        for row in abi[direction]:
            if row['slot'] != len(rows) or row['issues'] or not row['logical_bytes']:
                raise ValueError(f'Invalid compiled {direction} in {step["name"]}: {row}')
            dtype = {'FP16': 'float16', 'FP32': 'float32'}.get(row['dtype'], row['dtype'].lower())
            key = typed_view(dtype, row['shape'])
            name = row['node'] if direction == 'inputs' else row['edge']
            if direction == 'outputs' and len(step['outputs']) == 1 and wires[step['outputs'][0]]['shape'] is None:
                edge, = step['outputs']
                expected = wires[edge]['dtype']
                if (dtype not in ('float16', 'float32') or expected not in (None, dtype)
                        or not row['shape'] or min(row['shape']) < 1):
                    raise ValueError(f'{step["name"]}: unexpected compiled W4 probe output at {edge}: {row}')
                candidates = [edge]
            else:
                candidates = [e for e in pending if typed_view(wires[e]['dtype'], wires[e]['shape']) == key]
            if direction == 'inputs':
                candidates = [e for e in candidates if names.get(name) == e]
            elif name in pending:
                candidates = [e for e in candidates if e == name]
            if len(candidates) != 1:
                raise ValueError(f'{step["name"]}: cannot bind {direction} {row} to {candidates}')
            edge, = candidates
            pending.remove(edge)
            if prod(row['shape']) * np.dtype(dtype).itemsize != row['logical_bytes']:
                raise ValueError(f'Compiled byte extent differs at {edge}')
            rows.append(dict(name=name, edge=edge, shape=row['shape'], dtype=dtype, bytes=row['logical_bytes']))
        if pending:
            raise ValueError(f'{step["name"]}: omitted {direction}: {pending}')
        io[direction] = rows
    return io


def compile_stage(run, bundle, step, graph, raw, ddk, diagnosis, bundle_hash, shared_silu=None,
                  embed_decoder_hidden=False):
    name = step['name']
    wires = graph['wires']
    stage_air = (build_air(shape=wires[step['inputs'][0]]['shape']) if name.endswith('_silu')
                 else slice_air(raw, step, wires))
    shared = shared_silu is not None and name.endswith('_silu')
    embedded = step.get('weight_storage') == 'embedded' or (embed_decoder_hidden and name == 'decoder_hidden')
    dest = (shared_silu if shared else run / 'omc') / step.get('build_name', name)
    parent = dest.parent
    parent.mkdir(exist_ok=True)
    source = dict(bundle_sha256=bundle_hash, air_sha256=hashlib.sha256(stage_air).hexdigest(), ddk=str(ddk))
    if shared:
        del source['bundle_sha256']  # Singleton AIRs have no layer weights.
    if embedded:
        if dest.is_dir():
            # Older split runs serialized equivalent protobuf maps in a different order.
            # Compile the exact bytes used in the external-weight control.
            control = load(dest/'manifest.json')['source']
            retained = (dest/L0_DIR/'export.air').read_bytes()
            digest = hashlib.sha256(retained).hexdigest()
            if control != dict(source, air_sha256=digest):
                raise ValueError(f'{dest}: external-weight build belongs to a different AIR bundle or DDK')
            if digest != source['air_sha256']:
                old, _ = graph_message(retained, 'NetOutput')
                if old.SerializeToString(deterministic=True) != stage_air:
                    raise ValueError(f'{dest}: retained AIR differs from the current slice')
            stage_air = retained
            source['air_sha256'] = digest
        source['weight_storage'] = 'embedded'
        dest = dest.with_name(dest.name + '-embedded')
        print(f"{name}: embedded weights, AIR sha256={source['air_sha256']}", flush=True)
    if not dest.exists():
        out = Path(tempfile.mkdtemp(prefix=name + '-', dir=parent))
        folder = out / L0_DIR
        folder.mkdir()
        (folder/'export.air').write_bytes(stage_air)
        for p in (bundle / L0_DIR).iterdir():
            if p.name != 'export.air' and not name.endswith('_silu'):
                (folder / p.name).symlink_to(p, target_is_directory=p.is_dir())
        code, _, _ = compile_omg(out, L0_DIR + '/export.air', ddk, diagnosis, platform=None,
                                 external_weights=not embedded)
        if code:
            raise RuntimeError(f'{name}: OMG exited {code}\n{compile_log_text(out / "compile.log")}')
        model, = out.rglob('*.omc')
        weights = [identity(p, out) for p in sorted(model.parent.rglob('*.weight'))]
        if embedded and weights:
            raise ValueError(f'{name}: OMG emitted external weights for an embedded build in {out}')
        abi = inspect(read_omc(model, ddk_root=ddk)[0], enum=GE_PROTO)
        io = bind(step, wires, abi)
        save_json(out/'manifest.json', dict(source=source, model=identity(model, out), weights=weights, **io))
        out.rename(dest)
    meta = load(dest/'manifest.json')
    recorded = meta['source']
    if recorded != source:
        # Earlier slices wrote unordered protobuf maps. Verify the retained compiler input
        # against its original hash, then compare its canonical serialization to today's slice.
        retained = dest / L0_DIR / 'export.air'
        if (dict(recorded, air_sha256=source['air_sha256']) == source
                and retained.is_file() and sha(retained) == recorded['air_sha256']):
            old, _ = graph_message(retained.read_bytes(), 'NetOutput')
            if old.SerializeToString(deterministic=True) == stage_air:
                recorded = source
    if recorded != source:
        raise ValueError(f'{dest} belongs to a different AIR or DDK. Select a new L0_SPLIT_RUN.')
    if embedded and meta['weights']:
        raise ValueError(f'{dest}: embedded OMC manifest lists external weights')
    if shared:
        for direction in ('inputs', 'outputs'):
            row, = meta[direction]
            edge, = step[direction]
            if typed_view(row['dtype'], row['shape']) != typed_view(wires[edge]['dtype'], wires[edge]['shape']):
                raise ValueError(f'Shared {name} {direction} differs from {edge}')
            meta[direction] = [dict(row, edge=edge)]
    return meta, sized_file(dest, meta['model']), [sized_file(dest, row) for row in meta['weights']]


def compile_template(run, bundle, step, graph, raw, ddk, diagnosis, root):
    sliced = slice_air(raw, step, graph['wires'])
    air, call, specs, params = parameterize(sliced, step, graph['wires'], bundle/L0_DIR, run/'parameters')
    name = step['name']
    if name.startswith(('key_cache_', 'value_cache_')):
        name = name.rsplit('_', 1)[0]
    dest = root/name
    root.mkdir(parents=True, exist_ok=True)
    source = dict(air_sha256=hashlib.sha256(air).hexdigest(), ddk=str(ddk), external_weights=False)
    if not dest.exists():
        tmp = Path(tempfile.mkdtemp(prefix=name+'-', dir=root))
        (tmp/'export.air').write_bytes(air)
        code, _, _ = compile_omg(tmp, 'export.air', ddk, diagnosis, platform=None, external_weights=False)
        if code:
            raise RuntimeError(f'{name}: template OMG exited {code}\n{compile_log_text(tmp/"compile.log")}')
        model, = tmp.rglob('*.omc')
        if list(model.parent.rglob('*.weight')):
            raise ValueError(f'{name}: unexpected external template weights')
        abi = inspect(read_omc(model, ddk_root=ddk)[0], enum=GE_PROTO)
        bind(call, specs, abi)
        save_json(tmp/'manifest.json', dict(source=source, model=identity(model, tmp),
                                           abi={k: abi[k] for k in ('inputs','outputs')}))
        tmp.rename(dest)
    meta = load(dest/'manifest.json')
    if meta['source'] != source:
        raise ValueError(f'{run.name}/{name}: layer-specific computation or ABI differs from {dest}; '
                         'use --per-layer-omc for this export')
    io = bind(call, specs, meta['abi'])
    original = dict(zip(call['outputs'], step['outputs'], strict=True))
    for row in io['outputs']:
        row['edge'] = original[row['edge']]
    return dict(model=meta['model'], **io), checked(dest, meta['model']), params


def checkpoint_source(run, recovery, fixture_hash):
    if recovery is None:
        return None, []
    if recovery == 'auto':
        candidates = sorted(run.glob('replay-*/chain.json'), key=lambda f: f.stat().st_mtime_ns, reverse=True)
    else:
        candidates = [(run / recovery) / 'chain.json']
    for path in candidates:
        previous = load(path)
        if previous['bundle_sha256'] == fixture_hash:
            print(f'Recovering from {path.parent} ({len(previous["stages"])} completed stages)', flush=True)
            return path.parent, previous['stages']
    raise ValueError(f'No matching replay for --recovery {recovery} in {run}')


def restore_stage(previous, dest, record, model_hash, inputs, outputs, fixture_hash):
    from qwen35_moe_dflash.execution.serial_transport import _environment
    _, device, server = _environment()
    name = dest.name
    source = previous / name
    path = source / 'result.json'
    if (record['name'] != name or record['inputs'] != inputs or record['model_sha256'] != model_hash
            or sha(path) != record['result_sha256']):
        raise ValueError(f'{name}: checkpoint inputs, model or result changed in {previous}')
    result = load(path)
    expected = dict(case=name, status='EXECUTED', exitcode=0, omc_sha256=model_hash,
                    fixture_sha256=fixture_hash, device=device, server=server)
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{name}: checkpoint belongs to another run, model or board: {path}')
    paths = []
    for i, (row, spec) in enumerate(zip(result['outputs'], outputs, strict=True)):
        filename = 'output.bin' if len(outputs) == 1 else f'output_{i}.bin'
        if row['bytes'] != spec['bytes'] or row['path'] != filename:
            raise ValueError(f'{name}: unexpected saved output {row}')
        paths.append(checked(source, row))
    for output in paths:
        shutil.copyfile(output, dest / output.name)
    shutil.copyfile(path, dest / path.name)
    print(f'{name}: restored verified outputs from {previous.name}', flush=True)
    return result


def execute(run, bundle, meta, graph, raw, ddk, diagnosis, config, *, recovery=None, from_stage=None,
            entry=None, out=None, shared_silu=None, embed_decoder_hidden=False, template_root=None):
    runtime = load(config.parent/load(config)['assets'])['environment']
    os.environ.setdefault('BOARD_HIAI_LIB_DIR', str(Path(DEFAULT_TOOL).parent))
    fixture_hash = sha(bundle/'manifest.json')
    files = {s['path']: s for s in meta['files']}
    feeds = {}
    if meta.get('layer') and (entry is None) != meta['standalone']:
        raise ValueError('Standalone decoders require the measured layer entry; the 40-layer chain requires its previous CPro hidden')
    for edge, wire in graph['entries'].items():
        if meta.get('layer') and (wire['dtype'], wire['shape']) == ('float32', [1, 64, 2048]):
            feeds[edge] = entry if entry is not None else checked(bundle, files['inputs/layer_entry.bin'])
            continue
        filename = original_filename(wire['name'], wire, wire['dtype'], fresh_zero_states=True)
        feeds[edge] = checked(bundle, files['inputs/' + filename])
    if meta.get('layer') and sum((s['dtype'], s['shape']) == ('float32', [1, 64, 2048])
                                 for s in graph['entries'].values()) != 1:
        raise ValueError('Expected one FP32 layer entry')
    previous, checkpoints = checkpoint_source(run, recovery, fixture_hash)
    names = [s['name'] for s in graph['stages']]
    projections_split = any(s.startswith('decoder_w4_projection_') for s in names)
    residual_split = 'decoder_attention_residual' in names
    silu_count = sum(name.endswith('_silu') for name in names)
    print(f'L{meta.get("layer", 0)}: {len(names)} OMC stages including {silu_count} standalone SiluPatch',
          flush=True)
    if from_stage == 'decoder_hidden' and projections_split:
        from_stage = next(n for n in names if n.startswith('decoder_w4_'))
    elif from_stage == 'decoder_hidden_after_w4' and residual_split:
        from_stage = 'decoder_attention_residual'
    if from_stage:
        if previous is None or from_stage not in names or names.index(from_stage) > len(checkpoints):
            raise ValueError(f'--from-stage needs a verified prefix in the selected replay: {from_stage}; stages={names}')
        checkpoints = checkpoints[:names.index(from_stage)]
    if out is None:
        out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    else:
        out.mkdir()
    receipt = dict(status='RUNNING', bundle_sha256=fixture_hash, stages=[], outputs={},
                   planned_stage_count=len(names), silu_stage_count=silu_count,
                   embed_decoder_hidden=embed_decoder_hidden, split_decoder_projections=projections_split,
                   split_decoder_residual=residual_split)
    save_json(out/'chain.json', receipt)
    print('Decoder split replay:', out, flush=True)
    try:
        for i, step in enumerate(graph['stages']):
            name = step['name']
            print(f"{name}: {step['inputs']} -> {step['outputs']}", flush=True)
            if step.get('grouped_matmuls'):
                print(f"{name}: using L{meta['layer']:02d} weights for {', '.join(step['grouped_matmuls'])}", flush=True)
            if 'projection' in step:
                print(f"Isolated {step['projection']}, weight sources: {step['weight_sources']}", flush=True)
            if 'casts' in step:
                print(f"{name} source AIR Cast nodes and their inputs: {step['casts']}", flush=True)
            if (template_root and step.get('compiler') != 'per_layer' and
                    not name.endswith('_silu') and name not in ('embedding', 'logits')):
                build, model, params = compile_template(run, bundle, step, graph, raw, ddk, diagnosis, template_root)
                feeds.update(params)
                weights = []
            else:
                build, model, weights = compile_stage(run, bundle, step, graph, raw, ddk, diagnosis, fixture_hash,
                                                      shared_silu, embed_decoder_hidden)
            for spec in build['outputs']:
                w = graph['wires'][spec['edge']]
                if w['shape'] is None:
                    dims = w['source_shape']
                    if dims and min(dims) > 0:
                        if typed_view(spec['dtype'], dims) != typed_view(spec['dtype'], spec['shape']):
                            raise ValueError(f"{name}: AIR shape {dims} differs from compiled output {spec}")
                    else:
                        dims = spec['shape']
                    w.update(shape=dims, dtype=spec['dtype'])
                    print(f"{name}: {spec['edge']} compiled {w['dtype']} {w['shape']}", flush=True)
            operands, paths = [], []
            for s in build['inputs']:
                path = feeds[s['edge']]
                if path.stat().st_size != s['bytes']:
                    raise ValueError(f'{name}: wrong upstream bytes for {s["edge"]}')
                operands.append(dict(s, sha256=sha(path)))
                paths.append(path)
            stage = out/name
            stage.mkdir()
            links = [dict(edge=s['edge'], sha256=s['sha256']) for s in operands]
            if i < len(checkpoints):
                try:
                    result = restore_stage(previous, stage, checkpoints[i], build['model']['sha256'],
                                           links, build['outputs'], fixture_hash)
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    print(f'{name}: checkpoint invalid in {previous}: {exc}. Re-executing from this stage.', flush=True)
                    checkpoints = []
            if i >= len(checkpoints):
                result = run_v2(stage, name, model, operands, paths, build['outputs'], runtime,
                                external_weights=weights, build_records=(build['model'], build.get('weights', [])),
                                fixture_sha256=fixture_hash,
                                slot_config=config if len(operands) > 1 or len(build['outputs']) > 1 else None)
            for s, r in zip(build['outputs'], result['outputs'], strict=True):
                if r['bytes'] != s['bytes']:
                    raise ValueError(f'{name}: output length differs at {s["edge"]}')
                feeds[s['edge']] = checked(stage, r)
            receipt['stages'].append(dict(name=name, model_sha256=result['omc_sha256'],
                                          result_sha256=sha(stage/'result.json'),
                                          inputs=links))
            for key, edge in graph['results'].items():
                if edge in step['outputs']:
                    receipt['outputs'][key] = dict(identity(feeds[edge], out), **graph['wires'][edge])
            save_json(out/'chain.json', receipt)
        receipt['status'] = 'EXECUTED'
    except Exception as exc:
        receipt.update(status='FAILED', failed_stage=name, error=str(exc))
        if 'projection' in step:
            receipt['projection'] = step['projection']
            print(f"{name} failed while testing {step['projection']}. See {out}/chain.json.", flush=True)
        raise
    finally:
        save_json(out/'chain.json', receipt)
    return out


def ssh_cpu(client, remote, staged):
    staged.mkdir()
    manifest = staged/'manifest.json'
    ssh_file(client, Path(remote)/manifest.name, manifest)
    for role, row in load(manifest)['outputs'].items():
        filename = row['path']
        if Path(filename).name != filename or filename != role + '.bin':
            raise ValueError(f'Invalid CPU output path: {filename}')
        ssh_file(client, Path(remote)/filename, staged/filename, row)


def write_matrices(path, scores, *, display=True):
    backends = ('cpu', '310p', 'cpro')
    first = scores['cpro']['310p']
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=['operator', 'scope', 'metric', 'actual', *backends])
        writer.writeheader()
        for role, scopes in first.items():
            for scope in scopes:
                for field in (*THRESHOLDS, 'mean_abs', 'max_abs', *ERROR_RATIO_FIELDS, 'reference_metrics_passed'):
                    show = display and field in ('cosine', 'relative_l2', 'error_ratio')
                    if show:
                        print(f'\n{role}  {scope}  {field}\nactual\\reference              CPU           310P           CPro')
                    for actual in backends:
                        row = {ref: scores[actual][ref][role][scope][field] for ref in backends}
                        writer.writerow(dict(operator=role, scope=scope, metric=field, actual=actual, **row))
                        if show:
                            print(actual.ljust(16), *(format(value, '14.9g') if value is not None else '           n/a'
                                                      for value in row.values()))
    Path(f.name).replace(path)


def compare(bundle, replay, *, allow_failed=False, remote_air=None, prepare_mode=None):
    receipt = load(replay/'chain.json')
    partial = allow_failed and receipt['status'] == 'FAILED'
    if (receipt['status'] != 'EXECUTED' and not partial) or receipt['bundle_sha256'] != sha(bundle/'manifest.json'):
        raise ValueError(f'Incomplete or unrelated L0 run: {replay}')
    meta = load(bundle/'manifest.json')
    if 'stages' in receipt:
        completed = receipt['stages']
        print(f'L{meta.get("layer", 0)}: {len(completed)} completed OMC stages in {replay.name}, '
              f'{sum(s["name"].endswith("_silu") for s in completed)} standalone SiluPatch', flush=True)
    reference = meta['reference']
    specs = contracts(meta.get('layer', 0), meta.get('head', True))
    missing = sorted(specs.keys() - receipt['outputs'].keys())
    if partial:
        print(f"Incomplete replay at {receipt['failed_stage']}; missing outputs: {missing}", flush=True)
        specs = {k: v for k, v in specs.items() if k in receipt['outputs']}
    cpu_dir = bundle.parent/'cpu'
    mode = ('linear' if not meta.get('layer', 0) else 'full') if meta.get('standalone', True) else '40'
    if not cpu_dir.exists():
        remote_cpu = (remote_air or bundle).parent/'cpu'
        print(f'Reading CPU reference from {remote_cpu} on the export host', flush=True)
        try:
            pull_air(bundle.parent, remote_cpu, destination_name='cpu', ssh_download=ssh_cpu)
        except FileNotFoundError as exc:
            # Paramiko reports SFTP ENOENT without a filename. Keep local filesystem errors intact.
            if exc.errno != errno.ENOENT or exc.filename is not None:
                raise
            remote_run = remote_cpu.parent if mode == 'linear' else remote_cpu.parent.parent
            raise FileNotFoundError(
                f'CPU reference missing or incomplete at {remote_cpu} on the export host. '
                f'On that host run: bash targets/cpro/scripts/split_prefill_t64.sh cpu '
                f'--mode {prepare_mode or mode} --run {shlex.quote(str(remote_run))}\nSaved board outputs: {replay}'
            ) from exc
    cpu = load(cpu_dir/'manifest.json')
    if ((cpu['basis'], cpu['mode'], cpu['layer'], cpu['capture_sha256']) !=
            (CPU_BASIS, mode, meta.get('layer', 0), reference['capture_sha256'])):
        raise ValueError(f'{cpu_dir}: CPU reference does not match the mode, layer and 310P capture')
    if not partial and set(receipt['outputs']) != set(specs):
        raise ValueError(f'{replay}: incomplete decoder outputs')
    values = {backend: {k: read_array(folder, records[k]) for k in specs}
              for backend, folder, records in [('cpu', cpu_dir, cpu['outputs']), ('310p', bundle, reference['tensors']),
                                               ('cpro', replay, receipt['outputs'])]}
    pairs = {a: {b: score(x, y, specs, reference_is_310p=b == '310p') for b, y in values.items()}
             for a, x in values.items()}
    results = pairs['cpro']['310p']
    passed = not partial and all(row['passed'] for scopes in results.values() for row in scopes.values())
    status = 'INCOMPLETE' if partial else ('PASSED' if passed else 'FAILED_ACCURACY')
    report = dict(status=status, physical_rows=64, valid_rows=19,
                  output_pos=18, conv_state_control=meta['fixture']['conv_state_control'],
                  conv_history_310p=reference['conv_history'], chain_sha256=sha(replay/'chain.json'),
                  reference_capture_sha256=reference['capture_sha256'], outputs=receipt['outputs'], scores=results)
    if partial:
        report.update(failed_stage=receipt['failed_stage'], missing_outputs=missing)
    path = replay/('partial-comparison.json' if partial else 'comparison.json')
    matrices = replay/('partial-matrices.csv' if partial else 'matrices.csv')
    write_matrices(matrices, pairs)
    report.update(matrices=identity(matrices, replay), cpu_reference_sha256=sha(cpu_dir/'manifest.json'),
                  gate_comparison='cpro_vs_310p')
    if 'logits' in results:
        report['top1_token18'] = {b: int(v['logits'].argmax()) for b, v in values.items()}
    save_json(path, report)
    print('Three-backend matrices:', matrices, flush=True)
    if 'logits' in results:
        print('Logits token18 top1:', report['top1_token18'])
    if reference['conv_history'] is not None:
        print('ConvState control:', report['conv_state_control'], '310P history:', report['conv_history_310p'])
    print(report['status'], path, flush=True)
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['air', 'board', 'compare'])
    p.add_argument('--recovery', nargs='?', const='auto', help='reuse verified prefix from latest or named replay')
    p.add_argument('--from-stage', help='with --recovery, execute again starting at this stage')
    variant = p.add_mutually_exclusive_group()
    variant.add_argument('--embed-decoder-hidden', action='store_true',
                   help='compile only decoder_hidden with weights embedded in its OMC')
    variant.add_argument('--split-decoder-projections', action='store_true',
                         help='execute each decoder_hidden W4 projection in a separate OMC before the residual')
    variant.add_argument('--split-decoder-residual', action='store_true',
                         help='keep the W4 probes and split the remainder at the post-attention residual')
    for flag in ('run', 'baseline', 'source', 'config', 'capture', 'ddk', 'diagnosis', 'runtime-config',
                 'remote-air', 'replay'):
        p.add_argument('--' + flag, type=Path, required=flag == 'run')
    a = p.parse_args()
    if (a.embed_decoder_hidden or a.split_decoder_projections or a.split_decoder_residual) and a.mode != 'board':
        p.error('decoder variants apply to board')
    run = output_path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    bundle = run/'air'
    if a.mode == 'air':
        if (bundle/'manifest.json').is_file():
            meta = load(bundle/'manifest.json')
            if (not meta.get('softplus_patch', {}).get('nodes') or meta['kind'] != L0_KIND
                    or meta['original_manifest_sha256'] != sha(a.baseline/'air/manifest.json')
                    or meta['reference']['capture_sha256'] != sha(a.capture/'manifest.json')):
                raise ValueError(f'{bundle}: different export version, baseline or capture. Use a new --run.')
            print(f'Existing corrected L0 AIR: {bundle}', flush=True)
            return 0
        export(run, a.source, a.config, None, None, baseline=a.baseline, kind=L0_KIND, l0_capture=a.capture)
        return 0
    if a.mode == 'board':
        select_board_defaults(a)
        if not bundle.exists():
            pull_air(run, a.remote_air or bundle, ssh_download=partial(ssh_bundle, kind=L0_KIND))
        meta = load(bundle/'manifest.json')
        if meta['kind'] != L0_KIND or meta.get('layer', 0) != 0 or not meta.get('head', True):
            raise ValueError(f'Expected {L0_KIND} at {bundle}')
        for row in meta['files']:
            checked(bundle, row)
        raw = (bundle/L0_DIR/'export.air').read_bytes()
        graph = plan(raw, meta['silu_patch']['nodes'], split_projections=a.split_decoder_projections,
                     split_residual=a.split_decoder_residual)
        name = 'plan-w4-projections.json' if a.split_decoder_projections else 'plan.json'
        save_json(run/('plan-w4-residual.json' if a.split_decoder_residual else name), graph)
        replay = execute(run, bundle, meta, graph, raw, a.ddk.resolve(strict=True), a.diagnosis,
                         a.runtime_config.resolve(strict=True), recovery=a.recovery, from_stage=a.from_stage,
                         embed_decoder_hidden=a.embed_decoder_hidden)
    else:
        if a.replay:
            replay = a.replay
        else:
            files = sorted(run.glob('replay-*/chain.json'), key=lambda f: f.stat().st_mtime_ns, reverse=True)
            replay = next((f.parent for f in files if load(f)['status'] == 'EXECUTED'), None)
            if not replay:
                p.error(f'No completed L0 chain in {run}')
    return 0 if compare(bundle, replay, allow_failed=a.mode == 'compare' and a.replay is not None,
                        remote_air=a.remote_air) else 1


if __name__ == '__main__':
    sys.exit(main())

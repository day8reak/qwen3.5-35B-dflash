#!/usr/bin/env python3
"""Compile and replay the one-layer native CPro decoder with four singleton SiluPatch OMCs."""
import argparse
from functools import partial
import hashlib
import json
from math import prod
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from decoder4_original import (L0_DIR, L0_KIND, compile_log_text, compile_omg, export,
                               original_filename, ssh_bundle)
from decoder4_prefill import pull_air
from export_silu_patch_air import build_air
from parse_model_json import GE_PROTO, inspect, read_omc
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, read_array, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import run_v2
from qwen35_moe_dflash.target.prefill_t64.l0_split import plan, score, slice_air


def load(path):
    return json.loads(path.read_bytes())


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
            candidates = [e for e in pending if typed_view(wires[e]['dtype'], wires[e]['shape']) == key]
            if direction == 'inputs':
                candidates = [e for e in candidates if names.get(name) == e]
            elif name in pending:
                candidates = [e for e in candidates if e == name]
            if len(candidates) != 1:
                raise ValueError(f'{step["name"]}: cannot bind {direction} {row} to {candidates}')
            edge, = candidates
            pending.remove(edge)
            if prod(wires[edge]['shape']) * np.dtype(dtype).itemsize != row['logical_bytes']:
                raise ValueError(f'Compiled byte extent differs at {edge}')
            rows.append(dict(name=name, edge=edge, shape=row['shape'], dtype=dtype, bytes=row['logical_bytes']))
        if pending:
            raise ValueError(f'{step["name"]}: omitted {direction}: {pending}')
        io[direction] = rows
    return io


def compile_stage(run, bundle, step, graph, raw, ddk, diagnosis, bundle_hash):
    name = step['name']
    wires = graph['wires']
    stage_air = (build_air(shape=wires[step['inputs'][0]]['shape']) if name.endswith('_silu')
                 else slice_air(raw, step, wires))
    dest = run / 'omc' / name
    parent = dest.parent
    parent.mkdir(exist_ok=True)
    source = dict(bundle_sha256=bundle_hash, air_sha256=hashlib.sha256(stage_air).hexdigest(), ddk=str(ddk))
    if not dest.exists():
        out = Path(tempfile.mkdtemp(prefix=name + '-', dir=parent))
        folder = out / L0_DIR
        folder.mkdir()
        (folder/'export.air').write_bytes(stage_air)
        for p in (bundle / L0_DIR).iterdir():
            if p.name != 'export.air':
                (folder / p.name).symlink_to(p, target_is_directory=p.is_dir())
        code, _, _ = compile_omg(out, L0_DIR + '/export.air', ddk, diagnosis, platform=None)
        if code:
            raise RuntimeError(f'{name}: OMG exited {code}\n{compile_log_text(out / "compile.log")}')
        model, = out.rglob('*.omc')
        weights = [identity(p, out) for p in sorted(model.parent.rglob('*.weight'))]
        abi = inspect(read_omc(model, ddk_root=ddk)[0], enum=GE_PROTO)
        io = bind(step, wires, abi)
        save_json(out/'manifest.json', dict(source=source, model=identity(model, out), weights=weights, **io))
        out.rename(dest)
    meta = load(dest/'manifest.json')
    if meta['source'] != source:
        raise ValueError(f'{dest} belongs to a different AIR or DDK. Select a new L0_SPLIT_RUN.')
    return meta, checked(dest, meta['model']), [checked(dest, row) for row in meta['weights']]


def execute(run, bundle, meta, graph, raw, ddk, diagnosis, config):
    runtime = load(config.parent/load(config)['assets'])['environment']
    os.environ.setdefault('BOARD_HIAI_LIB_DIR', str(Path(DEFAULT_TOOL).parent))
    fixture_hash = sha(bundle/'manifest.json')
    files = {s['path']: s for s in meta['files']}
    feeds = {}
    for edge, wire in graph['entries'].items():
        filename = original_filename(wire['name'], wire, wire['dtype'], fresh_zero_states=True)
        feeds[edge] = checked(bundle, files['inputs/' + filename])
    out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    receipt = dict(status='RUNNING', bundle_sha256=fixture_hash, stages=[], outputs={})
    save_json(out/'chain.json', receipt)
    print('L0 split replay:', out, flush=True)
    try:
        for step in graph['stages']:
            name = step['name']
            print(f"{name}: {step['inputs']} -> {step['outputs']}", flush=True)
            build, model, weights = compile_stage(run, bundle, step, graph, raw, ddk, diagnosis, fixture_hash)
            operands, paths = [], []
            for s in build['inputs']:
                path = feeds[s['edge']]
                if path.stat().st_size != s['bytes']:
                    raise ValueError(f'{name}: wrong upstream bytes for {s["edge"]}')
                operands.append(dict(s, sha256=sha(path)))
                paths.append(path)
            stage = out/name
            stage.mkdir()
            result = run_v2(stage, name, model, operands, paths, build['outputs'], runtime,
                            external_weights=weights, fixture_sha256=fixture_hash,
                            slot_config=config if len(operands) > 1 or len(build['outputs']) > 1 else None)
            for s, r in zip(build['outputs'], result['outputs'], strict=True):
                if r['bytes'] != s['bytes']:
                    raise ValueError(f'{name}: output length differs at {s["edge"]}')
                feeds[s['edge']] = checked(stage, r)
            receipt['stages'].append(dict(name=name, model_sha256=result['omc_sha256'],
                                          result_sha256=sha(stage/'result.json'),
                                          inputs=[dict(edge=s['edge'], sha256=s['sha256']) for s in operands]))
            for key, edge in graph['results'].items():
                if edge in step['outputs']:
                    receipt['outputs'][key] = dict(identity(feeds[edge], out), **graph['wires'][edge])
            save_json(out/'chain.json', receipt)
        receipt['status'] = 'EXECUTED'
    except Exception as exc:
        receipt.update(status='FAILED', failed_stage=name, error=str(exc))
        raise
    finally:
        save_json(out/'chain.json', receipt)
    return out


def compare(bundle, replay):
    receipt = load(replay/'chain.json')
    if receipt['status'] != 'EXECUTED' or receipt['bundle_sha256'] != sha(bundle/'manifest.json'):
        raise ValueError(f'Incomplete or unrelated L0 run: {replay}')
    meta = load(bundle/'manifest.json')
    reference = meta['reference']
    results = score({k: read_array(replay, v) for k, v in receipt['outputs'].items()},
                    {k: read_array(bundle, v) for k, v in reference['tensors'].items()})
    passed = all(row['passed'] for scopes in results.values() for row in scopes.values())
    report = dict(status='PASSED' if passed else 'FAILED_ACCURACY', physical_rows=64, valid_rows=19,
                  output_pos=18, conv_state_control=meta['fixture']['conv_state_control'],
                  conv_history_310p=reference['conv_history'], chain_sha256=sha(replay/'chain.json'),
                  reference_capture_sha256=reference['capture_sha256'], outputs=receipt['outputs'], scores=results)
    path = replay/'comparison.json'
    save_json(path, report)
    for role, scopes in results.items():
        for scope, m in scopes.items():
            print(f'{role:20} {scope:10} cosine={m["cosine"]} rel_L2={m["relative_l2"]} '
                  f'RMSE={m["rmse"]} error_ratio={m["error_ratio"]} passed={m["passed"]}')
    logits = results['logits']['token18']
    print(f'Token 18 top1: CPro={logits["actual_top1"]} 310P={logits["native_top1"]}')
    print('ConvState control:', report['conv_state_control'], '; 310P history:', report['conv_history_310p'])
    print(report['status'], path, flush=True)
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['air', 'board', 'compare'])
    for flag in ('run', 'baseline', 'source', 'config', 'capture', 'ddk', 'diagnosis', 'runtime-config',
                 'remote-air', 'replay'):
        p.add_argument('--' + flag, type=Path, required=flag == 'run')
    a = p.parse_args()
    run = output_path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    bundle = run/'air'
    if a.mode == 'air':
        export(run, a.source, a.config, None, None, baseline=a.baseline, kind=L0_KIND, l0_capture=a.capture)
        return 0
    if a.mode == 'board':
        if not bundle.exists():
            pull_air(run, a.remote_air or bundle, ssh_download=partial(ssh_bundle, kind=L0_KIND))
        meta = load(bundle/'manifest.json')
        if meta['kind'] != L0_KIND:
            raise ValueError(f'Expected {L0_KIND} at {bundle}')
        for row in meta['files']:
            checked(bundle, row)
        raw = (bundle/L0_DIR/'export.air').read_bytes()
        graph = plan(raw, meta['silu_patch']['nodes'])
        save_json(run/'plan.json', graph)
        replay = execute(run, bundle, meta, graph, raw, a.ddk.resolve(strict=True), a.diagnosis,
                         a.runtime_config.resolve(strict=True))
    else:
        if a.replay:
            replay = a.replay
        else:
            files = sorted(run.glob('replay-*/chain.json'), key=lambda f: f.stat().st_mtime_ns, reverse=True)
            replay = next((f.parent for f in files if load(f)['status'] == 'EXECUTED'), None)
            if not replay:
                p.error(f'No completed L0 chain in {run}')
    return 0 if compare(bundle, replay) else 1


if __name__ == '__main__':
    sys.exit(main())

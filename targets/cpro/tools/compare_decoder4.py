#!/usr/bin/env python3
"""Compare saved native-boundary four-layer HIAI replays with measured 310P outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from decoder4_original import ALIGNED_KIND, EXPORT_DIR, KIND, PATCH_KIND, original_filename
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.target.layer_probe.common import checked, read_array, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.chain_reference import VALID19_IDS_SHA256
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import (
    OUTPUTS, inspect_air, latest_replay, output_roles, output_shape, same_boundary, view,
)
from qwen35_moe_dflash.target.prefill_t64.segment_export import compare_segment_output


def load(path):
    return json.loads(path.read_bytes())


def golden_records(manifest):
    if (manifest.get('status') != 'CAPTURED' or manifest.get('layers') != [0, 1, 2, 3]
            or (manifest.get('physical_rows'), manifest.get('valid_rows'), manifest.get('output_pos')) != (64, 19, 18)
            or manifest.get('logits_basis') != 'original final norm and LM head on L3 decoder output at index 18'
            or set(manifest.get('outputs', {})) != set(OUTPUTS)):
        raise ValueError('Expected a measured 310P four-layer prefix capture with token18 logits and eight states')
    embedding = manifest['inputs']['embedding']
    if embedding['shape'] != [1, 64, 2048] or embedding['dtype'] != 'float16':
        raise ValueError('Expected the measured 310P FP16 embedding at the L0 entry')
    rows = [manifest['inputs']['input_ids'], embedding]
    for name in OUTPUTS:
        row = manifest['outputs'][name]
        if (row['name'] != name or tuple(row['shape']) != output_shape(name) or
                row['dtype'] not in ('float16', 'float32')):
            raise ValueError(f'Unexpected 310P {name} descriptor: {row}')
        rows.append(row)
    for row in rows:
        if Path(row['path']).parts != (row['path'],) or row['path'] in ('', '.', '..', 'manifest.json'):
            raise ValueError(f'Unexpected capture tensor path: {row["path"]}')
    return rows


def read_golden(root):
    manifest = load(root / 'manifest.json')
    ids, entry, *outputs = golden_records(manifest)
    ids = read_array(root, ids)
    if ids.shape != (1, 64) or hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest() != VALID19_IDS_SHA256:
        raise ValueError('The 310P capture uses a different prompt')
    arrays = {s['name']: read_array(root, s) for s in outputs}
    if int(np.argmax(arrays['logits'])) != manifest['four_layer_top1']:
        raise ValueError('The 310P logits disagree with the captured four-layer top1')
    return arrays, read_array(root, entry), manifest


def publish_golden(capture, dest):
    manifest = capture / 'manifest.json'
    if dest.exists():
        if sha(dest / 'manifest.json') != sha(manifest):
            raise ValueError(f'Another capture is already published at {dest}')
        read_golden(dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            staged = Path(tmp) / 'golden'
            staged.mkdir()
            shutil.copyfile(manifest, staged / manifest.name)
            for row in golden_records(load(manifest)):
                shutil.copyfile(checked(capture, row), staged / row['path'])
            read_golden(staged)
            staged.rename(dest)
    share_for_transfer(dest)
    print(f'310P golden: {dest}', flush=True)


def fetch_golden(remote, dest):
    from decoder4_prefill import ssh_file
    from pull_artifacts import DEFAULT_CONFIG, connection, pull

    def fetch(client, folder, staged):
        staged.mkdir()
        manifest = staged / 'manifest.json'
        ssh_file(client, Path(folder) / manifest.name, manifest)
        for row in golden_records(load(manifest)):
            ssh_file(client, Path(folder) / row['path'], staged / row['path'], row)
        read_golden(staged)

    with connection(DEFAULT_CONFIG, 'air') as client:
        pull(client, dest, remote, download_to=fetch)


def read_replay(path, kinds):
    replay = latest_replay(path)
    root, air = replay.parent, replay.parent / 'air'
    run, io = load(replay / 'result.json'), load(replay / 'io.json')
    manifest, compiled = load(air / 'manifest.json'), load(root / 'omc/manifest.json')
    digest = sha(air / 'manifest.json')
    if (manifest['kind'] not in kinds or compiled['air_manifest_sha256'] != digest
            or compiled['primary']['sha256'] != run['omc_sha256'] or run.get('exitcode') != 0
            or run['input_hashes'] != [r['sha256'] for r in io['inputs']]):
        raise ValueError(f'Board receipt, inputs or model identity mismatch: {replay}')
    files = {r['path']: r for r in manifest['files']}
    boundary = inspect_air(checked(air, files[EXPORT_DIR + '/export.air']))
    inputs = {r['name']: r for r in boundary['inputs']}
    actual_inputs = {}
    for r in io['inputs']:
        src = inputs[r['name']]
        name = original_filename(r['name'], src, r['dtype'], fresh_zero_states=True)
        if files['inputs/' + name]['sha256'] != r['sha256']:
            raise ValueError(f'{replay}: {r["name"]} differs from the input recorded for the valid19 fixture ({name})')
        actual_inputs[r['name']] = (r['dtype'], view(r['shape']), r['sha256'])
    if len(actual_inputs) != len(io['inputs']):
        raise ValueError(f'Duplicate compiled inputs: {replay}')
    if manifest['fixture']['token_sha256'] != VALID19_IDS_SHA256:
        raise ValueError(f'{replay}: unexpected valid19 prompt')
    if len(io['outputs']) != len(run['outputs']):
        raise ValueError(f'Incomplete board outputs: {replay}')
    arrays, bindings = {}, []
    roles = output_roles(io['outputs'], boundary['outputs'])
    native_edges = {r['edge'] for r in boundary['outputs']}
    for i, (name, spec, record) in enumerate(zip(roles, io['outputs'], run['outputs'], strict=True)):
        if spec['dtype'] not in ('float16', 'float32') or record['bytes'] != spec['bytes']:
            raise ValueError(f'Wrong output {i} descriptor at {replay}')
        arrays[name] = read_array(replay, dict(record, dtype=spec['dtype'], shape=spec['shape'])).reshape(output_shape(name))
        bindings.append(dict(record, slot=i, name=name, edge=spec['name'], dtype=spec['dtype'], shape=spec['shape'],
                             binding='AIR producer' if spec['name'] in native_edges else 'NetOutput slot and shape'))
    info = dict(replay=str(replay), kind=manifest['kind'], air_manifest_sha256=digest, omc_sha256=run['omc_sha256'],
                original_manifest_sha256=manifest.get('original_manifest_sha256'), fixture=manifest['fixture'],
                outputs=bindings)
    return arrays, boundary, actual_inputs, info


def compare(original, modified, golden, out):
    ref, entry, capture = read_golden(golden)
    values, boundary, inputs, info = read_replay(original, {KIND})
    # Early original replays store this count instead of the measured entry hash.
    air = Path(info['replay']).parent / 'air'
    original_files = {r['path']: r for r in load(air / 'manifest.json')['files']}
    q = np.fromfile(checked(air, original_files['inputs/inputs_embeds.bin']), dtype='int8').reshape(1, 64, 2048)
    scales = np.fromfile(checked(air, original_files['inputs/embed_scale.bin']), dtype='<f4').reshape(1, 64, 1)
    rounded = (q.astype(np.float32) * scales).astype(np.float16)
    different = int(np.count_nonzero(rounded[:, :19].view(np.uint16) != entry[:, :19].view(np.uint16)))
    if different != info['fixture']['valid19_embedding_changed_fp16_elements']:
        raise ValueError('310P L0 embedding differs from the reference used to prepare the original board inputs')
    runs = {'original': info}
    pairs = [('original_vs_310p', values, ref)]
    if modified:
        alt, other, feeds, detail = read_replay(modified, {PATCH_KIND, ALIGNED_KIND})
        same_boundary(boundary, other)
        if (info['air_manifest_sha256'] != detail['original_manifest_sha256'] or inputs != feeds or
                any(values[k].dtype != alt[k].dtype for k in OUTPUTS)):
            raise ValueError('Modified replay has a different baseline, runtime inputs or output dtypes')
        runs['modified'] = detail
        pairs.extend([('modified_vs_310p', alt, ref), ('modified_vs_original', alt, values)])
    for run in runs.values():
        for name, digest in run['fixture'].get('reference_inputs', {}).items():
            if capture['inputs'][name]['sha256'] != digest:
                raise ValueError(f'310P {name} differs from the measured fixture used for {run["replay"]}')
    report = dict(golden=str(golden), golden_manifest_sha256=sha(golden / 'manifest.json'),
                  golden_resources=capture.get('resources'), golden_conv_history=capture.get('conv_history'),
                  valid19_embedding_changed_fp16_elements=different,
                  physical_rows=64, valid_rows=19, output_pos=18, runs=runs, comparisons={})
    for label, actual, expected in pairs:
        print(f'\n{label:<23} {"scope":<12} {"cosine":>12} {"rel_L2 %":>11} {"RMSE":>12} {"error_ratio %":>14}')
        scores = report['comparisons'][label] = {}
        for name in OUTPUTS:
            score = compare_segment_output(name, actual[name], expected[name], None,
                                           reference_is_310p=label.endswith('_310p'))
            if name == 'logits':
                score['reference_top1'] = score.pop('native_top1')
            scores[name] = score
            scope = 'token18' if name == 'logits' else 'valid19' if 'cache_3' in name else 'final_state'
            def number(metric, factor=1):
                v = score[metric]
                return f'{factor*v:.8g}' if v is not None else 'n/a'
            print(f'{name:<23} {scope:<12} {number("cosine"):>12} {number("relative_l2",100):>11} '
                  f'{number("rmse"):>12} {number("error_ratio",100):>14}')
            if name == 'logits':
                print(f'  top1 actual={score["actual_top1"]} reference={score["reference_top1"]}')
    if modified:
        delta = report['delta_relative_l2_pp'] = {}
        for name in OUTPUTS:
            a = report['comparisons']['original_vs_310p'][name]['relative_l2']
            b = report['comparisons']['modified_vs_310p'][name]['relative_l2']
            delta[name] = None if a is None or b is None else 100 * (b - a)
        if delta['logits'] is not None:
            print(f'\nLogits relative L2 change versus 310P: {delta["logits"]:+.6g} percentage points')
    if out is None:
        out = Path(tempfile.mkdtemp(prefix='comparison-', dir=output_path(Path(info['replay']).parent))) / 'comparison.json'
    save_json(out, report)
    print(f'\nComparison: {out}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command', required=True)
    pack = modes.add_parser('golden', help='publish the measured 310P four-layer readout and states')
    pack.add_argument('--capture', required=True, type=Path)
    pack.add_argument('--output', required=True, type=Path)
    p = modes.add_parser('compare', help='compare saved original and optional modified board outputs')
    p.add_argument('--original', required=True, type=Path, help='original run or explicit replay directory')
    p.add_argument('--modified', type=Path, help='modified run or explicit replay directory')
    p.add_argument('--golden', type=Path, required=True, help='local four-layer 310P capture or published subset')
    p.add_argument('--remote-golden', type=Path, help='fetch measured tensors from 78 if the local golden is absent')
    p.add_argument('--output', type=Path, help='write report to this JSON under AI_RUN_DIR')
    a = parser.parse_args()
    if a.command == 'golden':
        publish_golden(a.capture.resolve(), output_path(a.output))
        return
    if not a.golden.exists() and a.remote_golden:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        fetch_golden(a.remote_golden, output_path(a.golden))
    compare(a.original, a.modified, a.golden.resolve(), output_path(a.output) if a.output else None)


if __name__ == '__main__':
    main()

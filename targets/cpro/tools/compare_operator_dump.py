#!/usr/bin/env python3
"""Compare saved CPro operator dumps to CPU/310P intermediates, without running a model."""
import argparse
import csv
from graphlib import TopologicalSorter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'model'))
from gdn_matrices_txt import BACKENDS, render
from operator_dump_files import calibrate, decoded, index_files
from qwen35_moe_dflash.common.accuracy import THRESHOLDS, floating_metrics
from qwen35_moe_dflash.target.layer_probe.common import save_json


def air_points(config, root):
    from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message

    keys = set()
    for backend in BACKENDS[:2]:
        prefix = config.get(backend+'_prefix', '')
        with np.load(root/config[backend], allow_pickle=False) as archive:
            keys.update(k[len(prefix):] for k in archive.files if k.startswith(prefix))
    _, graph = graph_message((root/config['air']).read_bytes(), 'NetOutput')
    nodes = {op.name:op for op in graph.op}
    dependencies = {name:[e.rsplit(':',1)[0] for e in op.input if e and e.rsplit(':',1)[0] in nodes]
                    for name, op in nodes.items()}
    points = []
    for name in TopologicalSorter(dependencies).static_order():
        for port in range(len(nodes[name].output_desc)):
            edge = f'{name}:{port}'
            if edge in keys:
                points.append(dict(name=edge, edge=edge))
    return points


def mapped_profile(path):
    config = json.loads(path.read_bytes())
    root = path.parent
    inventory = json.loads((root/config['inventory']).read_bytes())
    if inventory['model_exitcode'] != 0 or inventory['parser_exitcode'] != 0:
        raise ValueError('The dump requires a successful model invocation and parser')
    points = config['points'] if 'points' in config else air_points(config, root)
    if not points or len({p['name'] for p in points}) != len(points):
        raise ValueError('Supply a nonempty, ordered list of unique comparison points')
    references = []
    for backend in BACKENDS[:2]:
        with np.load(root/config[backend], allow_pickle=False) as archive:
            refs = {}
            for p in points:
                key = p.get(backend, config.get(backend+'_prefix', '') + p['name'])
                if key in archive:
                    refs[p['name']] = archive[key]
            references.append(refs)

    def views(p, tensors, _peers):
        arrays = list(tensors)
        if 'cpro_permute' in p:
            arrays[2] = arrays[2].transpose(p['cpro_permute'])
        if 'shape' in p:
            arrays = [a.reshape(p['shape']) for a in arrays]
        elif len({a.shape for a in arrays}) > 1 and len({a.squeeze().shape for a in arrays}) == 1:
            arrays = [a.reshape(arrays[0].shape) for a in arrays]
        if len({a.shape for a in arrays}) != 1:
            raise ValueError(f'Logical shapes differ: {[a.shape for a in arrays]}')
        if 'token_axis' in p:
            axis, valid = p['token_axis'], p.get('valid_rows', config.get('valid_rows'))
            if valid is None or not 0 < valid <= arrays[0].shape[axis]:
                raise ValueError('Invalid valid_rows for token_axis')
            return [a.take(range(valid), axis=axis) for a in arrays], f'valid{valid}'
        return arrays, 'full_tensor'
    return dict(points=points, references=references, views=views, root=root/config['dump_dir'],
                files=inventory['files'], air_types={p['edge']:p.get('air_dtype') for p in points},
                notes=[], peers={}, anchors={}, observed={}, attempt=inventory['attempt'],
                output=root/'dump-comparison')


def pair_scores(a, b):
    if a.shape != b.shape or (a.dtype.kind in 'iub') != (b.dtype.kind in 'iub'):
        raise ValueError(f'Incompatible tensors: {a.shape}/{a.dtype} and {b.shape}/{b.dtype}')
    ints = a.dtype.kind in 'iub'
    if ints:
        if a.dtype != b.dtype:
            raise ValueError(f'Integer dtypes differ: {a.dtype}, {b.dtype}')
        mismatch = float(np.count_nonzero(a != b) / a.size)
        a, b = a.astype(np.float64), b.astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        s = floating_metrics(a, b)
    return {k:s[k] for k in THRESHOLDS} | {'error_ratio':mismatch if ints else s['error_ratio']}


def crosses(pair, args):
    cos, l2 = pair['cosine'], pair['relative_l2']
    return cos is None or l2 is None or cos <= args.min_cosine or l2 >= args.max_rel_l2


def run(profile, args):
    folder = args.output or profile['output']
    folder.mkdir(parents=True, exist_ok=True)
    index = index_files(profile['files'])
    root, air_types = profile['root'], profile['air_types']
    rules, evidence = calibrate(root, index, profile['anchors'], air_types)

    def candidates(edge):
        return decoded(root, index, edge, air_types.get(edge), rules, args.padding_offset)

    peers = {}
    for name, edge in profile['peers'].items():
        options, skipped = candidates(edge)
        if len(options) == 1 and not skipped:
            peers[name] = options[0][0]
    results, unresolved = {}, []
    first = first_candidate = None
    csv_path = folder/'matrices.csv'
    metrics = ('cosine', 'relative_l2', 'error_ratio', 'kl_divergence', 'rmse', 'mean_relative_error')
    with csv_path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('stage','tensor','source_variable','scope','metric','actual/reference',*BACKENDS))
        for point in profile['points']:
            name, edge = point['name'], point.get('edge')
            row = results[name] = dict(edge=edge, variants=[], issues=[])
            if name in profile['observed']:
                cpro, public_path = profile['observed'][name]
                variants = [(cpro, [dict(public_output=public_path)])]
            else:
                variants, skipped = candidates(edge)
                row['issues'].extend(f'unsupported byte length: {p}' for p in skipped)
            if not variants:
                row['issues'].append('No matching dump' if edge else 'AIR edge unresolved')
            if not all(name in ref for ref in profile['references']):
                row['issues'].append('CPU or 310P reference is missing')
                variants = []
            for i, (actual, sources) in enumerate(variants, 1):
                try:
                    tensors = [ref[name] for ref in profile['references']] + [actual]
                    arrays, scope = profile['views'](point, tensors, peers)
                    table = {b:{r:pair_scores(x,y) for r,y in zip(BACKENDS, arrays)}
                             for b,x in zip(BACKENDS, arrays)}
                except (KeyError, IndexError, ValueError) as exc:
                    row['issues'].append(dict(files=sources, reason=str(exc)))
                    continue
                bad = crosses(table['cpro']['310p'], args)
                row['variants'].append(dict(scope=scope, sources=sources, scores=table, crosses=bad))
                label = name if len(variants) == 1 else f'{name} [candidate {i}]'
                for metric in metrics:
                    for b in BACKENDS:
                        writer.writerow((point.get('stage','model'), label, point.get('description',name), scope,
                                         metric,b,*(table[b][r][metric] for r in BACKENDS)))
            rows = row['variants']
            if not rows or row['issues'] or len({v['crosses'] for v in rows}) > 1:
                row['status'] = 'AMBIGUOUS' if rows else 'UNAVAILABLE'
                unresolved.append(name)
            else:
                row['status'] = ('LARGE_ERROR' if rows[0]['crosses'] else 'WITHIN_COS_L2')
                if len(rows) > 1:
                    row['status'] += '_ALL_CANDIDATES'
            if rows and any(v['crosses'] for v in rows) and first_candidate is None:
                first_candidate = name
            if row['status'].startswith('LARGE_ERROR') and first is None:
                pairs = [r['scores']['cpro']['310p'] for r in rows]
                first = dict(name=name, edge=edge, scope=rows[0]['scope'], readings=len(rows),
                             cpro_vs_310p=[{k:v[k] for k in ('cosine','relative_l2')} for v in pairs],
                             earlier_unresolved=list(unresolved))
            print(f'{name:24} {edge or "?" :30} {row["status"]}', flush=True)
    status = ('FIRST_OBSERVED_WITH_GAPS' if first['earlier_unresolved'] else 'FIRST_LARGE_ERROR') if first else (
             'INCOMPLETE' if unresolved else 'WITHIN_COS_L2')
    result = dict(status=status, attempt=profile['attempt'], threshold=dict(min_cosine=args.min_cosine,
                  max_relative_l2=args.max_rel_l2), order=list(results), unresolved=unresolved, first_large_error=first,
                  first_candidate_crossing=first_candidate, calibration=evidence, mapping_notes=profile['notes'],
                  points=results)
    save_json(folder/'comparison.json', result)
    if any(r['variants'] for r in results.values()):
        render(csv_path, metrics=('cosine','relative_l2','error_ratio'))
    else:
        for metric in ('cosine','relative_l2','error_ratio'):
            (folder/(metric+'.txt')).write_text('没有可对齐的三端数据。\n')
    if first:
        print(f'首个可判定大偏差的观测点：{first["name"]} ({first["edge"]}, {first["scope"]}) '
              f'CPro/310P {first["cpro_vs_310p"]}')
        if first['earlier_unresolved']:
            print('此前尚无法判定：' + ', '.join(first['earlier_unresolved']))
    else:
        print(f'可判定的点未发现超限，未解决 {len(unresolved)} 项。')
    if first_candidate and (not first or first_candidate != first['name']):
        print(f'更早有候选读法超限：{first_candidate}，见候选矩阵。')
    for note in profile['notes']:
        print(note)
    print(f'{status}  {folder}/comparison.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', type=Path, help='auto-map a complete L0 linear prefill case')
    group.add_argument('--mapping', type=Path, help='ordered reference and AIR port mapping for any layer/model')
    p.add_argument('--output', type=Path)
    p.add_argument('--min-cosine', type=float, default=0.99)
    p.add_argument('--max-rel-l2', type=float, default=0.05)
    p.add_argument('--padding-offset', type=int, choices=(0,32), help='confirmed payload offset for +32-byte dumps')
    args = p.parse_args()
    if not -1 <= args.min_cosine <= 1 or not 0 < args.max_rel_l2 < float('inf'):
        p.error('invalid large-error threshold')
    if args.case:
        from decoder_dump_profile import load
        profile = load(args.case)
    else:
        profile = mapped_profile(args.mapping)
    run(profile, args)


if __name__ == '__main__':
    main()

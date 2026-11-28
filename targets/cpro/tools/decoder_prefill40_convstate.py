"""Check the public L0 physical ConvState against the same 40-layer replay's pre-convolution QKV."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from decoder4_original import FULL_DIR
from decoder_l0_convstate_dump import dump_encoding, view_chain
from operator_dump_files import FILENAME, decoded, index_files
from qwen35_moe_dflash.target.layer_probe.common import checked, metrics, read_tensor, save_json, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_steps import CONCATS, only, operands
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message


def dump_plan(bundle, meta):
    boundary = meta['boundary']
    air, = [r for r in meta['files'] if r['path'] == FULL_DIR + '/export.air']
    _, graph = graph_message(checked(bundle, air).read_bytes(), 'NetOutput')
    nodes = {n.name: n for n in graph.op}
    types = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number
    cache = only((r['name'] + ':0' for r in boundary['inputs'] if r['state'] == 'conv_state_0'), 'L0 ConvState input')
    join = nodes[only((n.name for n in graph.op if n.type in CONCATS and cache in operands(n)),
                      'L0 history concat')]
    current = only((e for e in operands(join) if e != cache and nodes[e.rsplit(':', 1)[0]].type != 'Const'),
                   'L0 projected QKV feeding the history concat')
    taps = {}

    def add_views(edge, role):
        for src in view_chain(nodes, edge):
            name, port = src.rsplit(':', 1)
            dtype = types[nodes[name].output_desc[int(port)].dtype].name
            taps[src] = dict(role=role, air_dtype=dtype)
        return nodes[name]

    last = add_views(current, 'channel_major_qkv')
    if last.type in ('Transpose', 'TransposeD', 'Permute'):
        add_views(operands(last)[0], 'token_major_qkv')
    # Public outputs corroborate the parser group even if the cache itself is wrong.
    for row in boundary['outputs']:
        if row['role'] in ('logits', 'conv_state_0'):
            add_views(row['edge'], row['role'])
    add_views(boundary['token_ids_input'] + ':0', 'input_ids')
    print(f'[L0 ConvState] {cache} → {join.name}, QKV dump edges: '
          f'{[e for e, t in taps.items() if t["role"].endswith("_qkv")]}', flush=True)
    return taps


def inspect(folder, receipt, outputs, taps, ids):
    inv = json.loads((folder/'dump-inventory.json').read_bytes())
    if (inv['attempt'] != receipt['attempt'] or inv['omc_sha256'] != receipt['omc_sha256']
            or inv['model_argv'] != receipt['argv'] or '--enable_item=2' not in inv['model_argv']
            or inv['model_exitcode'] != 0 or sorted(taps) != inv.get('requested_edges')):
        raise ValueError(f'{folder}: dump inventory does not belong to the requested instrumented replay')

    def public(role):
        row = outputs[role]
        return read_tensor(sized_file(folder, row), row).squeeze()

    cache = public('conv_state_0')
    if cache.shape != (8192, 4):
        raise ValueError(f'{folder}: unexpected public L0 ConvState axes {cache.shape}')
    anchors = dict(conv_state_0=cache, logits=public('logits'), input_ids=ids)
    groups = defaultdict(dict)
    for path, size in inv['files'].items():
        groups[str(Path(path).parent)][path] = size

    report = dict(status='INCOMPLETE', attempt=receipt['attempt'], omc_sha256=receipt['omc_sha256'],
                  valid_rows=64, token_range=[60, 63], public_state=outputs['conv_state_0'],
                  edges=taps, parser_status=inv['status'], parser_exitcode=inv['parser_exitcode'],
                  readings=[], issues=[], anchors=[])
    qkv = None
    corroborated = False
    for group, files in groups.items():
        index = index_files(files)
        indexed_paths = {dump.path for edge in taps for dump in index.get(edge, ())}
        unindexed = []
        for name in files.keys() - indexed_paths:
            match = FILENAME.fullmatch(Path(name).name)
            if match and f'{match[1]}:{int(match[5])}' in taps:
                unindexed.append(name)
        if unindexed:
            report['issues'].append(f'Files outside the expected AIR name, slot or size: {unindexed[:4]}')
        matched = []
        has_qkv = False
        for edge, tap in taps.items():
            role, dtype = tap['role'], tap['air_dtype']
            arrays, issues = decoded(folder/'dump', index, edge, dtype, {})
            if role in anchors:
                target = anchors[role]
                if (target.size and np.isfinite(target).all() and target.min() != target.max()
                        and any(np.array_equal(a.squeeze(), target) for a, _ in arrays)):
                    matched.append(edge)
                continue
            report['issues'].extend(issues)
            for raw, sources in arrays:
                # The parser adds singleton dimensions to logical ND dump files.
                value = raw.squeeze()
                if role == 'channel_major_qkv':
                    value = value.T
                encoding = dump_encoding(raw, dtype)
                report['readings'].append(dict(edge=edge, files=sources, encoding=encoding, shape=list(raw.shape)))
                if (value.shape != (64, 8192) or encoding['interpretation'] == 'unresolved'
                        or value.dtype.kind != 'f' or not np.isfinite(value).all()):
                    report['issues'].append(f'{sources[0]["path"]}: unresolved QKV axes, dtype or nonfinite values')
                elif qkv is not None and not np.array_equal(qkv, value):
                    report['issues'].append(f'{edge}: distinct QKV payloads in this export')
                else:
                    has_qkv = True
                    qkv = value
        if has_qkv and matched:
            corroborated = True
        if matched:
            report['anchors'].append(dict(directory=group, matching_edges=matched))

    if inv['parser_exitcode'] or inv['status'] != 'FILES_RECEIVED_REVIEW_REQUIRED':
        report['issues'].append('Parser did not return a complete selected dump')
    if qkv is None:
        report['issues'].append('No complete, graph-bound L0 QKV dump was recovered')
    if qkv is not None and not report['issues']:
        tail = qkv[60:64].T
        scores = metrics(cache[None], tail[None])
        exact = scores['exact_elements'] == scores['total_elements']
        report['state_vs_qkv_60_63'] = {k: scores[k] for k in
            ('exact_elements', 'total_elements', 'max_abs', 'cosine', 'nonfinite_actual', 'first_different_coordinates')}
        np.savez(folder/'l0-convstate.npz', mixed_qkv=qkv[None], conv_state_0=cache[None])
        if not np.any(tail):
            report['issues'].append('All four reference rows are zero')
        elif corroborated:
            report['status'] = 'PASS' if exact else 'FAIL'
        elif exact and tail.min() != tail.max():
            # All graph-bound candidates already agreed before looking at the public cache.
            report['anchors'].append(dict(role='conv_state_0', matching_qkv_tail_elements=int(tail.size)))
            report['status'] = 'PASS'
        else:
            report['issues'].append('No same-invocation tensor corroborates this QKV group')
        print(f'[L0 ConvState] exact={scores["exact_elements"]}/{scores["total_elements"]} '
              f'max_abs={scores["max_abs"]} cos={scores["cosine"]}', flush=True)
    destination = folder/'l0-convstate.json'
    save_json(destination, report)
    print(f'[L0 ConvState] {report["status"]} {destination}', flush=True)
    for issue in report['issues'][:4]:
        print(f'  {issue}', flush=True)
    return report['status'] == 'PASS'

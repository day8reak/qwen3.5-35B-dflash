"""Inspect the physical tail in a dump from one complete four-layer prefill."""
import json

import numpy as np

from decoder_gate_probe import air_bundle
from operator_dump_files import decoded, index_files
from qwen35_moe_dflash.common.accuracy import floating_metrics
from qwen35_moe_dflash.target.layer_probe.common import read_tensor, save_json, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_steps import AttentionSteps, MATMUL, NORMS, only, operands
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.gdn_layer_air import before_padding_mask


def measured_l0_reference(case, ref):
    """Use the sibling single-layer capture only when its L0 inputs and checkpoint match."""
    folder = case.parents[3]/'1-linear-prefill-t64/board/package/fixture'
    index = folder/'reference.json'
    if not index.is_file():
        return None
    other = json.loads(index.read_bytes())
    fields = ('valid_rows', 'physical_rows', 'model_file', 'quanted_pth', 'config_sha256', 'source_sha256')
    if any(other.get(k) != ref[k] for k in fields):
        return None
    for name in ('hidden', 'input_ids', 'effective_length', 'conv_state_0', 'recurrent_state_0'):
        if other['inputs'].get(name) != ref['inputs'][name]:
            return None
    if other['outputs'].get('conv_state_0') != ref['outputs']['conv_state_0'] or 'out_proj_0' not in other['outputs']:
        return None
    spec = other['outputs']['out_proj_0']
    return index, read_tensor(sized_file(folder, spec), spec)[0]


def report_padding(case, replay, fixture, ref, returned_cache):
    source, meta, record = air_bundle(case, case.parents[2]/'export/package'/case.name)
    _, graph = graph_message(sized_file(source, record).read_bytes(), 'ChunkGatedDeltaRule')
    walk = AttentionSteps(graph, {n.name: n for n in graph.op}, [])
    returned = {r['name']: r['edge'] for r in meta['outputs']}

    def entry_and_qkv(layer):
        history = returned[f'conv_state_{layer}']
        qkv = walk.nearest(history, {MATMUL})
        quant = walk.expect(walk.strip(walk.node(qkv).input[0]), {'DynamicQuant'})
        norm = walk.expect(walk.strip(quant.input[0]), NORMS)
        return walk.strip(norm.input[0]), qkv

    entry, _ = entry_and_qkv(0)
    block_out, qkv = entry_and_qkv(1)
    residual_branches = []
    for node in graph.op:
        inputs = operands(node)
        if node.type in ('Add', 'AddV2') and len(inputs) == 2 and [walk.strip(e) for e in inputs].count(entry) == 1:
            peer = next(e for e in inputs if walk.strip(e) != entry)
            if walk.node(before_padding_mask(walk, peer)).type == MATMUL and walk.depends(block_out, node.name + ':0'):
                residual_branches.append(peer)
    out = only(residual_branches, 'L0 attention contribution to its first residual')
    points = {'L0.out_proj': (before_padding_mask(walk, out), 2048), 'L0.to_residual': (out, 2048),
              'L0.block_out': (block_out, 2048), 'L1.mixed_qkv': (qkv, 8192)}
    inventory = json.loads((replay/'dump-inventory.json').read_bytes())
    index = index_files(inventory['files'])
    rows, valid = ref['physical_rows'], ref['valid_rows']
    refs = {}
    for backend, key in [('cpu', 'cpu'), ('310p', 'outputs')]:
        spec = ref[key]['conv_state_1']
        refs[backend] = read_tensor(sized_file(fixture, spec), spec)[0].T
    folder, spec = returned_cache
    refs['same_run_cpro_cache'] = read_tensor(sized_file(folder, spec), spec)[0].T
    taps = refs['310p'].shape[0]
    report = dict(attempt=inventory['attempt'], physical_rows=rows, valid_rows=valid, points={})
    print(f'[padding] 有效 {valid}/{rows} 行，ConvState 取行 {rows-taps}–{rows-1}。', flush=True)
    measured = measured_l0_reference(case, ref)
    if measured is not None:
        index_path, attention = measured
        report['single_layer_310p'] = dict(reference=str(index_path), tail_nonzero=int(np.count_nonzero(attention[-taps:])))
        print('[padding] 同入口的单层 310P out_proj_0:', report['single_layer_310p'], flush=True)
    for label, (edge, width) in points.items():
        node = walk.node(edge)
        slot = int(edge.rsplit(':', 1)[1])
        dtype = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number[node.output_desc[slot].dtype].name
        arrays, unsupported = decoded(replay/'dump', index, edge, dtype, {})
        readings = []
        for value, sources in arrays:
            matrix = value.squeeze()
            if matrix.shape != (rows, width) or matrix.dtype.kind != 'f':
                unsupported.extend(s['path'] for s in sources)
                continue
            tail = matrix[-taps:]
            reading = dict(padding_nonzero=int(np.count_nonzero(matrix[valid:])),
                           tail_nonzero=int(np.count_nonzero(tail)), tail_nonfinite=int((~np.isfinite(tail)).sum()))
            if label == 'L1.mixed_qkv':
                for backend, expected in refs.items():
                    scores = floating_metrics(tail, expected)
                    reading[backend] = {k: scores[k] for k in ('cosine', 'relative_l2')}
            if label == 'L0.to_residual' and measured is not None:
                scores = floating_metrics(tail, attention[-taps:])
                reading['vs_310p_tail_max_abs'] = scores['max_abs']
            print(f'  {label} {edge} {sources[0]["path"]}  {reading}', flush=True)
            readings.append(dict(sources=sources, **reading))
        if not readings:
            print(f'  {label} {edge}: 缺少可绑定的 [{rows},{width}] dump', flush=True)
        report['points'][label] = dict(edge=edge, readings=readings, unsupported=unsupported)
    report['status'] = 'COMPLETE' if all(p['readings'] and not p['unsupported'] for p in report['points'].values()) else 'INCOMPLETE'
    destination = replay/'gdn-padding.json'
    save_json(destination, report)
    print(f'[padding] {destination}', flush=True)

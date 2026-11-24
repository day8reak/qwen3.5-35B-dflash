"""Run the rest of a measured decoder tail with its original residual and saved CPro QLinear outputs."""
import math
import shutil

import numpy as np

from ...execution.serial_transport import _environment
from ..layer_probe.common import checked, save_json, sha
from . import whole_run
from .attention_slice import descriptor, extract
from .chain_hilog import diagnose_decoder_pair
from .decode_cache_control import graph_message
from .decoder_pair import load
from .decoder_projection import QUANT, wire_shape


def live_inputs(nodes, output, boundaries):
    """Stop at the nearest captured edges, including when branches share ancestors."""
    todo, visited, live = [output], set(), set()
    while todo:
        edge = todo.pop()
        if edge in boundaries:
            live.add(edge)
            continue
        op = nodes[edge.rsplit(':', 1)[0]]
        if op.name not in visited:
            visited.add(op.name)
            todo.extend(arg for arg in op.input if arg)
    return live


def projection_output(root, name, report, device, digest):
    case = root / name
    row = report['probes'][name]
    meta = load(case / 'air/manifest.json')
    compiled = load(case / 'omc/manifest.json')
    attempt = load(case / 'board/attempt.json')
    result = load(case / 'board/result.json')
    spec = result['output']
    input_spec, = compiled['io']['inputs']
    output_spec, = compiled['io']['outputs']
    if (meta['kind'] != whole_run._PROBE or meta['layer'] != report['layer']
            or meta['parent_air_sha256'] != report['tail_air_sha256']
            or meta['output_edge'] != row['output_edge'] or meta['input_edge'] != row['input_edge']
            or meta['quant_nodes'] != [meta['output_edge'].rsplit(':', 1)[0]]
            or compiled['air_manifest_sha256'] != sha(case / 'air/manifest.json')
            or input_spec['sha256'] != row['input_sha256']
            or attempt['omc_sha256'] != compiled['primary']['sha256']
            or attempt['gdr_source_sha256'] != digest or compiled['gdr_source_sha256'] != digest
            or (attempt['device'], attempt['server']) != device
            or row['status'] != 'EXECUTED' or result['status'] != 'EXECUTED' or row['output'] != spec
            or any(spec[k] != output_spec[k] for k in ('shape', 'dtype', 'bytes'))):
        raise ValueError(f'{name}: board output does not match this projection and tail')
    return checked(case / 'board', spec), spec


def prepare(probes, run, device, digest):
    report = load(probes / 'projections.json')
    layer = report['layer']
    if layer not in (0, 3):
        raise ValueError('Expected a native-SiLU L0 or L3 T64 decoder tail')
    case = probes.parent / f'L{layer}'
    folder = case / 'air'
    meta = load(folder / 'manifest.json')
    compiled = load(case / 'omc/manifest.json')
    attempt = load(case / 'board/attempt.json')
    spec, = meta['inputs']
    if (meta['kind'] != 'single-cpro-decoder-tail' or meta['layer'] != layer
            or meta['air']['sha256'] != report['tail_air_sha256'] or spec['sha256'] != report['residual_sha256']
            or sha(folder / 'manifest.json') != compiled['air_manifest_sha256']
            or compiled['primary']['sha256'] != report['tail_omc_sha256']
            or attempt['omc_sha256'] != report['tail_omc_sha256'] or compiled['gdr_source_sha256'] != digest
            or (attempt['device'], attempt['server']) != device):
        raise ValueError('Retained decoder tail, residual, projection report or board identity differs')
    msg, graph = graph_message(checked(folder, meta['air']).read_bytes(), 'NetOutput')
    nodes = {op.name: op for op in graph.op}
    expected = [(op.name, f'projection-{i:02d}') for i, op in
                enumerate(op for op in graph.op if op.type == QUANT)]
    if not expected or [(row['node'], row['probe']) for row in report['candidates']] != expected:
        raise ValueError('Projection report does not cover the quantized matmuls in this tail')
    candidates = {row['node'] + ':0': row for row in report['candidates']}
    residual, end = spec['name'] + ':0', meta['output_edge']
    live = live_inputs(nodes, end, {residual, *candidates})
    if residual not in live or len(live) < 2:
        raise ValueError(f'Expected a live residual and projection outputs, found {sorted(live)}')
    feeds = {residual: ('replay_residual', checked(folder, spec), spec)}
    reused = {}
    for edge, item in candidates.items():
        if edge not in live:
            continue
        name = item['probe']
        if item['status'] != 'EXECUTED' or report['probes'][name]['output_edge'] != edge:
            raise ValueError(f'No executed projection for required edge {edge}')
        path, rec = projection_output(probes, name, report, device, digest)
        prefix = item['input_probe']
        input_hash = spec['sha256']
        input_edge = residual
        if prefix != 'retained-input':
            prefix_row = report['probes'][prefix]
            if (prefix_row['status'] != 'EXECUTED' or prefix_row['input_sha256'] != input_hash
                    or prefix_row['input_edge'] != residual):
                raise ValueError(f'{prefix}: activation was not produced from the selected residual')
            input_hash, input_edge = prefix_row['output']['sha256'], prefix_row['output_edge']
        if (report['probes'][name]['input_sha256'] != input_hash or
                report['probes'][name]['input_edge'] != input_edge):
            raise ValueError(f'{name} used a different activation from {prefix}')
        feeds[edge] = (name.replace('-', '_'), path, rec)
        reused[edge] = dict(probe=name, sha256=rec['sha256'], weight_files=item['weight_files'])
    out = run / 'air'
    out.mkdir(parents=True)
    ports, inputs = {}, []
    for edge, (name, path, rec) in feeds.items():
        shape = wire_shape(rec, descriptor(nodes, edge))
        if math.prod(shape) * np.dtype(rec['dtype']).itemsize != rec['bytes']:
            raise ValueError(f'{edge}: incompatible tensor extent for {shape}')
        ports[edge] = name, rec['dtype'], shape
        copy = out / (name + '.bin')
        shutil.copyfile(path, copy)
        entry = dict(rec, path=copy.name, name=name, semantic_name=name, shape=shape)
        checked(out, entry)
        inputs.append(entry)
        print(f'Input {name}: {edge} {entry["dtype"]} {shape} ({path})', flush=True)
    cut = extract(msg, graph, folder, out, ports, end)
    kinds = {node['type'] for node in cut['nodes']}
    if QUANT in kinds or not any(kind.lower().startswith('groupedmatmul') for kind in kinds):
        raise ValueError(f'Unexpected routed-expert remainder topology: {sorted(kinds)}')
    weights = list(cut['weight_evidence'].values())
    inventory = {w['path']: w for w in meta['weights']}
    if any(w != inventory.get(w['path']) for w in weights):
        raise ValueError('Remainder weights differ from the original decoder tail')
    golden = out / 'golden.bin'
    shutil.copyfile(checked(folder, meta['golden']), golden)
    save_json(out / 'manifest.json', dict(kind=whole_run._REMAINDER, layer=layer, attention=meta['attention'],
              reference=meta['reference'], parent_air_sha256=report['tail_air_sha256'],
              projections_sha256=sha(probes / 'projections.json'), reused_outputs=reused, air=cut['air'],
              inputs=inputs, weights=weights, golden=dict(meta['golden'], path=golden.name),
              output_contract=meta['output_contract']))
    return layer


def run_remainder(probes, run, ddk, source, diagnosis, tool, config):
    _, serial, server = _environment()
    layer = prepare(probes, run, (serial, server), whole_run.kernel_identity(ddk, source))
    whole_run.compile_model(run, ddk, source, diagnosis)
    try:
        result = whole_run.board(run, tool, slot_config=config)
    except whole_run.HiaiRunFailure:
        diagnose_decoder_pair(run.parent, names=(run.name,))
        raise
    for scope in ('valid19', 'token18'):
        m = result[scope]
        print(f"L{layer} tail {scope}  cos={m['cosine']}  rel_L2={m['relative_l2']}  "
              f"RMSE={m['rmse']}  error_ratio={m['error_ratio']}", flush=True)
    print(f"HIAI executed; numerical validation {result['status']}: {run / 'board/accuracy.json'}", flush=True)
    return result['status'] == 'PASSED'

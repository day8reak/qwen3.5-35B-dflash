"""Compare complete native CPro prefill decoders, using only layer-entry 310P fixtures as inputs."""
import math
import shutil

import numpy as np

from ...execution.serial_transport import _environment
from ..layer_probe.common import checked, read_array, save_json, sha
from . import whole_run
from .attention_slice import descriptor, extract
from .chain_hilog import diagnose_decoder_pair
from .decode_cache_control import graph_message
from .decode_cache_trace import _project_input
from .decoder_pair import HIDDEN, load, score_hidden
from .decoder_projection import wire_shape
from .decoder_remainder import live_inputs
from .decoder_tail import quant_nodes, tail_edges


def plan(graph, entry):
    nodes = {op.name: op for op in graph.op}
    residual, _, final = tail_edges(graph)
    projections = quant_nodes(graph, residual, final)['tail']
    if not projections:
        raise ValueError('Expected quantized MoE projections in the complete decoder')
    stages = {residual: 'attention-residual', final: 'decoder-hidden'}
    for i, name in enumerate(projections):
        stages[name + ':0'] = f'projection-{i:02d}'
    for i, name in enumerate(projections):
        start = _project_input(nodes, name + ':0')
        if start not in entry:
            stages.setdefault(start, f'activation-{i:02d}')
    boundary = set(entry) | set(stages)
    order, planned = [], set()

    def visit(edge):
        if edge in entry or edge in planned:
            return
        inputs = sorted(live_inputs(nodes, edge, boundary - {edge}))
        if not inputs:
            raise ValueError(f'Stage {edge} has no layer-entry or upstream activation input')
        for dep in inputs:
            visit(dep)
        planned.add(edge)
        order.append((stages[edge], edge, inputs))

    visit(final)
    if residual not in planned:
        raise ValueError('The decoder output is disconnected from its post-attention residual')
    return order


def run_layer(case, root, layer, ddk, source, diagnosis, tool, config):
    air = case / 'air'
    meta = load(air / 'manifest.json')
    compiled = load(case / 'omc/manifest.json')
    attempt = load(case / 'board/attempt.json')
    _, device, server = _environment()
    attn = 'ChunkGatedDeltaRule' if layer == 0 else 'AdnFusedInferAttention'
    if (meta['kind'] != 'single-cpro-decoder' or meta['layer'] != layer or meta['attention'] != attn
            or sha(air / 'manifest.json') != compiled['air_manifest_sha256']
            or compiled['primary']['sha256'] != attempt['omc_sha256']
            or compiled['gdr_source_sha256'] != whole_run.kernel_identity(ddk, source)
            or (device, server) != (attempt['device'], attempt['server'])):
        raise ValueError(f'L{layer}: expected the retained complete native decoder on the selected board and DDK')
    raw = checked(air, meta['air']).read_bytes()
    _, graph = graph_message(raw, attn)
    entry = {s['name'] + ':0': (checked(air, s), s) for s in meta['inputs']}
    if set(entry) != {op.name + ':0' for op in graph.op if op.type == 'Data'}:
        raise ValueError('Layer input fixtures do not match the complete source AIR Data nodes')
    args = {s['semantic_name']: s for _, s in entry.values()}
    length = args['effective_length' if layer == 0 else 'all_length']
    if (args['hidden']['shape'] != list(HIDDEN) or args['hidden']['dtype'] != 'float16'
            or read_array(air, length).tolist() != [19]):
        raise ValueError('Expected a T64/valid19 captured layer entry')
    golden = checked(air, meta['golden'])
    weights = {s['path']: s for s in meta['weights']}
    stages = plan(graph, entry)
    trace = dict(layer=layer, mode='complete-decoder-split-omc', source_air_sha256=meta['air']['sha256'],
                 layer_entry={e: s for e, (_, s) in entry.items()}, stages=[])
    feeds = dict(entry)
    for name, edge, deps in stages:
        work = root / name
        folder = work / 'air'
        folder.mkdir(parents=True)
        message, graph = graph_message(raw, attn)
        nodes = {op.name: op for op in graph.op}
        ports, inputs, links = {}, [], []
        for i, dep in enumerate(deps):
            path, spec = feeds[dep]
            shape = wire_shape(spec, descriptor(nodes, dep))
            if math.prod(shape) * np.dtype(spec['dtype']).itemsize != spec['bytes']:
                raise ValueError(f'{dep}: byte count differs from its source view {shape}')
            port = f'block_input_{i:02d}'
            ports[dep] = port, spec['dtype'], shape
            dst = folder / (port + '.bin')
            shutil.copyfile(path, dst)
            row = dict(spec, name=port, semantic_name=port, path=dst.name, shape=shape)
            checked(folder, row)
            inputs.append(row)
            links.append(dict(port=port, edge=dep, sha256=row['sha256'],
                              source='310P_LAYER_ENTRY' if dep in entry else 'CPRO_STAGE_OUTPUT'))
        cut = extract(message, graph, air, folder, ports, edge)
        weight_rows = list(cut['weight_evidence'].values())
        if any(w != weights.get(w['path']) for w in weight_rows):
            raise ValueError(f'{name}: sliced weights differ from the complete layer')
        (folder / 'weight').mkdir(exist_ok=True)
        kinds = {op['type'] for op in cut['nodes']}
        if name == 'attention-residual' and attn not in kinds:
            raise ValueError(f'Missing original {attn} in the attention-to-residual stage')
        if name == 'decoder-hidden' and not any(k.lower().startswith('groupedmatmul') for k in kinds):
            raise ValueError('Expected routed experts in the final decoder stage')
        dtype = cut['output_desc']['dtype']
        declared = {'DT_FLOAT16': 'float16', 'DT_FLOAT': 'float32', 'DT_UNDEFINED': None, 0: None}
        save_json(folder / 'manifest.json', dict(kind=whole_run._BLOCK_STAGE, layer=layer, air=cut['air'],
                  parent_air_sha256=trace['source_air_sha256'], output_edge=edge, output_dtype=declared[dtype],
                  inputs=inputs, weights=weight_rows))
        stage = dict(name=name, output_edge=edge, inputs=links, status='COMPILING')
        trace['stages'].append(stage)
        save_json(root / 'stages.json', trace)
        print(f'L{layer} {name} on CPro: {deps} -> {edge}', flush=True)
        try:
            whole_run.compile_model(work, ddk, source, diagnosis)
            stage['status'] = 'RUNNING'
            save_json(root / 'stages.json', trace)
            receipt = whole_run.board(work, tool, slot_config=config if len(inputs) > 1 else None)
        except Exception as exc:
            stage.update(status='FAILED', failure_stage=getattr(exc, 'stage', stage['status']), error=str(exc))
            save_json(root / 'stages.json', trace)
            if isinstance(exc, whole_run.HiaiRunFailure):
                diagnose_decoder_pair(root, names=(name,))
            raise
        out = receipt['output']
        output = checked(work / 'board', out)
        feeds[edge] = output, out
        stage.update(status='EXECUTED', output=out)
        save_json(root / 'stages.json', trace)

    _, final_edge, _ = stages[-1]
    path, spec = feeds[final_edge]
    if [d for d in spec['shape'] if d != 1] != [64, 2048]:
        raise ValueError(f'Final hidden has an unexpected token or channel dimension: {spec["shape"]}')
    result = score_hidden(air, meta, path, spec)
    result.update(scope='complete decoder from its measured 310P layer entry through the final residual',
                  execution='split OMC on CPro', golden=str(golden), output=str(path), output_edge=final_edge,
                  stages_sha256=sha(root / 'stages.json'))
    save_json(root / 'accuracy.json', result)
    for scope in ('valid19', 'token18'):
        m = result[scope]
        print(f"L{layer} complete decoder {scope}: rel_L2={m['relative_l2']} cosine={m['cosine']} "
              f"RMSE={m['rmse']} error_ratio={m['error_ratio']}", flush=True)
    return result


def run_pair(parent, root, ddk, source, diagnosis, tool, config):
    results = {}
    for layer in (0, 3):
        key = f'L{layer}'
        try:
            result = run_layer(parent / key, root / key, layer, ddk, source, diagnosis, tool, config)
            results[key] = {k: result[k] for k in ('scope', 'execution', 'output', 'status', 'valid19', 'token18')}
        except Exception as exc:
            results[key] = dict(status='ERROR', error=str(exc))
            print(f'{key} complete decoder failed: {exc}', flush=True)
        save_json(root / 'comparison.json', results)
    print(root / 'comparison.json', flush=True)
    return all(r['status'] == 'PASSED' for r in results.values())

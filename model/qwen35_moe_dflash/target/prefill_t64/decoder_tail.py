"""Isolate the original decoder's post-attention norm, MoE and final residual."""
from collections import Counter
import shutil

from ..layer_probe.common import checked, save_json, sha
from . import whole_run
from .attention_slice import extract
from .chain_reference import HIDDEN, TAIL_BASIS
from .decoder_pair import load
from .decode_cache_control import graph_message


def tail_edges(graph, *, target=None, attention=None):
    nodes = {op.name: op for op in graph.op}
    if target is None:
        net, = [op for op in graph.op if op.type == 'NetOutput']
        target, = [e for e in net.input if e and not e.endswith(':-1')]

    def producer(edge):
        return nodes[edge.rsplit(':', 1)[0]]

    def source_aliases(edge, *, views=False):
        aliases = [edge]
        transparent = ('Identity', 'TensorMove', 'Cast')
        if views:
            transparent += ('Reshape', 'Squeeze', 'Unsqueeze', 'ExpandDims')
        while producer(edge).type in transparent:
            edge = producer(edge).input[0]
            aliases.append(edge)
        return aliases

    final = producer(source_aliases(target)[-1])
    adds = ('Add', 'AddV2')
    operands = [e for e in final.input if e and not e.endswith(':-1')]
    if final.type not in adds or len(operands) != 2:
        raise ValueError(f'Expected decoder output from final residual addition, got {final.name} {final.type}')
    # The split AIR exposes out_proj directly. This also identifies the skip
    # connection when post-attention normalization was lowered to several ops.
    if attention is not None:
        skips = []
        for operand in operands:
            skip = source_aliases(operand, views=True)[-1]
            if producer(skip).type in adds and any(attention in source_aliases(e, views=True)
                    for e in producer(skip).input if e and not e.endswith(':-1')):
                skips.append(skip)
        if len(skips) == 1:
            return skips[0], None, target
    matches = []
    for norm in graph.op:
        if 'rmsnorm' not in norm.type.lower() or not norm.input:
            continue
        path = source_aliases(norm.input[0], views=True)
        for operand in operands:
            skip = source_aliases(operand)[-1]
            if skip in path and producer(skip).type in adds:
                matches.append((skip, norm.name))
    if len(matches) != 1:
        raise ValueError(f'Expected one post-attention sum feeding RMSNorm and {final.name}, got {matches}')
    residual, norm_name = matches[0]
    return residual, norm_name, target


def quant_nodes(graph, residual, target):
    nodes = {op.name: op for op in graph.op}
    todo, tail = [target], set()
    while todo:
        edge = todo.pop()
        if edge == residual:
            continue
        op = nodes[edge.rsplit(':', 1)[0]]
        if op.name not in tail:
            tail.add(op.name)
            todo.extend(e for e in op.input if e)
    return {region: [op.name for op in graph.op if op.type == 'QuantBatchMatmulV4444'
                     and (op.name in tail) == retained]
            for region, retained in (('tail', True), ('outside_tail', False))}


def inspect_pair(parent):
    for layer in (0, 3):
        folder = parent / f'L{layer}/air'
        meta = load(folder / 'manifest.json')
        if meta['kind'] != 'single-cpro-decoder' or meta['layer'] != layer:
            raise ValueError(f'{folder}: expected a complete L{layer} decoder AIR')
        _, graph = graph_message(checked(folder, meta['air']).read_bytes(), 'NetOutput')
        residual, norm, target = tail_edges(graph)
        print(f'L{layer}: {residual} -> {norm} -> MoE -> {target}')
        for region, names in quant_nodes(graph, residual, target).items():
            print(f'  QuantBatchMatmulV4444 {region} ({len(names)}): {names}')


def prepare(parent, reference, run, layer):
    air = parent / 'air'
    original = load(air / 'manifest.json')
    ref = load(reference / 'manifest.json')
    if (original['kind'] != 'single-cpro-decoder' or original['layer'] != layer
            or ref['status'] != 'CAPTURED' or ref['reference'] != TAIL_BASIS
            or (ref['physical_rows'], ref['valid_rows'], ref['output_pos']) != (64, 19, 18)):
        raise ValueError('Expected matching native T64/valid19 decoder and measured 310P tail fixtures')
    entry, = [s for s in original['inputs'] if s['semantic_name'] == 'hidden']
    golden = original['golden']
    incoming = ref['inputs']['embedding' if layer == 0 else 'l03_entry']
    endpoint, = [b for b in ref['blocks'] if b['layer'] == layer]
    for earlier, current in ((entry, incoming), (golden, endpoint['hidden'])):
        checked(air, earlier)
        checked(reference, current)
        if any(earlier[k] != current[k] for k in ('shape', 'dtype', 'sha256')):
            raise ValueError(f'L{layer}: 310P {earlier["name"]} differs from the retained decoder fixture')
    residual = ref['tails'][str(layer)]
    if residual['dtype'] != 'float16' or residual['shape'] != list(HIDDEN) or residual['bytes'] != 64*2048*2:
        raise ValueError(f'L{layer}: unexpected residual descriptor')
    input_file = checked(reference, residual)
    msg, graph = graph_message(checked(air, original['air']).read_bytes(), 'NetOutput')
    edge, norm, target = tail_edges(graph)
    dest = run / 'air'
    dest.mkdir(parents=True, exist_ok=False)
    data_name = 'post_attention_residual'
    sliced = extract(msg, graph, air, dest, {edge: (data_name, 'float16', HIDDEN)}, target)
    ops = Counter(n['type'] for n in sliced['nodes'])
    if (norm not in {n['name'] for n in sliced['nodes']}
            or not any(op.lower().startswith('groupedmatmul') for op in ops)
            or ops['ChunkGatedDeltaRule'] or ops['AdnFusedInferAttention']):
        raise ValueError(f'L{layer}: unexpected decoder tail topology {ops}')
    inventory = {w['path']: w for w in original['weights']}
    weights = list(sliced['weight_evidence'].values())
    if not weights or any(w != inventory.get(w['path']) for w in weights):
        raise ValueError('Tail weights differ from retained complete decoder weights')
    shutil.copyfile(input_file, dest / residual['path'])
    shutil.copyfile(checked(air, golden), dest / golden['path'])
    save_json(dest / 'manifest.json', dict(kind='single-cpro-decoder-tail', layer=layer, attention=original['attention'],
        reference=original['reference'], measured_capture_sha256=sha(reference / 'manifest.json'),
        parent_air_sha256=original['air']['sha256'], input_edge=edge, post_norm=norm, output_edge=target,
        graph_ops=ops, air=sliced['air'], inputs=[dict(residual, name=data_name)], golden=golden,
        weights=weights, output_contract=original['output_contract']))
    print(f'L{layer}: {edge} -> {norm} -> MoE -> {target}; retained QuantBatchMatmulV4444 nodes: '
          f'{[n["name"] for n in sliced["nodes"] if n["type"] == "QuantBatchMatmulV4444"]}', flush=True)


def run_pair(parent, reference, root, ddk, source, diagnosis, tool):
    from ...execution.serial_transport import _environment
    _, device, server = _environment()
    selected = dict(device=device, server=server, gdr_source_sha256=whole_run.kernel_identity(ddk, source))
    rows = {}
    for layer in (0, 3):
        key = f'L{layer}'
        original, run = parent / key, root / key
        stage = 'prepare'
        try:
            compiled = load(original / 'omc/manifest.json')
            attempt = load(original / 'board/attempt.json')
            if (any(attempt[k] != v for k, v in selected.items())
                    or attempt['omc_sha256'] != compiled['primary']['sha256']
                    or compiled['air_manifest_sha256'] != sha(original / 'air/manifest.json')):
                raise ValueError(f'{key}: selected board or retained complete decoder build differs')
            prepare(original, reference, run, layer)
            stage = 'compile'
            whole_run.compile_model(run, ddk, source, diagnosis)
            stage = 'board'
            result = whole_run.board(run, tool)
            rows[key] = {k: result[k] for k in ('scope', 'status', 'valid19', 'token18')}
            for scope in ('valid19', 'token18'):
                m = result[scope]
                print(f"{key} tail {scope} rel_L2={m['relative_l2']:.4%} cos={m['cosine']:.9f} "
                      f"RMSE={m['rmse']:.6g} error_ratio={m['error_ratio']:.4%} "
                      f"elementwise_passed={m['error_ratio_passed']} five_metrics_passed={m['passed']}", flush=True)
        except Exception as exc:
            rows[key] = dict(status='FAILED', error=dict(stage=stage, message=str(exc)))
            print(f'{key} tail {stage} failed: {exc}', flush=True)
        save_json(root / 'comparison.json', rows)
    if any(row.get('error', {}).get('stage') == 'board' for row in rows.values()):
        from .chain_hilog import diagnose_decoder_pair
        diagnose_decoder_pair(root)
    print(root / 'comparison.json', flush=True)
    return all(row['status'] == 'PASSED' for row in rows.values())

"""Inspect the four-layer native CPro AIR boundary, including reordered state outputs."""
import json

from .decode_cache_control import graph_message
from .segment_export import OUTPUTS


def output_shape(name):
    if name == 'logits':
        return (1, 1, 248320)
    if name.startswith('conv_state_'):
        return (1, 8192, 4)
    if name.startswith('recurrent_state_'):
        return (1, 32, 128, 128)
    if name in ('key_cache_3', 'value_cache_3'):
        return (64, 32, 32, 16)
    raise ValueError(f'Unknown four-layer output: {name}')


def view(shape):
    shape = list(shape)
    while shape and shape[0] == 1:
        shape.pop(0)
    while shape and shape[-1] == 1:
        shape.pop()
    return tuple(shape)


def data_edges(op):
    return [edge for edge in op.input if edge and not edge.endswith(':-1')]


def inspect_air(path):
    _, graph = graph_message(path.read_bytes(), 'NetOutput')
    nodes = {n.name: n for n in graph.op}
    net, = [n for n in graph.op if n.type == 'NetOutput']
    outputs = data_edges(net)
    if len(outputs) != 9 or len(set(outputs)) != 9:
        raise ValueError(f'{path}: expected nine distinct native outputs, found {outputs}')
    conv = [n for n in graph.op if n.type.lower() == 'linearattentionconvstateupdate']
    gdr = [n for n in graph.op if n.type == 'ChunkGatedDeltaRule']
    full, = [n for n in graph.op if n.type == 'AdnFusedInferAttention']
    if len(conv) != 3 or len(gdr) != 3:
        raise ValueError(f'{path}: expected three native ConvState and GDR calls, got {len(conv)}, {len(gdr)}')

    def chain(edge):
        seen = set()
        while edge:
            if edge in seen:
                raise ValueError(f'{path}: cycle in state binding at {edge}')
            seen.add(edge)
            yield edge
            owner, _, port = edge.rpartition(':')
            node = nodes[owner]
            incoming = data_edges(node)
            if node.type in ('Assign', 'AssignVariableOp') and len(incoming) == 2:
                edge = incoming[1]
            elif incoming and port == '0' and node.type in (
                    'Cast', 'Identity', 'TensorMove', 'Reshape', 'View', 'Squeeze', 'BroadcastTo',
                    'ExpandDims', 'Unsqueeze', 'Transpose', 'Permute', 'TransData', 'Reformat', 'Flatten', 'MemCopyAsync'):
                edge = incoming[0]
            else:
                break

    aliases = {}
    for i, (c, g) in enumerate(zip(conv, gdr)):
        for op, input_port, output_port, role in (
                (c, 1, 0, f'conv_state_{i}'), (g, 5, 1, f'recurrent_state_{i}')):
            aliases[f'{op.name}:{output_port}'] = role
            # Some exporter revisions return the mutated cache input.
            for src in chain(data_edges(op)[input_port]):
                aliases[src] = role

    for key, edge in zip(('key_cache_3', 'value_cache_3'), data_edges(full)[1:3], strict=True):
        for src in chain(edge):
            aliases[src] = key
            op = nodes[src.rsplit(':', 1)[0]]
            if op.type == 'CacheUpdate':
                for cache_input in chain(data_edges(op)[0]):
                    aliases[cache_input] = key

    def descriptor(d):
        return dict(dtype=d.dtype, shape=list(d.shape.dim), layout=d.layout)

    graph_inputs = {edge.rsplit(':', 1)[0]: i for i, edge in enumerate(graph.input)}
    inputs = []
    for op in graph.op:
        if op.type in ('Data', 'RefData'):
            inputs.append(dict(name=op.name, kind=op.type, state=aliases.get(f'{op.name}:0'),
                               index=op.attr['index'].i if 'index' in op.attr else None,
                               graph_slot=graph_inputs.get(op.name),
                               **descriptor(op.output_desc[0])))
    records = []
    for i, edge in enumerate(outputs):
        name, port = edge.rsplit(':', 1)
        port = int(port)
        producer = nodes[name].output_desc
        desc = net.input_desc[i] if len(net.input_desc) == len(outputs) else producer[port]
        own = producer[port] if port < len(producer) else desc
        shape = list(desc.shape.dim or own.shape.dim)
        role = next((aliases[e] for e in chain(edge) if e in aliases), None)
        if role is None and view(shape) == (248320,):
            role = 'logits'
        records.append(dict(edge=edge, role=role, shape=shape, dtype=desc.dtype or own.dtype,
                            layout=desc.layout or own.layout))
    # In shape-stripped AIR, the only non-state edge is the native wrapper's logits.
    missing = set(OUTPUTS) - {r['role'] for r in records}
    unresolved = [r for r in records if r['role'] is None]
    if len(unresolved) == len(missing) == 1:
        unresolved[0]['role'] = missing.pop()
    roles = [r['role'] for r in records]
    if set(roles) != set(OUTPUTS):
        raise ValueError(f'{path}: cannot identify all native outputs by producer: {[(r["edge"], r["role"]) for r in records]}')
    for r in records:
        if r['shape'] and view(r['shape']) != view(output_shape(r['role'])):
            raise ValueError(f'{path}: unexpected shape for {r}')
    return dict(inputs=inputs, outputs=records)


def same_boundary(original, modified):
    # Names of internal result producers can change when SiLU is lowered.
    def signature(boundary):
        return (sorted(boundary['inputs'], key=lambda r: r['name']),
                [{k: r[k] for k in ('role', 'shape', 'dtype', 'layout')} for r in boundary['outputs']])
    if signature(original) != signature(modified):
        raise ValueError(f'Four-layer AIR boundary mismatch. Original: {original}; modified: {modified}')


def latest_replay(path):
    path = path.resolve(strict=True)
    if (path / 'result.json').is_file():
        if json.loads((path / 'result.json').read_bytes()).get('status') == 'EXECUTED':
            return path
        raise ValueError(f'Selected replay did not complete: {path}')
    for f in sorted(path.glob('replay-*/result.json'), key=lambda p: (p.stat().st_mtime_ns, p), reverse=True):
        if json.loads(f.read_bytes()).get('status') == 'EXECUTED':
            return f.parent
    raise ValueError(f'No successful four-layer replay in {path}')


def output_roles(compiled, source):
    if len(compiled) != len(source):
        raise ValueError('Compiled output count differs from source AIR')
    edges = {s['edge']: s['role'] for s in source}
    # The compiled NetOutput is the ordered public interface. Retained AIR edge
    # names give an independent binding when OMG preserves or reorders producers.
    roles = [edges.get(c['name']) for c in compiled]
    assigned = {r for r in roles if r}
    for i, spec in enumerate(compiled):
        if roles[i] is None:
            matches = [n for n in OUTPUTS if n not in assigned and view(output_shape(n)) == view(spec['shape'])]
            role = matches[0] if len(matches) == 1 else source[i]['role']
            if role not in matches:
                raise ValueError(f'Cannot bind compiled output {i} {spec} to the native AIR slots {matches}')
            roles[i] = role
            assigned.add(role)
    if set(roles) != set(OUTPUTS) or any(view(c['shape']) != view(output_shape(n))
                                          for n, c in zip(roles, compiled)):
        raise ValueError(f'Invalid compiled native output order or typed views: {roles}, {compiled}')
    return roles

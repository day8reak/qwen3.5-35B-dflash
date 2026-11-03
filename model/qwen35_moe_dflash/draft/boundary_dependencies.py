"""Trace explicit main-graph dependencies in an existing boundary inspection.

This is a metadata analysis, not a transfer trace or a latency attribution.
GraphOp internals remain opaque; a dependency ends at its output port.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import output_path
from .runtime_boundaries import KIND as BOUNDARY_KIND

KIND = 'qwen35-draft-boundary-dependencies'


def dependencies(source):
    if source.get('artifact_kind') != BOUNDARY_KIND or source.get('status') not in {'INSPECTED', 'PARTIAL'}:
        raise ValueError('need an existing runtime boundary inspection')
    graphs = source.get('main_graphs', [])
    if not 0 < len(graphs) <= 8:
        raise ValueError('need the full saved boundary JSON with main_graphs, not the printed summary')
    edges, unresolved = [], []
    budget = 65536
    graph_names = set()
    for graph in graphs:
        name = graph.get('name')
        if not isinstance(name, str) or not name or name in graph_names:
            raise ValueError('missing or duplicate main graph name')
        graph_names.add(name)
        rows = graph.get('nodes', [])
        nodes = {n['name']: n for n in rows}
        if not 0 < len(rows) <= 8192 or len(nodes) != len(rows):
            raise ValueError('missing or duplicate main graph nodes, or node budget exceeded')

        def endpoint(node, port, direction):
            return dict(name=node['name'], type=node['type'],
                        owner=node.get('task', {}).get('cl_name'), port=port,
                        descriptor={k: node[direction][port].get(k)
                                    for k in ('dtype', 'shape', 'layout', 'logical_bytes')})

        for consumer in rows:
            if consumer['type'] != 'GraphOp':
                continue
            for index, slot in enumerate(consumer['inputs']):
                target = endpoint(consumer, index, 'inputs')
                if consumer.get('port_reference_alignment') != 'POSITIONAL':
                    unresolved.append(dict(graph=name, consumer=target, reason='INPUT_REFERENCE_ALIGNMENT'))
                    continue
                stack = [(slot.get('reference'), [], frozenset())]
                while stack:
                    budget -= 1
                    if budget < 0:
                        raise ValueError('main graph dependency traversal exceeds budget')
                    ref, path, seen = stack.pop()
                    node_name, sep, output = ref.rpartition(':') if isinstance(ref, str) else ('', '', '')
                    producer = nodes.get(node_name)
                    reason = None
                    if (not sep or not output.isascii() or not output.isdecimal() or producer is None):
                        reason = 'UNRESOLVED_EXPLICIT_EDGE'
                    elif node_name in seen or len(path) >= 64:
                        reason = 'CYCLE_OR_DEPTH_LIMIT'
                    elif int(output) >= len(producer['outputs']):
                        reason = 'OUTPUT_PORT_NOT_DESCRIBED'
                    if reason:
                        unresolved.append(dict(graph=name, consumer=target, reference=ref,
                                               via=path, reason=reason))
                        continue
                    origin = endpoint(producer, int(output), 'outputs')
                    kind = producer['type']
                    if kind == 'GraphOp':
                        edges.append(dict(graph=name, producer=origin, consumer=target,
                                          via=list(reversed(path))))
                    elif kind in {'Data', 'Const', 'FileConstant'}:
                        continue
                    elif (producer.get('port_reference_alignment') != 'POSITIONAL'
                          or not producer['inputs'] or len(producer['outputs']) != 1):
                        # No inference of a multi-output node's input/output mapping.
                        unresolved.append(dict(graph=name, consumer=target, reference=ref,
                                               via=path, reason='MAIN_NODE_DEPENDENCY_UNRESOLVED'))
                    else:
                        # Follow every explicit data input, including Reshape shape
                        # inputs. Do not treat a Cast's device label as its owner.
                        for item in reversed(producer['inputs']):
                            stack.append((item.get('reference'), path + [origin], seen | {node_name}))
            # Control edges or descriptor/reference mismatches are retained as
            # unresolved above, never silently aligned with a tensor input.
    pairs = Counter((e['producer']['owner'], e['consumer']['owner']) for e in edges)
    bridges = []
    for graph in graphs:
        for node in graph['nodes']:
            if node['type'] != 'GraphOp' or node.get('task', {}).get('cl_name') != 'CPUCL':
                continue
            incoming = [i for i, e in enumerate(edges) if e['graph'] == graph['name']
                        and e['consumer']['name'] == node['name'] and e['producer']['owner'] == 'NPUCL']
            outgoing = [i for i, e in enumerate(edges) if e['graph'] == graph['name']
                        and e['producer']['name'] == node['name'] and e['consumer']['owner'] == 'NPUCL']
            if incoming and outgoing:
                bridges.append(dict(graph=graph['name'], cpu_partition=node['name'],
                                    incoming_edge_indices=incoming, outgoing_edge_indices=outgoing))
    return dict(artifact_kind=KIND, status='PARTIAL' if unresolved else 'ANALYZED',
        model=source.get('model'), source_numerical_validation=source.get('source_numerical_validation'),
        model_compilation='NOT_RUN', device_execution='NOT_RUN',
        scope='EXPLICIT_MAIN_GRAPH_PARTITION_DEPENDENCIES_NOT_RUNTIME_TIMING',
        edges=edges, owner_pairs=[dict(producer=a, consumer=b, paths=count)
                                 for (a, b), count in pairs.items()],
        intermediate_cpu_partitions=bridges, unresolved=unresolved,
        limitations=[
            'Model identity is inherited from the saved inspection; this analysis does not reopen the OMC.',
            'Paths follow explicit input ports only. src_name/src_index and device_type do not establish ownership.',
            'GraphOp internals are opaque. A CPU partition with NPU inputs and consumers is a scheduling candidate, not proof that every output depends on every input.',
            'Descriptor bytes are logical extents, not measured copies. Shared producers and multiple paths may recur; do not sum them as transfers.',
            'No timestamps, waits, transfers, allocations or CPU/NPU clock alignment are measured. This does not attribute the model/operator duration difference.',
        ])


def analyze_boundaries(source, output):
    path, destination = Path(source).resolve(), output_path(output)
    if path == destination or destination.is_symlink():
        raise ValueError('analysis must not overwrite the boundary inspection')
    with path.open('rb') as stream:
        raw = stream.read((8 << 20) + 1)
    if len(raw) > 8 << 20:
        raise ValueError('boundary inspection exceeds 8 MiB')
    result = dependencies(json.loads(raw))
    result['source'] = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    result['report'] = str(destination)
    if destination.exists():
        if json.loads(destination.read_text()).get('artifact_kind') != KIND:
            raise ValueError('refusing to replace unrelated analysis output')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    return {k: v for k, v in result.items() if k != 'edges'} | {
        'npu_to_cpu': [dict(edge_index=i, **e) for i, e in enumerate(result['edges'])
                      if e['producer']['owner'] == 'NPUCL' and e['consumer']['owner'] == 'CPUCL'],
        'intermediate_cpu_outputs': [dict(edge_index=i, **e) for i, e in enumerate(result['edges'])
            if any(i in b['outgoing_edge_indices'] for b in result['intermediate_cpu_partitions'])]}

"""Describe the retained GDN state path without changing or accepting its ABI."""
import json

from qwen35_moe_dflash.target.operator_air_review import _data_edges, _dtype, _edge


# Follow the data operand only. Arithmetic, writeback and opaque GraphOps stop
# the walk; their actual input edges/descriptors remain available in the report.
_WRAPPERS = {'Cast', 'CastT', 'Identity', 'Reshape', 'Squeeze', 'Unsqueeze',
             'ExpandDims', 'Transpose', 'TensorMove'}


def _port(desc):
    return dict(dtype=_dtype(desc), shape=desc.get('shape'),
                raw_dtype=desc.get('raw_dtype'), schema_dtype_names=desc.get('schema_dtype_names'))


def _node(node, nodes):
    inputs = []
    for slot, edge in enumerate(_data_edges(node)):
        parsed = _edge(edge)
        producer = nodes.get(parsed[0]) if parsed else None
        descs = producer.get('output_desc', []) if producer else []
        desc = node['input_desc'][slot] if slot < len(node['input_desc']) else {}
        inputs.append(dict(slot=slot, edge=edge, **_port(desc),
            producer_type=producer['type'] if producer else None,
            producer_output=_port(descs[parsed[1]]) if parsed and parsed[1] < len(descs) else None))
    return dict(name=node['name'], type=node['type'], inputs=inputs,
                outputs=[_port(d) for d in node['output_desc']], attributes=node.get('attributes', {}))


def _trace(edge, nodes):
    chain, seen = [], set()
    for _ in range(32):
        parsed = _edge(edge)
        if parsed is None or parsed[0] not in nodes:
            return dict(chain=chain, stop='UNRESOLVED_EDGE')
        if parsed in seen:
            return dict(chain=chain, stop='CYCLE')
        seen.add(parsed)
        node, slot = nodes[parsed[0]], parsed[1]
        chain.append(dict(_node(node, nodes), selected_output_slot=slot))
        if slot >= len(node['output_desc']):
            return dict(chain=chain, stop='MISSING_OUTPUT_DESCRIPTOR')
        if slot != 0 or node['type'] not in _WRAPPERS:
            return dict(chain=chain, stop='OPAQUE_GRAPHOP' if node['type'] == 'GraphOp' else 'PRODUCER_REACHED')
        edges = _data_edges(node)
        edge = edges[0] if edges else ''
    return dict(chain=chain, stop='DEPTH_LIMIT')


def state_context(metadata):
    """Slot 2 is the requested whole-GDN tuple slot, never guessed by size."""
    graphs = []
    for graph in metadata.get('graphs', []):
        nodes = {n['name']: n for n in graph['nodes']}
        row = dict(graph=graph.get('name'), outputs=[], gdn_nodes=[
            _node(n, nodes) for n in nodes.values() if n['type'] == 'ChunkGatedDeltaRule'])
        nets = [n for n in nodes.values() if n['type'] == 'NetOutput']
        row['netoutput_count'] = len(nets)
        if len(nets) == 1:
            net, = nets
            edges = _data_edges(net)
            row['outputs'] = [dict(slot=i, edge=edges[i] if i < len(edges) else None, **_port(d))
                              for i, d in enumerate(net['input_desc'])]
            if len(edges) > 2 and len(net['input_desc']) > 2:
                row['requested_state_slot'] = 2
                row['state_trace'] = _trace(edges[2], nodes)
        # Retained IR subgraphs may not have the whole-layer tuple. Do not infer
        # that their output 2 is state, or that a missing GDN means it was removed.
        graphs.append(row)
    return dict(status='INSPECTED' if graphs else 'UNAVAILABLE',
                metadata_status=metadata.get('status'), error=metadata.get('error'),
                scope='REQUESTED_NETOUTPUT_SLOT_2_AND_VISIBLE_GDN; SUBGRAPH_SLOT_SEMANTICS_UNVERIFIED',
                internal_accumulation='NOT_OBSERVED', graphs=graphs)


def state_summary(report):
    lines = ['Read-only GDN state inspection; no export, compilation, device execution or numerical validation.',
             'Output slots are zero-based. Slot 2 is the requested whole-GDN recurrent state.',
             'Source/IR/OMC descriptors are reported as observed; opaque kernel arithmetic remains unknown.',
             'IR subgraph slot 2 need not be whole-layer state. No dtype labels or ABI gates were changed.', '']
    for name, case in report['cases'].items():
        lines.append(f'[{name}] compilation={case["compilation"].get("status")}')
        contract = case.get('export_evidence', {}).get('output_contract')
        lines.append('source_output_contract=' + json.dumps(contract, ensure_ascii=False))
        if case.get('source_error'):
            lines.append('source_error=' + case['source_error'])
        lines.append('retained_ir=' + json.dumps(case['ir_inventory'], ensure_ascii=False))
        if case.get('uninspected_ir_dumps'):
            lines.append('uninspected_ir=' + json.dumps(case['uninspected_ir_dumps']))
        for item in case['state_chain']:
            lines.append(f'{item["stage"]}: {item["status"]}; metadata={item["metadata_status"]}')
            if item['error']:
                lines.append(item['error'])
            for graph in item['graphs']:
                lines.append(json.dumps(graph, ensure_ascii=False))
        lines.append('')
    return '\n'.join(lines) + '\n'

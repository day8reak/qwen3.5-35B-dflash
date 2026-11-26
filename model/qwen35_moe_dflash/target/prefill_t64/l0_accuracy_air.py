"""Cumulative L0 readouts from the retained native four-layer T64 AIR."""
from .attention_steps import AttentionSteps, CONVS, MATMUL, NORMS, only, operands
from .decode_cache_control import graph_message
from .decoder_four_silu import rms_gate_path
from .gdn_layer_air import four_layer_gdn
from .l0_split import slice_dependencies, wire


def readouts(raw, meta):
    block = four_layer_gdn(raw, meta)
    _, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    nodes = {n.name: n for n in graph.op}
    walk = AttentionSteps(graph, nodes, [])
    added = {'Add', 'AddV2'}
    l0 = set(block['nodes'])

    def node_in(region, types):
        return only((n + ':0' for n in region if nodes[n].type in types), '/'.join(sorted(types)))

    def consumer(edge, types):
        return walk.node(only((n.name + ':0' for n in graph.op if n.type in types
                               and any(walk.strip(e) == walk.strip(edge) for e in operands(n))),
                              f'{types} consuming {edge}'))

    def peer(op, edge):
        return only((e for e in operands(op) if walk.strip(e) != walk.strip(edge)), f'{op.name} other operand')

    hidden = block['inputs'][0]
    out, history, cache = block['outputs']
    gdr = walk.node(node_in(l0, {'ChunkGatedDeltaRule'}))
    def port(op, name):
        slots = dict(zip((v.decode() for v in op.attr['_input_name_key'].list.s),
                         op.attr['_input_name_value'].list.i, strict=True))
        return op.input[slots[name]]
    gdr_inputs = {k: port(gdr, k) for k in ('query', 'key', 'value', 'g', 'beta')}
    conv = walk.node(node_in(l0, CONVS))
    sites = {tuple(s['shape']): nodes[s['name']] for s in meta['silu_patch']['nodes'] if s['name'] in l0}
    conv_silu = sites[(1, 8192, 64)]
    gate = sites[(64, 4096)] if (64, 4096) in sites else sites[(2048, 128)]
    gate_path, rms = rms_gate_path(graph, gate.name + ':0')
    soft = walk.node(node_in(l0, {'SoftplusPatch'}))
    projections = dict(qkv=walk.nearest(history, {MATMUL}), z=walk.nearest(gate.input[0], {MATMUL}),
                       a=walk.nearest(soft.input[0], {MATMUL}), b=walk.nearest(gdr_inputs['beta'], {MATMUL}),
                       out=walk.strip(out))
    residual = consumer(out, added)
    residual_edge = residual.name + ':0'
    entry = peer(residual, out)
    final = consumer(residual_edge, added)
    moe = peer(final, residual_edge)
    post_norm = consumer(residual_edge, NORMS)
    prefix_inputs = {n+':0': role for n, role in meta['inputs'].items()}
    edges = dict(block_entry=entry, hidden=hidden, **projections, window=conv.input[0], conv=conv_silu.input[0],
                 conv_silu=conv_silu.name+':0', new_history=history, **gdr_inputs,
                 biased_a=walk.nearest(soft.input[0], added), softplus=soft.name+':0',
                 core=gdr.name+':0', final_state=gdr.name+':1', recurrent_cache=cache,
                 gate_silu=gate.name+':0', product=gate_path[-1].name+':0', rms=rms,
                 residual=residual_edge, post_norm=post_norm.name+':0', moe_out=moe, block_out=final.name+':0')
    def quant_edges(key, edge):
        dq = walk.expect(walk.strip(walk.node(edge).input[0]), {'DynamicQuant'})
        edges[key+'_a8'], edges[key+'_scale'] = dq.name+':0', dq.name+':1'
        if key == 'out':
            edges['gated'] = dq.input[0]
    for key, edge in projections.items():
        quant_edges(key, edge)
    groups = {
        'entry-projections': ['block_entry', 'hidden', *(p+s for p in ('qkv','z','a','b') for s in ('_a8','_scale',''))],
        'convolution': ['window','new_history','conv','conv_silu','query','key','value'],
        'gdr': ['biased_a','softplus','g','beta','core','final_state','recurrent_cache'],
        'attention-output': ['rms','gate_silu','product','gated','out_a8','out_scale','out','residual','post_norm'],
    }

    def cut(label, roles):
        # One physical edge can implement several source variables (e.g. a shared DQ or FP16 cache).
        aliases = {}
        for name in roles:
            aliases.setdefault(edges[name], []).append(name)
        kept, used = slice_dependencies(nodes, list(aliases), set(prefix_inputs), label)
        feeds = [edge for edge in prefix_inputs if edge in used]
        wires = {}
        for edge in feeds:
            role = prefix_inputs[edge]
            spec = meta['input_files'][role]
            wires[edge] = wire(role, spec['shape'], spec['dtype'])
        for edge, names in aliases.items():
            wires.setdefault(edge, dict(name=names[0], shape=None, dtype=None))
            wires[edge].update(aliases=names, edge=edge)
        return label, dict(inputs=feeds, outputs=list(aliases), nodes=sorted(kept)), wires

    def sections(table):
        for label, roles in table.items():
            try:
                yield cut(label, roles)
            except (ValueError, KeyError) as exc:
                yield label, dict(error=str(exc), inputs=[], outputs=[]), {}

    yield from sections(groups)
    expert_groups = {
        'shared-router': [*(p+s for p in ('shared_gate_proj','shared_up') for s in ('_a8','_scale','')),
                          'shared_silu','shared_product','shared_down_a8','shared_down_scale','shared_down',
                          'shared_gate_logit','shared_weighted','router_logits_a8','router_logits_scale','router_logits','expert_ids'],
        'experts-output': ['expert_ids','route_map','route_weights','expert_gate_up','expert_product','expert_down','routed','moe_out','block_out'],
    }
    try:
        # Limit the expert search to the first MoE, excluding its attention and every later layer.
        expert_nodes, _ = slice_dependencies(nodes, [moe], {residual_edge, *prefix_inputs}, 'L0 MoE')
        shared_silu = only((s['name'] + ':0' for s in meta['silu_patch']['nodes']
                            if s['name'] in expert_nodes and s['shape'] == [64, 512]), 'shared expert SiLU')
        shared_mul = consumer(shared_silu, {'Mul'})
        shared_product = shared_mul.name + ':0'
        shared_up = peer(shared_mul, shared_silu)
        shared_gate = walk.nearest(walk.node(shared_silu).input[0], {MATMUL})
        shared_down = only((n + ':0' for n in expert_nodes if nodes[n].type == MATMUL
                            and walk.depends(nodes[n].input[0], shared_product)), 'shared expert down_proj')
        shared_weight = consumer(shared_down, {'Mul'})
        sigmoid = walk.expect(walk.strip(peer(shared_weight, shared_down)), {'Sigmoid'})
        finalize = walk.node(node_in(expert_nodes, {'MoeFinalizeRouting', 'MoeFinalizeRoutingV2'}))
        routing = walk.node(node_in(expert_nodes, {'MoeGatingTopKSoftmax', 'MoeGatingTopKSoftmaxV2'}))
        init = walk.node(node_in(expert_nodes, {'MoeInitRoutingV2', 'MoeInitRoutingV3'}))
        grouped = [n+':0' for n in expert_nodes if 'groupedmatmul' in nodes[n].type.lower()]
        if len(grouped) != 2:
            raise ValueError(f'Expected two expert projections, got {grouped}')
        gmm2 = walk.node(only((e for e in grouped if walk.depends(finalize.input[0], e)
                              and any(walk.depends(e, other) for other in grouped if other != e)), 'expert down GMM'))
        gmm1 = walk.node(only((e for e in grouped if walk.node(e).name != gmm2.name), 'expert gate/up GMM'))
        expert_silu = only((s['name'] + ':0' for s in meta['silu_patch']['nodes']
                            if s['name'] in expert_nodes and s['name'] + ':0' != shared_silu), 'routed expert SiLU')
        routed_product = consumer(expert_silu, {'Mul'})
        weights = only((e for e in operands(finalize) if walk.depends(e, routing.name+':0')), 'routing scales')
        edges.update(shared_gate_proj=shared_gate,
                     shared_up=walk.nearest(shared_up, {MATMUL}), shared_silu=shared_silu,
                     shared_product=shared_product, shared_down=shared_down, shared_gate_logit=sigmoid.input[0],
                     shared_weighted=shared_weight.name+':0', router_logits=routing.input[0],
                     expert_ids=init.input[1], route_map=init.name+':1', route_weights=weights,
                     expert_gate_up=gmm1.name+':0', expert_product=routed_product.name+':0', expert_down=gmm2.name+':0',
                     routed=finalize.name+':0')
        for role in ('shared_gate_proj','shared_up','shared_down','router_logits'):
            quant_edges(role, walk.nearest(edges[role], {MATMUL}))
    except (ValueError, KeyError, IndexError) as exc:
        for label in expert_groups:
            yield label, dict(error=f'L0 MoE boundary: {exc}', inputs=[], outputs=[]), {}
        yield from sections({'block-output': ['moe_out', 'block_out']})
        return
    yield from sections(expert_groups)

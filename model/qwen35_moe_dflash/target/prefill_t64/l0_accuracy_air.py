"""Cumulative L0 readouts from the retained native four-layer T64 AIR."""
from .attention_steps import AttentionSteps, CONVS, MATMUL, NORMS, only, operands
from .decode_cache_control import graph_message
from .decoder_four_silu import rms_gate_path
from .gdn_layer_air import before_padding_mask, four_layer_gdn
from .l0_split import slice_dependencies, wire


GDN_GROUPS = {
    'entry-projections': ['block_entry', 'hidden', *(p+s for p in ('qkv','z','a','b') for s in ('_a8','_scale',''))],
    'convolution': ['window','new_history','conv','conv_silu','query','key','value'],
    'gdr': ['biased_a','softplus','g','beta','core','final_state','recurrent_cache'],
    'attention-output': ['rms','gate_silu','product','gated','out_a8','out_scale','out','residual','post_norm'],
}

EXPERT_GROUPS = {
    'shared-router': [*(p+s for p in ('shared_gate_proj','shared_up') for s in ('_a8','_scale','')),
                      'shared_silu','shared_product','shared_down_a8','shared_down_scale','shared_down',
                      'shared_gate_logit','shared_weighted','router_logits_a8','router_logits_scale','router_logits','expert_ids'],
    'experts-output': ['expert_ids','route_map','route_weights','expert_gate_up','expert_product','expert_down','routed','moe_out','block_out'],
}


def readouts(raw, meta, *, same_input=False, dump_edges=None):
    _, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    nodes = {n.name: n for n in graph.op}
    walk = AttentionSteps(graph, nodes, [])
    added = {'Add', 'AddV2'}
    edges = {} if dump_edges is None else dump_edges
    def quant_edges(key, edge):
        dq = walk.expect(walk.strip(walk.node(edge).input[0]), {'DynamicQuant'})
        edges[key+'_a8'], edges[key+'_scale'] = dq.name+':0', dq.name+':1'
        if key == 'out':
            edges['gated'] = dq.input[0]
    prefix_inputs = {n+':0': role for n, role in meta['inputs'].items()}
    if dump_edges is None:
        block = four_layer_gdn(raw, meta)
        l0 = set(block['nodes'])
        hidden = block['inputs'][0]
        out, history, cache = block['outputs']
    else:
        # Complete single-layer export exposes attention output and both states.
        # Only inspect its AIR; the offline comparison never cuts or compiles a graph.
        returned = {row['name']: row['edge'] for row in meta['outputs']}
        out, history, cache = (returned[k] for k in ('out_proj_0', 'conv_state_0', 'recurrent_state_0'))
        entry = only((e for e, name in prefix_inputs.items() if name == 'hidden'), 'L0 input')
        hidden = only((op.name+':0' for op in graph.op if op.type in NORMS
                       and walk.strip(op.input[0]) == entry), 'L0 input norm')
        dump_edges.update(block_entry=entry, hidden=hidden, out=out, new_history=history, recurrent_cache=cache,
                          block_out=returned['hidden_0'], logits=returned['logits'])
        l0, _ = slice_dependencies(nodes, [out, history, cache], set(prefix_inputs), 'single L0 attention')

    def node_in(region, types):
        return only((n + ':0' for n in region if nodes[n].type in types), '/'.join(sorted(types)))

    def consumer(edge, types):
        return walk.node(only((n.name + ':0' for n in graph.op if n.type in types
                               and any(walk.strip(e) == walk.strip(edge) for e in operands(n))),
                              f'{types} consuming {edge}'))

    def peer(op, edge):
        return only((e for e in operands(op) if walk.strip(e) != walk.strip(edge)), f'{op.name} other operand')

    gdr = walk.node(node_in(l0, {'ChunkGatedDeltaRule'}))
    def port(op, name):
        slots = dict(zip((v.decode() for v in op.attr['_input_name_key'].list.s),
                         op.attr['_input_name_value'].list.i, strict=True))
        return op.input[slots[name]]
    gdr_inputs = {k: port(gdr, k) for k in ('query', 'key', 'value', 'g', 'beta')}
    conv = walk.node(node_in(l0, CONVS))
    edges.update(gdr_inputs, core=gdr.name+':0', final_state=gdr.name+':1', window=conv.input[0])
    soft = walk.node(node_in(l0, {'SoftplusPatch'}))
    edges.update(softplus=soft.name+':0', biased_a=walk.nearest(soft.input[0], added))
    if dump_edges is None:
        sites = {tuple(s['shape']): nodes[s['name']] for s in meta['silu_patch']['nodes'] if s['name'] in l0}
        conv_silu = sites[(1, 8192, 64)]
        gate = sites[(64, 4096)] if (64, 4096) in sites else sites[(2048, 128)]
    else:
        conv_silu = walk.node(walk.nearest(gdr_inputs['query'], {'SiluPatch'}))
        gate = walk.node(only((n+':0' for n in l0 if nodes[n].type == 'SiluPatch' and n != conv_silu.name),
                              'L0 RMS gate activation'))
    projections = dict(qkv=walk.nearest(history, {MATMUL}), z=walk.nearest(gate.input[0], {MATMUL}),
                       a=walk.nearest(soft.input[0], {MATMUL}), b=walk.nearest(gdr_inputs['beta'], {MATMUL}),
                       out=before_padding_mask(walk, out))
    edges.update(projections, conv_silu=conv_silu.name+':0', conv=conv_silu.input[0], gate_silu=gate.name+':0')
    for key, edge in projections.items():
        quant_edges(key, edge)
    if projections['out'] != walk.strip(out):
        edges['out'] = out
    gate_path, rms = rms_gate_path(graph, gate.name + ':0')
    edges.update(product=gate_path[-1].name+':0', rms=rms)
    residual = consumer(out, added)
    residual_edge = residual.name + ':0'
    entry = peer(residual, out)
    final = consumer(residual_edge, added)
    moe = peer(final, residual_edge)
    post_norm = consumer(residual_edge, NORMS)
    input_specs = meta.get('input_files', {}).copy()
    if same_input:
        # Intercept the common embedding product before the residual and norm paths fork.
        original = {e for e, role in prefix_inputs.items() if role in ('inputs_embeds', 'embed_scale')}
        product = walk.strip(entry)
        multiply = walk.expect(product, {'Mul'})
        if (len(original) != 2 or {walk.strip(e) for e in operands(multiply)} != original
                or not walk.depends(hidden, product)):
            raise ValueError(f'{product}: expected the embedding × scale feeding L0 RMSNorm and residual')
        shape = input_specs['inputs_embeds']['shape']
        dtype = {1: 'float32', 2: 'float16'}.get(multiply.output_desc[0].dtype)
        if list(shape) != [1, 64, 2048] or not dtype:
            raise ValueError(f'{product}: unsupported L0 embedding shape or dtype')
        # Removing both public embedding inputs also rejects cuts that still reach either port.
        prefix_inputs = {e: role for e, role in prefix_inputs.items() if e not in original}
        prefix_inputs[product] = 'l0_entry'
        input_specs['l0_entry'] = dict(shape=shape, dtype=dtype)
    edges.update(block_entry=entry, hidden=hidden, new_history=history, recurrent_cache=cache,
                 residual=residual_edge, post_norm=post_norm.name+':0', moe_out=moe, block_out=final.name+':0')

    def cut(label, roles):
        if dump_edges is not None:
            return label, {}, {}
        # One physical edge can implement several source variables (e.g. a shared DQ or FP16 cache).
        aliases = {}
        for name in roles:
            aliases.setdefault(edges[name], []).append(name)
        kept, used = slice_dependencies(nodes, list(aliases), set(prefix_inputs), label)
        feeds = [edge for edge in prefix_inputs if edge in used]
        wires = {}
        for edge in feeds:
            role = prefix_inputs[edge]
            spec = input_specs[role]
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

    yield from sections(GDN_GROUPS)
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
        edges.update(shared_gate_proj=shared_gate, shared_up=walk.nearest(shared_up, {MATMUL}),
                     shared_down=shared_down, shared_silu=shared_silu, shared_product=shared_product,
                     shared_gate_logit=sigmoid.input[0], shared_weighted=shared_weight.name+':0')
        for role in ('shared_gate_proj','shared_up','shared_down'):
            quant_edges(role, edges[role])
        finalize = walk.node(node_in(expert_nodes, {'MoeFinalizeRouting', 'MoeFinalizeRoutingV2'}))
        routing = walk.node(node_in(expert_nodes, {'MoeGatingTopKSoftmax', 'MoeGatingTopKSoftmaxV2'}))
        init = walk.node(node_in(expert_nodes, {'MoeInitRoutingV2', 'MoeInitRoutingV3'}))
        edges.update(router_logits=routing.input[0], expert_ids=init.input[1], route_map=init.name+':1')
        quant_edges('router_logits', walk.nearest(edges['router_logits'], {MATMUL}))
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
        edges.update(route_weights=weights,
                     expert_gate_up=gmm1.name+':0', expert_product=routed_product.name+':0', expert_down=gmm2.name+':0',
                     routed=finalize.name+':0')
    except (ValueError, KeyError, IndexError) as exc:
        for label in EXPERT_GROUPS:
            yield label, dict(error=f'L0 MoE boundary: {exc}', inputs=[], outputs=[]), {}
        yield from sections({'block-output': ['moe_out', 'block_out']})
        return
    yield from sections(EXPERT_GROUPS)

"""Bind independent diagnostic OMCs to the corrected L0 decoder's original AIR edges."""
from .attention_slice import descriptor, port_info, DTYPES
from .attention_steps import AttentionSteps, CONVS, MATMUL, only
from .decode_cache_control import graph_message
from .decode_cache_trace import _project_input
from .gdn_layer_stages import SHAPES, STAGES, check_view
from .gdr_stage_air import stage_edges
from .l0_split import plan, slice_dependencies, wire


def stages(raw, meta):
    trace = meta['silu_patch']['nodes']
    decoder = plan(raw, trace, layer=0, head=True, verify_precision=True)
    _, graph = graph_message(raw, 'SiluPatch')
    nodes = {op.name: op for op in graph.op}
    walk = AttentionSteps(graph, nodes, [])
    sites = {tuple(s['shape']): nodes[s['name']] for s in trace}
    conv_patch, gate_patch = sites[(1,8192,64)], sites[(2048,128)]
    gate_edge = gate_patch.name + ':0'
    mul = only((op.name for op in graph.op if op.type == 'Mul' and gate_edge in op.input), 'gated Mul')
    rms = only((e for e in nodes[mul].input if e and e != gate_edge), 'normalized core')
    gdr = stage_edges(graph, 'gdr')
    conv = walk.node(walk.nearest(conv_patch.input[0], CONVS))
    projections = dict(qkv=walk.nearest(conv.input[0], {MATMUL}), z=walk.nearest(gate_patch.input[0], {MATMUL}),
                       a=walk.nearest(gdr['g'], {MATMUL}), b=walk.nearest(gdr['beta'], {MATMUL}),
                       out=walk.strip(decoder['results']['out_proj']))
    if len(set(projections.values())) != 5 or any(walk.node(e).type != MATMUL for e in projections.values()):
        raise ValueError(f'expected five distinct L0 W4 projections: {projections}')
    quant = {p: walk.node(walk.strip(walk.node(e).input[0])) for p,e in projections.items()}
    # Decay is a product of a weight-only factor and a data-dependent softplus term.
    g_product = walk.node(walk.strip(gdr['g']))
    if g_product.type != 'Mul':
        raise ValueError(f'expected decay multiplication at {g_product.name} ({g_product.type})')
    soft = only((e for e in g_product.input if e and walk.depends(e, projections['a'])), 'decay step size')
    bias = only((op.name+':0' for op in graph.op if op.type in ('Add','AddV2') and
                 any(e and walk.strip(e) == projections['a'] for e in op.input) and
                 walk.depends(soft, op.name+':0')), 'a + dt_bias')
    edges = dict(**projections, conv=conv_patch.input[0], conv_silu=conv_patch.name+':0', window=conv.input[0],
                 new_history=decoder['results']['conv_state_0'], biased_a=bias, softplus=soft, g=g_product.name+':0',
                 core=gdr['core'], final_state=gdr['final_state'], rms=rms, gate_silu=gate_edge, product=mul+':0',
                 gated=_project_input(nodes, projections['out']), state=gdr['initial_state'], length=gdr['effective_length'])
    public_state = decoder['results']['recurrent_state_0']
    while public_state != gdr['final_state']:
        alias = walk.node(public_state)
        if alias.type not in {'Cast', 'Identity', 'TensorMove', 'Reshape', 'View'}:
            raise ValueError(f'Unexpected recurrent cache producer {alias.name}')
        if alias.type == 'Cast' and port_info(graph, alias.output_desc[0])['dtype'] != 'DT_FLOAT16':
            raise ValueError(f'Unexpected recurrent cache cast at {alias.name}')
        public_state = alias.input[0]
    for k in ('query', 'key', 'value', 'beta'):
        edges[k] = gdr[k]
    edges['history'] = only((e for e,v in decoder['entries'].items() if v['shape'] == [1,8192,4]), 'conv history')
    for p, op in quant.items():
        if op.type != 'DynamicQuant':
            raise ValueError(f'{p}: expected DynamicQuant, found {op.type}')
        edges.update({p+'_a8': op.name+':0', p+'_scale': op.name+':1'})

    shapes = dict(SHAPES, gated=(64,4096), rms=(2048,128), product=(2048,128), gate_silu=(2048,128))
    types = {v:k for k,v in DTYPES.items()}
    specs = []
    for name, (inputs, outputs) in STAGES.items():
        selected = dict(edges)
        views = dict(shapes)
        if name.endswith('_quant'):
            p = outputs[0].removesuffix('_a8')
            selected[inputs[0]] = quant[p].input[0]
            views[inputs[0]] = SHAPES[p+'_a8']
        if name == 'gate_silu':
            # The source z branch includes a view and may round-trip FP16 through FP32.
            # Feeding this singleton with z directly preserves its bytes in both cases.
            edge = gate_patch.input[0]
            while edge != projections['z']:
                node = walk.node(edge)
                if node.type not in {'Reshape', 'View', 'Identity', 'TensorMove', 'Squeeze', 'Unsqueeze',
                                     'ExpandDims', 'Cast'}:
                    raise ValueError(f'Unaccounted gate operation {node.name} ({node.type})')
                if node.type == 'Cast' and port_info(graph, node.output_desc[0])['dtype'] not in ('DT_FLOAT','DT_FLOAT16'):
                    raise ValueError(f'Unexpected gate cast at {node.name}')
                edge = node.input[0]
            selected['z'] = gate_patch.input[0]
            views['z'] = (2048,128)  # z.view(-1, head_v_dim), same element order
        src = [selected[k] for k in inputs]
        target = [selected[k] for k in outputs]
        retained, used = slice_dependencies(nodes, target, set(src), name)
        if used != set(src):
            raise ValueError(f'{name}: unused or missing inputs: {set(src) ^ used}')
        patches = [n for n in retained if nodes[n].type == 'SiluPatch']
        if bool(patches) != name.endswith('_silu') or (patches and len(retained) != 1):
            raise ValueError(f'{name}: unexpected SiluPatch slice {patches}')
        wires = {}
        for key in (*inputs, *outputs):
            edge = selected[key]
            desc = port_info(graph, descriptor(nodes, edge))
            if name.endswith('_quant') and key == inputs[0] and quant[p].input_desc:
                desc = port_info(graph, quant[p].input_desc[0])
            shape = desc['shape'] if desc['shape'] and min(desc['shape']) > 0 else views[key]
            check_view(key, shape)
            # TorchAir frequently writes placeholder FLOAT descriptors for view outputs.
            # Use the measured, source-level contract on known typed values. For decay,
            # take the FP32 arithmetic result before the custom GDR FP16 input cast.
            dtype = ('int8' if key.endswith('_a8') else 'int16' if key == 'length' else
                     'float32' if key.endswith('_scale') or key in
                     ('conv_silu','gate_silu','rms','product','biased_a','softplus','g') else 'float16')
            if name.endswith('_quant') and key == inputs[0] and quant[p].input_desc:
                dtype = types[port_info(graph, quant[p].input_desc[0])['dtype']]
            if ((key in ('biased_a', 'softplus', 'g') or (name.endswith('_quant') and key == inputs[0]))
                    and desc['dtype'] in ('DT_FLOAT16', 'DT_FLOAT')):
                dtype = types[desc['dtype']]
            wires[edge] = wire(key,shape,dtype)
        specs.append(dict(name=name, inputs=src, outputs=target, nodes=sorted(retained), wires=wires,
                          weight_storage='embedded'))
    return specs

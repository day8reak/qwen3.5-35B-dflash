"""Bind independent diagnostic OMCs to the corrected L0 decoder's original AIR edges."""
from .attention_slice import descriptor, port_info, DTYPES
from .attention_steps import AttentionSteps, CONVS, MATMUL, only
from .decode_cache_control import graph_message
from .decode_cache_trace import _project_input
from .decoder_four_silu import gate_product
from .gdn_layer_stages import ENTRY, SHAPES, STAGES, WHOLE_OUTPUTS, check_view
from .gdr_stage_air import stage_edges
from .l0_split import plan, slice_dependencies, wire


def recurrent_writeback(graph, nodes, state, cache):
    """Prove that the exposed GDR state and the returned FP16 cache have the same values."""
    initial = port_info(graph, descriptor(nodes, state))['dtype']
    if initial != 'DT_FLOAT16':
        raise ValueError(f'{state}: expected FP16 GDR state, got {initial}')
    trace = []
    seen = set()
    edge = cache
    while edge != state:
        name, _, slot = edge.rpartition(':')
        op = nodes.get(name)
        if edge in seen or slot != '0' or op is None or not op.input or not op.input[0]:
            raise ValueError(f'{cache}: invalid recurrent writeback edge {edge}, expected a path to {state}')
        seen.add(edge)
        target = None
        if op.type == 'Cast':
            attr = op.attr.get('dst_type')
            code = attr.i if attr is not None and attr.HasField('i') else None
            target = {0: 'DT_FLOAT', 1: 'DT_FLOAT16'}.get(code)
            if target is None:
                raise ValueError(f'{edge}: unsupported recurrent cache Cast dst_type={code} on path to {cache}')
        elif op.type not in {'Identity', 'TensorMove', 'Reshape', 'View'}:
            raise ValueError(f'{edge}: unexpected {op.type} on recurrent cache path from {state} to {cache}')
        trace.append((edge, target))
        edge = op.input[0]
    # The export converter widens FP16 to match the stock FP32 fake meta. The
    # source then narrows to FP16 at writeback. Check the actual GE Cast attributes.
    dtype = initial
    path = [f'{state} ({dtype})']
    for edge, target in reversed(trace):
        dtype = target or dtype
        path.append(f'{edge} ({dtype})')
    if dtype != 'DT_FLOAT16':
        raise ValueError(f'Expected FP16 at returned recurrent cache: {" -> ".join(path)}')
    return ' -> '.join(path)


def whole_stage(graph, decoder, quant, conv_patch, gate_patch, gdr):
    nodes = {op.name: op for op in graph.op}

    def hidden_input(branch):
        edge = quant[branch].input[0]
        op = nodes[edge.rsplit(':', 1)[0]]
        while op.type in ('Reshape', 'View', 'Identity', 'TensorMove'):
            edge = op.input[0]
            op = nodes[edge.rsplit(':', 1)[0]]
        return edge

    history = only((e for e, spec in decoder['entries'].items() if spec['shape'] == [1,8192,4]), 'conv history')
    inputs = [only((hidden_input(p) for p in ('qkv', 'z', 'a', 'b')), 'shared GDN layer input'),
              history, gdr['initial_state'], gdr['effective_length']]
    outputs = [decoder['results'][k] for k in ('out_proj', 'conv_state_0', 'recurrent_state_0')]
    kept, used = slice_dependencies(nodes, outputs, set(inputs), 'whole_gdn')
    if used != set(inputs) or {n for n in kept if nodes[n].type == 'SiluPatch'} != {conv_patch.name, gate_patch.name}:
        raise ValueError('Expected four GDN inputs and both SiluPatch nodes')
    wires = {}
    for k, edge in zip(ENTRY, inputs, strict=True):
        desc = port_info(graph, descriptor(nodes, edge))
        shape = desc['shape'] if desc['shape'] and min(desc['shape']) > 0 else SHAPES[k]
        check_view(k, shape)
        dtype = 'int16' if k == 'length' else 'float16'
        wires[edge] = wire(k, shape, dtype)
    for k, edge in zip(WHOLE_OUTPUTS, outputs, strict=True):
        spec = decoder['wires'][edge]
        check_view(k, spec['shape'])
        wires[edge] = wire(k, spec['shape'], spec['dtype'])
    return dict(name='whole_gdn', inputs=inputs, outputs=outputs, nodes=sorted(kept), wires=wires)


def stages(raw, meta, *, whole=False):
    trace = meta['silu_patch']['nodes']
    decoder = plan(raw, trace, layer=0, head=True, verify_precision=True)
    _, graph = graph_message(raw, 'SiluPatch')
    nodes = {op.name: op for op in graph.op}
    walk = AttentionSteps(graph, nodes, [])
    sites = {tuple(s['shape']): nodes[s['name']] for s in trace}
    conv_patch, gate_patch = sites[(1,8192,64)], sites[(2048,128)]
    gate_edge = gate_patch.name + ':0'
    mul, rms = gate_product(graph, gate_edge)
    gdr = stage_edges(graph, 'gdr')
    writeback = recurrent_writeback(graph, nodes, gdr['final_state'], decoder['results']['recurrent_state_0'])
    print(f'L0 recurrent cache writeback: {writeback}', flush=True)
    conv = walk.node(walk.nearest(conv_patch.input[0], CONVS))
    projections = dict(qkv=walk.nearest(conv.input[0], {MATMUL}), z=walk.nearest(gate_patch.input[0], {MATMUL}),
                       a=walk.nearest(gdr['g'], {MATMUL}), b=walk.nearest(gdr['beta'], {MATMUL}),
                       out=walk.strip(decoder['results']['out_proj']))
    if len(set(projections.values())) != 5 or any(walk.node(e).type != MATMUL for e in projections.values()):
        raise ValueError(f'expected five distinct L0 W4 projections: {projections}')
    quant = {p: walk.node(walk.strip(walk.node(e).input[0])) for p,e in projections.items()}
    for p, op in quant.items():
        if op.type != 'DynamicQuant':
            raise ValueError(f'{p}: expected DynamicQuant, found {op.type}')
    if whole:
        return [whole_stage(graph, decoder, quant, conv_patch, gate_patch, gdr)]
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
                 core=gdr['core'], final_state=gdr['final_state'], rms=rms, gate_silu=gate_edge, product=mul.name+':0',
                 gated=_project_input(nodes, projections['out']), state=gdr['initial_state'], length=gdr['effective_length'])
    for k in ('query', 'key', 'value', 'beta'):
        edges[k] = gdr[k]
    edges['history'] = only((e for e,v in decoder['entries'].items() if v['shape'] == [1,8192,4]), 'conv history')
    for p, op in quant.items():
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
                     ('rms','product','biased_a','softplus','g') else 'float16')
            if name.endswith('_quant') and key == inputs[0] and quant[p].input_desc:
                dtype = types[port_info(graph, quant[p].input_desc[0])['dtype']]
            if ((key in ('biased_a', 'softplus', 'g') or (name.endswith('_quant') and key == inputs[0]))
                    and desc['dtype'] in ('DT_FLOAT16', 'DT_FLOAT')):
                dtype = types[desc['dtype']]
            wires[edge] = wire(key,shape,dtype)
        specs.append(dict(name=name, inputs=src, outputs=target, nodes=sorted(retained), wires=wires,
                          weight_storage='embedded'))
    return specs

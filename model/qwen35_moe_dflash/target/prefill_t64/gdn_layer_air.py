"""Bind independent diagnostic OMCs to the corrected L0 decoder's original AIR edges."""
from .attention_slice import descriptor, port_info, DTYPES
from .attention_steps import AttentionSteps, CONVS, MATMUL, only
from .decode_cache_control import graph_message
from .decode_cache_trace import _project_input
from .decoder_four_silu import gate_product
from .gdn_layer_stages import ENTRY, SHAPES, STAGES, WHOLE_OUTPUTS, check_view
from .gdr_stage_air import stage_edges
from .l0_split import plan, slice_dependencies, wire


def before_padding_mask(walk, edge):
    """Return the projected values selected for valid rows in the PyTorch where."""
    node = walk.node(walk.strip(edge))
    if node.type in ('Select', 'SelectV2'):
        return walk.strip(node.input[1])
    return walk.strip(edge)


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


def four_layer_gdn(raw, meta):
    """Cut L0 linear_attn directly from the native T64 four-layer physical-history graph."""
    from collections import Counter
    from .attention_steps import NORMS, operands
    from .decoder_four_silu import activation_port_shape

    if list(meta['layers']) != [0, 1, 2, 3] or not (
            meta.get('physical_convstate') or meta.get('native_convstate') is False):
        raise ValueError('Select the four-layer T64 AIR with physical ConvState')
    _, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    nodes = {op.name: op for op in graph.op}
    walk = AttentionSteps(graph, nodes, [])
    returned = {r['name']: r['edge'] for r in meta['outputs']}
    history_out, state_out = (returned[k] for k in ('conv_state_0', 'recurrent_state_0'))

    def value_edge(edge):
        op = walk.node(edge)
        return operands(op)[1] if op.type in ('Assign', 'AssignVariableOp') else edge

    history_out, state_out = map(value_edge, (history_out, state_out))
    gdr = walk.node(walk.nearest(state_out, {'ChunkGatedDeltaRule'}))
    if not walk.depends(state_out, gdr.name + ':1'):
        raise ValueError(f'{state_out} does not return the L0 recurrent state from {gdr.name}:1')
    qkv = walk.node(walk.nearest(history_out, {MATMUL}))
    quant = walk.expect(walk.strip(qkv.input[0]), {'DynamicQuant'})
    hidden = quant.input[0]
    while walk.node(hidden).type in ('Reshape', 'View', 'Identity', 'TensorMove'):
        hidden = walk.node(hidden).input[0]
    history = only((op.name + ':0' for op in graph.op if op.type in ('Data', 'RefData')
                    and list(op.output_desc[0].shape.dim) == [1, 8192, 4]
                    and walk.depends(history_out, op.name + ':0')), 'L0 history input')
    ports = dict(zip((s.decode() for s in gdr.attr['_input_name_key'].list.s),
                     gdr.attr['_input_name_value'].list.i, strict=True))
    state, length = (gdr.input[ports[k]] for k in ('initial_state', 'effective_length'))

    # Match the output projection by its activation path, including for native SiLU.
    # Its DynamicQuant reads the gated product whose RMS operand comes from this GDR.
    projections = []
    for op in graph.op:
        if op.type != MATMUL:
            continue
        dq = walk.node(walk.strip(op.input[0]))
        if dq.type != 'DynamicQuant':
            continue
        product = walk.node(walk.strip(dq.input[0]))
        if product.type == 'Mul':
            branches = (walk.node(walk.strip(e)) for e in operands(product))
            if any(n.type in NORMS and walk.strip(n.input[0]) == gdr.name + ':0' for n in branches):
                projections.append(op.name + ':0')
    projection = only(projections, 'L0 gated out_proj')
    residual_inputs = (e for op in graph.op if op.type in ('Add', 'AddV2') for e in operands(op)
                       if before_padding_mask(walk, e) == projection)
    out = only(residual_inputs, 'L0 out_proj output entering the residual')
    inputs, outputs = [hidden, history, state, length], [out, history_out, state_out]
    kept, used = slice_dependencies(nodes, outputs, set(inputs), 'L0 linear_attn')
    counts = Counter(nodes[n].type for n in kept)
    if used != set(inputs) or counts[MATMUL] != 5 or counts['ChunkGatedDeltaRule'] != 1 or counts['SoftplusPatch'] != 1:
        raise ValueError(f'Unexpected L0 attention boundary: inputs={used}, operators={dict(counts)}')
    selected = meta.get('rms_silu_layers')
    if selected:
        expected = 1 + int('0' in selected.split(','))
        if counts['SiluPatch'] != expected:
            raise ValueError(f'Expected {expected} L0 SiluPatch nodes for --rms-silu-layers {selected}, found {counts["SiluPatch"]}')

    wires = {}
    for role, edge in zip((*ENTRY, *WHOLE_OUTPUTS), inputs + outputs, strict=True):
        desc = descriptor(nodes, edge)
        shapes = activation_port_shape(desc)
        fallback = (1, 64, 2048) if role == 'out' else SHAPES[role]
        shape = next((v for v in (shapes['ge_dims'], shapes['fx_dims']) if v and min(v) > 0), fallback)
        check_view(role, shape)
        # GDR's typed ports can be more specific than an intermediate producer's FX descriptor.
        gdr_name = {'length': 'effective_length', 'state': 'initial_state'}.get(role)
        port = gdr.input_desc[ports[gdr_name]] if gdr_name else desc
        dtype = {v: k for k, v in DTYPES.items()}.get(port_info(graph, port)['dtype'])
        allowed = ('int16',) if role == 'length' else ('float16', 'float32')
        if dtype not in allowed:
            raise ValueError(f'{edge}: unexpected {role} descriptor {port.dtype}')
        wires[edge] = wire(role, shape, dtype)
    print(f'[L0 linear_attn] RMS SiluPatch 层选择={selected}，W4 投影=5，GDR=1，'
          f'SiluPatch={counts["SiluPatch"]}，SoftplusPatch={counts["SoftplusPatch"]}', flush=True)
    return dict(nodes=[op.name for op in graph.op if op.name in kept], inputs=inputs, outputs=outputs, wires=wires)


def gdn_load_sections(raw, meta):
    """Partition the retained standalone L0 AIR at the GDR core and gated product."""
    from .attention_steps import operands
    from .decoder_four_silu import activation_port_shape
    from .gdr_stage_data import GDR, shapes

    _, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    nodes = {op.name: op for op in graph.op}
    net, = [op for op in graph.op if op.type == 'NetOutput']
    gdr, = [op for op in graph.op if op.type == 'ChunkGatedDeltaRule']
    inputs = {spec['name']: name + ':0' for name, spec in meta['inputs'].items()}
    outputs = {spec['name']: edge for edge, spec in zip(operands(net), meta['outputs'], strict=True)}
    if set(inputs) != set(ENTRY) or set(outputs) != set(WHOLE_OUTPUTS):
        raise ValueError('Expected the four inputs and three outputs of the standalone L0 GDN')
    wires = {name + ':0': spec for name, spec in meta['inputs'].items()}
    wires.update(zip(operands(net), meta['outputs'], strict=True))
    edges = stage_edges(graph, 'gdr')
    walk = AttentionSteps(graph, nodes, [])
    projection = before_padding_mask(walk, outputs['out'])
    product = walk.strip(_project_input(nodes, projection))
    mul = walk.expect(product, {'Mul'})
    if not walk.depends(product, edges['core']):
        raise ValueError(f'{product} is not downstream of the GDR core')

    wire_errors = {}

    def add_wire(edge, role, desc, views, expected_dtype=None):
        info = activation_port_shape(desc)
        shape = next((v for v in (info['ge_dims'], info['fx_dims']) if v and min(v) > 0), views[0])
        if expected_dtype is None:
            expected_dtype = {1: 'float32', 2: 'float16'}.get(desc.dtype)
        if list(shape) not in views or not expected_dtype or port_info(graph, desc)['dtype'] != DTYPES[expected_dtype]:
            wire_errors[edge] = f'{edge}: unexpected {role} descriptor {port_info(graph, desc)}, FX {info["fx_dims"]}'
            return
        # state/length or a returned cache can also be a public port of the whole AIR.
        wires.setdefault(edge, wire(role, shape, expected_dtype))

    for role in GDR:
        edge = edges[role]
        if role in ('core', 'final_state'):
            desc = descriptor(nodes, edge)
        else:
            desc = gdr.input_desc[list(gdr.input).index(edge)]
        add_wire(edge, role, desc, shapes(role), 'int16' if role == 'effective_length' else 'float16')
    add_wire(product, 'product', mul.output_desc[0], [[2048, 128]])

    hidden, history = inputs['hidden'], inputs['history']
    entry = [inputs[k] for k in ENTRY]
    core = edges['core']
    caches = [outputs['new_history'], outputs['recurrent_cache']]
    gdr_inputs = [edges[k] for k in GDR[:-2]]
    qkv = walk.nearest(outputs['new_history'], {MATMUL})
    add_wire(qkv, 'qkv', descriptor(nodes, qkv), [[64, 8192], [1, 64, 8192]], 'float16')
    qkv_results = [edges[k] for k in ('query', 'key', 'value')] + [outputs['new_history']]
    decay_results = [edges['g'], edges['beta']]
    gate_operand = only((e for e in operands(mul) if not walk.depends(e, core)), 'z gate operand')
    add_wire(gate_operand, 'gate_operand', mul.input_desc[list(mul.input).index(gate_operand)], [[2048, 128]])
    out = outputs['out']
    mask_length = [inputs['length']] if projection != walk.strip(out) else []

    # Cuts follow data edges. Cross-cut cases keep both producers at a join or
    # replace only one side of the gate, which can change OMG task construction.
    cases = [
        ('gdn-front', entry, [core, *caches], 3, True),
        ('gdn-tail', [hidden, core, *mask_length], [out], 2, False),
        ('pre-gdr', [hidden, history], qkv_results + decay_results, 3, False),
        ('gdr-state', gdr_inputs, [core, outputs['recurrent_cache']], 0, True),
        ('qkv-path', [hidden, history], qkv_results, 1, False),
        ('decay-beta', [hidden], decay_results, 2, False),
        ('qkv-project', [hidden], [qkv], 1, False),
        ('conv-to-qkv', [qkv, history], qkv_results, 0, False),
        ('decay-path', [hidden], [edges['g']], 1, False),
        ('beta-path', [hidden], [edges['beta']], 1, False),
        ('gate-product', [hidden, core], [product], 1, False),
        ('out-proj', [product, *mask_length], [out], 1, False),
        ('z-gate', [hidden], [gate_operand], 1, False),
        ('rms-mul', [core, gate_operand], [product], 0, False),
        ('through-gate', entry, [product, *caches], 4, True),
        ('gdr-to-gate', [hidden, *gdr_inputs], [product], 1, True),
        ('gdr-to-out', [hidden, *gdr_inputs], [out], 2, True),
    ]
    if outputs['recurrent_cache'] != edges['final_state']:
        cases.append(('gdr-only', gdr_inputs, [core, edges['final_state']], 0, True))
    gate = walk.node(walk.strip(gate_operand))
    if gate.type == 'SiluPatch':
        before, = operands(gate)
        after = gate.name + ':0'
        kernel_shape = [64, 4096] if meta.get('rms_silu_token_view') else [2048, 128]
        add_wire(before, 'gate_input', gate.input_desc[0], [kernel_shape])
        add_wire(after, 'gate_silu', gate.output_desc[0], [kernel_shape], 'float16')
        cases.extend([
            ('z-project', [hidden], [before], 1, False),
            ('gate-activation', [before], [gate_operand], 0, False),
            ('cut-before-silu', [*entry, before], [out, *caches], 4, True),
            ('cut-after-silu', [*entry, after], [out, *caches], 4, True),
        ])
    print(f'[L0 GDN] RMS 门控源节点 {gate.name} ({gate.type})，计划编译 {len(cases)} 个切片 OMC', flush=True)
    for label, feeds, results, matmuls, has_gdr in cases:
        try:
            for edge in feeds + results:
                if edge in wire_errors:
                    raise ValueError(wire_errors[edge])
            kept, used = slice_dependencies(nodes, results, set(feeds), label)
            if (used != set(feeds) or sum(nodes[n].type == MATMUL for n in kept) != matmuls
                    or (gdr.name in kept) != has_gdr or (label == 'gdr-only' and kept != {gdr.name})):
                raise ValueError(f'{label}: cut differs from the expected L0 dependencies')
            step = dict(inputs=feeds, outputs=results, nodes=[name for name in nodes if name in kept])
        except ValueError as err:
            step = dict(inputs=feeds, outputs=results, error=str(err))
        yield label, step, wires


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
                       out=before_padding_mask(walk, decoder['results']['out_proj']))
    if len(set(projections.values())) != 5 or any(walk.node(e).type != MATMUL for e in projections.values()):
        raise ValueError(f'expected five distinct L0 W4 projections: {projections}')
    quant = {p: walk.node(walk.strip(walk.node(e).input[0])) for p,e in projections.items()}
    for p, op in quant.items():
        if op.type != 'DynamicQuant':
            raise ValueError(f'{p}: expected DynamicQuant, found {op.type}')
    softplus_nodes = {n.name for n in graph.op if n.type == 'SoftplusPatch'}
    recorded = meta.get('softplus_patch')
    expected = set() if recorded is None else {n['name'] for n in recorded['nodes']}
    if softplus_nodes != expected or (recorded is not None and len(expected) != 1):
        raise ValueError(f'L0 SoftplusPatch differs from its export manifest: {sorted(softplus_nodes)}')
    if softplus_nodes and not walk.depends(gdr['g'], next(iter(softplus_nodes)) + ':0'):
        raise ValueError('SoftplusPatch output must contribute to GDR decay g')
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
        if softplus_nodes and bool(softplus_nodes.intersection(retained)) != (name == 'decay_softplus'):
            raise ValueError(f'{name}: SoftplusPatch must belong to the decay_softplus stage')
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
            if ((key in ('rms', 'product', 'biased_a', 'softplus', 'g', 'conv') or (name.endswith('_quant') and key == inputs[0]))
                    and desc['dtype'] in ('DT_FLOAT16', 'DT_FLOAT')):
                dtype = types[desc['dtype']]
            wires[edge] = wire(key,shape,dtype)
        specs.append(dict(name=name, inputs=src, outputs=target, nodes=sorted(retained), wires=wires,
                          weight_storage='embedded'))
    return specs

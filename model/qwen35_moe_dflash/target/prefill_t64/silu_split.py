"""Cut one T64 L0 AIR around both SiLUs. All four models use the same source edges."""

from .attention_cpu_silu import replace_sites, silu_sites
from .attention_slice import DTYPES, descriptor
from .decode_cache_control import graph_message

STAGES = ('prefix', 'native_silu', 'cpu_silu', 'suffix')
VIEWS = {'conv': ((1, 8192, 64), (1, 8192, 1, 64), (8192, 64)),
         'gate': ((2048, 128), (1, 64, 32, 128), (64, 32, 128), (1, 64, 4096), (64, 4096))}


def wire_shape(site, desc):
    dims = list(desc.shape.dim)
    if not dims or dims == [-2]:
        return list(VIEWS[site][0])
    matches = [view for view in VIEWS[site] if len(view) == len(dims)
               and all(x < 0 or x == y for x, y in zip(dims, view))]
    if not matches:
        raise ValueError(f'{site}: unsupported SiLU shape {dims}')
    # The prefix ABI supplies the concrete view before building either consumer.
    return list(matches[0])


def cut(raw, stage, original_inputs, golden, *, prefix_shapes=None):
    read_proj_input = stage == 'suffix_proj_input'
    if read_proj_input:
        stage = 'suffix'
    if stage not in STAGES:
        raise ValueError(f'Unknown split stage {stage}')
    msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    sites = silu_sites(graph)
    # Resolve types exactly as for the full L0 comparison, including sparse AIR descriptors.
    if stage == 'cpu_silu':
        info = replace_sites(graph, sites)
    else:
        _, copy = graph_message(raw, 'ChunkGatedDeltaRule')
        info = replace_sites(copy, sites)
    nodes = {op.name: op for op in graph.op}
    net, = [op for op in graph.op if op.type == 'NetOutput']
    hidden, = [e for e in net.input if e and not e.endswith(':-1')]
    src = {s['name']: s for s in original_inputs}
    bindings, results = {}, []
    for site, (x, y) in sites.items():
        shape = wire_shape(site, descriptor(nodes, x))
        if prefix_shapes:
            shape = prefix_shapes[site]
            if tuple(shape) not in VIEWS[site]:
                raise ValueError(f'{site}: unsupported prefix output view {shape}')
        before = dict(name='split_' + site + '_in', shape=shape, dtype=info[site]['input_dtype'])
        after = dict(name='split_' + site + '_out', shape=shape, dtype=info[site]['output_dtype'])
        if stage == 'prefix':
            results.append((x, before))
        elif stage == 'suffix':
            bindings[y] = after
        else:
            bindings[x] = before
            results.append((y, after))
    controls = []
    if stage == 'suffix':
        results.append((hidden, golden))
        controls = [e for e in net.input if e.endswith(':-1')]
    required, used = set(), set()

    def visit(edge):
        if edge in bindings:
            used.add(edge)
            return
        op = nodes[edge.rsplit(':', 1)[0]]
        if op.name in required:
            return
        if op.subgraph_name:
            raise ValueError(f'Cannot cut through a nested subgraph at {op.name}')
        required.add(op.name)
        for e in op.input:
            if e:
                visit(e)

    for edge in [*(e for e, _ in results), *controls]:
        visit(edge)
    if used != set(bindings):
        raise ValueError(f'Unused SiLU cut inputs: {set(bindings) - used}')
    gdr_count = sum(nodes[n].type == 'ChunkGatedDeltaRule' for n in required)
    silu_producers = {y.rsplit(':', 1)[0] for _, y in sites.values()}
    if stage in ('prefix', 'suffix') and required & silu_producers:
        raise ValueError(f'{stage} includes an original SiLU across the cut: {required & silu_producers}')
    if gdr_count != int(stage == 'suffix'):
        raise ValueError(f'{stage}: unexpected GDR count {gdr_count}')
    codes = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    original_data = sorted((op for op in graph.op if op.type == 'Data' and op.name in required),
                           key=lambda op: op.attr['index'].i)
    inputs = [src[op.name] for op in original_data]
    if stage.endswith('_silu') and inputs:
        raise ValueError(f'{stage}: SiLU graph depends on additional original inputs')
    for index, op in enumerate(original_data):
        op.attr['index'].i = index
    additions = []
    for edge, spec in bindings.items():
        name = spec['name']
        if name in nodes:
            raise ValueError(f'Cut input collides with source node {name}')
        op = type(net)(name=name, type='Data', id=max(n.id for n in nodes.values()) + 1 + len(additions))
        op.attr['index'].i = len(inputs)
        if any(n.type == 'Data' and list(n.input) == [''] for n in nodes.values()):
            op.input.append('')
        for descs in (op.input_desc, op.output_desc):
            d = descs.add(dtype=codes[DTYPES[spec['dtype']]].number, layout='ND', device_type='NPU')
            d.shape.dim.extend(spec['shape'])
        additions.append(op)
        inputs.append(spec)
    body = []
    for op in graph.op:
        if op.name not in required:
            continue
        for i, e in enumerate(op.input):
            if e in bindings:
                name = bindings[e]['name']
                op.input[i] = name + ':0'
                if op.src_name:
                    op.src_name[i] = name
                if op.src_index:
                    op.src_index[i] = 0
        del op.dst_name[:]
        del op.dst_index[:]
        body.append(op)
    outputs = [dict(name=spec['name'], shape=spec['shape'], dtype=spec['dtype']) for _, spec in results]
    out = net
    if stage != 'suffix':
        out = type(net)(name=net.name, type='NetOutput', id=net.id)
        for edge, spec in results:
            out.input.append(edge)
            desc = out.input_desc.add()
            desc.CopyFrom(descriptor(nodes, edge))
            desc.dtype = codes[DTYPES[spec['dtype']]].number
    del graph.op[:]
    graph.op.extend([*additions, *body, out])
    del graph.input[:]
    graph.input.extend(s['name'] + ':0' for s in inputs)
    if graph.output:
        via_net = graph.output[0] == net.name + ':0'
        del graph.output[:]
        graph.output.extend([net.name + ':' + str(i) if via_net else edge for i, (edge, _) in enumerate(results)])
    edges = [e for e, _ in results]
    if read_proj_input:
        from .decode_cache_trace import _add_ports, _project_input
        nodes = {op.name: op for op in graph.op}
        net, = [op for op in graph.op if op.type == 'NetOutput']
        if not net.input_desc:
            net.input_desc.add().CopyFrom(descriptor(nodes, hidden))
        edge = _project_input(nodes, hidden)
        edges = _add_ports(graph, {'out_proj_input': edge})
        outputs.append(dict(name='out_proj_input', dtype='float16', shape=[64, 4096]))
        stage = 'suffix_proj_input'
    return msg.SerializeToString(), dict(stage=stage, inputs=inputs, outputs=outputs,
        output_edges=edges, cut_inputs={e: s['name'] for e, s in bindings.items()},
        silu_sites=info, gdr_nodes=gdr_count)

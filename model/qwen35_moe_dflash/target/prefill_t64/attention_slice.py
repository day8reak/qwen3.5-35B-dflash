"""Inspect original T64 attention AIR and expose one output with explicit runtime input edges."""
from pathlib import Path
import shutil

from ..layer_probe.common import checked, identity
from .decode_cache_control import graph_message
from .decode_cache_trace import _record


CASE_IDS = {0: 'l00_c00_gdn_layer_hidden_o0', 3: 'l03_c00_full_attention_layer_hidden_o0'}
DTYPES = {'float16': 'DT_FLOAT16', 'float32': 'DT_FLOAT', 'int8': 'DT_INT8', 'uint8': 'DT_UINT8',
          'int16': 'DT_INT16', 'int32': 'DT_INT32', 'int64': 'DT_INT64', 'bool': 'DT_BOOL'}


def source_graph(air, layer):
    _, case, folder, spec = _record(air, CASE_IDS[layer])
    if (case['layer'] != layer or case['point'] != ('gdn_layer' if layer == 0 else 'full_attention_layer')
            or (spec['physical_rows'], spec['valid_rows'], spec['reference_position']) != (64, 19, 18)):
        raise ValueError(f'Expected the original L{layer} T64/valid19 attention AIR')
    msg, graph = graph_message(checked(folder, spec['air']).read_bytes(), 'NetOutput')
    return folder, spec, msg, graph


def descriptor(nodes, edge):
    name, sep, slot = edge.rpartition(':')
    if not sep or not slot.isdecimal() or name not in nodes or int(slot) >= len(nodes[name].output_desc):
        raise ValueError(f'Unknown data edge {edge!r}; use list to inspect the source AIR')
    return nodes[name].output_desc[int(slot)]


def port_info(graph, desc):
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number.get(desc.dtype)
    return dict(dtype=enum.name if enum else desc.dtype, shape=list(desc.shape.dim), layout=desc.layout)


def listing(graph):
    for op in graph.op:
        yield dict(name=op.name, type=op.type, inputs=list(op.input),
                   outputs=[dict(edge=f'{op.name}:{slot}', **port_info(graph, desc))
                            for slot, desc in enumerate(op.output_desc)])


def extract(msg, graph, folder, dest, ports, target):
    """ports maps source edge to (new Data name, logical dtype, static shape). Keep original ops and constants."""
    nodes = {op.name: op for op in graph.op}
    net, = [op for op in graph.op if op.type == 'NetOutput']
    original_output, = [e for e in net.input if e and not e.endswith(':-1')]
    if list(graph.output) not in ([], [net.name + ':0'], [original_output]):
        raise ValueError('Expected one original attention output')
    if target in ports or not ports or len({p[0] for p in ports.values()}) != len(ports):
        raise ValueError('Select distinct runtime ports and an output computed inside the slice')
    out_desc = descriptor(nodes, target)
    codes = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    used, kept = set(), set()
    def visit(edge):
        if edge in ports:
            used.add(edge)
            return
        op = nodes[edge.rsplit(':', 1)[0]]
        if op.name in kept:
            return
        if op.type in ('Data', 'NetOutput') or op.subgraph_name:
            raise ValueError(f'Unbound dependency {edge} ({op.type}); specify all incoming data edges')
        kept.add(op.name)
        for arg in op.input:
            if arg:
                visit(arg)
    visit(target)
    if used != set(ports):
        raise ValueError(f'Inputs not used by the selected output: {sorted(set(ports) - used)}')
    new_ops = []
    ident = max(n.id for n in graph.op) + 1
    empty_arg = any(n.type == 'Data' and list(n.input) == [''] for n in graph.op)
    for idx, (edge, (name, dtype, shape)) in enumerate(ports.items()):
        desc = descriptor(nodes, edge)
        if name in nodes or dtype not in DTYPES or any(type(n) != int or n < 1 for n in shape):
            raise ValueError(f'Invalid new Data name, dtype or shape for {edge}')
        code = codes[DTYPES[dtype]].number
        dims = list(desc.shape.dim)
        if (desc.dtype not in (0, code) or (dims and dims != [-2] and (len(dims) != len(shape)
                or any(a > 0 and a != b for a, b in zip(dims, shape))))):
            raise ValueError(f'{edge}: requested {dtype} {shape}, source declares {port_info(graph, desc)}')
        data = type(net)(name=name, type='Data', id=ident+idx)
        data.attr['index'].i = idx
        if empty_arg:
            data.input.append('')
        for descriptors in (data.input_desc, data.output_desc):
            inp = descriptors.add(dtype=code, layout='ND', device_type='NPU')
            inp.shape.dim.extend(shape)
        new_ops.append(data)
    weights = {}
    for op in graph.op:
        if op.name not in kept:
            continue
        for idx, arg in enumerate(op.input):
            if arg in ports:
                name = ports[arg][0]
                op.input[idx] = name + ':0'
                if op.src_name:
                    op.src_name[idx] = name
                if op.src_index:
                    op.src_index[idx] = 0
        del op.dst_name[:]
        del op.dst_index[:]
        if op.type == 'FileConstant':
            rel = Path(op.attr['file_path'].s.decode())
            src = (folder/rel).resolve(strict=True)
            if rel.is_absolute() or '..' in rel.parts or folder.resolve() not in src.parents:
                raise ValueError(f'Invalid external weight path {rel}')
            if str(rel) not in weights:
                dst = dest/rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                weights[str(rel)] = identity(dst, dest)
        new_ops.append(op)
    final = type(net)(name=net.name, type='NetOutput', id=net.id, input=[target])
    final.input_desc.add().CopyFrom(out_desc)
    out = port_info(graph, out_desc)
    new_ops.append(final)
    del graph.op[:]
    graph.op.extend(new_ops)
    del graph.input[:]
    graph.input.extend(p[0] + ':0' for p in ports.values())
    if graph.output and graph.output[0] != net.name + ':0':
        graph.output[0] = target
    path = dest/'export.air'
    path.write_bytes(msg.SerializeToString())
    return dict(air=identity(path, dest), source_edge=target, output_desc=out, weight_evidence=weights,
                inputs=[dict(edge=e, name=p[0], dtype=p[1], shape=list(p[2])) for e, p in ports.items()],
                nodes=[dict(name=n.name, type=n.type, inputs=list(n.input)) for n in graph.op])

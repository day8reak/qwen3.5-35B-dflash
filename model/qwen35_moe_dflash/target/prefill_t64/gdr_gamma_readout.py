"""Read CPro RMSNorm gamma or SiLU activation rows through a runtime-indexed gather."""
import hashlib

import numpy as np

from ..layer_probe.common import identity
from .gdr_stage_data import shapes

READOUT_SHAPES = {'gamma': [128], 'silu_output': [64*32, 128]}


def indices(role='gamma'):
    return np.arange(READOUT_SHAPES[role][0], dtype='<i4')


def index_binding(folder, source_inputs, role='gamma'):
    values = indices(role)
    name = role + '_read_indices'
    path = folder / (name + '.bin')
    values.tofile(path)
    return dict(identity(path, folder), name=name, semantic_name=name,
                fixture_index=1 + max(s['fixture_index'] for s in source_inputs),
                dtype='int32', shape=list(values.shape))


def verify_index_input(binding, role='gamma'):
    values = indices(role)
    if binding != ('int32', list(values.shape), hashlib.sha256(values.tobytes()).hexdigest()):
        raise ValueError(f'Expected the executed {role} readout indices to be INT32 [0..{len(values)-1}]')


def verify_compiled_readout(contract, inputs, nodes, net, role='gamma'):
    readout = contract[role + '_readout']
    indices = [s for s in inputs if s['name'] == readout['index_input'] and s['semantic_name'] == role + '_read_indices']
    if len(indices) != 1:
        raise ValueError(f'Compiled {role} readout lost its runtime indices')
    spec = indices[0]
    verify_index_input((spec['dtype'], spec['shape'], spec['sha256']), role)

    slot = contract['order'].index(role)
    if contract['source_edges'][slot] != readout['output_edge']:
        raise ValueError(f'{role} output does not match the AIR readout edge')
    # Empty data edges keep their slot in NetOutput. Skipping them shifts every later port.
    refs = [e for e in net.get('input', []) if not e.endswith(':-1')]
    if len(refs) != len(contract['order']):
        raise ValueError('Compiled NetOutput does not identify every probe slot')
    owner, _, port = refs[slot].rpartition(':')
    producer = next((n for n in nodes if n['name'] == owner), None)
    if not producer or not port.isdecimal() or int(port) >= len(producer.get('output_desc', [])):
        raise ValueError(f'Compiled output {slot} ({role}) has no connected tensor producer')


def add_indexed_readout(graph, source_edge, role):
    shape_dims = READOUT_SHAPES[role]
    nodes = {n.name: n for n in graph.op}
    owner, port = source_edge.rsplit(':', 1)
    source = nodes[owner].output_desc[int(port)]
    types = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    if source.dtype not in (types['DT_FLOAT'].number, types['DT_FLOAT16'].number):
        raise ValueError(f'{source_edge}: expected a floating {role} tensor')
    if source.shape.dim and list(source.shape.dim) not in shapes(role):
        raise ValueError(f'{source_edge}: unexpected {role} shape {list(source.shape.dim)}')

    data = [n for n in graph.op if n.type == 'Data']
    if not data or any('index' not in n.attr or not n.attr['index'].HasField('i') for n in data):
        raise ValueError('Cannot assign the readout input an AIR Data index')
    added = []
    next_id = 1 + max(n.id for n in graph.op)

    def op(name, typ):
        if name in nodes:
            raise ValueError('Existing AIR node conflicts with indexed readout: ' + name)
        node = type(nodes[owner])(name=name, type=typ, id=next_id + len(added))
        added.append(node)
        return node

    def desc(dtype, dims):
        value = type(source)(dtype=dtype, layout='ND', device_type='NPU')
        value.shape.SetInParent()
        value.shape.dim.extend(dims)
        return value

    i32 = types['DT_INT32'].number

    def const(name, data):
        value = np.asarray(data, dtype='<i4')
        node = op(name, 'Const')
        descriptor = desc(i32, value.shape)
        node.output_desc.add().CopyFrom(descriptor)
        tensor = node.attr['value'].t
        tensor.desc.CopyFrom(descriptor)
        tensor.data = value.tobytes()
        return node

    index = op(role + '_read_indices', 'Data')
    index.attr['index'].i = 1 + max(n.attr['index'].i for n in data)
    if list(data[0].input) == ['']:
        index.input.append('')
    index.input_desc.add().CopyFrom(desc(i32, shape_dims[:1]))
    index.output_desc.add().CopyFrom(index.input_desc[0])
    shape = const(role + '_read_shape', shape_dims)
    axis = const(role + '_read_axis', 0)
    vector = op(role + '_read_vector', 'Reshape')
    vector.input.extend([source_edge, shape.name + ':0'])
    vector.input_desc.extend([source, shape.output_desc[0]])
    vector.output_desc.add().CopyFrom(desc(source.dtype, shape_dims))
    gather = op(role + '_read_gather', 'GatherV2')
    gather.input.extend([n.name + ':0' for n in (vector, index, axis)])
    gather.input_desc.extend([n.output_desc[0] for n in (vector, index, axis)])
    gather.output_desc.add().CopyFrom(vector.output_desc[0])
    gather.attr['batch_dims'].i = 0
    net_idx, = [i for i, n in enumerate(graph.op) if n.type == 'NetOutput']
    for offset, node in enumerate(added):
        graph.op.insert(net_idx + offset, node)
    if graph.input:
        graph.input.append(index.name + ':0')
    return dict(source_edge=source_edge, output_edge=gather.name + ':0', index_input=index.name)

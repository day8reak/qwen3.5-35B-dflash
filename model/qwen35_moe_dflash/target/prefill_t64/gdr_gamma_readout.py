"""Read the CPro RMSNorm weight through a runtime-indexed, diagnostic-only gather."""
import hashlib

import numpy as np

from ..layer_probe.common import identity
from .gdr_stage_data import shapes

INDEX_INPUT = 'gamma_read_indices'


def indices():
    return np.arange(128, dtype='<i4')


def index_binding(folder, source_inputs):
    values = indices()
    path = folder / (INDEX_INPUT + '.bin')
    values.tofile(path)
    return dict(identity(path, folder), name=INDEX_INPUT, semantic_name=INDEX_INPUT,
                fixture_index=1 + max(s['fixture_index'] for s in source_inputs),
                dtype='int32', shape=list(values.shape))


def verify_index_input(binding):
    values = indices()
    if binding != ('int32', list(values.shape), hashlib.sha256(values.tobytes()).hexdigest()):
        raise ValueError('Expected the executed gamma readout indices to be INT32 [0..127]')


def verify_compiled_readout(contract, inputs, nodes, net):
    readout = contract['gamma_readout']
    indices = [s for s in inputs if s['name'] == readout['index_input'] and s['semantic_name'] == INDEX_INPUT]
    if len(indices) != 1:
        raise ValueError('Compiled gamma readout lost its runtime indices')
    spec = indices[0]
    verify_index_input((spec['dtype'], spec['shape'], spec['sha256']))

    slot = contract['order'].index('gamma')
    if contract['source_edges'][slot] != readout['output_edge']:
        raise ValueError('Gamma output does not match the AIR readout edge')
    # Empty data edges keep their slot in NetOutput. Skipping them shifts every later port.
    refs = [e for e in net.get('input', []) if not e.endswith(':-1')]
    if len(refs) != len(contract['order']):
        raise ValueError('Compiled NetOutput does not identify every gamma probe slot')
    owner, _, port = refs[slot].rpartition(':')
    producer = next((n for n in nodes if n['name'] == owner), None)
    if not producer or not port.isdecimal() or int(port) >= len(producer.get('output_desc', [])):
        raise ValueError(f'Compiled output {slot} (gamma) has no connected tensor producer')


def add_gamma_readout(graph, source_edge):
    nodes = {n.name: n for n in graph.op}
    owner, port = source_edge.rsplit(':', 1)
    gamma = nodes[owner].output_desc[int(port)]
    types = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    if gamma.dtype not in (types['DT_FLOAT'].number, types['DT_FLOAT16'].number):
        raise ValueError(f'{source_edge}: expected a floating RMSNorm weight')
    if gamma.shape.dim and list(gamma.shape.dim) not in shapes('gamma'):
        raise ValueError(f'{source_edge}: unexpected RMSNorm weight shape {list(gamma.shape.dim)}')

    data = [n for n in graph.op if n.type == 'Data']
    if not data or any('index' not in n.attr or not n.attr['index'].HasField('i') for n in data):
        raise ValueError('Cannot assign the gamma readout input an AIR Data index')
    added = []
    next_id = 1 + max(n.id for n in graph.op)

    def op(name, typ):
        if name in nodes:
            raise ValueError('Existing AIR node conflicts with gamma readout: ' + name)
        node = type(nodes[owner])(name=name, type=typ, id=next_id + len(added))
        added.append(node)
        return node

    def desc(dtype, dims):
        value = type(gamma)(dtype=dtype, layout='ND', device_type='NPU')
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

    index = op(INDEX_INPUT, 'Data')
    index.attr['index'].i = 1 + max(n.attr['index'].i for n in data)
    if list(data[0].input) == ['']:
        index.input.append('')
    index.input_desc.add().CopyFrom(desc(i32, [128]))
    index.output_desc.add().CopyFrom(index.input_desc[0])
    shape = const('gamma_read_shape', [128])
    axis = const('gamma_read_axis', 0)
    vector = op('gamma_read_vector', 'Reshape')
    vector.input.extend([source_edge, shape.name + ':0'])
    vector.input_desc.extend([gamma, shape.output_desc[0]])
    vector.output_desc.add().CopyFrom(desc(gamma.dtype, [128]))
    gather = op('gamma_read_gather', 'GatherV2')
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

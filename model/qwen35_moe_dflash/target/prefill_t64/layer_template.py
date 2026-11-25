"""Bind learned tensors as runtime inputs of a static, layer-independent CPro stage."""
from hashlib import sha256
from math import prod
from pathlib import Path
import tempfile

import numpy as np

from ..layer_probe.common import sha
from ..silu_patch_air import nd_desc, port_names
from .attention_slice import DTYPES
from .decode_cache_control import graph_message


FILE_DTYPES = {0: 'float32', 1: 'float16', 2: 'int8', 3: 'int32', 6: 'int16', 9: 'int64'}
SUPPORTED = set(FILE_DTYPES.values())


def inline_file(directory, data):
    digest = sha256(data).hexdigest()
    path = directory / (digest + '.bin')
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha(path) != digest:
            raise ValueError(f'Changed template weight {path}')
    else:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
            f.write(data)
        Path(f.name).replace(path)
    return path


def parameter(op, types, folder, directory):
    if op.type == 'FileConstant':
        attrs = op.attr
        other = {k for k in attrs if not k.startswith('_')} - {'file_path', 'dtype', 'shape', 'file_id'}
        if other or ('file_id' in attrs and attrs['file_id'].s):
            raise ValueError(f'{op.name}: unsupported external weight attributes {sorted(other)}')
        shape = list(attrs['shape'].list.i)
        dtype = FILE_DTYPES.get(attrs['dtype'].i) if attrs['dtype'].HasField('i') else None
        rel = Path(attrs['file_path'].s.decode())
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'{op.name}: invalid external weight {rel}')
        # Original torchair exports can qualify file_path with their own
        # directory name (OMG resolves that spelling from its working dir).
        if rel.parts and rel.parts[0] == folder.name:
            rel = rel.relative_to(folder.name)
        path = (folder/rel).resolve(strict=True)
        path.relative_to(folder.resolve())
    elif op.type in ('Const', 'Constant') and 'value' in op.attr:
        tensor = op.attr['value'].t
        dtype = types.get(tensor.desc.dtype)
        shape = list(tensor.desc.shape.dim)
        if not shape and op.output_desc:
            shape = list(op.output_desc[0].shape.dim)
        if not shape and len(tensor.data) > 8:
            raise ValueError(f'{op.name}: missing static shape for an inline tensor')
        # Integer shape, axis and permutation literals stay compile-time
        # constants. Packed W4 matrices and floating parameter vectors become inputs.
        if not shape or (dtype not in ('float16', 'float32') and (len(shape) < 2 or len(tensor.data) < 32)):
            return None
        if tensor.desc.layout not in ('', 'ND') or tensor.desc.data_offset:
            raise ValueError(f'{op.name}: unsupported inline storage {tensor.desc.layout!r}')
        path = inline_file(directory, tensor.data)
    else:
        return None
    if (dtype not in SUPPORTED or not shape or min(shape) < 1 or not path.is_file() or
            path.stat().st_size != prod(shape) * np.dtype(dtype).itemsize):
        raise ValueError(f'{op.name}: cannot bind exact runtime weight {dtype} {shape} from {path}')
    if any(d.layout not in ('', 'ND') for d in op.output_desc):
        raise ValueError(f'{op.name}: unsupported weight output format')
    return dtype, shape, path


def parameterize(raw, step, wires, folder, directory):
    """Canonicalize one slice by ordered dependencies and preserve parameter roles at every call."""
    msg, graph = graph_message(raw, 'NetOutput')
    codes = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    types = {codes[v].number: k for k, v in DTYPES.items()}
    nodes = {n.name: n for n in graph.op}
    net, = [op for op in graph.op if op.type == 'NetOutput']
    old_data = {f'l0_feed_{i}': edge for i, edge in enumerate(step['inputs'])}
    aliases, ops, inputs, extra, specs = {}, [], [], {}, {}
    def visit(name):
        if name in aliases:
            return aliases[name]
        op = nodes[name]
        weight = parameter(op, types, folder, directory)
        if name in old_data or weight is not None:
            slot = len(inputs)
            tag = f'l0_feed_{slot}'
            if weight is None:
                edge = old_data[name]
                dtype, shape = wires[edge]['dtype'], wires[edge]['shape']
            else:
                dtype, shape, path = weight
                edge = 'weight/' + name
                extra[edge] = path
            inputs.append(edge)
            specs[edge] = dict(dtype=dtype, shape=shape)
            value = type(op)(name=tag, type='Data', input=[''])
            value.attr['index'].i = slot
            port_names(value, 'x', 'y')
            for desc in (value.input_desc, value.output_desc):
                nd_desc(desc.add(dtype=codes[DTYPES[dtype]].number), shape)
        else:
            upstream = []
            for e in op.input:
                if e:
                    src, port = e.rsplit(':', 1)
                    e = visit(src) + ':' + port
                upstream.append(e)
            tag = 'NetOutput' if op.type == 'NetOutput' else f'{op.type}_{len(ops)}'
            op.name = tag
            for key in ('_datadump_original_op_names', '_original_op_names'):
                if key in op.attr:
                    del op.attr[key]
            op.input[:] = upstream
            del op.src_name[:]
            del op.src_index[:]
            del op.dst_name[:]
            del op.dst_index[:]
            # A lifted operand is a Data node even when the original FX export
            # marked the corresponding argument as constant.
            for idx, edge in enumerate(upstream):
                if idx < len(op.is_input_const) and edge.startswith('l0_feed_'):
                    op.is_input_const[idx] = False
            value = op
        aliases[name] = tag
        value.id = len(ops)
        ops.append(value)
        return tag
    old_net = net.name
    visit(old_net)
    if set(aliases) != set(nodes) or set(old_data.values()) - set(inputs):
        raise ValueError(f'{step["name"]}: disconnected stage inputs or operators')
    outputs = []
    for edge in step['outputs']:
        name, port = edge.rsplit(':', 1)
        alias = aliases[name] + ':' + port
        outputs.append(alias)
        specs[alias] = wires[edge]
    graph.output[:] = [aliases[n] + ':' + port for n, port in (e.rsplit(':', 1) for e in graph.output)]
    graph.input[:] = [f'l0_feed_{i}:0' for i in range(len(inputs))]
    graph.name = 'decoder_stage'
    if msg is not graph:
        msg.name = graph.name
    del graph.op[:]
    graph.op.extend(ops)
    current = dict(name=step['name'], inputs=inputs, outputs=outputs)
    return msg.SerializeToString(deterministic=True), current, specs, extra

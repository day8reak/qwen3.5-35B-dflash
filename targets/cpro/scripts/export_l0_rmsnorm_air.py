#!/usr/bin/env python3
"""Write a GE AIR containing one replaceable L0 input RMSNorm calculation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


def schema_bytes():
    script = Path(__file__).resolve()
    for path in (script.with_name('torchair-ge-ir.pb'), script.parent/'../../../specs/torchair-ge-ir.pb'):
        if path.is_file():
            return path.read_bytes()
    from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef
    return ModelDef.DESCRIPTOR.file.serialized_pb


def build(args, schema):
    from google.protobuf import descriptor_pool, message_factory

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(schema)
    desc = pool.FindMessageTypeByName('ge.proto.ModelDef')
    cls = (message_factory.GetMessageClass(desc) if hasattr(message_factory, 'GetMessageClass')
           else message_factory.MessageFactory(pool).GetPrototype(desc))
    model = cls(name='l0_input_rmsnorm', version=0)
    for key in ('weight_size', 'memory_size', 'stream_num', 'event_num', 'p2p_memory_size', 'label_num'):
        model.attr[key].i = 0
    model.attr['target_type'].s = b'MINI'
    enum = desc.file.enum_types_by_name['DataType'].values_by_name
    dtype = {'float16': enum['DT_FLOAT16'].number, 'float32': enum['DT_FLOAT'].number}
    x_type = dtype[args.dtype]
    gamma_type = dtype[args.gamma_dtype or args.dtype]
    graph = model.graph.add(name='l0_input_rmsnorm')
    graph.attr['_executor_type'].i = 1

    def port(op, direction, name, shape, code):
        ports = getattr(op, direction + '_desc')
        index = len(ports)
        desc = ports.add(name=name, dtype=code, layout='ND', device_type='NPU')
        desc.shape.dim.extend(shape)
        keys = op.attr[f'_{direction}_name_key'].list
        keys.val_type = keys.VT_LIST_STRING
        keys.s.append(name.encode())
        slots = op.attr[f'_{direction}_name_value'].list
        slots.val_type = slots.VT_LIST_INT
        slots.i.append(index)

    shape = (1, 64, 2048)
    inputs = [('x', shape, x_type), ('gamma', (2048,), gamma_type)]
    for i, (name, dims, code) in enumerate(inputs):
        data = graph.op.add(name=name, type='Data', id=i)
        data.attr['index'].i = i
        data.input.append('')
        port(data, 'input', 'x', dims, code)
        port(data, 'output', 'y', dims, code)
        graph.input.append(name + ':0')

    norm = graph.op.add(name='norm', type=args.op_type, id=2)
    for (source, dims, code), name in zip(inputs, args.input_ports):
        norm.input.append(source + ':0')
        port(norm, 'input', name, dims, code)
    if args.epsilon_attr:
        norm.attr[args.epsilon_attr].f = args.epsilon
    port(norm, 'output', args.output_ports[0], shape, x_type)
    if len(args.output_ports) == 2:
        port(norm, 'output', args.output_ports[1], (1, 64, 1), dtype['float32'])

    net = graph.op.add(name='output', type='NetOutput', id=3)
    net.input.append('norm:0')
    port(net, 'input', 'input0', shape, x_type)
    graph.output.append('norm:0')
    return model


def package(case, info, dest, raw, schema, args):
    import numpy as np

    if dest.resolve() == case:
        raise ValueError('Choose a bundle directory separate from the source case to preserve its original gamma.bin')
    dest.mkdir(parents=True, exist_ok=True)
    feeds = [('x', info['captured'], (1, 64, 2048), args.dtype),
             ('gamma', info['real_inputs']['gamma'], (2048,), args.gamma_dtype or args.dtype)]
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        stage = Path(tmp)
        for name, spec, shape, dtype in feeds:
            source = case / spec['path']
            dims = tuple(spec['shape'])
            layout_matches = dims == shape if name == 'x' else tuple(d for d in dims if d != 1) == (2048,)
            if spec['dtype'] not in ('float16', 'float32') or not layout_matches:
                raise ValueError(f'{source}: unexpected input descriptor {spec}')
            source_dtype = np.dtype(spec['dtype']).newbyteorder('<')
            if source.stat().st_size != math.prod(shape) * source_dtype.itemsize:
                raise ValueError(f'{source}: input byte count differs from descriptor')
            target = np.fromfile(source, dtype=source_dtype).reshape(shape).astype(np.dtype(dtype).newbyteorder('<'))
            target.tofile(stage/(name + '.bin'))
        shutil.copyfile(__file__, stage/'export_l0_rmsnorm_air.py')
        (stage/'torchair-ge-ir.pb').write_bytes(schema)
        (stage/'export.air').write_bytes(raw)
        # Serialize repeat exports to the same directory; publish the model last.
        handle = os.open(dest, os.O_RDONLY)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            for name in ('x.bin', 'gamma.bin', 'export_l0_rmsnorm_air.py', 'torchair-ge-ir.pb', 'export.air'):
                (stage/name).replace(dest/name)
        finally:
            os.close(handle)
    for name, _, shape, dtype in feeds:
        path = dest/(name + '.bin')
        print(f'{path}: {dtype} {list(shape)}, {path.stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument('--output', type=Path, help='write just this AIR file')
    outputs.add_argument('--output-dir', type=Path, help='directory for the model, exporter and actual inputs')
    parser.add_argument('--case-dir', type=Path, help='L0 input RMSNorm capture directory with case.json')
    parser.add_argument('--op-type', default='RmsNorm', help='registered GE operator type')
    parser.add_argument('--input-ports', nargs=2, default=['x', 'gamma'], metavar=('X', 'GAMMA'),
                        help='registered activation and scale port names, in that order')
    parser.add_argument('--output-ports', nargs='+', default=['y', 'rstd'], metavar='NAME',
                        help='normalized output, optionally followed by FP32 reciprocal RMS; only slot 0 is returned')
    parser.add_argument('--epsilon-attr', default='epsilon',
                        help='registered floating epsilon attribute; pass an empty string to omit it')
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16',
                        help='activation dtype (default: float16, as captured on 310P)')
    parser.add_argument('--gamma-dtype', choices=('float16', 'float32'),
                        help='effective gamma dtype (defaults to --dtype)')
    parser.add_argument('--epsilon', type=float, help='defaults to the captured epsilon, or 1e-6 if absent')
    args = parser.parse_args()
    if (not args.op_type or args.op_type in ('Data', 'NetOutput') or len(args.output_ports) > 2
            or len(set(args.input_ports)) != 2 or len(set(args.output_ports)) != len(args.output_ports)
            or not all(args.input_ports + args.output_ports)):
        parser.error('expected one compute type, two distinct input ports, and one or two distinct output ports')
    if args.op_type == 'RmsNorm' and (args.input_ports != ['x', 'gamma'] or args.output_ports != ['y', 'rstd']
                                       or args.epsilon_attr != 'epsilon'):
        parser.error('the built-in RmsNorm uses x, gamma, y, rstd and epsilon')
    if args.output and args.case_dir:
        parser.error('use --output-dir with --case-dir to package the inputs')
    case = args.case_dir
    if args.output_dir and case is None:
        root = os.environ.get('AI_RUN_DIR')
        if not root:
            parser.error('set --case-dir or AI_RUN_DIR to locate the measured inputs')
        case = Path(root)/'target/tests/4-prefill-t64/board/L00-L03/l0-input-rmsnorm'
    if case is not None:
        case = case.expanduser().resolve(strict=True)
        info = json.loads((case/'case.json').read_bytes())
        if info.get('kind') != 'l0-input-rmsnorm':
            parser.error(f'{case}: expected the L0 RMSNorm capture')
        if args.epsilon is None:
            args.epsilon = info['norm'].get('epsilon')
    if args.epsilon is None:
        args.epsilon = 1e-6
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error('--epsilon must be finite and positive')

    schema = schema_bytes()
    model = build(args, schema)
    raw = model.SerializeToString()
    parsed = type(model).FromString(raw)
    graph, = parsed.graph
    if ([op.type for op in graph.op] != ['Data', 'Data', args.op_type, 'NetOutput']
            or list(graph.op[2].input) != ['x:0', 'gamma:0']
            or list(graph.op[3].input) != ['norm:0'] or list(graph.output) != ['norm:0']):
        raise ValueError('expected two Data ports, one RMSNorm calculation and its slot 0 as the public output')
    if case is not None:
        dest = (args.output_dir or case/'single-op').expanduser().absolute()
        package(case, info, dest, raw, schema, args)
        output = dest/'export.air'
    else:
        output = (args.output or Path('l0_input_rmsnorm.air')).expanduser().absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            air = Path(tmp) / 'export.air'
            air.write_bytes(raw)
            air.replace(output)
    print(f'{output}\ncompute_ops=1  op_type={args.op_type}'
          f'\nx -> {args.input_ports[0]} [1,64,2048] {args.dtype}'
          f'\ngamma -> {args.input_ports[1]} [2048] {args.gamma_dtype or args.dtype}'
          f'\npublic output: {args.output_ports[0]} [1,64,2048] {args.dtype}')
    if args.epsilon_attr:
        print(f'{args.epsilon_attr}={args.epsilon:g}')


if __name__ == '__main__':
    main()

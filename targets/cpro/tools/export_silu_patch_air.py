#!/usr/bin/env python3
"""Generate a standalone ND FP16 activation AIR for SiluPatch or SoftplusPatch."""
import argparse
from math import prod
import os
from pathlib import Path
import sys
import tempfile

from google.protobuf import descriptor_pool, message_factory

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'model'))
from qwen35_moe_dflash.target.silu_patch_air import GE_TYPE, nd_desc, port_names

OP_TYPES = {'silu_patch': GE_TYPE, 'softplus_patch': 'SoftplusPatch'}


def build_air(numel=None, *, shape=None, op='silu_patch'):
    op_type = OP_TYPES[op]
    if numel is not None and numel < 1:
        raise ValueError('numel must be positive')
    if shape is None:
        dims = [-1 if numel is None else numel]
    else:
        dims = list(shape)
        if not dims or any(d < 1 for d in dims) or (numel is not None and prod(dims) != numel):
            raise ValueError(f'inconsistent static shape {dims} and numel {numel}')
    pool = descriptor_pool.DescriptorPool()
    schema = pool.AddSerializedFile((REPO/'specs/torchair-ge-ir.pb').read_bytes())
    desc = schema.message_types_by_name['ModelDef']
    cls = (message_factory.GetMessageClass(desc) if hasattr(message_factory, 'GetMessageClass')
           else message_factory.MessageFactory(pool).GetPrototype(desc))
    model = cls(name=op, version=0)
    for key in ('weight_size', 'memory_size', 'stream_num', 'event_num', 'p2p_memory_size', 'label_num'):
        model.attr[key].i = 0
    model.attr['target_type'].s = b'MINI'
    graph = model.graph.add(name=op)
    graph.attr['_executor_type'].i = 1
    data = graph.op.add(name='x', type='Data', id=0)
    data.input.append('')
    data.attr['index'].i = 0
    port_names(data, 'x', 'y')
    graph.input.append('x:0')
    activation = graph.op.add(name=op, type=op_type, id=1)
    activation.input.append('x:0')
    port_names(activation, 'x', 'z')
    net = graph.op.add(name='NetOutput', type='NetOutput', id=2)
    net.input.append(f'{op}:0')
    graph.output.append(f'{op}:0')

    types = schema.enum_types_by_name['DataType'].values_by_name
    for ports in (data.input_desc, data.output_desc, activation.input_desc, activation.output_desc, net.input_desc):
        nd_desc(ports.add(dtype=types['DT_FLOAT16'].number), dims)
    return model.SerializeToString(deterministic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='output directory')
    parser.add_argument('--op', choices=OP_TYPES, default='silu_patch')
    parser.add_argument('--shape', help='static ND dimensions, e.g. 1,64,32')
    parser.add_argument('--numel', type=int, help='fix N for a static OMC test (default: dynamic N)')
    args = parser.parse_args()
    if args.numel is not None and args.numel <= 0:
        parser.error('--numel must be positive')
    shape = None
    if args.shape is not None:
        try:
            shape = tuple(int(d) for d in args.shape.split(','))
        except ValueError:
            parser.error('--shape expects comma-separated positive integers')
        if min(shape) < 1 or (args.numel is not None and prod(shape) != args.numel):
            parser.error('--shape must be positive and agree with --numel')
    elif args.op == 'softplus_patch' and args.numel is None:
        shape = (1, 64, 32)
    folder = args.output
    if folder is None:
        if not os.getenv('AI_RUN_DIR'):
            parser.error('set --output or AI_RUN_DIR')
        folder = Path(os.environ['AI_RUN_DIR'])/'target/tests'/f'{args.op}_fp16_fp16'
        if shape:
            folder = folder/('x'.join(map(str, shape)))
        elif args.numel is not None:
            folder = folder/f'n{args.numel}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{args.op}.air'
    raw = build_air(args.numel, shape=shape, op=args.op)
    # Publish complete bytes. Repeating the same command can reuse its AIR.
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        staged = Path(tmp)/path.name
        staged.write_bytes(raw)
        try:
            os.link(staged, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                parser.error(f'{path} already contains a different AIR; select a new --output directory')
    dims = ','.join(map(str, shape)) if shape else str(args.numel or 'N')
    print(f'{path.resolve()}  {OP_TYPES[args.op]}: x FP16[{dims}] -> z FP16[{dims}]')


if __name__ == '__main__':
    main()

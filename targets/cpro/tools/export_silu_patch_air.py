#!/usr/bin/env python3
"""Write one ND vector AIR for the GE SiluPatch(x: FP16[N]) -> z: FP32[N] kernel."""
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

NODE_NAME = 'silu_patch'


def build_air(numel=None, *, shape=None):
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
    model = cls(name=NODE_NAME, version=0)
    for key in ('weight_size', 'memory_size', 'stream_num', 'event_num', 'p2p_memory_size', 'label_num'):
        model.attr[key].i = 0
    model.attr['target_type'].s = b'MINI'
    graph = model.graph.add(name=NODE_NAME)
    graph.attr['_executor_type'].i = 1
    data = graph.op.add(name='x', type='Data', id=0)
    data.input.append('')
    data.attr['index'].i = 0
    port_names(data, 'x', 'y')
    graph.input.append('x:0')
    op = graph.op.add(name=NODE_NAME, type=GE_TYPE, id=1)
    op.input.append('x:0')
    port_names(op, 'x', 'z')
    net = graph.op.add(name='NetOutput', type='NetOutput', id=2)
    net.input.append('silu_patch:0')
    graph.output.append('silu_patch:0')

    types = schema.enum_types_by_name['DataType'].values_by_name
    fp16, fp32 = types['DT_FLOAT16'].number, types['DT_FLOAT'].number
    for ports, dtype in [(data.input_desc, fp16), (data.output_desc, fp16),
                         (op.input_desc, fp16), (op.output_desc, fp32), (net.input_desc, fp32)]:
        nd_desc(ports.add(dtype=dtype), dims)
    return model.SerializeToString(deterministic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='output directory')
    parser.add_argument('--numel', type=int, help='fix N for a static OMC test (default: dynamic N)')
    args = parser.parse_args()
    if args.numel is not None and args.numel <= 0:
        parser.error('--numel must be positive')
    folder = args.output
    if folder is None:
        if not os.getenv('AI_RUN_DIR'):
            parser.error('set --output or AI_RUN_DIR')
        folder = Path(os.environ['AI_RUN_DIR'])/'target/tests/silu_patch_fp16_fp32'
        if args.numel is not None:
            folder = folder/f'n{args.numel}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'silu_patch.air'
    raw = build_air(args.numel)
    # Publish complete bytes. Repeating the same command can reuse its AIR.
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        staged = Path(tmp)/path.name
        staged.write_bytes(raw)
        try:
            os.link(staged, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                parser.error(f'{path} already contains a different AIR; select a new --output directory')
    length = 'N' if args.numel is None else args.numel
    print(f'{path.resolve()}  {GE_TYPE}: x FP16[{length}] -> z FP32[{length}]')


if __name__ == '__main__':
    main()

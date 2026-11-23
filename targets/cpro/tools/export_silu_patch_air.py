#!/usr/bin/env python3
"""Write one GE AIR for silu_patch(x: FP16[N]) -> y: FP32[N]."""
import argparse
import os
from pathlib import Path
import tempfile

from google.protobuf import descriptor_pool, message_factory

REPO = Path(__file__).resolve().parents[3]
OP_TYPE = 'silu_patch'


def build_air(numel=None):
    if numel is not None and numel < 1:
        raise ValueError('numel must be positive')
    length = -1 if numel is None else numel
    pool = descriptor_pool.DescriptorPool()
    schema = pool.AddSerializedFile((REPO/'specs/torchair-ge-ir.pb').read_bytes())
    desc = schema.message_types_by_name['ModelDef']
    cls = (message_factory.GetMessageClass(desc) if hasattr(message_factory, 'GetMessageClass')
           else message_factory.MessageFactory(pool).GetPrototype(desc))
    model = cls(name=OP_TYPE)
    graph = model.graph.add(name=OP_TYPE)
    data = graph.op.add(name='x', type='Data', id=0)
    data.attr['index'].i = 0
    graph.input.append('x:0')
    op = graph.op.add(name=OP_TYPE, type=OP_TYPE, id=1)
    op.input.append('x:0')
    net = graph.op.add(name='NetOutput', type='NetOutput', id=2)
    net.input.append('silu_patch:0')
    graph.output.append('silu_patch:0')

    types = schema.enum_types_by_name['DataType'].values_by_name
    fp16, fp32 = types['DT_FLOAT16'].number, types['DT_FLOAT'].number
    for ports, port_name, dtype in [(data.input_desc, 'x', fp16), (data.output_desc, 'y', fp16),
                                    (op.input_desc, 'x', fp16), (op.output_desc, 'y', fp32),
                                    (net.input_desc, 'input0', fp32)]:
        port = ports.add(name=port_name, dtype=dtype, layout='ND', device_type='NPU')
        port.shape.dim.append(length)
    for direction, port_name in [('input', 'x'), ('output', 'y')]:
        keys = op.attr[f'_{direction}_name_key'].list
        slots = op.attr[f'_{direction}_name_value'].list
        keys.val_type = keys.VT_LIST_STRING
        slots.val_type = slots.VT_LIST_INT
        keys.s.append(port_name.encode())
        slots.i.append(0)
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
    print(f'{path.resolve()}  x: FP16[{length}] -> y: FP32[{length}]')


if __name__ == '__main__':
    main()

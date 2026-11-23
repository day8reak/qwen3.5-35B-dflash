#!/usr/bin/env python3
"""Export standalone unary SiLU AIRs for the B=1 CPro quantized text decoder."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from google.protobuf import descriptor_pool, message_factory

REPO = Path(__file__).resolve().parents[3]


def contracts(config):
    k = config['linear_num_key_heads'] * config['linear_key_head_dim']
    heads, width = config['linear_num_value_heads'], config['linear_value_head_dim']
    linear = config['layer_types'].count('linear_attention')
    layers = len(config['layer_types'])
    if config['hidden_act'] != 'silu':
        raise ValueError('Shared expert does not use SiLU in this config')
    for phase, tokens in [('prefill', 64), ('decode', 1)]:
        yield phase, 'gdn_conv', 'float16', [1, 2*k + heads*width, tokens], linear
        yield phase, 'gdn_gate', 'float32', [tokens*heads, width], linear
        yield phase, 'shared_expert', 'float16', [tokens, config['shared_expert_intermediate_size']], layers
        yield phase, 'routed_experts_w4', 'float16', [tokens*config['num_experts_per_tok'],
                                                       config['moe_intermediate_size']], layers


def build(schema, model_cls, name, dtype, shape):
    model = model_cls(name=name)
    graph = model.graph.add(name=name)
    data = graph.op.add(name='x', type='Data', id=0)
    data.attr['index'].i = 0
    graph.input.append('x:0')
    op = graph.op.add(name='silu_patch', type='silu_patch', id=1)
    op.input.append('x:0')
    net = graph.op.add(name='NetOutput', type='NetOutput', id=2)
    net.input.append('silu_patch:0')
    graph.output.append('silu_patch:0')
    enum = 'DT_FLOAT16' if dtype == 'float16' else 'DT_FLOAT'
    code = schema.enum_types_by_name['DataType'].values_by_name[enum].number
    for ports, port_name in [(data.input_desc, 'x'), (data.output_desc, 'y'), (op.input_desc, 'x'),
                             (op.output_desc, 'y'), (net.input_desc, 'input0')]:
        desc = ports.add(name=port_name, dtype=code, layout='ND', device_type='NPU')
        desc.shape.dim.extend(shape)
    for direction, port in [('input', 'x'), ('output', 'y')]:
        keys = op.attr[f'_{direction}_name_key'].list
        values = op.attr[f'_{direction}_name_value'].list
        keys.val_type = keys.VT_LIST_STRING
        values.val_type = values.VT_LIST_INT
        keys.s.append(port.encode())
        values.i.append(0)
    return model.SerializeToString(deterministic=True)


def export(output):
    pool = descriptor_pool.DescriptorPool()
    schema = pool.AddSerializedFile((REPO/'specs/torchair-ge-ir.pb').read_bytes())
    desc = schema.message_types_by_name['ModelDef']
    cls = (message_factory.GetMessageClass(desc) if hasattr(message_factory, 'GetMessageClass')
           else message_factory.MessageFactory(pool).GetPrototype(desc))
    config = json.loads((REPO/'specs/target-config.json').read_text())['text_config']
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Choose a new --output or reuse the existing AIRs: {output}')
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        ready = Path(tmp)/'air'
        ready.mkdir()
        rows = []
        for phase, site, dtype, shape, calls in contracts(config):
            name = 'silu_patch_' + phase + '_' + site
            raw = build(schema, cls, name, dtype, shape)
            filename = name + '.air'
            (ready/filename).write_bytes(raw)
            rows.append(dict(phase=phase, site=site, ge_type='silu_patch', input='x', output='y',
                             input_dtype=dtype, output_dtype=dtype, shape_x_and_y=json.dumps(shape),
                             layout='ND', bytes_each=math.prod(shape)*(2 if dtype == 'float16' else 4),
                             source_calls_per_forward=calls, air=filename, sha256=hashlib.sha256(raw).hexdigest()))
        with (ready/'contracts.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        ready.rename(output)
    print(f'{len(rows)} standalone silu_patch(x) -> y AIRs: {output}')
    print(output/'contracts.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new output directory (default: $AI_RUN_DIR/target/tests/silu_patch)')
    args = parser.parse_args()
    if args.output is None and not os.getenv('AI_RUN_DIR'):
        parser.error('set --output or AI_RUN_DIR')
    output = args.output or Path(os.environ['AI_RUN_DIR'])/'target/tests/silu_patch'
    export(output.resolve())


if __name__ == '__main__':
    main()

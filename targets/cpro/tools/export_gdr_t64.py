#!/usr/bin/env python3
"""Write a static seven-input CPro GDR AIR and matching raw input tensors."""
import argparse
import json
import tempfile
from pathlib import Path
import sys

import numpy as np
from google.protobuf import descriptor_pool, message_factory

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'model'))
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
from qwen35_moe_dflash.target.gdn_probe import fixture
from qwen35_moe_dflash.target.layer_probe.common import identity, read_array, save_json
from qwen35_moe_dflash.target.prefill_t64.gdr_math import compute
from qwen35_moe_dflash.target.prefill_t64.gdr_stage_data import GDR, TOKEN_OPERANDS, logical, shapes

INPUTS = GDR[:-2]
OUTPUTS = ('core_attn', 'last_recurrent_state')


def build_air():
    pool = descriptor_pool.DescriptorPool()
    schema = pool.AddSerializedFile((REPO / 'specs/torchair-ge-ir.pb').read_bytes())
    message = schema.message_types_by_name['ModelDef']
    cls = (message_factory.GetMessageClass(message) if hasattr(message_factory, 'GetMessageClass')
           else message_factory.MessageFactory(pool).GetPrototype(message))
    model = cls(name='gdr_t64')
    graph = model.graph.add(name='gdr_t64')
    types = schema.enum_types_by_name['DataType'].values_by_name

    def descriptor(port, role):
        enum = 'DT_INT16' if role == 'effective_length' else 'DT_FLOAT16'
        port.dtype = types[enum].number
        port.layout = 'ND'
        port.device_type = 'NPU'
        port.shape.dim.extend(shapes(role)[0])

    for i, name in enumerate(INPUTS):
        data = graph.op.add(name=name, type='Data', id=i)
        data.attr['index'].i = i
        descriptor(data.input_desc.add(name='x'), name)
        descriptor(data.output_desc.add(name='y'), name)
        graph.input.append(name + ':0')

    op = graph.op.add(name='gdr', type='ChunkGatedDeltaRule', id=len(INPUTS))
    op.attr['chunk_size'].i = 64
    op.attr['output_final_state'].i = 1
    op.attr['use_qk_l2norm_in_kernel'].i = 1
    for direction, names in [('input', INPUTS), ('output', OUTPUTS)]:
        keys = op.attr[f'_{direction}_name_key'].list
        slots = op.attr[f'_{direction}_name_value'].list
        keys.val_type = keys.VT_LIST_STRING
        slots.val_type = slots.VT_LIST_INT
        keys.s.extend(name.encode() for name in names)
        slots.i.extend(range(len(names)))
    for name in INPUTS:
        op.input.append(name + ':0')
        descriptor(op.input_desc.add(name=name), name)
    net = graph.op.add(name='NetOutput', type='NetOutput', id=len(INPUTS) + 1)
    for i, (port, role) in enumerate(zip(OUTPUTS, GDR[-2:])):
        descriptor(op.output_desc.add(name=port), role)
        net.input.append(f'gdr:{i}')
        descriptor(net.input_desc.add(name=f'input{i}'), role)
    graph.output.extend(net.input)
    return model.SerializeToString(deterministic=True)


def generate(args):
    root = output_path(args.output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f'Choose an empty output directory: {root}')
    if args.cpro_dump:
        manifest = json.loads((args.cpro_dump / 'manifest.json').read_text())
        layer = manifest['layers']['0']
        if manifest['physical_rows'] != 64 or manifest['chunk_size'] != 64:
            raise ValueError('Expected a T64 chunk64 L0 dump')
        specs = layer['tensors']
        values = {n: logical(read_array(args.cpro_dump, specs[n]['cpro']), n) for n in INPUTS}
        source = dict(kind='measured_l0_cpro', manifest=identity(args.cpro_dump / 'manifest.json', args.cpro_dump))
    else:
        values = dict(zip(INPUTS[:-1], fixture(64, nonzero_state=False)))
        if args.effective_length_bin:
            if args.effective_length_bin.stat().st_size != 2:
                raise ValueError('effective_length.bin must be exactly one little-endian INT16')
            length = np.fromfile(args.effective_length_bin, dtype='<i2')
        else:
            count = args.effective_length if args.effective_length is not None else 19
            if not 0 <= count <= 64:
                raise ValueError('effective length must be in [0, 64]')
            length = np.array([count], dtype='<i2')
        values['effective_length'] = length
        source = dict(kind='synthetic', seed=35064)
    for name, tensor in values.items():
        dtype = np.dtype('<i2' if name == 'effective_length' else '<f2')
        if list(tensor.shape) != shapes(name)[0] or tensor.dtype != dtype:
            raise ValueError(f'{name}: expected {dtype} {shapes(name)[0]}, got {tensor.dtype} {tensor.shape}')

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        stage = Path(temporary) / root.name
        (stage / 'inputs').mkdir(parents=True)
        (stage / 'cpu').mkdir()
        air = stage / 'gdr_t64.air'
        air.write_bytes(build_air())

        def write_tensor(folder, name, tensor):
            path = stage / folder / (name + '.bin')
            tensor.tofile(path)
            return dict(name=name, shape=list(tensor.shape), dtype=tensor.dtype.str, **identity(path, stage))

        inputs = [dict(air_index=i, **write_tensor('inputs', n, values[n])) for i, n in enumerate(INPUTS)]
        # The runtime length comes from the raw input, including for the CPU reference.
        valid = int(np.fromfile(stage / 'inputs/effective_length.bin', dtype='<i2').item())
        if not 0 <= valid <= 64:
            raise ValueError(f'effective_length.bin contains {valid}, expected 0..64')
        if valid:
            core, state = compute(*(values[n][:, :valid] for n in TOKEN_OPERANDS), values['initial_state'])
        else:
            core, state = values['value'][:, :0], values['initial_state']
        cpu = [write_tensor('cpu', 'core_valid', core),
               write_tensor('cpu', 'last_recurrent_state', state.astype('<f2'))]
        outputs = [dict(name=n, air_index=i, dtype='<f2', shape=shapes(role)[0])
                   for i, (n, role) in enumerate(zip(OUTPUTS, GDR[-2:]))]
        save_json(stage / 'manifest.json', dict(operator='ChunkGatedDeltaRule', physical_rows=64,
                  chunk_size=64, effective_length=valid, source=source, air=identity(air, stage),
                  inputs=inputs, outputs=outputs, cpu_golden=cpu))
        share_for_transfer(stage)
        stage.rename(root)
    print(f'{root / air.name}  ·  7 inputs  ·  2 outputs  ·  T64  ·  effective_length={valid}')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True, type=Path, help='empty directory under AI_RUN_DIR')
    group = cli.add_mutually_exclusive_group()
    group.add_argument('--cpro-dump', type=Path, help='use all seven inputs from an L0 stage dump')
    group.add_argument('--effective-length-bin', type=Path, help='runtime length input for the synthetic case')
    group.add_argument('--effective-length', type=int, help='generate the synthetic runtime length input (default 19)')
    args = cli.parse_args()
    try:
        generate(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        cli.exit(1, f'gdr-t64: {exc}\n')

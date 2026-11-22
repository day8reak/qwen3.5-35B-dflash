#!/usr/bin/env python3
"""Export a standalone T64 CPro GDR case using L0 operands from the real 310P prefill."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys

import numpy as np
from google.protobuf import descriptor_pool, message_factory

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'model'))
from qwen35_moe_dflash.common.artifacts import output_path, share_for_transfer
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


def reference_inputs(root_ref):
    manifest = json.loads((root_ref / 'manifest.json').read_text())
    record = manifest['gdr_stages']['0']
    attrs = record['attrs']
    if (manifest['status'] != 'CAPTURED' or manifest['physical_rows'] != 64
            or attrs['chunk_size'] != 64 or not attrs['output_final_state'] or not attrs['use_qk_l2norm_in_kernel']):
        raise ValueError('expected a completed 310P L0 T64 GDR stage capture')
    ids = read_array(root_ref, manifest['inputs']['input_ids'])
    values = {n: logical(read_array(root_ref, record['tensors'][n]), n) for n in INPUTS}
    if (hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest() != manifest['input_ids_i32_sha256']
            or values['effective_length'].tolist() != [manifest['valid_rows']]):
        raise ValueError('310P input_ids or GDR length differs from the stage manifest')
    source = dict(kind='measured_l0_310p', input_ids_source='310p_capture',
                  manifest=identity(root_ref / 'manifest.json', root_ref))
    return values, ids, source


def live_inputs(args, length):
    from capture_gdr_inputs import capture
    requested = None
    if args.input_ids:
        dtype = np.dtype({'int32': '<i4', 'int64': '<i8'}[args.ids_dtype])
        size = args.input_ids.stat().st_size
        if size % dtype.itemsize or not 1 <= size // dtype.itemsize <= 64:
            raise ValueError('input_ids must contain 1..64 complete integer tokens')
        requested = np.fromfile(args.input_ids, dtype=dtype)
        if length is None and requested.size == 64:
            raise ValueError('64 physical IDs: supply --effective-length-bin or --effective-length (64 if all are valid)')
        if length is not None and (length < 1 or requested.size not in (length, 64)):
            raise ValueError('provide the valid prefix or all 64 physical IDs with a matching length')
    return capture(args.source_root, args.config, requested, length)


def generate(args):
    root = output_path(args.output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f'Choose an empty output directory: {root}')
    ids = None
    originals = {}
    length = args.effective_length
    if args.effective_length_bin:
        if args.effective_length_bin.stat().st_size != 2:
            raise ValueError('effective_length.bin must contain one little-endian INT16')
        length = int(np.fromfile(args.effective_length_bin, dtype='<i2').item())
    if length is not None and not 0 <= length <= 64:
        raise ValueError('effective length must be in [0, 64]')

    if args.cpro_dump:
        manifest = json.loads((args.cpro_dump / 'manifest.json').read_text())
        layer = manifest['layers']['0']
        if manifest['physical_rows'] != 64 or manifest['chunk_size'] != 64:
            raise ValueError('Expected a T64 chunk64 L0 dump')
        specs = layer['tensors']
        values = {n: logical(read_array(args.cpro_dump, specs[n]['cpro']), n) for n in INPUTS}
        source = dict(kind='measured_l0_cpro', manifest=identity(args.cpro_dump / 'manifest.json', args.cpro_dump))
    elif args.synthetic:
        from qwen35_moe_dflash.target.gdn_probe import fixture
        values = dict(zip(INPUTS[:-1], fixture(64, nonzero_state=False)))
        values['effective_length'] = np.array([19 if length is None else length], dtype='<i2')
        source = dict(kind='synthetic', seed=35064)
    else:
        if args.reference:
            values, ids, source = reference_inputs(args.reference)
        else:
            values, ids, source = live_inputs(args, length)
        if ids.shape != (1, 64) or ids.dtype not in (np.int32, np.int64):
            raise ValueError('expected the actual padded integer input_ids at the 310P wrapper')
        # These ports enter the original 310P call in FP32. Preserve the raw
        # measurement as well as the FP16 buffer the CPro AIR actually accepts.
        for name in ('g', 'initial_state'):
            if values[name].dtype == np.float32:
                originals[name] = values[name]
                values[name] = values[name].astype('<f2')
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

        if ids is not None:
            source['input_ids'] = write_tensor('', 'input_ids', ids)
            source['input_ids_i32_sha256'] = hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest()
        if originals:
            (stage / 'source_310p').mkdir()
            source['fp32_before_cpro_cast'] = [write_tensor('source_310p', n, x) for n, x in originals.items()]
        inputs = [dict(air_index=i, **write_tensor('inputs', n, values[n])) for i, n in enumerate(INPUTS)]
        # The runtime length comes from the raw input, including for the CPU reference.
        valid = int(np.fromfile(stage / 'inputs/effective_length.bin', dtype='<i2').item())
        if length is not None and valid != length:
            raise ValueError(f'captured effective length {valid} differs from requested {length}')
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
    if ids is not None:
        print(f'310P padded input_ids SHA256 (int32): {source["input_ids_i32_sha256"]}')
    print(f'{root / air.name}  ·  7 inputs  ·  2 outputs  ·  T64  ·  effective_length={valid}  ·  {source["kind"]}')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True, type=Path, help='empty directory under AI_RUN_DIR')
    group = cli.add_mutually_exclusive_group()
    group.add_argument('--cpro-dump', type=Path, help='use all seven inputs from an L0 stage dump')
    group.add_argument('--source-root', type=Path, help='original 310P source with inference_segment.py (default: TARGET_SOURCE)')
    group.add_argument('--reference', type=Path, help='reuse a measured check_segment_prefill.py --gdr-stages capture')
    group.add_argument('--synthetic', action='store_true', help='explicitly generate deterministic synthetic operands')
    cli.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'), help='310P driver YAML')
    cli.add_argument('--input-ids', type=Path, help='use this raw little-endian token file in the live model (default: driver prompt)')
    cli.add_argument('--ids-dtype', choices=('int32', 'int64'), help='element type of the supplied raw token file')
    sizes = cli.add_mutually_exclusive_group()
    sizes.add_argument('--effective-length-bin', type=Path, help='one little-endian INT16; supply for already padded file inputs')
    sizes.add_argument('--effective-length', type=int, help='valid count for file inputs, or expected length for a captured prompt')
    args = cli.parse_args()
    if bool(args.input_ids) != bool(args.ids_dtype):
        cli.error('supply --input-ids and --ids-dtype together')
    if args.input_ids and (args.cpro_dump or args.synthetic or args.reference):
        cli.error('--input-ids drives the live 310P source')
    if not (args.source_root or args.synthetic or args.cpro_dump or args.reference):
        if not os.environ.get('TARGET_SOURCE'):
            cli.error('set TARGET_SOURCE, pass --source-root or --reference, or explicitly request --synthetic')
        args.source_root = Path(os.environ['TARGET_SOURCE'])
    try:
        generate(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        cli.exit(1, f'gdr-t64: {exc}\n')

"""Specialize the retained L0 64/19 AIR for nineteen state-linked S=1 calls."""
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import checked, identity, read_array, save_json, sha
from .decode_cache_control import graph_message
from .decode_cache_trace import _record
from .gdr_stage_data import BASIS, TOKEN_OPERANDS, case_id, shapes
from .segment_export import NATIVE_GDR_MODULE_BASIS

BASE = 'l00_c00_gdn_layer_hidden_o0'
GDR_CASE = case_id(0, 'gdr')
PAIR_CASE = case_id(0, 'norm_silu_pair')
SPLIT_ROLES = ('rms_input', 'gamma', 'rms_output', 'gate', 'mul_output', 'gated_return')
GROUPS = {'gamma-readout': (BASE, GDR_CASE),
          'norm-single-l0': tuple(case_id(0, 'norm_' + r) for r in SPLIT_ROLES),
          'l0-silu-pair': (PAIR_CASE,)}
MODE = 'S1_VALID19_STATE_CHAIN'


def unroll(raw, length_edge):
    msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
    original, = [n for n in graph.op if n.type == 'ChunkGatedDeltaRule']
    nodes = {n.name: n for n in graph.op}
    names = original.attr['_input_name_key'].list.s
    ports = dict(zip((x.decode() for x in names), original.attr['_input_name_value'].list.i, strict=True))
    flags = (original.attr[k] for k in ('output_final_state', 'use_qk_l2norm_in_kernel'))
    if (original.attr['chunk_size'].i != 64 or not all(v.b or v.i == 1 for v in flags)
            or original.input[ports['effective_length']] != length_edge
            or len(original.output_desc) != 2 or len(nodes) != len(graph.op)):
        raise ValueError('Expected one unscaled chunk64 GDR reading the original runtime length')
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    half, i16, i32 = (enum[n].number for n in ('DT_FLOAT16', 'DT_INT16', 'DT_INT32'))
    for role in (*TOKEN_OPERANDS, 'effective_length', 'initial_state'):
        d = original.input_desc[ports[role]]
        if d.dtype != (i16 if role == 'effective_length' else half) or (d.shape.dim and list(d.shape.dim) != shapes(role)[0]):
            raise ValueError(f'Unexpected GDR input descriptor for {role}')
    if any(d.dtype != half for d in original.output_desc):
        raise ValueError('Expected FP16 core and FP16 feedback state from the original CPro GDR')

    added = []
    first_id = max(n.id for n in graph.op) + 1

    def desc(dtype, dims):
        d = type(original.output_desc[0])(dtype=dtype, layout='ND', device_type='NPU')
        d.shape.SetInParent()
        d.shape.dim.extend(dims)
        return d

    def add(label, kind, inputs, out):
        name = 'l0s1_' + label
        if name in nodes:
            raise ValueError('Duplicate GE node ' + name)
        node = type(original)(name=name, type=kind, id=first_id + len(added))
        for edge in inputs:
            producer, port = edge.rsplit(':', 1)
            node.input.append(edge)
            node.input_desc.add().CopyFrom(nodes[producer].output_desc[int(port)])
        node.output_desc.extend(out)
        nodes[name] = node
        added.append(node)
        return node

    def constant(label, value, dtype):
        arr = np.asarray(value, dtype={half: '<f2', i16: '<i2', i32: '<i4'}[dtype])
        node = add(label, 'Const', [], [desc(dtype, arr.shape)])
        tensor = node.attr['value'].t
        tensor.desc.CopyFrom(node.output_desc[0])
        tensor.data = arr.tobytes()
        return node.name + ':0'

    def unary(label, kind, edges, dtype, dims, **attrs):
        node = add(label, kind, edges, [desc(dtype, dims)])
        for key, val in attrs.items():
            node.attr[key].i = val
        return node.name + ':0'

    one = constant('one', [1], i16)
    consumed = constant('consumed', [18], i32)
    # Nineteen is the checked fixture length. Preserve its runtime input in the
    # last call: after 18 updates its remaining valid length is exactly one.
    wide = unary('valid_length', 'Cast', [length_edge], i32, [1], dst_type=3)
    remain = unary('remaining', 'Sub', [wide, consumed], i32, [1])
    last_length = unary('last_length', 'Cast', [remain], i16, [1], dst_type=6)
    sizes = {rank: constant(f'size{rank}', shapes('g' if rank == 3 else 'core', 1)[0], i32) for rank in (3, 4)}
    state = original.input[ports['initial_state']]
    cores = []
    for t in range(19):
        offsets = {r: constant(f'offset{r}_{t}', [0, t] + [0] * (r-2), i32) for r in sizes}
        step = type(original)()
        step.CopyFrom(original)
        step.name = f'l0s1_gdr_{t:02}'
        if step.name in nodes:
            raise ValueError('Duplicate GE node ' + step.name)
        step.attr['chunk_size'].i = 1
        for role in TOKEN_OPERANDS:
            shape = shapes(role, 1)[0]
            rank = len(shape)
            slot = ports[role]
            step.input[slot] = unary(f'{role}_{t}', 'Slice', [original.input[slot], offsets[rank], sizes[rank]], half, shape)
            step.input_desc[slot].shape.CopyFrom(desc(half, shape).shape)
        step.input[ports['initial_state']] = state
        step.input[ports['effective_length']] = one if t < 18 else last_length
        step.output_desc[0].shape.CopyFrom(desc(half, shapes('core', 1)[0]).shape)
        # TorchAir port attributes describe which original operands are constants.
        for slot in ports.values():
            const = nodes[step.input[slot].rsplit(':', 1)[0]].type in ('Const', 'FileConstant')
            if step.is_input_const:
                step.is_input_const[slot] = const
            if 'is_input_const' in step.attr and step.attr['is_input_const'].list.b:
                step.attr['is_input_const'].list.b[slot] = const
        if step.src_name:
            for slot in ports.values():
                owner, port = step.input[slot].rsplit(':', 1)
                step.src_name[slot], step.src_index[slot] = owner, int(port)
        step.id = first_id + len(added)
        added.append(step)
        nodes[step.name] = step
        cores.append(step.name + ':0')
        state = step.name + ':1'

    padding = constant('padding', np.zeros(shapes('core', 45)[0], dtype='<f2'), half)
    axis = constant('time_axis', 1, i32)
    core = unary('core', 'ConcatV2', [*cores, padding, axis], half, shapes('core')[0], N=20)
    replacements = {original.name + ':0': core, original.name + ':1': state, original.name + ':-1': step.name + ':-1'}
    for node in graph.op:
        for i, edge in enumerate(node.input):
            if edge in replacements:
                node.input[i] = replacements[edge]
                if i < len(node.src_name) and i < len(node.src_index):
                    owner, port = node.input[i].rsplit(':', 1)
                    node.src_name[i], node.src_index[i] = owner, int(port)
    for i, edge in enumerate(graph.output):
        graph.output[i] = replacements.get(edge, edge)
    idx = next(i for i, n in enumerate(graph.op) if n.name == original.name)
    del graph.op[idx]
    for i, node in enumerate(added):
        graph.op.insert(idx + i, node)
    return msg.SerializeToString(), replacements


def prepare(group, output):
    capture = sha(group / 'reference/manifest.json')
    target = output_path(output)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        root = Path(tmp) / 'air'
        cases = []
        for batch, names in GROUPS.items():
            source = group / batch / 'air'
            if not (source / 'manifest.json').is_file():
                if batch == 'gamma-readout':
                    source = group / 'air'  # Both retained batches include the original hidden and GDR cases.
                else:
                    from .gdr_stage_air import prepare as prepare_chunk_taps
                    source = Path(tmp) / batch / 'air'
                    options = dict(paired_silu=True, indexed_silu_layer=0) if batch == 'l0-silu-pair' else dict(split_norm_layers=[0])
                    prepare_chunk_taps(group, group / 'reference', source, **options)
            for name in names:
                manifest, case, folder, record = _record(source, name)
                if record['capture_sha256'] != capture or (record['physical_rows'], record['valid_rows'], record['reference_position']) != (64, 19, 18):
                    raise ValueError(f'{name}: source does not match the 310P 64/19 fixture')
                length, = [s for s in record['source_inputs'] if s['semantic_name'] == 'effective_length']
                if length['dtype'] != 'int16' or read_array(folder, length).tolist() != [19] or record['length_edge'] != length['name'] + ':0':
                    raise ValueError(f'{name}: expected runtime INT16 [19]')
                if name == BASE:
                    if record['reference_basis'] != NATIVE_GDR_MODULE_BASIS:
                        raise ValueError('Expected the uninstrumented native GDR stage baseline')
                    chunk_hash = record['air']['sha256']
                elif record['gdr_stages']['source_air'] != chunk_hash or record['gdr_stages']['capture'] != capture:
                    raise ValueError(f'{name}: probe has a different chunk baseline or capture')
                src = checked(folder, record['air'])
                dest = root / 'models' / name
                shutil.copytree(folder, dest)
                raw, edges = unroll(src.read_bytes(), record['length_edge'])
                path = dest / record['air']['path']
                path.write_bytes(raw)
                record.update(chunk_air_sha256=record['air']['sha256'], air=identity(path, dest), gdr_mode=MODE)
                if name == BASE:
                    source_hash = record['air']['sha256']
                else:
                    meta = record['gdr_stages']
                    meta.update(source_air=source_hash, edges={r: edges.get(e, e) for r, e in meta['edges'].items()})
                contract = record['output_contract']
                contract['source_edges'] = [edges.get(e, e) for e in contract['source_edges']]
                save_json(dest / 'export.json', record)
                cases.append(dict(case, export=identity(dest / 'export.json', root)))
        # The paired probe owns both SiLU reads; exclude the obsolete direct Swish output.
        save_json(root / 'manifest.json', dict(artifact_kind=manifest['artifact_kind'], status='EXPORTED',
            cases=cases, capture_sha256=capture, golden_source=BASIS, reference_model_layers=4,
            scope=MODE, gdr_mode=MODE, split_norm_layers=[0]))
        root.rename(target)
    share_for_transfer(target)
    print(f'{target}: {len(cases)} L0 graphs, each with 19 state-linked ChunkGatedDeltaRule S=1 calls', flush=True)

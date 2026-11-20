"""Expose tensors on the existing unscaled whole-module AIR, using its producer ports."""
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import checked, identity, read_array, save_json, sha
from .decode_cache_control import graph_message
from .decode_cache_trace import _add_ports, _project_input, _record
from .gdr_stage_data import BASIS, GDR, GROUPS, case_id, logical, shapes
from .segment_export import MODULE_BASIS

VIEWS = {'Reshape', 'Identity', 'TensorMove', 'Unsqueeze', 'Squeeze', 'ExpandDims'}


def stage_edges(graph, group):
    ops = {n.name: n for n in graph.op}
    gdr, = [n for n in graph.op if n.type == 'ChunkGatedDeltaRule']
    if gdr.attr['chunk_size'].i != 64:
        raise ValueError('Expected one unscaled prefill GDR with chunk_size=64')
    def node(edge):
        return ops[edge.rsplit(':', 1)[0]]
    def strip(edge, types):
        while node(edge).type in types:
            edge = node(edge).input[0]
        return edge
    def inputs(op):
        return [s for s in op.input if not s.endswith(':-1')]
    @lru_cache(None)
    def from_core(edge):
        if node(edge).name == gdr.name:
            return edge == gdr.name + ':0'
        return any(from_core(s) for s in inputs(node(edge)) if s)
    def inner(edge, op_types):
        value = strip(edge, VIEWS | {'Cast'})
        op = node(value)
        if op.type not in op_types:
            raise ValueError(f'Expected {sorted(op_types)} upstream of {edge}, found {op.name} ({op.type})')
        return op
    if group == 'gdr':
        names = [n.decode() for n in gdr.attr['_input_name_key'].list.s]
        ports = dict(zip(names, gdr.attr['_input_name_value'].list.i, strict=True))
        edges = {name: inputs(gdr)[ports[name]] for name in GDR[:-2]}
        return dict(edges, core=gdr.name + ':0', final_state=gdr.name + ':1')
    net, = [n for n in graph.op if n.type == 'NetOutput']
    hidden, = inputs(net)
    proj_input = _project_input(ops, hidden)
    mul = inner(proj_input, {'Mul'})
    a, b = inputs(mul)
    if from_core(a) == from_core(b):
        raise ValueError('Could not separate gated product operands by GDR core dependency')
    lhs, rhs = (a, b) if from_core(a) else (b, a)
    act = inner(rhs, {'Silu', 'SiLU', 'Swish', 'SwishV2', 'Mul'})
    if act.type == 'Mul':
        x, y = inputs(act)
        if node(strip(x, VIEWS)).type == 'Sigmoid':
            x, y = y, x
        sigmoid = inner(y, {'Sigmoid'})
        if strip(x, VIEWS | {'Cast'}) != strip(sigmoid.input[0], VIEWS | {'Cast'}):
            raise ValueError('Expected x * sigmoid(x) for the gate SiLU')
        silu_input = x
    else:
        silu_input = act.input[0]
    rms_edge = strip(lhs, VIEWS | {'Cast'})
    rms = node(rms_edge)
    if rms.type in ('RmsNorm', 'RMSNorm', 'AdnRmsNorm'):
        rms_input, gamma = inputs(rms)[:2]
        if rms_edge != rms.name + ':0':
            raise ValueError('Gated output does not read the normalized activation')
    elif rms.type == 'Mul':
        # Decomposed RMSNorm: identify gamma on the independent branch and
        # the actual tensor squared for the variance, bounded by this norm's output.
        a, b = inputs(rms)
        if from_core(a) == from_core(b):
            raise ValueError('Could not identify the direct RMSNorm weight')
        gamma = b if from_core(a) else a
        def squares(edge, seen):
            if edge in seen or not from_core(edge) or node(edge).name == gdr.name:
                return set()
            seen.add(edge)
            op = node(edge)
            edges = inputs(op)
            if (op.type in ('Square', 'Pow', 'Power') or (op.type == 'Mul' and edges[0] == edges[1])) and (
                    strip(edges[0], VIEWS | {'Cast'}) == gdr.name + ':0'):
                return {edges[0]}
            return set().union(*(squares(s, seen) for s in edges if s))
        hits = squares(rms_edge, set())
        if len(hits) != 1:
            raise ValueError(f'Could not bind the variance input for decomposed RMSNorm: {hits}')
        rms_input, = hits
    else:
        raise ValueError(f'Unrecognized gated RMSNorm producer {rms.name} ({rms.type})')
    return dict(rms_input=rms_input, gamma=gamma, rms_output=rms_edge,
                gate=node(silu_input).input[0] if node(silu_input).type == 'Cast' else silu_input, silu_input=silu_input, silu_output=act.name + ':0',
                mul_lhs=lhs, mul_rhs=rhs, mul_output=mul.name + ':0', gated_return=proj_input)


def prepare(baseline, reference, output):
    capture = json.loads((reference / 'manifest.json').read_text())
    if (capture['status'], capture['physical_rows'], capture['valid_rows'], capture['output_pos']) != ('CAPTURED', 64, 19, 18):
        raise ValueError('Expected a completed physical64/valid19 prefill')
    target = output_path(output)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        root = Path(tmp) / 'air'
        cases = []
        for i in range(3):
            orig = f'l{i:02}_c00_gdn_layer_hidden_o0'
            manifest, case, folder, record = _record(baseline / 'air', orig)
            if (record['reference_basis'] != MODULE_BASIS
                    or (record['physical_rows'], record['valid_rows'], record['reference_position']) != (64, 19, 18)):
                raise ValueError(f'{orig}: expected original standalone valid19 hidden AIR')
            info = capture['attention_modules'][str(i)]
            gold = read_array(folder, record['golden'])
            if gold[:, :19].tobytes() != read_array(reference, info['golden_hidden'])[:, :19].tobytes():
                raise ValueError(f'{orig}: traced 310P valid hidden changed')
            for s in record['source_inputs']:
                role = s['semantic_name']
                old = read_array(folder, s)
                live = read_array(reference, info['inputs'][role])
                if role == 'hidden':
                    old, live = old[:, :19], live[:, :19]
                if old.shape != live.shape or old.tobytes() != live.astype(old.dtype).tobytes() or not np.array_equal(old, live):
                    raise ValueError(f'{orig}: captured {role} changed')
            raw = checked(folder, record['air']).read_bytes()
            native = capture['gdr_stages'][str(i)]
            for group in GROUPS:
                msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
                edges = stage_edges(graph, group)
                # Two boundaries may read exactly the same producer. Read it once
                # and retain the alias explicitly instead of relying on OMC duplicate-output handling.
                slots, taps, aliases = {}, {}, {}
                for role, edge in edges.items():
                    primary = slots.setdefault(edge, role)
                    aliases[role] = primary
                    taps[primary] = edge
                enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
                codes = {enum[n].number: typ for n, typ in [('DT_FLOAT16', 'float16'), ('DT_FLOAT','float32'), ('DT_INT16','int16')]}
                source_edges = _add_ports(graph, taps, allowed_dtypes=(0, *codes))
                norm_epsilon = None
                if group == 'norm':
                    owner = edges['rms_output'].rsplit(':', 1)[0]
                    norm_op = next(n for n in graph.op if n.name == owner)
                    if 'epsilon' in norm_op.attr:
                        norm_epsilon = norm_op.attr['epsilon'].f
                name = case_id(i, group)
                dest = root / 'models' / name
                shutil.copytree(folder, dest)
                air = dest / record['air']['path']
                air.write_bytes(msg.SerializeToString())
                goldens = [record['golden']]
                required = {'0': 'float16'}
                air_dtypes = {}
                for idx, role in enumerate(taps, 1):
                    ref = native['tensors'][role]
                    tensor = logical(read_array(reference, ref), role)
                    path = dest / ('trace_' + role + '.bin')
                    tensor.tofile(path)
                    goldens.append(dict(identity(path, dest), name=role, dtype=tensor.dtype.name, shape=list(tensor.shape)))
                    op, port = taps[role].rsplit(':', 1)
                    typ = next(n for n in graph.op if n.name == op).output_desc[int(port)].dtype
                    air_dtypes[role] = codes.get(typ, 'unspecified')
                    if role == 'effective_length':
                        required[str(idx)] = 'int16'
                contract = dict(order=[s['name'] for s in goldens], source_edges=source_edges, required_dtypes=required,
                                stage_views={k: shapes(k) for k in taps})
                meta = dict(source=orig, source_air=record['air']['sha256'], aliases=aliases, edges=edges,
                            group=group, air_dtypes=air_dtypes, air_rms_epsilon=norm_epsilon, capture=sha(reference / 'manifest.json'), epsilon=native['epsilon'])
                export = dest / 'export.json'
                save_json(export, dict(record, goldens=goldens, air=identity(air, dest), reference_basis=BASIS,
                                       output_contract=contract, gdr_stages=meta))
                cases.append(dict(case, id=name, kind='gdr_stages', point=group + '_stages', output_name='stages',
                                  export=identity(export, root)))
        save_json(root / 'manifest.json', dict(manifest, cases=cases, scope='THREE_LAYER_GDR_AND_GATED_NORM_DIAGNOSTIC', golden_source=BASIS))
        root.rename(target)
    share_for_transfer(target)
    print(target, flush=True)

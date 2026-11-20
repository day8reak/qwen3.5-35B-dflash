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
from .gdr_gamma_readout import add_indexed_readout, index_binding
from .gdr_stage_data import BASIS, GDR, GROUPS, case_id, logical, shapes
from .segment_export import MODULE_BASIS, NATIVE_GDR_MODULE_BASIS

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


def _trace_graphs(raw, split_norm, only_role=None):
    for group in (('norm',) if split_norm else GROUPS):
        msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
        edges = stage_edges(graph, group)
        if split_norm:
            # Each readout starts from the original AIR. Aliases of the same
            # operand share one diagnostic output and one invocation.
            selected = [edges[only_role]] if only_role else dict.fromkeys(edges.values())
            for edge in selected:
                roles = {name: e for name, e in edges.items() if e == edge}
                label = 'norm_' + next(iter(roles))
                msg, graph = graph_message(raw, 'ChunkGatedDeltaRule')
                yield group, label, msg, graph, roles
        else:
            yield group, group, msg, graph, edges


def prepare(baseline, reference, output, *, include_baseline=False, split_norm_layers=None, indexed_silu_layer=None):
    if indexed_silu_layer is not None:
        if split_norm_layers is not None:
            raise ValueError('Select either the split norm batch or the indexed SiLU readout')
        split_norm_layers = [indexed_silu_layer]
    split_norm = split_norm_layers is not None
    layers = list(split_norm_layers) if split_norm else list(range(3))
    if split_norm and (include_baseline or not layers or len(set(layers)) != len(layers) or set(layers) - {0, 1, 2}):
        raise ValueError('Single-output norm probes need distinct layers within 0..2 and a separate original baseline')
    capture_hash = sha(reference / 'manifest.json')
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
        for i in layers:
            orig = f'l{i:02}_c00_gdn_layer_hidden_o0'
            manifest, case, folder, record = _record(baseline / 'air', orig)
            if (record['reference_basis'] not in (MODULE_BASIS, NATIVE_GDR_MODULE_BASIS)
                    or (record['physical_rows'], record['valid_rows'], record['reference_position']) != (64, 19, 18)):
                raise ValueError(f'{orig}: expected original standalone valid19 hidden AIR')
            if (include_baseline or split_norm) and (record['reference_basis'] != NATIVE_GDR_MODULE_BASIS
                    or record['capture_sha256'] != capture_hash or manifest['capture_sha256'] != capture_hash):
                raise ValueError(f'{orig}: re-export the whole module from this native GDR stage capture')
            info = capture['attention_modules'][str(i)]
            gold = read_array(folder, record['golden'])
            hidden = read_array(reference, info['golden_hidden'])
            rows = slice(None) if include_baseline or split_norm else slice(0, 19)
            if gold.shape != hidden.shape or gold.dtype != hidden.dtype or gold[:, rows].tobytes() != hidden[:, rows].tobytes():
                raise ValueError(f'{orig}: traced 310P hidden changed')
            inputs = {}
            for spec in record['source_inputs']:
                role = spec['semantic_name']
                old = read_array(folder, spec)
                live = read_array(reference, info['inputs'][role])
                if old.shape != live.shape:
                    raise ValueError(f'{orig}: captured {role} shape changed')
                # The original CPro graph carries the fresh zero state in FP16.
                bound = live.astype(old.dtype, copy=False)
                if bound.dtype != live.dtype and not np.array_equal(bound.astype(live.dtype), live):
                    raise ValueError(f'{orig}: CPro input ABI loses values for {role}')
                baseline_values, current_values = (old[:, rows], bound[:, rows]) if role == 'hidden' else (old, bound)
                if baseline_values.tobytes() != current_values.tobytes():
                    raise ValueError(f'{orig}: captured {role} changed')
                inputs[role] = bound
            raw = checked(folder, record['air']).read_bytes()
            if include_baseline:
                dest = root / 'models' / orig
                shutil.copytree(folder, dest)
                cases.append(dict(case, export=identity(dest / 'export.json', root)))
            native = capture['gdr_stages'][str(i)]
            only_role = 'silu_output' if indexed_silu_layer is not None else None
            for group, label, msg, graph, edges in _trace_graphs(raw, split_norm, only_role):
                # Two boundaries may read exactly the same producer. Read it once
                # and retain the alias explicitly instead of relying on OMC duplicate-output handling.
                slots, taps, aliases = {}, {}, {}
                for role, edge in edges.items():
                    primary = slots.setdefault(edge, role)
                    aliases[role] = primary
                    taps[primary] = edge
                enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
                codes = {enum[n].number: typ for n, typ in [('DT_FLOAT16', 'float16'), ('DT_FLOAT','float32'), ('DT_INT16','int16')]}
                readout_role = only_role or 'gamma'
                readout = None
                if readout_role in taps:
                    readout = add_indexed_readout(graph, edges[readout_role], readout_role)
                    taps[readout_role] = readout['output_edge']
                source_edges = _add_ports(graph, taps, allowed_dtypes=(0, *codes))
                norm_epsilon = None
                if 'rms_output' in edges:
                    owner = edges['rms_output'].rsplit(':', 1)[0]
                    norm_op = next(n for n in graph.op if n.name == owner)
                    if 'epsilon' in norm_op.attr:
                        norm_epsilon = norm_op.attr['epsilon'].f
                name = case_id(i, label)
                dest = root / 'models' / name
                shutil.copytree(folder, dest)
                air = dest / record['air']['path']
                air.write_bytes(msg.SerializeToString())
                source_inputs = []
                for spec in record['source_inputs']:
                    path = dest / spec['path']
                    inputs[spec['semantic_name']].tofile(path)
                    source_inputs.append(dict(spec, **identity(path, dest)))
                if readout:
                    source_inputs.append(index_binding(dest, source_inputs, readout_role))
                hidden_path = dest / record['golden']['path']
                hidden.tofile(hidden_path)
                goldens = [dict(record['golden'], **identity(hidden_path, dest))]
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
                if readout:
                    contract[readout_role + '_readout'] = readout
                    required[str(contract['order'].index(readout_role))] = air_dtypes[readout_role]
                meta = dict(module_inputs='CURRENT_310P_CAPTURE', source=orig, source_air=record['air']['sha256'], aliases=aliases, edges=edges,
                            group=group, air_dtypes=air_dtypes, air_rms_epsilon=norm_epsilon, capture=capture_hash, epsilon=native['epsilon'])
                export = dest / 'export.json'
                save_json(export, dict(record, source_inputs=source_inputs, golden=goldens[0], goldens=goldens, air=identity(air, dest), reference_basis=BASIS,
                                       output_contract=contract, gdr_stages=meta))
                cases.append(dict(case, id=name, kind='gdr_stages', point=group + '_stages', output_name='stages',
                                  export=identity(export, root)))
        batch = dict(manifest, cases=cases, scope='THREE_LAYER_GDR_AND_GATED_NORM_DIAGNOSTIC', golden_source=BASIS)
        if split_norm:
            batch.update(scope='ONE_NORM_OPERAND_PORT_PLUS_ORIGINAL_HIDDEN', split_norm_layers=layers)
        if indexed_silu_layer is not None:
            if len(cases) != 1:
                raise ValueError('Expected one isolated SiLU readout')
            batch.update(scope='INDEXED_SILU_OUTPUT_PLUS_ORIGINAL_HIDDEN', indexed_silu_layer=indexed_silu_layer)
        save_json(root / 'manifest.json', batch)
        root.rename(target)
    share_for_transfer(target)
    print(target, flush=True)

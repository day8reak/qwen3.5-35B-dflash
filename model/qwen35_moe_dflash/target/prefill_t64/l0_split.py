"""One native L0 decoder with observable state updates and separately compiled SiLU calls."""
from hashlib import sha256
import json
from shutil import copyfile

import numpy as np

from ..layer_probe.common import checked, identity, metrics, read_array, sha
from ..silu_patch_air import nd_desc, port_names
from .attention_slice import DTYPES
from .chain_reference import VALID19_IDS_SHA256
from .decoder_four_boundary import data_edges
from .decode_cache_control import graph_message
from .segment_export import compare_segment_output


OUTPUTS = {
    'logits': ([1, 1, 248320], 'float32'),
    'conv_state_0': ([1, 8192, 4], 'float16'),
    'recurrent_state_0': ([1, 32, 128, 128], 'float16'),
    'decoder_hidden': ([1, 64, 2048], 'float32'),
    'out_proj': ([1, 64, 2048], 'float16'),
}
SITES = {(1, 8192, 64): 'conv', (2048, 128): 'gate', (64, 512): 'shared', (512, 512): 'experts'}


def expose_l0(folder):
    """Edit only the staged source. Return the values written by the fresh, single-call prefill."""
    def edit(file, scope, changes):
        path = folder / 'models' / file
        before, separator, after = path.read_text().partition('class ' + scope + '(')
        if not separator:
            raise ValueError(f'{path}: missing {scope}')
        body, following, rest = after.partition('\nclass ')
        for old, new in changes:
            if body.count(old) != 1:
                raise ValueError(f'{scope}: expected a single {old!r}')
            body = body.replace(old, new)
        path.write_text(before + separator + body + following + rest)

    model = 'modeling_qwen3_5_moe_hiai_cpro.py'
    edit(model, 'Qwen3_5MoeGatedDeltaNet', [
        ('        conv_state.copy_(new_conv_state)', ''),
        ('        recurrent_state.copy_(last_recurrent_state)', ''),
        ('return output, past_key_values', 'return output, (new_conv_state.to(conv_state.dtype), last_recurrent_state)'),
    ])
    edit(model, 'Qwen3_5MoeDecoderLayer', [
        ('        hidden_states = residual + hidden_states\n\n        # MoE',
         '        attention_output = hidden_states\n        hidden_states = residual + hidden_states\n\n        # MoE'),
        ('return hidden_states, present_key_value', 'return hidden_states, present_key_value, attention_output'),
    ])
    edit('export_model_wrapper_qwen3_5_moe.py', 'Qwen3_5MoeForCausalLMCproWrapper', [
        ('        hidden_states = hidden_states.to(torch.float32)',
         '        hidden_states = hidden_states.to(torch.float32)\n        decoder_hidden = hidden_states'),
        ('return lm_logits, *kv_caches_out', 'return lm_logits, *kv_caches_out, decoder_hidden, layer_outputs[2]'),
    ])


def pack_reference(root, work, fixture):
    m = json.loads((root / 'manifest.json').read_bytes())
    ids = read_array(root, m['inputs']['input_ids'])
    digest = sha256(ids.astype('<i4').tobytes()).hexdigest()
    if ((m['status'], m['physical_rows'], m['valid_rows'], m['output_pos']) != ('CAPTURED', 64, 19, 18)
            or digest != fixture['token_sha256'] or digest != VALID19_IDS_SHA256):
        raise ValueError('Expected the identical fresh 310P T64/valid19 prompt')
    for key, expected in fixture.get('reference_inputs', {}).items():
        if m['inputs'][key]['sha256'] != expected:
            raise ValueError(f'310P fixture differs at {key}')
    entry = read_array(root, m['inputs']['embedding'])
    quant = np.fromfile(work/'inputs/inputs_embeds.bin', dtype='i1').reshape(1, 64, 2048)
    scale = np.fromfile(work/'inputs/embed_scale.bin', dtype='<f4').reshape(1, 64, 1)
    changed = np.count_nonzero((quant * scale).astype('<f2')[:, :19].view('<u2') != entry[:, :19].view('<u2'))
    if changed != fixture['valid19_embedding_changed_fp16_elements']:
        raise ValueError('L0 reference entry differs from the original measured embedding')
    rows = dict(logits=m['l0_logits'], decoder_hidden=m['boundaries']['0'],
                out_proj=m['attention_modules']['0']['golden_hidden'],
                **{k: m['outputs'][k] for k in ('conv_state_0', 'recurrent_state_0')})
    dest = work / 'golden'
    dest.mkdir()
    reference = {}
    for key, row in rows.items():
        if row['shape'] != OUTPUTS[key][0] or row['dtype'] not in ('float16', 'float32'):
            raise ValueError(f'Unexpected 310P {key}: {row}')
        target = dest / (key + '.bin')
        copyfile(checked(root, row), target)
        reference[key] = dict(identity(target, work), shape=row['shape'], dtype=row['dtype'])
    return dict(capture_sha256=sha(root/'manifest.json'), conv_history=m['conv_history']['0'], tensors=reference)


def score(actual, golden):
    if set(actual) != set(OUTPUTS) or set(golden) != set(OUTPUTS):
        raise ValueError('Missing complete L0 readout or cache')
    report = {}
    for role, (shape, _) in OUTPUTS.items():
        a, g = actual[role], golden[role]
        if list(a.shape) != shape or list(g.shape) != shape:
            raise ValueError(f'{role}: expected {shape}, got {a.shape} and {g.shape}')
        if role in ('decoder_hidden', 'out_proj'):
            report[role] = dict(valid19=metrics(a[0, :19], g[0, :19]), token18=metrics(a[0, 18], g[0, 18]))
        else:
            report[role] = {('token18' if role == 'logits' else 'full_state'): compare_segment_output(role, a, g, None)}
    return report


def wire(name, shape, dtype):
    return dict(name=name, shape=list(shape), dtype=dtype)


def plan(raw, trace):
    _, graph = graph_message(raw, 'SiluPatch')
    nodes = {op.name: op for op in graph.op}
    def one(kind):
        op, = [n for n in graph.op if n.type.lower() == kind.lower()]
        return op
    conv, gdr, net = one('LinearAttentionConvstateUpdate'), one('ChunkGatedDeltaRule'), one('NetOutput')
    if any(n.type in ('Assign', 'AssignVariableOp', 'AdnFusedInferAttention') for n in graph.op):
        raise ValueError('Expected one functional linear-attention layer')
    def activation_edge(edge):
        while True:
            node = nodes[edge.rsplit(':', 1)[0]]
            if node.type not in ('Identity', 'TensorMove'):
                return edge
            edge, = data_edges(node)

    outputs = [activation_edge(e) for e in data_edges(net)]
    if len(outputs) != 5 or len(set(outputs)) != 5:
        raise ValueError(f'Expected five distinct L0 outputs, found {outputs}')
    state_ports = {conv.name + ':0': 'conv_state_0', gdr.name + ':1': 'recurrent_state_0'}
    result = {}
    for output in outputs:
        edge = output
        while edge not in state_ports:
            op = nodes[edge.rsplit(':', 1)[0]]
            if op.type not in ('Cast', 'Identity', 'TensorMove', 'Reshape', 'View'):
                break
            edge = data_edges(op)[0]
        if edge in state_ports:
            result[state_ports[edge]] = output
    if set(result) != set(state_ports.values()):
        raise ValueError(f'Cannot bind both updated states by their producer ports: {outputs}')

    # The three public activations form out_proj -> final residual -> norm/head.
    # Derive them by connectivity even if NetOutput reorders the equal-size hidden tensors.
    activations = [e for e in outputs if e not in result.values()]
    for output in activations:
        todo, seen = list(data_edges(nodes[output.rsplit(':', 1)[0]])), set()
        while todo:
            edge = todo.pop()
            if edge not in seen:
                seen.add(edge)
                todo.extend(data_edges(nodes[edge.rsplit(':', 1)[0]]))
        role = ('out_proj', 'decoder_hidden', 'logits')[len(seen.intersection(activations))]
        if role in result:
            raise ValueError(f'Expected a connected L0 readout; cannot distinguish {output} and {result[role]}')
        result[role] = output
    by_shape = {tuple(s['shape']): s['name'] for s in trace}
    if set(by_shape) != set(SITES) or set(by_shape.values()) != {n.name for n in graph.op if n.type == 'SiluPatch'}:
        raise ValueError('Expected one native patch at each of the four L0 SiLU sites')
    groups, wires = {}, {}
    for shape, name in SITES.items():
        op = nodes[by_shape[shape]]
        x, = data_edges(op)
        z = op.name + ':0'
        groups[name + '_input'] = [x]
        groups[name + '_silu'] = [z]
        wires[x], wires[z] = wire(name + '_x', shape, 'float16'), wire(name + '_z', shape, 'float32')
    # Carry the state and the activation from the *same* ConvState/GDR invocation.
    groups['conv_input'].append(result['conv_state_0'])
    gate, = groups['gate_silu']
    mul, = [n for n in graph.op if n.type == 'Mul' and gate in n.input]
    norm, = [e for e in data_edges(mul) if e != gate]
    groups['gdr_norm'] = [norm, result['recurrent_state_0']]
    wires[norm] = wire('norm', (2048, 128), 'float32')
    for role, edge in result.items():
        wires[edge] = wire(role, *OUTPUTS[role])
        if role not in ('conv_state_0', 'recurrent_state_0'):
            groups[role] = [edge]

    entries = {}
    reverse_types = {v: k for k, v in DTYPES.items()}
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number
    for n in graph.op:
        if n.type in ('Data', 'RefData'):
            desc, = n.output_desc
            entries[n.name + ':0'] = wire(n.name, desc.shape.dim, reverse_types[enum[desc.dtype].name])
    owners = {e: key for key, outputs in groups.items() for e in outputs}
    if len(owners) != sum(map(len, groups.values())):
        raise ValueError('Overlapping stage outputs')
    done, active, ordered, ancestors = set(), set(), [], {}
    def stage(key):
        if key in done:
            return
        if key in active:
            raise ValueError(f'Cyclic stage dependency at {key}')
        active.add(key)
        kept, deps = set(), set()
        def visit(edge):
            if edge in entries or (edge in owners and owners[edge] != key):
                deps.add(edge)
                return
            op = nodes[edge.rsplit(':', 1)[0]]
            if op.name in kept:
                return
            if op.type in ('Data', 'RefData', 'NetOutput') or op.subgraph_name:
                raise ValueError(f'Unexpected dependency {edge} in {key}')
            kept.add(op.name)
            for dep in op.input:
                if dep:
                    visit(dep)
        for edge in groups[key]:
            visit(edge)
        patches = [n for n in kept if nodes[n].type == 'SiluPatch']
        if key.endswith('_silu'):
            if len(kept) != 1 or len(patches) != 1 or len(deps) != 1:
                raise ValueError(f'{key}: custom OMC must have exactly one arithmetic node')
        elif patches:
            raise ValueError(f'{key}: unisolated SiluPatch {patches}')
        parents = {owners[e] for e in deps if e not in entries}
        ancestors[key] = set(parents)
        for parent in sorted(parents):
            stage(parent)
            ancestors[key].update(ancestors[parent])
        ordered.append(dict(name=key, inputs=sorted(deps), outputs=groups[key], nodes=sorted(kept)))
        done.add(key)
        active.remove(key)
    stage('logits')
    if done != set(groups) or sum(gdr.name in s['nodes'] for s in ordered) != 1 or sum(
            conv.name in s['nodes'] for s in ordered) != 1:
        raise ValueError('Incomplete L0 chain or repeated state update')
    expected = {'gdr_norm': {'conv_silu'}, 'out_proj': {'gate_silu', 'gdr_norm'},
                'decoder_hidden': {'out_proj', 'shared_silu', 'experts_silu'}, 'logits': {'decoder_hidden'}}
    for key, required in expected.items():
        if not required <= ancestors[key]:
            raise ValueError(f'{key}: missing upstream L0 edges for {required - ancestors[key]}')
    return dict(entries=entries, wires={**entries, **wires}, stages=ordered, results=result)


def slice_air(raw, step, wires):
    msg, graph = graph_message(raw, 'SiluPatch')
    net, = [n for n in graph.op if n.type == 'NetOutput']
    via_net = list(graph.output[:1]) == [net.name + ':0']
    body = []
    codes = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    bindings = {}
    next_id = max(n.id for n in graph.op) + 1
    for i, edge in enumerate(step['inputs']):
        w = wires[edge]
        name = f'l0_feed_{i}'
        bindings[edge] = name
        op = type(net)(name=name, type='Data', id=next_id + i)
        op.input.append('')
        op.attr['index'].i = i
        port_names(op, 'x', 'y')
        for ports in (op.input_desc, op.output_desc):
            nd_desc(ports.add(dtype=codes[DTYPES[w['dtype']]].number), w['shape'])
        body.append(op)
    for op in graph.op:
        if op.name not in step['nodes']:
            continue
        for i, edge in enumerate(op.input):
            if edge in bindings:
                op.input[i] = bindings[edge] + ':0'
                if op.src_name:
                    op.src_name[i] = bindings[edge]
                if op.src_index:
                    op.src_index[i] = 0
        del op.dst_name[:]
        del op.dst_index[:]
        body.append(op)
    out = type(net)(name=net.name, type='NetOutput', id=net.id)
    for edge in step['outputs']:
        out.input.append(edge)
        w = wires[edge]
        nd_desc(out.input_desc.add(dtype=codes[DTYPES[w['dtype']]].number), w['shape'])
    del graph.op[:]
    graph.op.extend([*body, out])
    graph.input[:] = [name + ':0' for name in bindings.values()]
    graph.output[:] = [f'{out.name}:{i}' if via_net else e for i, e in enumerate(step['outputs'])]
    return msg.SerializeToString()

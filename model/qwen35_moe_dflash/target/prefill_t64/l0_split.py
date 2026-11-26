"""Split native prefill decoders at SiLU and state updates and score their public outputs."""
from hashlib import sha256
import json
from shutil import copyfile

import numpy as np

from ..layer_probe.common import checked, identity, metrics, read_array, sha
from ..silu_patch_air import nd_desc, port_names
from .attention_slice import DTYPES, descriptor
from .chain_reference import VALID19_IDS_SHA256
from .decoder_four_boundary import data_edges
from .decoder_four_silu import gate_product
from .decode_cache_control import graph_message
from .segment_export import compare_segment_output


CPU_BASIS = 'CPU_W4_SPLIT_PREFILL_T64_VALID19_V1'
YAML_CPU_BASIS = 'CPU_W4_PREFILL_T64_YAML_V1'
OUTPUTS = {
    'logits': ([1, 1, 248320], 'float32'),
    'conv_state_0': ([1, 8192, 4], 'float16'),
    'recurrent_state_0': ([1, 32, 128, 128], 'float16'),
    'decoder_hidden': ([1, 64, 2048], 'float32'),
    'out_proj': ([1, 64, 2048], 'float16'),
}
SITES = {(1, 8192, 64): 'conv', (2048, 128): 'gate', (64, 512): 'shared', (512, 512): 'experts'}


def edit_class(path, scope, changes):
    before, separator, after = path.read_text().partition('class ' + scope + '(')
    if not separator:
        raise ValueError(f'{path}: missing {scope}')
    body, following, rest = after.partition('\nclass ')
    for old, new in changes:
        if body.count(old) != 1:
            raise ValueError(f'{scope}: expected a single {old!r}')
        body = body.replace(old, new)
    path.write_text(before + separator + body + following + rest)


def expose_projection(path):
    edit_class(path, 'Qwen3_5MoeDecoderLayer', [
        ('        hidden_states = residual + hidden_states\n\n        # MoE',
         '        attention_output = hidden_states\n        hidden_states = residual + hidden_states\n\n        # MoE'),
        ('return hidden_states, present_key_value', 'return hidden_states, present_key_value, attention_output'),
    ])


def expose_l0(folder, *, layer=0, head=True):
    """Edit only the staged source. Return the values written by the fresh, single-call prefill."""
    model = 'modeling_qwen3_5_moe_hiai_cpro.py'
    edit_class(folder / 'models' / model, 'Qwen3_5MoeGatedDeltaNet', [
        ('        conv_state.copy_(new_conv_state)', ''),
        ('        recurrent_state.copy_(last_recurrent_state)', ''),
        ('return output, past_key_values', 'return output, (new_conv_state.to(conv_state.dtype), last_recurrent_state)'),
    ])
    expose_projection(folder / 'models' / model)
    edit_class(folder / 'models/export_model_wrapper_qwen3_5_moe.py', 'Qwen3_5MoeForCausalLMCproWrapper', [
        ('        hidden_states = inputs_embeds\n', '        hidden_states = inputs_embeds.to(torch.float16)\n'),
        ('        hidden_states = hidden_states.to(torch.float32)',
         '        hidden_states = hidden_states.to(torch.float32)\n        decoder_hidden = hidden_states'),
        ('hidden_states = hidden_states[:, output_pos, :]',
         'hidden_states = torch.index_select(hidden_states, 1, output_pos.int())'),
        ('hidden_states = self.model.model.norm(hidden_states)',
         'hidden_states = self.model.model.norm(hidden_states.to(torch.float16))'),
        ('return lm_logits, *kv_caches_out', 'return lm_logits, *kv_caches_out, decoder_hidden, layer_outputs[2]'),
    ])
    # The one-block graph uses local index 0. Select the global weights both in the
    # owner's HF loader and in its W4 projection/expert loader.
    wrapper = [('layer_num=None, start_layer_idx=0', f'layer_num=1, start_layer_idx={layer}')]
    path = folder / 'export_glm_model_qwen3_5_moe.py'
    text = path.read_text()
    replacements = [('start_layer_idx=0, device=\'cpu\', quant_bits=quant_bits)',
                     f'start_layer_idx={layer}, device=\'cpu\', quant_bits=quant_bits)'),
                    ('output_pos = torch.tensor([seq_len], dtype=torch.int32)',
                     'output_pos = torch.tensor([18], dtype=torch.int32)'),
                    ('block_type = config.layer_types[layer_idx]', f'block_type = config.layer_types[layer_idx + {layer}]')]
    owner = (folder/'models/export_model_wrapper_qwen3_5_moe.py').read_text().split(
        'class Qwen3_5MoeForCausalLMCproWrapper(', 1)[1].split('\nclass ', 1)[0]
    if layer:
        # Later layers consume the FP32 residual directly; there is no second embedding lookup.
        begin = owner.index('        if not self.embedding_in_omc')
        wrapper.append((owner[begin:owner.index('        new_kv_cache_pos =', begin)], ''))
        old = '            inputs_embeds=inputs_embeds,\n            embed_scale=embed_scale,'
        replacements.append((old, '            inputs_embeds=inputs_embeds.float(),\n            embed_scale=None,'))
    if not head:
        begin = owner.index('        bsz, q_len, hidden_state_len = hidden_states.size()')
        wrapper.append((owner[begin:].rstrip(), '        return hidden_states.float(), *kv_caches_out, layer_outputs[2]'))
    edit_class(folder / 'models/export_model_wrapper_qwen3_5_moe.py', 'Qwen3_5MoeForCausalLMCproWrapper', wrapper)
    if layer % 4 == 3:
        # The owner's Functionalize rule lowers to this op then copy_ back into
        # the input. Use its returned CacheUpdate tensor for both attention and
        # readout so the same updated bytes can cross an OMC boundary.
        edit_class(folder / 'models' / model, 'Qwen3_5MoeAttention', [
            ('        torch_npu.npu_cache_update_(past_key_value.to(new_k.device),',
             '        return torch.ops.npu.npu_cache_update__functional(past_key_value.to(new_k.device),'),
            ('        return past_key_value\n', ''),
        ])
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f'{path}: cannot select global L{layer} at {old!r}')
        text = text.replace(old, new)
    path.write_text(text)


def pack_reference(root, work, fixture, *, layer=0, head=True, standalone=True):
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
    specs = contracts(layer, head)
    rows = {k: m['outputs'][k] for k in specs if k.endswith(f'_{layer}')}
    if 'blocks' in m:
        rows.update(decoder_hidden=m['blocks'][layer]['hidden'], out_proj=m['outputs'][f'out_proj_{layer}'])
    else:
        rows.update(decoder_hidden=m['boundaries'][str(layer)], out_proj=m['attention_modules'][str(layer)]['golden_hidden'])
    if head:
        rows['logits'] = m['l0_logits'] if layer == 0 else m['outputs']['logits']
    if standalone and layer:
        entry = read_array(root, m['boundaries'][str(layer-1)])
        if tuple(entry.shape) != (1, 64, 2048) or entry.dtype != np.float16:
            raise ValueError(f'Unexpected measured entry for L{layer}')
        entry.astype('<f4').tofile(work/'inputs/layer_entry.bin')
    dest = work / 'golden'
    dest.mkdir()
    reference = {}
    for key, row in rows.items():
        if row['shape'] != specs[key][0] or row['dtype'] not in ('float16', 'float32'):
            raise ValueError(f'Unexpected 310P {key}: {row}')
        target = dest / (key + '.bin')
        copyfile(checked(root, row), target)
        reference[key] = dict(identity(target, work), shape=row['shape'], dtype=row['dtype'])
    packed = dict(capture_sha256=sha(root/'manifest.json'), conv_history=m.get('conv_history', {}).get(str(layer)),
                  tensors=reference)
    if not standalone:
        packed['split_layer'] = m['split_layer']
    return packed


def score(actual, golden, contracts=OUTPUTS, *, reference_is_310p=True, valid_rows=19):
    if set(actual) != set(contracts) or set(golden) != set(contracts):
        raise ValueError('Missing complete L0 readout or cache')
    report = {}
    for role, (shape, _) in contracts.items():
        a, g = actual[role], golden[role]
        if list(a.shape) != shape or list(g.shape) != shape:
            raise ValueError(f'{role}: expected {shape}, got {a.shape} and {g.shape}')
        if role in ('decoder_hidden', 'out_proj'):
            report[role] = {f'valid{valid_rows}': metrics(a[0, :valid_rows], g[0, :valid_rows])}
        else:
            if role.startswith(('key_cache_', 'value_cache_')):
                report[role] = {f'valid{valid_rows}': compare_segment_output(role, a, g, None,
                                    reference_is_310p=reference_is_310p, valid_rows=valid_rows)}
            else:
                report[role] = {(f'token{valid_rows-1}' if role == 'logits' else 'full_state'): compare_segment_output(role, a, g, None)}
    return report


def wire(name, shape, dtype):
    return dict(name=name, shape=list(shape), dtype=dtype)


def contracts(layer, head):
    specs = {k: v for k, v in OUTPUTS.items() if k in ('decoder_hidden', 'out_proj') or (head and k == 'logits')}
    if layer % 4 == 3:
        specs.update({f'{k}_cache_{layer}': ([64, 32, 32, 16], 'float16') for k in ('key', 'value')})
    else:
        for k in ('conv_state', 'recurrent_state'):
            specs[f'{k}_{layer}'] = OUTPUTS[k + '_0']
    return specs


def physical_history(nodes, outputs):
    """Locate the returned time slice of the concat containing the actual input conv cache."""
    from .attention_steps import CONCATS, SPLITS
    matches = []
    for output in outputs:
        edge = output
        sliced = False
        while True:
            node = nodes[edge.rsplit(':', 1)[0]]
            if node.type in CONCATS and sliced:
                for operand in data_edges(node):
                    src = nodes[operand.rsplit(':', 1)[0]]
                    if (src.type in ('Data', 'RefData') and
                            tuple(src.output_desc[0].shape.dim) == (1, 8192, 4)):
                        matches.append((output, node))
                break
            if node.type not in SPLITS | {'Cast', 'Identity', 'TensorMove', 'Reshape', 'View'}:
                break
            sliced |= node.type in SPLITS
            edge = data_edges(node)[0]
    if len(matches) != 1:
        raise ValueError(f'Expected one returned slice of the physical conv history, found {[e for e, _ in matches]}')
    return matches[0]


def prefill_precision(graph, layers, head, *, rows=64):
    """Inspect the consumer port contracts on the exported AIR before OMG lowering."""
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number
    nodes = {n.name: n for n in graph.op}
    problems = []

    def dtype(desc):
        entry = enum.get(desc.dtype)
        return entry.name if entry else str(desc.dtype)

    def check(op, contract):
        keys = op.attr['_input_name_key'].list.s
        slots = dict(zip((k.decode() for k in keys), op.attr['_input_name_value'].list.i, strict=True))
        ports = {}
        for key, expected in contract.items():
            slot = slots.get(key, -1)
            if not 0 <= slot < min(len(op.input_desc), len(op.input)) or not op.input[slot]:
                problems.append(f'{op.name}.{key}: missing input or descriptor (slot {slot})')
                continue
            actual = dtype(op.input_desc[slot])
            ports[key] = dict(dtype=actual, edge=op.input[slot])
            if actual != expected:
                problems.append(f'{op.name}.{key}: {op.input[slot]} is {actual}, expected {expected}')
        return ports

    matmuls, gdrs = {}, {}
    for op in graph.op:
        if op.type == 'QuantBatchMatmulV4444':
            ports = check(op, dict(x1='DT_INT8', x2='DT_INT32', scale='DT_FLOAT16', pertoken_scale='DT_FLOAT16'))
            matmuls[op.name] = ports
            if 'pertoken_scale' in ports:
                edge = ports['pertoken_scale']['edge']
                cast = nodes.get(edge.rsplit(':', 1)[0])
                if (cast is None or cast.type != 'Cast' or not cast.input_desc or not cast.output_desc
                        or dtype(cast.input_desc[0]) != 'DT_FLOAT' or dtype(cast.output_desc[0]) != 'DT_FLOAT16'):
                    problems.append(f'{op.name}.pertoken_scale: expected FP32 to FP16 Cast at {edge}')
        elif op.type == 'ChunkGatedDeltaRule':
            expected = dict.fromkeys(('query', 'key', 'value', 'g', 'beta', 'initial_state'), 'DT_FLOAT16')
            ports = check(op, dict(expected, effective_length='DT_INT16'))
            state = dtype(op.output_desc[1]) if len(op.output_desc) > 1 else None
            chunk = op.attr['chunk_size'].i
            gdrs[op.name] = dict(inputs=ports, state_output=state, chunk_size=chunk)
            if chunk != rows or state != 'DT_FLOAT16':
                problems.append(f'{op.name}: chunk={chunk}, state output={state}; expected {rows} and DT_FLOAT16')

    linear = sum(layer % 4 != 3 for layer in layers)
    expected = 8 * len(layers) + linear + int(head)  # 4/5 attention + 4 MoE projections per layer, plus head
    if len(matmuls) != expected or len(gdrs) != linear:
        problems.append(f'Expected {expected} W4 matmuls and {linear} GDRs, found {len(matmuls)} and {len(gdrs)}')
    return dict(matmuls=matmuls, gdrs=gdrs, mismatches=problems)


def check_split_precision(graph, layer, head):
    report = prefill_precision(graph, [layer], head)
    if report['mismatches']:
        raise ValueError('\n'.join(report['mismatches']))
    matmuls, gdrs = report['matmuls'], report['gdrs']
    print(f'L{layer} AIR verified: {len(matmuls)} W4 matmuls with FP16 weight and per-token scales, {len(gdrs)} GDR',
          flush=True)
    states = next((g['inputs'] for g in gdrs.values()), {})
    return dict(matmuls=matmuls, gdr_inputs=states)


def slice_dependencies(nodes, outputs, boundaries, key):
    kept, deps = set(), set()
    pending = list(outputs)
    while pending:
        edge = pending.pop()
        if edge in boundaries:
            deps.add(edge)
            continue
        op = nodes[edge.rsplit(':', 1)[0]]
        if op.name in kept:
            continue
        if op.type in ('Data', 'RefData', 'NetOutput') or op.subgraph_name:
            raise ValueError(f'Unexpected dependency {edge} in {key}')
        kept.add(op.name)
        pending.extend(e for e in op.input if e)
    return kept, deps


def split_decoder_projections(chain, nodes):
    """Subdivide just the failed residual/MoE stage. Keep the original prefix and its checkpoints."""
    from .decode_cache_trace import _project_input
    from .decoder_projection import QUANT, weight_sources

    steps = chain['stages']
    index = next(i for i, s in enumerate(steps) if s['name'] == 'decoder_hidden')
    original = steps[index]
    retained = set(original['nodes'])
    incoming = set(original['inputs'])
    hidden, = original['outputs']
    projections = {n.name + ':0': n for n in nodes.values() if n.name in retained and n.type == QUANT}
    if not projections:
        raise ValueError('No W4 projections in decoder_hidden')
    cuts = {hidden: 'decoder_hidden_after_w4'}
    wires = chain['wires']
    for i, edge in enumerate(projections):
        activation = _project_input(nodes, edge)
        if activation not in incoming:
            cuts.setdefault(activation, f'decoder_w4_input_{i}')
        cuts[edge] = f'decoder_w4_projection_{i}'
    # Static graphs can omit intermediate descriptors' dimensions. Let the
    # producing OMC establish each new wire's actual floating ABI before reuse.
    for edge in cuts.keys() - {hidden}:
        if edge in projections:
            desc = descriptor(nodes, edge)
            dtype = 'float16'  # CPro W4 QLinear output_dtype
        else:
            quant = next(op for op in nodes.values() if op.type == 'DynamicQuant' and op.input[0] == edge)
            desc = quant.input_desc[0] if quant.input_desc else descriptor(nodes, edge)
            dtype = {1: 'float32', 2: 'float16'}.get(desc.dtype)
        wires[edge] = dict(name=cuts[edge], shape=None, dtype=dtype, source_shape=list(desc.shape.dim))

    boundary = incoming | cuts.keys()
    refined, finished = [], set()
    def stage(edge):
        if edge in finished:
            return
        name = cuts[edge]
        kept, deps = slice_dependencies(nodes, [edge], boundary - {edge}, name)
        expected = {projections[edge].name} if edge in projections else set()
        actual = {n for n in kept if nodes[n].type == QUANT}
        if actual != expected or not kept <= retained or not deps:
            raise ValueError(f'{name}: invalid cut at {edge}, W4={sorted(actual)}, inputs={sorted(deps)}')
        for dep in sorted(deps - incoming):
            stage(dep)
        step = dict(name=name, inputs=sorted(deps), outputs=[edge], nodes=sorted(kept))
        if edge != hidden:
            step['weight_storage'] = 'embedded'
        if edge in projections:
            step['projection'] = projections[edge].name
            step['weight_sources'] = weight_sources(nodes, projections[edge].input[1])
        refined.append(step)
        finished.add(edge)
    stage(hidden)
    if finished != cuts.keys() or {n for s in refined for n in s['nodes']} != retained:
        raise ValueError('W4 cuts did not cover the original decoder_hidden stage')
    steps[index:index + 1] = refined


def split_after_attention_residual(chain, graph, nodes):
    """Materialize the skip connection only inside the remaining MoE/residual stage."""
    from .decoder_tail import tail_edges

    steps = chain['stages']
    index = next(i for i, step in enumerate(steps) if step['name'] == 'decoder_hidden_after_w4')
    old = steps[index]
    hidden, = old['outputs']
    attention = chain['results']['out_proj']
    residual, _, _ = tail_edges(graph, target=hidden, attention=attention)
    incoming = set(old['inputs'])
    first = 'decoder_attention_residual'
    last = 'decoder_hidden_from_residual'
    left, left_deps = slice_dependencies(nodes, [residual], incoming, first)
    right, right_deps = slice_dependencies(nodes, [hidden], incoming | {residual}, last)
    if attention not in left_deps or residual not in right_deps or (left | right) != set(old['nodes']):
        raise ValueError(f'Cannot separate {old["name"]} at the post-attention sum {residual}')

    # Uninferred AIR output descriptors can say DT_FLOAT even on FP16 arithmetic.
    # Prefer the two actual Add operands when their types can be traced.
    def operand_dtype(edge):
        while edge not in chain['wires']:
            op = nodes[edge.rsplit(':', 1)[0]]
            if op.type == 'Cast' and 'dst_type' in op.attr and op.attr['dst_type'].HasField('i'):
                return {0: 'float32', 1: 'float16'}.get(op.attr['dst_type'].i)
            if op.type not in ('Identity', 'TensorMove', 'Reshape', 'View', 'Squeeze', 'Unsqueeze', 'ExpandDims'):
                return None
            edge = data_edges(op)[0]
        return chain['wires'][edge]['dtype']

    desc = descriptor(nodes, residual)
    operands = data_edges(nodes[residual.rsplit(':', 1)[0]])
    types = {operand_dtype(e) for e in operands}
    if len(operands) != 2 or (None not in types and len(types) != 1):
        raise ValueError(f'{residual}: unexpected post-attention Add operands {operands} ({types})')
    dtype = next(iter(types)) if None not in types else {0: None, 1: 'float32', 2: 'float16'}.get(desc.dtype)
    if dtype not in (None, 'float32', 'float16') or desc.dtype not in (0, 1, 2):
        raise ValueError(f'{residual}: expected a floating post-attention sum')
    shape = list(desc.shape.dim)
    out_shape = chain['wires'][attention]['shape']
    if not shape or min(shape) < 1:
        shape = out_shape
    if [d for d in shape if d != 1] != [d for d in out_shape if d != 1]:
        raise ValueError(f'{residual}: post-attention sum {shape} differs from attention output {out_shape}')
    chain['wires'][residual] = dict(name=first, shape=None, dtype=dtype, source_shape=shape)
    new_steps = []
    for name, kept, deps, output in ((first, left, left_deps, residual), (last, right, right_deps, hidden)):
        step = dict(name=name, inputs=sorted(deps), outputs=[output], nodes=sorted(kept))
        if name == first:
            step['weight_storage'] = 'embedded'
        step['casts'] = {n: list(nodes[n].input) for n in sorted(kept) if nodes[n].type == 'Cast'}
        new_steps.append(step)
    steps[index:index + 1] = new_steps


def plan(raw, trace, *, layer=0, head=True, verify_precision=False, split_projections=False, split_residual=False,
         separate_embedding=False, per_layer_experts=False):
    _, graph = graph_message(raw, 'SiluPatch')
    nodes = {op.name: op for op in graph.op}
    def one(kind):
        op, = [n for n in graph.op if n.type.lower() == kind.lower()]
        return op
    net = one('NetOutput')
    linear = layer % 4 != 3
    specs = contracts(layer, head)
    if any(n.type in ('Assign', 'AssignVariableOp') for n in graph.op):
        raise ValueError('Expected functional cache updates in the split AIR')
    def activation_edge(edge):
        while True:
            node = nodes[edge.rsplit(':', 1)[0]]
            if node.type not in ('Identity', 'TensorMove'):
                return edge
            edge, = data_edges(node)

    outputs = [activation_edge(e) for e in data_edges(net)]
    if len(outputs) != len(specs) or len(set(outputs)) != len(specs):
        raise ValueError(f'Expected {len(specs)} distinct L{layer} outputs, found {outputs}')
    if linear:
        gdr = one('ChunkGatedDeltaRule')
        legacy = [n for n in graph.op if n.type.lower() == 'linearattentionconvstateupdate']
        if legacy and not verify_precision:
            conv, = legacy
            conv_edge = conv.name + ':0'
        else:
            conv_edge, conv = physical_history(nodes, outputs)
        state_ports = {conv_edge: f'conv_state_{layer}', gdr.name + ':1': f'recurrent_state_{layer}'}
    else:
        attn = one('AdnFusedInferAttention')
        def cache_update(edge):
            while nodes[edge.rsplit(':', 1)[0]].type in ('Identity', 'TensorMove', 'Cast', 'Reshape', 'View'):
                edge = data_edges(nodes[edge.rsplit(':', 1)[0]])[0]
            if nodes[edge.rsplit(':', 1)[0]].type != 'CacheUpdate':
                raise ValueError(f'L{layer}: cannot identify updated K/V at {edge}')
            return edge
        attn_inputs = dict(zip(('key', 'value'), data_edges(attn)[1:3], strict=True))
        state_ports = {cache_update(e): f'{k}_cache_{layer}' for k, e in attn_inputs.items()}
        if len(state_ports) != 2:
            raise ValueError(f'L{layer}: K and V have the same CacheUpdate producer')
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
    sites = {s: n for s, n in SITES.items() if linear or n in ('shared', 'experts')}
    if set(by_shape) != set(sites) or set(by_shape.values()) != {n.name for n in graph.op if n.type == 'SiluPatch'}:
        raise ValueError(f'L{layer}: expected one native patch at each of {list(sites.values())}')
    groups, wires = {}, {}
    for shape, name in sites.items():
        op = nodes[by_shape[shape]]
        x, = data_edges(op)
        z = op.name + ':0'
        if [d.dtype for d in op.input_desc] != [2] or [d.dtype for d in op.output_desc] != [2]:
            raise ValueError(f'{op.name}: expected FP16 input and output. Re-export the CPro AIR.')
        groups[name + '_input'] = [x]
        groups[name + '_silu'] = [z]
        wires[x], wires[z] = wire(name + '_x', shape, 'float16'), wire(name + '_z', shape, 'float16')
    # Carry the state and the activation from the *same* ConvState/GDR invocation.
    if linear:
        if not legacy:
            from .attention_steps import AttentionSteps, CONVS
            topology = AttentionSteps(graph, nodes, [])
            source = topology.nearest(groups['conv_input'][0], CONVS)
            if not topology.depends(source, conv.name + ':0'):
                raise ValueError('Convolution and returned history must consume the same QKV history')
        groups['conv_input'].append(result[f'conv_state_{layer}'])
        gate, = groups['gate_silu']
        _, norm = gate_product(graph, gate)
        groups['gdr_norm'] = [norm, result[f'recurrent_state_{layer}']]
        wires[norm] = wire('norm', (2048, 128), 'float32')
    for role, edge in result.items():
        wires[edge] = wire(role, *specs[role])
        if role not in state_ports.values():
            groups[role] = [edge]
    if not linear:
        # Preserve K/V identities even when OMG renames equal-shaped subgraph outputs.
        # Publish each updated cache once, then feed the actual bytes to attention.
        for k, edge in attn_inputs.items():
            role = f'{k}_cache_{layer}'
            public = result[role]
            # A view or cast on the path into attention must still execute. Cut at
            # the actual public cache, and prove that this value feeds attention.
            while activation_edge(edge) != public:
                op = nodes[activation_edge(edge).rsplit(':', 1)[0]]
                if op.type not in ('Cast', 'Reshape', 'View'):
                    raise ValueError(f'L{layer}: {role} output {public} does not feed attention')
                edge = data_edges(op)[0]
            groups[role] = [public]

    entries = {}
    reverse_types = {v: k for k, v in DTYPES.items()}
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number
    for n in graph.op:
        if n.type in ('Data', 'RefData'):
            desc, = n.output_desc
            entries[n.name + ':0'] = wire(n.name, desc.shape.dim, reverse_types[enum[desc.dtype].name])
    if separate_embedding:
        # Materialize the original signed INT8 times per-token FP32 scale on
        # CPro once. All following linear bodies see a floating hidden entry.
        def embed_origin(edge):
            node = nodes[edge.rsplit(':', 1)[0]]
            if node.type == 'Cast' and node.attr['dst_type'].i == 0:
                edge, = data_edges(node)
            return edge
        sources = {edge for edge, w in entries.items() if (w['dtype'], tuple(w['shape'])) in {
            ('int8', (1, 64, 2048)), ('float32', (1, 64, 1))}}
        candidates = [op for op in graph.op if op.type == 'Mul' and
                      {embed_origin(e) for e in data_edges(op)} == sources]
        if layer != 0 or len(sources) != 2 or len(candidates) != 1:
            raise ValueError('Cannot identify the signed embedding × token scale at the L0 entry')
        edge = candidates[0].name + ':0'
        groups['embedding'] = [edge]
        wires[edge] = wire('embedding', (1, 64, 2048), 'float32')
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
        boundaries = set(entries) | (owners.keys() - set(groups[key]))
        kept, deps = slice_dependencies(nodes, groups[key], boundaries, key)
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
    stage('logits' if head else 'decoder_hidden')
    operators = (gdr.name, conv.name) if linear else (attn.name, *(e.rsplit(':', 1)[0] for e in state_ports))
    if done != set(groups) or any(sum(op in s['nodes'] for s in ordered) != 1 for op in operators):
        raise ValueError(f'Incomplete L{layer} chain or repeated state update')
    expected = {'decoder_hidden': {'out_proj', 'shared_silu', 'experts_silu'}}
    if head:
        expected['logits'] = {'decoder_hidden'}
    if linear:
        expected.update(gdr_norm={'conv_silu'}, out_proj={'gate_silu', 'gdr_norm'})
    else:
        expected['out_proj'] = set(state_ports.values())
    for key, required in expected.items():
        if not required <= ancestors[key]:
            raise ValueError(f'{key}: missing upstream L0 edges for {required - ancestors[key]}')
    chain = dict(entries=entries, wires={**entries, **wires}, stages=ordered, results=result)
    if split_projections or split_residual:
        split_decoder_projections(chain, nodes)
    if split_residual:
        split_after_attention_residual(chain, graph, nodes)
    if per_layer_experts:
        for step in chain['stages']:
            grouped = [name for name in step['nodes'] if nodes[name].type == 'GroupedMatmulV555']
            if grouped:
                step.update(compiler='per_layer', grouped_matmuls=grouped)
                if separate_embedding:
                    step['build_name'] = step['name'] + '-after-embedding'
    if verify_precision:
        chain['precision'] = check_split_precision(graph, layer, head)
    return chain


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
        if w['shape'] is None:
            # The original producer and its exact operands are retained. OMG
            # infers this NetOutput as it would an original intermediate readout.
            desc = out.input_desc.add()
            desc.CopyFrom(descriptor({op.name: op for op in body}, edge))
            if w['dtype']:
                desc.dtype = codes[DTYPES[w['dtype']]].number
        else:
            nd_desc(out.input_desc.add(dtype=codes[DTYPES[w['dtype']]].number), w['shape'])
    del graph.op[:]
    graph.op.extend([*body, out])
    graph.input[:] = [name + ':0' for name in bindings.values()]
    graph.output[:] = [f'{out.name}:{i}' if via_net else e for i, e in enumerate(step['outputs'])]
    return msg.SerializeToString(deterministic=True)

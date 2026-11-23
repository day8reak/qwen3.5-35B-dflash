"""Resolve source-level attention steps against producer ports of the original GE graph."""
from functools import cached_property

from .decode_cache_trace import _project_input
from .gdr_stage_air import stage_edges
from .qkv_silu import CONVS, activation_edges

VIEWS = {'Reshape', 'Cast', 'Identity', 'TensorMove', 'Transpose', 'TransData',
         'Permute', 'ExpandDims', 'Unsqueeze', 'Squeeze'}
SPLITS = {'Split', 'SplitV', 'SplitD', 'SplitVD', 'Slice', 'SliceD', 'StridedSlice', 'StridedSliceD', 'StridedSliceV2'}
CONCATS = {'Concat', 'ConcatD', 'ConcatV2'}
NORMS = {'RmsNorm', 'RMSNorm', 'AdnRmsNorm'}
MATMUL = 'QuantBatchMatmulV4444'


def only(values, label):
    values = list(dict.fromkeys(values))
    if len(values) != 1:
        raise ValueError(f'Expected one {label}, found {values}')
    return values[0]


def operands(op):
    return [e for e in op.input if e and not e.endswith(':-1')]


class AttentionSteps:
    def __init__(self, graph, nodes, bindings):
        self.graph = graph
        self.nodes = nodes
        self.sources = {s['semantic_name']: s['name'] + ':0' for s in bindings}

    def node(self, edge):
        return self.nodes[edge.rsplit(':', 1)[0]]

    def single(self, op_type):
        return only((n.name + ':0' for n in self.graph.op if n.type == op_type), op_type)

    def strip(self, edge):
        while edge.endswith(':0') and self.node(edge).type in VIEWS:
            edge = self.node(edge).input[0]
        return edge

    def expect(self, edge, types):
        op = self.node(edge)
        if op.type not in types:
            raise ValueError(f'Expected {sorted(types)} at {edge}, found {op.type}')
        return op

    def nearest(self, edge, types):
        """Stop at the first matching edge on each data path. Ambiguity is an error."""
        todo, visited, hits = [edge], set(), []
        while todo:
            current = todo.pop()
            if current in visited:
                continue
            visited.add(current)
            op = self.node(current)
            if op.type in types:
                hits.append(current)
            else:
                todo.extend(operands(op))
        return only(hits, f'{sorted(types)} upstream of {edge}')

    def depends(self, edge, source):
        todo, seen = [edge], set()
        while todo:
            current = todo.pop()
            if current == source:
                return True
            if current not in seen:
                seen.add(current)
                todo.extend(operands(self.node(current)))
        return False

    @cached_property
    def qkv_splits(self):
        activated = self.conv_silu[1]
        branches = {}
        for role in ('query', 'key', 'value'):
            todo, seen, hits = [self.gdr[role]], set(), []
            while todo:
                edge = todo.pop()
                if edge == activated or edge in seen:
                    continue
                seen.add(edge)
                op = self.node(edge)
                args = operands(op)
                if op.type in SPLITS:
                    feeds = [e for e in args if self.strip(e) == activated]
                    if feeds:
                        hits.append((only(feeds, 'Q/K/V split input'), edge))
                        continue
                todo.extend(args)
            branches[role] = only(hits, role + ' split of activated QKV')
        if len({pair[0] for pair in branches.values()}) != 1 or len(set(branches.values())) != 3:
            raise ValueError(f'Expected three distinct ports split from one QKV tensor: {branches}')
        return branches

    @cached_property
    def hidden(self):
        return only(operands(self.node(self.single('NetOutput'))), 'attention output')

    @cached_property
    def gdr(self):
        return stage_edges(self.graph, 'gdr')

    @cached_property
    def norm(self):
        return stage_edges(self.graph, 'norm')

    @cached_property
    def conv_silu(self):
        return activation_edges(self.graph)

    @cached_property
    def attention(self):
        return self.node(self.single('AdnFusedInferAttention'))

    def cache(self, kind):
        source = self.sources[kind]
        update = only((n.name for n in self.graph.op if n.type == 'CacheUpdate'
                       and self.strip(n.input[0]) == source), kind + ' CacheUpdate')
        consumer = self.attention.input[1 if kind == 'key' else 2]
        if self.strip(consumer) not in (source, update + ':0'):
            raise ValueError(f'{kind} cache does not supply {consumer}')
        return self.nodes[update]

    def full_branch(self, kind):
        return self.attention.input[0] if kind == 'q' else self.cache('key').input[1]

    def full_norm(self, kind):
        edge = self.nearest(self.full_branch(kind), NORMS)
        if not edge.endswith(':0'):
            raise ValueError(f'Expected normalized activation at {edge}')
        return self.node(edge)

    @cached_property
    def full_gate(self):
        mul = self.expect(self.strip(_project_input(self.nodes, self.hidden)), {'Mul'})
        gate = only((e for e in operands(mul) if self.node(self.strip(e)).type == 'Sigmoid'), 'sigmoid operand')
        core = only((e for e in operands(mul) if e != gate), 'attention operand')
        self.nearest(core, {'AdnFusedInferAttention'})
        return self.node(self.strip(gate)), core, mul

    def projection(self, name):
        if name == 'out_proj':
            mat = self.strip(self.hidden)
        elif name == 'in_proj_qkv':
            mat = self.nearest(self.conv_silu[0], {MATMUL})
        elif name == 'in_proj_z':
            mat = self.nearest(self.norm['gate'], {MATMUL})
        elif name in ('in_proj_a', 'in_proj_b'):
            mat = self.nearest(self.gdr['g' if name == 'in_proj_a' else 'beta'], {MATMUL})
        elif name in ('q_proj', 'k_proj'):
            mat = self.nearest(self.full_norm(name[0]).input[0], {MATMUL})
        elif name == 'v_proj':
            mat = self.nearest(self.cache('value').input[1], {MATMUL})
        else:
            raise ValueError(f'Unknown projection {name}')
        self.expect(mat, {MATMUL})
        return mat

    def project_step(self, name, part):
        mat = self.projection(name)
        input_edge = _project_input(self.nodes, mat)
        if part == 'matmul':
            return None, mat
        if part in ('a8', 'token_scale'):
            quant = self.node(self.strip(self.node(mat).input[0]))
            slot = 0 if part == 'a8' else 1
            target = f'{quant.name}:{slot}'
            if not any(self.strip(e) == target for e in operands(self.node(mat))):
                raise ValueError(f'Projection {mat} does not consume {target}')
            return [input_edge], target
        target = self.hidden if name == 'out_proj' else mat
        return [input_edge], target

    def linear(self, step):
        if step == 'qkv_conv_silu':
            x, y = self.conv_silu
            return [x], y
        if step in ('conv_window', 'depthwise_conv'):
            conv = self.expect(self.strip(self.conv_silu[0]), CONVS)
            if step == 'depthwise_conv':
                return [conv.input[0]], self.conv_silu[0]
            cuts = [self.projection('in_proj_qkv'), self.sources['conv']]
            if 'linear_update_tensor' in self.sources and any(
                    n.type == 'LinearAttentionConvstateUpdate' for n in self.graph.op):
                cuts.append(self.sources['linear_update_tensor'])
            return cuts, conv.input[0]
        if step == 'conv_history':
            return None, self.single('LinearAttentionConvstateUpdate')
        if step == 'qkv_transpose' or step.startswith(('split_', 'prepare_')):
            role = 'query' if step == 'qkv_transpose' else step.split('_', 1)[1]
            before, separated = self.qkv_splits[role]
            if step == 'qkv_transpose':
                return [self.conv_silu[1]], before
            if step.startswith('split_'):
                return None, separated
            return [separated], self.gdr[role]
        if step in ('beta_sigmoid', 'decay_g'):
            beta = step == 'beta_sigmoid'
            return [self.projection('in_proj_b' if beta else 'in_proj_a')], self.gdr['beta' if beta else 'g']
        if step == 'decay_softplus':
            return None, self.nearest(self.gdr['g'], {'Softplus', 'SoftplusV2'})
        if step in ('gdr_core', 'gdr_final_state'):
            return None, self.gdr['core' if step == 'gdr_core' else 'final_state']
        if step == 'core_to_rms':
            return [self.gdr['core']], self.norm['rms_input']
        if step == 'gated_rmsnorm':
            return [self.gdr['core']], self.norm['rms_output']
        if step == 'gate_reshape':
            return [self.projection('in_proj_z')], self.norm['silu_input']
        if step == 'gate_silu':
            return [self.norm['gate']], self.norm['silu_output']
        if step == 'gated_mul':
            return None, self.norm['mul_output']
        if step == 'gated_return':
            return [self.norm['mul_output']], self.norm['gated_return']
        raise ValueError(f'Unknown L0 step {step}')

    def rope(self, kind):
        target = self.strip(self.full_branch(kind))
        # K cache writes can select the batch row after flattening into blocks.
        # Keep that slicing in k_pack, outside the rotary arithmetic.
        if kind == 'k' and self.node(target).type in SPLITS | {'Gather', 'GatherV2', 'GatherV2D'}:
            target = self.strip(self.node(target).input[0])
        node = self.expect(target, CONCATS | {'RotaryMul', 'ApplyRotaryPosEmb'})
        if node.type in CONCATS and not any(self.node(self.strip(e)).type in ('Add', 'AddV2')
                                            for e in operands(node)):
            raise ValueError(f'{target} is not a partial RoPE concat with a rotated sum')
        return target

    def rope_terms(self, kind):
        # Match q_rot*cos and rotate_half(q_rot)*sin by their activation operands.
        # Both table lookups depend on position_ids. Only rotate_half contains a concat.
        node = self.expect(self.rope(kind), CONCATS)
        plus = only((self.strip(e) for e in operands(node)
                     if self.node(self.strip(e)).type in ('Add', 'AddV2')), 'rotary sum')
        terms = {}
        for arg in operands(self.node(plus)):
            mul = self.expect(self.strip(arg), {'Mul'})
            table = only((e for e in operands(mul) if self.depends(e, self.sources['positions'])),
                         'rotary table operand')
            activation = only((e for e in operands(mul) if e != table), 'rotary activation')
            self.expect(self.strip(activation), SPLITS | CONCATS)
            label = 'sin' if self.node(self.strip(activation)).type in CONCATS else 'cos'
            if label in terms:
                raise ValueError(f'Ambiguous {label} term at {plus}')
            terms[label] = (activation, table, mul.name + ':0')
        if terms.keys() != {'cos', 'sin'}:
            raise ValueError(f'Incomplete rotary terms at {plus}')
        return plus, terms

    def full(self, step):
        if step in ('attention_core', 'key_cache_update', 'value_cache_update'):
            op = self.attention if step == 'attention_core' else self.cache(step.split('_')[0])
            return None, op.name + ':0'
        if step in ('cache_blocks', 'cache_offsets'):
            index = 2 if step == 'cache_blocks' else 3
            return [self.sources['cache_positions']], self.cache('key').input[index]
        if step in ('q_split', 'gate_split'):
            target = self.full_norm('q').input[0] if step == 'q_split' else self.full_gate[0].input[0]
            return [self.projection('q_proj')], target
        if step in ('q_norm', 'k_norm'):
            return None, self.full_norm(step[0]).name + ':0'
        if step in ('q_rope', 'k_rope'):
            return [self.full_norm(step[0]).name + ':0', self.sources['positions']], self.rope(step[0])
        if step in ('rope_cos', 'rope_sin'):
            _, table, _ = self.rope_terms('q')[1][step[5:]]
            return [self.sources['positions']], table
        if step.endswith(('_rotate_half', '_mul_cos', '_mul_sin', '_rotary_sum')):
            plus, terms = self.rope_terms(step[0])
            if step.endswith('_rotate_half'):
                return [terms['cos'][0]], terms['sin'][0]
            if step.endswith('_rotary_sum'):
                return None, plus
            return None, terms[step[-3:]][2]
        if step in ('q_pack', 'k_pack', 'v_pack'):
            if step == 'v_pack':
                return [self.projection('v_proj')], self.cache('value').input[1]
            return [self.rope(step[0])], self.full_branch(step[0])
        if step == 'attention_unpack':
            return [self.attention.name + ':0'], self.full_gate[1]
        if step == 'gate_sigmoid':
            return None, self.full_gate[0].name + ':0'
        if step == 'gated_mul':
            return None, self.full_gate[2].name + ':0'
        raise ValueError(f'Unknown L3 step {step}')

    def boundaries(self, layer, step):
        base = step
        part = ''
        for suffix in ('a8', 'token_scale', 'matmul'):
            if step.endswith('_' + suffix):
                base = step[:-(len(suffix) + 1)]
                part = suffix
                break
        if base in ('in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z',
                    'q_proj', 'k_proj', 'v_proj', 'out_proj'):
            return self.project_step(base, part)
        return self.linear(step) if layer == 0 else self.full(step)

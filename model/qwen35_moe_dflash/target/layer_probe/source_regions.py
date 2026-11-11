"""Bind replay expressions to the supplied CPro source, without loading a layer.

Boundary inputs still come from the independent 310P capture. This proves which
Python expression is exported, not equivalence of whole-graph storage or kernels.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap


def definition(function):
    text = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(text).body[0]
    return tree, dict(file=inspect.getsourcefile(function),
        line=inspect.getsourcelines(function)[1], function=function.__qualname__,
        source_sha256=hashlib.sha256(text.encode()).hexdigest())


def one(values, label):
    values = list(values)
    if len(values) != 1:
        raise ValueError(f'CPro source needs one {label}; found {len(values)}')
    return values[0]


def assignment(tree, name):
    return one((n for n in tree.body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name), 'assignment to ' + name)


def call(tree, name):
    return one((n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and ast.unparse(n.func) == name), name)


def qlinear_factory(converters):
    """Follow the exporter's actual replacement function/global, not a namesake."""
    quantize = converters.quant_model_inplace
    tree, outer = definition(quantize)
    call(tree, 'replace_linear_to_QLinear')
    replace = quantize.__globals__['replace_linear_to_QLinear']
    tree, inner = definition(replace)
    call(tree, 'QLinear')
    cls = replace.__globals__['QLinear']
    _, forward = definition(cls.forward)
    return cls, dict(selection='EXPORTER_REPLACEMENT_FUNCTION_GLOBAL',
        export_quantization=outer, replacement=inner, forward=forward,
        factory=cls.__module__ + '.' + cls.__qualname__)


class BlockTableInput(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if ast.unparse(node) == 'self.block_table':
            return ast.copy_location(ast.Name(id='block_table', ctx=ast.Load()), node)
        return self.generic_visit(node)


def fragment(target, kind):
    """Extract only expressions whose dependencies are explicit boundary inputs."""
    owners = dict(convstate=target.Qwen3_5MoeGatedDeltaNet.forward,
        silu=target.Qwen3_5MoeGatedDeltaNet.forward,
        gates=target.Qwen3_5MoeGatedDeltaNet.forward,
        gdn=target.Qwen3_5MoeGatedDeltaNet.forward,
        state_cast=target.Qwen3_5MoeGatedDeltaNet.forward,
        attention=target.Qwen3_5MoeAttention.forward,
        attention_post=target.Qwen3_5MoeAttention.forward,
        add=target.Qwen3_5MoeDecoderLayer.forward)
    owner = owners[kind]
    tree, evidence = definition(owner)
    if kind == 'gates':
        args = ['self', 'a', 'b']
        body = [assignment(tree, 'beta'), assignment(tree, 'g')]
        result = ast.parse('(g, beta)', mode='eval').body
    elif kind == 'convstate':
        args = ['self', 'mixed_qkv', 'conv_state', 'linear_update_tensor']
        body, result = [], call(tree, 'torch.ops.my_ops.linear_attention_convstate_update')
    elif kind == 'silu':
        args = ['self', 'conv_out']
        body, result = [], assignment(tree, 'qkv_conv').value
    elif kind == 'gdn':
        args = ['self', 'query', 'key', 'value', 'g', 'beta', 'recurrent_state', 'effective_length', 'seq_len']
        body, result = [], call(tree, 'torch_npu.npu_chunk_gated_delta_rule')
    elif kind == 'state_cast':
        args = ['self', 'last_recurrent_state']
        body, result = [], assignment(tree, 'last_recurrent_state').value
    elif kind == 'attention':
        args = ['self', 'q2', 'k2', 'v2', 'attention_mask', 'block_table',
                'actual_seq_len_q', 'actual_seq_len_kv', 'allQLen', 'export_flag']
        route = one((n for n in tree.body if isinstance(n, ast.If)
            and ast.unparse(n.test) == 'export_flag'), 'attention export_flag branch')
        body = [assignment(tree, 'attn_params'), route]
        result = call(tree, 'torch_npu.adn_fused_infer_attention')
    elif kind == 'attention_post':
        args = ['self', 'attn_output', 'q_origin_shape', 'input_shape', 'gate']
        # Everything between the actual attention call and o_proj, including
        # original NZ recovery and gate order. Do not copy a second formula.
        begin = one((i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == 'torch_npu.adn_fused_infer_attention'), 'attention producer')
        end = one((i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == 'self.o_proj'), 'attention o_proj')
        body = tree.body[begin + 1:end]
        result = ast.Name(id='attn_output', ctx=ast.Load())
    else:
        args = ['self', 'residual', 'hidden_states']
        adds = [n.value for n in tree.body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Add)
            and any(isinstance(v, ast.Name) and v.id == 'residual' for v in ast.walk(n.value))]
        if len(adds) != 2 or ast.dump(adds[0]) != ast.dump(adds[1]):
            raise ValueError('attention/MLP residual expressions differ; review both boundaries')
        body, result = [], adds[0]
    fn = ast.parse('def replay(' + ', '.join(args) + '):\n    pass').body[0]
    fn.body = copy.deepcopy(body + [ast.Return(value=result)])
    if kind == 'attention':
        fn = BlockTableInput().visit(fn)
    # A new upstream variable must never be silently replaced with a default.
    assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    unexpected = loaded - set(args) - assigned - {'torch', 'torch_npu', 'F'}
    if unexpected:
        raise ValueError(f'CPro {kind} added dependencies: {sorted(unexpected)}')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(owner.__globals__)
    exec(compile(module, evidence['file'], 'exec'), namespace)
    evidence.update(mode='ORIGINAL_FORWARD_FRAGMENT', expression=ast.unparse(fn),
        boundary_adaptation=('self.block_table supplied as captured runtime input; export_flag=True'
                             if kind == 'attention' else 'CAPTURED_BOUNDARY_ARGUMENTS'))
    return namespace['replay'], evidence


def bind_region(region, target, record):
    """Set one source callable; remaining module/lowering paths are explicit."""
    kind = record['kind']
    region.source_contract = dict(kind=kind, input_scope='CAPTURED_310P_BOUNDARY',
        weight_scope='CAPTURED_MODULE_BUFFERS; NOT_RELOADED_FROM_CPRO_CHECKPOINT',
        captured_attributes=record['attrs'],
        output_scope='ONE_SELECTED_OUTPUT; NO_CPRO_STATE_FEEDBACK',
        compiled_storage='NOT_VERIFIED_BY_PYTHON_SOURCE')
    direct = dict(qlinear=lambda: region.op.forward,
        linear=lambda: region.op.forward,
        norm=lambda: target.Qwen3_5MoeRMSNorm.forward,
        norm_gated=lambda: region.op.forward,
        repeat_qk=lambda: target.repeat_qk,
        rope=lambda: target.apply_rotary_pos_emb,
        conv=lambda: target.Qwen3_5MoeGatedDeltaNet.forward,
        cache=lambda: target.Qwen3_5MoeAttention.update)
    if kind in direct:
        tree, evidence = definition(direct[kind]())
        evidence['mode'] = 'ORIGINAL_MODULE_OR_HELPER'
        if kind == 'conv':
            conv = call(tree, 'F.conv1d')
            if ast.unparse(conv) != 'F.conv1d(padded, self.conv1d.weight, groups=self.conv_dim)':
                raise ValueError('original Conv1d call changed; review explicit Conv2d lowering')
            evidence.update(mode='EXPLICIT_EQUIVALENT_CONV2D', expression=ast.unparse(conv))
        elif kind == 'cache':
            op = call(tree, 'torch_npu.npu_cache_update_')
            if len(op.args) != 4 or op.keywords:
                raise ValueError('original CacheUpdate signature changed')
            evidence.update(mode='KERNEL_BOUNDARY_CLONED_CACHE', expression=ast.unparse(op),
                excluded='position/block computation, RefData alias and inter-call feedback')
        elif kind == 'norm':
            evidence['mode'] = 'ZERO_CENTERED_GAMMA_FOLDED_AT_BOUND_CPRO_WEIGHT_DTYPE'
        if kind == 'rope':
            _, evidence['rotate_half'] = definition(target.rotate_half)
    else:
        region.source_forward, evidence = fragment(target, kind)
    if kind == 'state_cast':
        evidence['excluded'] = 'recurrent_state.copy_ and RefData alias; value conversion only'
    if kind == 'convstate':
        evidence['excluded'] = 'conv_state.copy_, Conv consumer and other output from NetOutput'
    if kind in ('attention', 'attention_post'):
        region.num_heads = record['attrs']['num_heads']
        if kind == 'attention':
            region.num_key_value_heads = record['attrs']['num_key_value_heads']
            region.scaling = record['attrs']['scale_value']
        else:
            _, evidence['nz_recovery'] = definition(target.Qwen3_5MoeAttention.transform_nz_2_nd)
    region.source_contract.update(status='BOUND_TO_SOURCE', source=evidence)
    return region.source_contract


def internal_boundaries(graph, schema):
    """Keep both descriptors on every internal op edge; never invent FP16 casts."""
    from qwen35_moe_dflash.target.operator_air_review import descriptor, port_names, trace_input
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    enums = schema.enums[schema.package + '.DataType']
    nodes = {n['name']: n for n in graph['op']}
    focus = {'DynamicQuant', 'LinearAttentionConvstateUpdate', 'ChunkGatedDeltaRule',
             'CacheUpdate', 'AdnFusedInferAttention', 'RmsNorm', 'Conv2D'}
    rows = []
    for node in graph['op']:
        if node['type'] not in focus and 'QuantBatchMat' not in node['type']:
            continue
        attrs = _attributes(node)
        ports = port_names(attrs, 'input')
        inputs = []
        for slot, edge in enumerate(e for e in node.get('input', []) if not e.endswith(':-1')):
            if not edge:
                continue
            descs = node.get('input_desc', [])
            inputs.append(dict(slot=slot, port=ports.get(slot), edge=edge,
                consumer=descriptor(descs[slot] if slot < len(descs) else {}, enums),
                producer=trace_input(edge, nodes, enums, limit=16)))
        rows.append(dict(name=node['name'], type=node['type'], inputs=inputs,
            outputs=[descriptor(d, enums) for d in node.get('output_desc', [])],
            attributes={k: v for k, v in attrs.items() if not k.startswith('_')}))
    return dict(nodes=rows, scope='SOURCE_AIR_EDGES_AND_CASTS; NOT_COMPILED_STORAGE_OR_RUNTIME_BYTES')

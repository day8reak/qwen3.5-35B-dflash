"""Adapt the operator boundaries of a supported private CPro model for T64 Prefill.

Source operations, model weights and layer definitions come from the supplied local
source at prepare time. The selected fixture determines the GDR chunk width.
"""
import ast
from hashlib import sha256


# Two supported inputs: original source and the earlier full-model CPro adaptation.
_ORIGINAL = 'bd716718818b6b152ece5a2e345a5b674f091f53ba82059c86cd37060511f07d'
_ADAPTED = '389d6dd4d523a683d378e89a30dd5e369322bf6505fccda8702a4e2d28a98ba7'


def _one(matches, label):
    values = list(matches)
    if len(values) != 1:
        raise ValueError(f'Expected one {label}, found {len(values)}')
    return values[0]


def _definition(parent, name):
    return _one((n for n in parent.body if getattr(n, 'name', None) == name), name)


def _call(parent, name):
    return _one((c for c in ast.walk(parent) if isinstance(c, ast.Call)
                 and ast.unparse(c.func) == name), name)


def _assignment(parent, target):
    return _one((s for s in parent.body if isinstance(s, ast.Assign)
                 and ast.unparse(s.targets[0]) == target), target)


def _expr(code):
    return ast.parse(code, mode='eval').body


def use_position_gathers(tree, class_name, tables, index):
    """Lower axis-1 row selections for the prepared nonnegative INT32 positions."""
    forward = _definition(_definition(tree, class_name), 'forward')
    parents = [node for node in ast.walk(forward) if isinstance(getattr(node, 'value', None), ast.Subscript)]
    for table in tables:
        selection = f'{table}[:, {index}, :]'
        parent = _one((node for node in parents if ast.unparse(node.value) == selection), selection)
        # X90's frontend accepts FP32 gather data. Select from the already
        # rounded values, then return rows in the source table's dtype.
        parent.value = _expr(f'torch.index_select({table}.float(), 1, {index}.int()).to({table}.dtype)')


def _convert(node, method, arguments=()):
    return ast.Call(func=ast.Attribute(value=node, attr=method, ctx=ast.Load()),
                    args=list(arguments), keywords=[])


def restore_qlinear_scales(tree):
    """Undo only our two weight-scale casts when reusing a prepared CPro source."""
    cls = _definition(tree, 'QLinear')
    init = _definition(cls, '__init__')
    scale = _one((call for call in ast.walk(init) if isinstance(call, ast.Call)
                  and ast.unparse(call.func) == 'self.register_buffer'
                  and isinstance(call.args[0], ast.Constant) and call.args[0].value == 'scale'),
                 'registered scale')
    if isinstance(scale.args[1], ast.Call) and ast.unparse(scale.args[1]).endswith('.to(torch.float16)'):
        scale.args[1] = scale.args[1].func.value
    scale_arg = _call(_definition(cls, 'forward'), 'torch_npu.npu_quant_matmul').args[2]
    if isinstance(scale_arg, ast.Call) and isinstance(scale_arg.func, ast.Attribute):
        source = scale_arg.func.value
        if isinstance(source, ast.Call) and ast.unparse(source).endswith('.to(torch.float16)'):
            scale_arg.func.value = source.func.value


def _rotate_fp32(tree, original):
    if original:
        rope = _definition(tree, 'apply_rotary_pos_emb')
        precision = ast.parse(
            'q_dtype = torch.promote_types(torch.promote_types(q.dtype, cos.dtype), sin.dtype)\n'
            'k_dtype = torch.promote_types(torch.promote_types(k.dtype, cos.dtype), sin.dtype)\n'
            'q, k = q.float(), k.float()').body
        first = rope.body.index(_assignment(rope, 'cos'))
        rope.body[first:first] = precision
        for name in ('cos', 'sin'):
            operation = _assignment(rope, name).value
            operation.func.value = _convert(operation.func.value, 'float')
        operands = rope.body[-1].value.elts
        for i, dtype in enumerate(('q_dtype', 'k_dtype')):
            operands[i] = _convert(operands[i], 'to', [_expr(dtype)])
    forward = _definition(_definition(tree, 'Qwen3_5MoeAttention'), 'forward')
    rotation = _call(forward, 'apply_rotary_pos_emb')
    rotation.args = [_convert(value, 'float') for value in rotation.args]
    site = _one((s for s in forward.body if rotation in ast.walk(s)), 'rotary assignment')
    casts = []
    for name in site.targets[0].elts:
        casts.extend(ast.parse(f'{name.id} = {name.id}.to(hidden_states.dtype)').body)
    where = forward.body.index(site) + 1
    forward.body[where:where] = casts


def physical_conv_history(forward):
    update = _call(forward, 'torch.ops.my_ops.linear_attention_convstate_update')
    site = _one((s for s in forward.body if getattr(s, 'value', None) is update), 'history update')
    projected, old = map(ast.unparse, update.args[:2])
    # Match the original wrapper’s physical-tail ConvState writeback.
    history, window = [target.id for target in site.targets[0].elts]
    index = forward.body.index(site)
    forward.body[index:index+1] = ast.parse(
        f'conv_width = {old}.shape[2]\n'
        f'joined = torch.cat(({old}, {projected}.transpose(1, 2)), dim=2)\n'
        f'{history} = joined[:, :, -conv_width:].contiguous()\n'
        f'{window} = joined[:, :, -(seq_len + conv_width - 1):].contiguous()').body


def explicit_gdn_conv2d(forward, *, original=True):
    """Represent the depthwise time convolution as NCHW with singleton height."""
    conv = _assignment(forward, 'conv_out')
    if original:
        call = _call(conv, 'F.conv1d')
        call.func.attr = 'conv2d'
        call.args[0] = _convert(call.args[0], 'reshape', [
            _expr('batch_size'), _expr('self.conv_dim'), ast.Constant(1), _expr('-1')])
        call.args[1] = _convert(call.args[1], 'unsqueeze', [ast.Constant(2)])
        conv.value = _convert(call, 'squeeze', [ast.Constant(2)])
    reshape = _call(conv, 'F.conv2d').args[0]
    x = reshape.func.value
    if not ast.unparse(x).endswith('.to(self.conv1d.weight.dtype)'):
        reshape.func.value = _convert(x, 'to', [_expr('self.conv1d.weight.dtype')])


def adapt_four_layer_convolution(source, *, explicit_conv2d, physical_convstate):
    """Adapt the history and convolution independently in the selected CPro source."""
    from .decoder_four_source import preserve_source
    tree = ast.parse(source)
    forward = _definition(_definition(tree, 'Qwen3_5MoeGatedDeltaNet'), 'forward')
    if physical_convstate and any(isinstance(n, ast.Call) and
                                  ast.unparse(n.func) == 'torch.ops.my_ops.linear_attention_convstate_update'
                                  for n in ast.walk(forward)):
        physical_conv_history(forward)
    if explicit_conv2d:
        conv = _assignment(forward, 'conv_out')
        if not any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'F.conv2d' for n in ast.walk(conv)):
            explicit_gdn_conv2d(forward)
    return preserve_source(source, ast.unparse(ast.fix_missing_locations(tree)))


def _adapt_gdn(tree, original, single_token):
    cls = _definition(tree, 'Qwen3_5MoeGatedDeltaNet')
    if single_token:
        tree.body.insert(tree.body.index(cls), ast.ImportFrom(
            module='qwen35_moe_dflash.target.layer_probe.whole_gdn',
            names=[ast.alias(name='single_token_gdr')], level=0))
        init = _definition(cls, '__init__')
        index = init.body.index(_assignment(init, 'self.conv_kernel_size'))
        init.body[index:index] = ast.parse(
            "self.register_buffer('gdr_step_length', torch.tensor([1], dtype=torch.int16), persistent=False)").body
    forward = _definition(cls, 'forward')
    if original:
        physical_conv_history(forward)
    explicit_gdn_conv2d(forward, original=original)
    silu = _assignment(forward, 'qkv_conv')
    silu.value = _convert(silu.value, 'to', [_expr('mixed_qkv.dtype')])
    gdr = _call(forward, 'torch_npu.npu_chunk_gated_delta_rule')
    if not single_token:
        return
    gdr.func = _expr('single_token_gdr')
    options = {kw.arg: kw for kw in gdr.keywords}
    options['chunk_size'].value = ast.Constant(64)
    length = options['effective_length']
    length.arg, length.value = 'step_length', _expr('self.gdr_step_length')


def _adapt_experts(tree):
    apply = _definition(_definition(tree, 'CompressedTensorW4A8Int8MoEGMMMethod'), 'apply')
    branches = [s for s in apply.body if isinstance(s, ast.If)]
    quantize = _one((s for s in branches if isinstance(s.test, ast.Compare)
                     and isinstance(s.test.ops[0], ast.Is)), 'conditional expert quantization')
    quantize.orelse = ast.parse(
        "if x.dtype != torch.int8:\n    raise ValueError('Prequantized expert input must be INT8')\n"
        "x_quant = x").body
    reshape = _one((s for s in branches if any(isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == 'x_quant' for n in s.body)), 'quantized expert reshape')
    flattened = _assignment(reshape, 'x_quant')
    flattened.value.func = ast.Attribute(value=_expr('x_quant'), attr='reshape', ctx=ast.Load())

    forward = _definition(_definition(tree, 'Qwen3_5MoeSparseMoeBlock'), 'forward')
    shared_gate = _call(forward, 'self.shared_expert_gate')
    shared_gate.args[0] = _convert(shared_gate.args[0], 'to', [_expr('self.shared_expert_gate.weight.dtype')])
    shared = _one((s for s in forward.body if shared_gate in ast.walk(s)), 'shared expert weighting')
    shared.value.left = _convert(shared.value.left, 'to', [_expr('shared_expert_output.dtype')])

    router = _assignment(forward, 'router_logits')
    router_arg = ast.unparse(router.value.args[0])
    position = forward.body.index(router)
    forward.body[position:position] = ast.parse(
        f'router_input = {router_arg}.to(self.gate.weight.dtype) '
        f'if isinstance(self.gate, nn.Linear) else {router_arg}').body
    router.value.args[0] = _expr('router_input')


def stage_model(text, *, single_token=True, trace_chunk=False):
    """Return the adapted private model source, leaving the supplied source file intact."""
    digest = sha256(text.replace('\r\n', '\n').encode('utf-8')).hexdigest()
    if digest not in (_ORIGINAL, _ADAPTED):
        raise ValueError('The CPro model source differs from both reviewed baselines')
    tree = ast.parse(text)
    original = digest == _ORIGINAL
    if not original:
        restore_qlinear_scales(tree)
    _rotate_fp32(tree, original)
    _adapt_gdn(tree, original, single_token)
    if trace_chunk:
        from .segment_gdn_trace import instrument_chunk
        forward = _definition(_definition(tree, 'Qwen3_5MoeGatedDeltaNet'), 'forward')
        instrument_chunk(forward, _call(forward, 'torch_npu.npu_chunk_gated_delta_rule'))
    if not single_token:
        use_position_gathers(tree, 'Qwen3_5MoeRotaryEmbedding', ('self.cos_cached', 'self.sin_cached'), 'position_ids')
        # CacheUpdate mutates its input. Keep it internal to the nine-output model,
        # just as in the whole-attention probe, and return the two updated caches.
        forward = _definition(_definition(tree, 'Qwen3_5MoeAttention'), 'forward')
        for site in ast.walk(forward):
            if isinstance(site, ast.Call) and ast.unparse(site.func) == 'self.update':
                site.args[2] = _convert(site.args[2], 'clone')
    _adapt_experts(tree)
    norm = _definition(_definition(tree, 'Qwen3_5MoeRMSNorm'), 'forward')
    _call(norm, 'torch_npu.npu_rms_norm').args[1] = _expr('self.export_gamma')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'

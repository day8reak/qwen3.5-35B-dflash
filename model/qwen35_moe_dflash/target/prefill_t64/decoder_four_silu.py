"""Stage the native CPro model with the selected FP16 activation operators."""
from collections import Counter, defaultdict
from difflib import unified_diff
from math import prod
import re
from shutil import copyfile

from ..original_target_source import EXPORTER, SOURCE_FILES
from ..silu_patch_air import GE_TYPE

MODELING = 'models/modeling_qwen3_5_moe_hiai_cpro.py'


# Insert beside ChunkGatedDeltaRule and use the same npu Meta library and converter registry.
def activation_registration(name, ge_type):
    return f'''if not hasattr(torch.ops.npu, "{name}"):
    m_fragment.define("{name}(Tensor x) -> Tensor")


@torch.library.impl(m, "{name}")
def {name}_meta(x):
    if x.dtype != torch.float16:
        raise ValueError("{ge_type} expects FP16 input")
    ret = x.size()
    z = torch.empty(ret, dtype=torch.float16, device='meta')
    return torch.empty_like(z)


@torchair.register_fx_node_ge_converter(torch.ops.npu.{name}.default)
def convert_{name}(x: Tensor):
    z = torchair.ge.custom_op("{ge_type}", inputs={{"x": x}}, outputs=["z"])
    # The AIR writer may omit intermediate shapes. Keep the converter's observation separately.
    trace = os.environ.get("QWEN35_SILU_SHAPES")
    if trace:
        with open(trace, "a", encoding="utf-8") as stream:
            print(z.node.name, *x.meta.shape, file=stream)
    return z


'''

# Recognize previously generated editable source trees when preparing a four-layer run.
_OLD_SILU_HELPERS = '''def silu_patch(x):
    return torch.ops.npu.silu_patch(x.to(torch.float16).contiguous())


def silu_same_dtype(x):
    return silu_patch(x).to(x.dtype)


'''


_CONV_SILU = 'torch.ops.npu.silu_patch(conv_out.to(torch.float16).contiguous())'
_GATE_SILU = 'torch.ops.npu.silu_patch(gate.to(torch.float16).contiguous())'
_GATE = ('F.silu(gate.to(torch.float32))', _GATE_SILU + '.to(torch.float32)')
_CONV = ('F.silu(conv_out)', _CONV_SILU + '.to(conv_out.dtype)')
_SHARED_ACT = ('self.act_fn = ACT2FN[config.hidden_act]',
               "self.act_fn = None if config.hidden_act in ('silu', 'swish') else ACT2FN[config.hidden_act]")
_SHARED = ('return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))',
           '''gate = self.gate_proj(x)
        if self.act_fn is None:
            gate = torch.ops.npu.silu_patch(gate.to(torch.float16).contiguous()).to(gate.dtype)
        else:
            gate = self.act_fn(gate)
        return self.down_proj(gate * self.up_proj(x))''')
_ROUTED = ('self.act_fn(x1) * x2',
           'torch.ops.npu.silu_patch(x1.to(torch.float16).contiguous()).to(x1.dtype) * x2')
MOE_SILU_CALLS = (_SHARED_ACT, _SHARED, _ROUTED)
SILU_CALLS = (_GATE, _CONV, *MOE_SILU_CALLS)
SOFTPLUS_CALL = ('F.softplus(a.float() + self.dt_bias)',
                 'torch.ops.npu.softplus_patch((a.float() + self.dt_bias).to(torch.float16).contiguous()).float()')
CONV_HELPER_CALL = ('F.silu(out[:, :, -seq_len:])',
                    'torch.ops.npu.silu_patch(out[:, :, -seq_len:].to(torch.float16).contiguous()).to(out.dtype)')


def inline_old_silu(model):
    """Inline this adapter's old wrappers in the export copy of an editable model."""
    if _OLD_SILU_HELPERS not in model:
        return model
    model = model.replace(_OLD_SILU_HELPERS, '')
    for old, new in (('silu_patch(gate).to(torch.float32)', _GATE[1]),
                     ('silu_same_dtype(conv_out)', _CONV[1]),
                     ('qkv_conv = silu_patch(conv_out)', 'qkv_conv = ' + _CONV_SILU),
                     ('silu_same_dtype(out[:, :, -seq_len:])', CONV_HELPER_CALL[1])):
        model = model.replace(old, new)
    for wrapped, assignment, call in (
            ("self.act_fn = silu_same_dtype if config.hidden_act in ('silu', 'swish') else ACT2FN[config.hidden_act]",
             _SHARED_ACT[1], _SHARED),
            ('self.act_fn = silu_same_dtype', "self.act_fn = ACT2FN['silu']", _ROUTED)):
        if wrapped in model:
            if model.count(call[0]) != 1:
                raise ValueError(f'Cannot inline SiLU in the editable model: expected one {call[0]!r}')
            model = model.replace(wrapped, assignment).replace(*call)
    import ast
    if any(isinstance(node, ast.Name) and node.id in ('silu_patch', 'silu_same_dtype')
           for node in ast.walk(ast.parse(model))):
        raise ValueError('The editable model has an additional call to a legacy SiLU wrapper')
    return model


def native_conv_silu(model):
    model = model.replace(_CONV[1], _CONV[0]).replace(CONV_HELPER_CALL[1], CONV_HELPER_CALL[0])
    # The aligned GDR converter requires half Q/K/V even when conv_out is FP32.
    return model.replace(_CONV_SILU, 'F.silu(conv_out).to(torch.float16)')


def select_gdn_silu(model, site):
    for original, patched in MOE_SILU_CALLS:
        model = model.replace(patched, original)
    if site == 'conv':
        return model.replace(_GATE[1], _GATE[0])
    if site == 'gate':
        return native_conv_silu(model)
    return model


def token_view_rms_gate(model):
    """Present each T64 token's 32 contiguous 128-wide heads to the FP16 kernel together."""
    import ast
    from .decoder_four_source import preserve_source
    from .source_model import _call, _definition

    tree = ast.parse(model)
    forward = _definition(_definition(tree, 'Qwen3_5MoeRMSNormGated'), 'forward')
    kernel = _call(forward, 'torch.ops.npu.silu_patch')
    value, = kernel.args

    class TokenView(ast.NodeTransformer):
        def visit_Call(self, node):
            if node is kernel:
                expr = f'torch.ops.npu.silu_patch(({ast.unparse(value)}).reshape(-1, 4096)).reshape(-1, 128)'
                return ast.parse(expr, mode='eval').body
            return self.generic_visit(node)

    TokenView().visit(forward)
    return preserve_source(model, ast.unparse(tree))


def omit_gate_output_cast(model, *, select_layers=False):
    import ast
    from .decoder_four_source import preserve_source
    from .source_model import _call, _definition

    tree = ast.parse(model)
    norm = _definition(tree, 'Qwen3_5MoeRMSNormGated')
    forward = _definition(norm, 'forward')
    gate = _call(forward, 'torch.ops.npu.silu_patch')

    class OmitGateCast(ast.NodeTransformer):
        def visit_Call(self, node):
            if node is gate and select_layers:
                native = ast.Call(func=ast.parse('F.silu', mode='eval').body, args=gate.args, keywords=[])
                return ast.IfExp(test=ast.parse('self.use_silu_patch', mode='eval').body, body=gate, orelse=native)
            func = node.func
            if isinstance(func, ast.Attribute) and func.value is gate and func.attr in ('float', 'to'):
                call = ast.unparse(node)
                suffix = call[len(ast.unparse(gate)):]
                if suffix not in ('.float()', '.to(torch.float32)', '.to(dtype=torch.float32)'):
                    raise ValueError(f'Unexpected GDN RMS gate cast in {norm.name}.forward: {call}')
                return self.visit(gate)
            return self.generic_visit(node)

    OmitGateCast().visit(forward)
    return preserve_source(model, ast.unparse(tree))


def set_rms_gate_layers(model, selected):
    """Set a static boolean on each original RMS gate before TorchAir traces the wrapper."""
    from models.modeling_qwen3_5_moe_hiai_cpro import Qwen3_5MoeRMSNormGated

    gates = {}
    for name, module in model.named_modules():
        if isinstance(module, Qwen3_5MoeRMSNormGated):
            match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
            if match is None or int(match[1]) in gates:
                raise ValueError(f'Cannot bind RMS gate to one decoder layer: {name}')
            gates[int(match[1])] = module
    if set(gates) != {0, 1, 2}:
        raise ValueError(f'Expected RMS gates on L0-L2, found {sorted(gates)}')
    for layer, module in sorted(gates.items()):
        module.use_silu_patch = layer in selected
        print(f'[GDN RMS L{layer}] {"SiluPatch" if module.use_silu_patch else "F.silu"} (FP16)', flush=True)


def configure_gate_export(exporter, selected):
    import ast
    from .source_model import _call

    call = _call(ast.parse(exporter), 'torchair.dynamo_export')
    wrapper, = [kw.value.id for kw in call.keywords if kw.arg == 'model' and isinstance(kw.value, ast.Name)]
    lines = exporter.splitlines(keepends=True)
    at = call.lineno - 1
    indent = lines[at][:call.col_offset]
    if indent.strip():
        raise ValueError('Expected a standalone torchair.dynamo_export call')
    lines[at:at] = [
        indent + 'from qwen35_moe_dflash.target.prefill_t64.decoder_four_silu import set_rms_gate_layers\n',
        f'{indent}set_rms_gate_layers({wrapper}, {selected!r})\n',
    ]
    return ''.join(lines)


def restore_native_activations(exporter, model, *, keep_silu=False, keep_softplus=False):
    """Undo only this adapter's activation edits in a copy of the selected aligned sources."""
    if not keep_softplus:
        exporter = exporter.replace(activation_registration('softplus_patch', 'SoftplusPatch'), '')
        original, patched = SOFTPLUS_CALL
        model = model.replace(patched, original)
        for symbol in ('softplus_patch', 'SoftplusPatch'):
            if symbol in model or symbol in exporter:
                raise ValueError(f'Unsupported activation edit in selected CPro source: {symbol} remains')
    if keep_silu:
        return exporter, model
    exporter = exporter.replace(activation_registration('silu_patch', GE_TYPE), '')
    model = native_conv_silu(model)
    for original, patched in (_GATE, *MOE_SILU_CALLS):
        model = model.replace(patched, original)
    for symbol in ('silu_patch', 'silu_same_dtype', GE_TYPE):
        if symbol in model or symbol in exporter:
            raise ValueError(f'Unsupported activation edit in selected CPro source: {symbol} remains')
    return exporter, model


def stage(source, dest, *, aligned=False, l0=False, layer=None, head=True, layers=4, native_convstate=False):
    (dest / 'models').mkdir(parents=True)
    for name in SOURCE_FILES:
        copyfile(source / name, dest / name)
    init = source / 'models/__init__.py'
    if init.is_file():
        copyfile(init, dest / 'models/__init__.py')
    else:
        (dest / 'models/__init__.py').touch()
    if l0:
        from .decoder_four_source import preserve_source
        from .source_model import stage_model

        original = (source / MODELING).read_text()
        (dest / MODELING).write_text(preserve_source(original, stage_model(original, single_token=False)))
    gdr_meta = '@torch.library.impl(m, "npu_chunk_gated_delta_rule")'
    registration = activation_registration("silu_patch", GE_TYPE)
    if aligned or l0:
        registration += activation_registration("softplus_patch", "SoftplusPatch")
    edits = {
        EXPORTER: [(gdr_meta, registration + gdr_meta)],
        MODELING: list(SILU_CALLS),
    }
    if aligned or l0:
        edits[MODELING].append(SOFTPLUS_CALL)
    for name, replacements in edits.items():
        path = dest / name
        text = path.read_text()
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError(f'Expected one {old!r} in original {name}')
            text = text.replace(old, new)
        if name == MODELING:
            # Some source revisions also include a standalone causal-convolution helper.
            text = text.replace(*CONV_HELPER_CALL)
        if name == MODELING and aligned and not l0:
            import ast
            from .decoder_four_source import preserve_source
            from .source_model import (_adapt_experts, _definition, _register_fp16_scale, _rotate_fp32,
                                       explicit_gdn_conv2d, physical_conv_history, use_position_gathers)

            # Keep the native wrapper and readout.
            # Feed GDR the FP16 activation that the custom SiLU already produced.
            changes = [('qkv_conv = ' + _CONV[1], 'qkv_conv = ' + _CONV_SILU)]
            if native_convstate or layers != 40:
                changes.append(('F.conv1d(padded, self.conv1d.weight,',
                                'F.conv1d(padded.to(self.conv1d.weight.dtype), self.conv1d.weight,'))
            for old, new in changes:
                if text.count(old) != 1:
                    raise ValueError(f'Expected one {old!r} in native GDN')
                text = text.replace(old, new)
            tree = ast.parse(text)
            if layers == 40 and not native_convstate:
                gdn = _definition(_definition(tree, 'Qwen3_5MoeGatedDeltaNet'), 'forward')
                physical_conv_history(gdn)
                explicit_gdn_conv2d(gdn)
            _register_fp16_scale(tree)
            _rotate_fp32(tree, original=True)
            use_position_gathers(tree, 'Qwen3_5MoeRotaryEmbedding',
                                 ('self.cos_cached', 'self.sin_cached'), 'position_ids')
            _adapt_experts(tree)
            text = preserve_source(text, ast.unparse(ast.fix_missing_locations(tree)))
        path.write_text(text)
    changed = [EXPORTER, MODELING]
    if l0:
        from .l0_split import expose_l0
        index = layer or 0
        expose_l0(dest, layer=index, head=head)
        changed.append('models/export_model_wrapper_qwen3_5_moe.py')
    if l0 or aligned:
        from .decoder_four_source import preserve_source
        from .source_exporter import stage_prefill_exporter, stage_utils
        path = dest / EXPORTER
        text = path.read_text()
        count = int(index % 4 != 3) if l0 else layers - layers // 4
        path.write_text(preserve_source(text, stage_prefill_exporter(text, gdn_layers=count, split=l0)))
        utils = dest / 'utils.py'
        utils.write_text(stage_utils(utils.read_text()))
        changed.append('utils.py')
    diff = []
    for name in changed:
        before = (source / name).read_text().splitlines(True)
        diff.extend(unified_diff(before, (dest / name).read_text().splitlines(True),
                                 fromfile='original/' + name, tofile='modified/' + name))
    (dest / 'source.diff').write_text(''.join(diff))
    return dest / EXPORTER


def vector_length(op, producers):
    """Read the length in historical, flattened FP32-output diagnostic AIRs."""
    inputs = [edge for edge in op.input if edge and not edge.endswith(':-1')]
    if len(inputs) != 1 or len(op.input_desc) != 1 or len(op.output_desc) != 1:
        raise ValueError(f'{op.name}: expected one input and one output')
    x, z = op.input_desc[0], op.output_desc[0]
    if (x.dtype, z.dtype) != (2, 1):  # GE proto DT_FLOAT16 / DT_FLOAT
        raise ValueError(f'{op.name}: expected FP16 x -> FP32 z, got {x.dtype} -> {z.dtype}')
    shapes = {tuple(d.shape.dim) for d in (x, z) if d.shape.dim}
    if any(len(shape) != 1 or shape[0] == 0 for shape in shapes):
        raise ValueError(f'{op.name}: expected vector ports, got {shapes}')
    lengths = {shape[0] for shape in shapes if shape[0] > 0}
    # TorchAir can omit intermediate dimensions even with dynamic=False.
    # The model calls view(x.numel()), which leaves a positive one-element Const.
    source = producers[inputs[0].rsplit(':', 1)[0]]
    if source.type == 'Reshape' and len(source.input) >= 2:
        shape_op = producers[source.input[1].rsplit(':', 1)[0]]
        if shape_op.type == 'Const' and 'value' in shape_op.attr:
            tensor = shape_op.attr['value'].t
            types = tensor.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
            for dtype, size in (('DT_INT32', 4), ('DT_INT64', 8)):
                if tensor.desc.dtype == types[dtype].number and len(tensor.data) == size:
                    n = int.from_bytes(tensor.data, 'little', signed=True)
                    if n > 0:
                        lengths.add(n)
    if len(lengths) != 1:
        raise ValueError(f'{op.name}: missing or inconsistent static vector lengths {lengths}')
    return lengths.pop()


def gate_product(graph, gate):
    """Resolve an FP16 activation through its widening cast to an FP32 product."""
    cast, = [op for op in graph.op if op.type == 'Cast' and list(op.input) == [gate]]
    if ([d.dtype for d in cast.input_desc], [d.dtype for d in cast.output_desc]) != ([2], [1]):
        raise ValueError(f'{cast.name}: expected FP16 gate to FP32')
    value = cast.name + ':0'
    mul, = [op for op in graph.op if op.type == 'Mul' and value in op.input]
    if [d.dtype for d in mul.input_desc] != [1, 1] or [d.dtype for d in mul.output_desc] != [1]:
        raise ValueError(f'{mul.name}: activation product must be FP32')
    norm, = [e for e in mul.input if e and e != value and not e.endswith(':-1')]
    return mul, norm


def rms_gate_path(graph, edge):
    """Follow the custom gate's data output across its views and FP32 cast to the RMS product."""
    chain = []
    widened = False
    while True:
        uses = [op for op in graph.op if (op.type == 'Mul' and edge in op.input)
                or (op.type in ('Reshape', 'View', 'Identity', 'TensorMove', 'Cast') and op.input and op.input[0] == edge)]
        if len(uses) != 1 or uses[0] in chain:
            raise ValueError(f'{edge}: expected one data path to the RMS gate product, found {[op.name for op in uses]}')
        op = uses[0]
        chain.append(op)
        if op.type == 'Mul':
            if not widened or [d.dtype for d in op.input_desc] != [1, 1] or [d.dtype for d in op.output_desc] != [1]:
                raise ValueError(f'{op.name}: expected FP32 multiplication after the FP16 gate')
            peer, = [e for e in op.input if e and e != edge and not e.endswith(':-1')]
            return chain, peer
        if op.type == 'Cast':
            if widened or (op.input_desc[0].dtype, op.output_desc[0].dtype) != (2, 1):
                raise ValueError(f'{op.name}: expected one FP16 to FP32 gate cast')
            widened = True
        edge = op.name + ':0'


def inspect_gdn_convs(graph, count, rows):
    nodes = {op.name: op for op in graph.op}
    convs = [op for op in graph.op if op.type == 'Conv2D']
    if len(convs) != count:
        raise ValueError(f'Expected {count} GDN Conv2D nodes, found {len(convs)}')
    expected = ((1, 8192, 1, rows + 3), (8192, 1, 1, 4))
    observed = 0
    for conv in convs:
        for i, dims in enumerate(expected):
            name, slot = conv.input[i].rsplit(':', 1)
            source = nodes[name]
            ports = [source.output_desc[int(slot)]] if int(slot) < len(source.output_desc) else []
            ports.extend(conv.input_desc[i:i+1])
            shapes = [tuple(d.shape.dim) for d in ports if d.shape.dim and tuple(d.shape.dim) != (-2,)]
            # A reshape target also records the rank when TorchAir omits the edge descriptor.
            if source.type == 'Reshape' and len(source.input) >= 2:
                target = nodes[source.input[1].rsplit(':', 1)[0]]
                if target.type == 'Const' and 'value' in target.attr:
                    t = target.attr['value'].t
                    enums = t.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number
                    width = {'DT_INT32': 4, 'DT_INT64': 8}.get(enums[t.desc.dtype].name)
                    if width and t.data:
                        shapes.append(tuple(int.from_bytes(t.data[j:j+width], 'little', signed=True)
                                            for j in range(0, len(t.data), width)))
            for shape in shapes:
                if len(shape) != 4 or any(x >= 0 and x != y for x, y in zip(shape, dims)):
                    raise ValueError(f'{conv.name} input[{i}] from {name}: found {shape}, expected {dims}')
            observed += bool(shapes)
    print(f'GDN Conv2D: {count} nodes; AIR records dimensions for {observed}/{2*count} operands', flush=True)
    return dict(nodes=count, operands_with_shapes=observed, expected_shapes=expected)


def activation_port_shape(desc):
    """Keep the GE descriptor and the FX annotation as independent observations."""
    meta = desc.attr['_meta'].s.decode(errors='replace') if '_meta' in desc.attr else ''
    match = re.search(r'shape\s*=\s*torch\.Size\(\[\s*((?:-?\d+\s*,\s*)*-?\d+)?\s*\]\)', meta)
    fx_dims = [int(d) for d in re.findall(r'-?\d+', match.group(0))] if match else None
    fx_tensor = desc.attr['_fx_tensor_name'].s.decode(errors='replace') if '_fx_tensor_name' in desc.attr else None
    return dict(ge_dims=list(desc.shape.dim), fx_dims=fx_dims, fx_tensor=fx_tensor)


def inspect_activation_air(path, shape_trace, *, layers=4, full_only=False, silu=True, softplus=False, rows=64,
                           check_convs=False, gdn_conv="conv2d", moe_silu=True, gdn_silu_site=None,
                           all_silu_no_gate_cast=False, rms_gate_count=None, rms_silu_token_view=False):
    """Cross-check static FX input shapes and serialized AIR edges, without modifying the graph."""
    from .decode_cache_control import graph_message

    _, graph = graph_message(path.read_bytes(), GE_TYPE if silu else 'SoftplusPatch')
    traced = {}
    for line in shape_trace.read_text().splitlines():
        name, *dims = line.split()
        shape = tuple(map(int, dims))
        if name in traced and traced[name] != shape:
            raise ValueError(f'{name}: inconsistent traced shapes {traced[name]} and {shape}')
        traced[name] = shape
    linear = 0 if full_only else layers - layers // 4
    expert_layers = layers if moe_silu else 0
    gate_count = 0 if gdn_silu_site == 'conv' else linear
    if rms_gate_count is not None:
        gate_count = rms_gate_count
    gate_shape = (rows, 4096) if rms_silu_token_view else (rows * 32, 128)
    expected = {GE_TYPE: Counter({(1, 8192, rows): 0 if gdn_silu_site == 'gate' else linear,
                                  gate_shape: gate_count,
                                  (rows, 512): expert_layers, (rows * 8, 512): expert_layers}),
                "SoftplusPatch": Counter({(1, rows, 32): linear if softplus else 0})}
    if not silu:
        expected[GE_TYPE].clear()
    ops = {op.name: op for op in graph.op if op.type in expected}
    if set(ops) != set(traced):
        raise ValueError(f'Activation AIR and converter names differ: {sorted(set(ops) ^ set(traced))}')
    for op_type, shapes in expected.items():
        counts = Counter(traced[name] for name, op in ops.items() if op.type == op_type)
        if counts != +shapes:
            raise ValueError(f'{op_type}: expected {dict(+shapes)} native shapes, found {dict(counts)}')

    readers = defaultdict(list)
    for consumer in graph.op:
        for slot, edge in enumerate(consumer.input):
            readers[edge].append((consumer, slot))

    def dtype(desc):
        enum = desc.DESCRIPTOR.file.enum_types_by_name['DataType']
        return enum.values_by_number[desc.dtype].name if desc.dtype else 'unrecorded'

    def read_edges(edge, output_dtype):
        rows = []
        for consumer, slot in readers[edge]:
            desc = consumer.input_desc[slot] if slot < len(consumer.input_desc) else None
            if output_dtype and desc is not None and desc.dtype not in (0, output_dtype):
                raise ValueError(f'{edge} dtype={output_dtype} feeds {consumer.name}:{slot} declared {dtype(desc)}')
            rows.append(dict(name=consumer.name, op=consumer.type, slot=slot,
                             input_dtype=dtype(desc) if desc is not None else 'unrecorded',
                             output_dtypes=[dtype(d) for d in consumer.output_desc]))
        return rows

    records = {op_type: [] for op_type in expected}
    by_name = {op.name: op for op in graph.op}
    for name, op in ops.items():
        x, = op.input_desc
        z, = op.output_desc
        shape = traced[name]
        consumers = read_edges(name + ':0', z.dtype)
        if (x.dtype, z.dtype) != (2, 2) or not consumers:
            raise ValueError(f'{name}: expected a connected FP16 -> FP16 {op.type}')
        if op.type == 'SoftplusPatch':
            # The decay product is computed in FP32 on both sides of this patch.
            cast_in = by_name[op.input[0].rsplit(':', 1)[0]]
            if (cast_in.type != 'Cast' or [d.dtype for d in cast_in.input_desc] != [1]
                    or [d.dtype for d in cast_in.output_desc] != [2]):
                raise ValueError(f'{name}: expected a FP32 to FP16 cast of biased_a')
            gate_product(graph, name + ':0')
        elif shape == gate_shape:
            if rms_silu_token_view:
                chain, _ = rms_gate_path(graph, name + ':0')
                print(f'[GDN 门控 AIR] {name} {list(shape)} → ' + ' → '.join(n.name for n in chain), flush=True)
            elif all_silu_no_gate_cast:
                # Type promotion at the unchanged multiplication can still generate a Cast.
                targets = ', '.join(f"{r['name']} ({r['op']}, {r['input_dtype']} → {', '.join(r['output_dtypes'])})"
                                    for r in consumers)
                print(f'[GDN 门控 AIR] {name} (FP16) → {targets}', flush=True)
            else:
                gate_product(graph, name + ':0')
        for row in consumers:
            if row['op'] == 'Cast':
                cast = by_name[row['name']]
                row['after_cast'] = read_edges(cast.name + ':0', cast.output_desc[0].dtype)
        for desc in (x, z):
            dims = tuple(desc.shape.dim)
            if dims and dims != (-2,):  # GE's unknown-rank marker
                if len(dims) != len(shape) or any(d != -1 and d != actual for d, actual in zip(dims, shape)):
                    raise ValueError(f'{name}: AIR port {dims} differs from traced shape {shape}')
        ports = dict(x=activation_port_shape(x), z=activation_port_shape(z))
        records[op.type].append(dict(name=name, shape=list(shape), numel=prod(shape), consumers=consumers,
                                     air_ports=ports))
        print(f'{name}: converter input={shape}; FP16 x={ports["x"]} -> FP16 z={ports["z"]}; consumers={consumers}', flush=True)
    result = {}
    if check_convs:
        if gdn_conv == 'taps':
            from .gdn_taps import inspect_taps
            conv_silus = [op.name for op in ops.values() if op.type == GE_TYPE and traced[op.name] == (1, 8192, rows)]
            result['gdn_convs'] = inspect_taps(graph, conv_silus, linear)
        else:
            result['gdn_convs'] = inspect_gdn_convs(graph, linear, rows)
    for name, op_type, enabled in (("silu_patch", GE_TYPE, silu), ("softplus_patch", "SoftplusPatch", softplus)):
        if not enabled:
            continue
        nodes = records[op_type]
        print(f'{op_type}: {len(nodes)} nodes, native shapes {dict(+expected[op_type])}', flush=True)
        result[name] = dict(type=op_type, nodes=nodes, input='x:float16', output='z:float16',
                            shape_source='converter')
    return result

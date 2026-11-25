"""Stage the native CPro model with the selected FP16 activation operators."""
from collections import Counter, defaultdict
from difflib import unified_diff
from math import prod
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

MODEL_HELPERS = '''def silu_patch(x):
    return torch.ops.npu.silu_patch(x.to(torch.float16).contiguous())


def silu_same_dtype(x):
    return silu_patch(x).to(x.dtype)


'''


def stage(source, dest, *, aligned=False, l0=False, layer=None, head=True):
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
        MODELING: [
            ('def rotate_half(', MODEL_HELPERS + 'def rotate_half('),
            ('F.silu(gate.to(torch.float32))', 'silu_patch(gate).to(torch.float32)'),
            ('F.silu(conv_out)', 'silu_same_dtype(conv_out)'),
            ('self.act_fn = ACT2FN[config.hidden_act]',
             "self.act_fn = silu_same_dtype if config.hidden_act in ('silu', 'swish') else ACT2FN[config.hidden_act]"),
            ("self.act_fn = ACT2FN['silu']", 'self.act_fn = silu_same_dtype'),
        ],
    }
    if aligned or l0:
        edits[MODELING].append((
            'F.softplus(a.float() + self.dt_bias)',
            'torch.ops.npu.softplus_patch((a.float() + self.dt_bias).to(torch.float16).contiguous()).float()',
        ))
    for name, replacements in edits.items():
        path = dest / name
        text = path.read_text()
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError(f'Expected one {old!r} in original {name}')
            text = text.replace(old, new)
        if name == MODELING:
            # Some source revisions also include a standalone causal-convolution helper.
            text = text.replace('F.silu(out[:, :, -seq_len:])', 'silu_same_dtype(out[:, :, -seq_len:])')
        if name == MODELING and aligned and not l0:
            import ast
            from .decoder_four_source import preserve_source
            from .source_model import _adapt_experts, _register_fp16_scale, _rotate_fp32, use_position_gathers

            # Keep native state updates, ConvState control and readout.
            # Feed GDR the FP16 activation that the custom SiLU already produced.
            for old, new in (
                ('F.conv1d(padded, self.conv1d.weight,',
                 'F.conv1d(padded.to(self.conv1d.weight.dtype), self.conv1d.weight,'),
                ('qkv_conv = silu_same_dtype(conv_out)', 'qkv_conv = silu_patch(conv_out)'),
            ):
                if text.count(old) != 1:
                    raise ValueError(f'Expected one {old!r} in native GDN')
                text = text.replace(old, new)
            tree = ast.parse(text)
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
        count = int(index % 4 != 3) if l0 else 3
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


def inspect_activation_air(path, shape_trace, *, layers=4, full_only=False, softplus=False):
    """Cross-check static FX input shapes and serialized AIR edges, without modifying the graph."""
    from .decode_cache_control import graph_message

    _, graph = graph_message(path.read_bytes(), GE_TYPE)
    traced = {}
    for line in shape_trace.read_text().splitlines():
        name, *dims = line.split()
        shape = tuple(map(int, dims))
        if name in traced and traced[name] != shape:
            raise ValueError(f'{name}: inconsistent traced shapes {traced[name]} and {shape}')
        traced[name] = shape
    linear = 0 if full_only else {1: 1, 4: 3}[layers]
    expected = {GE_TYPE: Counter({(1, 8192, 64): linear, (2048, 128): linear,
                                  (64, 512): layers, (512, 512): layers}),
                "SoftplusPatch": Counter({(1, 64, 32): linear if softplus else 0})}
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
        elif shape == (2048, 128):
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
        records[op.type].append(dict(name=name, shape=list(shape), numel=prod(shape), consumers=consumers))
        print(f'{name}: FP16{shape} -> FP16{shape}; consumers={consumers}', flush=True)
    result = {}
    for name, op_type in (("silu_patch", GE_TYPE), ("softplus_patch", "SoftplusPatch")):
        if name == "softplus_patch" and not softplus:
            continue
        nodes = records[op_type]
        print(f'{op_type}: {len(nodes)} nodes, native shapes {dict(+expected[op_type])}', flush=True)
        result[name] = dict(type=op_type, nodes=nodes, input='x:float16', output='z:float16',
                            shape_source='converter')
    return result

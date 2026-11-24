"""Integrate SiluPatch at the original CPro model's SiLU call sites."""
from collections import Counter
from difflib import unified_diff
from shutil import copyfile

from ..original_target_source import EXPORTER, SOURCE_FILES
from ..silu_patch_air import GE_TYPE

MODELING = 'models/modeling_qwen3_5_moe_hiai_cpro.py'


# Insert beside ChunkGatedDeltaRule and use the same npu Meta library and converter registry.
REGISTRATION = '''if not hasattr(torch.ops.npu, "silu_patch"):
    m_fragment.define("silu_patch(Tensor x) -> Tensor")


@torch.library.impl(m, "silu_patch")
def silu_patch_meta(x):
    if x.ndim != 1 or x.dtype != torch.float16:
        raise ValueError("SiluPatch expects a one-dimensional FP16 input")
    return torch.empty_like(x, dtype=torch.float32)


@torchair.register_fx_node_ge_converter(torch.ops.npu.silu_patch.default)
def convert_silu_patch(x: Tensor):
    return torchair.ge.custom_op("SiluPatch", inputs={"x": x}, outputs=["z"])


'''

MODEL_HELPERS = '''def silu_patch(x):
    flat = x.to(torch.float16).contiguous().view(x.numel())
    return torch.ops.npu.silu_patch(flat).reshape(x.shape)


def silu_same_dtype(x):
    return silu_patch(x).to(x.dtype)


'''


def stage(source, dest, *, aligned=False):
    (dest / 'models').mkdir(parents=True)
    for name in SOURCE_FILES:
        copyfile(source / name, dest / name)
    init = source / 'models/__init__.py'
    if init.is_file():
        copyfile(init, dest / 'models/__init__.py')
    else:
        (dest / 'models/__init__.py').touch()
    gdr_meta = '@torch.library.impl(m, "npu_chunk_gated_delta_rule")'
    edits = {
        EXPORTER: [(gdr_meta, REGISTRATION + gdr_meta)],
        MODELING: [
            ('def rotate_half(', MODEL_HELPERS + 'def rotate_half('),
            ('F.silu(gate.to(torch.float32))', 'silu_patch(gate)'),
            ('F.silu(conv_out)', 'silu_same_dtype(conv_out)'),
            ('self.act_fn = ACT2FN[config.hidden_act]',
             "self.act_fn = silu_same_dtype if config.hidden_act in ('silu', 'swish') else ACT2FN[config.hidden_act]"),
            ("self.act_fn = ACT2FN['silu']", 'self.act_fn = silu_same_dtype'),
        ],
    }
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
        if name == MODELING and aligned:
            import ast
            from .decoder_four_source import preserve_source
            from .source_model import _adapt_experts, _rotate_fp32, use_position_gathers

            # These reviewed internal fixes work with the native entry, state updates
            # and readout. Keep its quantized loader and ConvState control wire.
            tree = ast.parse(text)
            _rotate_fp32(tree, original=True)
            use_position_gathers(tree, 'Qwen3_5MoeRotaryEmbedding',
                                 ('self.cos_cached', 'self.sin_cached'), 'position_ids')
            _adapt_experts(tree)
            text = preserve_source(text, ast.unparse(ast.fix_missing_locations(tree)))
        path.write_text(text)
    diff = []
    for name in (EXPORTER, MODELING):
        before = (source / name).read_text().splitlines(True)
        diff.extend(unified_diff(before, (dest / name).read_text().splitlines(True),
                                 fromfile='original/' + name, tofile='modified/' + name))
    (dest / 'source.diff').write_text(''.join(diff))
    return dest / EXPORTER


def vector_length(op, producers):
    """Read the native AIR descriptors or the literal used to flatten the input."""
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


def inspect_silu_air(path):
    from .decode_cache_control import graph_message

    _, graph = graph_message(path.read_bytes(), GE_TYPE)
    producers = {op.name: op for op in graph.op}
    nodes = [dict(name=op.name, numel=vector_length(op, producers))
             for op in graph.op if op.type == GE_TYPE]
    sizes = Counter(n['numel'] for n in nodes)
    if sizes != {64*8192: 3, 64*4096: 3+4, 64*512: 4}:
        raise ValueError(f'Unexpected SiluPatch nodes in the four-layer W4 T64 AIR: {dict(sizes)}')
    print(f'{GE_TYPE}: {len(nodes)} nodes, FP16 x -> FP32 z, vector lengths {dict(sizes)}', flush=True)
    return dict(type=GE_TYPE, nodes=nodes, input='x:float16', output='z:float32')

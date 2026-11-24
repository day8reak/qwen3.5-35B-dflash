"""Integrate SiluPatch at the original CPro model's SiLU call sites."""
from collections import Counter
from difflib import unified_diff
from pathlib import Path
import re
from shutil import copyfile

from ..original_target_source import EXPORTER, SOURCE_FILES
from ..silu_patch_air import GE_TYPE, port_names, vector_desc

MODELING = 'models/modeling_qwen3_5_moe_hiai_cpro.py'


def stage(source, dest, *, aligned=False):
    (dest / 'models').mkdir(parents=True)
    for name in SOURCE_FILES:
        copyfile(source / name, dest / name)
    init = source / 'models/__init__.py'
    if init.is_file():
        copyfile(init, dest / 'models/__init__.py')
    else:
        (dest / 'models/__init__.py').touch()
    for name in ('silu_patch.py', 'silu_patch_air.py'):
        copyfile(Path(__file__).resolve().parents[1] / name, dest / 'models' / name)

    edits = {
        EXPORTER: [('\nimport torchair\n', '\nimport torchair\nfrom models.silu_patch import register_ge\n'
                   'register_ge(torchair)\n')],
        MODELING: [
            ('import torch.nn.functional as F\n', 'import torch.nn.functional as F\n'
             'from .silu_patch import silu_patch, silu_same_dtype\n'),
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
    for name in (EXPORTER, MODELING, 'models/silu_patch.py', 'models/silu_patch_air.py'):
        original = source / name
        before = original.read_text().splitlines(True) if original.is_file() else []
        diff.extend(unified_diff(before, (dest / name).read_text().splitlines(True),
                                 fromfile='original/' + name, tofile='modified/' + name))
    (dest / 'source.diff').write_text(''.join(diff))
    return dest / EXPORTER


def finish_air(path):
    from .decode_cache_control import graph_message

    message, graph = graph_message(path.read_bytes(), GE_TYPE)
    nodes = []
    changed = 0
    for op in graph.op:
        if op.type != GE_TYPE:
            continue
        if (len([e for e in op.input if e and not e.endswith(':-1')]) != 1
                or len(op.input_desc) != 1 or len(op.output_desc) != 1):
            raise ValueError(f'{op.name}: expected one input x and one output z')
        before = op.SerializeToString(deterministic=True)
        x, z = op.input_desc[0], op.output_desc[0]
        # Some TorchAir writers clear intermediate shapes. The converter names the
        # node from the actual flat input's FakeTensor length before serialization.
        tag = re.search(r'(?:^|/)silu_patch_n(\d+)(?:_\d+)?$', op.name)
        shapes = {tuple(d.shape.dim) for d in (x, z) if d.shape.dim}
        if tag:
            shapes.add((int(tag[1]),))
        if len(shapes) != 1:
            raise ValueError(f'{op.name}: inconsistent or missing x/z shapes {shapes}')
        shape = shapes.pop()
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError(f'{op.name}: expected a static vector, got {shape}')
        for desc, name, dtype in ((x, 'x', 2), (z, 'z', 1)):  # GE proto DT_FLOAT16 / DT_FLOAT
            if desc.dtype != dtype or (desc.name and desc.name != name):
                raise ValueError(f'{op.name}: expected {name} dtype={dtype}, got {desc.name!r} dtype={desc.dtype}')
            vector_desc(desc, shape[0])
        port_names(op, 'x', 'z')
        changed += before != op.SerializeToString(deterministic=True)
        nodes.append(dict(name=op.name, numel=shape[0]))
    # Three GDN convs, three GDN gates, four shared experts and four routed experts.
    sizes = Counter(n['numel'] for n in nodes)
    if sizes != {64*8192: 3, 64*4096: 3+4, 64*512: 4}:
        raise ValueError(f'Unexpected SiluPatch nodes in the four-layer W4 T64 AIR: {dict(sizes)}')
    if changed:
        # Restore the v3 port-format hints on the custom nodes before hashing the AIR.
        path.write_bytes(message.SerializeToString())
    print(f'{GE_TYPE}: {len(nodes)} nodes, FP16 x -> FP32 z, ND vectors {dict(sizes)}', flush=True)
    return dict(type=GE_TYPE, nodes=nodes, format='ND', input='x:float16', output='z:float32',
                normalized_nodes=changed)

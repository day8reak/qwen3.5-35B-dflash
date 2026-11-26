"""Export the four-tap GDN depthwise cross-correlation with slice, multiply and add."""
import ast
from pathlib import Path
import shutil
from textwrap import indent

from ..original_target_source import EXPORTER
from .decoder_four_silu import MODELING
from .source_model import _definition, _assignment


# The original convolution takes FP16 operands. Accumulate their products in FP32,
# then round once at the existing SiLU input boundary.
TAPS = """weight = self.conv1d.weight
if weight.shape[1:] != (1, 4):
    raise ValueError("GDN taps requires a depthwise filter [channels, 1, 4]")
window_fp32 = padded.to(weight.dtype).float()
weight_fp32 = weight.float()
conv_out = window_fp32[:, :, :seq_len] * weight_fp32[:, :, 0]
for tap in range(1, 4):
    conv_out = conv_out + window_fp32[:, :, tap:tap + seq_len] * weight_fp32[:, :, tap]
conv_out = conv_out.to(weight.dtype)"""


def taps_source(source, dest):
    """Derive a run-local variant without modifying the owner's reviewed source."""
    model = source / MODELING
    text = model.read_text()
    mixer = _definition(ast.parse(text), 'Qwen3_5MoeGatedDeltaNet')
    forward = _definition(mixer, 'forward')
    conv = _assignment(forward, 'conv_out')
    expected = ('F.conv2d(padded.to(self.conv1d.weight.dtype).reshape(batch_size, self.conv_dim, 1, -1), '
                'self.conv1d.weight.unsqueeze(2), groups=self.conv_dim).squeeze(2)')
    if ast.dump(conv.value) != ast.dump(ast.parse(expected, mode='eval').body):
        raise ValueError(f'{model}:{conv.lineno}: expected the reviewed GDN depthwise conv_out expression. '
                         'Check your CPRO_PATCHED_SOURCE before selecting taps.')
    lines = text.splitlines(keepends=True)
    lines[conv.lineno-1:conv.end_lineno] = [indent(TAPS, ' ' * conv.col_offset) + '\n']
    (dest/'models').mkdir(parents=True)
    # Keep the entry point physically in dest so Python imports the modified models package.
    # Link other resources, including custom local modules and data referenced from this tree.
    for folder in (Path('.'), Path('models')):
        for path in (source/folder).iterdir():
            rel = folder/path.name
            if path.name == '__pycache__' or rel in (Path('models'), Path(MODELING)):
                continue
            if rel == Path(EXPORTER):
                shutil.copyfile(path, dest/rel)
            else:
                (dest/rel).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (dest/MODELING).write_text(''.join(lines))
    print(f'GDN per-channel four-tap source: {dest/MODELING}', flush=True)
    return dest


def inspect_taps(graph, silus, count):
    """Check the four products feed each convolution SiLU in the exported AIR."""
    conv_types = {'Conv1D', 'Conv2D', 'Conv2DV2', 'Convolution', 'DepthwiseConv2D'}
    convs = [n.name for n in graph.op if n.type in conv_types]
    if convs or len(silus) != count:
        raise ValueError(f'Expected {count} GDN tap sums, found {len(silus)} convolution SiLUs and conv nodes {convs}')
    nodes = {n.name: n for n in graph.op}

    def products(edge):
        node = nodes[edge.rsplit(':', 1)[0]]
        inputs = [x for x in node.input if x and not x.endswith(':-1')]
        if node.type in ('Cast', 'Identity', 'TensorMove'):
            source, = inputs
            return products(source)
        if node.type in ('Add', 'AddV2', 'AddN'):
            return [name for src in inputs for name in products(src)]
        if node.type == 'Mul':
            return [node.name]
        raise ValueError(f'{node.name}: expected a GDN tap product or sum, found {node.type}')

    for silu in silus:
        names = products(nodes[silu].input[0])
        if len(names) != 4 or len(set(names)) != 4:
            raise ValueError(f'{silu}: expected four different GDN tap products, found {names}')
    print(f'GDN taps: {count} sums feeding SiluPatch, 4 Mul each, 0 convolution nodes', flush=True)
    return dict(implementation='taps', nodes=count, convolution_nodes=0, taps_per_channel=4)

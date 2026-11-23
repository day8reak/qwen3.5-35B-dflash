"""The L0 convolution SiLU, before the transpose and split into Q/K/V."""
from functools import lru_cache, wraps
import sys
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from ...common.artifacts import share_for_transfer
from ..layer_probe.common import KIND, checked, identity, read_array, save_json, sha
from .attention_slice import DTYPES, descriptor, extract, source_graph
from .decode_cache_trace import _record
from .gdn_tail_probe import case_id, tensor
from .gdr_stage_air import stage_edges

STAGE = 'qkv_conv_silu'
BASIS = 'NATIVE_L0_310P_CONV_OUT_SILU'
NCT = (1, 8192, 64)
CONVS = {'Conv1D', 'Conv2D', 'Convolution', 'DepthwiseConv2D', 'Conv2DV2'}
VIEWS = {'Cast', 'Reshape', 'Squeeze', 'Unsqueeze', 'ExpandDims', 'Identity', 'TensorMove'}


def activation_edges(graph):
    nodes = {n.name: n for n in graph.op}
    ports = stage_edges(graph, 'gdr')

    @lru_cache(None)
    def ancestors(edge):
        name = edge.rsplit(':', 1)[0]
        parents = [ancestors(e) for e in nodes[name].input if e and not e.endswith(':-1')]
        return {name}.union(*parents)

    def data(node):
        return [e for e in node.input if e and not e.endswith(':-1')]

    def producer(edge):
        return nodes[edge.rsplit(':', 1)[0]]

    def strip(edge, types):
        while producer(edge).type in types:
            edge = data(producer(edge))[0]
        return edge

    common = set.intersection(*(ancestors(ports[k]) for k in ('query', 'key', 'value')))
    found = []
    for name in sorted(common):
        n = nodes[name]
        args = data(n)
        if n.type in ('Silu', 'SiLU', 'Swish', 'SwishV2'):
            x = args[0]
        elif n.type == 'Mul' and len(args) == 2:
            xs = []
            for lhs, rhs in (args, args[::-1]):
                sigmoid = producer(strip(rhs, VIEWS))
                if sigmoid.type != 'Sigmoid':
                    continue
                # Retain differing internal casts on either branch of x * sigmoid(x).
                def cast_chain(edge):
                    chain = [edge]
                    while producer(edge).type == 'Cast':
                        edge = data(producer(edge))[0]
                        chain.append(edge)
                    return chain
                other = set(cast_chain(data(sigmoid)[0]))
                matches = [edge for edge in cast_chain(lhs) if edge in other]
                if matches:
                    xs.append(matches[0])
            if len(xs) != 1:
                continue
            x = xs[0]
        else:
            continue
        if producer(strip(x, VIEWS)).type in CONVS:
            found.append((x, name + ':0'))
    if len(found) != 1:
        raise ValueError(f'Expected one convolution SiLU shared by GDR Q/K/V, found {found}. Inspect with list.')
    return found[0]


def observe_conv(layer, snapshot, cleanup):
    import torch.nn.functional as F
    from ..layer_probe.capture import cpu

    original = F.silu

    @wraps(original)
    def observe(x, *args, **kwargs):
        if tuple(x.shape) != NCT:
            return original(x, *args, **kwargs)
        if 'conv_out' in snapshot or x.device.type != 'npu':
            raise ValueError('Expected a single native 310P L0 convolution SiLU call')
        snapshot['conv_out'] = cpu(x)
        frame = sys._getframe(1)
        snapshot['silu_site'] = f'{frame.f_code.co_filename}:{frame.f_lineno} ({frame.f_code.co_name})'
        y = original(x, *args, **kwargs)
        snapshot[STAGE] = cpu(y)
        return y

    # Limit the hook to the original L0 convolution update on the NZ driver.
    update = getattr(layer, 'causal_conv1d_update', None)
    if callable(update):
        @wraps(update)
        def tapped(*args, **kwargs):
            with patch.object(F, 'silu', observe):
                return update(*args, **kwargs)
        cleanup.enter_context(patch.object(layer, 'causal_conv1d_update', tapped))
    else:
        # CPro-style source has the call inline in L0.forward. Capture exits at L0 out_proj.
        cleanup.enter_context(patch.object(F, 'silu', observe))


def measure(source, config, expected_hidden):
    import torch.nn.functional as F
    from .gdn_tail_reference import capture_l0

    def evaluate(layer, snap, info):
        if 'conv_out' not in snap:
            raise ValueError('No visible native F.silu(conv_out) on this 310P path; a fused update may hide it')
        hidden = snap['output'][:, :19]
        expected = expected_hidden.reshape(1, 64, 2048)[:, :19]
        if hidden.dtype != expected.dtype or hidden.tobytes() != expected.tobytes():
            raise ValueError('Native valid19 L0 output differs from the selected attention AIR fixture')
        x, y = snap['conv_out'], snap[STAGE]
        info['silu_site'] = snap['silu_site']
        info['native_input_dtype'] = x.numpy().dtype.name
        return x.numpy(), y.numpy(), F.silu(x).numpy()

    return capture_l0(source, config, evaluate, observe=observe_conv)


def prepare(run, air, source, config, *, reuse=False):
    run.mkdir(parents=True, exist_ok=True)
    target = run/'air'
    if reuse and target.is_dir():
        manifest, _, folder, spec = _record(target, case_id(STAGE))
        if manifest['golden_source'] != BASIS or len(manifest['cases']) != 1:
            raise ValueError(f'{target}: expected the standalone convolution SiLU fixture')
        if (air/'manifest.json').is_file() and sha(air/'manifest.json') != manifest['source_manifest_sha256']:
            raise ValueError(f'{target}: stored SiLU data belongs to a different attention AIR batch')
        native = manifest['native_310p']
        if sha(source/'inference_segment.py') != native['driver_sha256'] or sha(source/config) != native['config_sha256']:
            raise ValueError(f'{target}: driver or config changed; prepare a new fixture directory')
        checked(target, manifest['boundaries'])
        for record in (spec['air'], *spec['source_inputs'], spec['golden'], spec['cpu'], *spec['weight_evidence'].values()):
            checked(folder, record)
        share_for_transfer(target)
        print(f'Reusing SiLU fixture: {target}', flush=True)
        return
    if target.exists():
        raise FileExistsError(f'{target} already exists')
    folder, original, msg, graph = source_graph(air, 0)
    x_edge, y_edge = activation_edges(graph)
    nodes = {n.name: n for n in graph.op}
    x_desc = descriptor(nodes, x_edge)
    dims = list(x_desc.shape.dim)
    shapes = (NCT, (1, 8192, 1, 64), (8192, 64))
    shape = NCT
    if dims and dims != [-2]:
        views = [s for s in shapes if len(s) == len(dims) and
                 all(a <= 0 or a == b for a, b in zip(dims, s))]
        if len(views) != 1:
            raise ValueError(f'{x_edge}: cannot map native NCT to source AIR shape {dims}')
        shape = views[0]
    (x, y, host), info = measure(source, config, read_array(folder, original['golden']))
    if any(a.shape != NCT or not np.isfinite(a[:, :, :19]).all() for a in (x, y, host)):
        raise ValueError('Expected finite valid rows on the last axis of all three native tensors')
    enums = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    if x_desc.dtype:
        cast_dtype = next((k for k in ('float16','float32') if enums[DTYPES[k]].number == x_desc.dtype), None)
        if cast_dtype is None:
            raise ValueError(f'{x_edge}: unsupported SiLU operand dtype {x_desc.dtype}')
        candidate = x.astype(cast_dtype, copy=False)
        if candidate.astype(x.dtype, copy=False).tobytes() != x.tobytes():
            raise ValueError(f'{x_edge}: CPro {cast_dtype} cannot represent the measured 310P inputs exactly')
        x = candidate
    name = case_id(STAGE)
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        root = Path(tmp)/'air'
        dest = root/'models'/name
        dest.mkdir(parents=True)
        ports = {x_edge: ('conv_out', x.dtype.name, shape)}
        sliced = extract(msg, graph, folder, dest, ports, y_edge)
        # A raw SiLU preserves element order and shape. Keep the original channel-first bytes on disk.
        out_type = sliced['output_desc']['dtype']
        required = {'0': k for k in ('float16','float32') if DTYPES[k] == out_type}
        save_json(dest/'export.json', dict(air=sliced['air'], source_edge=y_edge, source_data_edges={'conv_out': x_edge},
                  source_inputs=[dict(tensor(dest, 'conv_out', x.reshape(shape)), semantic_name='conv_out', fixture_index=0)],
                  golden=tensor(dest, STAGE, y.reshape(shape)), cpu=tensor(dest, 'cpu', host.reshape(shape)),
                  token_axis=-1, weight_evidence=sliced['weight_evidence'], reference_basis=BASIS,
                  output_contract=dict(stage_views={STAGE: [list(shape)]}, required_dtypes=required)))
        save_json(root/'boundaries.json', dict(source_air=original['air'], **sliced))
        case = dict(id=name, layer=0, call=0, phase='prefill', ordinal=0, kind='gdn_tail_prefix',
                    point=STAGE, output_name=STAGE, status='EXPORTED', export=identity(dest/'export.json', root))
        save_json(root/'manifest.json', dict(artifact_kind=KIND, status='EXPORTED', cases=[case],
                  golden_source=BASIS, reference_model_layers=1, native_310p=info, boundaries=identity(root/'boundaries.json', root),
                  source_manifest_sha256=sha(air/'manifest.json'), scope='L0 convolution SiLU on measured 310P input, NCT valid19'))
        root.rename(target)
    share_for_transfer(target)
    print(f'{x_edge} -> {y_edge}; {shape} {x.dtype}; {target}/boundaries.json', flush=True)

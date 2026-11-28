"""Compare L0 GDN linear operands in the precise layout consumed by A8W4 matmul."""
from itertools import combinations
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from ..layer_probe.qlinear_golden import _array_identity
from ..original_quant import pack_i4zn
from .attention_slice import DTYPES
from .decode_cache_control import graph_message
from .gdn_layer_stages import PROJECTIONS, SHAPES
from .layer_template import parameter

BACKENDS = ('cpu', '310p', 'cpro')
VIEWS = {'Reshape', 'View', 'Identity', 'TensorMove', 'Squeeze', 'Unsqueeze', 'ExpandDims'}


def operand_comparison(cpu, native, cpro):
    values = dict(zip(BACKENDS, (cpu, native, cpro), strict=True))
    pairs = {}
    for left, right in combinations(BACKENDS, 2):
        a, b = values[left], values[right]
        row = pairs[f'{left}/{right}'] = dict(shape_equal=a.shape == b.shape, dtype_equal=a.dtype == b.dtype)
        if not row['shape_equal']:
            row['status'] = 'SHAPE_MISMATCH'
            continue
        if a.dtype.kind == 'f' and (not np.isfinite(a).all() or not np.isfinite(b).all()):
            row.update(status='NONFINITE', nonfinite_left=int((~np.isfinite(a)).sum()),
                       nonfinite_right=int((~np.isfinite(b)).sum()))
            continue
        mismatch = int(np.count_nonzero(a != b))
        row.update(elements=a.size, different=mismatch, max_abs=float(np.max(np.abs(
            a.astype(np.float64) - b))) if mismatch else 0.0)
        if not mismatch:
            row['status'] = 'EXACT'
        elif {a.dtype.name, b.dtype.name} == {'float16', 'float32'} and np.array_equal(
                a.astype(np.float16), b.astype(np.float16)):
            row['status'] = 'FP16_ROUNDING_ONLY'
        else:
            row['status'] = 'DIFFERENT'
    return dict(operands={k: _array_identity(v) for k, v in values.items()}, pairs=pairs)


def static_operand(edge, nodes, types, folder, directory):
    n = nodes[edge.rsplit(':', 1)[0]]
    if not edge.endswith(':0'):
        raise ValueError(f'Expected a single-output weight at {edge}')
    if n.type == 'Cast':
        return static_operand(n.input[0], nodes, types, folder, directory).astype(types[n.output_desc[0].dtype])
    if n.type in ('Identity', 'TensorMove'):
        return static_operand(n.input[0], nodes, types, folder, directory)
    bound = parameter(n, types, folder, directory)
    if bound is None:
        raise ValueError(f'Cannot read static matmul operand from {edge} ({n.type})')
    dtype, shape, path = bound
    return np.fromfile(path, dtype=dtype).reshape(shape)


def fixed_operands(raw, folder, steps, params):
    _, graph = graph_message(raw, 'QuantBatchMatmulV4444')
    nodes = {n.name: n for n in graph.op}
    codes = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_name
    types = {codes[v].number: k for k, v in DTYPES.items()}
    plan = {s['name']: s for s in steps}
    result = {}

    def view_root(edge):
        while edge.endswith(':0') and nodes[edge[:-2]].type in VIEWS:
            edge = nodes[edge[:-2]].input[0]
        return edge

    with TemporaryDirectory() as tmp:
        for p in PROJECTIONS:
            name = 'out_proj' if p == 'out' else 'in_proj_' + p
            step = plan[name]
            edge, = step['outputs']
            op = nodes[edge.rsplit(':', 1)[0]]
            slots = dict(zip((k.decode() for k in op.attr['_input_name_key'].list.s),
                             op.attr['_input_name_value'].list.i, strict=True))
            ports = {k: op.input[slots[k]] for k in ('x1', 'x2', 'scale', 'pertoken_scale')}
            connected = {i for i, e in enumerate(op.input) if e and not e.endswith(':-1')}
            if op.type != 'QuantBatchMatmulV4444' or connected != {slots[k] for k in ports}:
                raise ValueError(f'{name}: expected A8W4 with no additional tensor operands at {edge}')
            scale_cast = nodes[ports['pertoken_scale'].rsplit(':', 1)[0]]
            scale_dtype = types[op.input_desc[slots['pertoken_scale']].dtype]
            x1, token = step['inputs']
            if (scale_cast.type != 'Cast' or scale_dtype != 'float16' or
                    types[scale_cast.output_desc[0].dtype] != scale_dtype or
                    view_root(scale_cast.input[0]) != token or view_root(ports['x1']) != x1):
                raise ValueError(f'{name}: unexpected activation or per-token scale path at {edge}')
            w, groups = params[p + '_weights']
            packed, scales = params[p + '_310p_operands']
            cpro_w = static_operand(ports['x2'], nodes, types, folder, Path(tmp))
            cpro_s = static_operand(ports['scale'], nodes, types, folder, Path(tmp))
            if cpro_w.dtype != np.int32 or cpro_s.dtype != np.float16:
                raise ValueError(f'{name}: constant dtypes disagree with the checked AIR input ports')
            result[name] = dict(node=op.name, ports=ports, pertoken_scale_cast=scale_cast.name,
                pertoken_scale_dtype=scale_dtype, optional_tensor_inputs=[],
                cpu_checkpoint=dict(weight_shape=list(w.shape), group_scale_shape=list(groups.shape),
                                    group_scale_dtype=groups.dtype.name),
                packed_w4=operand_comparison(pack_i4zn(w), packed, cpro_w),
                group_scale=operand_comparison(groups[:, :, 0].T.astype(np.float32), scales, cpro_s))
    return result


def dynamic_operands(p, root, ref, stage, checkpoint, static):
    from ..layer_probe.common import read_array

    result = {}
    for suffix, field in (('_a8', 'x_quant'), ('_scale', 'pertoken_scale')):
        role = p + suffix
        shape = SHAPES[role]
        slot, = [i for i, s in enumerate(checkpoint['source']['inputs']) if s['role'] == role]
        feed = checkpoint['source']['inputs'][slot]
        spec = dict(path=f'operand_{slot}.bin', dtype=feed['dtype'], shape=shape, sha256=feed['sha256'],
                    bytes=int(np.prod(shape)) * np.dtype(feed['dtype']).itemsize)
        cpro = read_array(stage, spec)
        if field == 'pertoken_scale':
            cpro = cpro.astype(static['pertoken_scale_dtype'])
        elif cpro.dtype != np.int8:
            raise ValueError(f'{stage}: expected INT8 activation')
        cpu, native = (read_array(root, ref['arrays'][b][role]) for b in ('cpu', '310p'))
        result[field] = operand_comparison(cpu[:19], native[:19], cpro[:19])
    return result

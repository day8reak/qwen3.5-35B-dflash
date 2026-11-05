"""Batch single-output source prefixes through the complete first Draft layer.

This diagnostic consumes retained AIR and inputs on the DDK host, using NumPy
only. Each checkpoint has its own NetOutput, avoiding paired FP16/FP32 aliases.
It does not change the production model, weights, acceptance or runtime.
GE slots/attributes follow the pinned source IR, not OMG's compiled op names:
https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py
"""
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, _read_air
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v, _new_id
from .hidden_probe import Prefix, GE_DTYPES
from .torch_export import _schema, _disjoint

KIND = 'qwen35-draft-layer0-prefix-suite'
SCOPE = 'SOURCE_LAYER0_SINGLE_OUTPUT_PREFIXES; NOT_FULL_MODEL_FIRST_DIVERGENCE_PROOF'
STEPS = (0, 2, 17)
MAX_POINTS = 64
CASES = tuple(f'hidden_layer0_c{s:02d}_p{p:02d}' for s in STEPS for p in range(MAX_POINTS))
GE_TYPES = {**GE_DTYPES, 12: '|b1'}
PROTO_TYPES = {'DT_FLOAT': '<f4', 'DT_FLOAT16': '<f2', 'DT_INT32': '<i4', 'DT_INT64': '<i8', 'DT_BOOL': '|b1'}
MATH = {'Conv2D', 'MatMul', 'BatchMatMul', 'BatchMatMulV2', 'RmsNorm', 'SoftmaxV2', 'Swish'}
ALLOWED = MATH | {'Data', 'Const', 'FileConstant', 'Cast', 'Identity', 'Reshape', 'Transpose',
    'Squeeze', 'Unsqueeze', 'ExpandDims', 'SplitV', 'ConcatV2', 'GatherV2', 'Neg',
    'Add', 'Sub', 'Mul', 'Muls', 'Less', 'LessEqual', 'Greater', 'GreaterEqual', 'Equal',
    'LogicalAnd', 'LogicalOr', 'LogicalNot', 'SelectV2', 'BroadcastTo', 'Sin', 'Cos',
    'Shape', 'Fill', 'ZerosLike', 'StridedSliceV2'}


class LayerTrace(Prefix):
    def node_port(self, edge):
        name, sep, port = edge.rpartition(':')
        if not sep or not port.isdecimal() or name not in self.nodes:
            raise ValueError('unresolved source edge: '+edge)
        node = self.nodes[name]
        if int(port) >= len(node.get('output_desc', [])):
            raise ValueError('source edge outside recorded output ports: '+edge)
        return node, int(port)

    def live_inputs(self, node):
        if node['type'] == 'MatMul':
            return [e for _, e in self.data_inputs(node)]
        edges = node.get('input', [])
        if node['type'] == 'Data':
            if edges not in ([], ['']):
                raise ValueError('Data has unexpected dependencies')
            return []
        optional = {'Conv2D': {2, 3}, 'StridedSliceV2': {3, 4}}.get(node['type'], set())
        descs = node.get('input_desc', [])
        if len(descs) != len(edges):
            raise ValueError('source input/descriptor count differs: '+node['name'])
        for i, edge in enumerate(edges):
            if edge:
                continue
            if (i not in optional or descs[i].get('dtype', 0) != self.types['DT_UNDEFINED']
                    or _attributes(descs[i]).get('_is_unfed_optional', {}).get('i') != 1):
                raise ValueError('unmarked absent source input: '+node['name'])
        return [e for e in edges if e]

    def dependencies(self, edge):
        seen, active, ordered = set(), set(), []
        def visit(e):
            node, _ = self.node_port(e); name = node['name']
            if name in seen:
                return
            if name in active or len(active) > 256 or len(seen) > 1024:
                raise ValueError('cyclic or oversized first-layer dependency graph')
            if node.get('subgraph_name'):
                raise ValueError('nested source graphs are unsupported')
            active.add(name)
            for arg in self.live_inputs(node):
                visit(arg)
            active.remove(name); seen.add(name); ordered.append(name)
        visit(edge)
        return ordered

    def select_layer(self):
        # The next layer's input norm locates layer zero's final residual;
        # numerical node suffixes are never used as semantic identities.
        next_norm = self.norm('draft.layers.1.input_layernorm.weight')
        edge = next_norm['input'][0]
        node, port = self.node_port(edge)
        if node['type'] != 'Cast' or _attributes(node).get('dst_type', {}).get('i') != 0 or port != 0:
            raise ValueError('next-layer RMS must consume the source FP32 cast of hidden')
        self.terminal = node['input'][0]
        self.ordered = self.dependencies(self.terminal)
        unknown = [(n, self.nodes[n]['type']) for n in self.ordered if self.nodes[n]['type'] not in ALLOWED]
        if unknown:
            raise ValueError('unsupported first-layer source operations (all found): '+str(unknown))
        counts = Counter(self.nodes[n]['type'] for n in self.ordered)
        expected = {'Conv2D': 1, 'MatMul': 5, 'RmsNorm': 6, 'SoftmaxV2': 1, 'Swish': 1}
        if any(counts[k] != v for k, v in expected.items()) or counts['BatchMatMul']+counts['BatchMatMulV2'] != 2:
            raise ValueError('first-layer topology differs from packed Draft: '+str(dict(counts)))
        self.counts = dict(counts)

    def evaluate_layer(self, inputs, weights, *, reference_gamma_boundary=False):
        self.values = {}
        proto = {self.types[k]: np.dtype(v) for k, v in PROTO_TYPES.items()}
        def value(edge):
            if edge in self.values:
                return self.values[edge]
            n, port = self.node_port(edge); kind, name = n['type'], n['name']
            attrs = _attributes(n)
            a = [value(e) if e else None for e in n.get('input', [])] if kind != 'Data' else []
            attr = lambda key, typ, default=None: attrs.get(key, {}).get(typ, default)
            ints = lambda key, default=None: attrs.get(key, {}).get('list', {}).get('i', default)
            if kind == 'Data':
                y = inputs[name]
            elif kind == 'FileConstant':
                y = weights[attr('file_path', 's')]
                if y.dtype != np.dtype(GE_TYPES[attr('dtype', 'i')]) or list(y.shape) != ints('shape'):
                    raise ValueError('FileConstant metadata conflicts with bytes: '+name)
            elif kind == 'Const':
                desc = attrs['value']['t']['desc']
                data = self.shapes.constant_data(n)
                y = np.frombuffer(data, dtype=proto[desc['dtype']]).reshape(desc.get('shape', {}).get('dim', []))
            elif kind == 'Cast':
                y = a[0].astype(GE_TYPES[attr('dst_type', 'i', 0)])
            elif kind == 'Identity':
                y = a[0]
            elif kind == 'Reshape':
                if attr('axis', 'i', 0) != 0:
                    raise ValueError('partial Reshape is unsupported')
                y = a[0].reshape(a[1].reshape(-1).tolist())
            elif kind == 'Transpose':
                y = a[0].transpose(a[1].reshape(-1).tolist())
            elif kind in {'Squeeze', 'Unsqueeze', 'ExpandDims'}:
                axes = (a[1].reshape(-1).tolist() if kind == 'ExpandDims' else
                        ints('axis' if kind == 'Squeeze' else 'axes', []))
                y = (np.squeeze(a[0], axis=tuple(axes) if axes else None) if kind == 'Squeeze'
                     else np.expand_dims(a[0], axis=tuple(axes)))
            elif kind == 'SplitV':
                sizes = a[1].reshape(-1).tolist(); axis = int(a[2].item())
                if sizes.count(-1) > 1:
                    raise ValueError('multiple inferred SplitV sizes')
                sizes = [a[0].shape[axis]-sum(s for s in sizes if s >= 0) if s == -1 else s for s in sizes]
                if (any(s <= 0 for s in sizes) or sum(sizes) != a[0].shape[axis]
                        or len(sizes) != attr('num_split', 'i') or len(sizes) != len(n['output_desc'])):
                    raise ValueError('invalid SplitV extents')
                ys = np.split(a[0], np.cumsum(sizes)[:-1], axis=axis)
                for i, part in enumerate(ys):
                    self.values[f'{name}:{i}'] = np.ascontiguousarray(part)
                y = ys[port]
            elif kind == 'ConcatV2':
                y = np.concatenate(a[:-1], axis=int(a[-1].item()))
            elif kind == 'GatherV2':
                if attr('batch_dims', 'i', 0) != 0:
                    raise ValueError('batched Gather not supported')
                y = np.take(a[0], a[1], axis=int(a[2].item()))
            elif kind in {'MatMul', 'BatchMatMul', 'BatchMatMulV2'}:
                x, w = a[:2]
                if len(a) > 2 and any(v is not None for v in a[2:]):
                    raise ValueError('unexpected matrix bias')
                if x.dtype != w.dtype or x.dtype not in (np.dtype('<f2'), np.dtype('<f4')):
                    raise ValueError('matrix input dtype mismatch')
                t = 'transpose_' if kind == 'MatMul' else 'adj_'
                if attr(t+'x1', 'b', False): x = x.swapaxes(-1, -2)
                if attr(t+'x2', 'b', False): w = w.swapaxes(-1, -2)
                y = (x.astype(np.float32) @ w.astype(np.float32)).astype(x.dtype)
            elif kind == 'Conv2D':
                x, w = a[:2]; stride, pads = ints('strides'), ints('pads')
                if (attr('data_format', 's') != 'NCHW' or attr('groups', 'i', 1) != 1
                        or ints('dilations', [1]*4) != [1]*4 or pads != [0]*4
                        or attr('offset_x', 'i', 0) != 0 or any(v is not None for v in a[2:])
                        or x.ndim != 4 or w.ndim != 4 or x.dtype != np.float16 or w.dtype != np.float16
                        or stride != [1, 1, 4, 4] or w.shape[-2:] != (4, 4)):
                    raise ValueError('only the retained spatial 4x4 full-dot Conv is supported')
                # Full K dot, no split-K rounding; original NCHW patch order.
                patches = np.lib.stride_tricks.sliding_window_view(x, (4, 4), axis=(2, 3))[:, :, ::4, ::4]
                matrix = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, w.shape[1]*16)
                y = (matrix.astype(np.float32) @ w.reshape(w.shape[0], -1).astype(np.float32).T).astype(np.float16)
                y = y.reshape(x.shape[0], x.shape[2]//4, x.shape[3]//4, w.shape[0]).transpose(0, 3, 1, 2)
            elif kind == 'RmsNorm':
                x, gamma = a; eps = attr('epsilon', 'f')
                if x.dtype != np.float32 or gamma.dtype != np.float32 or gamma.shape != (x.shape[-1],) or not 0 < eps < 1:
                    raise ValueError('source RMS dtype/gamma/epsilon mismatch')
                rstd = 1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32)+np.float32(eps))
                normalized = x*rstd
                y = ((normalized.astype(np.float16)*gamma.astype(np.float16)).astype(np.float32)
                     if reference_gamma_boundary else normalized*gamma)
                if len(n['output_desc']) == 2:
                    self.values[name+':1'] = np.ascontiguousarray(rstd)
                if port == 1:
                    self.values[name+':0'] = np.ascontiguousarray(y)
                    y = rstd
            elif kind == 'SoftmaxV2':
                if ints('axes') not in ([-1], [a[0].ndim-1]) or attr('half_to_float', 'b', False):
                    raise ValueError('unsupported Softmax attributes')
                x = a[0].astype(np.float32); ex = np.exp(x-np.max(x, axis=-1, keepdims=True))
                y = (ex/np.sum(ex, axis=-1, keepdims=True, dtype=np.float32)).astype(a[0].dtype)
            elif kind == 'Swish':
                x = a[0].astype(np.float32); scale = np.float32(attr('scale', 'f', 1.0))
                y = (x/(1+np.exp(-x*scale))).astype(a[0].dtype)
            elif kind in {'Sin', 'Cos', 'Neg', 'LogicalNot'}:
                y = {'Sin': np.sin, 'Cos': np.cos, 'Neg': np.negative, 'LogicalNot': np.logical_not}[kind](a[0])
            elif kind == 'Muls':
                y = (a[0].astype(np.float32)*np.float32(attr('value', 'f'))).astype(a[0].dtype)
            elif kind in {'Add', 'Sub', 'Mul', 'Less', 'LessEqual', 'Greater', 'GreaterEqual', 'Equal', 'LogicalAnd', 'LogicalOr'}:
                fn = {'Add': np.add, 'Sub': np.subtract, 'Mul': np.multiply, 'Less': np.less,
                    'LessEqual': np.less_equal, 'Greater': np.greater, 'GreaterEqual': np.greater_equal,
                    'Equal': np.equal, 'LogicalAnd': np.logical_and, 'LogicalOr': np.logical_or}[kind]
                y = fn(*a)
            elif kind == 'SelectV2':
                y = np.where(*a)
            elif kind == 'BroadcastTo':
                y = np.broadcast_to(a[0], a[1].reshape(-1).tolist())
            elif kind == 'Shape':
                y = np.array(a[0].shape, dtype=GE_TYPES[attr('dtype', 'i', 3)])
            elif kind == 'Fill':
                y = np.full(a[0].reshape(-1).tolist(), a[1].item(), dtype=a[1].dtype)
            elif kind == 'ZerosLike':
                y = np.zeros_like(a[0])
            elif kind == 'StridedSliceV2':
                if any(attr(k, 'i', 0) for k in ('begin_mask', 'end_mask', 'ellipsis_mask', 'new_axis_mask', 'shrink_axis_mask')):
                    raise ValueError('masked StridedSliceV2 is unsupported')
                begin, end = a[1].reshape(-1), a[2].reshape(-1)
                axes = a[3].reshape(-1) if len(a) > 3 and a[3] is not None else np.arange(len(begin))
                stride = a[4].reshape(-1) if len(a) > 4 and a[4] is not None else np.ones(len(begin), dtype=int)
                if not len(begin) == len(end) == len(axes) == len(stride):
                    raise ValueError('slice vectors have different lengths')
                slices = [slice(None)]*a[0].ndim
                for ax, lo, hi, st in zip(axes, begin, end, stride): slices[int(ax)] = slice(int(lo), int(hi), int(st))
                y = a[0][tuple(slices)]
            else:
                raise ValueError('unsupported source operation: '+kind)
            if y.size > 32 << 20 or np.isnan(y).any():
                raise ValueError('oversized or NaN source value: '+edge)
            desc = n['output_desc'][port]
            dims = desc.get('shape', {}).get('dim')
            if kind != 'FileConstant' and dims and dims != list(y.shape):
                raise ValueError('source/evaluated shape mismatch: '+edge)
            # Empty source descriptors are placeholders, not observed tensors.
            # Concrete shapes imply a concrete type; normative ops check inputs above.
            if kind != 'FileConstant' and dims and desc.get('dtype') in proto and y.dtype != proto[desc['dtype']]:
                raise ValueError('source/evaluated dtype mismatch: '+edge)
            self.values[edge] = np.ascontiguousarray(y) if y.ndim else np.array(y)
            return self.values[edge]
        with np.errstate(over='ignore', under='ignore', invalid='raise', divide='raise'):
            terminal = value(self.terminal)
        if terminal.shape != (1, 16, 2048) or terminal.dtype != np.float16 or not np.isfinite(terminal).all():
            raise ValueError('layer-zero residual output contract differs')
        return self.values

    def checkpoints(self):
        selected, excluded = [], []
        for name in self.ordered:
            n = self.nodes[name]; edge = name+':0'; a = self.values[edge]
            if n['type'] == 'SelectV2' and any(not np.isfinite(self.values[e]).all() for e in n['input'][1:]):
                excluded.append(dict(source_edge=edge, reason='NONFINITE_MASK_SENTINEL_BRANCH; COVERED_BY_DOWNSTREAM_SOFTMAX'))
                continue
            # Every arithmetic result, plus its actual source half-cast boundary.
            # Views and split ports are retained inside each dependency prefix.
            eligible = n['type'] in MATH | {'Add', 'Mul', 'Muls', 'SelectV2', 'Sin', 'Cos'}
            if n['type'] == 'Cast' and _attributes(n).get('dst_type', {}).get('i') == 1:
                p, _ = self.node_port(n['input'][0])
                eligible = p['type'] in MATH | {'Add', 'Mul', 'Muls', 'Sin', 'Cos'}
            if not eligible or a.dtype not in (np.dtype('<f2'), np.dtype('<f4')):
                continue
            if not np.isfinite(a).all():
                excluded.append(dict(source_edge=edge, reason='NONFINITE_MASK_SENTINELS; COVERED_BY_DOWNSTREAM_SOFTMAX'))
                continue
            point = dict(source_edge=edge, source_type=n['type'], dtype=str(a.dtype), shape=list(a.shape))
            producer = n
            if n['type'] == 'Cast':
                point['cast_of'] = n['input'][0]
                producer, _ = self.node_port(n['input'][0])
            if producer['type'] in {'RmsNorm', 'MatMul', 'Conv2D'}:
                current = producer['input'][1]
                for _ in range(8):
                    weight, _ = self.node_port(current)
                    if weight['type'] not in {'Cast', 'Identity', 'Reshape', 'Transpose'}: break
                    current = weight['input'][0]
                if weight['type'] == 'FileConstant':
                    point['source_weight'] = _attributes(weight)['file_path']['s']
            selected.append(point)
        if not selected or selected[-1]['source_edge'] != self.terminal or len(selected) > MAX_POINTS:
            raise ValueError('layer-zero checkpoint coverage or count is invalid: '+str(len(selected)))
        return selected, excluded

    def extract_checkpoint(self, point, inputs):
        schema = self.schema
        def blob(message, field, value, kind=9, repeated=False):
            return _blob(_field(schema, message, field, kind, repeated), value)
        edge = point['source_edge']; selected = set(self.dependencies(edge))
        datas = [n for n in self.ordered if n in selected and self.nodes[n]['type'] == 'Data']
        if not 1 <= len(datas) <= 8:
            raise ValueError('layer prefix exceeds runner input contract')
        attr_field = _field(schema, 'OpDef', 'attr', 11, True)
        encoded, preserved = [], {}
        for n, payload, _ in self.rows:
            if n['name'] not in selected: continue
            if n['type'] == 'Data':
                index = blob('OpDef.AttrEntry', 'key', 'index')+blob('OpDef.AttrEntry', 'value',
                    _v(_field(schema, 'AttrDef', 'i', 3, False) << 3)+_v(datas.index(n['name'])), 11)
                payload = b''.join(e for f, _, e, p in _parts(payload) if f != attr_field
                    or schema.decode(p, 'OpDef.AttrEntry')['key'] != 'index')+_blob(attr_field, index)
            else: preserved[n['name']] = hashlib.sha256(payload).hexdigest()
            encoded.append(_blob(self.op_field, payload))
        dtype = self.types['DT_FLOAT16' if point['dtype'] == 'float16' else 'DT_FLOAT']
        desc = (_v(_field(schema, 'TensorDescriptor', 'dtype', 14, False) << 3)+_v(dtype)
            +blob('TensorDescriptor', 'shape', blob('ShapeDef', 'dim', b''.join(_v(d) for d in point['shape']), 3, True), 11)
            +blob('TensorDescriptor', 'layout', 'ND')+blob('TensorDescriptor', 'device_type', 'NPU'))
        net = (blob('OpDef', 'name', 'NetOutput')+blob('OpDef', 'type', 'NetOutput')
               +blob('OpDef', 'input', edge, repeated=True)+blob('OpDef', 'input_desc', desc, 11, True))
        net = _new_id(net, schema, max(n.get('id', 0) for n in self.nodes.values())+1)
        graph = blob('GraphDef', 'name', 'draft_layer0_prefix')+b''.join(encoded)+_blob(self.op_field, net)
        raw = graph if self.graph_field is None else b''.join(_blob(f, graph) if f == self.graph_field else e
                                                            for f, _, e, _ in _parts(self.raw))
        _decode(schema, raw)
        declared = dict(inputs=[dict(name=n, dtype=str(inputs[n].dtype), shape=list(inputs[n].shape)) for n in datas],
                        outputs=[dict(name='checkpoint', **point)])
        files = sorted({_attributes(self.nodes[n])['file_path']['s'] for n in selected if self.nodes[n]['type'] == 'FileConstant'})
        return raw, dict(graph=declared, files=files, preserved_payload_sha256=preserved,
            changes='ONLY_DATA_INDICES_AND_SINGLE_NETOUTPUT; SOURCE_COMPUTE_BYTES_PRESERVED')


def prepare_air(raw, schema):
    from .air_reshape import fold_reshape_chains
    from .air_zeros import materialize_zeros
    from .air_fill import materialize_fills
    from .air_matmul import fold_weight_transposes
    result, reshapes = fold_reshape_chains(raw, schema)
    result, zeros = materialize_zeros(result, schema)
    result, fills = materialize_fills(result, schema)
    result, transposes = fold_weight_transposes(result, schema)
    if transposes['skipped']:
        raise ValueError('unhandled first-layer weight transpose: '+str(transposes['skipped']))
    return result, dict(policy='layer0-prefix-static-preparation-v1', reshapes=reshapes,
        zeros_like=zeros, fills=fills, weight_transposes=transposes,
        compiler_sha256=hashlib.sha256(result).hexdigest())


def compile_suite(air_export, source, result, output, *, ddk_root, ddk_id,
                  tbe_diagnosis, timeout_seconds=1800, progress=print, replace=False):
    from .accuracy import analyze_result
    from .optimization import manifest_policy
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    import stable_arg_max_ge
    root, air, source, result = map(lambda p: Path(p).resolve(), (output, air_export, source, result))
    if root != Path(output).absolute() or root.exists() or replace:
        raise ValueError('layer0 suite requires its own unused output directory; prior query evidence is retained')
    output_path(root); _disjoint(root, air, source, result, ddk_root, tbe_diagnosis)
    progress('verifying retained AIR, original compiler snapshots and all three real input cases')
    analyze_result(result)
    run = read_json(result/'result.json', 4 << 20)
    fixture = Path(run['fixture']); fixture = fixture if fixture.is_absolute() else result/fixture
    meta, binding = read_json(fixture/'manifest.json'), read_json(fixture/'pytorch-abi.json')
    exported, plan = read_json(air/'manifest.json'), read_json(source/'prepared.json')
    raw, _ = _read_air(air/'export.air')
    if (run.get('output_initialization') != 'untouched' or meta.get('frontend') != 'pytorch'
            or manifest_policy(exported) != 'packed-spatial-host-embedding'
            or exported.get('attention', {}).get('policy') != 'composed-attention-v1'
            or binding['source_air_sha256'] != hashlib.sha256(raw).hexdigest()
            or binding['prepared_sha256'] != sha256_file(source/'prepared.json')
            or binding['packed_model_sha256'] != meta['model']['sha256']):
        raise ValueError('layer suite requires the bound untouched packed Draft replay')
    compiled = Path(meta['model']['path']).parent
    _disjoint(root, fixture, compiled)
    packing = read_json(compiled/'weight-packing.json')
    receipt = read_json(compiled/'compilation/report.json', 8 << 20)
    if (packing['output']['sha256'] != binding['packed_model_sha256'] or packing['status'] != 'EMBEDDED'
            or receipt.get('status') != 'COMPILED' or receipt.get('ddk_id') != ddk_id
            or packing['source']['sha256'] != receipt['omc']['sha256']):
        raise ValueError('original compiler/packing receipts differ from retained model')
    snapshots = {r['source_relative']: r for r in receipt['input_snapshots']}
    if len(snapshots) != len(receipt['input_snapshots']) or snapshots['export.air']['sha256'] != binding['source_air_sha256']:
        raise ValueError('original compiler snapshot inventory is ambiguous')
    schema = _schema((air/'air-schema.pb').read_bytes(), exported['schema'])
    trace = LayerTrace(raw, schema); trace.select_layer()
    constants = {n: _attributes(trace.nodes[n]) for n in trace.ordered if trace.nodes[n]['type'] == 'FileConstant'}
    paths = {a['file_path']['s'] for a in constants.values()}
    if sum(snapshots[p]['bytes'] for p in paths) > 256 << 20:
        raise ValueError('source first-layer constants exceed the 256 MiB batch model bound')
    weights, records = {}, {}
    for n, attrs in constants.items():
        path = attrs['file_path']['s']; snap = snapshots[path]
        record = dict(path=path, bytes=snap['bytes'], sha256=snap['sha256'])
        file = checked_file(air, record)
        weights[path] = np.fromfile(file, dtype=GE_TYPES[attrs['dtype']['i']]).reshape(attrs['shape']['list']['i'])
        records[path] = record
    original_inputs = {s['name']: s for s in meta['inputs']}
    input_records = {(r['case'], r['index']): r for r in meta['input_files']}
    if len(input_records) != len(meta['input_files']):
        raise ValueError('duplicate source input records')
    samples, checkpoints, plans = {}, None, []
    for step in STEPS:
        cases = [c for c in meta['cases'] if c['index'] == step]
        labels = {0: 'empty-context', 2: 'partial-context', 17: 'capacity-boundary'}
        if len(cases) != 1 or (step in (0, 2) and cases[0]['label'] != labels[step]):
            raise ValueError('retained diagnostic case is absent or changed: '+str(step))
        if step == 17 and cases[0]['start']+cases[0]['valid'] != meta['cache_capacity']:
            raise ValueError('case17 must reach the retained cache capacity')
        inputs, hashes = {}, {}
        for semantic, data_name in binding['source_input_roles'].items():
            if data_name not in trace.ordered: continue
            spec = original_inputs[semantic]; rec = input_records[step, spec['index']]
            inputs[data_name] = np.fromfile(checked_file(fixture, rec), dtype=np.dtype(spec['dtype'])).reshape(spec['shape'])
            hashes[data_name] = dict(rec, role=semantic)
        progress(f'evaluating complete source layer0 on retained case {step}: {cases[0]["label"]}')
        values = trace.evaluate_layer(inputs, weights)
        points, excluded = trace.checkpoints()
        if checkpoints is not None and checkpoints != points:
            raise ValueError('checkpoint structure differs between input cases')
        checkpoints = points
        samples[step] = dict(case=cases[0], inputs=inputs, input_records=hashes,
            golden={p['source_edge']: values[p['source_edge']].copy() for p in points})
        alternate = trace.evaluate_layer(inputs, weights, reference_gamma_boundary=True)
        samples[step]['boundary'] = {p['source_edge']: alternate[p['source_edge']].copy() for p in points}
    # Construct and validate EVERY compiler input before the first OMG call.
    for index, point in enumerate(checkpoints):
        extracted, extraction = trace.extract_checkpoint(point, samples[0]['inputs'])
        prepared, preparation = prepare_air(extracted, schema)
        plans.append(dict(index=index, checkpoint=point, extracted=extracted, extraction=extraction,
                          prepared=prepared, preparation=preparation))
    selected_weight_bytes = sum(sum(records[p]['bytes'] for p in r['extraction']['files']) for r in plans)
    required = 4*selected_weight_bytes+(1 << 30)
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < required:
        raise OSError(f'layer0 batch needs at least {required} free host bytes for source/compiler/packed copies')
    with (root.parent/('.'+root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir()
        state = dict(artifact_kind=KIND, stage='compile', status='PARTIAL', scope=SCOPE,
            model_compilation='PARTIAL', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
            ddk_id=ddk_id, original_model_sha256=meta['model']['sha256'],
            source_air_sha256=binding['source_air_sha256'], source_result=str(result),
            source_compiler_receipt_sha256=sha256_file(compiled/'compilation/report.json'),
            source_weights=records, source_cases=[samples[s]['case'] for s in STEPS],
            output_initialization='untouched', comparison=meta['comparison'],
            reference='NUMPY_SOURCE_GRAPH_FP32_REDUCTIONS_WITH_EXPLICIT_CASTS; NOT_ORIGINAL_PYTORCH_GOLDEN',
            coverage=dict(checkpoints=len(checkpoints), input_cases=3, planned_device_cases=len(checkpoints)*3,
                node_type_counts=trace.counts, excluded_checkpoints=excluded,
                semantics='ALL_LAYER0_ARITHMETIC_RESULTS_AND_SOURCE_FP16_BOUNDARIES; SINGLE_OUTPUT_MODELS'),
            probe_source_sha256=sha256_file(Path(__file__)), cases=[])
        _save(root/'result.json', state)
        for p in plans:
            i = p['index']; name = f'p{i:02d}'; folder = root/name; (folder/'air/weight').mkdir(parents=True)
            (folder/'air/export.air').write_bytes(p['extracted'])
            for relative in p['extraction']['files']:
                shutil.copyfile(checked_file(air, records[relative]), folder/'air'/relative)
            fixtures = {}
            for step in STEPS:
                f = folder/f'c{step:02d}'; f.mkdir(); sample = samples[step]
                for j, spec in enumerate(p['extraction']['graph']['inputs']):
                    sample['inputs'][spec['name']].tofile(f/f'input_{j}.bin')
                sample['golden'][p['checkpoint']['source_edge']].tofile(f/'golden_0.bin')
                sample['boundary'][p['checkpoint']['source_edge']].tofile(f/'boundary_0.bin')
                fixtures[str(step)] = {x.name: dict(path=x.name, bytes=x.stat().st_size, sha256=sha256_file(x)) for x in f.iterdir()}
            row = dict(case=name, status='FAILED', model_compilation='NOT_RUN', checkpoint=p['checkpoint'],
                extraction=p['extraction'], fixtures=fixtures, source_inputs={str(s):samples[s]['input_records'] for s in STEPS})
            state['cases'].append(row); _save(root/'result.json', state)
            try:
                progress(f'compiling checkpoint {i+1}/{len(plans)}: {p["checkpoint"]["source_edge"]}')
                def transform(snapshot):
                    if snapshot.read_bytes() != p['extracted']:
                        raise ValueError('prefix compiler snapshot changed')
                    return p['prepared'], p['preparation']
                build = compile_existing_air(folder/'air/export.air', folder/'compilation', ddk_root,
                    ddk_id=ddk_id, tbe_diagnosis=tbe_diagnosis, timeout_seconds=timeout_seconds,
                    input_layout='air-directory', use_origin_format=True, save_weights_as_external_data=True,
                    progress=lambda r: progress(r['message']), air_transform=transform)
                row['model_compilation'] = build['model_compilation']
                if build['status'] != 'COMPILED': raise ValueError(str(build.get('error', build['status'])))
                sidecars = [dict(r, path=str(folder/'compilation'/r['path'])) for r in build['external_data_candidates']
                            if Path(r['path']).suffix == '.weight']
                packed = embed_external_weights(folder/'compilation'/build['omc']['path'], folder/'model.omc',
                    Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so',
                    expected_source_sha256=build['omc']['sha256'], sidecars=sidecars, progress=progress)
                _save(folder/'weight-packing.json', packed)
                if packed['status'] != 'EMBEDDED' or (folder/'model.omc').stat().st_size > 256 << 20:
                    raise ValueError('layer prefix packing failed or exceeded 256 MiB')
                inspection = stable_arg_max_ge.inspect_model(folder/'model.omc', ddk_root, row['extraction']['graph'],
                                                             input_order='graph-input')
                row.update(status='COMPILED', placement=inspection['placement'],
                    model=dict(path=name+'/model.omc', bytes=(folder/'model.omc').stat().st_size,
                               sha256=sha256_file(folder/'model.omc')))
            except Exception as error:
                row.update(error=f'{type(error).__name__}: {error}',
                    compilation_diagnostics=compilation_diagnostics(folder/'compilation'))
            _save(root/'result.json', state)
        good = sum(r['status'] == 'COMPILED' for r in state['cases'])
        state.update(status='COMPILED' if good == len(plans) else ('PARTIAL' if good else 'FAILED'),
            stage='complete', model_compilation='PASSED' if good == len(plans) else ('PARTIAL' if good else 'FAILED'))
        state['coverage'].update(compiled_checkpoints=good, missing_checkpoints=[r['checkpoint'] for r in state['cases'] if r['status'] != 'COMPILED'])
        _save(root/'result.json', state)
    return state


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    from validate_draft_vector import WARMUP, REPETITIONS
    import stable_arg_max_ge
    root, out = Path(root), Path(out); report = read_json(root/'result.json', 16 << 20)
    if (report.get('status') not in ('COMPILED', 'PARTIAL') or report.get('stage') != 'complete'
            or report.get('ddk_id') != ddk_id or selection is not None):
        raise ValueError('run the complete available layer0 batch from a completed compilation')
    cases = []
    for row in report['cases']:
        if row['status'] != 'COMPILED': continue
        model = checked_file(root, row['model'])
        if model.stat().st_size > 256 << 20:
            raise ValueError('layer prefix exceeds its declared model bound')
        inspection = stable_arg_max_ge.inspect_model(model, ddk, row['extraction']['graph'], input_order='graph-input')
        for step in STEPS:
            name = f'hidden_layer0_c{step:02d}_{row["case"]}'
            if name not in CASES: raise ValueError('unknown layer0 checkpoint case')
            bound = deepcopy(inspection['io']); folder = out/'fixture'/name; folder.mkdir(parents=True)
            plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}']
            for direction, stem in (('inputs', 'input'), ('outputs', 'golden')):
                for spec in bound[direction]:
                    key = f'{stem}_{spec["fixture_index"]}.bin'
                    source = checked_file(root/row['case']/f'c{step:02d}', row['fixtures'][str(step)][key])
                    if source.stat().st_size != spec['bytes']: raise ValueError('reference/ABI extent mismatch')
                    dest = folder/f'{stem}_{spec["index"]}.bin'; shutil.copyfile(source, dest)
                    spec['fixture_sha256'] = sha256_file(dest)
                    if direction == 'outputs':
                        rec = row['fixtures'][str(step)]['boundary_0.bin']
                        alt = checked_file(root/row['case']/f'c{step:02d}', rec)
                        shutil.copyfile(alt, folder/'boundary_0.bin'); spec['boundary_sha256'] = rec['sha256']
                    plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'], ','.join(map(str, spec['shape'])))))
            (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
            placement = dict(inspection['placement'], case=name)
            cases.append(dict(case=name, source_case=step, **bound, checkpoint=row['checkpoint'],
                model=_file_record(model, row['case']+'/model.omc'),
                placement=placement, graphop_counts=placement['graphop_counts']))
    if not cases: raise ValueError('no layer0 checkpoint compiled')
    meta = dict(cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=report['comparison']['atol'],
        rtol=report['comparison']['rtol'], scope=SCOPE, ddk_id=ddk_id,
        source_evidence=report, output_initialization='untouched')
    _save(out/'fixture/manifest.json', meta)
    return meta


def diagnose(fixture, native, meta, comparison, report):
    from .accuracy import metrics
    rows, captured = [], {}
    for case in meta['cases']:
        matches = [r for r in comparison['cases'] if r['case'] == case['case']]
        if not matches: continue  # Recovery may have a recorded native failure later in the batch.
        measured = matches[0]
        spec = case['outputs'][0]
        expected = np.fromfile(fixture/case['case']/'golden_0.bin', dtype=spec['dtype']).reshape(spec['shape'])
        alt = checked_file(fixture/case['case'], dict(path='boundary_0.bin', bytes=spec['bytes'], sha256=spec['boundary_sha256']))
        boundary = np.fromfile(alt, dtype=spec['dtype']).reshape(spec['shape'])
        calls, arrays = [], []
        for i in range(10):
            actual = np.fromfile(native/case['case']/f'{i}.bin', dtype=spec['dtype']).reshape(spec['shape'])
            arrays.append(actual)
            calls.append(dict(call=i, elements=int(actual.size), zero_elements=int(np.count_nonzero(actual == 0)),
                all_zero=bool(np.all(actual == 0)), nonfinite_elements=int(np.count_nonzero(~np.isfinite(actual))),
                vs_original_gamma_rounding=metrics(actual, boundary, meta['atol'], meta['rtol'])))
        captured[case['source_case'], case['checkpoint']['source_edge']] = arrays
        tensors = measured['tensors']
        suspect = bool(np.any(expected != 0)) and any(c['all_zero'] for c in calls)
        rows.append(dict(case=case['case'], source_case=case['source_case'], checkpoint=case['checkpoint'],
            passed=measured['passed'], repeat_bitwise_stable=measured['repeat_bitwise_stable'],
            all_zero_output_with_nonzero_reference=suspect, calls=calls,
            source_vs_original_gamma_rounding=metrics(expected, boundary, meta['atol'], meta['rtol']),
            max_abs_error=max((r['max_abs_error'] for r in tensors), key=lambda x: float('inf') if x is None else x),
            max_relative_l2_error=max((r['relative_l2_error'] for r in tensors), key=lambda x: float('inf') if x is None else x),
            placement=case['placement']))
    coverage = meta['source_evidence']['coverage']
    cast_pairs = []
    for row in rows:
        point = row['checkpoint']; step = row['source_case']; before = (step, point.get('cast_of'))
        after = (step, point['source_edge'])
        if before not in captured: continue
        pairs = [a.astype('<f2').tobytes() == b.tobytes() for a, b in zip(captured[before], captured[after])]
        cast_pairs.append(dict(source_case=step, source_edge=before[1], cast_edge=after[1],
            all_calls_bytes_equal=all(pairs), calls_bytes_equal=pairs,
            scope='SEPARATELY_COMPILED_PREFIXES; DIFFERENCE_CANNOT_ALONE_LOCALIZE_VISIBILITY_OR_COMPUTE'))
    first = []
    for step in STEPS:
        subset = [r for r in rows if r['source_case'] == step]
        bad = next((r for r in subset if not r['passed']), None)
        first.append(dict(source_case=step, first_observed_failure=bad['checkpoint'] if bad else None,
            output_visibility_suspects=[r['checkpoint'] for r in subset if r['all_zero_output_with_nonzero_reference']],
            attribution='INDEPENDENT_PREFIX_COMPARISON_ONLY; BRANCHES_HAVE_NO_SINGLE_LINEAR_FIRST_ERROR'))
    return dict(artifact_kind=KIND, status='ANALYZED', scope=SCOPE,
        numerical_validation=comparison['numerical_validation'], model_execution='NOT_RUN',
        repeat_bitwise_stable=comparison['repeat_bitwise_stable'], coverage=coverage,
        coverage_complete=not coverage['missing_checkpoints'] and len(rows) == coverage['planned_device_cases'],
        source_cases=first, independent_cast_pairs=cast_pairs, cases=rows, full_model_first_divergence='NOT_ESTABLISHED',
        limitations=['Single-output prefix cuts may change fusion/kernel choice.',
            'Source FP32 and boundary casts do not establish internal instruction precision.',
            'NumPy source replay is diagnostic; full-model PyTorch goldens and thresholds are unchanged.'])

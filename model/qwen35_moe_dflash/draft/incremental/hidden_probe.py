"""Bounded first-query prefix from retained AIR and real case-zero embeddings.

No Torch/TorchAir, new weights, full-model export, or device access here.
The normal-output control and tapped copy retain the original source op bytes;
compiler copies use the full Draft's static reshape/weight-transpose rules.
An isolated prefix can narrow a hypothesis, not prove full-graph placement.
"""
from copy import deepcopy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import shutil

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, _read_air
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v, _new_id
from .air_zeros import _Shapes
from .air_fill import _FillValues
from .input_probe import _unary_path
from .torch_export import _schema, _disjoint

KIND = 'qwen35-draft-hidden-prefix-probe'
CASES = ('hidden_first_query', 'hidden_first_query_taps')
CHECKPOINT_ORDER = ('input_rms_fp32', 'input_rms_fp16', 'query_projection_fp16',
                    'query_rms_fp32', 'query_rms_fp16')
SCOPE = 'SOURCE_LAYER0_INPUT_RMS_QUERY_PROJECTION_QUERY_RMS_PREFIX_NOT_FULL_DRAFT'
WEIGHTS = ('draft.layers.0.input_layernorm.weight',
           'draft.layers.0.self_attn.q_proj.weight', 'draft.layers.0.self_attn.q_norm.weight')
GE_DTYPES = {0: '<f4', 1: '<f2', 3: '<i4', 9: '<i8'}


def prepare_compiler_air(raw, schema):
    """Use the full Draft's exact static rewrites on the compiler copy only."""
    from .air_reshape import fold_reshape_chains
    from .air_matmul import fold_weight_transposes
    prepared, reshapes = fold_reshape_chains(raw, schema)
    prepared, transposes = fold_weight_transposes(prepared, schema)
    # This prefix has no Fill, ZerosLike or embedding scale operations. Do not
    # silently retry an unsupported FileConstant transpose on this DDK.
    if transposes['skipped']:
        raise ValueError('unsupported prefix weight transpose: '+str(transposes['skipped']))
    return prepared, dict(policy='hidden-prefix-static-preparation-v1',
        status='APPLIED' if prepared != raw else 'UNCHANGED',
        removed_count=reshapes['removed_count'], weight_transpose_count=transposes['weight_transpose_count'],
        reshapes=reshapes, weight_transposes=transposes,
        source_sha256=hashlib.sha256(raw).hexdigest(), compiler_sha256=hashlib.sha256(prepared).hexdigest(),
        scope='COMPILER_COPY_ONLY; FULL_DRAFT_REWRITE_RULES; NO_WEIGHT_DTYPE_OR_GOLDEN_CHANGE')


def _replace_failed(root, replace):
    """Called under the source-snapshot lock, after new input preflight succeeds."""
    if not root.exists():
        return None
    if not replace:
        raise ValueError('hidden probe output exists; use --replace for a completed failed host compilation')
    tests = Path(os.environ['AI_RUN_DIR']).resolve()/'draft/tests'
    if tests not in root.parents or not root.is_dir() or root.is_symlink():
        raise ValueError('replacement requires a hidden probe directory below draft/tests')
    if (root/'result.json').is_symlink() or set(p.name for p in root.iterdir()) - {*CASES, 'result.json', 'previous-failure.json'}:
        raise ValueError('foreign files in hidden probe output; retained')
    state = read_json(root/'result.json')
    if (state.get('artifact_kind') != KIND or state.get('stage') != 'complete'
            or state.get('status') not in {'FAILED', 'PARTIAL'}
            or state.get('device_execution') != 'NOT_RUN' or state.get('numerical_validation') != 'NOT_RUN'
            or [r.get('case') for r in state.get('cases', [])] != list(CASES)):
        raise ValueError('replacement requires both completed failed host cases; output retained')
    previous = dict(original_model_sha256=state.get('original_model_sha256'),
        source_air_sha256=state.get('source_air_sha256'), cases=[],
        scope='PREVIOUS_FAILED_COMPILATION_EXCERPTS; ORIGINAL_LOG_PATHS_REUSED')
    for row in state['cases']:
        folder = root/row['case']; report = folder/'compilation/report.json'
        if (folder.resolve() != folder or report.resolve() != report or (folder/'model.omc').exists()
                or (folder/'request.json').exists() or row.get('status') != 'FAILED'
                or row.get('model_compilation') != 'FAILED'):
            raise ValueError('successful, active or foreign case cannot be replaced')
        compiled = read_json(report)
        if (compiled.get('artifact_kind') != 'qwen35-existing-air-compilation'
                or compiled.get('status') != 'FAILED' or compiled.get('stage') != 'omg'
                or compiled.get('model_compilation') != 'FAILED' or compiled.get('device_execution') != 'NOT_RUN'
                or type(compiled.get('returncode')) is not int or compiled['returncode'] == 0):
            raise ValueError('replacement requires known failed OMG exits; output retained')
        previous['cases'].append(dict(case=row['case'], error=row.get('error'), returncode=compiled['returncode'],
            log_excerpt=row.get('compilation_diagnostics', {}).get('log_excerpt')))
    for proc in Path('/proc').glob('[0-9]*'):
        if proc.name == str(os.getpid()):
            continue
        try:
            args = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd = proc.joinpath('cwd').resolve(strict=True)
        except OSError:
            continue
        if cwd == root or root in cwd.parents or any(
                arg.split('=', 1)[-1] == str(root) or arg.split('=', 1)[-1].startswith(str(root)+'/') for arg in args):
            raise ValueError('process still references hidden probe output; retained')
    shutil.rmtree(root)
    return previous


class Prefix:
    """Evaluate only the source operations on the first query dependency path."""
    def __init__(self, raw, schema):
        if schema.package != 'ge.proto':
            raise ValueError('hidden prefix requires the retained TorchAir GE source schema')
        self.raw, self.schema = raw, schema
        self.kind, _, _ = _decode(schema, raw)
        graph, self.graph_field = _graph_raw(raw, schema, self.kind)
        self.rows, self.op_field = _node_bytes(graph, schema)
        self.nodes = {n['name']: n for n, _, _ in self.rows}
        if len(self.nodes) != len(self.rows):
            raise ValueError('duplicate source node name')
        self.shapes = _Shapes(self.rows, schema)
        self.types = schema.enums[schema.package+'.DataType']

    def norm(self, weight):
        matches = []
        for n in self.nodes.values():
            if n['type'] != 'RmsNorm' or len(n.get('input', [])) != 2:
                continue
            gamma, _ = _unary_path(self.shapes, n['input'][1], {'Cast', 'Identity'})
            if (gamma['type'] == 'FileConstant' and
                    _attributes(gamma).get('file_path', {}).get('s') == 'weight/'+weight.replace('.', '_')):
                matches.append(n)
        if len(matches) != 1:
            raise ValueError('cannot uniquely identify source RMS: '+weight)
        return matches[0]

    def half_output(self, norm):
        matches = []
        for n in self.nodes.values():
            if n['type'] == 'Cast' and _attributes(n).get('dst_type', {}).get('i') == 1:
                root, _ = _unary_path(self.shapes, n['input'][0], {'Identity'})
                if root['name'] == norm['name']:
                    matches.append(n['name']+':0')
        if len(matches) != 1:
            raise ValueError('expected one original FP16 Cast after '+norm['name'])
        return matches[0]

    def data_inputs(self, node):
        """Resolve live input slots without treating an absent MatMul bias as data."""
        edges, descs = node.get('input', []), node.get('input_desc', [])
        name, kind = node['name'], node['type']
        if kind == 'MatMul':
            if len(edges) not in (2, 3) or len(descs) != len(edges) or not all(edges[:2]):
                raise ValueError('source MatMul requires two connected matrix ports and matching descriptors: '+name)
            if len(edges) == 3:
                bias = descs[2]
                # TorchAir mm emits MatMul(x1, x2, None). The third slot is
                # get_invalid_desc(), not an ND tensor. Keep its raw bytes.
                # Pinned schema source: Ascend/torchair e1118e8b, MatMul and
                # ge/_ge_graph.py:get_invalid_desc (_is_unfed_optional=1).
                # Proto3 omits the default DT_UNDEFINED=0 enum value.
                if (edges[2] != '' or bias.get('dtype', self.types['DT_UNDEFINED']) != self.types['DT_UNDEFINED']
                        or _attributes(bias).get('_is_unfed_optional', {}).get('i') != 1
                        or bias.get('shape', {}).get('dim')
                        or bias.get('name') not in (None, 'bias')):
                    raise ValueError('source MatMul input[2] must be a marked absent optional bias: '+name)
            return list(enumerate(edges[:2]))
        if kind == 'Data' and edges in ([], ['']):
            return []
        if any(not e for e in edges):
            raise ValueError('empty required source input: '+name+' ('+kind+')')
        return list(enumerate(edges))

    def select(self, embedding_name):
        first, query = self.norm(WEIGHTS[0]), self.norm(WEIGHTS[2])
        source, _ = _unary_path(self.shapes, first['input'][0], {'Cast', 'Identity'})
        if source['name'] != embedding_name or source['type'] != 'Data':
            raise ValueError('first RMS must descend directly from the bound host embeddings')
        self.terminal = self.half_output(query)
        selected, active = set(), set()
        self.absent_optional_inputs = []
        allowed = {'Data', 'FileConstant', 'Const', 'Cast', 'Identity', 'Reshape',
                   'Transpose', 'MatMul', 'RmsNorm'}
        def visit(edge):
            n = self.shapes.node(edge); name = n['name']
            if name in selected:
                return
            if name in active or len(active) > 48 or len(selected) > 64:
                raise ValueError('cyclic or oversized query prefix')
            if n['type'] not in allowed or n.get('subgraph_name'):
                raise ValueError('unsupported query-prefix node: '+name+' ('+n['type']+')')
            inputs = self.data_inputs(n)
            absent = n['type'] == 'MatMul' and len(n['input']) == 3
            for side in ('input', 'output'):
                for index, d in enumerate(n.get(side+'_desc', [])):
                    if side == 'input' and absent and index == 2:
                        self.absent_optional_inputs.append(dict(node=name, input_index=index,
                            descriptor=deepcopy(d), reason='MARKED_ABSENT_MATMUL_BIAS; RAW_BYTES_PRESERVED'))
                        continue
                    if d.get('layout', 'ND') != 'ND':
                        raise ValueError(f'source prefix requires ND descriptors: {name} ({n["type"]}) '
                            f'{side}[{index}] layout={d.get("layout")!r} dtype={d.get("dtype")} '
                            f'shape={d.get("shape", {}).get("dim")}')
            active.add(name)
            for _, edge in inputs:
                visit(edge)
            active.remove(name); selected.add(name)
        visit(self.terminal)
        self.selected = selected
        data = [n['name'] for n in self.nodes.values() if n['name'] in selected and n['type'] == 'Data']
        matrices = [n for n in self.nodes.values() if n['name'] in selected and n['type'] == 'MatMul']
        norms = [n['name'] for n in self.nodes.values() if n['name'] in selected and n['type'] == 'RmsNorm']
        files = {_attributes(self.nodes[n]).get('file_path', {}).get('s') for n in selected
                 if self.nodes[n]['type'] == 'FileConstant'}
        if (data != [embedding_name] or len(matrices) != 1 or set(norms) != {first['name'], query['name']}
                or files != {'weight/'+w.replace('.', '_') for w in WEIGHTS}):
            raise ValueError('prefix must contain one embedding Data, two RMS, one MatMul and three source weights')
        self.input = embedding_name
        self.checkpoints = [
            ('input_rms_fp32', first['name']+':0'), ('input_rms_fp16', self.half_output(first)),
            ('query_projection_fp16', matrices[0]['name']+':0'),
            ('query_rms_fp32', query['name']+':0'), ('query_rms_fp16', self.terminal)]
        if any(edge.rsplit(':', 1)[0] not in selected for _, edge in self.checkpoints):
            raise ValueError('checkpoint is not on the first query dependency path')

    def evaluate(self, embeddings, weights, *, reference_gamma_boundary=False):
        """FP32 reductions/matmul, explicit source casts; separate rounding hypothesis."""
        values = {}
        def evaluate(edge):
            if edge in values:
                return values[edge]
            n = self.shapes.node(edge); name, kind = n['name'], n['type']
            attrs = _attributes(n)
            args = [evaluate(e) for _, e in self.data_inputs(n)]
            if kind == 'Data':
                y = embeddings
            elif kind == 'FileConstant':
                relative = attrs['file_path']['s']
                y = weights[relative]
                if (attrs.get('dtype', {}).get('i') != 1 or
                        attrs.get('shape', {}).get('list', {}).get('i') != list(y.shape)):
                    raise ValueError('source weight shape/dtype differs: '+relative)
            elif kind == 'Const':
                y = np.array(self.shapes.integers(edge), dtype=np.int64)
            elif kind == 'Cast':
                dt = GE_DTYPES.get(attrs.get('dst_type', {}).get('i'))
                if len(args) != 1 or dt is None:
                    raise ValueError('unsupported source Cast')
                y = args[0].astype(dt)
            elif kind == 'Identity':
                if len(args) != 1:
                    raise ValueError('invalid source Identity')
                y = args[0]
            elif kind == 'Reshape':
                if len(args) != 2 or attrs.get('axis', {}).get('i', 0) != 0:
                    raise ValueError('unsupported source Reshape')
                shape = args[1].reshape(-1).tolist()
                if shape.count(-1) > 1 or any(d == 0 or d < -1 for d in shape):
                    raise ValueError('dynamic/zero source reshape is unsupported')
                y = args[0].reshape(shape)
            elif kind == 'Transpose':
                if len(args) != 2:
                    raise ValueError('unsupported source Transpose')
                y = args[0].transpose(args[1].reshape(-1).tolist())
            elif kind == 'MatMul':
                if len(args) != 2 or any(a.ndim != 2 or a.dtype != np.float16 for a in args):
                    raise ValueError('first query MatMul must have two FP16 matrices')
                a, b = args
                if attrs.get('transpose_x1', {}).get('b', False):
                    a = a.T
                if attrs.get('transpose_x2', {}).get('b', False):
                    b = b.T
                if a.shape != (16, 2048) or b.shape != (2048, 4096):
                    raise ValueError('first query matrix geometry differs from 16x2048x4096')
                y = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
            elif kind == 'RmsNorm':
                if len(args) != 2 or any(a.dtype != np.float32 for a in args):
                    raise ValueError('source RMS must request FP32 input and gamma')
                x, gamma = args; eps = attrs.get('epsilon', {}).get('f')
                if (not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0
                        or gamma.shape != (x.shape[-1],)):
                    raise ValueError('unsupported RMS epsilon/gamma')
                rstd = np.float32(1) / np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32)+np.float32(eps))
                normalized = x*rstd
                y = ((normalized.astype(np.float16)*gamma.astype(np.float16)).astype(np.float32)
                     if reference_gamma_boundary else normalized*gamma)
            else:
                raise ValueError('unhandled prefix operation: '+kind)
            if y.size > (16 << 20) or not np.isfinite(y).all():
                raise ValueError('oversized/nonfinite prefix reference at '+name)
            # Source AIR may omit intermediate dimensions; explicit conflicts fail.
            if kind not in {'FileConstant', 'Const'}:
                descs = n.get('output_desc', [])
                if not descs:
                    raise ValueError('source operation has no output descriptor: '+name)
                recorded = descs[0].get('shape', {}).get('dim')
                if recorded and recorded != list(y.shape):
                    raise ValueError('source shape conflicts with evaluated shape: '+name)
                dtype = {'float16': 'DT_FLOAT16', 'float32': 'DT_FLOAT', 'int32': 'DT_INT32', 'int64': 'DT_INT64'}[str(y.dtype)]
                _FillValues(self.shapes).descriptor_dtype(descs[0], self.types[dtype], 'source output '+name,
                    allow_placeholder=kind in {'Cast', 'Reshape', 'Transpose', 'MatMul'})
            values[edge] = np.ascontiguousarray(y)
            return values[edge]
        evaluate(self.terminal)
        result = {name: evaluate(edge).copy() for name, edge in self.checkpoints}
        for name, shape in zip(result, ((1,16,2048), (1,16,2048), (16,4096),
                                       (1,16,32,128), (1,16,32,128))):
            if result[name].shape != shape:
                raise ValueError('source checkpoint geometry differs: '+name)
        return result

    def extract(self, reference, *, taps):
        schema = self.schema
        def blob(message, field, value, kind=9, repeated=False):
            return _blob(_field(schema, message, field, kind, repeated), value)
        def descriptor(a):
            dtype = self.types['DT_FLOAT16' if a.dtype == np.float16 else 'DT_FLOAT']
            return (_v(_field(schema, 'TensorDescriptor', 'dtype', 14, False) << 3)+_v(dtype)
                +blob('TensorDescriptor', 'shape', blob('ShapeDef', 'dim', b''.join(_v(d) for d in a.shape), 3, True), 11)
                +blob('TensorDescriptor', 'layout', 'ND')+blob('TensorDescriptor', 'device_type', 'NPU'))
        checkpoints = self.checkpoints if taps else self.checkpoints[-1:]
        net = blob('OpDef', 'name', 'NetOutput')+blob('OpDef', 'type', 'NetOutput')
        for name, edge in checkpoints:
            net += blob('OpDef', 'input', edge, repeated=True)+blob('OpDef', 'input_desc', descriptor(reference[name]), 11, True)
        net = _new_id(net, schema, max(n.get('id', 0) for n in self.nodes.values())+1)
        attr = _field(schema, 'OpDef', 'attr', 11, True)
        index = blob('OpDef.AttrEntry', 'key', 'index')+blob('OpDef.AttrEntry', 'value',
            _v(_field(schema, 'AttrDef', 'i', 3, False) << 3)+_v(0), 11)
        encoded, preserved = [], {}
        for n, payload, _ in self.rows:
            if n['name'] not in self.selected:
                continue
            if n['name'] == self.input:
                payload = b''.join(e for f, _, e, p in _parts(payload)
                    if f != attr or schema.decode(p, 'OpDef.AttrEntry')['key'] != 'index')+_blob(attr, index)
            else:
                preserved[n['name']] = hashlib.sha256(payload).hexdigest()
            encoded.append(_blob(self.op_field, payload))
        graph = blob('GraphDef', 'name', 'draft_hidden_prefix')+b''.join(encoded)+_blob(self.op_field, net)
        raw = graph if self.graph_field is None else b''.join(
            _blob(f, graph) if f == self.graph_field else e for f, _, e, _ in _parts(self.raw))
        _decode(schema, raw)
        declared = dict(inputs=[dict(name=self.input, dtype='float16', shape=[1, 16, 2048])],
            outputs=[dict(name=name, source_edge=edge, dtype=str(reference[name].dtype), shape=list(reference[name].shape))
                     for name, edge in checkpoints])
        signatures = {(s['dtype'], tuple(s['shape'])) for s in declared['outputs']}
        if len(signatures) != len(declared['outputs']):
            raise ValueError('tap shapes/dtypes are ambiguous for the existing probe ABI binder')
        return raw, dict(graph=declared, preserved_payload_sha256=preserved,
            absent_optional_inputs=self.absent_optional_inputs,
            changes='ONLY_DATA_INDEX_AND_NETOUTPUT; ORIGINAL_PREFIX_COMPUTE_PAYLOADS_PRESERVED',
            scope=SCOPE, full_graph_compilation_preserved=False, taps=taps)


def compile_probe(air_export, source, result, output, *, ddk_root, ddk_id,
                  tbe_diagnosis, timeout_seconds=1800, progress=print, replace=False):
    from .accuracy import analyze_result
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    root, air, prepared, recovered = output_path(output), Path(air_export).resolve(), Path(source).resolve(), Path(result).resolve()
    if Path(output).absolute() != root:
        raise ValueError('hidden probe output must not contain symlink or traversal components')
    _disjoint(root, air, prepared, recovered, ddk_root, tbe_diagnosis)
    if root.exists() and not replace:
        raise ValueError('hidden probe output exists; use --replace for a completed failed host compilation')
    progress('verifying completed untouched replay and the source AIR binding')
    analyze_result(recovered)
    run = read_json(recovered/'result.json', 4 << 20)
    fixture = Path(run['fixture']); fixture = fixture if fixture.is_absolute() else recovered/fixture
    meta, binding = read_json(fixture/'manifest.json'), read_json(fixture/'pytorch-abi.json')
    manifest, plan = read_json(air/'manifest.json'), read_json(prepared/'prepared.json')
    raw, _ = _read_air(air/'export.air')
    if (run.get('output_initialization') != 'untouched' or meta.get('frontend') != 'pytorch'
            or binding['source_air_sha256'] != hashlib.sha256(raw).hexdigest()
            or binding['prepared_sha256'] != sha256_file(prepared/'prepared.json')
            or binding['packed_model_sha256'] != meta['model']['sha256']
            or manifest.get('attention', {}).get('policy') != 'composed-attention-v1'):
        raise ValueError('requires the completed, matching untouched composed-attention replay')
    steps = [s for s in meta['cases'] if s['index'] == 0]
    if (len(steps) != 1 or any(steps[0][k] != v for k, v in
            dict(start=0, valid=0, proposals=15, label='empty-context').items())):
        raise ValueError('expected the original empty-context case zero')
    spec = next(s for s in meta['inputs'] if s['name'] == 'block_embeddings')
    if (spec['dtype'], spec['shape'], spec['bytes']) != ('float16', [1, 16, 2048], 65536):
        raise ValueError('host embedding contract differs')
    records = [r for r in meta['input_files'] if r['case'] == 0 and r['index'] == spec['index']]
    if len(records) != 1 or records[0]['bytes'] != spec['bytes']:
        raise ValueError('missing or duplicate real embedding fixture')
    x = np.fromfile(checked_file(fixture, records[0]), dtype='<f2').reshape(spec['shape'])
    schema = _schema((air/'air-schema.pb').read_bytes(), manifest['schema'])
    prefix = Prefix(raw, schema); prefix.select(binding['source_input_roles']['block_embeddings'])
    bound = {r['name']: r for r in binding['reference_weights']}
    if len(bound) != len(binding['reference_weights']):
        raise ValueError('duplicate bound reference weight')
    weights, weight_records = {}, {}
    for name, shape in zip(WEIGHTS, ((2048,), (4096, 2048), (128,))):
        record = plan['weights'][name]; relative = 'weight/'+name.replace('.', '_')
        if (record['sha256'] != bound[name]['sha256'] or record['bytes'] != math.prod(shape)*2
                or record.get('dtype') != 'float16' or record.get('shape') != list(shape)):
            raise ValueError('prepared first-query weight contract differs: '+name)
        checked_file(prepared, record)
        path = checked_file(air, dict(path=relative, bytes=record['bytes'], sha256=record['sha256']))
        weights[relative] = np.fromfile(path, dtype='<f2').reshape(shape)
        weight_records[relative] = dict(path=relative, bytes=record['bytes'], sha256=record['sha256'])
    progress('evaluating the two source RMS and query projection on real case-zero input')
    reference = prefix.evaluate(x, weights)
    alternate = prefix.evaluate(x, weights, reference_gamma_boundary=True)
    plans = []
    for index, name in enumerate(CASES):
        extracted, receipt = prefix.extract(reference, taps=bool(index))
        compiler_raw, preparation = prepare_compiler_air(extracted, schema)
        plans.append((name, extracted, receipt, compiler_raw, preparation))
        progress(f'prepared {name}: folded {preparation["weight_transpose_count"]} weight Transposes, '
                 f'{preparation["removed_count"]} intermediate Reshapes; original AIR and golden retained')
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 256 << 20:
        raise OSError('hidden prefix preparation requires 256 MiB free')
    with (root.parent/('.'+root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = _replace_failed(root, replace)
        root.mkdir()
        if previous is not None:
            _save(root/'previous-failure.json', previous)
        state = dict(artifact_kind=KIND, status='PARTIAL', scope=SCOPE, stage='compile',
            model_compilation='PARTIAL', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
            original_model_sha256=meta['model']['sha256'], source_air_sha256=binding['source_air_sha256'],
            source_result=str(recovered), source_case=steps[0], input_record=records[0],
            source_weights=weight_records, ddk_id=ddk_id, output_initialization='untouched',
            reference='NUMPY_FP32_RMS_AND_MATMUL_WITH_SOURCE_FP16_CASTS; CPU_REDUCTION_ORDER_MAY_DIFFER',
            comparison=meta['comparison'], cases=[])
        state['probe_source_sha256'] = sha256_file(Path(__file__))
        if previous is not None:
            state['replaced_failed_compilation'] = 'previous-failure.json'
        for name, extracted, receipt, compiler_raw, preparation in plans:
            folder = root/name; (folder/'air/weight').mkdir(parents=True)
            (folder/'air/export.air').write_bytes(extracted)
            for relative, record in weight_records.items():
                shutil.copyfile(checked_file(air, record), folder/'air'/relative)
            fixtures = folder/'fixture'; fixtures.mkdir()
            x.tofile(fixtures/'input_0.bin')
            for i, s in enumerate(receipt['graph']['outputs']):
                reference[s['name']].tofile(fixtures/f'golden_{i}.bin')
                alternate[s['name']].tofile(fixtures/f'boundary_{i}.bin')
            row = dict(case=name, status='FAILED', model_compilation='NOT_RUN', extraction=receipt,
                air_preparation={k: preparation[k] for k in ('policy', 'status', 'removed_count', 'weight_transpose_count')},
                fixtures={p.name: dict(path=p.name, bytes=p.stat().st_size, sha256=sha256_file(p)) for p in fixtures.iterdir()})
            state['cases'].append(row); _save(root/'result.json', state)
            try:
                def transform(snapshot):
                    if snapshot.read_bytes() != extracted:
                        raise ValueError('prefix AIR snapshot differs from the prepared source')
                    return compiler_raw, preparation
                progress('compiling '+name+'; only the small prefix is recompiled')
                compiled = compile_existing_air(folder/'air/export.air', folder/'compilation', ddk_root,
                    ddk_id=ddk_id, tbe_diagnosis=tbe_diagnosis, timeout_seconds=timeout_seconds,
                    input_layout='air-directory', use_origin_format=True, save_weights_as_external_data=True,
                    progress=lambda r: progress(r['message']), air_transform=transform)
                row['model_compilation'] = compiled['model_compilation']
                if compiled['status'] != 'COMPILED':
                    raise ValueError(str(compiled.get('error', compiled['status'])))
                sidecars = [dict(r, path=str(folder/'compilation'/r['path'])) for r in compiled['external_data_candidates']
                            if Path(r['path']).suffix == '.weight']
                packed = embed_external_weights(folder/'compilation'/compiled['omc']['path'], folder/'model.omc',
                    Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so',
                    expected_source_sha256=compiled['omc']['sha256'], sidecars=sidecars, progress=progress)
                _save(folder/'weight-packing.json', packed)
                if packed['status'] != 'EMBEDDED' or (folder/'model.omc').stat().st_size > 64 << 20:
                    raise ValueError('prefix packing failed or exceeded 64 MiB')
                import stable_arg_max_ge
                inspection = stable_arg_max_ge.inspect_model(folder/'model.omc', ddk_root, receipt['graph'])
                inspection['placement']['case'] = name
                row.update(status='COMPILED',
                    placement=inspection['placement'], model=dict(path=name+'/model.omc',
                        bytes=(folder/'model.omc').stat().st_size, sha256=sha256_file(folder/'model.omc')))
            except Exception as error:
                row.update(error=f'{type(error).__name__}: {error}',
                    compilation_diagnostics=compilation_diagnostics(folder/'compilation'))
            _save(root/'result.json', state)
        passed = all(r['status'] == 'COMPILED' for r in state['cases'])
        failed = all(r['model_compilation'] == 'FAILED' for r in state['cases'])
        state.update(status='COMPILED' if passed else ('FAILED' if failed else 'PARTIAL'), stage='complete',
                     model_compilation='PASSED' if passed else ('FAILED' if failed else 'PARTIAL'))
        _save(root/'result.json', state)
    return state


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    from validate_draft_vector import selected_cases, WARMUP, REPETITIONS
    import stable_arg_max_ge
    root, out = Path(root), Path(out)
    report = read_json(root/'result.json')
    if report.get('status') != 'COMPILED' or report.get('ddk_id') != ddk_id:
        raise ValueError('both prefix controls must compile for the selected DDK before execution')
    if selected_cases(CASES if selection is None else selection, CASES) != list(CASES):
        raise ValueError('run both ordinary and tapped prefix controls together')
    rows = {r['case']: r for r in report['cases']}
    if set(rows) != set(CASES) or len(report['cases']) != len(CASES):
        raise ValueError('missing/duplicate prefix cases')
    cases = []
    for name in CASES:
        row = rows[name]; model = checked_file(root, row['model'])
        inspection = stable_arg_max_ge.inspect_model(model, ddk, row['extraction']['graph'])
        inspection['placement']['case'] = name
        if row['status'] != 'COMPILED':
            raise ValueError('query prefix did not compile')
        bound = deepcopy(inspection['io']); folder = out/'fixture'/name; folder.mkdir(parents=True)
        plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}']
        for direction, stem in (('inputs', 'input'), ('outputs', 'golden')):
            for spec in bound[direction]:
                member = f'{stem}_{spec["fixture_index"]}.bin'
                path = checked_file(root/name/'fixture', row['fixtures'][member])
                if path.stat().st_size != spec['bytes']:
                    raise ValueError('prefix reference extent differs from compiled ABI')
                dest = folder/f'{stem}_{spec["index"]}.bin'; shutil.copyfile(path, dest)
                spec['fixture_sha256'] = sha256_file(dest)
                if direction == 'outputs':
                    alt = f'boundary_{spec["fixture_index"]}.bin'
                    dest = folder/f'boundary_{spec["index"]}.bin'
                    shutil.copyfile(checked_file(root/name/'fixture', row['fixtures'][alt]), dest)
                    spec['boundary_sha256'] = sha256_file(dest)
                plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'], ','.join(map(str, spec['shape'])))))
        (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
        cases.append(dict(case=name, **bound, model=_file_record(model, name+'/model.omc'),
            graphop_counts=inspection['placement']['graphop_counts'], placement=inspection['placement']))
    meta = dict(cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=report['comparison']['atol'],
        rtol=report['comparison']['rtol'], scope=SCOPE, ddk_id=ddk_id, source_evidence=report,
        output_initialization='untouched')
    _save(out/'fixture/manifest.json', meta)
    return meta


def diagnose(fixture, native, meta, comparison, report):
    from .accuracy import metrics
    summaries, terminal, captured = [], {}, {}
    for case in meta['cases']:
        name = case['case']; tensors = next(r['tensors'] for r in comparison['cases'] if r['case'] == name)
        rows = []
        for spec in case['outputs']:
            alt = checked_file(fixture/name, dict(path=f'boundary_{spec["index"]}.bin',
                bytes=spec['bytes'], sha256=spec['boundary_sha256']))
            boundary = np.fromfile(alt, dtype=GE_DTYPES[1 if spec['dtype'] == 'float16' else 0]).reshape(spec['shape'])
            expected = np.fromfile(fixture/name/f'golden_{spec["index"]}.bin', dtype=boundary.dtype).reshape(spec['shape'])
            offset = sum(s['bytes'] for s in case['outputs'][:spec['index']])
            actual = []
            for call in range(meta['repetitions']):
                raw = (native/name/f'{call}.bin').read_bytes()[offset:offset+spec['bytes']]
                actual.append(np.frombuffer(raw, dtype=boundary.dtype).reshape(spec['shape']))
            observed = [r for r in tensors if r['name'] == spec['name']]
            if len(actual) != 10 or len(observed) != 10:
                raise ValueError('hidden diagnosis requires every measured call')
            captured[name, spec['name']] = actual
            rows.append(dict(name=spec['name'], source_edge=spec['source_edge'],
                runtime_slot=spec['index'], dtype=spec['dtype'], shape=spec['shape'],
                snapshot_offset_bytes=offset, snapshot_bytes=spec['bytes'],
                actual_statistics=[dict(call=i, elements=int(a.size),
                    zero_elements=int(np.count_nonzero(a == 0)),
                    nonfinite_elements=int(np.count_nonzero(~np.isfinite(a))),
                    all_zero=bool(np.all(a == 0))) for i, a in enumerate(actual)],
                all_calls_passed=all(r['passed'] for r in observed),
                all_calls_bytes_equal=all(r['bytes_equal'] for r in observed),
                max_abs_error=max((r['max_abs_error'] for r in observed), key=lambda v: float('inf') if v is None else v),
                max_relative_l2_error=max((r['relative_l2_error'] for r in observed), key=lambda v: float('inf') if v is None else v),
                source_vs_original_rounding=metrics(expected, boundary, meta['atol'], meta['rtol']),
                call0_vs_original_rounding=metrics(actual[0], boundary, meta['atol'], meta['rtol'])))
            if spec['name'] == 'query_rms_fp16':
                terminal[name] = actual
        rows.sort(key=lambda r: CHECKPOINT_ORDER.index(r['name']))
        summaries.append(dict(case=name, placement=case['placement'], checkpoints=rows))
    same = all(a.tobytes() == b.tobytes() for a, b in zip(terminal[CASES[0]], terminal[CASES[1]]))
    tapped = next(r['checkpoints'] for r in summaries if r['case'] == CASES[1])
    pairs = []
    for label, left, right in (
            ('input_rms_cast_pair', (CASES[1], 'input_rms_fp32'), (CASES[1], 'input_rms_fp16')),
            ('query_rms_cast_pair', (CASES[1], 'query_rms_fp32'), (CASES[1], 'query_rms_fp16')),
            ('query_rms_recast_vs_normal', (CASES[1], 'query_rms_fp32'), (CASES[0], 'query_rms_fp16'))):
        calls = []
        for i, (a, b) in enumerate(zip(captured[left], captured[right])):
            cast = a.astype('<f2')
            calls.append(dict(call=i, bytes_equal=cast.tobytes() == b.tobytes(),
                different_elements=int(np.count_nonzero(cast.view('<u2') != b.view('<u2'))),
                **metrics(cast, b, meta['atol'], meta['rtol'])))
        pairs.append(dict(name=label, left=list(left), right=list(right),
            operation='CAST_RETRIEVED_FP32_TO_FP16; NOT_INTERNAL_PRECISION_PROOF',
            all_calls_bytes_equal=all(c['bytes_equal'] for c in calls), calls=calls))
    usable = same and comparison['repeat_bitwise_stable']
    first_failed = next((r['name'] for r in tapped if not r['all_calls_passed']), None)
    return dict(artifact_kind=KIND, status='ANALYZED', scope=SCOPE, model_execution='NOT_RUN',
        numerical_validation=comparison['numerical_validation'], repeat_bitwise_stable=comparison['repeat_bitwise_stable'],
        normal_vs_tapped_terminal_all_calls_bitwise_equal=same,
        checkpoint_attribution_usable=usable, retained_output_crosschecks=pairs,
        first_nonexact_checkpoint=next((r['name'] for r in tapped if not r['all_calls_bytes_equal']), None),
        observed_tapped_first_failed_checkpoint=first_failed,
        first_failed_checkpoint=first_failed if usable else None,
        cases=summaries, full_model_first_divergence='NOT_ESTABLISHED_BY_ISOLATED_PREFIX',
        next=('Resolve changed compilation or output visibility before attributing tapped values' if not same else
              'Use first failed checkpoint to choose the next bounded accuracy experiment; if all pass, extend through layer0 attention/residual'),
        limitations=['Extra outputs and prefix cutting can change kernel selection/fusion.',
            'Decoded placement is reported per prefix; CPU or unassigned casts are not silently treated as NPU computation.',
            'NumPy reduction order may differ from PyTorch and the NPU; original full-model goldens and gates are unchanged.',
            'Original-rounding comparisons are diagnostic alternatives, never acceptance replacements.'])

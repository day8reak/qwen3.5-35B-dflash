"""Two retained-source C16 feature-projection prefixes, not a full-model rewrite.

The candidate removes only the input layout path. Device CPU packing is timed
inside the probe. Original CPU-reference gates and device-baseline equivalence
are reported separately; neither establishes full Draft accuracy/performance.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, _read_air
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _parts, _v, _new_id
from .hidden_layer_probe import LayerTrace
from .hidden_probe import prepare_compiler_air
from .torch_export import _schema, _disjoint

KIND = 'qwen35-draft-feature-pack-prefix'
SCOPE = 'SOURCE_C16_FEATURE_PROJECTION_PREFIX; NOT_FULL_DRAFT'
STEPS = (2, 16, 17)
VARIANTS = ('baseline', 'packed')
CASES = tuple(f'feature_pack_c{s:02d}_{v}' for s in STEPS for v in VARIANTS)
WEIGHT = 'weight/draft_fc_weight'
INPUT_SHAPE = [1, 16, 16384]
PACKED_SHAPE = [1, 1024, 16, 16]
OUTPUT_SHAPE = [1, 16, 2048]
PREPARATION = dict(baseline='copy-c16-v1', packed='feature-pack-c16-v1')
TIMING = 'CPU_PACK_SETINPUT_EXECUTE_WALL; PREALLOCATED; SNAPSHOTS_EXCLUDED'


def input_contract(raw, schema, input_name, shape, *, input_device='NPU'):
    """Check both sides of Data, including its unconnected external input.

    TorchAir Data copies output_desc to input_desc (named y/x):
    Ascend/torchair e1118e8b, python/torchair/ge/_ge_graph.py, Data().
    The retained baseline may have a CPU external input. That source placement
    is not compiled-op placement; preserve it while checking shape/dtype/layout.
    The newly written packed descriptors must both declare NPU.
    """
    _, _, graph = _decode(schema, raw)
    nodes = [n for n in graph['op'] if n['type'] == 'Data']
    if len(nodes) != 1 or nodes[0]['name'] != input_name or nodes[0].get('input', []) not in ([], ['']):
        raise ValueError('projection requires one unconnected external Data input')
    if input_device not in ('CPU', 'NPU'):
        raise ValueError('unsupported source Data input device: '+repr(input_device))
    observed = {}
    half = schema.enums[schema.package+'.DataType']['DT_FLOAT16']
    for side in ('input_desc', 'output_desc'):
        expected = dict(shape=list(shape), dtype='float16', layout='ND',
            device_type=input_device if side == 'input_desc' else 'NPU')
        rows = nodes[0].get(side, [])
        actual = [dict(shape=d.get('shape', {}).get('dim'),
            dtype='float16' if d.get('dtype') == half else d.get('dtype'),
            layout=d.get('layout'), device_type=d.get('device_type')) for d in rows]
        if actual != [expected]:
            raise ValueError('projection Data descriptor mismatch: '+json.dumps(dict(
                graph=graph.get('name'), node=input_name, side=side, expected=expected, actual=actual)))
        observed[side] = actual[0]
    return observed


def pack_words(a):
    """Permutation oracle for arbitrary element indices or storage words."""
    return np.ascontiguousarray(a.reshape(4, 4, 1024, 4, 4).transpose(2, 0, 3, 1, 4)).reshape(PACKED_SHAPE)


class Projection(LayerTrace):
    def layout(self, edge, inputs):
        """Evaluate views on integer labels: prove every source element mapping."""
        if edge in inputs:
            return inputs[edge]
        n = self.shapes.node(edge); args = n.get('input', [])
        kind, attrs = n['type'], _attributes(n)
        if kind == 'Identity' and len(args) == 1:
            return self.layout(args[0], inputs)
        if kind in ('Reshape', 'Transpose') and len(args) == 2:
            x, axes = self.layout(args[0], inputs), self.shapes.integers(args[1])
            if kind == 'Reshape':
                if attrs.get('axis', {}).get('i', 0) != 0:
                    raise ValueError('partial input layout Reshape')
                return x.reshape(axes)
            return x.transpose(axes)
        raise ValueError('projection layout must contain only exact views: '+edge)

    def select_projection(self, input_name):
        norm = self.norm('draft.hidden_norm.weight')
        cast = self.shapes.node(norm['input'][0])
        if cast['type'] != 'Cast' or _attributes(cast).get('dst_type', {}).get('i', 0) != 0:
            raise ValueError('context RMS must consume the original FP32 Cast')
        self.terminal, self.input = cast['input'][0], input_name
        ordered = self.dependencies(self.terminal)
        nodes = [self.nodes[n] for n in ordered]
        convs = [n for n in nodes if n['type'] == 'Conv2D']
        if (len(convs) != 1 or [n['name'] for n in nodes if n['type'] == 'Data'] != [input_name]
                or any(n['type'] not in {'Data', 'Const', 'FileConstant', 'Identity', 'Reshape', 'Transpose', 'Conv2D'} for n in nodes)):
            raise ValueError('expected one spatial Conv with only source views/weights')
        self.conv = convs[0]
        if len(self.live_inputs(self.conv)) != 2:
            raise ValueError('projection Conv must not have bias/offset inputs')
        files = [n for n in nodes if n['type'] == 'FileConstant']
        if len(files) != 1 or _attributes(files[0]).get('file_path', {}).get('s') != WEIGHT:
            raise ValueError('projection must use only the original fc weight')
        attrs = _attributes(files[0])
        self.weight_shape = attrs.get('shape', {}).get('list', {}).get('i')
        if attrs.get('dtype', {}).get('i') != 1 or self.weight_shape not in ([2048, 16384], [2048, 1024, 4, 4]):
            raise ValueError('original FP16 feature weight geometry differs')
        if self.shapes.shape(input_name+':0') != INPUT_SHAPE:
            raise ValueError('source feature input must be C16 FP16')
        source_dtype = self.nodes[input_name]['output_desc'][0].get('dtype')
        if source_dtype != self.types['DT_FLOAT16']:
            raise ValueError('source feature input must be FP16')
        labels = np.arange(16*16384, dtype=np.int32).reshape(INPUT_SHAPE)
        actual = self.layout(self.conv['input'][0], {input_name+':0': labels})
        if actual.shape != tuple(PACKED_SHAPE) or not np.array_equal(actual, pack_words(labels)):
            raise ValueError('source input layout does not match C16 packing')
        if sum(self.nodes[n]['type'] == 'Transpose' for n in self.dependencies(self.conv['input'][0])) != 2:
            raise ValueError('expected the two source input Transposes')
        labels = np.arange(16*2048, dtype=np.int32).reshape(1, 2048, 4, 4)
        output = self.layout(self.terminal, {self.conv['name']+':0': labels})
        if output.shape != tuple(OUTPUT_SHAPE) or not np.array_equal(output, labels.transpose(0, 2, 3, 1).reshape(OUTPUT_SHAPE)):
            raise ValueError('projection output token layout differs')
        self.selected = set(ordered)

    def extract_projection(self, variant):
        if variant not in VARIANTS:
            raise ValueError('unknown feature prefix variant')
        schema = self.schema
        def blob(msg, field, value, kind=9, repeated=False):
            return _blob(_field(schema, msg, field, kind, repeated), value)
        def desc(shape):
            return (_v(_field(schema, 'TensorDescriptor', 'dtype', 14, False) << 3)+_v(self.types['DT_FLOAT16'])
                +blob('TensorDescriptor', 'shape', blob('ShapeDef', 'dim', b''.join(_v(d) for d in shape), 3, True), 11)
                +blob('TensorDescriptor', 'layout', 'ND')+blob('TensorDescriptor', 'device_type', 'NPU'))
        selected = set(self.selected)
        if variant == 'packed':
            # Revisit dependencies with only Conv's input[0] replaced. This
            # removes dead layout constants too, without relying on node suffixes.
            selected = {self.input}
            def visit(edge):
                n = self.shapes.node(edge)
                if n['name'] in selected:
                    return
                selected.add(n['name'])
                for i, arg in enumerate(n.get('input', [])):
                    if arg and not (n['name'] == self.conv['name'] and i == 0):
                        visit(arg)
            visit(self.terminal)
        shape = PACKED_SHAPE if variant == 'packed' else INPUT_SHAPE
        attr_field = _field(schema, 'OpDef', 'attr', 11, True)
        index = blob('OpDef.AttrEntry', 'key', 'index')+blob('OpDef.AttrEntry', 'value',
            _v(_field(schema, 'AttrDef', 'i', 3, False) << 3)+_v(0), 11)
        input_field = _field(schema, 'OpDef', 'input', 9, True)
        input_desc = _field(schema, 'OpDef', 'input_desc', 11, True)
        output_desc = _field(schema, 'OpDef', 'output_desc', 11, True)
        encoded, preserved = [], {}
        for n, payload, _ in self.rows:
            if n['name'] not in selected:
                continue
            if n['name'] == self.input:
                # Preserve the already compiled baseline byte-for-byte. The
                # candidate must also replace Data's external input descriptor;
                # its original shape otherwise wins during OMG Data inference.
                remove = {output_desc, input_desc} if variant == 'packed' else {output_desc}
                payload = b''.join(e for f, _, e, p in _parts(payload)
                    if f not in remove and (f != attr_field or schema.decode(p, 'OpDef.AttrEntry')['key'] != 'index'))
                if variant == 'packed':
                    payload += _blob(input_desc, blob('TensorDescriptor', 'name', 'x')+desc(shape))
                payload += _blob(attr_field, index)+_blob(output_desc, desc(shape))
            elif variant == 'packed' and n['name'] == self.conv['name']:
                count, parts = 0, []
                for f, _, e, _ in _parts(payload):
                    if f == input_field:
                        if count == 0: e = _blob(f, self.input+':0')
                        count += 1
                    parts.append(e)
                payload = b''.join(parts)
            else:
                preserved[n['name']] = hashlib.sha256(payload).hexdigest()
            encoded.append(_blob(self.op_field, payload))
        net = blob('OpDef', 'name', 'NetOutput')+blob('OpDef', 'type', 'NetOutput')
        net += blob('OpDef', 'input', self.terminal, repeated=True)+blob('OpDef', 'input_desc', desc(OUTPUT_SHAPE), 11, True)
        net = _new_id(net, schema, max(n.get('id', 0) for n in self.nodes.values())+1)
        graph = blob('GraphDef', 'name', 'draft_feature_pack_'+variant)+b''.join(encoded)+_blob(self.op_field, net)
        raw = graph if self.graph_field is None else b''.join(
            _blob(f, graph) if f == self.graph_field else e for f, _, e, _ in _parts(self.raw))
        # The compiled baseline kept the original external descriptor, whose
        # placement follows the exporter input (CPU in the retained Draft).
        source_inputs = self.nodes[self.input].get('input_desc', [])
        input_device = (source_inputs[0].get('device_type') if len(source_inputs) == 1 else None)
        contract = input_contract(raw, schema, self.input, shape,
            input_device=input_device if variant == 'baseline' else 'NPU')
        return raw, dict(graph=dict(inputs=[dict(name=self.input, dtype='float16', shape=shape)],
            outputs=[dict(name='projected_features', source_edge=self.terminal, dtype='float16', shape=OUTPUT_SHAPE)]),
            removed_nodes=sorted(self.selected-selected), preserved_payload_sha256=preserved,
            changes='DATA_INDEX_AND_NETOUTPUT' if variant == 'baseline' else 'DATA_INPUT_AND_OUTPUT_SHAPES_AND_CONV_INPUT_0; PRUNE_DEAD_LAYOUT',
            data_descriptor_contract=contract,
            input_mapping='ALL_262144_ELEMENT_INDICES_EQUAL_SOURCE_VIEWS', conv_attributes='UNCHANGED',
            scope=SCOPE, full_graph_compilation_preserved=False)


def _retry_failed(root, state, plans, inspect_model, ddk_root):
    """Under the writer lock, retain completed OMCs and reset known failed OMG jobs."""
    from qwen35_moe_dflash.toolchain.compiler_storage import require_idle, _plain, _tree_bytes
    _plain(root, directory=True); _plain(root/'result.json')
    old = read_json(root/'result.json')
    if (old.get('artifact_kind') != KIND or old.get('status') != 'PARTIAL' or old.get('stage') != 'complete'
            or old.get('device_execution') != 'NOT_RUN' or old.get('numerical_validation') != 'NOT_RUN'):
        raise ValueError('--retry-failed requires a completed host-only partial compilation')
    for key in ('original_model', 'source_air_sha256', 'source_weight', 'ddk_id', 'comparison', 'source_cases', 'cases'):
        if old.get(key) != state[key]:
            raise ValueError('retry source, inputs, reference or DDK changed: '+key)
    rows = old.get('models', [])
    if not 1 <= len(rows) <= 2 or [r.get('variant') for r in rows] != list(VARIANTS[:len(rows)]):
        raise ValueError('invalid feature-prefix retry inventory')
    allowed = {'result.json', 'fixtures', 'previous-failure.json', *(r['variant'] for r in rows)}
    if set(p.name for p in root.iterdir()) - allowed or (root/'previous-failure.json').is_symlink():
        raise ValueError('foreign files in feature-prefix output; retained')
    require_idle(root)
    for row in state['cases']:
        for record in row['fixtures'].values():
            checked_file(root, record)
    by_variant = {p[0]: p for p in plans}
    retained, failed = [], []
    for row in rows:
        variant = row['variant']; folder = root/variant
        _plain(folder, directory=True); _plain(folder/'air', directory=True); _plain(folder/'air/export.air')
        if row.get('status') == 'COMPILED' and row.get('model_compilation') == 'PASSED':
            if row.get('model', {}).get('path') != variant+'/model.omc':
                raise ValueError('unexpected retained model path')
            model = checked_file(root, row['model'])
            if (folder/'air/export.air').read_bytes() != by_variant[variant][1]:
                raise ValueError('successful prefix source changed; refusing to reuse its OMC')
            inspection = inspect_model(model, ddk_root, by_variant[variant][2]['graph'])
            retained.append(dict(row, placement=inspection['placement'], reuse='EXISTING_COMPILED_MODEL_UNCHANGED',
                compiled_probe_source_sha256=row.get('compiled_probe_source_sha256', old['probe_source_sha256'])))
            continue
        if (row.get('status') != 'FAILED' or row.get('model_compilation') != 'FAILED'
                or set(p.name for p in folder.iterdir()) != {'air', 'compilation'}):
            raise ValueError('only known failed OMG cases can be reset; all files retained')
        _plain(folder/'compilation', directory=True); _plain(folder/'compilation/report.json')
        build = read_json(folder/'compilation/report.json', 32 << 20)
        if (build.get('artifact_kind') != 'qwen35-existing-air-compilation' or build.get('status') != 'FAILED'
                or build.get('model_compilation') != 'FAILED' or build.get('device_execution') != 'NOT_RUN'
                or build.get('stage') != 'omg' or type(build.get('returncode')) is not int or build['returncode'] == 0
                or build.get('ddk_id') != state['ddk_id'] or build.get('platform') != 'kirinx90'
                or build.get('air_source') != str(folder/'air/export.air')):
            raise ValueError('retry requires a recorded failed OMG exit for this candidate and DDK')
        _tree_bytes(folder)  # Reject mounts/special files; rmtree never follows scratch symlinks.
        failed.append(dict(variant=variant, previous_model=row, compilation_report=build,
            previous_air_sha256=sha256_file(folder/'air/export.air'),
            replacement_air_sha256=hashlib.sha256(by_variant[variant][1]).hexdigest()))
    if not failed:
        raise ValueError('no failed OMG case to retry')
    require_idle(root)
    _save(root/'previous-failure.json', dict(artifact_kind=KIND, scope='LAST_KNOWN_FAILED_OMG_ATTEMPT',
        probe_source_sha256=old['probe_source_sha256'], models=failed))
    for failure in failed:
        shutil.rmtree(root/failure['variant'])
    state.update(models=retained, retry=dict(reused_variants=[r['variant'] for r in retained],
        reset_failed_variants=[r['variant'] for r in failed], previous_failure='previous-failure.json'))


def compile_probe(air_export, result, output, *, ddk_root, ddk_id, tbe_diagnosis,
                  timeout_seconds=1800, progress=print, retry_failed=False):
    from .accuracy_board import replay_fixture
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
    from qwen35_moe_dflash.toolchain.compiler_storage import release_packed_scratch
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    import stable_arg_max_ge
    root, air = output_path(output), Path(air_export).resolve()
    _disjoint(root, air, Path(result).resolve(), ddk_root, tbe_diagnosis)
    if root != Path(output).absolute():
        raise ValueError('feature-prefix output must not contain symlinks')
    if root.exists() != retry_failed:
        raise ValueError('use a new output, or --retry-failed on the existing completed failed build')
    fixture, meta = replay_fixture(result)
    binding, manifest = read_json(fixture/'pytorch-abi.json'), read_json(air/'manifest.json')
    raw, _ = _read_air(air/'export.air')
    if (meta.get('frontend') != 'pytorch' or binding['source_air_sha256'] != hashlib.sha256(raw).hexdigest()
            or binding['packed_model_sha256'] != meta['model']['sha256']
            or [meta[k] for k in ('context_rows', 'cache_capacity', 'max_draft_tokens')] != [16, 192, 15]):
        raise ValueError('retained source/ABI/C16 replay binding differs')
    schema = _schema((air/'air-schema.pb').read_bytes(), manifest['schema'])
    prefix = Projection(raw, schema); prefix.select_projection(binding['source_input_roles']['target_features'])
    weights = [r for r in binding['reference_weights'] if r['name'] == 'draft.fc.weight']
    if len(weights) != 1:
        raise ValueError('one bound source feature weight required')
    weight_record = dict(path=WEIGHT, bytes=2048*16384*2, sha256=weights[0]['sha256'])
    weight_path = checked_file(air, weight_record)
    weight = np.fromfile(weight_path, dtype='<f2').reshape(prefix.weight_shape)
    if not np.isfinite(weight).all():
        raise ValueError('nonfinite feature weights')
    spec = next(s for s in meta['inputs'] if s['name'] == 'target_features')
    if (spec['dtype'], spec['shape'], spec['bytes']) != ('float16', INPUT_SHAPE, 524288):
        raise ValueError('retained feature fixture geometry differs')
    fixtures, sources = {}, []
    for step, valid in zip(STEPS, (2, 16, 14)):
        case = next(c for c in meta['cases'] if c['index'] == step)
        if case['valid'] != valid:
            raise ValueError('retained partial/full context coverage differs')
        records = [r for r in meta['input_files'] if r['case'] == step and r['index'] == spec['index']]
        if len(records) != 1:
            raise ValueError('missing/duplicate feature input')
        x = np.fromfile(checked_file(fixture, records[0]), dtype='<f2').reshape(INPUT_SHAPE)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite retained feature input')
        progress(f'evaluating retained projection for case {step}, valid={valid}')
        values = prefix.evaluate_layer({prefix.input: x}, {WEIGHT: weight}, targets=[prefix.terminal])
        y = values[prefix.terminal]
        if list(y.shape) != OUTPUT_SHAPE or y.dtype != np.float16 or not np.isfinite(y).all():
            raise ValueError('source feature projection output contract differs')
        fixtures[step] = (x, y.copy())
        sources.append(dict(case=case, input=records[0]))
    plans = []
    for variant in VARIANTS:
        extracted, receipt = prefix.extract_projection(variant)
        prepared, preparation = prepare_compiler_air(extracted, schema)
        receipt['compiler_data_descriptor_contract'] = input_contract(prepared, schema, prefix.input,
            PACKED_SHAPE if variant == 'packed' else INPUT_SHAPE,
            input_device=receipt['data_descriptor_contract']['input_desc']['device_type'])
        plans.append((variant, extracted, receipt, prepared, preparation))
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 1 << 30:
        raise OSError('two feature prefixes need at least 1 GiB free; compiler scratch may need more')
    with (root.parent/('.'+root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(artifact_kind=KIND, scope=SCOPE, status='PARTIAL', stage='compile',
            model_compilation='PARTIAL', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
            original_model=meta['model'], source_air_sha256=binding['source_air_sha256'],
            source_result=str(Path(result).resolve()), source_cases=sources, source_weight=weight_record,
            ddk_id=ddk_id, comparison=meta['comparison'], output_initialization='untouched',
            reference='SOURCE_CONV_FULL_K_FP32_DOT_THEN_ORIGINAL_FP16; CPU_REDUCTION_ORDER_MAY_DIFFER',
            probe_source_sha256=sha256_file(Path(__file__)), cases=[], models=[])
        for step, (x, y) in fixtures.items():
            state['cases'].append(dict(source_case=step, fixtures={name: dict(
                path=f'fixtures/{step}/{name}', bytes=a.nbytes, sha256=hashlib.sha256(a.tobytes()).hexdigest())
                for name, a in (('input_0.bin', x), ('golden_0.bin', y))}))
        if retry_failed:
            _retry_failed(root, state, plans, stable_arg_max_ge.inspect_model, ddk_root)
        else:
            root.mkdir()
            for step, (x, y) in fixtures.items():
                folder = root/'fixtures'/str(step); folder.mkdir(parents=True)
                x.tofile(folder/'input_0.bin'); y.tofile(folder/'golden_0.bin')
        _save(root/'result.json', state)
        for variant, extracted, receipt, prepared, preparation in plans:
            if any(r['variant'] == variant for r in state['models']):
                progress('reusing unchanged feature prefix '+variant)
                continue
            folder = root/variant; (folder/'air/weight').mkdir(parents=True)
            (folder/'air/export.air').write_bytes(extracted)
            shutil.copyfile(weight_path, folder/'air'/WEIGHT)
            checked_file(folder/'air', weight_record)
            row = dict(variant=variant, status='FAILED', model_compilation='NOT_RUN', extraction=receipt)
            state['models'].append(row); _save(root/'result.json', state)
            build = None
            try:
                def transform(snapshot):
                    if snapshot.read_bytes() != extracted: raise ValueError('prefix AIR snapshot changed')
                    return prepared, preparation
                progress('compiling feature prefix '+variant)
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
                if packed['status'] != 'EMBEDDED' or (folder/'model.omc').stat().st_size > 96 << 20:
                    raise ValueError('feature prefix packing failed or exceeded 96 MiB')
                inspection = stable_arg_max_ge.inspect_model(folder/'model.omc', ddk_root, receipt['graph'])
                row.update(status='COMPILED', placement=inspection['placement'], model=dict(path=variant+'/model.omc',
                    bytes=(folder/'model.omc').stat().st_size, sha256=sha256_file(folder/'model.omc')))
            except Exception as error:
                row.update(error=f'{type(error).__name__}: {error}',
                    compilation_diagnostics=compilation_diagnostics(folder/'compilation'))
            if row['status'] == 'COMPILED':
                try:
                    row['storage'] = release_packed_scratch(folder, apply=True)
                    (folder/'air'/WEIGHT).unlink()
                except (OSError, ValueError, KeyError) as error:
                    row['storage_error'] = str(error)
            _save(root/'result.json', state)
            if row['status'] != 'COMPILED':
                break  # Retain failure; no second expensive compile after a failed/uncertain first.
        passed = len(state['models']) == 2 and all(r['status'] == 'COMPILED' for r in state['models'])
        state.update(status='COMPILED' if passed else 'PARTIAL', stage='complete',
                     model_compilation='PASSED' if passed else 'PARTIAL')
        _save(root/'result.json', state)
    return state


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    from validate_draft_vector import selected_cases, WARMUP, REPETITIONS
    import stable_arg_max_ge
    root, out = Path(root), Path(out)
    report = read_json(root/'result.json')
    if report.get('status') != 'COMPILED' or report.get('ddk_id') != ddk_id:
        raise ValueError('both feature prefixes must compile for the selected DDK')
    if selected_cases(CASES if selection is None else selection, CASES) != list(CASES):
        raise ValueError('run all three paired feature inputs together')
    models = {r['variant']: r for r in report['models']}
    sources = {r['source_case']: r for r in report['cases']}
    if len(report['models']) != 2 or set(models) != set(VARIANTS) or len(report['cases']) != 3 or set(sources) != set(STEPS):
        raise ValueError('feature-prefix compile inventory differs')
    inspected = {}
    for variant, row in models.items():
        if row['status'] != 'COMPILED': raise ValueError('feature prefix is not compiled')
        model = checked_file(root, row['model'])
        inspected[variant] = stable_arg_max_ge.inspect_model(model, ddk, row['extraction']['graph'])
    cases = []
    for step in STEPS:
        for variant in VARIANTS:
            name = f'feature_pack_c{step:02d}_{variant}'
            row, inspection = models[variant], inspected[variant]
            bound = deepcopy(inspection['io']); folder = out/'fixture'/name; folder.mkdir(parents=True)
            plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}',
                    'input_preparation '+PREPARATION[variant]]
            for direction, stem in (('inputs', 'input'), ('outputs', 'golden')):
                for spec in bound[direction]:
                    path = checked_file(root, sources[step]['fixtures'][f'{stem}_{spec["fixture_index"]}.bin'])
                    if path.stat().st_size != spec['bytes']: raise ValueError('feature fixture/ABI extent differs')
                    dest = folder/f'{stem}_{spec["index"]}.bin'; shutil.copyfile(path, dest)
                    spec['fixture_sha256'] = sha256_file(dest)
                    plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'], ','.join(map(str, spec['shape'])))))
            (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
            cases.append(dict(case=name, source_case=step, variant=variant, input_preparation=PREPARATION[variant],
                input_file_layout='ORIGINAL_ROW_MAJOR_C16', **bound,
                model=_file_record(checked_file(root, row['model']), variant+'/model.omc'),
                graphop_counts=inspection['placement']['graphop_counts'], placement=inspection['placement']))
    meta = dict(cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=report['comparison']['atol'],
        rtol=report['comparison']['rtol'], scope=SCOPE, ddk_id=ddk_id, source_evidence=report,
        output_initialization='untouched')
    _save(out/'fixture/manifest.json', meta)
    return meta


def summarize(meta, native, comparison):
    """Keep CPU reference failures; require finite stable bitwise device equivalence."""
    selected = set(comparison['validated_cases']); rows, pairs = {}, []
    for case in meta['cases']:
        name = case['case']
        if name not in selected: continue
        result = read_json(native/name/'result.json')
        if result.get('input_preparation') != case['input_preparation'] or result.get('prepared_timing_scope') != TIMING:
            raise ValueError('native probe did not time the declared CPU input preparation')
        for field in ('packing_us', 'set_input_us', 'prepared_call_us'):
            values = result.get(field, [])
            if len(values) != 10 or any(type(t) is not int or t < 0 for t in values):
                raise ValueError('invalid preparation timing: '+field)
        for i in range(10):
            if (result['prepared_call_us'][i] < result['api_us'][i] or
                    result['prepared_call_us'][i] < result['packing_us'][i]+result['set_input_us'][i]):
                raise ValueError('inconsistent preparation/API wall timing')
        def stats(values):
            a = np.asarray(values, dtype=np.float64)/1000
            return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)), min_ms=float(a.min()), max_ms=float(a.max()))
        row = next(r for r in comparison['cases'] if r['case'] == name)
        row['input_preparation'] = case['input_preparation']
        row['preparation_timing'] = {f: stats(result[f]) for f in ('packing_us', 'set_input_us', 'api_us', 'prepared_call_us')}
        rows[name] = row
    for step in STEPS:
        names = [f'feature_pack_c{step:02d}_{v}' for v in VARIANTS]
        if not set(names) <= selected: continue
        b, c = [rows[n] for n in names]
        equal = [(native/names[0]/f'{i}.bin').read_bytes() == (native/names[1]/f'{i}.bin').read_bytes() for i in range(10)]
        ok = all(equal) and all(r['repeat_bitwise_stable'] and r['numerics']['finite'] for r in (b, c))
        bt, ct = [r['preparation_timing']['prepared_call_us'] for r in (b, c)]
        pairs.append(dict(source_case=step, baseline_equivalence='PASSED' if ok else 'FAILED', calls_bytes_equal=equal,
            baseline=bt, candidate=ct, mean_saved_ms=bt['mean_ms']-ct['mean_ms'],
            median_saved_ms=bt['median_ms']-ct['median_ms'],
            observed_ranges_separated=ct['max_ms'] < bt['min_ms']))
    complete = selected == set(CASES) and len(pairs) == len(STEPS)
    equivalence = 'PASSED' if complete and all(p['baseline_equivalence'] == 'PASSED' for p in pairs) else ('FAILED' if complete else 'INCOMPLETE')
    return dict(coverage_complete=complete, baseline_equivalence=equivalence, pairs=pairs,
        timing_scope=TIMING, full_model_integration='NOT_RUN', production_promotion='NOT_PERFORMED',
        decision='REVIEW_PREFIX_TIMINGS_BEFORE_FULL_MODEL_BUILD',
        limitations=['CPU reference gates remain unchanged; equivalence is a separate measured-device comparison.',
                    'Separate model loads and snapshots between calls; no full-model or end-to-end speedup claim.',
                    'Candidate uses a preallocated CPU scratch plus SetInput copy; preparation is included.'])

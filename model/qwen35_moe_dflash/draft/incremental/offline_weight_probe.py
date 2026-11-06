"""Paired layout/transpose import matrix, built from the installed GE schema."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.run_layout import _save
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _new_id
from . import offline_weights as ow
from .torch_export import _producer_schema, _schema, _slot, _disjoint

KIND = 'qwen35-draft-offline-weight-import'
SCENARIOS = {
    # W[N,K], activation rows. Large axes are covered separately to avoid
    # replicating the 970 MiB head in each of six layout experiments.
    'mapping': ([48, 80], 95),
    'reverse': ([80, 48], 1),
    'k2048': ([48, 2048], 15),
    'k4096': ([48, 4096], 16),
    'k6144': ([48, 6144], 32),
    'q_width': ([4096, 32], 15),
    'kv_o_width': ([2048, 48], 16),
    'gate_up_width': ([12288, 32], 32),
    'head_width': ([248320, 16], 15),
}
CASES = tuple(f'offline_weight_{s}_{v}' for s in SCENARIOS for v in ow.VARIANTS)
SCOPE = 'FP16_RHS_LAYOUT_TRANSPOSE_MATRIX_AXES_COVERAGE; NOT_FULL_SHAPE_OR_SPEEDUP'
SUITE = dict(policy='static-rhs-import-matrix-v1', scenarios=SCENARIOS,
             variants={v: ow.contract(v) for v in ow.VARIANTS})


def fixture(scenario='mapping'):
    shape, rows = SCENARIOS[scenario]
    rng = np.random.default_rng(35008048)
    w = rng.integers(-2, 3, size=shape).astype('<f2')
    # Mapping case exposes every weight individually. Other cases exercise
    # actual K/N extents separately, including signed mixed accumulation.
    if scenario == 'mapping':
        x = np.concatenate((np.eye(shape[1], dtype='<f2'),
                            rng.integers(-1, 2, size=(15, shape[1])).astype('<f2')))
    else:
        x = rng.integers(-1, 2, size=(rows, shape[1])).astype('<f2')
        # Interior and boundary impulses, plus dense signed/mixed dot rows.
        for i, k in enumerate((0, 15, min(16, shape[1]-1), shape[1]//2-1, shape[1]//2, shape[1]-1)):
            if i >= rows: break
            x[i] = 0; x[i, k] = 1 if i % 2 == 0 else -1
        if rows > 6: x[6] = 1
        if rows > 7: x[7] = np.where(np.arange(shape[1]) % 2, -1, 1)
        if rows > 8: x[8] = 0
    y = (x.astype(np.float32) @ w.astype(np.float32).T).astype('<f2')
    if np.max(np.abs(y)) >= 1024:
        raise ValueError('integer import reference must remain exactly representable in FP16')
    return w, x, y


def graph(schema, scenario='mapping'):
    shape, rows = SCENARIOS[scenario]
    def blob(msg, name, value, kind=9, repeated=False):
        return _blob(_field(schema, msg, name, kind, repeated), value)
    def node(name, kind, inputs, ins, outs, attrs):
        return (blob('OpDef', 'name', name) + blob('OpDef', 'type', kind)
                + b''.join(blob('OpDef', 'input', e, repeated=True) for e in inputs)
                + b''.join(blob('OpDef', 'input_desc', d, 11, True) for d in ins)
                + b''.join(blob('OpDef', 'output_desc', d, 11, True) for d in outs)
                + b''.join(ow._attr(schema, 'OpDef', k, v) for k, v in attrs.items()))
    xshape, yshape = [rows, shape[1]], [rows, shape[0]]
    dx, dw, dy = [ow.descriptor(schema, list(a)) for a in (xshape, shape, yshape)]
    optional = (blob('TensorDescriptor', 'layout', 'FORMAT_RESERVED')
                + blob('TensorDescriptor', 'device_type', 'NPU')
                + ow._attr(schema, 'TensorDescriptor', '_is_unfed_optional', 1)
                + ow._attr(schema, 'TensorDescriptor', 'origin_format', 'ND')
                + ow._attr(schema, 'TensorDescriptor', 'origin_format_for_int', 2)
                + ow._attr(schema, 'TensorDescriptor', 'format_for_int', 40))
    nodes = [node('x', 'Data', [''], [dx], [dx], dict(index=0)),
             node('w', 'FileConstant', [], [], [dw],
                  dict(file_path='weight/canary', dtype=1, shape=shape, file_id='')),
             node('dot', 'MatMul', ['x:0', 'w:0', ''], [dx, dw, optional], [dy],
                  dict(transpose_x1=False, transpose_x2=True)),
             node('output', 'NetOutput', ['dot:0'], [dy], [], {})]
    graph_raw = (blob('GraphDef', 'name', 'offline_weight_import')
                 + b''.join(blob('GraphDef', 'op', _new_id(n, schema, i), 11, True)
                            for i, n in enumerate(nodes)))
    # OMG framework=1 loads ModelDef. A bare GraphDef can pass our general AIR
    # reader but is seen by the compiler as a model with zero graphs.
    raw = (blob('ModelDef', 'name', 'offline_weight_import')
           + blob('ModelDef', 'graph', graph_raw, 11, True))
    _check_model_container(raw, schema)
    declared = dict(inputs=[dict(name='x', shape=xshape, dtype='float16')],
                    outputs=[dict(name='y', shape=yshape, dtype='float16')])
    return raw, declared


def _check_model_container(raw, schema):
    """Check the compiler's container, without GraphDef autodetection."""
    graphs = schema.decode(raw, 'ModelDef').get('graph', [])
    if (len(graphs) != 1 or [(n.get('name'), n.get('type')) for n in graphs[0].get('op', [])]
            != [('x', 'Data'), ('w', 'FileConstant'), ('dot', 'MatMul'), ('output', 'NetOutput')]):
        raise ValueError('offline weight AIR must be a ModelDef containing the four-node probe graph')


def compile_probe(output, *, ddk_root, ddk_id, torchair_pb2=None, tbe_diagnosis=None,
                  timeout_seconds=1800, replace=False, progress=print):
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    import stable_arg_max_ge
    if not ddk_root or not ddk_id:
        raise ValueError('selected CPro DDK root and id are required')
    _disjoint(output, ddk_root)
    for relative in ('tools/tools_omg/master/omg', 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so'):
        if not (Path(ddk_root)/relative).is_file():
            raise FileNotFoundError(Path(ddk_root)/relative)
    descriptor, metadata = _producer_schema(torchair_pb2)
    schema = _schema(descriptor, metadata)
    ancestor = Path(output).resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    weights = sum(math.prod(s)*2 for s, _ in SCENARIOS.values())*len(ow.VARIANTS)
    outputs = sum(s[0]*m*2 for s, m in SCENARIOS.values())*len(ow.VARIANTS)
    required = 3*weights + 20*outputs + (48 << 20)*len(CASES) + (256 << 20)
    available = shutil.disk_usage(ancestor).free
    if available < required:
        raise OSError(f'layout suite needs at least {required} free bytes; available={available}; '
                      'includes later output archives/snapshots; compiler scratch may require more')
    with _slot(output, KIND, replace) as root:
        report = dict(artifact_kind=KIND, status='COMPILING', scope=SCOPE, ddk_id=ddk_id,
                      suite=SUITE, schema=metadata, air_container='ModelDef', cases=[],
                      model_compilation='NOT_RUN', compilation_calls=0,
                      storage=dict(required_free_bytes=required, available_free_bytes=available,
                                   learned_weight_bytes=0, synthetic_weight_bytes=weights,
                                   ten_repeat_output_bytes=10*outputs, scope='LOWER_BOUND_INCLUDING_RECOVERY'),
                      device_execution='NOT_RUN', numerical_validation='NOT_RUN')
        # _slot owns manifest.json; result.json is the existing vector-runner ABI.
        _save(root/'manifest.json', report)
        (root/'air-schema.pb').write_bytes(descriptor)
        blocked_by = None
        for case in CASES:
            scenario = next(s for s in SCENARIOS if case.startswith('offline_weight_'+s+'_'))
            variant = case.rsplit('_', 2)[-2]+'_'+case.rsplit('_', 1)[-1]
            row = dict(case=case, scenario=scenario, variant=variant, contract=ow.contract(variant),
                       status='COMPILING', model_compilation='NOT_RUN')
            report['cases'].append(row)
            if blocked_by:
                row.update(status='SKIPPED', blocked_by=blocked_by,
                           reason='ordinary ND baseline failed; layout support is undetermined')
                continue
            _save(root/'result.json', report)
            try:
                raw, declared = graph(schema, scenario)
                row['graph'] = declared
                plan = ow.inventory(raw, schema, draft=False, variant=variant)
                w, _, _ = fixture(scenario)
                folder = root/case
                (folder/'air/weight').mkdir(parents=True)
                w.tofile(folder/'air/weight/canary')
                (folder/'air/probe.air').write_bytes(raw)
                if shutil.disk_usage(root).free < 256 << 20:
                    raise OSError('less than 256 MiB free; compiler invocation skipped')
                def transform(snapshot):
                    if snapshot.read_bytes() != raw:
                        raise ValueError('canary snapshot changed')
                    (snapshot.parent/'before-offline-weights.air').write_bytes(raw)
                    proof = ow.pack_files(snapshot.parent, plan, progress=progress)
                    proof.update(before_air='before-offline-weights.air', scope=SCOPE)
                    prepared = ow.rewrite(raw, schema, plan)
                    _check_model_container(prepared, schema)
                    return prepared, dict(policy=ow.POLICY, status='APPLIED',
                        removed_count=0, offline_weights=proof)
                report['compilation_calls'] += 1
                result = compile_existing_air(folder/'air/probe.air', folder/'omc', ddk_root,
                    ddk_id=ddk_id, timeout_seconds=timeout_seconds, input_layout='air-directory',
                    use_origin_format=True, save_weights_as_external_data=False,
                    tbe_diagnosis=tbe_diagnosis, air_transform=transform if variant != 'nd_t' else None,
                    progress=lambda r: progress(case+': '+r['message']))
                row['model_compilation'] = result['model_compilation']
                if result['status'] != 'COMPILED':
                    row['compilation_diagnostics'] = compilation_diagnostics(folder/'omc')
                    raise RuntimeError(str(result.get('error', result['status'])))
                model = checked_file(folder/'omc', result['omc'])
                inspected = stable_arg_max_ge.inspect_model(model, ddk_root, declared)
                if not inspected['placement']['npu_only_candidate']:
                    raise ValueError('import canary is not entirely in NPU tasks at decoded graph level')
                row.update(status='COMPILED', placement=inspected['placement'],
                    model=dict(path=str(model.relative_to(root)), bytes=model.stat().st_size, sha256=sha256_file(model)))
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                row.update(status='FAILED', error=f'{type(error).__name__}: {error}')
                if variant == 'nd_t':
                    blocked_by = case
                    report['baseline_gate'] = dict(status='FAILED', case=case, error=row['error'])
                    progress(case+': ND baseline failed; skipping the remaining cases')
            _save(root/'result.json', report)
        success = all(r['status'] == 'COMPILED' for r in report['cases'])
        baseline = all(r['status'] == 'COMPILED' for r in report['cases'] if r['variant'] == 'nd_t')
        report.update(status='COMPILED' if success else 'PARTIAL' if baseline else 'FAILED', stage='complete',
                      model_compilation='PASSED' if success else 'PARTIAL' if baseline else 'FAILED',
                      compilation_matrix={v: {r['scenario']: r['status'] for r in report['cases'] if r['variant'] == v}
                                          for v in ow.VARIANTS},
                      skipped_cases=sum(r['status'] == 'SKIPPED' for r in report['cases']),
                      next=('Validate all compiled cases together; failed imports remain explicit in the matrix.'
                            if baseline else 'Fix the ordinary ND baseline failure and retry with --replace-failed; '
                            'do not run device validation or infer layout support from this suite.'))
        if baseline:
            report['baseline_gate'] = dict(status='PASSED', scenarios=len(SCENARIOS))
        _save(root/'result.json', report)
        _save(root/'manifest.json', report)
        return report


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from validate_draft_vector import selected_cases, WARMUP, REPETITIONS
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    import stable_arg_max_ge
    root, out = Path(root), Path(out)
    report = json.loads((root/'result.json').read_text())
    compiled_cases = [r['case'] for r in report['cases'] if r['status'] == 'COMPILED']
    if (report['status'] not in ('COMPILED', 'PARTIAL') or report['ddk_id'] != ddk_id
            or report['suite'] != json.loads(json.dumps(SUITE))
            or [r['case'] for r in report['cases']] != list(CASES)
            or selected_cases(compiled_cases if selection is None else selection, CASES) != compiled_cases):
        raise ValueError('run all compiled layout matrix cases together with the selected DDK')
    if any(r['status'] != 'COMPILED' for r in report['cases'] if r['variant'] == 'nd_t'):
        raise ValueError('all ordinary ND baseline scenarios must compile before device comparison')
    descriptor = (root/'air-schema.pb').read_bytes()
    if hashlib.sha256(descriptor).hexdigest() != report['schema']['descriptor_sha256']:
        raise ValueError('canary source schema changed')
    schema = _schema(descriptor, report['schema'])
    cases = []
    for row in report['cases']:
        if row['status'] != 'COMPILED': continue
        raw, declared = graph(schema, row['scenario'])
        w, x, y = fixture(row['scenario'])
        name = row['case']
        model = checked_file(root, row['model'])
        receipt = json.loads((root/name/'omc/report.json').read_text())
        if receipt['ddk_id'] != ddk_id or receipt['platform'] != 'kirinx90':
            raise ValueError('canary compiler receipt differs')
        snap = {r['source_relative']: r for r in receipt['input_snapshots']}
        original = checked_file(root/name/'omc', snap['probe.air'])
        weight = checked_file(root/name/'omc', snap['weight/canary'])
        if (original.read_bytes() != raw or weight.read_bytes() != w.tobytes() or row['graph'] != declared
                or row['model']['sha256'] != receipt['omc']['sha256']):
            raise ValueError('canary source recipe changed; recompile')
        if row['variant'] != 'nd_t':
            evidence = receipt['air_preparation']
            if evidence['offline_weights']['contract'] != ow.contract(row['variant']):
                raise ValueError('canary variant differs from the compiler layout contract')
            packed_air = checked_file(root/name/'omc', evidence['compiler_air'])
            ow.verify_binding(packed_air.parent, packed_air.read_bytes(), schema, evidence['offline_weights'])
        inspected = stable_arg_max_ge.inspect_model(model, ddk, declared)
        if not inspected['placement']['npu_only_candidate']:
            raise ValueError('canary requires NPU-only decoded placement')
        io = deepcopy(inspected['io'])
        folder = out/'fixture'/name
        folder.mkdir(parents=True)
        plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}']
        for direction, stem, array in (('inputs', 'input', x), ('outputs', 'golden', y)):
            for spec in io[direction]:
                spec['exact_bytes'] = True
                path = folder/f'{stem}_{spec["index"]}.bin'
                array.tofile(path)
                spec['fixture_sha256'] = sha256_file(path)
                plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'],
                                      ','.join(map(str, spec['shape'])))))
        (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
        cases.append(dict(case=name, scenario=row['scenario'], variant=row['variant'], **io,
                          model=_file_record(model, name+'/model.omc'),
                          graphop_counts=inspected['placement']['graphop_counts']))
    meta = dict(cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=0., rtol=0.,
                exact_bytes=True, scope=SCOPE, ddk_id=ddk_id, source_evidence=report,
                output_initialization='untouched')
    _save(out/'fixture/manifest.json', meta)
    return meta


def summarize(meta, comparison):
    """Classify every attempted setting; never turn unsupported rows into a pass."""
    results = {r['case']: r for r in comparison['cases']}
    matrix, passing = {}, []
    for variant in ow.VARIANTS:
        rows = {}
        for attempt in meta['source_evidence']['cases']:
            if attempt['variant'] != variant: continue
            observed = results.get(attempt['case'])
            rows[attempt['scenario']] = dict(
                compilation=attempt['model_compilation'], import_ready=attempt['status'] == 'COMPILED',
                numerical='NOT_RUN' if observed is None else 'PASSED' if observed['passed'] else 'FAILED',
                repeat_stable=None if observed is None else observed['repeat_bitwise_stable'],
                mean_api_ms=None if observed is None else observed['mean_ms'],
                error=attempt.get('error'))
        if (set(rows) == set(SCENARIOS) and all(r['compilation'] == 'PASSED' and r['import_ready'] and r['numerical'] == 'PASSED'
                                              and r['repeat_stable'] for r in rows.values())):
            passing.append(variant)
        matrix[variant] = rows
    return dict(suite=SUITE, passing_variants=passing, matrix=matrix,
                baseline_passed='nd_t' in passing, full_shape_validation='NOT_RUN',
                decision='CHOOSE_A_PASSING_VARIANT_FOR_FULL_DRAFT; NO_MICROBENCHMARK_SPEEDUP_CLAIM')


def check_device_canary(directory, ddk_id, variant='nz_t'):
    """Recompare retained device bytes; never accept just an edited status flag."""
    from validate_draft_vector import analyze
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    execution = Path(result['recovery_source'])
    fixture_meta = json.loads((execution/'fixture/manifest.json').read_text())
    source = fixture_meta['source_evidence']
    if (source.get('artifact_kind') != KIND or source.get('suite') != json.loads(json.dumps(SUITE))
            or fixture_meta.get('ddk_id') != ddk_id or fixture_meta.get('exact_bytes') is not True
            or fixture_meta.get('scope') != SCOPE or fixture_meta.get('atol') != 0 or fixture_meta.get('rtol') != 0
            or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'):
        raise ValueError('a completed device import canary for this DDK is required')
    expected_cases = [r['case'] for r in source['cases'] if r['status'] == 'COMPILED']
    if [c['case'] for c in fixture_meta['cases']] != expected_cases:
        raise ValueError('all compiled canary cases must be retained')
    for case in fixture_meta['cases']:
        _, x, y = fixture(case['scenario'])
        for direction, stem, array in (('inputs', 'input', x), ('outputs', 'golden', y)):
            for spec in case[direction]:
                path = execution/'fixture'/case['case']/f'{stem}_{spec["index"]}.bin'
                if path.read_bytes() != array.tobytes():
                    raise ValueError('canary fixtures no longer match the current recipe')
    comparison = analyze(directory)
    summary = comparison['offline_weight_layouts']
    if (comparison['validated_cases'] != expected_cases or not summary['baseline_passed']
            or variant not in summary['passing_variants']):
        raise ValueError('selected layout and ND baseline must pass every scenario and all ten repetitions')
    return dict(status='DEVICE_IMPORT_CANARY_PASSED', ddk_id=ddk_id, contract=ow.contract(variant),
                validation_sha256=sha256_file(directory/'result.json'),
                fixture_sha256=sha256_file(execution/'fixture/manifest.json'),
                scope=SCOPE)

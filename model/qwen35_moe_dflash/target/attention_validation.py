"""Bind Target operator fixtures to compiled X90 IO and compare device snapshots."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from . import gdn_probe, cache_probe

KIND = 'qwen35-target-attention-operator-validation'
MODULES = {'gdn': gdn_probe, 'cache': cache_probe}
KINDS = {module.KIND: family for family, module in MODULES.items()}
CASES = tuple('target_gdn_'+name for name in (
    't64_zero_state', 't64_nonzero_state', 't1_zero_state', 't1_nonzero_state'
)) + tuple('target_cache_'+name for name in cache_probe.CASES)
WARMUP, REPETITIONS = 1, 10


def compatibility_summary(compiled):
    result = {}
    for name, row in compiled.get('models', {}).items():
        evidence = row.get('tbe_compatibility')
        result[name] = (dict(status='NOT_REQUESTED') if not evidence else dict(
            status=(evidence.get('audit') or {}).get('status', 'NOT_AUDITED'),
            diagnosis=evidence.get('diagnosis'), compile=evidence.get('compile'),
            audit_log=(evidence.get('metadata') or {}).get('audit_log')))
    return result


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from validate_draft_vector import bind_io, selected_cases
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    from qwen35_moe_dflash.draft.incremental.torch_export import _compiled_observation

    root, out = Path(root).resolve(), Path(out).resolve()
    family = KINDS[read_json(root/'manifest.json')['artifact_kind']]
    _, manifest = MODULES[family].load_prepared(root)
    compiled = read_json(root/'compile.json')
    if (compiled.get('artifact_kind') != manifest['artifact_kind']
            or compiled.get('manifest_sha256') != sha256_file(root/'manifest.json')):
        raise ValueError('compilation is not bound to these prepared fixtures')
    available = {'target_'+family+'_'+c['id']: c for c in manifest['cases']
                 if compiled.get('models', {}).get(c['model'], {}).get('status') == 'COMPILED'}
    selected = selected_cases(available if selection is None else selection, available)
    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    enums, bindings, cases = _enum_fields(schema), {}, []
    for name in selected:
        case = available[name]
        model_key = case['model']
        if model_key not in bindings:
            record = compiled['models'][model_key]['omc']
            model = checked_file(root, record)
            compilation = read_json(root/'models'/model_key/'omc/report.json')
            if (compilation.get('status') != 'COMPILED' or compilation.get('platform') != 'kirinx90'
                    or compilation.get('ddk_id') != ddk_id):
                raise ValueError('probe compiler platform/DDK differs from runtime configuration')
            if model.stat().st_size > 64 << 20 or any(model.parent.glob('*.weight')):
                raise ValueError('operator probe must be small and have embedded weights')
            parts = [p for p in read_container(model)['partitions'] if p['type'] == 0]
            if len(parts) != 1:
                raise ValueError('expected one compiled model metadata partition')
            document = schema.decode(read_partition(model, parts[0]), 'ModelDef')
            io = bind_io(document, manifest['models'][model_key]['graph'], enums,
                         input_order='graph-input')
            observed = _compiled_observation(model, ddk)
            checked_file(root, record)
            bindings[model_key] = (io, model, observed)
        io, model, observed = bindings[model_key]
        bound = deepcopy(io)
        inputs = case['input'] if family == 'gdn' else case['inputs']
        golden = case['golden'] if family == 'gdn' else [case['golden']]
        folder = out/'fixture'/name
        folder.mkdir(parents=True)
        plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}']
        for direction, prefix, records in (('inputs', 'input', inputs), ('outputs', 'golden', golden)):
            for spec in bound[direction]:
                source = checked_file(root, records[spec['fixture_index']])
                if source.stat().st_size != spec['bytes']:
                    raise ValueError('fixture extent differs from compiled IO')
                destination = folder/f'{prefix}_{spec["index"]}.bin'
                shutil.copyfile(source, destination)
                spec['fixture_sha256'] = sha256_file(destination)
                plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'],
                    spec['dtype'], ','.join(map(str, spec['shape'])))))
        (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
        cases.append(dict(case=name, family=family, **bound,
            model=_file_record(model, family+'_'+model_key+'/model.omc'),
            graphop_counts=observed['graphop_counts'], compiled_observation=observed))
    meta = dict(artifact_kind=KIND, cases=cases, warmup=WARMUP, repetitions=REPETITIONS,
        planned_cases=['target_'+family+'_'+c['id'] for c in manifest['cases']],
        uncompiled_cases=['target_'+family+'_'+c['id'] for c in manifest['cases']
                          if 'target_'+family+'_'+c['id'] not in available],
        output_initialization='zero', ddk_id=ddk_id, schema=schema.evidence,
        scope='SYNTHETIC_OPERATOR_CASES_NOT_FULL_TARGET_EQUIVALENCE',
        source_evidence=dict(artifact_kind=manifest['artifact_kind'],
            manifest_sha256=sha256_file(root/'manifest.json'),
            compile_sha256=sha256_file(root/'compile.json'),
            tbe_compatibility=compatibility_summary(compiled),
            scope=manifest.get('scope'), master_differences=manifest.get('master_differences', [])),
        cache_alias_semantics='RETURNED_CACHE_ONLY_INPUT_MUTATION_NOT_PROVEN',
        gdn_feedback='ORIGINAL_STATE_REUSED_NOT_CONTINUOUS_STATE_FEEDBACK')
    _save(out/'fixture/manifest.json', meta)
    return meta


def compare(fixture, native, *, selection=None, allow_host_fake=False):
    from validate_draft_vector import selected_cases, member
    fixture, native = Path(fixture), Path(native)
    meta = read_json(fixture/'manifest.json')
    requested = [c['case'] for c in meta['cases']]
    selected = selected_cases(requested if selection is None else selection, requested)
    rows = []
    for case in meta['cases']:
        name = case['case']
        if name not in selected:
            continue
        # Also validate offline re-comparisons, outside the recovery path.
        for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
            for spec in case[direction]:
                checked_file(fixture, dict(path=f'{name}/{prefix}_{spec["index"]}.bin',
                    bytes=spec['bytes'], sha256=spec['fixture_sha256']))
        receipt_path = member(native, name+'/result.json')
        receipt = read_json(receipt_path)
        times = receipt.get('api_us', [])
        if (receipt.get('status') != 'EXECUTED'
                or receipt.get('evidence_kind') != ('host-fake-hiai' if allow_host_fake else 'device-hiai')
                or receipt.get('warmup') != WARMUP or receipt.get('repetitions') != REPETITIONS
                or receipt.get('output_initialization') != meta['output_initialization']
                or receipt.get('timing_scope') != 'SYNCHRONOUS_API_WALL_NOT_NPU_ONLY'
                or len(times) != REPETITIONS or any(type(t) is not int or t < 0 for t in times)):
            raise ValueError('runner receipt does not match the Target probe plan: '+name)
        written = None
        if case['family'] == 'cache':
            by_name = {s['name']: s for s in case['inputs']}
            controls = [np.fromfile(fixture/name/f'input_{by_name[n]["index"]}.bin', dtype='<i4')
                        for n in ('targetBlock', 'offsetInBlock')]
            written = np.zeros(cache_probe.SHAPE, dtype=bool)
            written[controls[0], :, controls[1], :] = True
        invocations = []
        for iteration in range(REPETITIONS):
            snapshot = member(native, f'{name}/{iteration}.bin')
            if snapshot.stat().st_size != sum(s['bytes'] for s in case['outputs']):
                raise ValueError('incorrect snapshot extent: '+name)
            raw, offset, outputs = snapshot.read_bytes(), 0, {}
            for spec in case['outputs']:
                expected_raw = (fixture/name/f'golden_{spec["index"]}.bin').read_bytes()
                actual_raw = raw[offset:offset+spec['bytes']]
                offset += spec['bytes']
                actual = np.frombuffer(actual_raw, dtype='<f2').reshape(spec['shape'])
                expected = np.frombuffer(expected_raw, dtype='<f2').reshape(spec['shape'])
                metrics = gdn_probe.metrics(actual, expected)
                metrics['bytes_equal'] = actual_raw == expected_raw
                if written is not None:
                    bad = actual.view('<u2') != expected.view('<u2')
                    metrics.update(mismatched_fp16_elements=int(bad.sum()),
                        mismatched_written_elements=int((bad & written).sum()),
                        mismatched_untouched_elements=int((bad & ~written).sum()),
                        first_mismatch_coordinates=np.argwhere(bad)[:8].tolist())
                outputs[spec['name']] = metrics
            invocations.append(dict(index=iteration, sha256=hashlib.sha256(raw).hexdigest(), outputs=outputs))
        values = [v for i in invocations for v in i['outputs'].values()]
        stable = len({i['sha256'] for i in invocations}) == 1
        finite = all(v['nonfinite_actual'] == v['nonfinite_golden'] == 0 for v in values)
        exact = all(v['bytes_equal'] for v in values)
        status = ('FAILED' if not finite or not stable or (written is not None and not exact)
                  else 'PASSED' if written is not None else 'METRICS_RECORDED')
        rows.append(dict(case=name, status=status, family=case['family'],
            repeat_bitwise_stable=stable, finite=finite, all_calls_bytes_equal=exact,
            numerical_gate='EXACT_WHOLE_CACHE_BITS' if written is not None else 'NO_CALIBRATED_ERROR_THRESHOLD',
            invocations=invocations, runner_receipt_sha256=sha256_file(receipt_path),
            graphop_counts=case['graphop_counts'], model=case['model'], api_us=times))
    failed = any(r['status'] == 'FAILED' for r in rows)
    metrics_only = any(r['family'] == 'gdn' for r in rows)
    complete = set(selected) == set(meta['planned_cases'])
    status = 'NUMERICAL_FAILED' if failed else 'METRICS_RECORDED' if metrics_only else 'PASSED'
    if not complete and not failed:
        status = 'PARTIAL'
    return dict(status=status, numerical_validation='FAILED' if failed else
        'METRICS_ONLY_NO_CALIBRATED_THRESHOLD' if metrics_only else 'PASSED',
        cases=rows, validated_cases=selected, requested_cases=requested,
        coverage_complete=complete, planned_cases=meta['planned_cases'],
        uncompiled_cases=meta['uncompiled_cases'],
        all_operators_passed=False, model_compilation='COMPILED',
        evidence_kind='host-fake-hiai' if allow_host_fake else 'device-hiai',
        source_evidence=meta['source_evidence'], scope=meta['scope'],
        tbe_compatibility=meta['source_evidence'].get('tbe_compatibility', {}),
        warmup=WARMUP, repetitions=REPETITIONS,
        output_initialization=meta['output_initialization'],
        cache_alias_semantics=meta['cache_alias_semantics'], gdn_feedback=meta['gdn_feedback'])

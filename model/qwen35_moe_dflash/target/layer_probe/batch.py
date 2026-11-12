"""Sequential export/compile, exact compiled binding, and batch numerical reports."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from .common import KIND, DTYPES, checked, identity, metrics, new_output, read_array, save_json, sha
from qwen35_moe_dflash.toolchain.compiler_entry import (
    DEFAULT_PROFILE, SOURCED_PROFILE, UNSPECIFIED_PLATFORM,
    sourced_wrapper_environment, validate_profile)

# Dependency order, independent of when a post-hook finishes writing its files.
POINT_ORDER = {name: i for i, name in enumerate((
    'input_layernorm', 'linear_attn_in_proj_qkv', 'linear_attn_in_proj_z',
    'linear_attn_in_proj_b', 'linear_attn_in_proj_a', 'convstate_update', 'conv1d',
    'conv_silu', 'decay_beta', 'repeat_qk', 'gdn', 'recurrent_writeback',
    'linear_attn_norm', 'linear_attn_out_proj', 'self_attn_q_proj', 'self_attn_q_norm',
    'self_attn_k_proj', 'self_attn_k_norm', 'self_attn_v_proj', 'rope', 'cache_k',
    'cache_v', 'fused_attention', 'attention_layout_gate', 'self_attn_o_proj',
    'residual_attention', 'post_attention_layernorm', 'residual_mlp'))}


def export_variants(region, cache_indexing):
    modes = ('float', 'integer') if region['kind'] == 'cache' and cache_indexing == 'both' else (
        cache_indexing if region['kind'] == 'cache' else 'kernel',)
    for mode in modes:
        suffix = '' if mode == 'kernel' else '_index_' + mode
        for index, tensor in enumerate(region['outputs']):
            yield mode, suffix, index, tensor


def export_batch(capture, output, source_root, converter_script, *, points=None, kinds=None,
                 timeout=900, torchair_pb2=None, target_config=None, qlinear_scales='source', cache_indexing='kernel'):
    if qlinear_scales not in ('source', 'fp16') or cache_indexing not in ('kernel', 'float', 'integer', 'both'):
        raise ValueError('unknown module candidate')
    capture = Path(capture).resolve()
    source = json.loads((capture/'manifest.json').read_text())
    if source.get('artifact_kind') != KIND or source.get('status') != 'CAPTURED':
        raise ValueError('requires completed 310P capture')
    regions = [r for r in source['regions'] if (not points or r['point'] in points)
               and (not kinds or r['kind'] in kinds)]
    if not regions:
        raise ValueError('no matching captured regions')
    if points and set(points) - {r['point'] for r in regions}:
        raise ValueError('requested point is missing from capture')
    if any('weights' in r for r in regions) and not target_config:
        raise ValueError('weighted regions require --target-config with original model_file and quanted_pth')
    root = new_output(output)
    report = dict(artifact_kind=KIND, status='EXPORTING', cases=[],
        capture_sha256=sha(capture/'manifest.json'), golden_source=source['reference'],
        layers=source['layers'], reference_sources=source['sources'],
        scope='ONE_CPRO_REGION_PER_GRAPH; GOLDEN_INPUT_REPLAY; NO_FULL_MODEL_EXPORT',
        golden_state_feedback='CAPTURED_310P_TRAJECTORY', cpro_state_feedback='NOT_TESTED',
        reference_quant_matmul_policy=source.get('quant_matmul', {}).get('policy', 'ORIGINAL_CALL'),
        reference_baseline_equivalence=source.get('quant_matmul', {}).get('baseline_equivalence', 'ORIGINAL_CALL_UNCHANGED'),
        reference_model_layers=source.get('placement', {}).get('executed_num_hidden_layers'))
    report['candidate_options'] = dict(qlinear_scales=qlinear_scales, cache_indexing=cache_indexing,
        reference='ORIGINAL_CAPTURE_UNCHANGED')
    save_json(root/'manifest.json', report)
    tool = Path(__file__).resolve().parents[4]/'targets/cpro/tools/target_layer_probe.py'
    for region in regions:
        for mode, suffix, index, tensor in export_variants(region, cache_indexing):
            name = region['id'] + suffix + f'_o{index}'
            folder = root/'models'/name
            folder.parent.mkdir(exist_ok=True)
            row = dict(id=name, region=region['id'], layer=region['layer'], call=region['call'],
                point=region['point'] + suffix, phase=region['phase'], kind=region['kind'],
                ordinal=region['ordinal'], output_index=index, output_name=tensor['name'], status='EXPORTING')
            report['cases'].append(row)
            save_json(root/'manifest.json', report)
            argv = [sys.executable, str(tool), 'export-one', '--capture', str(capture),
                '--region', region['id'], '--output-index', str(index), '--output', str(folder),
                '--source-root', str(Path(source_root).resolve()),
                '--converter-script', str(Path(converter_script).resolve())]
            if qlinear_scales != 'source':
                argv += ['--qlinear-scales', qlinear_scales]
            if mode != 'kernel':
                argv += ['--cache-indexing', mode]
            if torchair_pb2:
                argv += ['--torchair-pb2', str(Path(torchair_pb2).resolve())]
            if target_config:
                argv += ['--target-config', str(Path(target_config).resolve())]
            print('export', name, flush=True)
            log = root/(name+'.log')
            try:
                with log.open('w') as stream:
                    done = subprocess.run(argv, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                        timeout=timeout, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                if done.returncode:
                    raise RuntimeError(f'export worker exit {done.returncode}; see {log.name}')
                receipt = json.loads((folder/'export.json').read_text())
                if receipt['status'] != 'EXPORTED' or receipt['capture_manifest_sha256'] != report['capture_sha256']:
                    raise ValueError('export receipt differs from captured golden')
                if receipt.get('source_contract', {}).get('status') != 'BOUND_TO_SOURCE':
                    raise ValueError('export worker did not bind this region to original Target source')
                if ((region['kind'] == 'qlinear' and qlinear_scales == 'fp16') or mode != 'kernel'):
                    if not receipt.get('candidate') or receipt.get('candidate_graph_check', {}).get('status') != 'PASSED':
                        raise ValueError('candidate export is missing its source AIR checks')
                row.update(status='EXPORTED', export=identity(folder/'export.json', root))
            except Exception as exc:
                row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            save_json(root/'manifest.json', report)
    report['status'] = 'EXPORTED' if all(r['status'] == 'EXPORTED' for r in report['cases']) else 'PARTIAL'
    report['source_contracts'] = dict(scope='EVERY_EXPORTED_CASE; SEE_EXPORT_JSON',
        untested=['MoE internals', 'LM head', 'CPro state feedback', 'full-graph storage equivalence'])
    save_json(root/'manifest.json', report)
    return report


def compile_batch(root, output, ddk, ddk_id, *, tbe_diagnosis=None, timeout=900, use_origin_format=None,
                  cases=None, compiler_profile=DEFAULT_PROFILE):
    """Read an AIR batch and write a fresh, separate compilation batch."""
    from qwen35_moe_dflash.common.artifacts import output_path
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    root, ddk, output = Path(root).resolve(), Path(ddk).resolve(), output_path(output)
    for source in (root, ddk):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('compilation output must be separate from the AIR batch and DDK directories; '
                             'use --output with a fresh sibling directory')
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('artifact_kind') != KIND or manifest.get('status') not in ('EXPORTED', 'PARTIAL'):
        raise ValueError('requires region AIR batch')
    known = [c['id'] for c in manifest['cases']]
    selected = known if cases is None else list(cases)
    if (not selected or len(set(known)) != len(known) or len(set(selected)) != len(selected)
            or set(selected) - set(known)):
        raise ValueError('select distinct known AIR cases')
    validate_profile(compiler_profile, tbe_diagnosis)
    diagnostic = compiler_profile == SOURCED_PROFILE
    if diagnostic:
        # Fail before creating a batch if the caller/DDK entry is not bound.
        sourced_wrapper_environment(ddk)
    output = new_output(output)
    # Keep a small manifest snapshot for the board entry's batch detection. AIR,
    # weights and golden fixtures remain in the original, read-only source batch.
    shutil.copyfile(root/'manifest.json', output/'manifest.json')
    report = dict(artifact_kind=KIND, status='COMPILING', cases={}, ddk_id=ddk_id,
        platform=UNSPECIFIED_PLATFORM if diagnostic else 'kirinx90',
        compiler_profile=compiler_profile, selected_cases=selected,
        tbe_compatibility_requested=tbe_diagnosis is not None,
        coverage='ALL_MANIFEST_CASES' if set(selected) == set(known) else 'SELECTED_CASES_ONLY',
        status_scope='SELECTED_CASES', diagnostic_only=diagnostic,
        manifest_sha256=sha(output/'manifest.json'),
        source_batch=os.path.relpath(root, output))
    save_json(output/'compile.json', report)
    for case in manifest['cases']:
        row = report['cases'][case['id']] = dict(status='NOT_EXPORTED')
        if case['id'] not in selected:
            row['status'] = 'NOT_SELECTED'
            continue
        if case['status'] != 'EXPORTED':
            continue
        try:
            exported = json.loads(checked(root, case['export']).read_text())
            folder = root/'models'/case['id']
            air = checked(folder, exported['air'])
            destination = output/'models'/case['id']/'omc'
            print('compile', case['id'], flush=True)
            compiled = compile_existing_air(air, destination, ddk, ddk_id=ddk_id,
                timeout_seconds=timeout, save_weights_as_external_data=diagnostic,
                limit_partition_size=False if diagnostic else None, compiler_profile=compiler_profile,
                use_origin_format=use_origin_format, tbe_diagnosis=tbe_diagnosis)
            if compiled['status'] != 'COMPILED':
                raise RuntimeError(compiled.get('error', 'OMG failed; inspect retained omc/report.json'))
            row.update(status='COMPILED', omc=identity(destination/compiled['omc']['path'], output))
        except Exception as exc:
            row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        save_json(output/'compile.json', report)
    report['status'] = 'COMPILED' if all(report['cases'][name]['status'] == 'COMPILED' for name in selected) else 'PARTIAL'
    save_json(output/'compile.json', report)
    return report


def bind_compiled(document, source_inputs, golden, enums):
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    graphs = document.get('graph', [])
    if len(graphs) != 1:
        raise ValueError('requires one compiled main graph')
    graph = graphs[0]
    nodes = graph['op']
    by_name = {s['name']: s for s in source_inputs}
    data = {n['name']: n for n in nodes if n['type'] == 'Data'}
    if len(data) != len([n for n in nodes if n['type'] == 'Data']) or set(data) - set(by_name):
        raise ValueError('compiled Data names differ from source-bound inputs')
    references = graph.get('input', [])
    if len(references) != len(data) or len(set(references)) != len(data):
        raise ValueError('compiled GraphDef.input must explicitly order each Data')
    inputs = []
    for slot, reference in enumerate(references):
        if not reference.endswith(':0') or reference[:-2] not in data:
            raise ValueError('unbound compiled input edge: ' + reference)
        name = reference[:-2]
        spec = by_name[name]
        descs = data[name].get('output_desc', [])
        observed = [_observed_desc(desc, enums) for desc in descs]
        if len(observed) != 1 or any(observed[0][k] != spec[k] for k in ('shape', 'dtype')):
            for desc in observed:
                desc['schema_dtype_names'] = [alias for alias, code in enums.get('dtype', {}).get('values', {}).items()
                                              if code == desc['raw_dtype']]
            expected = {k: spec[k] for k in ('shape', 'dtype')}
            raise ValueError(f'compiled input shape/dtype changed: {name}; '
                             f'expected={json.dumps(expected)}; observed={json.dumps(observed)}')
        inputs.append(dict(spec, index=slot))
    nets = [n for n in nodes if n['type'] == 'NetOutput']
    if len(nets) != 1 or len(nets[0].get('input_desc', [])) != 1:
        raise ValueError('region must have exactly one compiled output; never guess K/V order')
    observed = _observed_desc(nets[0]['input_desc'][0], enums)
    if observed['shape'] != golden['shape'] or observed['dtype'] not in DTYPES:
        raise ValueError('compiled output shape or dtype is incompatible with captured boundary')
    output = dict(golden, index=0, dtype=observed['dtype'], golden_dtype=golden['dtype'],
                  bytes=math.prod(observed['shape'])*np.dtype(DTYPES[observed['dtype']]).itemsize)
    return dict(inputs=inputs, outputs=[output], source_dtype_preserved=observed['dtype'] == golden['dtype'],
                input_binding='SOURCE_ARGUMENT_NAME_THEN_COMPILED_GRAPHDEF_INPUT_ORDER',
                output_binding='SOLE_COMPILED_OUTPUT_NO_SHAPE_BASED_SLOT_GUESSING')


def check_io_budget(name, io):
    sizes = {d: sum(s['bytes'] for s in io[d]) for d in ('inputs', 'outputs')}
    total = sum(sizes.values())
    if total > 32 << 20 or any(len(io[d]) > 8 for d in sizes):
        raise ValueError(f'{name}: probe IO requires {total} bytes '
                         f'(inputs={sizes["inputs"]}, outputs={sizes["outputs"]}); '
                         'limit=33554432 bytes (32 MiB), at most 8 slots per direction')


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    from qwen35_moe_dflash.draft.incremental.torch_export import _compiled_observation
    root, out = Path(root).resolve(), Path(out)
    manifest = json.loads((root/'manifest.json').read_text())
    compiled = json.loads((root/'compile.json').read_text())
    if compiled.get('compiler_profile', DEFAULT_PROFILE) != DEFAULT_PROFILE or compiled.get('diagnostic_only'):
        raise ValueError('compiler-entry diagnostic batch has an unverified target; cannot prepare board fixtures')
    if compiled['manifest_sha256'] != sha(root/'manifest.json') or compiled['ddk_id'] != ddk_id:
        raise ValueError('compile source/DDK identity changed')
    # Legacy batches colocate fixtures and OMCs. New batches reference the source
    # relatively so both directories can be moved together without rewriting evidence.
    source_root = (root/compiled.get('source_batch', '.')).resolve()
    if not (source_root/'manifest.json').is_file() or sha(source_root/'manifest.json') != compiled['manifest_sha256']:
        raise ValueError('AIR source manifest is missing or changed; retain the original AIR batch '
                         'at the relative source_batch path recorded in compile.json')
    available = {r['id']: r for r in manifest['cases'] if compiled['cases'].get(r['id'], {}).get('status') == 'COMPILED'}
    selected = list(available) if selection is None else list(selection)
    if not selected or len(set(selected)) != len(selected) or set(selected)-set(available):
        raise ValueError('select unique compiled regions')
    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    enums, cases = _enum_fields(schema), []
    for name in selected:
        case = available[name]
        exported = json.loads(checked(source_root, case['export']).read_text())
        folder = source_root/'models'/name
        model = checked(root, compiled['cases'][name]['omc'])
        parts = [p for p in read_container(model)['partitions'] if p['type'] == 0]
        if len(parts) != 1 or any(model.parent.glob('*.weight')):
            raise ValueError('requires embedded weights and one main metadata partition')
        doc = schema.decode(read_partition(model, parts[0]), 'ModelDef')
        try:
            io = bind_compiled(doc, exported['source_inputs'], exported['golden'], enums)
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from exc
        check_io_budget(name, io)
        fixture = out/'fixture'/name
        fixture.mkdir(parents=True)
        plan = ['QWEN35_TENSOR_PROBE 1', 'iterations 3 10', 'input_reset copy-each-call']
        for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
            for spec in io[direction]:
                source = checked(folder, exported['golden'] if direction == 'outputs' else spec)
                destination = fixture/f'{prefix}_{spec["index"]}.bin'
                shutil.copyfile(source, destination)
                spec['fixture_sha256'] = sha(destination)
                spec['fixture_bytes'] = destination.stat().st_size
                spec['name'] = prefix + '_' + str(spec['index'])
                plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'], ','.join(map(str, spec['shape'])))))
        (fixture/'plan.txt').write_text('\n'.join(plan)+'\n')
        observed = _compiled_observation(model, ddk)
        cases.append(dict(case=name, **io, layer=case['layer'], call=case['call'], point=case['point'],
            kind=case['kind'], phase=case['phase'], ordinal=case['ordinal'], output_name=case['output_name'],
            model=_file_record(model, name+'/model.omc'), graphop_counts=observed['graphop_counts']))
    meta = dict(artifact_kind=KIND, cases=cases, warmup=3, repetitions=10, output_initialization='zero', input_reset='copy-each-call',
        planned_cases=[r['id'] for r in manifest['cases']], uncompiled_cases=[r['id'] for r in manifest['cases'] if r['id'] not in available],
        ddk_id=ddk_id, schema=schema.evidence, scope=manifest['scope'],
        source_evidence=dict(artifact_kind=KIND, manifest_sha256=sha(root/'manifest.json'), compile_sha256=sha(root/'compile.json')))
    meta['reference'] = {k: manifest.get(k) for k in ('golden_source', 'reference_quant_matmul_policy',
        'reference_baseline_equivalence', 'reference_model_layers')}
    save_json(out/'fixture/manifest.json', meta)
    return meta


def compare(fixture, native, *, selection=None, allow_host_fake=False, atol=None, rtol=None):
    fixture, native = Path(fixture), Path(native)
    meta = json.loads((fixture/'manifest.json').read_text())
    if (atol is None) != (rtol is None) or any(v is not None and (not math.isfinite(v) or v < 0) for v in (atol, rtol)):
        raise ValueError('atol/rtol must both be supplied and finite nonnegative')
    requested = {c['case'] for c in meta['cases']}
    if selection is not None and (not selection or set(selection) - requested or len(set(selection)) != len(selection)):
        raise ValueError('select distinct captured fixture cases')
    rows = []
    for case in meta['cases']:
        if selection is not None and case['case'] not in selection:
            continue
        row = {k: case[k] for k in ('case', 'layer', 'call', 'point', 'phase', 'ordinal', 'output_name')}
        try:
            folder = native/case['case']
            receipt = json.loads((folder/'result.json').read_text())
            if (receipt.get('status') != 'EXECUTED' or receipt.get('evidence_kind') != ('host-fake-hiai' if allow_host_fake else 'device-hiai')
                    or receipt.get('repetitions') != meta['repetitions'] or receipt.get('warmup') != meta['warmup']
                    or len(receipt.get('api_us', [])) != meta['repetitions']
                    or receipt.get('input_reset') != 'copy-each-call'):
                raise ValueError('runner did not complete the declared device invocations')
            for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
                for s in case[direction]:
                    checked(fixture, dict(path=f'{case["case"]}/{prefix}_{s["index"]}.bin', bytes=s['fixture_bytes'], sha256=s['fixture_sha256']))
            spec = case['outputs'][0]
            golden = np.fromfile(fixture/case['case']/'golden_0.bin', dtype=DTYPES[spec['golden_dtype']]).reshape(spec['shape'])
            tensors, hashes = [], []
            for index in range(meta['repetitions']):
                path = folder/f'{index}.bin'
                if path.stat().st_size != spec['bytes']:
                    raise ValueError('runtime output byte extent differs from compiled ABI')
                actual = np.fromfile(path, dtype=DTYPES[spec['dtype']]).reshape(spec['shape'])
                metric = metrics(actual, golden, atol=atol, rtol=rtol)
                if case['kind'] == 'cache':
                    identical = actual.dtype == golden.dtype and actual.tobytes() == golden.tobytes()
                    metric.update(gate='EXACT_WHOLE_CACHE_BITS', passed=identical)
                    source_spec = next(s for s in case['inputs'] if s['fixture_index'] == 0)
                    before = np.fromfile(fixture/case['case']/f'input_{source_spec["index"]}.bin', dtype=DTYPES[source_spec['dtype']]).reshape(golden.shape)
                    # Classify preserved positions using the executed golden, not an assumed scatter ABI.
                    if before.dtype != golden.dtype:
                        raise ValueError('cache golden dtype differs from its input boundary')
                    bits = np.dtype(f'u{golden.dtype.itemsize}')
                    preserved = before.view(bits) == golden.view(bits)
                    changed = actual != golden
                    metric.update(mismatched_preserved_elements=int((changed & preserved).sum()),
                                  mismatched_updated_elements=int((changed & ~preserved).sum()))
                tensors.append(metric)
                hashes.append(sha(path))
            stable = len(set(hashes)) == 1
            finite = all(r['nonfinite_actual'] == r['nonfinite_golden'] == 0 for r in tensors)
            passed = all(r['passed'] is True for r in tensors)
            failed = not stable or not finite or any(r['passed'] is False for r in tensors)
            row.update(status='FAILED' if failed else 'PASSED' if passed else 'METRICS_RECORDED',
                stable=stable, source_dtype_preserved=case['source_dtype_preserved'], invocations=tensors,
                output_sha256=hashes, api_us=receipt['api_us'], graphop_counts=case['graphop_counts'])
        except Exception as exc:
            row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    rows.sort(key=lambda r: (r['call'], r['layer'], POINT_ORDER.get(r['point'], 1000), r['ordinal'], r['case']))
    failed = [r for r in rows if r['status'] == 'FAILED']
    status = 'FAILED' if failed else 'PASSED' if rows and all(r['status']=='PASSED' for r in rows) else 'METRICS_RECORDED'
    complete = not meta.get('uncompiled_cases') and len(rows) == len(meta['cases'])
    return dict(status=status if complete or status == 'FAILED' else 'PARTIAL',
        numerical_validation=status, coverage_complete=complete, validated_cases=[r['case'] for r in rows],
        cases=rows, first_failed_boundary=failed[0]['case'] if failed else None,
        uncompiled_cases=meta.get('uncompiled_cases', []), all_model_layers_passed=False,
        reference=meta.get('reference', {}),
        scope='CPro boundary replay with 310P inputs; isolated failures are not proof of full-graph root cause',
        first_failure_order='CALL_LAYER_CPRO_DEPENDENCY_ORDER; INDEPENDENT_BRANCHES_HAVE_NO_UNIQUE_FIRST_CAUSE',
        cpro_continuous_state_feedback='NOT_TESTED', thresholds=dict(atol=atol, rtol=rtol))

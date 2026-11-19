"""Sequential export/compile, exact compiled binding, and batch numerical reports."""
from __future__ import annotations
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from qwen35_moe_dflash.common.accuracy import POLICY, THRESHOLDS
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
    if region['kind'] in ('gdn_layer', 'full_attention_layer'):
        yield 'kernel', '', None, dict(name='whole_layer')
        return
    modes = ('float', 'integer') if region['kind'] == 'cache' and cache_indexing == 'both' else (
        cache_indexing if region['kind'] == 'cache' else 'kernel',)
    for mode in modes:
        suffix = '' if mode == 'kernel' else '_index_' + mode
        for index, tensor in enumerate(region['outputs']):
            yield mode, suffix, index, tensor


def export_batch(capture, output, source_root, converter_script, *, points=None, kinds=None, phases=None,
                 timeout=900, torchair_pb2=None, target_config=None, qlinear_scales='source', cache_indexing='kernel',
                 rope_precision='source', gdn_state_dtype='source', gdn_convstate='kernel',
                 full_attention_reference='math-replay', gdr_mode='chunk', gdn_trace_layer=None, gdr_inputs=False):
    if (qlinear_scales not in ('source', 'fp16') or cache_indexing not in ('kernel', 'float', 'integer', 'both')
            or rope_precision not in ('source', 'fp32') or gdn_state_dtype not in ('source', 'float16', 'float32')
            or gdn_convstate not in ('kernel', 'reference') or gdr_mode not in ('chunk', 'single-token')
            or full_attention_reference not in ('math-replay', 'captured-audited')):
        raise ValueError('unknown module candidate')
    if phases is not None and (not phases or set(phases) - {'prefill', 'decode'}):
        raise ValueError('select prefill and/or decode phases')
    if gdr_inputs and (gdn_trace_layer is None or gdn_state_dtype != 'float16'):
        raise ValueError('--gdr-inputs requires --gdn-trace-layer and --gdn-state-dtype float16')
    if gdn_trace_layer is not None:
        if (points and set(points) != {'gdn_layer'} or kinds and set(kinds) != {'gdn_layer'}
                or phases and set(phases) != {'prefill'} or gdr_mode != 'single-token'):
            raise ValueError('--gdn-trace-layer requires a single-token whole GDN Prefill probe')
        points, phases = ['gdn_layer'], ['prefill']
    capture = Path(capture).resolve()
    source = json.loads((capture/'manifest.json').read_text())
    if source.get('artifact_kind') != KIND or source.get('status') != 'CAPTURED':
        raise ValueError('requires completed 310P capture')
    # Whole Attention needs a qualified reference and explicit selection;
    # ordinary boundary batches remain compatible with captures that include it.
    available_regions = [r for r in source['regions'] if r['kind'] != 'full_attention_layer']
    if points and 'gdn_layer' in points or kinds and 'gdn_layer' in kinds:
        from .whole_gdn import assemble
        available_regions += assemble(source, trace_layer=gdn_trace_layer, trace_inputs=gdr_inputs)
    if points and 'full_attention_layer' in points or kinds and 'full_attention_layer' in kinds:
        from .whole_attention import assemble
        available_regions += assemble(source, capture=capture, reference=full_attention_reference)
    regions = [r for r in available_regions if (not points or r['point'] in points)
               and (not kinds or r['kind'] in kinds)
               and (phases is None or r['phase'] in phases)]
    if not regions:
        raise ValueError('no matching captured regions')
    if gdr_mode == 'single-token' and any(r['phase'] != 'prefill' or not (
            r['kind'] == 'gdn_layer' or r['kind'] == 'gdn' and 'gdr_valid_rows' in r) for r in regions):
        raise ValueError('--gdr-mode single-token requires whole Prefill GDN or qualified segmented GDR cases')
    if full_attention_reference != 'math-replay' and not any(r['kind'] == 'full_attention_layer' for r in regions):
        raise ValueError('--full-attention-reference requires full_attention_layer selection')
    if gdn_convstate != 'kernel' and any(r['kind'] != 'gdn_layer' for r in regions):
        raise ValueError('--gdn-convstate reference requires only whole gdn_layer regions')
    if points and set(points) - {r['point'] for r in regions}:
        raise ValueError('requested point is missing from capture')
    if phases is not None and set(phases) - {r['phase'] for r in regions}:
        raise ValueError('requested phase is missing from selected captured regions')
    if any('weights' in r for r in regions) and not target_config:
        raise ValueError('weighted regions require --target-config with original model_file and quanted_pth')
    if any(r['kind'] == 'gdn_layer' for r in regions) and (
            gdn_state_dtype == 'source' or qlinear_scales != 'fp16' or not target_config):
        raise ValueError('whole GDN requires --gdn-state-dtype float16|float32 --qlinear-scales fp16 --target-config')
    if any(r['kind'] == 'full_attention_layer' for r in regions) and (
            qlinear_scales != 'fp16' or rope_precision != 'fp32' or not target_config):
        raise ValueError('whole FullAttention requires --qlinear-scales fp16 --rope-precision fp32 --target-config')
    if all('gdr_valid_rows' in r for r in regions) and gdn_state_dtype == 'source':
        gdn_state_dtype = 'float16'
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
        rope_precision=rope_precision, gdn_state_dtype=gdn_state_dtype, gdn_convstate=gdn_convstate,
        reference='ORIGINAL_CAPTURE_UNCHANGED')
    if any(r['kind'] == 'full_attention_layer' for r in regions):
        report['candidate_options']['full_attention_reference'] = full_attention_reference
    if any(r['kind'] == 'gdn_layer' or 'gdr_valid_rows' in r for r in regions):
        report['candidate_options']['gdr_mode'] = gdr_mode
    if gdn_trace_layer is not None:
        report['candidate_options']['gdn_trace_layer'] = gdn_trace_layer
    if gdr_inputs:
        report['candidate_options']['gdr_inputs'] = True
    report['selected_phases'] = sorted({r['phase'] for r in regions})
    save_json(root/'manifest.json', report)
    tool = Path(__file__).resolve().parents[4]/'targets/cpro/tools/target_layer_probe.py'
    for region in regions:
        for mode, suffix, index, tensor in export_variants(region, cache_indexing):
            name = region['id'] + suffix + (f'_o{index}' if index is not None else '_all')
            folder = root/'models'/name
            folder.parent.mkdir(exist_ok=True)
            row = dict(id=name, region=region['id'], layer=region['layer'], call=region['call'],
                point=region['point'] + suffix, phase=region['phase'], kind=region['kind'],
                ordinal=region['ordinal'], output_index=index, output_name=tensor['name'], status='EXPORTING')
            report['cases'].append(row)
            if region['kind'] == 'full_attention_layer':
                row['reference_basis'] = region['reference_basis']
            save_json(root/'manifest.json', report)
            argv = [sys.executable, str(tool), 'export-one', '--capture', str(capture),
                '--region', region['id'], '--output', str(folder),
                '--source-root', str(Path(source_root).resolve()),
                '--converter-script', str(Path(converter_script).resolve())]
            if index is not None:
                argv += ['--output-index', str(index)]
            else:
                row['output_names'] = [s['name'] for s in region['outputs']]
            if gdn_state_dtype != 'source':
                argv += ['--gdn-state-dtype', gdn_state_dtype]
            if gdn_convstate != 'kernel':
                argv += ['--gdn-convstate', gdn_convstate]
            if gdr_mode != 'chunk':
                argv += ['--gdr-mode', gdr_mode]
            if gdn_trace_layer is not None:
                argv += ['--gdn-trace-layer', str(gdn_trace_layer)]
            if gdr_inputs:
                argv.append('--gdr-inputs')
            if qlinear_scales != 'source':
                argv += ['--qlinear-scales', qlinear_scales]
            if mode != 'kernel':
                argv += ['--cache-indexing', mode]
            if region['kind'] in ('rope', 'full_attention_layer') and rope_precision != 'source':
                argv += ['--rope-precision', rope_precision]
            if region['kind'] == 'full_attention_layer':
                argv += ['--full-attention-reference', full_attention_reference]
            if torchair_pb2:
                argv += ['--torchair-pb2', str(Path(torchair_pb2).resolve())]
            if target_config:
                argv += ['--target-config', str(Path(target_config).resolve())]
            print('export', name, flush=True)
            log = root/(name+'.log')
            row['log'] = log.name
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
                if ((region['kind'] == 'qlinear' and qlinear_scales == 'fp16') or region['kind'] in ('gdn_layer', 'full_attention_layer') or 'gdr_valid_rows' in region or mode != 'kernel'
                        or (region['kind'] == 'rope' and rope_precision == 'fp32')):
                    if not receipt.get('candidate') or receipt.get('candidate_graph_check', {}).get('status') != 'PASSED':
                        raise ValueError('candidate export is missing its source AIR checks')
                row.update(status='EXPORTED', export=identity(folder/'export.json', root))
            except Exception as exc:
                row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
                # The worker retains the real tracing/conversion error. Carry
                # it into the batch instead of showing only its exit status.
                try:
                    failure = json.loads((folder/'export.json').read_text())
                except (OSError, ValueError):
                    failure = {}
                if failure.get('status') == 'FAILED' and failure.get('error'):
                    row['worker_error'] = row['error']
                    row['error'] = failure['error']
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
    from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
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
            row.update(source_status=case['status'], source_error=case.get('error'),
                       source_log=os.path.relpath(root/(case['id']+'.log'), output))
            continue
        try:
            exported = json.loads(checked(root, case['export']).read_text())
            folder = root/'models'/case['id']
            air = checked(folder, exported['air'])
            destination = output/'models'/case['id']/'omc'
            stream_weights = case.get('kind') == 'segment_prefill' and not diagnostic
            print('compile', case['id'], flush=True)
            compiled = compile_existing_air(air, destination, ddk, ddk_id=ddk_id,
                timeout_seconds=timeout, save_weights_as_external_data=diagnostic or stream_weights,
                limit_partition_size=False if diagnostic else None, compiler_profile=compiler_profile,
                use_origin_format=use_origin_format, tbe_diagnosis=tbe_diagnosis)
            if compiled['status'] != 'COMPILED':
                raise RuntimeError(compiled.get('error', 'OMG failed; inspect retained omc/report.json'))
            model = destination/compiled['omc']['path']
            if stream_weights:
                row.update(stage='pack-weights', compiler_status='COMPILED')
                print('pack weights', case['id'], flush=True)
                sidecars = [dict(item, path=str(destination/item['path']))
                            for item in compiled['external_data_candidates'] if Path(item['path']).suffix == '.weight']
                packed_model = destination.parent/'model.omc'
                packing = embed_external_weights(model, packed_model,
                    ddk/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so',
                    expected_source_sha256=compiled['omc']['sha256'], sidecars=sidecars)
                receipt = destination.parent/'weight-packing.json'
                save_json(receipt, packing)
                model_record = identity(packed_model, output)
                if any(model_record[k] != packing['output'][k] for k in ('sha256', 'bytes')):
                    raise ValueError('packed model differs from weight-packing receipt')
                row.update(stage='complete', weight_storage='embedded-stream', weight_packing=identity(receipt, output))
            else:
                model_record = identity(model, output)
            row.update(status='COMPILED', omc=model_record)
        except Exception as exc:
            row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        save_json(output/'compile.json', report)
    report['status'] = 'COMPILED' if all(report['cases'][name]['status'] == 'COMPILED' for name in selected) else 'PARTIAL'
    save_json(output/'compile.json', report)
    return report


def bind_compiled(document, source_inputs, golden, enums, output_contract=None):
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
    goldens = golden if isinstance(golden, list) else [golden]
    if len(goldens) > 1 and (not output_contract or output_contract.get('order') != [g['name'] for g in goldens]
            or len([e for e in output_contract.get('source_edges', []) if not e.endswith(':-1')]) != len(goldens)):
        raise ValueError('multi-output graph requires its explicit source tuple/NetOutput order')
    if len(nets) != 1 or len(nets[0].get('input_desc', [])) != len(goldens):
        raise ValueError('compiled output count differs from explicit captured output order')
    outputs = []
    for index, item in enumerate(goldens):
        observed = _observed_desc(nets[0]['input_desc'][index], enums)
        if observed['shape'] != item['shape'] or observed['dtype'] not in DTYPES:
            raise ValueError(f'compiled output {index} shape or dtype is incompatible with captured boundary')
        required = (output_contract or {}).get('required_dtypes', {})
        dtype = required.get(str(index), required.get(index))
        if dtype is not None and observed['dtype'] != dtype:
            raise ValueError(f'compiled output {index} must remain {dtype}, observed {observed["dtype"]}')
        outputs.append(dict(item, index=index, semantic_name=item.get('name', f'output_{index}'), dtype=observed['dtype'],
            golden_dtype=item['dtype'], bytes=math.prod(observed['shape'])*np.dtype(DTYPES[observed['dtype']]).itemsize))
    return dict(inputs=inputs, outputs=outputs, source_dtype_preserved=all(s['dtype'] == s['golden_dtype'] for s in outputs),
                input_binding='SOURCE_ARGUMENT_NAME_THEN_COMPILED_GRAPHDEF_INPUT_ORDER',
                output_binding=('EXPLICIT_SOURCE_TUPLE_THEN_COMPILED_NETOUTPUT_SLOT_ORDER' if len(outputs) > 1
                                else 'SOLE_COMPILED_OUTPUT_NO_SHAPE_BASED_SLOT_GUESSING'))


def check_io_budget(name, io, max_slots=8):
    sizes = {d: sum(s['bytes'] for s in io[d]) for d in ('inputs', 'outputs')}
    total = sum(sizes.values())
    if total > 32 << 20 or any(len(io[d]) > max_slots for d in sizes):
        raise ValueError(f'{name}: probe IO requires {total} bytes '
                         f'(inputs={sizes["inputs"]}, outputs={sizes["outputs"]}); '
                         f'limit=33554432 bytes (32 MiB), at most {max_slots} slots per direction')


def _external_prefill_files(root, case, compiled):
    from qwen35_moe_dflash.toolchain.existing_air import select_primary_omc
    name = case['id']
    if case['kind'] != 'segment_prefill' or compiled.get('weight_storage') != 'embedded-stream':
        raise ValueError('external weights require a segment prefill batch with a retained packing receipt')
    packing = json.loads(checked(root, compiled['weight_packing']).read_text())
    if (packing.get('status') != 'EMBEDDED' or packing.get('artifact_kind') != 'qwen35-omc-embedded-stream'
            or any(packing['output'][k] != compiled['omc'][k] for k in ('bytes', 'sha256'))):
        raise ValueError('external weight receipt differs from the previously packed model')
    compiler_dir = root/'models'/name/'omc'
    raw = select_primary_omc(compiler_dir, required=True)
    model = checked(compiler_dir, dict(packing['source'], path=str(raw.relative_to(compiler_dir))))
    weights = []
    for row in packing['weights']:
        # Packing already verified the graph owner and the exact weight range.
        # Resolve beside the retained model so moving the batch remains supported.
        filename = row['graphop_name'] + '.weight'
        if Path(row['path']).name != filename:
            raise ValueError('invalid external GraphOp weight filename')
        path = checked(model.parent, dict(row, path=filename))
        weights.append(dict(local=str(path), remote=name+'/'+filename, bytes=row['bytes'], sha256=row['sha256']))
    if not weights or len({r['remote'] for r in weights}) != len(weights):
        raise ValueError('missing or duplicate external GraphOp weight')
    return model, weights


def prepare_fixture(root, out, ddk, ddk_id, selection=None, *, external_weights=False):
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
        build = compiled['cases'][name]
        if external_weights:
            model, weights = _external_prefill_files(root, case, build)
        else:
            model = checked(root, build['omc'])
        partitions = read_container(model)['partitions']
        parts = [p for p in partitions if p['type'] == 0]
        if external_weights and ([p['type'] for p in partitions] != [0, 1, 2] or partitions[1]['bytes'] != 0):
            raise ValueError('original external prefill OMC must have an empty weight partition')
        if len(parts) != 1 or (not external_weights and any(model.parent.glob('*.weight'))):
            raise ValueError('requires embedded weights and one main metadata partition')
        doc = schema.decode(read_partition(model, parts[0]), 'ModelDef')
        try:
            io = bind_compiled(doc, exported['source_inputs'], exported.get('goldens', exported['golden']), enums,
                               exported.get('output_contract'))
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from exc
        check_io_budget(name, io, max_slots=16 if case['kind'] in ('segment_prefill', 'segment_gdn_trace') else 8)
        fixture = out/'fixture'/name
        fixture.mkdir(parents=True)
        plan = ['QWEN35_TENSOR_PROBE 1', 'iterations 3 10', 'input_reset copy-each-call']
        if external_weights:
            plan.append('model_load file-v2-external')
        for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
            for spec in io[direction]:
                source = checked(folder, exported.get('goldens', [exported['golden']])[spec['index']]
                                 if direction == 'outputs' else spec)
                destination = fixture/f'{prefix}_{spec["index"]}.bin'
                shutil.copyfile(source, destination)
                spec['fixture_sha256'] = sha(destination)
                spec['fixture_bytes'] = destination.stat().st_size
                spec['name'] = prefix + '_' + str(spec['index'])
                plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'], ','.join(map(str, spec['shape'])))))
        (fixture/'plan.txt').write_text('\n'.join(plan)+'\n')
        observed = _compiled_observation(model, ddk)
        state_contract = None
        if case['kind'] == 'gdn_layer':
            from .module_contracts import compiled_state_contract
            required = exported['output_contract']['required_dtypes']
            state_contract = compiled_state_contract(observed, state_dtype=required.get('2', required.get(2)))
        cases.append(dict(case=name, **io, layer=case['layer'], call=case['call'], point=case['point'],
            kind=case['kind'], phase=case['phase'], ordinal=case['ordinal'], output_name=case['output_name'],
            model=_file_record(model, name+'/model.omc'), graphop_counts=observed['graphop_counts']))
        if external_weights:
            cases[-1].update(model_load='file-v2-external', external_weights=weights)
        if state_contract:
            cases[-1]['compiled_state_contract'] = state_contract
            cases[-1]['gdr_mode'] = exported.get('candidate', {}).get('gdr_mode', 'chunk')
            cases[-1]['convstate_mode'] = exported.get('candidate', {}).get('convstate_mode', 'kernel')
        if case['kind'] == 'segment_gdn_trace':
            from ..prefill_t64.segment_gdn_trace import BASIS, TAPS
            if (exported['reference_basis'] != BASIS or exported['capture_sha256'] != manifest['capture_sha256']
                    or [s['semantic_name'] for s in io['outputs']] != ['golden_hidden', *TAPS]
                    or (exported['physical_rows'], exported['valid_rows'], exported['reference_position']) != (64, 19, 18)):
                raise ValueError('Unexpected chunk GDR trace outputs or reference')
            cases[-1]['reference_basis'] = BASIS
        if case['kind'] == 'segment_module_hidden':
            from ..prefill_t64.segment_export import MODULE_BASIS
            if (exported['reference_basis'] != MODULE_BASIS or exported['capture_sha256'] != manifest['capture_sha256']
                    or [s['semantic_name'] for s in io['outputs']] != ['golden_hidden']
                    or (exported['physical_rows'], exported['valid_rows'], exported['reference_position']) != (64, 19, 18)):
                raise ValueError('Unexpected valid19 standalone module reference')
            cases[-1]['reference_basis'] = MODULE_BASIS
        if case['kind'] == 'segment_prefill':
            from ..prefill_t64.segment_export import OUTPUTS
            if [s['semantic_name'] for s in io['outputs']] != list(OUTPUTS):
                raise ValueError('Expected the four-layer logits and eight ordered states')
            cases[-1]['reference_basis'] = exported['reference_basis']
            cpu_outputs = {}
            for spec in exported['cpu_outputs']:
                target = fixture / ('cpu_' + spec['name'] + '.bin')
                shutil.copyfile(checked(folder, spec), target)
                cpu_outputs[spec['name']] = dict(spec, **identity(target, out / 'fixture'))
            cases[-1]['cpu_outputs'] = cpu_outputs
        if 'attention_contract' in exported:
            cases[-1]['attention_contract'] = exported['attention_contract']
        if case['kind'] == 'gdn' and 'reference_basis' in exported:
            cases[-1].update(reference_basis=exported['reference_basis'], gdr_mode=exported['candidate']['gdr_mode'])
        if case['kind'] == 'full_attention_layer':
            from .whole_attention import reference_scope
            captured = exported.get('captured_goldens', [])
            if len(captured) != 3 or len(io['outputs']) != 3:
                raise ValueError('FullAttention requires primary and original three-output references')
            copied = []
            for i, spec in enumerate(captured):
                destination = fixture/f'captured_golden_{i}.bin'
                shutil.copyfile(checked(folder, spec), destination)
                copied.append(dict(spec, **identity(destination, out/'fixture')))
            cases[-1].update(captured_goldens=copied,
                reference_basis=exported['reference_basis'],
                reference_scope=reference_scope(exported['reference_basis']),
                full_attention_contract=exported['full_attention_contract'])
        if case['kind'] == 'rope' and 'rope_contract' in exported:
            contract = exported['rope_contract']
            cases[-1]['rope_contract'] = {k: contract[k] for k in (
                'precision', 'rotary_dim', 'output_dtype', 'coefficient_scope')}
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
        row['reference_basis'] = case.get('reference_basis', 'ORIGINAL_CAPTURE_UNCHANGED')
        if 'reference_scope' in case:
            row['reference_scope'] = case['reference_scope']
        try:
            folder = native/case['case']
            receipt = json.loads((folder/'result.json').read_text())
            if case.get('model_load', 'legacy-buffer') != receipt.get('model_load', 'legacy-buffer'):
                raise ValueError('runner model loading route differs from fixture')
            if (receipt.get('status') != 'EXECUTED' or receipt.get('evidence_kind') != ('host-fake-hiai' if allow_host_fake else 'device-hiai')
                    or receipt.get('repetitions') != meta['repetitions'] or receipt.get('warmup') != meta['warmup']
                    or len(receipt.get('api_us', [])) != meta['repetitions']
                    or receipt.get('input_reset') != 'copy-each-call'):
                raise ValueError('runner did not complete the declared device invocations')
            for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
                for s in case[direction]:
                    checked(fixture, dict(path=f'{case["case"]}/{prefix}_{s["index"]}.bin', bytes=s['fixture_bytes'], sha256=s['fixture_sha256']))
            total_bytes = sum(s['bytes'] for s in case['outputs'])
            output_rows, offset = [], 0
            for spec in case['outputs']:
                output_row = dict(row)
                if len(case['outputs']) > 1:
                    output_row.update(output_index=spec['index'], output_name=spec['semantic_name'])
                golden = np.fromfile(fixture/case['case']/f'golden_{spec["index"]}.bin', dtype=DTYPES[spec['golden_dtype']]).reshape(spec['shape'])
                captured_golden = None
                if case['kind'] == 'full_attention_layer':
                    captured_golden = read_array(fixture, case['captured_goldens'][spec['index']])
                attention = case.get('attention_contract', {})
                valid_rows = attention.get('audited_valid_query_rows')
                if valid_rows is not None:
                    from .attention_reference import unpack
                    if (case['kind'] != 'attention' or spec['shape'] != attention['query_shape']
                            or spec['dtype'] != spec['golden_dtype'] or not 0 < valid_rows <= golden.shape[2]
                            or attention['all_seq_lengths_q'] != [valid_rows]):
                        raise ValueError('invalid active query range or output ABI for the audited attention golden')
                    golden = unpack(golden, attention['num_heads'])[:, :, :valid_rows]
                    output_row.update(reference_basis='CPU_CHECKED_310P_SEGMENT_ATTENTION_VALID_ROWS',
                                      valid_query_rows=valid_rows, physical_query_rows=spec['shape'][2])
                cpu_reference = None
                if case['kind'] == 'segment_prefill' and spec['semantic_name'] in case['cpu_outputs']:
                    cpu_reference = read_array(fixture, case['cpu_outputs'][spec['semantic_name']])
                tensors, hashes = [], []
                valid_hashes = []
                for index in range(meta['repetitions']):
                    path = folder/f'{index}.bin'
                    if path.stat().st_size != total_bytes:
                        raise ValueError('runtime output byte extent differs from compiled ABI')
                    actual = np.fromfile(path, dtype=DTYPES[spec['dtype']], count=math.prod(spec['shape']), offset=offset).reshape(spec['shape'])
                    hashes.append(hashlib.sha256(actual.tobytes()).hexdigest())
                    if valid_rows is not None:
                        actual = unpack(actual, attention['num_heads'])[:, :, :valid_rows]
                        valid_hashes.append(hashlib.sha256(actual.tobytes()).hexdigest())
                    if case['kind'] == 'segment_module_hidden' or (case['kind'] == 'segment_gdn_trace' and spec['index'] == 0):
                        from ..prefill_t64.segment_export import score_module_hidden
                        metric = score_module_hidden(actual, golden, atol, rtol)
                        valid_hashes.append(hashlib.sha256(actual[:, :19].tobytes()).hexdigest())
                    elif case['kind'] == 'segment_gdn_trace':
                        metric = metrics(actual, golden, atol=atol, rtol=rtol)
                        last = metrics(actual[:, 18], golden[:, 18], atol=atol, rtol=rtol)
                        metric['token_18'] = last
                        if not last['passed']:
                            metric['passed'] = False
                            metric['failed_metrics'].extend('token_18_' + k for k in last['failed_metrics'])
                    elif case['kind'] == 'segment_prefill':
                        from ..prefill_t64.segment_export import compare_segment_output
                        metric = compare_segment_output(spec['semantic_name'], actual, golden, cpu_reference, atol, rtol)
                        if 'valid_rows' in metric:
                            active = actual[0, :, :metric['valid_rows'], :]
                            valid_hashes.append(hashlib.sha256(active.tobytes()).hexdigest())
                    else:
                        metric = metrics(actual, golden, atol=atol, rtol=rtol)
                    if valid_rows is not None:
                        metric['last_valid_query'] = metrics(actual[:, :, -1], golden[:, :, -1], atol=atol, rtol=rtol)
                    if captured_golden is not None:
                        metric['original_capture_comparison'] = metrics(actual, captured_golden, atol=atol, rtol=rtol)
                        metric['original_capture_role'] = (
                            'SAME_AS_PRIMARY_AUDITED_CAPTURE'
                            if case['reference_basis'] == 'ORIGINAL_FULL_ATTENTION_CAPTURE_WITH_AUDITED_CORE'
                            else 'OBSERVATION_ONLY; DOES_NOT_CONTROL_PRIMARY_VERDICT')
                        if spec['index'] in (1, 2):
                            from .full_attention_compare import cache_partitions
                            contract = case['full_attention_contract']
                            fixture_index = contract['cache_input_fixture_indices'][str(spec['index'])]
                            previous = next(s for s in case['inputs'] if s['fixture_index'] == fixture_index)
                            before = np.fromfile(fixture/case['case']/f'input_{previous["index"]}.bin',
                                                 dtype=DTYPES[previous['dtype']]).reshape(previous['shape'])
                            partitions = cache_partitions(actual, golden, before, contract, atol=atol, rtol=rtol)
                            metric['full_attention_cache'] = partitions
                            metric['passed'] = metric['passed'] and partitions['passed']
                            if not partitions['active_metrics']['passed']:
                                metric.setdefault('failed_metrics', []).append('active_cache_metrics')
                            if not partitions['untouched_equal_input_bits']:
                                metric.setdefault('failed_metrics', []).append('untouched_cache_bits')
                    if case['kind'] == 'rope' and 'rope_contract' in case:
                        from .rope import partition_metrics
                        metric['rope_partitions'] = partition_metrics(actual, golden, case['rope_contract']['rotary_dim'])
                    if (case['kind'] == 'gdn_layer' and case.get('convstate_mode') == 'reference'
                            and case['phase'] == 'decode' and spec['index'] == 1):
                        previous = next(s for s in case['inputs'] if s['fixture_index'] == 1)
                        before = np.fromfile(fixture/case['case']/f'input_{previous["index"]}.bin',
                                             dtype=DTYPES[previous['dtype']]).reshape(previous['shape'])
                        if actual.ndim != 3 or before.shape != actual.shape or actual.shape[-1] < 1:
                            raise ValueError('reference ConvState Decode history extent is invalid')
                        preserved = (actual.dtype == before.dtype and
                                     actual[..., :-1].tobytes() == before[..., 1:].tobytes())
                        metric['convstate_preserved_history_equal_bits'] = preserved
                        metric['passed'] = metric['passed'] and preserved
                        if not preserved:
                            metric.setdefault('failed_metrics', []).append('convstate_preserved_history_bits')
                    if case['kind'] == 'cache':
                        identical = actual.dtype == golden.dtype and actual.tobytes() == golden.tobytes()
                        metric.update(gate='EXACT_WHOLE_CACHE_BITS', passed=identical and metric['passed'])
                        source_spec = next(s for s in case['inputs'] if s['fixture_index'] == 0)
                        before = np.fromfile(fixture/case['case']/f'input_{source_spec["index"]}.bin', dtype=DTYPES[source_spec['dtype']]).reshape(golden.shape)
                        # Classify preserved positions using the executed golden, not an assumed scatter ABI.
                        if before.dtype != golden.dtype:
                            raise ValueError('cache golden dtype differs from its input boundary')
                        bits = np.dtype(f'u{golden.dtype.itemsize}')
                        preserved = before.view(bits) == golden.view(bits)
                        changed = (actual.view(bits) != golden.view(bits)
                                   if actual.dtype == golden.dtype else actual != golden)
                        metric.update(mismatched_preserved_elements=int((changed & preserved).sum()),
                                      mismatched_updated_elements=int((changed & ~preserved).sum()))
                    tensors.append(metric)
                stable = len(set(valid_hashes or hashes)) == 1
                if valid_hashes:
                    output_row.update(valid_output_sha256=valid_hashes, full_output_stable=len(set(hashes)) == 1)
                finite = all(r['nonfinite_actual'] == r['nonfinite_golden'] == 0 for r in tensors)
                passed = all(r['passed'] is True for r in tensors)
                failed = not stable or not finite or any(r['passed'] is False for r in tensors)
                output_row.update(status='FAILED' if failed else 'PASSED' if passed else 'METRICS_RECORDED',
                    actual_dtype=spec['dtype'], golden_dtype=spec['golden_dtype'],
                    stable=stable, source_dtype_preserved=spec['dtype'] == spec['golden_dtype'], invocations=tensors,
                    output_sha256=hashes, api_us=receipt['api_us'], graphop_counts=case['graphop_counts'])
                output_rows.append(output_row)
                offset += spec['bytes']
            rows.extend(output_rows)
            continue
        except Exception as exc:
            row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    rows.sort(key=lambda r: (r['call'], r['layer'], POINT_ORDER.get(r['point'], 1000), r['ordinal'], r['case']))
    failed = [r for r in rows if r['status'] == 'FAILED']
    status = 'FAILED' if failed else 'PASSED' if rows and all(r['status']=='PASSED' for r in rows) else 'METRICS_RECORDED'
    validated = list(dict.fromkeys(r['case'] for r in rows))
    complete = not meta.get('uncompiled_cases') and len(validated) == len(meta['cases'])
    return dict(status=status if complete or status == 'FAILED' else 'PARTIAL',
        numerical_validation=status, coverage_complete=complete, validated_cases=validated,
        cases=rows, first_failed_boundary=failed[0]['case'] if failed else None,
        uncompiled_cases=meta.get('uncompiled_cases', []), all_model_layers_passed=False,
        reference=meta.get('reference', {}),
        scope='CPro boundary replay with 310P inputs; isolated failures are not proof of full-graph root cause',
        first_failure_order='CALL_LAYER_CPRO_DEPENDENCY_ORDER; INDEPENDENT_BRANCHES_HAVE_NO_UNIQUE_FIRST_CAUSE',
        cpro_continuous_state_feedback='NOT_TESTED', numerical_policy=POLICY,
        thresholds=dict(metrics=THRESHOLDS, atol=atol, rtol=rtol))

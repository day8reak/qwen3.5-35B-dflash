"""Export measured first-prefill tensors and compare CPro to the same prefill on 310P."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import identity, metrics, read_array, save_json, sha
from .decode_cache_trace import _record
from .gdr_gamma_readout import verify_index_input
from .gdr_stage_data import DESCRIPTIONS, GROUPS, STATIC, case_id, logical
from .gdr_stage_summary import check_measured_product, write_inputs, write_overview
from .segment_trace_compare import board_case


def _compiled_air(group):
    build_file = group / 'omc/compile.json'
    build = json.loads(build_file.read_text())
    fixture = json.loads((group / 'execution/fixture/manifest.json').read_text())
    air = group / 'omc' / build['source_batch']
    if (fixture['source_evidence']['compile_sha256'] != sha(build_file)
            or build['manifest_sha256'] != sha(air / 'manifest.json')):
        raise ValueError('Recovered execution, compilation and source AIR differ: ' + str(group))
    return air


def _score(rows, layer, stage, label, scope, a, ref, *, reproduced):
    row = metrics(a, ref)
    match = a.dtype == ref.dtype and a.tobytes() == ref.tobytes()
    ref_subnormals = None
    if ref.dtype == np.float16:
        fp16_subnormal = (ref != 0) & (np.abs(ref) < np.finfo(np.float16).tiny)
        ref_subnormals = int(np.count_nonzero(fp16_subnormal & (a == 0)))
    rows.append(dict(layer=layer, stage=stage, comparison=label, scope=scope,
                     actual_dtype=a.dtype.name, reference_dtype=ref.dtype.name,
                     baseline_hidden_reproduced=reproduced, relative_l2=row.get('relative_l2'),
                     cosine=row.get('cosine'), rmse=row.get('rmse'), max_abs=row.get('max_abs'),
                     mean_relative_error=row.get('mean_relative_error'), kl_divergence=row.get('kl_divergence'),
                     actual_zeros=int(np.count_nonzero(a == 0)), reference_zeros=int(np.count_nonzero(ref == 0)),
                     reference_fp16_subnormals_with_actual_zero=ref_subnormals,
                     nonfinite_actual=row['nonfinite_actual'], nonfinite_reference=row['nonfinite_golden'],
                     equal_bits=match, passed=row['passed'], failed_metrics=','.join(row.get('failed_metrics', []))))
    if a.dtype.kind == 'f':
        def fmt(v):
            return 'undefined' if v is None else f'{v:.6g}'
        return fmt(row['relative_l2']) + '/' + fmt(row['cosine'])
    return 'equal' if row['passed'] else 'different'


def _check_silu_pair(rows, layer, graph_info, board, reference):
    for backend, tensors in [('cpro', board), ('310p', reference)]:
        z = logical(tensors['silu_input'], 'silu_input').astype(np.float64)
        formula = z * np.exp(-np.logaddexp(0., -z))
        measured = logical(tensors['silu_output'], 'silu_output')
        for scope, idx in [('valid19', slice(0, 19)), ('token18', 18)]:
            score = _score(rows, layer, 'silu_output', backend + '_vs_silu_of_own_input', scope,
                           measured[:, idx], formula[:, idx],
                           reproduced=graph_info['valid_hidden_equal_bits'] if backend == 'cpro' else '')
            print(f'{backend} measured SiLU / SiLU({backend} input)  {scope:8s}  {score}')
    failed = [dict(stage=r['stage'], comparison=r['comparison'], scope=r['scope'], failed_metrics=r['failed_metrics'])
              for r in rows if not r['passed'] and r['stage'] != 'mul_rhs']
    status = 'PASSED' if not failed else 'NUMERICAL_FAILED'
    if not (graph_info['valid_hidden_equal_bits'] and graph_info['module_inputs_equal_baseline']):
        status = 'READOUT_NOT_REPRESENTATIVE'
    return dict(status=status, layer=layer, valid_hidden_equal_bits=graph_info['valid_hidden_equal_bits'],
                module_inputs_equal_baseline=graph_info['module_inputs_equal_baseline'], failed_checks=failed)


def _groups(layer, baseline, trace, trace_batch, silu_readout):
    split = 'split_norm_layers' in trace_batch
    def load(case, batch):
        run, air = batch
        _, _, _, record = _record(air, case)
        return case, run, record

    if trace_batch.get('paired_silu'):
        name = case_id(layer, 'norm_silu_pair')
        case, run, record = load(name, trace)
        aliases = {'silu_input': 'silu_input', 'silu_output': 'silu_output', 'mul_rhs': 'silu_output'}
        if (trace_batch['indexed_silu_layer'] != layer or [c['id'] for c in trace_batch['cases']] != [name]
                or record['gdr_stages']['aliases'] != aliases
                or record['output_contract']['order'] != ['golden_hidden', 'silu_input', 'silu_output']
                or not record['output_contract'].get('silu_output_readout')):
            raise ValueError('Expected one indexed SiLU input/output pair and the original hidden')
        yield 'norm_silu_pair', aliases, case, run, record
        return

    yield 'gdr', GROUPS['gdr'], *load(case_id(layer, 'gdr'), baseline if split else trace)
    if not split:
        yield 'norm', GROUPS['norm'], *load(case_id(layer, 'norm'), trace)
        return
    seen = set()
    for case in trace_batch['cases']:
        if case['layer'] != layer:
            continue
        name = case['id']
        _, run, record = load(name, trace)
        if silu_readout and name == silu_readout[0]:
            _, run, replacement = load(name, silu_readout[1])
            if (not replacement['output_contract'].get('silu_output_readout') or
                    any(replacement['gdr_stages'][k] != record['gdr_stages'][k]
                        for k in ('aliases', 'edges', 'capture', 'source_air'))):
                raise ValueError(f'{name}: indexed readout does not observe the same source operands')
            record = replacement
        stages = record['gdr_stages']['aliases']
        if (record['gdr_stages']['group'] != 'norm' or len(record['goldens']) != 2
                or set(stages.values()) != {record['goldens'][1]['name']} or seen.intersection(stages)):
            raise ValueError(f'{name}: expected one distinct norm probe output and original hidden')
        seen.update(stages)
        yield 'norm_' + record['goldens'][1]['name'], stages, name, run, record
    if seen != set(GROUPS['norm']):
        raise ValueError(f'L{layer}: incomplete norm operand coverage')


def _combined_groups(layer, base, split, split_manifest, paired, paired_manifest, orig, source):
    pair, = _groups(layer, base, paired, paired_manifest, None)
    edges = pair[-1]['gdr_stages']['edges']
    replaced = set()
    for group in _groups(layer, base, split, split_manifest, None):
        roles = set(group[1])
        if roles & edges.keys():
            old = group[-1]['gdr_stages']['edges']
            if not roles <= edges.keys() or any(old[r] != edges[r] for r in roles):
                raise ValueError('Split and paired probes observe different SiLU operands')
            replaced.update(roles)
        else:
            yield group
    if replaced != set(edges):
        raise ValueError('Incomplete paired SiLU replacement')
    yield pair
    role = 'golden_hidden'
    edge, = [e for e in source['output_contract']['source_edges'] if not e.endswith(':-1')]
    meta = dict(aliases={role: role}, edges={role: edge}, capture=source['capture_sha256'],
                source_air=source['air']['sha256'], air_dtypes={}, air_rms_epsilon=None)
    yield 'module_hidden', (role,), orig, base[0], dict(source, gdr_stages=meta,
        output_contract=dict(source['output_contract'], source_edges=[edge]))


def compare(baseline, reference, trace, output, *, silu_readout=None, combined=False):
    target = output_path(output)
    if target.exists():
        raise FileExistsError(target)
    capture = json.loads((reference / 'manifest.json').read_text())
    source_air, trace_air = _compiled_air(baseline), _compiled_air(trace)
    trace_batch = json.loads((trace_air / 'manifest.json').read_text())
    alternate = None
    pair = trace_batch.get('paired_silu', False)
    if pair and trace_batch.get('split_norm_layers') != [trace_batch['indexed_silu_layer']]:
        raise ValueError('Expected a paired readout for exactly one layer')
    if 'indexed_silu_layer' in trace_batch and not pair:
        raise ValueError('Use the full split batch as --trace and the indexed SiLU case as --silu-readout')
    if combined:
        paired_air = _compiled_air(silu_readout)
        paired_manifest = json.loads((paired_air / 'manifest.json').read_text())
        if (trace_batch.get('split_norm_layers') != [0] or pair or not paired_manifest.get('paired_silu')
                or paired_manifest.get('split_norm_layers') != [0] or '0' not in capture['gdr_stages']):
            raise ValueError('Expected L0 split and paired SiLU runs with a matching L0 310P capture')
    elif silu_readout:
        if pair:
            raise ValueError('Paired SiLU uses the input and output from a single invocation')
        extra_air = _compiled_air(silu_readout)
        extra = json.loads((extra_air / 'manifest.json').read_text())
        layer = extra.get('indexed_silu_layer')
        if layer not in trace_batch.get('split_norm_layers', []):
            raise ValueError('The indexed SiLU graph must belong to a selected split layer')
        name = case_id(layer, 'norm_silu_output')
        if [c['id'] for c in extra['cases']] != [name] or name not in {c['id'] for c in trace_batch['cases']}:
            raise ValueError('Expected a matching single SiLU case')
        alternate = (name, (silu_readout, extra_air))
    if 'split_norm_layers' in trace_batch and baseline.resolve() == trace.resolve():
        raise ValueError('Pass --baseline pointing to the original recovered nine-case batch')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        root = Path(tmp) / 'dump'
        root.mkdir()
        rows, tensor_map, layers = [], [], {}
        for i in trace_batch.get('split_norm_layers', range(3)):
            orig = f'l{i:02}_c00_gdn_layer_hidden_o0'
            _, _, _, source = _record(source_air, orig)
            base_inputs, base_refs, base, base_fixture = board_case(baseline, orig)
            native = capture['gdr_stages'][str(i)]
            info = layers[str(i)] = dict(epsilon=native['epsilon'], gdr_attributes=native['attrs'], graphs={}, tensors={})
            groups = _groups(i, (baseline, source_air), (trace, trace_air), trace_batch, alternate)
            if combined:
                write_inputs(root, i, reference, capture['attention_modules'][str(i)], baseline, orig, base_fixture, info, tensor_map, rows, _score)
                groups = _combined_groups(i, (baseline, source_air), (trace, trace_air), trace_batch,
                                          (silu_readout, paired_air), paired_manifest, orig, source)
            for group, stages, case, run, record in groups:
                meta = record['gdr_stages']
                if meta['source_air'] != source['air']['sha256'] or meta['capture'] != sha(reference / 'manifest.json'):
                    raise ValueError(f'{case}: source AIR or 310P capture changed')
                trace_inputs, refs, actual, fixture_case = (base_inputs, base_refs, base, base_fixture) if group == 'module_hidden' else board_case(run, case)
                offsets = {}
                offset = 0
                for spec in fixture_case['outputs']:
                    offsets[spec['semantic_name']] = (spec['index'], offset, spec['bytes'])
                    offset += spec['bytes']
                module_inputs = trace_inputs.copy()
                contract = record['output_contract']
                for role in ('gamma', 'silu_output'):
                    readout = contract.get(role + '_readout')
                    if readout:
                        verify_index_input(module_inputs.pop(readout['index_input'], None), role)
                for role, (dtype, shape, digest) in module_inputs.items():
                    live = read_array(reference, capture['attention_modules'][str(i)]['inputs'][role])
                    bound = live.astype(dtype, copy=False)
                    if list(bound.shape) != shape or hashlib.sha256(bound.tobytes()).hexdigest() != digest:
                        raise ValueError(f'{case}: executed {role} differs from this 310P capture')
                if base_refs['golden_hidden'][:, :19].tobytes() != refs['golden_hidden'][:, :19].tobytes():
                    raise ValueError(f'{case}: different valid hidden references')
                matched = actual['golden_hidden'][:, :19].tobytes() == base['golden_hidden'][:, :19].tobytes()
                qualified = matched and module_inputs == base_inputs
                if combined and not qualified:
                    raise ValueError(f'{case}: selected L0 readout differs from the original module inputs or valid hidden')
                info['graphs'][group] = dict(case=case, module_inputs=meta.get('module_inputs', 'BASELINE_310P_CAPTURE'),
                                            full_inputs_equal_baseline=trace_inputs == base_inputs,
                                            module_inputs_equal_baseline=module_inputs == base_inputs,
                                            gamma_readout=contract.get('gamma_readout'), silu_output_readout=contract.get('silu_output_readout'),
                                            valid_hidden_equal_bits=matched, edges=meta['edges'], aliases=meta['aliases'],
                                            air_port_dtypes=meta['air_dtypes'], air_rms_epsilon=meta['air_rms_epsilon'],
                                            source_air_sha256=meta['source_air'], trace_air_sha256=record['air']['sha256'])
                print(f'L{i} {group} valid hidden equals original CPro run: {matched}')
                if not matched:
                    print('Diagnostic outputs only: valid hidden differs from the original CPro graph.')
                print('stage                 scope     CPro/310P L2/cos')
                start_row = len(rows)
                for stage in stages:
                    key = meta['aliases'][stage]
                    native_spec = capture['attention_modules'][str(i)]['golden_hidden'] if stage == 'golden_hidden' else native['tensors'][stage]
                    values = {'310p': read_array(reference, native_spec), 'cpro': actual[key]}
                    if (group == 'norm_silu_pair' or (combined and key == stage)) and logical(values['310p'], stage).tobytes() != logical(refs[key], stage).tobytes():
                        raise ValueError(f'{case}/{stage}: fixture reference differs from the measured 310P tensor')
                    slot, board_offset, board_size = offsets[key]
                    sources = {
                        '310p': reference / native_spec['path'],
                        'cpro': run / 'recovery/native' / case / '0.bin',
                    }
                    records = info['tensors'][stage] = {}
                    for backend, raw in values.items():
                        arr = logical(raw, stage)
                        values[backend] = arr
                        board = backend == 'cpro'
                        path = root / f'l{i}' / backend / (stage + '.bin')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        arr.tofile(path)
                        records[backend] = dict(identity(path, root), dtype=arr.dtype.name, shape=list(arr.shape),
                                                original_shape=list(raw.shape))
                        step, meaning = DESCRIPTIONS[stage]
                        tensor_map.append(dict(layer=i, backend=backend, tensor=stage, step=step, meaning=meaning,
                            baseline_hidden_reproduced=matched if board else '',
                            path=path.relative_to(root).as_posix(), shape=json.dumps(list(arr.shape)), dtype=arr.dtype.name,
                            bytes=arr.nbytes, source=str(sources[backend]), source_slot=slot if board else '',
                            source_offset_bytes=board_offset if board else 0, source_length_bytes=board_size if board else raw.nbytes,
                            module_edge=meta['edges'][stage] if board else '',
                            air_edge=record['output_contract']['source_edges'][slot] if board else '', slot_name=key if board else '',
                            fixture_golden=f'{case}/golden_{slot}.bin' if board else ''))
                        if combined:
                            measurement = 'REFERENCE' if not board else (
                                'BASELINE' if stage == 'golden_hidden' else 'QUALIFIED')
                            tensor_map[-1].update(measurement=measurement, source_case=case if board else '')
                    if stage == 'effective_length' and any(x.tolist() != [19] for x in values.values()):
                        print(f'L{i} effective_length differs: ' + str({k: v.tolist() for k, v in values.items()}))
                    scopes = [('tensor', None)] if stage in STATIC else [('valid19', slice(0, 19)), ('token18', 18)]
                    for scope, selection in scopes:
                        observed, golden = values['cpro'], values['310p']
                        if selection is not None:
                            observed, golden = observed[:, selection], golden[:, selection]
                        score = _score(rows, i, stage, 'cpro_vs_310p', scope, observed, golden, reproduced=matched)
                        print(f'{stage:20s}  {scope:8s}  {score}')
                if group == 'norm_silu_pair':
                    pair_rows = rows[start_row:]
                    verdict = _check_silu_pair(pair_rows, i, info['graphs'][group], actual, refs)
                    rows[start_row:] = pair_rows
                    save_json(root / 'silu-result.json', verdict)
                    if combined and verdict['status'] != 'PASSED':
                        raise ValueError(f'L{i}: paired SiLU did not pass: {verdict}')
        if combined:
            check_measured_product(root, layers['0'], rows, _score)
            write_overview(root, rows, layers)
        for name, entries in [('summary.csv', rows), ('tensor-map.csv', tensor_map)]:
            with (root / name).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for entry in entries for k in entry)))
                writer.writeheader()
                writer.writerows(entries)
        save_json(root / 'manifest.json', dict(capture_sha256=sha(reference / 'manifest.json'),
                  physical_rows=64, valid_rows=19, last_valid_index=18, chunk_size=64,
                  input_basis='CAPTURED_310P_MODULE_ENTRY', reference_backend='310p', layers=layers))
        root.rename(target)
    share_for_transfer(target)
    print(target / 'summary.csv')
    if combined:
        print(target / 'summary.md')
    if pair:
        print('SiLU:', json.dumps(verdict, ensure_ascii=False))
        return verdict['status'] == 'PASSED'

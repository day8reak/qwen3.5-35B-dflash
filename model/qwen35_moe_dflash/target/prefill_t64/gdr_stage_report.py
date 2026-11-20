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
from .gdr_stage_data import DESCRIPTIONS, GROUPS, STATIC, case_id, logical
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


def compare(baseline, reference, trace, output):
    target = output_path(output)
    if target.exists():
        raise FileExistsError(target)
    capture = json.loads((reference / 'manifest.json').read_text())
    source_air, trace_air = _compiled_air(baseline), _compiled_air(trace)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        root = Path(tmp) / 'dump'
        root.mkdir()
        rows, tensor_map, layers = [], [], {}
        for i in range(3):
            orig = f'l{i:02}_c00_gdn_layer_hidden_o0'
            _, _, _, source = _record(source_air, orig)
            base_inputs, base_refs, base, _ = board_case(baseline, orig)
            native = capture['gdr_stages'][str(i)]
            info = layers[str(i)] = dict(epsilon=native['epsilon'], gdr_attributes=native['attrs'], graphs={}, tensors={})
            for group, stages in GROUPS.items():
                case = case_id(i, group)
                _, _, _, record = _record(trace_air, case)
                meta = record['gdr_stages']
                if meta['source_air'] != source['air']['sha256'] or meta['capture'] != sha(reference / 'manifest.json'):
                    raise ValueError(f'{case}: source AIR or 310P capture changed')
                trace_inputs, refs, actual, fixture_case = board_case(trace, case)
                offsets = {}
                offset = 0
                for spec in fixture_case['outputs']:
                    offsets[spec['semantic_name']] = (spec['index'], offset, spec['bytes'])
                    offset += spec['bytes']
                for role, (dtype, shape, digest) in trace_inputs.items():
                    live = read_array(reference, capture['attention_modules'][str(i)]['inputs'][role])
                    bound = live.astype(dtype, copy=False)
                    if list(bound.shape) != shape or hashlib.sha256(bound.tobytes()).hexdigest() != digest:
                        raise ValueError(f'{case}: executed {role} differs from this 310P capture')
                if base_refs['golden_hidden'][:, :19].tobytes() != refs['golden_hidden'][:, :19].tobytes():
                    raise ValueError(f'{case}: different valid hidden references')
                matched = actual['golden_hidden'][:, :19].tobytes() == base['golden_hidden'][:, :19].tobytes()
                info['graphs'][group] = dict(module_inputs=meta.get('module_inputs', 'BASELINE_310P_CAPTURE'),
                                            full_inputs_equal_baseline=trace_inputs == base_inputs,
                                            valid_hidden_equal_bits=matched, edges=meta['edges'], aliases=meta['aliases'],
                                            air_port_dtypes=meta['air_dtypes'], air_rms_epsilon=meta['air_rms_epsilon'],
                                            source_air_sha256=meta['source_air'], trace_air_sha256=record['air']['sha256'])
                print(f'L{i} {group} valid hidden equals original CPro run: {matched}')
                print('stage                 scope     CPro/310P L2/cos')
                for stage in stages:
                    key = meta['aliases'][stage]
                    native_spec = native['tensors'][stage]
                    values = {'310p': read_array(reference, native_spec), 'cpro': actual[key]}
                    slot, board_offset, board_size = offsets[key]
                    sources = {
                        '310p': reference / native_spec['path'],
                        'cpro': trace / 'recovery/native' / case / '0.bin',
                    }
                    records = info['tensors'][stage] = {}
                    for backend, raw in values.items():
                        arr = logical(raw, stage)
                        values[backend] = arr
                        path = root / f'l{i}' / backend / (stage + '.bin')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        arr.tofile(path)
                        records[backend] = dict(identity(path, root), dtype=arr.dtype.name, shape=list(arr.shape),
                                                original_shape=list(raw.shape))
                        board = backend == 'cpro'
                        step, meaning = DESCRIPTIONS[stage]
                        tensor_map.append(dict(layer=i, backend=backend, tensor=stage, step=step, meaning=meaning,
                            path=path.relative_to(root).as_posix(), shape=json.dumps(list(arr.shape)), dtype=arr.dtype.name,
                            bytes=arr.nbytes, source=str(sources[backend]), source_slot=slot if board else '',
                            source_offset_bytes=board_offset if board else 0, source_length_bytes=board_size if board else raw.nbytes,
                            air_edge=meta['edges'][stage] if board else '', slot_name=key if board else '',
                            fixture_golden=f'{case}/golden_{slot}.bin' if board else ''))
                    if stage == 'effective_length' and any(x.tolist() != [19] for x in values.values()):
                        print(f'L{i} effective_length differs: ' + str({k: v.tolist() for k, v in values.items()}))
                    scopes = [('tensor', None)] if stage in STATIC else [('valid19', slice(0, 19)), ('token18', 18)]
                    for scope, selection in scopes:
                        observed, golden = values['cpro'], values['310p']
                        if selection is not None:
                            observed, golden = observed[:, selection], golden[:, selection]
                        score = _score(rows, i, stage, 'cpro_vs_310p', scope, observed, golden, reproduced=matched)
                        print(f'{stage:20s}  {scope:8s}  {score}')
        for name, entries in [('summary.csv', rows), ('tensor-map.csv', tensor_map)]:
            with (root / name).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
                writer.writeheader()
                writer.writerows(entries)
        save_json(root / 'manifest.json', dict(capture_sha256=sha(reference / 'manifest.json'),
                  physical_rows=64, valid_rows=19, last_valid_index=18, chunk_size=64,
                  input_basis='CAPTURED_310P_MODULE_ENTRY', reference_backend='310p', layers=layers))
        root.rename(target)
    share_for_transfer(target)
    print(target / 'summary.csv')

"""Read retained attention extents and length evidence; never guess NZ or padding."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .common import KIND, DTYPES, checked, identity, new_output, save_json, sha
from qwen35_moe_dflash.common.artifacts import output_path


def tensor_extent(path, spec, expected=None):
    """Keep the observed byte count visible even when its descriptor is wrong."""
    dtype = np.dtype(DTYPES[spec['dtype']])
    shape = spec['shape']
    if any(type(d) is not int or d < 0 for d in shape):
        raise ValueError('requires concrete nonnegative tensor dimensions')
    size = path.stat().st_size
    result = dict(path=str(path), shape=shape, dtype=spec['dtype'],
        descriptor_numel=math.prod(shape), descriptor_bytes=math.prod(shape)*dtype.itemsize,
        observed_bytes=size, observed_numel=size//dtype.itemsize if size % dtype.itemsize == 0 else None,
        byte_count_matches=size == math.prod(shape)*dtype.itemsize, sha256=sha(path))
    if expected is not None:
        result['identity_matches'] = size == expected['bytes'] and result['sha256'] == expected['sha256']
    return result


def inspect_attention(batch, fixture, native, output, capture=None):
    root, fixture, native = (Path(p).resolve() for p in (batch, fixture, native))
    manifest = json.loads((root/'manifest.json').read_text())
    compiled = json.loads((root/'compile.json').read_text())
    source = (root/compiled.get('source_batch', '.')).resolve()
    if (manifest.get('artifact_kind') != KIND or compiled.get('artifact_kind') != KIND
            or compiled['manifest_sha256'] != sha(root/'manifest.json')
            or compiled['manifest_sha256'] != sha(source/'manifest.json')):
        raise ValueError('compile/AIR manifest binding changed')
    selected = [c for c in manifest['cases'] if c['kind'] == 'attention']
    if not selected or len({c['id'] for c in selected}) != len(selected):
        raise ValueError('requires distinct attention cases; use the batch that produced the attention metrics')
    meta = json.loads((fixture/'manifest.json').read_text())
    expected = dict(artifact_kind=KIND, manifest_sha256=sha(root/'manifest.json'),
                    compile_sha256=sha(root/'compile.json'))
    if meta.get('artifact_kind') != KIND or meta.get('source_evidence') != expected:
        raise ValueError('fixture belongs to a different AIR/OMC batch')
    fixtures = {c['case']: c for c in meta['cases']}
    if len(fixtures) != len(meta['cases']):
        raise ValueError('duplicate fixture cases')
    capture = Path(capture).resolve() if capture else None
    regions = {}
    if capture:
        if sha(capture/'manifest.json') != manifest.get('capture_sha256'):
            raise ValueError('capture manifest differs from AIR batch')
        cap = json.loads((capture/'manifest.json').read_text())
        if cap.get('artifact_kind') != KIND or cap.get('status') != 'CAPTURED':
            raise ValueError('requires completed original capture')
        regions = {r['id']: r for r in cap['regions']}
        if len(regions) != len(cap['regions']):
            raise ValueError('duplicate captured regions')
    out = output_path(output)
    for original in (root, source, fixture, native, *([capture] if capture else [])):
        if out == original or original in out.parents or out in original.parents:
            raise ValueError('inspection output must be separate from retained inputs')
    out = new_output(out)
    report = dict(artifact_kind=KIND+'-attention-extents', status='INSPECTED',
        source_evidence=expected, fixture_manifest=identity(fixture/'manifest.json', fixture),
        capture_binding='MANIFEST_HASH_MATCH' if capture else 'NOT_SUPPLIED', cases=[],
        scope='RETAINED_FILES_AND_FIXTURE_DESCRIPTORS; NO_NEW_OMC_DECODE_OR_DEVICE_EXECUTION',
        layout_equivalence='UNVERIFIED', valid_query_region='UNVERIFIED',
        padding_hypothesis='UNVERIFIED; NO_TOKEN_COUNT_ASSUMED', numerical_validation='NOT_RUN',
        model_compilation='NOT_RUN', device_execution='NOT_RUN')
    for case in selected:
        name = case['id']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid case name')
        row = dict(case=name, phase=case['phase'], layer=case['layer'],
            compilation_status=compiled['cases'].get(name, {}).get('status'), errors=[], inputs=[], dumps=[])
        report['cases'].append(row)

        def observe(label, path, spec, expected_file=None):
            value = tensor_extent(path, spec, expected_file)
            if not value['byte_count_matches'] or value.get('identity_matches') is False:
                row['errors'].append(label + ': extent or identity mismatch')
            return value

        try:
            exp = json.loads(checked(source, case['export']).read_text())
            folder = source/'models'/name
            row['captured_attributes'] = exp.get('source_contract', {}).get('captured_attributes', {})
            row['source_air_boundaries'] = exp.get('internal_boundaries', {})
            row['golden'] = observe('export golden', checked(folder, exp['golden']), exp['golden'], exp['golden'])
            region = None
            if capture:
                region = regions[case['region']]
                row['captured_sequence_length'] = region['sequence_length']
                golden = region['outputs'][case['output_index']]
                row['capture_golden'] = observe('capture golden', checked(capture, golden), golden, golden)
                if any(golden[k] != exp['golden'][k] for k in ('sha256', 'bytes', 'shape', 'dtype')):
                    raise ValueError('export golden differs from captured output')
            for spec in exp['source_inputs']:
                item = dict(source_name=spec['name'], fixture_index=spec['fixture_index'])
                path = checked(folder, spec)
                item.update(observe('export input', path, spec, spec))
                if region is not None:
                    original = region['inputs'][spec['fixture_index']]
                    if any(original[k] != spec[k] for k in ('sha256', 'bytes', 'shape', 'dtype')):
                        raise ValueError('export input differs from captured argument')
                    item['captured_name'] = original['name']
                # Show small integer controls only; do not infer which dimension is valid.
                if item['byte_count_matches'] and np.dtype(DTYPES[spec['dtype']]).kind in 'iu' and item['descriptor_numel'] <= 128:
                    item['integer_values'] = np.fromfile(path, dtype=DTYPES[spec['dtype']]).reshape(spec['shape']).tolist()
                row['inputs'].append(item)
            fx = fixtures[name]
            omc = compiled['cases'][name]['omc']
            checked(root, omc)
            if any(fx['model'][k] != omc[k] for k in ('bytes', 'sha256')):
                raise ValueError('fixture model differs from compiled OMC')
            if len(fx['outputs']) != 1:
                raise ValueError('attention fixture must have exactly one output')
            spec = fx['outputs'][0]
            row['compiled_output_recorded_in_fixture'] = {k: spec[k] for k in ('shape', 'dtype', 'bytes', 'golden_dtype')}
            golden_path = fixture/name/'golden_0.bin'
            row['fixture_golden'] = observe('fixture golden', golden_path,
                dict(shape=spec['shape'], dtype=spec['golden_dtype']),
                dict(bytes=spec['fixture_bytes'], sha256=spec['fixture_sha256']))
            if (row['fixture_golden']['sha256'] != exp['golden']['sha256']
                    or spec['shape'] != exp['golden']['shape'] or spec['golden_dtype'] != exp['golden']['dtype']):
                row['errors'].append('fixture golden differs from export')
            for inp in fx['inputs']:
                original = [s for s in exp['source_inputs'] if s['fixture_index'] == inp['fixture_index']]
                if len(original) != 1 or any(inp[k] != original[0][k] for k in ('shape', 'dtype')):
                    raise ValueError('fixture input binding differs from export')
                artifact = checked(fixture, dict(path=f'{name}/input_{inp["index"]}.bin',
                    bytes=inp['fixture_bytes'], sha256=inp['fixture_sha256']))
                if sha(artifact) != original[0]['sha256']:
                    raise ValueError('fixture input bytes differ from export')
            row['fixture_binding'] = 'MANIFEST_MODEL_AND_INPUT_HASHES_MATCH'
            try:
                receipt = json.loads((native/name/'result.json').read_text())
                row['runner_receipt'] = receipt
                if (receipt.get('status') != 'EXECUTED' or receipt.get('evidence_kind') != 'device-hiai'
                        or receipt.get('repetitions') != meta['repetitions'] or receipt.get('warmup') != meta['warmup']
                        or len(receipt.get('api_us', [])) != meta['repetitions']
                        or receipt.get('input_reset') != 'copy-each-call'):
                    row['errors'].append('runner receipt does not meet comparison requirements')
            except (OSError, ValueError) as exc:
                row['errors'].append('runner receipt: ' + str(exc))
            for index in range(meta['repetitions']):
                path = native/name/f'{index}.bin'
                try:
                    dump = observe(f'dump {index}', path, spec)
                    dump['invocation'] = index
                    dump['matches_fixture_output_bytes'] = dump['observed_bytes'] == spec['bytes']
                    if not dump['matches_fixture_output_bytes']:
                        row['errors'].append(f'dump {index}: size differs from fixture output bytes')
                    row['dumps'].append(dump)
                except (OSError, ValueError) as exc:
                    row['errors'].append(f'dump {index}: {exc}')
            row['native_provenance'] = 'LOCAL_RECEIPT_AND_FILE_HASHES; NOT_INDEPENDENT_UPLOAD_ATTESTATION'
        except (KeyError, OSError, ValueError, TypeError) as exc:
            row['errors'].append(f'{type(exc).__name__}: {exc}')
        row['status'] = 'PARTIAL' if row['errors'] else 'INSPECTED'
    if any(row['errors'] for row in report['cases']):
        report['status'] = 'PARTIAL'
    save_json(out/'report.json', report)
    lines = ['Read-only attention extent inspection; no export, compilation or device execution.',
        'No cropping, padding, NZ conversion or new accuracy verdict. Equal extents do not prove equal layout.',
        'Compiled descriptors below are retained fixture records, not a new decode of OMC.',
        'Length attributes/controls are evidence only; no valid-query mask inferred.']
    for row in report['cases']:
        lines += ['', f'{row["case"]} ({row["phase"]}): {row["status"]}']
        for key in ('golden', 'fixture_golden'):
            item = row.get(key, {})
            lines.append(f'{key}: shape={item.get("shape")} dtype={item.get("dtype")} '
                f'numel={item.get("observed_numel")} bytes={item.get("observed_bytes")}')
        lines.append('compiled_output_recorded_in_fixture=' + json.dumps(row.get('compiled_output_recorded_in_fixture')))
        lines.append('dump_bytes=' + json.dumps([d['observed_bytes'] for d in row['dumps']]))
        lines.append('captured_attributes=' + json.dumps(row.get('captured_attributes'), ensure_ascii=False))
        if 'captured_sequence_length' in row:
            lines.append('captured_sequence_length=' + str(row['captured_sequence_length']))
        for inp in row['inputs']:
            lines.append(f'input {inp.get("captured_name", inp["source_name"])}: '
                f'shape={inp["shape"]} dtype={inp["dtype"]} integer_values={inp.get("integer_values", "NOT_SAMPLED")}')
        lines.extend('ERROR: ' + e for e in row['errors'])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    return dict(status=report['status'], cases=len(selected), report=str(out/'report.json'),
                summary=str(out/'summary.txt'), numerical_validation='NOT_RUN',
                model_compilation='NOT_RUN', device_execution='NOT_RUN')

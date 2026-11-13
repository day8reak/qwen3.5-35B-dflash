"""Read retained attention extents and length evidence; never guess NZ or padding."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from .common import KIND, DTYPES, checked, identity, new_output, save_json, sha
from qwen35_moe_dflash.common.artifacts import output_path


def attention_length_constant(edge, nodes, enums, lengths):
    """Verify Const payload identity; this does not establish host GetData access."""
    from qwen35_moe_dflash.target.operator_air_review import descriptor, _dtype
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
    name, slot = edge.rsplit(':', 1)
    node = nodes.get(name, {})
    if slot != '0' or node.get('type') != 'Const':
        raise ValueError('attention actual_seq_lengths_q must be a direct INT64 Const [S], not a runtime length input')
    tensor = _attributes(node).get('value', {}).get('t', {})
    desc = descriptor(tensor.get('desc', {}), enums)
    data = tensor.get('data', {})
    expected = struct.pack('<' + 'q' * len(lengths), *lengths)
    if (_dtype(desc) != 'int64' or desc.get('shape') != [len(lengths)]
            or data.get('bytes') != len(expected)
            or data.get('sha256') != hashlib.sha256(expected).hexdigest()):
        raise ValueError('attention query length Const dtype/shape/bytes differ from captured [S]')
    return dict(edge=edge, dtype='int64', shape=[len(lengths)], values=lengths,
        data_sha256=data['sha256'], status='SOURCE_CONST_BYTES_VERIFIED', host_tiling_data='NOT_OBSERVED')


def source_call_contract(folder, exported):
    """Read old or new source AIR; no dependency on Torch/TorchAir installation."""
    from .inspection import bounded_bytes
    from qwen35_moe_dflash.toolchain.omc_wire import Schema
    from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
    from qwen35_moe_dflash.target.operator_air_review import _data_edges, port_names
    schema_path = folder/'air-schema.pb'
    schema = Schema(bounded_bytes(schema_path, 8 << 20),
        expected_file=exported['schema']['schema_file'], expected_package=exported['schema']['schema_package'])
    air = checked(folder, exported['air'])
    _, _, graph = _decode(schema, bounded_bytes(air, 64 << 20))
    nodes = {n['name']: n for n in graph['op']}
    ops = [n for n in nodes.values() if n['type'] == 'AdnFusedInferAttention']
    if len(ops) != 1:
        raise ValueError('requires one source AdnFusedInferAttention')
    op = ops[0]
    attrs = _attributes(op)
    ports = port_names(attrs, 'input')
    edges = {ports[i]: edge for i, edge in enumerate(_data_edges(op)) if i in ports}
    heads = attrs.get('num_heads', {}).get('i')
    if (attrs.get('input_layout', {}).get('s') != 'BNSD'
            or type(heads) is not int or heads < 1):
        raise ValueError('source attention layout/heads do not match packed BNSD contract')
    query = [s for s in exported['source_inputs'] if s['name'] + ':0' == edges.get('query')]
    if len(query) != 1 or nodes.get(query[0]['name'], {}).get('type') != 'Data':
        raise ValueError('source query must bind directly to one captured Data argument')
    shape = query[0]['shape']
    if (len(shape) != 4 or any(type(d) is not int or d < 1 for d in shape)
            or shape[0] != 1 or shape[3] != 16 or shape[1] % heads):
        raise ValueError('requires B1 packed query [1, heads*D/16, S, 16]')
    result = dict(status='INSPECTED', air=identity(air, folder), schema=identity(schema_path, folder),
        schema_binding='RETAINED_DESCRIPTOR; LEGACY_EXPORT_NO_DESCRIPTOR_HASH',
        query_shape=shape, query_dtype=query[0]['dtype'], num_heads=heads, head_dim=shape[1]*16//heads,
        query_seq_step=16, query_step_basis='USER_CONFIRMED_OPERATOR_SOURCE; NOT_OBSERVED_TILING',
        expected_total_tasks=heads*((shape[2]+15)//16), observed_total_tasks=None,
        attributes={k: attrs.get(k) for k in ('input_layout','inner_precise','block_size','num_key_value_heads')})
    try:
        result['query_length_constant'] = attention_length_constant(edges['actual_seq_lengths_q'], nodes,
            schema.enums[schema.package + '.DataType'], [shape[2]])
    except (ValueError, KeyError, TypeError) as exc:
        result['query_length_constant'] = dict(status='REVIEW_REQUIRED', error=str(exc))
    return result


def output_pattern(actual, golden, contract):
    """Group all output elements using the source packed layout, with no cropping.

    Zero-valued groups are observations, not evidence that a task never ran.
    Actual device layout equivalence remains an independent hypothesis.
    """
    shape = contract['query_shape']
    if list(actual.shape) != shape or list(golden.shape) != shape:
        raise ValueError('output shape differs from source query; cannot assign packed heads')
    b, _, seq, _ = shape
    heads, dim = contract['num_heads'], contract['head_dim']
    def view(value):
        return value.reshape(b, heads, dim//16, seq, 16).transpose(0, 1, 3, 2, 4).reshape(b, heads, seq, dim)
    av, gv = view(actual), view(golden)

    def stats(a, g, query_start=0):
        af, gf = a.astype(np.float64), g.astype(np.float64)
        finite = bool(np.isfinite(af).all() and np.isfinite(gf).all())
        delta = af-gf
        gn, en = (float(np.linalg.norm(gf)), float(np.linalg.norm(delta))) if finite else (None, None)
        coord = np.unravel_index(np.abs(delta).argmax(), delta.shape) if finite else None
        return dict(elements=a.size, actual_nonzero=int(np.count_nonzero(a)), golden_nonzero=int(np.count_nonzero(g)),
            exact_elements=int(np.count_nonzero(a == g)), nonfinite_actual=int((~np.isfinite(a)).sum()),
            nonfinite_golden=int((~np.isfinite(g)).sum()), max_abs=float(np.abs(delta).max()) if finite else None,
            relative_l2=en/gn if finite and gn else 0.0 if finite and en == 0 else None,
            actual_abs_max=float(np.abs(af).max()) if np.isfinite(af).all() else None,
            golden_abs_max=float(np.abs(gf).max()) if np.isfinite(gf).all() else None,
            max_error_query=query_start+int(coord[0]) if coord is not None else None,
            max_error_channel=int(coord[1]) if coord is not None else None,
            actual_at_max_error=float(af[coord]) if coord is not None else None,
            golden_at_max_error=float(gf[coord]) if coord is not None else None)

    rows, blocks, queries, locations = [], [], [], []
    step = contract['query_seq_step']
    for head in range(heads):
        rows.append(dict(head=head, **stats(av[0,head], gv[0,head])))
        locations.append(dict(head=head,
            zero_actual_query_rows=np.flatnonzero(np.all(av[0,head] == 0, axis=1)).tolist(),
            zero_golden_query_rows=np.flatnonzero(np.all(gv[0,head] == 0, axis=1)).tolist(),
            equal_value_query_rows=np.flatnonzero(np.all(av[0,head] == gv[0,head], axis=1)).tolist()))
        for start in range(0,seq,step):
            stop = min(seq,start+step)
            blocks.append(dict(head=head, query_start=start, query_stop=stop,
                **stats(av[0,head,start:stop], gv[0,head,start:stop], start)))
        for query in range(seq):
            queries.append(dict(head=head, query=query,
                **stats(av[0,head,query:query+1], gv[0,head,query:query+1], query)))
    return dict(scope='SOURCE_PACKED_HEAD_AND_QUERY_BLOCK_VIEW; NO_TILING_EXECUTION_PROOF',
        coordinate_basis='LOGICAL_QUERY_AND_CHANNEL_WITHIN_HEAD; ZERO_BASED; SOURCE_PACKED_LAYOUT',
        heads=rows, query_blocks=blocks, query_rows=queries, row_locations=locations,
        zero_actual_nonzero_golden_heads=[r['head'] for r in rows if r['actual_nonzero']==0 and r['golden_nonzero']>0])


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
        row = dict(case=name, phase=case['phase'], layer=case['layer'], output_patterns=[],
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
            try:
                row['source_call_contract'] = source_call_contract(folder, exp)
            except (KeyError, OSError, ValueError, TypeError) as exc:
                row['source_call_contract'] = dict(status='UNAVAILABLE', error=str(exc))
                row['errors'].append('source call contract: ' + str(exc))
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
            binding_valid = not row['errors']
            patterns = {}
            for index in range(meta['repetitions']):
                path = native/name/f'{index}.bin'
                try:
                    dump = observe(f'dump {index}', path, spec)
                    dump['invocation'] = index
                    dump['matches_fixture_output_bytes'] = dump['observed_bytes'] == spec['bytes']
                    if not dump['matches_fixture_output_bytes']:
                        row['errors'].append(f'dump {index}: size differs from fixture output bytes')
                    row['dumps'].append(dump)
                    if binding_valid and dump['byte_count_matches'] and dump['matches_fixture_output_bytes']:
                        digest = dump['sha256']
                        if digest not in patterns:
                            actual = np.fromfile(path, dtype=DTYPES[spec['dtype']]).reshape(spec['shape'])
                            golden = np.fromfile(golden_path, dtype=DTYPES[spec['golden_dtype']]).reshape(spec['shape'])
                            pattern = output_pattern(actual, golden, row['source_call_contract'])
                            pattern.update(output_sha256=digest, invocations=[])
                            patterns[digest] = pattern
                            row['output_patterns'].append(pattern)
                        patterns[digest]['invocations'].append(index)
                except (OSError, ValueError) as exc:
                    row['errors'].append(f'dump {index}: {exc}')
            row['native_provenance'] = 'LOCAL_RECEIPT_AND_FILE_HASHES; NOT_INDEPENDENT_UPLOAD_ATTESTATION'
        except (KeyError, OSError, ValueError, TypeError) as exc:
            row['errors'].append(f'{type(exc).__name__}: {exc}')
        row['status'] = 'PARTIAL' if row['errors'] else 'INSPECTED'
    if any(row['errors'] for row in report['cases']):
        report['status'] = 'PARTIAL'
    csv_paths = {}
    for group, filename, prefix in (('heads','head-summary.csv',['head']),
                                    ('query_blocks','query-blocks.csv',['head','query_start','query_stop']),
                                    ('query_rows','query-rows.csv',['head','query'])):
        path = out/filename
        fields = ['case','phase','invocations'] + prefix + ['elements','actual_nonzero','golden_nonzero',
            'exact_elements','nonfinite_actual','nonfinite_golden','max_abs','relative_l2',
            'actual_abs_max','golden_abs_max','max_error_query','max_error_channel',
            'actual_at_max_error','golden_at_max_error']
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in report['cases']:
                for pattern in row['output_patterns']:
                    for item in pattern[group]:
                        writer.writerow(dict(case=row['case'],phase=row['phase'],
                            invocations=';'.join(map(str,pattern['invocations'])),**item))
        csv_paths[group] = str(path)
    report['output_pattern_csv'] = csv_paths
    save_json(out/'report.json', report)
    lines = ['Read-only attention extent inspection; no export, compilation or device execution.',
        'No cropping, padding or new accuracy verdict. Head/block statistics use the SOURCE packed layout.',
        'Query/channel coordinates are zero-based within each head; per-row statistics cover every element.',
        'Zero output does not prove a task was skipped. Actual tiling and device layout remain unverified.',
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
        contract = row.get('source_call_contract', {})
        lines.append('source_call_contract=' + json.dumps(contract, ensure_ascii=False))
        for pattern in row['output_patterns']:
            lines.append('invocations=' + str(pattern['invocations']) +
                '; zero_actual_nonzero_golden_heads=' + str(pattern['zero_actual_nonzero_golden_heads']))
            for head in pattern['heads']:
                lines.append(f'head {head["head"]}: actual_nonzero={head["actual_nonzero"]}/{head["elements"]} '
                    f'golden_nonzero={head["golden_nonzero"]}/{head["elements"]} relative_l2={head["relative_l2"]} '
                    f'actual_abs_max={head["actual_abs_max"]} golden_abs_max={head["golden_abs_max"]} '
                    f'max_error_at_q_channel=[{head["max_error_query"]},{head["max_error_channel"]}] '
                    f'actual_at_max_error={head["actual_at_max_error"]} golden_at_max_error={head["golden_at_max_error"]}')
                lines.append('row_locations=' + json.dumps(pattern['row_locations'][head['head']]))
        lines.extend('ERROR: ' + e for e in row['errors'])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    return dict(status=report['status'], cases=len(selected), report=str(out/'report.json'),
                summary=str(out/'summary.txt'), output_pattern_csv=csv_paths, numerical_validation='NOT_RUN',
                model_compilation='NOT_RUN', device_execution='NOT_RUN')

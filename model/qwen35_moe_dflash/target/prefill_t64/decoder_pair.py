"""Run independent complete L0/L3 decoders on their original 310P layer inputs."""
import json
import os
import re

import numpy as np

from ...execution.serial_transport import _environment
from ..layer_probe.common import checked, identity, metrics, read_array, save_json, sha
from .chain_compile import _payloads, _pull
from .chain_reference import BASIS, HIDDEN
from . import whole_run, whole_trace


def load(path):
    return json.loads(path.read_text())


def decoder_row(batch, layer):
    name = f'l{layer:02d}_c00_decoder_o0'
    row, = [r for r in batch['cases'] if r['id'] == name]
    if (row['layer'] != layer or row['point'] != 'decoder' or row['phase'] != 'prefill'
            or row['status'] != 'EXPORTED' or row['export']['path'] != f'models/{name}/export.json'):
        raise ValueError(f'Invalid decoder case: {name}')
    return row


def check_capture(batch, provenance):
    digest = batch['capture_sha256']
    if (batch['status'] != 'EXPORTED' or batch['golden_source'] != BASIS
            or not re.fullmatch(r'[0-9a-f]{64}', digest) or provenance['reference'] != digest
            or not provenance['sources']):
        raise ValueError('Expected decoder AIR with a matching T64/valid19 310P capture')


def check_pair_air(root):
    batch = load(root / 'manifest.json')
    check_capture(batch, load(root / 'source.json'))
    for layer in (0, 3):
        metadata = checked(root, decoder_row(batch, layer)['export'])
        export = load(metadata)
        for spec in [*export['source_inputs'], export['golden'], export['air']]:
            checked(metadata.parent, spec)
        weights = export['weight_files']
        if not weights:
            raise ValueError(f'{metadata}: missing FileConstant weights')
        for name, size in _payloads(metadata.parent, export).items():
            path = metadata.parent / name
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f'Incomplete decoder AIR payload: {path}')
    print(f'Complete L0 and L3 decoder fixtures ready: {root}', flush=True)


def prepare_case(run, layer, batch, remote_air):
    row = decoder_row(batch, layer)
    name = row['id']
    air = run / 'air'
    if not air.is_dir():
        _pull(air, air, remote_air / 'models' / name)
    export = load(checked(air, dict(row['export'], path='export.json')))
    inputs, golden = export['source_inputs'], export['golden']
    for spec in [golden, *inputs]:
        checked(air, spec)
    args = {s['semantic_name']: s for s in inputs}
    attention = 'ChunkGatedDeltaRule' if layer == 0 else 'AdnFusedInferAttention'
    control = 'effective_length' if layer == 0 else 'all_length'
    ops = export['graph_ops']
    if (ops.get(attention) != 1 or not any(op.lower().startswith('groupedmatmul') for op in ops)
            or args['hidden']['shape'] != list(HIDDEN) or args['hidden']['dtype'] != 'float16'
            or read_array(air, args[control]).tolist() != [19]
            or golden['shape'] != list(HIDDEN) or golden['dtype'] != 'float16' or golden['name'] != 'hidden'):
        raise ValueError(f'{name}: expected a complete T64/valid19 attention + MoE decoder')
    sizes = _payloads(air, export)
    weights = [identity(air / p, air) for p in sorted(export['weight_files'])]
    if not weights or any(w['bytes'] != sizes[w['path']] for w in weights):
        raise ValueError(f'{name}: incomplete AIR weights')
    manifest = dict(kind='single-cpro-decoder', layer=layer, attention=attention, reference=batch['capture_sha256'],
                    source_export_sha256=row['export']['sha256'], inputs=inputs, golden=golden, air=export['air'],
                    weights=weights, output_contract=export['output_contract'])
    path = air / 'manifest.json'
    if path.exists() and load(path) != manifest:
        raise ValueError(f'{name}: existing decoder manifest differs')
    if not path.exists():
        save_json(path, manifest)
    return manifest


def score_hidden(air, meta, output, spec):
    golden = read_array(air, meta['golden'])
    if (spec['dtype'] != 'float16' or golden.shape != HIDDEN or golden.dtype != np.float16
            or output.stat().st_size != golden.nbytes or spec['bytes'] != golden.nbytes):
        raise ValueError('Expected one FP16 [1,64,2048] decoder output')
    valid = np.fromfile(output, dtype='<f2').reshape(HIDDEN)[0, :19]
    reference = golden[0, :19]
    tokens = [metrics(a, b) for a, b in zip(valid, reference)]
    aggregate = metrics(valid, reference)
    return dict(layer=meta['layer'], attention=meta['attention'], reference_sha256=meta['reference'],
                hidden_sha256=sha(output), status='PASSED' if aggregate['passed'] and tokens[18]['passed'] else 'FAILED',
                valid19=aggregate, token18=tokens[18], per_token=tokens)


def trace_comparison(trace, meta, run, baseline):
    record = load(trace / 'air/manifest.json')
    whole_trace.check_baseline(record, baseline)
    attempt = load(trace / 'board/attempt.json')
    isolated = load(run / 'board/attempt.json')
    previous = load(trace / 'board/first-token.json')['decoder_checkpoints']
    if (record['reference'] != meta['reference']
            or not previous['baseline_logits_equal_bits']
            or any(attempt[k] != isolated[k] for k in ('device', 'server', 'gdr_source_sha256'))):
        raise ValueError('Whole-model trace is from a different capture, board or GDR, or changed the original logits')
    raw = trace / 'board/logits-and-hidden.bin'
    listing, = (trace / 'board/calls').glob('*-list-logits.json')
    call = load(listing)
    digests = re.findall(r'^([a-f0-9]{64})\s+\./[^\r\n]+', call['stdout'], re.M)
    if call['returncode'] or digests != [sha(raw)] or raw.stat().st_size != whole_trace.WIDTH * 4:
        raise ValueError('Whole-model readout differs from the device output')
    start = whole_trace.VOCAB + whole_trace.LAYERS.index(meta['layer']) * whole_trace.HIDDEN
    row = slice(start, start + whole_trace.HIDDEN)
    actual = np.fromfile(raw, dtype='<f4')[row]
    expected = read_array(trace / 'air', record['golden']).reshape(-1)[row]
    reference = read_array(run / 'air', meta['golden'])[0, 18].astype('<f4')
    if not np.array_equal(reference.view('<u4'), expected.view('<u4')):
        raise ValueError('Standalone and whole-model token18 golden bytes differ')
    single = np.fromfile(run / 'board/hidden.bin', dtype='<f2').reshape(HIDDEN)[0, 18].astype('<f4')
    return dict(continuous_vs_310p=metrics(actual, expected), isolated_vs_continuous=metrics(single, actual))


def select_board(baseline, digest, trace):
    compiled_path = baseline / 'omc/manifest.json'
    compiled = load(compiled_path) if compiled_path.is_file() else None
    if compiled is not None and (compiled['gdr_source_sha256'] != digest
                                 or compiled['air_manifest_sha256'] != sha(baseline / 'air/manifest.json')):
        raise ValueError(f'{compiled_path}: baseline OMC differs from the selected GDR or baseline AIR')

    baseline_receipt = baseline / 'board/attempt.json'
    receipt = baseline_receipt
    if not receipt.is_file() and trace and (trace / 'board/attempt.json').is_file():
        whole_trace.check_baseline(load(trace / 'air/manifest.json'), baseline)
        receipt = trace / 'board/attempt.json'

    board_source = 'SN/BIP'
    gdr_source = compiled_path if compiled is not None else 'current DDK and source only'
    if receipt.is_file():
        attempt = load(receipt)
        if receipt == baseline_receipt and compiled is not None and attempt['omc_sha256'] != compiled['primary']['sha256']:
            raise ValueError(f'{receipt}: board attempt differs from baseline OMC')
        if attempt['gdr_source_sha256'] != digest:
            raise ValueError(f'{receipt}: recorded whole-model GDR differs from selected GDR')
        for var, key in (('SN', 'device'), ('BIP', 'server')):
            if os.environ.setdefault(var, attempt[key]) != attempt[key]:
                raise ValueError(f'{var} differs from {receipt}')
        board_source = gdr_source = receipt
    elif not all(os.environ.get(k) for k in ('SN', 'BIP')):
        raise ValueError(f'Missing {receipt}. Set SN and BIP (HDC host:port) for the independent decoder run, '
                         'or pass --trace with the matching whole-model board attempt.')

    _, device, server = _environment()
    print(f'Decoder board {device} via {server} (from {board_source}). '
          f'GDR SHA256 {digest} (checked against {gdr_source}).', flush=True)
    return dict(device=device, server=server, gdr_source_sha256=digest)


def run_pair(root, remote_air, baseline, ddk, source, diagnosis, tool, trace=None):
    digest = whole_run.kernel_identity(ddk, source)
    if baseline:
        selected = select_board(baseline, digest, trace)
    else:
        _, device, server = _environment()
        selected = dict(device=device, server=server, gdr_source_sha256=digest)
        print(f'Decoder board {device} via {server}. Installed GDR SHA256 {digest}.', flush=True)
    batch_path = root / 'source/manifest.json'
    if not batch_path.exists():
        _pull(batch_path, batch_path.parent, remote_air)
    provenance_path = batch_path.with_name('source.json')
    if not provenance_path.exists():
        _pull(provenance_path, provenance_path.parent, remote_air)
    provenance = load(provenance_path)
    batch = load(batch_path)
    check_capture(batch, provenance)
    if baseline:
        base = load(baseline / 'air/manifest.json')
        if batch['capture_sha256'] != base['reference'] or provenance['sources'] != base['source']:
            raise ValueError('Decoder export must match the 40-layer run’s CPro sources and 310P capture')
    rows = {}
    for layer in (0, 3):
        key = f'L{layer}'
        run = root / key
        stage = 'prepare'
        try:
            meta = prepare_case(run, layer, batch, remote_air)
            stage = 'compile'
            if not (run / 'omc/manifest.json').is_file():
                whole_run.compile_model(run, ddk, source, diagnosis)
            compiled = load(run / 'omc/manifest.json')
            if compiled['gdr_source_sha256'] != digest or compiled['air_manifest_sha256'] != sha(run / 'air/manifest.json'):
                raise ValueError(f'{key}: OMC differs from the selected AIR or GDR')
            stage = 'board'
            report = run / 'board/accuracy.json'
            if report.is_file():
                attempt = load(run / 'board/attempt.json')
                result = load(report)
                if (attempt['omc_sha256'] != compiled['primary']['sha256']
                        or any(attempt[k] != v for k, v in selected.items())
                        or result['hidden_sha256'] != sha(run / 'board/hidden.bin')):
                    raise ValueError(f'{key}: retained board output differs from the original attempt')
            else:
                result = whole_run.board(run, tool)
            rows[key] = {k: result[k] for k in ('status', 'valid19', 'token18')}
            for scope in ('valid19', 'token18'):
                m = result[scope]
                print(f"{key} {scope}: L2={m['relative_l2']} cosine={m['cosine']} RMSE={m['rmse']} "
                      f"error_ratio={m['error_ratio']} five_metrics_passed={m['passed']}", flush=True)
            if trace:
                stage = 'compare-trace'
                rows[key]['trace_token18'] = trace_comparison(trace, meta, run, baseline)
                for boundary, m in rows[key]['trace_token18'].items():
                    print(f"{key} token18 {boundary}: L2={m['relative_l2']} cosine={m['cosine']}", flush=True)
        except Exception as exc:
            rows.setdefault(key, {})['error'] = dict(stage=stage, message=str(exc))
            print(f'{key} {stage} failed: {exc}', flush=True)
    save_json(root / 'comparison.json', rows)
    print(root / 'comparison.json', flush=True)
    return all(not r.get('error') and r['status'] == 'PASSED' for r in rows.values())

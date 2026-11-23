"""Sparse token-18 decoder readouts from a continuous T64/valid19 prefill."""
import json
from pathlib import Path
import re

import numpy as np

from ..layer_probe.common import identity, metrics, read_array, sha

LAYERS = (0, 3, 19, 39)
VOCAB = 248320
HIDDEN = 2048
POSITION = 18
WIDTH = VOCAB + len(LAYERS) * HIDDEN


def reference_output(root, meta, out):
    logits = read_array(root, meta['outputs']['logits'])
    if logits.shape != (1, 1, VOCAB) or logits.dtype != np.float32:
        raise ValueError('Expected one measured FP32 logits row')
    parts = [logits]
    for i in LAYERS:
        block = meta['blocks'][i]
        hidden = read_array(root, block['hidden'])
        if block['layer'] != i or hidden.shape != (1, 64, HIDDEN) or hidden.dtype != np.float16:
            raise ValueError(f'Expected measured FP16 decoder output at L{i}')
        parts.append(hidden[:, POSITION:POSITION + 1, :].astype('<f4'))
    path = out / 'trace-golden.bin'
    np.concatenate(parts, axis=-1).astype('<f4', copy=False).tofile(path)
    return dict(identity(path, out), name='logits_and_hidden', dtype='float32', shape=[1, 1, WIDTH])


def source_reference(baseline, reference):
    base = json.loads((baseline / 'air/manifest.json').read_text())
    candidates = [reference] if reference else [baseline / 'reference',
        baseline.parent / 'prefill-40l-t64-valid19-v1024-01/reference']
    for root in candidates:
        path = root / 'manifest.json'
        if path.is_file() and sha(path) == base['reference']:
            return root
    raise ValueError('Pass --reference with the original 310P capture whose manifest SHA256 is ' + base['reference'])


def check_baseline(meta, baseline):
    if baseline is None:
        raise ValueError('The checkpoint diagnostic needs --baseline pointing to the original single-model run')
    base = json.loads((baseline / 'air/manifest.json').read_text())
    def inputs(m):
        return sorted((s['semantic_name'], s['path'], s['sha256'], s['dtype'], s['shape']) for s in m['inputs'])
    if (base.get('trace_layers') or meta['trace_layers'] != list(LAYERS)
            or meta['baseline_air_manifest_sha256'] != sha(baseline / 'air/manifest.json')
            or any(meta[k] != base[k] for k in ('kind', 'reference', 'source', 'weights', 'token_text_map',
                                                'reference_top1', 'reference_text'))
            or inputs(meta) != inputs(base)):
        raise ValueError('Trace reference, weights, sources or input bytes differ from the logits-only baseline')
    return base


def baseline_logits(baseline, digest):
    omc = json.loads((baseline / 'omc/manifest.json').read_text())
    board = baseline / 'board'
    attempt = json.loads((board / 'attempt.json').read_text())
    if (omc['air_manifest_sha256'] != sha(baseline / 'air/manifest.json') or omc['gdr_source_sha256'] != digest
            or attempt['omc_sha256'] != omc['primary']['sha256'] or attempt['gdr_source_sha256'] != digest):
        raise ValueError('Baseline AIR, compiled GDR or board attempt identity differs')
    listing, = (board / 'calls').glob('*-list-logits.json')
    call = json.loads(listing.read_text())
    expected, = re.findall(r'^([a-f0-9]{64})\s+\./[^\r\n]+', call['stdout'], re.M)
    path = board / 'logits.bin'
    if call['returncode'] != 0 or path.stat().st_size != VOCAB * 4 or sha(path) != expected:
        raise ValueError('Baseline logits differ from the original HDC output hash')
    return path


def compare_rows(actual, reference, baseline):
    if any(x.dtype != np.float32 or x.shape != (WIDTH,) for x in (actual, reference)):
        raise ValueError('Expected one packed FP32 diagnostic row')
    if baseline.dtype != np.float32 or baseline.shape != (VOCAB,):
        raise ValueError('Expected the retained baseline vocabulary row')
    equal = actual[:VOCAB].view('uint32') == baseline.view('uint32')
    rows = {}
    for j, layer in enumerate(LAYERS):
        begin = VOCAB + j * HIDDEN
        rows[f'L{layer}'] = metrics(actual[begin:begin + HIDDEN], reference[begin:begin + HIDDEN])
    return dict(position=POSITION, baseline_logits_equal_bits=bool(equal.all()),
                changed_logits=int((~equal).sum()), total_logits=VOCAB, layers=rows)

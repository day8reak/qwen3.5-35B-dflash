"""Compare a retained whole-Prefill result with same-input CPU controls before GDR."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..layer_probe.common import read_array, sha
from ..layer_probe.target_weights import read_tensors
from ..original_quant import OriginalQuantWeights, pack_i4zn
from .audit_gdn import exactly_one
from .cpu_reference import linear, norm
from .gdn_tail import _same_bits, _summary, _tensor, _weights


def audit_input(capture, check, layer):
    capture, check = Path(capture).resolve(strict=True), Path(check).resolve(strict=True)
    fixture = capture.parent
    baseline = json.loads((fixture / 'manifest.json').read_text())
    report = json.loads((check / 'report.json').read_text())
    if (report.get('stage') != 'COMPLETE' or report.get('scope') != 'FOUR_LAYER_FRESH_T64_LAST_POSITION'
            or report.get('original_fixture_sha256') != sha(fixture / 'manifest.json')
            or fixture / baseline['boundary_capture']['path'] != capture / 'manifest.json'
            or baseline['sequence_length'] != 64):
        raise ValueError('Use the complete four-layer comparison and the T64 capture from its original fixture')
    regions = json.loads((capture / 'manifest.json').read_text())['regions']

    def region(point, index=layer):
        return exactly_one((r for r in regions if r['layer'] == index and r['point'] == point
                            and r['phase'] == 'prefill' and r['call'] == 0 and r['sequence_length'] == 64), point)

    state_name = f'conv_state_{layer}'

    def state(root, specs):
        item = exactly_one((s for s in specs if s.get('semantic_name', s['name']) == state_name), state_name)
        return read_array(root, item)

    historical = state(fixture, baseline['outputs'])
    native = state(check, report['native_outputs'])
    whole_cpu = state(check, report['cpu_outputs'])
    entry, proj = region('input_layernorm'), region('linear_attn_in_proj_qkv')
    hidden = _tensor(capture, entry, 'inputs', 'x')
    x = _tensor(capture, proj, 'inputs', 'x')
    y = _tensor(capture, proj, 'outputs', 'golden_y')
    if (x.ndim != 3 or x.shape[:2] != (1, 64) or hidden.shape != x.shape or y.shape[:2] != x.shape[:2]
            or historical.shape != (1, y.shape[-1], 4) or native.shape != historical.shape
            or whole_cpu.shape != historical.shape):
        raise ValueError('Expected T64 QKV and four-token ConvState')
    if layer and not _same_bits(hidden, _tensor(capture, region('residual_mlp', layer-1), 'outputs', 'golden_hidden')):
        raise ValueError('Captured previous decoder output differs from the current decoder input')
    if not _same_bits(x, _tensor(capture, entry, 'outputs', 'golden_y')):
        raise ValueError('Captured input RMSNorm output differs from the QKV projection input')

    def tail(values):
        return values[:, -4:, :].transpose(0, 2, 1)

    if not _same_bits(tail(y), historical):
        raise ValueError('Historical wrapper ConvState differs from the captured QKV tail')

    resources = baseline['resources']
    norm_name = report['floating_checkpoint_parameters'][f'layers.{layer}.input_layernorm.weight']
    weight = read_tensors(resources['model_file'], {norm_name})[norm_name]
    saved = _weights(capture, entry, ['weight'])['weight']
    gamma = torch.from_numpy(saved)
    if not torch.equal(weight.to(gamma.dtype), gamma):
        raise ValueError('Input RMSNorm weight differs from the checkpoint')
    raw, scales = OriginalQuantWeights(resources['quanted_pth']).logical(
        f'target.layers.{layer}.linear_attn.in_proj_qkv.weight', (y.shape[-1], x.shape[-1]))
    packed = _weights(capture, proj, ['W_q', 'scale'])
    if (not _same_bits(pack_i4zn(raw), packed['W_q'])
            or not _same_bits(scales[:, :, 0].T.astype(np.float32), packed['scale'])):
        raise ValueError('Captured QKV weight or scale differs from the checkpoint')

    norm_x = norm(torch.from_numpy(hidden), SimpleNamespace(weight=gamma, variance_epsilon=entry['attrs']['epsilon']))
    y_fixed = linear(torch.from_numpy(x), raw, scales).numpy()
    local_state = tail(linear(norm_x, raw, scales).numpy())
    scores = {
        '310p_norm_vs_cpu_same_layer_input': _summary(x, norm_x.numpy()),
        '310p_qkv_vs_cpu_same_projection_input': _summary(y, y_fixed),
        '310p_conv_vs_cpu_same_projection_input': _summary(historical, tail(y_fixed)),
        '310p_conv_vs_cpu_same_layer_input': _summary(historical, local_state),
        'cpu_fixed_layer_input_conv_vs_cpu_whole': _summary(local_state, whole_cpu),
        'native_conv_vs_cpu_whole': _summary(native, whole_cpu),
    }
    native_check = report['comparison']['native_310p']
    summary_keys = ('passed', 'relative_l2', 'cosine', 'failed_metrics')
    scores.update({f'whole_L{i}_decoder_hidden': {k: row[k] for k in summary_keys}
                   for i, row in enumerate(native_check['decoder_hidden'])})
    for run, cmp in report['comparison'].items():
        if run in ('native_310p', 'l3_math_310p') and 'outputs' in cmp:
            logits = cmp['outputs']['logits']
            scores[run + '_logits_vs_cpu'] = {k: logits[k] for k in (*summary_keys, 'cpu_top1', 'device_top1')}
    return {'layer': layer, 'tail_token_indices': [60, 61, 62, 63],
            'native_conv_equals_historical_bits': _same_bits(native, historical), 'comparisons': scores}

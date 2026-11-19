"""Qualify one original NZ L1 GDR call at 64 physical and 19 valid rows."""
import numpy as np
import torch

from ..layer_probe.capture import Capture
from ..layer_probe.common import metrics
from .gdr_math import compute


OPERANDS = ('query', 'key', 'value', 'g', 'beta', 'initial_state', 'effective_length')


def audit_operands(inputs, core, state):
    q, k, v, g, beta, old_state, length = [inputs[name].numpy() for name in OPERANDS]
    if (length.shape != (1,) or length.dtype != np.int16 or int(length[0]) != 19
            or q.shape[:2] != (1, 64) or k.shape != q.shape or v.shape[:3] != q.shape[:3]
            or g.shape != q.shape[:3] or beta.shape != g.shape or np.any(old_state)
            or q.dtype != np.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or g.dtype != np.float32 or beta.dtype != q.dtype
            or core.shape != v.shape or core.dtype != torch.float16
            or state.shape != old_state.shape or state.dtype != torch.float32):
        raise ValueError('expected fresh B1 L1 GDR with 64 physical rows and INT16 length [19]')
    valid = int(length[0])
    cpu_core, cpu_state = compute(*(a[:, :valid] for a in (q, k, v, g, beta)), old_state)
    active = core[:, :valid].numpy()
    comparisons = dict(valid_core=metrics(active, cpu_core),
                       last_valid_core=metrics(active[:, -1], cpu_core[:, -1]),
                       final_state=metrics(state.numpy(), cpu_state))
    passed = all(row['passed'] for row in comparisons.values())
    return comparisons, passed


def save(output, snapshot, report):
    for call in report['l1_gdr']['calls']:
        if 'error' in call:
            raise ValueError(f"GDR operand capture failed: {call['error']}")
    l1 = snapshot['gdr']
    inputs, (core, state) = l1['inputs'], l1['outputs']
    ids, prompt = snapshot['wrapper_ids'], snapshot['prompt_ids']
    if (len(report['l1_gdr']['calls']) != 1 or report['segment_entry'] != 'prefill_chunk_segment1'
            or report['wrapper_arguments']['effective_length'] != [19]
            or report['wrapper_arguments']['allQLen'] != [19]
            or tuple(ids.shape) != (1, 64) or tuple(prompt.shape) != (1, 19)
            or not torch.equal(ids[:, :19], prompt)
            or l1['attrs']['chunk_size'] != 64 or not l1['attrs']['output_final_state']
            or not l1['attrs']['use_qk_l2norm_in_kernel']):
        raise ValueError('expected the first 19-token segmented Prefill and one normalized chunk-64 GDR call')
    comparisons, passed = audit_operands(inputs, core, state)
    capture = Capture(output, layers=[1], calls=1)
    for path in (report['entry'], l1['source']):
        capture.source(path)
    capture.active = dict(layer=1, call=0, seq_len=64, execution_device=l1['device'])
    # Core after row 18 is padding. Keep the full original result as evidence;
    # the replay golden only contains rows used by this prompt.
    capture.emit('gdn', 'gdn', inputs, dict(core=core[:, :19], state_raw=state), attrs=l1['attrs'])
    capture.report['regions'][0]['gdr_valid_rows'] = 19
    capture.report.update(status='CAPTURED' if passed else 'NUMERICAL_FAILED',
                          scope='L1_SEGMENT_GDR_PHYSICAL64_VALID19',
                          reference_qualification='CPU_CHECKED' if passed else 'CPU_MISMATCH',
                          wrapper_arguments=report['wrapper_arguments'],
                          prompt_ids=capture.tensor(capture.root, 'prompt_ids', prompt),
                          wrapper_ids=capture.tensor(capture.root, 'wrapper_ids', ids),
                          full_observed_core=capture.tensor(capture.root, 'full_observed_core', core),
                          cpu_audit=comparisons)
    capture.flush()
    return dict(status=capture.report['reference_qualification'], manifest=str(capture.root/'manifest.json'),
                **{name: {k: row[k] for k in ('passed', 'relative_l2', 'cosine', 'failed_metrics')}
                   for name, row in comparisons.items()})


def verify_capture(capture, manifest, record, inputs):
    from ..layer_probe.common import read_array

    if (manifest.get('reference_qualification') != 'CPU_CHECKED'
            or [s['name'] for s in record['inputs']] != list(OPERANDS)
            or [s['name'] for s in record['outputs']] != ['golden_core', 'golden_state_raw']
            or record['attrs'].get('chunk_size') != 64 or not record['attrs'].get('output_final_state')
            or not record['attrs'].get('use_qk_l2norm_in_kernel') or record['gdr_valid_rows'] != 19):
        raise ValueError('expected a qualified 64/19 GDR capture')
    full = read_array(capture, manifest['full_observed_core'])
    valid, state = (read_array(capture, s) for s in record['outputs'])
    if valid.shape != full[:, :19].shape or valid.dtype != full.dtype or valid.tobytes() != full[:, :19].tobytes():
        raise ValueError('GDR valid-row golden differs from original core bytes')
    _, passed = audit_operands(dict(zip(OPERANDS, inputs)), torch.from_numpy(full), torch.from_numpy(state))
    if not passed:
        raise ValueError('retained 310P GDR core or state failed its CPU check')

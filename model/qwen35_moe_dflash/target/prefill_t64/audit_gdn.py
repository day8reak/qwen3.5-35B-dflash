"""Audit one T64 GDN layer using the retained 310P tensors. All replay arithmetic runs on CPU."""
import json
from pathlib import Path

import numpy as np


def exactly_one(items, what):
    result = list(items)
    if len(result) != 1:
        raise ValueError(f'Expected one {what}, found {len(result)}')
    return result[0]


def core_location(actual, reference):
    """Where the total squared core error occurs, using full head vectors in logical B,T,H,V order."""
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    energy = (delta[0] ** 2).sum(axis=-1)
    total = float(energy.sum())
    if not total:
        return None
    token, head = map(int, np.unravel_index(np.argmax(energy), energy.shape))
    return dict(token=token, head=head, squared_error_share=float(energy[token, head]/total),
                error_share_by_16_tokens=(energy.reshape(4,16,-1).sum(axis=(1,2))/total).tolist())


def audit(capture, layer=1, group=None, baseline_group=None):
    from qwen35_moe_dflash.target.layer_probe.common import KIND, metrics, read_array, sha
    from .gdr_math import compute
    from .gdn_tail import audit_tail

    capture = Path(capture).resolve(strict=True)
    manifest = capture/'manifest.json'
    doc = json.loads(manifest.read_text())
    if (doc.get('artifact_kind') != KIND or doc.get('reference') != 'ORIGINAL_310P_INFERENCE'
            or doc.get('status') != 'CAPTURED'):
        raise ValueError('Expected the completed original layer-boundary capture at prefill-fixture-t64/capture')
    same_call = [r for r in doc['regions'] if (r['layer'],r['call'],r['phase']) == (layer,0,'prefill')]
    core = exactly_one((r for r in same_call if r['point'] == 'gdn'), 'GDR region')
    if (core['sequence_length'] != 64 or core['attrs'].get('chunk_size') != 64
            or not core['attrs'].get('use_qk_l2norm_in_kernel') or not core['attrs'].get('output_final_state')):
        raise ValueError('Expected the T64 chunk-64 reference with Q/K normalization and a final state')

    def tensor(region, direction, name):
        spec = exactly_one((r for r in region[direction] if r['name'] == name), name)
        return read_array(capture, spec)

    q,k,v,g,b,state0,length = [tensor(core,'inputs',name) for name in
                              ('query','key','value','g','beta','initial_state','effective_length')]
    if (q.ndim != 4 or q.shape[:2] != (1,64) or not np.array_equal(length,[64])
            or length.dtype != np.int16 or state0.dtype != np.float16 or np.any(state0)):
        raise ValueError('Expected the fresh B1 T64 capture with length 64 and a zero FP16 cache')
    original_core, original_state = [tensor(core,'outputs',name) for name in ('golden_core','golden_state_raw')]
    if original_core.dtype != np.float16 or original_state.dtype != np.float32:
        raise ValueError('Unexpected original GDR core/state dtype')

    def row(actual, reference):
        m = metrics(actual,reference)
        return {k:m[k] for k in ('passed','relative_l2','cosine','rmse','max_abs','mean_relative_error','failed_metrics')}

    inputs = (q,k,v,g,b,state0)
    cpu_core, cpu_state = compute(*inputs)
    gate_core, gate_state = compute(*inputs,round_gate=True)
    rounded_core, rounded_state = compute(*inputs,round_gate=True,round_state=True)
    # Apply the recorded output boundary before comparing with the captured FP16 core.
    cpu_core,gate_core,rounded_core = [c.astype(original_core.dtype) for c in (cpu_core,gate_core,rounded_core)]
    scores = {
        'cpu_core_vs_310p_core': row(cpu_core,original_core),
        'cpu_state_vs_310p_raw_state': row(cpu_state,original_state),
        'gate_fp16_core_vs_cpu_core': row(gate_core,cpu_core),
        'gate_fp16_state_vs_cpu_state': row(gate_state,cpu_state),
        'gate_and_step_fp16_core_vs_gate_fp16_core': row(rounded_core,gate_core),
        'gate_and_step_fp16_state_vs_gate_fp16_state': row(rounded_state,gate_state),
        'gate_and_step_fp16_core_vs_310p_core': row(rounded_core,original_core),
        'gate_and_step_fp16_state_vs_310p_raw_state': row(rounded_state,original_state)
    }
    report = dict(layer=layer,call=0,tokens=64,capture_sha256=sha(manifest),gdr_comparisons=scores,
                  largest_core_error=core_location(cpu_core,original_core),
                  scope='CAPTURED_OPERANDS; CPU_ORIGINAL_MATH_AND_ROUNDING_CONTROLS')
    samples = {}
    if group:
        from .gdn_board import audit_board
        proj = exactly_one((r for r in same_call if r['point']=='linear_attn_out_proj'),'out_proj')
        hidden = tensor(proj,'outputs','golden_y')
        operands = (report['capture_sha256'], layer, hidden, original_state, cpu_state, rounded_state)
        report['board'], samples = audit_board(group, *operands, return_samples=True)
        if baseline_group:
            baseline = audit_board(baseline_group, *operands)
            board = report['board']
            if ('core' not in samples or not baseline['output_hashes'].keys() < board['output_hashes'].keys()
                    or board['input_hashes'] != baseline['input_hashes']):
                raise ValueError('Expected a traced group and a baseline with the same inputs and a subset of its outputs')
            report['original_outputs_equal_bits'] = {
                name: board['output_hashes'][name] == digest for name, digest in baseline['output_hashes'].items()}
    replay_cores = None
    if 'gdr_inputs' in samples:
        names = ('query', 'key', 'value', 'g', 'beta')
        board_inputs = [samples['gdr_inputs'][name] for name in names]
        for name, observed, captured in zip(names, board_inputs, inputs):
            if observed.shape != captured.shape or observed.dtype != np.float16:
                raise ValueError('CPro GDR operand shape/dtype mismatch: ' + name)
            scores['board_gdr_' + name + '_vs_capture'] = row(observed, captured)
        board_cpu_core, _ = compute(*board_inputs, state0, round_state=True, round_gate=True)
        scores['board_core_vs_cpu_on_board_gdr_inputs'] = row(samples['core'], board_cpu_core)
        replay_cores = (rounded_core, board_cpu_core)
    report['tail'] = audit_tail(capture, doc['regions'], layer, original_core, cpu_core,
                                board_hidden=samples.get('hidden'), board_stages=samples if 'core' in samples else None,
                                replay_cores=replay_cores)
    return report

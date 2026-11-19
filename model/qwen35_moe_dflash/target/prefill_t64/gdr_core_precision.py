"""Fixed-input numerical controls for a fresh valid19 GDR chunk with zero initial state."""
import math

import numpy as np

from .gdn_tail import _gate_norm
from .segment_gdn_trace import OPERANDS


def chunk_core(q, k, v, g, beta, *, round_readout=False, round_feedback=False):
    """Causal chunk equations in FP32. Each switch rounds one coefficient matrix to FP16."""
    dtype = q.dtype
    q, k, v = (x.astype(np.float32).swapaxes(1, 2) for x in (q, k, v))
    g, beta = (x.astype(np.float32).swapaxes(1, 2) for x in (g, beta))
    length, width = q.shape[-2:]
    # Use the same Q/K normalization, query scaling and core interface as the sequential oracle.
    q /= np.sqrt(np.sum(q*q, axis=-1, keepdims=True) + np.float32(1e-6))
    q *= np.float32(1. / math.sqrt(width))
    k /= np.sqrt(np.sum(k*k, axis=-1, keepdims=True) + np.float32(1e-6))
    prefix = np.cumsum(g, axis=-1)
    log_decay = prefix[..., :, None] - prefix[..., None, :]
    decay = np.zeros_like(log_decay)
    np.exp(log_decay, where=np.tri(length, dtype=bool), out=decay)
    feedback = (k @ k.swapaxes(-2, -1)) * decay
    readout = (q @ k.swapaxes(-2, -1)) * decay
    if round_feedback:
        feedback = feedback.astype(np.float16).astype(np.float32)
    if round_readout:
        readout = readout.astype(np.float16).astype(np.float32)
    # delta[t] = beta[t] * (V[t] - sum_{s<t} feedback[t,s] * delta[s]).
    delta = np.empty_like(v)
    for t in range(length):
        prediction = np.einsum('bhs,bhsv->bhv', feedback[..., t, :t], delta[..., :t, :])
        delta[..., t, :] = beta[..., t, None] * (v[..., t, :] - prediction)
    return (readout @ delta).swapaxes(1, 2).astype(dtype)


def precision_controls(board, refs, replay, gamma, epsilon, replay_state_half, project, show):
    """Compare CPU hypotheses to measured core values, with the board gate fixed throughout."""
    shape = replay.shape
    actual = board['core'].reshape(shape)
    gate = board['norm_gate'].reshape(shape)

    def norm(core):
        return _gate_norm(core, gate, gamma, epsilon, np.float16)

    def tail(core):
        return project({'out_proj_input': norm(core)})

    print('Whole hidden vs 310P. These two CPU replays share the CPro gate and output projection.')
    print('comparison                                  valid19_L2  token18_L2  five_metrics')
    golden = refs['golden_hidden'][:, :19]
    show('CPro core -> CPU tail', tail(actual), golden)
    show('CPU core on CPro operands -> CPU tail', tail(replay), golden)

    tiny = np.finfo(np.float16).tiny
    subnormal = (replay != 0) & (np.abs(replay) < tiny)
    print(f'FP16 core output: CPU subnormals={subnormal.sum()}, CPro zero at those elements='
          f'{np.count_nonzero(subnormal & (actual == 0))}, '
          f'CPro zero with CPU normal={np.count_nonzero((actual == 0) & (np.abs(replay) >= tiny))}')
    args = [board[n].reshape(shape) if n in ('query', 'key', 'value') else board[n] for n in OPERANDS]
    # Choose the row by the measured proj-input difference, independently of all hypotheses.
    observed = ((board['out_proj_input'].reshape(shape).astype(np.float64)
                 - refs['out_proj_input'].reshape(shape).astype(np.float64))[0])
    token, head = np.unravel_index(np.sum(observed**2, axis=-1).argmax(), observed.shape[:2])
    norm_actual = norm(actual)[0].astype(np.float64)
    base_gap = norm(replay)[0].astype(np.float64) - norm_actual
    print(f'Fixed-input residual vs measured CPro core. Row of largest measured proj input error: token={token} head={head}.')
    energy = np.sum(base_gap**2)
    if energy:
        print(f'Token 0 (fresh zero state): {np.sum(base_gap[0]**2)/energy:.2%} of same-input post-norm squared error.')
    print('After norm: remaining gap / sequential FP32 gap (1=baseline, 0=match, n/a=zero baseline gap).')
    print('CPU model                          raw_core_rel_L2      valid19    token0   token18   peak_row')

    def ratio(delta, denominator):
        size = np.linalg.norm(denominator)
        return f'{np.linalg.norm(delta) / size:9.5g}' if size else '      n/a'

    raw_actual = actual.astype(np.float64)

    def row(label, candidate):
        gap = norm(candidate)[0].astype(np.float64) - norm_actual
        cols = [ratio(candidate.astype(np.float64) - raw_actual, raw_actual), ratio(gap, base_gap),
                ratio(gap[0], base_gap[0]), ratio(gap[18], base_gap[18]),
                ratio(gap[token, head], base_gap[token, head])]
        print(f'{label:36s} ' + ' '.join(cols))

    row('sequential FP32', replay)
    row('flush final FP16 subnormals only', np.where(subnormal, np.float16(0), replay))
    row('FP16 state between tokens only', replay_state_half)
    row('causal chunk FP32', chunk_core(*args))
    row('chunk FP16 readout coefficients', chunk_core(*args, round_readout=True))
    row('chunk FP16 feedback coefficients', chunk_core(*args, round_feedback=True))

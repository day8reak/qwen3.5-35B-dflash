"""CPU math for a full valid sequence of captured [B,T,H,D] GDR operands."""
import math

import numpy as np


def compute(q, k, v, g, beta, initial_state, *, round_state=False, round_gate=False):
    """Return the per-token core and the final [B,H,Dk,Dv] state.

    Compute normalized Q/K and the delta recurrence in FP32. The core rounds to
    the captured Q dtype. The default returns the FP32 raw state used to audit
    310P. round_state models an FP16 state at each invocation boundary, including
    the input and final return. The current token's core reads the updated FP32
    accumulator before this boundary. round_gate rounds only g to FP16 before
    exponentiation. These two switches model visible buffer conversions.
    """
    q, k, v, g, beta, initial_state = (
        np.asarray(x) for x in (q, k, v, g, beta, initial_state)
    )
    if (q.ndim != 4 or k.shape != q.shape or v.ndim != 4
            or v.shape[:3] != q.shape[:3] or min(*q.shape, v.shape[-1]) < 1):
        raise ValueError('GDR expects compatible nonempty [B,T,H,D] Q/K/V')
    batch, length, heads, dk = q.shape
    if g.shape != q.shape[:3] or beta.shape != g.shape:
        raise ValueError('GDR expects g and beta of shape [B,T,H]')
    if initial_state.shape != (batch, heads, dk, v.shape[-1]):
        raise ValueError('GDR initial state must have shape [B,H,Dk,Dv]')
    for x in (q, k, v, g, beta, initial_state):
        if x.dtype.kind != 'f' or x.dtype.itemsize not in (2, 4) or not np.isfinite(x).all():
            raise ValueError('GDR operands must be finite FP16 or FP32 arrays')

    output_dtype = q.dtype
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        q, k, v, g, beta = (x.astype(np.float32, copy=False) for x in (q, k, v, g, beta))
        eps = np.float32(1e-6)
        q = q / np.sqrt(np.sum(q*q, axis=-1, keepdims=True) + eps)
        q *= np.float32(1.0 / math.sqrt(dk))
        k = k / np.sqrt(np.sum(k*k, axis=-1, keepdims=True) + eps)
        if round_gate:
            g = g.astype(np.float16).astype(np.float32)
        decay = np.exp(g)
        state = initial_state.astype(np.float16 if round_state else np.float32, copy=True)
        output = np.empty(v.shape, dtype=output_dtype)
        for t in range(length):
            decayed = decay[:, t, :, None, None] * state
            prediction = np.einsum('bhk,bhkv->bhv', k[:, t], decayed)
            residual = beta[:, t, :, None] * (v[:, t] - prediction)
            updated = decayed + k[:, t, :, :, None] * residual[:, :, None, :]
            output[:, t] = np.einsum('bhk,bhkv->bhv', q[:, t], updated)
            state = updated.astype(np.float16) if round_state else updated
    return output, state

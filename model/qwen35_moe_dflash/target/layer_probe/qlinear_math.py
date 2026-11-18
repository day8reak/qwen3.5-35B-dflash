"""Independent mathematical diagnostics for the original W4 QLinear path.

No packed weights or NPU operation are used here. ``raw_weight[n, k]`` is a
signed int4 value carried in int8; ``raw_scale[n, g, 0]`` scales K group g of
256 values. Original scales are converted to float32 before either reference.

Both references use an exact integer dot per group. In ascending group order,
ref64 widens the dot and float32 group scale to float64, multiplies, then adds;
ref32 performs the same separate operations in float32. Only after that sum is
the float32 token scale multiplied (widened for ref64), followed by one float16
cast. Neither result is selected according to agreement with a device golden.
These declared arithmetic orders and dynamic-quant rounding are diagnostic
conventions, not proof of the installed 310P kernel's internal arithmetic.
"""
from __future__ import annotations

import numpy as np


GROUP_SIZE = 256
_M_BLOCK = 32
_N_BLOCK = 128
_FLOAT_DTYPES = (np.dtype('float16'), np.dtype('float32'))


def _array(name, value, ndim, dtypes):
    if not isinstance(value, np.ndarray):
        raise TypeError(f'{name} must be a NumPy array')
    if value.dtype not in dtypes:
        raise TypeError(f'{name} has unsupported dtype {value.dtype}')
    if value.ndim != ndim or any(size <= 0 for size in value.shape):
        raise ValueError(f'{name} must have {ndim} nonempty dimensions')
    if value.dtype.kind == 'f' and not np.isfinite(value).all():
        raise ValueError(f'{name} contains nonfinite values')
    return value


def dynamic_quant(x):
    """Return int8 Q, float32 [M,1] scale, and the fixed quantization policy.

    x must be finite float16/float32 [M,K]. All math after the input cast is
    float32: scale = max(abs(x), axis=1) / 127, Q = rint(x / scale), clipped to
    [-127,127]. ``rint`` rounds ties to even. Exactly zero rows yield Q=0 and
    scale=0; a nonzero row whose scale underflows to zero is rejected.
    """
    x = _array('x', x, 2, _FLOAT_DTYPES)
    x32 = x.astype(np.float32, copy=False)
    row_max = np.max(np.abs(x32), axis=1, keepdims=True)
    with np.errstate(under='ignore'):
        scale = row_max / np.float32(127)
    if np.any((row_max != 0) & (scale == 0)):
        raise FloatingPointError('nonzero row dynamic-quant scale underflows float32')
    normalized = np.zeros_like(x32)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        np.divide(x32, scale, out=normalized, where=scale != 0)
        q = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
    info = {
        'input_shape': list(x.shape),
        'input_dtype': str(x.dtype),
        'arithmetic_dtype': 'float32',
        'scale_formula': 'max(abs(float32(x)), axis=1, keepdims=True) / float32(127)',
        'quant_formula': 'clip(rint(float32(x) / scale), -127, 127)',
        'rounding': 'nearest_ties_to_even',
        'zero_row_policy': 'q=0, scale=0; no epsilon',
        'zero_rows': int(np.count_nonzero(row_max == 0)),
        'nonzero_scale_underflow': 'error',
        'installed_kernel_rounding': 'NOT_VERIFIED',
        'scope': 'DECLARED_NUMPY_MATHEMATICAL_REFERENCE',
    }
    return q, scale, info


def group_matmul(q, token_scale, raw_weight, raw_scale):
    """Return (ref64_final_fp16, ref32_final_fp16), each shaped [M,N].

    Required inputs: int8 q[M,K] in [-128,127], float32 token_scale[M,1],
    int8 raw_weight[N,K] in [-8,7], and float16/float32 raw_scale[N,K/256,1].
    Scales must be finite and nonnegative. K must be divisible by 256. Dots
    are exact int32: abs(sum(q*w)) <= 256*128*8 = 262144, so int32 cannot
    overflow. Only bounded M/N tiles are converted to int32, without a full
    dequantized weight. Floating overflow, including the final float16 cast,
    is an error; representable underflow follows NumPy arithmetic.
    """
    q = _array('q', q, 2, (np.dtype('int8'),))
    token_scale = _array('token_scale', token_scale, 2, (np.dtype('float32'),))
    raw_weight = _array('raw_weight', raw_weight, 2, (np.dtype('int8'),))
    raw_scale = _array('raw_scale', raw_scale, 3, _FLOAT_DTYPES)
    m, k = q.shape
    n, weight_k = raw_weight.shape
    if k % GROUP_SIZE or weight_k != k:
        raise ValueError('q and raw_weight must share K divisible by 256')
    if token_scale.shape != (m, 1):
        raise ValueError('token_scale must have shape [M,1]')
    if raw_scale.shape != (n, k // GROUP_SIZE, 1):
        raise ValueError('raw_scale must have shape [N,K/256,1]')
    if int(raw_weight.min()) < -8 or int(raw_weight.max()) > 7:
        raise ValueError('raw_weight must contain signed int4 values [-8,7]')
    if np.any(token_scale < 0) or np.any(raw_scale < 0):
        raise ValueError('token_scale and raw_scale must be nonnegative')
    scales = raw_scale[:, :, 0].astype(np.float32, copy=False)
    ref64 = np.empty((m, n), dtype=np.float16)
    ref32 = np.empty_like(ref64)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        for mi in range(0, m, _M_BLOCK):
            ms = slice(mi, min(mi + _M_BLOCK, m))
            for ni in range(0, n, _N_BLOCK):
                ns = slice(ni, min(ni + _N_BLOCK, n))
                acc64 = np.zeros((ms.stop - mi, ns.stop - ni), dtype=np.float64)
                acc32 = np.zeros_like(acc64, dtype=np.float32)
                for g in range(k // GROUP_SIZE):
                    ks = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
                    dot = q[ms, ks].astype(np.int32) @ raw_weight[ns, ks].astype(np.int32).T
                    group_scale = scales[ns, g][None, :]
                    acc64 += dot.astype(np.float64) * group_scale.astype(np.float64)
                    acc32 += dot.astype(np.float32) * group_scale
                ref64[ms, ns] = (acc64 * token_scale[ms].astype(np.float64)).astype(np.float16)
                ref32[ms, ns] = (acc32 * token_scale[ms]).astype(np.float16)
    return ref64, ref32

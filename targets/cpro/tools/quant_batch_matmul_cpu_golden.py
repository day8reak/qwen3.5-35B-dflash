#!/usr/bin/env python3
"""NumPy CPU golden for the captured GDN QKV QuantBatchMatmulV4444 (A8W4).

Run with a directory containing x1.bin, x2.bin, scale.bin, pertoken_scale.bin:
    python quant_batch_matmul_cpu_golden.py --input-dir /path/to/dump

Or pass the four files explicitly:
    python quant_batch_matmul_cpu_golden.py --x1 x1.bin --x2 x2.bin \
        --scale scale.bin --pertoken-scale pertoken_scale.bin --output y.bin

All files are little-endian, C-order, with no header. The default M,K,N is
64,2048,8192. The output always contains all M rows. --compare accepts an
FP16[M,N] result from the device. No model, manifest or NPU runtime is needed.
"""
import argparse
from math import prod
import os
from pathlib import Path
import tempfile

import numpy as np


GROUP = 256
QKV_SHAPE = (64, 2048, 8192)


def unpack_i4zn_group(packed_tiles):
    """Decode four consecutive K64 tiles into signed W[N,256].

    Within each K64/N16 tile, one byte holds two adjacent K positions of the
    same output channel. Even K is the low nibble, odd K is the high nibble.
    """
    codes = np.empty((*packed_tiles.shape[:-1], 64), dtype=np.int8)
    codes[..., 0::2] = packed_tiles & 0x0F
    codes[..., 1::2] = packed_tiles >> 4
    codes[codes >= 8] -= 16
    # [4, N/16, 16, 64] -> [N/16, 16, 4, 64] -> [N, 256]
    return codes.transpose(1, 2, 0, 3).reshape(-1, GROUP)


def cpu_golden(x1, x2, scale, pertoken_scale):
    """INT8[M,K], packed I4ZN INT32[K,N/8], scale[K/256,N], token scale[M or M,1]."""
    if x1.ndim != 2 or x1.dtype != np.int8 or x2.ndim != 2 or x2.dtype != np.dtype('<i4'):
        raise ValueError('expected INT8 x1[M,K] and packed little-endian INT32 x2[K,N/8]')
    m, k = x1.shape
    n = x2.shape[1] * 8
    if min(m, k, n) <= 0 or x2.shape[0] != k or k % GROUP or n % 16:
        raise ValueError('I4ZN needs matching K, K divisible by 256, and N divisible by 16')
    if scale.shape != (k // GROUP, n) or pertoken_scale.shape not in ((m,), (m, 1)):
        raise ValueError('expected group scale[K/256,N] and pertoken_scale[M] or [M,1]')
    for values in (scale, pertoken_scale):
        if values.dtype not in (np.dtype('<f2'), np.dtype('<f4')):
            raise ValueError('scales must be FP16 or FP32')
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('scales must be finite and nonnegative')

    packed = np.ascontiguousarray(x2).view(np.uint8).reshape(k // 64, n // 16, 16, 32)
    acc = np.zeros((m, n), dtype=np.float32)
    with np.errstate(over='raise', invalid='raise'):
        for g, group_scale in enumerate(scale):
            x = x1[:, g * GROUP:(g + 1) * GROUP].astype(np.int32)
            w = unpack_i4zn_group(packed[4 * g:4 * (g + 1)]).astype(np.int32)
            dot = x @ w.T                                      # exact INT32 [M,N]
            acc += dot.astype(np.float32) * group_scale.astype(np.float32)
        acc *= pertoken_scale.astype(np.float32).reshape(m, 1)
        return acc.astype('<f2')                               # one final FP16 rounding


def read_bin(path, shape, *dtypes):
    """The element count distinguishes an FP16 scale file from an FP32 one."""
    count = prod(shape)
    size = path.stat().st_size
    for dtype in map(np.dtype, dtypes):
        if size == count * dtype.itemsize:
            return np.fromfile(path, dtype=dtype).reshape(shape)
    raise ValueError(f'{path}: {size} bytes do not match shape {shape}, allowed dtypes {dtypes}')


def compare(cpu, device, rows):
    # CPU is the actual value, device is the reference (same direction as the QKV capture report).
    actual = cpu[:rows].astype(np.float64).ravel()
    ref = device[:rows].astype(np.float64).ravel()
    if not np.isfinite(ref).all():
        raise ValueError(f'device output contains nonfinite values in the first {rows} rows')
    diff = actual - ref
    err, a_norm, r_norm = (np.linalg.norm(v) for v in (diff, actual, ref))
    cosine = (np.clip(np.dot(actual, ref) / (a_norm * r_norm), -1, 1)
              if a_norm and r_norm else float(a_norm == r_norm))
    rel_l2 = err / r_norm if r_norm else (0.0 if not err else None)
    # Same min/max KL convention and near-zero denominator rule as the capture report.
    def distribution(v):
        span = v.max() - v.min()
        weights = (v - v.min()) / span if span else v
        weights = np.maximum(weights, np.finfo(np.float32).eps)
        return weights / weights.sum()

    a, b = distribution(actual), distribution(ref)
    kl = max(0.0, np.sum(a * (np.log(a) - np.log(b))))
    keep = np.abs(ref) > np.finfo(np.float64).eps
    mean_relative = np.abs(actual[keep] / ref[keep] - 1).sum() / ref.size
    with np.errstate(over='ignore'):
        close = np.isclose(cpu[:rows], device[:rows], rtol=1e-3, atol=1e-5)
    print(f'CPU vs device, first {rows} rows: cosine={cosine:.9g} relative_l2={rel_l2} '
          f'kl_divergence={kl:.9g} rmse={np.sqrt(np.mean(diff**2)):.9g} '
          f'mean_relative_error={mean_relative:.9g} max_abs={np.max(np.abs(diff)):.9g} '
          f'error_ratio={np.count_nonzero(~close)/close.size:.9g}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', '--capture', type=Path, default=Path('.'), help='directory with the four named bins')
    for name in ('x1', 'x2', 'scale', 'pertoken-scale'):
        parser.add_argument('--' + name, type=Path, help='explicit path overriding the corresponding input')
    parser.add_argument('--shape', nargs=3, type=int, default=QKV_SHAPE, metavar=('M', 'K', 'N'))
    parser.add_argument('-o', '--output', type=Path, help='default: INPUT_DIR/golden_cpu.bin')
    parser.add_argument('--compare', type=Path, help='FP16[M,N] device output, e.g. golden_310p.bin')
    parser.add_argument('--valid-rows', type=int, help='comparison only; default 19 for L0 QKV, otherwise M')
    args = parser.parse_args()
    m, k, n = args.shape
    if min(m, k, n) <= 0 or k % GROUP or n % 16:
        parser.error('M,K,N must be positive, K divisible by 256 and N divisible by 16')
    rows = args.valid_rows if args.valid_rows is not None else (19 if tuple(args.shape) == QKV_SHAPE else m)
    if not 1 <= rows <= m:
        parser.error('--valid-rows must be between 1 and M')

    inputs = {key: getattr(args, key) or args.input_dir / (key + '.bin')
              for key in ('x1', 'x2', 'scale', 'pertoken_scale')}
    output = args.output or args.input_dir / 'golden_cpu.bin'
    reference = args.compare
    captured = args.input_dir / 'golden_310p.bin'
    if reference is None and captured.is_file() and all(getattr(args, k) is None for k in inputs):
        reference = captured
    protected = [*inputs.values(), captured]
    if reference is not None:
        protected.append(reference)
    if output.resolve() in {p.resolve() for p in protected}:
        parser.error('output must be separate from the operands and measured reference')
    try:
        x1 = read_bin(inputs['x1'], (m, k), 'i1')
        x2 = read_bin(inputs['x2'], (k, n // 8), '<i4')
        scale = read_bin(inputs['scale'], (k // GROUP, n), '<f2', '<f4')
        token = read_bin(inputs['pertoken_scale'], (m,), '<f2', '<f4')
        golden = cpu_golden(x1, x2, scale, token)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            staged = Path(tmp) / 'golden.bin'
            golden.tofile(staged)
            os.replace(staged, output)
        print(f'{output}: FP16[{m},{n}], {golden.nbytes} bytes')
        if reference is not None:
            print(f'Reference: {reference}')
            compare(golden, read_bin(reference, (m, n), '<f2'), rows)
    except (OSError, ValueError, FloatingPointError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    main()

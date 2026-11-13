"""Portable, hash-bound local tensor fixtures. No model or device imports."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.common.accuracy import POLICY, floating_metrics

KIND = 'qwen35-target-cpro-boundary-probe-v1'
DTYPES = {'float16': '<f2', 'float32': '<f4', 'int16': '<i2', 'int32': '<i4', 'int64': '<i8'}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(path, root):
    path = Path(path)
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def checked(root, record):
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('fixture path escapes its root or is absent')
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('fixture changed: ' + record['path'])
    return path


def new_output(path):
    root = output_path(path)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    return root


def metrics(actual, golden, *, atol=None, rtol=None):
    a, b = np.asarray(actual), np.asarray(golden)
    if a.shape != b.shape:
        raise ValueError('comparison shape mismatch')
    if (atol is None) != (rtol is None) or any(v is not None and (not math.isfinite(v) or v < 0) for v in (atol, rtol)):
        raise ValueError('atol/rtol must both be supplied and finite nonnegative')
    integer = a.dtype.kind in 'biu' and b.dtype.kind in 'biu'
    if integer and a.dtype != b.dtype:
        raise ValueError('integer dtype mismatch; implicit promotion is not a verified mapping')
    finite = np.isfinite(a) & np.isfinite(b)
    equal = a == b
    result = dict(shape=list(a.shape), actual_dtype=str(a.dtype), golden_dtype=str(b.dtype),
        dtype_equal=a.dtype == b.dtype, nonfinite_actual=int((~np.isfinite(a)).sum()),
        nonfinite_golden=int((~np.isfinite(b)).sum()), exact_elements=int(equal.sum()), total_elements=a.size)
    if integer:
        bad = ~equal
        result.update(gate='EXACT_INTEGER', passed=bool(a.size and not bad.any()))
    else:
        result.update(floating_metrics(a, b))
        result.update(gate=POLICY, passed=result['reference_metrics_passed'])
        bad = ~finite | ~equal
        af, bf = a.astype(np.float64), b.astype(np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            delta = np.abs(af-bf)
            if finite.all() and a.size and np.isfinite(delta).all():
                index = int(np.argmax(delta))
                result.update(max_error_coordinate=[int(x) for x in np.unravel_index(index, a.shape)],
                              actual_at_max=float(af.flat[index]), golden_at_max=float(bf.flat[index]))
            if atol is not None:
                bad = ~finite | ~np.isfinite(delta) | (delta > atol + rtol*np.abs(bf))
                result.update(gate=POLICY+'+DECLARED_ALLCLOSE',
                              passed=result['passed'] and not bool(bad.any()), atol=atol, rtol=rtol)
    result['first_different_coordinates'] = np.argwhere(bad)[:8].tolist()
    return result


def read_array(root, spec):
    path = checked(root, spec)
    dtype = np.dtype(DTYPES[spec['dtype']])
    if path.stat().st_size != math.prod(spec['shape']) * dtype.itemsize:
        raise ValueError('tensor extent differs from shape/dtype')
    return np.fromfile(path, dtype=dtype).reshape(spec['shape'])

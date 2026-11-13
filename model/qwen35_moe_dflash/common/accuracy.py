"""Backend-independent device accuracy metrics, accumulated in float64.

KL uses the MindStudio msaccucmp min/max distribution convention (actual ||
golden), not softmax. See docs/target-accuracy-status.md for formulas and scope.
"""
from __future__ import annotations

import math
import operator

import numpy as np

POLICY = 'DEVICE_FIVE_METRICS_V1'
THRESHOLDS = {
    'cosine': {'algorithm': 'CosineSimilarity', 'op': '>', 'value': 0.99},
    'relative_l2': {'algorithm': 'RelativeEuclideanDistance', 'op': '<', 'value': 0.05},
    'kl_divergence': {'algorithm': 'KullbackLeiblerDivergence', 'op': '<', 'value': 0.005},
    'rmse': {'algorithm': 'RootMeanSquareError', 'op': '<', 'value': 1.0},
    'mean_relative_error': {'algorithm': 'MeanRelativeError', 'op': '<', 'value': 1.0},
}
KL_EPSILON = float(np.finfo(np.float32).eps)
RELATIVE_EPSILON = float(np.finfo(np.float64).eps)


def evaluate(values):
    """Strict inequalities on unrounded values; missing metrics never pass."""
    checks = {}
    for key, spec in THRESHOLDS.items():
        value = values.get(key)
        checks[key] = bool(value is not None and math.isfinite(value) and
                          {'>': operator.gt, '<': operator.lt}[spec['op']](value, spec['value']))
    return dict(metric_checks=checks, failed_metrics=[k for k, ok in checks.items() if not ok],
                unavailable_metrics=[k for k in checks if values.get(k) is None],
                reference_metrics_passed=all(checks.values()), numerical_policy=POLICY)


def _chunks(actual, golden, chunk):
    for av, gv in np.nditer([actual, golden], flags=['external_loop', 'buffered', 'zerosize_ok'],
                           op_flags=[['readonly'], ['readonly']], order='C', buffersize=chunk):
        yield av.astype(np.float64), gv.astype(np.float64)


def _distribution_weights(values, low, high):
    # Constant tensors become uniform distributions after normalization.
    weights = (values - low) / (high - low) if high != low else values
    return np.maximum(weights, KL_EPSILON)


def floating_metrics(actual, golden, *, chunk=65536):
    """Compare the full, already aligned logical tensor; never trim padding.

    Three bounded-memory passes keep KL normalization global, independent of
    chunk size. Empty/nonfinite/unavailable results cannot pass. Both all-zero
    vectors use cosine=1, relative_l2=0; a single zero vector is not accepted.
    """
    actual, golden = np.asarray(actual), np.asarray(golden)
    if actual.shape != golden.shape:
        raise ValueError('comparison shape mismatch')
    if actual.dtype.kind != 'f' or golden.dtype.kind != 'f':
        raise ValueError('floating metrics require floating tensors; compare integers exactly')
    if type(chunk) is not int or chunk <= 0:
        raise ValueError('chunk must be a positive integer')
    result = dict.fromkeys(THRESHOLDS)
    result.update(max_abs=None, mean_abs=None, golden_l2=None, error_l2=None,
                  kl_normalization='PER_TENSOR_MINMAX_THEN_SUM; D_KL(ACTUAL||GOLDEN)',
                  kl_epsilon=KL_EPSILON, mean_relative_epsilon=RELATIVE_EPSILON,
                  mean_relative_zero_policy='ABS_GOLDEN_LE_EPS_CONTRIBUTES_ZERO',
                  near_zero_golden_elements=0, metric_issue=None)
    total = actual.size
    if not total:
        result['metric_issue'] = 'EMPTY_TENSOR'
        return dict(result, **evaluate(result))
    square_error = actual_sq = golden_sq = dot = absolute_error = relative_error = 0.0
    max_error = 0.0
    amin = gmin = math.inf
    amax = gmax = -math.inf
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        for a, g in _chunks(actual, golden, chunk):
            if not (np.isfinite(a).all() and np.isfinite(g).all()):
                result['metric_issue'] = 'NONFINITE_TENSOR'
                return dict(result, **evaluate(result))
            delta = a - g
            absolute = np.abs(delta)
            square_error += float(np.sum(delta * delta))
            actual_sq += float(np.sum(a * a))
            golden_sq += float(np.sum(g * g))
            dot += float(np.sum(a * g))
            absolute_error += float(np.sum(absolute))
            max_error = max(max_error, float(absolute.max()))
            valid = np.abs(g) > RELATIVE_EPSILON
            relative_error += float(np.sum(np.abs(a[valid] / g[valid] - 1)))
            result['near_zero_golden_elements'] += int(np.count_nonzero(~valid))
            amin, amax = min(amin, float(a.min())), max(amax, float(a.max()))
            gmin, gmax = min(gmin, float(g.min())), max(gmax, float(g.max()))
        error_norm, golden_norm = math.sqrt(square_error), math.sqrt(golden_sq)
        result.update(max_abs=max_error, mean_abs=absolute_error / total,
                      golden_l2=golden_norm, error_l2=error_norm,
                      relative_l2=error_norm / golden_norm if golden_norm else
                      (0.0 if not error_norm else None), rmse=math.sqrt(square_error / total),
                      mean_relative_error=relative_error / total)
        if actual_sq and golden_sq:
            cosine = dot / math.sqrt(actual_sq) / golden_norm
            result['cosine'] = min(1.0, max(-1.0, cosine)) if math.isfinite(cosine) else None
        elif actual_sq == golden_sq == 0.0:
            # Do not mistake reduction underflow on nonzero values for all-zero tensors.
            result['cosine'] = 1.0 if amin == amax == gmin == gmax == 0.0 else None
        else:
            result['cosine'] = 0.0
        if math.isfinite(amax - amin) and math.isfinite(gmax - gmin):
            asum = gsum = 0.0
            for a, g in _chunks(actual, golden, chunk):
                asum += float(_distribution_weights(a, amin, amax).sum())
                gsum += float(_distribution_weights(g, gmin, gmax).sum())
            if math.isfinite(asum) and math.isfinite(gsum) and asum > 0 and gsum > 0:
                kl = 0.0
                for a, g in _chunks(actual, golden, chunk):
                    aw, gw = _distribution_weights(a, amin, amax), _distribution_weights(g, gmin, gmax)
                    # Log weights avoid division underflow for very small probabilities.
                    kl += float(np.sum((aw / asum) *
                                (np.log(aw) - math.log(asum) - np.log(gw) + math.log(gsum))))
                result['kl_divergence'] = max(0.0, kl) if math.isfinite(kl) and kl >= -1e-12 else None
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
    if any(result[key] is None for key in THRESHOLDS):
        result['metric_issue'] = 'UNDEFINED_OR_NONFINITE_METRIC'
    return dict(result, **evaluate(result))

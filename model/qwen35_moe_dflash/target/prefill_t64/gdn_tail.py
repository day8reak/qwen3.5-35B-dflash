"""CPU replay of the T64 GDN gated RMSNorm and A8W4 output projection."""
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.target.layer_probe.common import checked, metrics, read_array
from qwen35_moe_dflash.target.layer_probe.qlinear_math import dynamic_quant, group_matmul
from qwen35_moe_dflash.target.original_quant import pack_i4zn


def _same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _tensor(root, row, direction, name):
    entries = [r for r in row[direction] if r['name'] == name]
    if len(entries) != 1:
        raise ValueError(f"{row['id']} requires exactly one {direction} tensor named {name}")
    return read_array(root, entries[0])


def _weights(root, row, keys):
    import torch

    saved = torch.load(checked(root, row['weights']), map_location='cpu', weights_only=True)
    if set(saved) != set(keys) or any(not isinstance(t, torch.Tensor) for t in saved.values()):
        raise ValueError(f"{row['id']} has an unexpected captured weight schema")
    return {name: t.detach().contiguous().numpy() for name, t in saved.items()}


def _unpack_i4zn(packed, k, n):
    if (packed.shape != (k, n // 8) or packed.dtype != np.int32
            or k % 256 or n % 16):
        raise ValueError('Expected an I4ZN INT32[K,N/8] group-256 projection')
    # The original packer stores adjacent K as low and high nibbles within K64/N16 tiles.
    raw_bytes = packed.astype('<i4', copy=False).view(np.uint8).reshape(k // 64, n // 16, 16, 32)
    tiles = np.empty((k // 64, n // 16, 16, 64), dtype=np.int8)
    tiles[..., ::2] = (raw_bytes & 15).astype(np.int8)
    tiles[..., 1::2] = (raw_bytes >> 4).astype(np.int8)
    tiles[tiles >= 8] -= 16
    logical = np.ascontiguousarray(tiles.transpose(1, 2, 0, 3).reshape(n, k))
    if not _same_bits(pack_i4zn(logical), packed):
        raise ValueError('Decoded I4ZN weight does not re-pack to the captured bytes')
    return logical


def _gate_norm(x, z, weight, eps, dtype):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('Invalid captured gated RMSNorm epsilon')
    a, gate = x.astype(np.float32), z.astype(np.float32)
    w = weight.astype(np.float32)
    if (w.shape != (a.shape[-1],) or a.shape != gate.shape or
            not all(np.isfinite(t).all() for t in (a, gate, w))):
        raise ValueError('Invalid gated RMSNorm inputs')
    # Stable sigmoid, multiplied by z to form SiLU(z); this norm uses gamma directly.
    exp_neg_abs = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1., exp_neg_abs) / (1. + exp_neg_abs)
    rms = np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + np.float32(eps))
    return ((a / rms) * w * (gate * sigmoid)).astype(dtype)


def _summary(a, b):
    m = metrics(a, b)
    return {name: m[name] for name in ('passed', 'relative_l2', 'cosine', 'rmse', 'max_abs',
                                        'mean_relative_error', 'kl_divergence', 'failed_metrics')}


def _norm_distances(capture, cpu_capture, cpu_board, board):
    """Measure each change using the same captured norm; distances need not add."""
    reference_norm = np.linalg.norm(capture)
    changes = dict(cpu_on_capture_vs_capture=cpu_capture - capture, change_gdr_inputs=cpu_board - cpu_capture,
                   board_vs_same_input_cpu=board - cpu_board, total=board - capture)
    return {k: float(np.linalg.norm(delta) / reference_norm) if reference_norm else None
            for k, delta in changes.items()}


def audit_tail(capture, regions, layer, core_310p, core_math, board_hidden=None, board_stages=None, replay_cores=None):
    """Compare fixed captures at each boundary, then propagate one alternate core to hidden.

    core_math is rounded to the recorded core interface before downstream arithmetic.
    board_hidden, when supplied, must already be bound to this same capture and layer.
    """
    root = Path(capture)
    if isinstance(regions, dict):
        regions = regions['regions']
    def find(point, kind):
        matches = [row for row in regions if row['point'] == point and row['layer'] == layer
                   and row['call'] == 0 and row['phase'] == 'prefill']
        if len(matches) != 1 or matches[0]['kind'] != kind or matches[0]['sequence_length'] != 64:
            raise ValueError(f'Expected a single L{layer} T64 {point} ({kind})')
        return matches[0]
    norm = find('linear_attn_norm', 'norm_gated')
    proj = find('linear_attn_out_proj', 'qlinear')
    z_proj = find('linear_attn_in_proj_z', 'qlinear')
    x = _tensor(root, norm, 'inputs', 'x')
    gate = _tensor(root, norm, 'inputs', 'gate')
    norm_y = _tensor(root, norm, 'outputs', 'golden_y')
    proj_x = _tensor(root, proj, 'inputs', 'x')
    hidden = _tensor(root, proj, 'outputs', 'golden_y')
    z = _tensor(root, z_proj, 'outputs', 'golden_y')
    if (core_310p.ndim != 4 or core_310p.shape != core_math.shape or core_310p.shape[:2] != (1, 64)
            or core_310p.dtype != np.float16 or not np.isfinite(core_math).all()):
        raise ValueError('Expected finite T64 core candidates with matching [1,S,H,D] shape')
    batch, seq, heads, dim = core_310p.shape
    k = heads * dim
    if (any(a.shape != (batch * seq * heads, dim) for a in (x, gate, norm_y))
            or proj_x.shape != (batch, seq, k) or z.shape != proj_x.shape
            or hidden.ndim != 3 or hidden.shape[:2] != (batch, seq)
            or any(a.dtype != np.float16 for a in (x, gate, norm_y, proj_x, z, hidden))):
        raise ValueError('Captured GDN tail shape or dtype mismatch')
    links = {
        'core_to_norm_x': _same_bits(core_310p.reshape(x.shape), x),
        'in_proj_z_to_norm_gate': _same_bits(z.reshape(gate.shape), gate),
        'norm_y_to_out_proj_x': _same_bits(norm_y.reshape(proj_x.shape), proj_x),
    }
    report = dict(links=links, comparisons={}, scope='CPU_FIXED_OPERANDS;CAPTURED_I4ZN_WEIGHTS;FP32_GATED_NORM;GROUP256_FP64_DOT_ACCUMULATION_FINAL_FP16')
    if not all(links.values()):
        return dict(report, status='CAPTURE_LINK_MISMATCH')
    w = _weights(root, norm, ['weight'])['weight']
    weights = _weights(root, proj, ['W_q', 'scale'])
    n = hidden.shape[-1]
    raw = _unpack_i4zn(weights['W_q'], k, n)
    if weights['scale'].shape != (k // 256, n) or weights['scale'].dtype != np.float32:
        raise ValueError('Expected group-256 FP32 out_proj scales')
    scale = np.ascontiguousarray(weights['scale'].T[:, :, None])

    def project(input_):
        q, token_scale, _ = dynamic_quant(input_.reshape(seq, k))
        output, _ = group_matmul(q, token_scale, raw, scale)
        return output.reshape(hidden.shape)

    eps = norm['attrs']['epsilon']
    local_norm = _gate_norm(x, gate, w, eps, x.dtype)
    alternate_norm = _gate_norm(core_math.astype(x.dtype).reshape(x.shape), gate, w, eps, x.dtype)
    local_proj = project(proj_x)
    from_captured_core = project(local_norm)
    from_math_core = project(alternate_norm)
    comparison = report['comparisons']
    comparison['norm_fixed_inputs_vs_310p_norm'] = _summary(local_norm, norm_y)
    comparison['out_proj_fixed_input_vs_310p_hidden'] = _summary(local_proj, hidden)
    comparison['captured_core_cpu_tail_vs_310p_hidden'] = _summary(from_captured_core, hidden)
    comparison['math_core_cpu_norm_vs_310p_norm'] = _summary(alternate_norm, norm_y)
    comparison['math_core_cpu_tail_vs_310p_hidden'] = _summary(from_math_core, hidden)
    comparison['math_core_vs_captured_core_after_same_cpu_tail'] = _summary(from_math_core, from_captured_core)
    if board_hidden is not None:
        if board_hidden.shape != hidden.shape or board_hidden.dtype != np.float16:
            raise ValueError('Board hidden shape or dtype differs from capture')
        comparison['board_hidden_vs_math_core_cpu_tail'] = _summary(board_hidden, from_math_core)
        comparison['board_hidden_vs_captured_core_cpu_tail'] = _summary(board_hidden, from_captured_core)
    if board_stages is not None:
        board_core, board_gate, board_input = (board_stages[n] for n in ('core', 'gate', 'proj_input'))
        for value, expected in ((board_core, core_310p), (board_gate, gate), (board_input, proj_x)):
            if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
                raise ValueError('Board tail output differs from the captured shape/dtype or is nonfinite')
        board_x = board_core.reshape(x.shape)
        core_only = _gate_norm(board_x, gate, w, eps, x.dtype)
        gate_only = _gate_norm(x, board_gate, w, eps, x.dtype)
        board_math_norm = _gate_norm(board_x, board_gate, w, eps, x.dtype)
        # All three controls use the same CPU arithmetic and the same reference.
        comparison['cpu_norm_board_core_only_vs_capture_inputs'] = _summary(core_only, local_norm)
        comparison['cpu_norm_board_gate_only_vs_capture_inputs'] = _summary(gate_only, local_norm)
        comparison['cpu_norm_board_both_vs_capture_inputs'] = _summary(board_math_norm, local_norm)
        delta = (board_input.astype(np.float64) - proj_x).reshape(seq, heads, dim)
        row_energy = np.sum(delta * delta, axis=-1)
        total_energy = row_energy.sum()
        if total_energy:
            token, head = map(int, np.unravel_index(row_energy.argmax(), row_energy.shape))
            def rms(vector):
                return float(np.sqrt(np.mean(np.square(vector, dtype=np.float64))))
            report['largest_norm_error_row'] = dict(
                token=token, head=head, squared_error_share=float(row_energy[token, head] / total_energy),
                captured_core_rms=rms(core_310p[0, token, head]), board_core_rms=rms(board_core[0, token, head]),
                sqrt_epsilon=float(np.sqrt(eps)))
        if replay_cores is not None:
            # Use the captured z for all four cores, including the two CPU recurrences.
            replay_norms = [_gate_norm(c.reshape(x.shape), gate, w, eps, x.dtype) for c in replay_cores]
            vectors = [a.astype(np.float64).reshape(seq, heads, dim) for a in (local_norm, *replay_norms, core_only)]
            effects = report['gdr_replay_norm_distances'] = dict(
                whole=_norm_distances(*vectors), denominator='L2_NORM_OF_CPU_GATED_NORM_ON_CAPTURED_CORE_AND_Z')
            if total_energy:
                effects['largest_row'] = dict(token=token, head=head,
                                              distances=_norm_distances(*(v[token, head] for v in vectors)))
        comparison['board_core_vs_310p_core'] = _summary(board_core, core_310p)
        comparison['board_norm_gate_vs_310p_gate'] = _summary(board_gate, gate)
        comparison['board_out_proj_input_vs_310p'] = _summary(board_input, proj_x)
        comparison['board_norm_vs_cpu_on_board_core_gate'] = _summary(board_input, board_math_norm.reshape(proj_x.shape))
        comparison['board_hidden_vs_cpu_on_board_proj_input'] = _summary(board_hidden, project(board_input))
    return dict(report, status='INSPECTED', epsilon=float(eps),
                quantizer='FP32_ABSMAX_DIV_127_RINT_TO_EVEN', weight='CAPTURED_PACKED_BYTES_I4ZN_ROUNDTRIP')

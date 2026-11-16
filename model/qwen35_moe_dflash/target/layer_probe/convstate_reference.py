"""Explicit BCK history update for an opt-in, full-length whole-GDN comparison.

This is a graph-level reference candidate, not an inferred contract for the
installed LinearAttentionConvstateUpdate. Captured golden bytes are read only.
"""
import numpy as np

from .common import checked, read_array


def tensor_update(mixed_qkv, old_state):
    import torch
    length, width = mixed_qkv.shape[1], old_state.shape[2]
    joined = torch.cat((old_state, mixed_qkv.transpose(1, 2)), dim=2)
    return (joined[:, :, -width:].contiguous(),
            joined[:, :, -(length + width - 1):].contiguous())


def validate_capture(record, capture):
    """Match saved bytes, including the reference's lossless padded widening."""
    components = {r['point']: r for r in record['components']}
    conv = components['convstate_update']
    inputs = {s['name']: s for s in conv['inputs']}
    outputs = {s['name']: s for s in conv['outputs']}
    qspec, sspec = inputs['mixed_qkv'], inputs['old_conv_state']
    conv_input = next(s for s in components['conv1d']['inputs'] if s['name'] == 'padded')
    if any(conv_input[k] != outputs['golden_padded'][k] for k in ('shape', 'dtype', 'bytes', 'sha256')):
        raise ValueError('ConvState padded capture must match the executed convolution input boundary')
    checked(capture, conv_input)
    projected = components['linear_attn_in_proj_qkv']['outputs'][0]
    if any(qspec[k] != projected[k] for k in ('shape', 'dtype', 'bytes', 'sha256')):
        raise ValueError('ConvState capture must use the same executed QKV projection')
    q, old = read_array(capture, qspec), read_array(capture, sspec)
    if (q.ndim != 3 or old.ndim != 3 or q.shape[0] != 1 or old.shape[0] != 1
            or q.shape[2] != old.shape[1] or q.dtype != np.float16 or old.dtype != q.dtype
            or q.shape[1] <= 0 or old.shape[2] != conv['attrs']['kernel_size']
            or old.shape[2] <= 0 or not np.isfinite(q).all() or not np.isfinite(old).all()):
        raise ValueError('reference ConvState requires finite FP16 [1,T,C], [1,C,K], kernel width K')
    length, width = q.shape[1], old.shape[2]
    effective = read_array(capture, record['inputs'][4])
    if (record['sequence_length'] != length or effective.shape != (1,)
            or effective.dtype.kind not in 'iu' or int(effective[0]) != length):
        raise ValueError('reference ConvState candidate only supports full captured sequence length')
    if record['phase'] == 'decode' and length != 1:
        raise ValueError('reference ConvState Decode comparison requires one captured token')
    joined = np.concatenate((old, q.transpose(0, 2, 1)), axis=2)
    expected = {'golden_new_conv_state': joined[:, :, -width:],
                'golden_padded': joined[:, :, -(length + width - 1):]}
    checks = {}
    for name, value in expected.items():
        saved = read_array(capture, outputs[name])
        compared, conversion = value, 'NONE'
        if name == 'golden_padded' and saved.dtype == np.float32:
            # The NZ reference concatenates FP16 data, then casts to the
            # convolution weight dtype. Widen the candidate, never round the
            # saved FP32 golden down; even a one-bit FP32 difference must fail.
            compared, conversion = value.astype(np.float32), 'LOSSLESS_FP16_TO_FP32'
        same = (saved.shape == compared.shape and saved.dtype == compared.dtype
                and saved.tobytes() == compared.tobytes())
        if not same:
            raise ValueError(f'captured {name} does not match reference concat/tail bytes; '
                f'candidate={list(value.shape)}/{value.dtype}, captured={list(saved.shape)}/{saved.dtype}, '
                f'comparison_cast={conversion}; keep original golden')
        checks[name] = dict(shape=list(saved.shape), dtype=str(saved.dtype), equal_bits=True,
            candidate_dtype=str(value.dtype), comparison_cast=conversion,
            comparison_scope='ORIGINAL_CAPTURE_DTYPE; SAVED_BYTES_UNCHANGED')
    return dict(status='PASSED', scope='CAPTURED_QKV_AND_OLD_STATE; NOT_TARGET_EXECUTION',
        batch=1, length=length, history_width=width, checks=checks,
        update='CONCAT_OLD_BCK_AND_QKV_BCT; LAST_K_STATE; LAST_T_PLUS_K_MINUS_1_CONV_INPUT',
        control_policy='LINEAR_UPDATE_TENSOR_UNUSED; FULL_LENGTH_ONLY', golden='UNCHANGED')


def check_graph(graph):
    """Check presence of the explicit candidate, independently of GDN dtype."""
    nodes = {n['name']: n for n in graph['op']}
    if any(n['type'] == 'LinearAttentionConvstateUpdate' for n in nodes.values()):
        raise ValueError('reference ConvState graph still contains the installed state-update operator')
    nets = [n for n in nodes.values() if n['type'] == 'NetOutput']
    convs = [n for n in nodes.values() if n['type'] == 'Conv2D']
    if len(nets) != 1 or len(convs) != 1:
        raise ValueError('reference whole GDN requires one NetOutput and one Conv2D')

    def ancestors(edge):
        pending, seen = [edge.rsplit(':', 1)[0]], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            node = nodes[name]
            pending.extend(e.rsplit(':', 1)[0] for e in node.get('input', []) if e and not e.endswith(':-1'))
        return {nodes[name]['type'] for name in seen}

    state_types = ancestors(nets[0]['input'][1])
    padded_types = ancestors(convs[0]['input'][0])
    slices = {'Slice', 'SliceD', 'StridedSlice', 'StridedSliceV2'}
    if (not state_types & slices or not padded_types & slices
            or not padded_types & {'Concat', 'ConcatD', 'ConcatV2'}):
        raise ValueError('reference ConvState AIR must retain state slicing and concat/slice convolution input')
    return dict(status='PASSED', convstate_operator_count=0,
        state_ancestor_types=sorted(state_types), padded_ancestor_types=sorted(padded_types),
        scope='SOURCE_AIR_STRUCTURE; TARGET_NUMERICS_PENDING')

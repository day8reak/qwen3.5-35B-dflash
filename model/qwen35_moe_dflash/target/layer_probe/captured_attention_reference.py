"""Qualify one original FullAttention capture using its same-call core audit.

This never substitutes a failed module replay or changes captured golden files.
Only the attention core has an independent mathematical reference; the enclosing
module remains the original execution, with unverified repeat reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .attention_reference import reference_math, unpack
from .common import KIND, identity, metrics, read_array

POLICY = 'ORIGINAL_FULL_ATTENTION_CAPTURE_WITH_AUDITED_CORE'
SCOPE = 'SAME_CALL_SOURCE_MODULE_OUTPUTS; CORE_MATH_AUDITED; WHOLE_MODULE_MATH_UNVERIFIED'
_IDENTITY = ('shape', 'dtype', 'bytes', 'sha256')


def _mask_binding(module_mask, core_mask):
    """Classify observed mask representations; never rewrite either capture."""
    def stats(value):
        finite = value[np.isfinite(value)]
        return dict(shape=list(value.shape), dtype=str(value.dtype),
            zero=int(np.count_nonzero(value == 0)),
            negative_zero=int(np.count_nonzero((value == 0) & np.signbit(value))),
            negative_infinity=int(np.isneginf(value).sum()),
            positive_infinity=int(np.isposinf(value).sum()), nan=int(np.isnan(value).sum()),
            finite_min=float(finite.min()) if finite.size else None,
            finite_max=float(finite.max()) if finite.size else None)

    diagnostic = dict(module=stats(module_mask), core=stats(core_mask))

    def reject(reason):
        raise ValueError('module/core mask binding failed: ' + reason + '; mask_diagnostics=' +
                         json.dumps(diagnostic, allow_nan=False))

    if module_mask.dtype not in (np.dtype('float16'), np.dtype('float32')) or core_mask.dtype != np.float16:
        reject('expected module FP16/FP32 and core FP16')
    if module_mask.shape != core_mask.shape:
        reject('shape mismatch')
    with np.errstate(over='ignore', invalid='ignore'):
        converted = module_mask.astype(np.float16)
    diagnostic['cpu_fp16'] = stats(converted)
    value_diff = converted != core_mask
    bit_diff = converted.view(np.uint16) != core_mask.view(np.uint16)
    diagnostic.update(value_mismatches=int(value_diff.sum()), bit_mismatches=int(bit_diff.sum()),
        negative_inf_to_min_finite=int(np.count_nonzero(np.isneginf(converted) & (core_mask == -65504))),
        zero_to_nonzero=int(np.count_nonzero((converted == 0) & (core_mask != 0))))
    def item(value):
        number = float(value)
        return dict(value=number if np.isfinite(number) else str(number), bits=value.tobytes().hex())
    diagnostic['first_differences'] = [dict(coordinate=list(map(int, coord)),
        module=item(module_mask[tuple(coord)]), cpu_fp16=item(converted[tuple(coord)]),
        core=item(core_mask[tuple(coord)])) for coord in np.argwhere(bit_diff)[:8]]
    if any(np.isnan(v).any() or np.isposinf(v).any() for v in (module_mask, core_mask, converted)):
        reject('NaN/+Inf mask is unsupported')
    if not bit_diff.any():
        mode = 'SOURCE_CAST_TO_FLOAT16'
    elif not value_diff.any():
        mode = 'EQUAL_CAST_VALUES_WITH_SIGNED_ZERO_DIFFERENCE'
    else:
        # A bounded representation alternative for this fresh causal capture.
        # It is accepted only after an additional same-QKV math check below.
        if (module_mask.ndim != 4 or module_mask.shape[:2] != (1, 1)
                or module_mask.shape[2] <= 1 or module_mask.shape[3] < module_mask.shape[2]):
            reject('sentinel alternative requires fresh B1 causal Prefill')
        causal = np.arange(module_mask.shape[3])[None, :] <= np.arange(module_mask.shape[2])[:, None]
        source, observed = module_mask[0, 0], core_mask[0, 0]
        if (not np.all(source[causal] == 0) or not np.all(np.isneginf(source[~causal]))
                or not np.all(observed[causal] == 0)
                or not np.all(np.isneginf(observed[~causal]) | (observed[~causal] == -65504))):
            reject('changed mask positions/values outside the reviewed 0/-Inf/-65504 representation')
        mode = 'OBSERVED_CAUSAL_NEG_INF_TO_FP16_MIN_FINITE'
    return converted, dict(transformation=mode, diagnostics=diagnostic,
        extra_reference_check_required=bool(bit_diff.any()))


def audited_capture_reference(capture, manifest, original, components):
    """Read and revalidate the parent Prefill audit; return evidence, not outputs.

    ``capture`` is the directory containing the boundary manifest. Its parent
    must contain the original Prefill manifest and the saved attention audit.
    The caller must explicitly select this policy; no fallback is performed.
    """
    capture = Path(capture).resolve(strict=True)
    parent = capture.parent
    capture_path, parent_path = capture/'manifest.json', parent/'manifest.json'
    stored = json.loads(capture_path.read_text())
    prefill = json.loads(parent_path.read_text())
    if stored != manifest:
        raise ValueError('captured-audited requires the unchanged on-disk boundary manifest')
    if (manifest.get('artifact_kind') != KIND or manifest.get('status') != 'CAPTURED'
            or manifest.get('reference') != 'ORIGINAL_310P_INFERENCE' or manifest.get('errors')
            or manifest.get('entrypoint') != 'DIRECT_ORIGINAL_SINGLE_PREFILL'):
        raise ValueError('captured-audited requires a complete original single-Prefill boundary capture')
    link = prefill.get('boundary_capture', {})
    if (prefill.get('artifact_kind') != 'qwen35-target-prefill-fixture-v1'
            or prefill.get('status') not in ('CAPTURED', 'PARTIAL') or prefill.get('error')
            or prefill.get('phase') != 'prefill' or prefill.get('source_wrapper_calls') != 1
            or prefill.get('golden_source') != 'ORIGINAL_310P_SAME_INVOCATION'
            or link.get('status') != 'CAPTURED'
            or not isinstance(link.get('path'), str)
            or (parent/link['path']).resolve() != capture_path):
        raise ValueError('captured-audited parent is not the original Prefill owning this boundary capture')
    sources = prefill.get('sources', [])
    if not sources or any(item not in manifest.get('sources', []) for item in sources):
        raise ValueError('Prefill and boundary capture source identities differ')
    modules = [r for r in manifest['regions'] if r['point'] == 'full_attention_layer']
    cores = [r for r in manifest['regions'] if r['point'] == 'fused_attention']
    if len(modules) != 1 or modules[0] != original or len(cores) != 1:
        raise ValueError('captured-audited currently requires one unambiguous FullAttention invocation')
    if (original.get('phase') != 'prefill' or original.get('call') != 0
            or original.get('sequence_length') != prefill.get('sequence_length')
            or original.get('sequence_length', 0) <= 1):
        raise ValueError('FullAttention capture is not this parent fresh Prefill invocation')
    by_point = {r['point']: r for r in components}
    if len(by_point) != len(components) or by_point.get('fused_attention') != cores[0]:
        raise ValueError('ambiguous or unrelated FullAttention components')
    for component in components:
        if (component not in manifest['regions'] or any(component.get(k) != original.get(k)
                for k in ('layer', 'call', 'phase', 'sequence_length'))):
            raise ValueError('FullAttention component is not from the same captured invocation')

    arrays, bindings = {}, []

    def array(root, spec):
        # read_array verifies file containment, hash, size and tensor extent.
        key = (str(root), spec['path'], *(str(spec[k]) for k in _IDENTITY))
        if key not in arrays:
            arrays[key] = read_array(root, spec)
        return arrays[key]

    def named(rows, name):
        found = [s for s in rows if s['name'] == name]
        if len(found) != 1:
            raise ValueError('missing/duplicate captured tensor: ' + name)
        return found[0]

    def bind(label, left, right, left_root=capture, right_root=capture):
        array(left_root, left)
        array(right_root, right)
        if any(left[k] != right[k] for k in _IDENTITY):
            raise ValueError('captured-audited tensor identity mismatch: ' + label)
        bindings.append(dict(binding=label, left_path=left['path'], right_path=right['path'],
            **{key: left[key] for key in _IDENTITY}))

    # Verify every tensor consumed by this module assembly, including snapshots
    # outside the independently audited core. Weights are checked by export.
    for component in [original, *components]:
        for spec in component['inputs'] + component['outputs']:
            array(capture, spec)
    module_inputs = {s['name']: s for s in original['inputs']}
    module_outputs = {s['name']: s for s in original['outputs']}
    for projection in ('q_proj', 'k_proj', 'v_proj'):
        bind('module_hidden_to_' + projection, module_inputs['hidden_states'],
             by_point['self_attn_' + projection]['inputs'][0])
    bind('o_proj_to_module_hidden', by_point['self_attn_o_proj']['outputs'][0],
         module_outputs['golden_hidden'])

    audit = prefill.get('attention_audit', {})
    calls = audit.get('calls', [])
    if (audit.get('status') != 'PASSED' or audit.get('scope') != 'ATTENTION_BOUNDARY_ONLY'
            or len(calls) != 1 or calls[0].get('status') != 'PASSED' or calls[0].get('call') != 0):
        raise ValueError('captured-audited requires the single same-invocation attention audit to pass')
    audited = calls[0]
    core = cores[0]
    if (sorted(s['name'] for s in core['inputs']) != sorted(
            ('query', 'atten_mask', 'block_table', 'key', 'value', 'all_seq_lengths_q'))
            or set(core['attrs']) != {'actual_seq_lengths_q', 'actual_seq_lengths_kv',
                'num_heads', 'num_key_value_heads', 'block_size', 'input_layout',
                'scale_value', 'inner_precise'}):
        raise ValueError('captured core input/attribute set differs from the reviewed source contract')
    core_inputs = {s['name']: s for s in core['inputs']}
    audit_names = {'query': 'query', 'key': 'key', 'value': 'value',
                   'mask': 'atten_mask', 'table': 'block_table'}
    if sorted(s['name'] for s in audited.get('inputs', [])) != sorted(audit_names):
        raise ValueError('attention audit input set changed')
    for audit_name, core_name in audit_names.items():
        bind('audit_' + audit_name + '_to_core', named(audited['inputs'], audit_name),
             core_inputs[core_name], left_root=parent)
    actual_spec = audited['actual']
    bind('audit_actual_to_core_output', actual_spec,
         named(core['outputs'], 'golden_attention_nz'), left_root=parent)
    # Bind the observed conversion; CPU and device cast bytes are not assumed identical.
    module_mask_spec, core_mask_spec = module_inputs['attention_mask'], core_inputs['atten_mask']
    module_mask = array(capture, module_mask_spec)
    core_mask = array(capture, core_mask_spec)
    source_mask, mask_binding = _mask_binding(module_mask, core_mask)
    mask_binding.update(binding='module_mask_to_core', left=module_mask_spec, right=core_mask_spec)
    bindings.append(mask_binding)
    bind('module_length_to_core', module_inputs['allQLen'], core_inputs['all_seq_lengths_q'])
    for role, suffix in (('key', 'k'), ('value', 'v')):
        cache = by_point['cache_' + suffix]
        bind(role + '_old_module_to_cache', module_inputs['old_' + role + '_cache'],
             named(cache['inputs'], 'cache'))
        bind(role + '_cache_update_to_core', named(cache['outputs'], 'golden_updated_cache'),
             core_inputs[role])
        bind(role + '_core_to_module_return', core_inputs[role],
             module_outputs['golden_' + role + '_cache'])
        parent_name = role + '_cache_' + str(original['layer'])
        bind(role + '_parent_to_module_input', named(prefill['inputs'], parent_name),
             module_inputs['old_' + role + '_cache'], left_root=parent)
        bind(role + '_module_to_parent_output', module_outputs['golden_' + role + '_cache'],
             named(prefill['outputs'], 'golden_' + parent_name), right_root=parent)

    values = {name: array(capture, spec) for name, spec in core_inputs.items()}
    attrs = core['attrs']
    length = values['all_seq_lengths_q']
    if (length.dtype != np.int64 or length.shape != (1,)
            or int(length[0]) != original['sequence_length']
            or values['query'].shape[2] != int(length[0])
            or attrs.get('input_layout') != 'BNSD' or attrs.get('inner_precise') != 2
            or attrs.get('actual_seq_lengths_q') != [int(length[0])]
            or attrs.get('actual_seq_lengths_kv') != [values['atten_mask'].shape[-1]]):
        raise ValueError('captured core lengths/layout differ from the fresh Prefill audit contract')
    for name in ('num_heads', 'num_key_value_heads', 'block_size', 'scale_value'):
        if attrs[name] != original['attrs'][name]:
            raise ValueError('module/core attribute mismatch: ' + name)
    prepared = reference_math(values['query'], values['key'], values['value'],
        values['atten_mask'], values['block_table'], int(length[0]),
        heads=attrs['num_heads'], kv_heads=attrs['num_key_value_heads'],
        block_size=attrs['block_size'], scale=attrs['scale_value'])
    if prepared['output'] is None:
        raise ValueError('independent core reference is undefined for this capture')
    if mask_binding['extra_reference_check_required']:
        source_prepared = reference_math(values['query'], values['key'], values['value'],
            source_mask, values['block_table'], int(length[0]),
            heads=attrs['num_heads'], kv_heads=attrs['num_key_value_heads'],
            block_size=attrs['block_size'], scale=attrs['scale_value'])
        source_output, core_output = source_prepared['output'], prepared['output']
        if (source_output is None or not np.isfinite(source_output).all()
                or not np.isfinite(core_output).all() or source_output.shape != core_output.shape
                or source_output.dtype != core_output.dtype or source_output.tobytes() != core_output.tobytes()):
            raise ValueError('module/core mask representations change the same-QKV reference output; mask_diagnostics=' +
                             json.dumps(mask_binding['diagnostics'], allow_nan=False))
        mask_binding.update(reference_outputs_equal_bits=True,
            equivalence_scope='THIS_CAPTURE_QKV_ONLY; NUMPY_FP64_THEN_FP16; NOT_GENERAL_DEVICE_CAST_EQUIVALENCE',
            source_mask_reference=source_prepared['evidence'])
    recorded_reference = array(parent, audited['independent_reference'])
    fresh_reference = prepared['output']
    if (recorded_reference.shape != fresh_reference.shape
            or recorded_reference.dtype != fresh_reference.dtype
            or recorded_reference.tobytes() != fresh_reference.tobytes()):
        raise ValueError('saved independent reference differs from recomputed captured core reference')
    recorded_evidence = audited.get('reference', {})
    for name in ('status', 'arithmetic', 'query_shape', 'num_heads', 'num_key_value_heads',
                 'head_dim', 'scale_value', 'kv_length', 'mask_stride'):
        if recorded_evidence.get(name) != prepared['evidence'][name]:
            raise ValueError('audit/reference contract differs on ' + name)
    actual = array(parent, actual_spec)
    if actual.dtype != np.float16 or actual.shape != values['query'].shape:
        raise ValueError('audited core output is not its packed FP16 query shape')
    compared = metrics(unpack(actual, attrs['num_heads']), fresh_reference)
    if not compared['passed']:
        raise ValueError('same-invocation core output fails recomputed independent attention metrics')
    replay = [r for r in manifest.get('attention_math_replays', []) if all(
        r.get(k) == original[k] for k in ('layer', 'call', 'phase'))]
    return dict(policy=POLICY, status='QUALIFIED', scope=SCOPE,
        original_outputs_unchanged=True, independent_whole_module_reference=False,
        reference_repeatability='NOT_ESTABLISHED_BY_SAME_INVOCATION_AUDIT',
        prefill_manifest=identity(parent_path, parent),
        capture_manifest=identity(capture_path, parent),
        layer=original['layer'], call=original['call'], phase=original['phase'],
        sequence_length=original['sequence_length'], bindings=bindings,
        independent_reference=prepared['evidence'], recomputed_core_metrics=compared,
        saved_independent_reference=audited['independent_reference'],
        prior_module_replays=[{k: r.get(k) for k in ('policy', 'status', 'error', 'cache_replay_bitwise_equal')}
                             for r in replay])

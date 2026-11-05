"""CPU hypotheses from retained single-output prefixes; no model execution.

Substitution uses an independently compiled RMS prefix, not the internal input
of the compiled Q prefix. A closer match is evidence for a hypothesis, never
proof of the target's accumulator precision or of the full model's first error.
"""
import hashlib
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _read_air
from .accuracy import metrics
from .hidden_layer_probe import GE_TYPES, STEPS, LayerTrace, tolerance_details
from .torch_export import _schema


def compare_arrays(actual, expected, atol, rtol):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError('math audit requires matching finite tensor contracts')
    return dict(metrics(actual, expected, atol, rtol),
        bitwise_equal=actual.tobytes() == expected.tobytes(),
        tolerance=tolerance_details(actual, expected, atol, rtol, examples=8))


def trig_candidates(phase, kind, dtype):
    """Separately test input phase rounding and output rounding, using NumPy."""
    if kind not in ('Sin', 'Cos') or phase.dtype != np.float32 or not np.isfinite(phase).all():
        raise ValueError('RoPE audit requires finite source FP32 phase')
    with np.errstate(over='ignore'):
        half_phase = phase.astype(np.float16).astype(np.float32)
    if not np.isfinite(half_phase).all():
        raise ValueError('RoPE phase exceeds finite FP16 range')
    fn = np.sin if kind == 'Sin' else np.cos
    original, rounded = fn(phase), fn(half_phase)
    return {name: value.astype(dtype) for name, value in (
        ('source_fp32', original), ('fp16_phase', rounded),
        ('fp16_output', original.astype(np.float16)),
        ('fp16_phase_and_output', rounded.astype(np.float16)))}


def _load_source(air, evidence):
    raw, _ = _read_air(air/'export.air')
    if hashlib.sha256(raw).hexdigest() != evidence['source_air_sha256']:
        raise ValueError('math audit AIR differs from the executed prefix source')
    exported = read_json(air/'manifest.json')
    trace = LayerTrace(raw, _schema((air/'air-schema.pb').read_bytes(), exported['schema']))
    trace.select_layer()
    attrs = [_attributes(trace.nodes[n]) for n in trace.ordered if trace.nodes[n]['type'] == 'FileConstant']
    records = evidence['source_weights']
    if ({a['file_path']['s'] for a in attrs} != set(records)
            or sum(r['bytes'] for r in records.values()) > 256 << 20):
        raise ValueError('math audit source weight inventory differs or exceeds bound')
    weights = {}
    for a in attrs:
        name = a['file_path']['s']
        if records[name]['path'] != name:
            raise ValueError('math audit weight path differs from source')
        path = checked_file(air, records[name])
        weights[name] = np.fromfile(path, dtype=GE_TYPES[a['dtype']['i']]).reshape(a['shape']['list']['i'])
    return trace, weights


def _inputs(fixture, cases):
    inputs, contracts = {}, {}
    for case in cases:
        for spec in case['inputs']:
            name = spec['name']
            contract = (spec['dtype'], tuple(spec['shape']), spec['bytes'], spec['fixture_sha256'])
            if name in contracts:
                if contracts[name] != contract:
                    raise ValueError('independent prefixes have inconsistent source inputs: '+name)
                continue
            # analyze() already verified each input hash against the launch.
            array = np.fromfile(fixture/case['case']/f'input_{spec["index"]}.bin', dtype=spec['dtype'])
            if array.nbytes != spec['bytes']:
                raise ValueError('retained source input extent changed')
            inputs[name] = array.reshape(spec['shape']); contracts[name] = contract
    return inputs


def _snapshot(fixture, native, case, measured):
    if not measured.get('repeat_bitwise_stable') or len(case['outputs']) != 1:
        raise ValueError('math audit requires stable single-output snapshots')
    spec = case['outputs'][0]
    values = []
    for path in (fixture/case['case']/f'golden_{spec["index"]}.bin', native/case['case']/'0.bin'):
        array = np.fromfile(path, dtype=spec['dtype'])
        if array.nbytes != spec['bytes'] or not np.isfinite(array).all():
            raise ValueError('invalid retained math audit tensor: '+str(path))
        values.append(array.reshape(spec['shape']))
    return values


def audit(air_export, fixture, native, meta, comparison, *, progress=lambda _: None):
    """Called after launch-bound fixture verification and all-call comparison."""
    evidence = meta['source_evidence']
    progress('verifying source AIR and original weight hashes; CPU analysis only')
    trace, weights = _load_source(Path(air_export).resolve(), evidence)
    rms_edge = trace.half_output(trace.norm('draft.layers.0.input_layernorm.weight'))
    q_points = {c['checkpoint']['source_edge'] for c in meta['cases']
                if c['checkpoint']['source_type'] == 'MatMul'
                and c['checkpoint'].get('source_weight') == 'weight/draft_layers_0_self_attn_q_proj_weight'}
    if len(q_points) != 1:
        raise ValueError('cannot identify the retained layer0 Q projection')
    q_edge = q_points.pop()
    if rms_edge.rsplit(':', 1)[0] not in trace.dependencies(q_edge):
        raise ValueError('Q projection does not depend on the selected input RMS')
    trig_edges = [n+':0' for n in trace.ordered if trace.nodes[n]['type'] in ('Sin', 'Cos')]
    if len(trig_edges) != 4:
        raise ValueError('expected both query/context Sin and Cos source branches')
    targets = [rms_edge, q_edge, *trig_edges]
    measured = {c['case']: c for c in comparison['cases']}
    contexts = []
    for step in STEPS:
        progress(f'context {step}: Q input propagation and four Sin/Cos rounding comparisons')
        cases = [c for c in meta['cases'] if c['source_case'] == step]
        by_edge = {c['checkpoint']['source_edge']: c for c in cases}
        if len(by_edge) != len(cases) or any(e not in by_edge for e in targets):
            raise ValueError('required retained checkpoint absent or ambiguous')
        if any(by_edge[e]['case'] not in measured for e in targets):
            raise ValueError('required checkpoint was not recovered')
        inputs = _inputs(fixture, cases)
        base = trace.evaluate_layer(inputs, weights, targets=targets)
        snapshots = {e: _snapshot(fixture, native, by_edge[e], measured[by_edge[e]['case']]) for e in targets}
        cmp = lambda a, b: compare_arrays(a, b, meta['atol'], meta['rtol'])
        replay = {e: cmp(base[e], snapshots[e][0]) for e in targets}
        if any(not r['passed'] for r in replay.values()):
            raise ValueError('CPU source replay no longer matches retained reference at the original tolerance')
        # Shape/dtype checked by replay comparison above and the single-output
        # contract. Preserve all source views, weight layout and MatMul flags.
        observed_rms, observed_q = snapshots[rms_edge][1], snapshots[q_edge][1]
        substituted = trace.evaluate_layer(inputs, weights, targets=[q_edge],
                                           substitutions={rms_edge: observed_rms})[q_edge]
        query = dict(input_edge=rms_edge, output_edge=q_edge,
            rms_device_vs_retained_reference=cmp(observed_rms, snapshots[rms_edge][0]),
            q_device_vs_retained_reference=cmp(observed_q, snapshots[q_edge][0]),
            q_device_vs_cpu_source_rms=cmp(observed_q, base[q_edge]),
            q_device_vs_cpu_device_rms=cmp(observed_q, substituted),
            q_cpu_device_rms_vs_retained_reference=cmp(substituted, snapshots[q_edge][0]))
        trigs = []
        for edge in trig_edges:
            node, _ = trace.node_port(edge)
            phase = base[node['input'][0]]
            golden, actual = snapshots[edge]
            candidates = trig_candidates(phase, node['type'], actual.dtype)
            trigs.append(dict(source_edge=edge, phase_edge=node['input'][0],
                source_type=node['type'], shape=list(actual.shape), phase_dtype=str(phase.dtype),
                phase_max_abs=float(np.max(np.abs(phase), initial=0)),
                phase_fp16_rounding_max_abs=float(np.max(np.abs(phase-phase.astype(np.float16).astype(np.float32)), initial=0)),
                device_vs_retained_reference=cmp(actual, golden),
                device_vs_candidates={name: cmp(actual, value) for name, value in candidates.items()},
                placement=by_edge[edge]['placement']))
        contexts.append(dict(source_case=step, source_replay_vs_retained_reference=replay,
                             query_projection=query, trigonometry=trigs))
    return dict(status='ANALYZED', model_execution='NOT_RUN', model_compilation='NOT_RUN',
        numerical_validation='NOT_EVALUATED', scope='CPU_Q_INPUT_PROPAGATION_AND_ROPE_ROUNDING_HYPOTHESES',
        original_model_sha256=evidence['original_model_sha256'], source_air_sha256=evidence['source_air_sha256'],
        snapshot='CALL_0_AFTER_VERIFIED_TEN_CALL_BITWISE_STABILITY',
        source_numerical_validation=comparison['numerical_validation'],
        comparison=dict(atol=meta['atol'], rtol=meta['rtol'], acceptance='UNCHANGED; NO_RELATIVE_L2_GATE'),
        contexts=contexts, limitations=[
            'The substituted RMS comes from another compiled prefix, not the internal Q-prefix input.',
            'Prefix cutting may change fusion/kernel choice; CPU BLAS reduction order may differ.',
            'Sin/Cos candidates isolate source phase/output rounding, not the device implementation of transcendental functions.',
            'Closer or exact candidate agreement does not prove internal instruction or accumulator precision.',
            'This audit does not change goldens, acceptance gates or establish full-model first divergence.'])

"""Compare full feature-pack or offline-weight candidates with a retained baseline."""
import numpy as np
from .accuracy import read_snapshot
from .accuracy_diagnose import tensor_difference
from .air_feature_pack import CONTRACT
from .kv_diagnose import _load, _same_inputs
from qwen35_moe_dflash.common.hashing import sha256_file


def compare_result(path, *, offline_weights=False):
    path, result, fixture, meta, analysis = _load(path)
    if meta.get('feature_pack') != CONTRACT or not result.get('replay_source'):
        raise ValueError('expected a completed full feature-pack candidate replay')
    old_path, original, baseline, old, old_analysis = _load(path.parent/result['replay_source'])
    replay = meta.get('candidate_replay', {})
    if offline_weights:
        from .offline_weights import check_replay_pair
        from qwen35_moe_dflash.execution.runtime_config import read_json
        check_replay_pair(old, meta, read_json(baseline/'pytorch-abi.json'), read_json(fixture/'pytorch-abi.json'))
        if replay.get('policy') != 'draft-offline-weights-replay-v1':
            raise ValueError('expected an offline-weight candidate replay')
    elif old.get('feature_pack') is not None or meta.get('offline_weights') is not None:
        raise ValueError('feature-pack comparison requires an unpacked baseline without offline weights')
    if (result.get('model') != meta['model'] or original.get('model') != old['model']
            or meta.get('precision_repair') != old.get('precision_repair')
            or result.get('baseline_model') != old['model'] or replay.get('baseline_model') != old['model']
            or replay.get('source_fixture_sha256') != sha256_file(baseline/'manifest.json')
            or result.get('output_initialization') != original.get('output_initialization')):
        raise ValueError('baseline model, RoPE repair, fixture or output initialization differs')
    _same_inputs(fixture, meta, baseline, old)
    rows = []
    for mode in ('isolated', 'feedback'):
        for case in meta['cases']:
            i, count = case['index'], case['proposals']
            a = read_snapshot(path.parent/'native'/f'{mode}-{i}.bin', meta['outputs'])
            b = read_snapshot(old_path.parent/'native'/f'{mode}-{i}.bin', old['outputs'])
            for j, spec in enumerate(meta['outputs']):
                if j < 2 and not count:
                    continue
                x, y = (a[j][:, :count], b[j][:, :count]) if j < 2 else (a[j], b[j])
                rows.append(dict(mode=mode, index=i, tensor=spec['name'],
                    finite=bool(np.isfinite(x).all() and np.isfinite(y).all()), **tensor_difference(x, y)))
    equivalent = all(r['finite'] and r['bitwise_equal'] for r in rows)
    timings = dict(status='UNAVAILABLE', reason='both runs need matching runner/runtime and preparation timing')
    a, b = analysis.get('prepared_call_latency'), old_analysis.get('prepared_call_latency')
    if (a and b and a['timing_scope'] == b['timing_scope']
            and result.get('runner') == original.get('runner') and result.get('runner')
            and result.get('runtime') == original.get('runtime') and result.get('runtime')):
        pairs = []
        for mode in ('isolated', 'feedback'):
            for phase in ('catchup', 'proposal'):
                x, y = a[mode][phase], b[mode][phase]
                if x['calls'] != y['calls']:
                    raise ValueError('prepared-call timing coverage differs')
                pairs.append(dict(mode=mode, phase=phase, baseline=y, candidate=x,
                    mean_saved_ms=y['mean_ms']-x['mean_ms'] if x['calls'] else None))
        timings = dict(status='COMPARED', timing_scope=a['timing_scope'], pairs=pairs,
            limitation='SEPARATE_MODEL_LOADS; ORIGINAL_DIAGNOSTIC_SCHEDULE; NOT_WARM_REPEATED_OR_END_TO_END_BENCHMARK')
    return dict(status='ANALYZED', numerical_validation=analysis['numerical_validation'],
        experiment='OFFLINE_STATIC_WEIGHT_TRANSPOSE' if offline_weights else 'CPU_FEATURE_PACK',
        baseline_equivalence='PASSED' if equivalent else 'FAILED', compared_regions=len(rows),
        scope='ALL_ACTIVE_TOP1_HIDDEN_AND_FULL_KV_BANKS; BOTH_ORIGINAL_SEQUENCES',
        candidate_model=meta['model'], baseline_model=old['model'],
        input_verification='ORIGINAL_LOGICAL_BYTES_AND_GOLDENS_HASH_EQUAL_BY_SEMANTIC_NAME',
        differences=[r for r in rows if not r['bitwise_equal'] or not r['finite']],
        candidate_case_counts=analysis['case_counts'], baseline_case_counts=old_analysis['case_counts'],
        candidate_relative_l2=analysis['relative_l2_ranges'], baseline_relative_l2=old_analysis['relative_l2_ranges'],
        feedback_repeats=analysis['feedback_repeats'], prepared_timings=timings,
        production_promotion='NOT_PERFORMED')

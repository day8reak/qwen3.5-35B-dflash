"""Bind RoPE, feature-pack or offline-weight candidates to retained accuracy data."""
from copy import deepcopy
from pathlib import Path
import shutil

from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save
from .air_rope import POLICY
from .torch_abi import bind_candidate
from .runtime import validate_abi


def rebind_candidate(fixture, original, compiled, source, air, ddk_root, output):
    """No Torch, new features, goldens or comparison thresholds are generated."""
    fixture, compiled, output = map(Path, (fixture, compiled, output))
    manifest = read_json(compiled/'manifest.json')
    receipt = read_json(compiled/'compilation/report.json', 8 << 20)
    repair = manifest.get('precision_repair', {})
    if (original.get('frontend') != 'pytorch' or repair.get('policy') != POLICY
            or repair.get('status') != 'APPLIED' or repair.get('trig_count') != 4
            or repair != receipt.get('air_preparation', {}).get('rope')):
        raise ValueError('candidate replay requires the completed RoPE precision repair')
    old_binding = read_json(fixture/'pytorch-abi.json')
    print('draft-accuracy: binding candidate ABI to the original source and retained reference', flush=True)
    candidate = bind_candidate(compiled, air, source, ddk_root, output)
    binding = read_json(output/'pytorch-abi.json')
    offline_replay = candidate.get('offline_weights') is not None
    if offline_replay:
        from .offline_weights import check_replay_pair
        check_replay_pair(original, candidate, old_binding, binding)
    elif original.get('feature_pack') is not None:
        raise ValueError('packed-to-packed replay requires a verified offline-weight candidate')
    for key in ('source_air_sha256', 'prepared_sha256', 'source_input_roles', 'cache_edges', 'reference_weights'):
        if binding[key] != old_binding[key]:
            raise ValueError('candidate changed retained source/reference binding: '+key)
    for key in ('context_rows', 'cache_capacity', 'max_draft_tokens', 'ddk_id', 'host_embedding'):
        if candidate.get(key) != original.get(key):
            raise ValueError('candidate changed retained geometry or input policy: '+key)
    for direction in ('inputs', 'outputs'):
        old = {s['name']: (s['dtype'], s['shape'], s['bytes']) for s in original[direction]}
        new = {s['name']: (s['dtype'], s['shape'], s['bytes']) for s in candidate[direction]}
        if direction == 'inputs' and candidate.get('feature_pack') is not None and not offline_replay:
            from .air_feature_pack import CONTRACT, LOGICAL_SHAPE, PHYSICAL_SHAPE
            if (original.get('feature_pack') is not None or candidate['feature_pack'] != CONTRACT
                    or original.get('precision_repair') != repair
                    or old.get('target_features') != ('float16', LOGICAL_SHAPE, 524288)):
                raise ValueError('feature-pack replay requires the original unpacked RoPE baseline')
            old['target_features'] = ('float16', PHYSICAL_SHAPE, 524288)
        if old != new or len(old) != len(original[direction]) or len(new) != len(candidate[direction]):
            raise ValueError('candidate changed semantic tensor contracts: '+direction)
    result = deepcopy(original)
    result.update(inputs=candidate['inputs'], outputs=candidate['outputs'],
        model=dict(candidate['model'], path=str((compiled/candidate['model']['path']).resolve())),
        host_embedding_binding=candidate.get('host_embedding_binding'),
        precision_repair=repair, input_files=[])
    if candidate.get('feature_pack') is not None:
        result['feature_pack'] = candidate['feature_pack']
    if offline_replay:
        result['offline_weights'] = candidate['offline_weights']
    validate_abi(result)
    old_slots = {s['name']: s['index'] for s in original['inputs']}
    old_files = {(r['case'], r['index']): r for r in original['input_files']}
    golden_paths = [f'golden/{c["index"]}/{name}.bin' for c in original['cases']
                    for name in ([*(['top1', 'hidden'] if c['proposals'] else []),
                                  *(f'kv_{i}' for i in range(12))])]
    expected_bytes = sum(r['bytes'] for r in original['input_files'])+sum((fixture/p).stat().st_size for p in golden_paths)
    # Fixture, upload archive and returned native snapshots coexist.
    required = expected_bytes*2+4*len(original['cases'])*sum(s['bytes'] for s in result['outputs'])+(256 << 20)
    if shutil.disk_usage(output).free < required:
        raise OSError(f'candidate replay and recovery need at least {required} free host bytes')
    def copy(relative, destination, expected=None):
        src, dst = fixture/relative, output/destination
        digest = sha256_file(src)
        if expected and (digest != expected['sha256'] or src.stat().st_size != expected['bytes']):
            raise ValueError('retained replay input changed before copying: '+relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(dst) != digest:
            raise ValueError('candidate replay copy differs from retained bytes: '+relative)
        return dict(path=destination, bytes=dst.stat().st_size, sha256=digest)
    for case in original['cases']:
        index = case['index']
        for spec in result['inputs']:
            record = old_files[index, old_slots[spec['name']]]
            copied = copy(record['path'], f'inputs/{index}/{spec["index"]}.bin', record)
            result['input_files'].append(dict(copied, case=index, index=spec['index']))
    goldens = [copy(p, p) for p in golden_paths]
    rows = [f"QWEN35_INCREMENTAL_ACCURACY {3 if result.get('feature_pack') else 2}",
            f"geometry {result['context_rows']} {result['cache_capacity']} {result['max_draft_tokens']}"]
    for direction in ('inputs', 'outputs'):
        rows += [' '.join((direction[:-1], str(s['index']), s['name'], s['dtype'], ','.join(map(str, s['shape']))))
                 for s in result[direction]]
    rows += [f"case {c['index']} {c['start']} {c['valid']} {c['anchor']} {c['proposals']}" for c in result['cases']]
    (output/'plan.txt').write_text('\n'.join(rows)+'\n')
    result['candidate_replay'] = dict(policy='draft-offline-weights-replay-v1' if offline_replay else
        result.get('feature_pack', {}).get('policy', POLICY), baseline_model=original['model'],
        source_fixture=str(fixture.resolve()), source_fixture_sha256=sha256_file(fixture/'manifest.json'),
        goldens=goldens, reference_generation='NOT_RUN', input_generation='NOT_RUN',
        input_binding='ORIGINAL_BYTES_REMAPPED_BY_VERIFIED_SEMANTIC_NAME',
        comparison='ORIGINAL_SCHEDULE_GOLDENS_AND_GATES_UNCHANGED')
    _save(output/'manifest.json', result)
    return output, result

"""T64 YAML-prompt references and public outputs for one original CPro L0–L39 prefill."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ..layer_probe.common import identity, read_tensor, save_json, sha, sized_file
from .chain_reference import YAML_BASIS, LAYERS
from .decoder_four_boundary import output_roles, output_shape, state_roles
from .l0_split import YAML_CPU_BASIS, score


def read(path):
    return json.loads(path.read_bytes())


def capture_and_cpu(run, reference, source, config):
    from .chain_reference import capture
    from .split_cpu_reference import prepare

    if reference == run / 'reference' and not reference.exists():
        with tempfile.TemporaryDirectory(dir=run, prefix='capture-') as tmp:
            staged = Path(tmp) / '310p'
            capture(source, config, None, staged, states=True, yaml_prompt=True)
            staged.rename(reference)
    ref = read(reference / 'manifest.json')
    states = {role for i in range(LAYERS) for role in state_roles(i)}
    valid = ref['valid_rows']
    if ((ref['reference'], ref['status'], ref['physical_rows'], ref['output_pos'])
            != (YAML_BASIS, 'CAPTURED', 64, valid-1) or not 1 <= valid <= 64
            or [b['layer'] for b in ref['blocks']] != list(range(LAYERS))
            or not states.issubset(ref['outputs'])
            or ref['source_sha256'] != sha(source / 'inference_segment.py')
            or ref['config_sha256'] != sha(source / config)):
        raise ValueError(f'Expected a matching 40-layer 310P capture with all states at {reference}')
    prepare(run, '40', reference, source, per_layer_air=False)


def pack_goldens(reference, run, work):
    ref = read(reference / 'manifest.json')
    capture_sha = sha(reference / 'manifest.json')
    records = {'cpu': {}, '310p': {}}
    for backend in records:
        (work / 'golden' / backend).mkdir(parents=True)
    for i in range(LAYERS):
        cpu = run / f'L{i:02d}/cpu'
        meta = read(cpu / 'manifest.json')
        if (meta['basis'], meta['mode'], meta['layer'], meta['capture_sha256']) != (YAML_CPU_BASIS, '40', i, capture_sha):
            raise ValueError(f'{cpu}: reference belongs to another layer or capture')
        roles = state_roles(i) + (('logits',) if i == LAYERS - 1 else ())
        for name in roles:
            for backend, folder, outputs in (('cpu', cpu, meta['outputs']), ('310p', reference, ref['outputs'])):
                row = outputs[name]
                if tuple(row['shape']) != output_shape(name) or row['dtype'] not in ('float16', 'float32'):
                    raise ValueError(f'{backend}: unexpected {name} descriptor {row}')
                dest = Path('golden') / backend / (name + '.bin')
                shutil.copyfile(sized_file(folder, row), work / dest)
                records[backend][name] = dict(row, path=str(dest))
    token_file = work / 'golden/token-text.json'
    shutil.copyfile(sized_file(reference, ref['token_text_map']), token_file)
    return records, identity(token_file, work)


def compare(bundle, replay, write_matrices):
    meta = read(bundle / 'manifest.json')
    result = read(replay / 'result.json')
    io = read(replay / 'io.json')
    compiled = read(replay.parent / 'omc/manifest.json')
    if (result['status'] != 'EXECUTED' or result['exitcode'] != 0
            or result['input_hashes'] != [r['sha256'] for r in io['inputs']]
            or result['omc_sha256'] != compiled['primary']['sha256']
            or compiled['air_manifest_sha256'] != sha(bundle / 'manifest.json')):
        raise ValueError(f'{replay}: no complete execution of this 40-layer OMC')
    roles = output_roles(io['outputs'], meta['boundary']['outputs'])
    pairs = {b: {r: {} for r in ('cpu', '310p', 'cpro')} for b in ('cpu', '310p', 'cpro')}
    predictions = {}
    valid = meta['fixture']['valid_rows']
    pieces = read(sized_file(bundle, meta['tokens']))['pieces']
    if len(pieces) != output_shape('logits')[-1]:
        raise ValueError('Vocabulary and single-token decode table sizes differ')
    for name, spec, record in zip(roles, io['outputs'], result['outputs'], strict=True):
        shape = output_shape(name)
        actual = read_tensor(sized_file(replay, record), spec).reshape(shape)
        values = {}
        for backend in ('cpu', '310p'):
            golden = meta['goldens'][backend][name]
            values[backend] = read_tensor(sized_file(bundle, golden), golden)
        values['cpro'] = actual
        if name == 'logits':
            for b, logits in values.items():
                token = int(logits.argmax())
                predictions[b] = dict(id=token, text=pieces[token]) if np.isfinite(logits).all() else None
        for b, x in values.items():
            for r, y in values.items():
                pairs[b][r].update(score({name: x}, {name: y}, {name: (list(shape), spec['dtype'])},
                                         reference_is_310p=r == '310p', valid_rows=valid))
    matrices = replay.parent / 'matrices.csv'
    print(f'[40/prefill] 汇总 logits 与各层状态的三端精度矩阵 → {matrices}', flush=True)
    write_matrices(matrices, pairs)
    scores = pairs['cpro']['310p']
    failed = [name for name, scopes in scores.items() if any(not m['passed'] for m in scopes.values())]
    report = dict(status='FAILED_ACCURACY' if failed else 'PASSED', physical_rows=64, valid_rows=valid,
                  prompt=meta['fixture']['prompt'], output_pos=valid-1, predictions=predictions, failed=failed, cpro_vs_310p=scores,
                  matrices=identity(matrices, replay.parent), conv_state_control=meta['fixture']['conv_state_control'])
    save_json(replay.parent / 'comparison.json', report)
    print(f'First token: {predictions}\n{report["status"]} {replay.parent / "comparison.json"}', flush=True)
    return not failed

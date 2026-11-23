"""Independent L0 RMSNorm, SiLU and quantized projection, all on native 310P operands."""
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ...common.artifacts import share_for_transfer
from ..layer_probe.common import KIND, checked, identity, metrics, read_array, save_json, sha
from .decode_cache_control import graph_message
from .decode_cache_trace import _record
from .gdn_tail_probe import BASIS as TAIL_BASIS, SOURCE_CASE, case_id, cut, tensor
from .standalone_gdr import FIELDS

BASIS = 'NATIVE_310P_L0_OPERATOR_INPUTS_AND_OUTPUTS'
ROLES = {'rms_output': 'core', 'silu_output': 'z', 'out_proj_only': 'proj_input'}
BACKENDS = ('cpu', '310p', 'cpro')


def boundary(air, stage):
    _, case, folder, spec = _record(air, case_id(stage))
    operand, = spec['source_inputs']
    role = ROLES[stage]
    if (operand['semantic_name'] != role or set(spec['source_data_edges']) != {role}
            or case['layer'] != 0 or case['point'] != stage or case['kind'] != 'gdn_tail_prefix'):
        raise ValueError(f'{stage}: unexpected L0 operand or case')
    _, graph = graph_message(checked(folder, spec['air']).read_bytes(), 'NetOutput')
    data, = [op for op in graph.op if op.type == 'Data']
    net, = [op for op in graph.op if op.type == 'NetOutput']
    types = [op.type for op in graph.op]
    if (operand['name'] != data.name or list(net.input) != [spec['source_edge']]
            or 'ChunkGatedDeltaRule' in types or any(op.subgraph_name for op in graph.op)
            or types.count('QuantBatchMatmulV4444') != int(stage == 'out_proj_only')
            or types.count('DynamicQuant') != int(stage == 'out_proj_only')):
        raise ValueError(f'{stage}: graph does not match its standalone export contract')
    return dict(air=spec['air'], role=role, input_edge=spec['source_data_edges'][role],
                input_shape=operand['shape'], input_dtype=operand['dtype'], output_edge=spec['source_edge'],
                output_shape=spec['golden']['shape'], output_dtype_310p=spec['golden']['dtype'],
                weights=spec['weight_evidence'],
                nodes=[dict(name=n.name, type=n.type, inputs=list(n.input)) for n in graph.op])


def review(air):
    boundaries = {stage: boundary(air, stage) for stage in ROLES}
    print(json.dumps(boundaries, indent=2), flush=True)
    return boundaries


def native_and_cpu(source, config):
    import torch.nn.functional as F
    import yaml
    from ..layer_probe.capture import cpu
    from ..original_quant import OriginalQuantWeights
    from .cpu_reference import linear
    from .gdn_tail_reference import capture_l0
    from .gdr_stage_capture import observe_norm

    config = (source/config).resolve(strict=True)
    quant_path = (source/yaml.safe_load(config.read_text())['quanted_pth']).resolve(strict=True)

    def evaluate(layer, orig, info):
        x, z, proj = (orig[k] for k in ('norm_input', 'norm_gate', 'out_proj_input'))
        dev = orig['device']
        taps = {}
        with ExitStack() as cleanup:
            observe_norm(layer, taps, cleanup)
            gated = layer.norm(x.to(dev), z.to(dev))
        if cpu(gated).numpy().tobytes() != orig['norm_return'].numpy().tobytes():
            raise ValueError('Tapping RMSNorm and SiLU changed the original 310P gated return')
        y = cpu(layer.out_proj(proj.to(dev))).numpy()
        if y.tobytes() != orig['output'].tobytes():
            raise ValueError('Replaying the exact projection operand changed the native 310P output')
        # The stage boundary is before the gated product. gamma is the actual L0 gated-norm weight.
        gamma = orig['gamma'].float()
        key = 'target.layers.0.linear_attn.out_proj.weight'
        weight, scale = OriginalQuantWeights(quant_path).logical(key, (2048, 4096))
        info.update(cpu_projection=dict(checkpoint=str(quant_path), key=key,
                    weight_sha256=hashlib.sha256(weight.tobytes()).hexdigest(),
                    scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(),
                    policy='per-token A8 RNE, signed W4 group256, int32 dots, FP32 scales/sum, FP16 output'))
        # Every CPU operand is the host copy taken directly at that operator's native entry.
        return {
            'rms_output': (x.numpy(), taps['norm']['rms_output'].numpy(),
                           F.rms_norm(x.float(), (128,), gamma, orig['epsilon']).numpy()),
            'silu_output': (z.numpy(), taps['norm']['silu_output'].numpy(), F.silu(z.float()).numpy()),
            'out_proj_only': (proj.numpy(), y, linear(proj, weight, scale).numpy()),
        }

    return capture_l0(source, config, evaluate)


def prepare(run, source_air, source, config):
    source_manifest = json.loads((source_air/'manifest.json').read_text())
    run.mkdir(parents=True, exist_ok=True)
    target = run/'air'
    if target.exists():
        raise FileExistsError(f'{target} exists; reuse with board or select a new run')
    reuse = source_manifest.get('golden_source') == TAIL_BASIS
    if reuse:
        for stage in ROLES:
            boundary(source_air, stage)
    else:
        _, case, original_folder, original = _record(source_air, SOURCE_CASE)
        if (case['point'] != 'gdn_layer' or
                (original['physical_rows'], original['valid_rows'], original['reference_position']) != (64, 19, 18)):
            raise ValueError('Expected original L0 attention AIR with T64 and valid19')
        raw = checked(original_folder, original['air']).read_bytes()
    datasets, info = native_and_cpu(source, config)
    if reuse:
        previous = source_manifest['provenance']['native_310p']
        for k in ('config_sha256', 'norm_source', 'gamma_sha256', 'epsilon', 'proj_class'):
            if info[k] != previous[k]:
                raise ValueError(f'Native 310P {k} differs from the source batch')
    else:
        actual = datasets['out_proj_only'][1].reshape(64, 2048)[:19]
        expected = read_array(original_folder, original['golden']).reshape(64, 2048)[:19]
        if actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
            raise ValueError('Native L0 projection does not reproduce the original AIR golden on valid19')
    cases = []
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        root = Path(tmp)/'air'
        root.mkdir()
        for ordinal, (stage, (x, native, host)) in enumerate(datasets.items()):
            name = case_id(stage)
            dest = root/'models'/name
            dest.mkdir(parents=True)
            if not all(np.isfinite(a.reshape(64, -1)[:19]).all() for a in (x, native, host)):
                raise ValueError(f'{stage}: nonfinite native input or output in valid19')
            if reuse:
                _, _, folder, spec = _record(source_air, name)
                for record in [spec['air'], *spec['weight_evidence'].values()]:
                    src = checked(folder, record)
                    dst = dest/record['path']
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    checked(dest, record)
                operand, = spec['source_inputs']
                if x.dtype.name != operand['dtype'] or native.dtype.name != spec['golden']['dtype']:
                    raise ValueError(f'{stage}: native dtype differs from the original fixture')
                spec['source_inputs'] = [dict(operand, **tensor(dest, operand['name'], x.reshape(operand['shape'])))]
                spec['golden'] = tensor(dest, stage, native.reshape(spec['golden']['shape']))
            else:
                if stage == 'rms_output':
                    x = x.reshape(1, 64, 32, 128)
                elif stage == 'out_proj_only':
                    x = x.reshape(64, 4096)
                spec = cut(raw, stage, native, {ROLES[stage]: x}, dest, original_folder)
            spec.update(cpu=tensor(dest, 'cpu', host.reshape(spec['golden']['shape'])), reference_basis=BASIS)
            save_json(dest/'export.json', spec)
            cases.append(dict(id=name, layer=0, call=0, phase='prefill', ordinal=ordinal, kind='gdn_tail_prefix',
                              point=stage, output_name=stage, status='EXPORTED', export=identity(dest/'export.json', root)))
        manifest = dict(artifact_kind=KIND, status='EXPORTED', cases=cases, golden_source=BASIS,
                        reference_model_layers=1, native_310p=info, source_manifest_sha256=sha(source_air/'manifest.json'),
                        scope='Independent L0 operators on native T64 inputs. Compare valid19 and individual tokens.')
        save_json(root/'manifest.json', manifest)
        save_json(root/'boundaries.json', {stage: boundary(root, stage) for stage in ROLES})
        manifest['boundaries'] = identity(root/'boundaries.json', root)
        save_json(root/'manifest.json', manifest)
        root.rename(target)
    share_for_transfer(target)
    print(target/'boundaries.json', flush=True)


def matrix_rows(stage, outputs):
    width = 2048 if stage == 'out_proj_only' else 4096
    values = {k: outputs[k].reshape(64, width)[:19] for k in BACKENDS}
    for token in (None, *range(19)):
        scope = 'valid19' if token is None else f'token{token}'
        scores = {}
        for actual in BACKENDS:
            a = values[actual]
            for reference in BACKENDS:
                b = values[reference]
                scores[actual, reference] = metrics(a, b) if token is None else metrics(a[token], b[token])
        for field in FIELDS:
            for actual in BACKENDS:
                yield dict(operator=stage, scope=scope, metric=field, actual=actual,
                           **{ref: scores[actual, ref][field] for ref in BACKENDS})


def summarize(run):
    from .segment_trace_compare import board_case

    air = run/'air'
    manifest = json.loads((air/'manifest.json').read_text())
    fixture = json.loads((run/'execution/fixture/manifest.json').read_text())
    if manifest['golden_source'] != BASIS or fixture['source_evidence']['manifest_sha256'] != sha(air/'manifest.json'):
        raise ValueError('Board execution and native 310P inputs belong to different batches')
    passed = True
    path = run/'matrices.csv'
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        staged = Path(tmp)/path.name
        with staged.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['operator','scope','metric','actual', *BACKENDS])
            writer.writeheader()
            for stage, role in ROLES.items():
                _, _, folder, spec = _record(air, case_id(stage))
                operand, = spec['source_inputs']
                inputs, gold, board, _ = board_case(run, case_id(stage))
                if (set(inputs) != {role} or inputs[role][0] != operand['dtype']
                        or inputs[role][2] != operand['sha256']):
                    raise ValueError(f'{stage}: board input differs from the native 310P operand')
                native = read_array(folder, spec['golden'])
                if gold[stage].dtype != native.dtype or gold[stage].tobytes() != native.tobytes():
                    raise ValueError(f'{stage}: board fixture and 310P output disagree')
                outputs = {'cpu': read_array(folder, spec['cpu']), '310p': native, 'cpro': board[stage]}
                for row in matrix_rows(stage, outputs):
                    writer.writerow(row)
                    if row['scope'] not in ('valid19', 'token18'):
                        continue
                    if row['metric'] in ('cosine','relative_l2','error_ratio'):
                        if row['actual'] == 'cpu':
                            print(f"\n{stage}  {row['scope']}  {row['metric']}\nactual\\reference          CPU           310P           CPro")
                        print(row['actual'].ljust(15), *(format(row[k], '14.9g') if row[k] is not None
                                                       else '     undefined' for k in BACKENDS))
                    if row['metric'] == 'reference_metrics_passed' and row['actual'] == 'cpro':
                        passed &= bool(row['310p'])
        staged.replace(path)
    print(path, flush=True)
    return passed

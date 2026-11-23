"""Same-entry L0 references at both SiLU calls and the quantized output projection."""
import json
from pathlib import Path
import tempfile

import numpy as np

from ...common.artifacts import share_for_transfer
from ..layer_probe.common import read_array, save_json, sha
from .attention_cpu_silu import CASE
from .decode_cache_trace import _record
from .gdn_tail_probe import case_id, tensor
from .gdr_stage_data import shapes
from .silu_split import VIEWS

WIDTH = dict(conv_silu_in=8192, conv_silu_out=8192, gate_silu_in=4096,
             gate_silu_out=4096, out_proj_input=4096, out_proj_output=2048)


def valid_rows(stage, value):
    x = np.asarray(value)
    width = WIDTH[stage]
    if x.dtype not in (np.float16, np.float32):
        raise ValueError(f'{stage}: unexpected dtype {x.dtype}')
    if stage.startswith('conv_'):
        if x.shape not in VIEWS['conv']:
            raise ValueError(f'{stage}: expected channel-first T64, got {x.shape}')
        return x.reshape(width, 64).T[:19]
    for t in (19, 64):
        views = ([[t, width], [1, t, width], [1, 1, t, width]] if width == 2048 else
                 [*shapes('core', t), [1, 1, t, width]])
        if list(x.shape) in views:
            return x.reshape(t, width)[:19]
    raise ValueError(f'{stage}: unexpected token/head layout {x.shape}')


def cpu_values(source, config, inputs):
    import torch
    import yaml
    from .. import prefill_fixture as pf
    from .cpu_reference import CpuPrefill, verify_floating_weights

    cfg = yaml.safe_load(config.read_text())
    model = (source/cfg['model_file']).resolve(strict=True)
    weights = (source/cfg['quanted_pth']).resolve(strict=True)
    facts = dict(model=str(model), weights=str(weights), quantization='CPU per-token A8, packed W4')
    impl = pf.import_source(source)
    with torch.inference_mode(), pf.working_directory(source), pf.hf_config_prefix(model, 4, facts):
        wrapper = pf.build_four_layers(impl, model, facts)
        wrapper.to(dtype=torch.float16)
        verify_floating_weights(wrapper.model, model)
        cpu = CpuPrefill(wrapper, weights)
        x = {k: torch.from_numpy(v) for k, v in inputs.items()}
        taps = {}
        cpu.gdn(x['hidden'], wrapper.model.model.layers[0].linear_attn, 'layers.0.linear_attn',
                (x['conv'], x['state']), stages=taps, valid_length=int(x['effective_length'].item()))
        return dict(conv_silu_in=taps['conv'].transpose(1, 2).numpy(),
                    conv_silu_out=taps['conv_silu_output'].transpose(1, 2).numpy(),
                    gate_silu_in=taps['silu_input'].numpy(), gate_silu_out=taps['silu_output'].numpy(),
                    out_proj_input=taps['out_proj_input'].numpy(), out_proj_output=taps['hidden'].numpy()), facts


def live_310p(source, config, inputs):
    from ..layer_probe.capture import bound, cpu
    from .gdn_tail_reference import capture_l0
    from .gdr_stage_capture import observe_norm
    from .qkv_silu import STAGE, observe_conv

    def observe(layer, snapshot, cleanup):
        def check_entry(mod, args, kwargs):
            supplied = bound(mod, args, kwargs)
            conv, state = supplied['past_key_values']
            actual = dict(supplied, hidden=supplied['hidden_states'], conv=conv, state=state)
            for name, expected in inputs.items():
                value = cpu(actual[name]).numpy()
                if name in ('conv', 'state') and not np.any(value):
                    value = value.astype(expected.dtype)
                if (expected.shape, expected.dtype, expected.tobytes()) != (value.shape, value.dtype, value.tobytes()):
                    raise ValueError(f'310P L0 input differs from the staged run: {name}')
        cleanup.callback(layer.register_forward_pre_hook(check_entry, with_kwargs=True).remove)
        observe_conv(layer, snapshot, cleanup)
        observe_norm(layer, snapshot, cleanup)

    def evaluate(layer, snap, info):
        return dict(conv_silu_in=snap['conv_out'].numpy(), conv_silu_out=snap[STAGE].numpy(),
                    gate_silu_in=snap['norm']['silu_input'].numpy(), gate_silu_out=snap['norm']['silu_output'].numpy(),
                    out_proj_input=snap['out_proj_input'].numpy(), out_proj_output=snap['output'])
    return capture_l0(source, config, evaluate, observe=observe)


def archived_310p(tests, case_digest, config, source):
    """Only reuse standalone native operands/outputs linked to this exact parent L0 export."""
    from .gdn_op_matrix import BASIS
    from .qkv_silu import BASIS as CONV_BASIS
    manifests, parents = [], set()
    for path in sorted(tests.glob('*/air/manifest.json')):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        cases = doc.get('cases', [])
        if not isinstance(cases, list) or not all(isinstance(c, dict) for c in cases):
            continue
        if any(c.get('id') == CASE and c.get('export', {}).get('sha256') == case_digest for c in cases):
            parents.add(sha(path))
        if doc.get('golden_source') in (BASIS, CONV_BASIS):
            manifests.append((path, doc))
    values, used = {}, {}
    for path, doc in manifests:
        native = doc['native_310p']
        if (doc.get('source_manifest_sha256') not in parents or native['config_sha256'] != sha(config)
                or native['driver_sha256'] != sha(source/'inference_segment.py')):
            continue
        for row in doc['cases']:
            mapping = {'qkv_conv_silu': ('conv_silu_in', 'conv_silu_out'),
                       'silu_output': ('gate_silu_in', 'gate_silu_out'),
                       'out_proj_only': ('out_proj_input', 'out_proj_output')}
            step = row['point']
            if step not in mapping or row['id'] != case_id(step):
                continue
            try:
                _, _, folder, spec = _record(path.parent, row['id'])
                operand, = spec['source_inputs']
                x, y = read_array(folder, operand), read_array(folder, spec['golden'])
            except (OSError, ValueError) as exc:
                print(f'Skipping incomplete native {step} at {path}: {exc}', flush=True)
                continue
            if step == 'silu_output':
                x = x.astype(np.float32)  # Native norm calls SiLU on gate.float().
            if step == 'qkv_conv_silu':
                x = x.astype(native['native_input_dtype'])
            for name, value in zip(mapping[step], (x, y)):
                if name in values and not same_bits(valid_rows(name, values[name]), valid_rows(name, value)):
                    raise ValueError(f'Different archived 310P results for {name}: {used[name]} and {path}')
                values[name], used[name] = value, str(path)
    return values, used


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_bundle(root, backend, digest):
    folder = root/backend
    record = json.loads((folder/'manifest.json').read_text())
    if (record['backend'], record['case_sha256'], set(record['stages'])) != (backend, digest, set(WIDTH)):
        raise ValueError(f'Wrong L0 fixture or stages at {folder}')
    values = {k: read_array(folder, spec) for k, spec in record['stages'].items()}
    if any(a.shape != (19, WIDTH[k]) or a.dtype not in (np.float16, np.float32) for k, a in values.items()):
        raise ValueError(f'Bad valid19 shape or dtype at {folder}')
    return values


def prepare_refs(backend, air, root, source, config, tests):
    _, case, folder, spec = _record(air, CASE)
    digest = case['export']['sha256']
    config = (source/config).resolve(strict=True)
    dest = root/backend
    if dest.exists():
        load_bundle(root, backend, digest)
        old = json.loads((dest/'manifest.json').read_text())
        if old['config_sha256'] != sha(config) or old['driver_sha256'] != sha(source/'inference_segment.py'):
            raise ValueError(f'Driver or config changed; select a new reference directory: {dest}')
        print('Reusing verified references:', dest, flush=True)
        share_for_transfer(dest)
        return
    inputs = {s['semantic_name']: read_array(folder, s) for s in spec['source_inputs']}
    if inputs['effective_length'].tolist() != [19] or np.any(inputs['conv']) or np.any(inputs['state']):
        raise ValueError('Expected the L0 valid19 fresh prefill fixture')
    if backend == 'cpu':
        values, facts = cpu_values(source, config, inputs)
    else:
        values, used = archived_310p(tests, digest, config, source)
        facts = dict(archived=used)
        if set(values) != set(WIDTH):
            print('One native L0 forward to fill:', sorted(set(WIDTH)-values.keys()), flush=True)
            live, facts['native'] = live_310p(source, config, inputs)
            for k, old in values.items():
                if not same_bits(valid_rows(k, old), valid_rows(k, live[k])):
                    raise ValueError(f'Archived and fresh native {k} disagree')
            values.update({k: v for k, v in live.items() if k not in values})
        if not same_bits(valid_rows('out_proj_output', values['out_proj_output']),
                         valid_rows('out_proj_output', read_array(folder, spec['golden']))):
            raise ValueError('Native projection does not reproduce the staged valid19 golden')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        tmp = Path(temp)/backend
        tmp.mkdir()
        stages = {k: tensor(tmp, k, valid_rows(k, values[k])) for k in WIDTH}
        save_json(tmp/'manifest.json', dict(backend=backend, case_sha256=digest, details=facts, stages=stages,
                  config_sha256=sha(config), driver_sha256=sha(source/'inference_segment.py')))
        tmp.rename(dest)
    share_for_transfer(dest)
    print(backend, dest, flush=True)

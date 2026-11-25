"""Capture the real decoder entry, layer-specific history, and matching output rows."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from .. import prefill_fixture as pf
from ..layer_probe.capture import bound, cpu
from ..layer_probe.common import read_array, save_json, sha
from .chain_reference import VALID19_IDS_SHA256, run_segment_prefill
from .cpu_reference import CpuPrefill, verify_floating_weights
from .decoder_cases import groups, output_names, state_roles
from .decoder_four_boundary import output_shape
from .segment_prefix import tensor_record
from .split_cpu_reference import decoder


def capture(root, mode, phase, source, config):
    schedule = groups(mode)
    layers = [i for group in schedule for i in group]
    rows, valid = (64, 19) if phase == 'prefill' else (1, 1)
    wanted = {k for group in schedule for k in output_names(group, group[-1] == layers[-1])}
    report = dict(mode=mode, phase=phase, layers=layers, inputs={}, outputs={},
                  config_sha256=sha(config), source_sha256=sha(source/'inference_segment.py'))
    for name in ('inputs', '310p'):
        (root/name).mkdir()
    active = False
    prefill_token = None
    prefill_cache = {}
    last_row = None

    def record(group, name, value):
        folder = '310p' if group == 'outputs' else 'inputs'
        if name in report[group]:
            raise ValueError(f'Repeated {group} {name}')
        row = tensor_record(root/folder, name, value)
        row['path'] = folder + '/' + row['path']
        report[group][name] = row

    class Complete(Exception):
        pass

    with ExitStack() as stack:
        def hook(mod, fn, before=False):
            register = mod.register_forward_pre_hook if before else mod.register_forward_hook
            stack.callback(register(fn, with_kwargs=True).remove)

        def arm(frame):
            driver = frame.f_back
            while driver and 'model2' not in driver.f_locals:
                driver = driver.f_back
            if driver is None:
                raise ValueError('Missing segmented inference driver')
            ctx = driver.f_locals
            if ctx['args'].pseudo_quant or ctx['args'].quant_bits != 4:
                raise ValueError('Expected original W4 inference')
            for key in ('model_file', 'quanted_pth'):
                report[key] = str(Path(ctx['cfg'][key]).resolve(strict=True))
            first, second = frame.f_locals['model_wrapper'], ctx['model2']
            split = report['split_layer'] = len(first.model.model.layers)
            if split + len(second.model.model.layers) != 40:
                raise ValueError('Expected the complete 40-layer segmented reference')

            def enter_wrapper(mod, args, kwargs):
                nonlocal active
                v = bound(mod, args, kwargs)
                ids = cpu(v['input_ids'])
                if tuple(ids.shape) == (1, 64):
                    if hashlib.sha256(ids.numpy().astype('<i4').tobytes()).hexdigest() != VALID19_IDS_SHA256:
                        raise ValueError('Prompt differs from the T64 valid19 fixture')
                elif tuple(ids.shape) != (1, 1) or int(ids.item()) != prefill_token:
                    raise ValueError('Decode input differs from the observed prefill prediction')
                active = ids.shape[1] == rows
                if not active:
                    return
                if (list(v['allQLen']) != [19 if rows == 64 else 20]
                        or cpu(v['effective_length']).reshape(-1).tolist() != [valid]):
                    raise ValueError('Unexpected query or effective length')
                positions = list(range(64)) if rows == 64 else [19]
                for key in ('position_ids', 'new_kv_cache_pos'):
                    if cpu(v[key]).flatten().tolist() != positions:
                        raise ValueError(f'Unexpected {phase} {key}')
                for key in ('effective_length', 'position_ids', 'new_kv_cache_pos', 'attention_mask'):
                    record('inputs', key, v[key])
                record('inputs', 'all_length', torch.as_tensor(v['allQLen'], dtype=torch.int64))
                record('inputs', 'input_ids', ids)
            hook(first, enter_wrapper, True)
            for start, wrapper in ((0, first), (split, second)):
                for i, block in enumerate(wrapper.model.model.layers, start):
                    if i not in layers:
                        continue
                    def enter(mod, args, kwargs, i=i):
                        if not active:
                            return
                        v = bound(mod, args, kwargs)
                        hidden = v['hidden_states']
                        if tuple(hidden.shape) != (1, rows, 2048) or hidden.dtype != torch.float16:
                            raise ValueError(f'L{i} reference hidden descriptor changed')
                        if i == layers[0]:
                            record('inputs', 'hidden', hidden)
                        if i % 4 == 3 and not torch.equal(cpu(mod.self_attn.block_table), torch.arange(64)[None]):
                            raise ValueError(f'L{i} requires a measured sequential page table')
                        for name, state in zip(state_roles(i), v['past_key_values'], strict=True):
                            if tuple(state.shape) != output_shape(name) or state.dtype != torch.float16:
                                raise ValueError(f'{name}: unexpected cache descriptor {state.dtype} {list(state.shape)}')
                            if rows == 64 and torch.count_nonzero(cpu(state)):
                                raise ValueError(f'Unexpected nonzero prefill input {name}')
                            record('inputs', name, state)
                            if rows == 1:
                                digest = report['inputs'][name]['sha256']
                                report.setdefault('decode_cache_matches_prefill', {})[name] = digest == prefill_cache[name]

                    def leave(mod, args, kwargs, result, i=i):
                        nonlocal last_row
                        if not active and phase == 'decode':
                            for role, value in zip(state_roles(i), result[1], strict=True):
                                prefill_cache[role] = hashlib.sha256(cpu(value).numpy().tobytes()).hexdigest()
                        if active:
                            name = f'hidden_{i}'
                            if name in wanted:
                                record('outputs', name, result[0])
                            if i == layers[-1]:
                                last_row = cpu(result[0][:, valid-1, :])
                            for role, value in zip(state_roles(i), result[1], strict=True):
                                record('outputs', role, value)
                    hook(block, enter, True)
                    hook(block, leave)
                    if f'out_proj_{i}' in wanted:
                        def projection(mod, args, kwargs, result, i=i):
                            if active:
                                record('outputs', f'out_proj_{i}', result[0])
                        hook(block.self_attn if i % 4 == 3 else block.linear_attn, projection)

            def readout(mod, args, kwargs, result):
                nonlocal prefill_token
                token = int(result[1][0, 0, 0].item())
                if not active:
                    prefill_token = token
                    return
                v = bound(mod, args, kwargs)
                with torch.no_grad():
                    x = last_row.to(v['hidden_states'].device)
                    logits = mod.model.lm_head(mod.model.model.norm(x)).reshape(1, 1, mod.vocab_size).float()
                if layers[-1] == 39 and logits.flatten()[token] != logits.max():
                    raise ValueError('Full head disagrees with the token from the live driver')
                record('outputs', 'logits', logits)
                tokenizer = ctx['tokenizer']
                pieces = [None] * mod.vocab_size
                ids = sorted({i for i in tokenizer.get_vocab().values() if 0 <= i < mod.vocab_size})
                for start in range(0, len(ids), 1024):
                    batch = ids[start:start+1024]
                    texts = tokenizer.batch_decode([[i] for i in batch], skip_special_tokens=False)
                    for index, piece in zip(batch, texts, strict=True):
                        pieces[index] = piece
                save_json(root/'tokens.json', pieces)
                report['prefill_input_token_for_decode'] = prefill_token
                raise Complete()
            hook(second, readout)

        try:
            run_segment_prefill(source/'inference_segment.py', config, arm, decode=rows == 1)
        except Complete:
            pass
    if set(report['outputs']) != wanted:
        raise ValueError(f'Missing reference outputs: {sorted(wanted - report["outputs"].keys())}')
    save_json(root/'reference.json', report)
    return report


@torch.inference_mode()
def cpu_reference(root, ref, source):
    (root/'cpu').mkdir()
    def tensor(key):
        return torch.from_numpy(read_array(root, ref['inputs'][key]))
    controls = {k: tensor(k) for k in ('attention_mask', 'new_kv_cache_pos', 'position_ids')}
    controls['new_kv_cache_pos'] = controls['new_kv_cache_pos'].flatten()
    if ref['phase'] == 'decode':
        controls['kv_length'] = 20
    hidden = tensor('hidden')
    owner = pf.import_source(source)
    loader = sys.modules[owner.Qwen3_5MoeForCausalLMWrapper.__module__].load_npu_model
    checkpoint = Path(ref['model_file'])
    layers = ref['layers']
    outputs = {}
    for offset in range(0, len(layers), 4):
        group = layers[offset:offset+4]
        with pf.working_directory(source), pf.hf_config_prefix(checkpoint, 4, {}):
            model = loader(str(checkpoint), 'cpu', torch.float16, variant='nz', layer_num=len(group),
                           start_layer_idx=group[0], quant_enable=True)
        model.eval().to(dtype=torch.float16)
        verify_floating_weights(model, checkpoint, start_layer=group[0])
        cpu_model = CpuPrefill(SimpleNamespace(model=model), ref['quanted_pth'])
        for i, block in zip(group, model.model.layers, strict=True):
            if i == ref['split_layer'] and ref['phase'] == 'prefill':
                hidden[:, 19:] = 0
            values = decoder(cpu_model, block, i, hidden, controls, i == layers[-1],
                             before=[tensor(name) for name in state_roles(i)])
            hidden = values.pop('decoder_hidden')
            values[f'hidden_{i}'] = hidden
            values[f'out_proj_{i}'] = values.pop('out_proj')
            for name, value in values.items():
                if name in ref['outputs']:
                    row = tensor_record(root/'cpu', name, value)
                    outputs[name] = dict(row, path='cpu/' + row['path'])
        del block, cpu_model, model
    if outputs.keys() != ref['outputs'].keys():
        raise ValueError('CPU output set differs from the 310P capture')
    ref['cpu'] = outputs
    save_json(root/'reference.json', ref)

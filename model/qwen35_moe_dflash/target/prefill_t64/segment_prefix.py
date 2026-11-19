"""Observe the first four real decoder layers of a fresh segmented 310P prefill."""
from pathlib import Path

import torch

from ..layer_probe.capture import bound, cpu
from ..layer_probe.common import identity, save_json, sha
from ..prefill_fixture import STATE_NAMES


def tensor_record(root, name, tensor):
    path = root / (name + '.bin')
    array = cpu(tensor).numpy()
    array.tofile(path)
    return dict(identity(path, root), name=name, dtype=array.dtype.name, shape=list(array.shape))


def same_bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and left.contiguous().numpy().tobytes() == right.contiguous().numpy().tobytes())


def arm_prefix(frame, root, cleanup, *, attention_layers=False):
    caller = frame.f_back
    while caller and (caller.f_code.co_filename != frame.f_code.co_filename or 'model2' not in caller.f_locals):
        caller = caller.f_back
    if caller is None:
        raise ValueError('could not locate the original segmented Prefill driver')
    driver = caller.f_locals
    first = frame.f_locals['model_wrapper']
    second = driver['model2']
    cfg = driver['cfg']
    prompt = cpu(frame.f_locals['input_ids'])
    layers = first.model.model.layers[:4]
    if (prompt.shape != (1, 19) or [layer.block_type for layer in layers] !=
            ['linear_attention'] * 3 + ['full_attention'] or
            driver['args'].quant_mode != 'enable' or driver['args'].quant_bits != 4
            or driver['args'].pseudo_quant):
        raise ValueError('expected the original W4 19-token segment and three GDN layers followed by full attention')

    config = Path(driver['args'].config).resolve(strict=True)
    entry = Path(frame.f_code.co_filename)
    sources = [entry, entry.parent / 'utils.py', entry.parent / 'models/export_model_wrapper_qwen3_5_moe.py',
               entry.parent / 'models/modeling_qwen3_5_moe_hiai_nz.py', config]
    manifest = dict(status='CAPTURING', physical_rows=64, valid_rows=19, output_pos=18,
                    layers=[0, 1, 2, 3], segment_layers=[len(first.model.model.layers), len(second.model.model.layers)],
                    resources={k: str(Path(cfg[k]).resolve(strict=True)) for k in ('model_file', 'quanted_pth')},
                    sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
                    inputs={}, outputs={}, boundaries={}, conv_history={}, selected_golden=None,
                    torch_version=str(torch.__version__))
    save_json(root / 'manifest.json', manifest)
    working = {}

    def hook(module, fn, *, before=False):
        register = module.register_forward_pre_hook if before else module.register_forward_hook
        def once(*args):
            handle.remove()
            fn(*args)
        handle = register(once, with_kwargs=True)
        cleanup.callback(handle.remove)

    def entry_hook(module, args, kwargs):
        values = bound(module, args, kwargs)
        ids = cpu(values['input_ids'])
        length = cpu(values['effective_length'])
        mask = cpu(values['attention_mask'])
        pos = torch.arange(64)
        expected_mask = (pos[:, None] < 19) & (torch.arange(2048)[None, :] <= pos[:, None])
        blocked = torch.isneginf(mask) | (mask == -65504) | (mask == torch.finfo(mask.dtype).min)
        if (ids.shape != (1, 64) or not torch.equal(ids[:, :19], prompt) or not (ids[:, 19:] == 1).all()
                or length.dtype != torch.int16 or length.tolist() != [19] or list(values['allQLen']) != [19]
                or values['embed_scale'] is not None or values['output_pos'] is not None
                or not torch.equal(cpu(values['position_ids']), pos[None])
                or not torch.equal(cpu(values['new_kv_cache_pos']), pos)
                or mask.shape != (1, 1, 64, 2048) or not torch.equal(mask[0, 0] == 0, expected_mask)
                or not torch.equal(blocked[0, 0], ~expected_mask)):
            raise ValueError('first wrapper inputs differ from the fresh physical64/valid19 contract')
        controls = {k: values[k] for k in ('input_ids', 'attention_mask', 'position_ids', 'new_kv_cache_pos', 'effective_length')}
        controls.update(allQLen=torch.as_tensor(values['allQLen']), block_table=layers[3].self_attn.block_table)
        for i, pair in enumerate(values['past_key_values'][:4]):
            for name, value in zip(STATE_NAMES[2*i:2*i+2], pair, strict=True):
                value = cpu(value)
                if torch.count_nonzero(value):
                    raise ValueError('initial state is nonzero: ' + name)
                controls['initial_' + name] = value
        for name, value in controls.items():
            manifest['inputs'][name] = tensor_record(root, name, value)

    hook(first, entry_hook, before=True)
    def layer_input(module, args, kwargs):
        x = bound(module, args, kwargs)['hidden_states']
        manifest['device'] = str(x.device)
        if x.shape != (1, 64, 2048) or x.dtype != torch.float16:
            raise ValueError('unexpected L0 embedding descriptor')
        manifest['inputs']['embedding'] = tensor_record(root, 'embedding', x)
    hook(layers[0], layer_input, before=True)

    def capture_layer(i, layer):
        if i < 3:
            def projection(mod, args, kwargs, out):
                working[i] = cpu(out).transpose(1, 2)
            hook(layer.linear_attn.in_proj_qkv, projection)

        def exit_layer(mod, args, kwargs, result):
            hidden, pair = result[:2]
            manifest['boundaries'][str(i)] = tensor_record(root, f'decoder_hidden_{i}', hidden)
            for name, value in zip(STATE_NAMES[2*i:2*i+2], pair, strict=True):
                manifest['outputs'][name] = tensor_record(root, name, value)
            if i < 3:
                qkv = working.pop(i)
                history = cpu(pair[0])
                width = history.shape[-1]
                manifest['conv_history'][str(i)] = dict(
                    equals_valid_tail=same_bits(history, qkv[:, :, 19-width:19]),
                    equals_physical_tail=same_bits(history, qkv[:, :, -width:]))
            else:
                working['last_valid_hidden'] = cpu(hidden[:, 18, :])
        hook(layer, exit_layer)

    for i, layer in enumerate(layers):
        capture_layer(i, layer)
        if attention_layers:
            from .segment_modules import observe_module
            observe_module(root, manifest, hook, i, layer)

    def readout(module, args, kwargs, result):
        values = bound(module, args, kwargs)
        if values['output_pos'] != 18 or list(values['allQLen']) != [19] or 'last_valid_hidden' not in working:
            raise ValueError('expected the second segment to read out token index 18')
        # The original second segment has finished its own forward. Reuse its resident
        # final norm and W4 LM head for a separate four-layer prefix readout.
        with torch.inference_mode():
            last = working.pop('last_valid_hidden').to(values['hidden_states'].device)
            logits = module.model.lm_head(module.model.model.norm(last)).reshape(1, 1, module.vocab_size).float()
        manifest['outputs']['logits'] = tensor_record(root, 'logits', logits)
        manifest['four_layer_top1'] = int(logits.argmax(-1).item())
        manifest['segmented_model_top1'] = int(result[1][0, 0, 0].item())
        manifest['logits_basis'] = 'original final norm and LM head on L3 decoder output at index 18'
    hook(second, readout)
    return manifest

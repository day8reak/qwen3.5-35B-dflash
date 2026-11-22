"""Observe complete decoder blocks and the actual vocab head in segmented 310P prefill."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

from ..layer_probe.common import identity, new_output, read_array, save_json, sha

BASIS = 'MEASURED_310P_40_DECODER_BLOCKS_T64_VALID19'
HIDDEN = (1, 64, 2048)
LAYERS = 40
VALID19_IDS_SHA256 = 'ec5119e4b49de3c9e0cd48b7cff7faf13b18a0a10dda89c2a5180b194f05eaf7'


def capture(source, config, baseline, output):
    import torch
    from ..layer_probe.capture import bound, cpu
    from ..layer_probe.steps import working_directory
    from .segment_prefix import tensor_record

    entry = (source / 'inference_segment.py').resolve(strict=True)
    config = (source / config).resolve(strict=True)
    prior = json.loads((baseline / 'manifest.json').read_text()) if baseline else None
    ids = read_array(baseline, prior['inputs']['input_ids']) if prior else None
    if prior and (prior['status'] != 'CAPTURED' or ids.shape != (1, 64)
                  or (prior['valid_rows'], prior['output_pos']) != (19, 18)):
        raise ValueError('expected the retained T64 valid19 prefill reference')
    root = new_output(output)
    report = dict(status='CAPTURING', reference=BASIS, physical_rows=64, valid_rows=19, output_pos=18,
                  baseline_sha256=sha(baseline / 'manifest.json') if prior else None, source_sha256=sha(entry),
                  config_sha256=sha(config), inputs={}, blocks=[], outputs={})
    save_json(root / 'manifest.json', report)
    seen = []
    previous = tokenizer = None

    class Captured(Exception):
        pass

    with ExitStack() as cleanup:
        def hook(module, fn, before=False):
            register = module.register_forward_pre_hook if before else module.register_forward_hook
            handle = register(fn, with_kwargs=True)
            cleanup.callback(handle.remove)

        def arm(frame):
            nonlocal tokenizer
            driver = frame.f_back
            while driver and 'model2' not in driver.f_locals:
                driver = driver.f_back
            if driver is None:
                raise ValueError('original 310P segmented driver not found')
            ctx = driver.f_locals
            tokenizer = ctx['tokenizer']
            wrappers = (frame.f_locals['model_wrapper'], ctx['model2'])
            sizes = [len(w.model.model.layers) for w in wrappers]
            if min(sizes) < 1 or sum(sizes) != LAYERS or ctx['args'].pseudo_quant:
                raise ValueError(f'expected a 40-layer real W4 segmented run, got {sizes}')
            split = report['split_layer'] = sizes[0]
            report['model_file'] = str(Path(ctx['cfg']['model_file']).resolve(strict=True))
            report['quanted_pth'] = str(Path(ctx['cfg']['quanted_pth']).resolve(strict=True))
            for start, wrapper in zip((0, split), wrappers):
                text_model = wrapper.model.model
                if text_model.start_layer_idx != start:
                    raise ValueError('incorrect 310P global weight offset')
                def controls(mod, args, kwargs, start=start):
                    v = bound(mod, args, kwargs)
                    if (list(v['allQLen']) != [19] or cpu(v['effective_length']).tolist() != [19]
                            or v['output_pos'] != (None if start == 0 else 18)):
                        raise ValueError(f'wrong prefill length or readout at segment {start}')
                    for name in ('position_ids', 'new_kv_cache_pos', 'attention_mask', 'effective_length'):
                        value = cpu(v[name])
                        if prior or start:
                            ref_dir, spec = (baseline, prior['inputs'][name]) if prior else (root, report['inputs'][name])
                            if not np.array_equal(value.numpy(), read_array(ref_dir, spec)):
                                raise ValueError(f'changed prefill control {name} at segment {start}')
                        if start == 0:
                            report['inputs'][name] = tensor_record(root, name, value)
                    if start == 0:
                        value = cpu(v['input_ids'])
                        if ids is not None and not np.array_equal(value.numpy(), ids):
                            raise ValueError('prompt differs from retained valid19 token IDs')
                        if ids is None and (tuple(value.shape) != (1, 64) or hashlib.sha256(
                                value.numpy().astype('<i4').tobytes()).hexdigest() != VALID19_IDS_SHA256):
                            raise ValueError('original driver prompt differs from the measured T64/valid19 case')
                        report['inputs']['input_ids'] = tensor_record(root, 'input_ids', value)
                hook(wrapper, controls, True)

                for i, layer in enumerate(text_model.layers, start):
                    def enter(mod, args, kwargs, i=i):
                        v = bound(mod, args, kwargs)
                        x = cpu(v['hidden_states'])
                        if (tuple(x.shape) != HIDDEN or x.dtype != torch.float16
                                or v['hidden_states'].device.type != 'npu' or i != len(seen)):
                            raise ValueError(f'unexpected 310P entry at L{i}')
                        if i == 0:
                            report['inputs']['embedding'] = tensor_record(root, 'embedding', x)
                        else:
                            expected = previous.clone()
                            if i == split:
                                expected[:, 19:] = 0
                            if not torch.equal(x.view(torch.int16), expected.view(torch.int16)):
                                raise ValueError(f'unaccounted transformation before L{i}')
                        if any(cpu(t).count_nonzero().item() for t in v['past_key_values']):
                            raise ValueError(f'L{i} prefill initial state is nonzero')
                        if mod.block_type == 'full_attention' and not torch.equal(
                                cpu(mod.self_attn.block_table), torch.arange(64)[None]):
                            raise ValueError(f'nonsequential 310P page table at L{i}')
                    def leave(mod, args, kwargs, result, i=i):
                        nonlocal previous
                        previous = cpu(result[0])
                        if tuple(previous.shape) != HIDDEN or previous.dtype != torch.float16:
                            raise ValueError(f'unexpected decoder output at L{i}')
                        report['blocks'].append(dict(layer=i, block_type=mod.block_type,
                            hidden=tensor_record(root, f'l{i:02d}_hidden', previous)))
                        seen.append(i)
                    hook(layer, enter, True)
                    hook(layer, leave)

            def head(mod, args, kwargs, result):
                logits = cpu(result).float().reshape(1, 1, -1)
                if seen != list(range(LAYERS)) or logits.shape != (1, 1, wrappers[1].vocab_size):
                    raise ValueError('expected one full vocabulary row after L39')
                report['outputs']['logits'] = tensor_record(root, 'logits', logits)
            hook(wrappers[1].model.lm_head, head)

            def first_token(mod, args, kwargs, result):
                logits = read_array(root, report['outputs']['logits']).reshape(-1)
                token = int(result[1][0, 0, 0].item())
                if not np.isfinite(logits).all() or not 0 <= token < logits.size or logits[token] != logits.max():
                    raise ValueError('original greedy token does not select a maximum of the captured head')
                report.update(status='CAPTURED', top1=token, token_text=ctx['tokenizer'].decode(
                    [token], skip_special_tokens=False))
                raise Captured()
            hook(wrappers[1], first_token)

        old_profile, old_argv = sys.getprofile(), sys.argv
        def entered(frame, event, _):
            if event == 'call' and frame.f_code.co_filename == str(entry) and frame.f_code.co_name == 'prefill_chunk_segment1':
                sys.setprofile(old_profile)
                arm(frame)
        cleanup.callback(sys.setprofile, old_profile)
        cleanup.callback(setattr, sys, 'argv', old_argv)
        sys.path.insert(0, str(entry.parent))
        cleanup.callback(sys.path.remove, str(entry.parent))
        sys.argv = [str(entry), '--max_token=0', '--quant_mode=enable', '--quant_bits=4',
                    '--prefill_chunk_size=64', '--config', str(config)]
        try:
            with working_directory(entry.parent):
                sys.setprofile(entered)
                runpy.run_path(str(entry), run_name='__main__')
        except Captured:
            pass
        if report['status'] != 'CAPTURED':
            raise ValueError('original segmented run did not return a full-prefill first token')
        vocab_size = report['outputs']['logits']['shape'][-1]
        mapping = [None] * vocab_size
        valid_ids = sorted({i for i in tokenizer.get_vocab().values() if 0 <= i < vocab_size})
        for begin in range(0, len(valid_ids), 1024):
            group = valid_ids[begin:begin + 1024]
            for i, piece in zip(group, tokenizer.batch_decode([[i] for i in group], skip_special_tokens=False), strict=True):
                mapping[i] = piece
        if mapping[report['top1']] != report['token_text']:
            raise ValueError('portable single-token decode differs from original tokenizer')
        ids = read_array(root, report['inputs']['input_ids'])
        token_map = root / 'token-text.json'
        save_json(token_map, dict(pieces=mapping, special_ids=tokenizer.all_special_ids,
                                  prompt=tokenizer.decode(ids[0, :19].tolist(), skip_special_tokens=False)))
        report['token_text_map'] = identity(token_map, root)
        save_json(root / 'manifest.json', report)
        print(f"310P 40 layers, first token {report['top1']} {report['token_text']!r}", flush=True)


def installed_vscale(ddk):
    """Fingerprint the two installed GDR C++ sources on the shared OMG host."""
    import re
    names = ['ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp',
             'tools/tools_ascendc/custom_op/ascend310b/src/chunk_gated_delta_rule.cpp']
    hashes = {name: sha(ddk / name) for name in names}
    if len(set(hashes.values())) != 1:
        raise ValueError(f'DDK installed GDR copies disagree: {hashes}')
    text = (ddk / names[0]).read_text()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    literal = r'1024(?:\.0*)?[fF]?'
    pattern = (r'\b(?:debug_)?v_?scale\s*=\s*(?:\(\s*half\s*\)\s*|'
               r'(?:static_cast\s*<\s*half\s*>|half)\s*\(\s*)?' + literal + r'\s*\)?\s*;')
    if not re.search(pattern, text, flags=re.I):
        raise ValueError('expected an active vscale=1024 literal in the installed chunk GDR source: ' + names[0])
    return hashes

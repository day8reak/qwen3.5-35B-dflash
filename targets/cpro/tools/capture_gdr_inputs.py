"""Capture the first L0 GDR call in the original 310P segmented prefill."""
from contextlib import ExitStack
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np

from qwen35_moe_dflash.target.layer_probe.common import sha


def capture(source_root, config, requested, length):
    import torch
    from qwen35_moe_dflash.target.layer_probe.capture import cpu
    from qwen35_moe_dflash.target.layer_probe.steps import working_directory
    from qwen35_moe_dflash.target.prefill_t64.segment_gdn_trace import observe_tail
    from trace_segment_attention import observe_l1_gdr_length

    root = source_root.resolve(strict=True)
    entry = root / 'inference_segment.py'
    config = (root / config).resolve(strict=True)
    argv = [str(entry), '--config', str(config), '--prefill_chunk_size=64',
            '--quant_mode=enable', '--quant_bits=4', '--max_token=0']
    report, snapshot = {}, {}

    class Captured(Exception):
        pass

    with ExitStack() as cleanup:
        cleanup.enter_context(working_directory(root))
        cleanup.enter_context(patch.object(sys, 'argv', argv))
        cleanup.enter_context(patch.object(sys, 'path', [str(root), *sys.path]))
        main = runpy.run_path(str(entry))['main']
        driver = main.__globals__
        chunk = driver['prefill_chunk_segment1']

        def observe_first(model_wrapper, input_ids, attention_mask, past_kv_1, max_len, chunk_size, device):
            wrapper = model_wrapper
            prompt = input_ids
            if requested is not None:
                valid = length if length is not None else requested.size
                if np.any(requested < 0) or np.any(requested >= wrapper.model.config.vocab_size):
                    raise ValueError('input_ids contains an ID outside the model vocabulary')
                if prompt.shape[1] != valid:
                    mask = driver['create_strict_causal_mask'](valid, max_len, device=device)
                    attention_mask = mask[None, None]
                prompt = torch.as_tensor(requested[:valid].copy()).reshape(1, -1).to(prompt)
            expected = cpu(prompt[:, :64])
            if not expected.numel() or (length is not None and expected.shape[1] != length):
                raise ValueError(f'prompt first chunk has {expected.shape[1]} valid tokens, requested {length}')

            def read_ids(module, pargs, kw):
                ids = kw['input_ids']
                if (tuple(ids.shape) != (1, 64) or not torch.equal(cpu(ids[:, :expected.shape[1]]), expected)
                        or cpu(kw['effective_length']).tolist() != [expected.shape[1]]
                        or kw.get('embed_scale') is not None or kw.get('inputs_embeds') is not None):
                    raise ValueError('segment wrapper input_ids or effective_length differ from the requested first chunk')
                if requested is not None and requested.size == 64:
                    # A physical dump may contain arbitrary padding IDs. Feed all of them to the embedding.
                    ids = torch.as_tensor(requested.copy()).reshape(1, 64).to(ids)
                    kw = dict(kw, input_ids=ids)
                snapshot['input_ids'] = cpu(ids)
                snapshot['valid'] = expected.shape[1]
                return pargs, kw

            def stop(module, args, output):
                raise Captured()

            cleanup.callback(wrapper.register_forward_pre_hook(read_ids, with_kwargs=True).remove)
            observe_l1_gdr_length(wrapper, report, cleanup, snapshot, layer_index=0)
            linear = wrapper.model.model.layers[0].linear_attn
            observe_tail(linear, snapshot, cleanup)
            cleanup.callback(linear.register_forward_hook(stop).remove)
            return chunk(wrapper, prompt, attention_mask, past_kv_1, max_len=max_len, chunk_size=chunk_size, device=device)

        cleanup.enter_context(patch.dict(driver, prefill_chunk_segment1=observe_first))
        try:
            main()
        except Captured:
            pass

    calls = report.get('l0_gdr', {}).get('calls', [])
    if len(calls) != 1 or calls[0].get('error') or 'gdr' not in snapshot or 'input_ids' not in snapshot:
        raise ValueError(f'L0 GDR capture incomplete: {report}')
    call = snapshot['gdr']
    inputs = {name: tensor.numpy() for name, tensor in call['inputs'].items()}
    if (inputs['effective_length'].shape != (1,) or int(inputs['effective_length'][0]) != snapshot['valid']
            or call['attrs']['chunk_size'] != 64 or not call['attrs']['output_final_state']
            or not call['attrs']['use_qk_l2norm_in_kernel']):
        raise ValueError('captured GDR attributes or length differ from the standalone AIR contract')
    core = call['outputs'][0]
    valid = snapshot['valid']
    if not torch.equal(core[:, :valid], snapshot['norm_input'].reshape_as(core)[:, :valid]):
        raise ValueError('Gated RMSNorm did not consume this L0 GDR core')
    context = (snapshot['norm_gate'].reshape_as(core).numpy(), snapshot['gamma'].numpy(), snapshot['epsilon'])
    return inputs, snapshot['input_ids'].numpy(), dict(kind='measured_l0_310p', device=call['device'],
        input_ids_source='file' if requested is not None else 'original_segment_prompt',
        driver_sha256=sha(entry), config_sha256=sha(config), layer_source_sha256=sha(Path(call['source']))), context

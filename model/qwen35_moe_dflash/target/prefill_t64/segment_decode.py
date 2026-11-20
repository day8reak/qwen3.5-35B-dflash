"""Capture one native NZ decode step following the retained physical64/valid19 prompt."""
from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import sys

import torch

from ..layer_probe.capture import bound, cpu
from ..layer_probe.common import read_array, save_json, sha
from .prepare import STATES
from .segment_modules import observe_module
from .segment_prefix import tensor_record


def capture_decode(root, entry, prior):
    previous = json.loads((prior / 'manifest.json').read_text())
    if (previous['status'] != 'CAPTURED' or previous['layers'] != [0, 1, 2, 3]
            or (previous['physical_rows'], previous['valid_rows'], previous['output_pos']) != (64, 19, 18)
            or Path(previous['sources'][0]['path']).resolve() != entry):
        raise ValueError('Expected the original segmented 64/19 four-layer prefill and its inference_segment.py')
    for source in previous['sources']:
        if sha(source['path']) != source['sha256']:
            raise ValueError('Prefill source or config changed: ' + source['path'])
    config, = [s['path'] for s in previous['sources'] if Path(s['path']).suffix in ('.yaml', '.yml')]
    ids = torch.from_numpy(read_array(prior, previous['inputs']['input_ids']))
    table = torch.from_numpy(read_array(prior, previous['inputs']['block_table']))
    report = dict(status='CAPTURING', phase='decode', physical_rows=1, valid_rows=1, position=19,
                  prefill_fixture=str(prior), prefill_sha256=sha(prior / 'manifest.json'),
                  layers=[0, 1, 2, 3], attention_modules={})
    save_json(root / 'manifest.json', report)
    with ExitStack() as cleanup:
        def once(mod, callback, *, before=False):
            def run(*args):
                handle.remove()
                callback(*args)
            register = mod.register_forward_pre_hook if before else mod.register_forward_hook
            handle = register(run, with_kwargs=True)
            cleanup.callback(handle.remove)

        def prefill(mod, args, kwargs):
            v = bound(mod, args, kwargs)
            if (ids.shape != (1, 64) or not torch.equal(cpu(v['input_ids']), ids) or list(v['allQLen']) != [19]
                    or cpu(v['effective_length']).tolist() != [19]):
                raise ValueError('Live prefill token IDs or valid length differ from retained fixture')
            report['prefill_ids_match'] = True

            def after_prefill(wrapper, _args, _kwargs, _result):
                once(wrapper, decode, before=True)
            once(mod, after_prefill)

        def decode(mod, args, kwargs):
            v = bound(mod, args, kwargs)
            token = cpu(v['input_ids'])
            if (token.shape != (1, 1) or int(token.item()) != previous['segmented_model_top1']
                    or list(v['allQLen']) != [20] or cpu(v['effective_length']).tolist() != [1]
                    or cpu(v['position_ids']).reshape(-1).tolist() != [19]
                    or cpu(v['new_kv_cache_pos']).reshape(-1).tolist() != [19]):
                raise ValueError('Expected the first generated token at position 19, effective_length=1 and allQLen=20')
            layers = mod.model.model.layers
            if not torch.equal(cpu(layers[3].self_attn.block_table), table):
                raise ValueError('Decode page table differs from the retained prefill')
            report['block_table'] = tensor_record(root, 'block_table', table)
            report.update(input_token=int(token.item()), effective_length=1, allQLen=20)
            for i in range(4):
                observe_module(root, report, once, i, layers[i])

        def entered(frame, event, _):
            if event != 'call' or frame.f_code.co_filename != str(entry):
                return
            name = frame.f_code.co_name
            if name == 'prefill_chunk_segment1':
                sys.setprofile(old_profile)
                once(frame.f_locals['model_wrapper'], prefill, before=True)

        old_profile, old_argv = sys.getprofile(), sys.argv
        cleanup.callback(sys.setprofile, old_profile)
        cleanup.callback(setattr, sys, 'argv', old_argv)
        sys.path.insert(0, str(entry.parent))
        cleanup.callback(sys.path.remove, str(entry.parent))
        sys.argv = [str(entry), '--max_token=1', '--quant_mode=enable', '--quant_bits=4',
                    '--prefill_chunk_size=64', '--config', str(config)]
        sys.setprofile(entered)
        runpy.run_path(str(entry), run_name='__main__')
    if set(report['attention_modules']) != {'0', '1', '2', '3'}:
        raise RuntimeError('Expected a completed first decode in the original segmented run for L0–L3')
    states = {}
    for i in range(4):
        record = report['attention_modules'][str(i)]
        if record['golden_hidden']['shape'] != [1, 1, 2048]:
            raise ValueError(f'Missing first decode hidden at L{i}')
        for key, name in zip(('conv', 'state') if i < 3 else ('key', 'value'), STATES[2*i:2*i+2]):
            old, live = previous['outputs'][name], record['inputs'][key]
            states[name] = all(old[k] == live[k] for k in ('shape', 'dtype', 'sha256'))
    report.update(status='CAPTURED', initial_state_matches_retained_prefill=states)
    save_json(root / 'manifest.json', report)
    print(json.dumps(dict(status='CAPTURED', token=report['input_token'], position=19, allQLen=20,
                          state_matches_retained_prefill=states, manifest=str(root / 'manifest.json'))), flush=True)

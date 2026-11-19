"""Compare four propagated CPU decoder layers to the same valid segmented prompt."""
import json
from pathlib import Path

import numpy as np
import torch

from .. import prefill_fixture as pf
from ..layer_probe.common import metrics, read_array, save_json, sha
from .cpu_reference import CpuPrefill, norm, verify_floating_weights
from .segment_prefix import tensor_record


@torch.inference_mode()
def compare(root, output):
    manifest_path = root / 'manifest.json'
    capture = json.loads(manifest_path.read_text())
    if (capture.get('status') != 'CAPTURED' or capture['layers'] != [0, 1, 2, 3]
            or (capture['physical_rows'], capture['valid_rows'], capture['output_pos']) != (64, 19, 18)):
        raise ValueError('expected the completed four-layer segmented 64/19 capture')
    for source in capture['sources']:
        if sha(source['path']) != source['sha256']:
            raise ValueError(f"source changed: {source['path']}")
    source_root = Path(capture['sources'][0]['path']).parent
    cfg = capture['resources']
    model_path = Path(cfg['model_file'])
    report = dict(scope='FOUR_LAYER_CPU_VS_NATIVE_310P_VALID19', capture_sha256=sha(manifest_path),
                  comparisons={}, cpu_outputs={}, selected_golden=None, conv_history=capture['conv_history'])
    source = pf.import_source(source_root)
    with pf.working_directory(source_root), pf.hf_config_prefix(model_path, 4, report):
        wrapper = pf.build_four_layers(source, model_path, report)
    wrapper.to(dtype=torch.float16)
    verify_floating_weights(wrapper.model, model_path)
    cpu = CpuPrefill(wrapper, cfg['quanted_pth'])
    inputs = {name: torch.from_numpy(read_array(root, spec)) for name, spec in capture['inputs'].items()}
    hidden = cpu.model.model.embed_tokens(inputs['input_ids'].long()).to(torch.float16)
    report['embedding_equal_bits'] = torch.equal(hidden.view(torch.int16), inputs['embedding'].view(torch.int16))
    hidden = hidden[:, :19]
    args = dict(position_ids=inputs['position_ids'][:, :19], new_kv_cache_pos=inputs['new_kv_cache_pos'][:19],
                attention_mask=inputs['attention_mask'][:, :, :19])
    table = inputs['block_table']
    if not torch.equal(cpu.model.model.layers[3].self_attn.block_table, table):
        raise ValueError('L3 CPU block table differs from capture')
    # Only valid prompt rows enter the independent model. Cache slots written
    # for padded tokens are recorded on device but are outside this comparison.
    def score(name, actual, expected):
        row = metrics(np.asarray(actual), np.asarray(expected))
        report['comparisons'][name] = row
        print(f"{name:24s} rel_L2={row.get('relative_l2')}  {'PASS' if row['passed'] else 'FAIL'}", flush=True)

    def compare_output(name, expected):
        native = torch.from_numpy(read_array(root, capture['outputs'][name]))
        report['cpu_outputs'][name] = tensor_record(output, name, expected)
        if name in ('key_cache_3', 'value_cache_3'):
            block = expected.shape[2]
            positions = torch.arange(19)
            pages = table[0, positions // block].long()
            score(name, native[pages, :, positions % block], expected[pages, :, positions % block])
            unchanged = torch.ones(expected.shape[0], block, dtype=torch.bool)
            physical = torch.arange(64)
            unchanged[table[0, physical // block].long(), physical % block] = False
            before = inputs['initial_' + name]
            keep = unchanged[:, None, :, None].expand_as(native)
            exact = torch.equal(native[keep].view(torch.int16), before[keep].view(torch.int16))
            report['comparisons'][name]['untouched_equal_bits'] = exact
            report['comparisons'][name]['passed'] &= exact
        else:
            score(name, native, expected)
        return native

    for i, layer in enumerate(cpu.model.model.layers):
        print(f'[target-prefill] CPU L{i}', flush=True)
        prefix = f'layers.{i}.'
        state_names = pf.STATE_NAMES[2*i:2*i+2]
        before = tuple(inputs['initial_' + name] for name in state_names)
        normalized = norm(hidden, layer.input_layernorm)
        if i < 3:
            mix, (history, state) = cpu.gdn(normalized, layer.linear_attn, prefix + 'linear_attn', before)
            # NZ writes the FP16-rounded final accumulator back to its cache tensor.
            states = (history, state.half().to(before[1].dtype))
        else:
            mix, states = cpu.attention(normalized, layer.self_attn, prefix + 'self_attn', before, args)
        hidden = hidden + mix
        hidden = hidden + cpu.moe(norm(hidden, layer.post_attention_layernorm), layer.mlp, prefix + 'mlp')
        actual = read_array(root, capture['boundaries'][str(i)])
        name = f'decoder_hidden_{i}'
        report['cpu_outputs'][name] = tensor_record(output, name, hidden)
        score(name, actual[:, :19], hidden)
        for name, expected in zip(state_names, states, strict=True):
            compare_output(name, expected)
    last = norm(hidden[:, -1:, :], cpu.model.model.norm)
    logits = cpu.project(last, 'lm_head', cpu.model.lm_head).float()
    native = compare_output('logits', logits)
    report.update(cpu_top1=int(logits.argmax()), native_top1=int(native.argmax()))
    report['failed'] = [k for k, v in report['comparisons'].items() if not v['passed']]
    if not report['embedding_equal_bits']:
        report['failed'].append('embedding')
    if report['cpu_top1'] != report['native_top1']:
        report['failed'].append('logits_top1')
    report['status'] = 'NUMERICAL_FAILED' if report['failed'] else 'PASSED'
    save_json(output / 'report.json', report)
    return dict(status=report['status'], failed=report['failed'], cpu_top1=report['cpu_top1'],
                native_top1=report['native_top1'], conv_history=report['conv_history'],
                report=str(output / 'report.json'))

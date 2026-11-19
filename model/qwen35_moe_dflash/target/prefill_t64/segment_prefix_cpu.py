"""Compare four propagated CPU decoder layers to the same valid segmented prompt."""
import json
from pathlib import Path

import numpy as np
import torch

from .. import prefill_fixture as pf
from ..layer_probe.common import metrics, read_array, save_json, sha
from .cpu_reference import CpuPrefill, norm, verify_floating_weights
from .segment_prefix import tensor_record


def _load(root, report):
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
    report.update(capture_sha256=sha(manifest_path), conv_history=capture['conv_history'])
    source = pf.import_source(source_root)
    with pf.working_directory(source_root), pf.hf_config_prefix(model_path, 4, report):
        wrapper = pf.build_four_layers(source, model_path, report)
    wrapper.to(dtype=torch.float16)
    verify_floating_weights(wrapper.model, model_path)
    cpu = CpuPrefill(wrapper, cfg['quanted_pth'])
    inputs = {name: torch.from_numpy(read_array(root, spec)) for name, spec in capture['inputs'].items()}
    args = dict(position_ids=inputs['position_ids'][:, :19], new_kv_cache_pos=inputs['new_kv_cache_pos'][:19],
                attention_mask=inputs['attention_mask'][:, :, :19])
    table = inputs['block_table']
    if not torch.equal(cpu.model.model.layers[3].self_attn.block_table, table):
        raise ValueError('L3 CPU block table differs from capture')
    return capture, cpu, inputs, args


def _decoder(cpu, i, hidden, inputs, args):
    layer = cpu.model.model.layers[i]
    prefix = f'layers.{i}.'
    names = pf.STATE_NAMES[2*i:2*i+2]
    before = tuple(inputs['initial_' + name] for name in names)
    normalized = norm(hidden, layer.input_layernorm)
    if i < 3:
        mix, (history, state) = cpu.gdn(normalized, layer.linear_attn, prefix + 'linear_attn', before)
        # The original wrapper rounds the final accumulator before writing its cache.
        states = (history, state.half().to(before[1].dtype))
    else:
        mix, states = cpu.attention(normalized, layer.self_attn, prefix + 'self_attn', before, args)
    residual = hidden + mix
    hidden = residual + cpu.moe(norm(residual, layer.post_attention_layernorm), layer.mlp, prefix + 'mlp')
    return hidden, dict(zip(names, states, strict=True))


def _readout(cpu, hidden):
    last = norm(hidden[:, 18:19, :], cpu.model.model.norm)
    return cpu.project(last, 'lm_head', cpu.model.lm_head).float()


@torch.inference_mode()
def compare(root, output):
    report = dict(scope='FOUR_LAYER_CPU_VS_NATIVE_310P_VALID19', comparisons={}, cpu_outputs={}, selected_golden=None)
    capture, cpu, inputs, args = _load(root, report)
    table = inputs['block_table']
    hidden = cpu.model.model.embed_tokens(inputs['input_ids'].long()).to(torch.float16)
    report['embedding_equal_bits'] = torch.equal(hidden.view(torch.int16), inputs['embedding'].view(torch.int16))
    hidden = hidden[:, :19]
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

    for i in range(4):
        print(f'[target-prefill] CPU L{i}', flush=True)
        hidden, states = _decoder(cpu, i, hidden, inputs, args)
        actual = read_array(root, capture['boundaries'][str(i)])
        name = f'decoder_hidden_{i}'
        report['cpu_outputs'][name] = tensor_record(output, name, hidden)
        score(name, actual[:, :19], hidden)
        for name, expected in states.items():
            compare_output(name, expected)
    logits = _readout(cpu, hidden)
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


@torch.inference_mode()
def diagnose(root, output):
    """Replay each decoder on its captured input and compare to the saved propagated CPU result."""
    prior_root = root / 'cpu'
    prior_path = prior_root / 'report.json'
    prior = json.loads(prior_path.read_text())
    report = dict(scope='FOUR_LAYER_FIXED_INPUT_CPU_VALID19', prior_report_sha256=sha(prior_path),
                  comparisons={}, conv_physical_tail={}, selected_golden=None)
    if (prior.get('scope') != 'FOUR_LAYER_CPU_VS_NATIVE_310P_VALID19'
            or prior.get('status') not in ('PASSED', 'NUMERICAL_FAILED')
            or prior['capture_sha256'] != sha(root / 'manifest.json')):
        raise ValueError('expected the completed cpu/report.json from this same 64/19 capture')
    capture, cpu, inputs, args = _load(root, report)
    table = inputs['block_table']
    report['distance_denominator'] = 'L2 norm of the corresponding captured 310P output'
    report['gate_reference'] = 'CPU for each replay; captured 310P is the actual output'
    print('output                     fixed     propagated    input_effect   fixed_gate')
    def compare_three(name, native, fixed, propagated):
        if native.shape != fixed.shape or native.shape != propagated.shape:
            raise ValueError(f'{name}: tensor shapes differ')
        fixed_metrics = metrics(native.numpy(), fixed.numpy())
        denom = torch.linalg.vector_norm(native.double()).item()

        def distance(a, b):
            delta = torch.linalg.vector_norm(a.double() - b.double()).item()
            if denom == 0:
                return 0. if delta == 0 else None
            return delta / denom if np.isfinite(delta / denom) else None

        distances = (distance(fixed, native), distance(propagated, native), distance(fixed, propagated))
        report['comparisons'][name] = dict(fixed_metrics=fixed_metrics, fixed=distances[0],
                                           propagated=distances[1], input_effect=distances[2])
        fields = ' '.join(f'{value:13.6g}' if value is not None else f"{'n/a':>13}" for value in distances)
        print(f"{name:22s}{fields}   {'PASS' if fixed_metrics['passed'] else 'FAIL'}", flush=True)

    def stored_cpu(name):
        return torch.from_numpy(read_array(prior_root, prior['cpu_outputs'][name]))

    def stored_native(name):
        return torch.from_numpy(read_array(root, capture['outputs'][name]))

    entry = inputs['embedding']
    for i in range(4):
        hidden, states = _decoder(cpu, i, entry[:, :19], inputs, args)
        if i < 3:
            name = f'conv_state_{i}'
            layer = cpu.model.model.layers[i]
            width = states[name].shape[-1]
            # This writeback uses pre-convolution QKV. Each projected row only
            # depends on its own decoder input, so the four physical rows suffice.
            pad_input = norm(entry[:, -width:], layer.input_layernorm)
            physical = cpu.project(pad_input, f'layers.{i}.linear_attn.in_proj_qkv', layer.linear_attn.in_proj_qkv)
            native = stored_native(name)
            physical_metrics = metrics(native.numpy(), physical.transpose(1, 2).contiguous().numpy())
            report['conv_physical_tail'][name] = physical_metrics
        for name, fixed in states.items():
            native, propagated = stored_native(name), stored_cpu(name)
            if i == 3:
                block = native.shape[2]
                pos = torch.arange(19)
                pages = table[0, pos // block].long()
                native, fixed, propagated = (t[pages, :, pos % block] for t in (native, fixed, propagated))
            compare_three(name, native, fixed, propagated)
        entry = torch.from_numpy(read_array(root, capture['boundaries'][str(i)]))
        name = f'decoder_hidden_{i}'
        compare_three(name, entry[:, :19], hidden, stored_cpu(name))

    logits = stored_native('logits')
    fixed_logits = _readout(cpu, entry)
    propagated_logits = stored_cpu('logits')
    compare_three('logits', logits, fixed_logits, propagated_logits)
    report['readout'] = {}
    for label, values in (('native', logits), ('fixed', fixed_logits), ('propagated', propagated_logits)):
        top = values.flatten().topk(2)
        report['readout'][label] = dict(top1=int(values.argmax()), top2_ids=top.indices.tolist(),
                                        top2_values=top.values.tolist(), margin=float(top.values[0] - top.values[1]))
    report['status'] = 'INSPECTED'
    save_json(output / 'report.json', report)
    return dict(status='INSPECTED', fixed_failed=[k for k, v in report['comparisons'].items() if not v['fixed_metrics']['passed']],
                physical_conv_relative_l2={k: v['relative_l2'] for k, v in report['conv_physical_tail'].items()},
                top1={k: v['top1'] for k, v in report['readout'].items()}, report=str(output / 'report.json'))

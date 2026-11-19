"""Same-call module inputs and post-projection hidden from the real valid19 segment."""
import torch

from ..layer_probe.capture import bound, cpu
from ..layer_probe.common import metrics, read_array, save_json
from .segment_prefix import tensor_record


def observe_module(root, manifest, hook, index, decoder):
    linear = index < 3
    member = 'linear_attn' if linear else 'self_attn'
    op = getattr(decoder, member)
    record = {'module': member, 'inputs': {}}
    manifest.setdefault('attention_modules', {})[str(index)] = record

    def enter(mod, args, kwargs):
        supplied = bound(mod, args, kwargs)
        x = supplied['hidden_states']
        if x.shape != (1, 64, 2048) or x.dtype != torch.float16 or x.device.type != 'npu':
            raise ValueError(f'L{index}: unexpected segmented attention-module entry')
        inputs = {'hidden': x}
        first, second = supplied['past_key_values']
        if linear:
            length = cpu(supplied['effective_length'])
            if length.dtype != torch.int16 or length.tolist() != [19]:
                raise ValueError(f'L{index}: expected GDN effective_length=int16[19]')
            inputs.update(conv=first, state=second, effective_length=length)
            if supplied.get('linear_update_tensor') is not None:
                inputs['linear_update_tensor'] = supplied['linear_update_tensor']
        else:
            length = torch.as_tensor(supplied['allQLen'])
            pos = cpu(supplied['position_ids'])
            cache_pos = cpu(supplied['new_kv_cache_pos'])
            mask = cpu(supplied['attention_mask'])
            visible = ((torch.arange(64)[:, None] < 19) &
                       (torch.arange(2048)[None, :] <= torch.arange(64)[:, None]))
            if (length.tolist() != [19] or not torch.equal(pos, torch.arange(64)[None])
                    or not torch.equal(cache_pos.reshape(-1), torch.arange(64))
                    or mask.shape != (1, 1, 64, 2048) or not torch.equal(mask[0, 0] == 0, visible)
                    or not torch.all((mask[0, 0][~visible] <= -65504))):
                raise ValueError('L3: expected fresh physical64/valid19 positions, length and causal mask')
            inputs.update(key=first, value=second, mask=mask, positions=pos,
                          cache_positions=cache_pos, all_length=length.to(torch.int64))
        for name, tensor in inputs.items():
            value = cpu(tensor)
            if name in ('conv', 'state', 'key', 'value') and torch.count_nonzero(value):
                raise ValueError(f'L{index}: nonzero initial {name}')
            record['inputs'][name] = tensor_record(root, f'l{index}_module_{name}', value)

    def leave(mod, args, kwargs, result):
        if not isinstance(result, (list, tuple)) or result[0].shape != (1, 64, 2048):
            raise ValueError(f'L{index}: unexpected attention-module return')
        record['golden_hidden'] = tensor_record(root, f'l{index}_module_hidden_out', result[0])

    hook(op, enter, before=True)
    hook(op, leave)


@torch.inference_mode()
def compare_cpu(root, output):
    from .segment_prefix_cpu import _load
    report = {'scope': 'SEGMENT_VALID19_FIXED_INPUT_ATTENTION_MODULE_HIDDEN', 'layers': {}}
    capture, cpu_model, _, _ = _load(root, report)
    for i in range(4):
        layer = cpu_model.model.model.layers[i]
        record = capture['attention_modules'][str(i)]
        values = {k: torch.from_numpy(read_array(root, spec)) for k, spec in record['inputs'].items()}
        x = values['hidden'][:, :19]
        prefix = f'layers.{i}.' + record['module']
        if i < 3:
            result, _ = cpu_model.gdn(x, layer.linear_attn, prefix, (values['conv'], values['state']))
        else:
            args = dict(position_ids=values['positions'][:, :19], new_kv_cache_pos=values['cache_positions'].reshape(-1)[:19],
                        attention_mask=values['mask'][:, :, :19])
            result, _ = cpu_model.attention(x, layer.self_attn, prefix, (values['key'], values['value']), args)
        native = read_array(root, record['golden_hidden'])
        # Score in the same direction as the existing 310P qualification (CPU as reference).
        cpu_hidden = result.contiguous().numpy()
        row = metrics(native[:, :19], cpu_hidden)
        last = metrics(native[:, 18], cpu_hidden[:, 18])
        report['layers'][str(i)] = dict(valid19=row, token_18=last,
            passed=row['passed'] and last['passed'], cpu_hidden=tensor_record(output, f'l{i}_cpu_hidden', result))
        print(f"L{i} {record['module']} 310P vs CPU: valid19 rel_L2={row['relative_l2']} "
              f"token18 rel_L2={last['relative_l2']} failed={row['failed_metrics']}/{last['failed_metrics']}", flush=True)
    report['status'] = 'PASSED' if all(r['passed'] for r in report['layers'].values()) else 'NUMERICAL_FAILED'
    save_json(output / 'report.json', report)
    return dict(status=report['status'], report=str(output / 'report.json'))

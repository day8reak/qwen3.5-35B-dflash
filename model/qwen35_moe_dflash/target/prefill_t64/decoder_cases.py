"""Public shapes and ordered outputs of the static decoder comparisons."""

MODES = ('linear', 'full', '4', '40', 'layerwise')
PHASES = ('prefill', 'decode')
FOUR_LAYER_CONTROLS = ('no_patch', 'native_activations', 'silu_only', 'softplus_only',
                       'gdn_silu_only', 'gdn_silu_site', 'all_silu_no_gate_cast', 'rms_silu_layers',
                       'rms_silu_token_view', 'physical_convstate')
EXPORT_CONTROLS = (*FOUR_LAYER_CONTROLS, 'rmsnorm_patch', 'shared_gate_fp16_sigmoid')


def directory_name(mode, phase):
    count = {'linear': '1-linear', 'full': '1-full', '4': '4', '40': '40', 'layerwise': '40x1'}[mode]
    return f'{count}-{phase}-t{64 if phase == "prefill" else 1}'


def cache_rows(values, start, stop):
    # Physical layout is [page, head-block, offset, lane], with sequential pages.
    import numpy as np
    positions = np.arange(start, stop)
    return values[positions // 32, :, positions % 32, :]


def state_roles(layer):
    names = ('key_cache', 'value_cache') if layer % 4 == 3 else ('conv_state', 'recurrent_state')
    return tuple(f'{name}_{layer}' for name in names)


def cpro_input_records(reference):
    return {**reference['inputs'], **reference.get('cpro_inputs', {})}


def groups(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown decoder mode {mode}')
    if mode == 'layerwise':
        return [(i,) for i in range(40)]
    return [{'linear': (0,), 'full': (3,), '4': tuple(range(4)), '40': tuple(range(40))}[mode]]


def case_name(group):
    return f'L{group[0]:02d}' if len(group) == 1 else f'L{group[0]:02d}-L{group[-1]:02d}'


def output_names(group, final):
    names = [f'hidden_{group[-1]}']
    if len(group) == 1:
        names.append(f'out_proj_{group[0]}')
    for i in group:
        names.extend(state_roles(i))
    if final:
        names.append('logits')
    return names


def comparison_view(name, values, phase, prompt_length):
    scope = f'valid{prompt_length}' if phase == 'prefill' else f'position{prompt_length}'
    if name == 'logits':
        return (f'token{prompt_length-1}' if phase == 'prefill' else scope), values
    if name.startswith(('conv_state_', 'recurrent_state_')):
        return 'full_state', values
    if name.startswith(('key_cache_', 'value_cache_')):
        # Sequential page table. [page, head-block, offset, lane].
        start, stop = (0, prompt_length) if phase == 'prefill' else (prompt_length, prompt_length + 1)
        return scope, cache_rows(values, start, stop)
    return scope, values[:, :prompt_length] if phase == 'prefill' else values


def preserved_cache(values, before, phase, prompt_length):
    """All bytes outside the positions physically updated by the measured driver."""
    import numpy as np
    old = before.reshape(64, 32, 32, 16)
    new = values.reshape(old.shape)
    if old.dtype != new.dtype:
        return False
    page, slot = divmod(prompt_length, 32)
    regions = [(new[2:], old[2:])] if phase == 'prefill' else [
        (new[:page], old[:page]), (new[page+1:], old[page+1:]),
        (new[page, :, :slot], old[page, :, :slot]), (new[page, :, slot+1:], old[page, :, slot+1:])]
    return all(np.array_equal(a.view(np.uint16), b.view(np.uint16)) for a, b in regions)

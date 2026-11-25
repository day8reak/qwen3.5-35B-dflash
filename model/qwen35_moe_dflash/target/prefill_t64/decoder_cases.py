"""Public shapes and ordered outputs of the static decoder comparisons."""

MODES = ('linear', 'full', '4', '40', 'layerwise')
PHASES = ('prefill', 'decode')


def state_roles(layer):
    names = ('key_cache', 'value_cache') if layer % 4 == 3 else ('conv_state', 'recurrent_state')
    return tuple(f'{name}_{layer}' for name in names)


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


def comparison_view(name, values, phase):
    if name == 'logits':
        return ('token18' if phase == 'prefill' else 'position19'), values
    if name.startswith(('conv_state_', 'recurrent_state_')):
        return 'full_state', values
    if name.startswith(('key_cache_', 'value_cache_')):
        # Sequential page table. [page, head-block, offset, lane].
        if phase == 'decode':
            return 'position19', values[0, :, 19:20, :]
        return 'valid19', values[0, :, :19, :]
    return ('valid19', values[:, :19]) if phase == 'prefill' else ('position19', values)


def preserved_cache(values, before, phase):
    """All bytes outside the positions physically updated by the measured driver."""
    import numpy as np
    old = before.reshape(64, 32, 32, 16)
    new = values.reshape(old.shape)
    if old.dtype != new.dtype:
        return False
    regions = [(new[2:], old[2:])] if phase == 'prefill' else [
        (new[1:], old[1:]), (new[0, :, :19], old[0, :, :19]), (new[0, :, 20:], old[0, :, 20:])]
    return all(np.array_equal(a.view(np.uint16), b.view(np.uint16)) for a, b in regions)

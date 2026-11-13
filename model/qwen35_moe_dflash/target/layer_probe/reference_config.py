"""Keep a bounded reference config consistent without writing the checkpoint."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path


LAYER_LISTS = ('layer_types', 'mlp_layer_types')


def set_layer_prefix(config, count):
    """Match the already-constructed prefix, preserving its original layer order."""
    if type(count) is not int or count < 1 or config.num_hidden_layers < count:
        raise ValueError('reference layer prefix exceeds the source config')
    selected = {}
    for name in LAYER_LISTS:
        values = getattr(config, name, None)
        if values is not None:
            if not isinstance(values, (list, tuple)) or len(values) < count:
                raise ValueError(f'{name} does not describe the requested layer prefix')
            selected[name] = list(values[:count])
    config.num_hidden_layers = count
    for name, values in selected.items():
        setattr(config, name, values)


def consistent_hf_dict(raw, count):
    """Repair only a Qwen text config already declaring the requested prefix.

    A valid full checkpoint remains full here. Its construction limit is applied
    by the original loader adapter, not by the tokenizer's config reader.
    """
    if raw.get('model_type') not in ('qwen3_5_moe', 'qwen3_5_moe_text'):
        return raw, []
    text = raw.get('text_config', raw)
    if not isinstance(text, dict) or text.get('num_hidden_layers') != count:
        return raw, []
    changes = []
    for name in LAYER_LISTS:
        values = text.get(name)
        if isinstance(values, list) and len(values) > count:
            changes.append(dict(field=name, original_count=len(values), selected_count=count))
    if not changes:
        return raw, []
    result = deepcopy(raw)
    selected = result.get('text_config', result)
    for change in changes:
        name = change['field']
        selected[name] = selected[name][:count]
    return result, changes


@contextmanager
def hf_config_prefix(model_path, count, report):
    """Normalize only this local checkpoint's HF reads, before strict validation."""
    if type(count) is not int or count < 1:
        raise ValueError('reference layer count must be positive')
    from transformers import PretrainedConfig
    from .common import sha
    root = Path(model_path).resolve()
    config_path = root/'config.json'
    record = dict(policy='PROCESS_LOCAL_QWEN_PREFIX_LISTS', requested_layers=count,
        source_config=dict(path=str(config_path), sha256=sha(config_path)),
        checkpoint_files_modified=False, hf_reads=0, adaptations=[])
    report['config_compatibility'] = record
    original = vars(PretrainedConfig)['get_config_dict']

    def read(cls, path, **kwargs):
        raw, unused = original.__get__(None, cls)(path, **kwargs)
        if Path(path).resolve() not in (root, config_path):
            return raw, unused
        record['hf_reads'] += 1
        fixed, changes = consistent_hf_dict(raw, count)
        for change in changes:
            if change not in record['adaptations']:
                record['adaptations'].append(change)
                print(f'[target-probe] in-memory {change["field"]}: '
                      f'{change["original_count"]} -> {count}; checkpoint unchanged', flush=True)
        return fixed, unused

    PretrainedConfig.get_config_dict = classmethod(read)
    try:
        yield
    finally:
        PretrainedConfig.get_config_dict = original

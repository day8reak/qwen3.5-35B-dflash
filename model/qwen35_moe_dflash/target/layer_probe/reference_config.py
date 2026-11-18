"""Keep a bounded reference config consistent without writing the checkpoint."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path


LAYER_LISTS = ('layer_types', 'mlp_layer_types')


def read_trace_config(yaml_path):
    """Resolve local resource paths from the original inference working directory."""
    import yaml

    path = yaml_path.resolve(strict=True)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get('model_file'), str) or not config['model_file'].strip():
        raise ValueError(f'{path}: missing model_file')
    model = Path(config['model_file']).expanduser().resolve()
    config_file = model / 'config.json'
    raw = json.loads(config_file.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'{config_file}: expected a config object')
    text = raw.get('text_config', raw)
    if not isinstance(text, dict) or type(text.get('num_hidden_layers')) is not int:
        raise ValueError(f'{config_file}: missing integer num_hidden_layers')
    lengths = {}
    for name in LAYER_LISTS:
        if text.get(name) is not None:
            if not isinstance(text[name], list):
                raise ValueError(f'{config_file}: {name} must be a list')
            lengths[name] = len(text[name])
    w4 = config.get('quanted_pth')
    return raw, dict(yaml=str(path), model_file=str(model), hf_config=str(config_file),
                     num_hidden_layers=text['num_hidden_layers'], list_lengths=lengths,
                     quanted_pth=str(Path(w4).expanduser().resolve()) if w4 else None)


def trace_layer_policy(raw, info, count):
    """Only adapt a declared, intentionally selected four-layer prefix."""
    description = f"{info['hf_config']}: num_hidden_layers={info['num_hidden_layers']}, {info['list_lengths']}"
    if info['num_hidden_layers'] != count:
        raise ValueError(f'{description}; requested {count} layers. Select the YAML used by that reference run.')
    lengths = info['list_lengths'].values()
    if all(length == count for length in lengths):
        return False
    if count == 4 and all(length >= count for length in lengths):
        _, changes = consistent_hf_dict(raw, count)
        if changes:
            return True
    raise ValueError(f'{description}; inconsistent layer lists for the requested model')


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

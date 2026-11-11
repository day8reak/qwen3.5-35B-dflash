"""Load only one boundary's original weights; keep the captured golden intact."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .common import sha


def parameter_dtypes(record, captured, target):
    """Evaluate only the original parameter constructors needed by this region."""
    from .source_regions import definition
    kind = record['kind']
    if kind in ('norm', 'norm_gated'):
        cls = target.Qwen3_5MoeRMSNorm if kind == 'norm' else target.Qwen3_5MoeRMSNormGated
        return {'weight': cls(captured['weight'].numel(), eps=record['attrs']['epsilon']).weight.dtype}
    if kind in ('conv', 'gates'):
        owner = target.Qwen3_5MoeGatedDeltaNet.__init__
        tree, evidence = definition(owner)
        state = (SimpleNamespace(conv_dim=captured['weight'].shape[0], conv_kernel_size=captured['weight'].shape[-1])
                 if kind == 'conv' else SimpleNamespace(num_v_heads=captured['A_log'].numel()))
        wanted = {'self.conv1d'} if kind == 'conv' else {'A', 'self.A_log', 'self.dt_bias'}
        body = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and ast.unparse(n.targets[0]) in wanted]
        if {ast.unparse(n.targets[0]) for n in body} != wanted:
            raise ValueError('CPro parameter constructor changed: ' + kind)
        namespace = dict(owner.__globals__, self=state)
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), evidence['file'], 'exec'), namespace)
        if kind == 'conv':
            return {name: getattr(state.conv1d, name).dtype for name in captured}
        return {name: getattr(state, name).dtype for name in captured}
    if kind == 'linear':
        w = captured['weight']
        op = torch.nn.Linear(w.shape[1], w.shape[0], bias='bias' in captured, device='meta')
        return {name: getattr(op, name).dtype for name in captured}
    raise ValueError('unsupported floating weight region: ' + kind)


def read_tensors(directory, keys):
    from safetensors import safe_open
    directory = Path(directory).resolve()
    indexes = sorted(directory.glob('*.safetensors.index.json'))
    locations = {}
    if indexes:
        for path in indexes:
            mapping = json.loads(path.read_text())['weight_map']
            for key in keys:
                if key in mapping:
                    if key in locations:
                        raise ValueError('duplicate checkpoint tensor: ' + key)
                    locations[key] = directory / mapping[key]
    else:
        for path in sorted(directory.glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as f:
                for key in keys.intersection(f.keys()):
                    if key in locations:
                        raise ValueError('duplicate checkpoint tensor: ' + key)
                    locations[key] = path
    values = {}
    for key, path in locations.items():
        if directory not in path.resolve().parents:
            raise ValueError('checkpoint index escapes its directory')
        with safe_open(path, framework='pt', device='cpu') as f:
            values[key] = f.get_tensor(key)
    return values


def load_weights(record, captured, target, converters, config_path, source_root):
    if not captured:
        return captured, dict(status='NO_WEIGHTS')
    if not config_path:
        raise ValueError('weighted replay requires --target-config with original model_file and quanted_pth; '
                         '310P captured weights alone do not establish CPro parameter dtype')
    import yaml
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())

    def directory(key):
        value = Path(config[key]).expanduser()
        # Match the original command's source-root cwd, not the worker's output.
        return (value if value.is_absolute() else Path(source_root) / value).resolve()

    kind, point = record['kind'], record['point']
    modules = {'conv1d': 'linear_attn.conv1d', 'decay_beta': 'linear_attn',
        'input_layernorm': 'input_layernorm', 'post_attention_layernorm': 'post_attention_layernorm',
        'linear_attn_norm': 'linear_attn.norm', 'self_attn_q_norm': 'self_attn.q_norm',
        'self_attn_k_norm': 'self_attn.k_norm'}
    name = modules.get(point)
    if name is None:
        for prefix in ('linear_attn_', 'self_attn_'):
            if point.startswith(prefix):
                name = prefix[:-1] + '.' + point[len(prefix):]
                break
    if name is None:
        raise ValueError('no original checkpoint mapping for boundary: ' + point)
    layer_name = f'layers.{record["layer"]}.{name}'
    evidence = dict(status='ORIGINAL_CHECKPOINT_BOUND', config=str(config_path), config_sha256=sha(config_path),
        parameter=layer_name, captured_dtype={k: str(v.dtype) for k, v in captured.items()})
    if kind == 'qlinear':
        if captured['W_q'].dtype != torch.int32 or captured['W_q'].ndim != 2:
            raise ValueError('this Target regeneration requires original W4 INT32 QLinear capture')
        keys = {f'model.{layer_name}_quant_{suffix}' for suffix in ('weight', 'scale')}
        tensors = read_tensors(directory('quanted_pth'), keys)
        if tensors.keys() != keys:
            raise ValueError('missing original quantized tensors: ' + str(sorted(keys - tensors.keys())))
        # Invoke the SAME replacement utility on ONE Linear named like the
        # original layer. It owns I4ZN packing, scale transpose and factory.
        shell = torch.nn.Module()
        parent = shell
        parts = ('model.' + layer_name).split('.')
        for part in parts[:-1]:
            child = torch.nn.Module()
            parent.add_module(part, child)
            parent = child
        k, packed_n = captured['W_q'].shape
        parent.add_module(parts[-1], torch.nn.Linear(k, packed_n * 8, bias=False, device='meta'))
        replace = converters.quant_model_inplace.__globals__['replace_linear_to_QLinear']
        replace(shell, tensors, start_layer_idx=0, quant_bits=4)
        op = shell.get_submodule('.'.join(parts))
        result = dict(W_q=op.W_q.detach().cpu().contiguous(), scale=op.scale.detach().cpu().contiguous())
        evidence['preparation'] = 'ORIGINAL_REPLACE_LINEAR_TO_QLINEAR; ONE_LINEAR_ONLY; QUANT_BITS_4'
    else:
        candidates = {name: [prefix + layer_name + '.' + name for prefix in ('model.language_model.', 'model.', '')]
                      for name in captured}
        tensors = read_tensors(directory('model_file'), {key for keys in candidates.values() for key in keys})
        result = {}
        dtypes = parameter_dtypes(record, captured, target)
        for name, keys in candidates.items():
            matches = [key for key in keys if key in tensors]
            if len(matches) != 1:
                raise ValueError('need one original HF tensor for ' + layer_name + '.' + name)
            value = tensors[matches[0]]
            if value.shape != captured[name].shape or not torch.equal(value.to(captured[name].dtype), captured[name]):
                raise ValueError('capture and original HF weights differ: ' + matches[0])
            result[name] = value.to(dtypes[name]).contiguous()
        evidence['preparation'] = 'RAW_HF_TO_CPRO_PARAMETER_DTYPE; NO_GAMMA_PLUS_ONE_ON_LOAD'
    if kind == 'qlinear':
        for name, value in result.items():
            if value.shape != captured[name].shape or value.dtype != captured[name].dtype or not torch.equal(value, captured[name]):
                raise ValueError('capture differs from original CPro quantized preparation: ' + name)
    evidence['target_dtype'] = {k: str(v.dtype) for k, v in result.items()}
    return result, evidence

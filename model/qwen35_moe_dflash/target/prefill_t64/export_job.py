"""Export the same four-layer, full-length T64 case as the retained 310P capture."""
import hashlib
import json

import numpy as np
import torch
import yaml

from ..layer_probe.steps import working_directory


def export_air(job):
    with working_directory(job):
        _export_air(job)


def _export_air(job):
    import export_glm_model_qwen3_5_moe as exporter
    from models.modeling_qwen3_5_moe_hiai_cpro import Qwen3_5MoeRMSNorm

    air = job / 'air'
    if air.exists():
        raise FileExistsError(f'Export already exists: {air}')
    config = yaml.safe_load((job / 'config.yaml').read_text())
    manifest = json.loads((job / 'inputs.json').read_text())
    data = {}
    for name, row in manifest.items():
        raw = (job / row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'Prepared input changed: {name}')
        dtype = np.dtype(row['dtype']).newbyteorder('<')
        array = np.frombuffer(raw, dtype=dtype).reshape(row['shape']).copy()
        data[name] = torch.from_numpy(array)

    wrapper = exporter.Qwen3_5MoeForCausalLMCproWrapper(
        model_path=config['model_file'], device='cpu', dtype=torch.float16,
        embedding_config={'embedding_in_omc': False, 'mul_twice': False}, quant_enable=True,
    ).eval()
    layers = wrapper.model.model.layers
    if len(layers) != 4 or [layer.block_type for layer in layers] != ['linear_attention'] * 3 + ['full_attention']:
        raise ValueError('Expected L0–L2 GDN followed by L3 full attention')
    if wrapper.model.config.kv_cache_max_len != 2048:
        raise ValueError('Capture requires KV capacity 2048')
    exporter.quant_model_inplace(wrapper, config['quanted_pth'], wrapper.model.config.num_experts,
                                device='cpu', quant_bits=4, qlinear_cls=exporter.CproQLinear)
    for i, layer in enumerate(layers):
        if i < 3:
            attention = layer.linear_attn
            names = ('in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z', 'out_proj')
        else:
            attention = layer.self_attn
            names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        if not all(isinstance(getattr(attention, name), exporter.CproQLinear) for name in names):
            raise ValueError(f'L{i} contains an unbound attention projection')
    if not isinstance(wrapper.model.lm_head, exporter.CproQLinear):
        raise ValueError('Missing CPro W4 LM head')
    with torch.no_grad():
        for module in wrapper.modules():
            if isinstance(module, Qwen3_5MoeRMSNorm):
                module.register_buffer('export_gamma', (module.weight + 1.0).detach().contiguous(), persistent=False)
        if not torch.equal(layers[3].self_attn.block_table.cpu(), torch.arange(64, dtype=torch.int32).reshape(1, 64)):
            raise ValueError('L3 page table differs from fresh sequential Prefill')
        states = [(data[f'conv_state_{i}'], data[f'recurrent_state_{i}']) for i in range(3)]
        states.append((data['key_cache_3'], data['value_cache_3']))
        with exporter.cpro_export_contracts(True):
            exporter.torchair.dynamo_export(
                model=wrapper, export_path='air', dynamic=False, input_ids=data['input_ids'],
                inputs_embeds=data['inputs_embeds'], embed_scale=data['embed_scale'],
                past_key_values=states, new_kv_cache_pos=data['new_kv_cache_pos'], allQLen=data['allQLen'],
                attention_mask=data['attention_mask'], position_ids=data['position_ids'], output_pos=data['output_pos'],
                effective_length=data['effective_length'], use_cache=True, export_flag=True, output_attentions=False,
                topK=None, linear_update_tensor=None)
    if not (air / 'export.air').is_file():
        raise FileNotFoundError(f'TorchAir completed but {air / "export.air"} is missing')
    print(f'Four-layer T64 AIR: {air / "export.air"}')

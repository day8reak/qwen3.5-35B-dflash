"""Export one static first-prefill graph by invoking the installed CPro wrapper."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

from ...common.artifacts import share_for_transfer
from ...draft.incremental.torch_export import _producer_schema, _schema
from ...draft.incremental.torch_weight_export import external_weight_export, validate_external_weights
from ..export_inventory import _file_constants
from ..layer_probe.common import checked, identity, new_output, read_array, save_json, sha
from ..layer_probe.replay import export_input_identity, source_bindings
from ..layer_probe.steps import working_directory
from ..operator_air_review import _data_edges, port_names
from ..original_air_roundtrip import _attributes, _decode, _read_air
from .chain_air import checkpoint_paths, export_progress, stage_cpro_source
from .chain_reference import BASIS, LAYERS, VALID19_IDS_SHA256
from .prepare import EXPORTER, MODELING, NZ, WRAPPER
from .source_wrapper import stage_wrapper


def logits_wrapper(base):
    class FirstToken(base):
        def forward(self, *, past_key_values, **kwargs):
            # The original GDN writes its two caches in-place. FullAttention's
            # staged update already uses private clones. Only logits escape.
            states = [pair if i % 4 == 3 else (pair[0].clone(), pair[1].clone())
                      for i, pair in enumerate(past_key_values)]
            return super().forward(past_key_values=states, **kwargs)[0]
    return FirstToken


def model_inputs(root, meta):
    def tensor(name):
        return torch.from_numpy(read_array(root, meta['inputs'][name]))

    data = dict(input_ids=tensor('input_ids').int(), position_ids=tensor('position_ids').int(),
                new_kv_cache_pos=tensor('new_kv_cache_pos').float(), attention_mask=tensor('attention_mask').float(),
                allQLen=torch.tensor([19], dtype=torch.int64), effective_length=tensor('effective_length').short(),
                output_pos=torch.tensor([18], dtype=torch.int32))
    pos = torch.arange(64)
    mask = data['attention_mask']
    allowed = (pos[:, None] < 19) & (torch.arange(2048)[None] <= pos[:, None])
    if (data['effective_length'].tolist() != [19] or tuple(data['input_ids'].shape) != (1, 64)
            or not (data['input_ids'][:, 19:] == 1).all()
            or not torch.equal(data['position_ids'], pos[None])
            or not torch.equal(data['new_kv_cache_pos'], pos)
            or mask.shape != (1, 1, 64, 2048) or not torch.equal(mask[0, 0] == 0, allowed)):
        raise ValueError('Expected the original T64/valid19 inputs')
    call = dict(data)
    states = []
    for i in range(LAYERS):
        shapes = [(1, 8192, 4), (1, 32, 128, 128)] if i % 4 != 3 else [(64, 32, 32, 16)] * 2
        pair = tuple(torch.zeros(shape, dtype=torch.float16) for shape in shapes)
        for j, value in enumerate(pair):
            data[f'l{i:02d}_state{j}'] = value
        states.append(pair)
    call.update(past_key_values=states, inputs_embeds=None, embed_scale=None, topK=None, use_cache=True,
                export_flag=True, output_attentions=False, linear_update_tensor=None)
    return data, call


@torch.no_grad()
def export(source, config, reference, output):
    reference = reference.resolve(strict=True)
    meta = json.loads((reference / 'manifest.json').read_text())
    if (meta['status'] != 'CAPTURED' or meta['reference'] != BASIS or
            (meta['physical_rows'], meta['valid_rows'], meta['output_pos']) != (64, 19, 18) or
            [b['layer'] for b in meta['blocks']] != list(range(LAYERS))):
        raise ValueError('Expected a complete 40-layer 310P reference for T64/valid19')
    paths = checkpoint_paths(source, (source / config).resolve(strict=True), meta)
    root = new_output(output)
    data, call = model_inputs(reference, meta)
    ids = data['input_ids'].numpy().astype('<i4', copy=False)
    if hashlib.sha256(ids.tobytes()).hexdigest() != VALID19_IDS_SHA256:
        raise ValueError('310P capture uses a different prompt from the valid19 case')
    with tempfile.TemporaryDirectory(dir=root.parent, prefix='.cpro40-source-') as tmp:
        stage = Path(tmp)
        stage_cpro_source(source, stage, lambda s: stage_wrapper(s, segmented=True, layers=40))
        sys.path.insert(0, tmp)
        try:
            import export_glm_model_qwen3_5_moe as exporter
            from models.modeling_qwen3_5_moe_hiai_cpro import Qwen3_5MoeRMSNorm
            print('Loading original CPro 40-layer wrapper and W4 weights', flush=True)
            model = logits_wrapper(exporter.Qwen3_5MoeForCausalLMCproWrapper)(
                model_path=str(paths['model_file']), device='cpu', dtype=torch.float16,
                embedding_config={'embedding_in_omc': True, 'mul_twice': False}, quant_enable=True).eval()
            layers = model.model.model.layers
            if ([x.block_type for x in layers] != [b['block_type'] for b in meta['blocks']]
                    or model.model.config.kv_cache_max_len != 2048):
                raise ValueError('Original wrapper layer or KV configuration differs from the reference')
            exporter.quant_model_inplace(model, str(paths['quanted_pth']), model.model.config.num_experts,
                                         device='cpu', quant_bits=4, qlinear_cls=exporter.CproQLinear)
            if not isinstance(model.model.lm_head, exporter.CproQLinear):
                raise ValueError('Original CPro W4 LM head was not loaded')
            for i, layer in enumerate(layers):
                if layer.block_type == 'linear_attention':
                    attn = layer.linear_attn
                    names = ('in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z', 'out_proj')
                else:
                    attn = layer.self_attn
                    names = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
                if not all(isinstance(getattr(attn, name), exporter.CproQLinear) for name in names):
                    raise ValueError(f'L{i} has an unquantized attention projection')
            model.model.to(dtype=torch.float16)
            for module in model.modules():
                if isinstance(module, Qwen3_5MoeRMSNorm):
                    module.register_buffer('export_gamma', (module.weight + 1.).detach().contiguous(), persistent=False)
            embedding = model.model.model.embed_tokens(data['input_ids']).half()
            if not np.array_equal(embedding.numpy(), read_array(reference, meta['inputs']['embedding'])):
                raise ValueError('CPro embedding differs from the first 310P decoder input')
            for layer in layers[3::4]:
                if not torch.equal(layer.self_attn.block_table.cpu(), torch.arange(64)[None]):
                    raise ValueError('Full attention page table differs from the reference')
            with working_directory(root), external_weight_export() as weights, export_input_identity(list(data.values())) as evidence:
                with exporter.cpro_export_contracts(True, chunk_size=64, gdn_layers=30), export_progress('single Prefill40', root / 'weight'):
                    exporter.torchair.dynamo_export(model=model, export_path='weight', dynamic=False, **call)
        finally:
            sys.path.remove(tmp)
    air = root / 'export.air'
    (root / 'weight/export.air').rename(air)
    _, _, graph = _decode(_schema(*_producer_schema()), _read_air(air)[0])
    file_nodes = {n['name']: str(Path(_attributes(n)['file_path']['s'])) for n in _file_constants(graph)}
    if file_nodes != {r['node']: r['path'] for r in weights['files']}:
        raise ValueError('AIR FileConstants differ from the tensors written by the original exporter')
    files = [identity(root / p, root) for p in sorted(set(file_nodes.values()))]
    validate_external_weights({'weight_files': {s['path']: s['bytes'] for s in files}}, weights)
    inputs = []
    (root / 'inputs').mkdir()
    for name, value in data.items():
        file = root / 'inputs' / (name + '.bin')
        a = value.numpy()
        a.tofile(file)
        inputs.append(dict(identity(file, root), semantic_name=name, dtype=a.dtype.name, shape=list(a.shape)))
    ports = source_bindings(graph, inputs, evidence['nodes'], 1)
    edges = {s['semantic_name']: s['name'] + ':0' for s in ports}
    counts = Counter(n['type'] for n in graph['op'])
    if not any(n.lower().startswith('groupedmatmul') for n in counts):
        raise ValueError('No grouped expert matmul in exported MoE graph')
    for kind, port, control, number in [('ChunkGatedDeltaRule', 'effective_length', 'effective_length', 30),
                                        ('AdnFusedInferAttention', 'all_seq_lengths_q', 'allQLen', 10)]:
        if counts[kind] != number:
            raise ValueError(f'Expected {number} {kind} calls, found {counts[kind]}')
        for op in (n for n in graph['op'] if n['type'] == kind):
            attrs = _attributes(op)
            slot, = [i for i, n in port_names(attrs, 'input').items() if n == port]
            if _data_edges(op)[slot] != edges[control] or (kind == 'ChunkGatedDeltaRule' and attrs['chunk_size']['i'] != 64):
                raise ValueError(f'Unexpected prefill controls on {op["name"]}')
    if set(edges) != set(data):
        raise ValueError(f'Unexpected AIR ports; missing {set(data) - set(edges)}')
    golden = meta['outputs']['logits']
    if golden['shape'] != [1, 1, 248320] or golden['dtype'] != 'float32':
        raise ValueError('Expected the 310P full vocabulary row')
    shutil.copyfile(checked(reference, golden), root / 'logits.bin')
    shutil.copyfile(checked(reference, meta['token_text_map']), root / 'token-text.json')
    save_json(root / 'manifest.json', dict(
        kind='single-cpro-prefill40', reference=sha(reference / 'manifest.json'),
        source={n: sha(source / n) for n in (MODELING, WRAPPER, EXPORTER, NZ, 'utils.py')},
        air=identity(air, root), weights=files, inputs=ports,
        golden=dict(golden, path='logits.bin', name='logits'), ops=dict(counts),
        token_text_map=identity(root / 'token-text.json', root),
        reference_top1=meta['top1'], reference_text=meta['token_text']))
    share_for_transfer(root)
    print(f'One AIR: {air}; 30 GDR, 10 full attention, 1 logits output', flush=True)

"""Capture L0 in the original segmented 310P call, plus an independent CPU W4 forward."""
import json
from functools import wraps
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from qwen35_moe_dflash.common.artifacts import share_for_transfer
from qwen35_moe_dflash.target import prefill_fixture as pf
from qwen35_moe_dflash.target.layer_probe.capture import cpu
from qwen35_moe_dflash.target.layer_probe.common import read_tensor, save_json, sized_file
from qwen35_moe_dflash.target.layer_probe.qlinear_math import dynamic_quant
from qwen35_moe_dflash.target.prefill_t64.cpu_reference import CpuPrefill, norm, verify_floating_weights
from qwen35_moe_dflash.target.prefill_t64.gdn_layer_reference import capture, evaluate
from qwen35_moe_dflash.target.prefill_t64.decoder_embedding import retained_embedding
from trace_segment_attention import observe_l1_gdr_length


def observe_decoder(block, stack, done, values):
    import torch_npu
    from torch.nn import functional as F

    def put(role, value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        values[role] = cpu(value).numpy()

    def hook(module, role, *, before=False):
        if before:
            def pre(mod, args, kwargs):
                put(role, args[0] if args else next(iter(kwargs.values())))
            handle = module.register_forward_pre_hook(pre, with_kwargs=True)
        else:
            handle = module.register_forward_hook(lambda mod, args, result: put(role, result))
        stack.callback(handle.remove)

    moe = block.mlp
    shared = moe.shared_expert
    for module, role in [(block.input_layernorm, 'block_entry'), (block.post_attention_layernorm, 'residual'),
                         (shared.down_proj, 'shared_product')]:
        hook(module, role, before=True)
    for module, role in [(block.post_attention_layernorm, 'post_norm'), (moe, 'moe_out'),
                         (moe.gate, 'router_logits'), (moe.shared_expert_gate, 'shared_gate_logit'),
                         (shared.gate_proj, 'shared_gate_proj'), (shared.up_proj, 'shared_up'),
                         (shared.down_proj, 'shared_down')]:
        hook(module, role)
    def quantized(module, role):
        forward = module.forward
        original = torch_npu.npu_dynamic_quant
        @wraps(forward)
        def tapped(*args, **kw):
            @wraps(original)
            def quant(*a, **options):
                result = original(*a, **options)
                put(role+'_a8', result[0])
                put(role+'_scale', result[1])
                return result
            with patch.object(torch_npu, 'npu_dynamic_quant', quant):
                return forward(*args, **kw)
        stack.enter_context(patch.object(module, 'forward', tapped))
    for module, role in [(shared.gate_proj, 'shared_gate_proj'), (shared.up_proj, 'shared_up'),
                         (shared.down_proj, 'shared_down'), (moe.gate, 'router_logits')]:
        quantized(module, role)

    if isinstance(getattr(shared, 'act_fn', None), torch.nn.Module):
        hook(shared.act_fn, 'shared_silu')

    # Observe actual operands/results, including the router decisions used for expert dispatch.
    def watch(name, collect):
        original = getattr(torch_npu, name, None)
        if original is None:
            return
        @wraps(original)
        def wrapped(*args, **kw):
            output = original(*args, **kw)
            collect(args, kw, output)
            return output
        stack.enter_context(patch.object(torch_npu, name, wrapped))
        namespace = getattr(type(moe).forward, '__globals__', {})
        for alias in [key for key, value in namespace.items() if value is original]:
            namespace[alias] = wrapped
            stack.callback(namespace.__setitem__, alias, original)


    def dispatch(args, kw, result):
        put('expert_ids', kw['expert_idx'] if 'expert_idx' in kw else args[1])
        put('route_map', result[1])

    def grouped(args, kw, result):
        put('expert_gate_up' if 'expert_gate_up' not in values else 'expert_down', result)

    def finalize(args, kw, result):
        put('routed', result)
        if 'scales' in kw or len(args) > 4:
            put('route_weights', kw['scales'] if 'scales' in kw else args[4])

    watch('npu_moe_init_routing_v2', dispatch)
    watch('npu_grouped_matmul', grouped)
    watch('npu_swiglu', lambda a, k, y: put('expert_product', y))
    watch('npu_moe_finalize_routing', finalize)

    # Read the shared contribution in the original Python forward after its sigmoid/multiply.
    # The model can use F.silu directly or store the activation as a module or callable.
    shared_forward = shared.forward
    @wraps(shared_forward)
    def observe_shared(*args, **kwargs):
        original = F.silu
        @wraps(original)
        def activated(*a, **kw):
            y = original(*a, **kw)
            put('shared_silu', y)
            return y
        with patch.object(F, 'silu', activated):
            if getattr(shared, 'act_fn', None) is original:
                with patch.object(shared, 'act_fn', activated):
                    return shared_forward(*args, **kwargs)
            return shared_forward(*args, **kwargs)
    stack.enter_context(patch.object(shared, 'forward', observe_shared))

    forward = moe.forward
    @wraps(forward)
    def observed(*args, **kwargs):
        previous = sys.getprofile()
        def record(frame, event, arg):
            if event == 'return' and frame.f_code == forward.__func__.__code__ and frame.f_locals.get('self') is moe:
                weighted = frame.f_locals.get('shared_expert_output')
                if isinstance(weighted, torch.Tensor):
                    put('shared_weighted', weighted)
            if previous is not None:
                previous(frame, event, arg)
        sys.setprofile(record)
        try:
            return forward(*args, **kwargs)
        finally:
            sys.setprofile(previous)
    stack.enter_context(patch.object(moe, 'forward', observed))

    def leave(mod, args, result):
        put('block_out', result)
        raise done()
    stack.callback(block.register_forward_hook(leave).remove)


def prepare(root, source, config):
    fixture = root/'export/fixture'
    capture_reference(fixture, root/'l0-steps-reference', source, config)


@torch.inference_mode()
def capture_reference(fixture, folder, source, config):
    ref = json.loads((fixture/'reference.json').read_bytes())
    valid = ref['valid_rows']
    def tensor(name):
        row = ref['inputs'][name]
        return read_tensor(sized_file(fixture, row), row)
    extra = {}
    native, params, info = capture(source, config, observe_l1_gdr_length, input_ids=tensor('input_ids'),
                                  embedding_rows=retained_embedding(fixture, ref),
                                  valid_rows=valid, observe_block=lambda b,s,d: observe_decoder(b,s,d,extra))
    if not np.array_equal(extra['block_entry'], tensor('hidden')):
        raise ValueError('L0 input hidden differs from the retained decoder reference')
    for native_key, role in [('new_history','conv_state_0'), ('recurrent_cache','recurrent_state_0')]:
        row = ref['outputs'][role]
        if not np.array_equal(native[native_key], read_tensor(sized_file(fixture, row), row)):
            raise ValueError(f'310P {role} differs from the retained decoder reference')
    if Path(info['quant_weights']).resolve() != Path(ref['quanted_pth']).resolve():
        raise ValueError('310P and the decoder reference use different W4 checkpoints')
    owner = pf.import_source(source)
    loader = sys.modules[owner.Qwen3_5MoeForCausalLMWrapper.__module__].load_npu_model
    checkpoint = Path(ref['model_file'])
    with pf.working_directory(source), pf.hf_config_prefix(checkpoint, 4, {}):
        model = loader(str(checkpoint), 'cpu', torch.float16, variant='nz', layer_num=1,
                       start_layer_idx=0, quant_enable=True)
    model.eval().to(dtype=torch.float16)
    verify_floating_weights(model, checkpoint)
    block = model.model.layers[0]
    reference = CpuPrefill(SimpleNamespace(model=model), ref['quanted_pth'])
    entry = torch.from_numpy(tensor('hidden'))
    golden = evaluate(dict(native, hidden=norm(entry, block.input_layernorm).numpy()), params)
    mix = torch.from_numpy(golden['out'].reshape(1,64,2048)).clone()
    mix[:,valid:] = 0
    residual = entry + mix
    post = norm(residual, block.post_attention_layernorm)
    stages = {}
    moe_out = reference.moe(post, block.mlp, 'layers.0.mlp', stages=stages)
    stages.update(block_entry=entry, residual=residual, post_norm=post, moe_out=moe_out, block_out=residual+moe_out)
    golden.update({k: v.detach().numpy() for k,v in stages.items()})
    for role in ('shared_gate_proj','shared_up','shared_down','router_logits'):
        x = stages['shared_product'] if role == 'shared_down' else post
        codes, scale, _ = dynamic_quant(x.reshape(64,-1).numpy())
        golden[role+'_a8'], golden[role+'_scale'] = codes, scale
    native.update(extra)
    arrays = {backend+'_'+k: a for backend, values in [('cpu', golden), ('310p', native)] for k,a in values.items()}
    folder.mkdir(exist_ok=True)
    dest = folder/'values.npz'
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        staged = Path(temp)/'values.npz'
        np.savez(staged, **arrays, valid=np.array(valid), ids=tensor('input_ids'),
                 fixture_hidden=np.array(ref['inputs']['hidden']['sha256']),
                 fixture_state=np.array(ref['outputs']['recurrent_state_0']['sha256']))
        staged.replace(dest)
    if ref.get('binding', {}).get('dump'):
        save_json(folder/'tensors.json', dict(valid_rows=valid, archive=dest.name,
                  tensors={k: dict(dtype=str(a.dtype), shape=list(a.shape)) for k, a in arrays.items()}))
    share_for_transfer(folder)
    print(f'L0 参考值 CPU {len(golden)} 项、310P {len(native)} 项：{dest}', flush=True)

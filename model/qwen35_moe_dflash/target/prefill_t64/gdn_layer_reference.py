"""Capture L0 NZ Prefill once and evaluate the declared CPU A8W4 reference."""
from contextlib import ExitStack
from functools import wraps
import hashlib
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from ..layer_probe.capture import bound, cpu
from ..layer_probe.common import sha
from ..layer_probe.qlinear_math import dynamic_quant, group_matmul
from ..original_quant import OriginalQuantWeights, pack_i4zn
from .chain_reference import VALID19_IDS_SHA256, run_segment_prefill
from .gdn_layer_stages import ENTRY, PROJECTIONS, SHAPES, STAGES, canonical
from .gdr_math import compute
from .gdr_stage_capture import observe_norm
from .qkv_silu import STAGE as CONV_SILU, observe_conv


def capture(source, config, observe_gdr):
    import torch_npu
    import yaml

    entry = (source / 'inference_segment.py').resolve(strict=True)
    config = (source / config).resolve(strict=True)
    weights = OriginalQuantWeights(source / yaml.safe_load(config.read_text())['quanted_pth'])
    snapshot, observed, params, log = {}, {}, {}, {}
    info = dict(driver_sha256=sha(entry), config_sha256=sha(config), quant_weights=str(weights.root), projections={})

    class Done(Exception):
        pass

    def put(name, tensor):
        if name in observed:
            raise ValueError(f'repeated L0 capture at {name}')
        observed[name] = canonical(name, cpu(tensor).numpy())

    with ExitStack() as stack:
        def arm(frame):
            wrapper = frame.f_locals['model_wrapper']
            text = wrapper.model.model
            if text.start_layer_idx != 0 or text.layers[0].block_type != 'linear_attention':
                raise ValueError('expected first layer of the original NZ linear-attention segment')
            layer = text.layers[0].linear_attn

            def verify_prompt(mod, args, kw):
                ids = cpu(bound(mod, args, kw)['input_ids']).numpy().astype('<i4')
                digest = hashlib.sha256(ids.tobytes()).hexdigest()
                if ids.shape != (1,64) or digest != VALID19_IDS_SHA256:
                    raise ValueError('unexpected T64 valid19 prompt')
                info['prompt_sha256'] = digest
            stack.callback(wrapper.register_forward_pre_hook(verify_prompt, with_kwargs=True).remove)

            def enter(mod, args, kw):
                vals = bound(mod, args, kw)
                x = vals['hidden_states']
                if x.device.type != 'npu' or tuple(x.shape) != SHAPES['hidden']:
                    raise ValueError('expected the 310P GDN module entry')
                info['device'] = str(x.device)
                history, state = vals['past_key_values']
                for name, value in zip(ENTRY, (x, history, state, vals['effective_length']), strict=True):
                    put(name, value)
                if observed['length'].tolist() != [19] or observed['history'].any() or observed['state'].any():
                    raise ValueError('expected fresh T64 prefill with 19 valid rows')
            stack.callback(layer.register_forward_pre_hook(enter, with_kwargs=True).remove)
            observe_gdr(wrapper, log, stack, snapshot, layer_index=0)
            observe_conv(layer, snapshot, stack)
            observe_norm(layer, snapshot, stack)
            impl = Path(inspect.getfile(type(layer)))
            info.update(model_source=str(impl), model_sha256=sha(impl),
                        cpu_policy='A8 ties-to-even, W4 group256 INT32 dots with FP32 scaling/accumulation')
            conv1d = F.conv1d
            @wraps(conv1d)
            def window(x, weight, *args, **kwargs):
                if x.shape[:2] == (1, 8192) and x.shape[-1] in (67, 68):
                    put('window', x[:, :, -67:])
                return conv1d(x, weight, *args, **kwargs)
            stack.enter_context(patch.object(F, 'conv1d', window))

            params.update(conv=cpu(layer.conv1d.weight), bias=cpu(layer.conv1d.bias),
                          gamma=cpu(layer.norm.weight), epsilon=float(layer.norm.variance_epsilon),
                          dt_bias=cpu(layer.dt_bias), A_log=cpu(layer.A_log))
            softplus = F.softplus
            @wraps(softplus)
            def step_size(x, *args, **kw):
                y = softplus(x, *args, **kw)
                if x.numel() == 64*32:
                    put('biased_a', x)
                    put('softplus', y)
                return y
            stack.enter_context(patch.object(F, 'softplus', step_size))

            def instrument(p, module):
                forward = module.forward
                matmul = torch_npu.npu_quant_matmul
                name = 'out_proj' if p == 'out' else 'in_proj_' + p
                key = 'target.layers.0.linear_attn.' + name + '.weight'
                @wraps(forward)
                def projected(x, *args, **kw):
                    def measured(x1, x2, scale, *tail, **options):
                        y = matmul(x1, x2, scale, *tail, **options)
                        put(p+'_a8', x1)
                        put(p+'_scale', options['pertoken_scale'])
                        put(p, y)
                        raw, groups = weights.logical(key, (PROJECTIONS[p], x1.shape[-1]))
                        packed = cpu(x2).numpy()
                        group_scale = cpu(scale).numpy()
                        if (packed.dtype != np.int32 or group_scale.dtype != np.float32
                                or not np.array_equal(pack_i4zn(raw), packed)
                                or not np.array_equal(groups[:, :, 0].T.astype('float32'), group_scale)):
                            raise ValueError(f'{name}: CPU checkpoint differs from live NZ weight or scale')
                        info['projections'][name] = dict(checkpoint_key=key,
                            packed_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
                            scale_sha256=hashlib.sha256(group_scale.tobytes()).hexdigest(),
                            packed_shape=list(packed.shape), group_scale_shape=list(group_scale.shape))
                        params[p+'_weights'] = raw, groups
                        return y
                    if p == 'out':
                        put('gated', x)
                    elif cpu(x).numpy().tobytes() != observed['hidden'].tobytes():
                        raise ValueError(f'{name} did not consume the shared GDN entry')
                    with patch.object(torch_npu, 'npu_quant_matmul', measured):
                        return forward(x, *args, **kw)
                stack.enter_context(patch.object(module, 'forward', projected))
            for p in PROJECTIONS:
                instrument(p, getattr(layer, 'out_proj' if p == 'out' else 'in_proj_'+p))

            def leave(mod, args, result):
                output, states = result
                put('new_history', states[0])
                put('recurrent_cache', states[1])
                s = snapshot['gdr']
                call, = log['l0_gdr']['calls']
                if (call.get('error') or s['attrs'] != dict(chunk_size=64, output_final_state=True,
                        use_qk_l2norm_in_kernel=True) or s['inputs']['effective_length'].tolist() != [19]):
                    raise ValueError(f'unexpected 310P GDR call: {call}')
                for k in ('query','key','value','g','beta'):
                    put(k, s['inputs'][k])
                put('core', s['outputs'][0])
                put('final_state', s['outputs'][1])
                if not np.array_equal(observed['final_state'].astype(np.float16), observed['recurrent_cache']):
                    raise ValueError('NZ returned recurrent cache differs from its FP16 GDR state writeback')
                put('conv', snapshot['conv_out'])
                put('conv_silu', snapshot[CONV_SILU])
                for k, name in [('rms', 'rms_output'), ('gate_silu','silu_output'), ('product','mul_output')]:
                    put(k, snapshot['norm'][name])
                # The NZ conv1d can process a 68-wide concat and trim one result.
                # Keep exactly the 67 input positions consumed by the retained 64 outputs.
                joined = np.concatenate([observed['history'], observed['qkv'].T[None]], axis=2)
                if (not np.array_equal(canonical('window', joined[:, :, -67:]), observed['window']) or
                        not np.array_equal(joined[:, :, -4:], observed['new_history']) or
                        not np.array_equal(cpu(output).reshape(64,2048).numpy()[:19], observed['out'][:19])):
                    raise ValueError('NZ physical history or out_proj return differs from the GDN module output')
                raise Done()
            stack.callback(layer.register_forward_hook(leave).remove)
        try:
            run_segment_prefill(entry, config, arm)
        except Done:
            pass
    if set(observed) != set(SHAPES) or 'prompt_sha256' not in info:
        raise ValueError(f'incomplete 310P capture, missing {set(SHAPES)-set(observed)}')
    return observed, params, info


def evaluate(native, params):
    """Only the four GDN entry tensors come from 310P. Every later input comes from CPU."""
    values = {k: native[k] for k in ENTRY}
    for name, (ins, outs) in STAGES.items():
        xs = [torch.from_numpy(values[k]) for k in ins]
        x = xs[0]
        if name.endswith('_quant'):
            a8, scales, _ = dynamic_quant(x.numpy().reshape(64,-1))
            result = a8, scales
        elif name.startswith('in_proj_') or name == 'out_proj':
            w, scales = params[outs[0]+'_weights']
            _, y = group_matmul(x.numpy(), xs[1].numpy().reshape(64,1), w, scales)
            result = (y,)
        elif name == 'conv_history':
            joined = torch.cat((xs[1], x.T[None]), dim=-1)
            result = joined[:,:,-67:], joined[:,:,-4:]
        elif name == 'depthwise_conv':
            result = (F.conv1d(x.squeeze(2).to(params['conv'].dtype), params['conv'], params['bias'], groups=8192),)
        elif name in ('conv_silu', 'gate_silu'):
            result = (F.silu(x if name == 'conv_silu' else x.float()),)
        elif name in ('query','key','value'):
            q, k, v = x.half().transpose(1,2).split((2048,2048,4096), dim=-1)
            part = {'query':q,'key':k,'value':v}[name].reshape(1,64,-1,128)
            result = (part.repeat_interleave(32//part.shape[2], dim=2),)
        elif name == 'decay_bias':
            result = (x.float() + params['dt_bias'].float(),)
        elif name == 'decay_softplus':
            result = (F.softplus(x),)
        elif name == 'decay_g':
            result = (-params['A_log'].float().exp() * x,)
        elif name == 'beta_sigmoid':
            result = (x.sigmoid(),)
        elif name == 'chunk_gdr':
            valid = int(xs[-1].item())
            core, state = compute(*(t.numpy()[:, :valid] for t in xs[:5]), xs[5].numpy())
            physical = np.zeros(SHAPES['core'], dtype=core.dtype)
            physical[:, :valid] = core
            result = physical, state
        elif name == 'gated_rmsnorm':
            result = (F.rms_norm(x.float(), (128,), params['gamma'].float(), params['epsilon']),)
        elif name == 'gated_mul':
            product = x * xs[1]
            result = product, product.half()
        else:
            raise ValueError(name)
        values.update((k, canonical(k, v)) for k, v in zip(outs, result, strict=True))
    values['recurrent_cache'] = values['final_state'].astype(np.float16)
    return values

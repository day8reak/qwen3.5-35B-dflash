"""Bound reference weight residency without truncating the original inference."""
from __future__ import annotations

import functools
import inspect

import torch


class LayerStreaming:
    """Construct on CPU; run each original decoder, norm and head on demand.

    Quantization remains the original inference.py operation, between wrapper
    construction and its first forward. No weight dtype or math is changed.
    Every layer in the declared reference scope executes, followed by its LM
    head. A four-layer reference must not be labelled full-model generation.
    This adapter requires the original external quantized embedding path.
    """

    def __init__(self, report, model_layers=None):
        if model_layers is not None and model_layers < 1:
            raise ValueError('model_layers must be positive')
        self.model_layers = model_layers
        self.record = dict(policy='CPU_LOAD_QUANTIZE_THEN_STREAM',
            requested_model_layers=model_layers, decoder_layer_limit=1,
            dtype_policy='DEVICE_ONLY_NO_CAST', constructed_on_cpu=0,
            max_active_units=0, wrappers=[], cleanup_errors=[])
        report['placement'] = self.record
        self.patches, self.handles, self.attached = [], [], set()
        self.active = None

    def patch(self, obj, name, callback):
        original = getattr(obj, name)
        own = name in vars(obj)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return callback(original, args, kwargs)
        setattr(obj, name, wrapped)
        self.patches.append((obj, name, original, own))

    def install(self, wrapper_module):
        cls = wrapper_module.Qwen3_5MoeForCausalLMWrapper
        if self.model_layers is not None:
            def load(original, args, kwargs):
                values = inspect.signature(original).bind(*args, **kwargs)
                values.apply_defaults()
                if 'layer_num' not in values.arguments or values.arguments.get('start_layer_idx', 0) != 0:
                    raise ValueError('four-layer reference requires original loader layer_num and start_layer_idx=0')
                values.arguments['layer_num'] = self.model_layers
                model = original(*values.args, **values.kwargs)
                total = model.config.num_hidden_layers
                if self.model_layers > total or len(model.model.layers) != self.model_layers:
                    raise ValueError('original loader did not construct the requested layer prefix')
                self.record.update(source_num_hidden_layers=total,
                    executed_num_hidden_layers=self.model_layers,
                    full_model_forward_preserved=self.model_layers == total,
                    decode_token_source=f'FIRST_{self.model_layers}_LAYERS_PLUS_ORIGINAL_NORM_LM_HEAD')
                # Original inference allocates caches and unpacks outputs using this field.
                model.config.num_hidden_layers = self.model_layers
                return model
            self.patch(wrapper_module, 'load_npu_model', load)

        def construct(original, args, kwargs):
            values = inspect.signature(original).bind(*args, **kwargs)
            values.apply_defaults()
            if 'device' not in values.arguments:
                raise ValueError('original wrapper constructor must expose device')
            requested = str(values.arguments['device'])
            values.arguments['device'] = 'cpu'
            original(*values.args, **values.kwargs)
            wrapper = args[0]
            self.record['constructed_on_cpu'] += 1
            self.record.setdefault('requested_devices', []).append(requested)
            # Catch later bulk .to(npu) in a changed inference script as well.
            self.patch(wrapper, 'to', self.cpu_only)
            self.patch(wrapper.model, 'to', self.cpu_only)
            print('[target-probe] model on CPU; original quantization runs before layer streaming', flush=True)

        def forward(original, args, kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError('layer streaming requires the original torch.no_grad() inference scope')
            values = inspect.signature(original).bind(*args, **kwargs)
            values.apply_defaults()
            wrapper = args[0]
            embeds = values.arguments.get('inputs_embeds')
            if (getattr(wrapper, 'embedding_in_omc', False)
                    or values.arguments.get('embed_scale') is None
                    or not isinstance(embeds, torch.Tensor)):
                raise ValueError('layer streaming requires external inputs_embeds + embed_scale from the original W4 inference')
            caches = values.arguments.get('past_key_values')
            if caches is None or len(caches) != len(wrapper.model.model.layers):
                raise ValueError('reference requires a cache entry for every constructed decoder layer')
            self.attach(wrapper, embeds.device)
            try:
                result = original(*args, **kwargs)
            except BaseException:
                # Keep the original failure if releasing a partial transfer fails.
                try:
                    self.offload()
                except Exception as exc:
                    self.record['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
                raise
            else:
                self.offload()
                return result

        self.patch(cls, '__init__', construct)
        self.patch(cls, 'forward', forward)

    @staticmethod
    def cpu_only(original, args, kwargs):
        device = kwargs.get('device', args[0] if args else None)
        if isinstance(device, torch.Tensor):
            device = device.device
        if device is not None and not isinstance(device, torch.dtype) and str(device) != 'cpu':
            raise RuntimeError('target probe blocks whole-model .to(device); weights are moved one layer at a time')
        return original(*args, **kwargs)

    @staticmethod
    def tensors(module):
        yield from module.parameters()
        yield from module.buffers()

    def attach(self, wrapper, device):
        if id(wrapper) in self.attached:
            return
        model = wrapper.model
        text = model.model
        layers = list(text.layers)
        expected = model.config.num_hidden_layers
        if len(layers) != expected or getattr(text, 'start_layer_idx', 0) != 0:
            raise ValueError('streaming golden requires all original decoder layers, not a truncated prefix')
        if any(t.device.type != 'cpu' for t in self.tensors(model)):
            raise ValueError('quantized model must remain on CPU before first streamed forward')
        # Reject missing W4 replacement before falling back to a large float LM.
        for name, module in [(f'layer.{i}', layer) for i, layer in enumerate(layers)] + [('lm_head', model.lm_head)]:
            quant = [m for m in module.modules() if type(m).__name__ == 'QLinear']
            if not quant or any(m.W_q.dtype != torch.int32 for m in quant):
                raise ValueError(f'{name}: original W4 QLinear replacement is missing or not INT32 packed')
        row = dict(total_decoder_layers=len(layers), execution_device=str(device), units={})
        self.record['wrappers'].append(row)
        units = [(f'layer.{i}', layer) for i, layer in enumerate(layers)]
        units += [('norm', text.norm), ('lm_head', model.lm_head)]
        for name, module in units:
            row['units'][name] = 0
            def before(m, args, name=name):
                if self.active is not None:
                    raise RuntimeError('overlapping streamed units would exceed the residency limit')
                # Track before transfer: cleanup also covers a partial failed .to.
                self.active = (m, device)
                self.move(m, device)
                row['units'][name] += 1
                self.record['max_active_units'] = max(self.record['max_active_units'], 1)
            def after(m, args, output):
                self.offload()
            self.handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
        self.attached.add(id(wrapper))
        print(f'[target-probe] streaming all {len(layers)} decoder layers, norm and head; one unit on device at a time', flush=True)

    @staticmethod
    def move(module, device):
        # Never pass dtype: packed integer weights and FP32 scales must survive.
        module.to(device)

    @staticmethod
    def synchronize(device):
        if device.type == 'npu':
            torch.npu.synchronize(device)

    @staticmethod
    def empty_cache(device):
        if device.type == 'npu':
            torch.npu.empty_cache()

    def offload(self):
        if self.active is None:
            return
        module, device = self.active
        # Drain queued kernels before releasing the weights they read.
        self.synchronize(device)
        self.move(module, torch.device('cpu'))
        self.active = None
        self.empty_cache(device)

    def close(self):
        try:
            self.offload()
        except Exception as exc:
            self.record['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            for handle in self.handles:
                handle.remove()
            for obj, name, original, own in reversed(self.patches):
                if own:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)

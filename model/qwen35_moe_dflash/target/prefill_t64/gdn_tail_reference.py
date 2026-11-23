"""Run both supplied cores through the live L0 modules of the original segmented 310P driver."""
from contextlib import ExitStack
import hashlib
import inspect
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import torch

from ..layer_probe.capture import cpu
from ..layer_probe.common import sha
from ..layer_probe.steps import working_directory
from .gdr_stage_capture import observe_norm
from .segment_gdn_trace import observe_tail


@torch.no_grad()
def capture_l0(source, config, evaluate):
    """Call evaluate while the original L0 modules and captured native inputs are available."""
    entry = (source/'inference_segment.py').resolve(strict=True)
    config = (source/config).resolve(strict=True)
    argv = [str(entry), '--config', str(config), '--quant_mode=enable', '--quant_bits=4',
            '--prefill_chunk_size=64', '--max_token=0']
    result = None
    info = dict(driver_sha256=sha(entry), config_sha256=sha(config))
    class PrefixDone(Exception):
        pass
    class BothDone(Exception):
        pass

    with working_directory(source), patch.object(sys, 'argv', argv), patch.object(sys, 'path', [str(source), *sys.path]):
        main = runpy.run_path(str(entry))['main']
        driver = main.__globals__
        chunk = driver['prefill_chunk_segment1']

        def replay(model_wrapper, input_ids, attention_mask, past_kv_1, max_len, chunk_size, device):
            nonlocal result
            layer = model_wrapper.model.model.layers[0].linear_attn
            if chunk_size != 64 or input_ids.shape[-1] != 19:
                raise ValueError('Expected original T64/valid19 prefill for L0')
            info.update(norm_source=sha(Path(inspect.getfile(inspect.unwrap(layer.norm.forward)))),
                        proj_class=type(layer.out_proj).__qualname__, epsilon=float(layer.norm.variance_epsilon),
                        gamma_sha256=hashlib.sha256(cpu(layer.norm.weight).numpy().tobytes()).hexdigest())
            original = {}
            def projected(module, args, output):
                if output.device.type != 'npu' or tuple(output.shape) not in ((1, 64, 2048), (64, 2048)):
                    raise ValueError(f'Unexpected 310P out_proj output: {output.device}, {output.shape}')
                original['device'] = output.device
                original['output'] = cpu(output).numpy().reshape(1, 64, 2048)
                raise PrefixDone()

            def norm_return(module, args, output):
                original['norm_return'] = cpu(output)

            with ExitStack() as cleanup:
                observe_tail(layer, original, cleanup)
                cleanup.callback(layer.norm.register_forward_hook(norm_return).remove)
                cleanup.callback(layer.out_proj.register_forward_hook(projected).remove)
                try:
                    chunk(model_wrapper, input_ids, attention_mask, past_kv_1,
                          max_len=max_len, chunk_size=chunk_size, device=device)
                except PrefixDone:
                    pass
            if 'output' not in original:
                raise ValueError('Native L0 replay did not reach out_proj')
            x, z, proj = (original[k] for k in ('norm_input', 'norm_gate', 'out_proj_input'))
            if (x.dtype != torch.float16 or z.dtype != x.dtype or x.shape != z.shape
                    or x.shape[-1] != 128 or x.numel() != 64*32*128
                    or proj.numpy().tobytes() != original['norm_return'].numpy().tobytes()):
                raise ValueError('Expected FP16 core/gate per head and gated norm return reshaped into out_proj')
            info['gate_shape'] = list(z.shape)
            info['input_ids'] = cpu(input_ids).reshape(-1).tolist()
            result = evaluate(layer, original, info)
            raise BothDone()

        with patch.dict(driver, prefill_chunk_segment1=replay):
            try:
                main()
            except BothDone:
                pass
    if result is None:
        raise ValueError('310P did not reach the requested L0 operator boundaries')
    return result, info


def measured(source, config, board_core, gold_core, gate, original_hidden):
    def evaluate(layer, original, info):
        x, z, proj = (original[k] for k in ('norm_input', 'norm_gate', 'out_proj_input'))
        results = {}
        info['supplied_gate_matches_native_valid19'] = (
            z.numpy().reshape(64, -1)[:19].tobytes() == gate.reshape(64, -1)[:19].tobytes())
        info['original_valid19_matches_air_fixture'] = (
            original['output'][:, :19].tobytes() == original_hidden[:, :19].tobytes())
        print('310P original L0 out_proj matches source AIR capture (valid19 bits):',
              info['original_valid19_matches_air_fixture'], flush=True)
        if not info['original_valid19_matches_air_fixture']:
            raise ValueError('Original 310P prompt/weights do not reproduce the selected L0 AIR capture')
        gate_tensor = torch.from_numpy(gate).reshape_as(z).to(original['device'])
        for label, core in [('same', board_core), ('gold', gold_core)]:
            core_tensor = torch.from_numpy(core).reshape_as(x).to(original['device'])
            plain = layer.norm(core_tensor, gate_tensor)
            snapshot = {}
            with ExitStack() as cleanup:
                observe_norm(layer, snapshot, cleanup)
                tapped = layer.norm(core_tensor, gate_tensor)
            if cpu(plain).numpy().tobytes() != cpu(tapped).numpy().tobytes():
                raise ValueError('Reading the 310P norm intermediates changed its return')
            stages = snapshot['norm']
            act = plain.reshape_as(proj)
            result = layer.out_proj(act)
            results[label] = {k: stages[k].numpy() for k in ('rms_output','silu_output','mul_output')}
            results[label].update(gated_return=cpu(act).numpy(), out_proj=cpu(result).numpy())
            for name, value in results[label].items():
                if not np.isfinite(value.reshape(64, -1)[:19]).all():
                    raise ValueError(f'{label}/{name}: nonfinite valid 310P output')
        return results

    return capture_l0(source, config, evaluate)

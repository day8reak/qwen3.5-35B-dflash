#!/usr/bin/env python3
"""Dump the first L0 GDN QKV matmul call of the original 310P W4 T64/valid19 prefill."""
import argparse
from contextlib import ExitStack
import hashlib
import inspect
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))

from qwen35_moe_dflash.target.layer_probe.common import new_output, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.chain_reference import VALID19_IDS_SHA256, run_segment_prefill


def dump_call(root, report, npu, original, args, kwargs):
    import torch
    from qwen35_moe_dflash.target.layer_probe.qlinear import describe
    from qwen35_moe_dflash.target.layer_probe.qlinear_backend import tensor_formats
    from qwen35_moe_dflash.target.prefill_t64.segment_prefix import tensor_record

    names = ('x1', 'x2', 'scale')
    if len(args) > len(names):
        schema = torch.ops.npu.npu_quant_matmul.default._schema
        names = [arg.name for arg in schema.arguments]
    positional = list(names[:len(args)])
    ports = dict(zip(positional, args, strict=True))
    ports.update(kwargs)
    for name in ('x1', 'x2', 'scale', 'pertoken_scale'):
        if not torch.is_tensor(ports.get(name)):
            raise ValueError(f'missing tensor operand {name} on the live matmul')

    def save(name, value):
        spec = tensor_record(root, name, value)
        spec.update(device=str(value.device), source_stride=list(value.stride()),
                    npu_format=tensor_formats(npu, {name: value})[name])
        return spec

    report.update(positional=positional, keyword=list(kwargs), options={}, inputs={})
    for name, value in ports.items():
        if torch.is_tensor(value):
            report['inputs'][name] = save(name, value)
        else:
            report['options'][name] = describe(value)
    report['status'] = 'INPUTS_SAVED'
    save_json(root / 'manifest.json', report)
    result = original(*args, **kwargs)
    report['output'] = dict(save('golden_310p', result), role='matmul_return', port='y')
    return result


def capture(source, config, output):
    import torch
    from qwen35_moe_dflash.target.layer_probe.capture import bound, cpu
    from qwen35_moe_dflash.target.prefill_t64.segment_prefix import tensor_record

    entry = (source / 'inference_segment.py').resolve(strict=True)
    config = (source / config).resolve(strict=True)
    root = new_output(output)
    report = dict(status='CAPTURING', module='layers.0.linear_attn.in_proj_qkv', call_index=0,
                  operator='torch_npu.npu_quant_matmul', physical_rows=64, valid_rows=19,
                  driver=str(entry), driver_sha256=sha(entry), config=str(config), config_sha256=sha(config),
                  torch_version=str(torch.__version__), byte_order=sys.byteorder,
                  serialization='Contiguous tensor elements in logical index order, original dtype')
    manifest = root / 'manifest.json'
    save_json(manifest, report)

    class Captured(Exception):
        pass

    with ExitStack() as cleanup:
        def arm(frame):
            import torch_npu

            wrapper = frame.f_locals['model_wrapper']
            model = wrapper.model.model
            if model.start_layer_idx != 0 or model.layers[0].block_type != 'linear_attention':
                raise ValueError('expected segment starting at L0 GDN')
            projection = model.layers[0].linear_attn.in_proj_qkv
            impl = Path(inspect.getfile(type(projection))).resolve()
            report.update(qlinear_class=type(projection).__qualname__, qlinear_source=str(impl),
                          qlinear_source_sha256=sha(impl), torch_npu_version=str(torch_npu.__version__))

            def check_prompt(mod, args, kwargs):
                values = bound(mod, args, kwargs)
                ids = cpu(values['input_ids'])
                digest = hashlib.sha256(ids.numpy().astype('<i4').tobytes()).hexdigest()
                if (ids.shape != (1, 64) or digest != VALID19_IDS_SHA256 or list(values['allQLen']) != [19]
                        or cpu(values['effective_length']).tolist() != [19]):
                    raise ValueError('original driver did not select the measured T64/valid19 prompt')
                report['input_ids_sha256'] = digest

            def enter_projection(mod, args, kwargs):
                x = next(iter(bound(mod, args, kwargs).values()))
                if x.device.type != 'npu' or 'input_ids_sha256' not in report:
                    raise ValueError('expected the first QKV forward on NPU for the measured prompt')
                report['device'] = str(x.device)
                report['device_name'] = torch.npu.get_device_name(x.device)
                report['projection_input'] = tensor_record(root, 'projection_input', x)
                original = torch_npu.npu_quant_matmul

                def matmul(*args, **kwargs):
                    if 'inputs' in report:
                        raise ValueError('more than one matmul in the first QKV projection')
                    return dump_call(root, report, torch_npu, original, args, kwargs)

                cleanup.enter_context(patch.object(torch_npu, 'npu_quant_matmul', matmul))

            def leave_projection(mod, args, result):
                if 'output' not in report:
                    raise RuntimeError('the QKV forward returned without an observed npu_quant_matmul')
                report['projection_output_shape'] = list(result.shape)
                raise Captured()

            cleanup.callback(wrapper.register_forward_pre_hook(check_prompt, with_kwargs=True).remove)
            cleanup.callback(projection.register_forward_pre_hook(enter_projection, with_kwargs=True).remove)
            cleanup.callback(projection.register_forward_hook(leave_projection).remove)

        try:
            run_segment_prefill(entry, config, arm)
            raise RuntimeError('original segmented driver did not reach L0 QKV')
        except Captured:
            report['status'] = 'CAPTURED'
        except Exception as exc:
            report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save_json(manifest, report)
    for spec in (report['projection_input'], *report['inputs'].values(), report['output']):
        print(f"{spec['path']:24} {spec['dtype']:8} {spec['shape']}  {spec['bytes']} bytes")
    print(f'CAPTURED {manifest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='310P NZ directory with inference_segment.py')
    parser.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    parser.add_argument('--output', type=Path, required=True, help='fresh directory under AI_RUN_DIR')
    args = parser.parse_args()
    capture(args.source_root, args.config, args.output)


if __name__ == '__main__':
    main()

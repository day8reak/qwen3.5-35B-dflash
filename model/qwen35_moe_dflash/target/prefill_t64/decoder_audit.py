"""Compare T64 GDN, MoE and decoder residuals against one historical 310P capture."""
import json
from pathlib import Path

import torch

from .. import prefill_fixture as pf
from ..layer_probe.common import KIND, metrics, read_array, sha
from ..layer_probe.whole_gdn import assemble
from .audit_gdn import exactly_one
from .cpu_reference import CpuPrefill, norm, verify_floating_weights
from .reference import source_config


def tensor(root, row, direction, name):
    spec = exactly_one((s for s in row[direction] if s['name'] == name), f"{row['id']}/{name}")
    return torch.from_numpy(read_array(root, spec))


def match(label, a, b):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
        raise ValueError('Captured decoder boundaries disagree at ' + label)


def scores(native, cpu):
    if native.ndim != 3 or native.shape[:2] != (1, 64) or cpu.shape != native.shape:
        raise ValueError('Expected matching B1 T64 hidden tensors')
    a, b = native.detach().numpy(), cpu.detach().numpy()
    result = {}
    for name, start in (('all', 0), ('last_four', 60)):
        row = metrics(a[:, start:], b[:, start:])
        result[name] = {k: row[k] for k in ('passed', 'relative_l2', 'cosine', 'failed_metrics')}
    return result


@torch.inference_mode()
def replay_layer(cpu, layer, index, captured, initial, prefix_input):
    """Branch controls take recorded inputs. Only the prefix consumes previous CPU decoder outputs."""
    entry, pre, gdn, after_gdn, after_norm, moe, output = captured
    base = f'layers.{index}.'
    mixer = base + 'linear_attn'
    mlp = base + 'mlp'
    local_gdn = cpu.gdn(pre, layer.linear_attn, mixer, initial)[0]
    local_moe = cpu.moe(after_norm, layer.mlp, mlp)

    prefix_pre = norm(prefix_input, layer.input_layernorm)
    prefix_gdn = cpu.gdn(prefix_pre, layer.linear_attn, mixer, initial)[0]
    prefix_residual = prefix_input + prefix_gdn
    prefix_moe = cpu.moe(norm(prefix_residual, layer.post_attention_layernorm), layer.mlp, mlp)
    prefix_out = prefix_residual + prefix_moe

    checks = {
        'prefix_entry': scores(entry, prefix_input),
        'fixed_input_pre_norm': scores(pre, norm(entry, layer.input_layernorm)),
        'fixed_input_gdn': scores(gdn, local_gdn),
        'fixed_input_post_norm': scores(after_norm, norm(after_gdn, layer.post_attention_layernorm)),
        'fixed_input_moe': scores(moe, local_moe),
        'hidden_replace_gdn_only': scores(output, (entry + local_gdn) + moe),
        'hidden_replace_moe_only': scores(output, after_gdn + local_moe),
        'prefix_after_gdn_residual': scores(after_gdn, prefix_residual),
        'prefix_decoder_out': scores(output, prefix_out),
        'captured_gdn_residual_readd': scores(after_gdn, entry + gdn),
        'captured_moe_residual_readd': scores(output, after_gdn + moe),
    }
    return checks, prefix_out


def audit(fixture, source_root, config):
    fixture = Path(fixture).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    baseline = json.loads((fixture / 'manifest.json').read_text())
    if (baseline.get('status') != 'CAPTURED' or baseline['sequence_length'] != 64
            or baseline['layers'] != [0, 1, 2, 3] or baseline['boundary_capture'].get('status') != 'CAPTURED'):
        raise ValueError('Use the completed four-layer T64 Prefill fixture with boundary capture')
    root = fixture / 'capture'
    manifest = root / 'manifest.json'
    doc = json.loads(manifest.read_text())
    if (fixture / baseline['boundary_capture']['path'] != manifest or doc.get('artifact_kind') != KIND
            or doc.get('status') != 'CAPTURED' or doc.get('entrypoint') != 'DIRECT_ORIGINAL_SINGLE_PREFILL'
            or doc.get('reference') != 'ORIGINAL_310P_INFERENCE'):
        raise ValueError('Expected original 310P single Prefill capture')
    cfg = source_config(source_root, (source_root / config).resolve(strict=True), baseline)
    source = pf.import_source(source_root)
    with pf.hf_config_prefix(cfg['model_file'], 4, {}), pf.working_directory(source_root):
        wrapper = pf.build_four_layers(source, cfg['model_file'], {})
    verify_floating_weights(wrapper.model, cfg['model_file'])
    cpu = CpuPrefill(wrapper, cfg['quanted_pth'])
    args = baseline['source_arguments']
    embeds, scale = [torch.from_numpy(read_array(fixture, args[name])) for name in ('inputs_embeds', 'embed_scale')]
    prefix_input = (embeds * scale).to(torch.float16)
    previous = None
    gdns = assemble(doc)
    result = {'comparison': '310P_CAPTURE_VS_CPU', 'capture_sha256': sha(manifest), 'layers': {}}
    for i in range(3):
        def region(point):
            return exactly_one((r for r in doc['regions'] if r['point'] == point and r['layer'] == i
                                and r['call'] == 0 and r['phase'] == 'prefill' and r['sequence_length'] == 64), point)

        row = exactly_one((r for r in gdns if r['layer'] == i and r['call'] == 0 and r['sequence_length'] == 64), 'GDN')
        attn, mlp = region('residual_attention'), region('residual_mlp')
        entry = tensor(root, attn, 'inputs', 'residual')
        pre = tensor(root, row, 'inputs', 'hidden_states')
        gdn = tensor(root, attn, 'inputs', 'branch')
        after_gdn = tensor(root, attn, 'outputs', 'golden_hidden')
        post_norm = region('post_attention_layernorm')
        after_norm = tensor(root, post_norm, 'outputs', 'golden_y')
        moe = tensor(root, mlp, 'inputs', 'branch')
        output = tensor(root, mlp, 'outputs', 'golden_hidden')

        if previous is not None:
            match('previous hidden to layer input', previous, entry)
        match('layer input to input norm', entry, tensor(root, region('input_layernorm'), 'inputs', 'x'))
        match('input norm to GDN', pre, tensor(root, region('input_layernorm'), 'outputs', 'golden_y'))
        match('GDN to first residual', gdn, tensor(root, row, 'outputs', 'golden_hidden'))
        match('first residual to post norm', after_gdn, tensor(root, post_norm, 'inputs', 'x'))
        match('first residual to second residual', after_gdn, tensor(root, mlp, 'inputs', 'residual'))
        initial = tuple(tensor(root, row, 'inputs', name) for name in ('old_conv_state', 'initial_state'))
        if any(v.dtype != torch.float16 or torch.count_nonzero(v) for v in initial):
            raise ValueError('Expected fresh FP16 GDN cache inputs')
        print(f'[target-prefill] CPU replay of L{i} GDN and MoE', flush=True)
        result['layers'][str(i)], prefix_input = replay_layer(cpu, cpu.model.model.layers[i], i,
            (entry, pre, gdn, after_gdn, after_norm, moe, output), initial, prefix_input)
        previous = output
    return result

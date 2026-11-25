"""Independent CPU W4 trajectory for the split S64 prefill, using the measured prompt and layer entry."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

from ...common.artifacts import output_path, share_for_transfer
from .. import prefill_fixture as pf
from ..layer_probe.common import checked, read_array, save_json, sha
from .chain_reference import VALID19_IDS_SHA256
from .cpu_reference import CpuPrefill, norm, verify_floating_weights
from .l0_split import CPU_BASIS, contracts
from . import last4
from .segment_prefix import tensor_record


def decoder(cpu, block, i, hidden, controls, with_head):
    names = (('key_cache', 'value_cache') if i % 4 == 3 else ('conv_state', 'recurrent_state'))
    roles = [f'{name}_{i}' for name in names]
    before = [torch.zeros(contracts(i, False)[role][0], dtype=torch.float16) for role in roles]
    x = norm(hidden, block.input_layernorm)
    prefix = f'layers.{i}.'
    if i % 4 == 3:
        attn = block.self_attn
        if not torch.equal(attn.block_table, torch.arange(64).reshape(1, 64)):
            raise ValueError(f'L{i}: expected the sequential fresh-prefill page table')
        mix, states = cpu.attention(x, attn, prefix + 'self_attn', before, controls, valid_length=19)
    else:
        # QKV and conv history see T64. Only 19 rows advance the recurrence.
        mix, states = cpu.gdn(x, block.linear_attn, prefix + 'linear_attn', before, valid_length=19)
    physical_mix = torch.zeros_like(hidden)
    physical_mix[:, :19] = mix
    residual = hidden + physical_mix
    final = residual + cpu.moe(norm(residual, block.post_attention_layernorm), block.mlp, prefix + 'mlp')
    outputs = dict(zip(roles, states, strict=True), out_proj=physical_mix, decoder_hidden=final)
    if with_head:
        last = norm(final[:, 18:19], cpu.model.model.norm)
        outputs['logits'] = cpu.project(last, 'lm_head', cpu.model.lm_head).float()
    return outputs


@torch.inference_mode()
def prepare(run, mode, capture, source, *, per_layer_air=True):
    m = json.loads((capture/'manifest.json').read_bytes())
    if (m['status'], m['physical_rows'], m['valid_rows'], m['output_pos']) != ('CAPTURED', 64, 19, 18):
        raise ValueError('Expected a complete T64/valid19 310P reference')
    cfg = m.get('resources', m)
    checkpoint = Path(cfg['model_file'])
    inputs = m['inputs']
    ids = read_array(capture, inputs['input_ids'])
    if hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest() != VALID19_IDS_SHA256:
        raise ValueError('Unexpected valid19 prompt')
    def tensor(row):
        return torch.from_numpy(read_array(capture, row))
    controls = {key: tensor(inputs[key]) for key in ('position_ids', 'new_kv_cache_pos', 'attention_mask')}
    layers = [0] if mode == 'linear' else ([3] if mode == 'full' else list(range(40)))
    entry = m['boundaries']['2'] if mode == 'full' else inputs['embedding']
    hidden = tensor(entry)
    if hidden.dtype != torch.float16 or tuple(hidden.shape) != (1, 64, 2048):
        raise ValueError('Expected a measured physical FP16 layer entry')
    capture_hash = sha(capture/'manifest.json')
    tail = mode == 'last4'
    if tail:
        # An existing 40-layer CPU reference already contains this exact decoder exit.
        # Otherwise compute the CPU prefix without requiring 36 AIR exports.
        saved = run/'L35/cpu'
        if (saved/'manifest.json').is_file():
            prior = json.loads((saved/'manifest.json').read_bytes())
            if (prior['basis'], prior['mode'], prior['layer'], prior['capture_sha256']) != (
                    CPU_BASIS, '40', 35, capture_hash):
                raise ValueError(f'{saved}: unrelated CPU L35 reference')
            hidden = torch.from_numpy(read_array(saved, prior['outputs']['decoder_hidden']))
            layers = last4.LAYERS
        elif (run/last4.BUNDLE/'manifest.json').exists():
            saved = run/last4.BUNDLE
            prior = json.loads((saved/'manifest.json').read_bytes())
            if (prior['cpu_basis'], prior['capture_sha256'], prior['source_layer']) != (CPU_BASIS, capture_hash, 35):
                raise ValueError(f'{saved}: unrelated CPU L35 entry')
            hidden = torch.from_numpy(read_array(saved, prior['inputs']['cpu'])).half()
            layers = last4.LAYERS
        if layers == last4.LAYERS:
            last4.publish(run, capture, m, hidden.numpy())
        mode = '40'
    owner = None
    # At most four floating decoder layers are resident. W4 projections use
    # original checkpoint weights and CPU math, including the vocabulary head.
    for offset in range(0, len(layers), 4):
        group = layers[offset:offset + 4]
        cpu = None
        for local, i in enumerate(group):
            if mode == '40' and i == m['split_layer']:
                hidden[:, 19:] = 0
            case = run if mode == 'linear' else run/f'L{i:02d}'
            head = i == layers[-1]
            prefix_only = tail and i < 36
            if not per_layer_air:
                case.mkdir(parents=True, exist_ok=True)
            elif not prefix_only:
                bundle = json.loads((case/'air/manifest.json').read_bytes())
                if (bundle['reference']['capture_sha256'], bundle['layer'], bundle['head'], bundle['standalone']) != (
                        capture_hash, i, head, mode != '40'):
                    raise ValueError(f'L{i}: AIR and CPU input boundaries differ')
            dest = case/'cpu'
            stamp = dict(basis=CPU_BASIS, capture_sha256=capture_hash, mode=mode, layer=i,
                         layer_entry_sha256=hashlib.sha256(hidden.contiguous().numpy().tobytes()).hexdigest())
            if dest.is_dir() and not prefix_only:
                old = json.loads((dest/'manifest.json').read_bytes())
                if (any(old.get(k) != v for k, v in stamp.items()) or set(old['outputs']) != set(contracts(i, head))):
                    raise ValueError(f'{dest}: CPU reference belongs to a different trajectory')
                for row in old['outputs'].values():
                    checked(dest, row)
                hidden = torch.from_numpy(read_array(dest, old['outputs']['decoder_hidden']))
                print(f'L{i} CPU reference reused: {dest}', flush=True)
                continue
            if cpu is None:
                if owner is None:
                    if 'sources' in m:
                        for row in m['sources']:
                            if sha(row['path']) != row['sha256']:
                                raise ValueError(f"Capture source changed: {row['path']}")
                        source = Path(m['sources'][0]['path']).parent
                    elif source is None or sha(source/'inference_segment.py') != m['source_sha256']:
                        raise ValueError('Set TARGET_SOURCE to the 310P source of this 40-layer capture')
                    source = source.resolve()
                    driver = pf.import_source(source)
                    owner = sys.modules[driver.Qwen3_5MoeForCausalLMWrapper.__module__]
                with pf.working_directory(source), pf.hf_config_prefix(checkpoint, 4, {}):
                    model = owner.load_npu_model(str(checkpoint), 'cpu', torch.float16, variant='nz',
                                                 layer_num=len(group), start_layer_idx=group[0], quant_enable=True)
                model.eval().to(dtype=torch.float16)
                if (model.model.start_layer_idx != group[0] or [b.block_type for b in model.model.layers] != [
                        'full_attention' if j % 4 == 3 else 'linear_attention' for j in group]):
                    raise ValueError(f'CPU loader selected the wrong decoder layers for {group}')
                verify_floating_weights(model, checkpoint, start_layer=group[0])
                cpu = CpuPrefill(SimpleNamespace(model=model), cfg['quanted_pth'])
                del model
            print(f'L{i} CPU W4 reference', flush=True)
            outputs = decoder(cpu, cpu.model.model.layers[local], i, hidden, controls, head)
            if prefix_only:
                hidden = outputs['decoder_hidden']
                if i == 35:
                    last4.publish(run, capture, m, hidden.numpy())
                continue
            with tempfile.TemporaryDirectory(prefix='cpu-', dir=case) as tmp:
                staged = Path(tmp)/'cpu'
                staged.mkdir()
                records = {}
                for role, (shape, _) in contracts(i, head).items():
                    value = outputs[role]
                    if list(value.shape) != shape or not torch.isfinite(value).all():
                        raise ValueError(f'L{i}: invalid CPU {role}')
                    records[role] = tensor_record(staged, role, value)
                save_json(staged/'manifest.json', dict(**stamp, outputs=records,
                          padding='zero attention contribution outside valid19; physical residual and MoE propagated'))
                share_for_transfer(staged)
                staged.rename(dest)
            hidden = outputs['decoder_hidden']


def select_capture(run, mode, capture):
    first = run if mode == 'linear' else run / ('L03' if mode == 'full' else ('L36' if mode == 'last4' else 'L00'))
    air_manifest = first/'air/manifest.json'
    expected = json.loads(air_manifest.read_bytes())['reference']['capture_sha256']
    if capture is not None:
        if sha(capture/'manifest.json') != expected:
            raise ValueError(f'{capture}: capture SHA-256 differs from {air_manifest}; expected {expected}')
        return capture
    # Existing exports can use a capture from an earlier run. Match its bytes,
    # since two captures with the same prompt can have different model inputs.
    pattern = 'capture-*/310p/manifest.json'
    for paths in (run.glob(pattern), run.parent.glob('*/' + pattern)):
        for path in sorted(paths):
            if path.is_file() and sha(path) == expected:
                return path.parent
    raise FileNotFoundError(f'No capture matching {expected} under {run.parent}. '
                            'Use --capture DIR with the original 310P capture on the export host.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('linear', 'full', '40', 'last4'), required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--capture', type=Path, help='original 310P capture (default: match the AIR manifest SHA-256)')
    p.add_argument('--source', type=Path)
    a = p.parse_args()
    run = output_path(a.run)
    capture = select_capture(run, a.mode, a.capture)
    print(f'Using retained 310P capture: {capture}', flush=True)
    with (run/'.split-cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(run, a.mode, capture, a.source)


if __name__ == '__main__':
    main()

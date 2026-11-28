"""Independent 40-block CPU W4 golden for the exact inputs of a completed CPro T64 replay."""
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

from decoder_golden_cache import reference_inputs
from decoder_valid64_inputs import FILES, complete as complete_inputs
from decoder_prefill40_valid64 import read
from qwen35_moe_dflash.target import prefill_fixture as pf
from qwen35_moe_dflash.target.layer_probe.common import checked, save_json
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import state_roles
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import output_shape
from qwen35_moe_dflash.target.prefill_t64.segment_prefix import tensor_record
from qwen35_moe_dflash.target.prefill_t64.split_cpu_reference import decoder, load_cpu_group


def replay_inputs(plan_path, folder):
    if plan_path.name != 'inputs.json':
        raise ValueError('Select inputs.json from the successful 40-layer valid64 attempt')
    folder.mkdir()
    (folder/'replay').mkdir()
    for name in FILES:
        shutil.copyfile(plan_path.parent/name, folder/name)
    plan = read(folder/'inputs.json')
    result = read(folder/'replay/result.json')
    if not complete_inputs(folder):
        raise ValueError(f'{plan_path}: inconsistent 40-layer valid64 execution records')
    ids = np.fromfile(folder/'input_ids.bin', '<i4')
    if (ids.size != 64 or not np.array_equal(ids, plan['token_ids']) or ids.min() < 0
            or ids.max() >= output_shape('logits')[-1]):
        raise ValueError('Saved input_ids differ from the 64 real token IDs in the replay plan')
    for name, dtype, value in [('effective_length', '<i2', 64), ('allQLen', '<i8', 64), ('output_pos', '<i4', 63)]:
        if np.fromfile(folder/(name+'.bin'), dtype).tolist() != [value]:
            raise ValueError(f'Unexpected valid64 control {name}')
    mask = np.fromfile(folder/'attention_mask.bin', '<f4').reshape(1, 1, 64, 2048)
    positions = torch.arange(64, dtype=torch.int32)
    allowed = np.arange(2048)[None, :] <= np.arange(64)[:, None]
    if not (np.all(mask[0, 0][allowed] == 0) and np.all(mask[0, 0][~allowed] < 0)):
        raise ValueError('Expected the replay mask to expose exactly the 64 causal query positions')
    controls = dict(position_ids=positions[None], new_kv_cache_pos=positions, valid_rows=64,
                    attention_mask=torch.from_numpy(mask))
    return plan, result['attempt'], torch.from_numpy(ids).long()[None], controls


@torch.inference_mode()
def compute(out, plan, attempt, ids, controls, source, checkpoint, weights):
    driver = pf.import_source(source)
    owner = sys.modules[driver.Qwen3_5MoeForCausalLMWrapper.__module__]
    with pf.hf_config_prefix(checkpoint, 4, {}):
        tokenizer = driver.AutoTokenizer.from_pretrained(str(checkpoint))
    records = {}

    def record(name, value, shape):
        if value.device.type != 'cpu' or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid CPU golden tensor {name}: {value.device} {value.shape}')
        records[name] = tensor_record(out, name, value)

    for start in range(0, 40, 4):
        group = range(start, start + 4)
        cpu = load_cpu_group(owner, source, checkpoint, weights, group)
        if start == 0:
            hidden = cpu.model.model.embed_tokens(ids)
            if tuple(hidden.shape) != (1, 64, 2048) or hidden.dtype != torch.float16:
                raise ValueError('Expected checkpoint lookup to return FP16[1,64,2048]')
        for i in group:
            print(f'[CPU golden] L{i:02d} / L39', flush=True)
            qkv_stages = {} if i == 0 else None
            values = decoder(cpu, cpu.model.model.layers[i-start], i, hidden, controls, i == 39,
                             gdn_stages=qkv_stages)
            hidden = values['decoder_hidden']
            record(f'hidden_{i}', hidden, (1, 64, 2048))
            for role in state_roles(i):
                record(role, values[role], output_shape(role))
            if i == 0:
                qkv = qkv_stages['in_proj_qkv']
                record('mixed_qkv_0', qkv, (1, 64, 8192))
                tail = qkv[:, 60:64].transpose(1, 2)
                matched = int((tail == values['conv_state_0']).sum())
                tail_check = dict(exact_elements=matched, total_elements=tail.numel())
                if matched != tail.numel():
                    raise ValueError(f'CPU conv_state_0 differs from the last four physical QKV rows: {tail_check}')
            if i == 39:
                logits = values['logits']
                record('logits', logits, output_shape('logits'))
        del cpu

    top1_id = int(logits.argmax())
    top1 = dict(id=top1_id, text=tokenizer.decode([top1_id], skip_special_tokens=False),
                logit=float(logits.reshape(-1)[top1_id]))
    save_json(out/'manifest.json', dict(status='COMPLETE', backend='cpu', mode='40', phase='prefill',
              basis='CPU_W4_PREFILL_T64_VALID64_CHECKPOINT_LOOKUP', source_attempt=attempt,
              input_plan='inputs/inputs.json', model_file=str(checkpoint), quanted_pth=str(weights),
              token_ids=plan['token_ids'], physical_rows=64, valid_rows=64, output_pos=63, layers=list(range(40)),
              l0_state_vs_qkv_tail=tail_check,
              outputs=records, top1=top1))


def checked_golden(folder, token_ids, checkpoint, weights):
    meta = read(folder/'manifest.json')
    if ((meta['backend'], meta['mode'], meta['phase'], meta['status']) != ('cpu', '40', 'prefill', 'COMPLETE')
            or meta['basis'] != 'CPU_W4_PREFILL_T64_VALID64_CHECKPOINT_LOOKUP'
            or (meta['physical_rows'], meta['valid_rows'], meta['output_pos']) != (64, 64, 63)
            or meta['layers'] != list(range(40)) or meta['token_ids'] != token_ids
            or (meta['model_file'], meta['quanted_pth']) != (str(checkpoint), str(weights))
            or meta['l0_state_vs_qkv_tail'] != dict(exact_elements=32768, total_elements=32768)):
        raise ValueError(f'{folder}: incomplete or unrelated CPU valid64 golden')
    shapes = dict(mixed_qkv_0=(1, 64, 8192), logits=output_shape('logits'))
    for i in range(40):
        shapes[f'hidden_{i}'] = (1, 64, 2048)
        shapes.update({role: output_shape(role) for role in state_roles(i)})
    if meta['outputs'].keys() != shapes.keys():
        raise ValueError(f'{folder}: missing or unexpected golden outputs')
    for name, shape in shapes.items():
        row = meta['outputs'][name]
        if (row['path'] != name+'.bin' or tuple(row['shape']) != shape or row['dtype'] not in ('float16', 'float32')
                or row['bytes'] != np.prod(shape) * np.dtype(row['dtype']).itemsize):
            raise ValueError(f'{folder}: invalid {name} descriptor')
        checked(folder, row)
    return meta


def generate(cache, case, plan_path, source, config, *, archive=False):
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.build-', dir=cache) as temp:
        staged = Path(temp)/'golden'
        staged.mkdir()
        plan, attempt, ids, controls = replay_inputs(plan_path, staged/'inputs')
        reference = reference_inputs(source, config, valid64_cpu=True)
        resources = reference['checkpoints']
        checkpoint, weights = [Path(resources[name]['directory']) for name in ('model_file', 'quanted_pth')]
        binding = dict(reference=reference, token_ids=plan['token_ids'], valid_rows=64, output_pos=63,
                       mask_sha256=hashlib.sha256(controls['attention_mask'].numpy().tobytes()).hexdigest(),
                       cpu_driver_sha256=hashlib.sha256(inspect.getsource(compute).encode()).hexdigest())
        key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        saved = cache/key
        with (cache/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if archive:
                # The v1 manifest records checkpoint paths. Its file revisions are bound at import.
                old = plan_path.parent.parent
                if not (old/'manifest.json').is_file():
                    raise ValueError('Archive a completed CPU golden by selecting its local inputs/inputs.json')
                previous = checked_golden(old, plan['token_ids'], checkpoint, weights)
                if (previous['source_attempt'] != attempt or previous['input_plan'] != 'inputs/inputs.json'
                        or plan_path.resolve() != (old/'inputs/inputs.json').resolve()):
                    raise ValueError(f'{plan_path}: select the completed CPU attempt’s own inputs/inputs.json')
            valid = False
            if saved.is_dir():
                try:
                    if (read(saved/'binding.json') == binding and complete_inputs(saved/'inputs')
                            and read(saved/'inputs/inputs.json')['token_ids'] == plan['token_ids']
                            and all((saved/'inputs'/name).read_bytes() == (staged/'inputs'/name).read_bytes()
                                    for name in FILES if name.endswith('.bin'))):
                        meta = checked_golden(saved, plan['token_ids'], checkpoint, weights)
                        valid = True
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f'[CPU golden] Cached version needs repair: {exc}', flush=True)
            if valid:
                if archive and previous['outputs'] != meta['outputs']:
                    raise ValueError(f'{old}: archived outputs differ from the existing golden {saved}')
                status = 'REUSED'
            else:
                if archive:
                    meta = previous
                    for row in meta['outputs'].values():
                        shutil.copyfile(old/row['path'], staged/row['path'])
                    meta['archived_from'] = str(old)
                    meta['reference_binding_at'] = 'archive_time'
                    save_json(staged/'manifest.json', meta)
                    status = 'ARCHIVED'
                else:
                    print(f'[CPU golden] Generating {saved}', flush=True)
                    compute(staged, plan, attempt, ids, controls, source, checkpoint, weights)
                    meta = read(staged/'manifest.json')
                    status = 'COMPLETE'
                if reference_inputs(source, config, valid64_cpu=True) != reference:
                    raise ValueError('Reference configuration or checkpoints changed during the run')
                save_json(staged/'binding.json', binding)
                if saved.exists():
                    saved.rename(Path(tempfile.mkdtemp(prefix='incomplete-', dir=cache))/key)
                staged.rename(saved)
    if case != cache:
        # Separate experiments only need a reference to the persistent tensors.
        case.mkdir(parents=True, exist_ok=True)
        print(f'[CPU golden] Experiment {case} uses version {key}', flush=True)
        with tempfile.TemporaryDirectory(dir=case) as tmp:
            pointer = Path(tmp)/'cpu-valid64.json'
            save_json(pointer, dict(manifest=str(saved/'manifest.json'), source_attempt=attempt))
            pointer.replace(case/'cpu-valid64.json')
    print(f'[CPU golden] {status} {saved}/manifest.json | top1={meta["top1"]}', flush=True)

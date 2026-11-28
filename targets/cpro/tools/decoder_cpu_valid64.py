"""Independent 40-block CPU W4 golden for the exact inputs of a completed CPro T64 replay."""
from contextlib import nullcontext
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
import yaml

from decoder_handoff import ssh_reader
from decoder_prefill40_valid64 import read
from qwen35_moe_dflash.target import prefill_fixture as pf
from qwen35_moe_dflash.target.layer_probe.common import save_json
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import state_roles
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import output_shape
from qwen35_moe_dflash.target.prefill_t64.segment_prefix import tensor_record
from qwen35_moe_dflash.target.prefill_t64.split_cpu_reference import decoder, load_cpu_group


def replay_inputs(plan_path, folder):
    if plan_path.name != 'inputs.json':
        raise ValueError('Select inputs.json from the successful 40-layer valid64 attempt')
    folder.mkdir()
    (folder/'replay').mkdir()
    reader = nullcontext(shutil.copyfile) if plan_path.is_file() else ssh_reader('omc')
    with reader as fetch:
        for name in ('inputs.json', 'replay/result.json', 'replay/io.json', 'replay/cpro-valid64.json',
                     'input_ids.bin', 'attention_mask.bin', 'effective_length.bin', 'allQLen.bin', 'output_pos.bin'):
            fetch(plan_path.parent/name, folder/name)
    plan = read(folder/'inputs.json')
    result = read(folder/'replay/result.json')
    run = read(folder/'replay/cpro-valid64.json')
    slots = read(folder/'replay/io.json')['inputs']
    if (result['status'] != 'EXECUTED' or result['exitcode'] != 0 or result['board_case'] != '40-prefill-t64'
            or (run['valid_rows'], run['output_pos'], plan['valid_rows'], plan['output_pos']) != (64, 63, 64, 63)
            or (run['attempt'], run['source_attempt'], run['model']['sha256']) !=
               (result['attempt'], plan['source_attempt'], result['omc_sha256'])
            or result['input_hashes'] != [row['sha256'] for row in slots]):
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
def generate(case, plan_path, source, config):
    cfg = yaml.safe_load(config.read_text())
    checkpoint = (source/cfg['model_file']).resolve(strict=True)
    weights = (source/cfg['quanted_pth']).resolve(strict=True)
    case.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='attempt-', dir=case))
    print(f'[CPU golden] {out} | input: {plan_path}', flush=True)
    plan, attempt, ids, controls = replay_inputs(plan_path, out/'inputs')
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
    print(f'[CPU golden] COMPLETE {out}/manifest.json | top1={top1}', flush=True)

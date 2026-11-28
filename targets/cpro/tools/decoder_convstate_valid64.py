"""Repeat the retained mountain prompt inside its chat template for a CPro-only T64 replay."""
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from decoder_cases_board import compiled, output_order, read, replay
from decoder_l0_convstate_dump import report_ports
from qwen35_moe_dflash.target.layer_probe.common import identity, read_array, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import output_shape


def repeat_positions(configured_prompt, tokens):
    phrase = '世界上最高的山'
    starts = [i+3 for i in range(len(tokens)-2) if tokens[i:i+3] == ['<|im_start|>', 'user', '\n']]
    if starts:
        if len(starts) != 1:
            raise ValueError(f'Cannot locate the user message in the retained token pieces: {tokens}')
        start, = starts
        stop = tokens.index('<|im_end|>', start)
        if '<|im_start|>' in tokens[start:stop]:
            raise ValueError('Unclosed user message in the retained chat')
    else:
        spans = [(i, j) for i in range(len(tokens)) for j in range(i+1, len(tokens)+1)
                 if ''.join(tokens[i:j]) == configured_prompt]
        if len(spans) != 1:
            raise ValueError(f'Cannot locate the configured prompt in the retained token pieces: {tokens}')
        start, stop = spans[0]
    source_text = ''.join(tokens[start:stop])
    # A saved vocabulary piece can contain U+FFFD for one byte of a multibyte character.
    if phrase not in source_text and not ('\ufffd' in source_text and phrase in configured_prompt):
        raise ValueError(f'Expected the mountain question in the retained user message, got {source_text!r}')
    choices = [(phrase, list(range(i, j))) for i in range(start, stop) for j in range(i+1, stop+1)
               if ''.join(tokens[i:j]) == phrase]
    choices.append((source_text, list(range(start, stop))))
    extra = 64 - len(tokens)
    for text, span in choices:
        copies, separators = divmod(extra, len(span))
        if copies < 1 or separators > copies or (separators and '\n' not in tokens):
            continue
        rows = list(range(start))
        for i in range(copies):
            rows.extend(span)
            if i < separators:
                rows.append(tokens.index('\n'))
        rows.extend(range(start, len(tokens)))
        if len(rows) == 64:
            return rows, source_text, (text+'\n')*separators + text*(copies-separators) + source_text
    raise ValueError('Cannot fill T64 with complete copies of the captured mountain text')


def prepare_inputs(folder, case, meta, fixture, bundle_stamp):
    ref = read(fixture/'reference.json')
    valid = ref['valid_rows']
    ids = read_array(fixture, ref['inputs']['input_ids'])
    if (ids.shape != (1, 64) or not 0 < valid < 64 or
            hashlib.sha256(ids.astype('<i4').tobytes()).hexdigest() != meta['fixture']['token_sha256']):
        raise ValueError('Captured token IDs or sequence length differ from the compiled input fixture')
    vocab = read(fixture/'tokens.json')
    positions, source_text, prompt = repeat_positions(ref['prompt'], [vocab[i] for i in ids[0, :valid]])
    specs = meta['input_files']

    def original(role):
        return read_array(case, specs[role])

    changes = {role: original(role)[:, positions] for role in ('inputs_embeds', 'embed_scale')}
    if 'input_ids' in specs:
        changes['input_ids'] = original('input_ids')[:, positions]
    mask = original('attention_mask')
    rows = np.arange(64)
    causal = np.arange(mask.shape[-1])[None, :] <= rows[:, None]
    blocked = mask[0, 0, 0, 1]
    if not blocked < 0 or not np.array_equal(mask[0, 0] == 0, (rows[:, None] < valid) & causal):
        raise ValueError('Unexpected source attention mask for a fresh causal prefill')
    # Retain the original mask convention (including any cache-capacity masking).
    # Copy the last measured query to new rows and open only the newly visible keys.
    full_mask = mask.copy()
    full_mask[:, :, valid:, :] = mask[:, :, valid-1:valid, :]
    for pos in range(valid, 64):
        full_mask[0, 0, pos, :pos+1] = 0
    if not np.array_equal(full_mask[0, 0] == 0, causal):
        raise ValueError('The generated mask does not expose exactly 64 causal query positions')
    changes['attention_mask'] = full_mask
    for role in ('position_ids', 'new_kv_cache_pos'):
        if role in specs and not np.array_equal(original(role).reshape(-1), rows):
            raise ValueError(f'{role} must enumerate T64 positions 0 through 63')
    controls = {}
    for role, old, new in [('effective_length', valid, 64), ('allQLen', valid, 64), ('output_pos', valid-1, 63),
                           ('conv_state_pos', valid, 64)]:
        if role not in specs and role != 'conv_state_pos':
            continue  # Only compiled inputs can be bound on the board.
        value = original(role)
        if value.flat[0] != old or np.any(value.reshape(-1)[1:]):
            raise ValueError(f'Unexpected {role} in the source fixture: {value}')
        value.flat[0] = new
        changes[role] = value
        controls[role] = int(value.flat[0])
    records = {}
    for role, value in changes.items():
        path = folder/(role+'.bin')
        value.tofile(path)
        records[role] = dict(identity(path, folder), dtype=value.dtype.name, shape=list(value.shape))
    plan = dict(valid_rows=64, physical_rows=64, source_valid_rows=valid, source_user_text=source_text,
                configured_prompt=ref['prompt'], prompt=prompt,
                token_ids=ids[0, positions].tolist(), source_positions=positions,
                source_bundle=bundle_stamp, controls=controls, inputs=records)
    save_json(folder/'inputs.json', plan)
    print(f'[L0 valid64] {prompt!r} | 64/64 real tokens | {controls}', flush=True)
    return plan


def full_replay(board, config, inspect=None, *, replace_inputs=False, replace_weights=False):
    case = board/'package/L00-L03'
    base = board/case.name
    stamp = sha(case.parent/'manifest.json')
    meta = read(case/'case.json')
    if meta['layers'] != [0, 1, 2, 3] or not meta.get('native_convstate'):
        raise ValueError(f'{case}: expected the four-layer native ConvState AIR')
    schema = {row['name']: dict(shape=output_shape(row['name'], 4)) for row in meta['outputs']}
    root = base/'operator-dump-valid64'
    if inspect is None:
        build, model, weights = compiled(case, base, None, None, stamp)
        root.mkdir(exist_ok=True)
        attempt = Path(tempfile.mkdtemp(dir=root, prefix='attempt-'))
        print(f'[L0 valid64] {attempt} | --inspect-attempt {attempt.name}', flush=True)
        plan = prepare_inputs(attempt, case, meta, case.parent/'fixture', stamp)
        runtime = read(config.parent/read(config)['assets'])['environment']
        inputs = {role: (attempt, row) for role, row in plan['inputs'].items()}
        try:
            replay(attempt, case, build, model, weights, {}, schema, runtime, sha(attempt/'inputs.json'), False,
                   board_case='4-prefill-t64-dump', replace_inputs=replace_inputs, replace_weights=replace_weights,
                   shape_config=config, operator_dump=True, input_overrides=inputs)
        except RuntimeError as exc:
            if 'remote release-global-model-io-lock failed' not in str(exc):
                raise
            print(f'[L0 valid64] {exc}. Checking the retained receipt and dump.', flush=True)
    else:
        attempt = root/inspect
        build = read(base/'omc/build.json')
        plan = read(attempt/'inputs.json')
    folder = attempt/'replay'
    receipt = read(folder/'result.json')
    inputs = {**meta['input_files'], **plan['inputs']}
    hashes = [inputs[meta['inputs'][p['name']]]['sha256'] for p in build['inputs']]
    if (receipt['status'] != 'EXECUTED' or receipt['exitcode'] != 0 or receipt['case'] != case.name or
            receipt['board_case'] != '4-prefill-t64-dump' or receipt['input_hashes'] != hashes or
            receipt['fixture_sha256'] != sha(attempt/'inputs.json') or plan['valid_rows'] != 64 or
            build['stamp']['bundle'] != stamp or plan['source_bundle'] != stamp or
            receipt['omc_sha256'] != build['model']['sha256'] or
            receipt['operator_dump']['status'] != 'FILES_RECEIVED_REVIEW_REQUIRED'):
        raise ValueError(f'{folder}: missing or mismatched valid64 dump receipt')
    names = output_order(build['outputs'], meta['outputs'], schema)
    if len(receipt['outputs']) != len(names):
        raise ValueError(f'{folder}: incomplete public outputs')
    slot = names.index('conv_state_0')
    public = dict(receipt['outputs'][slot], shape=schema['conv_state_0']['shape'], dtype=build['outputs'][slot]['dtype'])
    overrides = {role: (attempt, row) for role, row in plan['inputs'].items()}
    return report_ports(case, folder, None, plan, (folder, public), input_overrides=overrides)

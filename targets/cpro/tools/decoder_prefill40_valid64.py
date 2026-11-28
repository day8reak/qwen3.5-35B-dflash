"""Replay an existing L0–L39 physical-tail OMC with the token IDs from a valid64 attempt."""
import json
from pathlib import Path
import tempfile


def read(path):
    return json.loads(path.read_bytes())


def physical_tail(meta):
    boundary = meta.get('boundary', {})
    return (boundary.get('convstate_update_nodes') == 0 and boundary.get('physical_conv_caches') == 30
            and meta.get('fixture', {}).get('embedding') == 'lookup')


def select_case(parent, slug, *, hidden_trace=False):
    candidates = []
    for case in [parent/slug, *parent.glob('*/'+slug)]:
        site = case/'board'
        compiled, exported = site/'omc/manifest.json', site/'air/manifest.json'
        if compiled.is_file() and exported.is_file():
            meta = read(exported)
            if physical_tail(meta) and (not hidden_trace or meta.get('hidden_trace')):
                prior = site/'replay/result.json'
                executed = prior.is_file() and read(prior).get('status') == 'EXECUTED'
                candidates.append((case == parent/slug, bool(meta.get('hidden_trace')), executed,
                                   compiled.stat().st_mtime_ns, case))
    if not candidates:
        raise FileNotFoundError(f'No compiled 40-layer physical-tail OMC under {parent}. '
                                'Select its 40-prefill-t64 case with --run.')
    return max(candidates)[-1]


def execute(site, source, config, args):
    import numpy as np
    from decoder4_original import FULL_KIND, TAPS_FULL_KIND, board
    from qwen35_moe_dflash.target.layer_probe.common import read_array, read_tensor, save_json, sha, sized_file
    from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import output_roles

    bundle = site/'air'
    meta = read(bundle/'manifest.json')
    compiled = read(site/'omc/manifest.json')
    if (meta['kind'] not in (FULL_KIND, TAPS_FULL_KIND) or not physical_tail(meta) or
            compiled['air_manifest_sha256'] != sha(bundle/'manifest.json')):
        raise ValueError(f'{site}: expected a compiled 40-layer token-ID OMC with 30 explicit ConvState slices')
    if args.hidden_trace and not meta.get('hidden_trace'):
        raise ValueError(f'{site}: selected OMC does not expose hidden_layers')
    plan = read(source)
    prior = None
    if 'source' in plan:  # Reuse the exact model and inputs of an earlier 40-layer replay.
        prior = read(source.parent/'replay/result.json')
        if (prior['status'] != 'EXECUTED' or prior['exitcode'] != 0
                or prior['omc_sha256'] != compiled['primary']['sha256']):
            raise ValueError(f'{source}: the successful 40-layer replay must use this OMC')
        source = Path(plan['source'])
        original = read(source)
        if plan['token_ids'] != original['token_ids']:
            raise ValueError(f'{source}: token IDs differ from the previous 40-layer input')
        plan = original
    src_receipt = read(source.parent/'replay/result.json')
    if (src_receipt.get('status') != 'EXECUTED' or src_receipt.get('exitcode') != 0 or
            src_receipt.get('fixture_sha256') != sha(source) or (plan['valid_rows'], plan['physical_rows']) != (64, 64)):
        raise ValueError(f'{source}: expected the saved input of a completed valid64 CPro replay')
    ids = np.asarray(plan['token_ids'])
    pieces = read(sized_file(bundle, meta['tokens']))['pieces']
    if ids.shape != (64,) or ids.dtype.kind not in 'iu' or ids.min() < 0 or ids.max() >= len(pieces):
        raise ValueError(f'{source}: invalid 64-token sequence for the 40-layer vocabulary')
    mask = read_array(source.parent, plan['inputs']['attention_mask'])
    causal = np.arange(2048) <= np.arange(64)[:, None]
    if (mask.shape != (1, 1, 64, 2048) or mask.dtype != np.float32 or
            not np.array_equal(mask[0, 0] == 0, causal) or not np.all(mask[0, 0][~causal] < 0)):
        raise ValueError(f'{source}: expected a full 64-query causal mask')
    parent = site/'valid64'
    parent.mkdir(exist_ok=True)
    if args.dump:
        from decoder_prefill40_convstate import dump_plan, inspect
        taps = dump_plan(bundle, meta)
    attempt = parent/args.inspect_attempt if args.inspect_attempt else Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
    arrays = dict(input_ids=ids.reshape(1, 64).astype('<i4'), attention_mask=mask, effective_length=np.array([64], '<i2'),
                  allQLen=np.array([64], '<i8'), output_pos=np.array([63], '<i4'))
    overrides = {}
    for role, value in arrays.items():
        path = attempt/(role+'.bin')
        if args.inspect_attempt:
            if not np.array_equal(np.fromfile(path, dtype=value.dtype), value.reshape(-1)):
                raise ValueError(f'{path}: retained input differs from the requested valid64 replay')
        else:
            value.tofile(path)
        overrides[path.name] = path
    positions = np.arange(64, dtype='<i4')
    for name in ('position_ids', 'new_kv_cache_pos'):
        if not np.array_equal(np.fromfile(bundle/'inputs'/(name+'.bin'), '<i4'), positions):
            raise ValueError(f'{name}: expected positions 0 through 63 in the retained prefill fixture')
    input_plan = dict(source=str(source), source_attempt=src_receipt['attempt'],
                      prompt=plan['prompt'], token_ids=ids.tolist(), valid_rows=64, output_pos=63)
    if args.inspect_attempt:
        if read(attempt/'inputs.json') != input_plan:
            raise ValueError(f'{attempt}: retained input plan differs from the requested source')
    else:
        save_json(attempt/'inputs.json', input_plan)
    print(f'[40-layer valid64] {site}/omc | native ConvState updates=0, physical ConvState caches=30 | '
          f'64 tokens, output_pos=63 | {attempt}', flush=True)
    folder = attempt/'replay'
    if not args.inspect_attempt:
        try:
            board(site, None, config, (bundle/'inputs').resolve(), None, kind=meta['kind'],
                  replace_inputs=args.replace_inputs, replace_weights=args.replace_weights,
                  original_board_args=args.original_board_args, input_overrides=overrides, replay_dir=folder,
                  dump_edges=tuple(taps) if args.dump else ())
        except RuntimeError as exc:
            if 'remote release-global-model-io-lock failed' not in str(exc):
                raise
            print(f'[40-layer valid64] {exc}. Checking the retained execution receipt.', flush=True)
    result, io = read(folder/'result.json'), read(folder/'io.json')
    bindings = {Path(row['file']).name: row for row in io['inputs']}
    if (result['status'] != 'EXECUTED' or result['exitcode'] != 0 or
            result['omc_sha256'] != compiled['primary']['sha256'] or
            result['input_hashes'] != [r['sha256'] for r in io['inputs']] or
            not {'input_ids.bin', 'output_pos.bin', 'attention_mask.bin'} <= bindings.keys() or
            any(bindings[name]['sha256'] != sha(overrides[name]) for name in overrides.keys() & bindings.keys())):
        raise ValueError(f'{folder}: missing successful execution with the requested 64-token inputs')
    if prior and result['input_hashes'] != prior['input_hashes']:
        raise ValueError(f'{folder}: board input slots differ from the previous 40-layer replay')
    roles = output_roles(io['outputs'], meta['boundary']['outputs'])
    outputs = {}
    for role, spec, row in zip(roles, io['outputs'], result['outputs'], strict=True):
        sized_file(folder, row)
        outputs[role] = dict(row, shape=spec['shape'], dtype=spec['dtype'])
    logits = read_tensor(sized_file(folder, outputs['logits']), outputs['logits']).reshape(-1)
    if logits.size != len(pieces) or not np.isfinite(logits).all():
        raise ValueError(f'{folder}: expected one finite logit per vocabulary entry at position 63')
    token = int(np.argmax(logits))
    prediction = dict(id=token, text=pieces[token], logit=float(logits[token]))
    save_json(folder/'cpro-valid64.json', dict(model=compiled['primary'], input_plan='../inputs.json',
              source_attempt=src_receipt['attempt'], attempt=result['attempt'], valid_rows=64, output_pos=63,
              top1=prediction, outputs=outputs))
    print(f'[40-layer valid64] top1={prediction!r} | {folder}/cpro-valid64.json', flush=True)
    if args.dump:
        return inspect(folder, result, outputs, taps, ids)
    return True

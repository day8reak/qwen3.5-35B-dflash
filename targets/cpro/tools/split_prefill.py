#!/usr/bin/env python3
"""Run independent SiluPatch OMCs inside the L3 decoder or a continuous L0–L39 prefill."""
import argparse
from functools import partial
import fcntl
from pathlib import Path
import tempfile

from l0_silu_omcs import compare, execute, load, select_board_defaults
from decoder4_original import L0_DIR, SPLIT_KIND, export, ssh_bundle
from decoder4_prefill import pull_air, ssh_file
from qwen35_moe_dflash.target.prefill_t64 import last4
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.layer_probe.common import checked, read_array, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.l0_split import contracts, plan


def check_bundle(meta, layer, layers):
    if (meta['kind'], meta.get('layer'), meta.get('head'), meta.get('standalone')) != (
            SPLIT_KIND, layer, layer == layers[-1], len(layers) == 1):
        raise ValueError(f'L{layer}: expected a matching decoder, readout and input boundary')


def check_capture(capture, layers):
    ref = load(capture/'manifest.json')
    if (ref['status'], ref['physical_rows'], ref['valid_rows'], ref['output_pos']) != ('CAPTURED', 64, 19, 18):
        raise ValueError(f'{capture}: expected a completed 310P T64/valid19 capture')
    outputs = ref['outputs']
    if len(layers) == 1:
        if not ('2' in ref.get('boundaries', {}) and '3' in ref['boundaries'] and
                'golden_hidden' in ref.get('attention_modules', {}).get('3', {})):
            raise ValueError(f'{capture}: full mode needs the measured L3 entry and four-block prefix readout')
        required = {'logits', 'key_cache_3', 'value_cache_3'}
    else:
        blocks = [(b['layer'], b['block_type']) for b in ref.get('blocks', []) if 'hidden' in b]
        if (ref.get('split_layer') not in range(1, 40) or
                blocks != [(i, 'full_attention' if i % 4 == 3 else 'linear_attention') for i in range(40)]):
            raise ValueError(f'{capture}: expected all 40 ordered decoder boundaries from the same prefill')
        required = {'logits'}
        for i in layers:
            required.add(f'out_proj_{i}')
            required.update(k for k in contracts(i, False) if k.endswith(f'_{i}'))
    if required - outputs.keys():
        raise ValueError(f'{capture}: missing 310P outputs {sorted(required - outputs.keys())}. '
                         'Run the air action without --capture to collect the required states.')


def previous_replay(run, mode, selector, *, completed=False):
    if selector == 'auto':
        paths = sorted(run.glob('replay-*/prefill.json'), key=lambda f: f.stat().st_mtime_ns, reverse=True)
    else:
        paths = [(run / selector) / 'prefill.json']
    for path in paths:
        report = load(path)
        if report['mode'] != mode:
            continue
        if completed and report['status'] not in ('EXECUTED', 'PASSED', 'FAILED_ACCURACY'):
            # Older runs could stop while fetching the last layer's CPU reference.
            if mode == 'last4':
                continue
            last = path.parent / ('L03' if mode == 'full' else 'L39') / 'chain.json'
            if report['status'] != 'INTERRUPTED' or not last.is_file() or load(last)['status'] != 'EXECUTED':
                continue
        return path.parent
    raise ValueError(f'No {mode} replay matching {selector} in {run}')


def compare_replay(run, replay, layers, remote_run=None):
    manifest = replay/'prefill.json'
    # Comparisons publish into this replay. Allow independent replays under the same run to be read.
    with (replay/'.compare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return compare_layers(run, manifest, layers, remote_run)


def compare_layers(run, manifest, layers, remote_run):
    replay = manifest.parent
    summary = load(manifest)
    finished = summary['status'] in ('EXECUTED', 'PASSED', 'FAILED_ACCURACY')
    if finished and [s['layer'] for s in summary['layers']] != layers:
        raise ValueError(f'Incomplete layer list in {replay}')
    scores = []
    for i in layers:
        key = f'L{i:02d}'
        case, dst = run/key, replay/key
        if not finished and not (dst/'chain.json').is_file():
            break
        receipt = load(dst/'chain.json')
        options = dict(remote_air=(remote_run or run)/key/'air')
        if summary['mode'] == 'last4':
            options['prepare_mode'] = 'last4'
        if not finished and receipt['status'] != 'EXECUTED':
            if receipt['status'] == 'FAILED':
                compare(case/'air', dst, allow_failed=True, **options)
            break
        passed = compare(case/'air', dst, **options)
        scores.append(dict(layer=i, passed=passed, hidden_sha256=receipt['outputs']['decoder_hidden']['sha256'],
                           report=f'{key}/comparison.json', report_sha256=sha(dst/'comparison.json')))
    if len(scores) != len(layers) or summary['status'] == 'RUNNING':
        print(f'Incomplete {summary["mode"]} replay: {replay}', flush=True)
        return False
    summary.update(status='PASSED' if all(s['passed'] for s in scores) else 'FAILED_ACCURACY', layers=scores)
    summary.pop('error', None)
    summary.pop('layer', None)
    save_json(manifest, summary)
    print(summary['status'], manifest, flush=True)
    return summary['status'] == 'PASSED'


def run_board(args, run, layers, *, out=None, previous=None, entry=None, entry_capture=None):
    if out is None:
        previous = previous_replay(run, args.mode, args.recovery) if args.recovery else None
        out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    else:
        out.mkdir()
    start_layer = start_stage = None
    if args.from_stage and (entry is None or previous is not None):
        key, sep, start_stage = args.from_stage.partition('/')
        if not sep and len(layers) == 1:
            key, start_stage = f'L{layers[0]:02d}', key
        if not previous or key not in [f'L{i:02d}' for i in layers] or not start_stage:
            raise ValueError('--from-stage requires --recovery and an Lxx/stage name')
        start_layer = int(key[1:])
    summary = dict(mode=args.mode, status='RUNNING', layers=[], recovered_from=str(previous) if previous else None)
    manifest = out/'prefill.json'
    save_json(manifest, summary)
    print('Prefill replay:', out, flush=True)
    templates = getattr(args, 'templates', False)
    conv_weights = getattr(args, 'conv_input_weights', None)
    per_layer_experts = getattr(args, 'per_layer_experts', False)
    if entry is None and previous:
        prior = load(previous/'prefill.json')
        if not getattr(args, 'per_layer_omc', False):
            templates = prior.get('layer_templates', False)
        conv_weights = conv_weights or prior.get('conv_input_weights')
        per_layer_experts = per_layer_experts or prior.get('per_layer_experts', False)
    summary['conv_input_weights'] = conv_weights
    summary['per_layer_experts'] = per_layer_experts
    summary['layer_templates'] = templates
    hidden = entry
    if entry is not None:
        summary['initial_hidden'] = dict(source=out.name, after_layer=35, sha256=sha(entry))
    capture = baseline = None
    try:
        for i in layers:
            key = f'L{i:02d}'
            case, dst = run/key, out/key
            bundle = case/'air'
            remote_air = (args.remote_run or run)/key/'air'
            if not bundle.exists():
                pull_air(case, remote_air, ssh_download=partial(ssh_bundle, kind=SPLIT_KIND))
            meta = load(bundle/'manifest.json')
            check_bundle(meta, i, layers)
            origin = meta['original_manifest_sha256']
            current = meta['reference']['capture_sha256']
            if entry_capture is not None and current != entry_capture:
                raise ValueError(f'{key}: L35 entry and decoder fixture are from different captures')
            if capture is not None and (capture, baseline) != (current, origin):
                raise ValueError(f'{key}: reference or original CPro baseline changed mid-chain')
            capture, baseline = current, origin
            for row in meta['files']:
                checked(bundle, row)
            raw = (bundle/L0_DIR/'export.air').read_bytes()
            checkpoint = previous/key if previous and (previous/key/'chain.json').is_file() else None
            split = getattr(args, 'split_decoder_projections', False)
            residual = getattr(args, 'split_decoder_residual', False)
            if checkpoint and (start_layer is None or i < start_layer):
                prior = load(checkpoint/'chain.json')
                if prior['status'] == 'EXECUTED':
                    split = prior.get('split_decoder_projections', False)
                    residual = prior.get('split_decoder_residual', False)
            options = {'split_projections': True} if split else {}
            if residual:
                options['split_residual'] = True
            if templates and i == 0:
                options['separate_embedding'] = True
            if templates and per_layer_experts:
                options['per_layer_experts'] = True
            graph = plan(raw, meta['silu_patch']['nodes'], layer=i, head=meta['head'], **options)
            if conv_weights and i % 4 != 3:
                conv, = [s for s in graph['stages'] if s['name'] == 'conv_input']
                conv.update(compiler='per_layer', weight_storage=conv_weights)
                if options.get('separate_embedding'):
                    # L0's cut after embedding has a different input ABI.
                    conv['build_name'] = 'conv_input-after-embedding'
                print(f'{key}/conv_input: compiling the original layer slice with {conv_weights} weights', flush=True)
            plan_file = 'plan-w4-projections.json' if split else 'plan.json'
            save_json(case/('plan-w4-residual.json' if residual else plan_file), graph)
            if start_layer is not None:
                if i < start_layer and not checkpoint:
                    raise ValueError(f'Missing {key} checkpoint before requested restart')
                if i == start_layer and not checkpoint and start_stage != graph['stages'][0]['name']:
                    raise ValueError(f'Missing {key} checkpoint for {start_stage}')
                if i < start_layer and load(checkpoint/'chain.json')['status'] != 'EXECUTED':
                    raise ValueError(f'{key} did not complete before requested restart')
                if i > start_layer:
                    checkpoint = None
            if args.mode in ('40', 'last4') and i == meta['reference']['split_layer']:
                # The 310P driver trims and re-pads once between its two decoder segments.
                # Apply the same operation to the preceding CPro output, keeping all valid rows.
                import numpy as np
                previous_hidden = hidden
                values = np.fromfile(previous_hidden, dtype='<f4').reshape(1, 64, 2048)
                values[:, 19:] = 0
                hidden = out/f'{key}-entry.bin'
                values.tofile(hidden)
                summary['segment_entry'] = dict(layer=i, source_sha256=sha(previous_hidden),
                                                input_sha256=sha(hidden), valid_rows=19)
            template = run/'templates'/('full' if i % 4 == 3 else 'linear')
            extra = dict(template_root=template) if templates else {}
            execute(case, bundle, meta, graph, raw, args.ddk.resolve(strict=True), args.diagnosis,
                    args.runtime_config.resolve(strict=True), recovery=checkpoint, entry=hidden,
                    from_stage=start_stage if i == start_layer and checkpoint else None, out=dst, shared_silu=run/'silu-omc', **extra)
            receipt = load(dst/'chain.json')
            hidden = checked(dst, receipt['outputs']['decoder_hidden'])
            summary['layers'].append(dict(layer=i, hidden_sha256=receipt['outputs']['decoder_hidden']['sha256']))
            save_json(manifest, summary)
        summary['status'] = 'EXECUTED'
    except BaseException as exc:
        summary.update(status='INTERRUPTED', layer=i, error=str(exc))
        raise
    finally:
        save_json(manifest, summary)
        print(summary['status'], manifest, flush=True)
    if entry is not None:
        return True
    return compare_replay(run, out, layers, args.remote_run)


def ssh_last4(client, remote, staged):
    staged.mkdir()
    ssh_file(client, Path(remote)/'manifest.json', staged/'manifest.json')
    meta = load(staged/'manifest.json')
    for filename, row in [(k+'.bin', meta['inputs'][k]) for k in last4.SOURCES] + [
            ('token-text.json', meta['token_text_map'])]:
        if row['path'] != filename:
            raise ValueError(f'Unexpected last4 input {row}')
        ssh_file(client, Path(remote)/filename, staged/filename, row)


def tail_model_identity(lane):
    model = []
    for i in last4.LAYERS:
        chain = load(lane/f'L{i:02d}'/'chain.json')
        if chain['status'] != 'EXECUTED' or not chain['stages']:
            raise ValueError(f'{lane}: incomplete L{i} decoder')
        stages = [(row['name'], row['model_sha256'], [p for p in row['inputs'] if p['edge'].startswith('weight/')])
                  for row in chain['stages']]
        model.append((chain['bundle_sha256'], stages))
    return model


def paired_last4(args, run):
    folder = run/last4.BUNDLE
    if not folder.is_dir():
        pull_air(run, (args.remote_run or run)/last4.BUNDLE, destination_name=last4.BUNDLE, ssh_download=ssh_last4)
    meta = load(folder/'manifest.json')
    if (meta['source_layer'], meta['layers'], meta['output_pos'], meta['cpu_basis']) != (
            35, last4.LAYERS, 18, last4.CPU_BASIS):
        raise ValueError(f'{folder}: unexpected last4 inputs')
    vocab = load(checked(folder, meta['token_text_map']))
    executing = args.action == 'board'
    if executing:
        previous = previous_replay(run, 'last4', args.recovery) if args.recovery else None
        if previous:
            prior = load(previous/'prefill.json')
            if not getattr(args, 'per_layer_omc', False):
                args.templates = prior.get('layer_templates', args.templates)
            args.conv_input_weights = getattr(args, 'conv_input_weights', None) or prior.get('conv_input_weights')
            args.per_layer_experts = getattr(args, 'per_layer_experts', False) or prior.get('per_layer_experts', False)
        out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
        summary = dict(status='RUNNING', mode='last4', layer_templates=args.templates, inputs_sha256=sha(folder/'manifest.json'),
                       conv_input_weights=getattr(args, 'conv_input_weights', None),
                       per_layer_experts=getattr(args, 'per_layer_experts', False),
                       layers=last4.LAYERS, output_pos=18, prompt=vocab['prompt'], predictions={},
                       recovered_from=str(previous) if previous else None)
        save_json(out/'prefill.json', summary)
    else:
        out = previous_replay(run, 'last4', args.replay, completed=args.replay == 'auto')
        summary = load(out/'prefill.json')
    if summary['inputs_sha256'] != sha(folder/'manifest.json'):
        raise ValueError(f'{out}: L35 inputs or tokenizer changed')
    print('Last four decoder layers, paired replay:', out, flush=True)
    with (out/'.last4.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected_model = None
        try:
            for source in last4.SOURCES:
                lane = out/source
                if executing:
                    old = previous/source if previous and (previous/source/'prefill.json').is_file() else None
                    run_board(args, run, last4.LAYERS, out=lane, previous=old,
                              entry=checked(folder, meta['inputs'][source]), entry_capture=meta['capture_sha256'])
                chain = load(lane/'L39/chain.json')
                if chain['status'] != 'EXECUTED' or load(lane/'prefill.json')['initial_hidden'] != dict(
                        source=source, after_layer=35, sha256=meta['inputs'][source]['sha256']):
                    raise ValueError(f'{lane}: incomplete or unrelated CPro tail')
                model = tail_model_identity(lane)
                if expected_model is not None and model != expected_model:
                    raise ValueError('CPU and 310P prefixes were replayed with different decoder models or weights')
                expected_model = model
                logits = read_array(lane/'L39', chain['outputs']['logits'])
                token = last4.token_result(logits, vocab)
                summary['predictions'][source] = token
                save_json(out/'prefill.json', summary)
                text = repr(token['text']) if token['text'] is not None else '<id outside tokenizer vocabulary>'
                print(f"{source} L35 → CPro L36–L39: token_id={token['token_id']} text={text}", flush=True)
            summary['status'] = 'EXECUTED'
        except BaseException as exc:
            summary.update(status='INTERRUPTED', error=str(exc))
            raise
        finally:
            save_json(out/'prefill.json', summary)
        # A numerical threshold failure on one prefix still leaves both tokens observable.
        scores = [compare_replay(run, out/k, last4.LAYERS, args.remote_run) for k in last4.SOURCES]
        summary.update(status='PASSED' if all(scores) else 'FAILED_ACCURACY')
        summary.pop('error', None)
        save_json(out/'prefill.json', summary)
        print(summary['status'], out/'prefill.json', flush=True)
        return all(scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['capture', 'air', 'board', 'compare'])
    p.add_argument('--mode', choices=['full', '40', 'last4'], required=True)
    p.add_argument('--run', type=Path, required=True)
    for name in ('capture', 'source', 'config', 'baseline', 'ddk', 'diagnosis', 'runtime-config', 'remote-run'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--recovery', nargs='?', const='auto')
    p.add_argument('--from-stage')
    p.add_argument('--conv-input-weights', choices=('external', 'embedded'),
                   help='compile only conv_input with its own layer constants and selected OMG weight storage')
    p.add_argument('--per-layer-experts', action='store_true',
                   help='use layer-specific OMCs for slices containing GroupedMatmulV555')
    p.add_argument('--per-layer-omc', action='store_true',
                   help='use the original layer-specific compiler recipe instead of shared runtime-weight templates')
    variant = p.add_mutually_exclusive_group()
    variant.add_argument('--split-decoder-projections', action='store_true')
    variant.add_argument('--split-decoder-residual', action='store_true')
    p.add_argument('--replay', default='auto')
    a = p.parse_args()
    if (a.split_decoder_projections or a.split_decoder_residual) and a.action != 'board':
        p.error('decoder split variants apply to board')
    if (a.per_layer_omc or a.per_layer_experts or a.conv_input_weights) and (a.action != 'board' or a.mode not in ('40', 'last4')):
        p.error('layer-specific weight options apply to board --mode 40 or last4')
    a.templates = a.mode in ('40', 'last4') and not a.per_layer_omc
    layers = [3] if a.mode == 'full' else (last4.LAYERS if a.mode == 'last4' else list(range(40)))
    run = output_path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    if a.action == 'capture':
        from qwen35_moe_dflash.target.prefill_t64.chain_reference import capture
        capture(a.source.resolve(strict=True), a.config, None, a.capture, states=True)
        return True
    if a.action == 'compare':
        if a.mode == 'last4':
            return paired_last4(a, run)
        replay = previous_replay(run, a.mode, a.replay, completed=a.replay == 'auto')
        return compare_replay(run, replay, layers, a.remote_run)
    with (run/'.split-prefill.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == 'board':
            select_board_defaults(a)
            if a.mode == 'last4':
                return paired_last4(a, run)
            return run_board(a, run, layers)
        check_capture(a.capture, layers)
        capture_hash = sha(a.capture/'manifest.json')
        baseline_hash = sha(a.baseline/'air/manifest.json')
        for i in layers:
            case = run/f'L{i:02d}'
            if (case/'air/manifest.json').is_file():
                meta = load(case/'air/manifest.json')
                check_bundle(meta, i, layers)
                if i % 4 != 3 and not meta.get('softplus_patch', {}).get('nodes'):
                    raise ValueError(f'L{i}: existing AIR predates SoftplusPatch. Export into a new --run.')
                if (meta['reference']['capture_sha256'], meta['original_manifest_sha256']) != (capture_hash, baseline_hash):
                    raise ValueError(f'L{i}: existing AIR belongs to another reference or CPro baseline')
                continue
            export(case, a.source, a.config, None, None, kind=SPLIT_KIND, baseline=a.baseline,
                   l0_capture=a.capture, layer=i, head=i == layers[-1], standalone=len(layers) == 1)
        return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)

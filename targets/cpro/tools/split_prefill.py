#!/usr/bin/env python3
"""Run independent SiluPatch OMCs inside the L3 decoder or a continuous L0–L39 prefill."""
import argparse
from functools import partial
import fcntl
from pathlib import Path
import tempfile

from l0_silu_omcs import compare, execute, load
from decoder4_original import L0_DIR, SPLIT_KIND, export, ssh_bundle
from decoder4_prefill import pull_air
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.layer_probe.common import checked, save_json, sha
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
        if (ref.get('split_layer') not in layers[1:] or
                blocks != [(i, 'full_attention' if i % 4 == 3 else 'linear_attention') for i in layers]):
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
        if report['mode'] == mode and (not completed or report['status'] in ('PASSED', 'FAILED_ACCURACY')):
            return path.parent
    raise ValueError(f'No {mode} replay matching {selector} in {run}')


def run_board(args, run, layers):
    previous = previous_replay(run, args.mode, args.recovery) if args.recovery else None
    start_layer = start_stage = None
    if args.from_stage:
        key, sep, start_stage = args.from_stage.partition('/')
        if not sep and len(layers) == 1:
            key, start_stage = f'L{layers[0]:02d}', key
        if not previous or key not in [f'L{i:02d}' for i in layers] or not start_stage:
            raise ValueError('--from-stage requires --recovery and an Lxx/stage name')
        start_layer = int(key[1:])
    out = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    summary = dict(mode=args.mode, status='RUNNING', layers=[], recovered_from=str(previous) if previous else None)
    manifest = out/'prefill.json'
    save_json(manifest, summary)
    print('Prefill replay:', out, flush=True)
    hidden = capture = baseline = None
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
            graph = plan(raw, meta['silu_patch']['nodes'], layer=i, head=meta['head'], **options)
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
            if args.mode == '40' and i == meta['reference']['split_layer']:
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
            execute(case, bundle, meta, graph, raw, args.ddk.resolve(strict=True), args.diagnosis,
                    args.runtime_config.resolve(strict=True), recovery=checkpoint, entry=hidden,
                    from_stage=start_stage if i == start_layer and checkpoint else None, out=dst, shared_silu=run/'silu-omc')
            receipt = load(dst/'chain.json')
            hidden = checked(dst, receipt['outputs']['decoder_hidden'])
            passed = compare(bundle, dst, remote_air=remote_air)
            summary['layers'].append(dict(layer=i, passed=passed, hidden_sha256=receipt['outputs']['decoder_hidden']['sha256'],
                                          report=f'{key}/comparison.json', report_sha256=sha(dst/'comparison.json')))
            save_json(manifest, summary)
        summary['status'] = 'PASSED' if all(s['passed'] for s in summary['layers']) else 'FAILED_ACCURACY'
    except BaseException as exc:
        summary.update(status='INTERRUPTED', layer=i, error=str(exc))
        raise
    finally:
        save_json(manifest, summary)
        print(summary['status'], manifest, flush=True)
    return summary['status'] == 'PASSED'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['capture', 'air', 'board', 'compare'])
    p.add_argument('--mode', choices=['full', '40'], required=True)
    p.add_argument('--run', type=Path, required=True)
    for name in ('capture', 'source', 'config', 'baseline', 'ddk', 'diagnosis', 'runtime-config', 'remote-run'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--recovery', nargs='?', const='auto')
    p.add_argument('--from-stage')
    variant = p.add_mutually_exclusive_group()
    variant.add_argument('--split-decoder-projections', action='store_true')
    variant.add_argument('--split-decoder-residual', action='store_true')
    p.add_argument('--replay', default='auto')
    a = p.parse_args()
    if (a.split_decoder_projections or a.split_decoder_residual) and a.action != 'board':
        p.error('decoder split variants apply to board')
    layers = [3] if a.mode == 'full' else list(range(40))
    run = output_path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    if a.action == 'capture':
        from qwen35_moe_dflash.target.prefill_t64.chain_reference import capture
        capture(a.source.resolve(strict=True), a.config, None, a.capture, states=True)
        return True
    if a.action == 'compare':
        replay = previous_replay(run, a.mode, a.replay, completed=a.replay == 'auto')
        report = load(replay/'prefill.json')
        finished = report['status'] in ('PASSED', 'FAILED_ACCURACY')
        if finished and [s['layer'] for s in report['layers']] != layers:
            raise ValueError(f'Incomplete layer list in {replay}')
        passed = []
        for i in layers:
            key = f'L{i:02d}'
            dst = replay/key
            if not finished and not (dst/'chain.json').is_file():
                break
            passed.append(compare(run/key/'air', dst, allow_failed=not finished,
                                  remote_air=(a.remote_run/key/'air' if a.remote_run else None)))
        if not finished:
            print(f'Incomplete {a.mode} replay: {replay}', flush=True)
        return finished and all(passed)
    with (run/'.split-prefill.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == 'board':
            return run_board(a, run, layers)
        check_capture(a.capture, layers)
        capture_hash = sha(a.capture/'manifest.json')
        baseline_hash = sha(a.baseline/'air/manifest.json')
        for i in layers:
            case = run/f'L{i:02d}'
            if (case/'air/manifest.json').is_file():
                meta = load(case/'air/manifest.json')
                check_bundle(meta, i, layers)
                if (meta['reference']['capture_sha256'], meta['original_manifest_sha256']) != (capture_hash, baseline_hash):
                    raise ValueError(f'L{i}: existing AIR belongs to another reference or CPro baseline')
                continue
            export(case, a.source, a.config, None, None, kind=SPLIT_KIND, baseline=a.baseline,
                   l0_capture=a.capture, layer=i, head=i == layers[-1], standalone=len(layers) == 1)
        return True


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)

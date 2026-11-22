"""Feed recovered CPro hidden forward, including the original segment trim/pad."""
import csv
import fcntl
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import shlex
from types import SimpleNamespace

import numpy as np

from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import checked, metrics, save_json, sha
from .chain_reference import BASIS, HIDDEN, LAYERS


def load(path):
    return json.loads(path.read_text())


def hidden_input(previous, role, reset_padding):
    if previous.shape != HIDDEN or previous.dtype != np.dtype('<f2'):
        raise ValueError('expected the complete FP16 hidden returned by the preceding CPro model')
    if role == 'head':
        return previous[:, 18, :].copy()
    if reset_padding:
        previous = previous.copy()
        previous[:, 19:] = 0
    return previous


def bind_hidden(fixture, meta, case_id, value, predecessor):
    case, = meta['cases']
    slot, = [x for x in case['inputs'] if x['semantic_name'] == 'hidden']
    if case['case'] != case_id or tuple(slot['shape']) != value.shape or slot['dtype'] != value.dtype.name:
        raise ValueError('CPro feedback descriptor differs from the compiled input')
    path = fixture / case_id / f"input_{slot['index']}.bin"
    raw = value.tobytes()
    if len(raw) != slot['fixture_bytes'] or sha(path) != slot['fixture_sha256']:
        raise ValueError('unexpected original fixture input')
    path.write_bytes(raw)
    slot.update(fixture_sha256=sha(path), cpro_predecessor=predecessor)


def discard_board_weight_copies(api, directory, case_id):
    # Only temporary copies of the recovered job; outputs, logs and original OMC are retained.
    plan = load(directory / 'execution/request.json')
    result = load(directory / 'recovery/result.json')
    token = plan['token']
    board = PurePosixPath(plan['board_attempt'])
    if (not re.fullmatch('[0-9a-f]{32}', token) or str(board) != api.PREFIX + token
            or plan['cases'] != [case_id] or result['board_attempt'] != str(board)
            or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'):
        raise ValueError('cannot establish ownership of completed upload')
    exe, serial, server = api._environment()
    if (serial, server) != (plan['serial'], plan['hdc_server']):
        raise ValueError('board identity differs from the recovered run')
    hdc = api.Hdc(exe, serial, server, directory / 'recovery', 180)
    command = (api._guard(board, token) + ' && test "$(cat ' + shlex.quote(str(board/'complete')) + ')" = ' + token
               + ' && rm -f ' + shlex.join([str(board/'payload.tar'), str(board/case_id/'model.omc')]))
    hdc.shell(command, 'release-chain-upload')
    (directory / 'upload-released').touch()


def run(compiled, output, config, ddk):
    from .segment_trace_compare import board_case
    import validate_draft_vector as api

    output = output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = load(compiled / 'manifest.json')
        build = load(compiled / 'compile.json')
        expected = ['l00_c00_embedding_o0', *[f'l{i:02d}_c00_decoder_o0' for i in range(LAYERS)], 'l39_c00_head_o0']
        if (manifest.get('scope') != BASIS or [r['id'] for r in manifest['cases']] != expected
                or build['status'] != 'COMPILED' or any(build['cases'][n]['status'] != 'COMPILED' for n in expected)):
            raise ValueError('expected 42 compiled models: embedding, 40 complete decoder blocks, final norm/head')
        token_map = load(checked(compiled / build['source_batch'], manifest['token_text_map']))
        _, serial, server = api._environment()
        signature = dict(board_serial=serial, hdc_server=server, compile_sha256=sha(compiled/'compile.json'), config_sha256=sha(config),
                         capture_sha256=manifest['capture_sha256'], gdr_source_sha256=build['gdr_source_sha256'])
        marker = output / 'source.json'
        if marker.exists() and load(marker) != signature:
            raise ValueError('compiled models, reference or runtime configuration changed; choose another output')
        save_json(marker, signature)
        rows = []
        previous = predecessor = None
        for entry in manifest['cases']:
            name, role, i = entry['id'], entry['point'], entry['layer']
            base = output / name
            base.mkdir(exist_ok=True)
            execution, recovery = base/'execution', base/'recovery'
            carry = None if role == 'embedding' else hidden_input(previous, role, role == 'decoder' and i == manifest['split_layer'])
            if not execution.exists():
                args = SimpleNamespace(config=config, compiled=compiled, ddk_root=ddk, cases=[name], apply=True)
                hook = None if carry is None else lambda f, m: bind_hidden(f, m, name, carry, predecessor)
                launched = api.start(args, execution, bind_inputs=hook)
                if launched['status'] != 'VECTOR_VALIDATION_STARTED':
                    raise RuntimeError(f"{name}: {launched.get('error', launched['status'])}")
            if not (execution/'request.json').exists():
                raise RuntimeError(f'{name}: no detached request, inspect {execution}/result.json')
            if not (recovery/'result.json').is_file() or load(recovery/'result.json').get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED':
                # Recover the existing launch, including after an interrupted recovery.
                if recovery.exists():
                    attempt = 1
                    while (base/f'recovery-retry-{attempt}').exists():
                        attempt += 1
                    recovery.rename(base/f'recovery-retry-{attempt}')
                recovered = api.recover(execution, recovery, wait=True)
                if recovered.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED':
                    raise RuntimeError(f"{name}: {recovered.get('error', recovered['status'])}")
            inputs, refs, actual, case = board_case(base, name)
            if case['model']['sha256'] != build['cases'][name]['omc']['sha256']:
                raise ValueError(f'{name}: recovered another model')
            if carry is not None and inputs['hidden'][2] != hashlib.sha256(carry.tobytes()).hexdigest():
                raise ValueError(f'{name}: measured board run did not consume preceding CPro hidden')
            if role == 'head':
                a, b = (x['logits'].reshape(-1) for x in (actual, refs))
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('nonfinite final logits; greedy token unavailable')
                token, native = int(a.argmax()), manifest['reference_top1']
                if b[native] != b.max():
                    raise ValueError('reference head differs from original native top1')
                if len(token_map['pieces']) != a.size or token_map['pieces'][native] != manifest['reference_text']:
                    raise ValueError('token map differs from native readout or vocabulary')
                piece = token_map['pieces'][token]
                result = dict(prompt=token_map['prompt'], board_text=piece, board_token_in_vocab=piece is not None,
                    board_token_is_special=token in token_map['special_ids'], **signature, board_token=token, reference_token=native, top1_equal=token == native,
                    reference_text=manifest['reference_text'], logits=metrics(a, b), generated_tokens=1,
                    first_failed_decoder=next((r['layer'] for r in rows if r['role']=='decoder' and not r['passed']), None))
                print(f"first token: CPro={token} ({piece!r}), 310P={native} ({manifest['reference_text']!r}), equal={token == native}", flush=True)
                save_json(output/'report'/'first-token.json', result)
                share_for_transfer(output/'report')
            else:
                previous = actual['hidden'].reshape(HIDDEN)
                ref = refs['hidden'].reshape(HIDDEN)
                if not np.isfinite(previous[:, :19]).all() or not np.isfinite(ref[:, :19]).all():
                    raise ValueError(f'{name}: nonfinite valid hidden, see {recovery}/native')
                valid, last = (metrics(previous[:, part], ref[:, part]) for part in [slice(0,19), 18])
                rows.append(dict(role=role, layer=i, valid19_relative_l2=valid['relative_l2'], valid19_cosine=valid['cosine'],
                    token18_relative_l2=last['relative_l2'], token18_cosine=last['cosine'],
                    passed=valid['passed'] and last['passed'], valid19_failed=','.join(valid['failed_metrics']),
                    token18_failed=','.join(last['failed_metrics'])))
                report_dir = output/'report'
                report_dir.mkdir(exist_ok=True)
                temp = report_dir/'layers.csv.tmp'
                with temp.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                temp.replace(report_dir/'layers.csv')
                print(f"{role} {i:02d} valid19 L2/cos={valid['relative_l2']:.4%}/{valid['cosine']:.7f}"
                      f" token18={last['relative_l2']:.4%}/{last['cosine']:.7f} passed={rows[-1]['passed']}", flush=True)
            if not (base/'upload-released').exists():
                discard_board_weight_copies(api, base, name)
            # The detached request and recovered outputs are already verified.
            # A following layer builds its own tar. Keep the OMC and fixture for replay.
            if build.get('storage') == 'per-layer-v1':
                archive = execution / 'payload.tar'
                if archive.exists():
                    receipt = load(execution/'request.json')['payload']
                    if (archive.resolve() != archive or receipt['local'] != str(archive)
                            or archive.stat().st_size != receipt['bytes'] or sha(archive) != receipt['sha256']):
                        raise ValueError(f'{name}: local upload archive differs from the completed request')
                    archive.unlink()
            predecessor = name

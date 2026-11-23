"""Replay a completed, failed decoder request once with the device's HIAI V2 tool."""
from pathlib import PurePosixPath
import re
import shlex
import uuid

import numpy as np

from ...common.artifacts import output_path, share_for_transfer
from ..board_tool_inspect import DEFAULT_TOOL
from ..layer_probe.common import checked, metrics, save_json, sha
from ..original_target_board import _board_path
from ..prefill_smoke_plan import _TOOL_TYPES
from .chain_board import load
from .chain_reference import HIDDEN


def replay(case_dir, tool_path=DEFAULT_TOOL):
    import validate_draft_vector as api

    case_dir = output_path(case_dir)
    name = case_dir.name
    plan = load(case_dir / 'execution/request.json')
    recovered = load(case_dir / 'recovery/result.json')
    token = plan['token']
    board = PurePosixPath(plan['board_attempt'])
    failure = recovered.get('execution_diagnostics', {}).get('first_failure', {})
    if (not re.fullmatch(r'l\d\d_c00_decoder_o0', name) or not re.fullmatch('[0-9a-f]{32}', token)
            or str(board) != api.PREFIX + token or plan['cases'] != [name]
            or recovered.get('board_attempt') != str(board)
            or recovered.get('lock_state') != 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
            or failure.get('case') != name or not failure.get('exitcode')
            or 'HIAI Process(sync) failed:' not in failure.get('stderr_tail', '')):
        raise ValueError('Expected the original completed decoder request with a recorded V1 Process failure')
    fixture = case_dir / 'execution/fixture'
    manifest = fixture / 'manifest.json'
    if sha(manifest) != plan['fixture_sha256']:
        raise ValueError('Fixture manifest changed since the V1 launch')
    meta = load(manifest)
    case, = meta['cases']
    output, = case['outputs']
    inputs = case['inputs']
    if (case['case'] != name or case['kind'] != 'prefill_chain' or case['point'] != 'decoder'
            or case.get('external_weights') or case['model']['remote'] != f'{name}/model.omc'
            or [s['index'] for s in inputs] != list(range(len(inputs)))
            or output['semantic_name'] != 'hidden' or output['dtype'] != 'float16' or output['golden_dtype'] != 'float16'
            or output['bytes'] != np.prod(HIDDEN) * 2 or output['fixture_bytes'] != output['bytes']):
        raise ValueError('Unexpected embedded decoder fixture or hidden output contract')
    folder = board / name
    model = folder / 'model.omc'
    checks = [api._identity_guard(model, case['model'])]
    paths = []
    for s in inputs:
        filename = f"input_{s['index']}.bin"
        record = dict(bytes=s['fixture_bytes'], sha256=s['fixture_sha256'])
        local = checked(fixture/name, dict(path=filename, **record))
        values = np.fromfile(local, dtype=s['dtype'])
        summary = f"slot {s['index']} {s['semantic_name']} {s['dtype']} {s['shape']}"
        if s['semantic_name'] == 'effective_length':
            summary += f' = {values.tolist()}'
            if values.tolist() != [19]:
                raise ValueError(summary)
        print(summary, flush=True)
        paths.append(str(folder / filename))
        checks.append(api._identity_guard(folder / filename, record))
    golden = checked(fixture/name, dict(path='golden_0.bin', bytes=output['fixture_bytes'], sha256=output['fixture_sha256']))
    exe, serial, server = api._environment()
    if (serial, server) != (plan['serial'], plan['hdc_server']):
        raise ValueError('Select the board recorded in execution/request.json')
    token_v2 = uuid.uuid4().hex
    work = board / ('v2-' + token_v2)
    out = case_dir / 'recovery' / work.name
    out.mkdir()
    print(f'V2 replay: {out}', flush=True)
    hdc = api.Hdc(exe, serial, server, out, 700)
    tool = _board_path(str(tool_path))
    argv = [str(tool), '--model=' + str(model), '--input=' + ','.join(paths),
            '--input_type=' + ','.join(_TOOL_TYPES[s['dtype'].upper()] for s in inputs),
            '--output_type=FP16', '--times=1', '--modelmanager_itf=V2', '--direct_from_file=true',
            '--enable_item=1', '--output_dir=' + str(work / 'outputs')]
    report = dict(status='PREPARING', case=name, previous_attempt=str(board), device_work=str(work),
                  omc_sha256=case['model']['sha256'], fixture_sha256=plan['fixture_sha256'], argv=argv)
    save_json(out/'result.json', report)
    api._acquire_global(hdc, api.GLOBAL_LOCK, token_v2, 0)
    uncertain = False
    try:
        _, fw = hdc.shell('param get const.ohos.fullname', 'firmware')
        if fw.strip() != plan['runtime']['firmware_id'].strip():
            raise ValueError('Firmware changed since the V1 execution')
        hdc.shell(' && '.join([api._guard(board, token), f'test "$(cat {board}/complete)" = {token}',
                  f'test "$(cat {board}/{name}.exitcode)" = {int(failure["exitcode"])}',
                  f'test -x {tool}', *checks, f'mkdir {work}', f'mkdir {work}/outputs']), 'verify-retained-case')
        report['status'] = 'RUN_REQUESTED'
        save_json(out/'result.json', report)
        uncertain = True
        _, log = hdc.shell(f'cd {work} && touch tool.started && {{ timeout -k 5 600 env LD_LIBRARY_PATH={tool.parent} '
                           + shlex.join(argv) + ' >tool.stdout 2>tool.stderr; rc=$?; echo "$rc" >tool.exitcode; '
                           + 'tail -n 20 tool.stdout tool.stderr; }', 'v2-run')
        _, code = hdc.shell(f'cat {work}/tool.exitcode', 'v2-exitcode')
        report['tool_exitcode'] = int(code)
        uncertain = False
        if report['tool_exitcode']:
            raise RuntimeError(f'HIAI V2 exited {code}. Device logs: {work}/tool.stdout and tool.stderr\n{log}')
        _, listing = hdc.shell(f"cd {work}/outputs && find . -type f -size {output['bytes']}c -exec sha256sum {{}} \\;",
                               'v2-output-hash')
        matches = re.findall(r'^([0-9a-f]{64})\s+(\./[^\r\n]+)$', listing, re.M)
        if len(matches) != 1:
            raise ValueError('Expected one full decoder hidden of the declared byte length: ' + listing)
        digest, filename = matches[0]
        path = out / 'hidden.bin'
        hdc.receive(_board_path(str(work/'outputs'/filename)), path)
        if path.stat().st_size != output['bytes'] or sha(path) != digest:
            raise ValueError('V2 hidden readback size or SHA-256 differs')
        actual, ref = (np.fromfile(p, dtype='<f2').reshape(HIDDEN) for p in (path, golden))
        report.update(status='EXECUTED', output_sha256=digest, invocations=1)
        for scope, index in [('valid19', slice(0, 19)), ('token18', 18)]:
            m = report[scope] = metrics(actual[:, index], ref[:, index])
            print(f"{scope}: relative_L2={m['relative_l2']} cosine={m['cosine']} "
                  f"error_ratio={m['error_ratio']} five_metrics_passed={m['passed']}", flush=True)
    except BaseException as exc:
        report.update(status='EXECUTION_UNCERTAIN' if uncertain else 'FAILED', error=str(exc))
        raise
    finally:
        save_json(out/'result.json', report)
        if uncertain:
            print(f'Execution acknowledgement lost. Check {work}/tool.exitcode; board lock owner {token_v2}', flush=True)
        else:
            api._release_global(hdc, api.GLOBAL_LOCK, token_v2)
        share_for_transfer(out)
    return report

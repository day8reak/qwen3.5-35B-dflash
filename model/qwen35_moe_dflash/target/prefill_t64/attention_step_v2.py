"""Run one independent attention step with the X90 board's existing HIAI V2 tool."""
from pathlib import Path, PurePosixPath
import re
import shlex
import tempfile
import uuid

from ...common.command_trace import note
from ...draft.profile_board import GLOBAL_LOCK
from ...execution.hdc_run import Hdc
from ...execution.lock import _acquire_global, _release_global, _space
from ...execution.runtime_config import read_json
from ...execution.serial_transport import _environment
from ..board_tool_inspect import DEFAULT_TOOL
from ..layer_probe.batch import prepare_fixture
from ..layer_probe.common import checked, read_array, save_json, sha
from ..original_target_board import _board_path
from ..prefill_smoke_plan import _TOOL_TYPES


def execute(run, name, ddk, config):
    out = run/'v2'
    if (out/'result.json').is_file():
        receipt = read_json(out/'result.json')
        if receipt['status'] == 'EXECUTED':
            note(f'[reuse V2 output] {out / "result.json"}')
            return
        raise ValueError(f'Previous V2 attempt: {receipt["status"]}. See {out}/result.json. Use a new run to retry.')
    env = _environment()
    runtime = read_json(config.parent/read_json(config)['assets'])['environment']
    with tempfile.TemporaryDirectory(dir=run, prefix='.v2-fixture-') as tmp:
        prepared = Path(tmp)/'prepared'
        meta = prepare_fixture(run/'omc', prepared, ddk, runtime['ddk_id'], [name])
        for key in ('repetitions', 'warmup', 'output_initialization', 'input_reset'):
            meta.pop(key)
        meta.update(runner='model_run_tool_internal', invocations=1)
        save_json(prepared/'fixture/manifest.json', meta)
        prepared.rename(out)
    case, = meta['cases']
    operand, = case['inputs']
    result, = case['outputs']
    fixture = out/'fixture'
    model = checked(run/'omc', read_json(run/'omc/compile.json')['cases'][name]['omc'])
    infile = checked(fixture/name, dict(path='input_0.bin', bytes=operand['bytes'], sha256=operand['sha256']))
    token = uuid.uuid4().hex
    remote = PurePosixPath('/data/local/tmp') / ('qwen35_step_' + token)
    tool = _board_path(DEFAULT_TOOL)
    argv = [str(tool), f'--model={remote}/model.omc', f'--input={remote}/input_0.bin',
            '--input_type=' + _TOOL_TYPES[operand['dtype'].upper()],
            '--output_type=' + _TOOL_TYPES[result['dtype'].upper()], '--times=1', '--modelmanager_itf=V2',
            '--direct_from_file=true', '--enable_item=1', f'--output_dir={remote}/outputs']
    command = f'timeout -k 5 600 env LD_LIBRARY_PATH={tool.parent} ' + shlex.join(argv)
    script = out/'run.sh'
    script.write_text(f'cd {remote} || exit 1\n{command} >tool.stdout 2>tool.stderr\n'
                      'rc=$?\necho "$rc" >tool.exitcode\ntail -n 20 tool.stdout tool.stderr\nexit "$rc"\n')
    note(f'[V2 board script] {script}\n{script.read_text()}')
    hashes = out/'files.sha256'
    hashes.write_text(f"{case['model']['sha256']}  model.omc\n{operand['sha256']}  input_0.bin\n"
                      f"{sha(script)}  run.sh\n")
    receipt = dict(status='PREPARED', case=name, device=env[1], server=env[2], remote=str(remote),
                   argv=argv, omc_sha256=case['model']['sha256'], input_sha256=operand['sha256'],
                   fixture_sha256=sha(fixture/'manifest.json'))
    save_json(out/'result.json', receipt)
    hdc = Hdc(*env, out, 660)
    held = uncertain = False
    try:
        _acquire_global(hdc, GLOBAL_LOCK, token, 0)
        held = True
        _, fw = hdc.shell('param get const.ohos.fullname', 'firmware')
        if fw.strip() != runtime['firmware_id'].strip():
            raise ValueError(f'Board firmware differs from {config}')
        _, space = hdc.shell(f'test -x {tool} && df -Pk {remote.parent}', 'preflight')
        uploads = [(model, 'model.omc'), (infile, 'input_0.bin'), (script, 'run.sh'), (hashes, 'files.sha256')]
        needed = result['bytes'] + sum(source.stat().st_size for source, _ in uploads)
        if _space(space) < needed:
            raise OSError(f'Insufficient board storage for {name}: need {needed:,} bytes')
        hdc.shell(f'mkdir {remote} && mkdir {remote}/outputs', 'create-board-dir')
        for source, dest in uploads:
            hdc.send(source, remote/dest)
        hdc.shell(f'cd {remote} && sha256sum -c files.sha256', 'verify-upload')
        receipt['status'] = 'RUN_REQUESTED'
        save_json(out/'result.json', receipt)
        uncertain = True
        _, log = hdc.shell(f'cd {remote} && {{ sh run.sh; test -s tool.exitcode; }}', 'v2-execute')
        _, code = hdc.shell(f'cat {remote}/tool.exitcode', 'v2-exitcode')
        receipt['exitcode'] = int(code)
        uncertain = False
        if receipt['exitcode']:
            raise RuntimeError(f'HIAI V2 exited {code} in {remote}\n{log}')
        _, listing = hdc.shell(f"cd {remote}/outputs && find . -type f -size {result['bytes']}c "
                               + '-exec sha256sum {} \\;', 'output-hash')
        matches = re.findall(r'^([0-9a-f]{64})\s+(\./[^\r\n]+)$', listing, re.M)
        if len(matches) != 1:
            raise ValueError(f"Expected one {result['bytes']}-byte output in {remote}/outputs: {listing}")
        digest, filename = matches[0]
        hdc.receive(_board_path(str(remote/'outputs'/filename)), out/'output.bin')
        output = dict(path='output.bin', bytes=result['bytes'], sha256=digest)
        checked(out, output)
        receipt.update(status='EXECUTED', output=output)
    except BaseException as exc:
        receipt.update(status='EXECUTION_UNCERTAIN' if uncertain else 'FAILED', error=str(exc))
        raise
    finally:
        save_json(out/'result.json', receipt)
        if uncertain:
            print(f'HDC lost execution acknowledgement. Check {remote}/tool.exitcode. Board lock owner {token}', flush=True)
        elif held:
            _release_global(hdc, GLOBAL_LOCK, token)


def read_case(run, name, fixture):
    out = run/'v2'
    receipt = read_json(out/'result.json')
    case, = fixture['cases']
    if (receipt['status'] != 'EXECUTED' or receipt['case'] != name or case['case'] != name
            or receipt['fixture_sha256'] != sha(out/'fixture/manifest.json')
            or fixture['source_evidence']['compile_sha256'] != sha(run/'omc/compile.json')
            or receipt['omc_sha256'] != case['model']['sha256']):
        raise ValueError(f'V2 output belongs to a different model or fixture: {out}')
    operand, = case['inputs']
    result, = case['outputs']
    if receipt['input_sha256'] != operand['fixture_sha256']:
        raise ValueError('V2 input differs from the measured operand')
    inputs = {operand['semantic_name']: (operand['dtype'], operand['shape'], operand['fixture_sha256'])}
    golden = read_array(out/'fixture'/name, dict(path='golden_0.bin', dtype=result['golden_dtype'],
                        shape=result['shape'], bytes=result['fixture_bytes'], sha256=result['fixture_sha256']))
    key = result['semantic_name']
    return inputs, {key: golden}, {key: read_array(out, dict(receipt['output'], dtype=result['dtype'], shape=result['shape']))}

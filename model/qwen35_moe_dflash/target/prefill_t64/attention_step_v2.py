"""Run one independent attention step with the X90 board's existing HIAI V2 tool."""
from pathlib import Path, PurePosixPath
import math
import re
import shlex
import tempfile
import uuid

import numpy as np

from ...common.command_trace import note
from ...draft.profile_board import GLOBAL_LOCK
from ...execution.hdc_run import Hdc
from ...execution.lock import _acquire_global, _release_global, _space
from ...execution.runtime_config import read_json
from ...execution.serial_transport import _environment
from ..board_tool_inspect import DEFAULT_TOOL
from ..layer_probe.batch import prepare_fixture
from ..layer_probe.common import DTYPES, checked, read_array, save_json, sha
from ..original_target_board import _board_path
from ..prefill_smoke_plan import _TOOL_TYPES, _nd_view_key, _shape
from .whole_slots import observe_model_io


def execute(run, name, ddk, config):
    out = run/'v2'
    if (out/'result.json').is_file():
        receipt = read_json(out/'result.json')
        if receipt['status'] == 'EXECUTED':
            note(f'[reuse V2 output] {out / "result.json"}')
            return
        raise ValueError(f'Previous V2 attempt: {receipt["status"]}. See {out}/result.json. Use a new run to retry.')
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
    operands = case['inputs']
    result, = case['outputs']
    fixture = out/'fixture'
    model = checked(run/'omc', read_json(run/'omc/compile.json')['cases'][name]['omc'])
    if not operands or [s['index'] for s in operands] != list(range(len(operands))):
        raise ValueError('Expected inputs in compiled slot order')
    files = [checked(fixture/name, dict(s, path=f"input_{s['index']}.bin")) for s in operands]
    return run_v2(out, name, model, operands, files, [result], runtime,
                  fixture_sha256=sha(fixture/'manifest.json'))


def runtime_slots(native, direction, specs):
    """Bind each observed slot to its tensor by name and typed view, or by a unique typed view."""
    descs = native[direction]
    if len(descs) != len(specs) or native[direction[:-1] + '_count'] != len(specs):
        raise ValueError(f'HIAI {direction} count differs from compiled ABI')
    names = {s['name']: i for i, s in enumerate(specs)}
    if len(names) != len(specs):
        raise ValueError(f'Duplicate compiled {direction} names')
    types = native['dtype_enum_values']
    keys = [_nd_view_key((types[s['dtype'].upper()], _shape(s))) for s in specs]
    slots = []
    used = set()
    for slot, desc in enumerate(descs):
        if desc['index'] != slot or desc['raw_format'] != native['format_enum_values']['ND']:
            raise ValueError(f'HIAI {direction}[{slot}] has an unexpected index or format: {desc}')
        key = _nd_view_key((desc['raw_dtype'], _shape(desc)))
        matches = [i for i, expected in enumerate(keys) if expected == key]
        name = desc.get('tensor_name') if desc.get('tensor_name_status') == 'OBSERVED' else None
        if name in names:
            matches = [i for i in matches if i == names[name]]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError(f'Cannot bind HIAI {direction}[{slot}] {desc} to a unique compiled tensor: '
                             f'candidates={[specs[i]["name"] for i in matches]}')
        index, = matches
        used.add(index)
        spec = specs[index]
        size = math.prod(spec['shape']) * np.dtype(DTYPES[spec['dtype']]).itemsize
        slots.append(dict(slot=slot, compiled_index=index, name=spec['name'], runtime_name=name,
                          binding='name_and_typed_shape' if name in names else 'unique_typed_shape',
                          dtype=spec['dtype'], shape=spec['shape'], bytes=size))
    return slots


def run_v2(out, name, model, operands, files, outputs, runtime, *, fixture_sha256=None, slot_config=None):
    from ..prefill_output_inventory import _logged_slots

    env = _environment()
    input_names = [f'input_{i}.bin' for i in range(len(files))]
    if not operands or len(operands) != len(files) or not outputs:
        raise ValueError(f'{name}: missing model inputs or outputs')
    token = uuid.uuid4().hex
    remote = PurePosixPath('/data/local/tmp') / ('qwen35_step_' + token)
    tool = _board_path(DEFAULT_TOOL)
    model_sha = sha(model)
    output_order = list(range(len(outputs)))
    receipt = dict(status='PREPARED', case=name, device=env[1], server=env[2], remote=str(remote),
                   omc_sha256=model_sha, input_hashes=[s['sha256'] for s in operands],
                   fixture_sha256=fixture_sha256)
    if len(operands) == 1:
        receipt['input_sha256'] = operands[0]['sha256']
    save_json(out/'result.json', receipt)
    hdc = Hdc(*env, out, 660)
    held = uncertain = False
    try:
        _acquire_global(hdc, GLOBAL_LOCK, token, 0)
        held = True
        _, fw = hdc.shell('param get const.ohos.fullname', 'firmware')
        if fw.strip() != runtime['firmware_id'].strip():
            raise ValueError(f"Board firmware {fw.strip()!r} differs from {runtime['firmware_id']!r}")
        _, space = hdc.shell(f'test -x {tool} && df -Pk {remote.parent}', 'preflight')
        needed = model.stat().st_size + sum(p.stat().st_size for p in files) + sum(s['bytes'] for s in outputs)
        if _space(space) < needed:
            raise OSError(f'Insufficient board storage for {name}: need {needed:,} bytes')
        hdc.shell(f'mkdir {remote} && mkdir {remote}/outputs', 'create-board-dir')
        hdc.send(model, remote/'model.omc')
        if slot_config is not None:
            native = observe_model_io(hdc, slot_config, runtime, remote, out, remote/'model.omc', model_sha, tool.parent)
            slots = {k: runtime_slots(native, k, v) for k, v in [('inputs', operands), ('outputs', outputs)]}
            save_json(out/'slots.json', slots)
            for direction, entries in slots.items():
                for row in entries:
                    print(f"{name} {direction}[{row['slot']}] = {row['name']} "
                          f"{row['dtype']} {row['shape']} ({row['bytes']} bytes)", flush=True)
            input_order = [s['compiled_index'] for s in slots['inputs']]
            output_order = [s['compiled_index'] for s in slots['outputs']]
            operands = [operands[i] for i in input_order]
            files = [files[i] for i in input_order]
            outputs = [outputs[i] for i in output_order]
            for row, path in zip(slots['inputs'], files, strict=True):
                if path.stat().st_size != row['bytes']:
                    raise ValueError(f"{name}: {path} differs from the {row['bytes']}-byte {row['name']} input")
        argv = [str(tool), f'--model={remote}/model.omc', '--input=' + ','.join(str(remote/n) for n in input_names),
                '--input_type=' + ','.join(_TOOL_TYPES[s['dtype'].upper()] for s in operands),
                '--output_type=' + ','.join(_TOOL_TYPES[s['dtype'].upper()] for s in outputs), '--times=1', '--modelmanager_itf=V2',
                '--direct_from_file=true', '--enable_item=1', f'--output_dir={remote}/outputs']
        command = f'timeout -k 5 600 env LD_LIBRARY_PATH={tool.parent} ' + shlex.join(argv)
        script = out/'run.sh'
        script.write_text(f'cd {remote} || exit 1\n{command} >tool.stdout 2>tool.stderr\n'
                          'rc=$?\necho "$rc" >tool.exitcode\ntail -n 20 tool.stdout tool.stderr\nexit "$rc"\n')
        note(f'[V2 board script] {script}\n{script.read_text()}')
        hashes = out/'files.sha256'
        hashes.write_text(f"{model_sha}  model.omc\n"
                          + ''.join(f"{s['sha256']}  {n}\n" for s, n in zip(operands, input_names))
                          + f"{sha(script)}  run.sh\n")
        receipt.update(argv=argv, input_hashes=[s['sha256'] for s in operands])
        uploads = [*zip(files, input_names), (script, 'run.sh'), (hashes, 'files.sha256')]
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
            raise RuntimeError(f'{name}: HIAI V2 exited {code} in {remote}. Receipt: {out}/result.json\n{log}')
        size_filter = f" -size {outputs[0]['bytes']}c" if len(outputs) == 1 else ''
        _, listing = hdc.shell(f'cd {remote}/outputs && find . -type f{size_filter} -exec sha256sum {{}} \\;', 'output-hash')
        hash_by_file = {name[2:]: digest for digest, name in
                        re.findall(r'^([0-9a-f]{64})\s+(\./[^\r\n]+)$', listing, re.M)}
        if len(outputs) == 1 and len(hash_by_file) == 1:
            names = list(hash_by_file)
        else:
            # The two SiLU wires can have the same byte size. Use the board tool's
            # explicit output[index] mapping, even when filenames happen to be numbered.
            _, logs = hdc.shell(f'cat {remote}/tool.stdout', 'output-slots')
            _, sizes = hdc.shell(f'cd {remote}/outputs && find . -type f -exec wc -c {{}} \\;', 'output-sizes')
            inventory = [dict(path='outputs/' + n, bytes=int(sz)) for sz, n in
                         re.findall(r'^\s*(\d+)\s+\./([^\r\n]+)$', sizes, re.M) if n in hash_by_file]
            mapping = _logged_slots(logs.encode(), dict(board_attempt=str(remote)), outputs, inventory)
            if mapping['status'] != 'BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG':
                raise ValueError(f"{name}: {mapping['reason']}; device logs at {remote}/tool.stdout")
            names = [row['path'].removeprefix('outputs/') for row in mapping['entries']]
        retrieved = [None] * len(outputs)
        for i, (filename, spec) in enumerate(zip(names, outputs, strict=True)):
            compiled = output_order[i]
            path = 'output.bin' if len(outputs) == 1 else f'output_{compiled}.bin'
            hdc.receive(_board_path(str(remote/'outputs'/filename)), out/path)
            row = dict(path=path, bytes=spec['bytes'], sha256=hash_by_file[filename])
            checked(out, row)
            retrieved[compiled] = row
        receipt.update(status='EXECUTED', outputs=retrieved)
        if len(outputs) == 1:
            receipt['output'] = retrieved[0]
    except BaseException as exc:
        receipt.update(status='EXECUTION_UNCERTAIN' if uncertain else 'FAILED', error=str(exc))
        raise
    finally:
        save_json(out/'result.json', receipt)
        if uncertain:
            print(f'HDC lost execution acknowledgement. Check {remote}/tool.exitcode. Board lock owner {token}', flush=True)
        elif held:
            _release_global(hdc, GLOBAL_LOCK, token)
    return receipt


def read_case(run, name, fixture):
    out = run/'v2'
    receipt = read_json(out/'result.json')
    case, = fixture['cases']
    if receipt['status'] != 'EXECUTED':
        raise ValueError(f"Cannot compare V2 output: {receipt['status']} in {out/'result.json'}. "
                         f"{receipt.get('error', '')}")
    if (receipt['case'] != name or case['case'] != name
            or receipt['fixture_sha256'] != sha(out/'fixture/manifest.json')
            or fixture['source_evidence']['compile_sha256'] != sha(run/'omc/compile.json')
            or receipt['omc_sha256'] != case['model']['sha256']):
        raise ValueError(f'V2 output belongs to a different model or fixture: {out}')
    operands = case['inputs']
    result, = case['outputs']
    hashes = receipt.get('input_hashes', [receipt.get('input_sha256')])
    if hashes != [s['fixture_sha256'] for s in operands]:
        raise ValueError('V2 inputs differ from the measured operands')
    inputs = {s['semantic_name']: (s['dtype'], s['shape'], s['fixture_sha256']) for s in operands}
    golden = read_array(out/'fixture'/name, dict(path='golden_0.bin', dtype=result['golden_dtype'],
                        shape=result['shape'], bytes=result['fixture_bytes'], sha256=result['fixture_sha256']))
    key = result['semantic_name']
    return inputs, {key: golden}, {key: read_array(out, dict(receipt['output'], dtype=result['dtype'], shape=result['shape']))}

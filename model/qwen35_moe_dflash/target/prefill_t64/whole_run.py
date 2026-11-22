"""Compile the original whole-model prefill AIR and run one HIAI V2 invocation."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import uuid

import numpy as np

from ...draft.ir_inspect import _enum_fields
from ...execution.hdc_run import Hdc
from ...execution.lock import _acquire_global, _release_global, _space
from ...execution.serial_transport import _environment
from ...toolchain.compile import _omg_ipc_environment, omg_environment, prepare_tbe_compatibility
from ...toolchain.omc_container import read_container, read_partition
from ...toolchain.omc_wire import load_schema
from ..layer_probe.batch import bind_compiled
from ..layer_probe.common import checked, identity, metrics, new_output, read_array, save_json, sha
from ..original_target_board import _board_path
from ..prefill_smoke_plan import _TOOL_TYPES

_GDR_PATHS = ('ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp',
              'tools/tools_ascendc/custom_op/ascend310b/src/chunk_gated_delta_rule.cpp')


def kernel_identity(ddk, source):
    expected = sha(source)
    for name in _GDR_PATHS:
        if sha(ddk / name) != expected:
            raise ValueError(f'GDR in {ddk / name} differs from {source}')
    return expected


def abi(model, ddk, meta):
    schema = load_schema(ddk / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    part, = [p for p in read_container(model)['partitions'] if p['type'] == 0]
    graph = schema.decode(read_partition(model, part, max_bytes=max(1, part['bytes'])), 'ModelDef')
    return bind_compiled(graph, meta['inputs'], meta['golden'], _enum_fields(schema),
                         {'stage_views': {'logits': [[248320], [1, 248320]]}})


def compile_model(run, ddk, source, diagnosis):
    air = run / 'air'
    manifest = (air / 'manifest.json').read_bytes()
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    meta = json.loads(manifest)
    if meta['kind'] != 'single-cpro-prefill40':
        raise ValueError('Expected the single 40-layer CPro AIR manifest')
    digest = kernel_identity(ddk, source)
    air_file = checked(air, meta['air'])
    if not meta['weights']:
        raise ValueError('Missing W4 weight inventory')
    def stamp(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns

    original_files = {air_file: stamp(air_file)}
    for row in meta['weights']:
        path = checked(air, row)
        original_files[path] = stamp(path)
    omg, env = omg_environment(ddk)
    out = new_output(run / 'omc')
    work = out / 'generated'
    work.mkdir()
    # OMG resolves FileConstant paths relative to its working directory.
    (work / 'weight').symlink_to(air / 'weight', target_is_directory=True)
    (work / 'input.air').symlink_to(air_file)
    if diagnosis is not None:
        env, _ = prepare_tbe_compatibility(ddk, diagnosis, out / 'tbe-compat', inherited=env)
    shapes = ';'.join(s['name'] + ':' + ','.join(map(str, s['shape'])) for s in meta['inputs'])
    argv = [str(omg), '--framework=1', '--model=input.air', '--output=model', '--target=omc',
            '--platform=kirinx90', '--use_origin_format=true', '--weight_data_type=FP16',
            '--limit_partition_size=false', '--save_weights_as_external_data=true', '--input_shape=' + shapes]
    save_json(out / 'command.json', dict(argv=argv, cwd=str(work), gdr_source_sha256=digest))
    size = sum(row['bytes'] for row in meta['weights'])
    print(f'OMG starts. AIR weights {size / 2**30:.1f} GiB, free {shutil.disk_usage(work).free / 2**30:.1f} GiB. '
          f'Log: {out / "compile.log"}', flush=True)
    with _omg_ipc_environment(work, env) as (child, _), (out / 'compile.log').open('x') as log:
        subprocess.run(argv, cwd=work, env=child, stdout=log, stderr=subprocess.STDOUT, check=True)
    if kernel_identity(ddk, source) != digest:
        raise ValueError('GDR changed during compilation')
    if sha(air / 'manifest.json') != manifest_sha or any(stamp(p) != original for p, original in original_files.items()):
        raise ValueError('Compiler input bundle changed during compilation')
    models = sorted(work.rglob('*.omc'))
    if len(models) != 1:
        raise ValueError(f'Expected one compiled OMC, found {models}')
    model, = models
    weights = sorted(model.parent.rglob('*.weight'))
    if not weights:
        raise ValueError(f'OMG did not produce external .weight files alongside {model}')
    io = abi(model, ddk, meta)
    save_json(out / 'manifest.json', dict(air_manifest_sha256=manifest_sha,
              gdr_source_sha256=digest, primary=identity(model, out),
              weights=[identity(w, out) for w in weights], io=io))
    print(f'{model} with {len(weights)} external weight files', flush=True)


def score(air, meta, output, spec):
    ref = read_array(air, meta['golden']).reshape(-1)
    if output.stat().st_size != spec['bytes']:
        raise ValueError('Logits byte count differs from compiled ABI')
    actual = np.fromfile(output, dtype=spec['dtype'])
    if actual.size != ref.size or not (np.isfinite(actual).all() and np.isfinite(ref).all()):
        raise ValueError('Expected one finite full-vocabulary logits row from each backend')
    token = int(np.argmax(actual))
    reference = meta['reference_top1']
    vocab = json.loads(checked(air, meta['token_text_map']).read_text())
    if (len(vocab['pieces']) != ref.size or not 0 <= reference < ref.size or ref[reference] != ref.max()
            or vocab['pieces'][reference] != meta['reference_text']):
        raise ValueError('Tokenizer or reference token differs from the reference logits')
    return dict(prompt=vocab['prompt'], output_pos=18, valid_length=19,
                cpro_token=token, cpro_text=vocab['pieces'][token], cpro_special=token in vocab['special_ids'],
                reference_token=reference, reference_text=meta['reference_text'],
                top1_match=token == reference, logits=metrics(actual, ref))


def board(run, tool_path):
    air, omc = run / 'air', run / 'omc'
    compiled = json.loads((omc / 'manifest.json').read_text())
    if sha(air / 'manifest.json') != compiled['air_manifest_sha256']:
        raise ValueError('OMC and AIR manifest differ')
    meta = json.loads((air / 'manifest.json').read_text())
    tool = _board_path(tool_path)
    primary = checked(omc, compiled['primary'])
    files = [('payload/' + primary.name, primary, compiled['primary'])]
    for row in compiled['weights']:
        path = checked(omc, row)
        files.append(('payload/' + path.relative_to(primary.parent).as_posix(), path, row))
    inputs = compiled['io']['inputs']
    result_spec, = compiled['io']['outputs']
    for row in inputs:
        files.append((f"inputs/{row['index']:03d}.bin", checked(air, row), row))
    # Private attempt paths and a host-side script keep 80 cache names off HDC's shell command line.
    env = _environment()
    out = new_output(run / 'board')
    token = uuid.uuid4().hex
    remote = PurePosixPath('/data/local/tmp') / ('qwen35_prefill40_' + token)
    lock = remote.parent / '.qwen35-target-model-io-lock'
    hdc = Hdc(*env, out, 7200)
    save_json(out / 'attempt.json', dict(device=env[1], server=env[2], remote=str(remote),
              omc_sha256=compiled['primary']['sha256'], gdr_source_sha256=compiled['gdr_source_sha256']))
    _, space = hdc.shell(f'test -x {tool} && df -Pk {remote.parent}', 'preflight')
    needed = sum(row['bytes'] for _, _, row in files) + result_spec['bytes']
    if _space(space) < needed:
        raise OSError(f'Board requires at least {needed:,} bytes for model, inputs and logits')
    argv = [str(tool), '--model=' + str(remote / files[0][0]), '--weight_dir=' + str(remote / 'payload'),
            '--input=' + ','.join(str(remote / f"inputs/{s['index']:03d}.bin") for s in inputs),
            '--input_type=' + ','.join(_TOOL_TYPES[s['dtype'].upper()] for s in inputs),
            '--output_type=' + _TOOL_TYPES[result_spec['dtype'].upper()], '--times=1', '--modelmanager_itf=V2',
            '--direct_from_file=true', '--enable_item=1', '--output_dir=' + str(remote / 'outputs')]
    script = out / 'run.sh'
    script.write_text(f'cd {remote} || exit 1\nLD_LIBRARY_PATH={tool.parent} ' + shlex.join(argv)
                      + ' >tool.stdout 2>tool.stderr\nrc=$?\necho "$rc" > tool.exitcode\n'
                      + 'tail -n 10 tool.stdout tool.stderr\nexit "$rc"\n')
    checks = out / 'files.sha256'
    checks.write_text(''.join(row['sha256'] + '  ' + rel + '\n' for rel, _, row in files))
    _acquire_global(hdc, lock, token, 120)
    uncertain = False
    try:
        dirs = {remote / 'outputs', *((remote / rel).parent for rel, _, _ in files)}
        hdc.shell(f'mkdir {remote} && mkdir -p ' + shlex.join(sorted(map(str, dirs))), 'create-board-dir')
        print(f'Uploading model and inputs ({needed / 2**30:.1f} GiB) to {remote}', flush=True)
        for rel, path, _ in files:
            hdc.send(path, remote / rel)
        hdc.send(script, remote / 'run.sh')
        hdc.send(checks, remote / 'files.sha256')
        hdc.shell(f'cd {remote} && sha256sum -c files.sha256', 'verify-upload')
        print('Running one 40-layer HIAI V2 prefill', flush=True)
        uncertain = True
        # Use the exit record even for a tool failure. Some HDC clients turn a
        # nonzero remote status into a transport exception before returning its log.
        _, log = hdc.shell(f'cd {remote} && {{ sh run.sh; test -s tool.exitcode; }}', 'execute')
        _, code = hdc.shell(f'cat {remote}/tool.exitcode', 'exitcode')
        rc = int(code.strip())
        uncertain = False
        if rc:
            raise RuntimeError(f'HIAI exited {rc}; attempt {remote}\n{log}')
        _, listing = hdc.shell(f'cd {remote}/outputs && find . -type f -size {result_spec["bytes"]}c '
                               + '-exec sha256sum {} \\;', 'list-logits')
        matches = re.findall(r'^([a-f0-9]{64})\s+(\./[^\r\n]+)', listing, re.M)
        if len(matches) != 1:
            raise ValueError(f'Expected one {result_spec["bytes"]}-byte output in {remote}/outputs: {listing}')
        digest, name = matches[0]
        remote_file = _board_path(str(remote / 'outputs' / name))
        output = out / 'logits.bin'
        hdc.receive(remote_file, output)
        if sha(output) != digest:
            raise ValueError('HDC logits digest differs from device')
        result = score(air, meta, output, result_spec)
        save_json(out / 'first-token.json', result)
        print(f"310P {result['reference_token']} {result['reference_text']!r}; "
              f"CPro {result['cpro_token']} {result['cpro_text']!r}; top1_match={result['top1_match']}", flush=True)
        return result
    finally:
        if not uncertain:
            _release_global(hdc, lock, token)
        else:
            print(f'HDC lost execution acknowledgement. Check {remote}/tool.exitcode before releasing {lock}.',
                  flush=True)

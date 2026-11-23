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
from . import whole_trace

_PROBE = 'decoder-tail-projection-probe'
_DECODER_KINDS = ('single-cpro-decoder', 'single-cpro-decoder-tail')

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
    if meta['kind'] == _PROBE:
        from .decoder_projection import probe_abi
        return probe_abi(graph, _enum_fields(schema), meta)
    output = meta['golden']
    width = output['shape'][-1]
    contract = meta.get('output_contract', {'stage_views': {output['name']: [[width], [1, width]]}})
    if 'trace_layers' in meta:
        contract['required_dtypes'] = {'0': 'float32'}
    io = bind_compiled(graph, meta['inputs'], output, _enum_fields(schema), contract)
    if meta['kind'] in _DECODER_KINDS:
        from collections import Counter
        from ...toolchain.omc_cpu_inspect import _attr
        nodes = graph['graph'][0]['op']
        owners = Counter(_attr(op, 'cl_name').get('s', 'UNKNOWN') for op in nodes if op['type'] == 'GraphOp')
        io['graphop_counts'] = dict(owners)
        print(f"Compiled L{meta['layer']} decoder GraphOp placement: {dict(owners)}", flush=True)
    return io


def shared_assets(air, meta, baseline, digest):
    if 'trace_layers' in meta:
        whole_trace.check_baseline(meta, baseline)
        whole_trace.baseline_logits(baseline, digest)
        return baseline / 'air'
    if baseline is not None:
        raise ValueError('--baseline requires a diagnostic AIR with decoder readouts')
    return air


def compile_model(run, ddk, source, diagnosis, baseline=None):
    air = run / 'air'
    manifest = (air / 'manifest.json').read_bytes()
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    meta = json.loads(manifest)
    if meta['kind'] not in ('single-cpro-prefill40', _PROBE, *_DECODER_KINDS):
        raise ValueError('Expected a whole prefill or independent decoder AIR manifest')
    digest = kernel_identity(ddk, source)
    assets = shared_assets(air, meta, baseline, digest)
    air_file = checked(air, meta['air'])
    if not meta['weights'] and meta['kind'] != _PROBE:
        raise ValueError('Missing W4 weight inventory')
    def stamp(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns

    original_files = {air_file: stamp(air_file)}
    for row in meta['weights']:
        path = checked(assets, row)
        original_files[path] = stamp(path)
    omg, env = omg_environment(ddk)
    out = new_output(run / 'omc')
    work = out / 'generated'
    work.mkdir()
    # OMG resolves FileConstant paths relative to its working directory.
    (work / 'weight').symlink_to(assets / 'weight', target_is_directory=True)
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
    if not weights and meta['kind'] != _PROBE:
        raise ValueError(f'OMG did not produce external .weight files alongside {model}')
    io = abi(model, ddk, meta)
    save_json(out / 'manifest.json', dict(air_manifest_sha256=manifest_sha,
              gdr_source_sha256=digest, primary=identity(model, out),
              weights=[identity(w, out) for w in weights], io=io))
    print(f'{model} with {len(weights)} external weight files', flush=True)


def score(air, meta, output, spec, *, assets=None, original=None):
    ref = read_array(air, meta['golden']).reshape(-1)
    if output.stat().st_size != spec['bytes']:
        raise ValueError('Logits byte count differs from compiled ABI')
    actual = np.fromfile(output, dtype=spec['dtype'])
    trace = None
    if 'trace_layers' in meta:
        trace = whole_trace.compare_rows(actual, ref, np.fromfile(original, dtype='<f4'))
        actual, ref = actual[:whole_trace.VOCAB], ref[:whole_trace.VOCAB]
        base = json.loads((assets / 'manifest.json').read_text())
        if not np.array_equal(ref.view('uint32'), read_array(assets, base['golden']).reshape(-1).view('uint32')):
            raise ValueError('Diagnostic reference logits differ from the original 310P capture')
        actual.tofile(output.parent / 'logits.bin')
    if actual.size != ref.size or not (np.isfinite(actual).all() and np.isfinite(ref).all()):
        raise ValueError('Expected one finite full-vocabulary logits row from each backend')
    token = int(np.argmax(actual))
    reference = meta['reference_top1']
    vocab = json.loads(checked(assets or air, meta['token_text_map']).read_text())
    if (len(vocab['pieces']) != ref.size or not 0 <= reference < ref.size or ref[reference] != ref.max()
            or vocab['pieces'][reference] != meta['reference_text']):
        raise ValueError('Tokenizer or reference token differs from the reference logits')
    result = dict(prompt=vocab['prompt'], output_pos=18, valid_length=19,
                cpro_token=token, cpro_text=vocab['pieces'][token], cpro_special=token in vocab['special_ids'],
                reference_token=reference, reference_text=meta['reference_text'],
                top1_match=token == reference, logits=metrics(actual, ref))
    if trace is not None:
        result['decoder_checkpoints'] = trace
    return result


def board(run, tool_path, baseline=None):
    air, omc = run / 'air', run / 'omc'
    compiled = json.loads((omc / 'manifest.json').read_text())
    if sha(air / 'manifest.json') != compiled['air_manifest_sha256']:
        raise ValueError('OMC and AIR manifest differ')
    meta = json.loads((air / 'manifest.json').read_text())
    assets = shared_assets(air, meta, baseline, compiled['gdr_source_sha256'])
    tool = _board_path(tool_path)
    primary = checked(omc, compiled['primary'])
    files = [('payload/' + primary.name, primary, compiled['primary'])]
    for row in compiled['weights']:
        path = checked(omc, row)
        files.append(('payload/' + path.relative_to(primary.parent).as_posix(), path, row))
    inputs = compiled['io']['inputs']
    result_spec, = compiled['io']['outputs']
    for row in inputs:
        files.append((f"inputs/{row['index']:03d}.bin", checked(assets, row), row))
    # Private attempt paths and a host-side script keep 80 cache names off HDC's shell command line.
    env = _environment()
    out = new_output(run / 'board')
    token = uuid.uuid4().hex
    remote = PurePosixPath('/data/local/tmp') / ('qwen35_prefill40_' + token)
    lock = remote.parent / '.qwen35-target-model-io-lock'
    hdc = Hdc(*env, out, 7200)
    save_json(out / 'attempt.json', dict(device=env[1], server=env[2], remote=str(remote),
              omc_sha256=compiled['primary']['sha256'], gdr_source_sha256=compiled['gdr_source_sha256']))
    reusable = {}
    if baseline:
        old = json.loads((baseline / 'board/attempt.json').read_text())
        if (old['device'], old['server']) != tuple(env[1:]):
            raise ValueError('Trace and logits-only run must use the same board and HDC server')
        old_remote = _board_path(old['remote'])
        for line in (baseline / 'board/files.sha256').read_text().splitlines():
            digest, rel = line.split('  ', 1)
            if rel.startswith('payload/') and rel.endswith('.weight'):
                reusable[digest] = _board_path(str(old_remote) + '/' + rel)
    argv = [str(tool), '--model=' + str(remote / files[0][0]), '--weight_dir=' + str(remote / 'payload'),
            '--input=' + ','.join(str(remote / f"inputs/{s['index']:03d}.bin") for s in inputs),
            '--input_type=' + ','.join(_TOOL_TYPES[s['dtype'].upper()] for s in inputs),
            '--output_type=' + _TOOL_TYPES[result_spec['dtype'].upper()], '--times=1', '--modelmanager_itf=V2',
            '--direct_from_file=true', '--enable_item=1', '--output_dir=' + str(remote / 'outputs')]
    script = out / 'run.sh'
    report_logs = ('if [ "$rc" != 0 ]; then cat tool.stdout tool.stderr; else tail -n 10 tool.stdout tool.stderr; fi'
                   if meta['kind'] in (*_DECODER_KINDS, _PROBE) else 'tail -n 10 tool.stdout tool.stderr')
    script.write_text(f'cd {remote} || exit 1\ntouch tool.started || exit 1\nLD_LIBRARY_PATH={tool.parent} ' + shlex.join(argv)
                      + ' >tool.stdout 2>tool.stderr\nrc=$?\necho "$rc" > tool.exitcode\n'
                      + report_logs + '\nexit "$rc"\n')
    checks = out / 'files.sha256'
    checks.write_text(''.join(row['sha256'] + '  ' + rel + '\n' for rel, _, row in files))
    _acquire_global(hdc, lock, token, 120)
    uncertain = False
    try:
        dirs = {remote / 'outputs', *((remote / rel).parent for rel, _, _ in files)}
        hdc.shell(f'mkdir {remote} && mkdir -p ' + shlex.join(sorted(map(str, dirs))), 'create-board-dir')
        send = []
        for rel, path, row in files:
            if rel.endswith('.weight') and row['sha256'] in reusable:
                link = shlex.join(['ln', str(reusable[row['sha256']]), str(remote / rel)])
                _, outcome = hdc.shell(f'if {link} 2>/dev/null; then echo LINKED; else echo UPLOAD; fi', 'reuse-weight')
                if outcome == 'LINKED':
                    continue
            send.append((rel, path, row))
        needed = sum(row['bytes'] for _, _, row in send) + result_spec['bytes']
        _, space = hdc.shell(f'test -x {tool} && df -Pk {remote.parent}', 'preflight')
        if _space(space) < needed:
            raise OSError(f'Board requires at least {needed:,} additional bytes for this run')
        print(f'Uploading {len(send)} files ({(needed-result_spec["bytes"]) / 2**30:.2f} GiB) to {remote}', flush=True)
        for rel, path, _ in send:
            hdc.send(path, remote / rel)
        hdc.send(script, remote / 'run.sh')
        hdc.send(checks, remote / 'files.sha256')
        hdc.shell(f'cd {remote} && sha256sum -c files.sha256', 'verify-upload')
        decoder = meta['kind'] in _DECODER_KINDS
        label = f"L{meta['layer']} standalone decoder" if decoder else '40-layer'
        if meta['kind'] == 'single-cpro-decoder-tail':
            label = f"L{meta['layer']} post-attention decoder tail"
        if meta['kind'] == _PROBE:
            label = f"L{meta['layer']} {meta['output_edge']} probe"
        print(f'Running one {label} HIAI V2 prefill', flush=True)
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
        output = out / ('hidden.bin' if decoder else 'logits-and-hidden.bin' if baseline else 'logits.bin')
        if meta['kind'] == _PROBE:
            output = out / 'output.bin'
        hdc.receive(remote_file, output)
        if sha(output) != digest:
            raise ValueError('HDC output digest differs from device')
        if meta['kind'] == _PROBE:
            result = dict(status='EXECUTED', output=dict(identity(output, out), dtype=result_spec['dtype'],
                          shape=result_spec['shape']), numerical_validation='NOT_RUN')
            save_json(out / 'result.json', result)
            return result
        if decoder:
            from .decoder_pair import score_hidden
            result = score_hidden(air, meta, output, result_spec)
            if meta['kind'] == 'single-cpro-decoder-tail':
                result['scope'] = 'post_attention_layernorm input to final decoder hidden'
            save_json(out / 'accuracy.json', result)
            return result
        result = score(air, meta, output, result_spec, assets=assets,
                       original=baseline / 'board/logits.bin' if baseline else None)
        save_json(out / 'first-token.json', result)
        if baseline:
            trace = result['decoder_checkpoints']
            print(f"Original CPro logits equal bits: {trace['baseline_logits_equal_bits']}  "
                  f"({trace['changed_logits']}/{trace['total_logits']} changed)", flush=True)
            if not trace['baseline_logits_equal_bits']:
                print('Checkpoint graph changed the final logits. Results apply to this diagnostic graph only.')
            for layer, m in trace['layers'].items():
                print(f"{layer:3s} token18  L2={m['relative_l2']}  cosine={m['cosine']}  "
                      f"RMSE={m['rmse']}  error_ratio={m['error_ratio']}  failed={m['failed_metrics']}")
        print(f"310P {result['reference_token']} {result['reference_text']!r}; "
              f"CPro {result['cpro_token']} {result['cpro_text']!r}; top1_match={result['top1_match']}", flush=True)
        return result
    finally:
        if not uncertain:
            _release_global(hdc, lock, token)
        else:
            print(f'HDC lost execution acknowledgement. Check {remote}/tool.exitcode before releasing {lock}.',
                  flush=True)

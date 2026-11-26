"""Compile, execute, resume and compare whole-decoder OMCs on the OMG host."""
import json
import os
from pathlib import Path
import tempfile
import shutil

import numpy as np

from decoder4_original import EXPORT_DIR, compile_log_text, compile_native_prefill, compile_omg
from decoder4_prefill import pull_air, ssh_file
from l0_silu_omcs import write_matrices
from parse_model_json import GE_PROTO, inspect, named_shapes, read_omc
from qwen35_moe_dflash.execution.serial_transport import _environment
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, metrics, read_tensor, save_json, sha, sized_file
from qwen35_moe_dflash.target.prefill_t64.attention_step_v2 import run_v2
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import (
    case_name, comparison_view, cpro_input_records, directory_name, groups, preserved_cache,
)
from qwen35_moe_dflash.target.prefill_t64.decoder_four_boundary import view


def read(path):
    return json.loads(path.read_bytes())


def native_tool_args(case):
    # Older four-layer bundles selected the owner's OMG/V2 flags through the ConvState profile.
    return case.get('native_tool_args', case.get('native_convstate', case.get('gdn_conv') == 'original'))


def transfer_size(meta, dest):
    total = sum(row['bytes'] for row in meta['files'])
    free = shutil.disk_usage(dest).free
    print(f'Package {total / 2**30:.2f} GiB, available {free / 2**30:.2f} GiB before OMG', flush=True)
    if total > free:
        raise OSError(f'{dest}: insufficient space for the {total:,}-byte transfer')


def transfer(client, remote, dest):
    dest.mkdir()
    remote = Path(remote)
    ssh_file(client, remote/'manifest.json', dest/'manifest.json')
    meta = read(dest/'manifest.json')
    transfer_size(meta, dest)
    for row in meta['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Invalid artifact path {rel}')
        (dest/rel).parent.mkdir(parents=True, exist_ok=True)
        ssh_file(client, remote/rel, dest/rel, row)


def compiled(case, base, ddk, diagnosis, bundle_stamp):
    dest = base/'omc'
    stamp = dict(bundle=bundle_stamp)
    if ddk is not None:
        stamp['omg'] = sha(ddk/'tools/tools_omg/master/omg')
    if not dest.exists():
        if ddk is None:
            raise FileNotFoundError(f'Run compile and provide its output to execute: {dest}')
        meta = read(case/'case.json')
        folder = meta.get('air_directory', 'weight')
        if folder not in ('weight', EXPORT_DIR):
            raise ValueError(f'{case}: unexpected AIR directory {folder}')
        graph_size = sum(r['bytes'] for r in meta['files'] if r['path'].startswith(folder + '/'))
        print(f'{case.name}: AIR with weights {graph_size / 2**30:.2f} GiB, '
              f'available before OMG {shutil.disk_usage(base).free / 2**30:.2f} GiB', flush=True)
        work = base/'compile'
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        (work/folder).symlink_to(case/folder, target_is_directory=True)
        if native_tool_args(meta):
            code, argv, _ = compile_native_prefill(work, f'{folder}/export.air', ddk)
        else:
            code, argv, _ = compile_omg(work, f'{folder}/export.air', ddk, diagnosis, None)
        if code:
            raise RuntimeError(compile_log_text(work/'compile.log'))
        model, = work.rglob('*.omc')
        io = inspect(read_omc(model, ddk_root=ddk)[0], enum=GE_PROTO)
        data = dict(stamp=stamp, argv=argv, model=identity(model, work), output_shape=io['output_shape'],
                    weights=[identity(p, work) for p in sorted(model.parent.rglob('*.weight'))])
        for direction in ('inputs', 'outputs'):
            data[direction] = []
            for row in io[direction]:
                if row['slot'] != len(data[direction]) or row['issues'] or not row['logical_bytes']:
                    raise ValueError(f'Invalid compiled descriptor {row}')
                dtype = {'FP16': 'float16', 'FP32': 'float32'}.get(row['dtype'], row['dtype'].lower())
                data[direction].append(dict(name=row['node' if direction == 'inputs' else 'edge'],
                                            shape=row['shape'], dtype=dtype, bytes=row['logical_bytes']))
        save_json(work/'build.json', data)
        work.rename(dest)
    build = read(dest/'build.json')
    if any(build['stamp'].get(k) != v for k, v in stamp.items()):
        raise ValueError(f'{dest}: bundle or OMG changed. Run board again without --recovery.')
    model = sized_file(dest, build['model'])
    if 'output_shape' not in build:
        # Older build.json kept NetOutput edges but omitted the tool's output names.
        local_ddk = ddk or os.getenv('CPRO_DDK_ROOT') or os.getenv('DDK_PATH')
        if local_ddk and (Path(local_ddk)/'tools/tools_omg/master/omg').is_file():
            print(f'{case.name}: 从已有 OMC 补读输出名称和 shape', flush=True)
            try:
                io = inspect(read_omc(model, ddk_root=local_ddk)[0], enum=GE_PROTO)
                build['output_shape'] = io['output_shape']
            except (OSError, ValueError) as exc:
                print(f'{case.name}: OMG 无法补读端口，将从板端 OMC 读取：{exc}', flush=True)
    return build, model, [sized_file(dest, r) for r in build['weights']]


def output_order(descriptors, source, golden):
    if len(descriptors) != len(source):
        raise ValueError('Compiled output count differs from AIR')
    # Use surviving AIR edges when present. GraphOp outputs otherwise retain the public slot.
    names = [None] * len(source)
    by_edge = {row['edge']: row['name'] for row in source}
    for i, spec in enumerate(descriptors):
        names[i] = by_edge.get(spec['name'])
    for i, spec in enumerate(descriptors):
        if names[i] is None:
            candidates = [r['name'] for r in source if r['name'] not in names
                          and view(golden[r['name']]['shape']) == view(spec['shape'])]
            if len(candidates) == 1:
                names[i] = candidates[0]
            elif source[i]['name'] in candidates:
                names[i] = source[i]['name']
            else:
                raise ValueError(f'Cannot bind compiled output {i} {spec} to the AIR slots {candidates}')
    if sorted(names) != sorted(by_edge.values()):
        raise ValueError('Conflicting compiled output names and public AIR order')
    return names


def replay(base, case, build, model, weights, feeds, golden, runtime, stamp, recovery, board_case=None,
           replace_inputs=False, replace_weights=False, shape_config=None, original_board_args=False):
    meta = read(case/'case.json')
    original_board_args = original_board_args or native_tool_args(meta)
    if 'operator_counts' in meta:
        counts = meta['operator_counts']
        print(f'[{case.name}] 本次 AIR 配置：GDN 卷积={meta["gdn_conv"]}，SiluPatch={counts["SiluPatch"]}，'
              f'SoftplusPatch={counts["SoftplusPatch"]}', flush=True)
        if meta.get('gdn_silu_site'):
            print(f'[{case.name}] --gdn-silu-site {meta["gdn_silu_site"]}', flush=True)
        if meta.get('all_silu_no_gate_cast'):
            print(f'[{case.name}] --all-silu-no-gate-cast：GDN 和 MoE 全部使用 SiluPatch，GDN 门控源码省略输出 Cast。', flush=True)
        if meta.get('rms_silu_layers'):
            print(f'[{case.name}] RMS 门控 SiluPatch 层={meta["rms_silu_layers"]}', flush=True)
        if meta.get('physical_convstate'):
            print(f'[{case.name}] ConvState 使用本次 QKV 的物理行 60–63，卷积窗口为 67 行。', flush=True)
    if {r['name'] for r in build['inputs']} != set(meta['inputs']):
        raise ValueError(f'{case}: compiled input names differ from AIR')
    feeds = {**feeds, **{role: (case, row) for role, row in meta.get('input_files', {}).items()}}
    operands, paths = [], []
    for spec in build['inputs']:
        role = meta['inputs'][spec['name']]
        root, row = feeds[role]
        if (row['dtype'], view(row['shape']), row['bytes']) != (spec['dtype'], view(spec['shape']), spec['bytes']):
            raise ValueError(f'{case.name}: {role} compiled input differs from feed: {spec} vs {row}')
        paths.append(sized_file(root, row))
        operands.append(dict(spec, sha256=row['sha256']))
    if meta.get('native_activations') or meta.get('silu_only') or meta.get('softplus_only'):
        from decoder_four_inputs import compare_original_inputs
        compare_original_inputs(base.parent.parent, operands, meta['inputs'], [s['sha256'] for s in operands],
                                baseline='no_patch' if meta.get('native_activations') else 'native_activations',
                                gdn_conv=meta['gdn_conv'])
    names = output_order(build['outputs'], meta['outputs'], golden)
    for role, spec in zip(names, build['outputs'], strict=True):
        ref = golden[role]
        if view(ref['shape']) != view(spec['shape']) or spec['dtype'] not in ('float16', 'float32'):
            raise ValueError(f'{case.name}: invalid compiled {role} descriptor {spec}')
    _, serial, server = _environment()
    key = dict(bundle=stamp, model=build['model']['sha256'], weights=[w['sha256'] for w in build['weights']],
               inputs=[r['sha256'] for r in operands], runtime=runtime, serial=serial, server=server)
    saved = base/'completed.json'
    if original_board_args:
        key['original_board_args'] = True
    if recovery and saved.exists() and not (replace_inputs or replace_weights):
        record = read(saved)
        if record['binding'] != key:
            raise ValueError(f'{base}: saved result has different inputs, runtime, board or weights')
        folder = base/record['replay']
        result = read(checked(folder, record['receipt']))
        if result['status'] != 'EXECUTED' or result['exitcode'] != 0:
            raise ValueError(f'{folder}: incomplete replay')
        print(f'{case.name}: resumed verified board outputs', flush=True)
    else:
        folder = base/'replay'
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        try:
            result = run_v2(folder, case.name, model, operands, paths, build['outputs'], runtime,
                            decoder_shapes=(named_shapes(operands), build.get('output_shape')), shape_config=shape_config,
                            original_board_args=original_board_args,
                            fixture_sha256=stamp, external_weights=weights, build_records=(build['model'], build['weights']),
                            board_case=board_case, hdc_timeout=7200, replace_inputs=replace_inputs, replace_weights=replace_weights)
        except RuntimeError:
            try:
                from split_stage_log import diagnose
                receipt = folder/'result.json'
                if receipt.is_file() and read(receipt).get('exitcode'):
                    diagnose([folder])
            except Exception as exc:
                print(f'{case.name}: 无法读取本次板端失败日志：{exc}', flush=True)
            raise
        save_json(saved, dict(binding=key, replay=folder.name, receipt=identity(folder/'result.json', folder)))
    values = {}
    for role, spec, row in zip(names, build['outputs'], result['outputs'], strict=True):
        sized_file(folder, row)
        values[role] = (folder, dict(row, dtype=spec['dtype'], shape=golden[role]['shape']))
    return values


def board(run, mode, phase, remote, ddk, config, diagnosis, recovery, *, gdn_conv='conv2d', step='board',
          replace_inputs=False, replace_weights=False, original_board_args=False, no_patch=False,
          native_activations=False, silu_only=False, softplus_only=False, gdn_silu_only=False, gdn_silu_site=False,
          all_silu_no_gate_cast=False, physical_convstate=False, rms_silu_layers=False):
    bundle = run/'package'
    if step == 'execute' and not bundle.is_dir():
        raise FileNotFoundError(f'Provide the package and compiled OMCs from the compile stage: {bundle}')
    if step != 'execute' and recovery and bundle.is_dir() and not bundle.is_symlink():
        with tempfile.TemporaryDirectory(dir=run) as tmp:
            pull_air(Path(tmp), remote/'package/manifest.json', destination_name='manifest.json',
                     ssh_download=ssh_file)
            if sha(Path(tmp)/'manifest.json') != sha(bundle/'manifest.json'):
                raise ValueError('The AIR server has a new export. Run board without --recovery to use its inputs and model.')
    if not bundle.exists():
        pull_air(run, remote/'package', destination_name='package', ssh_download=transfer, before_download=transfer_size)
    manifest = read(bundle/'manifest.json')
    if (manifest['mode'], manifest['phase']) != (mode, phase):
        raise ValueError(f'{bundle}: mode/phase mismatch')
    if manifest.get('gdn_conv', 'conv2d') != gdn_conv:
        raise ValueError(f'{bundle}: expected GDN {gdn_conv}. Select the same --gdn-conv as the AIR export.')
    for key, selected in dict(no_patch=no_patch, native_activations=native_activations,
                              silu_only=silu_only, softplus_only=softplus_only, gdn_silu_only=gdn_silu_only,
                              gdn_silu_site=gdn_silu_site, all_silu_no_gate_cast=all_silu_no_gate_cast,
                              physical_convstate=physical_convstate, rms_silu_layers=rms_silu_layers).items():
        if manifest.get(key, False) != selected:
            raise ValueError(f'{bundle}: --{key.replace("_", "-")} selection differs from the AIR export')
    from decoder_handoff import runtime_file
    for row in manifest['files']:
        if step != 'execute' or runtime_file(row['path']):
            sized_file(bundle, row)
    fixture = bundle/'fixture'
    ref = read(fixture/'reference.json')
    length = ref['prompt_length']
    stamp = sha(bundle/'manifest.json')
    if step == 'compile':
        for group in groups(mode):
            case = bundle/case_name(group)
            base = run/case.name
            base.mkdir(exist_ok=True)
            compiled(case, base, ddk, diagnosis, stamp)
        return True
    runtime = read(config.parent/read(config)['assets'])['environment']
    pieces = read(fixture/'tokens.json')
    feeds = {k: (fixture, v) for k, v in cpro_input_records(ref).items()}
    scores = {b: {r: {} for r in ('cpu', '310p', 'cpro')} for b in ('cpu', '310p', 'cpro')}
    failures, unchanged, predictions = [], {}, {}
    for group in groups(mode):
        name = case_name(group)
        case = bundle/name
        base = run/name
        base.mkdir(parents=True, exist_ok=True)
        progress = dict(mode=mode, phase=phase, bundle_sha256=stamp, layer=group[0], case=name)
        save_json(run/'comparison.json', dict(status='RUNNING', **progress))
        try:
            build, model, weights = compiled(case, base, ddk, diagnosis, stamp)
            board_case = directory_name(mode, phase)
            if mode == 'layerwise':
                board_case += '-' + case.name
            values = replay(base, case, build, model, weights, feeds, ref['outputs'], runtime, stamp, recovery,
                            board_case, replace_inputs, replace_weights, shape_config=config,
                            original_board_args=original_board_args or gdn_conv == 'original')
        except Exception as exc:
            save_json(run/'comparison.json', dict(status='INTERRUPTED', message=str(exc), **progress))
            raise
        if mode == 'layerwise':
            feeds['hidden'] = values[f'hidden_{group[-1]}']
        for role, (folder, row) in values.items():
            sources = {'cpu': (fixture, ref['cpu'][role]), '310p': (fixture, ref['outputs'][role]), 'cpro': (folder, row)}
            arrays = {key: read_tensor(sized_file(root, spec), spec) for key, (root, spec) in sources.items()}
            if role.startswith(('key_cache_', 'value_cache_')):
                initial = ref['inputs'][role]
                before = read_tensor(sized_file(fixture, initial), initial)
                unchanged[role] = {b: preserved_cache(x, before, phase, length) for b, x in arrays.items()}
                if not all(unchanged[role].values()):
                    failures.append(role + '/unwritten_slots')
            if role == 'logits':
                for backend, logits in arrays.items():
                    token = int(logits.argmax())
                    predictions[backend] = dict(id=token, text=pieces[token]) if np.isfinite(logits).all() else None
            for b, x in arrays.items():
                scope, actual = comparison_view(role, x, phase, length)
                for r, y in arrays.items():
                    scores[b][r][role] = {scope: metrics(actual, comparison_view(role, y, phase, length)[1])}
            if not scores['cpro']['310p'][role][scope]['passed']:
                failures.append(role)
        # A later failed layer still leaves the completed layer comparisons accessible.
        write_matrices(base/'matrices.csv', {b: {r: {k: v for k, v in s.items() if k in values}
                                                for r, s in row.items()} for b, row in scores.items()}, display=False)
    print(f'[{mode}/{phase}] 汇总 CPU、310P、CPro 的 3×3 精度矩阵 → {run}', flush=True)
    write_matrices(run/'matrices.csv', scores)
    status = 'FAILED_ACCURACY' if failures else 'PASSED'
    save_json(run/'comparison.json', dict(mode=mode, phase=phase, prompt=ref['prompt'], prompt_length=length,
                                          logits_position=ref['logits_position'], bundle_sha256=stamp, failed=failures,
                                          unchanged_kv=unchanged, top1=predictions,
                                          status=status))
    print(f'Top1 {predictions}', flush=True)
    if failures:
        print('CPro 对 310P 比较或缓存写入检查未通过：' + ', '.join(failures), flush=True)
    print(f'{status} {run / "comparison.json"}', flush=True)
    return not failures

"""Recompile the failed L0 V2 decoder with OMG external weights on the same inputs."""
from pathlib import Path
import re
import shutil
import tempfile

from ...common.artifacts import output_path
from ...draft.ir_inspect import _enum_fields
from ...toolchain.compiler_entry import DEFAULT_PROFILE
from ...toolchain.existing_air import compile_existing_air, select_primary_omc
from ...toolchain.omc_container import read_container, read_partition
from ...toolchain.omc_embed_weights import _copy, _weight_plan
from ...toolchain.omc_wire import load_schema
from ..layer_probe.batch import bind_compiled
from ..layer_probe.common import checked, new_output, save_json, sha
from .chain_board import load
from .chain_compile import _payloads, _pull
from .chain_reference import installed_vscale


def _slots(specs, fields):
    return sorted(tuple(s[k] for k in fields) for s in specs)


def _compare_payloads(embedded, external, parts, weights):
    old = {p['type']: p for p in read_container(embedded)['partitions']}
    with embedded.open('rb') as a, external.open('rb') as b:
        a.seek(old[2]['offset'])
        b.seek(parts[2]['offset'])
        same_program = _copy(a, old[2]['bytes']) == _copy(b, parts[2]['bytes'])
        a.seek(old[1]['offset'])
        same_weights = (sum(w['bytes'] for w in weights) == old[1]['bytes']
                        and all(_copy(a, w['bytes']) == w['sha256'] for w in weights))
    print(f'Embedded/external byte comparison: target_partition_equal={same_program}, '
          f'weight_bytes_equal={same_weights}', flush=True)
    return dict(target_partition_equal=same_program, weight_bytes_equal=same_weights)


def compile_external(replay_dir, remote_air, output):
    replay_dir = output_path(replay_dir)
    case_dir = replay_dir.parent.parent
    name = case_dir.name
    run = case_dir.parent.parent
    baseline = load(replay_dir / 'result.json')
    if (name != 'l00_c00_decoder_o0' or replay_dir.parent.name != 'recovery'
            or not re.fullmatch('v2-[0-9a-f]{32}', replay_dir.name)
            or baseline['case'] != name or baseline.get('weight_storage') == 'external'
            or baseline['status'] != 'FAILED' or not baseline.get('tool_exitcode')):
        raise ValueError('Expected the failed embedded-weight L0 V2 replay directory')
    fixture = case_dir / 'execution/fixture/manifest.json'
    if sha(fixture) != baseline['fixture_sha256']:
        raise ValueError('Fixture has changed since the embedded V2 replay')
    case, = load(fixture)['cases']
    build = load(run / 'omc/compile.json')
    old = build['cases'][name]
    if (case['case'] != name or old['status'] != 'COMPILED'
            or old['omc']['sha256'] != case['model']['sha256']
            or old['omc']['sha256'] != baseline['omc_sha256']):
        raise ValueError('Compiled, deployed and replayed embedded OMC identities differ')
    embedded = checked(run / 'omc', old['omc'])
    jobs = sorted((run / 'omc/models' / name).glob('attempt-*/omc/report.json'))
    receipts = [r for file in jobs if (r := load(file)).get('status') == 'COMPILED'
                and r['omc']['sha256'] == baseline['omc_sha256']]
    if not receipts:
        raise ValueError('Missing original L0 OMG report for the replayed OMC')
    original = receipts[0]
    ddk = Path(original['ddk_root'])
    options = {'save_weights_as_external_data': 'false', 'use_origin_format': 'true'}
    if (original['requested_options'] != options or original['compiler_profile'] != DEFAULT_PROFILE
            or original['platform'] != 'kirinx90' or original['ddk_id'] != build['ddk_id']
            or sha(Path(original['omg_path'])) != original['omg_sha256']
            or installed_vscale(ddk) != build['gdr_source_sha256']):
        raise ValueError('Original L0 compiler recipe, OMG or installed GDR changed')
    diagnosis = original.get('tbe_compatibility', {}).get('diagnosis')
    manifest = run / 'air/manifest.json'
    if sha(manifest) != build['manifest_sha256']:
        raise ValueError('Original split AIR manifest changed')
    entry, = [c for c in load(manifest)['cases'] if c['id'] == name]
    metadata = checked(manifest.parent, entry['export'])
    export = load(metadata)
    expected = old['air_inputs']
    if {k: v['bytes'] for k, v in expected.items()} != _payloads(metadata.parent, export):
        raise ValueError('Incomplete AIR/weight inventory for original L0')

    out = new_output(output)
    job = out / 'omc'
    with tempfile.TemporaryDirectory(dir=out, prefix='source-') as tmp:
        folder = metadata.parent
        if any(not (folder / f).is_file() for f in expected):
            folder = Path(tmp) / 'models' / name
            _pull(folder, Path(tmp), remote_air)
        for file, spec in expected.items():
            checked(folder, dict(spec, path=file))
        print(f'Compiling {name}: same AIR and {len(expected) - 1} weights. OMG log: {job}/compile.log', flush=True)
        result = compile_existing_air(folder / 'export.air', job, ddk, ddk_id=original['ddk_id'],
            compiler_profile=DEFAULT_PROFILE, use_origin_format=True, save_weights_as_external_data=True,
            tbe_diagnosis=Path(diagnosis) if diagnosis else None, timeout_seconds=900)
    if result['status'] != 'COMPILED':
        raise RuntimeError(f'{result.get("error", "OMG failed")}; see {job}/compile.log')
    snapshots = {s['source_relative']: {k: s[k] for k in ('bytes', 'sha256')}
                 for s in result['input_snapshots'] if s['source_relative'] in expected}
    if (snapshots != expected or result['omg_sha256'] != original['omg_sha256']
            or installed_vscale(ddk) != build['gdr_source_sha256']
            or result['requested_options'] != dict(options, save_weights_as_external_data='true')):
        raise ValueError('Compiler snapshots, tool or options failed the embedded/external comparison')
    model = checked(job, result['omc'])
    parts = read_container(model)['partitions']
    if (model != select_primary_omc(job) or [p['type'] for p in parts] != [0, 1, 2]
            or parts[1]['bytes'] != 0):
        raise ValueError('Expected an external OMC with an empty internal weight partition')
    schema = load_schema(ddk / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    graph = schema.decode(read_partition(model, parts[0]), 'ModelDef')
    io = bind_compiled(graph, export['source_inputs'], export['golden'], _enum_fields(schema), export['output_contract'])
    fields = ('semantic_name', 'bytes', 'dtype')
    if _slots(io['inputs'], (*fields, 'shape')) != _slots(case['inputs'], (*fields, 'shape')):
        raise ValueError('External OMC inputs differ from the original fixture')
    # bind_compiled checked the permitted singleton views for the sole hidden output.
    if _slots(io['outputs'], fields) != _slots(case['outputs'], fields):
        raise ValueError('External OMC hidden output differs from the original fixture')
    weights = [dict(s, path=str(checked(job, s))) for s in result['external_data_candidates']
               if Path(s['path']).suffix == '.weight']
    weight_plan, _ = _weight_plan(graph, weights, model.parent)
    comparison = _compare_payloads(embedded, model, parts, weight_plan)
    files = []
    for path, record in [(model, result['omc']), *((Path(r['path']), r) for r in weight_plan)]:
        path.replace(out / path.name)
        files.append(dict(path=path.name, bytes=record['bytes'], sha256=record['sha256']))
    save_json(out / 'external.json', dict(case=name, baseline=replay_dir.name,
        baseline_sha256=sha(replay_dir / 'result.json'), fixture_sha256=baseline['fixture_sha256'],
        embedded_omc_sha256=baseline['omc_sha256'], files=files, payload_comparison=comparison,
        input_order=[s['semantic_name'] for s in io['inputs']]))
    shutil.rmtree(job / 'compile-work')
    print(f'External L0 OMC ready: {out / model.name}, {len(weights)} weight file(s)', flush=True)

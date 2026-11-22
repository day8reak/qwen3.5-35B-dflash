"""Compile the exported seven-input GDR AIR and score its two outputs against the supplied CPU bins."""
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from ...common.accuracy import ERROR_RATIO_FIELDS, THRESHOLDS
from ...execution.hdc_run import _file_record
from ...toolchain.existing_air import compile_existing_air
from ..layer_probe import batch
from ..layer_probe.common import KIND, checked, identity, metrics, save_json, sha
from .decode_cache_control import graph_message
from .gdr_stage_data import GDR, shapes

CASE = 'l00_c00_standalone_gdr_o0'
BASIS = 'CPU_GDR_RECURRENCE_ON_EXPORTED_INPUTS'
SCOPE = 'Standalone ChunkGatedDeltaRule vs supplied CPU golden, valid core prefix and complete final state'
RECEIPT = 'standalone-gdr.json'
OUTPUTS = ('core_attn', 'last_recurrent_state')
FIELDS = (*THRESHOLDS, 'mean_abs', 'max_abs', *ERROR_RATIO_FIELDS, 'reference_metrics_passed')


def read(path):
    return json.loads(path.read_text())


def bundle(root):
    manifest = read(root / 'manifest.json')
    if (manifest['operator'], manifest['chunk_size'], manifest['physical_rows']) != ('ChunkGatedDeltaRule', 64, 64):
        raise ValueError('Expected the standalone T64 ChunkGatedDeltaRule export')
    air = checked(root, manifest['air'])
    _, graph = graph_message(air.read_bytes(), 'ChunkGatedDeltaRule')
    op, = [n for n in graph.op if n.type == 'ChunkGatedDeltaRule']
    net, = [n for n in graph.op if n.type == 'NetOutput']
    edges = list(net.input)
    if (list(op.input) != [n + ':0' for n in GDR[:-2]] or edges != [op.name + ':0', op.name + ':1']
            or [op.attr[k].i for k in ('chunk_size', 'output_final_state', 'use_qk_l2norm_in_kernel')] != [64, 1, 1]):
        raise ValueError('AIR ports or GDR attributes differ from the exported CPU reference contract')

    def tensor(spec, name, shape, dtype):
        path = checked(root, spec)
        if (spec['name'] != name or spec['shape'] != shape or np.dtype(spec['dtype']) != np.dtype(dtype)
                or path.stat().st_size != math.prod(shape) * np.dtype(dtype).itemsize):
            raise ValueError(f'{path}: expected {name} {dtype} {shape}')
        return dict(spec, dtype=np.dtype(dtype).name)

    inputs = []
    for i, (spec, name) in enumerate(zip(manifest['inputs'], GDR[:-2], strict=True)):
        if spec['air_index'] != i:
            raise ValueError(f'Unexpected AIR index for {name}')
        dtype = '<i2' if name == 'effective_length' else '<f2'
        inputs.append(dict(tensor(spec, name, shapes(name)[0], dtype), semantic_name=name, fixture_index=i))
    valid = int(np.fromfile(root / inputs[-1]['path'], dtype='<i2').item())
    if not 0 <= valid <= 64 or valid != manifest['effective_length']:
        raise ValueError('effective_length.bin disagrees with the manifest or exceeds T64')
    outputs = []
    for i, (decl, cpu, name) in enumerate(zip(manifest['outputs'], manifest['cpu_golden'], OUTPUTS, strict=True)):
        shape = shapes(GDR[-2 + i])[0]
        if (decl['name'], decl['air_index'], decl['shape'], np.dtype(decl['dtype'])) != (name, i, shape, np.float16):
            raise ValueError(f'Unexpected exported ABI for {name}')
        ref_shape = [1, valid, *shape[2:]] if i == 0 else shape
        record = tensor(cpu, 'core_valid' if i == 0 else name, ref_shape, '<f2')
        outputs.append(dict(record, name=name, shape=shape, golden_shape=ref_shape))
    contract = dict(order=list(OUTPUTS), source_edges=edges, required_dtypes={'0': 'float16', '1': 'float16'})
    return air, inputs, outputs, contract, valid


def compile_case(root, out, ddk, ddk_id, source, diagnosis):
    manifest_hash = sha(root / 'manifest.json')
    air, _, _, _, _ = bundle(root)
    air_hash = sha(air)
    expected = sha(source)
    installed = [ddk / name / 'chunk_gated_delta_rule.cpp' for name in
                 ('ddk/ascendc/ops/impl/custom', 'tools/tools_ascendc/custom_op/ascend310b/src')]

    def verify_install():
        for path in installed:
            if sha(path) != expected:
                raise ValueError(f'{path} differs from the selected Tmatrix source {source}')
    verify_install()
    print(f'Compile {air} with GDR source SHA-256 {expected}', flush=True)
    report = compile_existing_air(air, out, ddk, ddk_id=ddk_id, tbe_diagnosis=diagnosis,
                                  use_origin_format=True, save_weights_as_external_data=False)
    if report['status'] != 'COMPILED':
        raise RuntimeError(f"{report.get('error', report['status'])}; see {out / 'compile.log'}")
    verify_install()
    if report['air']['sha256'] != air_hash or sha(root / 'manifest.json') != manifest_hash:
        raise ValueError('GDR AIR or bundle manifest changed during compilation')
    model = checked(out, report['omc'])
    save_json(out / RECEIPT, dict(bundle=os.path.relpath(root, out), manifest_sha256=manifest_hash,
              gdr_source_sha256=expected, compilation=identity(out / 'report.json', out)))
    print(model, flush=True)


def prepare_fixture(compiled, out, ddk, ddk_id, selection):
    from ...draft.incremental.torch_export import _compiled_observation
    from ...draft.ir_inspect import _enum_fields
    from ...toolchain.omc_container import read_container, read_partition
    from ...toolchain.omc_wire import load_schema
    receipt = read(compiled / RECEIPT)
    report = read(checked(compiled, receipt['compilation']))
    root = (compiled / receipt['bundle']).resolve()
    if ((selection is not None and list(selection) != [CASE])
            or report['status'] != 'COMPILED' or report['ddk_id'] != ddk_id
            or sha(root / 'manifest.json') != receipt['manifest_sha256']):
        raise ValueError('Standalone GDR selection, compiled DDK or exported bundle differs')
    air, inputs, outputs, contract, valid = bundle(root)
    if sha(air) != report['air']['sha256']:
        raise ValueError('GDR AIR differs from the compiled snapshot')
    model = checked(compiled, report['omc'])
    schema = load_schema(Path(ddk) / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    part, = [p for p in read_container(model)['partitions'] if p['type'] == 0]
    doc = schema.decode(read_partition(model, part), 'ModelDef')
    io = batch.bind_compiled(doc, inputs, outputs, _enum_fields(schema), contract)
    if len(io['inputs']) != len(inputs):
        raise ValueError('Expected all seven named GDR inputs in compiled GraphDef.input')
    batch.check_io_budget(CASE, io)
    fixture = out / 'fixture'
    batch.write_case_fixture(fixture / CASE, root, io, outputs)
    observation = _compiled_observation(model, ddk)
    case = dict(case=CASE, kind='standalone_gdr', point='gdn', call=0, layer=0, phase='prefill', ordinal=0,
                output_name=OUTPUTS[0], **io, effective_length=valid, reference_basis=BASIS,
                model=_file_record(model, CASE + '/model.omc'), graphop_counts=observation['graphop_counts'])
    meta = dict(artifact_kind=KIND, cases=[case], planned_cases=[CASE], uncompiled_cases=[],
                warmup=3, repetitions=10, input_reset='copy-each-call', output_initialization='zero',
                ddk_id=ddk_id, reference=dict(golden_source=BASIS), scope=SCOPE, comparison_scope=SCOPE,
                source_evidence=dict(artifact_kind=KIND, **receipt, omc_sha256=report['omc']['sha256']))
    save_json(fixture / 'manifest.json', meta)
    return meta


def score_core(actual, golden, atol, rtol):
    score = metrics(actual, golden, atol=atol, rtol=rtol)
    per_token = []
    for i in range(actual.shape[1]):
        m = metrics(actual[:, i], golden[:, i], atol=atol, rtol=rtol)
        per_token.append({k: m[k] for k in (*FIELDS, 'passed', 'failed_metrics')})
    score['per_token'] = per_token
    if not per_token:
        score['passed'] = None  # No valid core rows. The final state is still scored.
    elif not per_token[-1]['passed']:
        score['passed'] = False
        score['failed_metrics'].extend(f'last_token_{k}' for k in per_token[-1]['failed_metrics'])
    return score


def summarize(recovery):
    report = read(recovery / 'result.json')
    if report.get('stage') != 'complete' or report.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED':
        raise ValueError(f"{report.get('error') or report.get('device_execution')}; see {recovery / 'result.json'}")
    rows = report['cases']
    if ([r['output_name'] for r in rows] != list(OUTPUTS) or any(r.get('case') != CASE
            or r.get('reference_basis') != BASIS or len(r.get('invocations', [])) != 10 for r in rows)):
        raise ValueError(f'Missing standalone GDR comparison; see {recovery / "result.json"}')
    destination = recovery / 'accuracy.csv'
    with tempfile.NamedTemporaryFile(mode='w', dir=recovery, newline='', delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=['output', 'scope', 'call', 'stable', *FIELDS])
        writer.writeheader()
        for row in rows:
            for call, m in enumerate(row['invocations']):
                scores = [('tensor', m)]
                if row['output_name'] == 'core_attn':
                    tokens = m['per_token']
                    scores = [(f'valid{len(tokens)}', m)]
                    scores.extend((f'token{i}', v) for i, v in enumerate(tokens))
                for scope, v in scores:
                    values = {k: v[k] for k in FIELDS}
                    writer.writerow(dict(output=row['output_name'], scope=scope, call=call, stable=row['stable'], **values))
                    if call == 0 and (scope == scores[0][0] or scope == scores[-1][0]):
                        print(row['output_name'], scope, 'call=0', 'stable=' + str(row['stable']), json.dumps(values))
    Path(stream.name).replace(destination)
    print(f"Numerical validation: {report['numerical_validation']}\n{destination}")
    return report['numerical_validation']

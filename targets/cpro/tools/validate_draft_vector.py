#!/usr/bin/env python3
"""Validate existing X90 tensor probes; never compile or replace Draft models."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys
import tarfile
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from probe_draft_vector import CASES, KIND as COMPILE_KIND, graph_case
from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save, reuse_test_output
from qwen35_moe_dflash.execution.serial_transport import _environment, _build_payloads
from qwen35_moe_dflash.execution.hdc_run import Hdc, _file_record
from qwen35_moe_dflash.execution.probe_progress import Progress, local_status, pack_uploads, unpack_logs
from qwen35_moe_dflash.execution.lock import _acquire_global, _release_global, _space
from qwen35_moe_dflash.draft.profile_board import GLOBAL_LOCK, _guard, _identity_guard
from qwen35_moe_dflash.draft.board import _receive
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.draft.incremental.embedding_probe import CASE as EMBEDDING_CASE, KIND as EMBEDDING_KIND, DTYPES, STAGES
from qwen35_moe_dflash.draft.incremental.attention_probe import CASES as ATTENTION_CASES, KIND as ATTENTION_KIND
from qwen35_moe_dflash.draft.incremental.attention_matrix import CASES as MATRIX_CASES, KIND as MATRIX_KIND
from qwen35_moe_dflash.draft.incremental import top1_semantics

PYTORCH_CASES = set(STAGES.values())
VALIDATION_CASES = (*CASES, *STAGES.values(), *ATTENTION_CASES, *MATRIX_CASES, *top1_semantics.EXTRA_CASES)
WIDTHS = {'float32': 4, 'float16': 2, 'int32': 4, 'int64': 8}
DTYPES = {**DTYPES, 'int64': '<i8'}

KIND = 'qwen35-draft-vector-validation'
PREFIX = '/data/local/tmp/qwen35_vector_'
q = shlex.quote
WARMUP, REPETITIONS = 3, 10
ATOL, RTOL = 1e-5, 1e-4


def selected_cases(names, available=VALIDATION_CASES):
    names = list(names)
    if not names or any(name not in available for name in names) or len(set(names)) != len(names):
        raise ValueError('select one or more distinct available probe cases')
    return names


def member(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('probe member must be relative')
    path = root / relative
    if root.resolve() not in path.resolve().parents or path.is_symlink() or not path.is_file():
        raise ValueError('probe member is missing or outside its directory: ' + name)
    return path


def bind_io(document, declared, enums, *, input_order='data-index'):
    """Bind the public probe ABI, not compiler-internal producer connectivity.

    Each supported probe has unique output (shape, dtype) pairs. NetOutput's
    ordered descriptors therefore identify its reference buffers without
    requiring a particular compiler-internal input reference syntax.
    Attention probes opt into ordered GraphDef.input references: their compiled
    Data.index attributes retain source indices that differ from HIAI slots.
    Exact Data names bind even equal-shaped K/V and control inputs to fixtures.
    The runner still checks runtime counts/dimensions/storage before Process,
    and comparison covers every output. This is not a model KV binder.
    """
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    graphs = document.get('graph', [])
    if len(graphs) != 1:
        raise ValueError('probe needs one compiled main graph')
    if input_order not in ('data-index', 'graph-input'):
        raise ValueError('unknown probe input binding policy')
    nodes = graphs[0]['op']
    if len({n['name'] for n in nodes}) != len(nodes):
        raise ValueError('duplicate compiled node name')
    expected = {r['name']: (i, r) for i, r in enumerate(declared['inputs'])}
    if len(expected) != len(declared['inputs']):
        raise ValueError('duplicate declared probe input name')
    inputs = []
    for node in nodes:
        if node.get('type') != 'Data':
            continue
        if node['name'] not in expected:
            raise ValueError('unexpected compiled probe input: '+node['name'])
        source, spec = expected[node['name']]
        indices = [a['value']['i'] for a in node.get('attr', []) if a['key'] == 'index']
        if len(indices) != 1 or type(indices[0]) is not int:
            raise ValueError('missing or ambiguous compiled input index')
        descs = node['output_desc']
        if len(descs) != 1 or any(_observed_desc(descs[0], enums)[k] != spec[k] for k in ('shape', 'dtype')):
            raise ValueError('compiled probe input descriptor changed')
        inputs.append(dict(spec, index=indices[0], fixture_index=source, bytes=math.prod(spec['shape'])*WIDTHS[spec['dtype']]))
    if (len(inputs) != len(expected) or {r['name'] for r in inputs} != set(expected)
            or sorted(r['index'] for r in inputs) != list(range(len(expected)))):
        raise ValueError('compiled inputs are not a complete indexed mapping')
    inputs.sort(key=lambda r: r['index'])
    input_metadata = {}
    if input_order == 'graph-input':
        references = graphs[0].get('input')
        by_edge = {r['name']+':0': r for r in inputs}
        if (not isinstance(references, list) or len(references) != len(inputs)
                or any(not isinstance(ref, str) or ref not in by_edge for ref in references)
                or len(set(references)) != len(inputs)):
            raise ValueError('compiled GraphDef.input must reference every declared Data:0 exactly once')
        inputs = [dict(by_edge[ref], index=slot, data_index=by_edge[ref]['index'])
                  for slot, ref in enumerate(references)]
        input_metadata = dict(input_binding='COMPILED_GRAPHDEF_INPUT_REFERENCES',
                              graph_input_references=list(references))
    net = [n for n in nodes if n.get('type') == 'NetOutput']
    if len(net) != 1:
        raise ValueError('missing or ambiguous compiled NetOutput')
    descs = net[0].get('input_desc')
    if not isinstance(descs, list) or not descs or len(descs) != len(declared['outputs']):
        raise ValueError('compiled probe output count differs from reference')
    descriptors = [_observed_desc(d, enums) for d in descs]
    outputs = []
    for index, desc in enumerate(descriptors):
        references = net[0].get('input', [])
        if index < len(references) and references[index] in ('', None):
            raise ValueError('compiled probe output has no producer reference: '+json.dumps(dict(index=index, observed=desc)))
        # These small probes have unique output shapes (RMS y versus rstd).
        # Never extend shape-only matching to equal-shaped model KV outputs.
        matches = [(i, r) for i, r in enumerate(declared['outputs'])
                   if all(desc[k] == r[k] for k in ('shape', 'dtype'))]
        if len(matches) != 1:
            detail = dict(index=index, observed=desc, matches=[r['name'] for _, r in matches],
                          declared=[{k: r[k] for k in ('name', 'shape', 'dtype')} for r in declared['outputs']])
            raise ValueError('compiled output descriptor changed or is ambiguous: '+json.dumps(detail))
        source, spec = matches[0]
        outputs.append(dict(spec, index=index, fixture_index=source, bytes=math.prod(spec['shape'])*WIDTHS[spec['dtype']]))
    if sorted(r['fixture_index'] for r in outputs) != list(range(len(declared['outputs']))):
        raise ValueError('compiled outputs do not cover every probe output once')
    return dict(inputs=inputs, outputs=outputs, **input_metadata,
                output_binding='UNIQUE_SHAPE_DTYPE_IN_COMPILED_NETOUTPUT_ORDER',
                netoutput_input_metadata=net[0].get('input'))


def prepare(compiled, out, ddk, ddk_id, selection=None):
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    root = Path(compiled).resolve()
    report = read_json(root/'result.json')
    if report.get('artifact_kind') == MATRIX_KIND:
        from qwen35_moe_dflash.draft.incremental.attention_matrix import prepare_fixture
        return prepare_fixture(root, out, ddk, ddk_id, selection)
    if report.get('artifact_kind') == ATTENTION_KIND:
        from qwen35_moe_dflash.draft.incremental.attention_probe import prepare_fixture
        return prepare_fixture(root, out, ddk, ddk_id, selection)
    if report.get('artifact_kind') == EMBEDDING_KIND:
        from qwen35_moe_dflash.draft.incremental.embedding_probe import prepare_fixture
        return prepare_fixture(root, out, ddk, ddk_id, selection)
    if report.get('artifact_kind') != COMPILE_KIND or report.get('status') not in {'COMPILED', 'PARTIAL'}:
        raise ValueError('expected completed Draft region probe compilation')
    if report['status'] == 'PARTIAL' and selection is None:
        raise ValueError('partial compilation requires --cases selecting only compiled probes')
    rows = report['cases']
    available = selected_cases(r['case'] for r in rows)
    by_name = {r['case']: r for r in rows}
    rows = [by_name[name] for name in selected_cases(available if selection is None else selection, available)]
    if any(row.get('status') != 'COMPILED' or row.get('model_compilation') != 'PASSED' for row in rows):
        raise ValueError('every selected probe must have compiled successfully')
    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    enums = _enum_fields(schema)
    cases = []
    for row in rows:
        name = row['case']
        compilation = read_json(member(root, f'{name}/omc/report.json'))
        if (compilation.get('status') != 'COMPILED' or compilation.get('platform') != 'kirinx90'
                or compilation.get('ddk_id') != ddk_id):
            raise ValueError('probe compiler DDK differs from runtime configuration')
        graph, inputs, golden = graph_case(name)
        if row['status'] != 'COMPILED' or row['graph'] != graph.manifest():
            raise ValueError('probe recipe changed; recompile this probe before execution')
        model = member(root, row['omc'])
        if model.stat().st_size > 64 << 20:
            raise ValueError('small probe model exceeds 64 MiB')
        parts = read_container(model)['partitions']
        main = [p for p in parts if p['type'] == 0]
        if len(main) != 1 or any(model.parent.glob('*.weight')):
            raise ValueError('probe needs one embedded model without external weight sidecars')
        io = bind_io(schema.decode(read_partition(model, main[0]), 'ModelDef'), row['graph'], enums)
        if row.get('top1_contract', {}).get('acceptance') == 'EXACT_OUTPUT_BYTES_ALL_REPEATS':
            for spec in io['outputs']:
                spec['exact_bytes'] = True
        folder = out/'fixture'/name
        folder.mkdir(parents=True)
        for prefix, arrays in (('input', inputs), ('golden', golden)):
            for i, array in enumerate(arrays):
                path = member(root, f'{name}/fixture/{prefix}_{i}.bin')
                if path.stat().st_size != array.nbytes or path.read_bytes() != array.tobytes():
                    raise ValueError('compiled probe fixture differs: ' + str(path))
        plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}']
        for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
            for spec in io[direction]:
                plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'],
                                      ','.join(map(str, spec['shape'])))))
                destination = folder/f'{prefix}_{spec["index"]}.bin'
                shutil.copyfile(root/f'{name}/fixture/{prefix}_{spec["fixture_index"]}.bin', destination)
                spec['fixture_sha256'] = sha256_file(destination)
        (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
        cases.append(dict(case=name, **io, model=_file_record(model, name+'/model.omc'),
                          compiler_options=compilation.get('requested_options', {}),
                          graphop_counts=row['graphop_counts']))
    result = dict(cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=ATOL, rtol=RTOL,
                  scope='DECLARED_DTYPE_REGIONS_ONLY_NOT_FULL_DRAFT_VALIDATION', ddk_id=ddk_id, schema=schema.evidence)
    _save(out/'fixture/manifest.json', result)
    return result


def _exact_required(meta, spec):
    # Token indices need equality: relative float tolerance accepts nearby IDs.
    # FP64 comparison also cannot distinguish every adjacent INT64 value.
    return bool(meta.get('exact_bytes') or spec.get('exact_bytes') or spec['dtype'] in ('int32', 'int64'))


def compare(fixture, native, *, selection=None, allow_host_fake=False):
    import numpy as np
    meta = read_json(fixture/'manifest.json')
    requested = selected_cases(c['case'] for c in meta['cases'])
    selected = selected_cases(requested if selection is None else selection, requested)
    cases = []
    for case in meta['cases']:
        name = case['case']
        if name not in selected:
            continue
        result = read_json(native/name/'result.json')
        expected_kind = 'host-fake-hiai' if allow_host_fake else 'device-hiai'
        timings = result.get('api_us', [])
        if (result.get('status') != 'EXECUTED' or result.get('evidence_kind') != expected_kind
                or result.get('warmup') != WARMUP or result.get('repetitions') != REPETITIONS
                or len(timings) != REPETITIONS or any(type(t) is not int or t < 0 for t in timings)
                or result.get('timing_scope') != 'SYNCHRONOUS_API_WALL_NOT_NPU_ONLY'):
            raise ValueError('native report does not establish ten device calls after warmup')
        total = sum(r['bytes'] for r in case['outputs'])
        tensors, stable, first = [], True, None
        mismatches, examples = 0, []
        for iteration in range(REPETITIONS):
            snapshot = member(native, f'{name}/{iteration}.bin')
            if snapshot.stat().st_size != total:
                raise ValueError('probe snapshot has incorrect byte extent')
            raw = snapshot.read_bytes()
            if first is None:
                first = raw
            stable = stable and raw == first
            offset = 0
            for spec in case['outputs']:
                dtype = np.dtype(DTYPES[spec['dtype']])
                observed_bytes = raw[offset:offset+spec['bytes']]
                actual = np.frombuffer(observed_bytes, dtype=dtype).astype(np.float64)
                offset += spec['bytes']
                expected_bytes = (fixture/name/f'golden_{spec["index"]}.bin').read_bytes()
                expected = np.frombuffer(expected_bytes, dtype=dtype).astype(np.float64)
                if expected.size*dtype.itemsize != spec['bytes'] or not np.isfinite(expected).all():
                    raise ValueError('invalid retained reference')
                finite = bool(np.isfinite(actual).all())
                delta = actual-expected
                passed = finite and bool(np.all(np.abs(delta) <= meta['atol']+meta['rtol']*np.abs(expected)))
                exact = observed_bytes == expected_bytes
                if _exact_required(meta, spec):
                    passed = passed and exact
                if meta.get('top1_semantics'):
                    bad = np.flatnonzero(actual != expected)
                    mismatches += int(bad.size)
                    for index in bad[:max(0, 8-len(examples))]:
                        examples.append(dict(iteration=iteration, output=spec['name'],
                            flat_index=int(index), expected=int(expected[index]), actual=int(actual[index])))
                tensors.append(dict(iteration=iteration, name=spec['name'], passed=passed, finite=finite,
                    bytes_equal=exact,
                    max_abs_error=float(np.max(np.abs(delta))) if finite else None,
                    relative_l2_error=float(np.linalg.norm(delta)/max(float(np.linalg.norm(expected)), 1e-30)) if finite else None))
        passed = all(r['passed'] for r in tensors)
        finite = all(r['finite'] for r in tensors)
        cases.append(dict(case=name, passed=passed, repeat_bitwise_stable=stable,
            outputs=[dict(name=spec['name'], passed=all(t['passed'] for t in tensors if t['name'] == spec['name']),
                          exact_bytes_required=_exact_required(meta, spec),
                          all_calls_bytes_equal=all(t['bytes_equal'] for t in tensors if t['name'] == spec['name']))
                     for spec in case['outputs']],
            numerics=dict(finite=finite,
                max_abs_error=max(r['max_abs_error'] for r in tensors) if finite else None,
                max_relative_l2_error=max(r['relative_l2_error'] for r in tensors) if finite else None),
            tensors=tensors, graphop_counts=case['graphop_counts'], api_us=timings,
            compiler_options=case.get('compiler_options', {}),
            mean_ms=sum(timings)/REPETITIONS/1000, min_ms=min(timings)/1000, max_ms=max(timings)/1000))
        if meta.get('top1_semantics'):
            cases[-1].update(compiled_case=case['compiled_case'], scenario=case['scenario'],
                mismatched_elements_all_calls=mismatches, mismatch_examples=examples)
    passed = all(r['passed'] for r in cases)
    report = dict(status='PASSED' if passed else 'NUMERICAL_FAILED', numerical_validation='PASSED' if passed else 'FAILED',
        cases=cases, warmup=WARMUP, repetitions=REPETITIONS, atol=meta['atol'], rtol=meta['rtol'],
        scope=meta.get('scope'), exact_bytes_required=bool(meta.get('exact_bytes')),
        validated_cases=selected, requested_cases=requested,
        repeat_bitwise_stable=all(r['repeat_bitwise_stable'] for r in cases),
        timing_scope='SYNCHRONOUS_API_WALL_NOT_NPU_ONLY; SNAPSHOTS_BETWEEN_CALLS_EXCLUDED',
        full_draft_validation='NOT_RUN', internal_instruction_precision='NOT_PROVEN_BY_OUTPUT_MATCH')
    if meta.get('top1_semantics'):
        report['top1_semantics'] = top1_semantics.summary(meta, cases)
    return report


def analyze(directory):
    """Read retained snapshots only; never execute models or change pass gates."""
    import numpy as np
    directory = Path(directory).resolve()
    report = read_json(directory/'result.json')
    if report.get('artifact_kind') != KIND or not report.get('validated_cases'):
        raise ValueError('analysis requires recovered probe outputs')
    source = Path(report['recovery_source'])
    fixture, native = source/'fixture', directory/'native'
    request = read_json(source/'request.json')
    if sha256_file(fixture/'manifest.json') != request['fixture_sha256']:
        raise ValueError('retained fixture manifest changed')
    meta = read_json(fixture/'manifest.json')
    for case in meta['cases']:
        for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
            for spec in case[direction]:
                path = member(fixture, f'{case["case"]}/{prefix}_{spec["index"]}.bin')
                if path.stat().st_size != spec['bytes'] or sha256_file(path) != spec['fixture_sha256']:
                    raise ValueError('retained input/reference changed after launch')
    comparison = compare(fixture, native, selection=report['validated_cases'])
    if meta.get('top1_semantics'):
        return dict(comparison, artifact_kind=KIND, model_execution='NOT_RUN',
                    scope='RECOMPARE_RETAINED_TOP1_OUTPUTS_NO_DEVICE_EXECUTION')
    if meta.get('source_evidence',{}).get('artifact_kind') == MATRIX_KIND:
        from qwen35_moe_dflash.draft.incremental.attention_matrix import diagnose
        return diagnose(fixture, native, meta, comparison, report)
    if set(ATTENTION_CASES) & set(comparison['validated_cases']):
        from qwen35_moe_dflash.draft.incremental.attention_probe import diagnose
        return diagnose(fixture, native, meta, comparison, report)
    if PYTORCH_CASES & set(comparison['validated_cases']):
        from qwen35_moe_dflash.draft.incremental.embedding_diagnose import diagnose
        return diagnose(fixture, native, meta, comparison, report)

    def metrics(actual, expected):
        finite = bool(np.isfinite(actual).all())
        bad = ~np.isfinite(actual) | (np.abs(actual-expected) > meta['atol']+meta['rtol']*np.abs(expected))
        a, e = actual.reshape(-1), expected.reshape(-1)
        a_norm, e_norm = np.linalg.norm(a), np.linalg.norm(e)
        dot = float(a @ e) if finite else None
        return dict(finite=finite, failing_elements=int(bad.sum()), elements=int(actual.size),
            max_abs_error=float(np.max(np.abs(a-e))) if finite else None,
            relative_l2_error=float(np.linalg.norm(a-e)/e_norm) if finite and e_norm else None,
            least_squares_scale=dot/float(e_norm**2) if finite and e_norm else None,
            cosine=dot/float(a_norm*e_norm) if finite and a_norm and e_norm else None)

    def number(value):
        return float(value) if np.isfinite(value) else str(float(value))

    def round16(value):
        with np.errstate(over='ignore', invalid='ignore'):
            return value.astype(np.float16).astype('<f4')

    cases = []
    for case in meta['cases']:
        name = case['case']
        if name not in report['validated_cases']:
            continue
        input_spec = next(s for s in case['inputs'] if s['name'] == 'x')
        x = np.fromfile(fixture/name/f'input_{input_spec["index"]}.bin', dtype='<f4').reshape(input_spec['shape'])
        x_rows = x.reshape(-1, x.shape[-1])
        raw = (native/name/'0.bin').read_bytes()
        outputs, offset = [], 0
        for spec in case['outputs']:
            actual = np.frombuffer(raw[offset:offset+spec['bytes']], dtype='<f4').astype(np.float64).reshape(spec['shape'])
            offset += spec['bytes']
            expected = np.fromfile(fixture/name/f'golden_{spec["index"]}.bin', dtype='<f4').astype(np.float64).reshape(spec['shape'])
            a_rows, e_rows = actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, expected.shape[-1])
            if len(a_rows) != len(x_rows) or len(a_rows) < 4:
                raise ValueError('probe row analysis requires the original fixture shapes')
            groups = []
            for label, begin, end in (('zero_input_row', 0, 1), ('large_input_row', 1, 2),
                    ('small_input_row', 2, 3), ('ordinary_input_rows', 3, len(x_rows))):
                a, e = a_rows[begin:end], e_rows[begin:end]
                error = np.where(np.isfinite(a), np.abs(a-e), np.inf)
                row, column = np.unravel_index(int(np.argmax(error)), error.shape)
                row += begin
                groups.append(dict(group=label, **metrics(a, e),
                    worst=dict(row=int(row), column=int(column), actual=number(a_rows[row, column]),
                        expected=number(e_rows[row, column]),
                        input_finite_abs_max=(number(np.max(np.abs(x_rows[row][np.isfinite(x_rows[row])])))
                            if np.isfinite(x_rows[row]).any() else None),
                        input_nonfinite_elements=int((~np.isfinite(x_rows[row])).sum()),
                        actual_first4=[number(v) for v in a_rows[row, :4]],
                        expected_first4=[number(v) for v in e_rows[row, :4]])))
            a32 = actual.astype('<f4')
            finite = np.isfinite(a32)
            rounded = round16(a32)
            precision = dict(finite_elements=int(finite.sum()),
                fp16_roundtrip_exact_elements=int((finite & (rounded.view('<u4') == a32.view('<u4'))).sum()),
                scope='NUMERICAL_SIGNATURE_ONLY_NOT_INTERNAL_INSTRUCTION_PROOF')
            if name in {'residual_fp32', 'residual_addcustom_fp32'}:
                right_spec = next(s for s in case['inputs'] if s['name'] == 'residual')
                right = np.fromfile(fixture/name/f'input_{right_spec["index"]}.bin', dtype='<f4').reshape(right_spec['shape'])
                full = np.add(x, right, dtype=np.float32)
                rounded_sum = np.add(round16(x), round16(right), dtype=np.float32)
                hypotheses = {'fp32_add': full, 'output_round_only': round16(full),
                    'inputs_round_only': rounded_sum, 'inputs_and_output_round': round16(rounded_sum)}
                precision['rounding_models'] = {}
                for label, candidate in hypotheses.items():
                    both_finite = finite & np.isfinite(candidate)
                    precision['rounding_models'][label] = dict(elements=int(a32.size),
                        exact_fp32_bit_matches=int((both_finite & (candidate.view('<u4') == a32.view('<u4'))).sum()),
                        max_abs_error_to_actual=(float(np.max(np.abs(candidate.astype(np.float64)-actual)))
                            if both_finite.all() else None))
            outputs.append(dict(name=spec['name'], shape=spec['shape'], groups=groups, precision=precision))
        summary = next(c for c in comparison['cases'] if c['case'] == name)
        cases.append(dict(case=name, numerics=summary['numerics'],
            repeat_bitwise_stable=summary['repeat_bitwise_stable'],
            compiler_options=summary['compiler_options'], outputs=outputs))
    return dict(status='ANALYZED', artifact_kind=KIND, source_batch_status=report['status'],
        numerical_validation=comparison['numerical_validation'], validated_cases=comparison['validated_cases'],
        scope='LOCAL_RETAINED_OUTPUTS; ALL_CALLS_RECOMPARED; ROW_DETAILS_FROM_ITERATION_0',
        model_execution='NOT_RUN', cases=cases)


def supervisor(meta, board, token, env, search):
    lines = ['#!/bin/sh', 'cd '+q(str(board))+' || exit 91', _guard(board, token)+' || exit 92',
             _guard(GLOBAL_LOCK, token)+' || exit 93',
             'printf "%s\\n" '+q(token)+' > started', 'failed=0', 'for batch in once; do']
    for index, case in enumerate(meta['cases'], 1):
        name = case['case']
        argv = ['timeout', '-k', '5', '600', 'env', 'LD_LIBRARY_PATH='+':'.join([str(board/'lib'), *search]),
                str(board/'tensor-probe'), str(board/name/'model.omc'), env['runtime_version'],
                str(board/name), str(board/'outputs'/name)]
        lines += ['printf "%s\\n" '+q(f'CASE {index}/{len(meta["cases"])} {name}')
                  +' > progress.tmp && mv progress.tmp progress',
                  shlex.join(argv)+' > '+name+'.stdout 2> '+name+'.stderr', 'rc=$?',
                  'printf "%s\\n" "$rc" > '+name+'.exitcode',
                  'if test "$rc" -ne 0; then failed=1; break; fi']
    # Do not start another SDK process after a failed/timed-out case.
    return '\n'.join([*lines, 'done', 'printf "%s\\n" "$failed" > native.exitcode',
        'printf "ARCHIVING\\n" > progress.tmp && mv progress.tmp progress',
        'tar -cf outputs.tar -C outputs .', 'printf "%s\\n" "$?" > archive.exitcode',
        'set --', 'for f in '+shlex.join(log_names(meta['cases']))+'; do',
        '  if test -f "$f"; then set -- "$@" "$f"; fi', 'done',
        'tar -cf logs.tar "$@"', 'printf "%s\\n" "$?" > logs.exitcode',
        'printf "COMPLETE\\n" > progress.tmp && mv progress.tmp progress',
        'printf "%s\\n" '+q(token)+' > complete.tmp', 'mv complete.tmp complete', ''])


def log_names(cases):
    names = [c['case'] if isinstance(c, dict) else c for c in cases]
    return ['native.exitcode', 'archive.exitcode', 'supervisor.stdout', 'supervisor.stderr',
            *(name+'.'+suffix for name in names for suffix in ('stdout', 'stderr', 'exitcode'))]


def _check_probe_dtypes(meta, built):
    required = {s['dtype'] for c in meta['cases'] for direction in ('inputs', 'outputs') for s in c[direction]}
    missing = required - set(built.get('tensor_probe_dtypes', []))
    if missing:
        raise ValueError('rebuild runtime/native with build_hiai.py --replace; tensor probe lacks dtypes: '
                         + ', '.join(sorted(missing)))
    largest = max(n for c in meta['cases'] for direction in ('inputs', 'outputs')
                  for s in c[direction] for n in s['shape'])
    if largest > built.get('tensor_probe_limits', {}).get('max_dimension', 65536):
        raise ValueError('rebuild runtime/native with build_hiai.py --replace; tensor probe axis limit is too small')


def start(args, out):
    config = Path(args.config).resolve(); source = Path(args.compiled).resolve()
    for path in (config, source):
        if path == out or out in path.parents or source in out.parents:
            raise ValueError('execution slot must be separate from config and compiled probes')
    reuse_test_output(out, config=config)
    out.mkdir(parents=True)
    state = dict(artifact_kind=KIND, status='FAILED', stage='prepare', device_execution='NOT_RUN',
                 model_compilation='NOT_RUN', numerical_validation='NOT_RUN', lock_state='NO_KNOWN_OWNED_LOCK')
    hdc = None; held = uncertain = launched = False
    progress = Progress(out, state)
    try:
        progress.update('prepare', 'Checking compiled ABI and preparing fixtures', status='PREPARING')
        cfg = read_json(config)
        env = read_json(config.parent/cfg['assets'])['environment']
        with (source.parent/('.'+source.name+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            selection = top1_semantics.BASE_CASES if getattr(args, 'top1_semantics', False) else args.cases
            meta = prepare(source, out, args.ddk_root, env['ddk_id'], selection)
            if getattr(args, 'top1_semantics', False):
                meta = top1_semantics.expand(out, meta)
        state.update(status='PREPARED', stage='prepared', requested_cases=[c['case'] for c in meta['cases']])
        if meta.get('top1_semantics'):
            state['top1_semantics'] = meta['top1_semantics']
        if meta.get('source_evidence',{}).get('artifact_kind') == MATRIX_KIND:
            state['matrix_preparation_failures'] = meta['source_evidence'].get('preparation_failures',[])
        if not args.apply:
            return state
        progress.update('payload', f'Preparing deployment for {len(meta["cases"])} cases')
        native = (config.parent/cfg.get('native_directory', 'native')).resolve()
        binary = native/'build/qwen35_tensor_probe'
        built = read_json(native/'result.json')
        if not binary.is_file() or built.get('diagnostic_binaries', {}).get(binary.name) != {
                'bytes': binary.stat().st_size, 'sha256': sha256_file(binary)}:
            raise ValueError('rebuild runtime/native with build_hiai.py --replace for tensor probes')
        _check_probe_dtypes(meta, built)
        uploads, search, _ = _build_payloads(out, 600, resident='resident-mixed', native_directory=native)
        uploads = [r for r in uploads if r['remote'].startswith('lib/')]
        uploads.append(_file_record(binary, 'tensor-probe'))
        for case in meta['cases']:
            uploads.append(case['model'])
            for path in sorted((out/'fixture'/case['case']).iterdir()):
                if path.name.startswith('input_') or path.name == 'plan.txt':
                    uploads.append(_file_record(path, case['case']+'/'+path.name))
        exe, serial, server = _environment(); token = uuid.uuid4().hex
        board = PurePosixPath(PREFIX+token)
        (out/'launch.sh').write_text(supervisor(meta, board, token, env, search))
        uploads.append(_file_record(out/'launch.sh', 'launch.sh'))
        progress.update('pack', f'Packing {len(uploads)} files ({sum(r["bytes"] for r in uploads)} bytes) for one HDC transfer')
        payload = pack_uploads(out, uploads)
        plan = dict(artifact_kind=KIND, token=token, board_attempt=str(board), serial=serial,
                    hdc_server=server, runtime=env, cases=[c['case'] for c in meta['cases']],
                    fixture_sha256=sha256_file(out/'fixture/manifest.json'),
                    transport='archive-v1', payload=payload, payload_files=len(uploads))
        _save(out/'request.json', plan)
        progress.update('board-preflight', 'Checking board owner, firmware and free space', board_attempt=str(board))
        hdc = Hdc(exe, serial, server, out, 180)
        uncertain = True
        state['lock_state'] = 'RETAINED_OR_UNCERTAIN'
        _save(out/'result.json', state)
        try:
            _acquire_global(hdc, GLOBAL_LOCK, token, 0)
        except ValueError as error:
            if 'BOARD_MODEL_IO_BUSY:' in str(error):
                uncertain = False
            raise
        held = True; uncertain = False
        _, firmware = hdc.shell('param get const.ohos.fullname', 'vector-firmware')
        if firmware.strip() != env['firmware_id'].strip():
            raise ValueError('board firmware differs from runtime configuration')
        _, space = hdc.shell('df -Pk /data/local/tmp', 'vector-space')
        output_bytes = REPETITIONS*sum(s['bytes'] for c in meta['cases'] for s in c['outputs'])
        if _space(space) < payload['bytes']+sum(r['bytes'] for r in uploads)+2*output_bytes+(64 << 20):
            raise OSError('insufficient board space for small probe request')
        hdc.shell('mkdir '+q(str(board))+' && printf "%s\\n" '+q(token)+' > '+q(str(board/'owner'))
                  +' && mkdir '+shlex.join(str(board/p) for p in ['lib', 'outputs']), 'vector-directory')
        progress.update('upload', f'Uploading payload.tar ({payload["bytes"]} bytes); device computation has not been launched')
        hdc.send(payload['local'], board/'payload.tar')
        progress.update('verify-upload', f'Verifying archive and all {len(uploads)} extracted files')
        hdc.shell(_guard(board, token)+' && '+_guard(GLOBAL_LOCK, token)
            +' && '+_identity_guard(board/'payload.tar', payload)
            +' && cd '+q(str(board))+' && tar -xf payload.tar'
            +' && sha256sum -c payload.sha256 > payload-checks.stdout'
            +' && chmod +x tensor-probe', 'vector-payload-install')
        launched = True
        progress.update('launch', 'Requesting one detached board execution',
                        launch_requested=True, device_execution='RUNNING_OR_COMPLETION_PENDING')
        hdc.shell('cd '+q(str(board))+' && '+_guard(board, token)
                  +' && { nohup sh launch.sh < /dev/null > supervisor.stdout 2> supervisor.stderr & }', 'vector-launch')
        progress.update('running-detached', 'Launch acknowledged; use --recover on this execution directory',
                        status='VECTOR_VALIDATION_STARTED', next='Recover the same execution; no relaunch.')
    except Exception as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    finally:
        if hdc and held and not launched and not uncertain:
            try:
                _release_global(hdc, GLOBAL_LOCK, token); held = False
            except Exception as error:
                state['lock_release_error'] = str(error)
        state['lock_state'] = 'RETAINED_OR_UNCERTAIN' if held or uncertain else 'NO_KNOWN_OWNED_LOCK'
        _save(out/'result.json', state)
        progress.close()
    return state


def extract(archive, destination, meta, *, selection=None):
    requested = selected_cases(c['case'] for c in meta['cases'])
    selected = selected_cases(requested if selection is None else selection, requested)
    expected = {}
    for case in meta['cases']:
        expected[case['case']+'/result.json'] = None
        for i in range(REPETITIONS):
            expected[f'{case["case"]}/{i}.bin'] = sum(r['bytes'] for r in case['outputs'])
    required = {name for name in expected if name.split('/')[0] in selected}
    with tarfile.open(archive) as stream:
        members = [m for m in stream.getmembers() if not m.isdir()]
        names = [m.name.removeprefix('./') for m in members]
        if len(names) != len(set(names)) or not required <= set(names) <= set(expected):
            raise ValueError('archive has missing completed outputs or unexpected probe members')
        for m, name in zip(members, names):
            size = expected[name]
            limit = 65535 if size is None else size
            if not m.isfile() or not 0 <= m.size <= limit:
                raise ValueError('invalid archive member extent or type')
            if name not in required:
                continue  # Failed-case partial snapshots are never compared as completed calls.
            if (size is None and m.size == 0) or (size is not None and m.size != size):
                raise ValueError('completed probe output is truncated')
            path = destination/name; path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(m) as src, path.open('xb') as dst:
                shutil.copyfileobj(src, dst)


def execution_diagnostics(out, selection=CASES):
    """Explain a completed request from retained logs, without model execution."""
    def tail(name, limit=3072):
        path = out/name
        if not path.is_file():
            return ''
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-limit))
            return stream.read(limit).decode('utf-8', errors='replace').strip()

    def code(name):
        path = out/name
        if not path.is_file():
            return None
        with path.open() as stream:
            raw = stream.read(64).strip()
        return int(raw) if re.fullmatch(r'[0-9]{1,3}', raw) and int(raw) <= 255 else None

    codes = {name: code(name+'.exitcode') for name in selected_cases(selection)}
    report = dict(native_exitcode=code('native.exitcode'), archive_exitcode=code('archive.exitcode'),
                  case_exitcodes=codes, errors=[])
    failed = next((name for name, value in codes.items() if value not in (None, 0)), None)
    if failed is None:
        failed = next((name for name, value in codes.items() if value is None), None)
    if failed is not None:
        report['first_failure'] = dict(case=failed, exitcode=codes[failed],
            stderr_tail=tail(failed+'.stderr'), stdout_tail=tail(failed+'.stdout'))
        report['errors'].append(f'probe {failed}: exitcode={codes[failed]} (None means missing/invalid exit record)')
    if report['native_exitcode'] != 0:
        report['errors'].append('native execution summary: exitcode='+str(report['native_exitcode']))
    if report['archive_exitcode'] != 0:
        report['errors'].append('output archive: exitcode='+str(report['archive_exitcode']))
    if report['errors']:
        report['supervisor_stderr_tail'] = tail('supervisor.stderr')
    return report


def pending_progress(hdc, board, token, requested):
    """Markers are observations, not a process-liveness or numerical verdict."""
    command = [_guard(board, token)+' || exit 92', 'cd '+q(str(board))+' || exit 91',
        'if test -f started; then read -r probe_token < started; test "$probe_token" = '+q(token)
        +' || exit 93; echo STARTED; fi',
        'if test -f progress; then read -r probe_step < progress; printf "PROGRESS %s\\n" "$probe_step"; fi']
    # Keep the HDC command short even for the 129-case matrix. Only filenames
    # declared in the request are accepted when parsing the returned markers.
    command += ['for probe_file in *.exitcode; do',
        '  case "$probe_file" in native.exitcode|archive.exitcode|logs.exitcode) continue;; esac',
        '  if test -f "$probe_file"; then read -r probe_rc < "$probe_file"; '
        'printf "EXIT %s %s\\n" "${probe_file%.exitcode}" "$probe_rc"; fi', 'done',
        'for probe_file in *.stdout; do',
        '  case "$probe_file" in supervisor.stdout|payload-checks.stdout) continue;; esac',
        '  probe_case=${probe_file%.stdout}',
        '  if test -f "$probe_file" && test ! -f "$probe_case.exitcode"; then '
        'printf "CASE_STARTED %s\\n" "$probe_case"; fi', 'done']
    _, raw = hdc.shell('\n'.join(command), 'vector-progress', timeout=30)
    result = dict(supervisor_started=False, completed_cases=0, last_case_exit=None,
                  incomplete_case=None, progress=None,
                  scope='RETAINED_MARKERS; NOT_A_LIVE_PROCESS_CHECK')
    for line in raw.splitlines():
        if line == 'STARTED':
            result['supervisor_started'] = True
        elif line.startswith('PROGRESS ') and len(line) <= 256:
            result['progress'] = line[9:]
        elif line.startswith('CASE_STARTED ') and line[13:] in requested:
            result['incomplete_case'] = line[13:]
        elif line.startswith('EXIT '):
            parts = line.split()
            if len(parts) != 3 or parts[1] not in requested or not re.fullmatch(r'\d{1,3}', parts[2]) or int(parts[2]) > 255:
                raise ValueError('invalid case exit progress')
            result['completed_cases'] += 1
            if (result['last_case_exit'] is None or requested.index(parts[1]) >
                    requested.index(result['last_case_exit']['case'])):
                result['last_case_exit'] = dict(case=parts[1], exitcode=int(parts[2]))
        else:
            raise ValueError('unrecognized board progress response')
    return result


def recover(root, out):
    root = Path(root).resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('recovery needs a separate output slot')
    reuse_test_output(out, recovery_source=root); out.mkdir(parents=True)
    state = dict(artifact_kind=KIND, status='FAILED', stage='recover', model_compilation='NOT_RUN',
        numerical_validation='NOT_RUN', recovery_source=str(root), recovery_model_execution='FORBIDDEN',
        lock_state='RETAINED_OR_UNCERTAIN')
    complete = False; hdc = plan = None
    progress = Progress(out, state)
    try:
        progress.update('recover', 'Inspecting the existing request; no model relaunch', status='RECOVERING')
        state['local_status'] = local_status(root)
        plan = read_json(root/'request.json'); token = plan['token']
        if plan['artifact_kind'] != KIND or not re.fullmatch('[0-9a-f]{32}', token) or plan['board_attempt'] != PREFIX+token:
            raise ValueError('invalid vector request')
        board = PurePosixPath(plan['board_attempt']); state['board_attempt'] = str(board)
        requested = selected_cases(plan.get('cases', CASES))
        state['requested_cases'] = requested
        exe, serial, server = _environment()
        if (serial, server) != (plan['serial'], plan['hdc_server']):
            raise ValueError('recover requires the same board')
        hdc = Hdc(exe, serial, server, out, 180)
        _, done = hdc.shell(_guard(board, token)+' && if test -f '+q(str(board/'complete'))
            +'; then cat '+q(str(board/'complete'))+'; else echo PENDING; fi', 'vector-completion')
        if done == 'PENDING':
            state.update(status='VECTOR_VALIDATION_PENDING', stage='completion-unconfirmed',
                         board_progress=pending_progress(hdc, board, token, requested),
                         next='Inspect local_status and board_progress; resume a suspended launcher in its original shell. '
                              'Recover this same request; do not repeat --apply or remove its locks.')
            return state
        if done != token:
            raise ValueError('completion belongs to another request')
        complete = True; state['device_execution'] = 'FAILED_OUTPUTS_RETRIEVED'
        names = log_names(requested)
        if plan.get('transport') == 'archive-v1':
            progress.update('retrieve-logs', f'Retrieving log archive for {len(requested)} cases')
            _, log_info = hdc.shell('cd '+q(str(board))+' && cat logs.exitcode && wc -c < logs.tar', 'vector-log-size')
            fields = log_info.split()
            if len(fields) != 2 or fields[0] != '0' or not fields[1].isdigit() or not 0 < int(fields[1]) <= 64 << 20:
                raise ValueError('invalid or failed probe log archive')
            if _receive(hdc, board, out, 'logs.tar', timeout_seconds=180) is None:
                raise ValueError('missing probe log archive')
            unpack_logs(out/'logs.tar', out, names)
        else:
            # Legacy attempts retain their original layout; recovery never rewrites them.
            for index, name in enumerate(names, 1):
                if index == 1 or index % 10 == 0:
                    progress.update('retrieve-logs', f'Retrieving legacy logs {index}/{len(names)}: {name}')
                _receive(hdc, board, out, name)
        diagnostics = execution_diagnostics(out, requested)
        state['execution_diagnostics'] = diagnostics
        completed = [name for name in requested if diagnostics['case_exitcodes'][name] == 0]
        if diagnostics['archive_exitcode'] != 0 or not completed:
            raise ValueError('; '.join(diagnostics['errors']))
        fixture = root/'fixture'
        progress.update('verify-fixtures', 'Verifying retained input and reference hashes')
        if sha256_file(fixture/'manifest.json') != plan['fixture_sha256']:
            raise ValueError('retained fixture manifest changed')
        meta = read_json(fixture/'manifest.json')
        if [c['case'] for c in meta['cases']] != requested:
            raise ValueError('request case selection differs from retained fixtures')
        for case in meta['cases']:
            for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
                for spec in case[direction]:
                    path = fixture/case['case']/f'{prefix}_{spec["index"]}.bin'
                    if path.stat().st_size != spec['bytes'] or sha256_file(path) != spec['fixture_sha256']:
                        raise ValueError('retained input/reference changed after launch')
        maximum = 2*REPETITIONS*sum(r['bytes'] for c in meta['cases'] for r in c['outputs'])+(2 << 20)
        _, size = hdc.shell('wc -c < '+q(str(board/'outputs.tar')), 'vector-output-size')
        if not size.isdigit() or not 0 < int(size) < maximum:
            raise ValueError('unexpected output archive size')
        progress.update('retrieve-outputs', f'Retrieving outputs.tar ({size} bytes)')
        if _receive(hdc, board, out, 'outputs.tar', timeout_seconds=180) is None:
            raise ValueError('missing output archive')
        progress.update('compare', f'Extracting and comparing all ten calls for {len(completed)} completed cases')
        extract(out/'outputs.tar', out/'native', meta, selection=completed)
        state.update(compare(fixture, out/'native', selection=completed), stage='complete',
                     device_execution='EXECUTED_OUTPUTS_RETRIEVED')
        if diagnostics['errors']:
            state.update(status='FAILED', device_execution='PARTIAL_OUTPUTS_RETRIEVED',
                numerical_validation=('PARTIAL_PASSED' if state['numerical_validation'] == 'PASSED' else 'FAILED'),
                numerical_scope='COMPLETED_CASES_ONLY', error='; '.join(diagnostics['errors']))
    except Exception as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    finally:
        if complete and hdc and plan:
            try:
                _, owner = hdc.shell('if test ! -e '+q(str(GLOBAL_LOCK))+'; then echo ABSENT; elif '
                    +_guard(GLOBAL_LOCK, plan['token'])+'; then echo OWNED; else echo OTHER; fi', 'vector-lock-owner')
                if owner == 'OWNED':
                    _release_global(hdc, GLOBAL_LOCK, plan['token'])
                elif owner not in ('ABSENT', 'OTHER'):
                    raise ValueError('unknown lock owner')
                state['lock_state'] = 'NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
            except Exception as error:
                state['lock_release_error'] = str(error)
        _save(out/'result.json', state)
        progress.close()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--cases', nargs='+', choices=VALIDATION_CASES, help='Only execute these existing probes, in the supplied order')
    selection.add_argument('--top1-semantics', action='store_true',
        help='reuse nine compiled Top1 OMCs for 61 input cases; no compilation or native ABI change')
    parser.add_argument('--config')
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--recover')
    parser.add_argument('--status', help='Inspect local launcher state and last transport receipt; no HDC, writes or locks')
    parser.add_argument('--analyze', help='Read an existing recovery directory and print numerical details; no board access')
    parser.add_argument('--output')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.status:
        if args.analyze or args.recover or args.compiled or args.config or args.cases or args.top1_semantics or args.apply or args.output:
            parser.error('--status only accepts the original execution directory')
        print(json.dumps(local_status(args.status), ensure_ascii=False, allow_nan=False))
        return 0
    if args.analyze:
        if args.recover or args.compiled or args.config or args.cases or args.top1_semantics or args.apply or args.output:
            parser.error('--analyze only accepts a recovered result directory')
        print(json.dumps(analyze(args.analyze), ensure_ascii=False, allow_nan=False))
        return 0
    if not args.output:
        parser.error('start and recovery require --output')
    if args.recover and (args.apply or args.config or args.compiled or args.cases or args.top1_semantics):
        parser.error('recovery only accepts the original execution and separate output')
    if not args.recover and not all((args.config, args.compiled, args.ddk_root)):
        parser.error('start requires --config, --compiled and CPRO_DDK_ROOT/--ddk-root')
    if Path(args.output).is_symlink():
        raise ValueError('output must not be a symlink')
    out = output_path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent/('.'+out.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = recover(args.recover, out) if args.recover else start(args, out)
    shown = {k: v for k, v in result.items() if k != 'cases'}
    if 'cases' in result and not result.get('top1_semantics'):
        shown['cases'] = [{k: v for k, v in c.items() if k not in ('tensors', 'api_us')} for c in result['cases']]
    print(json.dumps(shown, ensure_ascii=False, allow_nan=False))
    return 1 if result['status'] in ('FAILED', 'NUMERICAL_FAILED') else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}))
        raise SystemExit(1)

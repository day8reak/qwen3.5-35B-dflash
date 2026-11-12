"""Read existing QLinear boundaries, constants and compiler arguments in a batch.

Descriptor observations, exact source bytes and compiler build arguments are
separate evidence. None is a dump of an executing kernel's GM addresses.
"""
from __future__ import annotations

import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.operator_air_review import (
    CHAIN_TYPES, _data_edges, _dtype, _edge, descriptor, port_names)
from qwen35_moe_dflash.target.original_air_inspect import _descriptor_metadata
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
from qwen35_moe_dflash.toolchain.omc_wire import Schema, load_schema
from .common import KIND, checked, identity, new_output, save_json, sha
from .inspection import bounded_bytes, metadata_attributes, retained_ir

LIMIT = 64 << 20
CHAIN = CHAIN_TYPES | {'TransData', 'Permute', 'ReFormat', 'Reformat'}
NUMPY_DTYPES = {'float16': '<f2', 'float32': '<f4', 'int8': 'i1',
                'int16': '<i2', 'int32': '<i4', 'int64': '<i8'}


class SampleSchema(Schema):
    """Keep only a prefix of TensorDef bytes in addition to the strict hash."""
    @staticmethod
    def _scalar(value, field, message):
        result = Schema._scalar(value, field, message)
        if field.kind == 12 and field.name == 'data' and message.endswith('.TensorDef'):
            result['prefix_hex'] = bytes(value[:512]).hex()
        return result


def focus(kind):
    return kind == 'DynamicQuant' or 'QuantBatchMat' in kind


def desc(raw, enums, stage):
    result = descriptor(raw, enums)
    result['dtype_evidence'] = stage + '_DESCRIPTOR_NOT_RUNTIME_BYTES'
    result['shape_evidence'] = ('DESCRIPTOR_DIMENSIONS' if result.get('shape')
        and all(d >= 0 for d in result['shape']) else 'UNRESOLVED_OR_SCALAR_DESCRIPTOR')
    result['dtype'] = _dtype(result)
    result['storage_metadata'] = _descriptor_metadata(raw)
    return result


def graphs(document):
    """Visit embedded GraphDefs, including attribute subgraphs, with bounds."""
    pending, count = [document], 0
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if isinstance(value.get('op'), list):
                count += 1
                if count > 128 or len(value['op']) > 8192:
                    raise ValueError('graph inspection budget exceeded')
                yield value
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def trace(edge, nodes, enums, stage):
    chain, seen = [], set()
    for _ in range(24):
        parsed = _edge(edge)
        if parsed is None or parsed[0] not in nodes or parsed in seen:
            return dict(chain=chain, stop='UNRESOLVED_OR_CYCLE')
        seen.add(parsed)
        name, slot = parsed
        node = nodes[name]
        outputs = node.get('output_desc', [])
        chain.append(dict(name=name, type=node['type'], output_slot=slot,
            descriptor=desc(outputs[slot] if slot < len(outputs) else {}, enums, stage),
            **metadata_attributes(node)))
        if node['type'] not in CHAIN or slot != 0:
            return dict(chain=chain, stop='PRODUCER_REACHED')
        edges = _data_edges(node)
        if not edges or not edges[0]:
            return dict(chain=chain, stop='MISSING_INPUT')
        edge = edges[0]
    return dict(chain=chain, stop='DEPTH_LIMIT')


def sample(blob, dtype):
    """Decode an explicitly typed prefix, never reinterpret FP32 as FP16."""
    raw = bytes.fromhex(blob.get('prefix_hex', ''))
    result = dict(scope='PREFIX_ONLY_NOT_WHOLE_TENSOR', raw_prefix_hex=raw[:32].hex())
    if dtype not in NUMPY_DTYPES:
        return result
    dt = np.dtype(NUMPY_DTYPES[dtype])
    values = np.frombuffer(raw[:len(raw)//dt.itemsize*dt.itemsize], dtype=dt)
    result.update(dtype=dtype, elements_sampled=int(values.size),
        first_values=[v.item() if np.isfinite(v) else str(v) for v in values[:16]],
        nonfinite_in_prefix=int((~np.isfinite(values)).sum()))
    if dtype == 'int32':
        # A byte/nibble preview, explicitly NOT a logical [K,N] unpack.
        nibbles = [n if n < 8 else n-16 for b in raw[:16] for n in (b & 15, b >> 4)]
        result['signed_i4_low_then_high_byte_order_preview'] = nibbles
        result['logical_i4zn_layout_validation'] = 'NOT_ESTABLISHED_BY_PREFIX'
    return result


def constants(graph, schema, folder, expected):
    enums = schema.enums[schema.package + '.DataType']
    result = []
    for node in graph.get('op', []):
        if node.get('type') not in ('Const', 'Constant', 'FileConstant'):
            continue
        row = dict(name=node['name'], type=node['type'], status='UNAVAILABLE')
        result.append(row)
        try:
            attrs = _attributes(node)
            outputs = node.get('output_desc', [])
            raw_desc = outputs[0] if outputs else {}
            if node['type'] == 'FileConstant':
                name = attrs.get('file_path', {}).get('s')
                if not isinstance(name, str):
                    raise ValueError('FileConstant path is not text')
                relative = PurePosixPath(name)
                path = (folder/name).resolve()
                if relative.is_absolute() or '..' in relative.parts or folder not in path.parents:
                    raise ValueError('FileConstant is outside this exported case')
                raw = bounded_bytes(path, LIMIT)
                extent = {k: attrs[k]['i'] for k in ('offset', 'length', 'size', 'file_offset', 'file_size')
                          if isinstance(attrs.get(k), dict) and 'i' in attrs[k]}
                # Only offset/length describe a known unambiguous range here.
                if set(extent) - {'offset', 'length'}:
                    raise ValueError('unsupported FileConstant extent: ' + str(extent))
                offset = extent.get('offset', 0)
                length = extent.get('length', len(raw)-offset)
                if (type(offset) is not int or type(length) is not int
                        or offset < 0 or length < 0 or offset+length > len(raw)):
                    raise ValueError('FileConstant extent outside file')
                payload = raw[offset:offset+length]
                blob = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                            prefix_hex=payload[:512].hex())
                row.update(file=identity(path, folder), extent=dict(offset=offset, length=length),
                           extent_source='EXPLICIT_OR_WHOLE_FILE; MATCH_HASH_BEFORE_IDENTIFYING_WEIGHT')
            else:
                tensors = [a['t'] for a in attrs.values() if isinstance(a, dict) and 't' in a]
                if len(tensors) != 1:
                    raise ValueError('no unique inline TensorDef')
                raw_desc = tensors[0].get('desc', raw_desc)
                blob = tensors[0].get('data', {})
                if not isinstance(blob, dict) or 'sha256' not in blob:
                    raise ValueError('no inline tensor payload')
            matches = [name for name, value in expected.items()
                       if value.get('sha256') == blob['sha256'] and value.get('bytes') == blob['bytes']]
            observed = desc(raw_desc, enums, 'SOURCE_AIR')
            typed = expected[matches[0]]['dtype'] if len(matches) == 1 else observed['dtype']
            row.update(status='READ', descriptor=observed,
                payload={k: blob[k] for k in ('bytes', 'sha256')}, matches_exported_weights=matches,
                sample=sample(blob, typed), sample_dtype_evidence=(
                    'EXACT_EXPORTED_WEIGHT_HASH_AND_RECORDED_DTYPE' if len(matches) == 1 else 'TENSOR_DESCRIPTOR'))
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
    return result


def boundaries(document, schema, stage):
    enums = schema.enums[schema.package + '.DataType']
    result = []
    for graph in graphs(document):
        nodes = {n['name']: n for n in graph['op']}
        if len(nodes) != len(graph['op']):
            raise ValueError('duplicate node names')
        for node in graph['op']:
            if not focus(node['type']):
                continue
            names = port_names(_attributes(node), 'input')
            bindings = []
            for slot, edge in enumerate(_data_edges(node)):
                inputs = node.get('input_desc', [])
                bindings.append(dict(slot=slot, port=names.get(slot), edge=edge,
                    consumer=desc(inputs[slot] if slot < len(inputs) else {}, enums, stage),
                    producer=trace(edge, nodes, enums, stage)))
            result.append(dict(graph=graph.get('name'), name=node['name'], type=node['type'],
                inputs=bindings, outputs=[desc(d, enums, stage) for d in node.get('output_desc', [])],
                **metadata_attributes(node)))
    return result


def build_arguments(path):
    """Extract only complete BuildSingleOp records; damaged interleaving stays unknown."""
    text = bounded_bytes(path, LIMIT).decode('utf-8', errors='replace')
    records, current, start = [], [], None
    def finish():
        if not current:
            return
        joined = ''.join(current)
        kernel = re.search(r'kernelName\[([^]]+)\]', joined)
        if not kernel or not any(s in kernel[1].lower() for s in ('quantbatchmat', 'dynamicquant')):
            return
        row = dict(kernel=kernel[1], line=start, status='UNPARSED')
        records.append(row)
        try:
            if len(joined) > 1 << 20:
                raise ValueError('build argument budget exceeded')
            match = re.search(r'op inputs: (.*), outputs: (.*), attrs: (.*)\.\s*$', joined, re.S)
            if not match:
                raise ValueError('incomplete or interleaved compiler argument record')
            inputs, outputs, attrs = (ast.literal_eval(part) for part in match.groups())
            if not isinstance(inputs, (tuple, list)) or not isinstance(outputs, (tuple, list)):
                raise ValueError('unexpected input/output argument structure')
            row.update(status='PARSED', inputs=inputs, outputs=outputs, attrs=attrs)
        except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
            row['error'] = str(exc)
    for number, line in enumerate(text.splitlines(), 1):
        match = re.search(r'BuildSingleOp\(\d+\)::"(.*)', line)
        if not match:
            continue
        chunk = match[1]
        if 'Single op building:' in chunk:
            finish()
            current, start = [], number
        if start is None:
            continue
        current.append(chunk[:-1] if chunk.endswith('"') else chunk)
        if (', attrs:' in ''.join(current) and current[-1].rstrip().endswith('.')) or len(current) > 512:
            finish()
            current, start = [], None
    finish()
    return dict(records=records, scope='COMPILER_BUILD_ARGUMENTS_NOT_RUNTIME_BUFFER_CONTENTS')


def fixture_binding(path, batch, source, compiled, manifest):
    meta = json.loads(bounded_bytes(path/'manifest.json', 8 << 20))
    expected = dict(artifact_kind=KIND, manifest_sha256=sha(batch/'manifest.json'),
                    compile_sha256=sha(batch/'compile.json'))
    if meta.get('source_evidence') != expected:
        raise ValueError('execution fixture belongs to a different AIR/OMC batch')
    cases = {c['id']: c for c in manifest['cases']}
    result = {}
    for case in meta['cases']:
        name = case['case']
        if name not in cases or cases[name]['kind'] != 'qlinear':
            continue
        exp = json.loads(checked(source, cases[name]['export']).read_text())
        if case['model']['sha256'] != compiled['cases'][name]['omc']['sha256']:
            raise ValueError('fixture model hash differs: ' + name)
        rows = []
        for spec in case['inputs']:
            artifact = checked(path, dict(path=f'{name}/input_{spec["index"]}.bin',
                bytes=spec['fixture_bytes'], sha256=spec['fixture_sha256']))
            captured = [s for s in exp['source_inputs'] if s['fixture_index'] == spec['fixture_index']]
            if len(captured) != 1 or any(spec[k] != captured[0][k] for k in ('shape', 'dtype')):
                raise ValueError('fixture input slot/shape/dtype mismatch: ' + name)
            if sha(artifact) != captured[0]['sha256']:
                raise ValueError('fixture input bytes differ from export: ' + name)
            rows.append(dict(slot=spec['index'], fixture_index=spec['fixture_index'], status='EXACT_SOURCE_BYTES'))
        result[name] = rows
    return dict(status='BOUND', cases=result,
                scope='PREPARED_HOST_FIXTURE; ACTUAL_DEVICE_UPLOAD_AND_INTERNAL_BUFFERS_NOT_OBSERVED')


def inspect_qlinear(batch, output, ddk_root, fixture=None):
    root, ddk = Path(batch).resolve(), Path(ddk_root).resolve()
    compiled = json.loads(bounded_bytes(root/'compile.json', 8 << 20))
    manifest = json.loads(bounded_bytes(root/'manifest.json', 8 << 20))
    source = (root/compiled.get('source_batch', '.')).resolve()
    if (manifest.get('artifact_kind') != KIND or compiled.get('artifact_kind') != KIND
            or compiled['manifest_sha256'] != sha(root/'manifest.json')
            or compiled['manifest_sha256'] != sha(source/'manifest.json')):
        raise ValueError('compile/AIR manifest binding changed')
    selected = [c for c in manifest['cases'] if c['kind'] == 'qlinear']
    if not selected or len({c['id'] for c in selected}) != len(selected):
        raise ValueError('requires distinct QLinear cases')
    out = output_path(output)
    fixture = Path(fixture).resolve() if fixture else None
    for original in (root, source, ddk, *([fixture] if fixture else [])):
        if out == original or original in out.parents or out in original.parents:
            raise ValueError('inspection output must be separate from input artifacts/DDK')
    out = new_output(out)
    report = dict(artifact_kind=KIND+'-qlinear-contracts', status='INSPECTED',
        manifest=identity(root/'manifest.json', root), compilation=identity(root/'compile.json', root),
        coverage='ALL_QLinear_CASES', cases={}, model_compilation='NOT_RUN', device_execution='NOT_RUN',
        numerical_validation='NOT_RUN', runtime_matmul_input_bytes='NOT_OBSERVED',
        abi_hypothesis='USER_KERNEL_REVIEW: INT8 x1, INT32 packed W4, FP16 scale/per-token; selected binary ABI unverified')
    compiler_schema = None
    try:
        compiler_schema = load_schema(ddk/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
        report['compiler_schema'] = compiler_schema.evidence
    except Exception as exc:
        report['compiler_schema_error'] = f'{type(exc).__name__}: {exc}'
        report['status'] = 'PARTIAL'
    if fixture:
        try:
            report['fixture_binding'] = fixture_binding(fixture, root, source, compiled, manifest)
        except Exception as exc:
            report.update(status='PARTIAL', fixture_binding=dict(status='UNBOUND', error=str(exc)))
    csv_rows = []
    for case in selected:
        name = case['id']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid case name')
        row = report['cases'][name] = dict(boundary={k: case.get(k) for k in ('layer', 'phase', 'point')},
            compilation_status=compiled['cases'].get(name, {}).get('status'), stages={}, errors=[])
        folder, job = source/'models'/name, root/'models'/name/'omc'
        try:
            exp = json.loads(checked(source, case['export']).read_text())
            air = checked(folder, exp['air'])
            schema_file = folder/'air-schema.pb'
            schema = SampleSchema(bounded_bytes(schema_file, 8 << 20),
                expected_file=exp['schema']['schema_file'], expected_package=exp['schema']['schema_package'])
            _, document, graph = _decode(schema, bounded_bytes(air, LIMIT))
            row['source_identity'] = dict(air=identity(air, folder), schema=identity(schema_file, folder),
                schema_binding='RETAINED_DESCRIPTOR; LEGACY_EXPORT_NO_DESCRIPTOR_HASH')
            row['export_evidence'] = {k: exp.get(k) for k in ('source_contract', 'qlinear_factory',
                'target_weights', 'captured_weights', 'exported_weights', 'source_files', 'torchair_version')}
            row['stages']['SOURCE_AIR'] = boundaries(document, schema, 'SOURCE_AIR')
            row['constants'] = constants(graph, schema, folder, exp.get('exported_weights', {}))
            row['input_files'] = []
            for spec in exp['source_inputs']:
                path = checked(folder, spec)
                if spec['dtype'] in NUMPY_DTYPES and path.stat().st_size != (
                        math.prod(spec['shape']) * np.dtype(NUMPY_DTYPES[spec['dtype']]).itemsize):
                    raise ValueError('input extent differs from shape/dtype: ' + spec['name'])
                with path.open('rb') as stream:
                    prefix = stream.read(512)
                row['input_files'].append(dict(spec, sample=sample(dict(prefix_hex=prefix.hex()), spec['dtype'])))
            # A match is full byte identity, not a proof of the consumer's layout.
            row['weight_matches'] = {w: [c['name'] for c in row['constants']
                if w in c.get('matches_exported_weights', [])] for w in exp.get('exported_weights', {})}
            row['weight_port_bindings'] = []
            constant_by_name = {c['name']: c for c in row['constants']}
            for node in row['stages']['SOURCE_AIR']:
                if 'QuantBatchMat' not in node['type']:
                    continue
                for port in node['inputs']:
                    expected_weight = {'x2': 'W_q', 'scale': 'scale'}.get(port['port'])
                    if not expected_weight:
                        continue
                    chain = port['producer']['chain']
                    constant = constant_by_name.get(chain[-1]['name'], {}) if chain else {}
                    matches = constant.get('matches_exported_weights', [])
                    status = ('EXACT_SOURCE_WEIGHT_BYTES' if expected_weight in matches else
                        'SOURCE_WEIGHT_BYTES_DIFFER' if constant.get('status') == 'READ' else 'UNRESOLVED')
                    row['weight_port_bindings'].append(dict(node=node['name'], port=port['port'],
                        expected_weight=expected_weight, constant=constant.get('name'), status=status,
                        transformations=[n['type'] for n in chain[:-1]],
                        scope='BEFORE_LISTED_TRANSFORMATIONS; NOT_COMPILED_CONSUMER_STORAGE'))
        except Exception as exc:
            row['errors'].append('source: ' + str(exc))
        try:
            if row['compilation_status'] == 'COMPILED' and compiler_schema:
                model = checked(root, compiled['cases'][name]['omc'])
                part = next(p for p in read_container(model)['partitions'] if p['type'] == 0)
                doc = compiler_schema.decode(read_partition(model, part), 'ModelDef')
                row['omc'] = identity(model, root)
                row['stages']['OMC_METADATA'] = boundaries(doc, compiler_schema, 'OMC_METADATA')
                row['omc_visibility'] = ('FOCUS_NODES_VISIBLE' if row['stages']['OMC_METADATA']
                    else 'OPAQUE_OR_FUSED; NO_INTERNAL_DTYPE_CONCLUSION')
        except Exception as exc:
            row['errors'].append('OMC metadata: ' + str(exc))
        try:
            inventory, dumps = retained_ir(job)
            row['ir_inventory'] = inventory
            row['ir_uninspected'] = [str(p.relative_to(root)) for p, _ in dumps[12:]]
            row['ir_errors'] = []
            if compiler_schema:
                for path, binding in dumps[:12]:
                    try:
                        if binding:
                            path = checked(path.parent, binding)
                        _, doc, _ = _decode(compiler_schema, bounded_bytes(path, LIMIT))
                        stage = 'IR:' + str(path.relative_to(job))
                        row['stages'][stage] = boundaries(doc, compiler_schema, 'IR')
                    except Exception as exc:
                        row['ir_errors'].append(dict(path=str(path), error=str(exc)))
        except Exception as exc:
            row['errors'].append('IR inventory: ' + str(exc))
        try:
            log = job/'compile.log'
            if log.is_file():
                row['compiler_arguments'] = dict(log=identity(log, root), **build_arguments(log))
            else:
                row['compiler_arguments'] = dict(records=[], status='LOG_NOT_FOUND')
            receipt = job/'report.json'
            if receipt.is_file():
                meta = json.loads(bounded_bytes(receipt, 8 << 20))
                row['compiler'] = {k: meta.get(k) for k in ('ddk_id', 'omg_path', 'omg_sha256',
                    'requested_options', 'compiler_profile', 'tbe_compatibility', 'returncode')}
        except Exception as exc:
            row['errors'].append('compiled: ' + str(exc))
        if row['errors']:
            report['status'] = 'PARTIAL'
        for stage, nodes in row['stages'].items():
            for node in nodes:
                for port in node['inputs']:
                    d = port['consumer']
                    chain = port['producer']['chain']
                    csv_rows.append(dict(case=name, phase=case['phase'], stage=stage, node=node['name'],
                        op=node['type'], slot=port['slot'], port=port['port'], dtype=d.get('dtype'),
                        shape=json.dumps(d.get('shape')), format=d.get('layout'),
                        producer_chain=' <- '.join(f"{v['name']}:{v['output_slot']}({v['type']},{v['descriptor'].get('dtype')},{v['descriptor'].get('layout')})" for v in chain)))
        for record in row.get('compiler_arguments', {}).get('records', []):
            if record['status'] == 'PARSED':
                for slot, d in enumerate(record['inputs']):
                    if not isinstance(d, dict):
                        continue
                    csv_rows.append(dict(case=name, phase=case['phase'], stage='COMPILER_BUILD_ARGUMENTS',
                        node=record['kernel'], op='BUILD_ARGUMENTS', slot=slot, port='', dtype=d.get('dtype'),
                        shape=json.dumps(d.get('shape')), format=d.get('format'), producer_chain='NOT_IN_BUILD_LOG'))
        save_json(out/'report.json', report)
    with (out/'ports.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'phase', 'stage', 'node', 'op', 'slot', 'port', 'dtype', 'shape', 'format', 'producer_chain'])
        writer.writeheader()
        writer.writerows(csv_rows)
    coverage = dict(cases=len(selected),
        source_cases=sum(bool(r['stages'].get('SOURCE_AIR')) for r in report['cases'].values()),
        parsed_build_argument_cases=sum(any(x['status'] == 'PARSED' for x in
            r.get('compiler_arguments', {}).get('records', [])) for r in report['cases'].values()),
        visible_omc_focus_cases=sum(r.get('omc_visibility') == 'FOCUS_NODES_VISIBLE' for r in report['cases'].values()),
        weight_ports=dict(Counter(p['status'] for r in report['cases'].values() for p in r.get('weight_port_bindings', []))))
    report['inspection_coverage'] = coverage
    save_json(out/'report.json', report)
    lines = ['Read-only QLinear inspection. No export, compile or device execution.',
        f'cases={len(selected)}; phases={dict(Counter(c["phase"] for c in selected))}',
        'inspection_coverage=' + json.dumps(coverage),
        'SOURCE_AIR / IR / OMC_METADATA / COMPILER_BUILD_ARGUMENTS are distinct evidence.',
        'Runtime MatMul inputs NOT dumped. No kernel correctness or ABI PASS inferred.']
    if 'fixture_binding' in report:
        lines.append('fixture_binding=' + report['fixture_binding']['status'])
    for name, row in report['cases'].items():
        lines += ['', name + ': ' + str(row['compilation_status']),
            'weight_matches=' + json.dumps(row.get('weight_matches', {})),
            'weight_port_bindings=' + json.dumps(row.get('weight_port_bindings', [])),
            'omc_visibility=' + row.get('omc_visibility', 'NOT_INSPECTED')]
        if not row.get('compiler_arguments', {}).get('records'):
            lines.append('BUILD_ARGUMENTS_NOT_FOUND; NOT_A_DTYPE_PASS')
        for port in [p for p in csv_rows if p['case'] == name and p['stage'] in ('SOURCE_AIR', 'COMPILER_BUILD_ARGUMENTS')]:
            lines.append(f"{port['stage']} {port['op']} input[{port['slot']}] {port['port'] or ''}: {port['dtype']} {port['shape']} {port['format']} | {port['producer_chain']}")
        for rec in row.get('compiler_arguments', {}).get('records', []):
            if rec['status'] != 'PARSED':
                lines.append(f"BUILD_LOG_UNPARSED L{rec['line']}: {rec['error']}")
        lines.extend(row['errors'])
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    return dict(status=report['status'], cases=len(selected), report=str(out/'report.json'),
                ports=str(out/'ports.csv'), summary=str(out/'summary.txt'),
                model_compilation='NOT_RUN', device_execution='NOT_RUN')

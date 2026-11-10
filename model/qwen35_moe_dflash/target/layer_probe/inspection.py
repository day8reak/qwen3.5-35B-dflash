"""Read retained small AIR/IR metadata after compilation; never execute a toolchain."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.common.tool_logs import log_excerpt
from qwen35_moe_dflash.target.operator_air_review import descriptor
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode, _digest
from qwen35_moe_dflash.toolchain.omc_wire import Schema, load_schema
from .common import KIND, checked, identity, new_output, save_json, sha


def bounded_bytes(path, limit):
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError(f'missing or excessive metadata file: {path}')
    with path.open('rb') as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f'metadata exceeds read budget: {path}')
    return value


def metadata_attributes(value):
    """Retain small scalar/list attributes, never tensor or nested graph payloads."""
    attrs = _attributes(value)
    metadata = {k: v for k, v in attrs.items()
        if isinstance(v, dict) and set(v) <= {'i', 'f', 'b', 's', 'list'}
        and ('list' not in v or isinstance(v['list'], dict)
             and set(v['list']) <= {'i', 'f', 'b', 's', 'val_type'})
        and len(json.dumps(v)) <= 4096}
    return dict(attributes=metadata, omitted_attributes=sorted(set(attrs)-set(metadata)))


def graph_metadata(path, schema, *, focus_type=None, raw=None, compare_interpretation=False):
    raw = bounded_bytes(path, 64 << 20) if raw is None else raw
    kind, decoded, _ = _decode(schema, raw)
    enums = schema.enums[schema.package + '.DataType']
    graphs = []

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('op'), list):
                if len(graphs) >= 32 or len(value['op']) > 4096:
                    raise ValueError('not a bounded region graph')
                nodes = []
                for node in value['op']:
                    nodes.append(dict(name=node.get('name'), type=node.get('type'),
                        inputs=node.get('input', []), **metadata_attributes(node),
                        input_desc=[descriptor(d, enums) for d in node.get('input_desc', [])],
                        output_desc=[descriptor(d, enums) for d in node.get('output_desc', [])]))
                counts = dict(Counter(n['type'] for n in nodes))
                total = len(nodes)
                if focus_type:
                    focused = {n['name'] for n in nodes if n['type'] == focus_type}
                    selected = set(focused)
                    for node in nodes:
                        producers = {edge.rsplit(':', 1)[0] for edge in node['inputs'] if edge}
                        if node['name'] in focused:
                            selected.update(producers)
                        if producers & focused:
                            selected.add(node['name'])
                    nodes = [n for n in nodes if n['name'] in selected]
                graphs.append(dict(name=value.get('name'), inputs=value.get('input', []),
                    outputs=value.get('output', []), **metadata_attributes(value),
                    type_counts=counts, nodes=nodes, nodes_omitted=total-len(nodes)))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(decoded)
    result = dict(status='INSPECTED', format=kind, graphs=graphs,
        decoded_file=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        model_attributes=metadata_attributes(decoded) if kind == 'ModelDef' else None,
        node_selection='ALL' if focus_type is None else f'{focus_type}_AND_DIRECT_NEIGHBORS',
        scope='OBSERVED_DESCRIPTORS; EMPTY_SHAPES_UNRESOLVED; NO_TENSOR_PAYLOADS_OR_GRAPH_EDITS')
    if compare_interpretation:
        # Compare the entire strict decode, including attributes omitted from
        # the public summary, plus the schema's names for observed port dtypes.
        result['interpretation_sha256'] = _digest(dict(format=kind, decoded=decoded, graphs=graphs))
    return result


def reference_metadata(path, candidates):
    """Try declared schemas, retaining failures and refusing conflicting interpretations."""
    raw = bounded_bytes(path, 64 << 20)
    result = dict(status='UNAVAILABLE',
        decoded_file=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        schema_attempts=[])
    compatible = []
    for key, (schema, evidence) in candidates.items():
        attempt = dict(candidate=key, schema=evidence)
        result['schema_attempts'].append(attempt)
        try:
            metadata = graph_metadata(path, schema, raw=raw,
                focus_type='LinearAttentionConvstateUpdate', compare_interpretation=True)
            attempt.update(status='COMPATIBLE', interpretation_sha256=metadata['interpretation_sha256'])
            compatible.append((key, metadata))
        except Exception as exc:
            attempt.update(status='REJECTED', error=f'{type(exc).__name__}: {exc}')
    if not compatible:
        result['error'] = 'No declared schema strictly decoded the reference; see schema_attempts.'
    elif len({m['interpretation_sha256'] for _, m in compatible}) != 1:
        result['error'] = 'Declared schemas produced conflicting interpretations; no graph selected.'
    else:
        key, metadata = compatible[0]
        result.update(metadata, selected_schema=key,
            compatible_schemas=[k for k, _ in compatible],
            schema_selection='UNIQUE_COMPATIBLE' if len(compatible) == 1 else 'EQUIVALENT_INTERPRETATIONS',
            schema_binding='STRUCTURAL_COMPATIBILITY_ONLY; REFERENCE_PRODUCER_UNVERIFIED')
    return result


def retained_ir(job):
    """Prefer the compiler's retained, hash-bound inventory over scratch paths."""
    folders = [job/'diagnostics', job/'compile-work/generated/dump_graph']
    files = [job/'diagnostics/dump_graph', job/'compile-work/generated/dump_graph']
    inventory = dict(searched_directories=[str(p.relative_to(job)) for p in folders],
        searched_files=[str(p.relative_to(job)) for p in files],
        dump_graph_path_kind=('SYMLINK' if files[1].is_symlink() else 'FILE' if files[1].is_file()
                              else 'DIRECTORY' if files[1].is_dir() else 'ABSENT'), skipped=[])
    manifest = folders[0]/'manifest.json'
    if manifest.is_file():
        retained = json.loads(bounded_bytes(manifest, 8 << 20))
        inventory.update(manifest=identity(manifest, job), retention_status=retained.get('status'))
        records = []
        for item in retained['files']:
            if item.get('role') != 'graph-dump':
                continue
            name = item['name']
            if Path(name).name != name or not (name.endswith('.irpb') or name == 'dump_graph'):
                raise ValueError('invalid retained IR name: ' + name)
            if item['status'] != 'RETAINED':
                inventory['skipped'].append({k: item[k] for k in ('name', 'status', 'bytes')})
                continue
            records.append((folders[0]/name, dict(path=name, bytes=item['bytes'], sha256=item['sha256'])))
        # Old retention manifests may contain a pre-check report but omit this
        # extensionless file entirely. Do not bypass an explicit budget skip.
        if not any(item.get('name') == 'dump_graph' for item in retained['files']):
            records.extend((p, None) for p in files if p.is_file() and not p.is_symlink())
    else:
        # Older runs may have only loose diagnostics or the original scratch files.
        records = [(p, None) for p in [*files, *(p for folder in folders if folder.is_dir()
                                                for p in sorted(folder.glob('*.irpb')))]
                   if p.is_file() and not p.is_symlink()]
    records.sort(key=lambda item: (0 if 'fail' in item[0].name else
                                   1 if item[0].name.startswith('05_') else 2, str(item[0])))
    inventory['status'] = ('PARTIAL' if inventory['skipped'] else 'FOUND') if records else 'NOT_FOUND'
    return inventory, records


def convstate_context(metadata):
    """Compact first-node context per graph; the full evidence stays in report.json."""
    contexts = []
    for graph in metadata.get('graphs', []):
        nodes = graph['nodes']
        node = next((n for n in nodes if n['type'] == 'LinearAttentionConvstateUpdate'), None)
        if node is None:
            continue
        producers = {edge.rsplit(':', 1)[0] for edge in node['inputs'] if edge}
        neighbors = [dict(name=n['name'], type=n['type']) for n in nodes
            if n['name'] in producers or any(edge.rsplit(':', 1)[0] == node['name']
                                             for edge in n['inputs'] if edge)]
        contexts.append(dict(graph=graph['name'], graph_attributes=graph['attributes'],
            model_attributes=metadata.get('model_attributes'),
            node=node, neighbors=neighbors,
            scope='FIRST_CONVSTATE_PER_GRAPH; SEE_REPORT_FOR_ALL_NODES'))
    return contexts


ENVIRONMENT_KEYS = ('PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME',
    'DDK_PATH', 'TOOLCHAIN_HOME', 'HIAI_VERSION', 'CCE_LIB_DIR', 'SOC_VERSION', 'KIRIN_CONF_JSON_PATH',
    'ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH', 'ASCEND_CUSTOM_OPP_PATH',
    'ASCEND_AICPU_PATH', 'CANN_ROOT', 'TMPDIR', 'TMP', 'TEMP', 'PYTHONDONTWRITEBYTECODE')


def compiler_entry_metadata(ddk, inherited=None):
    """Read exact entry files and compare environment construction without executing them."""
    from qwen35_moe_dflash.toolchain.compile import omg_environment
    ddk = Path(ddk).resolve()
    inherited = dict(os.environ if inherited is None else inherited)
    result = dict(status='INSPECTED', scripts={}, wrapper_execution='NOT_RUN',
        environment_source='CURRENT_PROCESS; NOT_A_RECORD_OF_THE_HISTORIC_SUCCESSFUL_RUN')
    for relative in ('tools/tools_ascendc/set_ascendc_env.sh', 'tools/tools_omg/omg'):
        path = ddk/relative
        try:
            with path.open('rb') as stream:
                signature = stream.read(4)
            item = dict(identity(path, ddk), resolved_path=str(path.resolve()))
            if signature == b'\x7fELF':
                item['format'] = 'ELF'
            else:
                item.update(format='TEXT', text=bounded_bytes(path, 256 << 10).decode('utf-8'))
            result['scripts'][relative] = item
        except Exception as exc:
            result['scripts'][relative] = dict(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}')
            result['status'] = 'PARTIAL'
    result['current_environment'] = {k: inherited[k] for k in ENVIRONMENT_KEYS if k in inherited}
    try:
        binary, prepared = omg_environment(ddk, inherited=inherited)
        result['probe_binary'] = identity(binary, ddk)
        result['probe_prepared_environment'] = {k: prepared[k] for k in ENVIRONMENT_KEYS if k in prepared}
        result['probe_environment_changes'] = {k: dict(inherited=inherited.get(k), prepared=prepared.get(k))
            for k in ENVIRONMENT_KEYS if inherited.get(k) != prepared.get(k)}
        result['scope'] = 'BASE_ENVIRONMENT_ONLY; RUN_LOCAL_TBE_AND_IPC_ADAPTATIONS_NOT_EXECUTED'
    except Exception as exc:
        result.update(status='PARTIAL', preparation_error=f'{type(exc).__name__}: {exc}')
    return result


def inspect_compilation(batch, output, cases=None, *, ddk_root=None, compiler_entry=False,
                        reference_ir=None):
    root = Path(batch).resolve()
    compiled = json.loads((root/'compile.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    source = (root/compiled.get('source_batch', '.')).resolve()
    if (compiled.get('artifact_kind') != KIND or manifest.get('artifact_kind') != KIND
            or compiled['manifest_sha256'] != sha(root/'manifest.json')
            or compiled['manifest_sha256'] != sha(source/'manifest.json')):
        raise ValueError('compile/AIR manifest binding changed')
    by_name = {c['id']: c for c in manifest['cases']}
    if cases is None:
        cases = [name for name in by_name if compiled['cases'].get(name, {}).get('status') == 'FAILED']
    if (not cases or len(set(cases)) != len(cases) or set(cases)-set(by_name)
            or any(Path(name).name != name or name in ('.', '..') for name in cases)):
        raise ValueError('select distinct cases from the AIR manifest')
    if compiler_entry and not ddk_root:
        raise ValueError('compiler-entry inspection requires the selected DDK root')
    references = [Path(p).resolve() for p in (reference_ir or [])]
    if references and (not ddk_root or len(references) > 8 or len(set(references)) != len(references)):
        raise ValueError('reference IR requires the selected DDK and at most 8 distinct files')
    output = output_path(output)
    if any(p == output or output in p.parents for p in references):
        raise ValueError('reference IR must be outside the new inspection directory')
    for original in (root, source, *([Path(ddk_root).resolve()] if ddk_root else [])):
        if output == original or original in output.parents or output in original.parents:
            raise ValueError('inspection output must be separate from source, compiled batch and DDK')
    output = new_output(output)
    result = dict(artifact_kind=KIND+'-compile-inspection', status='INSPECTED',
        manifest=identity(root/'manifest.json', root), compilation=identity(root/'compile.json', root),
        cases={}, model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN')
    compiler_schema = None
    reference_schemas = {}
    if compiler_entry:
        result['compiler_entry'] = compiler_entry_metadata(ddk_root)
        if result['compiler_entry']['status'] != 'INSPECTED':
            result['status'] = 'PARTIAL'
    if ddk_root:
        try:
            compiler_schema = load_schema(Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
            result['compiler_schema'] = compiler_schema.evidence
            reference_schemas[compiler_schema.evidence.get('descriptor_sha256', 'selected-ddk')] = (
                compiler_schema, dict(role='SELECTED_DDK', **compiler_schema.evidence))
        except Exception as exc:
            result['compiler_schema_error'] = f'{type(exc).__name__}: {exc}'
            result['status'] = 'PARTIAL'
    for name in cases:
        row = result['cases'][name] = dict(compilation=compiled['cases'].get(name, {}),
            boundary={k: by_name[name].get(k) for k in ('kind', 'point', 'phase', 'output_name')})
        try:
            exported = json.loads(checked(source, by_name[name]['export']).read_text())
            folder = source/'models'/name
            air = checked(folder, exported['air'])
            schema_path = folder/'air-schema.pb'
            schema_bytes = bounded_bytes(schema_path, 8 << 20)
            schema = Schema(schema_bytes,
                expected_file=exported['schema']['schema_file'], expected_package=exported['schema']['schema_package'])
            row.update(source_inputs=exported['source_inputs'], golden=exported['golden'],
                source_air=identity(air, folder), source_schema=identity(schema_path, folder),
                source_graph=graph_metadata(air, schema))
            row['export_evidence'] = {k: exported.get(k) for k in
                ('export_adjustments', 'export_graph_check', 'source_files', 'torch_version', 'torchair_version')}
            row['source_schema_binding'] = 'CURRENT_RETAINED_DESCRIPTOR; OLDER_EXPORT_HAS_NO_SCHEMA_HASH'
            if references:
                key = hashlib.sha256(schema_bytes).hexdigest()
                if key not in reference_schemas:
                    # At most the DDK plus eight distinct retained producers.
                    if sum(e['role'] == 'RETAINED_AIR_PRODUCER' for _, e in reference_schemas.values()) >= 8:
                        result.setdefault('untried_reference_schema_cases', []).append(name)
                    else:
                        reference_schemas[key] = (schema, dict(role='RETAINED_AIR_PRODUCER',
                            path=str(schema_path), descriptor_sha256=key,
                            descriptor_bytes=len(schema_bytes),
                            schema_file=exported['schema']['schema_file'],
                            schema_package=exported['schema']['schema_package'],
                            binding=row['source_schema_binding']))
        except Exception as exc:
            row['source_error'] = f'{type(exc).__name__}: {exc}'
            result['status'] = 'PARTIAL'
        job = root/'models'/name/'omc'
        receipt = job/'report.json'
        if receipt.is_file():
            report = json.loads(bounded_bytes(receipt, 8 << 20))
            row['compiler_report'] = dict(identity(receipt, root),
                **{k: report.get(k) for k in ('stage', 'returncode', 'error', 'requested_options', 'ddk_id',
                                             'omg_path', 'omg_sha256', 'compiler_profile', 'compiler_entry',
                                             'platform', 'tbe_compatibility', 'compiler_diagnostics')})
        environment = job/'omg-environment.json'
        if compiler_entry and environment.is_file():
            prior = json.loads(bounded_bytes(environment, 8 << 20)).get('environment', {})
            row['recorded_compiler_environment'] = dict(identity(environment, root),
                environment={k: prior[k] for k in ENVIRONMENT_KEYS if k in prior},
                unrecorded_keys=[k for k in ENVIRONMENT_KEYS if k not in prior],
                scope='RETAINED_FAILED_INVOCATION; UNRECORDED_KEYS_ARE_UNKNOWN')
        log = job/'compile.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - (24 << 10)))
                tail = stream.read(24 << 10).decode('utf-8', errors='replace')
            row['log'] = dict(identity(log, root), tail=tail, tail_bytes_limit=24 << 10)
            row['log']['diagnostic_excerpt'] = log_excerpt(log)
        try:
            row['ir_inventory'], dumps = retained_ir(job)
        except Exception as exc:
            row['ir_inventory'], dumps = dict(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}'), []
        row['ir_dumps'], row['uninspected_ir_dumps'] = [], [str(p.relative_to(root)) for p, _ in dumps[12:]]
        if row['ir_inventory']['status'] != 'FOUND' or len(dumps) > 12 or not compiler_schema:
            result['status'] = 'PARTIAL'
        for path, binding in dumps[:12]:
            item = dict(path=str(path.relative_to(root)), status='NOT_INSPECTED_NO_DDK_SCHEMA')
            row['ir_dumps'].append(item)
            try:
                if binding is not None:
                    path = checked(path.parent, binding)
                item.update(identity(path, root), retention_binding='SHA256_VERIFIED' if binding else 'UNBOUND_LEGACY_FILE')
                if compiler_schema:
                    item.update(graph_metadata(path, compiler_schema))
            except Exception as exc:
                item.update(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}')
                result['status'] = 'PARTIAL'
        save_json(output/'report.json', result)
    if references:
        result['reference_ir'] = []
        for path in references:
            item = dict(path=str(path), provenance='USER_SELECTED; HISTORIC_RUN_BINDING_UNVERIFIED')
            result['reference_ir'].append(item)
            try:
                if result.get('untried_reference_schema_cases'):
                    raise ValueError('reference schema candidate budget exceeded; select fewer cases')
                item.update(reference_metadata(path, reference_schemas))
            except Exception as exc:
                item.update(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}')
            if item['status'] != 'INSPECTED':
                result['status'] = 'PARTIAL'
        save_json(output/'report.json', result)
    # One compact index for a whole failed batch. Full source/IR and unshortened
    # bounded error contexts remain in report.json; these hints are not causes.
    lines = ['Read-only compile inspection; no export, OMG or device execution.',
             'Group by boundary and actual errors; a candidate rejection alone is not a root cause.', '']
    for item in result.get('reference_ir', []):
        lines.append(f'[reference IR] {item["path"]}: {item["status"]}')
        for attempt in item.get('schema_attempts', []):
            evidence = attempt['schema']
            lines.append('schema=' + json.dumps(dict(candidate=attempt['candidate'],
                role=evidence['role'], package=evidence.get('schema_package'),
                status=attempt['status'], error=attempt.get('error')), ensure_ascii=False))
        if item.get('selected_schema'):
            lines.append('schema_selection=' + item['schema_selection'] +
                         '; schema_binding=' + item['schema_binding'])
        if item.get('error'):
            lines.append(item['error'])
        for graph in item.get('graphs', []):
            lines.append('graph=' + str(graph.get('name')) + '; attributes=' +
                         json.dumps(graph['attributes'], ensure_ascii=False))
            lines.append('ConvState_nodes=' + str(graph['type_counts'].get('LinearAttentionConvstateUpdate', 0)) +
                         '; selected_nodes=' + json.dumps([dict(name=n['name'], type=n['type']) for n in graph['nodes']]))
        lines.append('convstate_context=' + json.dumps(convstate_context(item), ensure_ascii=False))
        lines.append('Full descriptors, graph/model attributes and direct neighbors: report.json; no fusion inferred.')
        lines.append('')
    for name, row in result['cases'].items():
        lines.append(f'[{name}] {json.dumps(row["boundary"], ensure_ascii=False)}')
        evidence = row.get('export_evidence', {})
        adjustments = evidence.get('export_adjustments')
        kinds = [a.get('kind') for a in adjustments] if isinstance(adjustments, list) else 'NOT_RECORDED'
        lines.append('export_adjustments=' + json.dumps(kinds) +
                     '; export_graph_check=' + json.dumps(evidence.get('export_graph_check')))
        graphs = row.get('source_graph', {}).get('graphs', [])
        lines.append('source_type_counts=' + json.dumps([g['type_counts'] for g in graphs]))
        for context_label, metadata in [('source', row.get('source_graph', {})),
                                        *[(item['path'], item) for item in row['ir_dumps']]]:
            contexts = convstate_context(metadata)
            if contexts:
                lines.append('convstate_context=' + json.dumps(dict(source=context_label, graphs=contexts),
                                                              ensure_ascii=False))
        report = row.get('compiler_report', {})
        lines.append(f'stage={report.get("stage")}; returncode={report.get("returncode")}')
        excerpt = row.get('log', {}).get('diagnostic_excerpt', '')
        hints = excerpt.split('Log tail:\n', 1)[0].splitlines()
        # Keep both ends: an exception's final line can follow many frames.
        chosen = hints if len(hints) <= 10 else hints[:5] + ['... see report.json ...'] + hints[-5:]
        lines.extend(line[:500] + (' [see report.json]' if len(line) > 500 else '') for line in chosen)
        if not hints:
            lines.append('No prioritized excerpt; inspect log.tail in report.json.')
        if row.get('source_error'):
            lines.append('source_error=' + row['source_error'])
        lines.append('')
    (output/'summary.txt').write_text('\n'.join(lines) + '\n')
    return result

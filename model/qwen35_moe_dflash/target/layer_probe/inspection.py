"""Read retained small AIR/IR metadata after compilation; never execute a toolchain."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.target.operator_air_review import descriptor
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
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


def graph_metadata(path, schema):
    kind, decoded, _ = _decode(schema, bounded_bytes(path, 64 << 20))
    enums = schema.enums[schema.package + '.DataType']
    graphs = []

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('op'), list):
                if len(graphs) >= 32 or len(value['op']) > 4096:
                    raise ValueError('not a bounded region graph')
                nodes = []
                for node in value['op']:
                    attrs = _attributes(node)
                    # Tensor payloads and nested graph attributes are never copied
                    # into the report. Nested graphs are inspected separately.
                    metadata = {k: v for k, v in attrs.items()
                        if isinstance(v, dict) and set(v) <= {'i', 'f', 'b', 's', 'list'}
                        and ('list' not in v or isinstance(v['list'], dict)
                             and set(v['list']) <= {'i', 'f', 'b', 's', 'val_type'})
                        and len(json.dumps(v)) <= 4096}
                    nodes.append(dict(name=node.get('name'), type=node.get('type'),
                        inputs=node.get('input', []), attributes=metadata,
                        omitted_attributes=sorted(set(attrs)-set(metadata)),
                        input_desc=[descriptor(d, enums) for d in node.get('input_desc', [])],
                        output_desc=[descriptor(d, enums) for d in node.get('output_desc', [])]))
                graphs.append(dict(name=value.get('name'), inputs=value.get('input', []),
                    outputs=value.get('output', []), type_counts=dict(Counter(n['type'] for n in nodes)), nodes=nodes))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(decoded)
    return dict(status='INSPECTED', format=kind, graphs=graphs,
        scope='OBSERVED_DESCRIPTORS; EMPTY_SHAPES_UNRESOLVED; NO_TENSOR_PAYLOADS_OR_GRAPH_EDITS')


def inspect_compilation(batch, output, cases, *, ddk_root=None):
    root = Path(batch).resolve()
    compiled = json.loads((root/'compile.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    source = (root/compiled.get('source_batch', '.')).resolve()
    if (compiled.get('artifact_kind') != KIND or manifest.get('artifact_kind') != KIND
            or compiled['manifest_sha256'] != sha(root/'manifest.json')
            or compiled['manifest_sha256'] != sha(source/'manifest.json')):
        raise ValueError('compile/AIR manifest binding changed')
    by_name = {c['id']: c for c in manifest['cases']}
    if (not cases or len(set(cases)) != len(cases) or set(cases)-set(by_name)
            or any(Path(name).name != name or name in ('.', '..') for name in cases)):
        raise ValueError('select distinct cases from the AIR manifest')
    output = output_path(output)
    for original in (root, source, *([Path(ddk_root).resolve()] if ddk_root else [])):
        if output == original or original in output.parents or output in original.parents:
            raise ValueError('inspection output must be separate from source, compiled batch and DDK')
    output = new_output(output)
    result = dict(artifact_kind=KIND+'-compile-inspection', status='INSPECTED',
        manifest=identity(root/'manifest.json', root), compilation=identity(root/'compile.json', root),
        cases={}, model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN')
    compiler_schema = None
    if ddk_root:
        try:
            compiler_schema = load_schema(Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
            result['compiler_schema'] = compiler_schema.evidence
        except Exception as exc:
            result['compiler_schema_error'] = f'{type(exc).__name__}: {exc}'
            result['status'] = 'PARTIAL'
    for name in cases:
        row = result['cases'][name] = dict(compilation=compiled['cases'].get(name, {}))
        try:
            exported = json.loads(checked(source, by_name[name]['export']).read_text())
            folder = source/'models'/name
            air = checked(folder, exported['air'])
            schema_path = folder/'air-schema.pb'
            schema = Schema(bounded_bytes(schema_path, 8 << 20),
                expected_file=exported['schema']['schema_file'], expected_package=exported['schema']['schema_package'])
            row.update(source_inputs=exported['source_inputs'], golden=exported['golden'],
                source_air=identity(air, folder), source_schema=identity(schema_path, folder),
                source_graph=graph_metadata(air, schema))
            row['source_schema_binding'] = 'CURRENT_RETAINED_DESCRIPTOR; OLDER_EXPORT_HAS_NO_SCHEMA_HASH'
        except Exception as exc:
            row['source_error'] = f'{type(exc).__name__}: {exc}'
            result['status'] = 'PARTIAL'
        job = root/'models'/name/'omc'
        receipt = job/'report.json'
        if receipt.is_file():
            report = json.loads(bounded_bytes(receipt, 8 << 20))
            row['compiler_report'] = dict(identity(receipt, root),
                **{k: report.get(k) for k in ('stage', 'returncode', 'error', 'requested_options', 'ddk_id', 'tbe_compatibility')})
        log = job/'compile.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - (24 << 10)))
                tail = stream.read(24 << 10).decode('utf-8', errors='replace')
            row['log'] = dict(identity(log, root), tail=tail, tail_bytes_limit=24 << 10)
        dumps = sorted((job/'compile-work/generated/dump_graph').glob('*.irpb'))
        # Keep the failing infershape and post-partition graphs ahead of early snapshots.
        dumps.sort(key=lambda p: (0 if 'fail' in p.name else 1 if p.name.startswith('05_') else 2, p.name))
        row['ir_dumps'], row['uninspected_ir_dumps'] = [], [p.name for p in dumps[12:]]
        for path in dumps[:12]:
            item = dict(identity(path, root), status='NOT_INSPECTED_NO_DDK_SCHEMA')
            row['ir_dumps'].append(item)
            if compiler_schema:
                try:
                    item.update(graph_metadata(path, compiler_schema))
                except Exception as exc:
                    item.update(status='UNAVAILABLE', error=f'{type(exc).__name__}: {exc}')
                    result['status'] = 'PARTIAL'
        save_json(output/'report.json', result)
    return result

"""Inspect a retained Draft's partition interfaces; never execute vendor code.

Logical port sizes are not measured transfers, allocations or elapsed time.
NPU tasks stay opaque; CPU GraphDef slices use explicit, checked task ownership.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.draft.compile_precision import _observed_desc
from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
from qwen35_moe_dflash.toolchain.omc_cpu_inspect import _task
from qwen35_moe_dflash.toolchain.omc_wire import load_schema

KIND = 'qwen35-draft-runtime-boundary-inspection'
WIDTHS = {'float16': 2, 'float32': 4, 'int32': 4, 'int64': 8, 'bool': 1}


def descriptor(raw, enums):
    row = _observed_desc(raw, enums)
    shape, width = row['shape'], WIDTHS.get(row['dtype'])
    # Empty/missing dimensions are not evidence of a scalar in sparse metadata.
    concrete = (isinstance(shape, list) and bool(shape) and width is not None
                and all(type(x) is int and x > 0 for x in shape))
    row.update(layout=raw.get('layout'), device_type=raw.get('device_type'),
               recorded_size=raw.get('size'),
               logical_bytes=math.prod(shape)*width if concrete else None)
    return row


def summarize_graph(graph, enums):
    nodes = graph.get('op', [])
    names = [n.get('name') for n in nodes]
    if (not 0 < len(nodes) <= 8192 or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('graph has missing/duplicate names or exceeds node budget')
    by_name = {n['name']: n for n in nodes}
    rows, counts, ports = [], Counter(), 0
    for node in nodes:
        kind = node.get('type')
        counts[kind] += 1
        row = dict(name=node['name'], type=kind, inputs=[], outputs=[],
                   src_name=node.get('src_name', []), src_index=node.get('src_index', []))
        if kind == 'GraphOp':
            row['task'] = _task(node)
        refs = node.get('input', [])
        if not isinstance(refs, list):
            raise ValueError('graph input references must be a list')
        # Retain control/unmatched edges; never silently align them to tensor slots.
        row['input_references'] = refs
        aligned = len(refs) == len(node.get('input_desc', []))
        row['port_reference_alignment'] = 'POSITIONAL' if aligned else 'UNRESOLVED_COUNT_MISMATCH'
        for direction, key in (('inputs', 'input_desc'), ('outputs', 'output_desc')):
            for i, raw in enumerate(node.get(key, [])):
                ports += 1
                if ports > 32768:
                    raise ValueError('graph exceeds descriptor budget')
                item = dict(index=i, **descriptor(raw, enums))
                if direction == 'inputs':
                    item['reference'] = refs[i] if aligned else None
                    item['producer_status'] = 'UNRESOLVED'
                    edge = item['reference']
                    if isinstance(edge, str) and ':' in edge:
                        name, port = edge.rsplit(':', 1)
                        producer = by_name.get(name)
                        if port.isascii() and port.isdecimal() and producer is not None:
                            index = int(port)
                            outputs = producer.get('output_desc', [])
                            if index < len(outputs):
                                item.update(producer_status='EXPLICIT_SAME_GRAPH_EDGE',
                                    producer=dict(name=name, port=index, type=producer['type'],
                                        owner=_task(producer)['cl_name'] if producer['type']=='GraphOp' else None,
                                        descriptor=descriptor(outputs[index], enums)))
                row[direction].append(item)
        # Constants are metadata only: no tensor data or attributes are copied.
        rows.append(row)
    return dict(name=graph.get('name'), node_type_counts=dict(counts), nodes=rows)


def _partition_summary(graphs):
    result = []
    for graph in graphs:
        for node in graph['nodes']:
            if node['type'] != 'GraphOp':
                continue
            row = dict(graph=graph['name'], name=node['name'], owner=node['task']['cl_name'])
            for direction in ('inputs', 'outputs'):
                known = [p['logical_bytes'] for p in node[direction] if p['logical_bytes'] is not None]
                row[direction] = dict(ports=len(node[direction]), known_logical_bytes_sum=sum(known),
                    unknown_size_ports=len(node[direction])-len(known))
            row['largest_inputs'] = sorted(node['inputs'],
                key=lambda p: p['logical_bytes'] if p['logical_bytes'] is not None else -1, reverse=True)[:8]
            result.append(row)
    return result


def inspect_boundaries(compiled, accuracy, ddk_root, output):
    root, retained = Path(compiled).resolve(), Path(accuracy).resolve()
    path = retained/'result.json' if retained.is_dir() else retained
    source = read_json(path, 4 << 20)
    manifest = read_json(root/'manifest.json')
    if (source.get('artifact_kind') != 'qwen35-incremental-draft-accuracy'
            or source.get('status') not in {'PASSED', 'NUMERICAL_FAILED'}
            or source.get('stage') != 'complete'
            or source.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'
            or manifest.get('artifact_kind') != 'qwen35-incremental-draft-pytorch-omc'
            or manifest.get('status') != 'PYTORCH_DRAFT_OMC_COMPILED'
            or manifest.get('omc') != 'model.omc'):
        raise ValueError('need completed PyTorch compilation and retrieved full accuracy result')
    destination = output_path(output)
    fixture = Path(source['fixture'])
    fixture = fixture if fixture.is_absolute() else path.parent/fixture
    if any(destination == p or p in destination.parents for p in (root, path.parent, fixture.resolve(), Path(ddk_root).resolve())):
        raise ValueError('inspection output must be separate from retained inputs and DDK')
    if destination.exists() and read_json(destination).get('artifact_kind') != KIND:
        raise ValueError('refusing to replace unrelated inspection output')
    if sha256_file(fixture/'manifest.json') != source['fixture_sha256']:
        raise ValueError('retained fixture changed')
    meta = read_json(fixture/'manifest.json')
    if meta['model'] != source['model'] or meta.get('ddk_id') != manifest.get('ddk_id'):
        raise ValueError('fixture and accuracy/compiler model or DDK identities disagree')
    model = root/'model.omc'
    identity = dict(bytes=model.stat().st_size, sha256=sha256_file(model))
    if any(identity[k] != source['model'][k] for k in identity):
        raise ValueError('compiled model differs from measured accuracy model')
    schema = load_schema(Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    packing = read_json(root/'weight-packing.json')
    if packing.get('status') != 'EMBEDDED' or packing.get('schema') != schema.evidence:
        raise ValueError('selected DDK schema differs from the retained model packing receipt')
    enums = _enum_fields(schema)
    container = read_container(model)
    main = next(p for p in container['partitions'] if p['type'] == 0)
    tasks = next(p for p in container['partitions'] if p['type'] == 2)
    raw = read_partition(model, main)
    document = schema.decode(raw, 'ModelDef')
    if not 1 <= len(document.get('graph', [])) <= 8:
        raise ValueError('expected 1..8 compiled main graphs')
    graphs = [summarize_graph(g, enums) for g in document['graph']]
    report = dict(artifact_kind=KIND, status='INSPECTED',
        model=dict(path=str(model), **identity), source_accuracy=str(path),
        source_numerical_validation=source['numerical_validation'],
        model_compilation='NOT_RUN', device_execution='NOT_RUN', production_promotion='NOT_PERFORMED',
        scope='COMPILED_PARTITION_METADATA_NOT_RUNTIME_TRANSFER_OR_WAIT_MEASUREMENT',
        schema=schema.evidence, container_layout=container['layout'],
        main_partition_sha256=hashlib.sha256(raw).hexdigest(), main_graphs=graphs,
        partitions=_partition_summary(graphs), cpu_subgraphs=[], report=str(destination))
    counts, spans, cpu_bytes = Counter(), [], 0
    owners = [n for g in graphs for n in g['nodes'] if n['type']=='GraphOp']
    # Validate every slice, including opaque NPU tasks, before reading CPU slices.
    for node in owners:
        task = node['task']; start, size = task['offset'], task['bytes']
        if not size or start > tasks['bytes'] or size > tasks['bytes']-start:
            raise ValueError('GraphOp task is outside the task partition')
        if any(start < end and left < start+size for left, end in spans):
            raise ValueError('GraphOp task slices overlap')
        spans.append((start, start+size)); counts[task['cl_name']] += 1
    for node in owners:
        task = node['task']
        if task['cl_name'] != 'CPUCL':
            continue
        entry = dict(owner=node['name'], task=task, status='NOT_DECODED')
        report['cpu_subgraphs'].append(entry)
        try:
            cpu_bytes += task['bytes']
            if cpu_bytes > 64 << 20:
                raise ValueError('CPU task metadata exceeds 64 MiB budget')
            payload = read_partition(model, dict(offset=tasks['offset']+task['offset'], bytes=task['bytes']), max_bytes=16 << 20)
            decoded = schema.decode(payload, 'GraphDef')
            if decoded.get('name') != node['name']:
                raise ValueError('CPU GraphDef name differs from GraphOp owner')
            entry.update(status='DECODED', sha256=hashlib.sha256(payload).hexdigest(),
                         graph=summarize_graph(decoded, enums))
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
            entry['error'] = f'{type(error).__name__}: {error}'
            report['status'] = 'PARTIAL'
    report['graphop_counts'] = dict(counts)
    report['limitations'] = [
        'Port sizes use logical descriptor dimensions; format padding and runtime allocations are not inferred.',
        'A producer consumed more than once is counted per input port; sums are not unique buffers or measured copies.',
        'Only explicit same-graph input references resolve producers; src_name/src_index remain uninterpreted metadata.',
        'CPU task decoding uses the reference container layout, selected schema, checked ranges and matching graph names; no OMG JSON cross-check was performed.',
        'NPU task internals remain opaque. Boundary dtypes do not establish internal arithmetic precision.',
        'This report cannot attribute the model/API versus operator duration gap or prove a performance cause.',
    ]
    if model.stat().st_size != identity['bytes'] or sha256_file(model) != identity['sha256']:
        raise ValueError('model changed during inspection')
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if len(encoded.encode()) > 8 << 20:
        raise ValueError('boundary report exceeds 8 MiB')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_text(encoded); temporary.replace(destination)
    return {**{k: report[k] for k in ('artifact_kind','status','model','source_numerical_validation',
        'model_compilation','device_execution','scope','graphop_counts','partitions','report','limitations')},
        'cpu_subgraphs': [{k:v for k,v in r.items() if k!='graph'} | {
            'node_type_counts': r.get('graph', {}).get('node_type_counts')} for r in report['cpu_subgraphs']]}

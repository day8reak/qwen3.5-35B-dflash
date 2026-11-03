"""Read an existing prefix probe's graph boundaries and retained compiler IR."""
import hashlib
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
from qwen35_moe_dflash.toolchain.omc_cpu_inspect import _task
from qwen35_moe_dflash.toolchain.omc_inspect import _preview
from qwen35_moe_dflash.draft.compile_precision import _observed_desc
from qwen35_moe_dflash.draft.ir_inspect import inspect_draft_ir
from .embedding_probe import _compiled_probe


def inspect_prefix(root, ddk, ddk_id):
    """No compiler, board, graph mutation, fixture generation or payload guessing."""
    root = Path(root).resolve()
    report, case, model, schema, document, enums = _compiled_probe(root, ddk, ddk_id)
    extraction = report['extraction']
    paths = extraction.get('float_paths')
    if not paths:
        raise ValueError('prefix inspection requires an input-RMS probe')
    names = list(dict.fromkeys([extraction['gather'],
        *[n['name'] for path in paths.values() for n in path], extraction['rms']]))
    if len(names) > 8:
        raise ValueError('prefix inspection exceeds eight source nodes')
    result = dict(artifact_kind='qwen35-draft-prefix-graph-inspection', status='INSPECTED',
        probe_case=case, model=dict(report['model'], path=str(model)), ddk_id=ddk_id,
        model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
        scope='READ_ONLY_RETAINED_PROBE; DESCRIPTORS_AND_ATTRIBUTES_NOT_DEVICE_VALUES',
        source=dict(source_air_sha256=extraction['source_air_sha256'],
                    probe_air_sha256=extraction['probe_air_sha256'],
                    scale=extraction['scale'], epsilon=extraction['epsilon'],
                    unit_scale_bypass=extraction.get('unit_scale_bypass'),
                    float_paths=paths, selected_node_names=names),
        main_graphs=[], cpu_graphs=[], opaque_tasks=[])

    def descriptor(d):
        return dict(_observed_desc(d, enums), layout=d.get('layout'),
                    device_type=d.get('device_type'), size=d.get('size'))

    def graph_summary(graph):
        nodes = graph.get('op', [])
        node_names = [n.get('name') for n in nodes]
        if (not nodes or len(nodes) > 256 or any(not isinstance(n, str) or not n for n in node_names)
                or len(set(node_names)) != len(node_names)):
            raise ValueError('prefix graph has missing/duplicate names or exceeds 256 nodes')
        rows = []
        for node in nodes:
            attrs = _attributes(node)
            # Keep arithmetic attrs including their raw i/f type. Binary payloads
            # and nested graphs have no role in this compact boundary report.
            attrs = {k: v for k, v in attrs.items()
                     if k in {'value', 'scale', 'shift', 'power', 'alpha', 'beta',
                              'dst_type', 'src_type', 'dtype', 'epsilon', 'index',
                              'cl_name', 'original_op_names', '_original_op_names'}}
            rows.append(dict(name=node['name'], type=node.get('type'), input=node.get('input', []),
                src_name=node.get('src_name', []), src_index=node.get('src_index', []),
                inputs=[descriptor(d) for d in node.get('input_desc', [])],
                outputs=[descriptor(d) for d in node.get('output_desc', [])],
                attrs=_preview(attrs, budget=[128])))
        return dict(name=graph.get('name'), nodes=rows)

    parts = read_container(model)['partitions']
    tasks = [p for p in parts if p['type'] == 2]
    if len(tasks) != 1:
        raise ValueError('prefix probe requires exactly one task partition')
    tasks = tasks[0]
    owners, spans = [], []
    graphs = document.get('graph', [])
    if not 1 <= len(graphs) <= 8:
        raise ValueError('prefix probe requires 1..8 main graphs')
    for graph in graphs:
        result['main_graphs'].append(graph_summary(graph))
        for node in graph['op']:
            if node.get('type') != 'GraphOp':
                continue
            task = _task(node)
            start, size = task['offset'], task['bytes']
            if not size or start > tasks['bytes'] or size > tasks['bytes'] - start:
                raise ValueError('prefix GraphOp task is outside the task partition')
            if any(start < end and left < start+size for left, end in spans):
                raise ValueError('prefix GraphOp tasks overlap')
            spans.append((start, start+size))
            owners.append((graph['name'], node['name'], task))
    total = 0
    for graph_name, name, task in owners:
        row = dict(main_graph=graph_name, owner=name, task=task)
        if task['cl_name'] != 'CPUCL':
            result['opaque_tasks'].append(dict(row, status='NOT_DECODED'))
            continue
        total += task['bytes']
        if total > 32 << 20:
            raise ValueError('prefix CPU tasks exceed 32 MiB')
        raw = read_partition(model, dict(offset=tasks['offset']+task['offset'], bytes=task['bytes']), max_bytes=16 << 20)
        graph = schema.decode(raw, 'GraphDef')
        if graph.get('name') != name:
            raise ValueError('CPU graph name differs from its GraphOp owner')
        result['cpu_graphs'].append(dict(row, sha256=hashlib.sha256(raw).hexdigest(),
                                         graph=graph_summary(graph)))
    # HIAI compiler IR may expose pre-task NPU nodes. It is a retained pass
    # snapshot, not the final opaque kernel. Missing names do not prove removal.
    result['compiler_ir'] = inspect_draft_ir(root/'compilation',
        Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so',
        node_name=names[0], node_names=names)
    def compact_ir_node(node):
        return dict(name=node.get('name'), type=node.get('type'), input=node.get('input'),
                    src_name=node.get('src_name'), src_index=node.get('src_index'),
                    attrs=node.get('attributes'),
                    inputs=[descriptor(d['raw']) for d in node.get('input_desc', [])],
                    outputs=[descriptor(d['raw']) for d in node.get('output_desc', [])])
    for dump in result['compiler_ir']['files']:
        for match in dump.get('matches', []):
            match['node'] = compact_ir_node(match['node'])
            match.pop('named_fc_weights', None)
            match.pop('named_fc_weight_count', None)
            for edge in match.get('input_producers', []):
                if 'producer' in edge:
                    edge['producer'] = compact_ir_node(edge['producer'])
    result['limitations'] = [
        'Source logical float types, protobuf descriptor enums and operator integer attrs have separate meanings.',
        'NPUCL tasks are opaque and are not decoded as CPU GraphDef.',
        'Compiler IR is reported with per-file hash/inventory status; it is not runtime tensor readback.',
        'Absent source names may reflect fusion, renaming or unavailable dumps; no root-cause verdict.',
    ]
    # Bind the complete inspection to the same model again after slice reads.
    checked_file(root, report['model'])
    return result

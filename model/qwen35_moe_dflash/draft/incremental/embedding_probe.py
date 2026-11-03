"""Extract the source AIR's embedding lookup or anchor-to-first-RMS prefix.

Lookup cuts before Gather; input-rms retains upstream ID construction and RMS.
Both retain original weights and remain separate from full-model validation.
No GE/PyTorch installation is needed on the selected DDK server.
"""
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v
from .air_zeros import _Shapes
from .torch_export import AIR_KIND, _schema, _disjoint, _save

KIND = 'qwen35-draft-pytorch-embedding-probe'
CASE = 'pytorch_embedding'
INPUT_CASE = 'pytorch_input_rms'
TAPS_CASE = 'pytorch_input_rms_taps'
STAGES = {'lookup': CASE, 'input-rms': INPUT_CASE, 'input-rms-taps': TAPS_CASE}
DTYPES = {'float16': '<f2', 'float32': '<f4', 'int32': '<i4'}


def extract_lookup(raw, schema, shape):
    kind, _, graph = _decode(schema, raw)
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, _ = _node_bytes(graph_raw, schema)
    nodes = {n['name']: (n, p) for n, p, _ in rows}
    if len(nodes) != len(rows):
        raise ValueError('duplicate source AIR node names')
    shapes = _Shapes(rows, schema)
    types = schema.enums[schema.package+'.DataType']
    matches = []
    for n, payload, _ in rows:
        if n['type'] == 'GatherV2' and len(n.get('input', [])) == 3:
            table = shapes.node(n['input'][0])
            if table['type'] == 'FileConstant' and shapes.shape(n['input'][0]) == shape:
                matches.append((n, payload, table))
    if len(matches) != 1:
        raise ValueError('need exactly one full-table source embedding GatherV2')
    gather, payload, table = matches[0]
    axis = shapes.node(gather['input'][2])
    attrs = _attributes(table)
    if (attrs.get('dtype', {}).get('i') != 1  # GE API FLOAT16, not protobuf enum 1.
            or shapes.integers(gather['input'][2]) != [0]
            or _attributes(gather).get('batch_dims', {}).get('i', 0) != 0
            or len(gather.get('input_desc', [])) != 3
            or len(gather.get('output_desc', [])) != 1
            or gather['input_desc'][0].get('dtype') != types['DT_FLOAT16']
            or gather['input_desc'][1].get('dtype') != types['DT_INT32']
            or gather['output_desc'][0].get('dtype') != types['DT_FLOAT16']):
        raise ValueError('source lookup is not FP16 full-table axis-zero INT32 Gather')
    selected = [table, axis, gather]
    for node in selected:
        if any(node.get(k) for k in ('src_name', 'src_index', 'dst_name', 'dst_index', 'subgraph_name')):
            raise ValueError('compiled adjacency is not supported in source lookup')
        if any('g' in a or a.get('list', {}).get('g') for a in _attributes(node).values()):
            raise ValueError('nested source graphs are not supported')
        if any(d.get('layout', 'ND') != 'ND' for d in node.get('input_desc', [])+node.get('output_desc', [])):
            raise ValueError('lookup requires source ND descriptors')
    if table.get('input') or axis.get('input'):
        raise ValueError('lookup table and axis must have no inputs')
    for d in (gather['input_desc'][0],):
        if d.get('shape', {}).get('dim') and d['shape']['dim'] != shape:
            raise ValueError('embedding table shape conflicts with consumer')

    def text(message, field, value):
        return _blob(_field(schema, message, field, 9, False), value)

    def integer(message, field, value, field_type):
        return _v(_field(schema, message, field, field_type, False) << 3) + _v(value)

    def desc(dtype, dims):
        dimensions = _blob(_field(schema, 'ShapeDef', 'dim', 3, True), b''.join(_v(d) for d in dims))
        return (integer('TensorDescriptor', 'dtype', types[dtype], 14)
                + _blob(_field(schema, 'TensorDescriptor', 'shape', 11, False), dimensions)
                + text('TensorDescriptor', 'layout', 'ND') + text('TensorDescriptor', 'device_type', 'NPU'))

    op_input = _field(schema, 'OpDef', 'input', 9, True)
    in_desc = _field(schema, 'OpDef', 'input_desc', 11, True)
    out_desc = _field(schema, 'OpDef', 'output_desc', 11, True)
    ids_desc, embedding_desc = desc('DT_INT32', [1, 16]), desc('DT_FLOAT16', [1, 16, shape[1]])
    index_attr = (text('OpDef.AttrEntry', 'key', 'index')
                  + _blob(_field(schema, 'OpDef.AttrEntry', 'value', 11, False), integer('AttrDef', 'i', 0, 3)))
    data = (text('OpDef', 'name', 'probe_ids') + text('OpDef', 'type', 'Data')
            + _blob(in_desc, ids_desc) + _blob(out_desc, ids_desc)
            + _blob(_field(schema, 'OpDef', 'attr', 11, True), index_attr))
    # Change only Gather's index reference and descriptor for this explicit cut.
    pieces, input_index, desc_index = [], 0, 0
    for field, _, encoded, _ in _parts(payload):
        if field == op_input:
            if input_index == 1:
                encoded = _blob(field, 'probe_ids:0')
            input_index += 1
        if field == in_desc:
            if desc_index == 1:
                encoded = _blob(field, ids_desc)
            desc_index += 1
        pieces.append(encoded)
    changed_gather = b''.join(pieces)
    output = text('OpDef', 'name', 'NetOutput') + text('OpDef', 'type', 'NetOutput')
    for edge, descriptor in ((gather['name']+':0', embedding_desc), ('probe_ids:0', ids_desc)):
        output += _blob(op_input, edge) + _blob(in_desc, descriptor)
    op_field = _field(schema, 'GraphDef', 'op', 11, True)
    new_nodes = [data, nodes[table['name']][1], nodes[axis['name']][1], changed_gather, output]
    # Keep unique IDs even when original source nodes carry explicit IDs.
    from qwen35_moe_dflash.target.original_ge_append_wire import _new_id
    used = [n.get('id', 0) for n, _, _ in rows]
    new_nodes[0] = _new_id(new_nodes[0], schema, max(used)+1)
    new_nodes[-1] = _new_id(new_nodes[-1], schema, max(used)+2)
    reduced = text('GraphDef', 'name', 'draft_embedding_probe') + b''.join(_blob(op_field, p) for p in new_nodes)
    result = reduced if graph_field is None else b''.join(
        _blob(f, reduced) if f == graph_field else encoded for f, _, encoded, _ in _parts(raw))
    _, _, parsed = _decode(schema, result)
    if len(parsed['op']) != 5:
        raise ValueError('lookup extraction did not produce exactly five nodes')
    receipt = dict(policy='source-gather-full-table-explicit-ids-v1',
        source_air_sha256=hashlib.sha256(raw).hexdigest(), probe_air_sha256=hashlib.sha256(result).hexdigest(),
        gather=gather['name'], table=table['name'], axis=axis['name'],
        preserved_payload_sha256={n['name']: hashlib.sha256(nodes[n['name']][1]).hexdigest() for n in (table, axis)},
        changed='Gather index input only; new Data and NetOutput; other source nodes excluded',
        excluded='upstream ID construction, embedding scale, RMSNorm, Attention, residuals, head',
        table_path=attrs['file_path']['s'], table_shape=shape,
        graph=dict(inputs=[dict(name='probe_ids', dtype='int32', shape=[1, 16])],
                   outputs=[dict(name='embedding', dtype='float16', shape=[1, 16, shape[1]]),
                            dict(name='ids_readback', dtype='int32', shape=[1, 16])]))
    return result, receipt


def compile_probe(air_export, source, output, *, result, ddk_root, ddk_id, stage='lookup', timeout_seconds=1800, progress=print):
    from qwen35_moe_dflash.toolchain.existing_air import _snapshot, compile_existing_air
    from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    air, prepared, root = Path(air_export).resolve(), Path(source).resolve(), output_path(output)
    if stage not in STAGES:
        raise ValueError('select lookup, input-rms or input-rms-taps probe stage')
    prefix_stage = stage != 'lookup'
    _disjoint(root, air, prepared, ddk_root)
    if root.exists():
        raise ValueError('probe output already exists; preserve it and use a new test directory')
    manifest = read_json(air/'manifest.json')
    if manifest.get('artifact_kind') != AIR_KIND or manifest.get('status') != 'PYTORCH_DRAFT_AIR_EXPORTED':
        raise ValueError('expected completed PyTorch AIR')
    plan = read_json(prepared/'prepared.json')
    cfg = Qwen35DFlashConfig.from_dict(plan['draft_config'])
    schema = _schema((air/'air-schema.pb').read_bytes(), manifest['schema'])
    with (air/'export.air').open('rb') as stream:
        raw = stream.read((64 << 20)+1)
    if len(raw) > 64 << 20:
        raise ValueError('source AIR exceeds 64 MiB')
    recovered = Path(result).resolve()
    recovered = recovered/'result.json' if recovered.is_dir() else recovered
    observed = read_json(recovered, 4 << 20)
    if observed.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED':
        raise ValueError('probe needs a recovered full-model execution')
    fixture = Path(observed['fixture'])
    fixture = fixture if fixture.is_absolute() else recovered.parent/fixture
    binding = read_json(fixture/'pytorch-abi.json')
    meta = read_json(fixture/'manifest.json')
    if (sha256_file(fixture/'manifest.json') != observed['fixture_sha256']
            or binding['source_air_sha256'] != hashlib.sha256(raw).hexdigest()
            or binding['prepared_sha256'] != sha256_file(prepared/'prepared.json')
            or binding['packed_model_sha256'] != meta['model']['sha256']):
        raise ValueError('source AIR/prepared/model differs from the recovered execution binding')
    extracted, receipt = extract_lookup(raw, schema, [cfg.vocab_size, cfg.hidden_size])
    if prefix_stage:
        from .input_probe import extract_prefix
        extracted, receipt = extract_prefix(raw, schema, receipt, cfg, capture_inputs=stage == 'input-rms-taps')
    record = plan['weights']['target.embed_tokens.weight']
    if (record['dtype'] != 'float16' or record['shape'] != receipt['table_shape']
            or record['bytes'] != math.prod(record['shape'])*2):
        raise ValueError('prepared embedding descriptor differs from source AIR')
    ref = checked_file(prepared, record)
    relative = Path(receipt['table_path'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'weight':
        raise ValueError('embedding sidecar must be a relative weight/ member')
    table = checked_file(air, dict(record, path=str(relative)))
    snapshots = [(table, relative, record['sha256'])]
    gamma = None
    if prefix_stage:
        from .input_probe import GAMMA
        gamma_record = plan['weights'][GAMMA]
        if gamma_record['dtype'] != 'float16' or gamma_record['shape'] != [cfg.hidden_size] or gamma_record['bytes'] != cfg.hidden_size*2:
            raise ValueError('prepared first-RMS gamma descriptor differs')
        gamma = np.fromfile(checked_file(prepared, gamma_record), dtype='<f2').astype('<f4')
        gamma_relative = Path(receipt['gamma_path'])
        gamma_file = checked_file(air, dict(gamma_record, path=str(gamma_relative)))
        mask_relative = Path(receipt['mask_path'])
        if mask_relative.is_absolute() or len(mask_relative.parts) != 2 or mask_relative.parts[0] != 'weight':
            raise ValueError('source mask must be a relative weight/ member')
        mask_bytes = np.full((1,15), cfg.mask_token_id, dtype='<i4').tobytes()
        mask_hash = hashlib.sha256(mask_bytes).hexdigest()
        mask_file = checked_file(air, dict(path=str(mask_relative), bytes=len(mask_bytes), sha256=mask_hash))
        snapshots += [(gamma_file, gamma_relative, gamma_record['sha256']), (mask_file, mask_relative, mask_hash)]
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 4*record['bytes']+(512 << 20):
        raise OSError('embedding probe needs space for source snapshot, compiler weights and packed model')
    ids = np.array([[17, 32, 34, 38, cfg.mask_token_id, 0, 1, cfg.vocab_size-1,
                     cfg.mask_token_id, 38, 34, 32, 17, cfg.mask_token_id, 17, cfg.mask_token_id]], dtype='<i4')
    if prefix_stage:
        ids = np.full((1,16), cfg.mask_token_id, dtype='<i4')
        ids[0,0] = 17
    if np.any(ids < 0) or np.any(ids >= cfg.vocab_size):
        raise ValueError('diagnostic IDs exceed source vocabulary')
    golden = np.empty((1, 16, cfg.hidden_size), dtype='<f2')
    with ref.open('rb') as stream:
        for i, token in enumerate(ids[0]):
            stream.seek(int(token)*cfg.hidden_size*2)
            golden[0, i] = np.frombuffer(stream.read(cfg.hidden_size*2), dtype='<f2')
    if not np.isfinite(golden).all():
        raise ValueError('prepared diagnostic embedding rows are nonfinite')
    arrays = [('input_0', ids), ('golden_0', golden), ('golden_1', ids)]
    if prefix_stage:
        # Same FP16 embedding/scaling boundary and independent FP32 RMS formula
        # as the full Draft reference; no target outputs enter this reference.
        x = (golden.astype('<f4')*np.float32(cfg.input_embedding_scale)).astype('<f2').astype('<f4')
        rstd = np.float32(1)/np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32)+np.float32(cfg.rms_norm_eps))
        normalized = np.asarray(x*rstd*gamma, dtype='<f4')
        if not np.isfinite(normalized).all():
            raise ValueError('first-RMS reference is nonfinite')
        arrays[0] = ('input_0', np.array([17], dtype='<i4'))
        arrays.append(('golden_2', normalized))
        if stage == 'input-rms-taps':
            factor = np.ones(cfg.hidden_size, dtype='<f4')
            arrays += [('input_1', factor), ('golden_3', x.reshape(16, cfg.hidden_size)),
                       ('golden_4', gamma), ('golden_5', factor.reshape(1, cfg.hidden_size))]
    root.mkdir(parents=True)
    state = dict(artifact_kind=KIND, status='FAILED', stage='extract', model_compilation='NOT_RUN',
        device_execution='NOT_RUN', numerical_validation='NOT_RUN', ddk_id=ddk_id,
        probe_case=STAGES[stage], scope=('SOURCE_INPUT_PREFIX_ONLY_NOT_FULL_DRAFT' if prefix_stage
            else 'SOURCE_EMBEDDING_LOOKUP_ONLY_NOT_FULL_DRAFT'), extraction=receipt,
        source_embedding_sha256=record['sha256'], prepared_sha256=sha256_file(prepared/'prepared.json'),
        original_model_sha256=meta['model']['sha256'], board_attempt=observed.get('board_attempt'))
    _save(root/'result.json', state)
    try:
        progress('snapshotting original prefix weights; no weight conversion')
        state['source_sidecars'] = {}
        for path, rel, digest in snapshots:
            snapshot = _snapshot(path, root/'air'/rel)
            if snapshot['sha256'] != digest:
                raise ValueError('source changed during probe snapshot: '+str(rel))
            state['source_sidecars'][str(rel)] = dict(bytes=snapshot['bytes'], sha256=digest)
        (root/'air/export.air').write_bytes(extracted)
        (root/'fixture').mkdir()
        for name, value in arrays:
            value.tofile(root/'fixture'/f'{name}.bin')
        state['fixtures'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in (root/'fixture').iterdir()}
        state['stage'] = 'compile'
        progress('compiling extracted PyTorch '+stage+' AIR with the selected DDK')
        compilation = compile_existing_air(root/'air/export.air', root/'compilation', ddk_root,
            ddk_id=ddk_id, timeout_seconds=timeout_seconds, input_layout='air-directory',
            use_origin_format=True, save_weights_as_external_data=True, progress=lambda r: progress(r['message']))
        state['model_compilation'] = compilation['model_compilation']
        if compilation['status'] != 'COMPILED':
            raise ValueError(str(compilation.get('error', compilation['status'])))
        state['stage'] = 'pack'
        sidecars = [dict(r, path=str(root/'compilation'/r['path'])) for r in compilation['external_data_candidates']
                    if Path(r['path']).suffix == '.weight']
        packing = embed_external_weights(root/'compilation'/compilation['omc']['path'], root/'model.omc',
            Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so',
            expected_source_sha256=compilation['omc']['sha256'], sidecars=sidecars, progress=progress)
        _save(root/'weight-packing.json', packing)
        if packing['status'] != 'EMBEDDED':
            raise ValueError('probe weight packing failed')
        state.update(status='COMPILED', stage='complete', model=dict(path='model.omc',
            bytes=(root/'model.omc').stat().st_size, sha256=sha256_file(root/'model.omc')))
    except Exception as error:
        state.update(error=f'{type(error).__name__}: {error}', compilation_diagnostics=compilation_diagnostics(root/'compilation'))
    _save(root/'result.json', state)
    return state


def _compiled_probe(root, ddk, ddk_id, selection=None):
    """Read the retained packed model with the selected DDK schema."""
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    root = Path(root).resolve()
    report = read_json(root/'result.json')
    case = report.get('probe_case', CASE)
    if (report.get('artifact_kind') != KIND or report.get('status') != 'COMPILED'
            or report.get('model_compilation') != 'PASSED' or report.get('ddk_id') != ddk_id
            or case not in STAGES.values() or selection not in (None, [case])):
        raise ValueError('expected the completed embedding probe for this DDK')
    model = checked_file(root, report['model'])
    if model.stat().st_size > 2 << 30:
        raise ValueError('embedding probe exceeds 2 GiB')
    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    main = [p for p in read_container(model)['partitions'] if p['type'] == 0]
    if len(main) != 1:
        raise ValueError('probe requires exactly one main model partition')
    document = schema.decode(read_partition(model, main[0]), 'ModelDef')
    return report, case, model, schema, document, _enum_fields(schema)


def inspect_compiled_probe(root, ddk, ddk_id):
    """Report descriptor mismatches without preparing files or starting a run."""
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    from qwen35_moe_dflash.toolchain.omc_inspect import _preview
    from validate_draft_vector import bind_io
    report, case, model, schema, document, enums = _compiled_probe(root, ddk, ddk_id)
    declared = report['extraction']['graph']
    result = dict(artifact_kind='qwen35-draft-probe-interface-inspection', status='INSPECTED',
                  probe_case=case, model=dict(report['model'], path=str(model)), ddk_id=ddk_id,
                  model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
                  scope='READ_ONLY_COMPILED_DESCRIPTORS; NO_SLOT_REMAP_OR_TYPE_CONVERSION',
                  declared=declared, source_taps=report['extraction'].get('taps', []), graphs=[])
    def describe(desc):
        return dict(_observed_desc(desc, enums), layout=desc.get('layout'),
                    device_type=desc.get('device_type'), recorded=_preview(desc, budget=[128]))
    graphs = document.get('graph', [])
    if len(graphs) > 8:
        raise ValueError('probe interface inspection exceeds eight graphs')
    for graph in graphs:
        nodes = {n['name']: n for n in graph.get('op', [])}
        inputs, nets = [], []
        for node in graph.get('op', []):
            if node.get('type') == 'Data':
                inputs.append(dict(name=node['name'],
                    indices=[a.get('value', {}).get('i') for a in node.get('attr', []) if a.get('key') == 'index'],
                    outputs=[describe(d) for d in node.get('output_desc', [])[:8]]))
            if node.get('type') != 'NetOutput':
                continue
            ports = []
            references = node.get('input', [])
            descriptors = node.get('input_desc', [])
            if len(descriptors) > 8 or len(references) > 16:
                raise ValueError('probe NetOutput inspection exceeds port budget')
            for index, descriptor in enumerate(descriptors):
                observed = describe(descriptor)
                edge = references[index] if index < len(references) else None
                row = dict(index=index, descriptor=observed, reference=edge,
                    matching_declared_names=[s['name'] for s in declared['outputs']
                                             if all(s[k] == observed[k] for k in ('shape', 'dtype'))])
                # Producer metadata is diagnostic, never a descriptor substitute.
                if isinstance(edge, str) and ':' in edge:
                    name, port = edge.rsplit(':', 1)
                    producer = nodes.get(name)
                    if producer is not None and port.isdecimal():
                        port = int(port)
                        descs = producer.get('output_desc', [])
                        row['producer'] = dict(name=name, type=producer.get('type'), output=port,
                            descriptor=describe(descs[port]) if port < len(descs) else None)
                ports.append(row)
            nets.append(dict(name=node['name'], inputs=ports, input_references=references,
                             output_descriptors=[describe(d) for d in node.get('output_desc', [])[:8]]))
        result['graphs'].append(dict(name=graph.get('name'), inputs=inputs, netoutputs=nets))
    try:
        result['binding'] = bind_io(document, declared, enums)
        result['binding_status'] = 'PASSED'
    except (KeyError, ValueError) as error:
        result.update(binding_status='FAILED', binding_error=f'{type(error).__name__}: {error}')
    return result


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    """Use the shared tensor-probe transport with a separately bound recipe."""
    from qwen35_moe_dflash.draft.incremental.torch_export import _compiled_observation
    from validate_draft_vector import bind_io, WARMUP, REPETITIONS
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    report, case, model, schema, document, enums = _compiled_probe(root, ddk, ddk_id, selection)
    io = bind_io(document, report['extraction']['graph'], enums)
    folder = out/'fixture'/case
    folder.mkdir(parents=True)
    plan = ['QWEN35_TENSOR_PROBE 1', f'iterations {WARMUP} {REPETITIONS}']
    for direction, prefix in (('inputs', 'input'), ('outputs', 'golden')):
        for spec in io[direction]:
            name = f'{prefix}_{spec["fixture_index"]}.bin'
            path = checked_file(root/'fixture', dict(report['fixtures'][name], path=name))
            if path.stat().st_size != spec['bytes']:
                raise ValueError('fixture extent differs from compiled probe')
            dest = folder/f'{prefix}_{spec["index"]}.bin'
            shutil.copyfile(path, dest)
            spec['fixture_sha256'] = sha256_file(dest)
            plan.append(' '.join((direction[:-1], str(spec['index']), spec['name'], spec['dtype'], ','.join(map(str, spec['shape'])))))
    (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
    observed = _compiled_observation(model, ddk)
    meta = dict(cases=[dict(case=case, **io, model=_file_record(model, case+'/model.omc'),
        graphop_counts=observed['graphop_counts'])], warmup=WARMUP, repetitions=REPETITIONS,
        atol=0.0 if case == CASE else 0.01, rtol=0.0 if case == CASE else 0.01,
        exact_bytes=case == CASE, scope=report['scope'],
        ddk_id=ddk_id, schema=schema.evidence, source_evidence=report)
    _save(out/'fixture/manifest.json', meta)
    return meta

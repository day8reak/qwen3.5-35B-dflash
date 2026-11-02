"""Extract the source AIR's embedding lookup, with explicit INT32 IDs.

This cuts before Gather. It tests the original full table and lookup operation,
not Draft's upstream ID construction, RMSNorm, attention or full-model accuracy.
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


def compile_probe(air_export, source, output, *, result, ddk_root, ddk_id, timeout_seconds=1800, progress=print):
    from qwen35_moe_dflash.toolchain.existing_air import _snapshot, compile_existing_air
    from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    air, prepared, root = Path(air_export).resolve(), Path(source).resolve(), output_path(output)
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
    record = plan['weights']['target.embed_tokens.weight']
    if (record['dtype'] != 'float16' or record['shape'] != receipt['table_shape']
            or record['bytes'] != math.prod(record['shape'])*2):
        raise ValueError('prepared embedding descriptor differs from source AIR')
    ref = checked_file(prepared, record)
    relative = Path(receipt['table_path'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'weight':
        raise ValueError('embedding sidecar must be a relative weight/ member')
    table = checked_file(air, dict(record, path=str(relative)))
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 4*record['bytes']+(512 << 20):
        raise OSError('embedding probe needs space for source snapshot, compiler weights and packed model')
    ids = np.array([[17, 32, 34, 38, cfg.mask_token_id, 0, 1, cfg.vocab_size-1,
                     cfg.mask_token_id, 38, 34, 32, 17, cfg.mask_token_id, 17, cfg.mask_token_id]], dtype='<i4')
    if np.any(ids < 0) or np.any(ids >= cfg.vocab_size):
        raise ValueError('diagnostic IDs exceed source vocabulary')
    golden = np.empty((1, 16, cfg.hidden_size), dtype='<f2')
    with ref.open('rb') as stream:
        for i, token in enumerate(ids[0]):
            stream.seek(int(token)*cfg.hidden_size*2)
            golden[0, i] = np.frombuffer(stream.read(cfg.hidden_size*2), dtype='<f2')
    if not np.isfinite(golden).all():
        raise ValueError('prepared diagnostic embedding rows are nonfinite')
    root.mkdir(parents=True)
    state = dict(artifact_kind=KIND, status='FAILED', stage='extract', model_compilation='NOT_RUN',
        device_execution='NOT_RUN', numerical_validation='NOT_RUN', ddk_id=ddk_id,
        scope='SOURCE_EMBEDDING_LOOKUP_ONLY_NOT_FULL_DRAFT', extraction=receipt,
        source_embedding_sha256=record['sha256'], prepared_sha256=sha256_file(prepared/'prepared.json'),
        original_model_sha256=meta['model']['sha256'], board_attempt=observed.get('board_attempt'))
    _save(root/'result.json', state)
    try:
        progress('snapshotting the original full embedding table; no weight conversion')
        snapshot = _snapshot(table, root/'air'/relative)
        if snapshot['sha256'] != record['sha256']:
            raise ValueError('embedding changed during probe snapshot')
        (root/'air/export.air').write_bytes(extracted)
        (root/'fixture').mkdir()
        for name, value in (('input_0', ids), ('golden_0', golden), ('golden_1', ids)):
            value.tofile(root/'fixture'/f'{name}.bin')
        state['fixtures'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in (root/'fixture').iterdir()}
        state['stage'] = 'compile'
        progress('compiling extracted PyTorch Gather AIR with the selected DDK')
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


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    """Use the shared tensor-probe transport with a separately bound recipe."""
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    from qwen35_moe_dflash.draft.incremental.torch_export import _compiled_observation
    from validate_draft_vector import bind_io, WARMUP, REPETITIONS
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    report = read_json(root/'result.json')
    if (report.get('artifact_kind') != KIND or report.get('status') != 'COMPILED'
            or report.get('model_compilation') != 'PASSED' or report.get('ddk_id') != ddk_id
            or selection not in (None, [CASE])):
        raise ValueError('expected the completed embedding probe for this DDK')
    model = checked_file(root, report['model'])
    if model.stat().st_size > 2 << 30:
        raise ValueError('embedding probe exceeds 2 GiB')
    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    main = [p for p in read_container(model)['partitions'] if p['type'] == 0]
    if len(main) != 1:
        raise ValueError('probe requires exactly one main model partition')
    io = bind_io(schema.decode(read_partition(model, main[0]), 'ModelDef'), report['extraction']['graph'], _enum_fields(schema))
    folder = out/'fixture'/CASE
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
    meta = dict(cases=[dict(case=CASE, **io, model=_file_record(model, CASE+'/model.omc'),
        graphop_counts=observed['graphop_counts'])], warmup=WARMUP, repetitions=REPETITIONS,
        atol=0.0, rtol=0.0, exact_bytes=True, scope='SOURCE_EMBEDDING_LOOKUP_ONLY_NOT_FULL_DRAFT',
        ddk_id=ddk_id, schema=schema.evidence, source_evidence=report)
    _save(out/'fixture/manifest.json', meta)
    return meta

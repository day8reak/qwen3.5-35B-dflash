"""Read the ordered AIR and OMC outputs of one GDR stage probe, before board execution."""
import json

from ...draft.compile_precision import _edges
from ...draft.ir_inspect import _enum_fields
from ...draft.runtime_boundaries import descriptor
from ...toolchain.omc_container import read_container, read_partition
from ...toolchain.omc_wire import load_schema
from ..layer_probe.batch import bind_compiled
from ..layer_probe.common import checked, sha
from .decode_cache_control import graph_message
from .decode_cache_trace import _record


def inspect_abi(compiled, ddk, case):
    build = json.loads((compiled / 'compile.json').read_text())
    batch = compiled / build['source_batch']
    if not all(sha(root / 'manifest.json') == build['manifest_sha256'] for root in (batch, compiled)):
        raise ValueError('AIR batch and compile manifest differ')
    _, row, folder, record = _record(batch, case)
    if row['kind'] != 'gdr_stages' or build['cases'][case]['status'] != 'COMPILED':
        raise ValueError(f'{case}: expected a compiled GDR stage probe')

    _, air = graph_message(checked(folder, record['air']).read_bytes(), 'ChunkGatedDeltaRule')
    contract = record['output_contract']
    air_net, = [n for n in air.op if n.type == 'NetOutput']
    air_edges = [e for e in air_net.input if not e.endswith(':-1')]
    if (air_edges != contract['source_edges'] or contract['order'] != [s['name'] for s in record['goldens']]
            or len(air_edges) != len(record['goldens'])):
        raise ValueError('AIR output order does not match its export contract')
    air_nodes = {n.name: n for n in air.op}
    air_types = air.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number

    omc = checked(compiled, build['cases'][case]['omc'])
    part, = [p for p in read_container(omc)['partitions'] if p['type'] == 0]
    schema = load_schema(ddk / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    document = schema.decode(read_partition(omc, part), 'ModelDef')
    enums = _enum_fields(schema)
    graph, = document['graph']
    nodes = {n['name']: n for n in graph['op']}
    net, = [n for n in graph['op'] if n['type'] == 'NetOutput']
    descs = net.get('input_desc', [])
    print(f"{case}   AIR outputs={len(air_edges)}   OMC outputs={len(descs)}")
    print(json.dumps(dict(omc_net_inputs=net.get('input'), omc_graph_outputs=graph.get('output'))))
    for i in range(max(len(air_edges), len(descs))):
        slot = dict(slot=i)
        if i < len(air_edges):
            edge = air_edges[i]
            owner, port = edge.rsplit(':', 1)
            src = air_nodes[owner]
            ref = record['goldens'][i]
            slot.update(tensor=ref['name'], golden_shape=ref['shape'], golden_dtype=ref['dtype'],
                        air_edge=edge, air_op=src.type, allowed_views=contract.get('stage_views', {}).get(ref['name'], []))
            if ref['name'] == 'gamma' and 'gamma_readout' in contract:
                slot['gamma_readout'] = contract['gamma_readout']
            if int(port) < len(src.output_desc):
                raw = src.output_desc[int(port)]
                slot.update(air_shape=list(raw.shape.dim) or None, air_dtype=air_types[raw.dtype].name,
                            air_layout=raw.layout or None)
        if i < len(descs):
            slot['omc'] = descriptor(descs[i], enums)
            for owner, port in _edges(net, i):
                producer = nodes.get(owner, {})
                out = producer.get('output_desc', [])
                slot.update(omc_edge=f'{owner}:{port}', omc_op=producer.get('type'),
                            producer_desc=descriptor(out[port], enums) if port < len(out) else None)
        print(json.dumps(slot, ensure_ascii=False))
    try:
        bind_compiled(document, record['source_inputs'], record['goldens'], enums, contract)
    except ValueError as exc:
        print('Fixture binding FAILED:', exc)
        return False
    print('Fixture binding PASSED')
    return True

"""Bind a retained PyTorch candidate for standalone numerical diagnosis.

AOT Data order is not forward() order. Recover roles from the source graph's
control and committed-cache edges, then join exact Data names to the compiled
Data.index. Ordered outputs are bound to the hash-checked compiler AIR. This
does not promote the candidate or certify its internal arithmetic precision.
"""
import json
import math
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from .abi import inspect_incremental_abi
from .runtime import GEOMETRY, validate_abi


def _slots(rows):
    return [dict(r, index=i, bytes=math.prod(r['shape']) * (2 if r['dtype'] == 'float16' else 4))
            for i, r in enumerate(rows)]


def source_input_roles(graph, declared):
    """Match this exporter's explicit topology; never sort arg names or KV shapes."""
    ops = graph.get('op', [])
    nodes = {n['name']: n for n in ops}
    if len(nodes) != len(ops):
        raise ValueError('duplicate source AIR node names')
    data = {n['name'] for n in ops if n['type'] == 'Data'}
    if len(data) != 17 or len(declared) != 17:
        raise ValueError('PyTorch Draft requires exactly 17 source Data nodes')

    def node(edge):
        name, sep, port = edge.rpartition(':')
        if not sep or not port.isdigit() or name not in nodes:
            raise ValueError('unresolved source data edge: ' + edge)
        return nodes[name], int(port)

    def root(edge, *, feature=False):
        seen = set()
        while True:
            n, port = node(edge)
            # Unrelated multi-output operators (e.g. SplitV for RoPE) can
            # appear while scanning Concat. They cannot identify a Data or
            # FileConstant role, and must not be mistaken for output zero.
            if port:
                return dict(name=edge, type='NONZERO_OUTPUT_PORT')
            if n['name'] in seen or len(seen) >= 64:
                raise ValueError('cyclic or excessive source passthrough chain')
            seen.add(n['name'])
            passthrough = {'Cast', 'Identity', 'Reshape'}
            if feature:
                # Spatial Conv permutes token patches before the input reshape.
                # This traversal identifies the feature Data, not its layout.
                passthrough.add('Transpose')
            if n['type'] not in passthrough:
                return n
            edge = n['input'][0]

    def buffer(n, name):
        return (n['type'] == 'FileConstant' and
                _attributes(n).get('file_path', {}).get('s') == 'weight/' + name)

    def unique(values, role):
        values = set(values)
        if len(values) != 1:
            raise ValueError(f'cannot uniquely bind source {role}: {sorted(values)}')
        return values.pop()

    roles = {}
    roles['target_features'] = unique(
        [root(n['input'][0], feature=True)['name'] for n in ops if n['type'] == 'Conv2D'
         and root(n['input'][0], feature=True)['type'] == 'Data'], 'target_features')
    roles['start_position'] = unique(
        [root(n['input'][1])['name'] for n in ops if n['type'] == 'Sub'
         and buffer(root(n['input'][0]), 'cache_positions')
         and root(n['input'][1])['type'] == 'Data'], 'start_position')
    pairs = [[root(e)['name'] for e in n.get('input', [])[:2]]
             for n in ops if n['type'] == 'Add']
    roles['valid_rows'] = unique(
        [b if a == roles['start_position'] else a for a, b in pairs
         if a in data and b in data and a != b and roles['start_position'] in (a, b)], 'valid_rows')
    roles['proposal_count'] = unique(
        [root(n['input'][1])['name'] for n in ops if n['type'] == 'LessEqual'
         and buffer(root(n['input'][0]), 'block_offsets')
         and root(n['input'][1])['type'] == 'Data'], 'proposal_count')
    if any(r['name'] == 'block_embeddings' for r in declared):
        # Prove the input feeds layer-zero candidate RMS with its original
        # gamma; equal tensor dimensions alone cannot identify this role.
        roles['block_embeddings'] = unique(
            [root(n['input'][0])['name'] for n in ops if n['type'] == 'RmsNorm'
             and len(n.get('input', [])) == 2
             and buffer(root(n['input'][1]), 'draft_layers_0_input_layernorm_weight')
             and root(n['input'][0])['type'] == 'Data'], 'block_embeddings')
    else:
        roles['anchor'] = unique(
            [root(n['input'][0])['name'] for n in ops if n['type'] == 'ConcatV2'
             and len(n.get('input', [])) >= 2 and buffer(root(n['input'][1]), 'mask_tokens')
             and root(n['input'][0])['type'] == 'Data'], 'anchor')
    outputs = [n for n in ops if n['type'] == 'NetOutput']
    if len(outputs) != 1 or len(outputs[0].get('input', [])) != 14:
        raise ValueError('source must expose the 14 ordered Draft outputs')
    cache_edges = []
    for i, edge in enumerate(outputs[0]['input'][2:]):
        select = root(edge)
        if select['type'] != 'SelectV2' or len(select.get('input', [])) != 3:
            raise ValueError('committed KV output must originate at SelectV2: ' + edge)
        old = root(select['input'][2])
        if old['type'] != 'Data':
            raise ValueError('committed KV false branch does not preserve its input: ' + edge)
        role = f"past_{('k', 'v')[i % 2]}_{i // 2}"
        roles[role] = old['name']
        cache_edges.append(dict(role=role, output_index=i+2, source_output=edge,
                                select=select['name'], previous_input=select['input'][2]))
    if set(roles) != {r['name'] for r in declared} or set(roles.values()) != data or len(set(roles.values())) != 17:
        raise ValueError('source semantic roles do not bijectively cover all Data nodes')
    return roles, cache_edges


def _netoutput(graph):
    rows = [n for n in graph['op'] if n['type'] == 'NetOutput']
    if len(rows) != 1:
        raise ValueError('expected one AIR NetOutput')
    return rows[0].get('input')


def bind_candidate(compiled, air_export, prepared, ddk_root, output):
    """Return a diagnostic ABI without changing the compiled candidate manifest."""
    from .torch_export import AIR_KIND, OMC_KIND, _schema
    root, air, source, out = map(Path, (compiled, air_export, prepared, output))
    meta, export = read_json(root/'manifest.json'), read_json(air/'manifest.json')
    if (meta.get('artifact_kind') != OMC_KIND or meta.get('status') != 'PYTORCH_DRAFT_OMC_COMPILED'
            or meta.get('model_compilation') != 'PASSED' or meta.get('omc') != 'model.omc'
            or export.get('artifact_kind') != AIR_KIND or export.get('status') != 'PYTORCH_DRAFT_AIR_EXPORTED'):
        raise ValueError('expected completed PyTorch AIR and OMC candidate bundles')
    if any(meta[k] != export[k] for k in (*GEOMETRY, 'inputs', 'outputs')):
        raise ValueError('candidate and source export contracts differ')
    from .optimization import manifest_policy, bind_weight_snapshots
    policy = manifest_policy(export)
    if manifest_policy(meta) != policy:
        raise ValueError('candidate and source optimization contracts differ')
    host_embedding = policy == 'packed-spatial-host-embedding'
    missing_host_binding = host_embedding and meta.get('host_embedding') is None
    if host_embedding and not isinstance(export.get('host_embedding'), dict):
        raise ValueError('source AIR is missing the host embedding binding')
    # Some retained compiled manifests omit this host-only input contract.
    # Defer only that missing duplicate: the AIR/compiler snapshots, prepared
    # table, source roles and packed OMC must all pass the checks below before
    # returning a diagnostic binding. An explicit conflicting value still fails.
    if not missing_host_binding and meta.get('host_embedding') != export.get('host_embedding'):
        raise ValueError('candidate and source host embedding bindings differ: ' + json.dumps(
            dict(compiled=meta.get('host_embedding'), air_export=export.get('host_embedding')),
            ensure_ascii=False, sort_keys=True))
    validate_abi(dict(meta, inputs=_slots(meta['inputs']), outputs=_slots(meta['outputs'])))
    compilation = root/'compilation'
    receipt = read_json(compilation/'report.json', 8 << 20)
    packing = read_json(root/'weight-packing.json')
    if (receipt.get('status') != 'COMPILED' or receipt.get('model_compilation') != 'PASSED'
            or receipt.get('platform') != 'kirinx90' or receipt.get('ddk_id') != meta['ddk_id']
            or packing.get('status') != 'EMBEDDED'
            or any(packing['source'][k] != receipt['omc'][k] for k in ('bytes', 'sha256'))):
        raise ValueError('compiler/packing/DDK receipts disagree')
    snapshots = {r['source_relative']: r for r in receipt['input_snapshots']}
    if len(snapshots) != len(receipt['input_snapshots']):
        raise ValueError('duplicate compiler input snapshots')
    source_air = checked_file(air, dict(snapshots['export.air'], path='export.air'))
    schema = _schema((air/'air-schema.pb').read_bytes(), export['schema'])
    _, _, graph = _decode(schema, source_air.read_bytes())
    roles, cache_edges = source_input_roles(graph, export['inputs'])
    compiler_air = checked_file(compilation, receipt.get('air_preparation', {}).get(
        'compiler_air', snapshots['export.air']))
    _, _, prepared_graph = _decode(schema, compiler_air.read_bytes())
    if _netoutput(prepared_graph) != _netoutput(graph):
        raise ValueError('compiler AIR changed the ordered source output contract')
    prepared_roles, _ = source_input_roles(prepared_graph, export['inputs'])
    if roles != prepared_roles:
        raise ValueError('compiler AIR changed source control/cache bindings')
    external = checked_file(compilation, receipt['omc'])
    if not ddk_root:
        raise ValueError('PyTorch candidate validation requires --ddk-root for the compiled ABI schema')
    library = Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so'
    declared = dict(export, inputs=[dict(r, name=roles[r['name']]) for r in export['inputs']])
    abi = inspect_incremental_abi(external, library, declared)
    if abi['status'] != 'PASSED':
        raise ValueError('PyTorch compiled ABI binding failed: ' + json.dumps(abi, ensure_ascii=False))
    if abi['schema'] != packing['schema']:
        raise ValueError('ABI inspection and packing used different DDK schemas')
    inverse = {v: k for k, v in roles.items()}
    for row in abi['inputs']:
        row['source_data'] = row['name']
        row['name'] = inverse[row['name']]
    # Bind the independent reference to actual compiler weight snapshot hashes,
    # not to matching tensor dimensions or an unrelated preparation directory.
    plan = read_json(source/'prepared.json')
    if (plan.get('artifact_kind') != 'qwen35-draft-ge-prepared' or plan.get('status') != 'PREPARED'
            or plan.get('precision') != 'float16' or plan.get('quantization') is not None):
        raise ValueError('expected original prepared FP16 reference weights')
    file_constants = [_attributes(n).get('file_path', {}).get('s')
                      for n in graph['op'] if n['type'] == 'FileConstant']
    weights = bind_weight_snapshots(plan['weights'], policy, file_constants, snapshots, compilation)
    if host_embedding:
        from .host_embedding import source_contract
        if export.get('host_embedding') != source_contract(plan):
            raise ValueError('host embedding differs from the export-bound original weight')
        if 'weight/embedding_weight' in file_constants:
            raise ValueError('host embedding candidate unexpectedly retains the full embedding table')
        record = plan['weights']['target.embed_tokens.weight']
        checked_file(source, record)
        weights.append(dict(name='target.embed_tokens.weight', sha256=record['sha256'],
                            role='HOST_INPUT_ROW_LOOKUP_NOT_COMPILED_WEIGHT'))
    # The reference loader verifies the complete topology and prepared file bytes.
    checked_file(root, dict(packing['output'], path='model.omc'))
    result = dict(meta, inputs=abi['inputs'], outputs=abi['outputs'],
                  model=dict(packing['output'], path='model.omc'), draft_config=plan['draft_config'])
    if host_embedding:
        result.update(host_embedding=export['host_embedding'], host_embedding_binding=(
            'RESTORED_FROM_VERIFIED_AIR_AND_PREPARED_WEIGHTS' if missing_host_binding else 'VERIFIED'))
    validate_abi(result)
    evidence = dict(abi, artifact_kind='qwen35-pytorch-draft-diagnostic-abi',
        source_input_roles=roles, cache_edges=cache_edges, reference_weights=weights,
        source_air_sha256=sha256_file(source_air), compiler_air_sha256=sha256_file(compiler_air),
        prepared_sha256=sha256_file(source/'prepared.json'), packed_model_sha256=packing['output']['sha256'],
        precision='UNVERIFIED_INTERNAL_ARITHMETIC', production_promotion='NOT_PERFORMED',
        output_binding='ORDERED_SOURCE_NETOUTPUT_PRESERVED_IN_HASH_BOUND_COMPILER_AIR')
    if host_embedding:
        evidence.update(host_embedding=result['host_embedding'],
                        host_embedding_binding=result['host_embedding_binding'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'pytorch-abi.json').write_text(json.dumps(evidence, indent=2)+'\n')
    result['diagnostic_frontend'] = 'pytorch'
    return result

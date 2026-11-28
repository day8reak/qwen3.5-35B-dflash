"""Map the complete decoder L0 replay to source variables for the generic dump comparison."""
import json

import numpy as np

from decoder_cases_board import output_order
from gdn_matrices_txt import BACKENDS
from l0_accuracy_report import EXPLAIN, aligned, reference_matches
from qwen35_moe_dflash.target.layer_probe.common import read_tensor, sized_file
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import cpro_input_records
from qwen35_moe_dflash.target.prefill_t64.decode_cache_control import graph_message
from qwen35_moe_dflash.target.prefill_t64.gdn_layer_stages import DESCRIPTION
from qwen35_moe_dflash.target.prefill_t64.l0_accuracy_air import EXPERT_GROUPS, GDN_GROUPS, readouts


def load(case):
    board = case / 'board'
    base, package = board/'L00', board/'package/L00'
    fixture = board/'package/fixture'
    replay = base/'replay'
    def read(path):
        return json.loads(path.read_bytes())
    meta, ref = read(package/'case.json'), read(fixture/'reference.json')
    build, receipt = read(base/'omc/build.json'), read(replay/'result.json')
    inventory = read(replay/'dump-inventory.json')
    if (meta['layers'] != [0] or ref['phase'] != 'prefill' or ref['physical_rows'] != 64
            or receipt['status'] != 'EXECUTED' or inventory['attempt'] != receipt['attempt']
            or receipt['omc_sha256'] != build['model']['sha256'] or inventory['omc_sha256'] != receipt['omc_sha256']
            or inventory['model_exitcode'] != 0 or inventory['parser_exitcode'] != 0):
        raise ValueError('Expected a completed single L0 prefill64 dump, build and replay from the same invocation')
    with np.load(fixture/'l0-steps-reference/values.npz', allow_pickle=False) as archive:
        if not reference_matches(archive, fixture, ref):
            raise ValueError('Intermediate CPU/310P archive and replay fixture differ')
        references = [{k[len(b)+1:]: archive[k] for k in archive.files if k.startswith(b+'_')} for b in BACKENDS[:2]]
    valid = ref['valid_rows']
    public_names = {'hidden_0':'block_out', 'out_proj_0':'out', 'conv_state_0':'new_history',
                    'recurrent_state_0':'recurrent_cache', 'logits':'logits'}
    edges = {public_names[r['name']]: r['edge'] for r in meta['outputs']}
    public_edges = edges.copy()
    entry, = [node+':0' for node, role in meta['inputs'].items() if role == 'hidden']
    edges['block_entry'] = entry
    for row in meta.get('rmsnorm_nodes', ()):
        if row['inputs'][0] == entry:
            edges['hidden'] = row['output']
    air = package/meta.get('air_directory', 'weight')/'export.air'
    raw = air.read_bytes()
    notes = []
    try:
        for label, section, _ in readouts(raw, meta, dump_edges=edges):
            if 'error' in section:
                notes.append(f'{label}: {section["error"]}')
    except (KeyError, ValueError, IndexError) as exc:
        notes.append(f'Partial AIR mapping: {exc}')
    edges.update(public_edges, block_entry=entry)
    _, graph = graph_message(raw, 'NetOutput')
    enum = graph.DESCRIPTOR.file.enum_types_by_name['DataType'].values_by_number
    air_types = {f'{op.name}:{i}': enum[desc.dtype].name for op in graph.op for i, desc in enumerate(op.output_desc)}
    names = output_order(build['outputs'], meta['outputs'], ref['outputs'])
    observed = {}
    feed = cpro_input_records(ref)['hidden']
    anchors = {entry:read_tensor(sized_file(fixture, feed), feed)}
    for name, spec, rec in zip(names, build['outputs'], receipt['outputs'], strict=True):
        role = public_names[name]
        array = read_tensor(sized_file(replay, rec), spec).reshape(ref['outputs'][name]['shape'])
        observed[role] = (array, str(replay/rec['path']))
        anchors[public_edges[role]] = array
        for table, reference in zip(references, (ref['cpu'], ref['outputs'])):
            item = reference[name]
            table[role] = read_tensor(sized_file(fixture, item), item)
    points = []
    for stage, roles in {**GDN_GROUPS, **EXPERT_GROUPS, 'head':['logits']}.items():
        for role in roles:
            if role == 'route_map' or any(p['name'] == role for p in points):
                continue
            points.append(dict(name=role, stage=stage, edge=edges.get(role),
                               description=EXPLAIN.get(role) or DESCRIPTION.get(role, role)))

    def views(point, tensors, observed_peers):
        role = point['name']
        if role == 'logits':
            return tensors, f'token{valid-1}'
        values = [dict(v, **{role:x}) for v, x in zip([*references, observed_peers], tensors)]
        return aligned(role, values, valid)

    return dict(points=points, references=references, root=replay/'dump', files=inventory['files'], views=views,
                air=air, valid_rows=valid, history_role='new_history',
                observed=observed, anchors=anchors, air_types=air_types, notes=notes,
                peers={k: edges[k] for k in ('expert_ids', 'route_map') if k in edges},
                attempt=receipt['attempt'], output=base/'dump-comparison')

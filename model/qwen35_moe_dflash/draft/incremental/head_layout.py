"""Read the retained full-vocabulary head and its compiler preparation.

No weight payload, compiler, Torch or device call. Source expression and
available compiler IR are separate evidence; neither proves internal tiling.
"""
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes
from qwen35_moe_dflash.draft.ir_inspect import _node, inspect_draft_ir
from .hidden_probe import Prefix
from .top1 import contract
from .torch_export import AIR_KIND, OMC_KIND, _schema

KIND = 'qwen35-draft-head-layout-inspection'
# Prepared semantic target.lm_head.weight is exported as IncrementalDraft.head.weight.
# Keep the same mapping as optimization.bind_weight_snapshots.
WEIGHT = 'weight/head_weight'


def head_expression(raw, schema, top1):
    if top1 != contract('stable-arg-max', [1, 15, 248320]):
        raise ValueError('head inspection requires the retained C16/K15 full-vocabulary Top1 contract')
    trace = Prefix(raw, schema)
    nodes = trace.nodes
    reductions = [n for n in nodes.values() if n['type'] == 'StableArgMax']
    outputs = [n for n in nodes.values() if n['type'] == 'NetOutput']
    if (len(reductions) != 1 or len(outputs) != 1 or len(reductions[0].get('input', [])) != 1
            or outputs[0].get('input', [])[:1] != [reductions[0]['name']+':0']):
        raise ValueError('cannot bind a single full-vocabulary reduction to public output zero')

    def walk(edge, allowed):
        chain, seen = [], set()
        while True:
            n = trace.shapes.node(edge)
            if n['name'] in seen or len(seen) >= 32:
                raise ValueError('cyclic or excessive head view chain')
            seen.add(n['name'])
            if n['type'] not in allowed:
                return n, chain
            row = dict(name=n['name'], type=n['type'])
            inputs = n.get('input', [])
            if len(inputs) != (1 if n['type'] == 'Identity' else 2):
                raise ValueError('invalid head view inputs')
            if n['type'] != 'Identity':
                row['literal'] = trace.shapes.integers(inputs[1])
            if n['type'] == 'Reshape' and _attributes(n).get('axis', {}).get('i', 0) != 0:
                raise ValueError('partial head reshape is unsupported')
            chain.append(row); edge = inputs[0]

    mm, output_views = walk(reductions[0]['input'][0], {'Identity', 'Reshape'})
    if mm['type'] != 'MatMul':
        raise ValueError('Top1 does not descend from a MatMul through supported views')
    trace.data_inputs(mm)  # Check absent optional bias, not just a tensor size.
    if len(mm.get('output_desc', [])) != 1:
        raise ValueError('head MatMul must have one output descriptor')
    weight, weight_views = walk(mm['input'][1], {'Identity', 'Reshape', 'Transpose'})
    attrs = _attributes(weight)
    if (weight['type'] != 'FileConstant' or attrs.get('file_path', {}).get('s') != WEIGHT
            or attrs.get('shape', {}).get('list', {}).get('i') != [248320, 2048]
            or attrs.get('dtype', {}).get('i') != 1):
        raise ValueError('head must use the original full FP16 weight')
    shape = [248320, 2048]
    for v in reversed(weight_views):
        if v['type'] == 'Transpose':
            if v['literal'] != [1, 0]: raise ValueError('unsupported head weight permutation')
            shape = shape[::-1]
        elif v['type'] == 'Reshape' and v['literal'] != shape:
            raise ValueError('head weight reshape changes its matrix dimensions')
    flags = {k: _attributes(mm).get(k, {}).get('b') for k in ('transpose_x1', 'transpose_x2')}
    if any(type(v) is not bool for v in flags.values()) or flags['transpose_x1']:
        raise ValueError('expected explicit head transpose flags and untransposed activations')
    effective = shape[::-1] if flags['transpose_x2'] else shape
    if effective != [2048, 248320]:
        raise ValueError('head right operand does not cover full hidden and vocabulary axes')
    for d, required in [(mm['input_desc'][1], shape), (mm['output_desc'][0], [15, 248320])]:
        recorded = d.get('shape', {}).get('dim')
        if recorded and recorded != required: raise ValueError('head descriptor conflicts with matrix geometry')
    half = trace.types['DT_FLOAT16']
    if any(d.get('dtype') != half for d in (*mm['input_desc'][:2], mm['output_desc'][0])):
        raise ValueError('head MatMul boundary must remain FP16')
    xshape = mm['input_desc'][0].get('shape', {}).get('dim') or None
    if xshape and xshape != [15, 2048]: raise ValueError('head activation descriptor differs')
    data_path = {mm['name'], *(v['name'] for v in output_views), reductions[0]['name']}
    extra = [dict(producer=e, consumer=n['name'], input_slot=i)
             for n in nodes.values() for i, e in enumerate(n.get('input', []))
             if e.rpartition(':')[0] in data_path
             and n['name'] not in data_path | {outputs[0]['name']}]
    # Audit weight consumers separately from logits consumers. Distinct
    # FileConstants may name the same payload; include all exact-path aliases.
    aliases = {n['name'] for n in nodes.values() if n['type'] == 'FileConstant'
               and _attributes(n).get('file_path', {}).get('s') == WEIGHT}
    pending, visited, consumers, terminal = sorted(aliases), set(), [], []
    while pending:
        producer = pending.pop()
        if producer in visited: continue
        visited.add(producer)
        for n in nodes.values():
            for slot, edge in enumerate(n.get('input', [])):
                if edge != producer+':0': continue
                row = dict(producer=producer, consumer=n['name'], type=n['type'], input_slot=slot)
                consumers.append(row)
                if n['type'] in {'Identity', 'Reshape', 'Transpose'} and slot == 0:
                    pending.append(n['name'])
                else:
                    terminal.append(row)
    usage = dict(file_constant_aliases=sorted(aliases), consumers=consumers,
        terminal_consumers=terminal, exclusive_head_use=bool(terminal) and all(
            r['consumer']==mm['name'] and r['input_slot']==1 for r in terminal),
        scope='ALL_EXACT_FILE_PATH_ALIASES_AND_VIEW_CONSUMERS_IN_THIS_SOURCE_GRAPH; NOT_OTHER_MODELS_OR_HOST_EMBEDDING_FILES')
    return dict(matmul=_node(mm, {}), weight=_node(weight, {}), weight_views=weight_views,
        weight_operand_shape=shape, effective_right_operand_shape=effective, transpose_flags=flags,
        activation_shape_observed=xshape, output_views=output_views,
        reduction=_node(reductions[0], {}), public_output_0=reductions[0]['name']+':0',
        activation_producer=_node(trace.shapes.node(mm['input'][0]), {}), additional_consumers=extra, weight_usage=usage,
        liveness=dict(scope='DECLARED_FULL_MODEL_CONTRACT_AND_TRACED_TOP1_PATH; NOT_FIXTURE_VALUES',
            rows=15, vocabulary=248320, terms_per_logit=2048, weight_bytes=1017118720,
            fp16_logits_bytes=7449600, removable_vocabulary_columns=0,
            note='All vocabulary logits participate in Top1; no vocabulary subset is a full-model equivalent.'))


def inspect_head_layout(air_export, compiled, ddk_root):
    air, root = Path(air_export).resolve(), Path(compiled).resolve()
    export, meta = read_json(air/'manifest.json'), read_json(root/'manifest.json')
    receipt = read_json(root/'compilation/report.json', 16 << 20)
    packing = read_json(root/'weight-packing.json')
    if (export.get('artifact_kind') != AIR_KIND or export.get('status') != 'PYTORCH_DRAFT_AIR_EXPORTED'
            or meta.get('artifact_kind') != OMC_KIND or meta.get('status') != 'PYTORCH_DRAFT_OMC_COMPILED'
            or receipt.get('status') != 'COMPILED' or receipt.get('platform') != 'kirinx90'
            or receipt.get('ddk_id') != meta.get('ddk_id') or packing.get('status') != 'EMBEDDED'
            or meta.get('top1') != export.get('top1')):
        raise ValueError('expected retained matching full Draft AIR and completed X90 compilation')
    if any(packing.get('source', {}).get(k) != receipt.get('omc', {}).get(k) for k in ('bytes', 'sha256')):
        raise ValueError('head compilation and packing receipts differ')
    snapshots = [r for r in receipt['input_snapshots'] if r.get('source_relative') == 'export.air']
    if len(snapshots) != 1: raise ValueError('missing unique compiler source AIR receipt')
    original = checked_file(air, dict(snapshots[0], path='export.air'))
    preparation = receipt.get('air_preparation', {})
    record = preparation.get('compiler_air', snapshots[0])
    prepared = checked_file(root/'compilation', record)
    if any(p.stat().st_size > 64 << 20 for p in (original, prepared)):
        raise ValueError('head AIR exceeds inspection budget')
    schema = _schema((air/'air-schema.pb').read_bytes(), export['schema'])
    source = head_expression(original.read_bytes(), schema, export['top1'])
    compiler = head_expression(prepared.read_bytes(), schema, meta['top1'])
    output = packing['output']
    model = root/'model.omc'
    if model.stat().st_size != output['bytes']:
        raise ValueError('retained model extent differs from packing receipt')
    library = Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so'
    # Read bounded retained IR if it exists. Do not launch OMG to obtain more.
    ir = inspect_draft_ir(root/'compilation', library,
        node_names=[compiler['matmul']['name'], compiler['reduction']['name']])
    if 'schema' in ir and ir['schema'] != packing.get('schema'):
        raise ValueError('selected DDK schema differs from retained model packing')
    matches = [m for f in ir['files'] for m in f.get('matches', [])]
    compiled_head = any(m['node']['name'] == compiler['matmul']['name'] for m in matches)
    folded = not compiler['weight_views'] and compiler['transpose_flags']['transpose_x2']
    return dict(artifact_kind=KIND, status='INSPECTED' if compiled_head and ir['status']=='INSPECTED' else 'PARTIAL',
        model_compilation='NOT_RUN', device_execution='NOT_RUN', weight_payloads='NOT_READ_OR_COPIED',
        model=dict(path=str(model), bytes=output['bytes'], recorded_sha256=output['sha256'],
                   verification='SIZE_ONLY; HASH_FROM_PACKING_RECEIPT_NOT_RECOMPUTED'),
        ddk_id=meta['ddk_id'], source_air_sha256=snapshots[0]['sha256'], compiler_air_sha256=record['sha256'],
        source=source, compiler_input=compiler, weight_transpose_folded_in_compiler_input=folded,
        retained_compiler_ir=ir,
        next='REVIEW_COMPILED_HEAD_ORIENTATION_BEFORE_ANY_HEAD_MICROBENCHMARK',
        limitations=['Compiler input AIR is not compiled NPU kernel code or physical traffic.',
            'Retained IR may be absent, incomplete or from an earlier compiler pass; names alone do not prove final tiling.',
            'No full-model liveness reduction, precision pass or speedup is claimed.'])

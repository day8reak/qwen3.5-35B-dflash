"""Small source-mask / fused-attention diagnostic, with no learned weights.

Keep the first source Attention's attributes, optional ports and mask ancestry.
Cut only packed Q/K/V, supply deterministic values, and compare against an
independent logical attention reference. This is not full-Draft validation.
"""
from copy import deepcopy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import shutil

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.target.original_air_roundtrip import _attributes, _decode
from qwen35_moe_dflash.target.original_ge_append_wire import _blob, _field, _graph_raw, _node_bytes, _parts, _v, _new_id
from .air_zeros import _Shapes, materialize_zeros
from .air_fill import materialize_fills
from .air_reshape import fold_reshape_chains
from .torch_abi import source_input_roles
from .torch_export import AIR_KIND, _schema, _disjoint, _save

KIND = 'qwen35-draft-source-attention-probe'
SCOPE = 'SOURCE_FIRST_MASK_AND_FUSED_ATTENTION_SYNTHETIC_PACKED_QKV_NOT_FULL_DRAFT'
CASES = tuple('pytorch_attention_'+s for s in
              ('empty', 'anchor', 'partial', 'short', 'full', 'weighted', 'catchup'))
CONTROLS = ('start_position', 'valid_rows', 'proposal_count')
BUFFERS = {'cache_positions', 'block_offsets', 'query_repeat', 'anchor_witness',
           'target_attention_mask_zeros', 'target_attention_mask_padding',
           'target_attention_block_table', 'target_attention_query_lengths'}
CONTRACT = dict(num_heads=32, num_key_value_heads=8, head_dim=128, block_size=32,
                input_layout='BNSD', inner_precise=2, scale_value=128**-0.5,
                query_shape=[1,256,16,16], kv_shape=[7,64,32,16],
                mask_shape=[1,1,16,224], block_table_shape=[1,7],
                logical_kv_rows=208, padded_kv_rows=224)
ATTENTION_CONTROLS = (
    dict(name='attention_key0', mask='key0', kv_length=224, shape=[32,16,128]),
    dict(name='attention_key15', mask='key15', kv_length=224, shape=[32,2048]),
    dict(name='attention_unmasked', mask='unmasked', kv_length=224, shape=[16,4096]),
    dict(name='attention_kv32', mask='unmasked', kv_length=32, shape=[65536]),
)
RUNTIME_LENGTH_CONTROLS = (
    dict(name='attention_runtime_q', mask='source', kv_length=224, runtime_ports=[16], shape=[32,16,128]),
    dict(name='attention_runtime_kv', mask='source', kv_length=224, runtime_ports=[6], shape=[32,2048]),
    dict(name='attention_runtime_both', mask='source', kv_length=224, runtime_ports=[6,16], shape=[16,4096]),
    dict(name='attention_runtime_unmasked', mask='unmasked', kv_length=224, runtime_ports=[6,16], shape=[65536]),
)
RUNTIME_LENGTH_INPUTS = (
    dict(name='probe_attention_all_query_length', role='all_query_length', value=16, port=16),
    dict(name='probe_attention_kv_length', role='kv_length', value=224, port=6),
)


def _control_variants(controls=False, runtime_lengths=False):
    if controls and runtime_lengths:
        raise ValueError('choose mask/length controls or runtime-length controls, not both')
    return RUNTIME_LENGTH_CONTROLS if runtime_lengths else ATTENTION_CONTROLS if controls else ()


def _control_visibility(control, source_visibility=None):
    if control['mask']=='source':
        if source_visibility is None or source_visibility.shape!=(16,224) or source_visibility.dtype!=np.bool_:
            raise ValueError('runtime length reference needs original source visibility')
        return source_visibility
    visible = np.zeros((16,224),dtype=bool)
    if control['mask']=='unmasked':
        visible[:,:control['kv_length']] = True
    else:
        visible[:,int(control['mask'][3:])] = True
    return visible


def _contract(manifest):
    if any(manifest.get(k) != v for k, v in
           (('context_rows',16), ('cache_capacity',192), ('max_draft_tokens',15))):
        raise ValueError('attention diagnostic requires C16/cap192/K15')
    actual = manifest.get('attention', {})
    for key, value in CONTRACT.items():
        if actual.get(key) != value:
            raise ValueError('source attention contract differs: '+key)


def extract_attention(raw, schema, manifest, *, controls=False, runtime_lengths=False):
    """Preserve source mask/length/table nodes; never copy QKV weight ancestry."""
    variants = _control_variants(controls, runtime_lengths)
    _contract(manifest)
    kind, _, graph = _decode(schema, raw)
    roles, _ = source_input_roles(graph, manifest['inputs'])
    graph_raw, graph_field = _graph_raw(raw, schema, kind)
    rows, op_field = _node_bytes(graph_raw, schema)
    nodes = {n['name']: n for n, _, _ in rows}
    payloads = {n['name']: p for n, p, _ in rows}
    if len(nodes) != len(rows):
        raise ValueError('duplicate source nodes')
    attention = [n for n, _, _ in rows if n['type'] == 'AdnFusedInferAttention']
    if len(attention) != 6 or attention[0]['name'] != 'AdnFusedInferAttention':
        raise ValueError('expected the six-layer source attention sequence')
    op = attention[0]
    attrs, edges = _attributes(op), op.get('input', [])
    types = schema.enums[schema.package+'.DataType']
    required = {0: ('DT_FLOAT16', CONTRACT['query_shape']),
                1: ('DT_FLOAT16', CONTRACT['kv_shape']), 2: ('DT_FLOAT16', CONTRACT['kv_shape']),
                4: ('DT_FLOAT16', CONTRACT['mask_shape']), 5: ('DT_INT64',[1]),
                6: ('DT_INT64',[1]), 14: ('DT_INT32',[1,7]), 16: ('DT_INT64',[1])}
    if len(edges) != 18 or len(op.get('input_desc', [])) != 18 or len(op.get('output_desc', [])) != 1:
        raise ValueError('source Attention needs its original 18 ports and one output')
    for key in ('num_heads', 'num_key_value_heads', 'block_size', 'inner_precise', 'input_layout', 'scale_value'):
        value = attrs.get(key, {}).get('s' if key == 'input_layout' else 'f' if key == 'scale_value' else 'i')
        expected = CONTRACT[key]
        if not (isinstance(value, (int,float)) and math.isclose(value, expected, rel_tol=1e-6)
                if key == 'scale_value' else value == expected):
            raise ValueError('source Attention attribute differs: '+key)
    for i, edge in enumerate(edges):
        if i not in required:
            if edge:
                raise ValueError('unexpected optional Attention input: '+str(i))
            continue
        dtype, shape = required[i]
        d = op['input_desc'][i]
        if (not edge or d.get('dtype') != types[dtype] or d.get('layout','ND') != 'ND'
                or (d.get('shape',{}).get('dim') and d['shape']['dim'] != shape)):
            raise ValueError('source Attention input descriptor differs: '+str(i))
    d = op['output_desc'][0]
    if (d.get('dtype') != types['DT_FLOAT16'] or d.get('layout','ND') != 'ND'
            or (d.get('shape',{}).get('dim') and d['shape']['dim'] != CONTRACT['query_shape'])):
        raise ValueError('source Attention output descriptor differs')
    shapes = _Shapes(rows, schema)
    selected, active = set(), set()
    allowed = {'Data','Const','FileConstant','Add','Sub','Less','LessEqual','Greater','GreaterEqual',
               'LogicalAnd','LogicalOr','LogicalNot','SelectV2','Cast','Identity','Reshape','GatherV2',
               'ConcatV2','ZerosLike','Shape','Fill','Equal'}
    def visit(edge):
        n = shapes.node(edge); name = n['name']
        if name in selected:
            return
        if name in active or len(active) > 64 or len(selected) > 256:
            raise ValueError('cyclic/oversized source mask ancestry')
        if n['type'] not in allowed:
            raise ValueError('unexpected mask/length operator: '+name+' '+n['type'])
        if (any(n.get(k) for k in ('src_name','src_index','dst_name','dst_index','subgraph_name'))
                or any('g' in a or a.get('list',{}).get('g') for a in _attributes(n).values())):
            raise ValueError('nested or compiled source ancestry unsupported')
        if n['type'] == 'FileConstant':
            path = _attributes(n).get('file_path',{}).get('s')
            if path not in {'weight/'+b for b in BUFFERS}:
                raise ValueError('unexpected source sidecar; no learned weights allowed: '+str(path))
        active.add(name)
        for e in n.get('input', []):
            if e or n['type'] != 'Data':
                visit(e)
        active.remove(name); selected.add(name)
    for slot in (4,5,6,14,16):
        visit(edges[slot])
    if {n for n in selected if nodes[n]['type']=='Data'} != {roles[k] for k in CONTROLS}:
        raise ValueError('mask must depend on exactly start_position, valid_rows and proposal_count')
    # These literal lengths are part of this source contract, not corrections.
    for slot, expected in ((5,[16]),(6,[224])):
        if shapes.integers(edges[slot]) != expected:
            raise ValueError('source Attention literal length differs: '+str(slot))
    all_query = shapes.node(edges[16]); all_query_attrs = _attributes(all_query)
    if (all_query['type']!='FileConstant'
            or all_query_attrs.get('file_path',{}).get('s')!='weight/target_attention_query_lengths'
            or all_query_attrs.get('dtype',{}).get('i')!=9
            or all_query_attrs.get('shape',{}).get('list',{}).get('i')!=[1]):
        raise ValueError('source all_seq_lengths_q must bind the verified INT64 query-length sidecar')
    if any(n.startswith('probe_attention_') for n in nodes):
        raise ValueError('probe namespace already exists')

    def blob(message, field, value, typ=9, repeated=False):
        return _blob(_field(schema,message,field,typ,repeated),value)
    def integer(message, field, value, typ=3):
        return _v(_field(schema,message,field,typ,False)<<3)+_v(value)
    def desc(dtype, dims):
        return (integer('TensorDescriptor','dtype',types[dtype],14)
                +blob('TensorDescriptor','shape',blob('ShapeDef','dim',b''.join(_v(d) for d in dims),3,True),11)
                +blob('TensorDescriptor','layout','ND')+blob('TensorDescriptor','device_type','NPU'))
    def attribute(key, value):
        return blob('OpDef','attr',blob('OpDef.AttrEntry','key',key)+blob('OpDef.AttrEntry','value',value,11),11,True)
    next_id = max(n.get('id',0) for n,_,_ in rows)+1
    def node(name, typ, ins, outs, extra=b''):
        nonlocal next_id
        p = blob('OpDef','name',name)+blob('OpDef','type',typ)
        for edge,d in ins:
            p += blob('OpDef','input',edge,repeated=True)+blob('OpDef','input_desc',d,11,True)
        for d in outs:
            p += blob('OpDef','output_desc',d,11,True)
        p = _new_id(p+extra,schema,next_id); next_id += 1
        return p
    attr_field = _field(schema,'OpDef','attr',11,True)
    inputs, changed = [], {}
    for i, role in enumerate(CONTROLS):
        name = roles[role]
        n = nodes[name]
        if shapes.shape(name+':0') != [1] or n['output_desc'][0].get('dtype') != types['DT_INT32']:
            raise ValueError('control must be scalar-shaped INT32: '+name)
        changed[name] = b''.join(e for f,_,e,p in _parts(payloads[name])
            if f != attr_field or schema.decode(p,'OpDef.AttrEntry').get('key') != 'index')
        changed[name] += attribute('index',integer('AttrDef','i',i))
        inputs.append(dict(name=name, role=role, dtype='int32', shape=[1]))
    added = []
    for i, label in enumerate(('q','k','v')):
        name = 'probe_attention_'+label
        dims = CONTRACT['query_shape'] if i==0 else CONTRACT['kv_shape']
        d = desc('DT_FLOAT16',dims)
        added.append(node(name,'Data',[('',d)],[d],attribute('index',integer('AttrDef','i',i+3))))
        inputs.append(dict(name=name, role=label, dtype='float16', shape=dims))
    input_field = _field(schema,'OpDef','input',9,True)
    input_desc_field = _field(schema,'OpDef','input_desc',11,True)
    input_index, desc_index, pieces = 0,0,[]
    for f,_,e,p in _parts(payloads[op['name']]):
        if f == input_field:
            if input_index < 3:
                e = _blob(f,inputs[3+input_index]['name']+':0')
            input_index += 1
        if f == input_desc_field:
            if desc_index < 3:
                e = _blob(f,desc('DT_FLOAT16',inputs[3+desc_index]['shape']))
            desc_index += 1
        pieces.append(e)
    changed[op['name']] = b''.join(pieces)
    expected = deepcopy(op)
    expected['input'][:3] = [s['name']+':0' for s in inputs[3:]]
    expected['input_desc'][:3] = [schema.decode(desc('DT_FLOAT16',s['shape']),'TensorDescriptor') for s in inputs[3:]]
    if schema.decode(changed[op['name']],'OpDef') != expected:
        raise ValueError('Attention extraction changed more than its QKV inputs')

    # Return the zero/visible predicate, keeping golden outputs finite. Exact
    # predicate equality plus numerical attention comparison checks masking;
    # this is not a raw -inf payload or a physical kernel-input readback.
    zero = 'probe_attention_zero'
    d0 = desc('DT_FLOAT16',[])
    tensor = blob('TensorDef','desc',d0,11)+blob('TensorDef','data',b'\0\0',12)
    extra = attribute('value',blob('AttrDef','t',tensor,11))
    added.append(node(zero,'Const',[],[d0],extra))
    maskd = desc('DT_FLOAT16',CONTRACT['mask_shape'])
    boold = desc('DT_BOOL',CONTRACT['mask_shape'])
    added.append(node('probe_attention_equal','Equal',[(edges[4],maskd),(zero+':0',d0)],[boold]))
    # The selected DDK cannot compile the added BOOL -> INT32 CastT.
    # Select finite FP16 literals instead; no arithmetic touches the -inf mask.
    for suffix, value in (('ones',1),('zeros',0)):
        tensor = blob('TensorDef','desc',maskd,11)+blob('TensorDef','data',
            np.full(CONTRACT['mask_shape'],value,dtype='<f2').tobytes(),12)
        added.append(node('probe_attention_'+suffix,'Const',[],[maskd],
                          attribute('value',blob('AttrDef','t',tensor,11))))
    added.append(node('probe_attention_visible','SelectV2',
        [('probe_attention_equal:0',boold),('probe_attention_ones:0',maskd),
         ('probe_attention_zeros:0',maskd)],[maskd]))
    # Preserve the packed value bytes in a unique public shape for ABI binding.
    shaped = desc('DT_INT64',[3]); viewdims = [7,64,512]
    tensor = blob('TensorDef','desc',shaped,11)+blob('TensorDef','data',np.array(viewdims,dtype='<i8').tobytes(),12)
    added.append(node('probe_attention_v_shape','Const',[],[shaped],attribute('value',blob('AttrDef','t',tensor,11))))
    viewd = desc('DT_FLOAT16',viewdims)
    added.append(node('probe_attention_v_view','Reshape',
        [('probe_attention_v:0',desc('DT_FLOAT16',CONTRACT['kv_shape'])),('probe_attention_v_shape:0',shaped)],[viewd]))
    control_receipts = []; control_outputs = []; runtime_edges = {}
    if runtime_lengths:
        # Preserve the source Attention's INT64 descriptors and values at the
        # runtime boundary. The selected DDK rejected an INT32->INT64 Cast.
        for spec in RUNTIME_LENGTH_INPUTS:
            d64 = desc('DT_INT64',[1])
            added.append(node(spec['name'],'Data',[('',d64)],[d64],
                              attribute('index',integer('AttrDef','i',len(inputs)))))
            runtime_edges[spec['port']] = spec['name']+':0'
            inputs.append(dict(name=spec['name'],role=spec['role'],dtype='int64',shape=[1]))
    if variants:
        # Compare valid mask/actual-KV-length changes without guessing the
        # meaning of all_seq_lengths_q. Both query lengths stay at source 16.
        # Keep the original Attention and source-mask readback in this graph.
        masks = {'source': (edges[4], None)}
        for control in variants:
            name = 'probe_'+control['name']
            if control['mask'] not in masks:
                mask = np.full(CONTRACT['mask_shape'],-np.inf,dtype='<f2')
                if control['mask']=='unmasked':
                    mask.fill(0)
                else:
                    mask[:,:,:,int(control['mask'][3:])] = 0
                mask_name = name+'_mask'
                tensor = blob('TensorDef','desc',maskd,11)+blob('TensorDef','data',mask.tobytes(),12)
                added.append(node(mask_name,'Const',[],[maskd],attribute('value',blob('AttrDef','t',tensor,11))))
                masks[control['mask']] = (mask_name+':0',hashlib.sha256(mask.tobytes()).hexdigest())
            mask_edge,mask_sha = masks[control['mask']]
            replacements = {} if mask_edge==edges[4] else {4:mask_edge}
            if control['kv_length']!=224:
                lengthd = desc('DT_INT64',[1]); length_name = name+'_length'
                tensor = blob('TensorDef','desc',lengthd,11)+blob('TensorDef','data',
                    np.array([control['kv_length']],dtype='<i8').tobytes(),12)
                added.append(node(length_name,'Const',[],[lengthd],attribute('value',blob('AttrDef','t',tensor,11))))
                replacements[6] = length_name+':0'
            replacements.update({port:runtime_edges[port] for port in control.get('runtime_ports',[])})
            parts = []; slot = 0
            for f,_,e,p in _parts(changed[op['name']]):
                if f==_field(schema,'OpDef','name',9,False):
                    e = _blob(f,name)
                elif f==input_field:
                    if slot in replacements:
                        e = _blob(f,replacements[slot])
                    slot += 1
                parts.append(e)
            clone = _new_id(b''.join(parts),schema,next_id); next_id += 1
            expected_clone = deepcopy(expected)
            expected_clone.update(name=name,id=next_id-1)
            for slot,edge in replacements.items():
                expected_clone['input'][slot] = edge
            if schema.decode(clone,'OpDef')!=expected_clone:
                raise ValueError('Attention control changed more than declared mask/length edges')
            added.append(clone)
            shape_name = name+'_shape'; dims = control['shape']; shaped = desc('DT_INT64',[len(dims)])
            tensor = blob('TensorDef','desc',shaped,11)+blob('TensorDef','data',np.array(dims,dtype='<i8').tobytes(),12)
            added.append(node(shape_name,'Const',[],[shaped],attribute('value',blob('AttrDef','t',tensor,11))))
            outputd = desc('DT_FLOAT16',dims)
            added.append(node(name+'_view','Reshape',[(name+':0',desc('DT_FLOAT16',CONTRACT['query_shape'])),
                (shape_name+':0',shaped)],[outputd]))
            control_outputs.append((name+'_view:0',outputd))
            control_receipts.append(dict(control,node=name,mask_edge=mask_edge,mask_sha256=mask_sha,
                changed_input_ports=sorted(replacements),actual_seq_lengths_q=16,all_seq_lengths_q=16,
                payload_sha256=hashlib.sha256(clone).hexdigest()))
    net = node('NetOutput','NetOutput',[(op['name']+':0',desc('DT_FLOAT16',CONTRACT['query_shape'])),
        ('probe_attention_visible:0',maskd),('probe_attention_v_view:0',viewd)]+control_outputs,[])
    # New QKV Data precedes Attention; auxiliary consumers follow it.
    selected.add(op['name'])
    kept = [changed.get(n['name'],p) for n,p,_ in rows if n['name'] in selected]
    reduced = blob('GraphDef','name','draft_source_attention_probe')+b''.join(
        _blob(op_field,p) for p in added[:3]+kept+added[3:]+[net])
    extracted = reduced if graph_field is None else b''.join(
        _blob(f,reduced) if f==graph_field else e for f,_,e,_ in _parts(raw))
    _decode(schema,extracted)
    original_probe_sha = hashlib.sha256(extracted).hexdigest()
    extracted, reshape = fold_reshape_chains(extracted,schema)
    extracted, zeros = materialize_zeros(extracted,schema)
    extracted, fills = materialize_fills(extracted,schema)
    prepared_kind,_,_ = _decode(schema,extracted)
    prepared_graph,_ = _graph_raw(extracted,schema,prepared_kind)
    prepared_rows,_ = _node_bytes(prepared_graph,schema)
    if _Shapes(prepared_rows,schema).shape(edges[4]) != CONTRACT['mask_shape']:
        raise ValueError('cannot prove prepared source mask shape')
    receipt = dict(policy='source-first-attention-mask-packed-qkv-v2',
        mask_observation='EQUAL_ZERO_SELECT_FP16_ONE_OR_ZERO; EXACT_BYTES; AUXILIARY_BRANCH',
        source_air_sha256=hashlib.sha256(raw).hexdigest(), extracted_air_sha256=original_probe_sha,
        probe_air_sha256=hashlib.sha256(extracted).hexdigest(), attention=op['name'],
        source_mask_edge=edges[4], source_input_roles={k:roles[k] for k in CONTROLS},
        preserved_payload_sha256={name:hashlib.sha256(payloads[name]).hexdigest() for name in sorted(selected-set(changed))},
        attention_before_sha256=hashlib.sha256(payloads[op['name']]).hexdigest(),
        attention_after_cut_sha256=hashlib.sha256(changed[op['name']]).hexdigest(),
        preparation=dict(reshape=reshape,zeros=zeros,fills=fills),
        source_sidecar_nodes=[nodes[n] for n in sorted(selected) if nodes[n]['type']=='FileConstant'],
        changed='Cut QKV at original packed Attention ports; renumber three controls; add mask predicate and V view outputs; apply existing static shape/zero/fill preparation',
        excluded='Embedding, projections, RoPE, QKV packing operators, residuals, MLP, head, full-graph partition and later Attention layers',
        limitation='Synthetic packed inputs test the source mask and Attention interface; extra outputs may change fusion/partition. Readbacks are logical edges, not physical kernel inputs.',
        graph=dict(inputs=inputs,outputs=[
            dict(name='attention',dtype='float16',shape=CONTRACT['query_shape'],exact_bytes=False),
            dict(name='mask_visible',dtype='float16',shape=CONTRACT['mask_shape'],exact_bytes=True),
            dict(name='packed_v',dtype='float16',shape=viewdims,exact_bytes=True)]))
    if variants:
        receipt.update(policy=('source-first-attention-runtime-lengths-v2' if runtime_lengths
                               else 'source-first-attention-mask-length-controls-v1'),
            control_experiment=dict(branches=control_receipts,
                original_attention='PRESERVED; BASELINE_MUST_BE_RECHECKED_IN_CHANGED_COMPILED_GRAPH',
                query_lengths='SOURCE_16_PRESERVED_AT_PORTS_5_AND_16',
                mask_visible='ORIGINAL_SOURCE_MASK_AUXILIARY_BRANCH; NOT_CONTROL_MASK_READBACK',
                scope='DIAGNOSTIC_BRANCHES_ONLY; NOT_FULL_DRAFT_FIX_OR_LENGTH_PORT_PROOF'))
        receipt['graph']['outputs'].extend(dict(name=c['name'],dtype='float16',shape=c['shape'],exact_bytes=False)
                                           for c in variants)
        receipt['changed'] += '; clone four Attention branches changing only mask and optional actual_seq_lengths_kv edges, with unique output views'
        if runtime_lengths:
            receipt['control_experiment'].update(
                runtime_length_inputs=[dict(s,dtype='int64',attention_dtype='int64',edge=runtime_edges[s['port']])
                                       for s in RUNTIME_LENGTH_INPUTS],
                query_lengths='VALUES_16_PRESERVED; PORT16_SOURCE_OR_RUNTIME_AS_DECLARED',
                scope='CONSTANT_VS_RUNTIME_LENGTH_SOURCES; VALUES_UNCHANGED; NOT_FULL_DRAFT',
                limitation='DIRECT_INT64_DATA_INPUTS; MAY_CHANGE_PLACEMENT; NO_PHYSICAL_LENGTH_READBACK')
            receipt['changed'] = ('Preserve original Attention/mask/V outputs; add two direct INT64 length inputs; '
                                  'clone four Attention branches with declared port6/16 sources and one zero-mask control; values remain16/224')
    return extracted,receipt


def reference_case(name, config, *, controls=False, runtime_lengths=False):
    """Independent ND reference, then explicit NZ coordinate assignment.

    Uniform-score cases expose cache/block visibility and anchor contribution.
    The weighted case additionally exercises QK scores and GQA heads.
    """
    control_variants = _control_variants(controls,runtime_lengths)
    variants = ((0,0,15,2,False),(0,0,15,-2,False),(16,2,15,2,False),
                (18,0,1,2,False),(192,0,15,2,False),(18,0,15,2,True),(0,16,0,2,False))
    if name not in CASES:
        raise ValueError('unknown attention case')
    start,valid,count,anchor,weighted = variants[CASES.index(name)]
    end = start+valid
    q = np.zeros((32,16,128),dtype='<f2'); k = np.zeros((8,224,128),dtype='<f2')
    if weighted:
        q[:,:,0] = (np.arange(32)[:,None]%4+1)/4
        q[:,:,1] = (np.arange(16)[None,:]-8)/8
        k[:,:,0] = (np.arange(224)[None,:]%13-6)/2
        k[:,:,1] = (np.arange(8)[:,None]+1)/4
    v = np.empty((8,224,128),dtype='<f2')
    for h in range(8):
        for row in range(224):
            base = ((row%17-8)/16 if row<192 else (row-192)/8 if row<208 else 8+(row-208)/4)
            v[h,row] = base+h/16+(np.arange(128)%16)/128
    v[:,192,:] += anchor
    visible = np.zeros((16,224),dtype=bool)
    for query in range(16):
        for key in range(208):
            pos = key if key<192 else end+key-192
            valid_key = key<end if key<192 else key-192<=count
            allowed = valid_key
            if config.layer_types[0]=='sliding_attention':
                distance = end+query-pos
                allowed &= distance>=0 and (not config.use_sliding_window or distance<config.sliding_window)
            visible[query,key] = allowed or (query>count and key==192)
    out = np.empty_like(q)
    for h in range(32):
        for row in range(16):
            indices = np.flatnonzero(visible[row])
            scores = k[h//4,indices].astype(np.float64)@q[h,row].astype(np.float64)*CONTRACT['scale_value']
            p = np.exp(scores-scores.max()); p /= p.sum()
            out[h,row] = p@v[h//4,indices].astype(np.float64)
    packed = []
    for logical in (q,out):
        p = np.empty(CONTRACT['query_shape'],dtype='<f2')
        for h in range(32):
            for d in range(128):
                p[0,h*8+d//16,:,d%16] = logical[h,:,d]
        packed.append(p)
    kv = []
    for logical in (k,v):
        p = np.empty(CONTRACT['kv_shape'],dtype='<f2')
        for h in range(8):
            for row in range(224):
                for d in range(128):
                    p[row//32,h*8+d//16,row%32,d%16] = logical[h,row,d]
        kv.append(p)
    inputs = [np.array([s],dtype='<i4') for s in (start,valid,count)]+[packed[0],*kv]
    outputs = [packed[1],visible.astype('<f2').reshape(CONTRACT['mask_shape']),kv[1].reshape(7,64,512)]
    if runtime_lengths:
        inputs.extend(np.array([s['value']],dtype='<i8') for s in RUNTIME_LENGTH_INPUTS)
    if control_variants:
        for control in control_variants:
            logical = _diagnostic_sdpa(q.astype(np.float64),k.astype(np.float64),v.astype(np.float64),
                                       _control_visibility(control,visible))
            # Same NZ output coordinates; the public view changes shape only.
            output = logical.reshape(32,16,8,16).transpose(0,2,1,3).copy()
            outputs.append(output.reshape(control['shape']))
    return inputs,outputs,dict(start_position=start,valid_rows=valid,context=end,proposal_count=count,
        synthetic_anchor_delta=anchor,weighted_scores=weighted,
        visible_keys_per_query=visible.sum(axis=1).tolist(),
        visible_temporary_keys_per_query=visible[:,192:208].sum(axis=1).tolist())


def _replace_failed(root, replace):
    """Reuse only a known failed, host-only compile; never reset a board run."""
    if not root.exists():
        return
    if not replace:
        raise ValueError('attention probe output exists; use --replace only for a failed host compilation')
    if root.parent != Path(os.environ['AI_RUN_DIR']).resolve()/'draft/tests':
        raise ValueError('failed attention probe replacement requires a direct draft/tests child')
    report_paths = (root/'result.json',root/'compilation/report.json')
    if (not root.is_dir() or any(p.is_symlink() or p.resolve()!=p for p in report_paths)
            or (root/'request.json').exists() or (root/'model.omc').exists()):
        raise ValueError('unexpected probe files or model/request present; output retained')
    state, compilation = (read_json(p) for p in report_paths)
    if (state.get('artifact_kind')!=KIND or state.get('status')!='FAILED' or state.get('stage')!='compile'
            or state.get('model_compilation')!='FAILED' or state.get('device_execution')!='NOT_RUN'
            or state.get('numerical_validation')!='NOT_RUN'
            or compilation.get('artifact_kind')!='qwen35-existing-air-compilation'
            or compilation.get('status')!='FAILED' or compilation.get('stage')!='omg'
            or compilation.get('model_compilation')!='FAILED' or compilation.get('device_execution')!='NOT_RUN'
            or type(compilation.get('returncode')) is not int or compilation['returncode']==0):
        raise ValueError('known failed OMG exit with no device execution is required; output retained')
    # Older probe versions did not hold the sibling lock. Also detect their
    # surviving OMG processes by command paths and compiler working directory.
    for proc in Path('/proc').glob('[0-9]*'):
        if proc.name==str(os.getpid()):
            continue
        try:
            args = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd = proc.joinpath('cwd').resolve(strict=True)
        except OSError:
            continue
        if (cwd==root or root in cwd.parents or any(
                arg.split('=',1)[-1]==str(root) or arg.split('=',1)[-1].startswith(str(root)+'/') for arg in args)):
            raise ValueError('process still references attention probe output; retained')
    shutil.rmtree(root)


def compile_probe(air_export, source, result, output, *, ddk_root, ddk_id, tbe_diagnosis,
                  timeout_seconds=1800, progress=print, replace=False, controls=False, runtime_lengths=False,
                  matrix_variant=None):
    _control_variants(controls,runtime_lengths)
    if matrix_variant is not None:
        from .attention_matrix import VARIANTS
        if matrix_variant not in VARIANTS or controls or runtime_lengths or replace:
            raise ValueError('matrix variant requires an independent fresh compile')
    root = output_path(output)
    if Path(output).absolute()!=root:
        raise ValueError('attention output must not contain symlink or traversal components')
    _disjoint(root,air_export,source,result,ddk_root,tbe_diagnosis)
    root.parent.mkdir(parents=True,exist_ok=True)
    # Shared with validate_draft_vector's source snapshot lock.
    with (root.parent/('.'+root.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _compile_probe(air_export,source,result,root,ddk_root=ddk_root,ddk_id=ddk_id,
            tbe_diagnosis=tbe_diagnosis,timeout_seconds=timeout_seconds,progress=progress,replace=replace,
            controls=controls,runtime_lengths=runtime_lengths,matrix_variant=matrix_variant)


def _compile_probe(air_export, source, result, output, *, ddk_root, ddk_id, tbe_diagnosis,
                   timeout_seconds, progress, replace, controls, runtime_lengths, matrix_variant=None):
    from qwen35_moe_dflash.toolchain.existing_air import _snapshot, compile_existing_air
    from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    air,prepared,root = Path(air_export).resolve(),Path(source).resolve(),output_path(output)
    _disjoint(root,air,prepared,result,ddk_root)
    if root.exists() and not replace:
        raise ValueError('attention probe output exists; use --replace only for a failed host compilation')
    manifest = read_json(air/'manifest.json'); plan = read_json(prepared/'prepared.json')
    if manifest.get('artifact_kind')!=AIR_KIND or manifest.get('status')!='PYTORCH_DRAFT_AIR_EXPORTED':
        raise ValueError('expected completed PyTorch AIR')
    cfg = Qwen35DFlashConfig.from_dict(plan['draft_config'])
    if (cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim,cfg.num_hidden_layers)!=(32,8,128,6):
        raise ValueError('reference configuration differs from bounded attention probe')
    with (air/'export.air').open('rb') as stream:
        raw = stream.read((64<<20)+1)
    if not 0<len(raw)<=64<<20:
        raise ValueError('AIR exceeds 64 MiB')
    recovered = Path(result).resolve(); observed = read_json(recovered/'result.json',4<<20)
    fixture = Path(observed['fixture']); fixture = fixture if fixture.is_absolute() else recovered/fixture
    binding = read_json(fixture/'pytorch-abi.json'); meta = read_json(fixture/'manifest.json')
    if (observed.get('device_execution')!='EXECUTED_OUTPUTS_RETRIEVED'
            or sha256_file(fixture/'manifest.json')!=observed['fixture_sha256']
            or binding['source_air_sha256']!=hashlib.sha256(raw).hexdigest()
            or binding['prepared_sha256']!=sha256_file(prepared/'prepared.json')
            or binding['packed_model_sha256']!=meta['model']['sha256']):
        raise ValueError('recovered full model source/config binding differs')
    schema = _schema((air/'air-schema.pb').read_bytes(),manifest['schema'])
    extracted,receipt = extract_attention(raw,schema,manifest,controls=controls,runtime_lengths=runtime_lengths)
    case_names = CASES
    if matrix_variant is not None:
        from .attention_matrix import graph, names
        extracted,receipt = graph(extracted,schema,receipt,matrix_variant)
        case_names = names(matrix_variant)
    snapshots = []
    widths = {0:4,1:2,3:4,9:8,12:1}  # GE operator attribute numbers.
    total = 0
    for node in receipt.pop('source_sidecar_nodes'):
        attrs = _attributes(node); relative = attrs['file_path']['s']
        shape = attrs.get('shape',{}).get('list',{}).get('i')
        dtype = attrs.get('dtype',{}).get('i')
        if (not isinstance(shape,list) or any(type(d)is not int or d<=0 for d in shape) or dtype not in widths):
            raise ValueError('small buffer has unsupported shape/dtype: '+relative)
        size = math.prod(shape)*widths[dtype]; total += size
        if size>128<<10 or total>256<<10:
            raise ValueError('source attention buffers exceed bounded small-probe budget')
        path = air/relative
        if path.is_symlink() or path.stat().st_size != size:
            raise ValueError('source buffer extent differs: '+relative)
        digest = sha256_file(path)
        checked_file(air,dict(path=relative,bytes=size,sha256=digest))
        data = path.read_bytes()
        if relative.endswith('target_attention_block_table') and data!=np.arange(7,dtype='<i4').tobytes():
            raise ValueError('source block table is not identity 0..6')
        if relative.endswith('target_attention_query_lengths') and data!=np.array([16],dtype='<i8').tobytes():
            raise ValueError('source all_seq_lengths_q differs')
        snapshots.append((path,relative,digest,size))
    root.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root.parent).free<128<<20:
        raise OSError('small attention probe needs 128 MiB free')
    _replace_failed(root,replace)
    root.mkdir()
    state = dict(artifact_kind=KIND,status='FAILED',stage='extract',model_compilation='NOT_RUN',
        device_execution='NOT_RUN',numerical_validation='NOT_RUN',ddk_id=ddk_id,scope=SCOPE,extraction=receipt,
        original_model_sha256=meta['model']['sha256'],board_attempt=observed.get('board_attempt'),
        prepared_sha256=sha256_file(prepared/'prepared.json'),cases=[],source_sidecars={})
    if matrix_variant is not None:
        from .attention_matrix import SCOPE as matrix_scope
        state['scope'] = matrix_scope
    _save(root/'result.json',state)
    try:
        for path,relative,digest,size in snapshots:
            snap = _snapshot(path,root/'air'/relative)
            if snap['sha256']!=digest:
                raise ValueError('source buffer changed during snapshot')
            state['source_sidecars'][relative] = dict(bytes=size,sha256=digest)
        (root/'air').mkdir(exist_ok=True); (root/'air/export.air').write_bytes(extracted)
        for name in case_names:
            if matrix_variant is None:
                ins,outs,detail = reference_case(name,cfg,controls=controls,runtime_lengths=runtime_lengths)
            else:
                from .attention_matrix import reference_case as matrix_case
                ins,outs,detail = matrix_case(matrix_variant,name,receipt['graph'])
            folder = root/'fixture'/name; folder.mkdir(parents=True)
            for prefix,arrays in (('input',ins),('golden',outs)):
                for i,array in enumerate(arrays):
                    array.tofile(folder/f'{prefix}_{i}.bin')
            state['cases'].append(dict(case=name,controls=detail,fixtures={p.name:dict(bytes=p.stat().st_size,
                sha256=sha256_file(p)) for p in folder.iterdir()}))
        state['stage']='compile'; _save(root/'result.json',state)
        progress('compiling one '+('matrix '+matrix_variant if matrix_variant else
                                  'source Attention plus four runtime-length controls' if runtime_lengths else
                                  'source Attention plus four mask/length controls' if controls else 'source-mask/Attention')
                 +f' model for {len(case_names)} fixtures; no learned weights')
        compilation = compile_existing_air(root/'air/export.air',root/'compilation',ddk_root,ddk_id=ddk_id,
            tbe_diagnosis=tbe_diagnosis,timeout_seconds=timeout_seconds,input_layout='air-directory',
            use_origin_format=True,save_weights_as_external_data=True,progress=lambda r:progress(r['message']))
        state['model_compilation']=compilation['model_compilation']
        if compilation['status']!='COMPILED':
            raise ValueError(str(compilation.get('error',compilation['status'])))
        state['stage']='pack'
        sidecars = [dict(r,path=str(root/'compilation'/r['path'])) for r in compilation['external_data_candidates']
                    if Path(r['path']).suffix=='.weight']
        packing = embed_external_weights(root/'compilation'/compilation['omc']['path'],root/'model.omc',
            Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so',
            expected_source_sha256=compilation['omc']['sha256'],sidecars=sidecars,progress=progress)
        _save(root/'weight-packing.json',packing)
        if packing['status']!='EMBEDDED' or (root/'model.omc').stat().st_size>16<<20:
            raise ValueError('attention probe packing failed or exceeded 16 MiB')
        state.update(status='COMPILED',stage='complete',model=dict(path='model.omc',
            bytes=(root/'model.omc').stat().st_size,sha256=sha256_file(root/'model.omc')))
    except Exception as error:
        state.update(error=f'{type(error).__name__}: {error}',compilation_diagnostics=compilation_diagnostics(root/'compilation'))
    _save(root/'result.json',state)
    return state


def _compiled_probe(root, ddk, ddk_id):
    from qwen35_moe_dflash.toolchain.omc_container import read_container,read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    root = Path(root)
    report = read_json(root/'result.json')
    if (report.get('artifact_kind')!=KIND or report.get('status')!='COMPILED'
            or report.get('model_compilation')!='PASSED' or report.get('ddk_id')!=ddk_id):
        raise ValueError('expected compiled source-attention probe for selected DDK')
    model = checked_file(root,report['model'])
    if model.stat().st_size>16<<20:
        raise ValueError('attention model exceeds 16 MiB')
    schema = load_schema(Path(ddk)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    main = [p for p in read_container(model)['partitions'] if p['type']==0]
    if len(main)!=1:
        raise ValueError('attention probe needs exactly one main graph')
    document = schema.decode(read_partition(model,main[0]),'ModelDef')
    return report,model,schema,document,_enum_fields(schema)


def inspect_compiled_probe(root, ddk, ddk_id):
    """Inspect compiled boundary references and the proposed runtime plan.

    Runtime dimensions cannot distinguish the equal-shaped K/V inputs or the
    three controls. Data.index, graph boundary references and serialized node
    order are reported separately. Only explicit boundary references bind the
    attention fixtures; runtime checks and numerical validation remain pending.
    """
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    from qwen35_moe_dflash.toolchain.omc_inspect import _preview
    from validate_draft_vector import bind_io
    report,model,schema,document,enums = _compiled_probe(root,ddk,ddk_id)
    graphs = document.get('graph',[])
    if len(graphs)!=1 or len(graphs[0].get('op',[]))>512:
        raise ValueError('attention interface inspection needs one bounded main graph')
    graph=graphs[0]; declared=report['extraction']['graph']
    expected={s['name']:s for s in declared['inputs']}
    data=[]; nets=[]
    def describe(d):
        return dict(_observed_desc(d,enums),recorded=_preview(d))
    for ordinal,node in enumerate(graph.get('op',[])):
        if node.get('type') in ('Data','RefData'):
            data.append(dict(op_ordinal=ordinal,data_ordinal=len(data),name=node['name'],type=node['type'],
                role=expected.get(node['name'],{}).get('role'),
                indices=[a.get('value',{}).get('i') for a in node.get('attr',[]) if a.get('key')=='index'],
                metadata=_preview({k:v for k,v in node.items() if k not in ('input_desc','output_desc')}),
                inputs=[describe(d) for d in node.get('input_desc',[])[:8]],
                outputs=[describe(d) for d in node.get('output_desc',[])[:8]]))
        elif node.get('type')=='NetOutput':
            nets.append(dict(name=node['name'],input=node.get('input',[]),
                             descriptors=[describe(d) for d in node.get('input_desc',[])[:8]]))
    result=dict(artifact_kind='qwen35-draft-attention-interface-inspection',status='INSPECTED',
        model=dict(report['model'],path=str(model)),ddk_id=ddk_id,schema=schema.evidence,
        model_compilation='NOT_RUN',device_execution='NOT_RUN',numerical_validation='NOT_RUN',
        scope='READ_ONLY_COMPILED_INPUT_METADATA; NO_FIXTURE_WRITES_NO_EXECUTION',
        declared=declared,model_metadata=_preview({k:v for k,v in document.items() if k!='graph'}),
        graph_metadata=_preview({k:v for k,v in graph.items() if k!='op'}),
        graph_input_references=graph.get('input',[]),data_nodes=data,netoutputs=nets,
        runtime_input_order='UNRESOLVED; DATA_INDEX_AND_SERIALIZED_NODE_ORDER_ARE_REPORTED_SEPARATELY',
        same_descriptor_input_groups=[names for names in (
            [s['name'] for s in declared['inputs'] if (s['dtype'],s['shape'])==(dtype,list(shape))]
            for dtype,shape in sorted({(s['dtype'],tuple(s['shape'])) for s in declared['inputs']})) if len(names)>1])
    try:
        result['current_binding']=bind_io(document,declared,enums,input_order='graph-input')
        result['current_binding_status']='GENERATED_NOT_RUNTIME_VALIDATED'
        result['runtime_input_order']='COMPILED_GRAPHDEF_INPUT_REFERENCES; RUNTIME_VALIDATION_PENDING'
    except (ValueError,KeyError,TypeError) as error:
        result.update(current_binding_status='FAILED',binding_error=f'{type(error).__name__}: {error}')
    return result


def prepare_fixture(root, out, ddk, ddk_id, selection=None):
    from qwen35_moe_dflash.execution.hdc_run import _file_record
    from .torch_export import _compiled_observation
    from validate_draft_vector import bind_io,selected_cases,WARMUP,REPETITIONS
    root,out = Path(root),Path(out)
    report,model,schema,document,enums = _compiled_probe(root,ddk,ddk_id)
    case_names = CASES
    variant = report['extraction'].get('matrix_variant')
    if variant is not None:
        from .attention_matrix import names
        case_names = names(variant)
    selected = selected_cases(case_names if selection is None else selection,case_names)
    io = bind_io(document,report['extraction']['graph'],enums,input_order='graph-input')
    rows = {r['case']:r for r in report['cases']}
    if len(rows)!=len(case_names) or set(rows)!=set(case_names):
        raise ValueError('missing/duplicate attention fixtures')
    observed = _compiled_observation(model,ddk)
    cases=[]
    for name in selected:
        bound = deepcopy(io); folder=out/'fixture'/name; folder.mkdir(parents=True)
        plan = ['QWEN35_TENSOR_PROBE 1',f'iterations {WARMUP} {REPETITIONS}']
        for direction,prefix in (('inputs','input'),('outputs','golden')):
            for spec in bound[direction]:
                member = f'{prefix}_{spec["fixture_index"]}.bin'
                path = checked_file(root/'fixture'/name,dict(rows[name]['fixtures'][member],path=member))
                if path.stat().st_size!=spec['bytes']:
                    raise ValueError('attention fixture extent differs from compiled ABI')
                dest=folder/f'{prefix}_{spec["index"]}.bin'; shutil.copyfile(path,dest)
                spec['fixture_sha256']=sha256_file(dest)
                plan.append(' '.join((direction[:-1],str(spec['index']),spec['name'],spec['dtype'],','.join(map(str,spec['shape'])))))
        (folder/'plan.txt').write_text('\n'.join(plan)+'\n')
        cases.append(dict(case=name,**bound,model=_file_record(model,name+'/model.omc'),
                          controls=rows[name]['controls'],graphop_counts=observed['graphop_counts']))
    meta = dict(cases=cases,warmup=WARMUP,repetitions=REPETITIONS,atol=0.01,rtol=0.01,
                exact_bytes=False,scope=SCOPE,ddk_id=ddk_id,schema=schema.evidence,source_evidence=report)
    _save(out/'fixture/manifest.json',meta)
    return meta


def _query_nd(packed):
    """Interpret the declared query/output NZ coordinates, not device memory."""
    if packed.shape != tuple(CONTRACT['query_shape']):
        raise ValueError('attention diagnosis query/output geometry differs')
    return packed.reshape(32,8,16,16).transpose(0,2,1,3).reshape(32,16,128)


def _kv_nd(packed):
    if packed.shape != tuple(CONTRACT['kv_shape']):
        raise ValueError('attention diagnosis KV geometry differs')
    return packed.reshape(7,8,8,32,16).transpose(1,0,3,2,4).reshape(8,224,128)


def _diagnostic_sdpa(q,k,v,visible):
    """FP64 logical SDPA, rounded once to FP16; no device-dependent reference."""
    if visible.shape != (16,224) or visible.dtype != np.bool_ or not visible.any(axis=1).all():
        raise ValueError('attention hypothesis requires at least one visible key per query')
    scores = np.matmul(q,k[np.arange(32)//4].transpose(0,2,1))*CONTRACT['scale_value']
    scores = np.where(visible[None],scores,-np.inf)
    p = np.exp(scores-scores.max(axis=-1,keepdims=True))
    p /= p.sum(axis=-1,keepdims=True)
    return np.matmul(p,v[np.arange(32)//4]).astype('<f2')


def _hypothesis_difference(actual, expected, atol, rtol):
    from .embedding_diagnose import _difference
    metrics = _difference(actual,expected,atol,rtol)
    metrics.update(bytes_equal=actual.tobytes()==expected.tobytes(),
        bitwise_mismatched_elements=int(np.count_nonzero(actual.view('<u2')!=expected.view('<u2'))),
        rmse=float(np.sqrt(np.mean((actual.astype(np.float64)-expected)**2))) if metrics['finite'] else None)
    return metrics


def _retained_qkv(fixture,case):
    packed = {}
    for role in ('q','k','v'):
        specs = [s for s in case['inputs'] if s.get('role')==role]
        shape = CONTRACT['query_shape'] if role=='q' else CONTRACT['kv_shape']
        if (len(specs)!=1 or specs[0]['dtype']!='float16' or specs[0]['shape']!=shape
                or specs[0]['bytes']!=math.prod(shape)*2):
            raise ValueError('attention diagnosis requires one hash-bound packed '+role+' input')
        spec=specs[0]
        path=checked_file(fixture/case['case'],dict(path=f'input_{spec["index"]}.bin',
            bytes=spec['bytes'],sha256=spec['fixture_sha256']))
        packed[role]=np.fromfile(path,dtype='<f2').reshape(shape)
        if not np.isfinite(packed[role]).all():
            raise ValueError('nonfinite attention diagnostic input: '+role)
    q=_query_nd(packed['q']).astype(np.float64)
    k,v=(_kv_nd(packed[role]).astype(np.float64) for role in ('k','v'))
    return q,k,v


def _attention_signatures(fixture,case,actual,reference,visible,atol,rtol):
    """Compare bounded hypotheses against every output coordinate in call 0."""
    q,k,v=_retained_qkv(fixture,case)
    a,e=_query_nd(actual),_query_nd(reference)
    source=_diagnostic_sdpa(q,k,v,visible)
    hypotheses=[]
    for length in (16,32,192,208,224):
        mask=np.broadcast_to(np.arange(224)<length,(16,224))
        prediction=_diagnostic_sdpa(q,k,v,mask)
        hypotheses.append(dict(hypothesis='unmasked_prefix_'+str(length),key_interval=[0,length],
            **_hypothesis_difference(a,prediction,atol,rtol),
            head0_channel0_first=float(prediction[0,0,0])))
    def distinct_rows(x):
        rows=x.transpose(1,0,2).copy().reshape(16,-1)
        return len({row.tobytes() for row in rows})
    return dict(scope='ITERATION_0_ALL_32_HEADS_16_QUERIES_128_CHANNELS; DIAGNOSTIC_HYPOTHESES_ONLY',
        input_scope='HASH_BOUND_HOST_FIXTURES_NOT_PHYSICAL_KERNEL_INPUT_READBACK',
        precision='FP64_LOGICAL_SDPA_ROUNDED_TO_FP16; INTERNAL_KERNEL_PRECISION_NOT_INFERRED',
        comparison_tolerances=dict(atol=atol,rtol=rtol),
        reference_reproduction=_hypothesis_difference(e,source,atol,rtol),
        declared_visibility=_hypothesis_difference(a,source,atol,rtol),
        unique_query_rows_bitwise=dict(actual=distinct_rows(a),reference=distinct_rows(e)),
        unmasked_prefix_hypotheses=hypotheses,
        interpretation='MATCH_IS_A_NUMERICAL_SIGNATURE_NOT_PROOF_OF_LENGTH_PORT_OR_MASK_KERNEL_BEHAVIOR; ORIGINAL_GATE_UNCHANGED')


def _control_observations(fixture,case,arrays,atol,rtol,*,runtime_lengths=False):
    length_values = []
    if runtime_lengths:
        for declared in RUNTIME_LENGTH_INPUTS:
            specs=[s for s in case['inputs'] if s.get('role')==declared['role']]
            if len(specs)!=1 or (specs[0]['dtype'],specs[0]['shape'],specs[0]['bytes'])!=('int64',[1],8):
                raise ValueError('runtime length fixture ABI differs: '+declared['role'])
            spec=specs[0]
            path=checked_file(fixture/case['case'],dict(path=f'input_{spec["index"]}.bin',
                bytes=8,sha256=spec['fixture_sha256']))
            value=int(np.fromfile(path,dtype='<i8')[0])
            if value!=declared['value']:
                raise ValueError('runtime length fixture value differs: '+declared['role'])
            length_values.append(dict(role=declared['role'],value=value,
                scope='HASH_BOUND_HOST_FIXTURE_NOT_DEVICE_LENGTH_READBACK'))
    q,k,v=_retained_qkv(fixture,case)
    prefix=_diagnostic_sdpa(q,k,v,np.broadcast_to(np.arange(224)<16,(16,224)))
    source=_query_nd(arrays['attention'][0]); rows=[]; nd={}
    visibility=arrays['mask_visible'][1].reshape(16,224)==1
    for control in _control_variants(not runtime_lengths,runtime_lengths):
        actual,expected=arrays[control['name']]
        actual,expected=(_query_nd(x.reshape(CONTRACT['query_shape'])) for x in (actual,expected))
        prediction=_diagnostic_sdpa(q,k,v,_control_visibility(control,visibility))
        nd[control['name']]=(actual,expected)
        rows.append(dict(name=control['name'],mask=control['mask'],actual_seq_lengths_kv=control['kv_length'],
            runtime_input_ports=control.get('runtime_ports',[]),
            comparison=_hypothesis_difference(actual,expected,atol,rtol),
            reference_reproduction=_hypothesis_difference(expected,prediction,atol,rtol),
            source_output_difference=_hypothesis_difference(actual,source,atol,rtol),
            unmasked_prefix_16=_hypothesis_difference(actual,prefix,atol,rtol),
            head0_channel0=dict(actual=[float(x) if np.isfinite(x) else None for x in actual[0,:,0]],
                               expected=expected[0,:,0].astype(float).tolist())))
    pairs=[]
    pairs_to_compare=(('attention_runtime_q','attention_runtime_both'),
                      ('attention_runtime_kv','attention_runtime_both'),
                      ('attention_runtime_both','attention_runtime_unmasked')) if runtime_lengths else (
                      ('attention_key0','attention_key15'),('attention_unmasked','attention_kv32'))
    for left,right in pairs_to_compare:
        a,e=nd[left]; b,f=nd[right]
        pairs.append(dict(outputs=[left,right],actual=_hypothesis_difference(a,b,atol,rtol),
                          reference=_hypothesis_difference(e,f,atol,rtol)))
    return dict(scope='ITERATION_0_ALL_65536_ELEMENTS_PER_BRANCH; TEN_CALL_GATES_IN_COMPARISON',
        query_lengths=dict(actual_seq_lengths_q=16,all_seq_lengths_q=16),branches=rows,pair_differences=pairs,
        runtime_length_inputs=length_values,
        limitation='ADDED_BRANCHES_MAY_CHANGE_COMPILATION; CONSTANT_MASKS_NOT_PHYSICAL_KERNEL_READBACK; NOT_FULL_DRAFT')


def diagnose(fixture,native,meta,comparison,report):
    """Expose which query rows lose temporary keys, with unchanged comparisons."""
    cases=[]; anchor_outputs={}
    for case in meta['cases']:
        if case['case'] not in comparison['validated_cases']:
            continue
        raw=(native/case['case']/'0.bin').read_bytes(); offset=0; arrays={}
        for spec in case['outputs']:
            dtype={'int32':'<i4','float16':'<f2'}[spec['dtype']]
            actual=np.frombuffer(raw[offset:offset+spec['bytes']],dtype=dtype).reshape(spec['shape'])
            expected=np.fromfile(fixture/case['case']/f'golden_{spec["index"]}.bin',dtype=dtype).reshape(spec['shape'])
            arrays[spec['name']]=(actual,expected); offset+=spec['bytes']
        a,e=arrays['mask_visible']; a,e=a.reshape(16,224),e.reshape(16,224)
        av,ev=a==1,e==1
        aq,eq=arrays['attention']
        def finite_values(x):
            return [float(v) if np.isfinite(v) else None for v in x]
        row=dict(case=case['case'],controls=case['controls'],
            comparison=next(c for c in comparison['cases'] if c['case']==case['case']),
            mask_visible=dict(mismatched_elements=int((a!=e).sum()),
                nonbinary_elements=int((~((a==0)|(a==1))).sum()),
                actual_keys_per_query=av.sum(axis=1).tolist(),expected_keys_per_query=ev.sum(axis=1).tolist(),
                actual_temporary_keys_per_query=av[:,192:208].sum(axis=1).tolist(),
                expected_temporary_keys_per_query=ev[:,192:208].sum(axis=1).tolist(),
                actual_visible_padding=int(av[:,208:].sum())),
            attention_head0_channel0=dict(actual=finite_values(aq[0,0,:,0]),expected=finite_values(eq[0,0,:,0])))
        row['comparison']={k:v for k,v in row['comparison'].items() if k not in ('tensors','api_us')}
        row['attention_signatures']=_attention_signatures(
            fixture,case,aq,eq,ev,meta['atol'],meta['rtol'])
        if meta['source_evidence']['extraction'].get('control_experiment'):
            row['attention_controls']=_control_observations(fixture,case,arrays,meta['atol'],meta['rtol'],
                runtime_lengths=meta['source_evidence']['extraction']['policy'].startswith('source-first-attention-runtime-lengths-'))
        if case['case'] in CASES[:2]:
            anchor_outputs[case['case']]=(aq,eq)
        cases.append(row)
    anchor_response=dict(status='NOT_EVALUATED_PAIR_NOT_SELECTED')
    if all(name in anchor_outputs for name in CASES[:2]):
        left,right=(anchor_outputs[name] for name in CASES[:2])
        anchor_response=dict(status='COMPARED',cases=list(CASES[:2]),
            scope='ITERATION_0_ALL_ATTENTION_ELEMENTS; PAIR_DIFFERENCE_NOT_ACCEPTANCE_GATE',
            actual=_hypothesis_difference(left[0],right[0],meta['atol'],meta['rtol']),
            reference=_hypothesis_difference(left[1],right[1],meta['atol'],meta['rtol']))
    return dict(artifact_kind='qwen35-draft-source-attention-diagnosis',status='ANALYZED',
        numerical_validation=comparison['numerical_validation'],model_execution='NOT_RUN',model_compilation='NOT_RUN',
        scope='RETAINED_SOURCE_ATTENTION_PROBE; TEN_CALLS_RECOMPARED; DETAILS_FROM_ITERATION_0',
        board_attempt=report.get('board_attempt'),model_sha256=meta['source_evidence']['model']['sha256'],
        original_model_sha256=meta['source_evidence']['original_model_sha256'],cases=cases,
        temporary_anchor_response=anchor_response,
        control_experiment=meta['source_evidence']['extraction'].get('control_experiment'),
        limitation=meta['source_evidence']['extraction']['limitation'])

"""Inventory the original Target's vendor calls before isolated validation.

Read Python as AST data and optional AIR with the selected static schema. No
model imports, weights, compilation or device calls. Source declarations and
DDK text matches are evidence to construct cases, never numerical passes.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import os
from pathlib import Path
import re


KIND = 'qwen35-target-operator-inventory'
FILES = ('export_glm_model_qwen3_5_moe.py', 'utils.py',
         'models/export_model_wrapper_qwen3_5_moe.py',
         'models/modeling_qwen3_5_moe_hiai_cpro.py',
         'models/modeling_qwen3_5_moe_hiai_nz.py')
# GE names for builtins are search candidates until observed in an actual AIR.
FAMILIES = {
    'conv_state': ('linear_attention_convstate_update', ('LinearAttentionConvstateUpdate',)),
    'gdn': ('npu_chunk_gated_delta_rule', ('ChunkGatedDeltaRule',)),
    'cache_update': ('npu_cache_update_', ('CacheUpdate',)),
    'attention': ('adn_fused_infer_attention', ('AdnFusedInferAttention',)),
    'rms': ('npu_rms_norm', ('RmsNorm',)),
    'dynamic_quant': ('npu_dynamic_quant', ('DynamicQuant',)),
    'quant_matmul': ('npu_quant_matmul', ('QuantBatchMatmul', 'QuantBatchMatmulV3', 'QuantBatchMatMul', 'QuantBatchMatMulV3')),
    'moe_router': ('npu_moe_gating_top_k_softmax', ('MoeGatingTopKSoftmax',)),
    'moe_init': ('npu_moe_init_routing_v2', ('MoeInitRoutingV2', 'MoeInitRoutingV3')),
    'moe_gmm': ('npu_grouped_matmul', ('GroupedMatmul', 'GroupedMatMul')),
    'moe_finalize': ('npu_moe_finalize_routing', ('MoeFinalizeRouting',)),
    'quant_param': ('npu_trans_quant_param', ('TransQuantParam',)),
}
SOURCE_CONTRACT = {
    'conv_state': 'x1 [1,T,8192]; x2 [1,8192,4]; INT32 x3[128]. Control semantics and installed y2 rank require evidence.',
    'gdn': 'Q/K/V [1,T,32,128]; g/beta [1,T,32]; initial FP16 state[1,32,128,128]; INT16 length[1]; chunk64/1, QK L2 enabled. Bind g/beta and last-state dtypes from AIR/DDK.',
    'cache_update': 'FP16 cache[64,32,32,16], updates[T,32,16], INT32 targetBlock/offset[T]; compare entire cache including untouched locations.',
    'attention': 'Packed FP16 query[1,256,T,16], two cache[64,32,32,16], mask[1,1,T,2048]; Q16/KV2/D256, block32, BNSD, scale1/16, inner_precise2. Export pse_shift maps to GE all_seq_lengths_q.',
    'rms': 'Effective gamma must come from master raw HF loading plus ordinary forward +1; gated gamma is raw HF. epsilon1e-6; freeze input/gamma/output casts.',
    'dynamic_quant': 'Capture original FP16/FP32 activations; compare integer output, per-token scale and rounding/clipping separately.',
    'quant_matmul': 'Original logical M/K/N, selected quant_bits and original packed carrier/scale. Do not infer W8 from int8 checkpoint storage or silently repack W4.',
    'moe_router': '[T,256] logits, top_k8; compare expert IDs and weights; near-tie cases need an explicit tie contract. Source subsequently renormalizes.',
    'moe_init': '[T,2048] activations, [T,8] IDs, 256 experts; quant_mode=-1, active_num=8T, expert_tokens_num_type1. Compare counts, maps and reconstructed rows.',
    'moe_gmm': 'All 256 experts; group_list_type1/split_item3/group_type0/act_type0; original W4 INT32 carrier and INT64 qparam. Preserve bias, SiLU and requantization boundaries.',
    'moe_finalize': '[8T,2048] expanded rows, [T,8] scales and source row map; drop_pad_mode2. Verify weighted reconstruction independently.',
    'quant_param': 'Weight-preparation operator, outside AIR inference. Compare original floating scale with transformed INT64 qparam and its GMM consumer.',
}
SCENARIOS = {
    'conv_state': ['zero_history', 'nonzero_asymmetric_history', 'partial_valid', 'feedback'],
    'gdn': ['zero_state', 'nonzero_asymmetric_state', 'partial_valid', 'feedback', 'output_fill_00_a5_5a'],
    'cache_update': ['encoded_rows', 'block_boundary31_32', 'block_boundary63_64', 'permuted_blocks', 'feedback'],
    'attention': ['select_one_key', 'causal_nonzero_qkv', 'partial_valid', 'permuted_blocks', 'cache_update_chain'],
    'rms': ['real_gamma', 'zero_and_small', 'large_magnitude', 'source_dtype_boundaries'],
    'dynamic_quant': ['real_activations', 'zero_rows', 'signed_extrema', 'rounding_boundaries'],
    'quant_matmul': ['real_weights_activations', 'encoded_layout', 'nonuniform_group_scales'],
    'moe_router': ['separated_logits', 'near_ties', 'skewed_experts'],
    'moe_init': ['empty_experts', 'skewed_experts', 'encoded_rows'],
    'moe_gmm': ['real_weights_activations', 'empty_experts', 'skewed_counts', 'tail_rows'],
    'moe_finalize': ['encoded_rows', 'nonuniform_scales', 'routing_gmm_chain'],
    'quant_param': ['real_scales', 'nonuniform_groups'],
}


def read_source(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 2 << 20:
        raise ValueError('missing or excessive source: ' + str(path))
    raw = path.read_bytes()
    if len(raw) > 2 << 20:
        raise ValueError('source exceeds 2 MiB')
    return raw, ast.parse(raw, filename=str(path))


class Calls(ast.NodeVisitor):
    def __init__(self):
        self.scope, self.calls, self.custom = [], [], []

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if re.fullmatch(r'(torch_npu|torch\.ops\.my_ops)\.[A-Za-z_]\w*', name):
            self.calls.append(dict(scope='.'.join(self.scope), line=node.lineno,
                call=name, positional=[ast.unparse(v) for v in node.args],
                keywords={k.arg or '**': ast.unparse(k.value) for k in node.keywords}))
        if name == 'torchair.ge.custom_op' and node.args and isinstance(node.args[0], ast.Constant):
            self.custom.append(dict(scope='.'.join(self.scope), line=node.lineno,
                ge_type=node.args[0].value,
                arguments={k.arg or '**': ast.unparse(k.value) for k in node.keywords}))
        self.generic_visit(node)


def source_inventory(root):
    root = Path(root).resolve()
    files, rows, custom, trees = [], [], [], {}
    for name in FILES:
        raw, tree = read_source(root / name)
        trees[name] = tree
        files.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        visitor = Calls()
        visitor.visit(tree)
        rows.extend(dict(file=name, **row) for row in visitor.calls)
        custom.extend(dict(file=name, **row) for row in visitor.custom)
    # This is the reviewed CPro quantized path. Every call outside it is shown
    # separately; matching a callee name alone does not prove path reachability.
    active_classes = {'Qwen3_5MoeRMSNorm', 'Qwen3_5MoeRMSNormGated',
        'Qwen3_5MoeAttention', 'Qwen3_5MoeGatedDeltaNet',
        'CompressedTensorW4A8Int8MoEGMMMethod', 'Qwen3_5MoeSparseMoeBlock'}
    active, excluded = [], []
    for row in rows:
        cls = row['scope'].split('.')[0]
        selected = (row['file'].endswith('_cpro.py') and cls in active_classes
            or row['file'].endswith('_nz.py') and cls == 'QLinear'
            or row['file'] == 'utils.py' and row['scope'] == 'combined_experts_weight_quant_3d')
        (active if selected else excluded).append(row)
    known = {call: family for family, (call, _) in FAMILIES.items()}
    grouped = {family: [] for family in FAMILIES}
    unknown = []
    for row in active:
        name = row['call'].rsplit('.', 1)[-1]
        if name in known:
            grouped[known[name]].append(row)
        else:
            unknown.append(row)
    # Explicitly verify the source facts on which the reviewed path depends.
    utils = trees['utils.py']
    imported = any(isinstance(n, ast.ImportFrom)
        and n.module == 'models.modeling_qwen3_5_moe_hiai_nz'
        and any(a.name == 'QLinear' and a.asname in (None, 'QLinear') for a in n.names)
        for n in utils.body)
    wrapper = trees['models/export_model_wrapper_qwen3_5_moe.py']
    loader = next((n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == 'load_npu_model'), None)
    norm_branch = next((n for n in ast.walk(loader) if isinstance(n, ast.If)
        and ast.unparse(n.test) == "'layernorm.weight' in name or name.endswith('norm.weight')"), None) if loader else None
    raw_load = norm_branch is not None and any(isinstance(n, ast.Call)
        and ast.unparse(n) == 'param.data.copy_(hf_val.to(param.dtype))'
        for statement in norm_branch.body for n in ast.walk(statement))
    observations = []
    cpro = trees['models/modeling_qwen3_5_moe_hiai_cpro.py']
    moe = next((n for n in cpro.body if isinstance(n, ast.ClassDef)
        and n.name == 'CompressedTensorW4A8Int8MoEGMMMethod'), None)
    if moe:
        for branch in ast.walk(moe):
            if isinstance(branch, ast.If) and ast.unparse(branch.test) == 'pertoken_scale.dim() > 1':
                for statement in branch.body:
                    if isinstance(statement, ast.Assign) and ast.unparse(statement) == 'x_quant = x.view(-1, hidden_size)':
                        observations.append(dict(id='MOE_QUANT_INPUT_OVERWRITE',
                            file='models/modeling_qwen3_5_moe_hiai_cpro.py', line=statement.lineno,
                            status='CONDITIONAL_SOURCE_ISSUE_RUNTIME_TRIGGER_UNVERIFIED',
                            observation='When pertoken_scale has rank >1, x_quant is replaced by original x, discarding dynamic-quant output.',
                            next='Capture actual DynamicQuant output scale rank and the first GMM input dtype before blaming the kernel.'))
    missing = [family for family, calls in grouped.items() if not calls]
    return dict(files=files, selected_path='MASTER_CPRO_QUANTIZED; QLINEAR_IMPORTED_FROM_NZ',
        path_observations=dict(quant_linear_import_from_nz=imported, master_norm_raw_load=raw_load),
        status='SOURCE_CALLS_ACCOUNTED' if imported and raw_load and not missing and not unknown else 'SOURCE_REVIEW_REQUIRED',
        families=grouped, unknown_selected_calls=unknown, missing_expected_families=missing,
        excluded_calls=excluded, explicit_ge_converters=custom,
        observations=observations,
        excluded_ge_types=['AdnRmsNorm'],
        excluded_reason='Registered AdnRmsNorm is not called by this CPro path; ordinary/gated norm call npu_rms_norm.',
        reachability='REVIEWED_SOURCE_PATH_ONLY; ACTUAL_AIR_RECONCILIATION_REQUIRED')


def case_matrix(config):
    """Logical shapes; never present these as observed runtime descriptors."""
    cfg = config.get('text_config', config)
    expected = dict(hidden_size=2048, num_hidden_layers=40, num_experts=256,
        num_experts_per_tok=8, moe_intermediate_size=512, vocab_size=248320,
        num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('operator matrix requires the original 40-layer Target configuration')
    projections = [('gdn_qkv',2048,8192), ('gdn_z',2048,4096), ('gdn_a',2048,32),
        ('gdn_b',2048,32), ('gdn_out',4096,2048), ('attention_q_gate',2048,8192),
        ('attention_k',2048,512), ('attention_v',2048,512), ('attention_out',4096,2048),
        ('shared_gate',2048,512), ('shared_up',2048,512), ('shared_down',512,2048),
        ('router',2048,256), ('lm_head',2048,248320)]
    cases = []
    def add(family, label, t, **shape):
        cases.append(dict(id=f'{family}_{label}_t{t}', family=family, sequence=t,
            logical_shapes=shape, scenarios=SCENARIOS[family],
            contract=SOURCE_CONTRACT[family], shape_evidence='SOURCE_CONFIG_LOGICAL_ONLY',
            fixture='NOT_CAPTURED', reference='NOT_GENERATED', model_compilation='NOT_RUN',
            device_execution='NOT_RUN', numerical='NOT_RUN'))
    for t in (64,1):
        add('conv_state','core',t,x=[1,t,8192],state=[1,8192,4],control=[128])
        add('gdn','core',t,qkv=[1,t,32,128],g_beta=[1,t,32],state=[1,32,128,128],length=[1])
        add('cache_update','core',t,cache=[64,32,32,16],updates=[t,32,16],blocks=[t],offsets=[t])
        add('attention','core',t,q=[1,256,t,16],kv=[64,32,32,16],mask=[1,1,t,2048],block_table=[1,64])
        for label,shape in [('hidden',[1,t,2048]),('query',[1,t,16,256]),
                            ('key',[1,t,2,256]),('gated',[t*32,128])]:
            add('rms',label,t,x=shape,gamma=[shape[-1]])
        for label,k,n in projections:
            m = 1 if label == 'lm_head' else t
            add('quant_matmul',label,t,x=[m,k],weight_logical=[n,k],output=[m,n])
        for label,shape in [('dense2048',[t,2048]),('dense4096',[t,4096]),
                            ('dense512',[t,512]),('head',[1,2048]),
                            ('expert_input',[8*t,2048]),('expert_intermediate',[8*t,512])]:
            add('dynamic_quant',label,t,x=shape)
        add('moe_router','core',t,logits=[t,256],selected=[t,8])
        add('moe_init','core',t,x=[t,2048],expert_ids=[t,8],expanded=[8*t,2048],counts=[256])
        add('moe_gmm','gate_up',t,x=[8*t,2048],weight_logical=[256,2048,1024],scale=[256,8,1024],counts=[256])
        add('moe_gmm','down',t,x=[8*t,512],weight_logical=[256,512,2048],scale=[256,2,2048],counts=[256])
        add('moe_finalize','core',t,expanded=[8*t,2048],scales=[t,8],output=[t,2048])
    add('quant_param','gate_up',0,scale=[256,8,1024])
    add('quant_param','down',0,scale=[256,2,2048])
    return cases


def inspect_air(path, schema):
    from qwen35_moe_dflash.target.original_air_roundtrip import _decode, _attributes
    from qwen35_moe_dflash.target.original_air_inspect import _metadata_desc
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 64 << 20:
        raise ValueError('AIR must be a nonempty file <=64 MiB')
    raw = path.read_bytes()
    kind, _, graph = _decode(schema, raw)
    enums = schema.enums[schema.package+'.DataType']
    known = {name: family for family,(_,names) in FAMILIES.items() for name in names}
    nodes = graph['op']
    by_name = {n['name']:n for n in nodes}
    roots = [n for n in nodes if n.get('type')=='NetOutput']
    if len(roots)!=1:
        raise ValueError('one explicit NetOutput required for reachability')
    reachable, pending = set(), [roots[0]['name']]
    while pending:
        name=pending.pop()
        if name in reachable: continue
        if name not in by_name: raise ValueError('unresolved graph edge: '+name)
        reachable.add(name)
        pending.extend(edge.rsplit(':',1)[0] for edge in by_name[name].get('input',[]) if edge)
    selected=[]
    counts=Counter()
    for n in nodes:
        if n['name'] not in reachable: continue
        counts[n.get('type','')]+=1
        if n.get('type') not in known: continue
        selected.append(dict(name=n['name'],type=n['type'],family=known[n['type']],
            inputs=n.get('input',[]),attributes=_attributes(n),
            input_desc=[_metadata_desc(d,enums) for d in n.get('input_desc',[])],
            output_desc=[_metadata_desc(d,enums) for d in n.get('output_desc',[])]))
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
        format=kind,node_count=len(nodes),reachable_count=len(reachable),
        reachable_type_counts=dict(sorted(counts.items())),operators=selected,
        unclassified_types=sorted(set(counts)-set(known)),
        status='GRAPH_OBSERVED',source_revision_binding='UNVERIFIED',
        scope='Source AIR metadata only; empty shapes are unknown, not scalars; no weights read or implementation/dtype guarantees.')


def ddk_inventory(root):
    """Find selected-DDK declarations in bounded text roots, without imports."""
    root=Path(root).resolve()
    if not root.is_dir(): raise ValueError('DDK root does not exist')
    roots=('ddk/tbe/ops','ddk/ascendcpp','tools/platform/kirinx90','tools/tools_ascendc')
    terms={family:re.compile('|'.join(re.escape(re.sub('[^a-z0-9]','',name.lower()))
        for name in (call,*names))) for family,(call,names) in FAMILIES.items()}
    matches=[]; files=0; consumed=0; limited=False; errors=[]; missing=[]
    for relative in roots:
        folder=root/relative
        if not folder.is_dir(): missing.append(relative);continue
        for directory,dirs,names in os.walk(folder,followlinks=False):
            dirs[:]=sorted(d for d in dirs if not (Path(directory)/d).is_symlink())
            for name in sorted(names):
                path=Path(directory)/name
                files+=1
                if files>50000 or consumed>=32<<20:
                    limited=True;break
                if path.is_symlink() or path.suffix.lower() not in {'.json','.h','.hpp','.py','.cpp','.cc'}:continue
                # Read registrations plus operator-named source files. Generic
                # unrelated kernel bodies do not consume the text budget.
                norm=re.sub('[^a-z0-9]','',name.lower())
                if path.suffix!='.json' and not any(p.search(norm) for p in terms.values()):continue
                try:
                    size=path.stat().st_size
                    if size>4<<20 or consumed+size>32<<20:
                        limited=True;continue
                    raw=path.read_bytes();consumed+=len(raw)
                    text=raw.decode('utf-8',errors='replace')
                    normalized=re.sub('[^a-z0-9]','',text.lower())
                    families=[f for f,p in terms.items() if p.search(normalized)]
                    if families:
                        lines=text.splitlines()
                        evidence=[]
                        for i,line in enumerate(lines):
                            cleaned=re.sub('[^a-z0-9]','',line.lower())
                            if any(terms[f].search(cleaned) for f in families):
                                evidence.append(dict(line=i+1,text='\n'.join(lines[max(0,i-2):i+7])[:2400]))
                                if len(evidence)==12:break
                        matches.append(dict(path=str(path.relative_to(root)),families=families,
                            bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),excerpts=evidence))
                        if len(matches)>=160:limited=True;break
                except OSError as error:
                    errors.append(dict(path=str(path),error=str(error)))
            if files>50000 or consumed>=32<<20 or len(matches)>=160:break
        if files>50000 or consumed>=32<<20 or len(matches)>=160:break
    status = ('UNAVAILABLE' if len(missing) == len(roots) else
              'PARTIAL' if limited or errors or missing else 'INSPECTED')
    return dict(root=str(root),status=status,
        files_visited=files,text_bytes=consumed,limited=limited,missing_roots=missing,
        matches=matches,errors=errors,
        scope='Text registrations/source candidates only; absence does not establish unsupported operators; presence does not establish numerical correctness.')


def inspect(source_root, config, *, ddk_root=None, airs=None, torchair_pb2=None):
    source=source_inventory(source_root)
    matrix=case_matrix(config)
    report=dict(schema_version=1,artifact_kind=KIND,status='INVENTORIED',source=source,
        operator_families=[dict(id=f,source_call=call,ge_search_candidates=list(names),
            source_call_count=len(source['families'][f]),contract=SOURCE_CONTRACT[f],
            phase='WEIGHT_PREPARATION' if f=='quant_param' else 'INFERENCE',
            numerical='NOT_RUN') for f,(call,names) in FAMILIES.items()],
        cases=matrix,airs={},ddk=None,model_compilation='NOT_RUN',device_execution='NOT_RUN',
        numerical_validation='NOT_RUN',all_operators_passed=False,
        next='Bind actual AIR/installed operator contracts and capture original tensors; then export isolated cases with independent references.')
    if source['status']!='SOURCE_CALLS_ACCOUNTED':report['status']='SOURCE_REVIEW_REQUIRED'
    if ddk_root:
        report['ddk']=ddk_inventory(ddk_root)
        if report['ddk']['status']!='INSPECTED' and report['status']=='INVENTORIED':
            report['status']='PARTIAL'
    if airs:
        from qwen35_moe_dflash.toolchain.air_schema import load_torchair_schema
        from qwen35_moe_dflash.toolchain.omc_wire import load_schema
        if not torchair_pb2 and not ddk_root:raise ValueError('AIR inspection needs --torchair-pb2 or --ddk-root')
        schema=(load_torchair_schema(torchair_pb2) if torchair_pb2 else
            load_schema(Path(ddk_root)/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so'))
        for label,path in airs.items():
            try:report['airs'][label]=inspect_air(path,schema)
            except (OSError,ValueError,KeyError,TypeError) as error:
                report['airs'][label]=dict(status='UNAVAILABLE',error=f'{type(error).__name__}: {error}')
                report['status']='PARTIAL'
    return report

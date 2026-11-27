"""Compile cumulative L0 readouts, replay the real prompt and compare three backends."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from decoder_cases_board import output_order, read
from decoder_gate_probe import air_bundle, compile_cases, execute_case
from decoder_handoff import ssh_reader
from gdn_matrices_txt import BACKENDS, render
from qwen35_moe_dflash.common.accuracy import floating_metrics
from qwen35_moe_dflash.target.layer_probe.common import identity, read_tensor, save_json, sized_file
from qwen35_moe_dflash.target.prefill_t64.gdn_layer_stages import DESCRIPTION, SHAPES, STATE, canonical

EXPERT = {'expert_gate_up', 'expert_product', 'expert_down'}
EXPLAIN = dict(block_entry='L0 输入，送入输入 RMSNorm 和第一条残差',
               residual='L0 输入加 GDN out_proj', post_norm='第一条残差经过 post_attention_layernorm',
               shared_gate_proj='共享专家 gate_proj', shared_up='共享专家 up_proj', shared_silu='共享专家门控 SiLU',
               shared_product='共享专家 SiLU(gate) × up', shared_down='共享专家 down_proj',
               shared_gate_logit='共享专家 sigmoid 前的单个门控分数', shared_weighted='乘 sigmoid 后的共享专家贡献',
               router_logits='路由器对全部专家的打分', expert_ids='逐 token 的已选专家集合（通过 0/1 指示向量比较）',
               route_weights='实际送入路由汇总的权重，按 token 和 expert ID 展开，未选专家为零',
               expert_gate_up='相同 token 和 expert 的第一跳 W4 gate/up 输出',
               expert_product='相同 token 和 expert 的 SiLU(gate) × up',
               expert_down='相同 token 和 expert 的第二跳 W4 down 输出',
               routed='按实际路由权重汇总的专家输出', moe_out='共享和路由专家合并后的输出',
               block_out='第二条残差相加，送入 L1 的 hidden')
WIDTHS = dict(block_entry=2048, residual=2048, post_norm=2048, shared_gate_proj=512, shared_up=512,
              shared_silu=512, shared_product=512, shared_down=2048, shared_gate_logit=1,
              shared_weighted=2048, router_logits=256, routed=2048, moe_out=2048, block_out=2048)


for projection, width in [('shared_gate_proj',2048), ('shared_up',2048), ('shared_down',512), ('router_logits',2048)]:
    for suffix, size, description in [('_a8',width,'DynamicQuant 激活编码'), ('_scale',1,'DynamicQuant 逐 token 尺度')]:
        name = projection + suffix
        WIDTHS[name] = size
        EXPLAIN[name] = EXPLAIN[projection] + ' 的 ' + description


def reference_matches(refs, directory, fixture):
    ids = fixture['inputs']['input_ids']
    return (int(refs['valid']) == fixture['valid_rows']
            and str(refs['fixture_hidden']) == fixture['inputs']['hidden']['sha256']
            and str(refs['fixture_state']) == fixture['outputs']['recurrent_state_0']['sha256']
            and np.array_equal(refs['ids'], read_tensor(sized_file(directory, ids), ids)))


def get_reference(root, remote, folder, filename='values.npz'):
    rel = Path('l0-steps-reference')/filename
    local_root = os.getenv('DECODER_GOLDEN_LOCAL_ROOT')
    mounted = Path(local_root).expanduser()/root.name/rel if local_root else root/rel
    if mounted.is_file():
        return mounted
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from pull_artifacts import DEFAULT_CONFIG
    role = 'golden' if 'golden' in read(DEFAULT_CONFIG) else 'air'
    configured = os.getenv('DECODER_' + role.upper() + '_REMOTE_ROOT')
    source = (Path(configured)/root.name if configured else remote)/rel
    print(f'接收 L0 CPU/310P 参考文件 {source}', flush=True)
    dest = folder/filename
    with tempfile.TemporaryDirectory(dir=folder) as tmp, ssh_reader(role) as fetch:
        staged = Path(tmp)/dest.name
        fetch(source, staged)
        staged.replace(dest)
    return dest


def compiled_output_slots(base, row):
    build = read(base/'omc/build.json')
    receipt = read(base/'replay/result.json')
    if receipt['status'] != 'EXECUTED' or receipt['omc_sha256'] != build['model']['sha256']:
        raise ValueError(f'{base}: output receipt does not match compiled model')
    outputs = row['outputs']
    by_edge = {w['edge']: w for w in outputs}
    assigned = set()
    for i, (spec, item, public) in enumerate(zip(build['outputs'], receipt['outputs'], outputs, strict=True)):
        wire = by_edge.get(spec['name'], public)
        if wire['edge'] in assigned:
            raise ValueError(f'{base}: ambiguous compiled output {i} {spec["name"]}')
        assigned.add(wire['edge'])
        yield i, wire, spec, sized_file(base/'replay', item)


def compiled_readouts(base, row):
    result = {}
    for _, wire, spec, path in compiled_output_slots(base, row):
        arr = read_tensor(path, spec)
        for role in wire['aliases']:
            result[role] = arr
    return result


def route_ids(values):
    ids = np.asarray(values['expert_ids'])
    if ids.dtype.kind not in 'iu' or ids.size % 64:
        raise ValueError(f'Invalid expert IDs {ids.shape}, {ids.dtype}')
    if tuple(d for d in ids.shape if d != 1) != (64, ids.size//64):
        raise ValueError(f'Expected [token, top_k] router layout, got {ids.shape}')
    ids = ids.reshape(64,-1)
    if ids.min() < 0 or ids.max() >= 256 or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError('Expert IDs are out of range or repeated for one token')
    return ids


def grouped_slots(ids, raw):
    """Resolve index direction/order against the invariant that physical GMM rows are expert-sorted."""
    order = np.asarray(raw).reshape(-1)
    if order.dtype.kind not in 'iu' or not np.array_equal(np.sort(order), np.arange(ids.size)):
        raise ValueError('MoE row map is not a permutation')
    logical = np.arange(ids.size).reshape(ids.shape)
    candidates = []
    for layout in (logical.ravel(), logical.T.ravel()):
        for mapping in (order, np.argsort(order)):
            physical = layout[mapping]
            sorted_ids = ids.ravel()[physical]
            if np.all(sorted_ids[1:] >= sorted_ids[:-1]):
                slots = np.argsort(physical).reshape(ids.shape)
                if not any(np.array_equal(slots, prev) for prev in candidates):
                    candidates.append(slots)
    if len(candidates) != 1:
        raise ValueError(f'{len(candidates)} valid expert row maps; cannot align GMM outputs')
    return candidates[0]


def aligned(role, values, valid):
    xs = [v[role] for v in values]
    if role.endswith('_a8') and any(x.dtype != np.int8 for x in xs):
        raise ValueError(f'{role}: expected INT8 codes on all three backends')
    if role in ('expert_ids', 'route_weights') or role in EXPERT:
        ids = [route_ids(v) for v in values]
        if role not in EXPERT:
            dense = []
            for keys, x in zip(ids, xs):
                matrix = np.zeros((valid, 256), np.float32)
                matrix[np.arange(valid)[:,None], keys[:valid]] = (1 if role == 'expert_ids' else x.reshape(keys.shape)[:valid])
                dense.append(matrix)
            return dense, f'valid{valid}'
        common = [(t,int(e)) for t in range(valid)
                  for e in sorted(set(ids[0][t]) & set(ids[1][t]) & set(ids[2][t]))]
        if not common:
            raise ValueError('No token/expert pair shared by all three backends')
        width = {'expert_gate_up': 1024, 'expert_product': 512, 'expert_down': 2048}[role]
        result = []
        for backend, v, keys, x in zip(BACKENDS, values, ids, xs):
            if x.shape[-1] != width or x.size != keys.size * width:
                raise ValueError(f'{backend}/{role}: unexpected expert shape {x.shape}')
            x = x.reshape(-1,width)
            slots = (np.arange(keys.size).reshape(keys.shape) if backend == 'cpu' else grouped_slots(keys,v['route_map']))
            result.append(np.stack([x[slots[t,np.flatnonzero(keys[t] == e)[0]]] for t,e in common]))
        return result, f'valid{valid}_common_{len(common)}_of_{valid*ids[0].shape[1]}'
    if role in SHAPES:
        xs = [canonical(role, x) for x in xs]
    else:
        width = WIDTHS[role]
        for x in xs:
            if tuple(n for n in x.shape if n != 1) != ((64,width) if width != 1 else (64,)):
                raise ValueError(f'{role}: unexpected logical axes {x.shape}')
    if role in STATE:
        return xs, 'full_state'
    if role == 'window':
        return [x.reshape(8192,67)[:,3:3+valid].T for x in xs], f'valid{valid}'
    if role in ('conv', 'conv_silu'):
        return [x.reshape(8192,64)[:,:valid].T for x in xs], f'valid{valid}'
    return [x.reshape(64,-1)[:valid] for x in xs], f'valid{valid}'


def scores(a, b, role):
    integer = a.dtype.kind in 'iu' and b.dtype.kind in 'iu'
    result = floating_metrics(a.astype(np.float64) if integer else a, b.astype(np.float64) if integer else b)
    if integer:
        result['error_ratio'] = np.count_nonzero(a != b)/a.size
    if role == 'expert_ids':
        union = np.count_nonzero(a.astype(bool) | b.astype(bool))
        result['expert_set_jaccard'] = np.count_nonzero(a.astype(bool) & b.astype(bool))/union
    return {m: result[m] for m in ('cosine','relative_l2','error_ratio','nonfinite_actual','nonfinite_golden','expert_set_jaccard')
            if m in result}


def original_caches(root, meta, fixture, values):
    case = root/'board/L00-L03'
    build = read(case/'omc/build.json')
    replay = case/'replay'
    receipt = read(replay/'result.json')
    if (receipt.get('status') != 'EXECUTED' or receipt.get('omc_sha256') != build['model']['sha256']
            or receipt.get('fixture_sha256') != build['stamp']['bundle']):
        raise ValueError('Original four-layer board replay is unavailable for this OMC')
    names = output_order(build['outputs'], meta['outputs'], fixture['outputs'])
    result = {}
    for name, spec, rec in zip(names, build['outputs'], receipt['outputs'], strict=True):
        role = {'conv_state_0':'new_history','recurrent_state_0':'recurrent_cache'}.get(name)
        if role and role in values:
            whole = read_tensor(sized_file(replay,rec), spec).reshape(SHAPES[role])
            sample = canonical(role, values[role])
            result[role] = dict(exact=bool(np.array_equal(whole,sample)), **scores(sample,whole,role))
    return result


def same_entry_feeds(root, case, folder, input_files, report):
    """Bind the captured FP16 rows at the original embedding product's declared dtype."""
    fixture = root/'board/package/fixture'
    spec = read(fixture/'reference.json')['inputs']['hidden']
    hidden_path = sized_file(fixture, spec)
    if spec['dtype'] != 'float16' or list(spec['shape']) != [1, 64, 2048]:
        raise ValueError(f'Unexpected 310P L0 entry: {spec}')
    records = {role: dict(row, path=str((case/row['path']).relative_to(root)))
               for role, row in input_files.items() if role not in ('inputs_embeds', 'embed_scale')}
    dtypes = {feed['dtype'] for row in report['cases'].values() if row['status'] == 'COMPILED'
              for feed in row['feeds'].values() if feed['name'] == 'l0_entry'}
    if not dtypes:
        return root, records  # Preserve individual cut/compile failures in the report.
    if dtypes == {'float16'}:
        entry = dict(spec, path=str(hidden_path.relative_to(root)))
    elif dtypes == {'float32'}:
        path = folder/'l0_entry_fp32.bin'
        read_tensor(hidden_path, spec).astype('<f4').tofile(path)
        entry = dict(identity(path, root), dtype='float32', shape=spec['shape'])
    else:
        raise ValueError(f'Inconsistent L0 entry dtypes in the compiled slices: {dtypes}')
    records['l0_entry'] = entry
    return root, records


def compare(root, folder, bundle, ref_path, meta, report, observed, *, same_input=False):
    fixture = read(bundle/'fixture/reference.json')
    missing = {}
    with np.load(ref_path, allow_pickle=False) as refs:
        valid = int(refs['valid'])
        if not reference_matches(refs, bundle/'fixture', fixture):
            raise ValueError('Reference captures belong to a different four-layer fixture')
        cpu, nz = [{key[len(b)+1:]:refs[key] for key in refs.files if key.startswith(b+'_')} for b in BACKENDS[:2]]
    matrices = {}
    cpro = {}
    first = {}
    csv_path = folder/'matrices.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('stage','tensor','source_variable','scope','metric','actual/reference',*BACKENDS))
        for label, row in report['cases'].items():
            if observed[label]['status'] != 'EXECUTED':
                missing[label] = observed[label]['status']
                continue
            try:
                batch = compiled_readouts(folder/label, row)
            except (OSError, ValueError, KeyError) as exc:
                missing[label] = str(exc)
                continue
            for role in cpro.keys() & batch.keys():
                if not np.array_equal(cpro[role], batch[role]):
                    missing[label+'/'+role] = 'Two CPro prefix OMCs returned different values at the same readout'
            cpro.update(batch)
            for role in batch:
                if role == 'route_map' or role in matrices:
                    continue
                try:
                    xs, scope = aligned(role, [cpu,nz,batch], valid)
                except (KeyError, ValueError) as exc:
                    missing[role] = str(exc)
                    continue
                explanation = EXPLAIN.get(role) or DESCRIPTION[role]
                views = [(role, scope, xs)]
                if role in ('block_entry','qkv','out','residual','post_norm','moe_out','block_out'):
                    tail = [v[role].reshape(64,-1)[-4:] for v in (cpu,nz,batch)]
                    views.append((role+'@60-63', 'physical60_63', tail))
                for key, scope, tensors in views:
                    table = {b:{r:scores(x,y,role) for r,y in zip(BACKENDS,tensors)}
                             for b,x in zip(BACKENDS,tensors)}
                    matrices[key] = dict(scope=scope, stage=label, scores=table, description=explanation)
                    for metric in ('cosine','relative_l2','error_ratio'):
                        for b in BACKENDS:
                            writer.writerow((label,role,explanation,scope,metric,b,*(table[b][r][metric] for r in BACKENDS)))
                    pair = table['cpro']['310p']
                    print(f'{role:18} {scope:28} CPU/310P cos={table["cpu"]["310p"]["cosine"]}  '
                          f'CPro/310P cos={pair["cosine"]}  rel_L2={pair["relative_l2"]}', flush=True)
                    if role == 'expert_ids':
                        print(f'  CPro/310P 路由专家集合 Jaccard={pair["expert_set_jaccard"]:.6f}', flush=True)
                    if scope.startswith('valid') and (pair['cosine'] is None or pair['relative_l2'] is None or
                            pair['cosine'] <= 0.99 or pair['relative_l2'] >= 0.05):
                        first.setdefault(label, role)
    if matrices:
        render(csv_path)
    if same_input:
        anchors = {'not_applicable': 'The original four-layer replay used the quantized embedding.'}
        shape = (64, 2048)
        entry = cpro.get('block_entry')
        exact = (entry is not None and entry.size == np.prod(shape)
                 and all(np.array_equal(entry.reshape(shape), v['block_entry'].reshape(shape)) for v in (cpu, nz)))
        if not exact:
            missing['block_entry_control'] = 'CPro L0 entry missing or differs from the common CPU/310P input'
        print(f'三端 L0 入口全部 64 行逐元素相同：{exact}', flush=True)
    else:
        try:
            anchors = original_caches(root, meta, fixture, cpro)
        except (OSError, ValueError, KeyError) as err:
            anchors = {'unavailable':str(err)}
    status = 'INCOMPLETE' if missing else 'COMPARED'
    save_json(folder/'comparison.json', dict(status=status, valid_rows=valid, source_air=report['air'],
              input_basis='310p_block_entry' if same_input else 'cpro_embedding',
              **({'block_entry_exact': exact} if same_input else {}),
              missing=missing, four_layer_cache_control=anchors, first_cos_or_l2_crossing_by_group=first, tensors=matrices))
    print(f'相对四层原 OMC 的缓存对照：{anchors}\n分组首个超过精度门限的观测点：{first}\n'
          f'{status} {folder}/comparison.json', flush=True)
    if missing:
        print('缺少可比较数据：' + json.dumps(missing,ensure_ascii=False), flush=True)
    return not missing


def run(root, remote, action, ddk, config, args):
    parent = root/'board/L00-L03'
    bundle = root/'board/package'
    case, meta, record = air_bundle(bundle/'L00-L03', root/'export/package/L00-L03' if action=='compile' else None)
    mode = args.gate_probe
    same_input = mode == 'l0-steps-same-input'
    folder = parent/mode
    if action == 'compile':
        compile_cases(folder, case, meta, record, ddk, mode)
        return True   # Replay successful groups even when another group failed to build.
    report = read(folder/'compile.json')
    if report['air'] != record or not report['complete']:
        raise ValueError('L0 readouts must be compiled from the current four-layer AIR')
    for filename in ('comparison.json','matrices.csv','cosine.txt','error_ratio.txt','result.json'):
        (folder/filename).unlink(missing_ok=True)
    old_reference = parent/'l0-steps/values.npz'
    ref_path = None
    if same_input and old_reference.is_file():
        fixture = read(bundle/'fixture/reference.json')
        with np.load(old_reference, allow_pickle=False) as refs:
            if reference_matches(refs, bundle/'fixture', fixture):
                ref_path = old_reference
    if ref_path is None:
        ref_path = get_reference(root, remote, folder)
    runtime = read(config.parent/read(config)['assets'])['environment']
    feeds = same_entry_feeds(root, case, folder, meta['input_files'], report) if same_input else (case, meta['input_files'])
    observed = {}
    for label, row in report['cases'].items():
        if row['status'] == 'COMPILED':
            try:
                observed[label] = execute_case(folder/label, row, runtime, config, real_inputs=feeds,
                      replace_inputs=args.replace_inputs, replace_weights=args.replace_weights,
                      board_case=f'4-prefill-t64-{mode}-{label}')
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                if 'BOARD_MODEL_IO_BUSY' in str(exc):
                    raise
                observed[label] = dict(status='REPLAY_ERROR', error=str(exc))
        else:
            observed[label] = dict(status=row['status'], error=row.get('error'))
        save_json(folder/'result.json', observed)
    return compare(root, folder, bundle, ref_path, meta, report, observed, same_input=same_input)

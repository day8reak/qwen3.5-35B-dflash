"""Inspect an existing complete L0 dump on the board host, starting at input RMSNorm."""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))
from qwen35_moe_dflash.target.layer_probe.common import metrics, read_tensor, save_json, sized_file
from qwen35_moe_dflash.target.prefill_t64.decoder_cases import cpro_input_records


def token_rows(array):
    if array.shape not in ((1, 64, 2048), (64, 2048)):
        raise ValueError(f'Unexpected L0 token tensor shape {array.shape}')
    return array.reshape(64, 2048)


def compare(actual, reference, valid):
    return metrics(token_rows(actual)[:valid], token_rows(reference)[:valid])


def inspect_entry_dumps(replay, files, nodes, feed, refs, valid):
    """Compare plausible byte views of exact AIR node names without assigning dump slots."""
    # The numbers surrounding the name are opaque; extract the shape and type
    # only from the observed filename convention. Extra singleton axes are common.
    pattern = re.compile(r'^_\d+_\d+_(.+?)_(float|float16|float32|half|fp16|fp32)_'
                         r'(\d+(?:_\d+)*)_(\d+)_(\d+)\.dump$', re.I)
    targets = {'feed': feed, 'cpu_hidden': refs['cpu_hidden'], '310p_hidden': refs['310p_hidden']}
    groups = {}
    skipped = {}
    named_files = []
    for name, size in sorted(files.items()):
        match = pattern.fullmatch(Path(name).name)
        if not match or match[1] not in nodes:
            continue
        named_files.append(name)
        dims = [int(dim) for dim in match[3].split('_')]
        if [dim for dim in dims if dim != 1] != [64, 2048]:
            skipped[name] = f'filename dimensions {dims}'
            continue
        if size not in (feed.size * 2, feed.size * 4, feed.size * 2 + 32, feed.size * 4 + 32):
            skipped[name] = f'file size {size}'
            continue
        path = sized_file(replay / 'dump', dict(path=name, bytes=size))
        raw = path.read_bytes()
        groups.setdefault((match[1], match[2], raw), []).append(name)
    print(f'精确节点 {sorted(nodes)}，匹配文件 {len(named_files)} 个，符合维度和长度的独立内容组 {len(groups)} 个')
    if not groups:
        print('上述节点没有可按完整 [64,2048] 检查的文件，输入输出端口仍待定位。')
    for name, why in skipped.items():
        print(f'  跳过 {name} ({why})')
    report = []
    for (node, tag, raw), names in groups.items():
        print(f'  {node}  文件名类型={tag}  字节={len(raw)}  相同内容={names}')
        payload = feed.size * (4 if len(raw) >= feed.size * 4 else 2)
        extra = len(raw) - payload
        views = []
        for offset in ([0, extra] if extra else [0]):
            # A previous shared-gate output had a float-sized buffer containing
            # packed halves at the start. Compare that hypothesis independently.
            dtypes = ('<f4', '<f2') if payload == feed.size * 4 else ('<f2',)
            for dtype in dtypes:
                array = np.frombuffer(raw, dtype=dtype, count=feed.size, offset=offset).reshape(64, 2048)
                scores = {}
                for key, target in targets.items():
                    result = compare(array, target, valid)
                    scores[key] = {k: result[k] for k in ('cosine', 'relative_l2', 'nonfinite_actual',
                                                           'exact_elements', 'total_elements')}
                    scores[key]['exact_all64'] = bool(np.array_equal(array, token_rows(target)))
                views.append(dict(dtype=array.dtype.name, offset=offset, bytes_used=array.nbytes,
                                  unused_bytes=len(raw) - array.nbytes, comparisons=scores))
                print(f'    {array.dtype.name}@{offset}: feed全64行相等={scores["feed"]["exact_all64"]}, '
                      f'有效行与feed相等元素={scores["feed"]["exact_elements"]}/{scores["feed"]["total_elements"]}, '
                      f'有效{valid}行 cos/rel_L2 ' + '  '.join(
                          f'{key}={s["cosine"]}/{s["relative_l2"]}' for key, s in scores.items()))
        report.append(dict(node=node, filename_type=tag, files=names, bytes=len(raw), views=views))
    return dict(matched_files=named_files, skipped=skipped, byte_views=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path, help='1-linear-prefill-t64 directory on the board host')
    parser.add_argument('--rms-io', nargs=2, metavar=('X', 'Z'),
                        help='confirmed dense FP16 ND input and output files of the first RMSNorm, relative to replay/dump')
    args = parser.parse_args()
    board = args.case / 'board'
    replay = board / 'L00/replay'
    fixture = board / 'package/fixture'
    inventory = json.loads((replay / 'dump-inventory.json').read_text())
    meta = json.loads((board / 'package/L00/case.json').read_text())
    ref = json.loads((fixture / 'reference.json').read_text())
    receipt = json.loads((replay / 'result.json').read_text())
    if (inventory['attempt'] != receipt['attempt'] or receipt['status'] != 'EXECUTED'
            or inventory['model_exitcode'] != 0 or inventory['parser_exitcode'] != 0):
        raise ValueError('Dump inventory does not match a successfully executed and parsed replay')
    files = inventory['files']
    norms = meta['rmsnorm_nodes']
    hidden_edges = {name + ':0' for name, role in meta['inputs'].items() if role == 'hidden'}
    entry_norms = [row['node'] for row in norms if row['inputs'][0] in hidden_edges]
    print(f'同次执行回收 {len(files)} 个文件，AIR hidden={sorted(hidden_edges)}')
    print(f'直接从 hidden 取 x 的 RMSNorm 节点：{entry_norms}')
    for row in norms:
        print(f'  {row["node"]}: {row["inputs"]} → {row["output"]}  {row["air_dtypes"]}')

    # The parser may split a tensor across dumps or emit fused node names.
    # Keep filenames and sizes verbatim; numeric filename fields have no verified slot semantics.
    candidates = {name: size for name, size in files.items()
                  if re.search(r'rms.?norm|dynamic.?quant', Path(name).name, re.I)}
    for label, pattern in [('RMSNorm', r'rms.?norm'), ('DynamicQuant', r'dynamic.?quant')]:
        names = sorted(n for n in candidates if re.search(pattern, Path(n).name, re.I))
        print(f'文件名含 {label} 的候选共 {len(names)} 个，示例：')
        for name in names[:6]:
            print(f'  {candidates[name]:>10}  {name}')
    row = cpro_input_records(ref)['hidden']
    feed = read_tensor(sized_file(fixture, row), row)
    results = {}
    with np.load(fixture / 'l0-steps-reference/values.npz', allow_pickle=False) as refs:
        valid = int(refs['valid'].item())
        if (valid != ref['valid_rows'] or not 0 < valid <= 64
                or str(refs['fixture_hidden'].item()) != ref['inputs']['hidden']['sha256']
                or str(refs['fixture_state'].item()) != ref['outputs']['recurrent_state_0']['sha256']):
            raise ValueError('Intermediate archive does not belong to this reference fixture')
        for backend in ('cpu', '310p'):
            results[f'host_feed_vs_{backend}_entry'] = compare(feed, refs[backend + '_block_entry'], valid)
        entry_nodes = {edge.rsplit(':', 1)[0] for edge in hidden_edges} | set(entry_norms)
        first_norm = inspect_entry_dumps(replay, files, entry_nodes, feed, refs, valid)
        if args.rms_io:
            if args.rms_io[0] == args.rms_io[1]:
                parser.error('Provide separate confirmed x and z dump paths')
            def dense(name):
                if name not in files:
                    raise ValueError(f'Dump is absent from this inventory: {name}')
                spec = dict(path=name, bytes=files[name], dtype='float16', shape=[1, 64, 2048])
                return read_tensor(sized_file(replay / 'dump', spec), spec)
            x, z = map(dense, args.rms_io)
            results['dump_x_vs_host_feed'] = compare(x, feed, valid)
            for backend in ('cpu', '310p'):
                results[f'dump_z_vs_{backend}_hidden'] = compare(z, refs[backend + '_hidden'], valid)
    for key, result in results.items():
        print(f'{key} valid{valid}  cos={result["cosine"]}  rel_L2={result["relative_l2"]}')
    dest = replay / 'l0-input-rmsnorm-dump.json'
    save_json(dest, dict(attempt=inventory['attempt'], valid_rows=valid, dense_fp16_x_z=args.rms_io,
                         input_norms=entry_norms, air_norms=norms, candidate_files=candidates,
                         entry_dump_candidates=first_norm, comparisons=results))
    print(f'完整候选清单和对比：{dest}')


if __name__ == '__main__':
    main()

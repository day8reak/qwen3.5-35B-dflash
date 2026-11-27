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
                         comparisons=results))
    print(f'完整候选清单和对比：{dest}')


if __name__ == '__main__':
    main()

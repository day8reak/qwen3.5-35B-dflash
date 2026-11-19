"""Compare complete Prefill outputs before selecting a case-scoped golden."""
import torch

from ..layer_probe.common import identity, metrics, read_array
from ..layer_probe.full_attention_compare import cache_partitions


def output_records(out, folder, values, specs):
    if not isinstance(values, (list, tuple)) or len(values) != len(specs):
        count = len(values) if isinstance(values, (list, tuple)) else type(values).__name__
        raise ValueError(f'Wrapper returned {count}; expected {len(specs)} ordered output tensors')
    (out / folder).mkdir(parents=True)
    records = []
    for tensor, spec in zip(values, specs):
        name = spec['semantic_name']
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f'Expected a tensor at output {name}')
        arr = tensor.detach().cpu().contiguous().numpy()
        if list(arr.shape) != spec['shape'] or arr.dtype.name != spec['dtype']:
            raise ValueError(f'Unexpected shape or dtype in {name}: {arr.shape}, {arr.dtype}')
        path = out / folder / (name + '.bin')
        arr.tofile(path)
        records.append(dict(identity(path, out), name=name, shape=list(arr.shape), dtype=arr.dtype.name))
    return records


def compare_outputs(cpu_root, cpu, other_root, other, cache_before, table):
    names = [r['name'] for r in cpu]
    compared_names = [s.get('semantic_name', s['name']) for s in other]
    if names != compared_names or len(names) != 9 or names[0] != 'logits':
        raise ValueError(f'Prefill output names or order differ: {names} vs {compared_names}')
    rows = {}
    for a, b in zip(cpu, other):
        name = a['name']
        expected, actual = read_array(cpu_root, a), read_array(other_root, b)
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError(f'CPU and 310P descriptors differ for {name}')
        row = rows[name] = metrics(actual, expected)
        if name == 'logits':
            row['cpu_top1'] = int(expected[0, 0].argmax())
            row['device_top1'] = int(actual[0, 0].argmax())
            row['top1_equal'] = row['cpu_top1'] == row['device_top1']
            row['passed'] = row['passed'] and row['top1_equal']
        if name in ('key_cache_3', 'value_cache_3'):
            before = cache_before[name]
            partition = cache_partitions(actual, expected, before,
                dict(block_table=table, sequence_length=64, block_size=before.shape[2]))
            row['active_relative_l2'] = partition['active_metrics']['relative_l2']
            row['active_passed'] = partition['active_metrics']['passed']
            row['untouched_equal_bits'] = partition['untouched_equal_input_bits']
            row['passed'] = row['passed'] and partition['passed']
    failed = [name for name, row in rows.items() if not row['passed']]
    return dict(status='FAILED' if failed else 'PASSED', failed_outputs=failed, outputs=rows)


def select_golden(report, out):
    native = report['comparison']['native_310p']
    repeat = report['comparison'].get('native_repeat')
    if (report['native_core_audit'] != 'PASSED' or native['status'] != 'PASSED'
            or report['native_repeat']['status'] != 'PASSED' or not repeat or repeat['status'] != 'PASSED'
            or any(len(run['decoder_hidden']) != 4 or not all(row['passed'] for row in run['decoder_hidden'])
                   for run in (native, repeat))):
        return None
    return dict(basis='ORIGINAL_NATIVE_310P', root=str(out), outputs=report['native_outputs'])

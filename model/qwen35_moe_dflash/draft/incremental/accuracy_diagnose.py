"""Read retained proposal tensors to distinguish zero/repeated hidden outputs.

No model execution, graph edits or acceptance changes. Source embedding rows,
when requested, describe prepared bytes, not the board's Gather output.
"""
import math
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.execution.runtime_config import read_json
from .accuracy import analyze_result, metrics, read_snapshot


def hidden_stats(value):
    rows = np.ascontiguousarray(value).reshape(-1, value.shape[-1])
    finite = bool(np.isfinite(rows).all())
    bits = [row.tobytes() for row in rows]
    norms = np.linalg.norm(rows.astype(np.float64), axis=1) if finite else None
    return dict(elements=int(rows.size), nonfinite_elements=int((~np.isfinite(rows)).sum()),
        zero_elements=int((rows == 0).sum()), zero_rows=int(np.all(rows == 0, axis=1).sum()),
        rows=len(rows), unique_rows_bitwise=len(set(bits)),
        max_abs=float(np.abs(rows.astype(np.float64)).max()) if finite else None,
        row_l2_min=float(norms.min()) if finite else None,
        row_l2_max=float(norms.max()) if finite else None,
        first_row_first8=[float(v) if np.isfinite(v) else None for v in rows[0, :8]])


def prefix_difference(actual, other):
    count = min(actual.shape[1], other.shape[1])
    a, b = actual[:, :count], other[:, :count]
    return dict(rows=count, bitwise_equal=a.tobytes() == b.tobytes(),
        max_abs_difference=(float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max())
                            if np.isfinite(a).all() and np.isfinite(b).all() else None))


def tensor_difference(left, right):
    """Bounded bitwise difference, including signed zero; no acceptance gate."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('snapshot tensor descriptors differ')
    a, b = np.ascontiguousarray(left), np.ascontiguousarray(right)
    unequal = np.any(a.view(np.uint8).reshape(-1, a.dtype.itemsize)
                     != b.view(np.uint8).reshape(-1, b.dtype.itemsize), axis=1)
    positions = np.flatnonzero(unequal)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(bitwise_equal=not bool(positions.size), different_elements=int(positions.size),
        first_different_flat_index=int(positions[0]) if positions.size else None,
        last_different_flat_index=int(positions[-1]) if positions.size else None,
        max_abs_difference=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0))
            if finite else None)


def initial_call_comparison(fixture, native, meta):
    """Compare the two case-zero snapshots after checking their shared inputs.

    The runner sets every input for isolated-0 and feedback-0 on one model
    instance. Later feedback cases reuse KV and are not same-input repeats.
    Retained files prove the fixture, not what the device actually consumed.
    """
    step = meta['cases'][0]
    if step['index'] != 0 or step['start'] != 0 or step['valid'] != 0 or step['proposals'] < 1:
        return dict(status='NOT_APPLICABLE', reason='first case is not an empty-context proposal')
    records = {r['index']: r for r in meta['input_files'] if r['case'] == 0}
    if len(records) != len(meta['inputs']):
        raise ValueError('initial case input inventory is incomplete')
    zero_kv = []
    for spec in meta['inputs']:
        record = records[spec['index']]
        if record['path'] != f"inputs/0/{spec['index']}.bin" or record['bytes'] != spec['bytes']:
            raise ValueError('initial case input path/extent differs: ' + spec['name'])
        path = checked_file(fixture, record)
        if spec['name'].startswith('past_'):
            zero_kv.append(not bool(np.any(np.fromfile(path, dtype='<f2'))))
    values = {mode: read_snapshot(native/f'{mode}-0.bin', meta['outputs'])
              for mode in ('isolated', 'feedback')}
    count = step['proposals']
    hidden = {mode: tensors[1][:, :count] for mode, tensors in values.items()}
    golden = np.fromfile(fixture/'golden/0/hidden.bin', dtype='<f2').reshape(hidden['isolated'].shape)
    atol, rtol = (meta['comparison'][key] for key in ('atol', 'rtol'))
    rows = []
    for i, (a, b, g) in enumerate(zip(hidden['isolated'].reshape(-1, golden.shape[-1]),
                                     hidden['feedback'].reshape(-1, golden.shape[-1]),
                                     golden.reshape(-1, golden.shape[-1]))):
        rows.append(dict(proposal_flat_row=i, between_modes=tensor_difference(a, b),
            isolated_vs_reference=metrics(a, g, atol, rtol),
            feedback_vs_reference=metrics(b, g, atol, rtol)))
    return dict(status='COMPARED', case=step,
        input_contract='RUNNER_RESETS_ALL_INPUTS_FROM_CASE_ZERO_FILES_IN_BOTH_MODES',
        input_verification='RETAINED_FILES_HASH_AND_EXTENT; NOT_DEVICE_READBACK',
        verified_input_count=len(records), kv_input_count=len(zero_kv),
        all_kv_inputs_zero=all(zero_kv) if zero_kv else None,
        call_ordinals_zero_based={'isolated': 0, 'feedback': len(meta['cases'])},
        comparison_direction='LEFT_ISOLATED_RIGHT_FEEDBACK',
        outputs=[dict(name=spec['name'], **tensor_difference(a, b)) for spec, a, b in
                 zip(meta['outputs'], values['isolated'], values['feedback'])],
        hidden_rows=rows, reference_tolerances=dict(atol=atol, rtol=rtol),
        limitation='OUTPUT_DIFFERENCES_DO_NOT_LOCALIZE_COMPUTE_BUFFER_LIFETIME_OR_VISIBILITY_CAUSE')


def _controls(fixture, meta):
    records = {(r['case'], r['index']): r for r in meta['input_files']}
    if len(records) != len(meta['input_files']):
        raise ValueError('duplicate fixture input records')
    slots = {r['name']: r['index'] for r in meta['inputs']}
    controls = [('start_position', 'start'), ('valid_rows', 'valid'),
                ('anchor', 'anchor'), ('proposal_count', 'proposals')]
    block = next((r for r in meta['inputs'] if r['name'] == 'block_embeddings'), None)
    if block is not None:
        controls.remove(('anchor', 'anchor'))
    for step in meta['cases']:
        for role, key in controls:
            slot = slots[role]
            record = records[(step['index'], slot)]
            if record['path'] != f"inputs/{step['index']}/{slot}.bin" or record['bytes'] != 4:
                raise ValueError('fixture control path/extent differs: ' + role)
            value = int(np.fromfile(checked_file(fixture, record), dtype='<i4')[0])
            if value != step[key]:
                raise ValueError('fixture control differs from schedule: ' + role)
        if block is not None:
            record = records[(step['index'], block['index'])]
            if (record['path'] != f"inputs/{step['index']}/{block['index']}.bin"
                    or record['bytes'] != block['bytes']):
                raise ValueError('fixture embedding path/extent differs')
            checked_file(fixture, record)
    result = dict(cases=len(meta['cases']), slots={n: slots[n] for n, _ in controls},
        status='HASH_AND_VALUE_MATCH', scope='RETAINED_INPUT_FILES_NOT_DEVICE_READBACK')
    if block is not None:
        result['block_embeddings'] = dict(slot=block['index'], status='HASH_AND_EXTENT_MATCH',
            anchor='SCHEDULE_ID; NO_SCALAR_ANCHOR_INPUT_IN_THIS_ABI')
    return result


def _embedding_rows(source, fixture, meta):
    binding = read_json(fixture/'pytorch-abi.json')
    source = Path(source)
    if (sha256_file(source/'prepared.json') != binding['prepared_sha256']
            or binding['packed_model_sha256'] != meta['model']['sha256']):
        raise ValueError('prepared source/model differs from the retained PyTorch ABI binding')
    plan = read_json(source/'prepared.json')
    config = Qwen35DFlashConfig.from_dict(plan['draft_config'])
    name = 'target.embed_tokens.weight'
    record = plan['weights'][name]
    bound = [r for r in binding['reference_weights'] if r['name'] == name]
    shape = [config.vocab_size, config.hidden_size]
    if (len(bound) != 1 or bound[0]['sha256'] != record['sha256']
            or record['dtype'] != 'float16' or record['shape'] != shape
            or record['bytes'] != math.prod(shape)*2):
        raise ValueError('embedding descriptor/hash differs from the bound FP16 source')
    path = checked_file(source, record)
    mask = config.mask_token_id
    ids = sorted({mask, *(s['anchor'] for s in meta['cases'] if s['proposals'])})
    if any(type(i) is not int or not 0 <= i < shape[0] for i in ids):
        raise ValueError('embedding diagnostic token outside vocabulary')
    rows = []
    with path.open('rb') as stream:
        for token in ids:
            stream.seek(token*shape[1]*2)
            value = np.frombuffer(stream.read(shape[1]*2), dtype='<f2').reshape(1, -1)
            if value.size != shape[1]:
                raise ValueError('incomplete embedding row')
            rows.append(dict(token=token, role='mask' if token == mask else 'anchor',
                             **hidden_stats(value)))
    return dict(scope='HASH_BOUND_PREPARED_ROWS_NOT_DEVICE_GATHER_OUTPUT',
                weight_sha256=record['sha256'], rows=rows)


def diagnose_result(path, *, source=None):
    # Recompare both complete sequences with the frozen reference and gates first.
    summary = analyze_result(path)
    path = Path(path)
    path = path/'result.json' if path.is_dir() else path
    result = read_json(path, 4 << 20)
    fixture = Path(result['fixture'])
    if not fixture.is_absolute():
        fixture = path.parent/fixture
    meta = read_json(fixture/'manifest.json')
    if len(meta['cases']) > 64 or meta['cache_capacity'] > 512:
        raise ValueError('diagnosis requires the bounded standalone accuracy fixture')
    report = dict(artifact_kind='qwen35-draft-output-diagnosis',
        source_status=summary['status'], numerical_validation=summary['numerical_validation'],
        scope='HOST_RETAINED_ACTIVE_PROPOSAL_ROWS_NO_EXECUTION_OR_GATE_CHANGE',
        model_execution='NOT_RUN', model_compilation='NOT_RUN',
        board_attempt=result.get('board_attempt'), model_sha256=meta['model']['sha256'],
        controls=_controls(fixture, meta), cases=[])
    report['initial_call_comparison'] = initial_call_comparison(fixture, path.parent/'native', meta)
    for mode in ('isolated', 'feedback'):
        previous = None
        for step in meta['cases']:
            i, count = step['index'], step['proposals']
            if not count:
                continue
            values = read_snapshot(path.parent/'native'/f'{mode}-{i}.bin', meta['outputs'])
            hidden = values[1][:, :count]
            golden = np.fromfile(fixture/'golden'/str(i)/'hidden.bin', dtype='<f2').reshape(hidden.shape)
            row = dict(mode=mode, index=i, label=step['label'], context=step['start']+step['valid'],
                anchor=step['anchor'], proposals=count,
                top1_unique=np.unique(values[0][:, :count]).tolist(),
                hidden=hidden_stats(hidden), reference_hidden=hidden_stats(golden))
            if previous is not None:
                j, old, old_golden = previous
                row['previous_proposal'] = dict(index=j, actual=prefix_difference(hidden, old),
                                                reference=prefix_difference(golden, old_golden))
            if mode == 'feedback':
                isolated = read_snapshot(path.parent/'native'/f'isolated-{i}.bin', meta['outputs'])
                row['same_case_isolated_hidden'] = prefix_difference(hidden, isolated[1][:, :count])
            report['cases'].append(row)
            previous = (i, hidden.copy(), golden)
    if source is not None:
        report['embedding'] = _embedding_rows(source, fixture, meta)
    return report

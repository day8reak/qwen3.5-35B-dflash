"""Offline head ranking for retained Top1 mismatches; no target or Torch import.

CPU FP32 accumulation and FP16 output rounding are diagnostic arithmetic,
not a reconstruction of the compiled MatMul or a new acceptance reference.
"""
import json
import math
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.execution.runtime_config import read_json
from .accuracy import analyze_result, read_snapshot

KIND = 'qwen35-draft-retained-head-audit'


def full_vocab_scores(hidden, weight, progress=lambda x: None, chunk=8192):
    """Bound weight conversion memory; retain only small selected-row logits."""
    if (hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]
            or not 1 <= len(hidden) <= 64 or chunk < 1):
        raise ValueError('head audit requires 1..64 compatible hidden rows')
    if not np.isfinite(hidden).all():
        raise ValueError('nonfinite retained hidden')
    h = hidden.astype(np.float32)
    scores = np.empty((len(h), len(weight)), dtype=np.float32)
    for start in range(0, len(weight), chunk):
        w = weight[start:start+chunk].astype(np.float32)
        if not np.isfinite(w).all():
            raise ValueError('nonfinite prepared head weight')
        scores[:, start:start+len(w)] = h @ w.T
        progress(f'full-vocabulary head: {start+len(w)}/{len(weight)} tokens, {len(h)} hidden rows')
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite CPU FP32 head logits')
    with np.errstate(over='ignore'):
        rounded = scores.astype(np.float16)
    if not np.isfinite(rounded).all():
        raise ValueError('CPU head logits overflow finite FP16 contract')
    return scores, rounded


def ranking(scores, expected, actual):
    first = int(np.argmax(scores))  # Smallest index among equal scores.
    rest = scores.astype(np.float64)
    rest[first] = -np.inf
    second = int(np.argmax(rest))
    return dict(top1=first, top2=second,
        top2_margin=float(scores[first])-float(scores[second]),
        maximum_tie_count=int(np.count_nonzero(scores == scores[first])),
        expected_logit=float(scores[expected]), actual_logit=float(scores[actual]),
        expected_minus_actual=float(scores[expected])-float(scores[actual]))


def audit_head(path, source, output, progress=lambda x: None):
    path, source = Path(path).resolve(), Path(source).resolve()
    path = path/'result.json' if path.is_dir() else path
    destination = output_path(output)
    result = read_json(path, 4 << 20)
    fixture = (path.parent/Path(result['fixture'])).resolve()
    if any(destination == p or p in destination.parents for p in (path.parent, fixture, source)):
        raise ValueError('head audit output must be separate from retained result/fixture/weights')
    if destination.exists() and read_json(destination).get('artifact_kind') != KIND:
        raise ValueError('refusing to overwrite an unrelated head audit output')
    progress('rechecking retained complete accuracy comparison; no model execution')
    original = analyze_result(path)
    meta = read_json(fixture/'manifest.json')
    binding = read_json(fixture/'pytorch-abi.json')
    plan = read_json(source/'prepared.json')
    if (meta.get('frontend') != 'pytorch' or result['model'] != meta['model']
            or binding.get('packed_model_sha256') != meta['model']['sha256']
            or binding.get('prepared_sha256') != sha256_file(source/'prepared.json')
            or plan.get('artifact_kind') != 'qwen35-draft-ge-prepared'
            or plan.get('status') != 'PREPARED' or plan.get('precision') != 'float16'
            or plan.get('quantization') is not None):
        raise ValueError('head audit requires the matching prepared FP16 PyTorch model binding')
    config = Qwen35DFlashConfig.from_dict(plan['draft_config'])
    name = 'target.lm_head.weight'
    record = plan['weights'][name]
    bound = [r for r in binding['reference_weights'] if r['name'] == name]
    shape = [config.vocab_size, config.hidden_size]
    if (len(bound) != 1 or bound[0]['sha256'] != record['sha256']
            or record['shape'] != shape or record['dtype'] != 'float16'
            or record['bytes'] != math.prod(shape)*2
            or shape[1] != meta['outputs'][1]['shape'][-1] or shape[0] < 2):
        raise ValueError('head descriptor/hash differs from retained model binding')
    progress('verifying original FP16 LM head; no other weights are loaded')
    weight_path = checked_file(source, record)
    weight = np.memmap(weight_path, dtype='<f2', mode='r', shape=tuple(shape))
    selected, rows, row_ids, files = [], [], {}, {}

    def hidden_id(value):
        key = value.tobytes()
        if key not in row_ids:
            row_ids[key] = len(rows)
            rows.append(value.copy())
        return row_ids[key]

    # Read actual snapshot IDs again, rather than interpreting rounded JSON values.
    for case in result['cases']:
        if not case.get('top1_mismatches'):
            continue
        mode, index = case['mode'], case['index']
        count = meta['cases'][index]['proposals']
        snapshot = path.parent/'native'/f'{mode}-{index}.bin'
        ids_path, hidden_path = (fixture/'golden'/str(index)/n for n in ('top1.bin', 'hidden.bin'))
        device = read_snapshot(snapshot, meta['outputs'])
        expected = np.fromfile(ids_path, dtype='<i4')
        hidden = np.fromfile(hidden_path, dtype='<f2').reshape(count, shape[1])
        actual = device[0].reshape(-1)[:count]
        if (len(expected) != count or actual.tolist() != case['actual_top1']
                or expected.tolist() != case['expected_top1']
                or any(np.any((a < 0) | (a >= shape[0])) for a in (actual, expected))):
            raise ValueError('retained token values/extent differ or exceed vocabulary')
        for p in np.flatnonzero(actual != expected):
            if len(selected) == 32:
                raise ValueError('bounded head audit supports at most 32 mismatched positions')
            selected.append(dict(mode=mode, index=index, label=case['label'], position=int(p),
                actual=int(actual[p]), expected=int(expected[p]),
                device_hidden_row=hidden_id(device[1].reshape(-1, shape[1])[p]),
                reference_hidden_row=hidden_id(hidden[p])))
        for f in (snapshot, ids_path, hidden_path):
            files[str(f)] = dict(bytes=f.stat().st_size, sha256=sha256_file(f))

    report = dict(artifact_kind=KIND, status='ANALYZED' if selected else 'NO_TOP1_MISMATCHES',
        scope='CPU_FULL_VOCABULARY_RANKING_AT_RETAINED_TOP1_MISMATCH_POSITIONS',
        device_execution='NOT_RUN', model_compilation='NOT_RUN', numerical_validation='NOT_EVALUATED',
        source_numerical_validation=original['numerical_validation'], production_promotion='NOT_PERFORMED',
        model_sha256=meta['model']['sha256'], board_attempt=result.get('board_attempt'),
        output_initialization=result.get('output_initialization', 'zero'), comparison=meta['comparison'],
        result_sha256=sha256_file(path), fixture_sha256=result['fixture_sha256'],
        prepared_sha256=binding['prepared_sha256'], binding_sha256=sha256_file(fixture/'pytorch-abi.json'),
        head_weight=record, retained_files=files, unique_hidden_rows=len(rows), numpy_version=np.__version__,
        audit_source_sha256=sha256_file(Path(__file__)),
        arithmetic='NUMPY_CPU_FP32_MATMUL; ALSO_ROUND_FINAL_LOGITS_TO_FP16; FP64_EXPECTED_ACTUAL_PAIR_ONLY',
        report=str(destination), rows=[],
        limitations=[
            'CPU BLAS reduction order/rounding need not match PyTorch or compiled NPU MatMul.',
            'Snapshot hidden may differ from the internal head input; agreement cannot prove buffer visibility.',
            'Only original Top1 mismatch positions are ranked; this is not an accuracy acceptance run.',
            'No device logits are available: this cannot isolate MatMul error from ArgMax or identify an upstream layer.',
            'Original golden tensors and acceptance thresholds are unchanged.'])
    if selected:
        scores, rounded = full_vocab_scores(np.stack(rows), weight, progress)
        for item in selected:
            row = dict(item)
            expected, actual = item['expected'], item['actual']
            pair = weight[[expected, actual]].astype(np.float64)
            for label in ('device_hidden', 'reference_hidden'):
                i = item[label+'_row']
                pair_logits = rows[i].astype(np.float64) @ pair.T
                row[label] = dict(fp32=ranking(scores[i], expected, actual),
                    fp16=ranking(rounded[i], expected, actual),
                    expected_minus_actual_fp64_pair=float(pair_logits[0]-pair_logits[1]))
            row['reference_head_matches_expected'] = row['reference_hidden']['fp16']['top1'] == expected
            row['device_hidden_head_matches_actual'] = row['device_hidden']['fp16']['top1'] == actual
            report['rows'].append(row)
    report['counts'] = dict(positions=len(selected),
        reference_head_matches_expected=sum(r['reference_head_matches_expected'] for r in report['rows']),
        device_hidden_head_matches_actual=sum(r['device_hidden_head_matches_actual'] for r in report['rows']))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(destination)
    summary = {k: report[k] for k in ('artifact_kind', 'status', 'scope', 'device_execution',
        'model_compilation', 'numerical_validation', 'source_numerical_validation',
        'model_sha256', 'output_initialization', 'counts', 'report', 'limitations')}
    summary['rows'] = [{**{k: r[k] for k in ('mode', 'index', 'position', 'actual', 'expected')},
        **{label: dict(top1=r[label]['fp16']['top1'], top2=r[label]['fp16']['top2'],
            top2_margin=r[label]['fp16']['top2_margin'],
            expected_minus_actual=r[label]['fp16']['expected_minus_actual'],
            rounding_changes_top1=r[label]['fp32']['top1'] != r[label]['fp16']['top1'])
            for label in ('reference_hidden', 'device_hidden')}} for r in report['rows']]
    return summary

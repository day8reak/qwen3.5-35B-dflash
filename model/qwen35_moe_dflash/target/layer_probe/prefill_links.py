"""Read-only Prefill capture links; no model, QLinear, or kernel qualification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import KIND, identity, metrics, new_output, read_array, save_json
from .convstate_reference import validate_capture


def inspect_prefill_links(capture, output):
    root, output = Path(capture).resolve(strict=True), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('link inspection output must be separate from capture')
    output = new_output(output)
    report = dict(status='PARTIAL', captured_link_checks='UNAVAILABLE', checks=[], errors=[],
        sequential_write_hypothesis=dict(status='UNAVAILABLE'),
        scope='CAPTURED_PREFILL_LINKS_ONLY; NOT_QLINEAR_NUMERICS_OR_WHOLE_GOLDEN_QUALIFICATION',
        model_compilation='NOT_RUN', device_execution='NOT_RUN')

    def check(label, actual, expected):
        row = dict(name=label, status='FAILED', equal_bits=False, actual_shape=list(actual.shape),
            expected_shape=list(expected.shape), actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype))
        if actual.shape == expected.shape and actual.dtype == expected.dtype:
            row['equal_bits'] = actual.tobytes() == expected.tobytes()
            row['metrics'] = metrics(actual, expected)
            row['status'] = 'PASSED' if row['equal_bits'] else 'FAILED'
        report['checks'].append(row)
        return row

    try:
        manifest = json.loads((root/'manifest.json').read_text())
        report['capture_manifest'] = identity(root/'manifest.json', root)
        regions = manifest['regions']
        lengths = {r['sequence_length'] for r in regions}
        if (manifest.get('artifact_kind') != KIND or manifest.get('status') != 'CAPTURED'
                or manifest.get('reference') != 'ORIGINAL_310P_INFERENCE' or manifest.get('errors')
                or manifest.get('entrypoint') != 'DIRECT_ORIGINAL_SINGLE_PREFILL'
                or set(manifest.get('layers', [])) != {0, 1, 2, 3}
                or not regions or len(lengths) != 1 or type(next(iter(lengths))) is not int or next(iter(lengths)) <= 1
                or any(r['layer'] not in range(4) or r['call'] != 0 or r['phase'] != 'prefill' for r in regions)):
            raise ValueError('requires CAPTURED original L0-L3 single-call fresh Prefill')
        length = next(iter(lengths))
        report['sequence_length'] = length
        report['layers'] = [0, 1, 2, 3]

        def region(layer, point):
            found = [r for r in regions if r['layer'] == layer and r['point'] == point]
            if len(found) != 1:
                raise ValueError(f'need exactly one L{layer} {point}')
            return found[0]

        def spec(record, direction, name):
            found = [s for s in record[direction] if s['name'] == name]
            if len(found) != 1:
                raise ValueError(f'{record["id"]}: missing/duplicate {direction}/{name}')
            return found[0]

        def array(record, direction, name):
            return read_array(root, spec(record, direction, name))

        for layer in range(3):
            try:
                conv, core = region(layer, 'convstate_update'), region(layer, 'gdn')
                projection = region(layer, 'linear_attn_in_proj_qkv')
                hidden = array(projection, 'inputs', 'x')
                for member in ('z', 'b', 'a'):
                    other = region(layer, 'linear_attn_in_proj_' + member)
                    check(f'l{layer}_hidden_to_in_proj_{member}', array(other, 'inputs', 'x'), hidden)
                for source, direction, name in ((conv, 'inputs', 'old_conv_state'), (core, 'inputs', 'initial_state')):
                    old = array(source, direction, name)
                    allowed = (np.float16, np.float32) if name == 'initial_state' else (np.float16,)
                    if old.dtype not in allowed or np.count_nonzero(old):
                        raise ValueError(f'L{layer} requires fresh zero floating {name}, ConvState must be FP16')
                # Reuse the existing exact QKV/concat/tail contract without importing Torch.
                record = dict(sequence_length=length, phase='prefill',
                    components=[conv, projection, region(layer, 'conv1d')],
                    inputs=[None]*4 + [spec(core, 'inputs', 'effective_length')])
                for source in record['components']:
                    for tensor in source['inputs'] + source['outputs']:
                        read_array(root, tensor)
                try:
                    evidence = validate_capture(record, root)
                    report['checks'].append(dict(name=f'l{layer}_qkv_to_convstate_and_padded',
                        status='PASSED', evidence=evidence))
                except ValueError as exc:
                    report['checks'].append(dict(name=f'l{layer}_qkv_to_convstate_and_padded',
                        status='FAILED', error=str(exc)))
            except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
                report['errors'].append(f'L{layer}: {type(exc).__name__}: {exc}')

        try:
            module, projection = region(3, 'full_attention_layer'), region(3, 'self_attn_v_proj')
            cache, core = region(3, 'cache_v'), region(3, 'fused_attention')
            hidden = array(module, 'inputs', 'hidden_states')
            check('l3_module_hidden_to_v_proj', array(projection, 'inputs', 'x'), hidden)
            projected, updates = array(projection, 'outputs', 'golden_y'), array(cache, 'inputs', 'updates')
            if (projected.dtype != np.float16 or projected.ndim != 3 or projected.shape[:2] != (1, length)
                    or projected.shape[2] % 16 or not np.isfinite(projected).all()):
                raise ValueError('V projection requires FP16 [1,T,C], C divisible by 16')
            check('l3_v_proj_reshape_to_cache_updates', updates, projected.reshape(length, -1, 16))
            old, updated = array(cache, 'inputs', 'cache'), array(cache, 'outputs', 'golden_updated_cache')
            check('l3_module_old_value_to_cache_input', old, array(module, 'inputs', 'old_value_cache'))
            core_value = array(core, 'inputs', 'value')
            check('l3_cache_output_to_core_value', updated, core_value)
            check('l3_core_value_to_module_return', core_value, array(module, 'outputs', 'golden_value_cache'))
            positions = array(module, 'inputs', 'new_kv_cache_pos')
            ids, qlen = array(module, 'inputs', 'position_ids'), array(module, 'inputs', 'allQLen')
            table = array(core, 'inputs', 'block_table')
            blocks, offsets = array(cache, 'inputs', 'target_block'), array(cache, 'inputs', 'offset_in_block')
            size = module['attrs']['block_size']
            if (old.dtype != np.float16 or old.ndim != 4 or old.shape[2:] != (size, 16)
                    or np.count_nonzero(old) or updated.shape != old.shape or updated.dtype != old.dtype
                    or updates.dtype != np.float16 or updates.shape != (length, old.shape[1], 16)
                    or not np.isfinite(updates).all() or not np.isfinite(updated).all()
                    or table.dtype != np.int32 or table.ndim != 2 or table.shape[0] != 1
                    or size != 32 or core['attrs']['block_size'] != size or length > table.shape[1]*size):
                raise ValueError('sequential-write hypothesis requires fresh FP16 cache and source INT32 B1 page table')
            if (positions.dtype.kind not in 'iu' or positions.shape != (length, 1)
                    or not np.array_equal(positions[:, 0], np.arange(length))
                    or ids.dtype.kind not in 'iu' or ids.shape != (1, length)
                    or not np.array_equal(ids[0], positions[:, 0])
                    or qlen.dtype != np.int64 or qlen.shape != (1,) or qlen[0] != length):
                raise ValueError('sequential-write hypothesis requires source fresh positions 0..T-1 and INT64[1] length T')
            pages = table[0, np.arange(length)//size]
            if np.any(pages < 0) or np.any(pages >= old.shape[0]) or len(np.unique(table[0, :((length+size-1)//size)])) != (length+size-1)//size:
                raise ValueError('active physical pages must be distinct and in cache range')
            if (blocks.dtype != np.int32 or offsets.dtype != np.int32 or blocks.ndim != 1
                    or offsets.shape != blocks.shape or blocks.size not in (1, length)
                    or (blocks.size == 1 and length > size)):
                raise ValueError('source indices must be INT32 [T] or one start index within one page')
            selected = np.arange(length) if blocks.size == length else np.array([0])
            if not np.array_equal(blocks, pages[selected]) or not np.array_equal(offsets, selected % size):
                raise ValueError('source cache indices disagree with captured fresh positions/page table')
            expected = old.copy()
            active = np.zeros((old.shape[0], size), dtype=bool)
            for token, page in enumerate(pages):
                expected[page, :, token % size, :] = updates[token]
                active[page, token % size] = True
            actual_rows, expected_rows = updated.transpose(0, 2, 1, 3), expected.transpose(0, 2, 1, 3)
            hypothesis = report['sequential_write_hypothesis']
            hypothesis.update(status='PASSED' if updated.tobytes() == expected.tobytes() else 'FAILED',
                scope='SEQUENTIAL_WRITE_MATH_HYPOTHESIS; NOT_INSTALLED_KERNEL_ABI_OR_QLINEAR_QUALIFICATION',
                source_index_count=int(blocks.size), physical_pages=np.unique(pages).tolist(),
                active=dict(equal_bits=actual_rows[active].tobytes() == expected_rows[active].tobytes(),
                            metrics=metrics(actual_rows[active], expected_rows[active])),
                untouched=dict(equal_bits=actual_rows[~active].tobytes() == expected_rows[~active].tobytes(),
                               elements=int(expected_rows[~active].size), nonzero=int(np.count_nonzero(actual_rows[~active]))))
        except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
            report['errors'].append(f'L3: {type(exc).__name__}: {exc}')
    except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    failed = any(r['status'] == 'FAILED' for r in report['checks'])
    report['captured_link_checks'] = 'FAILED' if failed else 'UNAVAILABLE' if report['errors'] else 'PASSED'
    report['status'] = ('FAILED' if failed or report['sequential_write_hypothesis']['status'] == 'FAILED'
                        else 'PARTIAL' if report['errors'] else 'INSPECTED')
    save_json(output/'report.json', report)
    lines = [report['scope'], f'status={report["status"]}; captured_link_checks={report["captured_link_checks"]}']
    if 'sequence_length' in report:
        lines.append(f'sequence_length={length}; layers={report["layers"]}')
    for row in report['checks']:
        detail = {k: row[k] for k in ('equal_bits', 'error') if k in row}
        if row['status'] == 'FAILED':
            detail.update({k: row['metrics'].get(k) for k in ('max_abs', 'relative_l2')} if 'metrics' in row else {})
        lines.append(f'{row["name"]}: {row["status"]}; ' + json.dumps(detail, allow_nan=False))
    hypothesis = report['sequential_write_hypothesis']
    if hypothesis['status'] != 'UNAVAILABLE':
        lines.append('sequential_write_hypothesis: ' + json.dumps(hypothesis, allow_nan=False))
    lines.extend('UNAVAILABLE: ' + error for error in report['errors'])
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
    return report

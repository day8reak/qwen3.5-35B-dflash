"""Retained row patterns and V fingerprints; no device access or new reference."""
import hashlib

import numpy as np

from .accuracy import metrics
from .accuracy_diagnose import tensor_difference


def value_rows(banks):
    """One row joins all six layers' V heads; K is excluded because of RoPE."""
    if len(banks) != 12 or any(a.shape != banks[0].shape for a in banks):
        raise ValueError('row audit requires twelve equal-shaped ND KV banks')
    if banks[0].ndim != 4 or banks[0].shape[0] != 1:
        raise ValueError('row audit requires batch-one KV')
    return np.stack([a[0].transpose(1, 0, 2) for a in banks[1::2]], axis=1)


def changed_rows(left, right, start):
    """Report every changed appended row, including per-head channel runs."""
    if left.shape != right.shape or left.dtype != right.dtype or left.ndim != 4 or left.shape[0] != 1:
        raise ValueError('row audit requires matching batch-one KV regions')
    different = left.view('<u2') != right.view('<u2')
    rows = []
    for row in np.flatnonzero(different.any(axis=(0, 1, 3))):
        heads = []
        for head in np.flatnonzero(different[0, :, row].any(axis=-1)):
            mask = different[0, head, row]
            edges = np.diff(np.r_[False, mask, False].astype(np.int8))
            heads.append(dict(head=int(head), different_elements=int(mask.sum()),
                channel_intervals=[[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]))
        rows.append(dict(cache_position=start+int(row), local_row=int(row), heads=heads,
            between_modes=tensor_difference(left[:, :, row], right[:, :, row])))
    return rows


def distance(a, b):
    a, b = a.astype(np.float64).reshape(-1), b.astype(np.float64).reshape(-1)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(finite=False, relative_l2_error=None, cosine=None)
    delta = a-b; na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(finite=True, relative_l2_error=float(np.linalg.norm(delta)/nb) if nb else None,
        absolute_l2_error=float(np.linalg.norm(delta)), max_abs_error=float(np.max(np.abs(delta), initial=0)),
        query_l2_norm=na, reference_l2_norm=nb,
        cosine=float(np.clip(np.dot(a, b)/(na*nb), -1., 1.)) if na and nb else None)


def paired_keys(current, values, mode, position, start, end, context_rows):
    """Compare K only where same-call six-layer V is an exact row duplicate."""
    query = values[mode][position]
    same = np.all(values[mode][:end].view('<u2') == query.view('<u2'), axis=(1, 2, 3))
    same[position] = False
    positions = [int(p) for p in np.flatnonzero(same)]
    other = 'feedback' if mode == 'isolated' else 'isolated'
    rows = []
    for p in positions[:8]:
        keys = [dict(layer=j//2,
            same_call=tensor_difference(current[mode][j+2][:, :, position], current[mode][j+2][:, :, p]),
            other_mode_matched_position=tensor_difference(current[mode][j+2][:, :, position], current[other][j+2][:, :, p]))
            for j in range(0, 12, 2)]
        rows.append(dict(matched_cache_position=p,
            source_index_formula='clamp(cache_position - start, 0, context_rows - 1)',
            expected_query_index=max(0, min(position-start, context_rows-1)),
            expected_matched_index=max(0, min(p-start, context_rows-1)),
            all_key_layers_same_call_bitwise_equal=all(r['same_call']['bitwise_equal'] for r in keys),
            keys=keys))
    return dict(exact_value_match_positions=positions, comparisons=rows,
        comparison_coverage=dict(total=len(positions), analyzed=len(rows), omitted=max(0, len(positions)-8)),
        limitation='K equality locates a repeated post-RoPE value, not the exact failing operator or device index bytes.')


class ValueCorpus:
    """Deduplicate full six-layer V rows while preserving occurrence identities."""
    def __init__(self):
        self.entries = {}
        self.occurrences = 0

    def add(self, banks, end, *, dataset, mode, index, call_ordinal=None):
        for position, row in enumerate(value_rows(banks)[:end]):
            raw = row.tobytes(); key = hashlib.sha256(raw).hexdigest()
            if key not in self.entries:
                self.entries[key] = dict(value=row.copy(), occurrences=[])
            self.entries[key]['occurrences'].append(dict(dataset=dataset, mode=mode, index=index,
                cache_position=position, call_ordinal_zero_based=call_ordinal))
            self.occurrences += 1

    def nearest(self, query, *, mode, index, position, ordinal):
        groups = dict(earlier_candidate_calls=[], same_call_other_positions=[], other_positions=[])
        for sha, entry in self.entries.items():
            refs = entry['occurrences']
            eligible = dict(
                earlier_candidate_calls=[r for r in refs if r['dataset'] == 'candidate'
                    and r['call_ordinal_zero_based'] < ordinal],
                same_call_other_positions=[r for r in refs if r['dataset'] == 'candidate'
                    and r['mode'] == mode and r['index'] == index and r['cache_position'] != position],
                other_positions=[r for r in refs if r['cache_position'] != position])
            if not any(eligible.values()):
                continue
            observed = distance(query, entry['value'])
            if not observed['finite']:
                continue
            # Rank by actual vector distance; relative error is also reported.
            for name, matches in eligible.items():
                if matches:
                    groups[name].append((observed['absolute_l2_error'], sha, observed, matches))
        result = {}
        for name, entries in groups.items():
            found = []
            for _, sha, observed, refs in sorted(entries, key=lambda r:(r[0], r[1]))[:3]:
                value = self.entries[sha]['value']
                found.append(dict(row_sha256=sha, bitwise_equal=query.tobytes() == value.tobytes(),
                    distance=observed, representative=refs[0], matching_scope_occurrences=len(refs),
                    per_layer=[dict(layer=i, **distance(a, b)) for i, (a, b) in enumerate(zip(query, value))]))
            result[name] = found
        return result


class RowAudit:
    def __init__(self, cases, atol, rtol, context_rows):
        self.count = len(cases)
        self.context_rows = context_rows
        self.atol, self.rtol = atol, rtol
        self.corpus = ValueCorpus()
        self.patterns, self.queries = [], []

    def observe(self, step, current, previous, golden, specs):
        index, start, valid = (step[k] for k in ('index', 'start', 'valid'))
        end = start+valid
        for dataset, collection in (('candidate', current), ('baseline', previous)):
            for mode, outputs in collection.items():
                self.corpus.add(outputs[2:], end, dataset=dataset, mode=mode, index=index,
                    call_ordinal=index+(self.count if mode == 'feedback' else 0))
        self.corpus.add(golden, end, dataset='reference', mode=None, index=index)
        positions = set()
        for j, spec in enumerate(specs[2:]):
            a, b = (current[m][j+2][:, :, start:end] for m in ('isolated', 'feedback'))
            changes = changed_rows(a, b, start)
            if changes:
                unchanged = np.ones(valid, dtype=bool)
                unchanged[[r['local_row'] for r in changes]] = False
                remaining = {m:metrics(current[m][j+2][:, :, start:end][:, :, unchanged],
                    golden[j][:, :, start:end][:, :, unchanged], self.atol, self.rtol)
                    for m in current} if unchanged.any() else None
                self.patterns.append(dict(index=index, label=step['label'], tensor=spec['name'],
                    layer=j//2, kind='K' if j%2 == 0 else 'V', changed_rows=changes,
                    unchanged_rows=int(unchanged.sum()), unchanged_rows_vs_reference=remaining))
                positions.update(r['cache_position'] for r in changes)
        if not positions:
            return
        actual = {m:value_rows(v[2:]) for m, v in current.items()}
        reference = value_rows(golden)
        for position in sorted(positions):
            per_mode = {m:dict(value=v[position].copy(),
                vs_reference=metrics(v[position], reference[position], self.atol, self.rtol),
                paired_keys=paired_keys(current, actual, m, position, start, end, self.context_rows))
                for m, v in actual.items()}
            delta = distance(actual['isolated'][position], actual['feedback'][position])
            self.queries.append(dict(index=index, label=step['label'], cache_position=position,
                local_row=position-start, between_modes=delta, modes=per_mode))

    def report(self, fixture, meta):
        # All changed-row patterns are retained. Bound expensive row searches.
        ordered = sorted(self.queries, key=lambda q:q['between_modes'].get('absolute_l2_error', float('inf')), reverse=True)
        matches = []
        feature_slot = next(s for s in meta['inputs'] if s['name'] == 'target_features')
        for query in ordered[:32]:
            index, position = query['index'], query['cache_position']
            row = {k:v for k, v in query.items() if k != 'modes'}
            row['value_modes'] = {}
            for mode, detail in query['modes'].items():
                value = detail['value']
                row['value_modes'][mode] = dict(vs_reference=detail['vs_reference'],
                    paired_keys=detail['paired_keys'],
                    row_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                    nearest=self.corpus.nearest(value, mode=mode, index=index, position=position,
                        ordinal=index+(self.count if mode == 'feedback' else 0)))
            features = np.fromfile(fixture/f'inputs/{index}/{feature_slot["index"]}.bin', dtype='<f2').reshape(feature_slot['shape'])[0]
            feature = features[query['local_row']]
            row['retained_feature_row'] = dict(input_slot=feature_slot['index'],
                local_row=query['local_row'], bytes=feature.nbytes,
                sha256=hashlib.sha256(feature.tobytes()).hexdigest(),
                finite=bool(np.isfinite(feature).all()), scope='FILE_BYTES_NOT_DEVICE_READBACK')
            matched = {p for detail in query['modes'].values()
                       for p in detail['paired_keys']['exact_value_match_positions']}
            start = position-query['local_row']
            row['retained_feature_pair_comparisons'] = [dict(cache_position=p, local_row=p-start,
                **tensor_difference(feature, features[p-start])) for p in sorted(matched)
                if start <= p < start+len(features)]
            past = []
            for spec in meta['inputs']:
                if spec['name'].startswith('past_'):
                    data = np.fromfile(fixture/f'inputs/{index}/{spec["index"]}.bin', dtype='<f2').reshape(spec['shape'])[:, :, position]
                    past.append(dict(tensor=spec['name'], elements=int(data.size),
                        zero_elements=int((data == 0).sum()), finite=bool(np.isfinite(data).all())))
            row['retained_past_rows'] = past
            matches.append(row)
        return dict(status='ANALYZED', pattern_scope='ALL_APPENDED_KV_ROWS_DIFFERING_BETWEEN_MODES',
            changed_case_indices=sorted({r['index'] for r in self.patterns}),
            changed_tensor_regions=len(self.patterns), changed_case_positions=len(self.queries),
            patterns=self.patterns, value_fingerprints=matches,
            fingerprint_coverage=dict(total=len(ordered), analyzed=len(matches), omitted=max(0, len(ordered)-32)),
            corpus=dict(unique_value_rows=len(self.corpus.entries), occurrences=self.corpus.occurrences,
                scope='COMMITTED_ROWS_FROM_BOTH_MODELS_BOTH_MODES_AND_REFERENCE_AT_APPENDING_CASES'),
            limitations=['V fingerprints join six layers and all heads; K is excluded because positions change RoPE.',
                'A nearest row is not proof of origin; inspect distance, per-layer agreement and call order.',
                'Later snapshots may repeat an earlier corrupted row; only earlier candidate calls can support stale-output hypotheses.',
                'Retained feature hashes do not establish what the NPU consumed.',
                'Only completed retained snapshots are searched; transient internal buffers were not captured.'])

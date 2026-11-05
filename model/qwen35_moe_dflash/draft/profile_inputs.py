"""Create a new profile input plan after full content revalidation.

Old real-feature receipts are immutable. Their inode/timestamp observations can
be stale even when all bytes match. This path records fresh identities, reruns
the slot/ABI checks and copies only the three small inputs into the new run.
It never accepts changed metadata, weights or input payloads as the old case.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.draft import real_feature_draft as real
from qwen35_moe_dflash.common.file_io import _relative, _safe


def pack_accuracy_features(raw, source_record, output, *, fixture):
    """Write one physical profiling input; retain the logical fixture unchanged.

    Vendor tools take ready-to-use tensor bytes and do not execute the native
    runner's preparation. This offline permutation is excluded from their timer.
    """
    import numpy as np
    from qwen35_moe_dflash.draft.incremental.air_feature_pack import CONTRACT, pack_words
    from qwen35_moe_dflash.target.source_inventory import _read_small
    out = output_path(output)
    fixture = Path(fixture).resolve()
    if out == fixture or fixture in out.parents or out in fixture.parents:
        raise ValueError('packed profile input must be outside the retained fixture')
    if (len(raw) != CONTRACT['bytes'] or source_record['bytes'] != len(raw)
            or source_record['sha256'] != hashlib.sha256(raw).hexdigest()):
        raise ValueError('logical profile feature bytes differ from their verified source')
    words = np.frombuffer(raw, dtype='<u2').reshape(CONTRACT['logical_shape'])
    packed = pack_words(words).tobytes()
    out.mkdir(parents=True, exist_ok=False)
    path = out/'target_features.bin'
    with path.open('xb') as stream:
        stream.write(packed)
    _, record = _read_small(path, CONTRACT['bytes'])
    receipt = dict(policy=CONTRACT['policy'], input=CONTRACT['input'],
        logical_source=source_record, physical_file=record,
        logical_shape=CONTRACT['logical_shape'], physical_shape=CONTRACT['physical_shape'],
        arithmetic=CONTRACT['arithmetic'], status='PACKED_FOR_VENDOR_PROFILE',
        preparation_scope='HOST_OFFLINE; EXCLUDED_FROM_VENDOR_MODEL_AND_OPERATOR_TIMINGS',
        retained_fixture='UNCHANGED', model_compilation='NOT_RUN')
    save_json(out/'manifest.json', receipt)
    return record, receipt


def refresh_profile_inputs(prepared, output):
    root, _ = _safe(prepared, directory=True)
    out = output_path(output)
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('fresh profiling inputs must be outside the old preparation')
    out.mkdir(parents=True, exist_ok=False)
    before, current = {}, {}

    def observe(path, row):
        # Drop only the historic stat comparison for this explicit new
        # preparation. Byte length and SHA256 remain mandatory and unchanged.
        raw, identity = real._same_file(path, {key: row[key] for key in ('bytes', 'sha256')})
        key = identity['path']
        if key in current and identity != current[key]:
            raise ValueError('source changed during profile content revalidation: ' + key)
        current[key] = identity
        if before.get(key) is None:
            before[key] = row.get('stat')
        return raw, identity

    old, old_id = real._metadata(root / 'manifest.json')
    if (old.get('artifact_kind') != real._KIND or old.get('status') != real._STATUS
            or old.get('output') != str(root) or old.get('sequence_length') != 64 or old.get('past_tokens') != 0):
        raise ValueError('requires an existing complete S64 real-feature preparation')
    observe(root / 'manifest.json', old_id)
    bindings = old.get('source_bindings')
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 1024:
        raise ValueError('original real-feature source inventory is missing or unbounded')
    for row in bindings:
        observe(row['path'], row)
    raw, _ = observe(old['source_inspection']['path'], old['source_inspection'])
    prior = json.loads(raw)
    _, ids = real._inspection_contract(prior)
    if len(old.get('inputs', [])) != 3 or len(old.get('upload_payloads', [])) != 5:
        raise ValueError('original real-feature input/ABI inventory is incomplete')
    for row in [*old['inputs'], old['draft_abi'], old['step_plan'], *old['upload_payloads']]:
        observe(_relative(root, row['path']), row)
    bundle_manifest = str(Path(old['source_bundle']) / 'manifest.json')
    if bundle_manifest not in current:
        raise ValueError('original source inventory must bind the selected bundle manifest bytes')
    fresh_inspection = real.inspect_target_prefill_outputs(prior['execution'], out / 'inspection',
        draft_bundle=old['source_bundle'], input_dir=Path(ids['path']).parent)
    real._inspection_contract(fresh_inspection)
    fresh = real.prepare_real_feature_draft(out / 'inspection', old['source_bundle'], out / 'prepared')
    for key in ('ddk_id', 'runtime_version', 'firmware_id', 'draft_config', 'direct_spec', 'model', 'outputs',
                'source_bundle', 'weight_storage', 'past_tokens'):
        if old.get(key) != fresh.get(key):
            raise ValueError('fresh profile input contract differs from the original: ' + key)
    for key in ('path', 'bytes', 'sha256'):
        if old['model_source'].get(key) != fresh['model_source'].get(key):
            raise ValueError('fresh model payload differs from the original profile source')
    model_path = fresh['model_source']['path']
    before[model_path] = old['model_source'].get('stat')
    current[model_path] = fresh['model_source']
    for old_row, fresh_row in zip(old['inputs'], fresh['inputs']):
        for key in ('index', 'name', 'shape', 'dtype', 'bytes', 'sha256'):
            if old_row.get(key) != fresh_row.get(key):
                raise ValueError('fresh input is not byte-identical to the original: ' + key)
    for path, row in current.items():
        if _safe(path)[1] != row['stat']:
            raise ValueError('source changed during profile preparation: ' + path)
    changes = [{'path': path, 'old_stat': before[path], 'current_stat': row['stat'],
        'bytes': row['bytes'], 'sha256': row['sha256']} for path, row in current.items()
        if before[path] is not None and before[path] != row['stat']]
    receipt = {'status': 'PROFILE_INPUTS_REBOUND_IDENTICAL_BYTES', 'original_prepared': old_id,
        'prepared': str(out / 'prepared'), 'source_file_count': len(current),
        'identity_change_count': len(changes), 'identity_changes': changes,
        'model_sha256': fresh['model_source']['sha256'], 'input_sha256': [r['sha256'] for r in fresh['inputs']],
        'old_receipts': 'UNCHANGED', 'model_copies': 0, 'model_execution': 'NOT_RUN',
        'scope': 'Full original metadata/input content revalidation, fresh slot/ABI binding and model SHA; no accuracy pass'}
    save_json(out / 'report.json', receipt)
    return out / 'prepared', fresh, receipt

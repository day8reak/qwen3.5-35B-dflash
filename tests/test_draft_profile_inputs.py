"""Fresh profile preparations tolerate physical drift, never content drift."""
import os

import pytest

from qwen35_moe_dflash import draft_profile_inputs as profile
from qwen35_moe_dflash import real_feature_draft as real
from qwen35_moe_dflash import draft_profile_plan
from test_real_feature_draft import fixture, prepare, retained_case
from test_draft_profile_plan import inspection


def replace_same_bytes(path):
    replacement = path.with_name(path.name + '.replacement')
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)


def test_bundle_manifest_inode_drift_rebinds_without_rewriting_old_receipts(fixture):
    original = prepare(fixture)
    old_manifest = (fixture['out'] / 'manifest.json').read_bytes()
    old_inspection = (fixture['out'] / 'source-validation/report.json').read_bytes()
    bundle_manifest = fixture['bundle'] / 'manifest.json'
    replace_same_bytes(bundle_manifest)
    with pytest.raises(ValueError, match='source identity changed'):
        real.validate_real_feature_draft(fixture['out'])
    tool, _ = inspection(fixture)
    bound = draft_profile_plan.bind_profile(fixture['out'], tool,
        refresh_output=fixture['out'].parent / 'profile-rebound')
    receipt = bound['input_revalidation']
    assert receipt['status'] == 'PROFILE_INPUTS_REBOUND_IDENTICAL_BYTES'
    assert any(row['path'] == str(bundle_manifest) for row in receipt['identity_changes'])
    assert bound['model']['sha256'] == original['model_source']['sha256']
    assert [row['sha256'] for row in bound['inputs']] == [row['sha256'] for row in original['inputs']]
    assert (fixture['out'] / 'manifest.json').read_bytes() == old_manifest
    assert (fixture['out'] / 'source-validation/report.json').read_bytes() == old_inspection
    assert real.validate_real_feature_draft(receipt['prepared'])['model'] == original['model']


def test_identical_input_and_model_replacement_keeps_bytes_and_new_identity(fixture):
    original = prepare(fixture)
    replace_same_bytes(fixture['out'] / original['inputs'][0]['path'])
    replace_same_bytes(fixture['bundle'] / 'model/model.omc')
    root, fresh, receipt = profile.refresh_profile_inputs(fixture['out'], fixture['out'].parent / 'rebound')
    assert fresh['model_source']['sha256'] == original['model_source']['sha256']
    assert fresh['model_source']['stat'] != original['model_source']['stat']
    assert receipt['identity_change_count'] >= 1
    assert not (root / 'model').exists()


@pytest.mark.parametrize('changed', ['bundle-manifest', 'prepared-input', 'source-feature', 'model'])
def test_any_changed_content_is_rejected_before_device_work(fixture, changed):
    original = prepare(fixture)
    path = {'bundle-manifest': fixture['bundle'] / 'manifest.json',
        'prepared-input': fixture['out'] / original['inputs'][0]['path'],
        'source-feature': fixture['case']['board'] / fixture['case']['paths'][81],
        'model': fixture['bundle'] / 'model/model.omc'}[changed]
    raw = path.read_bytes()
    path.write_bytes(raw + b' ' if changed == 'bundle-manifest' else bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match='SHA256|extent|size'):
        profile.refresh_profile_inputs(fixture['out'], fixture['out'].parent / 'rebound')


def test_metadata_changed_during_refresh_is_rejected(fixture, monkeypatch):
    prepare(fixture)
    original = real.prepare_real_feature_draft
    path = fixture['bundle'] / 'manifest.json'
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        replace_same_bytes(path)
        return result
    monkeypatch.setattr(real, 'prepare_real_feature_draft', changed)
    with pytest.raises(ValueError, match='source changed during'):
        profile.refresh_profile_inputs(fixture['out'], fixture['out'].parent / 'rebound')

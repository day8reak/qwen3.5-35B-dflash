"""Fresh prefill trials revalidate only historical small-primary ctime drift."""
import os
from pathlib import Path

import pytest

from qwen35_moe_dflash import target_prefill_smoke_plan as planner
from test_target_prefill_smoke_plan import case, prepare
from test_original_target_primary_revalidation import ctime_only


def test_new_prefill_plan_freshly_rehashes_ctime_only_primary_and_preserves_old_receipt(case):
    report = case['compilation'] / 'report.json'
    saved = report.read_bytes()
    ctime_only(case['model'])
    plan = prepare(case)
    assert plan['status'] == 'PREFILL_SMOKE_PLANNED'
    observations = plan['primary_omc_revalidation_observations']
    assert len(observations) == 1 and observations[0]['status'] == 'SHA256_VERIFIED'
    assert observations[0]['bytes'] == case['model'].stat().st_size
    assert report.read_bytes() == saved
    assert planner.validate_prefill_smoke_plan(plan) == plan


@pytest.mark.parametrize('fault', ['content', 'inode', 'mtime'])
def test_prefill_does_not_accept_other_primary_changes_as_historical_ctime(case, fault):
    path = case['model']; before = path.stat(); raw = path.read_bytes()
    if fault == 'content':
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif fault == 'inode':
        path.rename(path.with_name('retained-old-model.omc'))
        path.write_bytes(raw)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
    with pytest.raises(ValueError):
        prepare(case)


def test_current_primary_change_during_input_planning_remains_fatal(case, monkeypatch):
    ctime_only(case['model'])
    source = planner._source
    changed = False
    def mutate(path, size):
        nonlocal changed
        row = source(path, size)
        if Path(path).name == 'inputs_embeds.bin' and not changed:
            changed = True
            ctime_only(case['model'])
        return row
    monkeypatch.setattr(planner, '_source', mutate)
    with pytest.raises(ValueError, match='primary OMC changed'):
        prepare(case)

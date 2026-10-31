import hashlib
import json

import pytest

from qwen35_moe_dflash import draft_profile_plan as profile
from test_real_feature_draft import fixture, prepare, retained_case


def inspection(fixture):
    text = 'Usage: ' + ' '.join('--' + flag for flag in profile.FLAGS) + '\nV1;V2;ROMC\n1:Enable profiling; 2:Enable dump'
    row = {'artifact_kind': 'qwen35-target-board-tool-inspection', 'status': 'INSPECTED',
        'tool_identity': 'SHA256_MATCHED_BEFORE_AND_AFTER',
        'tool_sha256_before': 'a' * 64, 'tool_sha256_after': 'a' * 64,
        'tool': '/data/local/tmp/tools_sysdbg/model_run_tool_internal',
        'firmware': {'text': 'fixture-firmware'},
        'help': {'text': text, 'utf8_bytes': len(text.encode()),
            'utf8_sha256': hashlib.sha256(text.encode()).hexdigest(), 'truncated': False,
            'observation': 'HELP_TEXT_OBSERVED'}}
    path = fixture['out'].parent / 'tool.json'
    path.write_text(json.dumps(row))
    return path, row


def test_real_preparation_and_matching_tool_help_bind_current_model_inputs(fixture):
    prepare(fixture)
    path, _ = inspection(fixture)
    bound = profile.bind_profile(fixture['out'], path)
    assert bound['model']['sha256'] == fixture['bundle_manifest']['model']['sha256']
    assert [r['dtype'] for r in bound['inputs']] == ['float16', 'int32', 'int32']
    assert bound['times'] == 1
    assert 'NOT_EXACT' in bound['input_scope']


@pytest.mark.parametrize('mutation', ['firmware', 'help', 'v1', 'hash'])
def test_changed_tool_contract_prevents_profile(fixture, mutation):
    prepare(fixture)
    path, row = inspection(fixture)
    if mutation == 'firmware':
        row['firmware']['text'] = 'other-firmware'
    elif mutation == 'help':
        row['help']['text'] += ' changed'
    elif mutation == 'v1':
        text = row['help']['text'].replace('V1;V2', 'V2')
        row['help'].update(text=text, utf8_bytes=len(text.encode()), utf8_sha256=hashlib.sha256(text.encode()).hexdigest())
    else:
        row['tool_sha256_after'] = 'b' * 64
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='tool help/hash/firmware'):
        profile.bind_profile(fixture['out'], path)

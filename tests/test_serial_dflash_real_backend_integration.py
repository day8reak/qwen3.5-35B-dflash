"""Actual Python adapters/controller/CLI with fake native transport boundaries.

The full40/six-layer tensor extents are retained. Device execution and numerical
accuracy are never claimed by these host fixtures.
"""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

from qwen35_moe_dflash import serial_dflash_smoke as entry
from qwen35_moe_dflash.existing_draft_smoke_backend import ExistingDraftSmokeBackend
from qwen35_moe_dflash.original_target_smoke_backend import OriginalTargetSmokeBackend
from qwen35_moe_dflash.target_compile_failure import _read_small
from test_existing_draft_smoke_backend import case as draft_case
from test_original_target_smoke_backend import case as target_case


def test_cli_actual_adapters_p8_g8_keep_all_native_receipts_and_refuse_host_device_promotion(tmp_path, monkeypatch, capsys):
    target_root = tmp_path / 'target-fixture'; target_root.mkdir()
    draft_root = tmp_path / 'draft-fixture'; draft_root.mkdir()
    target_fixture = target_case.__wrapped__(target_root, monkeypatch, SimpleNamespace(param=False))
    draft_fixture = draft_case.__wrapped__(draft_root, monkeypatch)
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    schema = target_root / 'producer_pb2.py'; schema.write_text('# host fixture metadata; never imported\n')
    target_fixture['current']['source_bindings'].append(_read_small(schema, 4096)[1])
    prompt = tmp_path / 'prompt-ids.bin'
    prompt.write_bytes(b''.join(value.to_bytes(4, 'little', signed=True) for value in range(3, 11)))
    instances = {}
    def make_target(*args, **options):
        instances['target'] = OriginalTargetSmokeBackend(*args, **options, evidence_kind='host-fake-hiai')
        return instances['target']
    def make_draft(*args, **options):
        instances['draft'] = ExistingDraftSmokeBackend(*args, **options, _host_test=True)
        return instances['draft']
    monkeypatch.setattr(entry, '_backends', lambda: (make_target, make_draft))
    script = Path(__file__).parents[1] / 'targets/cpro/scripts/run_serial_dflash_smoke.py'
    main = runpy.run_path(str(script))['main']
    output = tmp_path / 'combined-serial-smoke'
    args = {'decode-job': target_fixture['job'], 'runtime-probe': target_fixture['probe'],
        'state-lineage': target_fixture['lineage'], 'source-torchair-pb2': schema,
        'draft-bundle': draft_fixture['root'], 'weight-table': target_fixture['lookup']._tables[0]['path'],
        'scale-table': target_fixture['lookup']._tables[1]['path'], 'prompt-input-ids': prompt,
        'output': output, 'prompt-length': 8, 'max-new-tokens': 8, 'max-proposals-per-round': 2,
        'minimum-draft-rounds': 2, 'lock-wait-seconds': 0}
    argv = [part for key, value in args.items() for part in ('--' + key, str(value))] + ['--apply']
    assert main(argv) == 1  # Host-native doubles cannot become device success.
    shown = json.loads(capsys.readouterr().out.splitlines()[-1])
    result = json.loads((output / 'result.json').read_text())
    loop = json.loads((output / 'loop.json').read_text())
    assert shown['status'] == result['status'] == 'FAILED'
    assert 'native execution receipts' in result['error']
    assert result['device_execution'] != 'EXECUTED_OUTPUTS_RETRIEVED'
    assert loop['status'] == 'EXECUTION_COMPLETE' and loop['multi_round_observed']
    assert loop['generated_token_ids'] == list(range(11, 19))
    assert loop['evidence_kinds'] == ['host-fake-hiai']
    target, draft = instances['target'], instances['draft']
    assert target.prefix == list(range(3, 19)) and target.features.shape == (1, 16, 16384)
    assert target.execution_summary()['native_calls'] == len(target_fixture['calls']) == 16
    assert loop['counts']['target_calls'] == 9
    assert draft.execution_summary()['native_calls'] == loop['counts']['draft_calls'] >= 2
    assert draft.execution_summary()['runner_build_count'] == len(draft_fixture['builds']) == 1
    assert result['native_executions']['target']['native_calls'] == 16
    prefill = json.loads((output / 'target/prefill-receipt.json').read_text())
    assert prefill['native_call_count'] == len(prefill['native_calls']) == 8
    previous = None
    for step in target_fixture['calls']:
        chain = json.loads((step['root'].parent / 'state-chain.json').read_text())
        assert chain['feedback_input_count'] == 80 and chain['bound_child_output_count'] == 82
        assert chain['status'] == 'ALL80_PARENT_OUTPUT_BYTES_USED_BY_CHILD_INPUTS'
        if previous is not None:
            old = {row['source_refdata']: row['child_output']['sha256'] for row in previous['states']}
            assert all(row['parent_output']['sha256'] == old[row['source_refdata']] for row in chain['states'])
        previous = chain
    for call in sorted((output / 'draft').glob('call-*')):
        proposal = json.loads((call / 'proposal.json').read_text())
        manifest = json.loads((call / 'prepared/manifest.json').read_text())
        assert proposal['proposal_count'] == len(proposal['draft_top1']) == 15
        count = manifest['logical_prefix_length']
        assert manifest['token_provenance']['anchor_position'] == count - 1
        assert manifest['token_provenance']['prefix_policy'].endswith('NO_TRUNCATION_OR_REBASE')
        assert count <= 64 and manifest['sequence_length'] == 64
    assert result['model_compilation'] == 'NOT_RUN' and result['performance'] == 'NOT_EVALUATED'

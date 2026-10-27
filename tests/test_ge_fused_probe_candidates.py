"""Candidate operator interfaces only; no installed DDK or device execution."""
import json

import pytest

from qwen35_moe_dflash.ge_fused.bindings import document_bindings
from qwen35_moe_dflash.ge_fused import probes
from qwen35_moe_dflash.hashing import sha256_file


@pytest.fixture
def bindings(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    path = tmp_path / 'bindings.json'
    path.write_text(json.dumps(document_bindings(target='kirinx90', ddk_id='SIMULATED-DDK',
                                                 gdn_initial_state_dtype='float16')))
    return path


def test_document_conv_candidate_keeps_default_width_three(bindings, tmp_path):
    result = probes.build_fused_probes(tmp_path / 'default', bindings, cases='conv_prefill,conv_decode')
    assert result['status'] == 'SOURCE_READY'
    for case in result['cases']:
        assert case['inputs'][1]['shape'] == [1, 8192, 3]
        assert case['outputs'][0]['reference'] is None
        assert case['outputs'][1]['reference'] is not None


def test_installed_width_four_probe_has_no_assumed_state_or_padded_reference(bindings, tmp_path):
    digest = sha256_file(bindings)
    root = tmp_path / 'width-four'
    result = probes.build_fused_probes(root, bindings, cases='conv_prefill,conv_decode', conv_history_width=4)
    assert result['status'] == 'SOURCE_READY'
    assert sha256_file(bindings) == digest
    assert json.loads(bindings.read_text())['operators']['conv_state']['inputs']['history']['shape'] == [1, 8192, 3]
    for case in result['cases']:
        assert case['inputs'][1]['shape'] == [1, 8192, 4]
        assert case['outputs'][0]['shape'] == [1, 8192, 4]
        assert case['outputs'][1]['shape'] == [1, 8192, case['sequence_length'] + 3]
        assert all(output['reference'] is None for output in case['outputs'])
        assert all('SHAPE_ONLY' in output['comparison'] for output in case['outputs'])
        abi = json.loads((root / case['operator_abi']['path']).read_text())
        assert abi['probe_override']
        assert not list((root / 'cases' / case['name']).glob('reference_*.bin'))


def test_missing_rms_registration_does_not_stop_later_operator_compilation(bindings, tmp_path, monkeypatch):
    calls = []

    def export(folder, *unused):
        (folder / 'probe.air').write_bytes(b'SIMULATED AIR; no real GE export')
        return {'simulation': True}

    def compile_case(folder, *unused):
        calls.append(folder.name)
        if folder.name == 'rms':
            raise RuntimeError('SIMULATED AdnRmsNorm infershape func failed')
        destination = folder / 'omc'
        destination.mkdir()
        (destination / 'probe.omc').write_bytes(b'SIMULATED OMC; no real compiler')
        return {'simulation': True}

    monkeypatch.setattr(probes, '_export_case', export)
    monkeypatch.setattr(probes, '_compile_case', compile_case)
    root = tmp_path / 'continue-after-rms'
    result = probes.build_fused_probes(root, bindings, ge_root=tmp_path, ddk_root=tmp_path,
                                       ddk_id='SIMULATED-DDK', cases='rms,cache_prefill,attention_prefill')
    assert result['status'] == 'PARTIAL'
    assert calls == ['rms', 'cache_prefill', 'attention_prefill']
    rms, cache, attention = result['cases']
    assert rms['operator_type'] == 'AdnRmsNorm' and rms['export'] == 'PASSED' and rms['compile'] == 'FAILED'
    assert 'infershape func failed' in rms['error']
    assert cache['operator_type'] == 'CacheUpdate' and attention['operator_type'] == 'AdnFusedInferAttention'
    assert cache['compile'] == attention['compile'] == 'PASSED'
    script = (root / 'bundle/run-board.sh').read_text()
    assert 'rms/step.tsv' not in script
    assert 'cache_prefill/step.tsv' in script and 'attention_prefill/step.tsv' in script

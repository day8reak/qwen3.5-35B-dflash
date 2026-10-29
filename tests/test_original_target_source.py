"""The source adapter cannot silently truncate, overwrite or rebuild a target."""
import ast
import pytest

from qwen35_moe_dflash.original_target_source import (
    EXPORTER, LAYER_IDS, SOURCE_FILES, WRAPPER,
    patch_cpro_wrapper, prepare_original_target_source,
)


# An independent small ABI stand-in, never copied from private vendor code.
WRAPPER_SOURCE = '''import torch
class Unrelated:
    untouched = 42
class Qwen3_5MoeForCausalLMCproWrapper:
    def forward(self, hidden_states, past_key_values, use_cache=True):
        kv_caches_out = []
        for idx, decoder_layer in enumerate(self.model.model.layers):
            layer_outputs = decoder_layer(hidden_states, past_key_values=past_key_values[idx],
                layer_idx=idx, new_kv_cache_pos=None, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            kv_caches_out.extend(layer_outputs[1])
        lm_logits = hidden_states
        return lm_logits, *kv_caches_out
'''


def test_minimal_three_edits_preserve_every_original_statement():
    patched = patch_cpro_wrapper(WRAPPER_SOURCE)
    old, new = ast.parse(WRAPPER_SOURCE), ast.parse(patched)
    assert ast.dump(old.body[1]) == ast.dump(new.body[1])
    assert patched.count('.clone()') == 1 and patched.count('torch.cat(') == 1
    assert str(LAYER_IDS) in patched
    assert 'len(self.model.model.layers) != 40' in patched
    assert 'len(past_key_values) != 40' in patched
    assert 'len(kv_caches_out) != 80' in patched
    assert 'return lm_logits, *kv_caches_out, torch.cat(' in patched
    with pytest.raises(ValueError, match='already patched'):
        patch_cpro_wrapper(patched)


@pytest.mark.parametrize('before,after', [
    ('CproWrapper', 'FirstWrapper'),
    ('idx, decoder_layer', 'index, decoder_layer'),
    ('enumerate(self.model.model.layers)', 'enumerate(self.model.model.layers[:38])'),
    ('hidden_states = layer_outputs[0]', 'hidden_states = layer_outputs[1]'),
    ('kv_caches_out.extend(layer_outputs[1])', 'pass'),
    ('new_kv_cache_pos=None, ', ''),
    ('return lm_logits, *kv_caches_out', 'return lm_logits'),
])
def test_unknown_source_variants_fail_closed(before, after):
    with pytest.raises(ValueError):
        patch_cpro_wrapper(WRAPPER_SOURCE.replace(before, after))


@pytest.fixture
def source_case(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    source = tmp_path / 'private-source'
    monkeypatch.setenv('AI_RUN_DIR', str(run))
    for name in SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(WRAPPER_SOURCE if name == WRAPPER else 'value = 1\n')
    (source / EXPORTER).write_text('torchair.dynamo_export(model=model_wrapper, dynamic=False)\n')
    config = source / 'config/qwen3_5_moe.yaml'
    config.parent.mkdir()
    config.write_text('model_file: /private/original-weights\n')
    # Large source siblings are not copied or inspected.
    (source / 'weights.bin').write_bytes(b'weights must stay outside derived source')
    return source, run / 'original-target-features'


def test_prepare_copies_only_small_source_and_config_reuses_identical_receipt(source_case):
    source, output = source_case
    original = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    receipt = prepare_original_target_source(source, output)
    assert receipt['status'] == 'PREPARED_SOURCE_ONLY'
    assert receipt['output_count'] == 82 and receipt['feature_output_index'] == 81
    assert receipt['feature_shape'] == [1, 'T', 16384]
    assert not list(output.rglob('*.bin'))
    assert not list(output.rglob('*.air'))
    for name in SOURCE_FILES:
        if name != WRAPPER:
            assert (source/name).read_bytes() == (output/'source'/name).read_bytes()
    assert original == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert prepare_original_target_source(source, output) == receipt
    assert before == {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    script = (output/'export.sh').read_text()
    assert 'test ! -e glm_export_air_cpro_40block_55_quant' in script
    assert 'num_hidden_layers=40' in script and 'quant_mode=enable' in script


@pytest.mark.parametrize('change', ['source', 'config', 'output', 'sequence'])
def test_reuse_rejects_changed_identity_without_overwriting(source_case, change):
    source, output = source_case
    prepare_original_target_source(source, output)
    target = {'source': source/'utils.py', 'config': source/'config/qwen3_5_moe.yaml',
              'output': output/'source'/WRAPPER}.get(change)
    if target:
        target.write_text(target.read_text() + '\n# changed\n')
    before = (output/'receipt.json').read_bytes()
    with pytest.raises(ValueError, match='changed'):
        prepare_original_target_source(source, output, sequence_length=54 if change=='sequence' else 55)
    assert (output/'receipt.json').read_bytes() == before


def test_output_ownership_and_partial_evidence_preserved(source_case):
    source, output = source_case
    with pytest.raises(ValueError, match='below AI_RUN_DIR'):
        prepare_original_target_source(source, source/'unsafe-output')
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match='without a completed'):
        prepare_original_target_source(source, output)


def test_unknown_exporter_refused_before_any_output(source_case):
    source, output = source_case
    (source/EXPORTER).write_text('torch.export.export(model)\n')
    with pytest.raises(ValueError, match='unknown original TorchAir'):
        prepare_original_target_source(source, output)
    assert not output.exists()


def test_source_symlink_escape_refused(source_case):
    source, output = source_case
    external = source.parent/'outside.py'
    external.write_text('value = 2\n')
    (source/'utils.py').unlink()
    (source/'utils.py').symlink_to(external)
    with pytest.raises(ValueError, match='bounded local file'):
        prepare_original_target_source(source, output)
    assert not output.exists()

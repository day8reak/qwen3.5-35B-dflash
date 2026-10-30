"""Synthetic regular/sparse files exercise exact bounded lookup, no model calls."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import struct
import time

import pytest

from qwen35_moe_dflash import original_embedding_lookup as lookup


def stamp(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode,
        stat.st_uid, stat.st_gid, stat.st_nlink)


def tables(tmp_path, vocab, hidden, tokens=None):
    paths = tmp_path / 'declared-int8.bin', tmp_path / 'declared-scale.bin'
    for path, width in zip(paths, (hidden, 4)):
        with path.open('wb') as stream: stream.truncate(vocab * width)
    rows, scales = {}, {}
    for token in tokens if tokens is not None else range(vocab):
        rows[token] = bytes((token * 17 + index) % 256 for index in range(hidden))
        scales[token] = struct.pack('<f', (token % 17 + 1) * .001)
        for path, width, raw in zip(paths, (hidden, 4), (rows[token], scales[token])):
            with path.open('r+b') as stream: stream.seek(token * width); stream.write(raw)
    return paths, rows, scales


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    # Test-only geometry keeps full initialization-hash cases tiny. There is no
    # public geometry flag; production always uses248320x2048 and248320x4.
    monkeypatch.setattr(lookup, '_VOCAB', 8); monkeypatch.setattr(lookup, '_HIDDEN', 16)
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    paths, rows, scales = tables(tmp_path, 8, 16)
    return {'paths': paths, 'rows': rows, 'scales': scales, 'tmp': tmp_path}


def test_default_uses_only_unique_selected_rows_preserves_duplicate_order_and_raw_separate_scale(tiny, monkeypatch):
    calls, pread = [], os.pread
    def bounded(fd, width, offset):
        assert width in {16, 4}
        calls.append((width, offset)); return pread(fd, width, offset)
    monkeypatch.setattr(os, 'pread', bounded)
    before = [stamp(path) for path in tiny['paths']]
    reader = lookup.OriginalEmbeddingLookup(*tiny['paths'])
    assert calls == []
    ids = [3, 0, 3, 7, 0]
    result = reader.lookup(ids)
    assert result.token_ids == tuple(ids) and result.input_ids_bytes == struct.pack('<5i', *ids)
    assert result.embedding_bytes == b''.join(tiny['rows'][token] for token in ids)
    assert result.scale_bytes == b''.join(tiny['scales'][token] for token in ids)
    assert result.observation['source_shapes'] == {'input_ids': [1, 5], 'inputs_embeds': [1, 5, 16], 'embed_scale': [1, 5, 1]}
    assert result.observation['selected_table_read_bytes'] == 3 * 20
    assert len(calls) == 6 and calls == [(16, 48), (4, 12), (16, 0), (4, 0), (16, 112), (4, 28)]
    assert result.observation['initialization']['full_table_hash'] == 'NOT_REQUESTED'
    assert result.observation['source_authenticity'] == 'NOT_ATTESTED_BY_ROW_LOOKUP'
    assert before == [stamp(path) for path in tiny['paths']]


def test_production_source_prefix64_and_single_token_shapes_use_exact_full_vocab_bounds(tmp_path, monkeypatch):
    assert (lookup._VOCAB, lookup._HIDDEN) == (248320, 2048)
    paths, rows, scales = tables(tmp_path, 248320, 2048, [0, 269, 248319])
    calls, pread = [], os.pread
    def bounded(fd, width, offset):
        assert width in {2048, 4}
        assert offset <= os.fstat(fd).st_size - width
        calls.append((width, offset)); return pread(fd, width, offset)
    monkeypatch.setattr(os, 'pread', bounded)
    reader = lookup.OriginalEmbeddingLookup(*paths)
    ids = ([269, 0, 248319] * 21) + [269]
    prefix = reader.lookup(ids)
    assert prefix.observation['source_shapes']['inputs_embeds'] == [1, 64, 2048]
    assert prefix.observation['source_shapes']['embed_scale'] == [1, 64, 1]
    assert len(prefix.embedding_bytes) == 131072 and len(prefix.scale_bytes) == 256
    assert prefix.embedding_bytes == b''.join(rows[token] for token in ids)
    assert prefix.observation['selected_table_read_bytes'] == 6156
    one = reader.lookup([248319])
    assert one.embedding_bytes == rows[248319] and one.scale_bytes == scales[248319]
    assert one.observation['source_shapes']['inputs_embeds'] == [1, 1, 2048]
    assert one.observation['source_shapes']['embed_scale'] == [1, 1, 1]
    assert len(calls) == 8  # No cross-call payload cache or hidden full-table read.


@pytest.mark.parametrize('ids', [[], [True], [False], [1.0], ['1'], [-1], [8], [2**64], b'\x01', '1', {1: 2}, {1, 2}, frozenset({1, 2}), None, [1] * 4097])
def test_bad_ids_fail_before_any_lookup_read(tiny, monkeypatch, ids):
    reader = lookup.OriginalEmbeddingLookup(*tiny['paths'])
    monkeypatch.setattr(os, 'pread', lambda *args: pytest.fail('invalid token IDs reached table payload'))
    with pytest.raises(ValueError): reader.lookup(ids)


def test_token_iterator_is_bounded_before_payload_and_retains_explicit_order(tiny, monkeypatch):
    reader = lookup.OriginalEmbeddingLookup(*tiny['paths'])
    assert reader.lookup(iter([7, 3, 7])).token_ids == (7, 3, 7)
    consumed = 0
    def too_many():
        nonlocal consumed
        while True:
            consumed += 1
            yield 3
    monkeypatch.setattr(os, 'pread', lambda *args: pytest.fail('oversized iterator reached table payload'))
    with pytest.raises(ValueError): reader.lookup(too_many())
    assert consumed == 4097


@pytest.mark.parametrize('role', [0, 1])
def test_wrong_table_extent_is_rejected_before_payload_read(tiny, monkeypatch, role):
    tiny['paths'][role].write_bytes(b'x')
    monkeypatch.setattr(os, 'pread', lambda *args: pytest.fail('invalid table extent reached payload'))
    with pytest.raises(ValueError, match='extent differs'):
        lookup.OriginalEmbeddingLookup(*tiny['paths'])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -.01])
def test_corrupt_selected_scale_refuses_returning_rows(tiny, value):
    with tiny['paths'][1].open('r+b') as stream: stream.seek(3 * 4); stream.write(struct.pack('<f', value))
    reader = lookup.OriginalEmbeddingLookup(*tiny['paths'])
    with pytest.raises(ValueError, match='finite and nonnegative'): reader.lookup([3])


@pytest.mark.parametrize('value', [0.0, -0.0])
def test_finite_zero_scale_bytes_are_preserved_without_multiplication(tiny, value):
    raw = struct.pack('<f', value)
    with tiny['paths'][1].open('r+b') as stream: stream.seek(3 * 4); stream.write(raw)
    selected = lookup.OriginalEmbeddingLookup(*tiny['paths']).lookup([3])
    assert selected.scale_bytes == raw and selected.embedding_bytes == tiny['rows'][3]


@pytest.mark.parametrize('fault', ['symlink_file', 'symlink_directory', 'mode', 'ctime', 'inode', 'hardlink'])
def test_live_table_full9_and_non_symlink_path_guards_remain_strict(tiny, fault):
    reader = lookup.OriginalEmbeddingLookup(*tiny['paths'])
    path = tiny['paths'][0]
    if fault == 'symlink_file':
        moved = path.with_suffix('.held'); path.rename(moved); path.symlink_to(moved)
    elif fault == 'symlink_directory':
        folder = tiny['tmp'] / 'tables'; folder.mkdir()
        for current in tiny['paths']: current.rename(folder / current.name)
        for current in tiny['paths']: current.symlink_to(folder / current.name)
    elif fault == 'mode': os.chmod(path, 0o777)
    elif fault == 'ctime':
        before = path.stat().st_ctime_ns; time.sleep(.02); os.chmod(path, path.stat().st_mode & 0o7777)
        assert path.stat().st_ctime_ns != before
    elif fault == 'hardlink': os.link(path, path.with_suffix('.alias'))
    else:
        raw = path.read_bytes(); path.rename(path.with_suffix('.held')); path.write_bytes(raw)
    with pytest.raises(ValueError): reader.lookup([3])


@pytest.mark.parametrize('fault', ['truncate_read', 'table_mutation', 'inode_replacement', 'mode_change'])
def test_truncated_read_or_active_identity_change_never_returns_partial_success(tiny, monkeypatch, fault):
    reader = lookup.OriginalEmbeddingLookup(*tiny['paths'])
    pread, done = os.pread, False
    def changed(fd, width, offset):
        nonlocal done
        result = pread(fd, width, offset)
        if not done:
            done = True
            if fault == 'truncate_read': return result[:-1]
            path = tiny['paths'][0]
            if fault == 'table_mutation':
                time.sleep(.02)
                with path.open('r+b') as stream: stream.write(b'x')
            elif fault == 'mode_change': os.chmod(path, 0o777)
            else:
                saved = path.read_bytes(); path.rename(path.with_suffix('.held')); path.write_bytes(saved)
        return result
    monkeypatch.setattr(os, 'pread', changed)
    with pytest.raises(ValueError): reader.lookup([3, 4])


def test_explicit_hash_pins_read_each_requested_whole_table_once_only_at_initialization(tiny, monkeypatch):
    pins = [hashlib.sha256(path.read_bytes()).hexdigest() for path in tiny['paths']]
    calls, pread = [], os.pread
    def observed(fd, width, offset):
        calls.append((width, offset)); return pread(fd, width, offset)
    monkeypatch.setattr(os, 'pread', observed)
    reader = lookup.OriginalEmbeddingLookup(*tiny['paths'], expected_weight_sha256=pins[0], expected_scale_sha256=pins[1])
    assert calls == [(128, 0), (32, 0)]
    assert reader.initialization['full_table_read_bytes'] == 160
    assert reader.initialization['full_table_hash'] == 'EXPLICIT_REQUESTED_TABLE_PINS_VERIFIED_ONCE'
    reader.lookup([3]); reader.lookup([4])
    assert calls == [(128, 0), (32, 0), (16, 48), (4, 12), (16, 64), (4, 16)]


@pytest.mark.parametrize('pin', ['not-sha', True, 'f' * 64])
def test_invalid_or_wrong_explicit_initialization_pin_fails(tiny, pin):
    with pytest.raises(ValueError):
        lookup.OriginalEmbeddingLookup(*tiny['paths'], expected_weight_sha256=pin)


def input_files(tiny, ids):
    root = tiny['tmp'] / 'provided-inputs'; root.mkdir()
    (root / 'input_ids.bin').write_bytes(struct.pack('<' + str(len(ids)) + 'i', *ids))
    (root / 'inputs_embeds.bin').write_bytes(b''.join(tiny['rows'][token] for token in ids))
    (root / 'embed_scale.bin').write_bytes(b''.join(tiny['scales'][token] for token in ids))
    return root


def inspect(tiny, root, **options):
    return lookup.inspect_original_embedding_inputs(*tiny['paths'], root, 64, tiny['tmp'] / 'inspection', **options)


def test_given64_input_prefix_exact_match_and_duplicate_order_are_described_without_execution_or_mutation(tiny):
    ids = [7, 3, 0, 3] * 16
    root = input_files(tiny, ids)
    before = {path: stamp(path) for path in [*tiny['paths'], *root.iterdir()]}
    report = inspect(tiny, root)
    assert report['status'] == 'ORIGINAL_EMBEDDING_INPUT_ROWS_MATCHED', report
    assert report['token_ids'] == ids and report['matched_position_count'] == 64 and not report['mismatch_positions']
    assert report['lookup_observation']['selected_table_read_bytes'] == 60
    assert len(report['active_source_identities']) == 5
    assert report['model_execution_binding'].endswith('GIVEN_INPUT_DIRECTORY_ONLY')
    assert report['numerical_validation'] == 'NOT_EVALUATED' and report['replacement_input_generation'] == 'NOT_RUN'
    assert before == {path: stamp(path) for path in before}
    assert {path.name for path in (tiny['tmp'] / 'inspection').iterdir()} == {'report.json'}


def test_input_embedding_and_scale_mismatches_keep_separate_positions_and_do_not_generate_inputs(tiny):
    root = input_files(tiny, [3, 4] * 32)
    with (root / 'inputs_embeds.bin').open('r+b') as stream: stream.seek(7 * 16); stream.write(b'x')
    with (root / 'embed_scale.bin').open('r+b') as stream: stream.seek(9 * 4); stream.write(struct.pack('<f', .75))
    result = inspect(tiny, root)
    assert result['status'] == 'ORIGINAL_EMBEDDING_INPUT_ROWS_MISMATCH'
    assert result['mismatch_positions'] == [7, 9]
    assert result['embedding_mismatch_positions'] == [7] and result['scale_mismatch_positions'] == [9]
    assert result['matched_position_count'] == 62


@pytest.mark.parametrize('file', ['input_ids.bin', 'inputs_embeds.bin', 'embed_scale.bin'])
def test_wrong_input_extents_are_not_truncated_or_reinterpreted_as_another_sequence(tiny, monkeypatch, file):
    root = input_files(tiny, [3] * 64)
    with (root / file).open('ab') as stream: stream.write(b'x')
    monkeypatch.setattr(os, 'pread', lambda *args: pytest.fail('bad input extents reached table rows'))
    result = inspect(tiny, root)
    assert result['status'] == 'FAILED' and 'expected_bytes' in result['error']


def test_active_original_input_change_during_row_lookup_is_rejected(tiny, monkeypatch):
    root = input_files(tiny, [3] * 64)
    original = lookup.OriginalEmbeddingLookup.lookup
    def changed(self, ids):
        result = original(self, ids)
        path = root / 'input_ids.bin'; os.chmod(path, 0o777)
        return result
    monkeypatch.setattr(lookup.OriginalEmbeddingLookup, 'lookup', changed)
    result = inspect(tiny, root)
    assert result['status'] == 'FAILED' and 'changed' in result['error']


def test_thin_cli_has_selected_rows_default_and_explicit_whole_hash_flags(tiny, capsys):
    root = input_files(tiny, [3, 4] * 32)
    module = runpy.run_path(str(Path(__file__).parents[1] / 'targets/cpro/scripts/inspect_original_embedding_inputs.py'))
    output = tiny['tmp'] / 'cli-output'
    assert module['main'](['--weight-table', str(tiny['paths'][0]), '--scale-table', str(tiny['paths'][1]),
        '--input-dir', str(root), '--sequence-length', '64', '--output', str(output)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown['status'] == 'ORIGINAL_EMBEDDING_INPUT_ROWS_MATCHED' and shown['selected_table_read_bytes'] == 40
    assert shown['full_table_hash'] == 'NOT_REQUESTED' and shown['device_execution'] == 'NOT_RUN'
    pins = [hashlib.sha256(path.read_bytes()).hexdigest() for path in tiny['paths']]
    assert module['main'](['--weight-table', str(tiny['paths'][0]), '--scale-table', str(tiny['paths'][1]),
        '--input-dir', str(root), '--sequence-length', '64', '--output', str(tiny['tmp'] / 'cli-hashed'),
        '--expected-weight-sha256', pins[0], '--expected-scale-sha256', pins[1]]) == 0
    assert json.loads(capsys.readouterr().out)['full_table_hash'] == 'EXPLICIT_REQUESTED_TABLE_PINS_VERIFIED_ONCE'

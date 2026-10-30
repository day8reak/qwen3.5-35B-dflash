"""Bounded exact lookup of declared INT8 embeddings and separate FP32 scales.

Default lookup reads only requested rows. Explicit expected SHA256 parameters
request one full-table verification at initialization, never per token. Source
paths and hashes are caller assertions, not proof of original model authenticity.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import islice
import math
from numbers import Integral
import os
from pathlib import Path
import re
import struct

from .artifacts import output_path, save_json
from .target_compile_failure import _safe
from .target_prefill_smoke_plan import _identity


_VOCAB = 248320
_HIDDEN = 2048
_MAX_TOKENS = 4096
_HASH_CHUNK = 1 << 20


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _current(path):
    path, stamp = _safe(path)
    identity = _identity(path.stat())
    if [identity[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
        raise ValueError('file changed before its current full identity snapshot: ' + str(path))
    return path, identity


def _tokens(values, vocab):
    if isinstance(values, (str, bytes, bytearray, memoryview, Mapping, set, frozenset)):
        raise ValueError('token IDs must be explicit ordered integers, not encoded bytes, text, a mapping or a set')
    try:
        values = list(islice(values, _MAX_TOKENS + 1))
    except TypeError as error:
        raise ValueError('token IDs must be an ordered sequence of integers') from error
    if (not 1 <= len(values) <= _MAX_TOKENS or any(not isinstance(token, Integral)
            or isinstance(token, bool) or not 0 <= int(token) < vocab for token in values)):
        raise ValueError('token IDs require1..4096 strictly integral in-vocabulary entries; booleans are invalid')
    return tuple(int(token) for token in values)


def _unchanged(path, expected, fd=None):
    current = _current(path)[1]
    if current != expected or fd is not None and _identity(os.fstat(fd)) != expected:
        raise ValueError('declared embedding source changed since its current baseline: ' + str(path))


@dataclass(frozen=True)
class OriginalEmbeddingRows:
    """Source-layout bytes: [1,S,2048] INT8 and [1,S,1] FP32 little endian."""
    token_ids: tuple[int, ...]
    embedding_bytes: bytes
    scale_bytes: bytes
    observation: dict

    @property
    def input_ids_bytes(self):
        return struct.pack('<' + str(len(self.token_ids)) + 'i', *self.token_ids)


class OriginalEmbeddingLookup:
    """Fixed original-vocabulary row layout, with strict live file identities.

    The object stores no table payload or cross-call row cache. Each lookup
    opens current files and reads one row per unique ID, then restores caller
    order and duplicates without numeric conversion or multiplying scales.
    """
    def __init__(self, weight_table, scale_table, *, expected_weight_sha256=None, expected_scale_sha256=None):
        self.vocab_size, self.hidden_size = _VOCAB, _HIDDEN
        self._tables = []
        for role, path, width, expected in (
                ('weight', weight_table, self.hidden_size, expected_weight_sha256),
                ('scale', scale_table, 4, expected_scale_sha256)):
            if expected is not None and (not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None):
                raise ValueError('expected ' + role + ' SHA256 must be explicit64 lowercase hexadecimal characters')
            path, identity = _current(path)
            if identity['bytes'] != self.vocab_size * width:
                raise ValueError('declared ' + role + ' table extent differs from the fixed original row layout: '
                    + str(path) + '; bytes=' + str(identity['bytes']) + '; expected_bytes=' + str(self.vocab_size * width))
            self._tables.append({'role': role, 'path': str(path), 'bytes': identity['bytes'], 'identity': identity,
                'row_bytes': width, 'expected_sha256': expected})
        if len({(row['identity']['device'], row['identity']['inode']) for row in self._tables}) != 2:
            raise ValueError('embedding and scale tables must be separate regular files')
        self.initialization = {'full_table_hash': 'NOT_REQUESTED', 'full_table_read_bytes': 0, 'tables': []}
        with self._opened() as fds:
            for table, fd in zip(self._tables, fds):
                observed = {key: deepcopy(table[key]) for key in ('role', 'path', 'bytes', 'identity', 'row_bytes')}
                observed.update(full_table_hash='NOT_REQUESTED', full_table_read_bytes=0)
                expected = table['expected_sha256']
                if expected is not None:
                    digest, offset = hashlib.sha256(), 0
                    while offset < table['bytes']:
                        width = min(_HASH_CHUNK, table['bytes'] - offset)
                        raw = os.pread(fd, width, offset)
                        if len(raw) != width:
                            raise ValueError('declared table truncated during explicit initialization SHA verification')
                        digest.update(raw); offset += len(raw)
                    _unchanged(table['path'], table['identity'], fd)
                    if digest.hexdigest() != expected:
                        raise ValueError('explicit initialization ' + table['role'] + ' table SHA256 does not match the expected pin')
                    observed.update(full_table_hash='MATCHED_EXPLICIT_EXPECTED_SHA256', sha256=expected,
                        full_table_read_bytes=offset)
                    self.initialization['full_table_read_bytes'] += offset
                self.initialization['tables'].append(observed)
        if any(table['expected_sha256'] is not None for table in self._tables):
            self.initialization['full_table_hash'] = 'EXPLICIT_REQUESTED_TABLE_PINS_VERIFIED_ONCE'
        self.initialization.update(table_authenticity='CALLER_PATHS_AND_OPTIONAL_HASH_PINS_ONLY_ORIGIN_NOT_ATTESTED',
            model_execution='NOT_RUN', source_preservation='CURRENT_TABLE_FULL9_IDENTITIES_UNCHANGED')

    @contextmanager
    def _opened(self):
        with ExitStack() as stack:
            fds = []
            for table in self._tables:
                _unchanged(table['path'], table['identity'])
                fd = os.open(table['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                stack.callback(os.close, fd)
                _unchanged(table['path'], table['identity'], fd)
                fds.append(fd)
            try:
                yield fds
            finally:
                for table, fd in zip(self._tables, fds):
                    _unchanged(table['path'], table['identity'], fd)

    def lookup(self, token_ids):
        tokens = _tokens(token_ids, self.vocab_size)
        unique = tuple(dict.fromkeys(tokens))
        selected, count = {}, 0
        with self._opened() as fds:
            for token in unique:
                parts = []
                for table, fd in zip(self._tables, fds):
                    width = table['row_bytes']
                    offset = token * width
                    if not 0 <= offset <= table['bytes'] - width:
                        raise ValueError('selected original row would escape its declared table')
                    raw = os.pread(fd, width, offset)
                    if len(raw) != width:
                        raise ValueError('selected original ' + table['role'] + ' row was truncated')
                    parts.append(raw); count += len(raw)
                factor, = struct.unpack('<f', parts[1])
                if not math.isfinite(factor) or factor < 0:
                    raise ValueError('selected original scale must be finite and nonnegative')
                selected[token] = parts
        observation = {'status': 'DECLARED_ORIGINAL_ROWS_READ', 'sequence_length': len(tokens),
            'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size,
            'source_shapes': {'input_ids': [1, len(tokens)], 'inputs_embeds': [1, len(tokens), self.hidden_size],
                'embed_scale': [1, len(tokens), 1]},
            'source_dtypes': {'input_ids': 'INT32', 'inputs_embeds': 'INT8', 'embed_scale': 'FLOAT32'}, 'byte_order': 'little',
            'token_count': len(tokens), 'unique_token_count': len(unique), 'duplicate_order': 'PRESERVED_EXACTLY',
            'selected_table_read_bytes': count, 'maximum_selected_table_read_bytes': len(unique) * (self.hidden_size + 4),
            'selected_rows': [{'token_id': token, 'embedding_sha256': _sha(selected[token][0]),
                'scale_sha256': _sha(selected[token][1])} for token in unique],
            'tables': [{key: deepcopy(table[key]) for key in ('role', 'path', 'bytes', 'identity', 'row_bytes')} for table in self._tables],
            'initialization': deepcopy(self.initialization), 'source_preservation': 'CURRENT_TABLE_FULL9_IDENTITIES_UNCHANGED',
            'numerical_validation': 'NOT_EVALUATED', 'source_authenticity': 'NOT_ATTESTED_BY_ROW_LOOKUP',
            'scope': 'Raw selected INT8 row bytes with separate original FP32 scale bytes; no requantization, scale multiplication, tokenizer/history proof, compiled ABI inference or device execution'}
        return OriginalEmbeddingRows(tokens, b''.join(selected[token][0] for token in tokens),
            b''.join(selected[token][1] for token in tokens), observation)


def _input_bytes(path, size, snapshots):
    path, before = _current(path)
    if before['bytes'] != size:
        raise ValueError('original input extent differs from its exact sequence/source layout: '
            + str(path) + '; bytes=' + str(before['bytes']) + '; expected_bytes=' + str(size))
    if str(path) in snapshots and snapshots[str(path)] != before:
        raise ValueError('original input changed within this inspection')
    snapshots[str(path)] = before
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        _unchanged(path, before, stream.fileno())
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError('original input truncated during bounded read')
        _unchanged(path, before, stream.fileno())
    return raw, {'path': str(path), 'bytes': size, 'sha256': _sha(raw), 'identity': before}


def inspect_original_embedding_inputs(weight_table, scale_table, input_dir, sequence_length, output, *,
        expected_weight_sha256=None, expected_scale_sha256=None):
    """Compare given S1/S64 ID/embedding/scale files; do not create replacements."""
    out = output_path(output); out.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'artifact_kind': 'qwen35-original-embedding-input-row-inspection',
        'status': 'FAILED', 'output': str(out), 'device_execution': 'NOT_RUN', 'model_compilation': 'NOT_RUN',
        'input_mutation': 'NONE', 'replacement_input_generation': 'NOT_RUN', 'numerical_validation': 'NOT_EVALUATED',
        'source_authenticity': 'NOT_ATTESTED', 'model_execution_binding': 'NOT_EVALUATED_GIVEN_INPUT_DIRECTORY_ONLY'}
    snapshots = {}
    try:
        if type(sequence_length) is not int or sequence_length not in {1, 64}:
            raise ValueError('original input inspection requires an explicit source sequence length of1 or64')
        root, _ = _safe(input_dir, directory=True)
        sizes = {'input_ids.bin': sequence_length * 4, 'inputs_embeds.bin': sequence_length * _HIDDEN,
            'embed_scale.bin': sequence_length * 4}
        # Establish all three current baselines before the first payload read.
        for name, size in sizes.items():
            path, current = _current(root / name)
            if current['bytes'] != size:
                raise ValueError('original input extent differs from its exact sequence/source layout: '
                    + str(path) + '; bytes=' + str(current['bytes']) + '; expected_bytes=' + str(size))
            snapshots[str(path)] = current
        payloads, sources = {}, []
        for name, size in sizes.items():
            payloads[name], identity = _input_bytes(root / name, size, snapshots)
            sources.append(identity)
        tokens = _tokens(struct.unpack('<' + str(sequence_length) + 'i', payloads['input_ids.bin']), _VOCAB)
        lookup = OriginalEmbeddingLookup(weight_table, scale_table,
            expected_weight_sha256=expected_weight_sha256, expected_scale_sha256=expected_scale_sha256)
        selected = lookup.lookup(tokens)
        embedding_bad, scale_bad = [], []
        positions = []
        for index, token in enumerate(tokens):
            embedding = payloads['inputs_embeds.bin'][index * _HIDDEN:(index + 1) * _HIDDEN]
            scale = payloads['embed_scale.bin'][index * 4:(index + 1) * 4]
            expected_embedding = selected.embedding_bytes[index * _HIDDEN:(index + 1) * _HIDDEN]
            expected_scale = selected.scale_bytes[index * 4:(index + 1) * 4]
            ematch, smatch = embedding == expected_embedding, scale == expected_scale
            if not ematch: embedding_bad.append(index)
            if not smatch: scale_bad.append(index)
            positions.append({'position': index, 'token_id': token, 'embedding_match': ematch, 'scale_match': smatch,
                'input_embedding_sha256': _sha(embedding), 'table_embedding_sha256': _sha(expected_embedding),
                'input_scale_sha256': _sha(scale), 'table_scale_sha256': _sha(expected_scale)})
        for path, identity in snapshots.items(): _unchanged(path, identity)
        for table in selected.observation['tables']: _unchanged(table['path'], table['identity'])
        bad = sorted(set(embedding_bad) | set(scale_bad))
        result.update(status='ORIGINAL_EMBEDDING_INPUT_ROWS_MATCHED' if not bad else 'ORIGINAL_EMBEDDING_INPUT_ROWS_MISMATCH',
            input_dir=str(root), sequence_length=sequence_length, token_ids=list(tokens), position_count=sequence_length,
            matched_position_count=sequence_length - len(bad), mismatch_positions=bad,
            embedding_mismatch_positions=embedding_bad, scale_mismatch_positions=scale_bad,
            position_index_base=0, position_observations=positions, source_inputs=sources,
            lookup_observation=selected.observation, source_preservation='CURRENT_INPUT_AND_TABLE_FULL9_IDENTITIES_UNCHANGED',
            active_source_identities=[{'path': path, 'identity': deepcopy(identity)} for path, identity in snapshots.items()]
                + [{'path': table['path'], 'identity': deepcopy(table['identity'])} for table in selected.observation['tables']],
            scope='Exact row correspondence for only these three explicitly supplied input files and declared tables; no successful-execution binding, table-origin authenticity, control/history, numerical accuracy or language-quality acceptance')
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, struct.error) as error:
        result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    save_json(out / 'report.json', result)
    return result

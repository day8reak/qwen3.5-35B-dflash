"""Execute the original S64 full-prefix drafter inside the serial controller.

The OMC is pinned once in its original bundle, never copied on the host. Each
call owns three padded inputs, unchanged native ABI and a step plan. This is a
real execution adapter; successful transport/native receipts do not establish
numerical accuracy, acceptance quality or acceleration.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from itertools import islice
import json
from pathlib import Path
import struct

import numpy as np

from .artifacts import output_path, save_json
from .draft_board import _io, _wire
from .hashing import sha256_file
from .incremental_smoke import (ModelIdentity, CallEvidence, DraftObservation,
    _ids, feature_sha256, prefix_sha256)
from .real_feature_draft import _bundle, _write
from .target_compile_failure import _read_small, _relative, _safe
from .target_prefill_smoke_plan import _identity


def _current(path):
    path, stamp = _safe(path)
    identity = _identity(path.stat())
    if [identity[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
        raise ValueError('draft source changed during full identity capture: ' + str(path))
    return path, identity


def _small(path, maximum=8 << 20):
    path, before = _current(path)
    raw, record = _read_small(path, maximum)
    if _current(path)[1] != before or [before[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != record['stat']:
        raise ValueError('draft small source changed during read: ' + str(path))
    return raw, {**record, 'identity': before}


class ExistingDraftSmokeBackend:
    """Original three-input, two-output S64 OMC; no context truncation/rebase."""
    def __init__(self, bundle_dir, output_dir, *, runtime_version, firmware_id,
                 timeout_seconds=1800, lock_wait_seconds=1800, _host_test=False):
        if type(_host_test) is not bool:
            raise ValueError('host-test scope must be explicit')
        for value in (runtime_version, firmware_id):
            if not isinstance(value, str) or not value or any(char in value for char in '\0\r\n'):
                raise ValueError('runtime and firmware identities must be explicit single-line strings')
        bundle, _ = _safe(bundle_dir, directory=True)
        out = output_path(output_dir)
        if out == bundle or bundle in out.parents:
            raise ValueError('backend output must be outside the immutable source bundle')
        raw, bundle_id = _small(bundle / 'manifest.json')
        selected = json.loads(raw)
        self._source_identities = {}
        for path in (bundle / 'manifest.json', *[_relative(bundle, selected[field]['path'])
                for field in ('model', 'draft_abi', 'weight_packing', 'step_plan')]):
            checked, identity = _current(path)
            self._source_identities[str(checked)] = identity
        observed = {'status': 'MANIFEST_OBSERVED_ONLY', 'manifest': {key: bundle_id[key]
            for key in ('path', 'bytes', 'sha256', 'stat')}}
        source, model_source, abi_raw, source_bindings = _bundle(bundle, observed)
        self.bundle, self.out, self._source = bundle, out, source
        self._model_source, self._abi_raw = model_source, abi_raw
        self._source_bindings, self._bundle_identity = source_bindings, bundle_id
        self.runtime_version, self.firmware_id = runtime_version, firmware_id
        self.timeout_seconds, self.lock_wait_seconds = timeout_seconds, lock_wait_seconds
        self._host_test = _host_test
        self._runner = None
        self._steps, self._calls, self._seen_receipts = {}, [], set()
        self.native_call_count, self._failed = 0, False
        self.identity = ModelIdentity('draft', 6, 248320, 16384, source['model']['sha256'],
            bundle_id['sha256'], 'host-fake-hiai' if _host_test else 'device-hiai')
        self._check_sources()
        out.mkdir(parents=True, exist_ok=False)
        save_json(out / 'session.json', {'schema_version': 1, 'artifact_kind': 'qwen35-existing-draft-smoke-session',
            'status': 'ORIGINAL_MODEL_AND_ABI_PINNED', 'source_bundle': str(bundle), 'source_manifest': bundle_id,
            'model_source': model_source, 'model_sha256': self.identity.model_sha256,
            'source_full_identities': deepcopy(self._source_identities), 'evidence_kind': self.identity.evidence_kind,
            'sequence_capacity': 64, 'draft_width': 15, 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
            'host_model_copy_bytes': 0, 'host_model_links_created': 0, 'model_hash_policy': 'FRESH_SHA256_ONCE_THEN_STRICT_FULL9',
            'runtime_version': runtime_version, 'firmware_id': firmware_id,
            'scope': 'Complete original S64 recompute prefix only; no truncation, position rebase, numerical or acceptance pass'})

    def _check_sources(self):
        for path, identity in self._source_identities.items():
            if _current(path)[1] != identity:
                raise ValueError('pinned original draft source changed: ' + path)
        if self._runner is not None:
            for row in [*self._runner['pins'], *self._runner.get('library_pins', [])]:
                if _current(row['path'])[1] != row['identity']:
                    raise ValueError('pinned linked runner or library changed during the draft session')

    def preflight(self):
        """Host-only model/ABI pin summary; runner builds lazily on first call."""
        self._check_sources()
        return {'artifact_kind': 'qwen35-existing-draft-backend-preflight', 'status': 'HOST_SOURCE_PINNED',
            'source_bundle': str(self.bundle), 'source_manifest_sha256': self._bundle_identity['sha256'],
            'model_source': deepcopy(self._model_source), 'model_identity': self.identity.__dict__.copy(),
            'sequence_capacity': 64, 'complete_prefix_required': True, 'proposal_outputs': 15,
            'execution_allowed': not self._failed,
            'runner': 'PINNED_LINKED_RUNNER' if self._runner else 'BUILD_CURRENT_RUNNER_ON_FIRST_PROPOSE',
            'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN', 'host_model_copy_bytes': 0}

    @property
    def runner_is_pinned(self):
        return self._runner is not None

    @property
    def runner_build_directory(self):
        """The original LINKED build; later attempts never relabel its receipts."""
        self._check_sources()
        if self._runner is None:
            raise ValueError('draft runner has not been built and pinned')
        return Path(self._runner['build_directory'])

    def runner_for_attempt(self, output, timeout, builder):
        self._check_sources()
        if self._runner is None:
            path = Path(builder(output, timeout))
            _, identity = _current(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if _current(path)[1] != identity:
                raise ValueError('linked runner changed during its initial pin')
            report = Path(output) / 'native/result.json'
            raw, report_id = _small(report)
            linked = json.loads(raw)
            command = report.parent / 'command.json'
            _, command_id = _small(command)
            if (linked.get('status') != 'LINKED' or linked.get('binary_sha256') != digest
                    or linked.get('command_sha256') != command_id['sha256']):
                raise ValueError('existing-draft session requires the actual LINKED runner build receipt')
            self._runner = {'path': str(path), 'sha256': digest, 'build_directory': str(report.parent),
                'build_result': report_id, 'build_command': command_id,
                'pins': [{'path': str(path), 'identity': identity}, {'path': str(report), 'identity': report_id['identity']},
                    {'path': str(command), 'identity': command_id['identity']}]}
        self._check_sources()
        return Path(self._runner['path'])

    def pin_runner_libraries(self, discovery):
        """Keep the first actual discovery tied to that immutable build and SOs."""
        self._check_sources()
        if (self._runner is None or Path(discovery['build_dir']).resolve() != self.runner_build_directory.resolve()
                or discovery['metadata'].get('command_sha256') != self._runner['build_command']['sha256']):
            raise ValueError('draft library discovery differs from the pinned original runner build')
        rows, pins = [], []
        metadata = {row['name']: row for row in discovery['metadata']['libraries']}
        for selected in discovery['libraries']:
            selected = Path(selected)
            source, identity = _current(selected.resolve(strict=True))
            proof = metadata.get(selected.name, {})
            if (Path(proof.get('path', '')).absolute() != selected.absolute()
                    or sha256_file(source) != proof.get('sha256') or _current(source)[1] != identity):
                raise ValueError('draft library differs from its fresh discovery SHA or identity')
            rows.append({'name': selected.name, 'selected_path': str(selected.absolute()),
                'path': str(source), 'sha256': proof['sha256'], 'identity': identity})
            pins.append({'path': str(source), 'identity': identity})
        rows.sort(key=lambda row: row['name'])
        if len(rows) != len(metadata) or len({row['name'] for row in rows}) != len(rows):
            raise ValueError('draft library discovery is incomplete or ambiguous')
        if 'libraries' in self._runner and rows != self._runner['libraries']:
            raise ValueError('draft library selection changed after the first runner build')
        self._runner['libraries'], self._runner['library_pins'] = rows, pins
        self._check_sources()

    def _prepare(self, prefix, features, out):
        out.mkdir(parents=True, exist_ok=False)
        count = len(prefix)
        padded = np.zeros((1, 64, 16384), dtype='<f2'); padded[:, :count] = features
        ids = np.zeros((1, 64), dtype='<i4'); ids[:, :count] = prefix
        mask = np.zeros((1, 64), dtype='<i4'); mask[:, :count] = 1
        inputs, _ = _io(64, direct_spec=self._source['direct_spec'])
        manifest = {'schema_version': 1, 'artifact_kind': 'qwen35-existing-draft-smoke-step', 'status': 'PREPARED',
            'output': str(out), 'sequence_length': 64, 'past_tokens': 0,
            'platform': self._source['platform'], 'ddk_id': self._source['ddk_id'],
            'runtime_version': self.runtime_version, 'firmware_id': self.firmware_id,
            'draft_config': self._source['draft_config'], 'direct_spec': self._source['direct_spec'],
            'weight_storage': 'embedded-stream', 'model': self._source['model'], 'model_source': self._model_source,
            'source_bundle': str(self.bundle), 'model_upload_source': self._model_source['path'],
            'source_bindings': self._source_bindings, 'inputs': [], 'outputs': self._source['outputs'], 'upload_payloads': [],
            'logical_prefix_length': count, 'logical_prefix_sha256': prefix_sha256(prefix),
            'logical_feature_sha256': feature_sha256(features),
            'token_provenance': {'valid_length': count, 'anchor_position': count - 1, 'anchor_token_id': prefix[-1],
                'prefix_policy': 'COMPLETE_CONTROLLER_PREFIX_RIGHT_PADDED_NO_TRUNCATION_OR_REBASE'},
            'feature_provenance': {'logical_shape': [1, count, 16384], 'wire_shape': [1, 64, 16384],
                'transformation': 'BYTE_EXACT_PREFIX_THEN_ZERO_PADDING_NO_CLIPPING_OR_NORMALIZATION'},
            'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN', 'numerical_validation': 'NOT_EVALUATED',
            'host_model_copy_bytes': 0, 'evidence_kind': self.identity.evidence_kind}
        for desc, data in zip(inputs, (padded.tobytes(), ids.tobytes(), mask.tobytes())):
            record, upload = _write(out, f"inputs/input_{desc['index']:04d}.bin", data, desc['name'])
            manifest['inputs'].append({**desc, **record}); manifest['upload_payloads'].append(upload)
        manifest['draft_abi'], upload = _write(out, 'draft-abi.json', self._abi_raw, 'draft_abi')
        manifest['upload_payloads'].append(upload)
        manifest['step_plan'], upload = _write(out, 'step.tsv', _wire(manifest), 'step_plan')
        manifest['upload_payloads'].append(upload)
        save_json(out / 'manifest.json', manifest)
        pins = []
        for path in [out / 'manifest.json', *[out / row['path'] for row in manifest['upload_payloads']]]:
            _, row = _small(path); pins.append(row)
        self._steps[str(out)] = {'manifest': deepcopy(manifest), 'pins': pins}
        self.validate_step(out)
        return manifest

    def validate_step(self, prepared_dir):
        if self._failed:
            raise RuntimeError('failed draft session cannot validate another executable step')
        self._check_sources()
        root, _ = _safe(prepared_dir, directory=True)
        selected = self._steps.get(str(root))
        if selected is None:
            raise ValueError('prepared draft step does not belong to this immutable session')
        for row in selected['pins']:
            raw, current = _small(row['path'])
            if any(current[key] != row[key] for key in ('bytes', 'sha256', 'identity')):
                raise ValueError('prepared original-draft step bytes or full identity changed')
        manifest = selected['manifest']
        if (root / 'model').exists() or (root / 'model').is_symlink():
            raise ValueError('smoke step must not contain a copied or linked local model')
        raw = [(root / row['path']).read_bytes() for row in manifest['inputs']]
        count = manifest['logical_prefix_length']
        size = count * 16384 * 2
        ids = struct.unpack('<64i', raw[1])
        if (hashlib.sha256(raw[0][:size]).hexdigest() != manifest['logical_feature_sha256']
                or raw[0][size:] != bytes((64 - count) * 16384 * 2)
                or prefix_sha256(ids[:count]) != manifest['logical_prefix_sha256'] or any(ids[count:])
                or raw[2] != struct.pack('<64i', *([1] * count + [0] * (64 - count)))
                or (root / manifest['draft_abi']['path']).read_bytes() != self._abi_raw
                or (root / manifest['step_plan']['path']).read_bytes() != _wire(manifest)):
            raise ValueError('prepared bytes do not preserve the complete logical prefix and right padding')
        for row in selected['pins']:
            if _current(row['path'])[1] != row['identity']:
                raise ValueError('prepared draft step changed within validation')
        self._check_sources()
        return deepcopy(manifest)

    def propose(self, prefix, features):
        if self._failed:
            raise RuntimeError('failed draft session cannot execute another native attempt; preserve the original evidence')
        try:
            if isinstance(prefix, (str, bytes, Mapping, set, frozenset)):
                raise ValueError('draft prefix must be an ordered complete token sequence')
            tokens = tuple(_ids(islice(prefix, 65), 248320, 'draft prefix'))
            if not 1 <= len(tokens) <= 64:
                raise ValueError('original S64 draft requires the complete prefix in1..64 rows; truncation/rebasing is forbidden')
            if (not isinstance(features, np.ndarray) or features.dtype != np.dtype('<f2')
                    or features.shape != (1, len(tokens), 16384) or not features.flags.c_contiguous):
                raise ValueError('draft requires byte-exact contiguous FP16 features for the complete prefix')
            digest = feature_sha256(features)
            frozen = features.copy(order='C')
            if feature_sha256(frozen) != digest or feature_sha256(features) != digest:
                raise ValueError('target feature prefix changed while preparing draft inputs')
            self._check_sources()
            call = self.out / ('call-%04d' % (len(self._calls) + 1)); call.mkdir()
            entry = {'output': str(call), 'validated': False}
            self._calls.append(entry)
            manifest = self._prepare(tokens, frozen, call / 'prepared')
            from .real_feature_draft_board import execute_existing_draft_smoke, _observe_native
            execution = execute_existing_draft_smoke(call / 'prepared', call / 'execution', session=self,
                timeout_seconds=self.timeout_seconds, lock_wait_seconds=self.lock_wait_seconds)
            entry['execution'] = execution
            if (execution.get('status') != 'REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED'
                    or execution.get('native_status') != 'EXECUTED' or execution.get('remote_exitcode') != 0
                    or type(execution.get('remote_exitcode')) is not int
                    or execution.get('retrieval_errors') or execution.get('lock_state') != 'NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT'):
                raise ValueError('actual original-draft native execution was not completely validated: ' + str(execution.get('error', execution.get('evidence_error', execution.get('status')))))
            native_path = call / 'execution/results/result.json'
            raw, native_id = _small(native_path)
            if execution.get('retrieved_sha256', {}).get('results/result.json') != native_id['sha256']:
                raise ValueError('native draft receipt differs from its actual retrieved SHA')
            native = json.loads(raw)
            output_pins = [(str(native_path), native_id['identity'])]
            for name in ('output_0000.bin', 'output_0001.bin'):
                path = native_path.parent / name
                output_pins.append((str(path), _current(path)[1]))
            observed = _observe_native(manifest, native, native_path.parent, runtime=self.runtime_version,
                firmware=self.firmware_id, pid=execution['native_process_id'])
            if observed.get('native_status') != 'EXECUTED' or native_id['sha256'] in self._seen_receipts:
                raise ValueError('native draft receipt is failed or reused')
            for row in native['outputs']:
                _, proof = _small(native_path.parent / row['path'])
                if any(proof[key] != row[key] for key in ('bytes', 'sha256')):
                    raise ValueError('retrieved native output differs from its native receipt')
            self.validate_step(call / 'prepared')
            proposals = tuple(observed['draft_top1'][0])
            receipt = CallEvidence('existing-s64-' + str(len(self._calls)) + '-' + native_id['sha256'][:24],
                self.identity.model_sha256, native_id['sha256'], prefix_sha256(tokens), self.identity.evidence_kind,
                feature_input_sha256=digest)
            save_json(call / 'proposal.json', {'status': 'ACTUAL_NATIVE_PROPOSALS_OBSERVED_UNVALIDATED',
                'evidence_kind': self.identity.evidence_kind, 'test_only': self._host_test,
                'logical_prefix_length': len(tokens), 'logical_prefix_sha256': prefix_sha256(tokens),
                'logical_feature_sha256': digest, 'wire_feature_sha256': manifest['inputs'][0]['sha256'],
                'model_sha256': self.identity.model_sha256, 'native_receipt': native_id,
                'draft_top1': list(proposals), 'proposal_count': 15, 'host_model_copy_bytes': 0,
                'numerical_validation': 'NOT_EVALUATED', 'model_compilation': 'NOT_RUN'})
            self.validate_step(call / 'prepared')
            for path, identity in output_pins:
                if _current(path)[1] != identity:
                    raise ValueError('native draft receipt/output changed before proposal return')
            self._seen_receipts.add(native_id['sha256']); self.native_call_count += 1
            entry.update(validated=True, receipt_path=str(native_path), receipt_sha256=native_id['sha256'])
            return DraftObservation(proposals, len(tokens), prefix_sha256(tokens), digest, receipt)
        except Exception:
            self._failed = True
            raise

    def execution_summary(self):
        return {'native_calls': self.native_call_count,
            'native_calls_scope': 'SUCCESSFULLY_VALIDATED_UNIQUE_NATIVE_RECEIPTS',
            'proposal_requests': len(self._calls),
            'native_call_attempts': sum(bool(row.get('execution', {}).get('native_process_id'))
                or row.get('execution', {}).get('stage') == 'execute' for row in self._calls),
            'known_executed_native_calls': sum(row.get('execution', {}).get('native_status') == 'EXECUTED' for row in self._calls),
            'receipt_paths': [row['receipt_path'] for row in self._calls if row.get('validated')],
            'validated_native_calls': bool(self.native_call_count) and not self._failed and all(row['validated'] for row in self._calls),
            'evidence_kind': self.identity.evidence_kind, 'model_sha256': self.identity.model_sha256,
            'source_bundle_manifest_sha256': self._bundle_identity['sha256'], 'runner_build_count': int(self._runner is not None),
            'host_model_copy_bytes': 0, 'model_compilation': 'NOT_RUN', 'numerical_validation': 'NOT_EVALUATED',
            'capacity_policy': 'COMPLETE_PREFIX_LE64_NO_TRUNCATION_OR_REBASE',
            'calls': [{**{key: row[key] for key in ('output', 'validated', 'receipt_path', 'receipt_sha256') if key in row},
                'execution_report': str(Path(row['output']) / 'execution/result.json'),
                'execution_status': row.get('execution', {}).get('status'),
                'native_process_id': row.get('execution', {}).get('native_process_id')} for row in self._calls]}


def validate_existing_draft_smoke_step(prepared_dir, session):
    if not isinstance(session, ExistingDraftSmokeBackend):
        raise TypeError('an explicit pinned ExistingDraftSmokeBackend session is required')
    return session.validate_step(prepared_dir)

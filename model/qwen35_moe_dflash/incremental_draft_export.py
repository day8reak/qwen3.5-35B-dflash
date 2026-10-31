"""Export a new six-layer cached Draft AIR, reusing the original 71 weights."""
from __future__ import annotations

import json
import uuid

from .artifacts import output_path, save_json
from .config import Qwen35DFlashConfig
from .draft_capacity_export import _bind_source_job, _export_source, _stable
from .ge_export import configurations
from .ge_incremental_draft import _validate
from .target_compile_failure import _read_small, _safe


def export_incremental_draft(job, output, *, context_rows=16, cache_capacity=192,
                             max_draft_tokens=15, ge_root=None, cxx='c++',
                             timeout_seconds=3600, progress=None):
    """Export one static C16/C64 and candidate-width gear; never relabel S64."""
    _validate(Qwen35DFlashConfig.from_dict(configurations()[1]),
              context_rows, cache_capacity, True, max_draft_tokens)
    job, _ = _safe(job, directory=True)
    destination = output_path(output)
    if destination == job or job in destination.parents:
        raise ValueError('incremental output must be outside the retained full-draft job')
    if destination.exists():
        raise FileExistsError('incremental output already exists: ' + str(destination))
    bound = _bind_source_job(job, ge_root)
    invocation = uuid.uuid4().hex
    try:
        result = _export_source(bound['prepared_root'], bound['prepared'], destination, bound['selected_ge'],
            sequence_length=context_rows, residual_policy=bound['original']['residual_policy'],
            cxx=cxx, cxx11_abi=bound['original']['cxx11_abi'], timeout_seconds=timeout_seconds,
            progress=progress, _invocation_token=invocation,
            _incremental_options={'cache_capacity': cache_capacity, 'max_draft_tokens': max_draft_tokens})
        sources = {row['path']: row['identity'] for row in result['source_weight_identities']}
        sources[str(bound['library'])] = bound['library_id']
        _stable(bound['bindings'], sources)
        result.update(schema_version=1, artifact_kind='qwen35-incremental-draft-air',
            output=str(destination), invocation_token=invocation,
            source_job=bound['source_id'], source_prepared=bound['prepared_id'],
            source_air_manifest=bound['original_id'], source_bundle_manifest=bound['bundle_id'],
            source_bindings=bound['bindings'], ge_root=str(bound['selected_ge']),
            libgraph_sha256=bound['original']['libgraph_sha256'], cxx11_abi=bound['original']['cxx11_abi'],
            draft_config=bound['prepared']['draft_config'],
            source_preservation='ORIGINAL_RECEIPT_AND_WEIGHT_FULL_IDENTITIES_UNCHANGED_NO_LINK_COPY_OR_MANIFEST_REWRITE')
        save_json(destination / 'manifest.json', result)
        return result
    except Exception as error:
        # Never write to an existing or foreign attempt, including on failure.
        try:
            raw, _ = _read_small(destination / 'export-request.json', 16 << 10)
            if json.loads(raw).get('invocation_token') == invocation:
                save_json(destination / 'manifest.json', {
                    'schema_version': 1, 'artifact_kind': 'qwen35-incremental-draft-air',
                    'status': 'FAILED', 'output': str(destination), 'invocation_token': invocation,
                    'error': f'{type(error).__name__}: {error}',
                    'context_rows': context_rows, 'cache_capacity': cache_capacity,
                    'max_draft_tokens': max_draft_tokens, 'compilation_ready': False,
                    'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
                    'source_preservation': 'NOT_CERTIFIED_AFTER_FAILURE'})
        except (OSError, ValueError, TypeError, KeyError):
            pass
        raise

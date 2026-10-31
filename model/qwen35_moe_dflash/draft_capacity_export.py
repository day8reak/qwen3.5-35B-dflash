"""Export a new full-prefix draft AIR using an immutable prepared weight root.

This builds a real GE serializer and a new capacity graph. It does not run OMG,
copy/link weights, rewrite the original preparation, or create a deployable OMC.
The AIR retains explicit FileConstant references relative to the old prepared
root, so a future compiler must use a separately verified readonly input view.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import uuid

from .artifacts import files_under, output_path, save_json
from .draft_standalone import _graph, CURRENT_DRAFT_EXPORT_CONTRACT
from .ge_export import configurations, weight_identity, _logged
from .hashing import sha256_file
from .target_compile_failure import _read_small, _safe
from .target_prefill_smoke_plan import _identity


_SMALL_OUTPUT_BUDGET = 256 << 20
_RESERVE = 128 << 20


def _sdk_path(path, *, directory=False):
    """Resolve SDK installation links only; artifact/weight checks stay strict."""
    resolved = Path(path).expanduser().resolve(strict=True)
    return _safe(resolved, directory=directory)[0]


def _sdk_layout(ge_root):
    ge = _sdk_path(ge_root, directory=True)
    include = _sdk_path(ge / 'include', directory=True)
    lib = _sdk_path(ge / 'lib64', directory=True)
    references = [ge / 'include', ge / 'lib64',
                  ge / 'include/graph/graph.h', ge / 'include/graph/operator.h',
                  ge / 'lib64/libgraph.so']
    base = ge / 'lib64/libgraph_base.so'
    if base.exists() or base.is_symlink():
        references.append(base)
    paths, files = {}, {}
    for index, reference in enumerate(references):
        resolved = _sdk_path(reference, directory=index < 2)
        paths[str(reference)] = {'resolved': str(resolved), 'directory': index < 2}
        if index >= 2:
            files[str(resolved)] = _current(resolved)[1]
    return include, lib, paths, files


def _stable_sdk_paths(paths):
    for reference, row in paths.items():
        if str(_sdk_path(reference, directory=row['directory'])) != row['resolved']:
            raise ValueError('selected GE SDK path changed during export: ' + reference)


def _current(path):
    path, stamp = _safe(path)
    current = _identity(path.stat())
    if [current[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != stamp:
        raise ValueError('draft capacity source changed during identity capture: ' + str(path))
    return path, current


def _metadata(path, bindings):
    _, before = _current(path)
    raw, row = _read_small(path, 16 << 20)
    if _current(path)[1] != before or [before[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')] != row['stat']:
        raise ValueError('draft capacity metadata changed during its fresh read')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('draft capacity metadata must be an object')
    row['full_identity'] = before
    bindings.append(row)
    return value, row


def _bound(root, row, bindings):
    if not isinstance(row, dict) or not isinstance(row.get('path'), str):
        raise ValueError('missing draft component file identity')
    relative = Path(row['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('draft component file path escapes its root')
    value, identity = _metadata(Path(root) / relative, bindings)
    if any(identity[key] != row.get(key) for key in ('bytes', 'sha256')):
        raise ValueError('draft component metadata differs from its saved identity')
    return value, identity


def _stable(bindings, sources):
    for row in bindings:
        if _current(row['path'])[1] != row['full_identity']:
            raise ValueError('draft capacity metadata changed within this invocation: ' + row['path'])
    for path, identity in sources.items():
        if _current(path)[1] != identity:
            raise ValueError('original prepared weight identity changed: ' + path)


def _export_source(prepared, plan, output, ge_root, *, sequence_length, residual_policy,
                   cxx='c++', cxx11_abi=0, timeout_seconds=3600, progress=None, _invocation_token=None,
                   _incremental_options=None):
    """Shared readonly-weight exporter; cached graphs retain all twelve KV outputs."""
    from .ge_cpp import emit_cpp
    from .draft_direct_trace import instrument_draft_direct_outputs
    from .ge_residual import EXPORT_SUFFIX
    root = output_path(output)
    if root == prepared or Path(prepared) in root.parents:
        raise ValueError('capacity output must be outside the original prepared root')
    if root.exists():
        raise FileExistsError('capacity output already exists: ' + str(root))
    incremental = _incremental_options is not None
    if not incremental and (type(sequence_length) is not int or not plan['sequence_length'] < sequence_length <= 16384):
        raise ValueError('new capacity must exceed the original capacity and fit the native 16384-token bound')
    if type(cxx11_abi) is not int or cxx11_abi not in (0, 1) or type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError('invalid exporter ABI or timeout')
    if residual_policy != 'cpu-fp32-add':
        raise ValueError('capacity export preserves the successful cpu-fp32-add residual recipe')
    if (plan.get('artifact_kind') != 'qwen35-draft-ge-prepared' or plan.get('status') != 'PREPARED'
            or plan.get('external_weights') is not True or plan.get('include_hidden') is not True
            or plan.get('draft_config') != configurations(tiny=plan.get('tiny'))[1]):
        raise ValueError('capacity export requires the unchanged locked draft preparation')
    if incremental:
        from .config import Qwen35DFlashConfig
        from .ge_incremental_draft import build_incremental_draft, incremental_draft_abi
        config = Qwen35DFlashConfig.from_dict(plan['draft_config'])
        graph = build_incremental_draft(config, context_rows=sequence_length, **_incremental_options)
        spec = incremental_draft_abi(config, context_rows=sequence_length, **_incremental_options)
    else:
        _, config, graph = _graph(sequence_length, tiny=plan['tiny'], draft_config=plan['draft_config'])
        graph, spec = instrument_draft_direct_outputs(graph, names=('draft_top1', 'draft_hidden'))
    if config.num_hidden_layers != 6:
        raise ValueError('capacity export requires all six draft layers')
    if set(graph.weights) != set(plan.get('weights', {})) or plan['source'].get('fp16_payload_sha256') != weight_identity(plan['weights']):
        raise ValueError('prepared weights do not bind the complete capacity graph')
    sources, paths = {}, {}
    for key, tensor in graph.weights.items():
        row = plan['weights'][key]
        relative = Path(row['path'])
        if (relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != 'weights' or '..' in relative.parts
                or row.get('dtype') != 'float16' or tuple(row.get('shape', [])) != tensor.shape
                or row.get('bytes') != math.prod(tensor.shape) * 2):
            raise ValueError('prepared weight shape/type/path differs from capacity graph: ' + key)
        path, identity = _current(Path(prepared) / relative)
        if identity['bytes'] != row['bytes'] or sha256_file(path) != row['sha256'] or _current(path)[1] != identity:
            raise ValueError('prepared weight bytes or active identity changed: ' + key)
        paths[key], sources[str(path)] = str(path), identity
    include, lib, sdk_paths, sdk_files = _sdk_layout(ge_root)
    compiler = shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError('C++ compiler is unavailable: ' + cxx)
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / 'export-request.json', {'artifact_kind': 'qwen35-draft-incremental-export-request' if incremental else 'qwen35-draft-capacity-export-request',
        'invocation_token': _invocation_token,
        'sequence_length': sequence_length, 'source_sequence_length': plan['sequence_length'],
        'incremental_options': _incremental_options,
        'source_prepared_root': str(prepared), 'weight_files_copied': 0, 'weight_links_created': 0,
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN'})
    if shutil.disk_usage(root).free < _SMALL_OUTPUT_BUDGET + _RESERVE:
        raise OSError('capacity AIR export requires 384 MiB of free space; no weights have been copied')
    folder = root / 'draft'; folder.mkdir()
    reduction, row_mul, projection, residual = {}, {}, {}, {}
    source = folder / 'draft.cpp'
    source.write_text(emit_cpp(graph, paths, external_weights=True, recipe_dir=folder / 'recipe',
        fp16_reductions='matmul', reduction_report=reduction, fp16_row_mul='compact', row_mul_report=row_mul,
        file_constant_policy='cpro-relative-int', file_constant_root=prepared,
        draft_fc_policy='conv2d', draft_fc_report=projection,
        draft_residual_policy=residual_policy, draft_residual_report=residual))
    if len(residual['replaced']) != 12 or len(projection['replaced']) != (0 if plan['tiny'] else 1):
        raise ValueError('capacity lowering does not cover the locked six-layer graph')
    for name, value in [('reduction', reduction), ('row-mul', row_mul), ('projection', projection), ('residual', residual)]:
        save_json(folder / (name + '-lowering.json'), value)
    construction = files_under(folder, root)
    binary, air = folder / 'draft_exporter', root / 'draft.air'
    build = [compiler, '-std=c++17', f'-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}', '-I', str(include),
        str(source), '-L', str(lib), f'-Wl,-rpath,{lib}', '-lgraph']
    if (lib / 'libgraph_base.so').is_file(): build.append('-lgraph_base')
    build += ['-o', str(binary)]
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(filter(None, (str(lib), os.environ.get('LD_LIBRARY_PATH', '')))))
    save_json(folder / 'build-command.json', {'argv': build})
    if progress: progress('Building the new six-layer GE serializer; no OMG compilation')
    _logged(build, folder=folder, env=env, log=folder / 'build.log', timeout=timeout_seconds)
    _safe(binary)
    export = [str(binary), str(air)]
    save_json(folder / 'export-command.json', {'argv': export, 'cwd': str(prepared),
        'file_constant_root': str(prepared), 'scope': 'Original weight files are read only; output AIR is a new absolute path'})
    if progress: progress('Serializing the new AIR from the unchanged prepared weight root')
    _logged(export, folder=Path(prepared), env=env, log=folder / 'export.log', timeout=timeout_seconds)
    _, air_stat = _safe(air)
    if not 0 < air_stat[2] <= 64 << 20:
        raise ValueError('capacity AIR must be nonempty bounded external-weight metadata')
    _stable([], sources)
    _stable_sdk_paths(sdk_paths)
    for path, identity in sdk_files.items():
        if _current(path)[1] != identity:
            raise ValueError('selected GE SDK file changed during export: ' + path)
    for row in construction:
        path = root / row['path']
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError('capacity construction source changed during exporter execution')
    generated = files_under(root, root)
    observed = sum(row['bytes'] for row in generated)
    if observed > _SMALL_OUTPUT_BUDGET:
        raise ValueError('capacity source/AIR/exporter exceeded its 256 MiB small-output budget')
    status = ('HOST_FIXTURE_' if plan['tiny'] else 'DRAFT_') + ('INCREMENTAL' if incremental else 'CAPACITY') + '_AIR_EXPORTED'
    result = {'status': status,
        'sequence_length': sequence_length, 'original_sequence_length': plan['sequence_length'],
        'draft_layers': 6, 'draft_width': graph.outputs[0].shape[1], 'feature_width': config.feature_size,
        'inputs': [{'name': value.name, 'shape': list(value.shape), 'dtype': value.dtype} for value in graph.inputs],
        'outputs': [{'name': value.name, 'shape': list(value.shape), 'dtype': value.dtype} for value in graph.outputs],
        'direct_spec': spec, 'sdk_path_bindings': sdk_paths,
        'source_export_contract': CURRENT_DRAFT_EXPORT_CONTRACT + EXPORT_SUFFIX + '-direct-outputs-v1',
        'source_prepared_root': str(prepared), 'file_constant_root': str(prepared), 'file_constant_policy': 'cpro-relative-int',
        'file_constant_path_base': 'ORIGINAL_PREPARED_ROOT_EXPLICIT_REFERENCE',
        'source_weight_bytes': sum(row['bytes'] for row in plan['weights'].values()), 'weight_files_copied': 0,
        'weight_links_created': 0, 'weight_payload_access': 'READ_ONLY_FRESH_SHA256_AND_EXPORTER_SIZE_CHECKS',
        'source_weight_identities': [{'path': name, 'identity': identity} for name, identity in sorted(sources.items())],
        'air': {'path': 'draft.air', 'bytes': air_stat[2], 'sha256': sha256_file(air)}, 'generated_files': generated,
        'observed_new_file_bytes': observed, 'small_output_budget_bytes': _SMALL_OUTPUT_BUDGET,
        'compilation_ready': False, 'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'numerical_validation': 'NOT_RUN_NEW_CAPACITY', 'residual_policy': residual_policy,
        'next': 'Bind an explicit readonly compiler input view for these original-root FileConstants and budget a new OMG/packed OMC; this AIR is not an air-directory deployment bundle'}
    if incremental:
        result.pop('sequence_length')
        result.pop('direct_spec')
        result.update(context_rows=sequence_length, **_incremental_options,
            incremental_abi=spec, export_contract=CURRENT_DRAFT_EXPORT_CONTRACT + EXPORT_SUFFIX + '-context-kv-v2',
            numerical_validation='NOT_RUN_INCREMENTAL_DEVICE',
            runtime_integration='PENDING_NEW_COMPILED_NAMED_ABI_NOT_OLD_S64_BUNDLE',
            cache_bank_bytes=12 * math.prod(graph.inputs[5].shape) * 2)
    return result


def _bind_source_job(job, ge_root=None):
    """Bind a complete original preparation, successful recipe and selected GE library."""
    bindings = []
    source, source_id = _metadata(job / 'result.json', bindings)
    component = source.get('components', {}).get('prepared', {})
    if source.get('artifact_kind') != 'qwen35-full-draft-job' or not component.get('manifest'):
        raise ValueError('capacity export requires the original full-draft prepared component')
    prepared_root = job / 'prepared'
    prepared, prepared_id = _bound(prepared_root, component.get('manifest'), bindings)
    original, original_id = _metadata(job / 'residual-fix/air/manifest.json', bindings)
    bundle, bundle_id = _metadata(job / 'residual-fix/bundle/manifest.json', bindings)
    from .ge_residual import EXPORT_SUFFIX
    expected_contract = CURRENT_DRAFT_EXPORT_CONTRACT + EXPORT_SUFFIX + '-direct-outputs-v1'
    if (prepared.get('tiny') is not False or original.get('artifact_kind') != 'qwen35-cpro-draft-air'
            or original.get('status') != 'EXPORTED' or original.get('residual_policy') != 'cpu-fp32-add'
            or original.get('export_contract') != expected_contract or original.get('fp16_reductions') != 'matmul'
            or original.get('fp16_row_mul') != 'compact'
            or original.get('sequence_length') != prepared.get('sequence_length')
            or original.get('prepared_manifest_sha256') != prepared_id['sha256']
            or bundle.get('artifact_kind') != 'qwen35-standalone-draft-direct-bundle' or bundle.get('status') != 'PREPARED'
            or bundle.get('draft_config') != prepared.get('draft_config') or bundle.get('sequence_length') != prepared.get('sequence_length')
            or bundle.get('export_manifest_sha256') != original_id['sha256'] or bundle.get('weight_storage') != 'embedded-stream'):
        raise ValueError('source residual-fix AIR and embedded bundle do not bind this complete preparation')
    selected_ge = _sdk_path(ge_root or original['ge_root'], directory=True)
    selected_library = selected_ge / 'lib64/libgraph.so'
    library, library_id = _current(_sdk_path(selected_library))
    if sha256_file(library) != original.get('libgraph_sha256') or _current(library)[1] != library_id:
        raise ValueError('selected GE library differs from the successful source exporter')
    return dict(bindings=bindings, source_id=source_id, prepared_root=prepared_root,
        prepared=prepared, prepared_id=prepared_id, original=original, original_id=original_id,
        bundle_id=bundle_id, selected_ge=selected_ge, library=library, library_id=library_id)


def export_draft_capacity(job, output, *, sequence_length=128, prompt_length=64, max_new_tokens=32,
                          ge_root=None, cxx='c++', timeout_seconds=3600, progress=None):
    """Bind the successful residual-fix lane, then export only a new capacity AIR."""
    if (any(type(value) is not int or value <= 0 for value in (sequence_length, prompt_length, max_new_tokens))
            or sequence_length > 16384 or prompt_length + max_new_tokens > sequence_length):
        raise ValueError('capacity must cover the complete prompt plus the generation budget')
    job, _ = _safe(job, directory=True)
    destination = output_path(output)
    if destination == job or job in destination.parents:
        raise ValueError('capacity output must be outside the retained full-draft job')
    bound = _bind_source_job(job, ge_root)
    bindings, source_id = bound['bindings'], bound['source_id']
    prepared_root, prepared, prepared_id = bound['prepared_root'], bound['prepared'], bound['prepared_id']
    original, original_id, bundle_id = bound['original'], bound['original_id'], bound['bundle_id']
    selected_ge, library, library_id = bound['selected_ge'], bound['library'], bound['library_id']
    invocation = uuid.uuid4().hex
    try:
        result = _export_source(prepared_root, prepared, output, selected_ge, sequence_length=sequence_length,
            residual_policy=original['residual_policy'], cxx=cxx, cxx11_abi=original['cxx11_abi'],
            timeout_seconds=timeout_seconds, progress=progress, _invocation_token=invocation)
        original_sources = {row['path']: row['identity'] for row in result['source_weight_identities']}
        original_sources[str(library)] = library_id
        _stable(bindings, original_sources)
        if _sdk_path(selected_ge / 'lib64/libgraph.so') != library:
            raise ValueError('selected GE library link changed during export')
        result.update(schema_version=1, artifact_kind='qwen35-draft-capacity-air', output=str(destination),
            invocation_token=invocation, source_job=source_id, source_prepared=prepared_id,
            source_air_manifest=original_id, source_bundle_manifest=bundle_id,
            source_bindings=bindings, prompt_length=prompt_length, max_new_tokens=max_new_tokens,
            ge_root=str(selected_ge), libgraph_sha256=original['libgraph_sha256'], cxx11_abi=original['cxx11_abi'],
            source_preservation='ORIGINAL_RECEIPT_AND_WEIGHT_FULL_IDENTITIES_UNCHANGED_NO_LINK_COPY_OR_MANIFEST_REWRITE')
        save_json(destination / 'manifest.json', result)
        return result
    except Exception as error:
        # Only this invocation's freshly created directory can receive a failure
        # receipt. Existing/foreign attempts and every source receipt stay intact.
        try:
            raw, _ = _read_small(destination / 'export-request.json', 16 << 10)
            request = json.loads(raw)
            if request.get('invocation_token') == invocation:
                save_json(destination / 'manifest.json', {'schema_version': 1, 'artifact_kind': 'qwen35-draft-capacity-air',
                    'status': 'FAILED', 'output': str(destination), 'invocation_token': invocation,
                    'error': f'{type(error).__name__}: {error}', 'model_compilation': 'NOT_RUN',
                    'device_execution': 'NOT_RUN', 'compilation_ready': False,
                    'source_preservation': 'NOT_CERTIFIED_AFTER_FAILURE', 'weight_links_created': 0,
                    'weight_files_copied': 0, 'scope': 'Retained partial source/AIR/logs; not a usable capacity artifact'})
        except (OSError, ValueError, TypeError, KeyError):
            pass
        raise

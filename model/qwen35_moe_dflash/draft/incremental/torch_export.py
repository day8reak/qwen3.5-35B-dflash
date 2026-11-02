"""Portable PyTorch AIR export and selected-DDK OMC candidate compilation.

The compiler side imports no PyTorch. Candidates stay separate from the current
runtime model until physical IO binding and device precision are validated.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil

from qwen35_moe_dflash.common.artifacts import output_path, working_directory

AIR_KIND = 'qwen35-incremental-draft-pytorch-air'
OMC_KIND = 'qwen35-incremental-draft-pytorch-omc'
GE_AIR_KIND = 'qwen35-incremental-draft-air'


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def _slot(output, kind, replace=False):
    if Path(output).is_symlink():
        raise ValueError('candidate output must not be a symlink')
    root = output_path(output)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ('.' + root.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('candidate slot is active; retained') from None
        if root.exists():
            marker = root / 'manifest.json'
            old = json.loads(marker.read_text()) if marker.is_file() else {}
            if (not replace or old.get('artifact_kind') != kind
                    or old.get('status') not in {'FAILED', 'PYTORCH_DRAFT_AIR_EXPORTED', 'PYTORCH_DRAFT_OMC_COMPILED'}):
                raise ValueError('output is active, foreign or already exists; --replace only replaces a finished owned candidate')
            shutil.rmtree(root)
        root.mkdir()
        yield root


def _disjoint(output, *sources):
    destination = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError('candidate output must be separate from inputs and SDK directories')


@contextmanager
def _air_locks(*folders):
    with ExitStack() as stack:
        for folder in sorted(set(folders)):
            if folder.is_symlink():
                raise ValueError('AIR directory must not be a symlink')
            folder.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(folder.parent / ('.' + folder.name + '.lock'),
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            lock = stack.enter_context(os.fdopen(fd, 'a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('AIR directory is active; retained') from None
        from qwen35_moe_dflash.execution.run_layout import _active_processes
        if _active_processes(Path(os.environ['AI_RUN_DIR']).resolve()):
            raise ValueError('finish active local model jobs before moving shared AIR')
        yield


def _replaceable_air(folder, replace):
    if folder.is_symlink():
        raise ValueError('AIR directory must not be a symlink')
    if not folder.exists():
        return
    marker = folder / 'manifest.json'
    old = json.loads(marker.read_text()) if marker.is_file() and not marker.is_symlink() else {}
    terminal = {AIR_KIND: {'FAILED', 'PYTORCH_DRAFT_AIR_EXPORTED'},
                GE_AIR_KIND: {'FAILED', 'DRAFT_INCREMENTAL_AIR_EXPORTED'}}
    if not replace or old.get('status') not in terminal.get(old.get('artifact_kind'), set()):
        raise ValueError('AIR destination is active or unknown; --replace requires a finished Draft AIR bundle')


def _completed_air(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest.get('artifact_kind') != AIR_KIND or manifest.get('status') != 'PYTORCH_DRAFT_AIR_EXPORTED':
        raise ValueError('import requires a completed PyTorch Draft AIR bundle')
    schema = _schema((folder / 'air-schema.pb').read_bytes(), manifest['schema'])
    observed = inspect_air(folder / 'export.air', schema, expected_norms=32, require_int32_indices=True)
    recorded = json.loads((folder / 'air-inspection.json').read_text())
    if (len(manifest.get('inputs', [])) != 17 or len(manifest.get('outputs', [])) != 14
            or observed['input_count'] != 17 or observed['output_counts'] != [14]
            or not observed['weight_files'] or observed['weight_files'] != recorded.get('weight_files')):
        raise ValueError('AIR inputs/outputs or external weight extents differ; retain the previous AIR')
    return manifest


def _install_air(source, destination, replace):
    _replaceable_air(destination, replace)
    previous = destination.with_name('.' + destination.name + '-replaced')
    if previous.exists() or previous.is_symlink():
        raise ValueError('interrupted AIR replacement exists; retain it and recover before replacing again: ' + str(previous))
    # Use rename only: no second copy of the multi-GB external weights. The
    # current OMC and its runtime configuration do not depend on these AIR files.
    had_previous = destination.exists()
    if had_previous:
        destination.rename(previous)
    try:
        source.rename(destination)
    except BaseException:
        if had_previous:
            previous.rename(destination)
        raise
    if had_previous:
        shutil.rmtree(previous)


def import_pytorch_air(source, output, *, replace=False):
    """Move a completed portable AIR into its shared location; no TorchAir/DDK."""
    if Path(source).is_symlink() or Path(output).is_symlink():
        raise ValueError('AIR directory must not be a symlink')
    source, destination = output_path(source), output_path(output)
    if source != destination:
        _disjoint(destination, source)
    with _air_locks(source, destination):
        manifest = _completed_air(source)
        if source != destination:
            _install_air(source, destination, replace)
    return dict(status='PYTORCH_DRAFT_AIR_INSTALLED', air=str(destination / 'export.air'),
                output=str(destination), moved=source != destination,
                input_count=len(manifest['inputs']), output_count=len(manifest['outputs']),
                rms_nodes=32, weight_files=manifest['weight_files'], model_compilation='NOT_RUN',
                device_execution='NOT_RUN', numerical_validation='NOT_RUN', runtime_changed=False)


@contextmanager
def _air_export_slot(output, prepared, replace):
    if Path(output).is_symlink():
        raise ValueError('AIR directory must not be a symlink')
    destination = output_path(output)
    if destination.name != 'air':
        with _slot(output, AIR_KIND, replace) as root:
            yield root
        return
    if (destination.parent / 'tests').is_symlink() or (destination.parent / 'tests/export').is_symlink():
        raise ValueError('AIR export scratch must not be a symlink')
    scratch = output_path(destination.parent / 'tests/export')
    _disjoint(scratch, destination, prepared)
    with _air_locks(destination, scratch):
        _replaceable_air(destination, replace)
        if scratch.exists():
            old = json.loads((scratch / 'manifest.json').read_text())
            if not replace or old.get('artifact_kind') != AIR_KIND or old.get('status') != 'FAILED':
                raise ValueError('export scratch is active or completed; import a completed AIR instead of re-exporting: ' + str(scratch))
            shutil.rmtree(scratch)
        scratch.mkdir()
        yield scratch
        if json.loads((scratch / 'manifest.json').read_text()).get('status') == 'PYTORCH_DRAFT_AIR_EXPORTED':
            _completed_air(scratch)
            _install_air(scratch, destination, replace)


def _producer_schema(pb2=None):
    from qwen35_moe_dflash.toolchain.air_schema import discover_torchair_schemas, extract_torchair_descriptor
    if not pb2:
        candidates = [r for r in discover_torchair_schemas()['candidates'] if r['status'] == 'LITERAL_DESCRIPTOR']
        if not candidates or len({r['descriptor_sha256'] for r in candidates}) != 1:
            raise ValueError('select the installed TorchAir ge_ir_by_protoc_*_pb2.py with --torchair-pb2')
        pb2 = min(candidates, key=lambda r: r['pb2_bytes'])['path']
    return extract_torchair_descriptor(pb2)


def _schema(descriptor, metadata):
    from qwen35_moe_dflash.toolchain.omc_wire import Schema
    return Schema(descriptor, expected_file=metadata['schema_file'], expected_package=metadata['schema_package'])


def inspect_air(air, schema, *, expected_norms=None, require_int32_indices=False):
    """Read only graph metadata and sidecar extents, without loading weight data."""
    from qwen35_moe_dflash.target.original_air_roundtrip import _decode, _attributes, MAX_AIR_BYTES
    path = Path(air)
    with path.open('rb') as stream:
        raw = stream.read(MAX_AIR_BYTES + 1)
    if not raw or len(raw) > MAX_AIR_BYTES:
        raise ValueError('expected an external-weight AIR graph at most 64 MiB')
    _, _, graph = _decode(schema, raw)
    nodes = graph['op']
    counts = Counter(n['type'] for n in nodes)
    if expected_norms is not None and (counts['RmsNorm'] != expected_norms or counts['AdnRmsNorm']):
        raise ValueError(f'expected {expected_norms} installed-converter RmsNorm nodes; observed {counts["RmsNorm"]}, AdnRmsNorm={counts["AdnRmsNorm"]}')
    types = schema.enums.get(schema.package + '.DataType', {})
    def desc(d):
        return dict(shape=d.get('shape', {}).get('dim'), layout=d.get('layout'), raw_dtype=d.get('dtype'),
                    dtype_names=[name for name, value in types.items() if value == d.get('dtype')])
    norms, sidecars, gathers = [], {}, []
    for n in nodes:
        attrs = _attributes(n)
        if n['type'] in {'Gather', 'GatherV2', 'GatherV2D'}:
            inputs = n.get('input_desc', [])
            indices = desc(inputs[1]) if len(inputs) > 1 else {}
            gathers.append(dict(name=n['name'], type=n['type'], indices=indices))
            if require_int32_indices and 'DT_INT32' not in indices.get('dtype_names', []):
                raise ValueError(f"X90 Draft Gather indices must be INT32: {n['name']} "
                                 f"has {indices.get('dtype_names', [])}; re-export with INT32 embedding/KV indices")
        if n['type'] in {'RmsNorm', 'AdnRmsNorm'}:
            norms.append(dict(name=n['name'], type=n['type'], input=n.get('input', []),
                inputs=[desc(d) for d in n.get('input_desc', [])],
                outputs=[desc(d) for d in n.get('output_desc', [])], epsilon=attrs.get('epsilon')))
        if n['type'] != 'FileConstant':
            continue
        name = attrs.get('file_path', {}).get('s')
        if not isinstance(name, str):
            raise ValueError('FileConstant without file_path')
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or len(relative.parts) != 2 or relative.parts[0] != 'weight'):
            raise ValueError('export must reference portable ./weight/<file>: ' + name)
        member = path.parent / str(relative)
        if member.is_symlink() or path.parent.resolve() not in member.resolve().parents or not member.is_file() or member.stat().st_size <= 0:
            raise ValueError('missing external weight: ' + name)
        sidecars[str(relative)] = member.stat().st_size
    inputs = [n for n in nodes if n['type'] == 'Data']
    outputs = [n for n in nodes if n['type'] == 'NetOutput']
    return dict(node_types=dict(counts), rms=norms, gather_indices=gathers, weight_files=sidecars,
                input_count=len(inputs), output_counts=[len(n.get('input_desc', [])) for n in outputs],
                scope='SOURCE_GRAPH_ONLY_NOT_COMPILED_PRECISION_OR_RUNTIME_ABI')


def export_pytorch(prepared, output, *, context_rows=16, cache_capacity=192, max_draft_tokens=15,
                   replace=False, torchair_pb2=None, progress=print):
    from qwen35_moe_dflash.toolchain.export import require_torchair
    torchair = require_torchair()
    import torch
    from .torch_model import load_prepared
    # Use the installed converter, as the original CPro Target does. No Adn
    # replacement, monkey-patched converter or CPU fallback on export failure.
    rms = torch.ops.npu.npu_rms_norm.default
    descriptor, metadata = _producer_schema(torchair_pb2)
    _disjoint(output, prepared)
    with _air_export_slot(output, prepared, replace) as root:
        result = dict(artifact_kind=AIR_KIND, status='EXPORTING', stage='weights', frontend='pytorch',
            context_rows=context_rows, cache_capacity=cache_capacity, max_draft_tokens=max_draft_tokens,
            model_compilation='NOT_RUN', device_execution='NOT_RUN', numerical_validation='NOT_RUN',
            rms='torch.ops.npu.npu_rms_norm.default -> RmsNorm', gamma='DIRECT_DRAFT_GAMMA_NO_ADD_ONE',
            internal_precision='UNVERIFIED_AFTER_COMPILATION', runtime_integration='PENDING_PHYSICAL_ABI_AND_DEVICE_VALIDATION')
        _save(root / 'manifest.json', result)
        try:
            weights = json.loads((Path(prepared) / 'prepared.json').read_text())['weights']
            required = sum(r['bytes'] for r in weights.values()) + (256 << 20)
            if shutil.disk_usage(root).free < required:
                raise OSError(f'AIR export needs at least {required} free bytes for weights and reserve')
            progress('loading Draft, embedding and head from shared prepared FP16 weights')
            model = load_prepared(prepared, context_rows=context_rows, cache_capacity=cache_capacity,
                                  max_draft_tokens=max_draft_tokens, rms=rms)
            result.update(**model.abi(), schema={k: metadata[k] for k in ('schema_file', 'schema_package')},
                torch_version=torch.__version__, torchair_version=getattr(torchair, '__version__', 'unknown'),
                projection='PYTORCH_CONV2D_4X4_FULL_K', stage='torchair-export')
            _save(root / 'manifest.json', result)
            (root / 'air-schema.pb').write_bytes(descriptor)
            progress('TorchAir export: 17 inputs, 14 outputs, tensor-controlled committed KV')
            with working_directory(root), torch.inference_mode():
                torchair.dynamo_export(*model.example_inputs(), model=model, export_path='weight', export_name='export', dynamic=False)
            # Relative export_path keeps FileConstant paths portable on the DDK
            # server; move only graph metadata, leaving the weight files in place.
            (root / 'weight/export.air').replace(root / 'export.air')
            for name in ('dynamo.pbtxt', 'dynamo.txt'):
                if (root / 'weight' / name).is_file():
                    (root / 'weight' / name).replace(root / name)
            inspection = inspect_air(root / 'export.air', _schema(descriptor, metadata),
                                     expected_norms=32, require_int32_indices=True)
            if inspection['input_count'] != 17 or inspection['output_counts'] != [14]:
                raise ValueError('TorchAir source boundary does not contain all 17 inputs / 14 outputs')
            _save(root / 'air-inspection.json', inspection)
            result.update(status='PYTORCH_DRAFT_AIR_EXPORTED', stage='complete', air='export.air',
                schema_file='air-schema.pb', rms_nodes=len(inspection['rms']),
                weight_files=len(inspection['weight_files']), gather_index_dtype='int32',
                gather_nodes=len(inspection['gather_indices']),
                next='Transfer this whole AIR directory; compile it on the selected DDK server. Candidate is not runtime-bound.')
        except Exception as error:
            result.update(status='FAILED', error=f'{type(error).__name__}: {error}',
                          manifest=str(root / 'manifest.json'))
        _save(root / 'manifest.json', result)
        return result


def _compiled_observation(omc, ddk):
    from qwen35_moe_dflash.toolchain.omc_container import read_container, read_partition
    from qwen35_moe_dflash.toolchain.omc_wire import load_schema
    from qwen35_moe_dflash.toolchain.omc_cpu_inspect import _task
    from qwen35_moe_dflash.draft.ir_inspect import _enum_fields
    from qwen35_moe_dflash.draft.compile_precision import _observed_desc
    schema = load_schema(Path(ddk) / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    parts = [p for p in read_container(omc)['partitions'] if p['type'] == 0]
    if len(parts) != 1:
        raise ValueError('expected one compiled model metadata partition')
    model = schema.decode(read_partition(omc, parts[0]), 'ModelDef')
    enums, owners, nodes = _enum_fields(schema), Counter(), []
    for graph in model.get('graph', []):
        for node in graph.get('op', []):
            if node['type'] not in {'Data', 'NetOutput', 'GraphOp', 'Cast', 'CastT'}:
                continue
            row = dict(name=node['name'], type=node['type'], input=node.get('input', []),
                       inputs=[_observed_desc(d, enums) for d in node.get('input_desc', [])],
                       outputs=[_observed_desc(d, enums) for d in node.get('output_desc', [])])
            if node['type'] == 'GraphOp':
                row['owner'] = _task(node)['cl_name']
                owners[row['owner']] += 1
            nodes.append(row)
    return dict(graphop_counts=dict(owners), nodes=nodes,
                scope='COMPILED_DESCRIPTORS_ONLY_NOT_RUNTIME_SLOT_BINDING_OR_INTERNAL_INSTRUCTION_PRECISION')


def inspect_pytorch_compilation(directory):
    """Inspect a retained candidate without the AIR, SDK or compiled payloads."""
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('artifact_kind') != OMC_KIND:
        raise ValueError('expected a PyTorch Draft OMC compilation directory')
    return dict(status='INSPECTED', source_status=manifest.get('status'),
                source_stage=manifest.get('stage'), error=manifest.get('error'),
                compilation_diagnostics=compilation_diagnostics(root / 'compilation'),
                scope='RETAINED_COMPILER_REPORT_AND_LOGS_ONLY_ROOT_CAUSE_NOT_CLASSIFIED',
                model_compilation='NOT_RUN', device_execution='NOT_RUN')


def compile_pytorch(air_export, output, *, ddk_root, ddk_id, replace=False,
                    timeout_seconds=7200, tbe_diagnosis=None, progress=print):
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics

    def compiler_progress(row):
        message = row['message']
        if row.get('stage') == 'input-snapshot' and message == 'Snapshotting input':
            index, total = row['completed'], row['total']
            if index not in (0, total - 1) and (index + 1) % 10:
                return
            message = f"Snapshotting input {index + 1}/{total}: {row['source_relative']} ({row['bytes']} bytes)"
        progress(message)

    source = Path(air_export).resolve(strict=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('artifact_kind') != AIR_KIND or manifest.get('status') != 'PYTORCH_DRAFT_AIR_EXPORTED':
        raise ValueError('expected a completed PyTorch Draft AIR bundle')
    if not ddk_root or not ddk_id or not Path(ddk_root).is_dir():
        raise ValueError('supply the selected --ddk-root and --ddk-id')
    _disjoint(output, source, ddk_root)
    metadata = manifest['schema']
    descriptor = (source / 'air-schema.pb').read_bytes()
    inspection = inspect_air(source / 'export.air', _schema(descriptor, metadata),
                             expected_norms=32, require_int32_indices=True)
    with _slot(output, OMC_KIND, replace) as root:
        result = {k: manifest[k] for k in ('context_rows', 'cache_capacity', 'max_draft_tokens', 'inputs', 'outputs', 'rms', 'gamma')}
        result.update(artifact_kind=OMC_KIND, status='COMPILING', stage='compile', frontend='pytorch',
            model_compilation='NOT_RUN', compiled_precision='NOT_VALIDATED', compiled_abi='NOT_BOUND',
            device_execution='NOT_RUN', numerical_validation='NOT_RUN', ddk_id=ddk_id,
            runtime_integration='PENDING_PHYSICAL_ABI_AND_DEVICE_VALIDATION')
        _save(root / 'manifest.json', result)
        try:
            # Snapshot + generated sidecars + packed model coexist. Reuse the
            # proven external-compile/stream-pack path, not OMG's large embedded
            # serialization. Compiler scratch may exceed this lower bound.
            required = 3 * (sum(inspection['weight_files'].values()) + (source / 'export.air').stat().st_size) + (1 << 30)
            if shutil.disk_usage(root).free < required:
                raise OSError(f'OMC candidate compilation needs at least {required} free bytes; compiler scratch may need more')
            progress('compiling portable PyTorch AIR using selected OMG; no CPU customization')
            compiled = compile_existing_air(source / 'export.air', root / 'compilation', ddk_root,
                ddk_id=ddk_id, timeout_seconds=timeout_seconds, input_layout='air-directory',
                use_origin_format=True, save_weights_as_external_data=True,
                tbe_diagnosis=tbe_diagnosis, progress=compiler_progress)
            result['model_compilation'] = compiled['model_compilation']
            if compiled['status'] != 'COMPILED':
                raise RuntimeError(str(compiled.get('error', compiled['status'])))
            model_path = root / 'compilation' / compiled['omc']['path']
            try:
                observed = _compiled_observation(model_path, ddk_root)
                _save(root / 'compiled-interface.json', observed)
                result['graphop_counts'] = observed['graphop_counts']
                result['interface_inspection'] = 'OBSERVED'
            except (OSError, ValueError, KeyError) as error:
                result.update(interface_inspection='UNAVAILABLE', interface_error=str(error))
            result['stage'] = 'pack-model'
            _save(root / 'manifest.json', result)
            from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
            destination = root / 'model.omc'
            sidecars = [dict(row, path=str(root / 'compilation' / row['path']))
                        for row in compiled['external_data_candidates'] if Path(row['path']).suffix == '.weight']
            packing = embed_external_weights(model_path, destination,
                Path(ddk_root) / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so',
                expected_source_sha256=compiled['omc']['sha256'], sidecars=sidecars, progress=progress)
            _save(root / 'weight-packing.json', packing)
            if packing['status'] != 'EMBEDDED':
                raise ValueError('candidate streaming weight packing did not complete')
            result.update(status='PYTORCH_DRAFT_OMC_COMPILED', stage='complete', omc='model.omc',
                          weight_storage='embedded-stream')
            result['next'] = 'Review compiled-interface.json; bind semantic KV slots and validate on device before promoting this candidate.'
        except Exception as error:
            result.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            result['compilation_diagnostics'] = compilation_diagnostics(root / 'compilation')
            result['next'] = 'Inspect compilation_diagnostics and the retained compile.log before retrying this same build directory.'
        _save(root / 'manifest.json', result)
        return result

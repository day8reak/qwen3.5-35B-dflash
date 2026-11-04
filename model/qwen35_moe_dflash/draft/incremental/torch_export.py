"""Portable PyTorch AIR export and selected-DDK OMC candidate compilation.

The compiler side imports no PyTorch. Candidates stay separate from the current
runtime model until physical IO binding and device precision are validated.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil

from qwen35_moe_dflash.common.artifacts import output_path, working_directory

AIR_KIND = 'qwen35-incremental-draft-pytorch-air'
OMC_KIND = 'qwen35-incremental-draft-pytorch-omc'
GE_AIR_KIND = 'qwen35-incremental-draft-air'
COMPOSED_ATTENTION = 'composed-attention-v1'
# Exactly one key/value tensor in the original Target CPro converter. Empty
# optional inputs retain their slots even when AIR omits descriptor names.
_ATTENTION_PORTS = ('query', 'key0', 'value0', 'pse_shift', 'atten_mask',
    'actual_seq_lengths_q', 'actual_seq_lengths_kv', 'dequant_scale1', 'quant_scale1',
    'dequant_scale2', 'quant_scale2', 'quant_offset2', 'antiquant_scale',
    'antiquant_offset', 'block_table', 'kv_padding_size', 'all_seq_lengths_q',
    'actual_seq_lengths_q_back')


class AirValidationError(ValueError):
    """Retain the observation even when an export cannot be published."""
    def __init__(self, inspection):
        self.inspection = inspection
        first = inspection['validation_errors'][0]
        super().__init__(f"AIR validation: {first['node']} {first['field']}: "
                         f"expected {first['expected']!r}, observed {first['actual']!r}")


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
    observed = inspect_air(folder / 'export.air', schema, expected_norms=32, require_int32_indices=True,
                           expected_attention=manifest.get('attention'))
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
                attention_nodes=manifest.get('attention_nodes', 0),
                composed_attention_layers=manifest.get('composed_attention_layers', 0),
                rms_nodes=32, weight_files=manifest['weight_files'], model_compilation='NOT_RUN',
                device_execution='NOT_RUN', numerical_validation='NOT_RUN', runtime_changed=False,
                air_validation=manifest.get('air_validation', 'NOT_RECORDED'),
                deferred_shape_count=manifest.get('deferred_shape_count', 0))


def _finish_air_manifest(result, inspection):
    for key in ('error', 'first_mismatch', 'manifest', 'inspection'):
        result.pop(key, None)
    result.update(status='PYTORCH_DRAFT_AIR_EXPORTED', stage='complete', air='export.air',
        schema_file='air-schema.pb', rms_nodes=len(inspection['rms']),
        weight_files=len(inspection['weight_files']), gather_index_dtype='int32',
        gather_nodes=len(inspection['gather_indices']), attention_nodes=len(inspection['attention']),
        composed_attention_layers=len(inspection.get('composed_attention', [])),
        air_validation=inspection['validation_status'],
        deferred_shape_count=len(inspection['deferred_shapes']),
        next='Transfer this whole AIR directory; compile it on the selected DDK server. Missing intermediate shapes require OMG inference; candidate is not runtime-bound.')


def finalize_pytorch_air(source, output, *, replace=False):
    """Recheck saved AIR after a post-export failure; never rerun TorchAir."""
    if Path(source).is_symlink() or Path(output).is_symlink():
        raise ValueError('AIR directory must not be a symlink')
    source, destination = output_path(source), output_path(output)
    if source != destination:
        _disjoint(destination, source)
    with _air_locks(source, destination):
        result = json.loads((source / 'manifest.json').read_text())
        if result.get('artifact_kind') != AIR_KIND:
            raise ValueError('finalize requires a retained PyTorch Draft AIR bundle')
        if result.get('status') == 'PYTORCH_DRAFT_AIR_EXPORTED':
            _completed_air(source)
        else:
            if (result.get('status') != 'FAILED'
                    or result.get('stage') not in {'torchair-export', 'air-validation'}
                    or result.get('attention', {}).get('policy') not in
                       {'target-cpro-fused-attention-v1', COMPOSED_ATTENTION}):
                raise ValueError('only a finished supported Draft AIR export failure can be finalized')
            if source != destination:
                _replaceable_air(destination, replace)
            inspection = inspect_air(source / 'export.air',
                _schema((source / 'air-schema.pb').read_bytes(), result['schema']),
                expected_norms=32, require_int32_indices=True, expected_attention=result['attention'])
            if (len(result.get('inputs', [])) != 17 or len(result.get('outputs', [])) != 14
                    or inspection['input_count'] != 17 or inspection['output_counts'] != [14]
                    or not inspection['weight_files']):
                raise ValueError('saved AIR is incomplete: expected 17 inputs, 14 outputs and external weights')
            result['recovered_export_failure'] = dict(stage=result['stage'], error=result.get('error'))
            _finish_air_manifest(result, inspection)
            _save(source / 'air-inspection.json', inspection)
            _save(source / 'manifest.json', result)
        if source != destination:
            _install_air(source, destination, replace)
    return dict(status='PYTORCH_DRAFT_AIR_INSTALLED', output=str(destination),
        air=str(destination / 'export.air'), moved=source != destination,
        input_count=len(result['inputs']), output_count=len(result['outputs']),
        rms_nodes=result['rms_nodes'], attention_nodes=result.get('attention_nodes', 0),
        composed_attention_layers=result.get('composed_attention_layers', 0),
        weight_files=result['weight_files'], air_validation=result.get('air_validation'),
        deferred_shape_count=result.get('deferred_shape_count', 0),
        model_export='NOT_RUN_REUSED_RETAINED_AIR', model_compilation='NOT_RUN',
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


def inspect_air(air, schema, *, expected_norms=None, require_int32_indices=False,
                expected_attention=None, strict=True):
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
    errors, deferred_shapes = [], []
    def mismatch(node, field, expected, actual):
        errors.append(dict(node=node, field=field, expected=expected, actual=actual))
    if expected_norms is not None and (counts['RmsNorm'] != expected_norms or counts['AdnRmsNorm']):
        mismatch('graph', 'norm_counts', {'RmsNorm': expected_norms, 'AdnRmsNorm': 0},
                 {'RmsNorm': counts['RmsNorm'], 'AdnRmsNorm': counts['AdnRmsNorm']})
    composed = expected_attention is not None and expected_attention.get('policy') == COMPOSED_ATTENTION
    if expected_attention is not None and not composed and (expected_attention.get('policy') != 'target-cpro-fused-attention-v1'
            or counts['AdnFusedInferAttention'] != 6):
        mismatch('graph', 'attention_policy_and_count',
                 {'policy': 'target-cpro-fused-attention-v1', 'nodes': 6},
                 {'policy': expected_attention.get('policy'), 'nodes': counts['AdnFusedInferAttention']})
    types = schema.enums.get(schema.package + '.DataType', {})
    def desc(d):
        return dict(shape=d.get('shape', {}).get('dim'), layout=d.get('layout'), raw_dtype=d.get('dtype'),
                    dtype_names=[name for name, value in types.items() if value == d.get('dtype')])
    names = Counter(n['name'] for n in nodes)
    producers = {n['name']: n for n in nodes if names[n['name']] == 1}
    def port(d, index, references):
        row = dict(desc(d), slot=index, name=d.get('name'),
                   reference=references[index] if index < len(references) else None)
        reference = row['reference']
        if reference:
            name, separator, output = reference.rpartition(':')
            producer = producers.get(name)
            if separator and output.isdecimal() and producer and int(output) < len(producer.get('output_desc', [])):
                row['producer'] = dict(node=name, type=producer['type'], output=int(output),
                                       descriptor=desc(producer['output_desc'][int(output)]))
            else:
                row['producer_resolution'] = 'UNRESOLVED_EXPLICIT_EDGE'
        return row
    norms, sidecars, gathers, attention, zeros = [], {}, [], [], []
    for n in nodes:
        attrs = _attributes(n)
        if n['type'] == 'ZerosLike':
            zeros.append(dict(name=n['name'],
                inputs=[port(d, i, n.get('input', [])) for i, d in enumerate(n.get('input_desc', []))],
                outputs=[desc(d) for d in n.get('output_desc', [])]))
        if n['type'] == 'AdnFusedInferAttention':
            inputs = [port(d, i, n.get('input', [])) for i, d in enumerate(n.get('input_desc', []))]
            # TorchAir stores names in attributes before AIR serialization. The
            # Target's fixed slot signature is still usable if names are absent.
            keys = attrs.get('_input_name_key', {}).get('list', {}).get('s', [])
            values = attrs.get('_input_name_value', {}).get('list', {}).get('i', [])
            for key, index in zip(keys, values):
                if isinstance(index, int) and 0 <= index < len(inputs):
                    inputs[index]['mapped_name'] = key
            for i, d in enumerate(inputs):
                d['target_role'] = _ATTENTION_PORTS[i] if i < len(_ATTENTION_PORTS) else None
            observed_attrs = {key: attrs.get(key, {}).get('s' if key == 'input_layout' else 'f' if key == 'scale_value' else 'i')
                              for key in ('num_heads', 'num_key_value_heads', 'block_size', 'input_layout', 'inner_precise', 'scale_value')}
            attention.append(dict(name=n['name'], inputs=inputs,
                outputs=[desc(d) for d in n.get('output_desc', [])], attrs=observed_attrs))
            if expected_attention is not None and not composed:
                for key, actual in observed_attrs.items():
                    expected = expected_attention[key]
                    matched = (isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-6)
                               if key == 'scale_value' else actual == expected)
                    if not matched:
                        mismatch(n['name'], 'attrs.' + key, expected, actual)
                if len(inputs) != 18 or len(n.get('input', [])) != 18:
                    mismatch(n['name'], 'input_slots', {'descriptors': 18, 'references': 18},
                             {'descriptors': len(inputs), 'references': len(n.get('input', []))})
                required = {0: (expected_attention['query_shape'], 'DT_FLOAT16'),
                    1: (expected_attention['kv_shape'], 'DT_FLOAT16'),
                    2: (expected_attention['kv_shape'], 'DT_FLOAT16'),
                    4: (expected_attention['mask_shape'], 'DT_FLOAT16'),
                    5: ([1], 'DT_INT64'), 6: ([1], 'DT_INT64'),
                    14: (expected_attention['block_table_shape'], 'DT_INT32'), 16: ([1], 'DT_INT64')}
                for i, (shape, dtype) in required.items():
                    actual = inputs[i] if i < len(inputs) else {}
                    role = _ATTENTION_PORTS[i]
                    if not actual.get('producer'):
                        mismatch(n['name'], f'inputs[{i}].reference', 'RESOLVED_' + role, actual.get('reference'))
                    for label in ('name', 'mapped_name'):
                        allowed_names = (role, role[:-1]) if role in {'key0', 'value0'} else (role,)
                        if actual.get(label) and actual[label] not in allowed_names:
                            mismatch(n['name'], f'inputs[{i}].' + label, role, actual[label])
                    if actual.get('shape') is None:
                        deferred_shapes.append(dict(node=n['name'], slot=i, role=role, expected=shape,
                            actual=None, status='DEFERRED_TO_OMG_SHAPE_INFERENCE',
                            reason='SOURCE_DESCRIPTOR_DIMENSIONS_NOT_RECORDED'))
                    elif actual['shape'] != shape:
                        mismatch(n['name'], f'inputs[{i}].shape', shape, actual.get('shape'))
                    if dtype not in actual.get('dtype_names', []):
                        mismatch(n['name'], f'inputs[{i}].dtype', dtype, actual.get('dtype_names'))
        if n['type'] in {'Gather', 'GatherV2', 'GatherV2D'}:
            inputs = n.get('input_desc', [])
            indices = desc(inputs[1]) if len(inputs) > 1 else {}
            gathers.append(dict(name=n['name'], type=n['type'], indices=indices))
            if require_int32_indices and 'DT_INT32' not in indices.get('dtype_names', []):
                mismatch(n['name'], 'Gather.indices.dtype', 'DT_INT32', indices.get('dtype_names', []))
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
    composed_attention = []
    if composed:
        from .torch_attention_inspect import inspect_composed_attention
        composed_attention = inspect_composed_attention(nodes, expected_attention, types, mismatch)
    inputs = [n for n in nodes if n['type'] == 'Data']
    outputs = [n for n in nodes if n['type'] == 'NetOutput']
    result = dict(node_types=dict(counts), rms=norms, gather_indices=gathers, attention=attention,
                composed_attention=composed_attention, zeros_like=zeros, weight_files=sidecars,
                validation_status='FAILED' if errors else 'PASSED_WITH_DEFERRED_SHAPES' if deferred_shapes else 'PASSED',
                validation_errors=errors, deferred_shapes=deferred_shapes,
                input_count=len(inputs), output_counts=[len(n.get('input_desc', [])) for n in outputs],
                scope='SOURCE_GRAPH_ONLY_NOT_COMPILED_PRECISION_OR_RUNTIME_ABI')
    if strict and errors:
        raise AirValidationError(result)
    return result


def inspect_pytorch_air(directory):
    """Read a successful or failed export in place; no TorchAir, model or writes."""
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('artifact_kind') != AIR_KIND:
        raise ValueError('expected a retained PyTorch Draft AIR export directory')
    inspection = inspect_air(root / 'export.air', _schema((root / 'air-schema.pb').read_bytes(), manifest['schema']),
        expected_norms=32, require_int32_indices=True, expected_attention=manifest.get('attention'), strict=False)
    errors = inspection['validation_errors']
    first = next((node for node in inspection['attention'] if errors and node['name'] == errors[0]['node']),
                 inspection['attention'][0] if inspection['attention'] else None)
    return dict(status='INSPECTED', source_status=manifest.get('status'), source_stage=manifest.get('stage'),
        source_error=manifest.get('error'), air=str(root / 'export.air'),
        validation_status=inspection['validation_status'], first_mismatch=errors[0] if errors else None,
        mismatch_count=len(errors), mismatches=errors[:24], mismatches_truncated=len(errors) > 24,
        deferred_shape_count=len(inspection['deferred_shapes']), deferred_shapes=inspection['deferred_shapes'],
        attention_nodes=len(inspection['attention']), first_attention=first,
        composed_attention=inspection['composed_attention'],
        zeros_like=inspection['zeros_like'],
        node_types=inspection['node_types'], input_count=inspection['input_count'], output_counts=inspection['output_counts'],
        weight_files=len(inspection['weight_files']), expected_attention=manifest.get('attention'),
        scope='READ_ONLY_RETAINED_AIR; PRODUCER_DESCRIPTORS_ARE_DIAGNOSTIC_ONLY_NOT_SUBSTITUTED',
        model_export='NOT_RUN', model_compilation='NOT_RUN', device_execution='NOT_RUN')


def export_pytorch(prepared, output, *, context_rows=16, cache_capacity=192, max_draft_tokens=15,
                   replace=False, torchair_pb2=None, progress=print, optimization='baseline'):
    from .optimization import optimization_contract
    contract = optimization_contract(optimization)
    from qwen35_moe_dflash.toolchain.export import require_torchair
    torchair = require_torchair()
    import torch
    from .torch_model import load_prepared
    # Keep the installed RMS converter. Attention exports ordinary Torch
    # MatMul/Select/Softmax operations; do not register the failed fused call.
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
                                  max_draft_tokens=max_draft_tokens, rms=rms, optimization=optimization)
            result.update(**model.abi(), schema={k: metadata[k] for k in ('schema_file', 'schema_package')},
                torch_version=torch.__version__, torchair_version=getattr(torchair, '__version__', 'unknown'),
                attention=model.attention_contract(),
                optimization=contract, projection=contract['projection'], stage='torchair-export')
            _save(root / 'manifest.json', result)
            (root / 'air-schema.pb').write_bytes(descriptor)
            progress('TorchAir export: 17 inputs, 14 outputs, 6 composed MatMul/Softmax attention layers; committed ND KV retained')
            with working_directory(root), torch.inference_mode():
                torchair.dynamo_export(*model.example_inputs(), model=model, export_path='weight', export_name='export', dynamic=False)
            # Relative export_path keeps FileConstant paths portable on the DDK
            # server; move only graph metadata, leaving the weight files in place.
            (root / 'weight/export.air').replace(root / 'export.air')
            for name in ('dynamo.pbtxt', 'dynamo.txt'):
                if (root / 'weight' / name).is_file():
                    (root / 'weight' / name).replace(root / name)
            result['stage'] = 'air-validation'
            inspection = inspect_air(root / 'export.air', _schema(descriptor, metadata),
                                     expected_norms=32, require_int32_indices=True, expected_attention=result['attention'])
            if inspection['input_count'] != 17 or inspection['output_counts'] != [14]:
                raise ValueError('TorchAir source boundary does not contain all 17 inputs / 14 outputs')
            _save(root / 'air-inspection.json', inspection)
            _finish_air_manifest(result, inspection)
        except Exception as error:
            if isinstance(error, AirValidationError):
                _save(root / 'air-inspection.json', error.inspection)
                result['first_mismatch'] = error.inspection['validation_errors'][0]
                result['inspection'] = str(root / 'air-inspection.json')
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
    from qwen35_moe_dflash.toolchain.omc_embed_weights import _attribute
    schema = load_schema(Path(ddk) / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    parts = [p for p in read_container(omc)['partitions'] if p['type'] == 0]
    if len(parts) != 1:
        raise ValueError('expected one compiled model metadata partition')
    model = schema.decode(read_partition(omc, parts[0]), 'ModelDef')
    enums, owners, nodes, fused_attention = _enum_fields(schema), Counter(), [], []
    for graph in model.get('graph', []):
        for node in graph.get('op', []):
            if 'FusedInferAttention' in node['type'] or node['type'] in {
                    'FlashAttentionScore', 'PromptFlashAttention', 'IncreFlashAttention'}:
                fused_attention.append(dict(name=node['name'], type=node['type']))
            if node['type'] not in {'Data', 'NetOutput', 'GraphOp', 'Cast', 'CastT'}:
                continue
            row = dict(name=node['name'], type=node['type'], input=node.get('input', []),
                       inputs=[_observed_desc(d, enums) for d in node.get('input_desc', [])],
                       outputs=[_observed_desc(d, enums) for d in node.get('output_desc', [])])
            if node['type'] == 'GraphOp':
                row['owner'] = _task(node)['cl_name']
                owners[row['owner']] += 1
                row['weight_bytes'] = (_attribute(node, 'graphop_weight_size')
                    if any(a.get('key') == 'graphop_weight_size' for a in node.get('attr', [])) else None)
            nodes.append(row)
    return dict(graphop_counts=dict(owners), nodes=nodes, fused_attention_nodes=fused_attention,
                attention_inspection_scope='MAIN_GRAPH_METADATA_ONLY; OPAQUE_GRAPHOP_KERNELS_NOT_DECODED',
                scope='COMPILED_DESCRIPTORS_ONLY_NOT_RUNTIME_SLOT_BINDING_OR_INTERNAL_INSTRUCTION_PRECISION')


def _partition_weights(observed):
    # Diagnostic threshold: the public Secure C implementation rejects a
    # destMax above 0x7fffffff. Actual runtime copy arguments are not observed.
    limit = 0x7fffffff
    rows = [dict(name=n['name'], owner=n['owner'], bytes=n.get('weight_bytes'))
            for n in observed.get('nodes', []) if n['type'] == 'GraphOp']
    known = [r for r in rows if type(r['bytes']) is int and r['bytes'] >= 0]
    oversized = [r['name'] for r in known if r['bytes'] > limit]
    missing = [r['name'] for r in rows if r not in known]
    complete = bool(rows) and not missing
    return dict(status=('EXCEEDS_DIAGNOSTIC_LIMIT' if oversized else
                        'WITHIN_DIAGNOSTIC_LIMIT' if complete else 'INCOMPLETE'),
                diagnostic_limit_bytes=limit, partitions=rows,
                max_known_bytes=max((r['bytes'] for r in known), default=None),
                oversized=oversized, missing=missing,
                scope='COMPILED_GRAPHOP_WEIGHT_EXTENTS_NOT_RUNTIME_ALLOCATION_OR_COPY_ARGUMENTS')


def inspect_pytorch_compilation(directory):
    """Inspect a retained candidate without the AIR, SDK or compiled payloads."""
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('artifact_kind') != OMC_KIND:
        raise ValueError('expected a PyTorch Draft OMC compilation directory')
    return dict(status='INSPECTED', source_status=manifest.get('status'),
                source_stage=manifest.get('stage'), error=manifest.get('error'),
                air_preparation=manifest.get('air_preparation'),
                compilation_diagnostics=compilation_diagnostics(root / 'compilation'),
                scope='RETAINED_COMPILER_REPORT_AND_LOGS_ONLY_ROOT_CAUSE_NOT_CLASSIFIED',
                model_compilation='NOT_RUN', device_execution='NOT_RUN')


def compile_pytorch(air_export, output, *, ddk_root, ddk_id, replace=False,
                    timeout_seconds=7200, tbe_diagnosis=None, progress=print,
                    limit_partition_size=None):
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics
    if limit_partition_size is not None and type(limit_partition_size) is not bool:
        raise ValueError('limit_partition_size must be bool or None')

    def compiler_progress(row):
        message = row['message']
        if row.get('stage') == 'air-preparation':
            message += (f": {row['status']}, removed {row['removed_count']} intermediate Reshapes, "
                        f"materialized {row.get('zero_count', 0)} ZerosLike constants, "
                        f"materialized {row.get('fill_count', 0)} Fill constants, "
                        f"folded {row.get('weight_transpose_count', 0)} weight Transposes, "
                        f"bypassed {row.get('unit_scale_count', 0)} embedding unit scales; {row['path']}")
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
    schema = _schema(descriptor, metadata)
    inspection = inspect_air(source / 'export.air', schema,
                             expected_norms=32, require_int32_indices=True, expected_attention=manifest.get('attention'))
    def prepare_air(raw):
        from .air_reshape import fold_reshape_chains
        from .air_zeros import materialize_zeros
        from .air_fill import materialize_fills
        from .air_matmul import fold_weight_transposes
        from .air_embedding import fold_embedding_unit_scale
        prepared, evidence = fold_reshape_chains(raw, schema)
        prepared, zeros = materialize_zeros(prepared, schema)
        prepared, fills = materialize_fills(prepared, schema)
        prepared, transposes = fold_weight_transposes(prepared, schema)
        prepared, embedding = fold_embedding_unit_scale(prepared, schema)
        evidence.update(reshape_policy=evidence['policy'], policy='draft-static-preparation-v5',
                        zero_count=zeros['zero_count'], zeros_like=zeros,
                        fill_count=fills['fill_count'], fills=fills,
                        weight_transpose_count=transposes['weight_transpose_count'], weight_transposes=transposes,
                        unit_scale_count=embedding['unit_scale_count'], embedding_unit_scale=embedding,
                        compiler_sha256=embedding['compiler_sha256'])
        if zeros['zero_count'] or fills['fill_count'] or transposes['weight_transpose_count'] or embedding['unit_scale_count']:
            evidence['status'] = 'APPLIED'
        return prepared, evidence
    with _slot(output, OMC_KIND, replace) as root:
        result = {k: manifest[k] for k in ('context_rows', 'cache_capacity', 'max_draft_tokens', 'inputs', 'outputs', 'rms', 'gamma')}
        for key in ('attention', 'optimization', 'projection'):
            if key in manifest:
                result[key] = manifest[key]
        result.update(artifact_kind=OMC_KIND, status='COMPILING', stage='compile', frontend='pytorch',
            source_air_validation=inspection['validation_status'],
            source_deferred_shape_count=len(inspection['deferred_shapes']),
            model_compilation='NOT_RUN', compiled_precision='NOT_VALIDATED', compiled_abi='NOT_BOUND',
            device_execution='NOT_RUN', numerical_validation='NOT_RUN', ddk_id=ddk_id,
            runtime_integration='PENDING_PHYSICAL_ABI_AND_DEVICE_VALIDATION')
        if limit_partition_size is not None:
            result['compiler_options'] = dict(limit_partition_size=limit_partition_size)
        _save(root / 'manifest.json', result)
        try:
            transform = None
            if manifest.get('attention', {}).get('policy') in {'target-cpro-fused-attention-v1', COMPOSED_ATTENTION}:
                # Validate the small source graph before copying multi-GB
                # sidecars or invoking OMG --help. The snapshot must still
                # match the bytes that were prepared here.
                result['stage'] = 'air-preparation'
                _save(root / 'manifest.json', result)
                progress('Preparing static AIR before weight snapshots or OMG invocation')
                with (source / 'export.air').open('rb') as stream:
                    source_raw = stream.read((64 << 20) + 1)
                prepared_raw, preparation = prepare_air(source_raw)
                def transform(snapshot):
                    with snapshot.open('rb') as stream:
                        if stream.read((64 << 20) + 1) != source_raw:
                            raise ValueError('source AIR changed after static preparation; snapshot differs')
                    return prepared_raw, preparation
            result['stage'] = 'compile'
            _save(root / 'manifest.json', result)
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
                limit_partition_size=limit_partition_size,
                tbe_diagnosis=tbe_diagnosis, progress=compiler_progress,
                air_transform=transform)
            result['model_compilation'] = compiled['model_compilation']
            if 'air_preparation' in compiled:
                evidence = compiled['air_preparation']
                result['air_preparation'] = {key: evidence[key] for key in
                    ('policy', 'status', 'removed_count', 'zero_count', 'fill_count', 'weight_transpose_count', 'unit_scale_count')}
                result['air_preparation']['report'] = 'compilation/air-preparation.json'
            if compiled['status'] != 'COMPILED':
                raise RuntimeError(str(compiled.get('error', compiled['status'])))
            model_path = root / 'compilation' / compiled['omc']['path']
            result['stage'] = 'compiled-interface'
            try:
                observed = _compiled_observation(model_path, ddk_root)
                _save(root / 'compiled-interface.json', observed)
                result['graphop_counts'] = observed['graphop_counts']
                result['partition_weights'] = _partition_weights(observed)
                result['interface_inspection'] = 'OBSERVED'
            except (OSError, ValueError, KeyError) as error:
                result.update(interface_inspection='UNAVAILABLE', interface_error=str(error))
                if manifest.get('attention', {}).get('policy') == COMPOSED_ATTENTION:
                    raise
            if manifest.get('attention', {}).get('policy') == COMPOSED_ATTENTION:
                result['composed_attention'] = dict(source_layers=len(inspection['composed_attention']),
                    source_fused_nodes=len(inspection['attention']),
                    compiled_main_graph_fused_nodes=observed['fused_attention_nodes'],
                    scope=observed['attention_inspection_scope'])
                if observed['fused_attention_nodes']:
                    raise ValueError('compiler main graph contains a fused attention operator; retain this candidate for inspection')
            if limit_partition_size is True:
                result['stage'] = 'partition-weight-check'
                if result.get('partition_weights', {}).get('status') != 'WITHIN_DIAGNOSTIC_LIMIT':
                    raise ValueError('bounded-partition candidate requires known GraphOp weight extents <= 0x7fffffff; '
                                     'see partition_weights and compiled-interface.json; packed model not published')
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
            if result['stage'] == 'air-preparation':
                result['next'] = 'Inspect the reported source AIR node descriptors; no weight snapshot or OMG invocation occurred.'
            elif result['stage'] == 'partition-weight-check':
                result['next'] = 'Inspect compiled partition weight extents; this requested limit did not produce a bounded candidate. No packed model or device execution.'
            else:
                result['compilation_diagnostics'] = compilation_diagnostics(root / 'compilation')
                result['next'] = 'Inspect compilation_diagnostics and the retained compile.log before retrying this same build directory.'
        _save(root / 'manifest.json', result)
        return result

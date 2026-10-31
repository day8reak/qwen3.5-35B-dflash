"""Compile and pack one incremental Draft gear; no model or board execution."""
from __future__ import annotations

from pathlib import Path
import shutil

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.draft.capacity_export import _metadata, _stable, _sdk_path
from qwen35_moe_dflash.draft.compile_precision import collect_precision_plan, verify_compiled_precision
from qwen35_moe_dflash.draft.precision_policy import customization_bytes
from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.draft.incremental.abi import inspect_incremental_abi
from qwen35_moe_dflash.draft.incremental.inputs import bind_incremental_export, _payload
from qwen35_moe_dflash.toolchain.omc_embed_weights import embed_external_weights
from qwen35_moe_dflash.execution.run_layout import relocated_path


def _compiler_source(bound, *, ddk_root=None, ddk_id=None):
    manifest, bindings = bound['manifest'], bound['bindings']
    declared = manifest.get('compiler', {})
    if ddk_root is not None or ddk_id is not None or declared:
        root, identity = ddk_root or declared.get('ddk_root'), ddk_id or declared.get('ddk_id')
        if not root or not identity:
            raise ValueError('declare both --ddk-root and --ddk-id for a standalone AIR')
        ddk = _sdk_path(root, directory=True)
        omg = _sdk_path(ddk / 'tools/tools_omg/master/omg')
        schema = _sdk_path(ddk / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
        if ddk not in omg.parents or ddk not in schema.parents:
            raise ValueError('compiler and schema must belong to the selected DDK')
        return dict(ddk_id=identity, omg_sha256=sha256_file(omg)), ddk, omg, schema
    if 'source_bundle_manifest' not in manifest:
        raise ValueError('standalone AIR requires --ddk-root and --ddk-id')
    bundle, bundle_id = _metadata(relocated_path(manifest['source_bundle_manifest']['path'], bound['root']), bindings)
    if any(bundle_id[k] != manifest['source_bundle_manifest'].get(k) for k in ('bytes', 'sha256', 'full_identity')):
        raise ValueError('original successful bundle metadata changed')
    previous = Path(bundle_id['path']).parent.parent / 'compilation/report.json'
    compiled, compiled_id = _metadata(previous, bindings)
    if (compiled_id['sha256'] != bundle.get('compiler_report_sha256')
            or compiled.get('status') != 'COMPILED' or compiled.get('model_compilation') != 'PASSED'):
        raise ValueError('selected compiler must bind the successful original Draft compilation')
    ddk = _sdk_path(compiled['ddk_root'], directory=True)
    omg = _sdk_path(compiled['omg_path'])
    if ddk not in omg.parents or sha256_file(omg) != compiled['omg_sha256']:
        raise ValueError('original selected OMG changed or is outside the DDK')
    schema = _sdk_path(ddk / 'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
    if ddk not in schema.parents:
        raise ValueError('schema must belong to the selected compiler DDK')
    return compiled, ddk, omg, schema


def _record(path, root):
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def compile_incremental_draft(export_dir, output, *, timeout_seconds=7200, tbe_diagnosis=None,
                              progress=None, ddk_root=None, ddk_id=None):
    root = output_path(output)
    export = Path(export_dir).absolute()
    if root == export or export in root.parents:
        raise ValueError('compile output must be outside the incremental AIR directory')
    if root.exists():
        raise FileExistsError('use a new incremental compilation output: ' + str(root))
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be a positive integer')
    # Check protected source directories before creating any output directory.
    exported_meta, _ = _metadata(export / 'manifest.json', [])
    prepared_path = exported_meta['source_prepared_root']
    protected_roots = [(export / prepared_path).resolve() if exported_meta.get('source_policy') == 'portable-prepared-weights-v1'
                       else relocated_path(prepared_path, export)]
    if exported_meta.get('source_job', {}).get('path'):
        protected_roots.append(relocated_path(exported_meta['source_job']['path'], export).parent)
    for protected in protected_roots:
        if root == protected or protected in root.parents:
            raise ValueError('compile output must be outside the original source job/preparation')
    root.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'artifact_kind': 'qwen35-incremental-draft-compiled',
        'status': 'FAILED', 'stage': 'bind-export', 'output': str(root),
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN',
        'numerical_validation': 'NOT_RUN_INCREMENTAL_DEVICE',
        'runtime_integration': 'PENDING_COMPILED_MODEL_AND_DEVICE_RUNTIME_CHECK',
        'full_loop_smoke': 'NOT_RUN_NEW_INCREMENTAL_MODEL', 'weight_files_copied': 0,
        'next': 'Complete compilation/ABI/packing before running the incremental board-local smoke'}
    bound = None
    notify = progress or (lambda message: None)
    try:
        notify('Binding new AIR, selected gear and all original weights; no weight copies')
        bound = bind_incremental_export(export)
        manifest = bound['manifest']
        result.update(**{k: manifest[k] for k in ('context_rows', 'cache_capacity', 'max_draft_tokens', 'draft_config', 'incremental_abi')},
            export_manifest=bound['identity'])
        previous, ddk, omg, schema = _compiler_source(bound, ddk_root=ddk_root, ddk_id=ddk_id)
        schema_hash = sha256_file(schema)
        result.update(ddk_id=previous['ddk_id'], ddk_root=str(ddk), platform='kirinx90',
            schema_library={'path': str(schema), 'sha256': schema_hash})
        # Output weights and a second, packed container must coexist. This is a
        # preflight allowance, not a promise about arbitrary compiler scratch.
        required = 2 * manifest['source_weight_bytes'] + (1 << 30)
        free = shutil.disk_usage(root).free
        result['disk_budget'] = {'available_bytes': free, 'required_bytes': required,
            'scope': 'Two weight extents plus 1 GiB metadata/scratch allowance; not a compiler peak-space guarantee'}
        if free < required:
            raise OSError(f'incremental compilation and packing need at least {required} free bytes; available={free}')
        result['stage'] = 'precision-policy'
        plan = collect_precision_plan(bound['recipe'])
        residuals = [r for r in plan['required_nodes'] if r['name'].startswith('__draft_residual/') and r['type'] == 'Add']
        if len(residuals) != 12:
            raise ValueError('new recipe must preserve all twelve FP32 residual sums')
        save_json(root / 'precision-plan.json', plan)
        policy = root / 'cl-customization.json'
        policy.write_bytes(customization_bytes(plan))
        result['stage'] = 'omg'
        notify('Compiling new incremental AIR with the successful Draft DDK/options')
        compile_root = root / 'compilation'
        selected_diagnosis = tbe_diagnosis or previous.get('tbe_compatibility', {}).get('diagnosis')
        if selected_diagnosis:
            selected_diagnosis = relocated_path(selected_diagnosis, export)
        compiled = compile_existing_air(bound['air'], compile_root, ddk, ddk_id=previous['ddk_id'],
            omg_path=omg, tbe_diagnosis=selected_diagnosis, timeout_seconds=timeout_seconds,
            use_origin_format=True, weight_data_type='FP16', limit_partition_size=False,
            save_weights_as_external_data=True, cl_customization=policy,
            incremental_draft_export=export, progress=lambda row: notify(row['message']))
        result['compilation_report'] = _record(compile_root / 'report.json', root)
        result['model_compilation'] = compiled['model_compilation']
        if compiled['status'] != 'COMPILED':
            raise RuntimeError('incremental OMG failed at ' + str(compiled.get('stage')) + ': ' + str(compiled.get('error')))
        if compiled.get('incremental_input_view', {}).get('status') != 'PASSED':
            raise ValueError('incremental original-weight reference audit did not pass')
        if sha256_file(omg) != previous['omg_sha256'] or sha256_file(schema) != schema_hash:
            raise ValueError('selected compiler or schema changed')
        omc = compile_root / compiled['omc']['path']
        _payload(omc, compiled['omc'])
        result['stage'] = 'compiled-precision'
        notify('Checking compiled FP32 arithmetic and complete incremental input/output ABI')
        precision = verify_compiled_precision(omc, schema, plan)
        save_json(root / 'compiled-precision.json', precision)
        result['compiled_precision'] = precision['status']
        # Inspect ABI even if precision failed, so one compiler run yields both
        # useful reports. Never pack/promote a failed contract.
        abi = inspect_incremental_abi(omc, schema, manifest)
        save_json(root / 'compiled-abi.json', abi)
        result['compiled_abi'] = abi['status']
        if precision['status'] != 'PASSED':
            raise ValueError('compiled FP32 placement/dtype gate failed; see compiled-precision.json')
        result['stage'] = 'compiled-abi'
        if abi['status'] != 'PASSED':
            raise ValueError('compiled incremental ABI failed: ' + abi.get('error', 'unknown error'))
        result['stage'] = 'pack-model'
        notify('Streaming new compiled weights into the new embedded OMC')
        destination = root / 'model/model.omc'
        destination.parent.mkdir()
        sidecars = [dict(row, path=str(compile_root / row['path'])) for row in compiled['external_data_candidates']
                    if Path(row['path']).suffix == '.weight']
        packing = embed_external_weights(omc, destination, schema,
            expected_source_sha256=compiled['omc']['sha256'], expected_schema_sha256=schema_hash,
            sidecars=sidecars, progress=notify)
        save_json(root / 'weight-packing.json', packing)
        model = _record(destination, root)
        if packing.get('status') != 'EMBEDDED' or any(packing['output'][k] != model[k] for k in ('bytes', 'sha256')):
            raise ValueError('new packed model differs from packing receipt')
        result.update(model=model, weight_storage='embedded-stream',
            inputs=abi['inputs'], outputs=abi['outputs'], compiled_abi_file=_record(root / 'compiled-abi.json', root),
            weight_packing=_record(root / 'weight-packing.json', root),
            status='DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND', stage='complete',
            runtime_integration='READY_FOR_INCREMENTAL_RESIDENT_ADAPTER_DEVICE_NOT_RUN',
            next='Use run_board_serial_dflash.py --incremental-draft with this directory and its exact --max-draft-tokens gear; retain the original S64 bundle')
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if bound is not None:
            try:
                _stable(bound['bindings'], bound['sources'])
                result['source_preservation'] = 'ORIGINAL_EXPORT_AND_PREPARED_IDENTITIES_UNCHANGED'
            except (OSError, ValueError) as error:
                result.update(status='FAILED', stage='source-recheck', error=str(error), source_preservation='CHANGED')
        save_json(root / 'manifest.json', result)
    return result

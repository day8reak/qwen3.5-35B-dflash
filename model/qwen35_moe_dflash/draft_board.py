"""Deploy one hash-bound complete draft OMC without loading a target model.

The native step runner executes three inputs and two outputs, including the
pre-head hidden diagnostic. Synthetic-feature agreement is not language quality,
acceptance-rate or accelerator-placement evidence.
"""
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import uuid

import numpy as np

from .artifacts import output_path, save_json
from .config import Qwen35DFlashConfig
from .elf_dynamic import dynamic_info
from .existing_air import _snapshot
from .hashing import sha256_file
from .hdc_run import Hdc, _file_record, _relative, _text, _upload
from .hiai_deployment import BOARD_LIBRARY_SEARCH_PATHS, DDK_HCL_EXCLUDED, candidate_blockers, discover_libraries
from .resources import repository_root


def _json(path):
    return json.loads(Path(path).read_text())


def _file(root, record):
    relative = _relative(record['path'])
    path = Path(root) / relative
    for part in [path, *path.parents]:
        if part == Path(root).parent:
            break
        if part.is_symlink():
            raise ValueError('draft payload must not use symlinks: ' + str(path))
    if (not path.is_file() or type(record.get('bytes')) is not int or record['bytes'] < 0
            or path.stat().st_size != record['bytes'] or sha256_file(path) != record.get('sha256')):
        raise ValueError('draft payload SHA256/size mismatch: ' + str(path))
    return path


def _record(path, root):
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def _io(sequence):
    return ([{'index': 0, 'name': 'target_features', 'dtype': 'float16', 'shape': [1,sequence,16384]},
             {'index': 1, 'name': 'input_ids', 'dtype': 'int32', 'shape': [1,sequence]},
             {'index': 2, 'name': 'attention_mask', 'dtype': 'int32', 'shape': [1,sequence]}],
            [{'index': 0, 'name': 'draft_top1', 'dtype': 'int32', 'shape': [1,15]},
             {'index': 1, 'name': 'draft_hidden', 'dtype': 'float16', 'shape': [1,15,2048]}])


def _wire(manifest):
    model, abi = manifest['model'], manifest['draft_abi']
    lines = ['QWEN35_DRAFT_STEP\t1', f"model\t{model['path']}\t{model['sha256']}",
             f"draft_abi\t{abi['path']}\t{abi['sha256']}", f"tokens\t{manifest['sequence_length']}\t0"]
    for row in manifest['inputs']:
        lines.append('\t'.join(('input', str(row['index']), row['name'], row['dtype'],
                                 ','.join(map(str,row['shape'])), row['path'], row['sha256'])))
    for row in manifest['outputs']:
        lines.append('\t'.join(('output', str(row['index']), row['name'], row['dtype'], ','.join(map(str,row['shape'])))))
    return ('\n'.join(lines)+'\n').encode('ascii')


def _compiler_payload_layout(report):
    """Preserve the compiler's generated tree, including nested OMC sidecars."""
    records = report.get('generated_artifacts', [])
    by_path, planned = {}, []
    for record in records:
        source_relative = _relative(record['path'])
        try:
            generated_relative = source_relative.relative_to('compile-work/generated')
        except ValueError as error:
            raise ValueError('compiler payload is outside the generated tree: '+record['path']) from error
        if not generated_relative.parts or record['path'] in by_path:
            raise ValueError('duplicate or invalid compiler payload path: '+record['path'])
        by_path[record['path']] = record
        planned.append((record, PurePosixPath('model') / generated_relative))
    primary = report.get('omc', {})
    model = by_path.get(primary.get('path'))
    if (model is None or PurePosixPath(model['path']).suffix != '.omc'
            or type(model.get('bytes')) is not int or model['bytes'] <= 0
            or any(model.get(key) != primary.get(key) for key in ('bytes', 'sha256'))):
        raise ValueError('primary OMC missing from compiler output inventory')
    for sidecar in report.get('external_data_candidates', []):
        retained = by_path.get(sidecar.get('path'))
        if retained is None or any(retained.get(key) != sidecar.get(key) for key in ('bytes', 'sha256')):
            raise ValueError('external compiler payload missing from generated inventory')
    return planned


def prepare_draft_bundle(compilation_dir, fixture_dir, output_dir, *, config, ddk_id, export_dir=None,
                         weight_storage=None, schema_library=None):
    """Snapshot the existing compiler payloads and fixture; never compile a model.

    Pass export_dir to bind compiler AIR snapshots to the exported draft, and
    the reference fixture to the same prepared checkpoint payload identity.
    """
    compiler, fixture = Path(compilation_dir).resolve(), Path(fixture_dir).resolve()
    report, reference = _json(compiler/'report.json'), _json(fixture/'report.json')
    cfg = Qwen35DFlashConfig.from_dict(config)
    if (cfg.num_hidden_layers, cfg.feature_size, cfg.hidden_size, cfg.block_size, cfg.vocab_size) != (6,16384,2048,16,248320):
        raise ValueError('standalone native runner requires the locked full six-layer draft')
    if (report.get('artifact_kind') != 'qwen35-existing-air-compilation' or report.get('status') != 'COMPILED'
            or report.get('model_compilation') != 'PASSED' or report.get('ddk_id') != ddk_id
            or report.get('platform') != 'kirinx90'):
        raise ValueError('draft requires successful matching-DDK kirinx90 compilation')
    planned_payloads = _compiler_payload_layout(report)
    compiled_storage = ('embedded' if report.get('requested_options', {}).get(
        'save_weights_as_external_data') == 'false' else 'external')
    weight_storage = weight_storage or compiled_storage
    if weight_storage not in {'external', 'embedded', 'embedded-stream'}:
        raise ValueError('unknown draft bundle weight storage')
    if weight_storage != compiled_storage and weight_storage != 'embedded-stream':
        raise ValueError('bundle weight storage differs from compiler output')
    if weight_storage == 'embedded-stream':
        if report.get('requested_options', {}).get('save_weights_as_external_data') != 'true':
            raise ValueError('embedded-stream requires the retained external-weight compilation')
        # Bind the decoder to the selected compiler installation. Do not import
        # or execute vendor libraries, and do not infer a protobuf schema.
        ddk = Path(report['ddk_root']).resolve()
        omg = ddk/'tools/tools_omg/master/omg'
        if sha256_file(omg) != report.get('omg_sha256'):
            raise ValueError('selected OMG changed since this compilation')
        schema_library = Path(schema_library).resolve() if schema_library else (
            ddk/'tools/tools_omg/master/lib64/libai_fmk_graph_proto_host.so')
        if not schema_library.is_relative_to(ddk):
            raise ValueError('streamed OMC schema must belong to the recorded compiler DDK')
    if (reference.get('artifact_kind') != 'qwen35-draft-reference'
            or reference.get('status') not in {'REFERENCE_COMPLETE','INPUTS_PREPARED'}
            or reference.get('draft_config') != config):
        raise ValueError('draft fixture configuration or status mismatch')
    sequence = reference.get('sequence_length')
    if type(sequence) is not int or not 0 < sequence <= 16384 or sequence + cfg.block_size > cfg.max_position_embeddings:
        raise ValueError('draft sequence exceeds position/input-buffer bounds')
    expected_inputs, outputs = _io(sequence)
    if len(reference.get('inputs', [])) != 3:
        raise ValueError('draft fixture must provide all three ordered inputs')
    for actual, expected in zip(reference['inputs'], expected_inputs):
        if any(actual.get(key) != value for key,value in expected.items()):
            raise ValueError('draft fixture ordered input ABI mismatch')
        _file(fixture, actual)
    source_identity = 'UNVERIFIED'
    exported_hash = None
    if export_dir is not None:
        exported = Path(export_dir).resolve()
        exported_manifest = _json(exported/'manifest.json')
        if (exported_manifest.get('artifact_kind') != 'qwen35-cpro-draft-air'
                or exported_manifest.get('status') != 'EXPORTED'
                or len(exported_manifest.get('graphs', [])) != 1
                or exported_manifest['graphs'][0].get('name') != 'draft'):
            raise ValueError('export identity does not describe one complete draft')
        graph = exported_manifest['graphs'][0]
        if graph.get('outputs') != ['draft_top1','draft_hidden']:
            raise ValueError('standalone draft export must include pre-head hidden diagnostics')
        air = next((row for row in graph['files'] if row['path'] == graph['air']), None)
        if air is None:
            raise ValueError('draft AIR missing from exported inventory')
        _file(exported, air)
        snapshots = [row for row in report.get('input_snapshots',[]) if row.get('role') == 'air']
        if len(snapshots) != 1 or any(snapshots[0].get(k) != air[k] for k in ('bytes','sha256')):
            raise ValueError('compiled AIR differs from exported draft')
        snapshots_by_relative={row['source_relative']:row for row in report.get('input_snapshots',[])}
        for payload in graph['files']:
            _file(exported,payload)
            snapshot=snapshots_by_relative.get(payload['path'])
            if snapshot is None or any(snapshot.get(k)!=payload[k] for k in ('bytes','sha256')):
                raise ValueError('compiler snapshot differs from exported draft payload: '+payload['path'])
            _file(compiler,snapshot)
        prepared_hash = exported_manifest.get('prepared_manifest_sha256')
        if not prepared_hash or prepared_hash != reference.get('prepared_manifest_sha256'):
            raise ValueError('draft export and reference use different prepared weight manifests')
        source_identity = 'MATCHED_EXPORT_AND_REFERENCE'
        exported_hash = sha256_file(exported/'manifest.json')
    root = output_path(output_dir)
    root.mkdir(parents=True,exist_ok=False)
    payloads, by_path = [], {}
    packing = None
    if weight_storage == 'embedded-stream':
        from .omc_embed_weights import embed_external_weights
        source = _file(compiler, report['omc'])
        sidecars = [{**record, 'path': str(_file(compiler, record))}
                    for record, _ in planned_payloads if Path(record['path']).suffix == '.weight']
        destination = root/'model/model.omc'
        destination.parent.mkdir(parents=True)
        print('draft-bundle: streaming compiled weights into one OMC; no AIR export or OMG compilation', flush=True)
        packing = embed_external_weights(source, destination, schema_library,
            expected_source_sha256=report['omc']['sha256'], sidecars=sidecars,
            progress=lambda message: print('draft-bundle: '+message, flush=True))
        model = _record(destination, root)
        if (packing.get('status') != 'EMBEDDED'
                or any(packing.get('output', {}).get(key) != model[key] for key in ('bytes','sha256'))):
            raise ValueError('streamed OMC differs from weight-packing receipt')
        save_json(root/'weight-packing.json', packing)
        payloads.append(model)
    else:
        for record, relative in planned_payloads:
            source = _file(compiler, record)
            # Preserve every relative path under generated. In particular, an OMG
            # model/model.omc keeps its sibling model/SubGraph_N.weight files.
            destination = root/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            _snapshot(source,destination)
            copied = _record(destination,root)
            if any(copied[k] != record[k] for k in ('bytes','sha256')):
                raise ValueError('copied compiler payload differs')
            payloads.append(copied)
            by_path[record['path']] = copied
        model = by_path.get(report.get('omc',{}).get('path'))
        if model is None or model['bytes'] <= 0 or any(model[k] != report['omc'][k] for k in ('bytes','sha256')):
            raise ValueError('primary OMC missing from compiler output inventory')
    inputs = []
    for row in reference['inputs']:
        destination = root/f"input_{row['index']:04d}.bin"
        _snapshot(_file(fixture,row),destination)
        inputs.append({**row,**_record(destination,root)})
    expected = []
    for key, desc in zip(('expected_top1','expected_hidden'),outputs):
        row = reference.get(key)
        if row is None:
            expected.append(None)
            continue
        if any(row.get(k) != value for k,value in desc.items()):
            raise ValueError('reference output ABI mismatch: '+key)
        destination=root/(key+'.bin')
        _snapshot(_file(fixture,row),destination)
        expected.append({**row,**_record(destination,root)})
    abi={'schema_version':1,'artifact_kind':'qwen35-standalone-draft-abi','inputs':expected_inputs,'outputs':outputs,
         'draft_config':config,'sequence_length':sequence,'state_policy':'prefix-recompute'}
    save_json(root/'draft-abi.json',abi)
    manifest={'schema_version':1,'artifact_kind':'qwen35-standalone-draft-bundle','status':'PREPARED',
        'weight_storage':weight_storage,
        'ddk_id':_text(ddk_id,'DDK identity'),'platform':'kirinx90','sequence_length':sequence,'vocab_size':cfg.vocab_size,
        'draft_config':config,'model':model,'model_payloads':payloads,'draft_abi':_record(root/'draft-abi.json',root),
        'inputs':inputs,'outputs':outputs,'expected':expected,'source_identity':source_identity,
        'compiler_report_sha256':sha256_file(compiler/'report.json'),'fixture_report_sha256':sha256_file(fixture/'report.json'),
        'export_manifest_sha256':exported_hash,'reference_scope':'Same complete draft and fixed feature inputs; no target execution or acceptance evaluation'}
    if packing is not None:
        manifest['weight_packing'] = _record(root/'weight-packing.json', root)
    (root/'step.tsv').write_bytes(_wire(manifest))
    manifest['step_plan']=_record(root/'step.tsv',root)
    save_json(root/'manifest.json',manifest)
    return manifest


def _validate_bundle(root):
    manifest = _json(root/'manifest.json')
    if (manifest.get('artifact_kind') != 'qwen35-standalone-draft-bundle' or manifest.get('status') != 'PREPARED'
            or manifest.get('platform') != 'kirinx90'):
        raise ValueError('expected a prepared standalone draft bundle')
    inputs,outputs=_io(manifest['sequence_length'])
    if manifest['outputs'] != outputs or len(manifest['inputs']) != 3:
        raise ValueError('draft bundle IO contract changed')
    for row,desc in zip(manifest['inputs'],inputs):
        if any(row.get(k)!=v for k,v in desc.items()):
            raise ValueError('draft bundle input order/shape/dtype mismatch')
    abi=_json(_file(root,manifest['draft_abi']))
    if abi['inputs']!=inputs or abi['outputs']!=outputs or abi['draft_config']!=manifest['draft_config']:
        raise ValueError('draft ABI differs from bundle')
    if manifest['model'] not in manifest['model_payloads']:
        raise ValueError('model missing from payloads')
    records=[manifest['draft_abi'],manifest['step_plan'],*manifest['inputs'],*manifest['model_payloads']]
    if manifest.get('weight_storage') == 'embedded-stream':
        packing = _json(_file(root, manifest['weight_packing']))
        if (packing.get('status') != 'EMBEDDED'
                or any(packing.get('output', {}).get(key) != manifest['model'][key] for key in ('bytes','sha256'))):
            raise ValueError('streamed OMC differs from weight-packing receipt')
    for record in [*records,*[r for r in manifest['expected'] if r is not None]]:
        _file(root,record)
    if _file(root,manifest['step_plan']).read_bytes()!=_wire(manifest):
        raise ValueError('draft plan differs from hash-bound manifest')
    uploads=[_file_record(root/'manifest.json','bundle/manifest.json')]
    uploads.extend(_file_record(_file(root,row),'bundle/'+row['path']) for row in records)
    return manifest,uploads


def compare_draft_result(bundle_dir, result_dir, *, allow_host_fake=False, expected_model_load_api=None):
    """Validate native identity/readback before comparing Top1 and hidden values."""
    root, result_root = Path(bundle_dir).resolve(), Path(result_dir).resolve()
    manifest,_ = _validate_bundle(root)
    native=_json(result_root/'result.json')
    report={'status':'FAILED','device_execution':'UNVERIFIED','numerical_validation':'NOT_EVALUATED',
            'runtime_placement':'UNVERIFIED','reference_scope':manifest['reference_scope']}
    identity={'schema_version':1,'scope':'standalone-draft-step','model_sha256':manifest['model']['sha256'],
              'draft_abi_sha256':manifest['draft_abi']['sha256'],'step_plan_sha256':manifest['step_plan']['sha256'],
              'ddk_id':manifest['ddk_id'],'target_calls':0}
    if any(native.get(k)!=v for k,v in identity.items()):
        raise ValueError('native result does not bind the supplied draft bundle')
    if expected_model_load_api is not None:
        if expected_model_load_api not in {'file-v2','legacy-buffer'}:
            raise ValueError('invalid expected model load API')
        # Historical results predate this field and are legacy-only. A V2 run
        # must name its route, so an old runner cannot silently stand in for it.
        if native.get('model_load_api','legacy-buffer')!=expected_model_load_api:
            raise ValueError('native model load API differs from this invocation')
        report['model_load_api']=expected_model_load_api
    if any(type(native.get(k)) is not int for k in ('schema_version','target_calls','draft_calls')):
        raise ValueError('native call counters must be integers')
    evidence=native.get('evidence_kind')
    if evidence!='device-hiai' and not (allow_host_fake and evidence=='host-fake-hiai'):
        raise ValueError('device evidence required; host-fake is test-only')
    report['evidence_kind']=evidence
    runtime_stage=native.get('runtime_stage')
    if runtime_stage is not None:
        if not isinstance(runtime_stage,str) or not runtime_stage or len(runtime_stage)>256:
            raise ValueError('invalid native runtime stage')
        report['runtime_stage']=runtime_stage
    if native.get('status')=='RUNTIME_FAILED':
        if (native['draft_calls'] != 0 or native.get('outputs') != []
                or native.get('failure_stage') not in {'LOAD_OR_SETUP','INPUT_BINDING','PROCESS'}
                or not isinstance(native.get('reason'),str) or not native['reason'].strip()):
            raise ValueError('native runtime failure record has invalid stage, reason, counters or outputs')
        report.update(device_execution='FAILED',failure_stage=native.get('failure_stage'),error=native.get('reason'))
        return report
    if native.get('status')!='EXECUTED' or native.get('draft_calls')!=1:
        raise ValueError('native result must record exactly one draft execution')
    if len(native.get('inputs',[]))!=3 or len(native.get('outputs',[]))!=2:
        raise ValueError('native draft IO count mismatch')
    for actual,expected in zip(native['inputs'],manifest['inputs']):
        if any(actual.get(k)!=expected[k] for k in ('index','name','dtype','shape','bytes','sha256')):
            raise ValueError('native input bytes differ from prepared fixture')
    diagnostic=native.get('io_diagnostic',{})
    if (diagnostic.get('enabled') is not True or diagnostic.get('process_called') is not True
            or len(diagnostic.get('inputs',[]))!=3):
        raise ValueError('draft input buffer readback missing')
    for actual,expected in zip(diagnostic['inputs'],manifest['inputs']):
        if (actual.get('index')!=expected['index'] or actual.get('readback_matches_source') is not True
                or actual.get('readback_sha256')!=expected['sha256']):
            raise ValueError('draft host buffer readback differs from source')
    if len(diagnostic.get('outputs',[]))!=2 or any(
        row.get('index')!=index or row.get('checked_after_process') is not True
        or row.get('object_same') is not True or row.get('buffer_same') is not True
        for index,row in enumerate(diagnostic['outputs'])):
        raise ValueError('draft output buffer identity was not preserved through Process')
    values=[]
    for row,desc in zip(native['outputs'],manifest['outputs']):
        if any(row.get(k)!=v for k,v in desc.items()):
            raise ValueError('native draft output ABI mismatch')
        dtype=np.dtype('<i4' if row['dtype']=='int32' else '<f2')
        if row.get('bytes') != math.prod(row['shape'])*dtype.itemsize:
            raise ValueError('native output byte count mismatch')
        values.append(np.fromfile(_file(result_root,row),dtype=dtype).reshape(row['shape']))
    report.update(status='EXECUTION_COMPLETE',device_execution='PASSED',host_input_readback='MATCHED',
                  numerical_validation='NOT_EVALUATED',runtime_version=native.get('runtime_version'),firmware_id=native.get('firmware_id'))
    if np.any(values[0]<0) or np.any(values[0]>=manifest['vocab_size']) or not np.isfinite(values[1]).all():
        report.update(status='FAILED',numerical_validation='FAILED',error='invalid token IDs or nonfinite draft hidden output')
        return report
    top_record,hidden_record=manifest['expected']
    if top_record is not None:
        expected=np.fromfile(_file(root,top_record),dtype='<i4').reshape(values[0].shape)
        mismatch=int(np.count_nonzero(expected!=values[0]))
        report.update(token_mismatches=mismatch,actual_top1=values[0].tolist(),expected_top1=expected.tolist(),
                      numerical_validation='TOP1_MATCHED' if mismatch==0 else 'FAILED')
        if mismatch:
            report.update(status='FAILED',error='draft Top1 differs from PyTorch reference on identical inputs')
    if hidden_record is not None:
        expected=np.fromfile(_file(root,hidden_record),dtype='<f2').reshape(values[1].shape).astype(np.float64)
        actual=values[1].astype(np.float64)
        if not np.isfinite(expected).all():
            raise ValueError('nonfinite reference hidden output')
        error=actual-expected
        report['hidden_diagnostic']={'max_abs_error':float(np.max(np.abs(error))),
            'relative_l2':float(np.linalg.norm(error)/max(float(np.linalg.norm(expected)),np.finfo(np.float64).tiny)),
            'acceptance_threshold':'NOT_DEFINED'}
    report['accuracy_scope']='Top1 agreement and hidden diagnostics on this fixture only; full numerical accuracy and language quality remain unverified'
    return report


def _build_runner(out,timeout):
    command=[sys.executable,str(repository_root()/'targets/cpro/scripts/build_hiai.py'),'--output',str(out/'native')]
    save_json(out/'runner-build-command.json',{'argv':command})
    with (out/'runner-build.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    runner=out/'native/build/qwen35_dflash_hiai'
    record=_json(out/'native/result.json')
    if (record.get('status')!='LINKED' or record.get('binary_sha256')!=sha256_file(runner)
            or record.get('command_sha256')!=sha256_file(out/'native/command.json')):
        raise ValueError('runner differs from its LINKED build record')
    dynamic_info(runner)
    return runner


def _receive(hdc,board,out,name,*,timeout_seconds=None):
    _relative(name)
    timeout=min(30,hdc.timeout)
    if timeout_seconds is not None:
        if (isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float))
                or not math.isfinite(timeout_seconds) or timeout_seconds<=0):
            raise ValueError('receive timeout_seconds must be finite and positive')
        timeout=min(timeout,timeout_seconds)
    remote=shlex.quote(str(board/name))
    _,text=hdc.shell(f"if test -f {remote}; then sha256sum {remote}; else printf '__MISSING__\\n'; fi",'hash-result',timeout=timeout)
    if text=='__MISSING__':
        return None
    if len(text.splitlines())!=1 or not re.match(r'^[0-9a-f]{64} [ *]',text):
        raise ValueError('invalid remote SHA256: '+name)
    path=out/name
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.receiving')
    hdc.receive(board/name,temporary,timeout=timeout)
    if sha256_file(temporary)!=text[:64]:
        raise ValueError('retrieved SHA256 mismatch: '+name)
    temporary.rename(path)
    return text[:64]


def execute_draft_bundle(bundle_dir,output_dir,*,timeout_seconds=3600,model_load_api='legacy-buffer'):
    """Build one runner, probe HDC, deploy one OMC, execute once and compare."""
    if (isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float))
            or not math.isfinite(timeout_seconds) or timeout_seconds < .001):
        raise ValueError('timeout_seconds must be finite and at least one millisecond')
    if model_load_api not in {'file-v2','legacy-buffer'}:
        raise ValueError('model_load_api must be file-v2 or legacy-buffer')
    native_timeout_ms=min(int(timeout_seconds*1000),2**32-1)
    root=Path(bundle_dir).resolve()
    out=output_path(output_dir)
    if root==out or root in out.parents:
        raise ValueError('board result must be separate from immutable bundle')
    out.mkdir(parents=True,exist_ok=False)
    result={'artifact_kind':'qwen35-standalone-draft-execution','status':'FAILED','stage':'preflight',
            'device_execution':'NOT_RUN','numerical_validation':'NOT_EVALUATED','runtime_placement':'UNVERIFIED',
            'board_cleanup':'RETAINED','output':str(out),'retrieved_sha256':{},'retrieval_errors':[],
            'model_load_api':model_load_api}
    hdc=board=None
    attempted=False
    def collect(names):
        for name in names:
            if name in result['retrieved_sha256']:
                continue
            try:
                digest=_receive(hdc,board,out,name)
                if digest:
                    result['retrieved_sha256'][name]=digest
            except Exception as error:
                result['retrieval_errors'].append({'path':name,'error':f'{type(error).__name__}: {error}'})
    try:
        manifest,uploads=_validate_bundle(root)
        if manifest.get('source_identity')!='MATCHED_EXPORT_AND_REFERENCE':
            raise ValueError('board execution requires matching exported draft and reference identities')
        digest=sha256_file(root/'manifest.json')
        ddk=manifest['ddk_id']
        if os.environ.get('CPRO_DDK_ID') and os.environ['CPRO_DDK_ID']!=ddk:
            raise ValueError('CPRO_DDK_ID differs from compiled draft')
        sn,bip=_text(os.environ.get('SN'),'SN'),_text(os.environ.get('BIP'),'BIP')
        host,separator,port=bip.rpartition(':')
        if (sn.startswith('-') or bip.startswith('-') or not separator or not host
                or any(c.isspace() for c in host) or not port.isdigit() or not 1<=int(port)<=65535):
            raise ValueError('SN must be a device ID; BIP must be an HDC server host:port')
        hdc=Hdc(_text(os.environ.get('HDC','hdc'),'HDC'),sn,bip,out,timeout_seconds)
        board=PurePosixPath('/data/local/tmp/qwen35-draft-'+uuid.uuid4().hex)
        result.update(bundle_manifest_sha256=digest,ddk_id=ddk,serial=sn,hdc_server=bip,board_root=str(board))
        private=os.environ.get('BOARD_HIAI_LIB_DIR')
        paths=list(BOARD_LIBRARY_SEARCH_PATHS)
        if private:
            _text(private,'BOARD_HIAI_LIB_DIR')
            if not private.startswith('/') or ':' in private or '..' in PurePosixPath(private).parts:
                raise ValueError('BOARD_HIAI_LIB_DIR must be an absolute board path')
            paths.insert(0,private)
        print('draft-board: checking HDC and building current runner',flush=True)
        hdc.shell('command -v sha256sum >/dev/null && '+' && '.join('test -d '+shlex.quote(p)+' && test -r '+shlex.quote(p) for p in paths),'preflight')
        result['stage']='build-runner'
        runner=_build_runner(out,timeout_seconds)
        uploads.append(_file_record(runner,'runner'))
        result['stage']='select-libraries'
        if private:
            result['libraries']={'mode':'existing-board-directory','path':private,'compatibility':'UNVERIFIED'}
        else:
            discovery=discover_libraries(runner,out/'native')
            libraries=[_file_record(path.resolve(),'lib/'+str(_relative(path.name))) for path in discovery['libraries']]
            dependencies=[{'name':PurePosixPath(row['remote']).name,'kind':'runner' if row['remote']=='runner' else 'library',
                           'sha256':row['sha256'],**dynamic_info(row['local'])} for row in [uploads[-1],*libraries]]
            blockers=candidate_blockers(dependencies,require_pair=False)
            if blockers:
                raise ValueError('cannot select board deployment: '+'; '.join(blockers))
            uploads.extend(row for row in libraries if PurePosixPath(row['remote']).name not in DDK_HCL_EXCLUDED)
            paths.insert(0,str(board/'lib'))
            result['libraries']={'mode':'candidate_without_ddk_hcl','discovery':discovery['metadata'],
                                 'dependencies':dependencies,'backend_selection':'UNVERIFIED'}
        result['stage']='upload'
        save_json(out/'upload-plan.json',{'board_root':str(board),'files':uploads})
        hdc.shell('mkdir '+shlex.quote(str(board)),'create-root')
        hdc.shell('mkdir -p '+' '.join(shlex.quote(x) for x in sorted({str((board/row['remote']).parent) for row in uploads})),'create-subdirs')
        _upload(hdc,board,uploads,out/'payloads.sha256',checksum_connection_retries=2)
        hdc.shell('chmod 755 '+shlex.quote(str(board/'runner')),'runner-permissions')
        environment='LD_LIBRARY_PATH='+shlex.quote(':'.join(paths))+' '
        result['stage']='runtime-probe'
        try:
            hdc.shell('cd '+shlex.quote(str(board))+' && '+environment+'./runner --probe > runtime.json 2> runtime.stderr; '
                      'rc=$?; printf \'%s\\n\' "$rc" > '+shlex.quote(str(board/'runtime.exitcode'))+'; exit "$rc"','runtime-probe')
        finally:
            collect(('runtime.json','runtime.stderr','runtime.exitcode'))
        if any(x not in result['retrieved_sha256'] for x in ('runtime.json','runtime.stderr','runtime.exitcode')):
            raise RuntimeError('runtime probe evidence incomplete')
        probe=_json(out/'runtime.json')
        if (out/'runtime.exitcode').read_text().strip()!='0' or probe.get('npu_supported') is not True:
            raise RuntimeError('runtime capability probe failed')
        runtime=_text(probe.get('runtime_version'),'runtime version')
        if os.environ.get('HIAI_RUNTIME_VERSION') and os.environ['HIAI_RUNTIME_VERSION']!=runtime:
            raise ValueError('runtime version differs from HIAI_RUNTIME_VERSION')
        firmware=os.environ.get('FIRMWARE_ID')
        if not firmware:
            _,firmware=hdc.shell('param get const.ohos.fullname','firmware')
            firmware=firmware.strip()
            if re.search(r'errNum|Get parameter.*fail|error:',firmware,re.I):
                raise ValueError('firmware query failed; supply actual FIRMWARE_ID')
        firmware=_text(firmware,'firmware identity')
        result.update(runtime_version=runtime,firmware_id=firmware,stage='execute',device_execution='UNVERIFIED')
        result['execution_timeout']={'host_hdc_seconds':timeout_seconds,
            'native_process_timeout_ms':native_timeout_ms if model_load_api=='legacy-buffer' else None,
            'scope':('HDC wait and legacy Process timeout' if model_load_api=='legacy-buffer' else
                     'HDC wait only; synchronous V2 Run has no timeout argument; remote cancellation is unverified')}
        if model_load_api=='legacy-buffer':
            result['native_timeout_ms']=native_timeout_ms
        command=environment+shlex.join([str(board/'runner'),'--draft-step',str(board/'bundle/step.tsv'),
            '--output',str(board/'results'),'--expected-runtime-version',runtime,'--ddk-id',ddk,'--firmware-id',firmware,
            '--timeout-ms',str(native_timeout_ms),'--model-load-api',model_load_api])
        print(f'draft-board: executing complete draft once with {model_load_api}; target is not loaded',flush=True)
        attempted=True
        status,_=hdc.shell('cd '+shlex.quote(str(board))+' && '+command+' > draft.stdout 2> draft.stderr; '
                          'rc=$?; printf \'%s\\n\' "$rc" > '+shlex.quote(str(board/'draft.exitcode'))+'; exit "$rc"',
                          'execute',allow_failure=True)
        result['remote_exitcode']=status
    except Exception as error:
        result['transport_error' if attempted else 'error']=f'{type(error).__name__}: {error}'
    finally:
        if attempted:
            required=['draft.stdout','draft.stderr','draft.exitcode','results/result.json']
            collect(required)
            comparison=None
            try:
                if sha256_file(root/'manifest.json')!=result['bundle_manifest_sha256']:
                    raise ValueError('draft bundle changed during execution')
                if 'results/result.json' not in result['retrieved_sha256']:
                    raise ValueError('hash-verified native result is unavailable')
                native=_json(out/'results/result.json')
                runtime_key='runtime_version' if native.get('status')=='EXECUTED' else 'expected_runtime_version'
                if native.get(runtime_key)!=result['runtime_version'] or native.get('firmware_id')!=result['firmware_id']:
                    raise ValueError('native runtime/firmware differs from this invocation')
                if native.get('status')=='EXECUTED':
                    outputs=['results/output_0000.bin','results/output_0001.bin']
                    required.extend(outputs)
                    collect(outputs)
                    if any(name not in result['retrieved_sha256'] for name in outputs):
                        raise ValueError('hash-verified native outputs are unavailable')
                comparison=compare_draft_result(root,out/'results',expected_model_load_api=model_load_api)
                save_json(out/'comparison.json',comparison)
                result.update(native_status=native['status'],comparison=str(out/'comparison.json'))
                if comparison.get('runtime_stage'):
                    result['runtime_stage']=comparison['runtime_stage']
                pid=native.get('process_id')
                if type(pid) is int and 0 < pid <= 2**31-1:
                    result['native_process_id']=pid
                if native['status']=='RUNTIME_FAILED':
                    # A verified failure has no successful output binaries.
                    # Preserve its cause even if separate transport evidence
                    # is incomplete; never promote unbound native messages.
                    result.update(device_execution='FAILED',failure_stage=comparison['failure_stage'],
                                  error=comparison['error'],stage='runtime-failed')
            except Exception as error:
                result['comparison_error']=f'{type(error).__name__}: {error}'
                comparison=None
            transport=[]
            if result.get('transport_error'):
                transport.append(result['transport_error'])
            missing=[name for name in required if name not in result['retrieved_sha256']]
            if missing:
                transport.append('missing verified execution evidence: '+', '.join(missing))
            if result['retrieval_errors']:
                transport.append('execution evidence retrieval failed; see retrieval_errors')
            remote_exit=result.get('remote_exitcode')
            if type(remote_exit) is not int or not 0 <= remote_exit <= 255:
                transport.append('verified remote exit code is unavailable')
            if 'draft.exitcode' in result['retrieved_sha256']:
                exit_text=(out/'draft.exitcode').read_bytes().strip()
                if not re.fullmatch(rb'0|[1-9][0-9]{0,2}',exit_text) or int(exit_text)>255:
                    transport.append('invalid retained draft exit code')
                elif int(exit_text)!=remote_exit:
                    transport.append('remote exit code differs from retained draft exit code')
                elif result.get('native_status')=='EXECUTED' and remote_exit!=0:
                    transport.append('native EXECUTED conflicts with nonzero process exit')
                elif result.get('native_status')=='RUNTIME_FAILED' and remote_exit==0:
                    transport.append('native RUNTIME_FAILED conflicts with zero process exit')
            if transport:
                result['transport_error']='; '.join(transport)
            elif comparison is not None and result.get('native_status')=='EXECUTED':
                result.update(status=comparison['status'],stage='complete',
                              device_execution=comparison['device_execution'],
                              numerical_validation=comparison['numerical_validation'])
                if comparison.get('error'):
                    result['error']=comparison['error']
            if result['status']=='FAILED':
                # Read one diagnostic snapshot; never retry the model, clear
                # device logs, or let collection replace the original failure.
                try:
                    from .draft_hilog import capture_draft_hilog
                    result['hilog']=capture_draft_hilog(
                        hdc,board,out,receive=_receive,recorded_pid=result.get('native_process_id'))
                except Exception as error:
                    result['hilog']={'status':'FAILED','error':f'{type(error).__name__}: {error}'}
        result['hdc_invocations']=hdc.count if hdc else 0
        save_json(out/'result.json',result)
    return result

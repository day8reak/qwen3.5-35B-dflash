"""Replay retained draft inputs on CPU without replacing goldens or board evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from .artifacts import output_path
from .draft_board import _file, _hidden_comparison, _hidden_summary
from .draft_reference import _validated_prepared, _load_graph
from .draft_review import _json, review_draft_result
from .hashing import sha256_file


_KIND = 'qwen35-draft-reference-replay'
_MODES = ('legacy-fp16', 'fp32-stats')
_NAMES = ('draft_top1', 'draft_hidden')
_FILES = {'report.json', '.owner.json'} | {
    f'{mode}.{name}.bin' for mode in _MODES for name in _NAMES
} | {f'{mode}.trace.json' for mode in _MODES}
_JSON_LIMIT = 8 << 20


def _atomic(root, name, data):
    if name not in _FILES:
        raise ValueError('unknown reference replay output')
    path, temporary = root/name, root/('.'+name+'.tmp')
    for candidate in (path, temporary):
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_file()):
            raise ValueError('reference replay output must be a regular file')
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def _save(root, name, value):
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if len(data)>_JSON_LIMIT:
        raise ValueError('reference replay metadata exceeds the bounded report size')
    return _atomic(root,name,data)


def _output(job, directory):
    raw=Path(directory).absolute()
    if raw.is_symlink() or raw.resolve()!=raw or raw!=job/'reference-replay':
        raise ValueError('reference replay output must be the real job/reference-replay directory')
    root=output_path(raw)
    owner={'artifact_kind':_KIND+'-files','job':str(job),'output':str(root)}
    if root.exists():
        if not root.is_dir():
            raise ValueError('reference replay output is not a directory')
        existing=list(root.iterdir())
        if existing and (not (root/'.owner.json').is_file() or _json(root/'.owner.json')!=owner):
            raise ValueError('reference replay output has no matching ownership marker')
        allowed=_FILES | {'.'+name+'.tmp' for name in _FILES}
        if any(path.name not in allowed or path.is_symlink() or not path.is_file() for path in existing):
            raise ValueError('reference replay output contains unrelated files or directories')
    else:
        root.mkdir()
    if not (root/'.owner.json').exists():
        _save(root,'.owner.json',owner)
    return root


def _component(job, name, filename):
    row=job.get('components',{}).get(name)
    if not isinstance(row,dict) or not isinstance(row.get('directory'),str):
        raise ValueError('reference replay requires completed '+name+' component')
    root=Path(row['directory'])
    if not root.is_absolute() or root.resolve()!=root or root.is_symlink():
        raise ValueError('reference replay component directory must be real and absolute')
    path=_file(root,row['manifest'])
    if path.name!=filename:
        raise ValueError('unexpected '+name+' manifest filename')
    return root,path,_read_json_bound(path,row['manifest']['sha256'])


def _read_json_bound(path, digest):
    if path.stat().st_size > 16 << 20:
        raise ValueError('retained replay metadata exceeds bounded size')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError('replay metadata changed after verification: '+str(path))
    return json.loads(data)


def _read_array_bound(path, row):
    dtype=np.dtype('<i4' if row['dtype']=='int32' else '<f2')
    values=np.fromfile(path,dtype=dtype).reshape(row['shape'])
    if hashlib.sha256(values.tobytes()).hexdigest()!=row['sha256']:
        raise ValueError('replay tensor changed after verification: '+str(path))
    return values


def _same_tensor(left, right, label):
    if not isinstance(left,dict) or not isinstance(right,dict) or any(
            left.get(key)!=right.get(key) for key in ('index','name','dtype','shape','bytes','sha256')):
        raise ValueError(label+' differs between retained fixture and executed bundle')


def _identity(path):
    info=Path(path).lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('replay source is not a regular file: '+str(path))
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def _arrays(outputs, manifest):
    if not isinstance(outputs,dict) or set(outputs)!=set(_NAMES):
        raise ValueError('reference replay must return the two ordered draft outputs')
    result={}
    for row in manifest['outputs']:
        values=outputs[row['name']]
        dtype=np.dtype('<i4' if row['dtype']=='int32' else '<f2')
        if not isinstance(values,np.ndarray) or values.dtype!=dtype or list(values.shape)!=row['shape']:
            raise ValueError('reference replay output shape/dtype differs from retained ABI')
        result[row['name']]=np.ascontiguousarray(values)
    return result


def replay_reference(job_dir, output_dir, *, progress=None):
    """Run both diagnostic modes once under the caller's DraftWorkspace lock.

    The old CPU goldens must reproduce byte-for-byte before a new RMS-statistics
    candidate can establish Top1 agreement. Neither candidate replaces them.
    """
    job_root=Path(job_dir).resolve()
    notify=progress or (lambda message: None)
    notify('Verifying retained execution, reference receipts and exact inputs')
    review=review_draft_result(job_root)
    if review.get('device_execution')!='PASSED' or review.get('native_status')!='EXECUTED':
        raise ValueError('reference replay requires verified completed device execution')
    job=_read_json_bound(job_root/'result.json',review['job_result_sha256'])
    prepared,prepared_file,plan=_component(job,'prepared','prepared.json')
    fixture,fixture_file,reference=_component(job,'fixture','report.json')
    bundle,bundle_file,manifest=_component(job,'bundle','manifest.json')
    if (sha256_file(job_root/'result.json')!=review['job_result_sha256']
            or sha256_file(bundle_file)!=review['bundle_manifest_sha256']
            or manifest.get('fixture_report_sha256')!=sha256_file(fixture_file)
            or reference.get('prepared_manifest_sha256')!=sha256_file(prepared_file)):
        raise ValueError('reference replay source manifests differ from retained execution')
    if reference.get('reference')!='PYTORCH_FP16_PRIMITIVES':
        raise ValueError('reference replay diagnoses legacy PYTORCH_FP16_PRIMITIVES fixtures only; '
                         'retained reference is '+repr(reference.get('reference'))+'. '
                         'Use --compare-only to compare this fixture with its retained board outputs.')
    if (reference.get('status')!='REFERENCE_COMPLETE'
            or plan.get('tiny') is not False or plan.get('draft_config')!=manifest.get('draft_config')
            or reference.get('draft_config')!=plan.get('draft_config')
            or reference.get('sequence_length')!=manifest.get('sequence_length')
            or plan.get('sequence_length')!=manifest.get('sequence_length')):
        raise ValueError('reference replay requires the same complete FP16 primitive draft fixture')
    if len(reference.get('inputs',[]))!=3 or len(manifest.get('inputs',[]))!=3:
        raise ValueError('reference replay requires three retained inputs')
    source_paths=[job_root/'result.json',job_root/'board/result.json',prepared_file,fixture_file,bundle_file]
    inputs=[]
    for original,executed in zip(reference['inputs'],manifest['inputs']):
        _same_tensor(original,executed,'input '+str(executed.get('name')))
        source_paths.append(_file(fixture,original))
        path=_file(bundle,executed);source_paths.append(path)
        inputs.append(_read_array_bound(path,executed))
    goldens=[]
    for index,key in enumerate(('expected_top1','expected_hidden')):
        original=reference.get(key)
        executed=manifest['expected'][index]
        _same_tensor(original,executed,key)
        source_paths.append(_file(fixture,original))
        path=_file(bundle,executed);source_paths.append(path)
        goldens.append(_read_array_bound(path,executed).tobytes())
    native_file=job_root/'board/results/result.json'
    retained={row['path']:row for row in review['retrieved_files']}
    native=_read_json_bound(native_file,retained['results/result.json']['sha256']);source_paths.append(native_file)
    if (sha256_file(job_root/'board/result.json')!=review['board_result_sha256']
            or sha256_file(native_file)!=retained['results/result.json']['sha256']):
        raise ValueError('retained native execution changed after verification')
    board_values={}
    for row in native['outputs']:
        path=_file(native_file.parent,row);source_paths.append(path)
        board_values[row['name']]=_read_array_bound(path,row)
    source_dir=Path(__file__).resolve().parent
    code_files=[source_dir/name for name in ('draft_reference.py','draft_reference_trace.py',
                'model.py','ops.py','graphs.py','draft_reference_replay.py')]
    source_paths.extend(code_files)
    identities={str(path):_identity(path) for path in source_paths}
    code_records=[{'path':path.name,'bytes':path.stat().st_size,'sha256':sha256_file(path)} for path in code_files]
    for path in code_files:
        if _identity(path)!=identities[str(path)]:
            raise ValueError('replay source code changed during provenance verification')
    # Ownership validation precedes model construction; unrelated files are not
    # removed even when a previous diagnostic attempt failed.
    root=_output(job_root,output_dir)
    result={'schema_version':1,'artifact_kind':_KIND,'status':'RUNNING',
            'operation':'CPU_REFERENCE_REPLAY','job':str(job_root),'output':str(root),
            'device_execution':'PASSED','new_device_execution':'NOT_RUN','new_compilation':'NOT_RUN',
            'target_execution':'NOT_RUN','runtime_placement':'UNVERIFIED',
            'numerical_validation':'NOT_ESTABLISHED','legacy_reproduced':False,'modes':{},
            'provenance':{'source_directory':str(source_dir),'source_files':code_records,'torch_version':None},
            'sources':{'job_result_sha256':review['job_result_sha256'],
                'board_result_sha256':review['board_result_sha256'],
                'bundle_manifest_sha256':review['bundle_manifest_sha256'],
                'prepared_manifest_sha256':sha256_file(prepared_file),'fixture_report_sha256':sha256_file(fixture_file),
                'model_sha256':manifest['model']['sha256'],'native_result_sha256':sha256_file(native_file),
                'inputs':manifest['inputs'],'original_goldens':manifest['expected'],
                'retained_board_outputs':native['outputs']},
            'scope':'Same prepared FP16 weights and exact retained inputs. RMS-only counterfactual restores FP32 RMS normalization order; other graph math stays unchanged. This is not a complete upstream BF16 oracle. Original goldens, compiled model and board evidence remain unchanged.',
            'accuracy_scope':'Top1 agreement and hidden diagnostics on this fixture only; full numerical accuracy, language quality and speed remain unverified.'}
    _save(root,'report.json',result)
    try:
        notify('Verifying prepared weights and loading the CPU draft once')
        # Capture every prepared payload before the expensive hash pass, so a
        # weight replacement between verification and graph loading is detected.
        for row in plan.get('weights',{}).values():
            path=prepared/row['path']
            identities[str(path)]=_identity(path)
        validated,validated_plan,config,paths=_validated_prepared(prepared)
        if validated!=prepared or validated_plan!=plan:
            raise ValueError('prepared manifest changed before CPU graph loading')
        for path in paths.values():
            if str(path) not in identities or _identity(path)!=identities[str(path)]:
                raise ValueError('prepared payload changed during validation: '+str(path))
        graph=_load_graph(config,paths)
        from .draft_reference_trace import replay_graph
        input_hashes=[hashlib.sha256(values.tobytes()).hexdigest() for values in inputs]
        for mode in _MODES:
            notify('Replaying '+mode+' on the retained input bytes')
            outputs,trace=replay_graph(graph,inputs,rms_mode=mode,progress=notify)
            arrays=_arrays(outputs,manifest)
            if [hashlib.sha256(values.tobytes()).hexdigest() for values in inputs]!=input_hashes:
                raise ValueError('CPU replay modified retained input arrays')
            if not isinstance(trace,dict) or not isinstance(trace.get('validation'),dict):
                raise ValueError('reference replay trace lacks explicit validation')
            version=trace.get('torch_version')
            if not isinstance(version,str) or not version or (result['provenance']['torch_version'] is not None
                    and result['provenance']['torch_version']!=version):
                raise ValueError('reference replay trace lacks a consistent PyTorch version')
            result['provenance']['torch_version']=version
            records=[]
            for index,name in enumerate(_NAMES):
                values=arrays[name]
                records.append({**manifest['outputs'][index],**_atomic(root,f'{mode}.{name}.bin',values.tobytes())})
            mode_result={'outputs':records,'trace':_save(root,mode+'.trace.json',trace),
                         'validation':trace['validation'],'first_collapse':trace.get('first_collapse')}
            if mode=='legacy-fp16':
                matches=[arrays[name].tobytes()==goldens[index] for index,name in enumerate(_NAMES)]
                result['legacy_reproduced']=all(matches)
                mode_result['original_golden_byte_matches']={name:matches[index] for index,name in enumerate(_NAMES)}
                mode_result['hidden_summary']=_hidden_summary(arrays['draft_hidden'].astype(np.float64))
                notify('Legacy output bytes '+('match' if result['legacy_reproduced'] else 'differ from')+' the retained goldens')
                collapse=trace.get('first_collapse')
                if isinstance(collapse,dict):
                    notify('First RMS collapse: '+str(collapse.get('name','unknown'))+'; '+str(collapse.get('cause_candidate','unclassified')))
            else:
                candidate=arrays['draft_hidden'].astype(np.float64)
                hidden=_hidden_comparison(candidate,board_values['draft_hidden'].astype(np.float64))
                top1=arrays['draft_top1']
                valid=(trace['validation'].get('status')=='PASSED' and trace['validation'].get('finite') is True
                       and trace['validation'].get('nondegenerate') is True and np.isfinite(candidate).all()
                       and bool(np.count_nonzero(candidate)) and np.all(top1>=0) and np.all(top1<manifest['vocab_size'])
                       and np.isfinite(board_values['draft_hidden']).all())
                mismatch=int(np.count_nonzero(top1!=board_values['draft_top1']))
                mode_result.update(candidate_valid=bool(valid),token_mismatches=mismatch,
                    candidate_top1=top1.tolist(),retained_board_top1=board_values['draft_top1'].tolist(),hidden_diagnostic=hidden)
                if result['legacy_reproduced']:
                    result['numerical_validation']='TOP1_MATCHED' if valid and mismatch==0 else 'FAILED'
                notify(f'FP32 RMS candidate valid={bool(valid)}; retained board Top1 mismatches={mismatch}; hidden relative_l2={hidden.get("relative_l2")}')
            result['modes'][mode]=mode_result
            for path,identity in identities.items():
                if _identity(path)!=identity:
                    raise ValueError('reference replay source changed during execution: '+path)
        result['status']='DIAGNOSTIC_COMPLETE'
        _save(root,'report.json',result)
        notify('Reference replay complete: '+result['numerical_validation']+'; '+str(root/'report.json'))
        return result
    except Exception as error:
        result.update(status='FAILED',numerical_validation='NOT_ESTABLISHED',error=f'{type(error).__name__}: {error}')
        _save(root,'report.json',result)
        raise

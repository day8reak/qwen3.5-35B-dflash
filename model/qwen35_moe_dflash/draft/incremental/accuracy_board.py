"""Detached standalone accuracy request with a hash-addressed Draft model cache."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import re
import shlex
import tarfile
import uuid

from qwen35_moe_dflash.common.artifacts import output_path, checked_file
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save, reuse_test_output
from qwen35_moe_dflash.execution.serial_transport import _environment, _build_payloads
from qwen35_moe_dflash.execution.hdc_run import Hdc, _file_record
from qwen35_moe_dflash.execution.lock import _acquire_global, _release_global
from qwen35_moe_dflash.draft.board import _receive
from qwen35_moe_dflash.draft.profile_board import GLOBAL_LOCK, _guard, _identity_guard
from .accuracy import KIND, prepare, compare
from .runtime import validate_abi

PREFIX = '/data/local/tmp/qwen35_draft_accuracy_'
q = shlex.quote


def replay_fixture(path):
    """Reuse a completed accuracy fixture without Torch or reference generation."""
    print('draft-accuracy: checking retained replay inputs and reference; no model or reference generation', flush=True)
    path = Path(path).resolve()
    path = path/'result.json' if path.is_dir() else path
    result = read_json(path, 4 << 20)
    if (result.get('artifact_kind') != KIND or result.get('stage') != 'complete'
            or result.get('status') not in ('PASSED', 'NUMERICAL_FAILED')
            or result.get('device_execution') != 'EXECUTED_OUTPUTS_RETRIEVED'):
        raise ValueError('replay requires a completed, retrieved accuracy result')
    fixture = (path.parent/Path(result['fixture'])).resolve()
    if sha256_file(fixture/'manifest.json') != result['fixture_sha256']:
        raise ValueError('replay fixture changed after the original comparison')
    meta = read_json(fixture/'manifest.json')
    validate_abi(meta)
    if (meta['model'] != result['model'] or meta['comparison'] != result['comparison']
            or not 1 <= len(meta['cases']) <= 64 or meta['cache_capacity'] > 512):
        raise ValueError('replay model, gates or fixture bounds differ')
    expected = {(s['index'], t['index']) for s in meta['cases'] for t in meta['inputs']}
    records = meta['input_files']
    if len(records) != len(expected) or {(r['case'], r['index']) for r in records} != expected:
        raise ValueError('replay input inventory is incomplete or duplicated')
    for record in records:
        if (record['path'] != f"inputs/{record['case']}/{record['index']}.bin"
                or record['bytes'] != meta['inputs'][record['index']]['bytes']):
            raise ValueError('replay input path/extent differs')
        checked_file(fixture, record)
    # Verify both the plan and the retained golden/result relationship before
    # asking for another execution. This is explicit diagnosis, never inference.
    rows = [f"QWEN35_INCREMENTAL_ACCURACY {2 if 'host_embedding' in meta else 1}",
            f"geometry {meta['context_rows']} {meta['cache_capacity']} {meta['max_draft_tokens']}"]
    for direction in ('inputs', 'outputs'):
        rows += [' '.join((direction[:-1], str(s['index']), s['name'], s['dtype'],
                           ','.join(map(str, s['shape'])))) for s in meta[direction]]
    rows += [f"case {s['index']} {s['start']} {s['valid']} {s['anchor']} {s['proposals']}"
             for s in meta['cases']]
    if (fixture/'plan.txt').read_text() != '\n'.join(rows)+'\n':
        raise ValueError('replay plan differs from fixture metadata')
    from .accuracy import analyze_result
    analyze_result(path)
    print('draft-accuracy: replay fixture verified; original schedule and comparison gates retained', flush=True)
    return fixture, meta


def extract_outputs(archive, destination, meta, reproduction=None):
    snapshot_bytes = sum(r['bytes'] for r in meta['outputs'])
    expected = {f'{mode}-{s["index"]}.bin': snapshot_bytes
                for mode in ('isolated', 'feedback') for s in meta['cases']}
    expected['result.json'] = None
    with tarfile.open(archive) as stream:
        members = [m for m in stream.getmembers() if not (m.name in ('.', './') and m.isdir())]
        names = [m.name[2:] if m.name.startswith('./') else m.name for m in members]
        if reproduction:
            from .kv_reproduction import validate_report
            reports = [m for m, n in zip(members, names) if n == 'kv-reproduction.json']
            if len(reports) != 1 or not reports[0].isfile() or not 0 < reports[0].size < 1 << 20:
                raise ValueError('missing/invalid native KV reproduction report')
            with stream.extractfile(reports[0]) as source:
                report = json.load(source)
            expected.update({n: snapshot_bytes for n in validate_report(report, reproduction, meta)})
            expected['kv-reproduction.json'] = None
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('native archive does not contain the complete two-sequence result')
        destination.mkdir()
        for member, name in zip(members, names):
            size = expected[name]
            if not member.isfile() or (size is None and not 0 < member.size < 1 << 20) or (size is not None and member.size != size):
                raise ValueError('invalid native result member: '+name)
            with stream.extractfile(member) as source, (destination / name).open('xb') as out:
                import shutil
                shutil.copyfileobj(source, out)


def start(config, compiled, source, air, output, *, apply=False, atol=0.01, rtol=0.01,
          timeout_seconds=1800, ddk_root=None, replay=None, output_initialization='zero',
          kv_reproduction_case=None):
    if output_initialization not in ('zero', 'untouched') or (not replay and output_initialization != 'zero'):
        raise ValueError('output initialization experiment requires replay and zero or untouched')
    candidate_replay = bool(replay and compiled)
    if replay and any((compiled, source, air, ddk_root)) and not all((compiled, source, air, ddk_root)):
        raise ValueError('candidate replay requires compiled, source, air and ddk_root together')
    out = output_path(output)
    config_path = Path(config).resolve()
    fixture, meta = replay_fixture(replay) if replay else (out/'fixture', None)
    reproduction = None
    if kv_reproduction_case is not None:
        if not replay or compiled or output_initialization != 'untouched':
            raise ValueError('KV reproduction requires untouched replay of the same retained OMC')
        from .kv_reproduction import request
        reproduction = request(meta, kv_reproduction_case)
    # These are public shared inputs, never descendants of the disposable slot.
    shared = [config_path, *(Path(v).resolve() for v in (compiled, source, air, replay) if v)]
    if replay:
        shared += [fixture, Path(meta['model']['path']).resolve()]
    for value in shared:
        if value == out or out in value.parents:
            raise ValueError('accuracy output contains a shared source')
    reuse_test_output(out, config=config_path)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(artifact_kind=KIND, status='FAILED', stage='reference', device_execution='NOT_RUN',
        model_compilation='NOT_RUN', numerical_validation='NOT_EVALUATED',
        lock_state='NO_KNOWN_OWNED_LOCK', output=str(out),
        output_initialization=output_initialization, replay_source=str(Path(replay).resolve()) if replay else None)
    if reproduction:
        state['kv_reproduction'] = dict(reproduction, status='NOT_RUN')
    hdc=None; held=[]; uncertain=False; launched=False
    try:
        cfg = read_json(config_path)
        env = read_json(config_path.parent / cfg['assets'])['environment']
        if candidate_replay:
            from .accuracy_replay import rebind_candidate
            fixture, meta = rebind_candidate(fixture, meta, compiled, source, air, ddk_root, out/'fixture')
        if not replay:
            meta = prepare(compiled, source, air, fixture, atol=atol, rtol=rtol, ddk_root=ddk_root)
        if meta['ddk_id'] != env['ddk_id']:
            raise ValueError('compiled Draft and runtime configuration use different DDKs')
        state.update(status='REFERENCE_PREPARED', stage='prepared', frontend=meta.get('frontend', 'ge'),
                     internal_precision=meta.get('internal_precision'), production_promotion='NOT_PERFORMED')
        if candidate_replay:
            state['precision_repair'] = meta['precision_repair']
        if 'host_embedding_binding' in meta:
            state['host_embedding_binding'] = meta['host_embedding_binding']
        if not apply:
            return state
        if not 1 <= timeout_seconds <= 7200:
            raise ValueError('accuracy timeout must be 1..7200 seconds')
        native = (config_path.parent / cfg.get('native_directory', 'native')).resolve()
        binary = native / 'build/qwen35_draft_accuracy'
        if not binary.is_file():
            raise ValueError('rebuild runtime/native with build_hiai.py --replace for the accuracy runner')
        built=read_json(native/'result.json')
        observed=dict(bytes=binary.stat().st_size,sha256=sha256_file(binary))
        if built.get('diagnostic_binaries',{}).get(binary.name) != observed:
            raise ValueError('accuracy runner differs from its native build; rebuild runtime/native')
        reports_initialization = output_initialization in built.get('accuracy_output_initialization', [])
        if replay and not reports_initialization:
            raise ValueError('rebuild runtime/native: the accuracy runner lacks output initialization replay support')
        if reproduction and built.get('accuracy_kv_reproduction_version') != reproduction['version']:
            raise ValueError('rebuild runtime/native: the accuracy runner lacks KV reproduction support')
        uploads, search, _ = _build_payloads(out, 600, resident='resident-mixed', native_directory=native)
        uploads = [r for r in uploads if r['remote'].startswith('lib/')]
        uploads.append(_file_record(binary, 'draft-accuracy'))
        archive = out / 'inputs.tar'
        with tarfile.open(archive, 'w') as tar:
            tar.add(fixture/'plan.txt', arcname='plan.txt')
            tar.add(fixture/'inputs', arcname='inputs')
        uploads.append(_file_record(archive, 'inputs.tar'))
        executable, serial, server = _environment()
        token = uuid.uuid4().hex
        board = PurePosixPath(PREFIX+token)
        cache = PurePosixPath('/data/local/tmp/qwen35_real_draft_'+meta['model']['sha256'])
        model = cache / 'model/model.omc'
        locks = [GLOBAL_LOCK, cache / '.execution-lock']
        plan = dict(artifact_kind=KIND, token=token, board_attempt=str(board),
            board_model_cache=str(cache), serial=serial, hdc_server=server, runtime=env,
            model=meta['model'], timeout_seconds=timeout_seconds, fixture=str(fixture),
            fixture_sha256=sha256_file(fixture/'manifest.json'), runner=observed,
            output_initialization=output_initialization, replay_source=state['replay_source'],
            output_initialization_reported=reports_initialization)
        if reproduction:
            plan['kv_reproduction'] = reproduction
        _save(out / 'request.json', plan)
        state.update(board_attempt=str(board), stage='board-preflight')
        hdc=Hdc(executable, serial, server, out, 180)
        for lock in locks:
            if lock == locks[1]:
                # The global lock protects creation; the cache owner binds it
                # to this exact OMC. Old and new candidates cannot collide.
                owner = cache / '.model-sha256'
                hdc.shell('if mkdir '+q(str(cache))+'; then printf "%s\\n" '+q(meta['model']['sha256'])
                    +' > '+q(str(owner))+'; else test -d '+q(str(cache))+' && test ! -L '+q(str(cache))
                    +' && test ! -L '+q(str(owner))+' && read -r draft_cache_owner < '+q(str(owner))
                    +' && test "$draft_cache_owner" = '+q(meta['model']['sha256'])+'; fi', 'accuracy-model-cache')
            uncertain=True
            try: _acquire_global(hdc, lock, token, 0)
            except ValueError as error:
                if 'BOARD_MODEL_IO_BUSY:' in str(error): uncertain=False
                raise
            held.append(lock); uncertain=False
        _, firmware=hdc.shell('param get const.ohos.fullname','accuracy-firmware')
        if firmware.strip()!=env['firmware_id'].strip():
            raise ValueError('board firmware differs from the runtime configuration')
        output_calls=2*len(meta['cases'])+(reproduction['planned_calls'] if reproduction else 0)
        required=archive.stat().st_size*2+output_calls*sum(r['bytes'] for r in meta['outputs'])*2+(128<<20)
        from qwen35_moe_dflash.draft.real_feature_draft_board import _cache_model
        model, state['model_cache'] = _cache_model(hdc, cache, meta['model'], out, extra_bytes=required)
        hdc.shell('mkdir '+q(str(board))+' && printf "%s\\n" '+q(token)+' > '+q(str(board/'owner'))
            +' && mkdir '+q(str(board/'lib'))+' '+q(str(board/'fixture')), 'accuracy-directory')
        for row in uploads:
            hdc.send(row['local'],board/row['remote'])
            hdc.shell(_identity_guard(board/row['remote'],row),'accuracy-upload')
        hdc.shell('cd '+q(str(board))+' && chmod +x draft-accuracy && tar -xf inputs.tar -C fixture',
                  'accuracy-unpack')
        command = ['timeout','-k','5',str(timeout_seconds),'env',
            'LD_LIBRARY_PATH='+':'.join([str(board/'lib'),*search]),
            str(board/'draft-accuracy'),str(model),env['runtime_version'],str(board/'fixture'),str(board/'outputs')]
        if replay:
            command.append(output_initialization)
        if reproduction:
            command.append(str(reproduction['case']))
        script='\n'.join(['#!/bin/sh','cd '+q(str(board))+' || exit 91',
            _guard(board,token)+' || exit 92',*[_guard(lock,token)+' || exit 93' for lock in locks],
            shlex.join(command)+' > native.stdout 2> native.stderr',
            'rc=$?', 'printf "%s\\n" "$rc" > native.exitcode',
            'if test "$rc" -eq 0; then tar -cf outputs.tar -C outputs .; printf "%s\\n" "$?" > archive.exitcode; fi',
            'printf "%s\\n" '+q(token)+' > complete.tmp','mv complete.tmp complete',''])
        (out/'launch.sh').write_text(script)
        hdc.send(out/'launch.sh',board/'launch.sh')
        hdc.shell(_identity_guard(board/'launch.sh',dict(bytes=len(script.encode()),sha256=sha256_file(out/'launch.sh'))),
                  'accuracy-launcher')
        launched=True;state.update(launch_requested=True,device_execution='RUNNING_OR_COMPLETION_PENDING')
        if reproduction:
            state['kv_reproduction']['status'] = 'RUNNING_OR_COMPLETION_PENDING'
        _save(out/'result.json',state)
        hdc.shell('cd '+q(str(board))+' && '+_guard(board,token)
            +' && { nohup sh launch.sh < /dev/null > supervisor.stdout 2> supervisor.stderr & }','accuracy-launch')
        state.update(status='DRAFT_ACCURACY_STARTED',stage='running-detached',
            next='Recover this same execution; no model relaunch.')
    except Exception as error:
        state.update(status='FAILED',error=f'{type(error).__name__}: {error}')
    finally:
        if not launched and not uncertain and hdc:
            for lock in reversed(held[:]):
                try: _release_global(hdc,lock,token);held.remove(lock)
                except Exception as error: state['lock_release_error']=str(error);break
        state['lock_state']='RETAINED_OR_UNCERTAIN' if held or uncertain else 'NO_KNOWN_OWNED_LOCK'
        _save(out/'result.json',state)
    return state


def recover(execution, output):
    root=Path(execution).resolve();out=output_path(output)
    if root==out or root in out.parents or out in root.parents:
        raise ValueError('recovery must use a separate test slot')
    reuse_test_output(out,recovery_source=root);out.mkdir(parents=True)
    result=dict(artifact_kind=KIND,status='FAILED',stage='recover',model_compilation='NOT_RUN',
        numerical_validation='NOT_EVALUATED',recovery_model_execution='FORBIDDEN',
        recovery_source=str(root),lock_state='RETAINED_OR_UNCERTAIN')
    complete=False;hdc=None;plan=None
    try:
        plan=read_json(root/'request.json');token=plan['token']
        if (plan.get('artifact_kind')!=KIND or not re.fullmatch('[0-9a-f]{32}',token)
            or plan['board_attempt']!=PREFIX+token
            or not re.fullmatch('[0-9a-f]{64}',plan['model']['sha256'])
            or plan['board_model_cache']!='/data/local/tmp/qwen35_real_draft_'+plan['model']['sha256']):
            raise ValueError('invalid retained accuracy request')
        board=PurePosixPath(plan['board_attempt']);result['board_attempt']=str(board)
        exe,serial,server=_environment()
        if (serial,server)!=(plan['serial'],plan['hdc_server']):raise ValueError('recover requires the same board')
        hdc=Hdc(exe,serial,server,out,180)
        _,text=hdc.shell(_guard(board,token)+' && if test -f '+q(str(board/'complete'))
            +'; then cat '+q(str(board/'complete'))+'; else printf "PENDING\\n"; fi','accuracy-completion')
        if text=='PENDING':
            result.update(status='DRAFT_ACCURACY_PENDING',stage='running');return result
        if text!=token:raise ValueError('completion belongs to another request')
        complete=True
        result['device_execution']='FAILED_OUTPUTS_RETRIEVED'
        for name in ('native.stdout','native.stderr','native.exitcode'):
            if _receive(hdc,board,out,name) is None:raise ValueError('missing '+name)
        if (out/'native.exitcode').read_text().strip()!='0':
            result['native_failure'] = dict(exitcode=(out/'native.exitcode').read_text().strip(),
                stderr_tail=(out/'native.stderr').read_text(errors='replace')[-4000:])
            # Load returns only a generic status in the native log. Recover
            # the bounded system-log evidence without loading the model again.
            # Old attempts have no recorded native PID: selection remains a
            # shared-log candidate, not an attributed cause of this failure.
            try:
                from qwen35_moe_dflash.draft.hilog import capture_draft_hilog
                result['hilog'] = capture_draft_hilog(hdc, board, out, receive=_receive,
                                                      application_names=('draft-accuracy', 'qwen35_draft_accuracy'))
            except Exception as error:
                result['hilog'] = dict(status='UNAVAILABLE', error=f'{type(error).__name__}: {error}')
            raise ValueError('native accuracy execution failed; see native.stderr')
        if _receive(hdc,board,out,'archive.exitcode') is None or (out/'archive.exitcode').read_text().strip()!='0':
            raise ValueError('native output archive failed')
        fixture = root/plan.get('fixture', 'fixture')
        if sha256_file(fixture/'manifest.json') != plan['fixture_sha256']:
            raise ValueError('reference fixture changed after device launch')
        meta=read_json(fixture/'manifest.json')
        reproduction = plan.get('kv_reproduction')
        if reproduction:
            from .kv_reproduction import request
            if reproduction != request(meta, reproduction['case']) or plan.get('output_initialization') != 'untouched':
                raise ValueError('invalid retained KV reproduction plan')
        output_calls=2*len(meta['cases'])+(reproduction['planned_calls'] if reproduction else 0)
        maximum=output_calls*sum(s['bytes'] for s in meta['outputs'])+(3<<20)
        _,size=hdc.shell('wc -c < '+q(str(board/'outputs.tar')),'accuracy-output-size')
        if not size.strip().isdigit() or not 0<int(size)<maximum:raise ValueError('unexpected accuracy archive extent')
        if _receive(hdc,board,out,'outputs.tar',timeout_seconds=180) is None:raise ValueError('missing output archive')
        extract_outputs(out/'outputs.tar',out/'native',meta,reproduction)
        if (plan.get('output_initialization_reported', False) and read_json(out/'native/result.json').get('output_initialization')
                != plan['output_initialization']):
            raise ValueError('native output initialization differs from the requested experiment')
        result.update(compare(fixture,out/'native'),stage='complete',device_execution='EXECUTED_OUTPUTS_RETRIEVED',
                      fixture=str(fixture),fixture_sha256=sha256_file(fixture/'manifest.json'),
                      output_initialization=plan.get('output_initialization', 'zero'),
                      replay_source=plan.get('replay_source'), runner=plan.get('runner'),
                      model=meta['model'],runtime=plan['runtime'], frontend=meta.get('frontend', 'ge'),
                      internal_precision=meta.get('internal_precision'), production_promotion='NOT_PERFORMED')
        if 'precision_repair' in meta:
            result['precision_repair'] = meta['precision_repair']
            result['baseline_model'] = meta['candidate_replay']['baseline_model']
        if reproduction:
            from .kv_reproduction import analyze
            diagnostic = analyze(fixture, out/'native', reproduction)
            _save(out/'kv-reproduction.json', diagnostic)
            result['kv_reproduction'] = dict(diagnostic['summary'], report=str(out/'kv-reproduction.json'))
    except Exception as error:
        result.update(status='FAILED',error=f'{type(error).__name__}: {error}')
    finally:
        if complete and hdc and plan:
            errors=[]
            for lock in (PurePosixPath(plan['board_model_cache'])/'.execution-lock',GLOBAL_LOCK):
                try:
                    _,owner=hdc.shell('if test ! -e '+q(str(lock))+'; then echo ABSENT; elif '
                        +_guard(lock,plan['token'])+'; then echo OWNED; else echo OTHER; fi','accuracy-lock-owner')
                    if owner=='OWNED':_release_global(hdc,lock,plan['token'])
                    elif owner not in ('ABSENT','OTHER'):raise ValueError('unknown lock owner')
                except Exception as error:errors.append(str(error))
            if errors:result['lock_release_errors']=errors
            else:result['lock_state']='NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION'
        _save(out/'result.json',result)
    return result

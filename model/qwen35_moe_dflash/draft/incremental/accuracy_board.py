"""Detached standalone accuracy request; reuse the already deployed Draft OMC."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile
import uuid

from qwen35_moe_dflash.common.artifacts import output_path
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.execution.runtime_config import read_json
from qwen35_moe_dflash.execution.run_layout import _save, reuse_test_output
from qwen35_moe_dflash.execution.serial_transport import _environment, _build_payloads
from qwen35_moe_dflash.execution.hdc_run import Hdc, _file_record
from qwen35_moe_dflash.execution.lock import _acquire_global, _release_global, _space
from qwen35_moe_dflash.draft.board import _receive
from qwen35_moe_dflash.draft.profile_board import GLOBAL_LOCK, _guard, _identity_guard
from .accuracy import KIND, prepare, compare

PREFIX = '/data/local/tmp/qwen35_draft_accuracy_'
q = shlex.quote


def extract_outputs(archive, destination, meta):
    snapshot_bytes = sum(r['bytes'] for r in meta['outputs'])
    expected = {f'{mode}-{s["index"]}.bin': snapshot_bytes
                for mode in ('isolated', 'feedback') for s in meta['cases']}
    expected['result.json'] = None
    with tarfile.open(archive) as stream:
        members = [m for m in stream.getmembers() if not (m.name in ('.', './') and m.isdir())]
        names = [m.name[2:] if m.name.startswith('./') else m.name for m in members]
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
          timeout_seconds=1800):
    out = output_path(output)
    config_path = Path(config).resolve()
    # These are public shared inputs, never descendants of the disposable slot.
    for value in (config_path, Path(compiled).resolve(), Path(source).resolve(), Path(air).resolve()):
        if value == out or out in value.parents:
            raise ValueError('accuracy output contains a shared source')
    reuse_test_output(out, config=config_path)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(artifact_kind=KIND, status='FAILED', stage='reference', device_execution='NOT_RUN',
        model_compilation='NOT_RUN', numerical_validation='NOT_EVALUATED',
        lock_state='NO_KNOWN_OWNED_LOCK', output=str(out))
    hdc=None; held=[]; uncertain=False; launched=False
    try:
        cfg = read_json(config_path)
        env = read_json(config_path.parent / cfg['assets'])['environment']
        meta = prepare(compiled, source, air, out / 'fixture', atol=atol, rtol=rtol)
        if meta['ddk_id'] != env['ddk_id']:
            raise ValueError('compiled Draft and runtime configuration use different DDKs')
        state.update(status='REFERENCE_PREPARED', stage='prepared')
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
        uploads, search, _ = _build_payloads(out, 600, resident='resident-mixed', native_directory=native)
        uploads = [r for r in uploads if r['remote'].startswith('lib/')]
        uploads.append(_file_record(binary, 'draft-accuracy'))
        archive = out / 'inputs.tar'
        with tarfile.open(archive, 'w') as tar:
            tar.add(out / 'fixture/plan.txt', arcname='plan.txt')
            tar.add(out / 'fixture/inputs', arcname='inputs')
        uploads.append(_file_record(archive, 'inputs.tar'))
        executable, serial, server = _environment()
        token = uuid.uuid4().hex
        board = PurePosixPath(PREFIX+token)
        cache = PurePosixPath('/data/local/tmp/qwen35_real_draft_'+meta['model']['sha256'])
        model = cache / 'model/model.omc'
        locks = [GLOBAL_LOCK, cache / '.execution-lock']
        plan = dict(artifact_kind=KIND, token=token, board_attempt=str(board),
            board_model_cache=str(cache), serial=serial, hdc_server=server, runtime=env,
            model=meta['model'], timeout_seconds=timeout_seconds, fixture='fixture',
            fixture_sha256=sha256_file(out/'fixture/manifest.json'))
        _save(out / 'request.json', plan)
        state.update(board_attempt=str(board), stage='board-preflight')
        hdc=Hdc(executable, serial, server, out, 180)
        for lock in locks:
            uncertain=True
            try: _acquire_global(hdc, lock, token, 0)
            except ValueError as error:
                if 'BOARD_MODEL_IO_BUSY:' in str(error): uncertain=False
                raise
            held.append(lock); uncertain=False
        # One explicit diagnostic content check. This is outside inference timing.
        hdc.shell(_identity_guard(model,meta['model']), 'accuracy-existing-omc', timeout=180)
        _, firmware=hdc.shell('param get const.ohos.fullname','accuracy-firmware')
        if firmware.strip()!=env['firmware_id'].strip():
            raise ValueError('board firmware differs from the runtime configuration')
        _, space=hdc.shell('df -Pk /data/local/tmp','accuracy-space')
        required=archive.stat().st_size*2+2*len(meta['cases'])*sum(r['bytes'] for r in meta['outputs'])*2+(128<<20)
        if _space(space)<required: raise OSError(f'accuracy needs {required} free board bytes')
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
            raise ValueError('native accuracy execution failed; see native.stderr')
        if _receive(hdc,board,out,'archive.exitcode') is None or (out/'archive.exitcode').read_text().strip()!='0':
            raise ValueError('native output archive failed')
        if sha256_file(root/'fixture/manifest.json') != plan['fixture_sha256']:
            raise ValueError('reference fixture changed after device launch')
        meta=read_json(root/'fixture/manifest.json')
        maximum=2*len(meta['cases'])*sum(s['bytes'] for s in meta['outputs'])+(2<<20)
        _,size=hdc.shell('wc -c < '+q(str(board/'outputs.tar')),'accuracy-output-size')
        if not size.strip().isdigit() or not 0<int(size)<maximum:raise ValueError('unexpected accuracy archive extent')
        if _receive(hdc,board,out,'outputs.tar',timeout_seconds=180) is None:raise ValueError('missing output archive')
        extract_outputs(out/'outputs.tar',out/'native',meta)
        result.update(compare(root/'fixture',out/'native'),stage='complete',device_execution='EXECUTED_OUTPUTS_RETRIEVED',
                      fixture=str(root/'fixture'),fixture_sha256=sha256_file(root/'fixture/manifest.json'),
                      model=meta['model'],runtime=plan['runtime'])
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

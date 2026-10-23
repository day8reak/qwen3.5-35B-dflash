"""Host-only HDC transport tests; fake shell/runner provide no NPU evidence."""
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

import pytest

from qwen35_moe_dflash.op_probe_hdc import _run_script, run_board


HDC = r'''
import json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[:4] == ['-t', os.environ['SN'], '-s', os.environ['BIP']]
with open(os.environ['FAKE_CALLS'], 'a') as f: f.write(json.dumps(args)+'\n')
deployment = json.loads(Path(os.environ['FAKE_DEPLOYMENT']).read_text())
remote, local = deployment['board_root'], os.environ['FAKE_BOARD']
def translated(value):
    assert value == remote or value.startswith(remote+'/')
    return Path(local + value[len(remote):])
mode = os.environ.get('FAKE_MODE', '')
args = args[4:]
if args[:2] == ['file', 'send']:
    shutil.copyfile(args[2], translated(args[3]))
elif args[:2] == ['file', 'recv']:
    shutil.copyfile(translated(args[2]), args[3])
    if ((mode == 'corrupt-output' or (mode == 'corrupt-later' and '/cases/later/' in args[2]))
            and args[2].endswith('/output_0.bin')):
        with open(args[3], 'ab') as f: f.write(b'corrupt')
    if mode == 'owner-mismatch' and args[2].endswith('/hilog.exitcode'):
        (Path(local)/'.qwen35-hdc-owner').write_text('another-owner\n')
elif args[:1] == ['shell']:
    assert len(args) == 2
    wrapper = shlex.split(args[1]); assert wrapper[:2] == ['sh', '-c']
    payload = wrapper[2].replace(remote, local)
    if 'param get const.ohos.fullname' in payload:
        replacement = "printf '%s\\n' FAKE-HARMONYOS" if os.environ.get('FAKE_FIRMWARE') != 'empty' else 'false'
        payload = payload.replace('param get const.ohos.fullname', replacement)
    if mode == 'no-marker':
        print('[Fail]Device not founded or connected'); sys.exit(0)
    for name in ('good', 'bad', 'later'):
        script = Path(local)/'cases'/name/'run.sh'
        invocation = 'sh '+shlex.quote(str(script))
        if invocation in payload and script.exists():
            if mode == 'timeout': time.sleep(5)
            content = script.read_text().replace(remote, local)
            payload = payload.replace(invocation, 'sh -c '+shlex.quote(content))
    if 'hilog -x' in payload:
        lines = ['03-06 06:32:41.000 999999 999999 E Other: unrelated-private-entry']
        for row in Path(os.environ['FAKE_NATIVE_RUNS']).read_text().splitlines():
            r = json.loads(row)
            lines.append('03-06 06:32:41.001 %s %s E HIAI: case=%s' % (r['pid'], r['pid'], r['name']))
        if mode == 'hilog-argument':
            lines = ["Mutlti commands can't be used in combination [CODE: -31]"]
        guard = 'test "$#" = 1 && test "$1" = -x || return 9; '
        body = shlex.join(['printf', '%s\n', '\n'.join(lines)])+'; return 0;'
        payload = 'hilog() { '+guard+body+' };\n'+payload
    completed = subprocess.run(['sh', '-c', payload], capture_output=True)
    sys.stdout.buffer.write(completed.stdout); sys.stderr.buffer.write(completed.stderr)
    sys.exit(0) # HDC itself reports zero even when the remote command failed.
else: raise RuntimeError('unexpected HDC command '+repr(args))
'''


RUNNER = r'''
import json, os, sys
from pathlib import Path
flags = dict(zip(sys.argv[1::2], sys.argv[2::2]))
assert set(flags) == {'--model', '--io', '--output'}
assert flags['--model']=='model.omc' and flags['--io']=='io.tsv' and flags['--output']=='results'
assert not Path('results').exists()
name = Path.cwd().name
with open(os.environ['FAKE_NATIVE_RUNS'], 'a') as f:
    f.write(json.dumps({'name':name, 'pid':os.getpid(), 'ld':os.environ['LD_LIBRARY_PATH']})+'\n')
if name == 'bad':
    print(json.dumps({'status':'FAILED', 'stage':'load', 'error':'fake rejection'}))
    print('fake Load rejection', file=sys.stderr); sys.exit(1)
Path('results').mkdir()
Path('results/output_0.bin').write_bytes(Path('input_0.bin').read_bytes())
print(json.dumps({'status':'EXECUTED', 'runtime_version':'FAKE', 'placement':'UNVERIFIED'}))
'''


def _executable(path, body):
    path.write_text('#!'+sys.executable+'\n'+body)
    path.chmod(0o700)
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('SN', 'FAKE-SERIAL')
    monkeypatch.setenv('BIP', '127.0.0.2:8790')
    output = tmp_path/'report'
    monkeypatch.setenv('FAKE_DEPLOYMENT', str(output/'board/deployment.json'))
    monkeypatch.setenv('FAKE_BOARD', str(tmp_path/'fake-board'))
    monkeypatch.setenv('FAKE_CALLS', str(tmp_path/'calls.jsonl'))
    monkeypatch.setenv('FAKE_NATIVE_RUNS', str(tmp_path/'native.jsonl'))
    monkeypatch.delenv('FAKE_MODE', raising=False)
    hdc = _executable(tmp_path/'hdc', HDC)
    runner = _executable(tmp_path/'runner', RUNNER)
    library = tmp_path/'libhiai.so'; library.write_bytes(b'host-only unused library fixture')
    records = []
    for name in ('good', 'bad', 'later'):
        folder = tmp_path/'cases'/name; folder.mkdir(parents=True)
        model, io, data = folder/'model.omc', folder/'io.tsv', folder/'input_0.bin'
        model.write_bytes(b'test fixture, not OMC')
        io.write_text('input\tfloat16\t1,2\tinput_0.bin\noutput\tfloat16\t1,2\toutput_0.bin\n')
        data.write_bytes(b'\x00\x3c\x00\x40')
        records.append(dict(name=name, directory=str(folder), model=str(model), io=str(io), inputs=[str(data)]))
    return dict(output=output, runner=runner, libraries=[library], records=records,
                hdc_executable=str(hdc), timeout=10)


def test_native_failures_continue_and_owned_board_is_cleaned(case, tmp_path, capsys):
    result = run_board(**case)
    assert result['status']=='TRANSPORT_COMPLETE' and result['native_failures']==1
    assert result['board_cleanup']=='REMOVED' and not (tmp_path/'fake-board').exists()
    assert result['firmware']=={'value':'FAKE-HARMONYOS', 'returncode':0}
    assert result['library_search_paths'] == []
    assert json.loads((case['output']/'board/deployment.json').read_text())['library_search_paths'] == []
    progress = capsys.readouterr().err
    assert '[1/3] good: running' in progress and '[2/3] bad: native exit 1' in progress
    assert '[3/3] later: native exit 0' in progress
    actual = {row['name']: row for row in map(json.loads, (tmp_path/'native.jsonl').read_text().splitlines())}
    for name, row in result['cases'].items():
        assert row['pid']==actual[name]['pid']
        assert actual[name]['ld'].split(':')[0]==str(tmp_path/'fake-board/lib')
        assert row['exitcode']==(1 if name=='bad' else 0)
        assert row['hilog_lines']==1
        text = Path(row['hilog']).read_text()
        assert 'case='+name in text and 'unrelated-private' not in text
        assert len(row['output_files'])==(0 if name=='bad' else 1)
        assert all('_evidence' not in Path(file['path']).parts for file in row['files'])
        if name!='bad':
            assert (Path(row['directory'])/'output_0.bin').read_bytes()==b'\x00\x3c\x00\x40'
    assert 'unrelated-private-entry' in (case['output']/'board/_evidence/hilog.txt').read_text()
    calls = list(map(json.loads, (tmp_path/'calls.jsonl').read_text().splitlines()))
    assert all(row[:4]==['-t','FAKE-SERIAL','-s','127.0.0.2:8790'] for row in calls)
    assert 'param get const.ohos.fullname' in ' '.join(calls[0])
    assert sum('hilog -x' in ' '.join(row) for row in calls)==1
    assert 'ring buffer' in result['hilog']['scope']
    assert all('hilog -r' not in ' '.join(row) for row in calls)
    assert not any('&& test -r' in ' '.join(row) for row in calls)


def test_default_library_environment_is_unchanged():
    board = PurePosixPath('/data/local/tmp/probe')
    remote_case = board/'cases/good'
    default = _run_script(board, remote_case)
    assert default == _run_script(board, remote_case, library_search_paths=())
    assert default.splitlines()[2] == (
        'export LD_LIBRARY_PATH=/data/local/tmp/probe/lib"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
    )
    candidate = _run_script(board, remote_case, library_search_paths=('/vendor/lib64/chipsetsdk',))
    assert candidate.splitlines()[2] == (
        'export LD_LIBRARY_PATH=/data/local/tmp/probe/lib:/vendor/lib64/chipsetsdk'
        '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
    )
    assert candidate.splitlines()[:2] + candidate.splitlines()[3:] == default.splitlines()[:2] + default.splitlines()[3:]


def test_existing_library_search_paths_preserve_order_and_are_read_only(case, tmp_path, monkeypatch):
    paths = [tmp_path/'vendor/lib64/chipsetsdk', tmp_path/'vendor/lib64/extra']
    for directory in paths:
        directory.mkdir(parents=True)
        (directory/'sentinel.so').write_bytes(b'board-owned library')
    supplied = tuple(map(str, paths))
    monkeypatch.setenv('LD_LIBRARY_PATH', '/inherited/first:/inherited/second')
    result = run_board(**case, library_search_paths=supplied)
    assert result['library_search_paths'] == list(supplied)
    for basename in ('deployment.json', 'result.json'):
        saved = json.loads((case['output']/'board'/basename).read_text())
        assert saved['library_search_paths'] == list(supplied)
    for row in map(json.loads, (tmp_path/'native.jsonl').read_text().splitlines()):
        assert row['ld'].split(':') == [str(tmp_path/'fake-board/lib'), *supplied,
                                       '/inherited/first', '/inherited/second']
    calls = list(map(json.loads, (tmp_path/'calls.jsonl').read_text().splitlines()))
    checks = [row for row in calls if '&& test -r' in ' '.join(row)]
    assert len(checks) == 2
    for directory, row in zip(paths, checks):
        text = ' '.join(row)
        assert f'test -d {directory} && test -r {directory} && test -x {directory}' in text
    first_mutation = next(i for i, row in enumerate(calls) if 'mkdir ' in ' '.join(row))
    assert all(calls.index(row) < first_mutation for row in checks)
    # No send, mkdir, removal or runner output targets these existing paths.
    for row in calls:
        if any(path in ' '.join(row) for path in supplied):
            assert row in checks
    assert result['board_cleanup'] == 'REMOVED'
    for directory in paths:
        assert list(directory.iterdir()) == [directory/'sentinel.so']
        assert (directory/'sentinel.so').read_bytes() == b'board-owned library'


@pytest.mark.parametrize('kind', ['missing', 'file'])
def test_unavailable_library_directory_fails_before_remote_creation(case, tmp_path, kind):
    path = tmp_path/'candidate-backend'
    if kind == 'file':
        path.write_bytes(b'not a directory')
    with pytest.raises(RuntimeError, match='library-search-path-1'):
        run_board(**case, library_search_paths=(str(path),))
    failure = json.loads((case['output']/'board/failure.json').read_text())
    assert failure['stage'] == 'library-search-paths'
    assert failure['board_cleanup'] == 'NOT_CREATED'
    assert failure['library_search_paths'] == [str(path)]
    assert not (tmp_path/'fake-board').exists()
    assert not (tmp_path/'native.jsonl').exists()
    calls = list(map(json.loads, (tmp_path/'calls.jsonl').read_text().splitlines()))
    assert len(calls) == 2  # firmware query, then the read-only directory check
    assert not any('mkdir ' in ' '.join(row) or row[4:6] == ['file', 'send'] for row in calls)
    if kind == 'file':
        assert path.read_bytes() == b'not a directory'


@pytest.mark.parametrize('paths', [
    '/vendor/lib64', None, {'/vendor/lib64'}, (1,), ('relative/path',),
    ('/vendor/../system',), ('/vendor/lib64:/other',), ('/vendor/with space',),
    ('/vendor/\nlib64',), ('/vendor/\x00lib64',), ('/vendor/\x7flib64',),
    ('/vendor/$(id)',), ('/vendor/`id`',), ('/vendor/a;id',), ("/vendor/a'b",),
    ('//vendor/lib64',), ('/vendor//lib64',), ('/vendor/./lib64',),
    ('/vendor/lib64', '/vendor/lib64'),
])
def test_library_search_path_injection_is_rejected_before_hdc(case, tmp_path, paths):
    with pytest.raises(ValueError, match='library'):
        run_board(**case, library_search_paths=paths)
    with pytest.raises(ValueError, match='library'):
        _run_script(PurePosixPath('/data/local/tmp/probe'), PurePosixPath('/case'),
                    library_search_paths=paths)
    assert not (tmp_path/'calls.jsonl').exists()
    assert not (case['output']/'board').exists()


def test_keep_board_preserves_inputs_outputs_and_scripts(case, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FIRMWARE", "empty")
    result = run_board(**case, keep_board=True)
    assert result['board_cleanup']=='RETAINED'
    assert result['firmware']=={'value':'', 'returncode':1}
    assert (tmp_path/'fake-board/cases/good/model.omc').exists()
    assert (tmp_path/'fake-board/cases/good/run.sh').exists()
    assert (tmp_path/'fake-board/cases/good/results/output_0.bin').exists()


@pytest.mark.parametrize('mode, expected', [
    ('corrupt-output','SHA256'), ('hilog-argument','CODE -31'), ('owner-mismatch','cleanup-board'),
    ('no-marker','missing unique remote exit marker'),
])
def test_transport_and_diagnostic_failures_never_clean_remote(case, tmp_path, monkeypatch, mode, expected):
    monkeypatch.setenv('FAKE_MODE', mode)
    with pytest.raises(RuntimeError, match=expected):
        run_board(**case)
    failed = json.loads((case['output']/'board/failure.json').read_text())
    assert failed['status']=='FAILED'
    if mode!='no-marker':
        assert (tmp_path/'fake-board/.qwen35-hdc-owner').exists()
    calls = list(map(json.loads, (tmp_path/'calls.jsonl').read_text().splitlines()))
    if mode!='owner-mismatch':
        assert not any('rm -rf' in ' '.join(row) for row in calls)
    if mode=='corrupt-output':
        assert not (case['output']/'board/good/output_0.bin').exists()
        assert (case['output']/'board/good/output_0.bin.received').exists()


def test_remote_timeout_retains_owned_directory(case, tmp_path, monkeypatch):
    monkeypatch.setenv('FAKE_MODE', 'timeout')
    case['timeout']=0.8
    with pytest.raises(subprocess.TimeoutExpired):
        run_board(**case)
    assert (tmp_path/'fake-board/.qwen35-hdc-owner').exists()
    failed = json.loads((case['output']/'board/failure.json').read_text())
    assert failed['stage']=='case:good' and failed['board_cleanup']=='RETAINED'


def test_io_traversal_rejected_before_any_hdc_call(case, tmp_path):
    Path(case['records'][0]['io']).write_text('input\tfloat16\t1,2\tinput_0.bin\noutput\tfloat16\t1,2\t../escape.bin\n')
    with pytest.raises(ValueError, match='basename'):
        run_board(**case)
    assert not (tmp_path/'calls.jsonl').exists()


def _prepare_report_records(case, *, mismatch_later=False):
    import numpy as np
    from qwen35_moe_dflash.ge_ir import Graph
    from qwen35_moe_dflash.op_probe import prepare_case

    records = []
    value = np.array([[1, 2]], dtype=np.float16)
    for name in ('good', 'bad', 'later'):
        graph = Graph(name)
        x = graph.input('x', value.shape)
        graph.outputs = [graph.op('cast', x, shape=x.shape, dtype='float16', name='result')]
        expected = value * 2 if mismatch_later and name == 'later' else value
        record = prepare_case(dict(name=name, graph=graph, inputs={'x': value},
            expected=[expected], tolerances=[{'rtol': 0.0, 'atol': 0.0}],
            description='Host-only transport/report integration fixture'), case['output'])
        Path(record['model']).write_bytes(b'host fixture, not an OMC')
        record['compile'] = {'status': 'COMPILED', 'stage': 'complete',
                             'scope': 'host fixture; no real compiler execution'}
        records.append(record)
    case['records'] = records
    return records


def test_prepared_contract_hdc_and_report_integration(case, tmp_path):
    import tarfile
    from qwen35_moe_dflash.op_probe_report import write_report

    records = _prepare_report_records(case, mismatch_later=True)
    board = run_board(**case)
    report = write_report(case['output'], {'status': 'COLLECTED', 'cases': records}, board)
    rows = {row['name']: row for row in report['cases']}
    assert report['board_transport_status'] == 'TRANSPORT_COMPLETE'
    assert report['board_transport']['board_cleanup'] == 'REMOVED'
    assert report['board_transport']['firmware']['value'] == 'FAKE-HARMONYOS'
    assert rows['good']['load'] == rows['good']['run'] == 'PASSED'
    assert rows['good']['numerical']['status'] == 'PASSED'
    assert rows['bad']['load'] == 'FAILED' and rows['bad']['run'] == 'NOT_RUN'
    assert rows['bad']['numerical']['status'] == 'NOT_RUN'
    assert rows['later']['numerical']['outputs'][0]['reason'] == 'MISMATCH'
    assert all(row['placement'] == 'UNVERIFIED' for row in rows.values())
    assert report['summary']['run_passed'] == 2 and report['summary']['numerical_passed'] == 1
    with tarfile.open(case['output']/'results.tar.gz') as archive:
        names = archive.getnames()
        assert 'board/good/hilog.pid.txt' in names
        assert not any('_evidence' in n or '/calls/' in n or n.endswith(('.bin', '.omc')) for n in names)
        logs = ''.join(archive.extractfile(n).read().decode() for n in names if n.endswith('hilog.pid.txt'))
        assert 'unrelated-private-entry' not in logs


def test_partial_hdc_failure_report_recovers_verified_earlier_cases(case, tmp_path, monkeypatch):
    from qwen35_moe_dflash.op_probe_report import write_report

    records = _prepare_report_records(case)
    monkeypatch.setenv('FAKE_MODE', 'corrupt-later')
    with pytest.raises(RuntimeError, match='SHA256'):
        run_board(**case)
    assert not (case['output']/'board/result.json').exists()
    assert (tmp_path/'fake-board/.qwen35-hdc-owner').is_file()
    report = write_report(case['output'], {'status': 'INFRASTRUCTURE_FAILED', 'cases': records})
    rows = {row['name']: row for row in report['cases']}
    assert report['board_transport_status'] == 'FAILED'
    assert report['board_transport']['stage'] == 'case:later'
    assert report['board_transport']['board_cleanup'] == 'RETAINED'
    assert rows['good']['numerical']['status'] == 'PASSED'
    assert rows['bad']['load'] == 'FAILED'
    assert rows['later']['numerical']['outputs'][0]['reason'] == 'MISSING_OUTPUT'
    assert all(row['placement'] == 'UNVERIFIED' for row in rows.values())

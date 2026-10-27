"""OMG child isolation and retained diagnostics; no real vendor compiler claim."""
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
import zipfile

import pytest

from qwen35_moe_dflash.compile import (
    _environment_record, _retain_failure_diagnostics, check_tbe_platform, compile_air, omg_environment,
    prepare_tbe_compatibility, verify_tbe_compatibility,
)
from qwen35_moe_dflash.hashing import sha256_file


@pytest.fixture
def ddk(tmp_path):
    root = tmp_path / "selected-ddk"
    binary = root / "tools/tools_omg/master/omg"
    binary.parent.mkdir(parents=True)
    binary.write_text("unused test converter\n")
    binary.chmod(0o700)
    return root


def _module(root, body):
    package = root / "te_fusion"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "log_util.py").write_text(body)


def test_ddk_te_precedes_conflicting_cann_and_retains_internal_extensions(ddk, tmp_path):
    cann = tmp_path / "server-cann"
    foreign = cann / "python/site-packages"
    selected = ddk / "tools/tools_ascendc/package/python"
    tbe_ops = ddk / "ddk/tbe/ops"
    tbe_ops.mkdir(parents=True)
    extension = tmp_path / "internal-extension"
    extension.mkdir()
    (extension / "internal_marker.py").write_text("VALUE = 'retained'\n")
    _module(foreign, "IDENTITY = 'wrong-cann-version'\n")
    _module(selected, "IDENTITY = 'selected-ddk'\ndef get_py_exception_dict(*args): return {}\n")
    simulator = ddk / "ddk/npu_simulator"
    inherited = dict(os.environ, CANN_ROOT=str(cann),
                     PYTHONPATH=os.pathsep.join([str(foreign), str(extension)]),
                     LD_LIBRARY_PATH=os.pathsep.join([str(cann / "lib64"), str(simulator)]))
    before = dict(inherited)
    _, env = omg_environment(ddk, inherited)
    assert inherited == before
    assert "CANN_ROOT" not in env
    assert env["PYTHONPATH"].split(os.pathsep) == [str(selected), str(tbe_ops), str(extension)]
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [str(ddk / "tools/tools_omg/master/lib64"), str(simulator)]
    assert env["HIAI_VERSION"] == str(ddk / "tools/tools_omg/master")
    assert env["DDK_PATH"] == env["TOOLCHAIN_HOME"] == str(ddk)
    child = subprocess.run([sys.executable, "-c", "from te_fusion import log_util; import internal_marker; "
                            "assert log_util.IDENTITY == 'selected-ddk'; "
                            "assert callable(log_util.get_py_exception_dict); "
                            "assert internal_marker.VALUE == 'retained'"], env=env, check=True,
                           capture_output=True, text=True)
    assert child.returncode == 0
    assert not list(selected.rglob("*.pyc"))


@pytest.mark.parametrize("root_key", ["CANN_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"])
def test_cann_symlink_paths_removed_without_prefix_collisions(ddk, tmp_path, root_key):
    cann = tmp_path / "cann"
    cann.mkdir()
    alias = tmp_path / "latest"
    alias.symlink_to(cann, target_is_directory=True)
    legitimate = tmp_path / "cann-custom"
    values = os.pathsep.join([str(alias / "lib"), str(legitimate)])
    inherited = {root_key: str(cann), "PATH": values, "LD_LIBRARY_PATH": values,
                 "PYTHONPATH": values, "ASCEND_OPP_PATH": str(alias / "opp"),
                 "ASCEND_AICPU_PATH": str(alias / "aicpu")}
    _, env = omg_environment(ddk, inherited)
    for key in ("PATH", "PYTHONPATH"):
        assert env[key] == str(legitimate)
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[-1] == str(legitimate)
    assert "ASCEND_OPP_PATH" not in env and "ASCEND_AICPU_PATH" not in env
    assert root_key not in env


def test_selected_ddk_paths_survive_when_cann_root_is_parent(ddk):
    internal = str(ddk / "ddk/tbe/ops")
    _, env = omg_environment(ddk, {"CANN_ROOT": str(ddk.parent), "PYTHONPATH": internal,
                                  "ASCEND_OPP_PATH": str(ddk / "opp")})
    assert env["PYTHONPATH"] == internal
    assert env["ASCEND_OPP_PATH"] == str(ddk / "opp")


def test_environment_diagnostics_list_candidates_without_import(ddk, tmp_path):
    selected = ddk / "tools/tools_ascendc/package/python"
    _module(selected, "raise RuntimeError('diagnostics must not import vendor code')\n")
    unknown = tmp_path / "unknown-package-root"
    _module(unknown, "raise RuntimeError('must not import')\n")
    _, env = omg_environment(ddk, {"PYTHONPATH": str(unknown), "PRIVATE_TOKEN": "do-not-record"})
    record = _environment_record(ddk, env)
    assert record["te_fusion_log_util_candidates"] == [str(selected / "te_fusion/log_util.py"),
                                                       str(unknown / "te_fusion/log_util.py")]
    assert "PRIVATE_TOKEN" not in record["environment"]
    assert "site-packages may differ" in record["candidate_note"]


@pytest.mark.parametrize("native_return", ["success", "MOCK unknown soc kirinx90"])
def test_platform_check_keeps_raw_native_return_in_isolated_real_child(ddk, tmp_path, monkeypatch, native_return):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    package = ddk / "tools/tools_ascendc/package/python/tbe"
    platform = package / "common/platform"
    platform.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "common/__init__.py").write_text("")
    (platform / "__init__.py").write_text("def get_soc_spec(key): return 'MOCK_INITIAL_SOC'\n")
    source = platform / "platform_info.py"
    source.write_text("import json, pathlib\n"
                      "def get_global_func(name):\n"
                      "    assert name == 'cce.te_update_version'\n"
                      "    def native(*args):\n"
                      "        pathlib.Path('calls.json').write_text(json.dumps([args]))\n"
                      f"        return {native_return!r}\n"
                      "    return native\n")
    before = source.read_bytes(), dict(os.environ)
    output = tmp_path / "check"
    result = check_tbe_platform(ddk, output, python=sys.executable)
    native = result["native"]
    assert native["native_return_repr"] == repr(native_return)
    assert native["native_return_type"] == "str"
    assert native["before"]["SOC_VERSION"] == 'MOCK_INITIAL_SOC'
    assert native["initialization"] == "IMPORT_ONLY" and native["worker_reproduction"] == "NOT_ESTABLISHED"
    assert result["status"] == ("RAW_UPDATE_SUCCEEDED" if native_return == "success" else "FAILED")
    assert result["compiler_execution"] == result["device_execution"] == "NOT_RUN"
    assert json.loads((output / 'calls.json').read_text()) == [["kirinx90", "AiCore", "", "false"]]
    assert native["modules"]["tbe.common.platform.platform_info"]["sha256"] == sha256_file(source)
    assert source.read_bytes() == before[0] and dict(os.environ) == before[1]
    assert not list(package.rglob('*.pyc')) and not list(output.glob('.ipc-*'))
    assert (output / 'report.txt').is_file() and (output / 'omg-environment.json').is_file()


def test_platform_check_retains_import_errors_and_timeout(ddk, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    package = ddk / "tools/tools_ascendc/package/python/tbe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('MOCK import error')\n")
    result = check_tbe_platform(ddk, tmp_path / "import-failed", python=sys.executable)
    assert result["status"] == "FAILED" and result["native"]["stage"] == "IMPORT"
    assert 'MOCK import error' in result["native"]["traceback"]
    def timeout(argv, **kwargs):
        kwargs['stdout'].write('MOCK hung import\n')
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    result = check_tbe_platform(ddk, tmp_path / 'timeout', python=sys.executable, timeout_seconds=1)
    assert result['status'] == 'FAILED' and result['returncode'] is None
    assert 'timed out' in result['error']
    assert 'MOCK hung import' in (tmp_path / 'timeout/report.txt').read_text()
    assert not list((tmp_path / 'timeout').glob('.ipc-*'))


def _platform_comparison_packages(ddk):
    python = ddk / "tools/tools_ascendc/package/python"
    for relative in ("tbe", "tbe/common", "tbe/common/platform", "te", "te/platform"):
        package = python / relative
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
    (python / "tbe/common/platform/__init__.py").write_text(
        "initialized = False\ndef get_soc_spec(key): return 'kirinx90' if initialized else 'MOCK_IMPORT_SOC'\n")
    (python / "tbe/common/platform/platform_info.py").write_text(
        "import os, json, pathlib\nfrom tbe.common import platform\n"
        "def get_global_func(name):\n"
        "    def native(*args):\n"
        "        pathlib.Path('calls.json').write_text(json.dumps([args]))\n"
        "        first = os.environ['LD_LIBRARY_PATH'].split(os.pathsep)[0]\n"
        "        return 'success' if first.endswith('/ddk/tbe/lib64') and platform.initialized else 'error'\n"
        "    return native\n")
    return python


def test_platform_comparison_uses_four_fresh_children_and_same_target_only(ddk, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    python = _platform_comparison_packages(ddk)
    source = python / "te/platform/cce_conf.py"
    source.write_text("from tbe.common import platform\n"
                      "def te_set_version(soc, core_type='AiCore'):\n"
                      "    assert soc == 'kirinx90' and not platform.initialized\n"
                      "    platform.initialized = True\n")
    (ddk / "ddk/tbe/lib64").mkdir(parents=True)
    before = dict(os.environ)
    result = check_tbe_platform(ddk, tmp_path / "comparison", python=sys.executable, compare_libraries=True)
    assert result['status'] == 'COMPARISON_COMPLETE'
    assert result['successful_arms'] == ['tbe_first_initialized']
    assert len(result['cases']) == 4 and dict(os.environ) == before
    for case in result['cases']:
        assert case['native']['before']['SOC_VERSION'] == 'MOCK_IMPORT_SOC'
        assert case['native']['native_arguments'] == ['kirinx90', 'AiCore', '', 'false']
        folder = tmp_path / 'comparison' / case['name']
        assert json.loads((folder / 'calls.json').read_text()) == [['kirinx90', 'AiCore', '', 'false']]
        env = json.loads((folder / 'omg-environment.json').read_text())['environment']
        first = 'ddk/tbe/lib64' if case['name'].startswith('tbe_first') else 'tools/tools_omg/master/lib64'
        assert env['LD_LIBRARY_PATH'].split(os.pathsep)[0] == str(ddk / first)
        if case['name'].endswith('_initialized'):
            assert case['native']['after_init']['SOC_VERSION'] == 'kirinx90'
            assert case['native']['init_api']['sha256'] == sha256_file(source)
    assert not list(python.rglob('*.pyc')) and not list((tmp_path / 'comparison').rglob('.ipc-*'))


def test_platform_initialization_rejects_unknown_signature_and_records_missing_library_arm(ddk, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    python = _platform_comparison_packages(ddk)
    (python / "te/platform/cce_conf.py").write_text(
        "def te_set_version(soc, required_unknown):\n    raise AssertionError('MUST NOT CALL')\n")
    result = check_tbe_platform(ddk, tmp_path / 'comparison', python=sys.executable, compare_libraries=True)
    assert [case['status'] for case in result['cases']] == ['FAILED', 'INIT_API_UNRESOLVED', 'UNAVAILABLE', 'UNAVAILABLE']
    assert not (tmp_path / 'comparison/omg_first_initialized/calls.json').exists()
    assert 'required_unknown' in result['cases'][1]['native']['init_api']['signature']
    assert 'missing' in result['cases'][1]['native']['error']


@pytest.mark.parametrize("init_raises", [False, True])
def test_platform_comparison_all_failures_are_collected_and_init_failure_skips_update(ddk, tmp_path, monkeypatch, init_raises):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    python = _platform_comparison_packages(ddk)
    body = "raise RuntimeError('MOCK init failed')" if init_raises else "return None"
    (python / 'te/platform/cce_conf.py').write_text("def te_set_version(soc):\n    " + body + "\n")
    (ddk / 'ddk/tbe/lib64').mkdir(parents=True)
    result = check_tbe_platform(ddk, tmp_path / 'comparison', python=sys.executable, compare_libraries=True)
    assert result['status'] == 'COMPARISON_COMPLETE' and result['successful_arms'] == []
    assert [case['status'] for case in result['cases']] == ['FAILED'] * 4
    for case in result['cases']:
        if init_raises and case['name'].endswith('_initialized'):
            assert case['native']['stage'] == 'INITIALIZE'
            assert not (tmp_path / 'comparison' / case['name'] / 'calls.json').exists()
        else:
            assert case['native']['stage'] == 'COMPLETE' and case['native']['native_return_repr'] == "'error'"


def test_platform_diagnosis_batches_alias_package_library_variants_with_real_children(ddk, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    python = _platform_comparison_packages(ddk)
    (python / 'te_fusion').mkdir()
    (python / 'te_fusion/__init__.py').write_text('')
    (python / 'tbe/common/platform/__init__.py').write_text(
        "soc = 'unset'\ndef get_soc_spec(key): return soc\n")
    (python / 'te/platform/cce_conf.py').write_text(
        "from tbe.common import platform\ndef te_set_version(soc):\n"
        "    assert soc == 'kirinx90'\n    platform.soc = 'InstalledAlias'\n")
    (python / 'tbe/common/platform/platform_info.py').write_text(
        "import os, time, json, pathlib, te_fusion\nfrom tbe.common import platform\n"
        "def get_global_func(name):\n"
        "    def native(soc, *args):\n"
        "        if pathlib.Path.cwd().name == 'current_tbe_first_requested': os._exit(7)\n"
        "        start = time.time()\n        time.sleep(.2)\n"
        "        pathlib.Path('window.json').write_text(json.dumps([start, time.time()]))\n"
        "        if soc == '': platform.soc = 'cleared'\n"
        "        return 'success' if soc in ('InstalledAlias', '') else 'error'\n"
        "    return native\n")
    folder = ddk / 'ddk/tbe/lib64'
    folder.mkdir(parents=True)
    archive = folder / 'te-test.whl'
    with zipfile.ZipFile(archive, 'w') as wheel:
        for path in python.rglob('*.py'):
            wheel.write(path, path.relative_to(python))
    before = {path: path.read_bytes() for path in ddk.rglob('*') if path.is_file()}
    output = tmp_path / 'diagnosis'
    result = check_tbe_platform(ddk, output, python=sys.executable, diagnose=True, jobs=4)
    assert result['status'] == 'DIAGNOSIS_COMPLETE' and len(result['cases']) == 12
    assert len(result['candidate_arms']) == 4 and all(name.endswith('_reported') for name in result['candidate_arms'])
    assert result['wheel']['sha256'] == sha256_file(archive)
    assert len((output / 'progress.jsonl').read_text().splitlines()) == 12
    assert len(capsys.readouterr().out.splitlines()) == 12
    events = []
    for case in result['cases']:
        native = case['native']
        assert native['init_api']['arguments'] == ['kirinx90']
        if case['name'] == 'current_tbe_first_requested':
            assert case['returncode'] == 7 and case['status'] == 'FAILED'
        else:
            start, end = json.loads((output / case['name'] / 'window.json').read_text())
            events += [(start, 1), (end, -1)]
        if case['update_mode'] == 'reported':
            assert native['native_arguments'][0] == 'InstalledAlias'
        elif case['update_mode'] == 'empty':
            assert native['native_arguments'][0] == '' and native['initialized_soc_preserved'] is False
        if case['package'] == 'wheel':
            assert native['package_provenance']['status'] == 'ISOLATED'
            assert all(Path(desc['path']).is_relative_to(output / 'wheel-python')
                       for desc in native['modules'].values() if desc['path'])
    active = maximum = 0
    for _, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    assert 2 <= maximum <= 4
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not list(ddk.rglob('*.pyc')) and not list(output.rglob('.ipc-*'))


@pytest.mark.parametrize('kind', ['traversal', 'symlink', 'multiple', 'incomplete'])
def test_platform_diagnosis_rejects_unsafe_wheels_and_never_falls_back_vendor_packages(ddk, tmp_path, monkeypatch, kind):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    python = _platform_comparison_packages(ddk)
    (python / 'te/platform/cce_conf.py').write_text('def te_set_version(soc): pass\n')
    folder = ddk / 'ddk/tbe/lib64'
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / 'te-test.whl', 'w') as wheel:
        if kind == 'traversal':
            wheel.writestr('../escaped.py', 'invalid')
        elif kind == 'symlink':
            info = zipfile.ZipInfo('tbe')
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            wheel.writestr(info, '/outside')
        else:
            wheel.writestr('te/__init__.py', '')  # no tbe: existing vendor package must not supply it
    if kind == 'multiple':
        (folder / 'te-other.whl').write_bytes(b'not used')
    output = tmp_path / 'diagnosis'
    result = check_tbe_platform(ddk, output, python=sys.executable, diagnose=True, jobs=2)
    wheel_cases = [case for case in result['cases'] if case['package'] == 'wheel']
    if kind == 'incomplete':
        assert result['wheel']['status'] == 'READY'
        assert all(case['status'] == 'FAILED' and 'wheel isolation: missing vendor module tbe' in case['native']['error']
                   for case in wheel_cases)
    else:
        assert result['wheel']['status'] == 'UNAVAILABLE'
        assert all(case['status'] == 'UNAVAILABLE' for case in wheel_cases)
    assert not (output / 'escaped.py').exists()


@pytest.mark.parametrize('soc', ['unknown', 'error'])
def test_diagnosis_raw_success_with_unresolved_soc_is_not_a_candidate(ddk, tmp_path, monkeypatch, soc):
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    python = _platform_comparison_packages(ddk)
    body = "raise RuntimeError('lookup failed')" if soc == 'error' else "return 'unknown'"
    (python / 'tbe/common/platform/__init__.py').write_text('def get_soc_spec(key): ' + body + '\n')
    (python / 'te/platform/cce_conf.py').write_text('def te_set_version(soc): pass\n')
    (python / 'tbe/common/platform/platform_info.py').write_text("def get_global_func(name): return lambda *a: 'success'\n")
    result = check_tbe_platform(ddk, tmp_path / 'diagnosis', python=sys.executable, diagnose=True)
    assert result['candidate_arms'] == []
    cases = {case['name']: case for case in result['cases']}
    assert cases['current_omg_first_reported']['status'] == 'FAILED'
    for mode in ('requested', 'empty'):
        native = cases['current_omg_first_' + mode]['native']
        assert native['native_success'] is True and native['initialized_soc_preserved'] is False


def _compatibility_diagnosis(ddk, tmp_path, monkeypatch, *, chained=False, duplicate=False):
    import _ctypes
    monkeypatch.setenv('AI_RUN_DIR', str(tmp_path))
    python = _platform_comparison_packages(ddk)
    library = ddk / 'tools/tools_omg/master/lib64/libplatform.so'
    library.parent.mkdir(parents=True)
    library.write_bytes(Path(_ctypes.__file__).read_bytes())  # real ELF mapping, mocked platform behavior
    (python / 'tbe/common/platform/__init__.py').write_text("soc='unset'\ndef get_soc_spec(key): return soc\n")
    (python / 'te/platform/cce_conf.py').write_text(
        "from tbe.common import platform\ndef te_set_version(soc):\n"
        "    assert soc == 'kirinx90'\n    platform.soc = 'InstalledAlias'\n")
    native = ("import pathlib, json, ctypes\nfrom tbe.common import platform\n"
              f"mapped_test_library = ctypes.CDLL({str(library)!r})\n"
              "def get_global_func(name):\n"
              "    def native(*args):\n"
              "        pathlib.Path('called-args.json').write_text(json.dumps(args))\n"
              "        return 'success' if args[0] == 'InstalledAlias' else 'error'\n"
              "    return native\n"
              "def te_update_version(soc_version, core_type='AiCore', aicore_num='', l1_fusion='false'):\n")
    call = ("    value = get_global_func('cce.te_update_version')(soc_version, core_type, aicore_num, l1_fusion)\n"
            if chained else "    func = get_global_func('cce.te_update_version')\n"
                            "    value = func(soc_version, core_type, aicore_num, l1_fusion)\n")
    (python / 'tbe/common/platform/platform_info.py').write_text(native + call * (2 if duplicate else 1) + '    return value\n')
    folder = ddk / 'ddk/tbe/lib64'
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / 'te-test.whl', 'w') as wheel:
        for path in python.rglob('*.py'):
            wheel.write(path, path.relative_to(python))
    diagnosis = tmp_path / 'diagnosis'
    check_tbe_platform(ddk, diagnosis, python=sys.executable, diagnose=True)
    return diagnosis


@pytest.mark.parametrize('chained', [False, True])
def test_run_local_compatibility_applies_only_to_initialized_target_and_audits_real_child(ddk, tmp_path, monkeypatch, chained):
    diagnosis = _compatibility_diagnosis(ddk, tmp_path, monkeypatch, chained=chained)
    before = {path: path.read_bytes() for path in ddk.rglob('*') if path.is_file()}
    inherited = omg_environment(ddk)[1]
    env, metadata = prepare_tbe_compatibility(ddk, diagnosis, tmp_path / 'compat', inherited=inherited)
    assert metadata['status'] == 'PREPARED' and verify_tbe_compatibility(metadata)['status'] == 'NOT_OBSERVED'
    assert env['SOC_VERSION'] == 'kirinx90' and env['LD_LIBRARY_PATH'] == inherited['LD_LIBRARY_PATH']
    assert env['PYTHONPATH'] != inherited.get('PYTHONPATH')
    script = ("from te.platform import cce_conf\nfrom tbe.common.platform import platform_info as pi\n"
              "cce_conf.te_set_version('kirinx90')\n"
              "assert pi.te_update_version('kirinx90','VectorCore','9','true') == 'success'\n")
    subprocess.run([sys.executable, '-c', script], cwd=tmp_path / 'compat', env=env, check=True, capture_output=True, text=True)
    assert json.loads((tmp_path / 'compat/called-args.json').read_text()) == ['InstalledAlias', 'VectorCore', '9', 'true']
    audit = verify_tbe_compatibility(metadata)
    assert audit['status'] == 'APPLIED' and audit['events'][0]['pid'] != os.getpid()
    assert audit['events'][0]['module_path'] == metadata['adapted_module']['path']
    subprocess.run([sys.executable, '-c', "from tbe.common.platform import platform_info as pi; pi.te_update_version('other','X','4','false')"],
                   cwd=tmp_path / 'compat', env=env, check=True, capture_output=True, text=True)
    assert json.loads((tmp_path / 'compat/called-args.json').read_text()) == ['other', 'X', '4', 'false']
    assert verify_tbe_compatibility(metadata)['events'][-1]['event'] == 'PASSTHROUGH'
    failed = subprocess.run([sys.executable, '-c', "from tbe.common.platform import platform_info as pi; pi.te_update_version('kirinx90')"],
                            cwd=tmp_path / 'compat', env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and 'requires the evidence-bound initialized SOC' in failed.stderr
    assert verify_tbe_compatibility(metadata)['status'] == 'REJECTED'
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not list(ddk.rglob('*.pyc'))


def test_compatibility_rejects_ambiguous_source_and_changed_wheel(ddk, tmp_path, monkeypatch):
    diagnosis = _compatibility_diagnosis(ddk, tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match='ambiguous native|handle is rebound'):
        prepare_tbe_compatibility(ddk, diagnosis, tmp_path / 'ambiguous')
    archive = ddk / 'ddk/tbe/lib64/te-test.whl'
    archive.write_bytes(archive.read_bytes() + b'changed wheel identity')
    with pytest.raises(ValueError, match='wheel differs'):
        prepare_tbe_compatibility(ddk, diagnosis, tmp_path / 'changed')


def test_compatibility_rejects_mixed_vendor_module_origins_in_actual_worker(ddk, tmp_path, monkeypatch):
    diagnosis = _compatibility_diagnosis(ddk, tmp_path, monkeypatch)
    env, metadata = prepare_tbe_compatibility(ddk, diagnosis, tmp_path / 'compat')
    script = ("import sys, types\nfrom te.platform import cce_conf\nfrom tbe.common.platform import platform_info as pi\n"
              "cce_conf.te_set_version('kirinx90')\nforeign=types.ModuleType('tbe.foreign')\n"
              "foreign.__file__='/foreign/tbe/foreign.py'\nsys.modules['tbe.foreign']=foreign\npi.te_update_version('kirinx90')\n")
    child = subprocess.run([sys.executable, '-c', script], cwd=tmp_path / 'compat', env=env, capture_output=True, text=True)
    assert child.returncode != 0 and 'module provenance mismatch' in child.stderr
    assert verify_tbe_compatibility(metadata)['status'] == 'REJECTED'
    assert not (tmp_path / 'compat/called-args.json').exists()


def test_diagnostic_retention_prioritizes_failure_and_respects_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    scratch = tmp_path / "scratch"
    source = scratch / "dump_graph"
    source.mkdir(parents=True)
    output = tmp_path / "output"
    output.mkdir()
    (source / "01_intermediate.irpb").write_bytes(b"too large" * 5)
    (source / "02_fail.irpb").write_bytes(b"failure graph")
    secret = tmp_path / "input-weight"
    secret.write_bytes(b"must not collect")
    (source / "03_symlink.irpb").symlink_to(secret)
    (source / "not-a-graph.bin").write_bytes(b"not a graph")
    _retain_failure_diagnostics(scratch, output, max_bytes=16)
    destination = output / "diagnostics"
    assert (destination / "02_fail.irpb").read_bytes() == b"failure graph"
    assert sorted(path.name for path in destination.iterdir()) == ["02_fail.irpb", "manifest.json"]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["retained_bytes"] == 13
    assert [row["status"] for row in manifest["files"]] == ["RETAINED", "SKIPPED_SIZE_BUDGET"]
    assert manifest["files"][0]["sha256"] == sha256_file(destination / "02_fail.irpb")


def test_diagnostic_retention_does_not_follow_dump_directory_symlink(tmp_path):
    (tmp_path / "scratch").mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "input.irpb").write_bytes(b"source model")
    (tmp_path / "scratch/dump_graph").symlink_to(private, target_is_directory=True)
    _retain_failure_diagnostics(tmp_path / "scratch", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("diagnostics_fail", [False, True])
def test_converter_failure_keeps_cause_and_cleans_scratch(ddk, tmp_path, monkeypatch, diagnostics_fail):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    bundle = tmp_path / "air"
    bundle.mkdir()
    keys = ("sequence_length", "draft_width", "feature_width", "vocab_size", "precision", "state_policy",
            "expert_dispatch", "source", "pad_token_id", "eos_token_ids", "precision_audit", "quantization")
    manifest = dict.fromkeys(keys)
    manifest.update(artifact_kind="qwen35-cpro-air-pair", status="EXPORTED", runtime="hiai",
                    abi_version=2, feature_dtype="torch.float16", export_root=str(bundle), graphs=[])
    for name in ("target", "draft"):
        air = bundle / f"{name}.air"
        air.write_bytes(b"test double, not AIR")
        manifest["graphs"].append({"name": name, "air": air.name,
                                   "files": [{"path": air.name, "sha256": sha256_file(air), "bytes": air.stat().st_size}]})
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    def failed_run(command, *, cwd, stdout, **kwargs):
        dump = cwd / "dump_graph"
        dump.mkdir()
        (dump / "02_fail.irpb").write_bytes(b"fake failed graph")
        stdout.write("mock OMG first error\n")
        raise subprocess.CalledProcessError(19, command)

    monkeypatch.setattr("qwen35_moe_dflash.compile.subprocess.run", failed_run)
    if diagnostics_fail:
        def fail_copy(*args):
            raise OSError("test diagnostic disk failure")
        monkeypatch.setattr("qwen35_moe_dflash.compile.shutil.copyfile", fail_copy)
    with pytest.raises(RuntimeError) as caught:
        compile_air(bundle, tmp_path / "omc", ddk, ddk_id="test-double")
    assert caught.value.__cause__.returncode == 19
    failure = json.loads((tmp_path / "omc/failure.json").read_text())
    assert failure["returncode"] == 19 and failure["stage"] == "target-omg"
    assert not list((tmp_path / "omc").glob("omg-*"))
    assert (tmp_path / "omc/target/omg-environment.json").is_file()
    if diagnostics_fail:
        assert "test diagnostic disk failure" in str(caught.value)
    else:
        assert (tmp_path / "omc/target/diagnostics/02_fail.irpb").read_bytes() == b"fake failed graph"

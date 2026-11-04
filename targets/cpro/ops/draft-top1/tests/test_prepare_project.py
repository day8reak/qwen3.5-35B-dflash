#!/usr/bin/env python3
"""Exercise preparation with a synthetic generator; never run a vendor build."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_project


def check(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    ddk = root / 'ddk'
    ddk.mkdir()
    checks = []
    def generator(mode):
        path = root / ('msopgen-' + mode)
        path.write_text('''#!/usr/bin/env python3
from pathlib import Path
import json,sys
mode=Path(sys.argv[0]).name.removeprefix('msopgen-')
p=Path(sys.argv[sys.argv.index('-out')+1])
if mode=='nested': p=p/'StableArgMax'
(p/'op_host').mkdir(parents=True)
(p/'op_kernel').mkdir()
for name in ['op_host/stable_arg_max.cpp','op_host/stable_arg_max_tiling.h',
             'op_kernel/stable_arg_max.cpp','CMakeLists.txt','build.sh']:
 (p/name).write_text('stable_arg_max')
if mode=='missing': (p/'op_host/stable_arg_max_tiling.h').unlink()
if mode=='wrong-symbol': (p/'op_kernel/stable_arg_max.cpp').write_text('wrong_entry')
if mode=='linked':
 f=p/'op_kernel/stable_arg_max.cpp'; f.unlink(); f.symlink_to(p/'op_host/stable_arg_max.cpp')
variables={'ASCEND_COMPUTE_UNIT':{'type':'STRING','value':'kirin9020' if mode=='default-unit' else 'kirinx90'}}
if mode=='no-unit': variables={}
(p/'CMakePresets.json').write_text(json.dumps({'configurePresets':[{'cacheVariables':variables}]}))
''')
        path.chmod(0o755)
        return str(path)
    for mode in ('valid', 'nested', 'default-unit'):
        report = prepare_project.prepare(root / mode, ddk, generator(mode))
        assert report['status'] == 'SOURCE_PROJECT_PREPARED'
        assert report['ddk_installation'] == report['kernel_compilation'] == 'NOT_RUN'
        project = Path(report['project'])
        presets = json.loads((project / 'CMakePresets.json').read_text())
        assert presets['configurePresets'][0]['cacheVariables']['ASCEND_COMPUTE_UNIT']['value'] == 'kirinx90'
        assert report['source_sha256']['op_kernel/stable_arg_max.cpp'] == report['copied_sha256']['op_kernel/stable_arg_max.cpp']
        assert list(ddk.iterdir()) == []
        checks.append(mode)
    for mode in ('missing', 'wrong-symbol', 'linked', 'no-unit'):
        try:
            prepare_project.prepare(root / mode, ddk, generator(mode))
        except ValueError:
            assert json.loads((root / mode / 'prepare-report.json').read_text())['status'] == 'PREPARATION_FAILED'
            checks.append('reject/' + mode)
        else:
            raise AssertionError('malformed template accepted: ' + mode)
    for name, out, gen in [('overwrite', root / 'valid', generator('valid')),
                           ('sdk-output', ddk / 'output', generator('valid')),
                           ('ancestor-output', root, generator('valid')),
                           ('missing-generator', root / 'no-generator', str(root / 'absent'))]:
        try:
            prepare_project.prepare(out, ddk, gen)
        except (ValueError, FileExistsError):
            checks.append('reject/' + name)
        else:
            raise AssertionError('unsafe preparation accepted: ' + name)
    assert not (root / 'no-generator').exists()
    report = {'status': 'HOST_PASSED', 'scope': 'SYNTHETIC_GENERATOR_ONLY',
        'checks_passed': len(checks), 'checks': checks,
        'prepare_source_sha256': prepare_project.sha(Path(prepare_project.__file__)),
        'model_compilation': 'NOT_RUN', 'device_execution': 'NOT_RUN'}
    (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(check(parser.parse_args().output)))

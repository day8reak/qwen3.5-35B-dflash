#!/usr/bin/env python3
"""Generate a kirinx90 project using the user's selected msopgen; overlay sources.

Does not run vendor build/install scripts. The generated CMake scaffold belongs
to that SDK, not a hand-written approximation of a different Ascend platform.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


FILES = ('stable_arg_max.json', 'op_host/stable_arg_max.cpp',
         'op_host/stable_arg_max_tiling.h', 'op_kernel/stable_arg_max.cpp')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, ddk_root, msopgen='msopgen'):
    source = Path(__file__).resolve().parents[1]
    ddk = Path(ddk_root).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if not ddk.is_dir():
        raise ValueError('DDK root must be a directory')
    for protected in (ddk, source):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Output must be outside source and selected DDK trees')
    generator = shutil.which(msopgen)
    if generator is None:
        raise ValueError('msopgen unavailable; source the selected CANN Kit environment first')
    if output.exists():
        raise FileExistsError('Use a new output directory; existing projects are never replaced')
    prototype = json.loads((source / FILES[0]).read_text())
    expected = [{'op': 'StableArgMax',
        'input_desc': [{'name': 'x', 'param_type': 'required', 'format': ['ND'], 'type': ['fp16']}],
        'output_desc': [{'name': 'indices', 'param_type': 'required', 'format': ['ND'], 'type': ['int32']}]}]
    if prototype != expected:
        raise ValueError('Prototype name/ports/dtypes differ from the frozen registration')
    output.mkdir(parents=True, exist_ok=False)
    command = [generator, 'gen', '-i', str(source / FILES[0]), '-c', 'ai_core-kirinx90',
               '-lan', 'cpp', '-out', str(output / 'project')]
    report = {'artifact_kind': 'stable-arg-max-source-project', 'status': 'PREPARING',
        'platform': 'kirinx90', 'ddk_root': str(ddk), 'generator': str(Path(generator).resolve()),
        'generator_sha256': sha(Path(generator).resolve()), 'command': command,
        'source_sha256': {name: sha(source / name) for name in FILES},
        'host_compilation': 'NOT_RUN', 'kernel_compilation': 'NOT_RUN',
        'ddk_installation': 'NOT_RUN', 'device_execution': 'NOT_RUN'}
    report_file = output / 'prepare-report.json'
    try:
        with (output / 'msopgen.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        candidates = [p.parent for p in output.rglob('CMakePresets.json')
                      if (p.parent / 'op_host').is_dir() and (p.parent / 'op_kernel').is_dir()]
        if len(candidates) != 1:
            raise ValueError('Expected one generated project containing op_host/op_kernel/CMakePresets.json')
        project = candidates[0].resolve()
        if output not in project.parents:
            raise ValueError('Generated project escapes the requested output directory')
        for name in ('CMakeLists.txt', 'build.sh', *FILES[1:]):
            target = project / name
            if target.is_symlink() or not target.is_file() or project not in target.resolve().parents:
                raise ValueError('Missing or linked generated file: ' + name)
        # Require the generator's actual symbol spelling before replacing code.
        if 'stable_arg_max' not in (project / FILES[-1]).read_text():
            raise ValueError('Generated kernel entry differs from stable_arg_max')
        presets_path = project / 'CMakePresets.json'
        if presets_path.is_symlink():
            raise ValueError('Refusing to edit a linked CMakePresets.json')
        presets = json.loads(presets_path.read_text())
        matched = False
        prior_units = []
        for preset in presets.get('configurePresets', []):
            variables = preset.get('cacheVariables', {})
            if 'ASCEND_COMPUTE_UNIT' in variables:
                current = variables['ASCEND_COMPUTE_UNIT']
                value = current.get('value') if isinstance(current, dict) else current
                prior_units.append(value)
                # Official templates may retain their example's default unit.
                # Pin the generated project to the explicitly requested X90.
                if isinstance(current, dict):
                    current['value'] = 'kirinx90'
                else:
                    variables['ASCEND_COMPUTE_UNIT'] = 'kirinx90'
                matched = True
        if not matched:
            raise ValueError('Selected template did not declare ASCEND_COMPUTE_UNIT; inspect msopgen.log')
        presets_path.write_text(json.dumps(presets, indent=2) + '\n')
        for name in FILES[1:]:
            shutil.copyfile(source / name, project / name)
        report.update(status='SOURCE_PROJECT_PREPARED', project=str(project),
            copied_sha256={name: sha(project / name) for name in FILES[1:]},
            cmake_presets_sha256=sha(presets_path),
            original_compute_units=prior_units,
            next='Run the generated build.sh in the selected SDK environment; it installs into that DDK. '
                 'Source publication does not prove kernel compilation; build_devices.sh/OMG and device checks remain required.',
            build_command='cd ' + shlex.quote(str(project)) + ' && bash ./build.sh')
    except Exception as error:
        report.update(status='PREPARATION_FAILED', error=str(error))
        raise
    finally:
        report_file.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ddk-root', default=os.environ.get('CPRO_DDK_ROOT'))
    parser.add_argument('--msopgen', default='msopgen')
    args = parser.parse_args()
    if not args.ddk_root:
        parser.error('--ddk-root or CPRO_DDK_ROOT is required')
    try:
        print(json.dumps(prepare(args.output, args.ddk_root, args.msopgen)))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()

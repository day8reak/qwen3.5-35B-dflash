"""Collect data_proc_tool's export after one successful V2 decoder invocation."""
from pathlib import PurePosixPath
import re
import shlex
import shutil

from ...execution.hdc_run import _relative
from ..layer_probe.common import save_json


def collect(hdc, remote, out, tool_dir, receipt):
    destination = remote / 'operator-dump'
    parser = tool_dir / 'data_proc_tool_internal'
    argv = [str(parser), f'--result_path={destination}/']
    report = dict(status='PARSER_REQUESTED', argv=argv, device_dir=str(destination), files={})
    receipt['operator_dump'] = report
    save_json(out/'result.json', receipt)
    # The model process has exited and we still own the board lock. The parser can
    # export service history, so keep its directory structure for model attribution.
    print(f'解析算子 dump → {destination}', flush=True)
    _, code = hdc.shell(f'cd {remote} && mkdir {destination} && '
                        f'{{ env LD_LIBRARY_PATH={tool_dir} {shlex.join(argv)} >parser.log 2>&1; echo $?; }}',
                        'parse-operator-dump')
    report['parser_exitcode'] = int(code.strip())
    _, log = hdc.shell(f'cat {remote}/parser.log', 'dump-parser-log')
    (out/'parser.log').write_text(log + '\n')
    _, listing = hdc.shell(f'cd {destination} && test -z "$(find . -type l -print)" && '
                           r'find . -type f -exec stat -c "%s %n" {} \;', 'dump-files')
    files = report['files']
    for line in listing.splitlines():
        size, name = line.split(' ./', 1)
        files[str(_relative(name))] = int(size)
    if files:
        if sum(files.values()) > shutil.disk_usage(out).free:
            raise OSError(f'Insufficient host space to retrieve {destination}')
        hdc.receive_tree(destination, out/'dump', files)
        for name, size in files.items():
            if (out/'dump'/name).stat().st_size != size:
                raise ValueError(f'Incomplete dumped file: {name}')
    candidates = []
    for name, size in files.items():
        path = PurePosixPath(name.lower())
        tensor_file = path.suffix in ('.bin', '.npy', '.npz', '.data', '.dump', '.tensor')
        named_tensor = re.search(r'(?:^|[._-])(?:input|output)[_.-]?\d+$', path.name)
        if size and (tensor_file or named_tensor):
            candidates.append(name)
    report['tensor_candidates'] = candidates
    if report['parser_exitcode']:
        report['status'] = 'PARSER_FAILED'
    elif not candidates:
        report['status'] = 'NO_IDENTIFIED_TENSOR_FILES'
    else:
        report['status'] = 'FILES_RECEIVED_REVIEW_REQUIRED'
    save_json(out/'result.json', receipt)
    print(f"{report['status']}：回收 {len(files)} 个文件，{len(candidates)} 个张量候选。{out}/result.json", flush=True)
    if report['status'] != 'FILES_RECEIVED_REVIEW_REQUIRED':
        raise RuntimeError(f"{report['status']}: inspect {out}/parser.log and {out}/result.json")

"""Collect data_proc_tool's export after one successful V2 decoder invocation."""
from collections import Counter
from glob import escape
from pathlib import PurePosixPath
import re
import shlex
import shutil

from ...execution.hdc_run import _relative
from ..layer_probe.common import save_json
from .chain_hilog import tool_client_tids


def collect(hdc, remote, out, tool_dir, receipt, *, edges=()):
    destination = remote / 'operator-dump'
    parser = tool_dir / 'data_proc_tool_internal'
    argv = [str(parser), f'--result_path={destination}/']
    report = dict(status='PARSER_REQUESTED', argv=argv, device_dir=str(destination), files={})
    receipt['operator_dump'] = report
    save_json(out/'result.json', receipt)
    # The model process has exited and we still own the board lock. The parser can
    # export service history, so keep its directory structure for model attribution.
    print(f'解析算子 dump → {destination}', flush=True)
    _, code = hdc.shell(f'mkdir {destination} && cd {destination} && '
                        f'{{ env LD_LIBRARY_PATH={tool_dir} {shlex.join(argv)} >../parser.log 2>&1; echo $?; }}',
                        'parse-operator-dump')
    report['parser_exitcode'] = int(code.strip())
    _, log = hdc.shell(f'cat {remote}/parser.log', 'dump-parser-log')
    (out/'parser.log').write_text(log + '\n')
    predicate = ''
    if edges:
        report['requested_edges'] = sorted(edges)
        patterns = []
        for edge in sorted(edges):
            name, slot = edge.rsplit(':', 1)
            pattern = f'_*_*_{escape(name)}_[a-zA-Z]*_[0-9]*_{int(slot)}.dump'
            patterns.append('-name ' + shlex.quote(pattern))
        predicate = r'\( ' + ' -o '.join(patterns) + r' \) '
    _, listing = hdc.shell(f'cd {destination} && test -z "$(find . -type l -print)" && '
                           'find . -type f ' + predicate + r'-exec stat -c "%s %n" {} \;', 'dump-files')
    files = report['files']
    for line in listing.splitlines():
        size, name = line.split(' ./', 1)
        files[str(_relative(name))] = int(size)
    if files:
        if sum(files.values()) > shutil.disk_usage(out).free:
            raise OSError(f'Insufficient host space to retrieve {destination}')
        if edges:
            for name in files:
                local = out/'dump'/name
                local.parent.mkdir(parents=True, exist_ok=True)
                hdc.receive(destination/name, local)
        else:
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
    elif not any(files.values()):
        report['status'] = 'NO_DUMP_DATA'
    else:
        report['status'] = 'FILES_RECEIVED_REVIEW_REQUIRED'
    _, window = hdc.shell(f'cd {remote} && stat -c "%n %y" tool.started tool.exitcode', 'dump-device-window')
    _, stdout = hdc.shell(f'cat {remote}/tool.stdout', 'dump-model-stdout')
    tids = tool_client_tids(stdout)
    inventory = out/'dump-inventory.json'
    save_json(inventory, dict(**report, attempt=receipt['attempt'], model_argv=receipt['argv'],
                              omc_sha256=receipt['omc_sha256'], model_exitcode=receipt['exitcode'],
                              device_window=window.splitlines(), client_tids=tids, total_bytes=sum(files.values())))
    save_json(out/'result.json', receipt)
    print(f"[dump] {report['status']}，parser 退出码 {report['parser_exitcode']}，回收 {len(files)} 个文件，"
          f"{sum(files.values()):,} 字节。完整清单 {inventory}", flush=True)
    examples = Counter()
    for name, size in sorted(files.items()):
        path = PurePosixPath(name)
        kind = path.suffix.lower()
        if examples[kind] < 3 and sum(examples.values()) < 20:
            print(f'  {size:>12,}  {name}', flush=True)
            examples[kind] += 1
